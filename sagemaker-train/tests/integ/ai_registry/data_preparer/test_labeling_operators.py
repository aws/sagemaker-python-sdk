# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Integration tests: the auto-labeling operators, as a labeling project calls them.

``test_labeling_happy_path`` submits one job through the typed options. This module
drives the client-side operators of a labeling project against real S3, Bedrock and
labeling jobs, in the order a customer runs them:

    inspect_labeling_data -> prepare_labeling_seed -> draft_labeling_spec
    -> tune_labeling_spec (a dev and a val job per round) -> select_labeling_spec
    -> label_corpus (the certified corpus job) -> write_report

The seed is ``labeling_seed.jsonl``: 30 banking-support messages, 6 per intent, labeled
in ``gt_label``, over the five intents of ``labeling_spec.md``. The corpus is
``labeling_corpus.jsonl``. With so few rows nothing certifies; what is checked is that
each operator reads what the previous one wrote, the jobs complete, and the client and
the job agree on the scores.

``TestClientOperators`` makes no job (S3 reads and one Bedrock call, seconds).
``TestRefinementLoop`` submits three labeling jobs and waits for them (about ten
minutes): scheduled CI only, not PR checks (see the ``gpu_intensive`` marker in
``sagemaker-train/pyproject.toml``).
"""

from __future__ import absolute_import

import json
import os
import uuid

import pytest

from sagemaker.ai_registry.data_preparer.auto_labeling import (
    cleanup_scratch,
    draft_labeling_spec,
    inspect_labeling_data,
    label_corpus,
    parse_label_set,
    prepare_labeling_seed,
    score,
    select_labeling_spec,
    tune_labeling_spec,
    write_report,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.split import (
    SplitGuardError,
)

REGION_ENV = "AWS_REGION"
LABELING_MODEL = "openai.gpt-oss-120b-1:0"
# A classification job that routes its low-confidence tail to review votes a jury,
# so the corpus job needs at least two models.
JURY_MODELS = "openai.gpt-oss-120b-1:0,qwen.qwen3-235b-a22b-2507-v1:0"
INTENTS = [
    "card_arrival",
    "card_not_working",
    "declined_card_payment",
    "lost_or_stolen_card",
    "top_up_failed",
]
SEED_ROWS = 30


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _region(sagemaker_session):
    return os.environ.get(REGION_ENV) or sagemaker_session.boto_region_name


@pytest.fixture(scope="module")
def spec_path():
    """The fixture spec, read from disk: every project-level operator takes a local spec."""
    return os.path.join(os.path.dirname(__file__), "fixtures", "labeling_spec.md")


@pytest.fixture(scope="module")
def data(sagemaker_session, s3_fixtures, spec_path):
    """``inspect_labeling_data`` over the S3 corpus and seed: the report every later call takes."""
    return inspect_labeling_data(
        spec=spec_path,
        corpus=s3_fixtures["labeling_corpus.jsonl"],
        seed=s3_fixtures["labeling_seed.jsonl"],
        region=_region(sagemaker_session),
    )


@pytest.mark.integ
class TestClientOperators:
    """The operators that run on the client: they read S3 and call Bedrock, and make no job."""

    def test_inspect_reads_the_s3_seed_and_spec(self, data):
        """The task, fields and label set come from the spec; the gate counts the S3 seed."""
        assert data.task == "classification"
        assert data.text_fields == ["text"]
        assert data.label_field == "gt_label"
        assert data.labels == INTENTS
        assert data.gate is not None
        assert data.gate["coverage"]["per_class"] == {label: 6 for label in INTENTS}
        assert data.gate["coverage"]["unknown_seed_labels"] == []

    def test_prepare_seed_splits_once_and_guards_the_split(self, data, tmp_path):
        """The first call splits and freezes; the second keeps the split; an edit is refused."""
        project = str(tmp_path / "project")
        seed = prepare_labeling_seed(project, data)

        assert seed.status == "ready"
        assert seed.frozen_now
        assert sum(seed.row_counts.values()) == SEED_ROWS
        assert set(seed.row_counts) == {"dev", "val", "test"}
        splits = {
            name: _read_jsonl(os.path.join(project, "splits", f"{name}.jsonl"))
            for name in ("dev", "val", "test")
        }
        ids = [row["id"] for rows in splits.values() for row in rows]
        assert len(ids) == len(set(ids)) == SEED_ROWS  # every seed row in exactly one split
        # Stratified: every intent is in dev, so the rounds can draw demos for each.
        assert {row["gt_label"] for row in splits["dev"]} == set(INTENTS)
        demo_labels = {row["gt_label"] for row in _read_jsonl(seed.demos)}
        assert demo_labels == set(INTENTS)

        again = prepare_labeling_seed(project, data)
        assert again.status == "ready"
        assert not again.frozen_now
        assert again.row_counts == seed.row_counts

        # A frozen split no longer matches its hash once a row changes, and the next
        # call refuses rather than scoring against an edited gold.
        test_file = os.path.join(project, "splits", "test.jsonl")
        rows = splits["test"]
        rows[0]["gt_label"] = next(label for label in INTENTS if label != rows[0]["gt_label"])
        with open(test_file, "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(row) + "\n" for row in rows)
        with pytest.raises(SplitGuardError):
            prepare_labeling_seed(project, data)

    def test_cold_start_draws_a_sample_from_the_s3_corpus(
        self, sagemaker_session, s3_fixtures, spec_path, tmp_path
    ):
        """With no seed, the operator samples the corpus for people to label."""
        cold = inspect_labeling_data(
            spec=spec_path,
            corpus=s3_fixtures["labeling_corpus.jsonl"],
            region=_region(sagemaker_session),
            text_fields=["text"],
        )
        project = str(tmp_path / "cold")
        seed = prepare_labeling_seed(project, cold)

        assert seed.status == "needs_labels"
        assert seed.sample and os.path.isfile(seed.sample)
        sample = _read_jsonl(seed.sample)
        assert len(sample) == seed.sample_rows > 0
        assert all(row.get("text") for row in sample)
        assert not os.path.exists(os.path.join(project, "splits", "dev.jsonl"))

    def test_draft_spec_from_dev_keeps_the_label_set(self, data, tmp_path):
        """The induced spec is written from dev through Bedrock and keeps the spec's labels."""
        project = str(tmp_path / "draft")
        prepare_labeling_seed(project, data)
        draft = draft_labeling_spec(
            project, data, task_hint="Which banking intent does the message express?"
        )

        assert os.path.isfile(draft.path)
        with open(draft.path, encoding="utf-8") as fh:
            text = fh.read()
        assert sorted(name for name, _ in parse_label_set(text)) == INTENTS


@pytest.mark.integ
@pytest.mark.gpu_intensive
class TestRefinementLoop:
    """One refinement round, the spec decision and the corpus job, sharing one project.

    The tests run in file order and each reads what the previous one wrote; a failure in
    one fails the rest by ``pytest.skip`` rather than by a confusing second error.
    """

    state = {}

    @pytest.fixture(scope="class")
    def project(self, tmp_path_factory, data):
        """The project folder, with the seed prepared; scratch DataSets are deleted after."""
        path = str(tmp_path_factory.mktemp("refinement") / "tickets")
        prepare_labeling_seed(path, data)
        yield path
        cleanup_scratch(path, delete=True)

    @pytest.fixture(scope="class")
    def s3_prefix(self, integ_bucket):
        return f"s3://{integ_bucket}/data-preparer-integ-operators/{uuid.uuid4().hex[:8]}"

    def _need(self, key):
        if key not in self.state:
            pytest.skip(f"an earlier step in {type(self).__name__} did not produce {key}")
        return self.state[key]

    def test_01_round_labels_dev_and_val(self, project, data, spec_path, s3_prefix, dt_role):
        """``next()`` runs a dev and a val job on the spec and returns the round's findings."""
        rounds = tune_labeling_spec(
            project,
            data,
            spec=spec_path,
            s3_prefix=s3_prefix,
            role_arn=dt_role,
            generation_model_id=LABELING_MODEL,
            max_iterations=0,
        )
        findings = next(rounds)

        assert findings.iteration == 0
        assert findings.decision in ("accept", "refine", "surface", "stop")
        for result, stage in ((findings.dev, "dev"), (findings.val, "val")):
            assert result.stage == stage
            assert result.summary["job_status"] == "Completed"
            assert result.summary["failed_rows"] == 0
            assert result.dataset_arn
            assert result.accuracy is not None and 0.0 <= result.accuracy <= 1.0
            assert os.path.isfile(result.out)
        assert findings.val_accuracy == findings.val.accuracy
        assert os.path.isfile(os.path.join(project, "work", "labeling_spec_iter_0.md"))
        # The loop stops after max_iterations, whatever the routing said.
        with pytest.raises(StopIteration):
            next(rounds)
        self.state["rounds"] = rounds
        self.state["findings"] = findings

    def test_02_client_score_matches_the_job(self, project):
        """``score`` over the val job's labels and the val split gives the job's own accuracy."""
        findings = self._need("findings")
        predictions = _read_jsonl(findings.val.out)
        gold = _read_jsonl(os.path.join(project, "splits", "val.jsonl"))
        local = score("classification", predictions, gold, gt_field="gt_label")
        assert local.accuracy == pytest.approx(findings.val.accuracy)

    def test_03_select_ships_a_spec(self, project):
        """``select_labeling_spec`` decides and writes the project's ``labeling_spec.md``."""
        decision = select_labeling_spec(project, self._need("rounds"))

        assert decision.ship
        assert decision.spec and os.path.isfile(decision.spec)
        shipped = os.path.join(project, "labeling_spec.md")
        assert os.path.isfile(shipped)
        with open(shipped, encoding="utf-8") as a, open(decision.spec, encoding="utf-8") as b:
            assert a.read() == b.read()
        assert os.path.isfile(os.path.join(project, "work", "decision.json"))
        self.state["decision"] = decision

    def test_04_label_corpus_registers_the_labeled_dataset(
        self, project, data, s3_fixtures, s3_prefix, dt_role, sagemaker_session
    ):
        """The corpus job runs the shipped spec, scores the test split and registers the DataSet."""
        self._need("decision")
        name = f"integ-operators-{uuid.uuid4().hex[:8]}"
        result = label_corpus(
            project,
            data,
            corpus=s3_fixtures["labeling_corpus.jsonl"],
            dataset_name=name,
            s3_prefix=s3_prefix,
            role_arn=dt_role,
            target=0.85,
            jury_models=JURY_MODELS,
        )
        try:
            assert result.stage == "corpus"
            assert result.summary["job_status"] == "Completed"
            assert result.dataset_arn and f"/DataSet/{name}/" in result.dataset_arn
            assert result.review_dataset_arn  # the target routes the tail to review
            # Five test rows are under the certificate's minimum, so nothing certifies.
            assert result.summary["status"] == "NotCertified"
            assert result.summary["held_out_rows"] == len(
                _read_jsonl(os.path.join(project, "splits", "test.jsonl"))
            )
            assert os.path.isfile(os.path.join(project, "labeled_dataset.json"))
            self.state["dataset_name"] = name
        finally:
            _delete_datasets(sagemaker_session, result.dataset_arn, result.review_dataset_arn)

    def test_05_report_renders_the_run(self, project):
        """``write_report`` renders the corpus run as HTML and Markdown."""
        name = self._need("dataset_name")
        written = write_report(project, dataset=name)

        assert written["html"] and os.path.isfile(written["html"])
        assert written["md"] and os.path.isfile(written["md"])
        with open(written["md"], encoding="utf-8") as fh:
            assert name in fh.read()


def _delete_datasets(sagemaker_session, *arns):
    """Delete the labeled and review DataSet versions the corpus job registered."""
    client = sagemaker_session.sagemaker_client
    for arn in arns:
        if not arn:
            continue
        # arn:...:hub-content/<hub>/DataSet/<name>/<version>
        hub, content_type, name, version = arn.split(":hub-content/", 1)[1].split("/")
        try:
            client.delete_hub_content(
                HubName=hub,
                HubContentType=content_type,
                HubContentName=name,
                HubContentVersion=version,
            )
        except Exception:  # noqa: BLE001 - cleanup must not mask the test's own failure
            pass
