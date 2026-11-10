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
"""The project-level flow.

inspect_labeling_data -> prepare_labeling_seed -> tune_labeling_spec -> select_labeling_spec -> label_corpus.

The job client is the fake of test_round.py; what is checked is that the flow drives the
same operators with the same arguments the skill's templates pass, writes the same files,
and routes refinement as refinement-routing.md says.
"""

from __future__ import absolute_import

import json
from pathlib import Path

import pytest

from sagemaker.ai_registry.data_preparer.auto_labeling import (
    select_labeling_spec,
    inspect_labeling_data,
    label_corpus,
    run_labeling_round,
    prepare_labeling_seed,
    tune_labeling_spec,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _run_log
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import client as J
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import round as R
from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError

from .test_round import ROLE, SPEC, _Fake, _jsonl


def _seed_rows(per_class=8, classes=("a", "b")):
    rows = []
    for c in classes:
        for i in range(per_class):
            rows.append({"id": f"{c}{i}", "text": f"{c} text {i} about topic {c}", "gt_label": c})
    return rows


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "labeling_spec.md").write_text(SPEC)
    _jsonl(root / "seed.jsonl", _seed_rows())
    _jsonl(root / "corpus.jsonl", [{"id": f"c{i}", "text": f"corpus row {i}"} for i in range(10)])
    return root


def _data(project):
    return inspect_labeling_data(
        spec=str(project / "labeling_spec.md"),
        seed=str(project / "seed.jsonl"),
        corpus=str(project / "corpus.jsonl"),
        region="us-west-2",
    )


class TestInspectData:
    def test_fields_task_labels_and_gate(self, project):
        data = _data(project)
        assert data.task == "classification" and data.text_fields == ["text"]
        assert data.label_field == "gt_label" and data.labels == ["a", "b"]
        assert data.gate["decision"] in ("go", "review-before-spend")
        assert data.region == "us-west-2" and data.spec.endswith("labeling_spec.md")

    def test_needs_a_corpus_or_a_seed(self, project):
        with pytest.raises(DataPreparerOperatorError, match="corpus, a seed"):
            inspect_labeling_data(spec=str(project / "labeling_spec.md"))


class TestPrepareSeed:
    def test_splits_freezes_and_writes_demos_and_folds_once(self, project):
        data = _data(project)
        first = prepare_labeling_seed(str(project), data, demos_per_class=2, k=2)
        assert first.status == "ready" and first.frozen_now and first.split_strategy == "label"
        assert sum(first.row_counts.values()) == 16 and all(first.row_counts.values())
        for name in ("dev.jsonl", "val.jsonl", "test.jsonl", "split_state.json", "manifest.json"):
            assert (project / "splits" / name).is_file()
        assert Path(first.demos).is_file() and (Path(first.folds_dir) / "manifest.json").is_file()
        kinds = [(e["kind"], e["step"], e["status"]) for e in _run_log.read_entries(str(project))]
        assert ("artifact", "split", "written") in kinds and (
            "artifact",
            "split",
            "frozen",
        ) in kinds
        # a second call finds the frozen split and refreshes the previews only
        second = prepare_labeling_seed(str(project), data, demos_per_class=2, k=2)
        assert second.status == "ready" and not second.frozen_now and second.split_strategy is None
        assert second.row_counts == first.row_counts

    def test_a_seed_row_without_its_label_is_refused(self, project):
        _jsonl(project / "seed.jsonl", [{"id": "x", "text": "t"}])
        with pytest.raises(DataPreparerOperatorError, match="no 'gt_label'"):
            prepare_labeling_seed(str(project), _data(project))

    def test_cold_start_writes_the_sample_and_asks_for_labels(self, project):
        (project / "seed.jsonl").unlink()
        data = inspect_labeling_data(
            spec=str(project / "labeling_spec.md"), corpus=str(project / "corpus.jsonl")
        )
        res = prepare_labeling_seed(str(project), data, target=0.9)
        assert res.status == "needs_labels" and res.sample_rows == 10  # the whole 10-row corpus
        rows = [json.loads(x) for x in Path(res.sample).read_text().splitlines()]
        assert rows and set(rows[0]) == {"id", "text"}
        assert any("call prepare_labeling_seed again" in w for w in res.warnings)
        assert not (project / "splits").exists()


class TestLabelRound:
    def test_dev_round_passes_the_skills_arguments_and_writes_the_files(self, project, monkeypatch):
        data = _data(project)
        prepare_labeling_seed(str(project), data, demos_per_class=2, k=2)
        fake = _Fake().install(monkeypatch)
        res = run_labeling_round(
            str(project),
            data,
            spec=str(project / "labeling_spec.md"),
            stage="dev",
            s3_prefix="s3://b/p/",
            role_arn=ROLE,
            demos_per_class=2,
        )
        assert res.stage == "dev" and res.iteration == 0
        assert (
            res.out == str(project / "work" / "dev_pred_iter_0.jsonl") and Path(res.out).is_file()
        )
        assert res.run_id.startswith("dev-iter0-") and res.accuracy == 0.9 and res.macro_f1 == 0.8
        kw = fake.submits[0]
        doc = kw["document"]
        cfg = doc["GenerationConfig"]["DatasetOptions"]["Config"]
        assert kw["role_arn"] == ROLE and "EvaluationData" in cfg  # the split is its own gold
        assert "Demos" in cfg  # --demos splits/dev.jsonl --demos-per-class 2
        assert Path(project / "work" / "dev_pred_iter_0.summary.json").is_file()
        kinds = [(e["kind"], e["status"]) for e in _run_log.read_entries(str(project))]
        assert ("job", "completed") in kinds and (
            "artifact",
            "scratch",
        ) in kinds  # a refine round is scratch
        assert res.run_log_entry and res.run_log_entry.startswith("artifact-auto-label-iter0")

    def test_the_same_spec_reattaches_under_the_same_run_id(self, project, monkeypatch):
        data = _data(project)
        prepare_labeling_seed(str(project), data, k=2)
        fake = _Fake().install(monkeypatch)
        kw = dict(
            spec=str(project / "labeling_spec.md"),
            stage="dev",
            s3_prefix="s3://b/p/",
            role_arn=ROLE,
        )
        first = run_labeling_round(str(project), data, **kw)
        second = run_labeling_round(str(project), data, **kw)
        assert first.run_id == second.run_id and second.summary["reattached"] is True
        assert (
            len(fake.submits) == 2 and fake.submits[1]["resume_job"]["job_name"] == first.job_name
        )


class _Scores:
    """Val HeldOutAccuracy per round, so the routing can be exercised."""

    def __init__(self, fake, *accuracies):
        self.fake, self.accuracies, self.calls = fake, list(accuracies), 0
        self._read = _Fake.read.__get__(fake)
        J.read_job_results = (
            self.read
        )  # install() bound the original; the test's monkeypatch restores it

    def read(self, **kw):
        n_round = (len(self.fake.submits) - 1) // 2  # dev then val per round
        acc = self.accuracies[min(n_round, len(self.accuracies) - 1)]
        self.fake.summary = {
            "Status": "NotCertified",
            "RowsLabeled": 8,
            "RowsMissing": 0,
            "HeldOutAccuracy": acc,
            "HeldOutLowerBound95": acc - 0.1,
            "HeldOutRows": 8,
            "JurorModelIds": ["m"],
            "Warnings": [],
        }
        return self._read(**kw)


def _breakdown(monkeypatch, per_class):
    """The held-out report the fake writes carries a per-class breakdown."""
    orig = _Fake.read

    def read(self, **kw):
        out = orig(self, **kw)
        path = out["files"].get("HeldOutReport")
        if path:
            Path(path).write_text(
                json.dumps(
                    {
                        "macro_f1": 0.8,
                        "per_class_f1": per_class,  # the job's key
                        "confusion_pairs": [["a", "b", 2]],
                    }
                )
            )
        return out

    monkeypatch.setattr(_Fake, "read", read)


class TestRefinementRounds:
    def _project(self, project, monkeypatch, per_class_val=6):
        _jsonl(project / "seed.jsonl", _seed_rows(per_class=per_class_val * 4))  # val gets ~1/4
        data = _data(project)
        prepare_labeling_seed(str(project), data, k=2)
        return data, _Fake().install(monkeypatch)

    def test_a_round_is_dev_then_val_on_a_snapshot_and_routes(self, project, monkeypatch):
        data, fake = self._project(project, monkeypatch)
        _breakdown(monkeypatch, {"a": 0.7, "b": 0.95})
        _Scores(fake, 0.8, 0.9)
        rounds = tune_labeling_spec(
            str(project),
            data,
            spec=str(project / "labeling_spec.md"),
            s3_prefix="s3://b/p/",
            role_arn=ROLE,
            gate=0.85,
        )
        r0 = next(rounds)
        assert r0.iteration == 0 and r0.spec == str(project / "work" / "labeling_spec_iter_0.md")
        assert len(fake.submits) == 2  # dev, then val
        assert r0.dev.stage == "dev" and r0.val.stage == "val" and r0.val_accuracy == 0.8
        assert r0.decision == "refine" and "below gate" in r0.decision_reason
        assert "a" in r0.classes_below_gate and "iter_0" in r0.summary()
        # the person edits the spec; the next round snapshots the new text
        (project / "labeling_spec.md").write_text(SPEC + "\nRule: b when the text says topic b.\n")
        r1 = next(rounds)
        assert r1.iteration == 1 and Path(r1.spec).read_text().endswith("topic b.\n")
        assert r1.spec_sha256 != r0.spec_sha256 and r1.val_gain == pytest.approx(0.1)
        assert (
            r1.decision == "stop" or r1.decision == "refine"
        )  # 0.9 >= gate but class a < gate -> refine/stop rule
        scores = rounds.scores()
        assert scores["initial"]["spec"] == r0.spec and scores["iters"][0]["val"] == 0.9
        assert [
            e["status"] for e in _run_log.read_entries(str(project)) if e["step"] == "refine"
        ] == [r0.decision, r1.decision]

    def test_accept_stops_the_iteration(self, project, monkeypatch):
        data, fake = self._project(project, monkeypatch)
        _breakdown(monkeypatch, {"a": 0.9, "b": 0.95})
        _Scores(fake, 0.92)
        rounds = tune_labeling_spec(
            str(project),
            data,
            spec=str(project / "labeling_spec.md"),
            s3_prefix="s3://b/p/",
            role_arn=ROLE,
            gate=0.85,
        )
        r0 = next(rounds)
        assert r0.decision == "accept"
        with pytest.raises(StopIteration):
            next(rounds)
        assert len(fake.submits) == 2

    def test_thin_val_never_refines(self, project, monkeypatch):
        data, fake = self._project(
            project, monkeypatch, per_class_val=2
        )  # 8 per class -> 2 val rows each
        _Scores(fake, 0.5)
        rounds = tune_labeling_spec(
            str(project),
            data,
            spec=str(project / "labeling_spec.md"),
            s3_prefix="s3://b/p/",
            role_arn=ROLE,
        )
        r0 = next(rounds)
        assert r0.decision == "surface" and "per class" in r0.decision_reason
        with pytest.raises(StopIteration):
            next(rounds)

    def test_an_unchanged_spec_is_flagged_and_the_cap_ends_the_loop(self, project, monkeypatch):
        data, fake = self._project(project, monkeypatch)
        _breakdown(monkeypatch, {"a": 0.5, "b": 0.5})
        _Scores(fake, 0.5, 0.6, 0.7)
        rounds = tune_labeling_spec(
            str(project),
            data,
            spec=str(project / "labeling_spec.md"),
            s3_prefix="s3://b/p/",
            role_arn=ROLE,
            max_iterations=1,
        )
        next(rounds)
        r1 = next(rounds)
        assert any("unchanged since iter_0" in w for w in r1.warnings)
        with pytest.raises(StopIteration):
            next(rounds)


class TestEvaluateAndCorpus:
    def test_evaluate_writes_scores_decision_and_the_shipped_spec(self, project, monkeypatch):
        data = _data(project)
        prepare_labeling_seed(str(project), data, k=2)
        fake = _Fake().install(monkeypatch)
        _Scores(fake, 0.80, 0.90)
        rounds = tune_labeling_spec(
            str(project),
            data,
            spec=str(project / "labeling_spec.md"),
            s3_prefix="s3://b/p/",
            role_arn=ROLE,
            gate=0.95,
            max_iterations=2,
        )
        rounds.run_round()
        (project / "labeling_spec.md").write_text(SPEC + "\nedit\n")
        rounds.run_round()
        decision = select_labeling_spec(str(project), rounds)
        assert decision.ship == "refined" and decision.spec.endswith("labeling_spec_iter_1.md")
        scores = json.loads((project / "scores.json").read_text())
        assert scores["initial"]["val"] == 0.8 and scores["iters"][0]["val"] == 0.9
        assert json.loads((project / "work" / "decision.json").read_text())["ship"] == "refined"
        assert (project / "labeling_spec.md").read_text() == Path(decision.spec).read_text()
        # the do-no-harm floor keeps the initial spec when the gain is below it
        decision2 = select_labeling_spec(
            str(project),
            scores={"initial": {"spec": "i0", "val": 0.9}, "iters": [{"spec": "i1", "val": 0.905}]},
        )
        assert decision2.ship == "initial"

    def test_label_corpus_certifies_on_test_and_flags_an_overfit(self, project, monkeypatch):
        data = _data(project)
        prepare_labeling_seed(str(project), data, k=2)
        fake = _Fake().install(monkeypatch)
        _Scores(fake, 0.7)  # the test score the corpus job reports
        (project / "work").mkdir(exist_ok=True)
        (project / "work" / "decision.json").write_text(
            json.dumps({"ship": "initial", "val": 0.95})
        )
        res = label_corpus(
            str(project),
            data,
            corpus=str(project / "corpus.jsonl"),
            dataset_name="tickets",
            s3_prefix="s3://b/p/",
            role_arn=ROLE,
            target=0.9,
            jury_models="m1,m2",
        )
        doc = fake.submits[-1]["document"]
        cfg = doc["GenerationConfig"]["DatasetOptions"]["Config"]
        assert (
            doc["DatasetName"] == "tickets" and "EvaluationData" in cfg
        )  # --eval splits/test.jsonl
        assert (
            res.stage == "corpus"
            and res.dataset_arn
            and res.out == str(project / "labeled_dataset.json")
        )
        assert not list(project.rglob("labels.jsonl")), "the corpus labels are never downloaded"
        assert any("review flag" in w for w in res.warnings)  # test 0.7 vs the selecting val 0.95
        kinds = [(e["kind"], e["step"], e["status"]) for e in _run_log.read_entries(str(project))]
        assert ("analysis", "certificate", "test-read") in kinds

    def test_label_corpus_without_a_seed_needs_the_opt_out(self, project, monkeypatch):
        data = _data(project)
        _Fake().install(monkeypatch)
        with pytest.raises(DataPreparerOperatorError, match="no labeled seed"):
            label_corpus(
                str(project),
                data,
                corpus=str(project / "corpus.jsonl"),
                dataset_name="t",
                s3_prefix="s3://b/p/",
                role_arn=ROLE,
            )


class TestRouting:
    @pytest.mark.parametrize(
        "acc,breakdown,counts,prev,it,expected",
        [
            (0.9, {"a": 0.9, "b": 0.95}, {"a": 6, "b": 6}, None, 0, "accept"),
            (0.8, {"a": 0.6, "b": 0.95}, {"a": 6, "b": 6}, None, 0, "refine"),
            (0.8, {"a": 0.6, "b": 0.95}, {"a": 2, "b": 2}, None, 0, "surface"),
            (0.8, {"a": 0.6, "b": 0.95}, {"a": 6, "b": 6}, 0.85, 1, "stop"),
            (None, {}, {"a": 6, "b": 6}, None, 0, "surface"),
        ],
    )
    def test_refinement_routing(self, acc, breakdown, counts, prev, it, expected):
        decision, reason = R.route_refinement(
            acc, breakdown, [["a", "b", 2]], counts, 0.85, prev, it
        )
        assert decision == expected and reason
