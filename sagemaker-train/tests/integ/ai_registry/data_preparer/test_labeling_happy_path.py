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
"""Integration test: an auto-labeling job through the typed options, read back
through ``LabelingSummary``.

One classification job over a 10-row corpus whose rows are also the gold
(``labeling_corpus.jsonl`` / ``labeling_gold.jsonl``, five banking intents), with the
opt-in ``Confidence`` object set. The job labels every row, scores them against the
gold, and the SDK reads its summary back off the labeled DataSet. Ten gold rows are
below the certificate's minimum, so the status is ``NotCertified``; what the test
checks is the request round-trip and the result read-back, not the accuracy.

Submits a real job and waits for it (about three minutes): scheduled CI only, not
PR checks (see the ``gpu_intensive`` marker in ``sagemaker-train/pyproject.toml``).
"""

from __future__ import absolute_import

import json
from urllib.parse import urlparse

import boto3
import pytest

from sagemaker.ai_registry import (
    ClassificationLabelingOptions,
    Confidence,
    DataPreparer,
    DatasetJobConfig,
    EvaluationData,
    JobCategory,
    LabelingSummary,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.client import (
    _sdg_output_field,
)

pytestmark = pytest.mark.gpu_intensive

GENERATION_MODEL = "openai.gpt-oss-120b-1:0"


def _s3_json_lines(sagemaker_session, uri):
    parsed = urlparse(uri)
    s3 = boto3.client("s3", region_name=sagemaker_session.boto_region_name)
    body = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"]
    return [json.loads(line) for line in body.read().decode("utf-8").splitlines() if line.strip()]


@pytest.mark.integ
class TestLabelingHappyPath:
    """A typed labeling request runs end to end and its summary reads back."""

    def test_labeling_job_with_confidence_reads_back(
        self, sagemaker_session, dt_role, s3_fixtures, integ_bucket
    ):
        """Submit a classification labeling job and read its LabelingSummary."""
        config = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            dataset_name="integ-labeling-happy-path",
            generation_config={
                "InputDataS3Uri": s3_fixtures["labeling_corpus.jsonl"],
                "DatasetSpecification": {"SpecS3Uri": s3_fixtures["labeling_spec.md"]},
                "ModelConfig": {"SampleGenerationModelConfigs": [{"ModelId": GENERATION_MODEL}]},
            },
            dataset_options=ClassificationLabelingOptions(
                text_fields=["text"],
                evaluation_data=EvaluationData(
                    s3_uri=s3_fixtures["labeling_gold.jsonl"], label_field="gt_label"
                ),
                # The opt-in object this PR types. ``verbalized`` labels with the one
                # model and asks it for a confidence, so no jury is needed.
                confidence=Confidence(method="verbalized"),
            ),
            output_data_config={
                _sdg_output_field(
                    sagemaker_session
                ): f"s3://{integ_bucket}/data-preparer-integ-labeling-output/",
            },
        )
        # The typed options serialize to the wire shape the service reads.
        sent = config.dataset_options.to_dataset_options()
        assert sent["Type"] == "labeling.classification"
        assert sent["Config"]["Confidence"] == {"Method": "verbalized"}
        assert sent["Config"]["EvaluationData"]["LabelField"] == "gt_label"

        preparer = DataPreparer(sagemaker_session=sagemaker_session, role=dt_role)
        dataset = preparer.generate(config)
        assert dataset is not None and dataset.arn

        summary = LabelingSummary.from_dataset(dataset, sagemaker_session=sagemaker_session)

        # The container wrote a summary for this job and the SDK found it off the DataSet.
        assert summary.type == "labeling.classification"
        assert summary.labeled_dataset_arn == dataset.arn
        assert summary.review_dataset_arn is None  # no Routing, so no review DataSet
        assert summary.status in ("Certified", "NotCertified", "Partial")
        assert summary.rows_labeled == 10
        assert summary.held_out_rows == 10
        assert summary.held_out_accuracy is not None
        assert 0.0 <= summary.held_out_accuracy <= 1.0
        assert summary.juror_model_ids == [GENERATION_MODEL]

        # The index names the labels file, and it carries one labeled row per input row
        # under the classification prediction field.
        labels_uri = summary.artifacts.get("Labels")
        assert labels_uri, f"index names no Labels artifact: {summary.artifacts!r}"
        labels = _s3_json_lines(sagemaker_session, labels_uri)
        assert len(labels) == 10
        allowed = {
            "card_arrival",
            "card_not_working",
            "declined_card_payment",
            "lost_or_stolen_card",
            "top_up_failed",
        }
        assert all(row.get("label") in allowed or row.get("label") is None for row in labels)
        # verbalized confidence: every labeled row carries the model's stated score.
        assert all(
            row.get("label") is None or 0.0 <= float(row["confidence_score"]) <= 1.0
            for row in labels
        )
