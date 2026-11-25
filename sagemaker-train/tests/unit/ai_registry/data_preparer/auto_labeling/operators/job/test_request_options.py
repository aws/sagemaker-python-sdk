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
"""job.request.labeling_options_from_document: every Config key reaches the job."""

from __future__ import absolute_import

import copy
import importlib

import pytest

R = importlib.import_module(
    "sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.request"
)


def _doc(task="classification", **config):
    doc = R.build_labeling_request(
        task=task,
        text_fields=["text"],
        spec_s3_uri="s3://b/spec.md",
        model_ids=["model-a"],
        dataset_name="tickets",
        output_s3_uri="s3://b/out/",
        input_s3_uri="s3://b/in.jsonl",
    )
    doc["GenerationConfig"]["DatasetOptions"]["Config"].update(copy.deepcopy(config))
    return doc


def _round_trip(doc):
    return R.labeling_options_from_document(doc).to_dataset_options()


@pytest.mark.parametrize(
    "config",
    [
        {"Review": {"LabelField": "category", "Scope": "seed"}},
        {
            "Review": {
                "LabelField": "gt_label",
                "Scope": "golden",
                "PredictionsS3Uris": ["s3://b/p1.jsonl", "s3://b/p2.jsonl"],
                "RulesS3Uri": "s3://b/rules.json",
                "JudgeModels": [{"ModelId": "judge-a"}],
                "JudgeScope": "flagged",
                "MaxJudgeRows": 200,
                "EstimateOnly": True,
            }
        },
        {"Routing": {"ReviewDatasetName": "tickets-review"}, "Scoring": {"Target": 0.9}},
        {"Demos": {"S3Uri": "s3://b/demos.jsonl"}},
        {"EvaluationData": {"LabelField": "gt_label", "S3Uri": "s3://b/gold.jsonl"}},
        {"Confidence": {"Method": "verbalized"}},
        {"Optimizer": {"Mode": "protegi", "Elicitation": "plain"}},
    ],
)
def test_every_option_the_document_carries_reaches_the_typed_options(config):
    doc = _doc(**config)
    assert _round_trip(doc) == doc["GenerationConfig"]["DatasetOptions"]


def test_a_review_request_is_not_submitted_as_a_labeling_job():
    # The regression this pins: Config.Review was dropped, so a label review ran as a
    # billed labeling job over the seed.
    doc = _doc(Review={"LabelField": "category", "Scope": "seed"})
    assert _round_trip(doc)["Config"]["Review"] == {"LabelField": "category", "Scope": "seed"}


def test_a_config_key_the_typed_options_do_not_take_is_refused():
    doc = _doc(SomethingNew={"x": 1})
    with pytest.raises(R.LabelingRequestError, match="SomethingNew"):
        R.labeling_options_from_document(doc)
