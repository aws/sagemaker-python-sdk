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
"""Tests for ``LLMAsJudgeEvaluator.validate_custom_metrics``.

Before this change ``_validate_custom_metrics_json`` only parsed the JSON. A metric with no
``{{prompt}}`` or ``{{prediction}}`` placeholder, 11 metrics, a 101-character rating
definition, or a rating value with neither ``floatValue`` nor ``stringValue`` reached the
evaluation job and failed there. ``LLMAsJudgeEvaluator`` had no ``validate_custom_metrics``.
"""

from __future__ import absolute_import

import copy
import json
from unittest.mock import Mock, patch

import pytest

from sagemaker.train.evaluate.llm_as_judge_evaluator import LLMAsJudgeEvaluator

DEFAULT_REGION = "us-west-2"
DEFAULT_ROLE = "arn:aws:iam::123456789012:role/test-role"
DEFAULT_MODEL = "llama3-2-1b-instruct"
DEFAULT_DATASET = "s3://test-bucket/dataset.jsonl"
DEFAULT_S3_OUTPUT = "s3://test-bucket/outputs/"
DEFAULT_MLFLOW_ARN = "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/test-server"
DEFAULT_MODEL_PACKAGE_GROUP_ARN = (
    "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group"
)
DEFAULT_BASE_MODEL_ARN = "arn:aws:sagemaker:us-west-2:aws:hub-content/SageMakerPublicHub/Model/llama3-2-1b-instruct/1.0.0"  # noqa: E501
DEFAULT_ARTIFACT_ARN = "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact"
DEFAULT_EVALUATOR_MODEL = "anthropic.claude-sonnet-4-5-20250929-v1:0"

VALID_METRIC = {
    "customMetricDefinition": {
        "name": "PositiveSentiment",
        "instructions": "Rate the sentiment. Prompt: {{prompt}} Response: {{prediction}}",
        "ratingScale": [
            {"definition": "Positive", "value": {"floatValue": 1.0}},
            {"definition": "Negative", "value": {"floatValue": 0.0}},
        ],
    }
}


def _metric(**definition_changes):
    metric = copy.deepcopy(VALID_METRIC)
    metric["customMetricDefinition"].update(definition_changes)
    return metric


def _validate(metrics):
    return LLMAsJudgeEvaluator.validate_custom_metrics(metrics)


def test_valid_payload_returns_no_violation():
    string_scale = _metric(
        name="Tone",
        ratingScale=[
            {"definition": "Formal", "value": {"stringValue": "formal"}},
            {"definition": "Casual", "value": {"stringValue": "casual"}},
        ],
    )
    no_scale = _metric(name="Free")
    del no_scale["customMetricDefinition"]["ratingScale"]

    assert _validate([VALID_METRIC, string_scale, no_scale]) == []
    assert _validate(json.dumps([VALID_METRIC])) == []


def test_invalid_json_string_is_one_violation():
    violations = _validate("not valid json {")

    assert len(violations) == 1
    assert violations[0].startswith("Invalid JSON:")


def test_non_array_is_rejected():
    assert _validate(VALID_METRIC) == ["custom_metrics must be a JSON array of metric definitions."]


def test_empty_array_is_rejected():
    assert _validate([]) == ["custom_metrics is empty. Define at least one metric."]


def test_more_than_ten_metrics_is_rejected():
    violations = _validate([_metric(name=f"m{index}") for index in range(11)])

    assert violations == ["custom_metrics has 11 metrics, the maximum is 10."]
    assert _validate([_metric(name=f"m{index}") for index in range(10)]) == []


def test_item_without_custom_metric_definition_is_rejected():
    violations = _validate([{"name": "test"}])

    assert violations == ["Metric [0] must be an object with a customMetricDefinition key."]


def test_name_must_be_a_string():
    violations = _validate([_metric(name=7)])

    assert violations == ["Metric [0]: name must be a string."]


def test_instructions_longer_than_5000_characters_are_rejected():
    long_instructions = "{{prompt}}" + "x" * (5000 - len("{{prompt}}"))

    assert len(long_instructions) == 5000
    assert _validate([_metric(instructions=long_instructions)]) == []
    violations = _validate([_metric(instructions=long_instructions + "y")])
    assert violations == ["Metric [0]: instructions is 5001 characters, the limit is 5000."]


def test_instructions_without_a_placeholder_are_rejected():
    violations = _validate([_metric(instructions="Rate the answer from 1 to 5.")])

    assert violations == ["Metric [0]: instructions must contain {{prompt}} or {{prediction}}."]


def test_instructions_with_only_prediction_placeholder_are_valid():
    assert _validate([_metric(instructions="Rate {{prediction}}.")]) == []


def test_rating_definition_longer_than_100_characters_is_rejected():
    scale = [
        {"definition": "d" * 100, "value": {"floatValue": 1.0}},
        {"definition": "d" * 101, "value": {"floatValue": 0.0}},
    ]

    violations = _validate([_metric(ratingScale=scale)])

    assert violations == [
        "Metric [0]: ratingScale[1].definition is 101 characters, the limit is 100."
    ]


def test_rating_value_with_neither_or_two_values_is_rejected():
    scale = [
        {"definition": "Neither", "value": {}},
        {"definition": "Two", "value": {"floatValue": 1.0, "stringValue": "one"}},
    ]

    violations = _validate([_metric(ratingScale=scale)])

    assert violations == [
        "Metric [0]: ratingScale[0].value must set exactly one of floatValue or stringValue.",
        "Metric [0]: ratingScale[1].value must set exactly one of floatValue or stringValue.",
    ]


def test_rating_scale_with_mixed_value_types_is_rejected():
    scale = [
        {"definition": "Good", "value": {"floatValue": 1.0}},
        {"definition": "Bad", "value": {"stringValue": "bad"}},
    ]

    violations = _validate([_metric(ratingScale=scale)])

    assert violations == ["Metric [0]: ratingScale mixes floatValue and stringValue. Use one type."]


def test_violations_of_every_metric_are_reported():
    violations = _validate([_metric(instructions="no placeholder"), _metric(name=None)])

    assert violations == [
        "Metric [0]: instructions must contain {{prompt}} or {{prediction}}.",
        "Metric [1]: name must be a string.",
    ]


@patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
@patch("sagemaker.core.resources.Artifact")
def test_validate_custom_metrics_json_raises_with_the_joined_messages(mock_artifact, mock_resolve):
    mock_info = Mock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info

    mock_artifact.get_all.return_value = iter([])
    mock_artifact_instance = Mock()
    mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
    mock_artifact.create.return_value = mock_artifact_instance

    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE

    evaluator = LLMAsJudgeEvaluator(
        evaluator_model=DEFAULT_EVALUATOR_MODEL,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )

    valid_json = json.dumps([VALID_METRIC])
    assert evaluator._validate_custom_metrics_json(valid_json) == valid_json

    with pytest.raises(ValueError) as excinfo:
        evaluator._validate_custom_metrics_json(json.dumps([{"name": "test"}]))

    assert str(excinfo.value) == (
        "Invalid custom_metrics:\n"
        "  - Metric [0] must be an object with a customMetricDefinition key."
    )
