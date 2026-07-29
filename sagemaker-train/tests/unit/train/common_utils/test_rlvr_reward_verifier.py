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
"""Tests for the RLVR reward function verifier."""
from __future__ import absolute_import

import pytest
import logging
import tempfile
from pathlib import Path


from sagemaker.core.training.configs import TrainingJobCompute, HyperPodCompute
from sagemaker.train.common_utils.rlvr_reward_verifier import verify_reward_function


def _write_reward_fn(reward_code):
    """Write reward code to a temp .py file and return its path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        return f.name


def test_verify_with_valid_rft_format():
    """Test verification with valid RFT format data."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        messages = sample.get("messages", [])
        last_msg = messages[-1] if messages else {}
        response_len = len(last_msg.get("content", ""))
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": float(response_len) / 10.0
        })
    return results
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ],
                "reference_answer": "4",
            },
            {
                "id": "sample_2",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
                "reference_answer": "Hi there!",
            },
        ]

        result = verify_reward_function(reward_function=reward_file, sample_data=sample_data)

        assert result["success"] is True
        assert result["total_samples"] == 2
        assert result["successful_samples"] == 2

    finally:
        Path(reward_file).unlink()


def test_verify_with_validation_disabled():
    """Test that output validation can be disabled."""
    reward_code = """
def lambda_handler(event, context):
    return [{"id": "test", "aggregate_reward_score": 1.0}]
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [{"prompt": "test", "response": "answer"}]

        result = verify_reward_function(
            reward_function=reward_file, sample_data=sample_data, validate_format=False
        )

        assert result["success"] is True

    finally:
        Path(reward_file).unlink()


def test_verify_output_format_validation():
    """Test that output format is validated."""
    reward_code = """
def lambda_handler(event, context):
    return [{"wrong_field": "value"}]
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        with pytest.raises(ValueError) as exc_info:
            verify_reward_function(reward_function=reward_file, sample_data=sample_data)

        error_msg = str(exc_info.value)
        assert "id - Field required" in error_msg
        assert "aggregate_reward_score - Field required" in error_msg

    finally:
        Path(reward_file).unlink()


def test_verify_with_transformed_dataset_format():
    """Test verification with data that looks like SDK-transformed RFT dataset."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        messages = sample.get("messages", [])
        prediction = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                prediction = msg.get("content", "")
                break
        ref_answer = sample.get("reference_answer", "")
        reward = 1.0 if prediction.strip() == ref_answer.strip() else -1.0
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": reward
        })
    return results
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "Paris"},
                ],
                "reference_answer": "Paris",
            },
            {
                "id": "custom_id_123",
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ],
                "reference_answer": "4",
            },
        ]

        result = verify_reward_function(reward_function=reward_file, sample_data=sample_data)

        assert result["success"] is True
        assert result["total_samples"] == 2
        assert result["successful_samples"] == 2

    finally:
        Path(reward_file).unlink()


def test_verify_evaluation_mode_with_metrics():
    """Test verification with metrics_list."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.75,
            "metrics_list": [
                {"name": "accuracy", "value": 0.85, "type": "Metric"},
                {"name": "fluency", "value": 0.90, "type": "Reward"}
            ]
        })
    return results
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        result = verify_reward_function(reward_function=reward_file, sample_data=sample_data)

        assert result["success"] is True
        assert result["total_samples"] == 1
        assert result["successful_samples"] == 1

    finally:
        Path(reward_file).unlink()


def test_verify_evaluation_mode_without_metrics():
    """Test that metrics_list is optional."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.75
        })
    return results
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        result = verify_reward_function(reward_function=reward_file, sample_data=sample_data)

        assert result["success"] is True

    finally:
        Path(reward_file).unlink()


def test_verify_evaluation_mode_invalid_metrics_list():
    """Test that invalid metrics_list structure raises error."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.75,
            "metrics_list": "not a list"
        })
    return results
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        with pytest.raises(ValueError, match="Input should be a valid list"):
            verify_reward_function(reward_function=reward_file, sample_data=sample_data)

    finally:
        Path(reward_file).unlink()


def test_verify_evaluation_mode_missing_metric_fields():
    """Test that missing metric fields raise errors."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.75,
            "metrics_list": [
                {"name": "accuracy"}
            ]
        })
    return results
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        with pytest.raises(ValueError, match="Field required"):
            verify_reward_function(reward_function=reward_file, sample_data=sample_data)

    finally:
        Path(reward_file).unlink()


def test_verify_evaluation_mode_invalid_metric_type():
    """Test that invalid metric type raises error."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.75,
            "metrics_list": [
                {"name": "accuracy", "value": 0.85, "type": "InvalidType"}
            ]
        })
    return results
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        with pytest.raises(ValueError, match="Input should be 'Metric' or 'Reward'"):
            verify_reward_function(reward_function=reward_file, sample_data=sample_data)

    finally:
        Path(reward_file).unlink()


def test_verify_empty_results_raises_error():
    """Test that empty results (fewer than samples sent) raises error."""
    reward_code = """
def lambda_handler(event, context):
    return []
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        with pytest.raises(ValueError) as exc_info:
            verify_reward_function(reward_function=reward_file, sample_data=sample_data)

        assert "1/1 sample(s) failed validation" in str(exc_info.value)
        assert "0/1 sample(s) passed" in str(exc_info.value)

    finally:
        Path(reward_file).unlink()


def test_verify_evaluation_mode_with_multiple_metrics():
    """Test evaluation mode with multiple metrics of different types."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.82,
            "metrics_list": [
                {"name": "accuracy", "value": 0.85, "type": "Metric"},
                {"name": "fluency", "value": 0.90, "type": "Reward"},
                {"name": "coherence", "value": 0.78, "type": "Metric"},
                {"name": "relevance", "value": 0.95, "type": "Reward"}
            ]
        })
    return results
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        result = verify_reward_function(reward_function=reward_file, sample_data=sample_data)

        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["output"]["metrics_list"] is not None
        assert len(result["results"][0]["output"]["metrics_list"]) == 4

    finally:
        Path(reward_file).unlink()


def test_verify_evaluation_mode_empty_metrics_list():
    """Test that empty metrics_list is valid."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.75,
            "metrics_list": []
        })
    return results
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        result = verify_reward_function(reward_function=reward_file, sample_data=sample_data)

        assert result["success"] is True

    finally:
        Path(reward_file).unlink()


def test_verify_evaluation_mode_metric_with_wrong_value_type():
    """Test that metric value must be a number."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.75,
            "metrics_list": [
                {"name": "accuracy", "value": [1, 2], "type": "Metric"}
            ]
        })
    return results
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        with pytest.raises(ValueError, match="Input should be a valid"):
            verify_reward_function(reward_function=reward_file, sample_data=sample_data)

    finally:
        Path(reward_file).unlink()


def test_verify_evaluation_mode_metric_with_wrong_name_type():
    """Test that metric name must be a string."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.75,
            "metrics_list": [
                {"name": ["not", "a", "string"], "value": 0.85, "type": "Metric"}
            ]
        })
    return results
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        with pytest.raises(ValueError, match="Input should be a valid string"):
            verify_reward_function(reward_function=reward_file, sample_data=sample_data)

    finally:
        Path(reward_file).unlink()


def test_verify_evaluation_mode_multiple_samples():
    """Test evaluation mode with multiple samples."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for idx, sample in enumerate(event):
        results.append({
            "id": sample.get("id", f"sample_{idx}"),
            "aggregate_reward_score": 0.5 + (idx * 0.1),
            "metrics_list": [
                {"name": "accuracy", "value": 0.6 + (idx * 0.1), "type": "Metric"}
            ]
        })
    return results
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test1"}],
                "reference_answer": "answer1",
            },
            {
                "id": "sample_2",
                "messages": [{"role": "user", "content": "test2"}],
                "reference_answer": "answer2",
            },
            {
                "id": "sample_3",
                "messages": [{"role": "user", "content": "test3"}],
                "reference_answer": "answer3",
            },
        ]

        result = verify_reward_function(reward_function=reward_file, sample_data=sample_data)

        assert result["success"] is True
        assert result["total_samples"] == 3
        assert result["successful_samples"] == 3
        assert len(result["results"]) == 3

    finally:
        Path(reward_file).unlink()


def test_verify_evaluation_mode_mixed_valid_invalid_metrics():
    """Test evaluation mode with mix of valid and invalid metrics."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.75,
            "metrics_list": [
                {"name": "accuracy", "value": 0.85, "type": "Metric"},
                {"name": "fluency", "value": 0.90, "type": "InvalidType"}
            ]
        })
    return results
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        with pytest.raises(ValueError, match="Input should be 'Metric' or 'Reward'"):
            verify_reward_function(reward_function=reward_file, sample_data=sample_data)

    finally:
        Path(reward_file).unlink()


def test_verify_aggregate_reward_score_as_int():
    """Test that aggregate_reward_score can be an integer."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 1
        })
    return results
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        result = verify_reward_function(reward_function=reward_file, sample_data=sample_data)

        assert result["success"] is True
        assert result["results"][0]["output"]["aggregate_reward_score"] == 1

    finally:
        Path(reward_file).unlink()


def test_verify_metric_value_as_int():
    """Test that metric value can be an integer."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.75,
            "metrics_list": [
                {"name": "count", "value": 5, "type": "Metric"}
            ]
        })
    return results
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        result = verify_reward_function(reward_function=reward_file, sample_data=sample_data)

        assert result["success"] is True
        assert result["results"][0]["output"]["metrics_list"][0]["value"] == 5

    finally:
        Path(reward_file).unlink()


def test_verify_evaluation_mode_metric_not_dict():
    """Test that metric must be a dict."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.75,
            "metrics_list": [
                "not a dict"
            ]
        })
    return results
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        with pytest.raises(
            ValueError, match="Input should be a valid dictionary or instance of RewardMetric"
        ):
            verify_reward_function(reward_function=reward_file, sample_data=sample_data)

    finally:
        Path(reward_file).unlink()


# ---------------------------------------------------------------------------
# Platform / Lambda ARN validation
# ---------------------------------------------------------------------------

def test_verify_smhp_compute_invalid_lambda_arn():
    """Test that HyperPod compute rejects a Lambda ARN without 'SageMaker' in the name."""
    sample_data = [
        {
            "id": "sample_1",
            "messages": [{"role": "user", "content": "test"}],
            "reference_answer": "answer",
        }
    ]

    invalid_arn = "arn:aws:lambda:us-east-1:123456789012:function:my-reward-function"

    with pytest.raises(ValueError, match="Lambda ARN for HyperPod compute.*must contain 'SageMaker'"):
        verify_reward_function(
            reward_function=invalid_arn,
            sample_data=sample_data,
            compute=HyperPodCompute(cluster_name="test-cluster", instance_type="ml.p5.48xlarge"),
        )


def test_verify_smhp_compute_valid_lambda_arn_case_insensitive():
    """Test that HyperPod compute accepts Lambda ARNs with 'sagemaker' regardless of case."""
    sample_data = [
        {
            "id": "sample_1",
            "messages": [{"role": "user", "content": "test"}],
            "reference_answer": "answer",
        }
    ]

    valid_arns = [
        "arn:aws:lambda:us-east-1:123456789012:function:MySageMakerReward",
        "arn:aws:lambda:us-east-1:123456789012:function:my-sagemaker-reward",
        "arn:aws:lambda:us-east-1:123456789012:function:SageMaker-reward-function",
        "arn:aws:lambda:us-east-1:123456789012:function:reward-Sagemaker",
    ]

    compute = HyperPodCompute(cluster_name="test-cluster", instance_type="ml.p5.48xlarge")
    for arn in valid_arns:
        # These are fake ARNs, so invocation fails - but it must NOT fail on ARN validation.
        try:
            verify_reward_function(reward_function=arn, sample_data=sample_data, compute=compute)
        except Exception as e:  # noqa: BLE001
            assert "must contain 'SageMaker'" not in str(e)
            assert "'compute' parameter is required" not in str(e)


def test_verify_smtj_compute_no_lambda_arn_validation():
    """Test that SMTJ (TrainingJob) compute doesn't validate the Lambda ARN function name."""
    sample_data = [
        {
            "id": "sample_1",
            "messages": [{"role": "user", "content": "test"}],
            "reference_answer": "answer",
        }
    ]

    arn = "arn:aws:lambda:us-east-1:123456789012:function:my-reward-function"

    try:
        verify_reward_function(
            reward_function=arn,
            sample_data=sample_data,
            compute=TrainingJobCompute(instance_type="ml.p4d.24xlarge"),
        )
    except Exception as e:  # noqa: BLE001
        assert "must contain 'SageMaker'" not in str(e)


def test_verify_smhp_arn_with_version_qualifier():
    """Test HyperPod validation with a Lambda ARN that includes a version qualifier."""
    sample_data = [
        {
            "id": "sample_1",
            "messages": [{"role": "user", "content": "test"}],
            "reference_answer": "answer",
        }
    ]

    arn_with_version = "arn:aws:lambda:us-east-1:123456789012:function:MySageMakerReward:1"

    try:
        verify_reward_function(
            reward_function=arn_with_version,
            sample_data=sample_data,
            compute=HyperPodCompute(cluster_name="test-cluster", instance_type="ml.p5.48xlarge"),
        )
    except Exception as e:  # noqa: BLE001
        assert "must contain 'SageMaker'" not in str(e)
        assert "'compute' parameter is required" not in str(e)


def test_verify_smhp_compute_with_local_file(caplog):
    """Test that HyperPod compute logs a warning for local files."""
    caplog.set_level(logging.WARNING)

    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 1.0
        })
    return results
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        result = verify_reward_function(
            reward_function=reward_file,
            sample_data=sample_data,
            compute=HyperPodCompute(cluster_name="test-cluster", instance_type="ml.p5.48xlarge"),
        )

        assert result["success"] is True

        # Verify the warning was logged with file name and guidance
        log_text = caplog.text
        assert "Skipping Lambda function name validation" in log_text
        assert reward_file in log_text
        assert "Nova RLVR jobs on HyperPod" in log_text
        assert "'SageMaker'" in log_text
        assert "arn:aws:lambda:*:*:function:*SageMaker*" in log_text

    finally:
        Path(reward_file).unlink()


def test_verify_combined_evaluation_mode_and_smhp_compute():
    """Test using both metrics_list and HyperPod compute together with a local file."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.85,
            "metrics_list": [
                {"name": "accuracy", "value": 0.90, "type": "Metric"}
            ]
        })
    return results
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        result = verify_reward_function(
            reward_function=reward_file,
            sample_data=sample_data,
            compute=HyperPodCompute(cluster_name="test-cluster", instance_type="ml.p5.48xlarge"),
        )

        assert result["success"] is True
        assert result["results"][0]["output"]["metrics_list"] is not None

    finally:
        Path(reward_file).unlink()


# ---------------------------------------------------------------------------
# Logging behavior
# ---------------------------------------------------------------------------
def test_logging_shows_input_and_output_for_all_samples(caplog):
    """Test that logging shows input and output for all validated samples."""
    import logging

    caplog.set_level(logging.INFO)

    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "unknown"),
            "aggregate_reward_score": 1.0
        })
    return results
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [
                    {"role": "user", "content": "test1"},
                    {"role": "assistant", "content": "response1"},
                ],
                "reference_answer": "answer1",
            },
            {
                "id": "sample_2",
                "messages": [
                    {"role": "user", "content": "test2"},
                    {"role": "assistant", "content": "response2"},
                ],
                "reference_answer": "answer2",
            },
        ]

        result = verify_reward_function(reward_function=reward_file, sample_data=sample_data)

        assert result["success"] is True

        log_text = caplog.text
        assert "Sample 0 INPUT:" in log_text
        assert "Sample 0 OUTPUT [PASS]:" in log_text
        assert "Sample 1 INPUT:" in log_text
        assert "Sample 1 OUTPUT [PASS]:" in log_text
        assert "sample_1" in log_text
        assert "sample_2" in log_text
        assert "test1" in log_text
        assert "test2" in log_text

    finally:
        Path(reward_file).unlink()


def test_logging_shows_validation_errors(caplog):
    """Test that logging shows validation errors for failed samples."""
    import logging

    caplog.set_level(logging.INFO)

    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "unknown"),
        })
    return results
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [
                    {"role": "user", "content": "test"},
                    {"role": "assistant", "content": "response"},
                ],
                "reference_answer": "answer",
            },
        ]

        with pytest.raises(ValueError):
            verify_reward_function(reward_function=reward_file, sample_data=sample_data)

        log_text = caplog.text
        assert "Sample 0 OUTPUT" in log_text
        assert "FAIL" in log_text
        assert "Sample 0 validation errors:" in log_text
        assert "aggregate_reward_score - Field required" in log_text

    finally:
        Path(reward_file).unlink()


def test_logging_shows_summary(caplog):
    """Test that logging shows a summary of validation results."""
    import logging

    caplog.set_level(logging.INFO)

    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "unknown"),
            "aggregate_reward_score": 1.0
        })
    return results
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "id": f"sample_{i}",
                "messages": [
                    {"role": "user", "content": f"test{i}"},
                    {"role": "assistant", "content": f"response{i}"},
                ],
                "reference_answer": f"answer{i}",
            }
            for i in range(5)
        ]

        result = verify_reward_function(reward_function=reward_file, sample_data=sample_data)

        assert result["success"] is True

        log_text = caplog.text
        assert "Testing local Python file:" in log_text
        assert "Number of samples: 5" in log_text
        assert "Lambda returned list with 5 result(s)" in log_text
        assert "All 5 sample(s) passed validation" in log_text

    finally:
        Path(reward_file).unlink()


def test_logging_handler_execution_error(caplog):
    """Test that handler execution errors are surfaced clearly."""
    import logging

    caplog.set_level(logging.INFO)

    reward_code = """
def lambda_handler(event, context):
    raise TypeError("RewardOutput.__init__() got an unexpected keyword argument 'aggregate_score'")
"""
    reward_file = _write_reward_fn(reward_code)

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [
                    {"role": "user", "content": "test"},
                    {"role": "assistant", "content": "response"},
                ],
                "reference_answer": "answer",
            },
        ]

        with pytest.raises(ValueError) as exc_info:
            verify_reward_function(reward_function=reward_file, sample_data=sample_data)

        error_msg = str(exc_info.value)
        assert "Handler execution failed" in error_msg
        assert "aggregate_score" in error_msg

    finally:
        Path(reward_file).unlink()
