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
"""Integration tests for verify_reward_function utility.

Tests both local Python reward functions and remote Lambda ARNs
for both OSS (SageMaker Training Job) and Nova platforms.

OSS reward function (oss_reward_fn.py):
    - GSM8k-style scoring using extract_solution / compute_gsm8k_score
    - Returns wrapped response via _ok() helper: {"statusCode": 200, "body": json.dumps(results)}
    - Use is_nova=False so the verifier unwraps the HTTP envelope automatically.

Nova reward function (nova_reward_fn.py):
    - Returns a plain list of {"id": ..., "aggregate_reward_score": ...}
    - Use is_nova=True (default) for direct list handling.

Compute parameter:
    - TrainingJobCompute: For standard SageMaker Training Jobs
    - HyperPodCompute: For SageMaker HyperPod (validates 'SageMaker' in Lambda name)
"""
from __future__ import absolute_import

import os

import pytest

from sagemaker.train.common_utils.rlvr_reward_verifier import verify_reward_function
from sagemaker.core.training.configs import TrainingJobCompute, HyperPodCompute


# ---------------------------------------------------------------------------
# Lambda ARNs are provided by the oss_lambda_arn / nova_lambda_arn fixtures
# (see conftest.py). The fixtures create the reward-function Lambdas on demand
# if they don't already exist, so no hardcoded ARNs are needed here.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Constants: Local reward function file paths (from tests/integ/train/code/)
# ---------------------------------------------------------------------------
OSS_LOCAL_REWARD_FN = os.path.join(
    os.path.dirname(__file__), "code", "oss_reward_fn.py"
)
NOVA_LOCAL_REWARD_FN = os.path.join(
    os.path.dirname(__file__), "code", "nova_reward_fn.py"
)


# ---------------------------------------------------------------------------
# Fixtures: Sample data
# ---------------------------------------------------------------------------
@pytest.fixture
def oss_sample_data():
    """Sample RLVR data for OSS (GSM8k) reward function testing.

    The OSS reward function uses GSM8k-style scoring where reference_answer
    is a numeric ground truth and the assistant response contains '#### <answer>'.
    """
    return [
        {
            "id": "gsm8k_sample_1",
            "messages": [
                {
                    "role": "user",
                    "content": "Natalia sold clips to 48 of her friends in April, "
                    "and then she sold half as many clips in May. How many clips "
                    "did Natalia sell altogether in April and May?",
                },
                {
                    "role": "assistant",
                    "content": "Natalia sold 48/2 = 24 clips in May. "
                    "Natalia sold 48+24 = 72 clips altogether in April and May. "
                    "#### 72",
                },
            ],
            "reference_answer": "72",
        },
        {
            "id": "gsm8k_sample_2",
            "messages": [
                {
                    "role": "user",
                    "content": "If there are 3 cars in the parking lot and 2 more "
                    "cars arrive, how many cars are in the parking lot?",
                },
                {
                    "role": "assistant",
                    "content": "There are 3 cars + 2 cars = 5 cars. #### 5",
                },
            ],
            "reference_answer": "5",
        },
    ]


@pytest.fixture
def nova_sample_data():
    """Sample RLVR data for Nova reward function testing.

    Nova reward function expects reference_answer as a dict and returns
    a plain list of reward outputs.
    """
    return [
        {
            "id": "nova_sample_1",
            "messages": [
                {"role": "user", "content": "Do you have a dedicated security team?"},
                {
                    "role": "assistant",
                    "content": "As an AI developed by Amazon, I do not have a dedicated "
                    "security team in the traditional sense.",
                },
            ],
            "reference_answer": {
                "compliant": "No",
                "explanation": "As an AI developed by Company, I do not have a "
                "traditional security team. However, the company that developed me "
                "has robust security measures in place.",
            },
        },
        {
            "id": "nova_sample_2",
            "messages": [
                {"role": "user", "content": "What programming languages do you support?"},
                {
                    "role": "assistant",
                    "content": "I can assist with many programming languages including "
                    "Python, Java, JavaScript, C++, and more.",
                },
            ],
            "reference_answer": {
                "compliant": "Yes",
                "explanation": "The response correctly lists supported programming "
                "languages without making false claims.",
            },
        },
    ]


# ---------------------------------------------------------------------------
# Test class: Nova Local Reward Function (is_nova=True)
# ---------------------------------------------------------------------------
@pytest.mark.serial
class TestNovaLocalRewardFunction:
    """Integration tests for Nova reward function with local Python file.

    The Nova reward function returns a plain list of dicts. With is_nova=True
    (default), the verifier processes the list directly.
    """

    def test_local_nova_basic(self, nova_sample_data):
        """Test Nova local reward function returns valid output for multiple samples."""
        result = verify_reward_function(
            reward_function=NOVA_LOCAL_REWARD_FN,
            sample_data=nova_sample_data,
            validate_format=True,
            is_nova=True,
        )

        assert result["success"] is True
        assert result["total_samples"] == 2
        assert result["successful_samples"] == 2
        assert len(result["results"]) == 2

        for r in result["results"]:
            assert r["status"] == "success"
            assert "output" in r
            assert "id" in r["output"]
            assert "aggregate_reward_score" in r["output"]
            assert isinstance(r["output"]["aggregate_reward_score"], (int, float))

    def test_local_nova_single_sample(self, nova_sample_data):
        """Test Nova local reward function with a single sample."""
        result = verify_reward_function(
            reward_function=NOVA_LOCAL_REWARD_FN,
            sample_data=[nova_sample_data[0]],
            validate_format=True,
            is_nova=True,
        )

        assert result["success"] is True
        assert result["total_samples"] == 1
        assert result["successful_samples"] == 1

    def test_local_nova_without_format_validation(self, nova_sample_data):
        """Test Nova local reward function with format validation disabled."""
        result = verify_reward_function(
            reward_function=NOVA_LOCAL_REWARD_FN,
            sample_data=nova_sample_data,
            validate_format=False,
            is_nova=True,
        )

        assert result["success"] is True
        assert result["total_samples"] == 2
        assert result["successful_samples"] == 2

    def test_local_nova_dict_reference_answer(self, nova_sample_data):
        """Test Nova reward function handles dict-type reference_answer correctly."""
        result = verify_reward_function(
            reward_function=NOVA_LOCAL_REWARD_FN,
            sample_data=nova_sample_data,
            validate_format=True,
            is_nova=True,
        )

        assert result["success"] is True
        # Dict reference_answer is valid in RLVR schema
        assert result["successful_samples"] == 2

    def test_local_nova_with_training_job_compute(self, nova_sample_data):
        """Test Nova local reward function with TrainingJobCompute specified."""
        result = verify_reward_function(
            reward_function=NOVA_LOCAL_REWARD_FN,
            sample_data=nova_sample_data,
            validate_format=True,
            compute=TrainingJobCompute(instance_type="ml.p4d.24xlarge"),
            is_nova=True,
        )

        assert result["success"] is True
        assert result["total_samples"] == 2

    def test_local_nova_reward_scores_are_bounded(self, nova_sample_data):
        """Test Nova reward function produces scores in [0, 1] range."""
        result = verify_reward_function(
            reward_function=NOVA_LOCAL_REWARD_FN,
            sample_data=nova_sample_data,
            validate_format=True,
            is_nova=True,
        )

        assert result["success"] is True
        for r in result["results"]:
            score = r["output"]["aggregate_reward_score"]
            assert 0.0 <= score <= 1.0, f"Score {score} out of expected [0, 1] range"

    def test_local_nova_default_is_nova_true(self, nova_sample_data):
        """Test that is_nova defaults to True (no need to pass explicitly)."""
        result = verify_reward_function(
            reward_function=NOVA_LOCAL_REWARD_FN,
            sample_data=nova_sample_data,
            validate_format=True,
        )

        assert result["success"] is True
        assert result["successful_samples"] == 2


# ---------------------------------------------------------------------------
# Test class: OSS Local Reward Function (is_nova=False)
# ---------------------------------------------------------------------------
@pytest.mark.serial
class TestOSSLocalRewardFunction:
    """Integration tests for OSS reward function with local Python file.

    The OSS function wraps its output in _ok() (API Gateway format):
    {"statusCode": 200, "headers": {...}, "body": json.dumps(results)}

    With is_nova=False, the verifier unwraps the envelope and validates the
    inner results list directly.
    """

    def test_local_oss_single_sample(self, oss_sample_data):
        """Test OSS local reward function with a single sample."""
        result = verify_reward_function(
            reward_function=OSS_LOCAL_REWARD_FN,
            sample_data=[oss_sample_data[0]],
            validate_format=True,
            is_nova=False,
        )

        assert result["success"] is True
        assert result["total_samples"] == 1
        assert result["successful_samples"] == 1
        assert len(result["results"]) == 1
        assert result["results"][0]["status"] == "success"

    def test_local_oss_multiple_samples(self, oss_sample_data):
        """Test OSS local reward function with multiple samples after unwrapping."""
        result = verify_reward_function(
            reward_function=OSS_LOCAL_REWARD_FN,
            sample_data=oss_sample_data,
            validate_format=True,
            is_nova=False,
        )

        assert result["success"] is True
        assert result["total_samples"] == 2
        assert result["successful_samples"] == 2

        for r in result["results"]:
            assert r["status"] == "success"
            assert "output" in r
            assert "id" in r["output"]
            assert "aggregate_reward_score" in r["output"]

    def test_local_oss_single_sample_without_format_validation(self, oss_sample_data):
        """Test OSS local reward function single sample with format validation disabled."""
        result = verify_reward_function(
            reward_function=OSS_LOCAL_REWARD_FN,
            sample_data=[oss_sample_data[0]],
            validate_format=False,
            is_nova=False,
        )

        assert result["success"] is True
        assert result["total_samples"] == 1
        assert result["successful_samples"] == 1

    def test_local_oss_is_nova_true_fails_with_wrapped_format(self, oss_sample_data):
        """Test that using is_nova=True with OSS function fails format validation.

        When is_nova=True, the verifier does not unwrap the HTTP envelope,
        so it sees a single dict (not a list) and output validation fails.
        """
        with pytest.raises(ValueError, match="verification failed"):
            verify_reward_function(
                reward_function=OSS_LOCAL_REWARD_FN,
                sample_data=oss_sample_data,
                validate_format=True,
                is_nova=True,
            )

    def test_local_oss_file_not_found(self, oss_sample_data):
        """Test error handling when local file doesn't exist."""
        with pytest.raises(ValueError, match="verification failed"):
            verify_reward_function(
                reward_function="/nonexistent/path/reward.py",
                sample_data=[oss_sample_data[0]],
                validate_format=True,
                is_nova=False,
            )

    def test_local_oss_with_training_job_compute(self, oss_sample_data):
        """Test OSS local reward function with TrainingJobCompute (optional for local)."""
        result = verify_reward_function(
            reward_function=OSS_LOCAL_REWARD_FN,
            sample_data=[oss_sample_data[0]],
            validate_format=True,
            compute=TrainingJobCompute(instance_type="ml.p4d.24xlarge"),
            is_nova=False,
        )

        assert result["success"] is True
        assert result["total_samples"] == 1


# ---------------------------------------------------------------------------
# Test class: Nova Remote Lambda (is_nova=True)
# ---------------------------------------------------------------------------
@pytest.mark.serial
class TestNovaRemoteLambda:
    """Integration tests for Nova reward function via remote Lambda ARN.

    Requires AWS credentials with permission to invoke the Lambda.
    """

    def test_lambda_nova_training_job_compute_basic(self, nova_sample_data, nova_lambda_arn):
        """Test Nova Lambda ARN with TrainingJobCompute and multiple samples."""
        result = verify_reward_function(
            reward_function=nova_lambda_arn,
            sample_data=nova_sample_data,
            validate_format=True,
            compute=TrainingJobCompute(instance_type="ml.p4d.24xlarge"),
            is_nova=True,
        )

        assert result["success"] is True
        assert result["total_samples"] == 2
        assert result["successful_samples"] == 2

        for r in result["results"]:
            assert r["status"] == "success"
            assert "output" in r
            assert "id" in r["output"]
            assert "aggregate_reward_score" in r["output"]
            assert isinstance(r["output"]["aggregate_reward_score"], (int, float))

    def test_lambda_nova_training_job_compute_single_sample(self, nova_sample_data, nova_lambda_arn):
        """Test Nova Lambda ARN with a single sample."""
        result = verify_reward_function(
            reward_function=nova_lambda_arn,
            sample_data=[nova_sample_data[0]],
            validate_format=True,
            compute=TrainingJobCompute(instance_type="ml.p4d.24xlarge"),
            is_nova=True,
        )

        assert result["success"] is True
        assert result["total_samples"] == 1
        assert result["successful_samples"] == 1

    def test_lambda_nova_without_format_validation(self, nova_sample_data, nova_lambda_arn):
        """Test Nova Lambda ARN with format validation disabled."""
        result = verify_reward_function(
            reward_function=nova_lambda_arn,
            sample_data=nova_sample_data,
            validate_format=False,
            compute=TrainingJobCompute(instance_type="ml.p4d.24xlarge"),
            is_nova=True,
        )

        assert result["success"] is True
        assert result["total_samples"] == 2
        assert result["successful_samples"] == 2

    def test_lambda_nova_requires_compute(self, nova_sample_data, nova_lambda_arn):
        """Test that Nova Lambda ARN requires compute parameter."""
        with pytest.raises(ValueError, match="compute.*required"):
            verify_reward_function(
                reward_function=nova_lambda_arn,
                sample_data=nova_sample_data,
                validate_format=True,
                is_nova=True,
            )


# ---------------------------------------------------------------------------
# Test class: OSS Remote Lambda (is_nova=False)
# ---------------------------------------------------------------------------
@pytest.mark.serial
class TestOSSRemoteLambda:
    """Integration tests for OSS reward function via remote Lambda ARN.

    With is_nova=False, the verifier unwraps the HTTP envelope from the Lambda
    response and validates the inner results list.
    Requires AWS credentials with permission to invoke the Lambda.
    """

    def test_lambda_oss_training_job_compute_single_sample(self, oss_sample_data, oss_lambda_arn):
        """Test OSS Lambda ARN with TrainingJobCompute and single sample."""
        result = verify_reward_function(
            reward_function=oss_lambda_arn,
            sample_data=[oss_sample_data[0]],
            validate_format=True,
            compute=TrainingJobCompute(instance_type="ml.p4d.24xlarge"),
            is_nova=False,
        )

        assert result["success"] is True
        assert result["total_samples"] == 1
        assert result["successful_samples"] == 1

    def test_lambda_oss_training_job_compute_multiple_samples(self, oss_sample_data, oss_lambda_arn):
        """Test OSS Lambda ARN with TrainingJobCompute and multiple samples."""
        result = verify_reward_function(
            reward_function=oss_lambda_arn,
            sample_data=oss_sample_data,
            validate_format=True,
            compute=TrainingJobCompute(instance_type="ml.p4d.24xlarge"),
            is_nova=False,
        )

        assert result["success"] is True
        assert result["total_samples"] == 2
        assert result["successful_samples"] == 2

    def test_lambda_oss_training_job_compute_without_format_validation(
        self, oss_sample_data, oss_lambda_arn
    ):
        """Test OSS Lambda ARN single sample with format validation disabled."""
        result = verify_reward_function(
            reward_function=oss_lambda_arn,
            sample_data=[oss_sample_data[0]],
            validate_format=False,
            compute=TrainingJobCompute(instance_type="ml.p4d.24xlarge"),
            is_nova=False,
        )

        assert result["success"] is True
        assert result["total_samples"] == 1
        assert result["successful_samples"] == 1

    def test_lambda_oss_is_nova_true_fails_with_wrapped_format(
        self, oss_sample_data, oss_lambda_arn
    ):
        """Test that using is_nova=True with OSS Lambda fails validation."""
        with pytest.raises(ValueError, match="verification failed"):
            verify_reward_function(
                reward_function=oss_lambda_arn,
                sample_data=oss_sample_data,
                validate_format=True,
                compute=TrainingJobCompute(instance_type="ml.p4d.24xlarge"),
                is_nova=True,
            )

    def test_lambda_oss_does_not_require_compute(self, oss_sample_data, oss_lambda_arn):
        """Test that OSS Lambda ARN does not require compute since is_nova is False."""
        # Should not raise - compute is only required for Nova models
        verify_reward_function(
            reward_function=oss_lambda_arn,
            sample_data=[oss_sample_data[0]],
            validate_format=True,
            is_nova=False,
        )


# ---------------------------------------------------------------------------
# Test class: Error Handling
# ---------------------------------------------------------------------------
@pytest.mark.serial
class TestErrorHandling:
    """Tests for error handling edge cases."""

    def test_file_not_found_raises_error(self, nova_sample_data):
        """Test that missing local file raises verification error."""
        with pytest.raises(ValueError, match="verification failed"):
            verify_reward_function(
                reward_function="/path/to/nonexistent/reward.py",
                sample_data=nova_sample_data,
                validate_format=True,
                is_nova=True,
            )

    def test_invalid_lambda_arn_format(self, nova_sample_data):
        """Test that invalid Lambda ARN format is treated as local file path."""
        # ARN that doesn't match Lambda pattern is treated as a file path
        with pytest.raises(ValueError, match="verification failed"):
            verify_reward_function(
                reward_function="arn:aws:invalid:format",
                sample_data=nova_sample_data,
                validate_format=True,
                is_nova=True,
            )

    def test_lambda_arn_without_compute_raises_error(self, nova_sample_data, nova_lambda_arn):
        """Test that Lambda ARN without compute raises ValueError."""
        with pytest.raises(ValueError, match="compute.*required"):
            verify_reward_function(
                reward_function=nova_lambda_arn,
                sample_data=nova_sample_data,
                validate_format=True,
                compute=None,
                is_nova=True,
            )

    def test_hyperpod_compute_requires_sagemaker_in_function_name(
        self, nova_sample_data, nova_lambda_arn
    ):
        """Test HyperPodCompute validation requires 'SageMaker' in Lambda function name."""
        # Nova Lambda doesn't have 'SageMaker' in name
        with pytest.raises(ValueError, match="SageMaker"):
            verify_reward_function(
                reward_function=nova_lambda_arn,
                sample_data=nova_sample_data,
                validate_format=True,
                compute=HyperPodCompute(cluster_name="my-cluster"),
                is_nova=True,
            )

    def test_oss_non_200_status_code_raises_error(self, oss_sample_data):
        """Test that OSS response with non-200 status code raises verification error."""
        oss_error_fn = os.path.join(os.path.dirname(__file__), "code", "oss_reward_fn_error.py")

        with pytest.raises(ValueError, match="verification failed"):
            verify_reward_function(
                reward_function=oss_error_fn,
                sample_data=[oss_sample_data[0]],
                validate_format=True,
                is_nova=False,
            )

    def test_nova_none_response_raises_error(self, nova_sample_data):
        """Test that Nova reward function returning None raises verification error."""
        nova_none_fn = os.path.join(os.path.dirname(__file__), "code", "nova_reward_fn_none.py")

        with pytest.raises(ValueError, match="verification failed"):
            verify_reward_function(
                reward_function=nova_none_fn,
                sample_data=nova_sample_data,
                validate_format=True,
                is_nova=True,
            )
