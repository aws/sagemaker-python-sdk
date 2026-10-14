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
"""Unit tests for workflow model_step."""

from __future__ import absolute_import

from unittest.mock import patch


def test_model_step_properties():
    """Test ModelStep has properties"""
    from sagemaker.mlops.workflow.model_step import ModelStep

    step_args = {"ModelName": "test-model"}

    with patch("sagemaker.core.workflow.utilities.validate_step_args_input"):
        step = ModelStep(name="model-step", step_args=step_args)
        assert step.name == "model-step"
        assert hasattr(step, "properties")


def _create_model_step(retry_policies):
    from sagemaker.mlops.workflow.model_step import ModelStep

    with patch("sagemaker.core.workflow.utilities.validate_step_args_input"):
        return ModelStep(
            name="model-step",
            step_args={"ModelName": "test-model"},
            retry_policies=retry_policies,
        )


def test_model_step_list_retry_policies():
    """A list of retry policies applies to the model step and the repack step."""
    from sagemaker.mlops.workflow.retry import StepRetryPolicy, StepExceptionTypeEnum

    policy = StepRetryPolicy(exception_types=[StepExceptionTypeEnum.THROTTLING], max_attempts=3)

    step = _create_model_step([policy])

    assert step.to_request()["RetryPolicies"] == [policy.to_request()]
    assert step._repack_model_retry_policies == [policy]


def test_model_step_dict_retry_policies():
    """A dict of retry policies is split between the model step and the repack step."""
    from sagemaker.mlops.workflow.retry import StepRetryPolicy, StepExceptionTypeEnum

    create_policy = StepRetryPolicy(
        exception_types=[StepExceptionTypeEnum.THROTTLING], max_attempts=3
    )
    repack_policy = StepRetryPolicy(
        exception_types=[StepExceptionTypeEnum.SERVICE_FAULT], max_attempts=5
    )

    step = _create_model_step(
        {
            "create_model_retry_policies": [create_policy],
            "repack_model_retry_policies": [repack_policy],
        }
    )

    assert step.to_request()["RetryPolicies"] == [create_policy.to_request()]
    assert step._repack_model_retry_policies == [repack_policy]


def test_model_step_dict_retry_policies_rejects_sagemaker_job_policy():
    """SageMakerJobStepRetryPolicy is rejected for the model step in dict form too."""
    import pytest
    from sagemaker.mlops.workflow.retry import (
        SageMakerJobStepRetryPolicy,
        SageMakerJobExceptionTypeEnum,
    )

    policy = SageMakerJobStepRetryPolicy(
        exception_types=[SageMakerJobExceptionTypeEnum.INTERNAL_ERROR], max_attempts=3
    )

    with pytest.raises(ValueError, match="SageMakerJobStepRetryPolicy is not allowed"):
        _create_model_step({"create_model_retry_policies": [policy]})
