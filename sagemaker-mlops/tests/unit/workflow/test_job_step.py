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
"""Unit tests for `StepTypeEnum.JOB` and `JobStep`."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock

from sagemaker.core.workflow.pipeline_context import _JobStepArguments, _StepArguments
from sagemaker.mlops.workflow.retry import (
    SageMakerJobExceptionTypeEnum,
    SageMakerJobStepRetryPolicy,
)
from sagemaker.mlops.workflow.steps import CacheConfig, ConfigurableRetryStep, JobStep, StepTypeEnum

# --- StepTypeEnum membership -------------------------------------------------
#
# The member set is pinned as a whole. A new step type must be added here in the
# same change that adds it to the enum, so an accidental addition or removal
# fails rather than passing silently. This tree does not carry PR #6224, which
# would add ENDPOINT_CONFIG, ENDPOINT, INFERENCE_COMPONENT and LINEAGE.
EXPECTED_STEP_TYPES = {
    "CONDITION": "Condition",
    "CREATE_MODEL": "Model",
    "PROCESSING": "Processing",
    "REGISTER_MODEL": "RegisterModel",
    "TRAINING": "Training",
    "TRANSFORM": "Transform",
    "CALLBACK": "Callback",
    "TUNING": "Tuning",
    "LAMBDA": "Lambda",
    "QUALITY_CHECK": "QualityCheck",
    "CLARIFY_CHECK": "ClarifyCheck",
    "EMR": "EMR",
    "EMR_SERVERLESS": "EMRServerless",
    "FAIL": "Fail",
    "AUTOML": "AutoML",
    "JOB": "Job",
}


def test_step_type_enum_member_set_is_exact():
    assert {member.name: member.value for member in StepTypeEnum} == EXPECTED_STEP_TYPES


def test_step_type_enum_has_job_member():
    assert StepTypeEnum.JOB.value == "Job"
    # The value must match the service's own name for the step, which is how a
    # definition round-trips through StepTypeEnum(request_dict["Type"]).
    assert StepTypeEnum("Job") is StepTypeEnum.JOB


# --- JobStep construction ---------------------------------------------------


def test_job_step_is_configurable_retry_step():
    # Retryable like TrainingStep, not a plain Step: CreateJob is a synchronous
    # create that can hit a resource limit.
    step = JobStep(name="my-job")
    assert isinstance(step, ConfigurableRetryStep)
    assert step.step_type is StepTypeEnum.JOB
    assert step.name == "my-job"


def test_job_step_accepts_sagemaker_job_step_retry_policy():
    policy = SageMakerJobStepRetryPolicy(
        exception_types=[SageMakerJobExceptionTypeEnum.RESOURCE_LIMIT],
        max_attempts=2,
    )
    step = JobStep(name="my-job", retry_policies=[policy])
    assert step.retry_policies == [policy]
    step.add_retry_policy(policy)
    assert len(step.retry_policies) == 2


def test_job_step_optional_metadata():
    step = JobStep(
        name="my-job",
        display_name="My Job",
        description="a job step",
        depends_on=["upstream"],
    )
    assert step.display_name == "My Job"
    assert step.description == "a job step"
    assert step.depends_on == ["upstream"]


# --- properties -------------------------------------------------------------


def test_job_step_properties_expose_describe_job_response_members():
    step = JobStep(name="my-job")
    properties = step.properties

    # Walked from the botocore DescribeJobResponse shape, not hand-listed.
    for member in ("JobName", "JobArn", "JobCategory", "JobStatus", "JobConfigDocument"):
        assert hasattr(properties, member), member

    assert properties.JobName.expr == {"Get": "Steps.my-job.JobName"}


def test_job_config_document_is_a_string_property():
    # JobConfigDocument is a JSON string in the describe response, so it has no
    # modeled sub-members. A reference descending into it is resolved by the
    # pipeline service at execution time, never type-checked here.
    from sagemaker.core.workflow.properties import Properties

    step = JobStep(name="my-job")
    document = step.properties.JobConfigDocument

    assert document.expr == {"Get": "Steps.my-job.JobConfigDocument"}
    assert [key for key, value in document.__dict__.items() if isinstance(value, Properties)] == []

    # Contrast: a modeled structure member does get walked.
    assert isinstance(step.properties.SecondaryStatusTransitions, Properties)


def test_job_step_properties_reference_the_step_instance():
    step = JobStep(name="my-job")
    assert step.properties._referenced_steps == [step]


# --- step_args validation ---------------------------------------------------


def test_job_step_rejects_non_step_args():
    with pytest.raises(TypeError, match="must be obtained from a producer"):
        JobStep(name="my-job", step_args={"JobName": "not-step-args"})


@pytest.mark.parametrize("caller_name", ["train", "transform", "run", "tune", "create_model"])
def test_job_step_rejects_other_producers(caller_name):
    # Every currently capturable producer builds a different create request.
    step_args = _JobStepArguments(caller_name, {"JobName": "j"})
    with pytest.raises(ValueError, match="must be obtained from a producer"):
        JobStep(name="my-job", step_args=step_args)


def test_job_step_accepts_create_job_caller():
    step_args = _JobStepArguments("create_job", {"JobName": "j"})
    step = JobStep(name="my-job", step_args=step_args)
    assert step.step_args is step_args


# --- arguments --------------------------------------------------------------


def test_job_step_arguments_requires_step_args():
    step = JobStep(name="my-job", step_args=None)
    with pytest.raises(ValueError, match="step_args input is required"):
        _ = step.arguments


def _capturing_step_args(request):
    """Build step_args whose func writes `request` into the session context.

    This stands in for `@runnable_by_pipeline` + `_intercept_create_request`,
    which no CreateJob producer implements yet.
    """
    producer = Mock()
    producer.sagemaker_session.context.args = request

    def capture(_producer):
        return None

    return _StepArguments("create_job", capture, producer)


def test_job_step_arguments_trims_job_name_by_default():
    request = {
        "JobName": "my-job-2026-09-10-22-31-57-000",
        "RoleArn": "arn:aws:iam::123456789012:role/JobRole",
        "JobCategory": "SyntheticDataGeneration",
        "JobConfigSchemaVersion": "1.0",
        "JobConfigDocument": '{"Foo": "bar"}',
    }
    step = JobStep(name="my-job", step_args=_capturing_step_args(request))

    arguments = step.arguments

    # No custom job prefix opted in, so the generated name is dropped from the
    # persisted definition.
    assert "JobName" not in arguments
    assert arguments["JobCategory"] == "SyntheticDataGeneration"
    assert arguments["JobConfigDocument"] == '{"Foo": "bar"}'


def test_job_step_to_request_shape():
    request = {"JobName": "my-job", "JobCategory": "DataQualityEvaluation"}
    step = JobStep(
        name="my-job",
        step_args=_capturing_step_args(request),
        display_name="My Job",
        description="a job step",
        depends_on=["upstream"],
    )

    step_request = step.to_request()

    assert step_request["Name"] == "my-job"
    assert step_request["Type"] == "Job"
    assert step_request["DependsOn"] == ["upstream"]
    assert step_request["DisplayName"] == "My Job"
    assert step_request["Description"] == "a job step"
    assert "CacheConfig" not in step_request


def test_job_step_to_request_includes_cache_config():
    request = {"JobName": "my-job"}
    step = JobStep(
        name="my-job",
        step_args=_capturing_step_args(request),
        cache_config=CacheConfig(enable_caching=True, expire_after="P30D"),
    )

    step_request = step.to_request()

    assert step_request["CacheConfig"] == {"Enabled": True, "ExpireAfter": "P30D"}


def test_job_step_retry_policies_land_in_request():
    request = {"JobName": "my-job"}
    policy = SageMakerJobStepRetryPolicy(
        exception_types=[SageMakerJobExceptionTypeEnum.RESOURCE_LIMIT],
        max_attempts=3,
    )
    step = JobStep(
        name="my-job",
        step_args=_capturing_step_args(request),
        retry_policies=[policy],
    )

    step_request = step.to_request()

    assert step_request["RetryPolicies"] == [policy.to_request()]


# --- exports ----------------------------------------------------------------


def test_job_step_is_exported_from_workflow_package():
    import sagemaker.mlops.workflow as workflow

    assert workflow.JobStep is JobStep
    assert "JobStep" in workflow.__all__
