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
"""Under a ``PipelineSession``, ``DataPreparer`` must capture rather than submit.

The producer half of composing a ``JobStep`` over ``CreateJob`` for data
preparation: declare a caller name the resolver recognises, assemble the request
``Job.create`` would have assembled, and hand it to the session instead of the
service. Mirrors the MTRL capture suite; ``_intercept_create_request`` is
deliberately NOT mocked, so assertions run against the ``_JobStepArguments`` the
session really builds.
"""
from __future__ import absolute_import

import json
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.ai_registry.data_preparer.configs import DatasetJobConfig
from sagemaker.ai_registry.data_preparer.constants import JobCategory
from sagemaker.ai_registry.data_preparer.defaults import DataPreparerDefaults
from sagemaker.ai_registry.data_preparer.preparer import DataPreparer
from sagemaker.core.workflow.pipeline_context import (
    PipelineSession,
    _DECLARABLE_CALLER_NAMES,
    _JobStepArguments,
    _StepArguments,
    retrieve_caller_name,
)
from sagemaker.core.workflow.utilities import execute_job_functions

ROLE_ARN = "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole"
ACCOUNT_DETAILS = {"AccountId": "123456789012"}
SCHEMA_VERSION = "1.0"
DOCUMENT = {
    "InputDataConfig": {"DatasetArn": "arn:aws:sagemaker:us-west-2:123456789012:dataset/d1"},
    "OutputDataConfig": {"OutputS3Uri": "s3://bucket/dq-output/"},
}
PINNED_JOB_NAME = "dq-job-pinned"


def _resolved_config(category=JobCategory.DATA_QUALITY_EVALUATION):
    """A ``DatasetJobConfig`` in the state ``_resolve_and_validate`` leaves it in.

    The resolution walk needs the live schema registry, so the post-conditions the
    capture path reads are set directly: the validated payload, its serialized
    form, and the resolved schema version.
    """
    config = DatasetJobConfig(job_category=category)
    config._validated_payload = dict(DOCUMENT)
    config._serialized_document = json.dumps(DOCUMENT, ensure_ascii=False)
    config._resolved_schema_version = SCHEMA_VERSION
    return config


def _pipeline_session():
    """A real ``PipelineSession`` with only its outbound edges stubbed."""
    session = PipelineSession()
    session.sagemaker_client = MagicMock()
    session.sagemaker_config = {}
    return session


def _preparer(session):
    return DataPreparer(sagemaker_session=session, role=ROLE_ARN)


def _static_defaults():
    """Pin the definition-time resolutions that would otherwise call AWS."""
    return (
        patch.object(
            DataPreparerDefaults, "get_sagemaker_session", side_effect=lambda s: s
        ),
        patch.object(
            DataPreparerDefaults, "get_customer_details", return_value=dict(ACCOUNT_DETAILS)
        ),
        patch.object(
            DataPreparerDefaults, "resolve_job_name", return_value=PINNED_JOB_NAME
        ),
        patch.object(DatasetJobConfig, "_resolve_and_validate"),
    )


def _capture(session, tags=None):
    """Drive the decorator through the public method, then replay as a step would."""
    preparer = _preparer(session)
    config = _resolved_config()
    patches = _static_defaults()
    for p in patches:
        p.start()
    try:
        step_args = preparer.evaluate(config, tags=tags)
        assert isinstance(step_args, _StepArguments)
        execute_job_functions(step_args)
    finally:
        for p in patches:
            p.stop()
    return session.context


class TestDeclaredCallerName:
    def test_declares_the_create_job_caller_name(self):
        """``JobStep``'s ``expected_caller={"create_job"}`` guard accepts only this value."""
        assert DataPreparer._pipeline_caller_name == "create_job"
        assert DataPreparer._pipeline_caller_name in _DECLARABLE_CALLER_NAMES

    def test_resolver_maps_the_instance_to_create_job(self):
        assert retrieve_caller_name(_preparer(None)) == "create_job"


class TestPipelineCapture:
    def test_capture_produces_job_step_arguments(self):
        session = _pipeline_session()
        context = _capture(session)
        assert isinstance(context, _JobStepArguments)
        assert context.caller_name == "create_job"

    def test_captured_request_is_the_direct_path_envelope_minus_job_arn(self):
        """Same envelope ``_create_job`` assembles, except the derivable ``JobArn``.

        ``JobArn`` embeds the job name, which the pipeline path replaces per
        execution, so a definition-time ARN would be stale and is omitted.
        """
        session = _pipeline_session()
        request = _capture(session).args
        assert request["JobName"] == PINNED_JOB_NAME
        assert request["RoleArn"] == ROLE_ARN
        assert request["JobCategory"] == JobCategory.DATA_QUALITY_EVALUATION.value
        assert request["JobConfigSchemaVersion"] == SCHEMA_VERSION
        assert request["CustomerDetails"] == ACCOUNT_DETAILS
        assert "JobArn" not in request

    def test_captured_document_matches_what_the_direct_path_serializes(self):
        """``JobStep`` encodes the dict at definition time; it must decode to the
        exact string ``to_job_config_document`` would have submitted."""
        session = _pipeline_session()
        request = _capture(session).args
        assert request["JobConfigDocument"] == DOCUMENT

    def test_dict_form_tags_reach_the_wire_form(self):
        """Dict-form tags are coerced through ``Tag`` so ``serialize`` emits the
        same ``{"Key", "Value"}`` wire form as the ``Job.create`` path."""
        session = _pipeline_session()
        request = _capture(
            session, tags=[{"key": "sagemaker:project-id", "value": "proj-123"}]
        ).args
        assert request["Tags"] == [{"Key": "sagemaker:project-id", "Value": "proj-123"}]

    def test_no_job_is_created_under_a_pipeline_session(self):
        session = _pipeline_session()
        with patch(
            "sagemaker.ai_registry.data_preparer.preparer._get_job_resource"
        ) as job_resource:
            _capture(session)
        job_resource.assert_not_called()


class TestDirectPathUnchanged:
    def test_direct_session_still_submits(self):
        """A non-pipeline session must take the ``Job.create`` path, not capture."""
        session = MagicMock()
        session.sagemaker_config = {}
        preparer = _preparer(session)
        config = _resolved_config()
        job = MagicMock()
        job.job_name = PINNED_JOB_NAME
        job.job_arn = "arn:aws:sagemaker:us-west-2:123456789012:job/dq/" + PINNED_JOB_NAME
        patches = _static_defaults()
        for p in patches:
            p.start()
        try:
            with patch.object(DataPreparer, "_create_job", return_value=job) as create:
                result = preparer.evaluate(config, wait=False)
        finally:
            for p in patches:
                p.stop()
        create.assert_called_once()
        assert result is job
