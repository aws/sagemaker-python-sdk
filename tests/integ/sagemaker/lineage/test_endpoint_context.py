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
"""This module contains code to test SageMaker ``Contexts``"""
from __future__ import absolute_import
import time

import pytest

SLEEP_TIME_ONE_SECONDS = 1
SLEEP_TIME_THREE_SECONDS = 3


@pytest.mark.skip("recurring failures due to existing ARN V739948996")
def test_model(endpoint_context_associate_with_model, model_obj, endpoint_action_obj):
    model_list = endpoint_context_associate_with_model.models()
    for model in model_list:
        assert model.source_arn == endpoint_action_obj.action_arn
        assert model.destination_arn == model_obj.artifact_arn
        assert model.source_type == "ModelDeployment"
        assert model.destination_type == "Model"


@pytest.mark.skip("recurring failures due to existing ARN V739948996")
def test_model_v2(endpoint_context_associate_with_model, model_obj, sagemaker_session):
    time.sleep(SLEEP_TIME_ONE_SECONDS)
    model_list = endpoint_context_associate_with_model.models_v2()
    assert len(model_list) == 1
    for model in model_list:
        assert model.artifact_arn == model_obj.artifact_arn
        assert model.artifact_name == model_obj.artifact_name
        assert model.artifact_type == "Model"
        assert model.properties == model_obj.properties


@pytest.mark.skip("data inconsistency P61661075")
def test_dataset_artifacts(static_endpoint_context):
    artifacts_from_query = static_endpoint_context.dataset_artifacts()

    assert len(artifacts_from_query) > 0
    for artifact in artifacts_from_query:
        assert artifact.artifact_type == "DataSet"


@pytest.mark.skip("data inconsistency P61661075")
def test_training_job_arns(
    static_endpoint_context,
):
    training_job_arns = static_endpoint_context.training_job_arns()

    assert len(training_job_arns) > 0
    for arn in training_job_arns:
        assert "training-job" in arn


@pytest.mark.skip("data inconsistency P61661075")
def test_pipeline_execution_arn(static_endpoint_context, static_pipeline_execution_arn):
    pipeline_execution_arn = static_endpoint_context.pipeline_execution_arn()

    assert pipeline_execution_arn == static_pipeline_execution_arn


@pytest.mark.skip("data inconsistency P61661075")
def test_transform_jobs(
    sagemaker_session, static_transform_job_trial_component, static_endpoint_context
):
    sagemaker_session.sagemaker_client.add_association(
        SourceArn=static_transform_job_trial_component.trial_component_arn,
        DestinationArn=static_endpoint_context.context_arn,
        AssociationType="ContributedTo",
    )
    time.sleep(SLEEP_TIME_THREE_SECONDS)
    transform_jobs_from_query = static_endpoint_context.transform_jobs()

    assert len(transform_jobs_from_query) > 0
    for transform_job in transform_jobs_from_query:
        assert "transform-job" in transform_job.trial_component_arn
        assert "TransformJob" in transform_job.source.get("SourceType")

    sagemaker_session.sagemaker_client.delete_association(
        SourceArn=static_transform_job_trial_component.trial_component_arn,
        DestinationArn=static_endpoint_context.context_arn,
    )


@pytest.mark.skip("data inconsistency P61661075")
def test_processing_jobs(
    sagemaker_session, static_transform_job_trial_component, static_endpoint_context
):
    processing_jobs_from_query = static_endpoint_context.processing_jobs()
    assert len(processing_jobs_from_query) > 0
    for processing_job in processing_jobs_from_query:
        assert "processing-job" in processing_job.trial_component_arn
        assert "ProcessingJob" in processing_job.source.get("SourceType")


@pytest.mark.skip("data inconsistency P61661075")
def test_trial_components(
    sagemaker_session, static_transform_job_trial_component, static_endpoint_context
):
    trial_components_from_query = static_endpoint_context.trial_components()

    assert len(trial_components_from_query) > 0
    for trial_component in trial_components_from_query:
        assert "job" in trial_component.trial_component_arn
        assert "Job" in trial_component.source.get("SourceType")
