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


def test_model(endpoint_context_associate_with_model, model_obj, endpoint_action_obj):
    model_list = endpoint_context_associate_with_model.models()
    for model in model_list:
        assert model.source_arn == endpoint_action_obj.action_arn
        assert model.destination_arn == model_obj.context_arn
        assert model.source_type == "ModelDeployment"
        assert model.destination_type == "Model"


def test_dataset_artifacts(static_endpoint_context):
    artifacts_from_query = static_endpoint_context.dataset_artifacts()

    assert len(artifacts_from_query) > 0
    for artifact in artifacts_from_query:
        assert artifact.artifact_type == "DataSet"


def test_training_job_arns(
    static_endpoint_context,
):
    training_job_arns = static_endpoint_context.training_job_arns()

    assert len(training_job_arns) > 0
    for arn in training_job_arns:
        assert "training-job" in arn


def test_pipeline_execution_arn(static_endpoint_context, static_pipeline_execution_arn):
    pipeline_execution_arn = static_endpoint_context.pipeline_execution_arn()

    assert pipeline_execution_arn == static_pipeline_execution_arn
