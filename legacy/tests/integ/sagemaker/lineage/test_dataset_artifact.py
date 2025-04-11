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
"""This module contains code to test SageMaker ``DatasetArtifact``"""
from __future__ import absolute_import

import pytest


def test_trained_models(
    dataset_artifact_associated_models,
    trial_component_obj,
    model_artifact_obj1,
):

    model_list = dataset_artifact_associated_models.trained_models()
    for model in model_list:
        assert model.source_arn == trial_component_obj.trial_component_arn
        assert model.destination_arn == model_artifact_obj1.artifact_arn
        assert model.destination_type == "Context"


@pytest.mark.skip("data inconsistency P61661075")
def test_endpoint_contexts(
    static_dataset_artifact,
):
    contexts_from_query = static_dataset_artifact.endpoint_contexts()

    assert len(contexts_from_query) > 0
    for context in contexts_from_query:
        assert context.context_type == "Endpoint"


@pytest.mark.skip("data inconsistency P61661075")
def test_get_upstream_datasets(static_dataset_artifact, sagemaker_session):
    artifacts_from_query = static_dataset_artifact.upstream_datasets()
    assert len(artifacts_from_query) > 0
    for artifact in artifacts_from_query:
        assert artifact.artifact_type == "DataSet"
        assert "artifact" in artifact.artifact_arn


@pytest.mark.skip("data inconsistency P61661075")
def test_get_down_datasets(static_dataset_artifact, sagemaker_session):
    artifacts_from_query = static_dataset_artifact.downstream_datasets()
    assert len(artifacts_from_query) > 0
    for artifact in artifacts_from_query:
        assert artifact.artifact_type == "DataSet"
        assert "artifact" in artifact.artifact_arn
