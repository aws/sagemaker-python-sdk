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
from __future__ import absolute_import

import unittest.mock

import pytest
from sagemaker.lineage import artifact, _api_types


@pytest.fixture
def sagemaker_session():
    return unittest.mock.Mock()


def test_trained_models(sagemaker_session):
    # dataset artifact ---- tc --- artifact model
    dataset_artifact_obj = artifact.DatasetArtifact(
        sagemaker_session,
        artifact_arn="dataset-artifact-arn",
        artifact_name="dataset-artifact-name",
    )
    sagemaker_session.sagemaker_client.list_associations.side_effect = [
        {
            "AssociationSummaries": [
                {
                    "SourceArn": dataset_artifact_obj.artifact_arn,
                    "SourceName": "X1",
                    "DestinationArn": "experiment-trial-component",
                    "DestinationName": "Y1",
                    "SourceType": "C1",
                    "DestinationType": "ModelDeployment",
                    "AssociationType": "E1",
                    "CreationTime": None,
                    "CreatedBy": None,
                }
            ],
        },
        {
            "AssociationSummaries": [
                {
                    "SourceArn": "experiment-trial-component",
                    "SourceName": "X2",
                    "DestinationArn": "B2",
                    "DestinationName": "Y2",
                    "SourceType": "C2",
                    "DestinationType": "Context",
                    "AssociationType": "E2",
                    "CreationTime": None,
                    "CreatedBy": None,
                }
            ]
        },
    ]

    model_list = dataset_artifact_obj.trained_models()
    expected_calls = [
        unittest.mock.call(SourceArn=dataset_artifact_obj.artifact_arn),
        unittest.mock.call(SourceArn="experiment-trial-component", DestinationType="Context"),
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.list_associations.mock_calls
    expected_model_list = [
        _api_types.AssociationSummary(
            source_arn="experiment-trial-component",
            source_name="X2",
            destination_arn="B2",
            destination_name="Y2",
            source_type="C2",
            destination_type="Context",
            association_type="E2",
            creation_time=None,
            created_by=None,
        )
    ]
    assert expected_model_list == model_list
