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
from sagemaker.lineage import artifact
from sagemaker.lineage.query import LineageQueryDirectionEnum


@pytest.fixture
def sagemaker_session():
    return unittest.mock.Mock()


def test_datasets(sagemaker_session):
    artifact_arn = (
        "arn:aws:sagemaker:us-west-2:123456789012:artifact/lineage-unit-3b05f017-0d87-4c37"
    )
    artifact_dataset_arn = "arn:aws:sagemaker:us-west-2:123456789012:artifact/datasets"
    artifact_dataset_name = "myDataset"

    obj = artifact.ImageArtifact(sagemaker_session, artifact_name="foo", artifact_arn=artifact_arn)

    sagemaker_session.sagemaker_client.query_lineage.return_value = {
        "Vertices": [
            {"Arn": artifact_dataset_arn, "Type": "DataSet", "LineageType": "Artifact"},
        ],
        "Edges": [{"SourceArn": "arn1", "DestinationArn": "arn2", "AssociationType": "Produced"}],
    }
    sagemaker_session.sagemaker_client.describe_artifact.return_value = {
        "ArtifactName": artifact_dataset_name,
        "ArtifactArn": artifact_dataset_arn,
    }

    dataset_list = obj.datasets(direction=LineageQueryDirectionEnum.DESCENDANTS)
    expected_calls = [
        unittest.mock.call(
            Direction="Descendants",
            Filters={"Types": ["DataSet"], "LineageTypes": ["Artifact"]},
            IncludeEdges=False,
            MaxDepth=10,
            StartArns=[artifact_arn],
        ),
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.query_lineage.mock_calls
    expected_dataset_list = [
        artifact.DatasetArtifact(
            artifact_name=artifact_dataset_name,
            artifact_arn=artifact_dataset_arn,
        )
    ]
    assert expected_dataset_list[0].artifact_arn == dataset_list[0].artifact_arn
    assert expected_dataset_list[0].artifact_name == dataset_list[0].artifact_name
