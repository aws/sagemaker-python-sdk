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

from sagemaker.lineage import context, _api_types
from sagemaker.lineage._api_types import ArtifactSource
from sagemaker.lineage.artifact import ModelArtifact


def test_models(sagemaker_session):
    obj = context.EndpointContext(sagemaker_session, context_name="foo", context_arn="bazz")

    sagemaker_session.sagemaker_client.list_associations.side_effect = [
        {
            "AssociationSummaries": [
                {
                    "SourceArn": "bazz",
                    "SourceName": "X1",
                    "DestinationArn": "B0",
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
                    "SourceArn": "B0",
                    "SourceName": "X2",
                    "DestinationArn": "B2",
                    "DestinationName": "Y2",
                    "SourceType": "C2",
                    "DestinationType": "Model",
                    "AssociationType": "E2",
                    "CreationTime": None,
                    "CreatedBy": None,
                }
            ]
        },
    ]

    model_list = obj.models()

    expected_calls = [
        unittest.mock.call(SourceArn=obj.context_arn, DestinationType="ModelDeployment"),
        unittest.mock.call(SourceArn="B0", DestinationType="Model"),
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.list_associations.mock_calls

    expected_model_list = [
        _api_types.AssociationSummary(
            source_arn="B0",
            source_name="X2",
            destination_arn="B2",
            destination_name="Y2",
            source_type="C2",
            destination_type="Model",
            association_type="E2",
            creation_time=None,
            created_by=None,
        )
    ]
    assert expected_model_list == model_list


def test_models_v2(sagemaker_session):
    arn1 = "arn:aws:sagemaker:us-west-2:123456789012:context/lineage-integ-3b05f017-0d87-4c37"

    obj = context.EndpointContext(sagemaker_session, context_name="foo", context_arn=arn1)

    sagemaker_session.sagemaker_client.query_lineage.return_value = {
        "Vertices": [
            {"Arn": arn1, "Type": "Model", "LineageType": "Artifact"},
        ],
        "Edges": [{"SourceArn": "arn1", "DestinationArn": "arn2", "AssociationType": "Produced"}],
    }

    sagemaker_session.sagemaker_client.describe_context.return_value = {
        "ContextName": "MyContext",
        "ContextArn": arn1,
        "Source": {
            "SourceUri": "arn:aws:sagemaker:us-west-2:0123456789012:endpoint/myendpoint",
            "SourceType": "ARN",
            "SourceId": "Thu Dec 17 17:16:24 UTC 2020",
        },
        "ContextType": "Endpoint",
        "Properties": {
            "PipelineExecutionArn": "arn:aws:sagemaker:us-west-2:0123456789012:\
                pipeline/mypipeline/execution/0irnteql64d0",
            "PipelineStepName": "MyStep",
            "Status": "Completed",
        },
        "CreationTime": 1608225384.0,
        "CreatedBy": {},
        "LastModifiedTime": 1608225384.0,
        "LastModifiedBy": {},
    }

    sagemaker_session.sagemaker_client.describe_artifact.return_value = {
        "ArtifactName": "MyArtifact",
        "ArtifactArn": arn1,
        "Source": {
            "SourceUri": "arn:aws:sagemaker:us-west-2:0123456789012:model/mymodel",
            "SourceType": "ARN",
            "SourceId": "Thu Dec 17 17:16:24 UTC 2020",
        },
        "ArtifactType": "Model",
        "Properties": {
            "PipelineExecutionArn": "arn:aws:sagemaker:us-west-2:0123456789012:\
                pipeline/mypipeline/execution/0irnteql64d0",
            "PipelineStepName": "MyStep",
            "Status": "Completed",
        },
        "CreationTime": 1608225384.0,
        "CreatedBy": {},
        "LastModifiedTime": 1608225384.0,
        "LastModifiedBy": {},
    }

    model_list = obj.models_v2()

    expected_calls = [
        unittest.mock.call(
            Direction="Descendants",
            Filters={"Types": ["ModelDeployment"], "LineageTypes": ["Action"]},
            IncludeEdges=False,
            MaxDepth=10,
            StartArns=[arn1],
        ),
        unittest.mock.call(
            Direction="Descendants",
            Filters={"Types": ["Model"], "LineageTypes": ["Artifact"]},
            IncludeEdges=False,
            MaxDepth=10,
            StartArns=[arn1],
        ),
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.query_lineage.mock_calls

    expected_model_list = [
        ModelArtifact(
            artifact_arn=arn1,
            artifact_name="MyArtifact",
            source=ArtifactSource(
                source_uri="arn:aws:sagemaker:us-west-2:0123456789012:model/mymodel",
                source_types=None,
                source_type="ARN",
                source_id="Thu Dec 17 17:16:24 UTC 2020",
            ),
            artifact_type="Model",
            properties={
                "PipelineExecutionArn": "arn:aws:sagemaker:us-west-2:0123456789012:\
                pipeline/mypipeline/execution/0irnteql64d0",
                "PipelineStepName": "MyStep",
                "Status": "Completed",
            },
            creation_time=1608225384.0,
            created_by={},
            last_modified_time=1608225384.0,
            last_modified_by={},
        )
    ]

    assert expected_model_list[0].artifact_arn == model_list[0].artifact_arn
    assert expected_model_list[0].artifact_name == model_list[0].artifact_name
    assert expected_model_list[0].source == model_list[0].source
    assert expected_model_list[0].artifact_type == model_list[0].artifact_type
    assert expected_model_list[0].artifact_type == "Model"
    assert expected_model_list[0].properties == model_list[0].properties
    assert expected_model_list[0].creation_time == model_list[0].creation_time
    assert expected_model_list[0].created_by == model_list[0].created_by
    assert expected_model_list[0].last_modified_time == model_list[0].last_modified_time
    assert expected_model_list[0].last_modified_by == model_list[0].last_modified_by
