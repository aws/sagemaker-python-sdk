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

from sagemaker.lineage import artifact, _api_types


def test_trained_models(sagemaker_session):
    model_artifact_obj = artifact.ModelArtifact(
        sagemaker_session, artifact_arn="model-artifact-arn"
    )

    sagemaker_session.sagemaker_client.list_associations.side_effect = [
        {
            "AssociationSummaries": [
                {
                    "SourceArn": model_artifact_obj.artifact_arn,
                    "SourceName": "X1",
                    "DestinationArn": "action-arn",
                    "DestinationName": "Y1",
                    "SourceType": "C1",
                    "DestinationType": "Action",
                    "AssociationType": "E1",
                    "CreationTime": None,
                    "CreatedBy": None,
                }
            ],
        },
        {
            "AssociationSummaries": [
                {
                    "SourceArn": "action-arn",
                    "SourceName": "X2",
                    "DestinationArn": "endpoint-context-arn",
                    "DestinationName": "Y2",
                    "SourceType": "Action",
                    "DestinationType": "Context",
                    "AssociationType": "E2",
                    "CreationTime": None,
                    "CreatedBy": None,
                }
            ]
        },
    ]

    endpoint_context_list = model_artifact_obj.endpoints()
    expected_calls = [
        unittest.mock.call(SourceArn=model_artifact_obj.artifact_arn, DestinationType="Action"),
        unittest.mock.call(SourceArn="action-arn", DestinationType="Context"),
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.list_associations.mock_calls
    expected_model_list = [
        _api_types.AssociationSummary(
            source_arn="action-arn",
            source_name="X2",
            destination_arn="endpoint-context-arn",
            destination_name="Y2",
            source_type="Action",
            destination_type="Context",
            association_type="E2",
            creation_time=None,
            created_by=None,
        )
    ]
    assert expected_model_list == endpoint_context_list
