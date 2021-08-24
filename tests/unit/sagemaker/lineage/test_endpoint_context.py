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
