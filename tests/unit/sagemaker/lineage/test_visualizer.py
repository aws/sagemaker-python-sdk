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
from sagemaker.lineage import visualizer
import pandas as pd
from collections import OrderedDict


@pytest.fixture
def sagemaker_session():
    return unittest.mock.Mock()


@pytest.fixture
def vizualizer(sagemaker_session):
    return visualizer.LineageTableVisualizer(sagemaker_session)


def test_friendly_name_short_uri(vizualizer, sagemaker_session):
    uri = "s3://f-069083975568/train.txt"
    arn = "test_arn"
    sagemaker_session.sagemaker_client.describe_artifact.return_value = {
        "Source": {"SourceUri": uri, "SourceTypes": ""}
    }
    actual_name = vizualizer._get_friendly_name(name=None, arn=arn, entity_type="artifact")
    assert uri == actual_name


def test_friendly_name_long_uri(vizualizer, sagemaker_session):
    uri = (
        "s3://flintstone-end-to-end-tests-gamma-us-west-2-069083975568/results/canary-auto-1608761252626/"
        "preprocessed-data/tuning_data/train.txt"
    )
    arn = "test_arn"
    sagemaker_session.sagemaker_client.describe_artifact.return_value = {
        "Source": {"SourceUri": uri, "SourceTypes": ""}
    }
    actual_name = vizualizer._get_friendly_name(name=None, arn=arn, entity_type="artifact")
    expected_name = "s3://.../preprocessed-data/tuning_data/train.txt"
    assert expected_name == actual_name


def test_trial_component_name(sagemaker_session, vizualizer):
    name = "tc-name"

    sagemaker_session.sagemaker_client.describe_trial_component.return_value = {
        "TrialComponentArn": "tc-arn",
    }

    sagemaker_session.sagemaker_client.list_associations.side_effect = [
        {
            "AssociationSummaries": [
                {
                    "SourceArn": "a:b:c:d:e:artifact/src-arn-1",
                    "SourceName": "source-name-1",
                    "SourceType": "source-type-1",
                    "DestinationArn": "a:b:c:d:e:artifact/dest-arn-1",
                    "DestinationName": "dest-name-1",
                    "DestinationType": "dest-type-1",
                    "AssociationType": "type-1",
                }
            ]
        },
        {
            "AssociationSummaries": [
                {
                    "SourceArn": "a:b:c:d:e:artifact/src-arn-2",
                    "SourceName": "source-name-2",
                    "SourceType": "source-type-2",
                    "DestinationArn": "a:b:c:d:e:artifact/dest-arn-2",
                    "DestinationName": "dest-name-2",
                    "DestinationType": "dest-type-2",
                    "AssociationType": "type-2",
                }
            ]
        },
    ]

    df = vizualizer.show(trial_component_name=name)

    sagemaker_session.sagemaker_client.describe_trial_component.assert_called_with(
        TrialComponentName=name,
    )

    expected_calls = [
        unittest.mock.call(
            DestinationArn="tc-arn",
        ),
        unittest.mock.call(
            SourceArn="tc-arn",
        ),
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.list_associations.mock_calls

    expected_dataframe = pd.DataFrame.from_dict(
        OrderedDict(
            [
                ("Name/Source", ["source-name-1", "dest-name-2"]),
                ("Direction", ["Input", "Output"]),
                ("Type", ["source-type-1", "dest-type-2"]),
                ("Association Type", ["type-1", "type-2"]),
                ("Lineage Type", ["artifact", "artifact"]),
            ]
        )
    )

    pd.testing.assert_frame_equal(expected_dataframe, df)


def test_model_package_arn(sagemaker_session, vizualizer):
    name = "model_package_arn"

    sagemaker_session.sagemaker_client.list_artifacts.return_value = {
        "ArtifactSummaries": [{"ArtifactArn": "artifact-arn"}]
    }

    sagemaker_session.sagemaker_client.list_associations.side_effect = [
        {
            "AssociationSummaries": [
                {
                    "SourceArn": "a:b:c:d:e:artifact/src-arn-1",
                    "SourceName": "source-name-1",
                    "SourceType": "source-type-1",
                    "DestinationArn": "a:b:c:d:e:artifact/dest-arn-1",
                    "DestinationName": "dest-name-1",
                    "DestinationType": "dest-type-1",
                    "AssociationType": "type-1",
                }
            ]
        },
        {
            "AssociationSummaries": [
                {
                    "SourceArn": "a:b:c:d:e:artifact/src-arn-2",
                    "SourceName": "source-name-2",
                    "SourceType": "source-type-2",
                    "DestinationArn": "a:b:c:d:e:artifact/dest-arn-2",
                    "DestinationName": "dest-name-2",
                    "DestinationType": "dest-type-2",
                    "AssociationType": "type-2",
                }
            ]
        },
    ]

    df = vizualizer.show(model_package_arn=name)

    sagemaker_session.sagemaker_client.list_artifacts.assert_called_with(
        SourceUri=name,
    )

    expected_calls = [
        unittest.mock.call(
            DestinationArn="artifact-arn",
        ),
        unittest.mock.call(
            SourceArn="artifact-arn",
        ),
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.list_associations.mock_calls

    expected_dataframe = pd.DataFrame.from_dict(
        OrderedDict(
            [
                ("Name/Source", ["source-name-1", "dest-name-2"]),
                ("Direction", ["Input", "Output"]),
                ("Type", ["source-type-1", "dest-type-2"]),
                ("Association Type", ["type-1", "type-2"]),
                ("Lineage Type", ["artifact", "artifact"]),
            ]
        )
    )

    pd.testing.assert_frame_equal(expected_dataframe, df)


def test_endpoint_arn(sagemaker_session, vizualizer):
    name = "endpoint_arn"

    sagemaker_session.sagemaker_client.list_contexts.return_value = {
        "ContextSummaries": [{"ContextArn": "context-arn"}]
    }

    sagemaker_session.sagemaker_client.list_associations.side_effect = [
        {
            "AssociationSummaries": [
                {
                    "SourceArn": "a:b:c:d:e:context/src-arn-1",
                    "SourceName": "source-name-1",
                    "SourceType": "source-type-1",
                    "DestinationArn": "a:b:c:d:e:context/dest-arn-1",
                    "DestinationName": "dest-name-1",
                    "DestinationType": "dest-type-1",
                    "AssociationType": "type-1",
                }
            ]
        },
        {
            "AssociationSummaries": [
                {
                    "SourceArn": "a:b:c:d:e:context/src-arn-2",
                    "SourceName": "source-name-2",
                    "SourceType": "source-type-2",
                    "DestinationArn": "a:b:c:d:e:context/dest-arn-2",
                    "DestinationName": "dest-name-2",
                    "DestinationType": "dest-type-2",
                    "AssociationType": "type-2",
                }
            ]
        },
    ]

    df = vizualizer.show(endpoint_arn=name)

    sagemaker_session.sagemaker_client.list_contexts.assert_called_with(
        SourceUri=name,
    )

    expected_calls = [
        unittest.mock.call(
            DestinationArn="context-arn",
        ),
        unittest.mock.call(
            SourceArn="context-arn",
        ),
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.list_associations.mock_calls

    expected_dataframe = pd.DataFrame.from_dict(
        OrderedDict(
            [
                ("Name/Source", ["source-name-1", "dest-name-2"]),
                ("Direction", ["Input", "Output"]),
                ("Type", ["source-type-1", "dest-type-2"]),
                ("Association Type", ["type-1", "type-2"]),
                ("Lineage Type", ["context", "context"]),
            ]
        )
    )

    pd.testing.assert_frame_equal(expected_dataframe, df)
