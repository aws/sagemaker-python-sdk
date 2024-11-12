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

import pandas as pd
from collections import OrderedDict


def test_friendly_name_short_uri(viz, sagemaker_session):
    uri = "s3://f-069083975568/train.txt"
    arn = "test_arn"
    sagemaker_session.sagemaker_client.describe_artifact.return_value = {
        "Source": {"SourceUri": uri, "SourceTypes": ""}
    }
    actual_name = viz._get_friendly_name(name=None, arn=arn, entity_type="artifact")
    assert uri == actual_name


def test_friendly_name_long_uri(viz, sagemaker_session):
    uri = (
        "s3://flintstone-end-to-end-tests-gamma-us-west-2-069083975568/results/canary-auto-1608761252626/"
        "preprocessed-data/tuning_data/train.txt"
    )
    arn = "test_arn"
    sagemaker_session.sagemaker_client.describe_artifact.return_value = {
        "Source": {"SourceUri": uri, "SourceTypes": ""}
    }
    actual_name = viz._get_friendly_name(name=None, arn=arn, entity_type="artifact")
    expected_name = "s3://.../preprocessed-data/tuning_data/train.txt"
    assert expected_name == actual_name


def test_trial_component_name(viz, sagemaker_session):
    name = "tc-name"

    sagemaker_session.sagemaker_client.describe_trial_component.return_value = {
        "TrialComponentArn": "tc-arn",
    }

    get_list_associations_side_effect(sagemaker_session)

    df = viz.show(trial_component_name=name)

    sagemaker_session.sagemaker_client.describe_trial_component.assert_called_with(
        TrialComponentName=name,
    )

    assert_list_associations_mock_calls(sagemaker_session)

    pd.testing.assert_frame_equal(get_expected_dataframe(), df)


def test_model_package_arn(viz, sagemaker_session):
    name = "model_package_arn"

    sagemaker_session.sagemaker_client.list_artifacts.return_value = {
        "ArtifactSummaries": [{"ArtifactArn": "artifact-arn"}]
    }

    get_list_associations_side_effect(sagemaker_session)

    df = viz.show(model_package_arn=name)

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

    pd.testing.assert_frame_equal(get_expected_dataframe(), df)


def test_endpoint_arn(viz, sagemaker_session):
    name = "endpoint_arn"

    sagemaker_session.sagemaker_client.list_contexts.return_value = {
        "ContextSummaries": [{"ContextArn": "context-arn"}]
    }

    get_list_associations_side_effect(sagemaker_session)

    df = viz.show(endpoint_arn=name)

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

    pd.testing.assert_frame_equal(get_expected_dataframe(), df)


def test_processing_job_pipeline_execution_step(viz, sagemaker_session):

    sagemaker_session.sagemaker_client.list_trial_components.return_value = {
        "TrialComponentSummaries": [{"TrialComponentArn": "tc-arn"}]
    }

    get_list_associations_side_effect(sagemaker_session)

    step = {"Metadata": {"ProcessingJob": {"Arn": "proc-job-arn"}}}

    df = viz.show(pipeline_execution_step=step)

    sagemaker_session.sagemaker_client.list_trial_components.assert_called_with(
        SourceArn="proc-job-arn",
    )

    assert_list_associations_mock_calls(sagemaker_session)

    pd.testing.assert_frame_equal(get_expected_dataframe(), df)


def test_training_job_pipeline_execution_step(viz, sagemaker_session):

    sagemaker_session.sagemaker_client.list_trial_components.return_value = {
        "TrialComponentSummaries": [{"TrialComponentArn": "tc-arn"}]
    }

    get_list_associations_side_effect(sagemaker_session)

    step = {"Metadata": {"TrainingJob": {"Arn": "training-job-arn"}}}

    df = viz.show(pipeline_execution_step=step)

    sagemaker_session.sagemaker_client.list_trial_components.assert_called_with(
        SourceArn="training-job-arn",
    )

    assert_list_associations_mock_calls(sagemaker_session)

    pd.testing.assert_frame_equal(get_expected_dataframe(), df)


def test_transform_job_pipeline_execution_step(viz, sagemaker_session):

    sagemaker_session.sagemaker_client.list_trial_components.return_value = {
        "TrialComponentSummaries": [{"TrialComponentArn": "tc-arn"}]
    }

    get_list_associations_side_effect(sagemaker_session)

    step = {"Metadata": {"TransformJob": {"Arn": "transform-job-arn"}}}

    df = viz.show(pipeline_execution_step=step)

    sagemaker_session.sagemaker_client.list_trial_components.assert_called_with(
        SourceArn="transform-job-arn",
    )

    assert_list_associations_mock_calls(sagemaker_session)

    pd.testing.assert_frame_equal(get_expected_dataframe(), df)


def get_list_associations_side_effect(sagemaker_session):

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


def assert_list_associations_mock_calls(sagemaker_session):

    expected_calls = [
        unittest.mock.call(
            DestinationArn="tc-arn",
        ),
        unittest.mock.call(
            SourceArn="tc-arn",
        ),
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.list_associations.mock_calls


def get_expected_dataframe():

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

    return expected_dataframe
