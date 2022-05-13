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
from sagemaker.lineage import artifact, lineage_trial_component


@pytest.fixture
def sagemaker_session():
    return unittest.mock.Mock()


def test_dataset_artifacts(sagemaker_session):
    trial_component_arn = (
        "arn:aws:sagemaker:us-west-2:123456789012:trial_component/lineage-unit-3b05f017-0d87-4c37"
    )
    artifact_dataset_arn = "arn:aws:sagemaker:us-west-2:123456789012:artifact/datasets"
    artifact_dataset_name = "myDataset"

    obj = lineage_trial_component.LineageTrialComponent(
        sagemaker_session, trial_component_name="foo", trial_component_arn=trial_component_arn
    )

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

    dataset_list = obj.dataset_artifacts()
    expected_calls = [
        unittest.mock.call(
            Direction="Ascendants",
            Filters={"Types": ["DataSet"], "LineageTypes": ["Artifact"]},
            IncludeEdges=False,
            MaxDepth=10,
            StartArns=[trial_component_arn],
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


def test_models(sagemaker_session):
    trial_component_arn = (
        "arn:aws:sagemaker:us-west-2:123456789012:trial_component/lineage-unit-3b05f017-0d87-4c37"
    )
    model_arn = "arn:aws:sagemaker:us-west-2:123456789012:context/models"
    model_name = "myDataset"

    obj = lineage_trial_component.LineageTrialComponent(
        sagemaker_session, trial_component_name="foo", trial_component_arn=trial_component_arn
    )

    sagemaker_session.sagemaker_client.query_lineage.return_value = {
        "Vertices": [
            {"Arn": model_arn, "Type": "Model", "LineageType": "Artifact"},
        ],
        "Edges": [{"SourceArn": "arn1", "DestinationArn": "arn2", "AssociationType": "Produced"}],
    }

    sagemaker_session.sagemaker_client.describe_artifact.return_value = {
        "ArtifactName": model_name,
        "ArtifactArn": model_arn,
    }

    model_list = obj.models()
    expected_calls = [
        unittest.mock.call(
            Direction="Descendants",
            Filters={"Types": ["Model"], "LineageTypes": ["Artifact"]},
            IncludeEdges=False,
            MaxDepth=10,
            StartArns=[trial_component_arn],
        ),
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.query_lineage.mock_calls
    expected_model_list = [
        artifact.DatasetArtifact(
            artifact_name=model_name,
            artifact_arn=model_arn,
        )
    ]
    assert expected_model_list[0].artifact_arn == model_list[0].artifact_arn
    assert expected_model_list[0].artifact_name == model_list[0].artifact_name


def test_pipeline_execution_arn(sagemaker_session):
    trial_component_arn = (
        "arn:aws:sagemaker:us-west-2:123456789012:trial_component/lineage-unit-3b05f017-0d87-4c37"
    )
    training_job_arn = (
        "arn:aws:sagemaker:us-west-2:123456789012:training-job/pipelines-bs6gaeln463r-abalonetrain"
    )
    context = lineage_trial_component.LineageTrialComponent(
        sagemaker_session,
        trial_component_name="foo",
        trial_component_arn=trial_component_arn,
        source={
            "SourceArn": training_job_arn,
            "SourceType": "SageMakerTrainingJob",
        },
    )
    obj = {
        "TrialComponentName": "pipelines-bs6gaeln463r-AbaloneTrain-A0QiDGuY6z-aws-training-job",
        "TrialComponentArn": trial_component_arn,
        "DisplayName": "pipelines-bs6gaeln463r-AbaloneTrain-A0QiDGuY6z-aws-training-job",
        "Source": {
            "SourceArn": training_job_arn,
            "SourceType": "SageMakerTrainingJob",
        },
    }
    sagemaker_session.sagemaker_client.describe_trial_component.return_value = obj

    sagemaker_session.sagemaker_client.list_tags.return_value = {
        "Tags": [
            {"Key": "sagemaker:pipeline-execution-arn", "Value": "tag1"},
        ],
    }
    expected_calls = [
        unittest.mock.call(ResourceArn=training_job_arn),
    ]
    pipeline_execution_arn_result = context.pipeline_execution_arn()

    assert pipeline_execution_arn_result == "tag1"
    assert expected_calls == sagemaker_session.sagemaker_client.list_tags.mock_calls


def test_no_pipeline_execution_arn(sagemaker_session):
    trial_component_arn = (
        "arn:aws:sagemaker:us-west-2:123456789012:trial_component/lineage-unit-3b05f017-0d87-4c37"
    )
    training_job_arn = (
        "arn:aws:sagemaker:us-west-2:123456789012:training-job/pipelines-bs6gaeln463r-abalonetrain"
    )
    context = lineage_trial_component.LineageTrialComponent(
        sagemaker_session,
        trial_component_name="foo",
        trial_component_arn=trial_component_arn,
        source={
            "SourceArn": training_job_arn,
            "SourceType": "SageMakerTrainingJob",
        },
    )
    obj = {
        "TrialComponentName": "pipelines-bs6gaeln463r-AbaloneTrain-A0QiDGuY6z-aws-training-job",
        "TrialComponentArn": trial_component_arn,
        "DisplayName": "pipelines-bs6gaeln463r-AbaloneTrain-A0QiDGuY6z-aws-training-job",
        "Source": {
            "SourceArn": training_job_arn,
            "SourceType": "SageMakerTrainingJob",
        },
    }
    sagemaker_session.sagemaker_client.describe_trial_component.return_value = obj

    sagemaker_session.sagemaker_client.list_tags.return_value = {
        "Tags": [
            {"Key": "abcd", "Value": "efg"},
        ],
    }
    expected_calls = [
        unittest.mock.call(ResourceArn=training_job_arn),
    ]
    pipeline_execution_arn_result = context.pipeline_execution_arn()
    expected_result = None
    assert pipeline_execution_arn_result == expected_result
    assert expected_calls == sagemaker_session.sagemaker_client.list_tags.mock_calls


def test_no_source_arn_pipeline_execution_arn(sagemaker_session):
    trial_component_arn = (
        "arn:aws:sagemaker:us-west-2:123456789012:trial_component/lineage-unit-3b05f017-0d87-4c37"
    )
    training_job_arn = (
        "arn:aws:sagemaker:us-west-2:123456789012:training-job/pipelines-bs6gaeln463r-abalonetrain"
    )
    context = lineage_trial_component.LineageTrialComponent(
        sagemaker_session,
        trial_component_name="foo",
        trial_component_arn=trial_component_arn,
        source={
            "SourceArn": training_job_arn,
            "SourceType": "SageMakerTrainingJob",
        },
    )
    obj = {
        "TrialComponentName": "pipelines-bs6gaeln463r-AbaloneTrain-A0QiDGuY6z-aws-training-job",
        "TrialComponentArn": trial_component_arn,
        "DisplayName": "pipelines-bs6gaeln463r-AbaloneTrain-A0QiDGuY6z-aws-training-job",
        "Source": {
            "SourceArn": None,
            "SourceType": None,
        },
    }
    sagemaker_session.sagemaker_client.describe_trial_component.return_value = obj

    sagemaker_session.sagemaker_client.list_tags.return_value = {
        "Tags": [
            {"Key": "abcd", "Value": "efg"},
        ],
    }
    expected_calls = []
    pipeline_execution_arn_result = context.pipeline_execution_arn()
    expected_result = None
    assert pipeline_execution_arn_result == expected_result
    assert expected_calls == sagemaker_session.sagemaker_client.list_tags.mock_calls
