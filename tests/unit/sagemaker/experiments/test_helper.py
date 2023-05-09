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

import json
import os
import shutil
import tempfile

from mock import Mock, PropertyMock, call
import pytest

from src.sagemaker.experiments._helper import (
    _LineageArtifactTracker,
    _ArtifactUploader,
)
from src.sagemaker.experiments._utils import resolve_artifact_name
from src.sagemaker.session import Session


@pytest.fixture
def client():
    """Mock client.

    Considerations when appropriate:

        * utilize botocore.stub.Stubber
        * separate runtime client from client
    """
    client_mock = Mock()
    client_mock._client_config.user_agent = (
        "Boto3/1.14.24 Python/3.8.5 Linux/5.4.0-42-generic Botocore/1.17.24 Resource"
    )
    return client_mock


@pytest.fixture
def boto_session(client):
    role_mock = Mock()
    type(role_mock).arn = PropertyMock(return_value="DummyRole")

    resource_mock = Mock()
    resource_mock.Role.return_value = role_mock

    session_mock = Mock(region_name="us-west-2")
    session_mock.resource.return_value = resource_mock
    session_mock.client.return_value = client

    return session_mock


@pytest.fixture
def sagemaker_session(client, boto_session):
    return Session(
        sagemaker_client=client,
        boto_session=boto_session,
    )


@pytest.fixture
def lineage_artifact_tracker(sagemaker_session):
    return _LineageArtifactTracker("test_trial_component_arn", sagemaker_session)


def test_lineage_artifact_tracker(lineage_artifact_tracker, sagemaker_session):
    client = sagemaker_session.sagemaker_client
    lineage_artifact_tracker.add_input_artifact(
        "input_name", "input_source_uri", "input_etag", "text/plain"
    )
    lineage_artifact_tracker.add_output_artifact(
        "output_name", "output_source_uri", "output_etag", "text/plain"
    )
    client.create_artifact.side_effect = [
        {"ArtifactArn": "created_arn_1"},
        {"ArtifactArn": "created_arn_2"},
    ]

    lineage_artifact_tracker.save()

    expected_calls = [
        call(
            ArtifactName="input_name",
            ArtifactType="text/plain",
            Source={
                "SourceUri": "input_source_uri",
                "SourceTypes": [{"SourceIdType": "S3ETag", "Value": "input_etag"}],
            },
        ),
        call(
            ArtifactName="output_name",
            ArtifactType="text/plain",
            Source={
                "SourceUri": "output_source_uri",
                "SourceTypes": [{"SourceIdType": "S3ETag", "Value": "output_etag"}],
            },
        ),
    ]
    assert expected_calls == client.create_artifact.mock_calls

    expected_calls = [
        call(
            SourceArn="created_arn_1",
            DestinationArn="test_trial_component_arn",
            AssociationType="ContributedTo",
        ),
        call(
            SourceArn="test_trial_component_arn",
            DestinationArn="created_arn_2",
            AssociationType="Produced",
        ),
    ]
    assert expected_calls == client.add_association.mock_calls


@pytest.fixture
def artifact_uploader(sagemaker_session):
    return _ArtifactUploader(
        trial_component_name="trial_component_name",
        artifact_bucket="artifact_bucket",
        artifact_prefix="artifact_prefix",
        sagemaker_session=sagemaker_session,
    )


@pytest.fixture
def tempdir():
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir)


def test_artifact_uploader_init(artifact_uploader):
    assert "trial_component_name" == artifact_uploader.trial_component_name
    assert "artifact_bucket" == artifact_uploader.artifact_bucket
    assert "artifact_prefix" == artifact_uploader.artifact_prefix


def test_artifact_uploader_upload_artifact_file_not_exists(tempdir, artifact_uploader):
    not_exist_file = os.path.join(tempdir, "not.exists")
    with pytest.raises(ValueError) as error:
        artifact_uploader.upload_artifact(not_exist_file)
    assert "does not exist or is not a file" in str(error)


def test_artifact_uploader_upload_artifact(tempdir, artifact_uploader):
    path = os.path.join(tempdir, "exists")
    with open(path, "a") as f:
        f.write("boo")

    name = resolve_artifact_name(path)
    artifact_uploader._s3_client.head_object.return_value = {"ETag": "etag_value"}

    s3_uri, etag = artifact_uploader.upload_artifact(path)
    expected_key = "{}/{}/{}".format(
        artifact_uploader.artifact_prefix, artifact_uploader.trial_component_name, name
    )

    artifact_uploader._s3_client.upload_file.assert_called_with(
        path, artifact_uploader.artifact_bucket, expected_key
    )

    expected_uri = "s3://{}/{}".format(artifact_uploader.artifact_bucket, expected_key)
    assert expected_uri == s3_uri


def test_artifact_uploader_upload_object_artifact(tempdir, artifact_uploader):
    artifact_uploader._s3_client.head_object.return_value = {"ETag": "etag_value"}

    artifact_name = "my-artifact"
    artifact_object = {"key": "value"}
    file_extension = ".csv"
    s3_uri, etag = artifact_uploader.upload_object_artifact(
        artifact_name, artifact_object, file_extension
    )
    name = artifact_name + file_extension
    expected_key = "{}/{}/{}".format(
        artifact_uploader.artifact_prefix, artifact_uploader.trial_component_name, name
    )

    artifact_uploader._s3_client.put_object.assert_called_with(
        Body=json.dumps(artifact_object), Bucket=artifact_uploader.artifact_bucket, Key=expected_key
    )

    expected_uri = "s3://{}/{}".format(artifact_uploader.artifact_bucket, expected_key)
    assert expected_uri == s3_uri
