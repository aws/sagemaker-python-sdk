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

from unittest.mock import call

import pytest
from botocore.exceptions import ClientError
from mock import Mock, patch
from sagemaker import Session
from sagemaker.lineage.artifact import ArtifactSummary, Artifact
from sagemaker.lineage.query import LineageSourceEnum

from sagemaker.serve.utils.lineage_constants import (
    MLFLOW_RUN_ID,
    MLFLOW_MODEL_PACKAGE_PATH,
    MLFLOW_S3_PATH,
    MLFLOW_LOCAL_PATH,
    LINEAGE_POLLER_MAX_TIMEOUT_SECS,
    MODEL_BUILDER_MLFLOW_MODEL_PATH_LINEAGE_ARTIFACT_TYPE,
    CONTRIBUTED_TO,
    MLFLOW_REGISTRY_PATH,
)
from sagemaker.serve.utils.lineage_utils import (
    _load_artifact_by_source_uri,
    _poll_lineage_artifact,
    _get_mlflow_model_path_type,
    _create_mlflow_model_path_lineage_artifact,
    _add_association_between_artifacts,
    _retrieve_and_create_if_not_exist_mlflow_model_path_lineage_artifact,
    _maintain_lineage_tracking_for_mlflow_model,
)


@patch("sagemaker.lineage.artifact.Artifact.list")
def test_load_artifact_by_source_uri(mock_artifact_list):
    source_uri = "s3://mybucket/mymodel"
    sagemaker_session = Mock(spec=Session)

    mock_artifact_1 = Mock(spec=ArtifactSummary)
    mock_artifact_1.artifact_type = LineageSourceEnum.MODEL_DATA.value
    mock_artifact_2 = Mock(spec=ArtifactSummary)
    mock_artifact_2.artifact_type = LineageSourceEnum.IMAGE.value
    mock_artifacts = [mock_artifact_1, mock_artifact_2]
    mock_artifact_list.return_value = mock_artifacts

    result = _load_artifact_by_source_uri(
        source_uri, LineageSourceEnum.MODEL_DATA.value, sagemaker_session
    )

    mock_artifact_list.assert_called_once_with(
        source_uri=source_uri, sagemaker_session=sagemaker_session
    )
    assert result == mock_artifact_1


@patch("sagemaker.lineage.artifact.Artifact.list")
def test_load_artifact_by_source_uri_no_match(mock_artifact_list):
    source_uri = "s3://mybucket/mymodel"
    sagemaker_session = Mock(spec=Session)

    mock_artifact_1 = Mock(spec=ArtifactSummary)
    mock_artifact_1.artifact_type = LineageSourceEnum.IMAGE.value
    mock_artifact_2 = Mock(spec=ArtifactSummary)
    mock_artifact_2.artifact_type = LineageSourceEnum.IMAGE.value
    mock_artifacts = [mock_artifact_1, mock_artifact_2]
    mock_artifact_list.return_value = mock_artifacts

    result = _load_artifact_by_source_uri(
        source_uri, LineageSourceEnum.MODEL_DATA.value, sagemaker_session
    )

    mock_artifact_list.assert_called_once_with(
        source_uri=source_uri, sagemaker_session=sagemaker_session
    )
    assert result is None


@patch("sagemaker.serve.utils.lineage_utils._load_artifact_by_source_uri")
def test_poll_lineage_artifact_found(mock_load_artifact):
    s3_uri = "s3://mybucket/mymodel"
    sagemaker_session = Mock(spec=Session)
    mock_artifact = Mock(spec=ArtifactSummary)

    with patch("time.time") as mock_time:
        mock_time.return_value = 0

        mock_load_artifact.return_value = mock_artifact

        result = _poll_lineage_artifact(
            s3_uri, LineageSourceEnum.MODEL_DATA.value, sagemaker_session
        )

    assert result == mock_artifact
    mock_load_artifact.assert_has_calls(
        [
            call(s3_uri, LineageSourceEnum.MODEL_DATA.value, sagemaker_session),
        ]
    )


@patch("sagemaker.serve.utils.lineage_utils._load_artifact_by_source_uri")
def test_poll_lineage_artifact_not_found(mock_load_artifact):
    s3_uri = "s3://mybucket/mymodel"
    artifact_type = LineageSourceEnum.MODEL_DATA.value
    sagemaker_session = Mock(spec=Session)

    with patch("time.time") as mock_time:
        mock_time_values = [0.0, 1.0, LINEAGE_POLLER_MAX_TIMEOUT_SECS + 1.0]
        mock_time.side_effect = mock_time_values

        with patch("time.sleep"):
            mock_load_artifact.side_effect = [None, None, None]

            result = _poll_lineage_artifact(s3_uri, artifact_type, sagemaker_session)

    assert result is None


@pytest.mark.parametrize(
    "mlflow_model_path, expected_output",
    [
        ("runs:/abc123", MLFLOW_RUN_ID),
        ("models:/my-model/1", MLFLOW_REGISTRY_PATH),
        (
            "arn:aws:sagemaker:us-west-2:123456789012:model-package/my-model-package",
            MLFLOW_MODEL_PACKAGE_PATH,
        ),
        ("s3://my-bucket/path/to/model", MLFLOW_S3_PATH),
    ],
)
def test_get_mlflow_model_path_type_valid(mlflow_model_path, expected_output):
    result = _get_mlflow_model_path_type(mlflow_model_path)
    assert result == expected_output


@patch("os.path.exists")
def test_get_mlflow_model_path_type_valid_local_path(mock_path_exists):
    valid_path = "/path/to/mlflow_model"
    mock_path_exists.side_effect = lambda path: path == valid_path
    result = _get_mlflow_model_path_type(valid_path)
    assert result == MLFLOW_LOCAL_PATH


def test_get_mlflow_model_path_type_invalid():
    invalid_path = "invalid_path"
    with pytest.raises(ValueError, match=f"Invalid MLflow model path: {invalid_path}"):
        _get_mlflow_model_path_type(invalid_path)


@patch("sagemaker.serve.utils.lineage_utils._get_mlflow_model_path_type")
@patch("sagemaker.lineage.artifact.Artifact.create")
def test_create_mlflow_model_path_lineage_artifact_success(
    mock_artifact_create, mock_get_mlflow_path_type
):
    mlflow_model_path = "runs:/Ab12Cd34"
    sagemaker_session = Mock(spec=Session)
    mock_artifact = Mock(spec=Artifact)
    mock_get_mlflow_path_type.return_value = "mlflow_run_id"
    mock_artifact_create.return_value = mock_artifact

    result = _create_mlflow_model_path_lineage_artifact(mlflow_model_path, sagemaker_session)

    assert result == mock_artifact
    mock_get_mlflow_path_type.assert_called_once_with(mlflow_model_path)
    mock_artifact_create.assert_called_once_with(
        source_uri=mlflow_model_path,
        artifact_type=MODEL_BUILDER_MLFLOW_MODEL_PATH_LINEAGE_ARTIFACT_TYPE,
        artifact_name="mlflow_run_id",
        properties={"model_builder_input_model_data_type": "mlflow_run_id"},
        sagemaker_session=sagemaker_session,
    )


@patch("sagemaker.serve.utils.lineage_utils._get_mlflow_model_path_type")
@patch("sagemaker.lineage.artifact.Artifact.create")
def test_create_mlflow_model_path_lineage_artifact_validation_exception(
    mock_artifact_create, mock_get_mlflow_path_type
):
    mlflow_model_path = "runs:/Ab12Cd34"
    sagemaker_session = Mock(spec=Session)
    mock_get_mlflow_path_type.return_value = "mlflow_run_id"
    mock_artifact_create.side_effect = ClientError(
        error_response={"Error": {"Code": "ValidationException"}}, operation_name="CreateArtifact"
    )

    result = _create_mlflow_model_path_lineage_artifact(mlflow_model_path, sagemaker_session)

    assert result is None


@patch("sagemaker.serve.utils.lineage_utils._get_mlflow_model_path_type")
@patch("sagemaker.lineage.artifact.Artifact.create")
def test_create_mlflow_model_path_lineage_artifact_other_exception(
    mock_artifact_create, mock_get_mlflow_path_type
):
    mlflow_model_path = "runs:/Ab12Cd34"
    sagemaker_session = Mock(spec=Session)
    mock_get_mlflow_path_type.return_value = "mlflow_run_id"
    mock_artifact_create.side_effect = ClientError(
        error_response={"Error": {"Code": "SomeOtherException"}}, operation_name="CreateArtifact"
    )

    with pytest.raises(ClientError):
        _create_mlflow_model_path_lineage_artifact(mlflow_model_path, sagemaker_session)


@patch("sagemaker.serve.utils.lineage_utils._create_mlflow_model_path_lineage_artifact")
@patch("sagemaker.serve.utils.lineage_utils._load_artifact_by_source_uri")
def test_retrieve_and_create_if_not_exist_mlflow_model_path_lineage_artifact_existing(
    mock_load_artifact, mock_create_artifact
):
    mlflow_model_path = "runs:/Ab12Cd34"
    sagemaker_session = Mock(spec=Session)
    mock_artifact_summary = Mock(spec=ArtifactSummary)
    mock_load_artifact.return_value = mock_artifact_summary

    result = _retrieve_and_create_if_not_exist_mlflow_model_path_lineage_artifact(
        mlflow_model_path, sagemaker_session
    )

    assert result == mock_artifact_summary
    mock_load_artifact.assert_called_once_with(
        mlflow_model_path, MODEL_BUILDER_MLFLOW_MODEL_PATH_LINEAGE_ARTIFACT_TYPE, sagemaker_session
    )
    mock_create_artifact.assert_not_called()


@patch("sagemaker.serve.utils.lineage_utils._create_mlflow_model_path_lineage_artifact")
@patch("sagemaker.serve.utils.lineage_utils._load_artifact_by_source_uri")
def test_retrieve_and_create_if_not_exist_mlflow_model_path_lineage_artifact_create(
    mock_load_artifact, mock_create_artifact
):
    mlflow_model_path = "runs:/Ab12Cd34"
    sagemaker_session = Mock(spec=Session)
    mock_artifact = Mock(spec=Artifact)
    mock_load_artifact.return_value = None
    mock_create_artifact.return_value = mock_artifact

    result = _retrieve_and_create_if_not_exist_mlflow_model_path_lineage_artifact(
        mlflow_model_path, sagemaker_session
    )

    assert result == mock_artifact
    mock_load_artifact.assert_called_once_with(
        mlflow_model_path, MODEL_BUILDER_MLFLOW_MODEL_PATH_LINEAGE_ARTIFACT_TYPE, sagemaker_session
    )
    mock_create_artifact.assert_called_once_with(mlflow_model_path, sagemaker_session)


@patch("sagemaker.lineage.association.Association.create")
def test_add_association_between_artifacts_success(mock_association_create):
    mlflow_model_path_artifact_arn = "arn:aws:sagemaker:us-west-2:123456789012:artifact/123"
    autogenerated_model_data_artifact_arn = "arn:aws:sagemaker:us-west-2:123456789012:artifact/456"
    sagemaker_session = Mock(spec=Session)

    _add_association_between_artifacts(
        mlflow_model_path_artifact_arn,
        autogenerated_model_data_artifact_arn,
        sagemaker_session,
    )

    mock_association_create.assert_called_once_with(
        source_arn=mlflow_model_path_artifact_arn,
        destination_arn=autogenerated_model_data_artifact_arn,
        association_type=CONTRIBUTED_TO,
        sagemaker_session=sagemaker_session,
    )


@patch("sagemaker.lineage.association.Association.create")
def test_add_association_between_artifacts_validation_exception(mock_association_create):
    mlflow_model_path_artifact_arn = "arn:aws:sagemaker:us-west-2:123456789012:artifact/123"
    autogenerated_model_data_artifact_arn = "arn:aws:sagemaker:us-west-2:123456789012:artifact/456"
    sagemaker_session = Mock(spec=Session)
    mock_association_create.side_effect = ClientError(
        error_response={"Error": {"Code": "ValidationException"}},
        operation_name="CreateAssociation",
    )

    _add_association_between_artifacts(
        mlflow_model_path_artifact_arn,
        autogenerated_model_data_artifact_arn,
        sagemaker_session,
    )


@patch("sagemaker.lineage.association.Association.create")
def test_add_association_between_artifacts_other_exception(mock_association_create):
    mlflow_model_path_artifact_arn = "arn:aws:sagemaker:us-west-2:123456789012:artifact/123"
    autogenerated_model_data_artifact_arn = "arn:aws:sagemaker:us-west-2:123456789012:artifact/456"
    sagemaker_session = Mock(spec=Session)
    mock_association_create.side_effect = ClientError(
        error_response={"Error": {"Code": "SomeOtherException"}}, operation_name="CreateAssociation"
    )

    with pytest.raises(ClientError):
        _add_association_between_artifacts(
            mlflow_model_path_artifact_arn,
            autogenerated_model_data_artifact_arn,
            sagemaker_session,
        )


@patch("sagemaker.serve.utils.lineage_utils._poll_lineage_artifact")
@patch(
    "sagemaker.serve.utils.lineage_utils._retrieve_and_create_if_not_exist_mlflow_model_path_lineage_artifact"
)
@patch("sagemaker.serve.utils.lineage_utils._add_association_between_artifacts")
def test_maintain_lineage_tracking_for_mlflow_model_success(
    mock_add_association, mock_retrieve_create_artifact, mock_poll_artifact
):
    mlflow_model_path = "runs:/Ab12Cd34"
    s3_upload_path = "s3://mybucket/path/to/model"
    sagemaker_session = Mock(spec=Session)
    mock_model_data_artifact = Mock(spec=ArtifactSummary)
    mock_mlflow_model_artifact = Mock(spec=Artifact)
    mock_poll_artifact.return_value = mock_model_data_artifact
    mock_retrieve_create_artifact.return_value = mock_mlflow_model_artifact

    _maintain_lineage_tracking_for_mlflow_model(
        mlflow_model_path, s3_upload_path, sagemaker_session
    )

    mock_poll_artifact.assert_called_once_with(
        s3_uri=s3_upload_path,
        artifact_type=LineageSourceEnum.MODEL_DATA.value,
        sagemaker_session=sagemaker_session,
    )
    mock_retrieve_create_artifact.assert_called_once_with(
        mlflow_model_path=mlflow_model_path, sagemaker_session=sagemaker_session
    )
    mock_add_association.assert_called_once_with(
        mlflow_model_path_artifact_arn=mock_mlflow_model_artifact.artifact_arn,
        autogenerated_model_data_artifact_arn=mock_model_data_artifact.artifact_arn,
        sagemaker_session=sagemaker_session,
    )


@patch("sagemaker.serve.utils.lineage_utils._poll_lineage_artifact")
@patch(
    "sagemaker.serve.utils.lineage_utils._retrieve_and_create_if_not_exist_mlflow_model_path_lineage_artifact"
)
@patch("sagemaker.serve.utils.lineage_utils._add_association_between_artifacts")
def test_maintain_lineage_tracking_for_mlflow_model_no_model_data_artifact(
    mock_add_association, mock_retrieve_create_artifact, mock_poll_artifact
):
    mlflow_model_path = "runs:/Ab12Cd34"
    s3_upload_path = "s3://mybucket/path/to/model"
    sagemaker_session = Mock(spec=Session)
    mock_poll_artifact.return_value = None
    mock_retrieve_create_artifact.return_value = None

    _maintain_lineage_tracking_for_mlflow_model(
        mlflow_model_path, s3_upload_path, sagemaker_session
    )

    mock_poll_artifact.assert_called_once_with(
        s3_uri=s3_upload_path,
        artifact_type=LineageSourceEnum.MODEL_DATA.value,
        sagemaker_session=sagemaker_session,
    )
    mock_retrieve_create_artifact.assert_not_called()
    mock_add_association.assert_not_called()
