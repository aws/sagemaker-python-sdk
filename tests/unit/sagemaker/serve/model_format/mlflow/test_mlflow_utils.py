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

import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pytest
import yaml

from sagemaker.serve import ModelServer
from sagemaker.serve.model_format.mlflow.constants import (
    MLFLOW_PYFUNC,
    TENSORFLOW_SAVED_MODEL_NAME,
)
from sagemaker.serve.model_format.mlflow.utils import (
    _get_default_model_server_for_mlflow,
    _get_default_image_for_mlflow,
    _generate_mlflow_artifact_path,
    _get_all_flavor_metadata,
    _get_framework_version_from_requirements,
    _get_deployment_flavor,
    _get_python_version_from_parsed_mlflow_model_file,
    _mlflow_input_is_local_path,
    _download_s3_artifacts,
    _select_container_for_mlflow_model,
    _validate_input_for_mlflow,
    _copy_directory_contents,
    _move_contents,
    _get_saved_model_path_for_tensorflow_and_keras_flavor,
)


def test_get_default_model_server_for_mlflow():
    assert _get_default_model_server_for_mlflow("pyfunc") == ModelServer.TORCHSERVE


@patch("sagemaker.image_uris.retrieve")
def test_get_default_image_for_mlflow(mock_image_uris_retrieve):
    python_version = "3.8.6"
    region = "us-east-1"
    instance_type = "ml.m5.xlarge"

    mock_image_uri = "mock_image_uri"
    mock_image_uris_retrieve.return_value = mock_image_uri
    assert _get_default_image_for_mlflow(python_version, region, instance_type) == mock_image_uri


@patch("sagemaker.image_uris.retrieve")
def test_get_default_image_for_mlflow_invalid_input(mock_image_uris_retrieve):
    python_version = "3.8.6"
    region = "us-east-1"
    instance_type = "ml.m5.xlarge"

    mock_image_uri = None
    mock_image_uris_retrieve.return_value = mock_image_uri
    with pytest.raises(ValueError, match="Unable to find default image based on"):
        _get_default_image_for_mlflow(python_version, region, instance_type)


@patch("os.path.isfile")
def test_generate_mlflow_artifact_path_exists(mock_path_isfile):
    mock_path_isfile.side_effect = lambda path: path != src_folder
    src_folder = "/path/to/source/folder"
    artifact_name = "model.pkl"
    expected_path = os.path.join(src_folder, artifact_name)

    assert _generate_mlflow_artifact_path(src_folder, artifact_name) == expected_path


@patch("os.path.isfile")
def test_generate_mlflow_artifact_path_not_found(mock_path_isfile):
    mock_path_isfile.side_effect = lambda path: path == src_folder
    src_folder = "/nonexistent/path"
    artifact_name = "model.pkl"

    with pytest.raises(FileNotFoundError):
        _generate_mlflow_artifact_path(src_folder, artifact_name)


@patch("yaml.safe_load")
@patch("builtins.open", new_callable=mock_open, read_data="")
@patch("os.path.isfile")
def test_get_all_flavor_metadata_exists(mock_path_isfile, mock_file, mock_safe_load):
    mock_path_isfile.side_effect = lambda path: path == mlmodel_path
    mlmodel_path = "/path/to/mlmodel"
    mlmodel_content = {"flavors": "test"}
    mock_safe_load.return_value = mlmodel_content

    actual_value = _get_all_flavor_metadata(mlmodel_path)
    mock_file.assert_called_once_with(mlmodel_path, "r")
    assert actual_value == mlmodel_content["flavors"]


@patch("os.path.isfile")
def test_get_all_flavor_metadata_missing_file(mock_path_isfile):
    mock_path_isfile.side_effect = lambda path: path != mlmodel_path
    mlmodel_path = "/nonexistent/path"

    with pytest.raises(ValueError, match="File does not exist"):
        _get_all_flavor_metadata(mlmodel_path)


@patch("yaml.safe_load")
@patch("builtins.open", new_callable=mock_open, read_data="")
@patch("os.path.isfile")
def test_get_all_flavor_metadata_missing_flavors_key(mock_path_isfile, mock_file, mock_safe_load):
    mock_path_isfile.side_effect = lambda path: path == mlmodel_path
    mlmodel_path = "/path/to/mlmodel"
    mock_safe_load.return_value = {}

    with pytest.raises(ValueError, match="The 'flavors' key is missing in the MLmodel file."):
        _get_all_flavor_metadata(mlmodel_path)
        mock_file.assert_called_once_with(mlmodel_path, "r")


@patch("yaml.safe_load")
@patch("builtins.open", new_callable=mock_open, read_data="")
@patch("os.path.isfile")
def test_get_all_flavor_metadata_invalid_yaml(mock_path_isfile, mock_file, mock_safe_load):
    mock_path_isfile.side_effect = lambda path: path == mlmodel_path
    mlmodel_path = "/path/to/mlmodel"
    mock_safe_load.side_effect = yaml.YAMLError("Invalid YAML")

    with pytest.raises(ValueError, match="Error parsing the file as YAML"):
        _get_all_flavor_metadata(mlmodel_path)
        mock_file.assert_called_once_with(mlmodel_path, "r")


@patch("builtins.open", new_callable=mock_open, read_data="tensorflow==2.3.0")
def test_get_framework_version_from_requirements_found(mock_file):
    flavor = "tensorflow"
    requirements_path = "/path/to/requirements.txt"
    expected_version = "2.3.0"

    assert _get_framework_version_from_requirements(flavor, requirements_path) == expected_version
    mock_file.assert_called_once_with(requirements_path, "r")


@patch("builtins.open", new_callable=mock_open, read_data="tensorflow")
def test_get_framework_version_from_requirements_malformed(mock_file):
    flavor = "tensorflow"
    requirements_path = "/path/to/requirements.txt"

    assert _get_framework_version_from_requirements(flavor, requirements_path) is None
    mock_file.assert_called_once_with(requirements_path, "r")


@patch("builtins.open", new_callable=mock_open, read_data="")
def test_get_framework_version_from_requirements_file_not_found(mock_file):
    flavor = "tensorflow"
    requirements_path = "/nonexistent/path/requirements.txt"
    mock_file.side_effect = FileNotFoundError

    with pytest.raises(ValueError, match="File not found"):
        _get_framework_version_from_requirements(flavor, requirements_path)
        mock_file.assert_called_once_with(requirements_path, "r")


@patch("builtins.open", new_callable=mock_open, read_data="")
def test_get_framework_version_from_requirements_not_found(mock_file):
    flavor = "tensorflow"
    requirements_path = "/nonexistent/path/requirements.txt"

    assert _get_framework_version_from_requirements(flavor, requirements_path) is None
    mock_file.assert_called_once_with(requirements_path, "r")


def test_get_deployment_flavor_metadata_none():
    assert _get_deployment_flavor({MLFLOW_PYFUNC: ""}) == MLFLOW_PYFUNC
    assert _get_deployment_flavor({"tensorflow": "", "xgboost": ""}) == "tensorflow"

    with pytest.raises(ValueError, match="Flavor metadata is not found"):
        _get_deployment_flavor(None)


def test_get_python_version_from_parsed_mlflow_model_file():
    assert (
        _get_python_version_from_parsed_mlflow_model_file(
            {MLFLOW_PYFUNC: {"python_version": "3.8.6"}}
        )
        == "3.8.6"
    )

    with pytest.raises(ValueError, match=f"{MLFLOW_PYFUNC} cannot be found in MLmodel file."):
        _get_python_version_from_parsed_mlflow_model_file({})


@patch("os.path.exists")
def test_mlflow_input_is_local_path(mock_path_exists):
    valid_path = "/path/to/mlflow_model"
    mock_path_exists.side_effect = lambda path: path == valid_path

    assert not _mlflow_input_is_local_path("s3://my_bucket/path/to/model")
    assert not _mlflow_input_is_local_path("runs:/run-id/run/relative/path/to/model")
    assert not _mlflow_input_is_local_path("/invalid/path")
    assert _mlflow_input_is_local_path(valid_path)


def test_download_s3_artifacts():
    pass


def test_download_s3_artifacts_invalid_s3_path():
    with pytest.raises(ValueError, match="Invalid S3 path provided"):
        _download_s3_artifacts("invalid_path", "/destination/path", MagicMock())


@patch("sagemaker.Session")
@patch("os.makedirs")
def test_download_s3_artifacts_valid_s3_path(mock_os_makedirs, mock_session):
    s3_path = "s3://bucket/key"
    dst_path = "/destination/path"

    mock_s3_client = MagicMock()
    mock_s3_client.get_paginator.return_value.paginate.return_value = [
        {"Contents": [{"Key": "key/file1.txt"}, {"Key": "key/file2.txt"}]}
    ]
    mock_session.boto_session.client.return_value = mock_s3_client

    _download_s3_artifacts(s3_path, dst_path, mock_session)

    mock_os_makedirs.assert_called_with(dst_path, exist_ok=True)
    mock_s3_client.download_file.assert_any_call(
        "bucket", "key/file1.txt", os.path.join(dst_path, "file1.txt")
    )
    mock_s3_client.download_file.assert_any_call(
        "bucket", "key/file2.txt", os.path.join(dst_path, "file2.txt")
    )


@patch("sagemaker.image_uris.retrieve")
@patch("sagemaker.serve.model_format.mlflow.utils._cast_to_compatible_version")
@patch("sagemaker.serve.model_format.mlflow.utils._get_framework_version_from_requirements")
@patch(
    "sagemaker.serve.model_format.mlflow.utils._get_python_version_from_parsed_mlflow_model_file"
)
@patch("sagemaker.serve.model_format.mlflow.utils._get_all_flavor_metadata")
@patch("sagemaker.serve.model_format.mlflow.utils._generate_mlflow_artifact_path")
def test_select_container_for_mlflow_model_with_framework_specific_dlc(
    mock_generate_mlflow_artifact_path,
    mock_get_all_flavor_metadata,
    mock_get_python_version_from_parsed_mlflow_model_file,
    mock_get_framework_version_from_requirements,
    mock_cast_to_compatible_version,
    mock_image_uris_retrieve,
):
    mlflow_model_src_path = "/path/to/mlflow_model"
    deployment_flavor = "pytorch"
    region = "us-west-2"
    instance_type = "ml.m5.xlarge"

    mock_requirements_path = "/path/to/requirements.txt"
    mock_metadata_path = "/path/to/mlmodel"
    mock_flavor_metadata = {"pytorch": {"some_key": "some_value"}}
    mock_python_version = "3.8.6"
    mock_framework_version = "1.8.0"
    mock_casted_version = "2.0.1"
    mock_image_uri = "mock_image_uri"

    mock_generate_mlflow_artifact_path.side_effect = lambda path, artifact: (
        mock_requirements_path if artifact == "requirements.txt" else mock_metadata_path
    )
    mock_get_all_flavor_metadata.return_value = mock_flavor_metadata
    mock_get_python_version_from_parsed_mlflow_model_file.return_value = mock_python_version
    mock_get_framework_version_from_requirements.return_value = mock_framework_version
    mock_cast_to_compatible_version.return_value = (mock_casted_version,)
    mock_image_uris_retrieve.return_value = mock_image_uri

    assert (
        _select_container_for_mlflow_model(
            mlflow_model_src_path, deployment_flavor, region, instance_type
        )
        == mock_image_uri
    )

    mock_generate_mlflow_artifact_path.assert_any_call(mlflow_model_src_path, "requirements.txt")
    mock_generate_mlflow_artifact_path.assert_any_call(mlflow_model_src_path, "MLmodel")
    mock_get_all_flavor_metadata.assert_called_once_with(mock_metadata_path)
    mock_get_framework_version_from_requirements.assert_called_once_with(
        deployment_flavor, mock_requirements_path
    )
    mock_cast_to_compatible_version.assert_called_once_with(
        deployment_flavor, mock_framework_version
    )
    mock_image_uris_retrieve.assert_called_once_with(
        framework=deployment_flavor,
        region=region,
        version=mock_casted_version,
        image_scope="inference",
        py_version="py38",
        instance_type=instance_type,
    )


@patch("sagemaker.serve.model_format.mlflow.utils._get_default_image_for_mlflow")
@patch(
    "sagemaker.serve.model_format.mlflow.utils._get_python_version_from_parsed_mlflow_model_file"
)
@patch("sagemaker.serve.model_format.mlflow.utils._get_all_flavor_metadata")
@patch("sagemaker.serve.model_format.mlflow.utils._generate_mlflow_artifact_path")
def test_select_container_for_mlflow_model_with_no_framework_specific_dlc(
    mock_generate_mlflow_artifact_path,
    mock_get_all_flavor_metadata,
    mock_get_python_version_from_parsed_mlflow_model_file,
    mock_get_default_image_for_mlflow,
):
    mlflow_model_src_path = "/path/to/mlflow_model"
    deployment_flavor = "scikit-learn"
    region = "us-west-2"
    instance_type = "ml.m5.xlarge"

    mock_requirements_path = "/path/to/requirements.txt"
    mock_metadata_path = "/path/to/mlmodel"
    mock_flavor_metadata = {"scikit-learn": {"some_key": "some_value"}}
    mock_python_version = "3.9.6"
    mock_image_uri = "mock_image_uri"

    mock_generate_mlflow_artifact_path.side_effect = lambda path, artifact: (
        mock_requirements_path if artifact == "requirements.txt" else mock_metadata_path
    )
    mock_get_all_flavor_metadata.return_value = mock_flavor_metadata
    mock_get_python_version_from_parsed_mlflow_model_file.return_value = mock_python_version
    mock_get_default_image_for_mlflow.return_value = mock_image_uri

    with patch("sagemaker.serve.model_format.mlflow.utils.logger") as mock_logger:
        assert (
            _select_container_for_mlflow_model(
                mlflow_model_src_path, deployment_flavor, region, instance_type
            )
            == mock_image_uri
        )

    mock_generate_mlflow_artifact_path.assert_any_call(mlflow_model_src_path, "requirements.txt")
    mock_generate_mlflow_artifact_path.assert_any_call(mlflow_model_src_path, "MLmodel")
    mock_get_all_flavor_metadata.assert_called_once_with(mock_metadata_path)
    mock_get_python_version_from_parsed_mlflow_model_file.assert_called_once_with(
        mock_flavor_metadata
    )
    mock_logger.warning.assert_called_once_with(
        f"{deployment_flavor} flavor currently doesn't have optimized framework specific DLC support. "
        f"Defaulting to generic image..."
    )


@patch("sagemaker.image_uris.retrieve")
@patch("sagemaker.serve.model_format.mlflow.utils._cast_to_compatible_version")
@patch("sagemaker.serve.model_format.mlflow.utils._get_framework_version_from_requirements")
@patch(
    "sagemaker.serve.model_format.mlflow.utils._get_python_version_from_parsed_mlflow_model_file"
)
@patch("sagemaker.serve.model_format.mlflow.utils._get_all_flavor_metadata")
@patch("sagemaker.serve.model_format.mlflow.utils._generate_mlflow_artifact_path")
def test_select_container_for_mlflow_model_no_dlc_detected(
    mock_generate_mlflow_artifact_path,
    mock_get_all_flavor_metadata,
    mock_get_python_version_from_parsed_mlflow_model_file,
    mock_get_framework_version_from_requirements,
    mock_cast_to_compatible_version,
    mock_image_uris_retrieve,
):
    mlflow_model_src_path = "/path/to/mlflow_model"
    deployment_flavor = "pytorch"
    region = "us-west-2"
    instance_type = "ml.m5.xlarge"

    mock_requirements_path = "/path/to/requirements.txt"
    mock_metadata_path = "/path/to/mlmodel"
    mock_flavor_metadata = {"pytorch": {"some_key": "some_value"}}
    mock_python_version = "3.8.6"
    mock_framework_version = "1.8.0"
    mock_casted_version = "2.0.1"
    mock_image_uri = None

    mock_generate_mlflow_artifact_path.side_effect = lambda path, artifact: (
        mock_requirements_path if artifact == "requirements.txt" else mock_metadata_path
    )
    mock_get_all_flavor_metadata.return_value = mock_flavor_metadata
    mock_get_python_version_from_parsed_mlflow_model_file.return_value = mock_python_version
    mock_get_framework_version_from_requirements.return_value = mock_framework_version
    mock_cast_to_compatible_version.return_value = (mock_casted_version,)
    mock_image_uris_retrieve.return_value = mock_image_uri

    with pytest.raises(ValueError, match="Unable to auto detect a DLC for framework"):
        _select_container_for_mlflow_model(
            mlflow_model_src_path, deployment_flavor, region, instance_type
        )

        mock_generate_mlflow_artifact_path.assert_any_call(
            mlflow_model_src_path, "requirements.txt"
        )
        mock_generate_mlflow_artifact_path.assert_any_call(mlflow_model_src_path, "MLmodel")
        mock_get_all_flavor_metadata.assert_called_once_with(mock_metadata_path)
        mock_get_framework_version_from_requirements.assert_called_once_with(
            deployment_flavor, mock_requirements_path
        )
        mock_cast_to_compatible_version.assert_called_once_with(
            deployment_flavor, mock_framework_version
        )
        mock_image_uris_retrieve.assert_called_once_with(
            framework=deployment_flavor,
            region=region,
            version=mock_casted_version,
            image_scope="inference",
            py_version="py38",
            instance_type=instance_type,
        )


def test_validate_input_for_mlflow():
    _validate_input_for_mlflow(ModelServer.TORCHSERVE, "pytorch")

    with pytest.raises(ValueError):
        _validate_input_for_mlflow(ModelServer.DJL_SERVING, "pytorch")


def test_validate_input_for_mlflow_non_supported_flavor_with_tf_serving():
    with pytest.raises(ValueError):
        _validate_input_for_mlflow(ModelServer.TENSORFLOW_SERVING, "pytorch")


@patch("sagemaker.serve.model_format.mlflow.utils.shutil.copy2")
@patch("sagemaker.serve.model_format.mlflow.utils.os.makedirs")
@patch("sagemaker.serve.model_format.mlflow.utils.os.walk")
def test_copy_directory_contents_preserves_structure(
    mock_os_walk, mock_os_makedirs, mock_shutil_copy2
):
    src_dir = "/fake/source/dir"
    dest_dir = "/fake/dest/dir"

    mock_os_walk.return_value = [
        (src_dir, ["dir1"], ["file1.txt"]),
        (f"{src_dir}/dir1", [], ["file2.txt"]),
    ]

    _copy_directory_contents(src_dir, dest_dir)

    mock_os_makedirs.assert_any_call(f"{dest_dir}/dir1", exist_ok=True)

    mock_shutil_copy2.assert_any_call(f"{src_dir}/file1.txt", f"{dest_dir}/file1.txt")
    mock_shutil_copy2.assert_any_call(f"{src_dir}/dir1/file2.txt", f"{dest_dir}/dir1/file2.txt")


@patch("sagemaker.serve.model_format.mlflow.utils.shutil.copy2")
@patch("sagemaker.serve.model_format.mlflow.utils.os.makedirs")
@patch("sagemaker.serve.model_format.mlflow.utils.os.walk")
def test_copy_directory_contents_handles_empty_source_dir(
    mock_os_walk, mock_os_makedirs, mock_shutil_copy2
):
    src_dir = "/fake/empty/source/dir"
    dest_dir = "/fake/dest/dir"

    mock_os_walk.return_value = [(src_dir, [], [])]

    _copy_directory_contents(src_dir, dest_dir)

    mock_shutil_copy2.assert_not_called()


@patch("sagemaker.serve.model_format.mlflow.utils.shutil.copy2")
@patch("sagemaker.serve.model_format.mlflow.utils.os.makedirs")
@patch("sagemaker.serve.model_format.mlflow.utils.os.walk")
def test_copy_directory_contents_handles_same_src_dst(
    mock_os_walk, mock_os_makedirs, mock_shutil_copy2
):
    src_dir = "/fake/empty/source/dir"
    dest_dir = "/fake/empty/source/./dir"

    _copy_directory_contents(src_dir, dest_dir)
    mock_os_walk.assert_not_called()
    mock_os_makedirs.assert_not_called()
    mock_shutil_copy2.assert_not_called()


@patch("os.path.abspath")
@patch("os.walk")
def test_get_saved_model_path_found(mock_os_walk, mock_os_abspath):
    mock_os_walk.return_value = [
        ("/root/folder1", ("subfolder",), ()),
        ("/root/folder1/subfolder", (), (TENSORFLOW_SAVED_MODEL_NAME,)),
    ]
    expected_path = "/root/folder1/subfolder"
    mock_os_abspath.return_value = expected_path

    # Call the function
    result = _get_saved_model_path_for_tensorflow_and_keras_flavor("/root/folder1")

    # Assertions
    mock_os_walk.assert_called_once_with("/root/folder1")
    mock_os_abspath.assert_called_once_with("/root/folder1/subfolder")
    assert result == expected_path


@patch("os.path.abspath")
@patch("os.walk")
def test_get_saved_model_path_not_found(mock_os_walk, mock_os_abspath):
    mock_os_walk.return_value = [
        ("/root/folder2", ("subfolder",), ()),
        ("/root/folder2/subfolder", (), ("not_saved_model.pb",)),
    ]

    result = _get_saved_model_path_for_tensorflow_and_keras_flavor("/root/folder2")

    mock_os_walk.assert_called_once_with("/root/folder2")
    mock_os_abspath.assert_not_called()
    assert result is None


@patch("sagemaker.serve.model_format.mlflow.utils.shutil.move")
@patch("sagemaker.serve.model_format.mlflow.utils.Path.iterdir")
@patch("sagemaker.serve.model_format.mlflow.utils.Path.mkdir")
def test_move_contents_handles_same_src_dst(mock_mkdir, mock_iterdir, mock_shutil_move):
    src_dir = "/fake/source/dir"
    dest_dir = "/fake/source/./dir"

    mock_iterdir.return_value = []

    _move_contents(src_dir, dest_dir)

    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_shutil_move.assert_not_called()


@patch("sagemaker.serve.model_format.mlflow.utils.shutil.move")
@patch("sagemaker.serve.model_format.mlflow.utils.Path.iterdir")
@patch("sagemaker.serve.model_format.mlflow.utils.Path.mkdir")
def test_move_contents_with_actual_files(mock_mkdir, mock_iterdir, mock_shutil_move):
    src_dir = Path("/fake/source/dir")
    dest_dir = Path("/fake/destination/dir")

    file_path = src_dir / "testfile.txt"
    mock_iterdir.return_value = [file_path]

    _move_contents(src_dir, dest_dir)

    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_shutil_move.assert_called_once_with(str(file_path), str(dest_dir / "testfile.txt"))
