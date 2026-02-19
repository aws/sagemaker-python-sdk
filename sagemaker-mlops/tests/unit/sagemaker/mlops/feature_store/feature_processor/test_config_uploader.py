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

import pytest
from mock import Mock, patch
from sagemaker.mlops.feature_store.feature_processor._config_uploader import (
    ConfigUploader,
)
from sagemaker.mlops.feature_store.feature_processor._constants import (
    SPARK_JAR_FILES_PATH,
    SPARK_FILES_PATH,
    SPARK_PY_FILES_PATH,
    SAGEMAKER_WHL_FILE_S3_PATH,
)
from sagemaker.core.remote_function.job import (
    _JobSettings,
    RUNTIME_SCRIPTS_CHANNEL_NAME,
    REMOTE_FUNCTION_WORKSPACE,
    SPARK_CONF_CHANNEL_NAME,
)
from sagemaker.core.remote_function.spark_config import SparkConfig
from sagemaker.core.helper.session_helper import Session


@pytest.fixture
def sagemaker_session():
    return Mock(Session)


@pytest.fixture
def wrapped_func():
    return Mock()


@pytest.fixture
def runtime_env_manager():
    mocked_runtime_env_manager = Mock()
    mocked_runtime_env_manager.snapshot.return_value = "some_dependency_path"
    return mocked_runtime_env_manager


def custom_file_filter():
    pass


@pytest.fixture
def remote_decorator_config(sagemaker_session):
    return Mock(
        _JobSettings,
        sagemaker_session=sagemaker_session,
        s3_root_uri="some_s3_uri",
        s3_kms_key="some_kms",
        spark_config=SparkConfig(),
        dependencies=None,
        include_local_workdir=True,
        workdir_config=None,
        pre_execution_commands="some_commands",
        pre_execution_script="some_path",
        python_sdk_whl_s3_uri=SAGEMAKER_WHL_FILE_S3_PATH,
        custom_file_filter=None,
    )


@pytest.fixture
def config_uploader(remote_decorator_config, runtime_env_manager):
    return ConfigUploader(remote_decorator_config, runtime_env_manager)


@pytest.fixture
def remote_decorator_config_with_filter(sagemaker_session):
    return Mock(
        _JobSettings,
        sagemaker_session=sagemaker_session,
        s3_root_uri="some_s3_uri",
        s3_kms_key="some_kms",
        spark_config=SparkConfig(),
        dependencies=None,
        include_local_workdir=True,
        pre_execution_commands="some_commands",
        pre_execution_script="some_path",
        python_sdk_whl_s3_uri=SAGEMAKER_WHL_FILE_S3_PATH,
        custom_file_filter=custom_file_filter,
    )


@patch("sagemaker.mlops.feature_store.feature_processor._config_uploader.StoredFunction")
def test_prepare_and_upload_callable(mock_stored_function, config_uploader, wrapped_func):
    mock_stored_function.save(wrapped_func).return_value = None
    config_uploader._prepare_and_upload_callable(wrapped_func, "s3_base_uri", sagemaker_session)
    assert mock_stored_function.called_once_with(
        s3_base_uri="s3_base_uri",
        s3_kms_key=config_uploader.remote_decorator_config.s3_kms_key,
        sagemaker_session=sagemaker_session,
    )


@patch(
    "sagemaker.mlops.feature_store.feature_processor._config_uploader._prepare_and_upload_workspace",
    return_value="some_s3_uri",
)
def test_prepare_and_upload_workspace(mock_upload, config_uploader):
    remote_decorator_config = config_uploader.remote_decorator_config
    s3_path = config_uploader._prepare_and_upload_workspace(
        local_dependencies_path="some/path/to/dependency",
        include_local_workdir=True,
        pre_execution_commands=remote_decorator_config.pre_execution_commands,
        pre_execution_script_local_path=remote_decorator_config.pre_execution_script,
        s3_base_uri=remote_decorator_config.s3_root_uri,
        s3_kms_key=remote_decorator_config.s3_kms_key,
        sagemaker_session=sagemaker_session,
    )
    assert s3_path == mock_upload.return_value
    mock_upload.assert_called_once_with(
        local_dependencies_path="some/path/to/dependency",
        include_local_workdir=True,
        pre_execution_commands=remote_decorator_config.pre_execution_commands,
        pre_execution_script_local_path=remote_decorator_config.pre_execution_script,
        s3_base_uri=remote_decorator_config.s3_root_uri,
        s3_kms_key=remote_decorator_config.s3_kms_key,
        sagemaker_session=sagemaker_session,
        custom_file_filter=None,
    )


@patch(
    "sagemaker.mlops.feature_store.feature_processor._config_uploader._prepare_and_upload_workspace",
    return_value="some_s3_uri",
)
def test_prepare_and_upload_workspace_with_filter(
    mock_job_upload, remote_decorator_config_with_filter, runtime_env_manager
):
    config_uploader_with_filter = ConfigUploader(
        remote_decorator_config=remote_decorator_config_with_filter,
        runtime_env_manager=runtime_env_manager,
    )
    remote_decorator_config = config_uploader_with_filter.remote_decorator_config
    config_uploader_with_filter._prepare_and_upload_workspace(
        local_dependencies_path="some/path/to/dependency",
        include_local_workdir=True,
        pre_execution_commands=remote_decorator_config.pre_execution_commands,
        pre_execution_script_local_path=remote_decorator_config.pre_execution_script,
        s3_base_uri=remote_decorator_config.s3_root_uri,
        s3_kms_key=remote_decorator_config.s3_kms_key,
        sagemaker_session=sagemaker_session,
        custom_file_filter=remote_decorator_config_with_filter.custom_file_filter,
    )

    mock_job_upload.assert_called_once_with(
        local_dependencies_path="some/path/to/dependency",
        include_local_workdir=True,
        pre_execution_commands=remote_decorator_config.pre_execution_commands,
        pre_execution_script_local_path=remote_decorator_config.pre_execution_script,
        s3_base_uri=remote_decorator_config.s3_root_uri,
        s3_kms_key=remote_decorator_config.s3_kms_key,
        sagemaker_session=sagemaker_session,
        custom_file_filter=custom_file_filter,
    )


@patch(
    "sagemaker.mlops.feature_store.feature_processor._config_uploader._prepare_and_upload_runtime_scripts",
    return_value="some_s3_uri",
)
def test_prepare_and_upload_runtime_scripts(mock_upload, config_uploader):
    s3_path = config_uploader._prepare_and_upload_runtime_scripts(
        spark_config=config_uploader.remote_decorator_config.spark_config,
        s3_base_uri=config_uploader.remote_decorator_config.s3_root_uri,
        s3_kms_key=config_uploader.remote_decorator_config.s3_kms_key,
        sagemaker_session=sagemaker_session,
    )
    assert s3_path == mock_upload.return_value
    mock_upload.assert_called_once_with(
        spark_config=config_uploader.remote_decorator_config.spark_config,
        s3_base_uri=config_uploader.remote_decorator_config.s3_root_uri,
        s3_kms_key=config_uploader.remote_decorator_config.s3_kms_key,
        sagemaker_session=sagemaker_session,
    )


@patch(
    "sagemaker.mlops.feature_store.feature_processor._config_uploader._prepare_and_upload_spark_dependent_files",
    return_value=("path_a", "path_b", "path_c", "path_d"),
)
def test_prepare_and_upload_spark_dependent_files(mock_upload, config_uploader):
    s3_paths = config_uploader._prepare_and_upload_spark_dependent_files(
        spark_config=config_uploader.remote_decorator_config.spark_config,
        s3_base_uri=config_uploader.remote_decorator_config.s3_root_uri,
        s3_kms_key=config_uploader.remote_decorator_config.s3_kms_key,
        sagemaker_session=sagemaker_session,
    )
    assert s3_paths == mock_upload.return_value
    mock_upload.assert_called_once_with(
        spark_config=config_uploader.remote_decorator_config.spark_config,
        s3_base_uri=config_uploader.remote_decorator_config.s3_root_uri,
        s3_kms_key=config_uploader.remote_decorator_config.s3_kms_key,
        sagemaker_session=sagemaker_session,
    )


@patch("sagemaker.mlops.feature_store.feature_processor._config_uploader.Channel")
@patch("sagemaker.mlops.feature_store.feature_processor._config_uploader.DataSource")
@patch("sagemaker.mlops.feature_store.feature_processor._config_uploader.S3DataSource")
@patch(
    "sagemaker.mlops.feature_store.feature_processor._config_uploader._prepare_and_upload_spark_dependent_files",
    return_value=("path_a", "path_b", "path_c", "path_d"),
)
@patch(
    "sagemaker.mlops.feature_store.feature_processor._config_uploader._prepare_and_upload_workspace",
    return_value="some_s3_uri",
)
@patch(
    "sagemaker.mlops.feature_store.feature_processor._config_uploader._prepare_and_upload_runtime_scripts",
    return_value="some_s3_uri",
)
@patch("sagemaker.mlops.feature_store.feature_processor._config_uploader.StoredFunction")
def test_prepare_step_input_channel(
    mock_upload_callable,
    mock_script_upload,
    mock_dependency_upload,
    mock_spark_dependency_upload,
    mock_s3_data_source,
    mock_data_source,
    mock_channel,
    config_uploader,
    wrapped_func,
):
    (
        input_data_config,
        spark_dependency_paths,
    ) = config_uploader.prepare_step_input_channel_for_spark_mode(
        wrapped_func,
        config_uploader.remote_decorator_config.s3_root_uri,
        sagemaker_session,
    )
    remote_decorator_config = config_uploader.remote_decorator_config

    assert mock_upload_callable.called_once_with(wrapped_func)

    mock_script_upload.assert_called_once_with(
        spark_config=config_uploader.remote_decorator_config.spark_config,
        s3_base_uri=config_uploader.remote_decorator_config.s3_root_uri,
        s3_kms_key="some_kms",
        sagemaker_session=sagemaker_session,
    )

    mock_dependency_upload.assert_called_once_with(
        local_dependencies_path="some_dependency_path",
        include_local_workdir=True,
        pre_execution_commands=remote_decorator_config.pre_execution_commands,
        pre_execution_script_local_path=remote_decorator_config.pre_execution_script,
        s3_base_uri=remote_decorator_config.s3_root_uri,
        s3_kms_key="some_kms",
        sagemaker_session=sagemaker_session,
        custom_file_filter=None,
    )

    mock_spark_dependency_upload.assert_called_once_with(
        spark_config=config_uploader.remote_decorator_config.spark_config,
        s3_base_uri=config_uploader.remote_decorator_config.s3_root_uri,
        s3_kms_key="some_kms",
        sagemaker_session=sagemaker_session,
    )

    # Verify input_data_config is a list of Channel objects
    assert isinstance(input_data_config, list)
    # 3 channels: runtime scripts, workspace, spark conf
    assert len(input_data_config) == 3

    # Verify each channel was constructed with the correct data
    # Channel 1: runtime scripts
    mock_s3_data_source.assert_any_call(
        s3_uri="some_s3_uri",
        s3_data_type="S3Prefix",
        s3_data_distribution_type="FullyReplicated",
    )
    # Channel 2: workspace
    mock_s3_data_source.assert_any_call(
        s3_uri=f"{config_uploader.remote_decorator_config.s3_root_uri}/sm_rf_user_ws",
        s3_data_type="S3Prefix",
        s3_data_distribution_type="FullyReplicated",
    )
    # Channel 3: spark conf
    mock_s3_data_source.assert_any_call(
        s3_uri="path_d",
        s3_data_type="S3Prefix",
        s3_data_distribution_type="FullyReplicated",
    )

    assert mock_s3_data_source.call_count == 3
    assert mock_data_source.call_count == 3
    assert mock_channel.call_count == 3

    # Verify channel names and input_mode
    channel_call_kwargs = [call.kwargs for call in mock_channel.call_args_list]
    channel_names = [kw["channel_name"] for kw in channel_call_kwargs]
    assert RUNTIME_SCRIPTS_CHANNEL_NAME in channel_names
    assert REMOTE_FUNCTION_WORKSPACE in channel_names
    assert SPARK_CONF_CHANNEL_NAME in channel_names
    for kw in channel_call_kwargs:
        assert kw["input_mode"] == "File"

    assert spark_dependency_paths == {
        SPARK_JAR_FILES_PATH: "path_a",
        SPARK_PY_FILES_PATH: "path_b",
        SPARK_FILES_PATH: "path_c",
    }
