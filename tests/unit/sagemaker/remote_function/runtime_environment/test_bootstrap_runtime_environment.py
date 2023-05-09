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

from mock import patch, Mock
from sagemaker.remote_function.runtime_environment.runtime_environment_manager import (
    RuntimeEnvironmentError,
)

import sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment as bootstrap
import pathlib

TEST_JOB_CONDA_ENV = "conda_env"
CURR_WORKING_DIR = "/user/set/workdir"
TEST_DEPENDENCIES_PATH = "/user/set/workdir/sagemaker_remote_function_workspace"
TEST_PYTHON_VERSION = "3.10"
TEST_WORKSPACE_ARCHIVE_DIR_PATH = "/opt/ml/input/data/sm_rf_user_ws"
TEST_WORKSPACE_ARCHIVE_PATH = "/opt/ml/input/data/sm_rf_user_ws/workspace.zip"


def mock_args():
    args = Mock()
    args.job_conda_env = TEST_JOB_CONDA_ENV
    args.client_python_version = TEST_PYTHON_VERSION
    return args


@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment._parse_agrs",
    new=mock_args,
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager._validate_python_version"
)
@patch("sys.exit")
@patch("shutil.unpack_archive", Mock())
@patch("os.getcwd", return_value=CURR_WORKING_DIR)
@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
@patch("os.listdir", return_value=["fileA.py", "fileB.sh", "requirements.txt"])
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager.run_pre_exec_script"
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager.bootstrap"
)
def test_main_success(
    bootstrap_runtime,
    run_pre_exec_script,
    list_dir,
    file_exists,
    path_exists,
    getcwd,
    _exit_process,
    validate_python,
):
    bootstrap.main()
    validate_python.assert_called_once_with(TEST_PYTHON_VERSION, TEST_JOB_CONDA_ENV)
    path_exists.assert_called_once_with(TEST_WORKSPACE_ARCHIVE_DIR_PATH)
    file_exists.assert_called_once_with(TEST_WORKSPACE_ARCHIVE_PATH)
    getcwd.assert_called()
    list_dir.assert_called_once_with(pathlib.Path(TEST_DEPENDENCIES_PATH))
    run_pre_exec_script.assert_called(),
    bootstrap_runtime.assert_called()
    _exit_process.assert_called_with(0)


@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment._parse_agrs",
    new=mock_args,
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager._validate_python_version"
)
@patch("sys.exit")
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment._write_failure_reason_file"
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager.run_pre_exec_script"
)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment._bootstrap_runtime_environment"
)
def test_main_failure(
    bootstrap_runtime, run_pre_exec_script, write_failure, _exit_process, validate_python
):
    runtime_err = RuntimeEnvironmentError("some failure reason")
    bootstrap_runtime.side_effect = runtime_err

    bootstrap.main()

    validate_python.assert_called_once_with(TEST_PYTHON_VERSION, TEST_JOB_CONDA_ENV)
    run_pre_exec_script.assert_not_called()
    bootstrap_runtime.assert_called()
    write_failure.assert_called_with(str(runtime_err))
    _exit_process.assert_called_with(1)


@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment._parse_agrs",
    new=mock_args,
)
@patch("sys.exit")
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager._validate_python_version"
)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment._write_failure_reason_file"
)
@patch("os.path.exists", return_value=False)
def test_main_channel_folder_does_not_exist(
    path_exists, write_failure, validate_python, _exit_process
):
    bootstrap.main()
    path_exists.assert_called_once_with(TEST_WORKSPACE_ARCHIVE_DIR_PATH)
    validate_python.assert_called_once_with(TEST_PYTHON_VERSION, TEST_JOB_CONDA_ENV)
    write_failure.assert_not_called()
    _exit_process.assert_called_with(0)


@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment._parse_agrs",
    new=mock_args,
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager"
    ".RuntimeEnvironmentManager._validate_python_version"
)
@patch("sys.exit")
@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=False)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager.run_pre_exec_script"
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager.bootstrap"
)
def test_main_no_workspace_archive(
    bootstrap_runtime, run_pre_exec_script, file_exists, path_exists, _exit_process, validate_python
):
    bootstrap.main()
    validate_python.assert_called_once_with(TEST_PYTHON_VERSION, TEST_JOB_CONDA_ENV)
    path_exists.assert_called_once_with(TEST_WORKSPACE_ARCHIVE_DIR_PATH)
    file_exists.assert_called_once_with(TEST_WORKSPACE_ARCHIVE_PATH)
    run_pre_exec_script.assert_not_called()
    bootstrap_runtime.assert_not_called()
    _exit_process.assert_called_with(0)


@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment._parse_agrs",
    new=mock_args,
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager._validate_python_version"
)
@patch("sys.exit")
@patch("shutil.unpack_archive", Mock())
@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
@patch("os.getcwd", return_value=CURR_WORKING_DIR)
@patch("os.listdir", return_value=["fileA.py", "fileB.sh"])
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager.run_pre_exec_script"
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager.RuntimeEnvironmentManager.bootstrap"
)
def test_main_no_dependency_file(
    bootstrap_runtime,
    run_pre_exec_script,
    list_dir,
    get_cwd,
    file_exists,
    path_exists,
    _exit_process,
    validate_python,
):
    bootstrap.main()
    validate_python.assert_called_once_with(TEST_PYTHON_VERSION, TEST_JOB_CONDA_ENV)
    path_exists.assert_called_once_with(TEST_WORKSPACE_ARCHIVE_DIR_PATH)
    file_exists.assert_called_once_with(TEST_WORKSPACE_ARCHIVE_PATH)
    get_cwd.assert_called_once()
    list_dir.assert_called_once_with(pathlib.Path(TEST_DEPENDENCIES_PATH))
    run_pre_exec_script.assert_called()
    bootstrap_runtime.assert_not_called()
    _exit_process.assert_called_with(0)
