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

from mock import patch
from mock.mock import MagicMock

from sagemaker.remote_function.runtime_environment.runtime_environment_manager import (
    RuntimeEnvironmentError,
    _DependencySettings,
)

import sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment as bootstrap
import pathlib

TEST_JOB_CONDA_ENV = "conda_env"
CURR_WORKING_DIR = "/user/set/workdir"
TEST_DEPENDENCIES_PATH = "/user/set/workdir/sagemaker_remote_function_workspace"
TEST_PYTHON_VERSION = "3.10"
TEST_SAGEMAKER_PYSDK_VERSION = "2.205.0"
TEST_WORKSPACE_ARCHIVE_DIR_PATH = "/opt/ml/input/data/sm_rf_user_ws"
TEST_WORKSPACE_ARCHIVE_PATH = "/opt/ml/input/data/sm_rf_user_ws/workspace.zip"
TEST_EXECUTION_ID = "test_execution_id"
TEST_BASE_CHANNEL_PATH = "/opt/ml/input/data/"
REMOTE_FUNCTION_CHANNEL = "sm_rf_user_ws"
PIPELINE_STEP_CHANNEL = "pre_exec_script_and_dependencies"
TEST_DEPENDENCY_FILE_NAME = "requirements.txt"
PRE_EXECUTION_SCRIPT_NAME = "pre_exec.sh"
FUNC_STEP_WORKSPACE = "workspace_folder"


def args_for_remote():
    return [
        "--job_conda_env",
        TEST_JOB_CONDA_ENV,
        "--client_python_version",
        TEST_PYTHON_VERSION,
        "--client_sagemaker_pysdk_version",
        TEST_SAGEMAKER_PYSDK_VERSION,
        "--dependency_settings",
        _DependencySettings(TEST_DEPENDENCY_FILE_NAME).to_string(),
    ]


def args_for_step():
    return [
        "--job_conda_env",
        TEST_JOB_CONDA_ENV,
        "--client_python_version",
        TEST_PYTHON_VERSION,
        "--client_sagemaker_pysdk_version",
        TEST_SAGEMAKER_PYSDK_VERSION,
        "--pipeline_execution_id",
        TEST_EXECUTION_ID,
        "--func_step_s3_dir",
        FUNC_STEP_WORKSPACE,
    ]


@patch("sys.exit")
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager._validate_sagemaker_pysdk_version"
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager._validate_python_version"
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager.bootstrap"
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager.run_pre_exec_script"
)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment."
    "_bootstrap_runtime_env_for_remote_function"
)
@patch("getpass.getuser", MagicMock(return_value="root"))
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager.change_dir_permission"
)
def test_main_success_remote_job_with_root_user(
    change_dir_permission,
    bootstrap_remote,
    run_pre_exec_script,
    bootstrap_runtime,
    validate_python,
    validate_sagemaker,
    _exit_process,
):
    bootstrap.main(args_for_remote())

    change_dir_permission.assert_not_called()
    validate_python.assert_called_once_with(TEST_PYTHON_VERSION, TEST_JOB_CONDA_ENV)
    validate_sagemaker.assert_called_once_with(TEST_SAGEMAKER_PYSDK_VERSION)
    bootstrap_remote.assert_called_once_with(
        TEST_PYTHON_VERSION,
        TEST_JOB_CONDA_ENV,
        _DependencySettings(TEST_DEPENDENCY_FILE_NAME),
    )
    run_pre_exec_script.assert_not_called()
    bootstrap_runtime.assert_not_called()
    _exit_process.assert_called_with(0)


@patch("sys.exit")
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager._validate_sagemaker_pysdk_version"
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager._validate_python_version"
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager.bootstrap"
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager.run_pre_exec_script"
)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment."
    "_bootstrap_runtime_env_for_remote_function"
)
@patch("getpass.getuser", MagicMock(return_value="root"))
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager.change_dir_permission"
)
def test_main_success_with_obsoleted_args_that_missing_sagemaker_version(
    change_dir_permission,
    bootstrap_remote,
    run_pre_exec_script,
    bootstrap_runtime,
    validate_python,
    validate_sagemaker,
    _exit_process,
):
    # This test is to test the backward compatibility
    # In old version of SDK, the client side sagemaker_pysdk_version is not passed to job
    # thus it would be None and would not lead to the warning
    obsoleted_args = [
        "--job_conda_env",
        TEST_JOB_CONDA_ENV,
        "--client_python_version",
        TEST_PYTHON_VERSION,
        "--dependency_settings",
        _DependencySettings(TEST_DEPENDENCY_FILE_NAME).to_string(),
    ]
    bootstrap.main(obsoleted_args)

    change_dir_permission.assert_not_called()
    validate_python.assert_called_once_with(TEST_PYTHON_VERSION, TEST_JOB_CONDA_ENV)
    validate_sagemaker.assert_called_once_with(None)
    bootstrap_remote.assert_called_once_with(
        TEST_PYTHON_VERSION,
        TEST_JOB_CONDA_ENV,
        _DependencySettings(TEST_DEPENDENCY_FILE_NAME),
    )
    run_pre_exec_script.assert_not_called()
    bootstrap_runtime.assert_not_called()
    _exit_process.assert_called_with(0)


@patch("sys.exit")
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager._validate_sagemaker_pysdk_version"
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager._validate_python_version"
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager.bootstrap"
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager.run_pre_exec_script"
)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment."
    "_bootstrap_runtime_env_for_pipeline_step"
)
@patch("getpass.getuser", MagicMock(return_value="root"))
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager.change_dir_permission"
)
def test_main_success_pipeline_step_with_root_user(
    change_dir_permission,
    bootstrap_step,
    run_pre_exec_script,
    bootstrap_runtime,
    validate_python,
    validate_sagemaker,
    _exit_process,
):
    bootstrap.main(args_for_step())
    change_dir_permission.assert_not_called()
    validate_python.assert_called_once_with(TEST_PYTHON_VERSION, TEST_JOB_CONDA_ENV)
    validate_sagemaker.assert_called_once_with(TEST_SAGEMAKER_PYSDK_VERSION)
    bootstrap_step.assert_called_once_with(
        TEST_PYTHON_VERSION,
        FUNC_STEP_WORKSPACE,
        TEST_JOB_CONDA_ENV,
        None,
    )
    run_pre_exec_script.assert_not_called()
    bootstrap_runtime.assert_not_called()
    _exit_process.assert_called_with(0)


@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager._validate_sagemaker_pysdk_version"
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
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment."
    "_bootstrap_runtime_env_for_remote_function"
)
@patch("getpass.getuser", MagicMock(return_value="root"))
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager.change_dir_permission"
)
def test_main_failure_remote_job_with_root_user(
    change_dir_permission,
    bootstrap_runtime,
    run_pre_exec_script,
    write_failure,
    _exit_process,
    validate_python,
    validate_sagemaker,
):
    runtime_err = RuntimeEnvironmentError("some failure reason")
    bootstrap_runtime.side_effect = runtime_err

    bootstrap.main(args_for_remote())

    change_dir_permission.assert_not_called()
    validate_python.assert_called_once_with(TEST_PYTHON_VERSION, TEST_JOB_CONDA_ENV)
    validate_sagemaker.assert_called_once_with(TEST_SAGEMAKER_PYSDK_VERSION)
    run_pre_exec_script.assert_not_called()
    bootstrap_runtime.assert_called()
    write_failure.assert_called_with(str(runtime_err))
    _exit_process.assert_called_with(1)


@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager._validate_sagemaker_pysdk_version"
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
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment."
    "_bootstrap_runtime_env_for_pipeline_step"
)
@patch("getpass.getuser", MagicMock(return_value="root"))
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager.change_dir_permission"
)
def test_main_failure_pipeline_step_with_root_user(
    change_dir_permission,
    bootstrap_runtime,
    run_pre_exec_script,
    write_failure,
    _exit_process,
    validate_python,
    validate_sagemaker,
):
    runtime_err = RuntimeEnvironmentError("some failure reason")
    bootstrap_runtime.side_effect = runtime_err

    bootstrap.main(args_for_step())

    change_dir_permission.assert_not_called()
    validate_python.assert_called_once_with(TEST_PYTHON_VERSION, TEST_JOB_CONDA_ENV)
    validate_sagemaker.assert_called_once_with(TEST_SAGEMAKER_PYSDK_VERSION)
    run_pre_exec_script.assert_not_called()
    bootstrap_runtime.assert_called()
    write_failure.assert_called_with(str(runtime_err))
    _exit_process.assert_called_with(1)


@patch("sys.exit")
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager._validate_sagemaker_pysdk_version"
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager._validate_python_version"
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager.bootstrap"
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager.run_pre_exec_script"
)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment."
    "_bootstrap_runtime_env_for_remote_function"
)
@patch("getpass.getuser", MagicMock(return_value="non_root"))
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager.change_dir_permission"
)
def test_main_remote_job_with_non_root_user(
    change_dir_permission,
    bootstrap_remote,
    run_pre_exec_script,
    bootstrap_runtime,
    validate_python,
    validate_sagemaker,
    _exit_process,
):
    bootstrap.main(args_for_remote())

    change_dir_permission.assert_called_once_with(
        dirs=bootstrap.JOB_OUTPUT_DIRS, new_permission="777"
    )
    validate_python.assert_called_once_with(TEST_PYTHON_VERSION, TEST_JOB_CONDA_ENV)
    validate_sagemaker.assert_called_once_with(TEST_SAGEMAKER_PYSDK_VERSION)
    bootstrap_remote.assert_called_once_with(
        TEST_PYTHON_VERSION,
        TEST_JOB_CONDA_ENV,
        _DependencySettings(TEST_DEPENDENCY_FILE_NAME),
    )
    run_pre_exec_script.assert_not_called()
    bootstrap_runtime.assert_not_called()
    _exit_process.assert_called_with(0)


@patch("sys.exit")
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager._validate_sagemaker_pysdk_version"
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager._validate_python_version"
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager.bootstrap"
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager.run_pre_exec_script"
)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment."
    "_bootstrap_runtime_env_for_pipeline_step"
)
@patch("getpass.getuser", MagicMock(return_value="non_root"))
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager.change_dir_permission"
)
def test_main_pipeline_step_with_non_root_user(
    change_dir_permission,
    bootstrap_step,
    run_pre_exec_script,
    bootstrap_runtime,
    validate_python,
    validate_sagemaker,
    _exit_process,
):
    bootstrap.main(args_for_step())

    change_dir_permission.assert_called_once_with(
        dirs=bootstrap.JOB_OUTPUT_DIRS, new_permission="777"
    )
    validate_python.assert_called_once_with(TEST_PYTHON_VERSION, TEST_JOB_CONDA_ENV)
    validate_sagemaker.assert_called_once_with(TEST_SAGEMAKER_PYSDK_VERSION)
    bootstrap_step.assert_called_once_with(
        TEST_PYTHON_VERSION,
        FUNC_STEP_WORKSPACE,
        TEST_JOB_CONDA_ENV,
        None,
    )
    run_pre_exec_script.assert_not_called()
    bootstrap_runtime.assert_not_called()
    _exit_process.assert_called_with(0)


@patch("shutil.unpack_archive")
@patch("os.getcwd", return_value=CURR_WORKING_DIR)
@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_unpack_user_workspace(file_exists, path_exists, get_cwd, archive):
    expected_workspace_path = pathlib.Path(TEST_DEPENDENCIES_PATH)

    actual_workspace_path = bootstrap._unpack_user_workspace()
    assert expected_workspace_path == actual_workspace_path

    path_exists.assert_called_once_with(TEST_WORKSPACE_ARCHIVE_DIR_PATH)
    file_exists.assert_called_once_with(TEST_WORKSPACE_ARCHIVE_PATH)
    get_cwd.assert_called()
    archive.assert_called()


@patch("os.listdir", return_value=[TEST_DEPENDENCY_FILE_NAME])
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager"
    ".RuntimeEnvironmentManager.bootstrap"
)
def test_install_dependencies_with_requirements_txt(bootstrap_runtime, list_dir):
    dependency_file = TEST_DEPENDENCIES_PATH + "/" + TEST_DEPENDENCY_FILE_NAME
    bootstrap._install_dependencies(
        TEST_DEPENDENCIES_PATH, TEST_JOB_CONDA_ENV, TEST_PYTHON_VERSION, REMOTE_FUNCTION_CHANNEL
    )
    list_dir.assert_called_once_with(TEST_DEPENDENCIES_PATH)
    bootstrap_runtime.assert_called_once_with(
        local_dependencies_file=dependency_file,
        conda_env=TEST_JOB_CONDA_ENV,
        client_python_version=TEST_PYTHON_VERSION,
    )


@patch("os.listdir", return_value=["conda_file.yaml"])
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager"
    ".RuntimeEnvironmentManager.bootstrap"
)
def test_install_dependencies_with_conda_yml(bootstrap_runtime, list_dir):
    dependency_file = TEST_DEPENDENCIES_PATH + "/conda_file.yaml"
    bootstrap._install_dependencies(
        TEST_DEPENDENCIES_PATH, TEST_JOB_CONDA_ENV, TEST_PYTHON_VERSION, REMOTE_FUNCTION_CHANNEL
    )
    list_dir.assert_called_once_with(TEST_DEPENDENCIES_PATH)
    bootstrap_runtime.assert_called_once_with(
        local_dependencies_file=dependency_file,
        conda_env=TEST_JOB_CONDA_ENV,
        client_python_version=TEST_PYTHON_VERSION,
    )


@patch("os.listdir", return_value=["requirements.py"])
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager"
    ".RuntimeEnvironmentManager.bootstrap"
)
def test_install_dependencies_no_dependencies_file_found(bootstrap_runtime, list_dir):
    bootstrap._install_dependencies(
        TEST_DEPENDENCIES_PATH, TEST_JOB_CONDA_ENV, TEST_PYTHON_VERSION, REMOTE_FUNCTION_CHANNEL
    )
    list_dir.assert_called_once_with(TEST_DEPENDENCIES_PATH)
    bootstrap_runtime.assert_not_called()


@patch("os.listdir", return_value=[])
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager"
    ".RuntimeEnvironmentManager.bootstrap"
)
def test_install_dependencies_no_file_exists(bootstrap_runtime, list_dir):
    bootstrap._install_dependencies(
        TEST_DEPENDENCIES_PATH, TEST_JOB_CONDA_ENV, TEST_PYTHON_VERSION, REMOTE_FUNCTION_CHANNEL
    )
    list_dir.assert_called_once_with(TEST_DEPENDENCIES_PATH)
    bootstrap_runtime.assert_not_called()


@patch("os.listdir", return_value=[TEST_DEPENDENCY_FILE_NAME])
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager"
    ".RuntimeEnvironmentManager.bootstrap"
)
def test_install_dependencies_filename_provided(bootstrap_runtime, list_dir):
    dependency_file = TEST_DEPENDENCIES_PATH + "/" + TEST_DEPENDENCY_FILE_NAME
    bootstrap._install_dependencies(
        TEST_DEPENDENCIES_PATH,
        TEST_JOB_CONDA_ENV,
        TEST_PYTHON_VERSION,
        REMOTE_FUNCTION_CHANNEL,
        _DependencySettings(TEST_DEPENDENCY_FILE_NAME),
    )
    list_dir.assert_not_called()
    bootstrap_runtime.assert_called_once_with(
        local_dependencies_file=dependency_file,
        conda_env=TEST_JOB_CONDA_ENV,
        client_python_version=TEST_PYTHON_VERSION,
    )


@patch("os.listdir", return_value=[TEST_DEPENDENCY_FILE_NAME])
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager"
    ".RuntimeEnvironmentManager.bootstrap"
)
def test_empty_dependencies_filename_provided(bootstrap_runtime, list_dir):
    bootstrap._install_dependencies(
        TEST_DEPENDENCIES_PATH,
        TEST_JOB_CONDA_ENV,
        TEST_PYTHON_VERSION,
        REMOTE_FUNCTION_CHANNEL,
        _DependencySettings(),
    )
    list_dir.assert_not_called()
    bootstrap_runtime.assert_not_called()


@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager.run_pre_exec_script"
)
def test_handle_pre_exec_scripts(run_scrips):
    bootstrap._handle_pre_exec_scripts(TEST_DEPENDENCIES_PATH)
    run_scrips.assert_called_once_with(pre_exec_script_path=TEST_DEPENDENCIES_PATH + "/pre_exec.sh")


@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment._install_dependencies"
)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment._handle_pre_exec_scripts"
)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment._unpack_user_workspace",
    return_value=pathlib.Path(TEST_DEPENDENCIES_PATH),
)
def test_bootstrap_runtime_env_for_remote(unpack_workspace, install_scripts, install_dependencies):
    bootstrap._bootstrap_runtime_env_for_remote_function(
        TEST_PYTHON_VERSION,
        TEST_JOB_CONDA_ENV,
        TEST_DEPENDENCY_FILE_NAME,
    )

    unpack_workspace.assert_called_once()
    install_scripts.assert_called_once_with(unpack_workspace.return_value)
    install_dependencies.assert_called_once_with(
        unpack_workspace.return_value,
        TEST_JOB_CONDA_ENV,
        TEST_PYTHON_VERSION,
        REMOTE_FUNCTION_CHANNEL,
        TEST_DEPENDENCY_FILE_NAME,
    )


@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment._install_dependencies"
)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment._handle_pre_exec_scripts"
)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment._unpack_user_workspace",
    return_value=None,
)
def test_bootstrap_runtime_env_for_remote_no_workspace(
    unpack_workspace, install_scripts, install_dependencies
):
    bootstrap._bootstrap_runtime_env_for_remote_function(TEST_PYTHON_VERSION, TEST_JOB_CONDA_ENV)

    unpack_workspace.assert_called_once()
    install_scripts.assert_not_called()
    install_dependencies.assert_not_called()


@patch("shutil.copy")
@patch("os.listdir", return_value=[PRE_EXECUTION_SCRIPT_NAME, TEST_DEPENDENCY_FILE_NAME])
@patch("os.path.exists", return_value=True)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment._install_dependencies"
)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment._handle_pre_exec_scripts"
)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment._unpack_user_workspace",
    return_value=pathlib.Path(TEST_DEPENDENCIES_PATH),
)
def test_bootstrap_runtime_env_for_step(
    unpack_workspace, install_scripts, install_dependencies, path_exists, list_dir, copy_files
):
    bootstrap._bootstrap_runtime_env_for_pipeline_step(
        TEST_PYTHON_VERSION,
        FUNC_STEP_WORKSPACE,
        TEST_JOB_CONDA_ENV,
        TEST_DEPENDENCY_FILE_NAME,
    )

    unpack_workspace.assert_called_once()
    dependency_dir = TEST_BASE_CHANNEL_PATH + PIPELINE_STEP_CHANNEL
    path_exists.assert_called_once_with(dependency_dir)
    list_dir.assert_called_once_with(dependency_dir)
    assert copy_files.call_count == 2
    install_scripts.assert_called_once_with(unpack_workspace.return_value)
    install_dependencies.assert_called_once_with(
        unpack_workspace.return_value,
        TEST_JOB_CONDA_ENV,
        TEST_PYTHON_VERSION,
        PIPELINE_STEP_CHANNEL,
        TEST_DEPENDENCY_FILE_NAME,
    )


@patch("os.getcwd", return_value=CURR_WORKING_DIR)
@patch("os.mkdir")
@patch("shutil.copy")
@patch("os.listdir", return_value=[PRE_EXECUTION_SCRIPT_NAME, TEST_DEPENDENCY_FILE_NAME])
@patch("os.path.exists", return_value=True)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment._install_dependencies"
)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment._handle_pre_exec_scripts"
)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment._unpack_user_workspace",
    return_value=None,
)
def test_bootstrap_runtime_env_for_step_no_workspace(
    unpack_workspace,
    install_scripts,
    install_dependencies,
    path_exists,
    list_dir,
    copy_files,
    mkdir,
    get_cwd,
):
    bootstrap._bootstrap_runtime_env_for_pipeline_step(
        TEST_PYTHON_VERSION,
        None,
        TEST_JOB_CONDA_ENV,
        TEST_DEPENDENCY_FILE_NAME,
    )

    unpack_workspace.assert_called_once()
    mkdir.assert_called_once_with("sagemaker_remote_function_workspace")
    get_cwd.assert_called_once()

    dependency_dir = TEST_BASE_CHANNEL_PATH + PIPELINE_STEP_CHANNEL
    path_exists.assert_called_once_with(dependency_dir)
    list_dir.assert_called_once_with(dependency_dir)
    assert copy_files.call_count == 2
    install_scripts.assert_called_once_with(pathlib.Path(TEST_DEPENDENCIES_PATH))
    install_dependencies.assert_called_once_with(
        pathlib.Path(TEST_DEPENDENCIES_PATH),
        TEST_JOB_CONDA_ENV,
        TEST_PYTHON_VERSION,
        PIPELINE_STEP_CHANNEL,
        TEST_DEPENDENCY_FILE_NAME,
    )


@patch("os.getcwd", return_value=CURR_WORKING_DIR)
@patch("os.mkdir")
@patch("os.path.exists", return_value=False)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment._install_dependencies"
)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment._handle_pre_exec_scripts"
)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment._unpack_user_workspace",
    return_value=None,
)
def test_bootstrap_runtime_env_for_step_no_dependencies_dir(
    unpack_workspace, install_scripts, install_dependencies, path_exists, mkdir, get_cwd
):
    bootstrap._bootstrap_runtime_env_for_pipeline_step(
        TEST_PYTHON_VERSION,
        TEST_JOB_CONDA_ENV,
        TEST_DEPENDENCY_FILE_NAME,
    )

    unpack_workspace.assert_called_once()
    mkdir.assert_called_once_with("sagemaker_remote_function_workspace")
    get_cwd.assert_called_once()
    dependency_dir = TEST_BASE_CHANNEL_PATH + PIPELINE_STEP_CHANNEL
    path_exists.assert_called_once_with(dependency_dir)
    install_scripts.assert_not_called()
    install_dependencies.assert_not_called()
