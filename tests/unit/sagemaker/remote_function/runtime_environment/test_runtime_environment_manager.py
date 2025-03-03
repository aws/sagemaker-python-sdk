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

import subprocess

import pytest
from mock import patch, Mock
import sys
import shlex
import os

from mock.mock import MagicMock

from sagemaker.remote_function.runtime_environment.runtime_environment_manager import (
    RuntimeEnvironmentManager,
    RuntimeEnvironmentError,
)

TEST_REQUIREMENTS_TXT = "usr/local/requirements.txt"
TEST_CONDA_YML = "usr/local/conda_env.yml"
CLIENT_PYTHON_VERSION = "3.10"
JOB_SAGEMAKER_PYSDK_VERSION = "2.205.0"


def test_snapshot_no_dependencies():
    response = RuntimeEnvironmentManager().snapshot(dependencies=None)
    assert response is None


@patch("os.path.isfile", return_value=True)
def test_snapshot_with_requirements_txt(isfile):
    response = RuntimeEnvironmentManager().snapshot(TEST_REQUIREMENTS_TXT)
    isfile.assert_called_once_with(TEST_REQUIREMENTS_TXT)
    assert response == TEST_REQUIREMENTS_TXT


@patch("os.path.isfile", return_value=True)
def test_snapshot_with_conda_yml(isfile):
    response = RuntimeEnvironmentManager().snapshot(TEST_CONDA_YML)
    isfile.assert_called_once_with(TEST_CONDA_YML)
    assert response == TEST_CONDA_YML


@patch("os.path.isfile", return_value=False)
def test_snapshot_file_not_exists(isfile):
    with pytest.raises(ValueError):
        RuntimeEnvironmentManager().snapshot(TEST_REQUIREMENTS_TXT)

    isfile.assert_called_once_with(TEST_REQUIREMENTS_TXT)


def test_snapshot_invalid_depedencies():

    # scenario 1: invalid file format
    invalid_dependencies_file = "usr/local/requirements.py"
    with pytest.raises(ValueError):
        RuntimeEnvironmentManager().snapshot(invalid_dependencies_file)

    # scenario 2: invalid keyword
    invalid_dependencies = "from_some_invalid_keyword"
    with pytest.raises(ValueError):
        RuntimeEnvironmentManager().snapshot(invalid_dependencies)


def test__get_conda_env_name():
    with patch("os.getenv") as getenv_patch:
        getenv_patch.return_value = "some-conda-env-name"

        result = RuntimeEnvironmentManager()._get_active_conda_env_name()

        assert result == "some-conda-env-name"
        call_arg = getenv_patch.call_args[0][0]
        assert call_arg == "CONDA_DEFAULT_ENV"


def test__get_active_conda_env_prefix():
    with patch("os.getenv") as getenv_patch:
        getenv_patch.return_value = "some-conda-prefix"

        result = RuntimeEnvironmentManager()._get_active_conda_env_prefix()

        assert result == "some-conda-prefix"
        call_arg = getenv_patch.call_args[0][0]
        assert call_arg == "CONDA_PREFIX"


@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager._log_error", Mock()
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager._log_output", Mock()
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager"
    ".RuntimeEnvironmentManager._get_conda_exe",
    return_value="some_exe",
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager"
    ".RuntimeEnvironmentManager._get_active_conda_env_name",
    return_value="test_env",
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager"
    ".RuntimeEnvironmentManager._get_active_conda_env_prefix",
    return_value="/some/conda/env/prefix",
)
def test_snapshot_from_active_conda_env_when_name_available(
    conda_env_prefix, conda_default_env, stub_conda_exe
):
    expected_result = os.path.join(os.getcwd(), "env_snapshot.yml")
    with patch("subprocess.Popen") as popen:
        popen.return_value.wait.return_value = 0

        result = RuntimeEnvironmentManager().snapshot("auto_capture")
        assert result == expected_result

        call_args = popen.call_args[0][0]
        assert call_args is not None
        expected_cmd = (
            f"{stub_conda_exe.return_value} env export -p {conda_env_prefix.return_value} "
            f"--no-builds > {expected_result}"
        )
        assert call_args == expected_cmd


@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager._log_error", Mock()
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager._log_output", Mock()
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager"
    ".RuntimeEnvironmentManager._get_conda_exe",
    return_value="conda",
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager"
    ".RuntimeEnvironmentManager._get_active_conda_env_name",
    return_value=None,
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager"
    ".RuntimeEnvironmentManager._get_active_conda_env_prefix",
    return_value="/some/conda/env/prefix",
)
def test_snapshot_from_active_conda_env_when_prefix_available(
    conda_env_prefix, no_conda_env_name, conda_env
):
    expected_result = os.path.join(os.getcwd(), "env_snapshot.yml")
    with patch("subprocess.Popen") as popen:
        popen.return_value.wait.return_value = 0

        result = RuntimeEnvironmentManager().snapshot("auto_capture")
        assert result == expected_result

        call_args = popen.call_args[0][0]
        assert call_args is not None
        expected_cmd = "{} env export -p {} --no-builds > {}".format(
            conda_env.return_value, conda_env_prefix.return_value, expected_result
        )
        assert call_args == expected_cmd


@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager._log_error", Mock()
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager._log_output", Mock()
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager"
    ".RuntimeEnvironmentManager._get_active_conda_env_name",
    return_value=None,
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager"
    ".RuntimeEnvironmentManager._get_active_conda_env_prefix",
    return_value=None,
)
def test_snapshot_auto_capture_no_active_conda_env(no_conda_env_prefix, no_conda_env_name):
    with pytest.raises(ValueError):
        RuntimeEnvironmentManager().snapshot("auto_capture")


@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager._log_error", Mock()
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager._log_output", Mock()
)
def test_bootstrap_req_txt():
    with patch("subprocess.Popen") as popen:
        popen.return_value.wait.return_value = 0
        RuntimeEnvironmentManager().bootstrap(TEST_REQUIREMENTS_TXT, CLIENT_PYTHON_VERSION)
        python_exe = sys.executable
        call_args = popen.call_args[0][0]
        assert call_args is not None

        expected_cmd = "{} -m pip install -r {} -U".format(python_exe, TEST_REQUIREMENTS_TXT)
        assert call_args == expected_cmd


@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager._log_error", Mock()
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager._log_output", Mock()
)
def test_bootstrap_req_txt_error():
    with patch("subprocess.Popen") as popen:
        popen.return_value.wait.return_value = 1

        with pytest.raises(RuntimeEnvironmentError):
            RuntimeEnvironmentManager().bootstrap(TEST_REQUIREMENTS_TXT, CLIENT_PYTHON_VERSION)

        python_exe = sys.executable
        call_args = popen.call_args[0][0]
        assert call_args is not None

        expected_cmd = "{} -m pip install -r {} -U".format(python_exe, TEST_REQUIREMENTS_TXT)
        assert call_args == expected_cmd


@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager._log_error", Mock()
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager._log_output", Mock()
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager"
    ".RuntimeEnvironmentManager._write_conda_env_to_file",
    Mock(),
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager"
    ".RuntimeEnvironmentManager._get_conda_exe",
    return_value="some_exe",
)
def test_bootstrap_req_txt_with_conda_env(mock_conda_exe):
    with patch("subprocess.Popen") as popen:
        popen.return_value.wait.return_value = 0
        job_conda_env = "conda_env"
        RuntimeEnvironmentManager().bootstrap(
            TEST_REQUIREMENTS_TXT, CLIENT_PYTHON_VERSION, job_conda_env
        )

        call_args = popen.call_args[0][0]
        assert call_args is not None

        expected_cmd = f"{mock_conda_exe.return_value} run -n conda_env pip install -r usr/local/requirements.txt -U"
        assert call_args == expected_cmd


@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager._log_error", Mock()
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager._log_output", Mock()
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager"
    ".RuntimeEnvironmentManager._write_conda_env_to_file",
    Mock(),
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager._validate_python_version"
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager"
    ".RuntimeEnvironmentManager._get_conda_exe",
    return_value="some_exe",
)
def test_bootstrap_conda_yml_create_env(mock_conda_exe, mock_validate_python):
    with patch("subprocess.Popen") as popen:
        popen.return_value.wait.return_value = 0

        RuntimeEnvironmentManager().bootstrap(TEST_CONDA_YML, CLIENT_PYTHON_VERSION)

        call_args = popen.call_args[0][0]
        assert call_args is not None

        expected_cmd = f"{mock_conda_exe.return_value} env create -n sagemaker-runtime-env --file {TEST_CONDA_YML}"
        assert call_args == expected_cmd
        mock_validate_python.assert_called_once_with(CLIENT_PYTHON_VERSION, "sagemaker-runtime-env")


@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager._log_error", Mock()
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager._log_output", Mock()
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager"
    ".RuntimeEnvironmentManager._write_conda_env_to_file",
    Mock(),
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager"
    ".RuntimeEnvironmentManager._get_conda_exe",
    return_value="conda",
)
def test_bootstrap_conda_yml_update_env(mock_conda_exe):
    with patch("subprocess.Popen") as popen:
        popen.return_value.wait.return_value = 0
        job_conda_env = "conda_env"

        RuntimeEnvironmentManager().bootstrap(TEST_CONDA_YML, CLIENT_PYTHON_VERSION, job_conda_env)

        call_args = popen.call_args[0][0]
        assert call_args is not None

        expected_cmd = "{} env update -n {} --file {}".format(
            mock_conda_exe.return_value, job_conda_env, TEST_CONDA_YML
        )
        assert call_args == expected_cmd


@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager"
    ".RuntimeEnvironmentManager._get_conda_exe",
    return_value="conda",
)
def test_python_version_in_conda(mock_conda_exe):
    with patch("subprocess.check_output") as check_output:
        check_output.return_value = b"Python 3.10.7"

        job_conda_env = "conda_env"
        version = RuntimeEnvironmentManager()._python_version_in_conda_env(job_conda_env)
        call_args = check_output.call_args[0][0]
        assert call_args is not None

        expected_cmd = f"{mock_conda_exe.return_value} run -n {job_conda_env} python --version"
        assert call_args == shlex.split(expected_cmd)
        assert version == "3.10"


@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager._python_version_in_conda_env",
    return_value="3.10",
)
def test_validate_python_version(python_version_in_conda_env):
    try:
        RuntimeEnvironmentManager()._validate_python_version(CLIENT_PYTHON_VERSION, "conda_env")
    except Exception:
        pytest.raises("Unexpected error")


@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager._python_version_in_conda_env",
    return_value="3.9",
)
def test_validate_python_version_error(python_version_in_conda_env):
    with pytest.raises(RuntimeEnvironmentError):
        RuntimeEnvironmentManager()._validate_python_version(CLIENT_PYTHON_VERSION, "conda_env")


@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager._current_sagemaker_pysdk_version",
    return_value=JOB_SAGEMAKER_PYSDK_VERSION,
)
def test_validate_sagemaker_pysdk_version(mock_sagemaker_version_in_job):
    # If the client sagemaker version differs from the job's, a warning is printed
    RuntimeEnvironmentManager()._validate_sagemaker_pysdk_version(
        "version-not-the-same-and-get-a-warning"
    )
    mock_sagemaker_version_in_job.assert_called_once()


@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager."
    "RuntimeEnvironmentManager._current_sagemaker_pysdk_version",
    return_value=JOB_SAGEMAKER_PYSDK_VERSION,
)
def test_validate_sagemaker_pysdk_version_with_none_input(mock_sagemaker_version_in_job):
    # This test is to test the backward compatibility
    # In old version of SDK, the client side sagemaker_pysdk_version is not passed to job
    # thus it would be None and would not lead to the warning
    RuntimeEnvironmentManager()._validate_sagemaker_pysdk_version(None)
    mock_sagemaker_version_in_job.assert_called_once()


@patch("os.path.isfile", return_value=True)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager._log_error", Mock()
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager._log_output", Mock()
)
def test_run_pre_exec_script(isfile):
    with patch("subprocess.Popen") as popen:
        popen.return_value.wait.return_value = 0
        RuntimeEnvironmentManager().run_pre_exec_script(pre_exec_script_path="path/to/pre_exec.sh")
        call_args = popen.call_args[0][0]
        expected_cmd = ["/bin/bash", "-eu", "path/to/pre_exec.sh"]
        assert call_args == expected_cmd


@patch("os.path.isfile", return_value=False)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager._log_error", Mock()
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager._log_output", Mock()
)
def test_run_pre_exec_script_no_script(isfile):
    with patch("subprocess.Popen") as popen:
        RuntimeEnvironmentManager().run_pre_exec_script(pre_exec_script_path="path/to/pre_exec.sh")
        popen.assert_not_called()


@patch("os.path.isfile", return_value=True)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager._log_error", Mock()
)
@patch(
    "sagemaker.remote_function.runtime_environment.runtime_environment_manager._log_output", Mock()
)
def test_run_pre_exec_script_cmd_error(isfile):
    with patch("subprocess.Popen") as popen:
        popen.return_value.wait.return_value = 1
        with pytest.raises(RuntimeEnvironmentError):
            RuntimeEnvironmentManager().run_pre_exec_script(
                pre_exec_script_path="path/to/pre_exec.sh"
            )
        call_args = popen.call_args[0][0]
        expected_cmd = ["/bin/bash", "-eu", "path/to/pre_exec.sh"]
        assert call_args == expected_cmd


@patch("subprocess.run")
def test_change_dir_permission(mock_subprocess_run):
    RuntimeEnvironmentManager().change_dir_permission(dirs=["a", "b", "c"], new_permission="777")
    expected_command = ["sudo", "chmod", "-R", "777", "a", "b", "c"]
    assert mock_subprocess_run.assert_called_once_with(
        expected_command, check=True, stderr=subprocess.PIPE
    )


@patch(
    "subprocess.run",
    MagicMock(side_effect=FileNotFoundError("[Errno 2] No such file or directory: 'sudo'")),
)
def test_change_dir_permission_and_no_sudo_installed():
    with pytest.raises(RuntimeEnvironmentError) as error:
        RuntimeEnvironmentManager().change_dir_permission(
            dirs=["a", "b", "c"], new_permission="777"
        )
    assert (
        "Please contact the image owner to install 'sudo' in the job container "
        "and provide sudo privilege to the container user."
    ) in str(error)


@patch("subprocess.run", MagicMock(side_effect=FileNotFoundError("Other file not found error")))
def test_change_dir_permission_and_sudo_installed_but_other_file_not_found_error():
    with pytest.raises(RuntimeEnvironmentError) as error:
        RuntimeEnvironmentManager().change_dir_permission(
            dirs=["a", "b", "c"], new_permission="777"
        )
    assert "Other file not found error" in str(error)


@patch("subprocess.run")
def test_change_dir_permission_and_dir_not_exist(mock_subprocess_run):
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        returncode=1,
        cmd="sudo chmod ...",
        stderr=b"chmod: cannot access ...: No such file or directory",
    )
    with pytest.raises(RuntimeEnvironmentError) as error:
        RuntimeEnvironmentManager().change_dir_permission(
            dirs=["a", "b", "c"], new_permission="777"
        )
    assert "chmod: cannot access ...: No such file or directory" in str(error)
