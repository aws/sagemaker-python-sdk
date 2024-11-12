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
"""An entry point for runtime environment. This must be kept independent of SageMaker PySDK"""
from __future__ import absolute_import

import argparse
import getpass
import sys
import os
import shutil
import pathlib

if __package__ is None or __package__ == "":
    from runtime_environment_manager import (
        RuntimeEnvironmentManager,
        _DependencySettings,
        get_logger,
    )
else:
    from sagemaker.remote_function.runtime_environment.runtime_environment_manager import (
        RuntimeEnvironmentManager,
        _DependencySettings,
        get_logger,
    )

SUCCESS_EXIT_CODE = 0
DEFAULT_FAILURE_CODE = 1

REMOTE_FUNCTION_WORKSPACE = "sm_rf_user_ws"
BASE_CHANNEL_PATH = "/opt/ml/input/data"
FAILURE_REASON_PATH = "/opt/ml/output/failure"
JOB_OUTPUT_DIRS = ["/opt/ml/output", "/opt/ml/model", "/tmp"]
PRE_EXECUTION_SCRIPT_NAME = "pre_exec.sh"
JOB_REMOTE_FUNCTION_WORKSPACE = "sagemaker_remote_function_workspace"
SCRIPT_AND_DEPENDENCIES_CHANNEL_NAME = "pre_exec_script_and_dependencies"


logger = get_logger()


def main(sys_args=None):
    """Entry point for bootstrap script"""

    exit_code = DEFAULT_FAILURE_CODE

    try:
        args = _parse_args(sys_args)
        client_python_version = args.client_python_version
        client_sagemaker_pysdk_version = args.client_sagemaker_pysdk_version
        job_conda_env = args.job_conda_env
        pipeline_execution_id = args.pipeline_execution_id
        dependency_settings = _DependencySettings.from_string(args.dependency_settings)
        func_step_workspace = args.func_step_s3_dir

        conda_env = job_conda_env or os.getenv("SAGEMAKER_JOB_CONDA_ENV")

        RuntimeEnvironmentManager()._validate_python_version(client_python_version, conda_env)

        user = getpass.getuser()
        if user != "root":
            log_message = (
                "The job is running on non-root user: %s. Adding write permissions to the "
                "following job output directories: %s."
            )
            logger.info(log_message, user, JOB_OUTPUT_DIRS)
            RuntimeEnvironmentManager().change_dir_permission(
                dirs=JOB_OUTPUT_DIRS, new_permission="777"
            )

        if pipeline_execution_id:
            _bootstrap_runtime_env_for_pipeline_step(
                client_python_version, func_step_workspace, conda_env, dependency_settings
            )
        else:
            _bootstrap_runtime_env_for_remote_function(
                client_python_version, conda_env, dependency_settings
            )

        RuntimeEnvironmentManager()._validate_sagemaker_pysdk_version(
            client_sagemaker_pysdk_version
        )

        exit_code = SUCCESS_EXIT_CODE
    except Exception as e:  # pylint: disable=broad-except
        logger.exception("Error encountered while bootstrapping runtime environment: %s", e)

        _write_failure_reason_file(str(e))
    finally:
        sys.exit(exit_code)


def _bootstrap_runtime_env_for_remote_function(
    client_python_version: str,
    conda_env: str = None,
    dependency_settings: _DependencySettings = None,
):
    """Bootstrap runtime environment for remote function invocation.

    Args:
        client_python_version (str): Python version at the client side.
        conda_env (str): conda environment to be activated. Default is None.
        dependency_settings (dict): Settings for installing dependencies.
    """

    workspace_unpack_dir = _unpack_user_workspace()
    if not workspace_unpack_dir:
        logger.info("No workspace to unpack and setup.")
        return

    _handle_pre_exec_scripts(workspace_unpack_dir)

    _install_dependencies(
        workspace_unpack_dir,
        conda_env,
        client_python_version,
        REMOTE_FUNCTION_WORKSPACE,
        dependency_settings,
    )


def _bootstrap_runtime_env_for_pipeline_step(
    client_python_version: str,
    func_step_workspace: str,
    conda_env: str = None,
    dependency_settings: _DependencySettings = None,
):
    """Bootstrap runtime environment for pipeline step invocation.

    Args:
        client_python_version (str): Python version at the client side.
        func_step_workspace (str): s3 folder where workspace for FunctionStep is stored
        conda_env (str): conda environment to be activated. Default is None.
        dependency_settings (dict): Name of the dependency file. Default is None.
    """

    workspace_dir = _unpack_user_workspace(func_step_workspace)
    if not workspace_dir:
        os.mkdir(JOB_REMOTE_FUNCTION_WORKSPACE)
        workspace_dir = pathlib.Path(os.getcwd(), JOB_REMOTE_FUNCTION_WORKSPACE).absolute()

    pre_exec_script_and_dependencies_dir = os.path.join(
        BASE_CHANNEL_PATH, SCRIPT_AND_DEPENDENCIES_CHANNEL_NAME
    )

    if not os.path.exists(pre_exec_script_and_dependencies_dir):
        logger.info("No dependencies to bootstrap")
        return
    for file in os.listdir(pre_exec_script_and_dependencies_dir):
        src_path = os.path.join(pre_exec_script_and_dependencies_dir, file)
        dest_path = os.path.join(workspace_dir, file)
        shutil.copy(src_path, dest_path)

    _handle_pre_exec_scripts(workspace_dir)

    _install_dependencies(
        workspace_dir,
        conda_env,
        client_python_version,
        SCRIPT_AND_DEPENDENCIES_CHANNEL_NAME,
        dependency_settings,
    )


def _handle_pre_exec_scripts(script_file_dir: str):
    """Run the pre execution scripts.

    Args:
       script_file_dir (str): Directory in the container where pre-execution scripts exists.
    """

    path_to_pre_exec_script = os.path.join(script_file_dir, PRE_EXECUTION_SCRIPT_NAME)
    RuntimeEnvironmentManager().run_pre_exec_script(pre_exec_script_path=path_to_pre_exec_script)


def _install_dependencies(
    dependency_file_dir: str,
    conda_env: str,
    client_python_version: str,
    channel_name: str,
    dependency_settings: _DependencySettings = None,
):
    """Install dependencies in the job container

    Args:
        dependency_file_dir (str): Directory in the container where dependency file exists.
        conda_env (str): conda environment to be activated.
        client_python_version (str): Python version at the client side.
        channel_name (str): Channel where dependency file was uploaded.
        dependency_settings (dict): Settings for installing dependencies.
    """

    if dependency_settings is not None and dependency_settings.dependency_file is None:
        # an empty dict is passed when no dependencies are specified
        logger.info("No dependencies to install.")
    elif dependency_settings is not None:
        dependencies_file = os.path.join(dependency_file_dir, dependency_settings.dependency_file)
        RuntimeEnvironmentManager().bootstrap(
            local_dependencies_file=dependencies_file,
            conda_env=conda_env,
            client_python_version=client_python_version,
        )
    else:
        # no dependency file name is passed when an older version of the SDK is used
        # we look for a file with .txt, .yml or .yaml extension in the workspace directory
        dependencies_file = None
        for file in os.listdir(dependency_file_dir):
            if file.endswith(".txt") or file.endswith(".yml") or file.endswith(".yaml"):
                dependencies_file = os.path.join(dependency_file_dir, file)
                break

        if dependencies_file:
            RuntimeEnvironmentManager().bootstrap(
                local_dependencies_file=dependencies_file,
                conda_env=conda_env,
                client_python_version=client_python_version,
            )
        else:
            logger.info(
                "Did not find any dependency file in the directory at '%s'."
                " Assuming no additional dependencies to install.",
                os.path.join(BASE_CHANNEL_PATH, channel_name),
            )


def _unpack_user_workspace(func_step_workspace: str = None):
    """Unzip the user workspace"""

    workspace_archive_dir_path = (
        os.path.join(BASE_CHANNEL_PATH, REMOTE_FUNCTION_WORKSPACE)
        if not func_step_workspace
        else os.path.join(BASE_CHANNEL_PATH, func_step_workspace)
    )
    if not os.path.exists(workspace_archive_dir_path):
        logger.info(
            "Directory '%s' does not exist.",
            workspace_archive_dir_path,
        )
        return None

    workspace_archive_path = os.path.join(workspace_archive_dir_path, "workspace.zip")
    if not os.path.isfile(workspace_archive_path):
        logger.info(
            "Workspace archive '%s' does not exist.",
            workspace_archive_dir_path,
        )
        return None

    workspace_unpack_dir = pathlib.Path(os.getcwd()).absolute()
    shutil.unpack_archive(filename=workspace_archive_path, extract_dir=workspace_unpack_dir)
    logger.info("Successfully unpacked workspace archive at '%s'.", workspace_unpack_dir)
    workspace_unpack_dir = pathlib.Path(workspace_unpack_dir, JOB_REMOTE_FUNCTION_WORKSPACE)
    return workspace_unpack_dir


def _write_failure_reason_file(failure_msg):
    """Create a file 'failure' with failure reason written if bootstrap runtime env failed.

    See: https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html
    Args:
        failure_msg: The content of file to be written.
    """
    if not os.path.exists(FAILURE_REASON_PATH):
        with open(FAILURE_REASON_PATH, "w") as f:
            f.write("RuntimeEnvironmentError: " + failure_msg)


def _parse_args(sys_args):
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_conda_env", type=str)
    parser.add_argument("--client_python_version", type=str)
    parser.add_argument("--client_sagemaker_pysdk_version", type=str, default=None)
    parser.add_argument("--pipeline_execution_id", type=str)
    parser.add_argument("--dependency_settings", type=str)
    parser.add_argument("--func_step_s3_dir", type=str)
    args, _ = parser.parse_known_args(sys_args)
    return args


if __name__ == "__main__":
    main(sys.argv[1:])
