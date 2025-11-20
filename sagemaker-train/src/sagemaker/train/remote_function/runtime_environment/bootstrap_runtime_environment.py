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
import json
import multiprocessing
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Any, Dict

if __package__ is None or __package__ == "":
    from runtime_environment_manager import (
        RuntimeEnvironmentManager,
        _DependencySettings,
        get_logger,
    )
else:
    from sagemaker.train.remote_function.runtime_environment.runtime_environment_manager import (
        RuntimeEnvironmentManager,
        _DependencySettings,
        get_logger,
    )

SUCCESS_EXIT_CODE = 0
DEFAULT_FAILURE_CODE = 1

REMOTE_FUNCTION_WORKSPACE = "sm_rf_user_ws"
BASE_CHANNEL_PATH = "/opt/ml/input/data"
FAILURE_REASON_PATH = "/opt/ml/output/failure"
JOB_OUTPUT_DIRS = ["/opt/ml/input", "/opt/ml/output", "/opt/ml/model", "/tmp"]
PRE_EXECUTION_SCRIPT_NAME = "pre_exec.sh"
JOB_REMOTE_FUNCTION_WORKSPACE = "sagemaker_remote_function_workspace"
SCRIPT_AND_DEPENDENCIES_CHANNEL_NAME = "pre_exec_script_and_dependencies"

SM_MODEL_DIR = "/opt/ml/model"

SM_INPUT_DIR = "/opt/ml/input"
SM_INPUT_DATA_DIR = "/opt/ml/input/data"
SM_INPUT_CONFIG_DIR = "/opt/ml/input/config"

SM_OUTPUT_DIR = "/opt/ml/output"
SM_OUTPUT_FAILURE = "/opt/ml/output/failure"
SM_OUTPUT_DATA_DIR = "/opt/ml/output/data"

SM_MASTER_ADDR = "algo-1"
SM_MASTER_PORT = 7777

RESOURCE_CONFIG = f"{SM_INPUT_CONFIG_DIR}/resourceconfig.json"
ENV_OUTPUT_FILE = "/opt/ml/input/sm_training.env"

SENSITIVE_KEYWORDS = ["SECRET", "PASSWORD", "KEY", "TOKEN", "PRIVATE", "CREDS", "CREDENTIALS"]
HIDDEN_VALUE = "******"

SM_EFA_NCCL_INSTANCES = [
    "ml.g4dn.8xlarge",
    "ml.g4dn.12xlarge",
    "ml.g5.48xlarge",
    "ml.p3dn.24xlarge",
    "ml.p4d.24xlarge",
    "ml.p4de.24xlarge",
    "ml.p5.48xlarge",
    "ml.trn1.32xlarge",
]

SM_EFA_RDMA_INSTANCES = [
    "ml.p4d.24xlarge",
    "ml.p4de.24xlarge",
    "ml.trn1.32xlarge",
]

logger = get_logger()


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
        # no dependency file name is passed when an legacy version of the SDK is used
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
    parser.add_argument("--distribution", type=str, default=None)
    parser.add_argument("--user_nproc_per_node", type=str, default=None)
    args, _ = parser.parse_known_args(sys_args)
    return args


def log_key_value(key: str, value: str):
    """Log a key-value pair, masking sensitive values if necessary."""
    if any(keyword.lower() in key.lower() for keyword in SENSITIVE_KEYWORDS):
        logger.info("%s=%s", key, HIDDEN_VALUE)
    elif isinstance(value, dict):
        masked_value = mask_sensitive_info(value)
        logger.info("%s=%s", key, json.dumps(masked_value))
    else:
        try:
            decoded_value = json.loads(value)
            if isinstance(decoded_value, dict):
                masked_value = mask_sensitive_info(decoded_value)
                logger.info("%s=%s", key, json.dumps(masked_value))
            else:
                logger.info("%s=%s", key, decoded_value)
        except (json.JSONDecodeError, TypeError):
            logger.info("%s=%s", key, value)


def log_env_variables(env_vars_dict: Dict[str, Any]):
    """Log Environment Variables from the environment and an env_vars_dict."""
    for key, value in os.environ.items():
        log_key_value(key, value)

    for key, value in env_vars_dict.items():
        log_key_value(key, value)


def mask_sensitive_info(data):
    """Recursively mask sensitive information in a dictionary."""
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                data[k] = mask_sensitive_info(v)
            elif isinstance(v, str) and any(
                keyword.lower() in k.lower() for keyword in SENSITIVE_KEYWORDS
            ):
                data[k] = HIDDEN_VALUE
    return data


def num_cpus() -> int:
    """Return the number of CPUs available in the current container.

    Returns:
        int: Number of CPUs available in the current container.
    """
    return multiprocessing.cpu_count()


def num_gpus() -> int:
    """Return the number of GPUs available in the current container.

    Returns:
        int: Number of GPUs available in the current container.
    """
    try:
        cmd = ["nvidia-smi", "--list-gpus"]
        output = subprocess.check_output(cmd).decode("utf-8")
        return sum(1 for line in output.splitlines() if line.startswith("GPU "))
    except (OSError, subprocess.CalledProcessError):
        logger.info("No GPUs detected (normal if no gpus installed)")
        return 0


def num_neurons() -> int:
    """Return the number of neuron cores available in the current container.

    Returns:
        int: Number of Neuron Cores available in the current container.
    """
    try:
        cmd = ["neuron-ls", "-j"]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8")
        j = json.loads(output)
        neuron_cores = 0
        for item in j:
            neuron_cores += item.get("nc_count", 0)
        logger.info("Found %s neurons on this instance", neuron_cores)
        return neuron_cores
    except OSError:
        logger.info("No Neurons detected (normal if no neurons installed)")
        return 0
    except subprocess.CalledProcessError as e:
        if e.output is not None:
            try:
                msg = e.output.decode("utf-8").partition("error=")[2]
                logger.info(
                    "No Neurons detected (normal if no neurons installed). \
                    If neuron installed then %s",
                    msg,
                )
            except AttributeError:
                logger.info("No Neurons detected (normal if no neurons installed)")
        else:
            logger.info("No Neurons detected (normal if no neurons installed)")

        return 0


def safe_serialize(data):
    """Serialize the data without wrapping strings in quotes.

    This function handles the following cases:
    1. If `data` is a string, it returns the string as-is without wrapping in quotes.
    2. If `data` is serializable (e.g., a dictionary, list, int, float), it returns
       the JSON-encoded string using `json.dumps()`.
    3. If `data` cannot be serialized (e.g., a custom object), it returns the string
       representation of the data using `str(data)`.

    Args:
        data (Any): The data to serialize.

    Returns:
        str: The serialized JSON-compatible string or the string representation of the input.
    """
    if isinstance(data, str):
        return data
    try:
        return json.dumps(data)
    except TypeError:
        return str(data)


def set_env(
    resource_config: Dict[str, Any],
    distribution: str = None,
    user_nproc_per_node: bool = None,
    output_file: str = ENV_OUTPUT_FILE,
):
    """Set environment variables for the training job container.

    Args:
        resource_config (Dict[str, Any]): Resource configuration for the training job.
        output_file (str): Output file to write the environment variables.
    """
    # Constants
    env_vars = {
        "SM_MODEL_DIR": SM_MODEL_DIR,
        "SM_INPUT_DIR": SM_INPUT_DIR,
        "SM_INPUT_DATA_DIR": SM_INPUT_DATA_DIR,
        "SM_INPUT_CONFIG_DIR": SM_INPUT_CONFIG_DIR,
        "SM_OUTPUT_DIR": SM_OUTPUT_DIR,
        "SM_OUTPUT_FAILURE": SM_OUTPUT_FAILURE,
        "SM_OUTPUT_DATA_DIR": SM_OUTPUT_DATA_DIR,
        "SM_MASTER_ADDR": SM_MASTER_ADDR,
        "SM_MASTER_PORT": SM_MASTER_PORT,
    }

    # Host Variables
    current_host = resource_config["current_host"]
    current_instance_type = resource_config["current_instance_type"]
    hosts = resource_config["hosts"]
    sorted_hosts = sorted(hosts)

    env_vars["SM_CURRENT_HOST"] = current_host
    env_vars["SM_CURRENT_INSTANCE_TYPE"] = current_instance_type
    env_vars["SM_HOSTS"] = sorted_hosts
    env_vars["SM_NETWORK_INTERFACE_NAME"] = resource_config["network_interface_name"]
    env_vars["SM_HOST_COUNT"] = len(sorted_hosts)
    env_vars["SM_CURRENT_HOST_RANK"] = sorted_hosts.index(current_host)

    env_vars["SM_NUM_CPUS"] = num_cpus()
    env_vars["SM_NUM_GPUS"] = num_gpus()
    env_vars["SM_NUM_NEURONS"] = num_neurons()

    # Misc.
    env_vars["SM_RESOURCE_CONFIG"] = resource_config

    if user_nproc_per_node is not None and int(user_nproc_per_node) > 0:
        env_vars["SM_NPROC_PER_NODE"] = int(user_nproc_per_node)
    else:
        if int(env_vars["SM_NUM_GPUS"]) > 0:
            env_vars["SM_NPROC_PER_NODE"] = int(env_vars["SM_NUM_GPUS"])
        elif int(env_vars["SM_NUM_NEURONS"]) > 0:
            env_vars["SM_NPROC_PER_NODE"] = int(env_vars["SM_NUM_NEURONS"])
        else:
            env_vars["SM_NPROC_PER_NODE"] = int(env_vars["SM_NUM_CPUS"])

    # All Training Environment Variables
    env_vars["SM_TRAINING_ENV"] = {
        "current_host": env_vars["SM_CURRENT_HOST"],
        "current_instance_type": env_vars["SM_CURRENT_INSTANCE_TYPE"],
        "hosts": env_vars["SM_HOSTS"],
        "host_count": env_vars["SM_HOST_COUNT"],
        "nproc_per_node": env_vars["SM_NPROC_PER_NODE"],
        "master_addr": env_vars["SM_MASTER_ADDR"],
        "master_port": env_vars["SM_MASTER_PORT"],
        "input_config_dir": env_vars["SM_INPUT_CONFIG_DIR"],
        "input_data_dir": env_vars["SM_INPUT_DATA_DIR"],
        "input_dir": env_vars["SM_INPUT_DIR"],
        "job_name": os.environ["TRAINING_JOB_NAME"],
        "model_dir": env_vars["SM_MODEL_DIR"],
        "network_interface_name": env_vars["SM_NETWORK_INTERFACE_NAME"],
        "num_cpus": env_vars["SM_NUM_CPUS"],
        "num_gpus": env_vars["SM_NUM_GPUS"],
        "num_neurons": env_vars["SM_NUM_NEURONS"],
        "output_data_dir": env_vars["SM_OUTPUT_DATA_DIR"],
        "resource_config": env_vars["SM_RESOURCE_CONFIG"],
    }

    if distribution and distribution == "torchrun":
        logger.info("Distribution: torchrun")

        instance_type = env_vars["SM_CURRENT_INSTANCE_TYPE"]
        network_interface_name = env_vars.get("SM_NETWORK_INTERFACE_NAME", "eth0")

        if instance_type in SM_EFA_NCCL_INSTANCES:
            # Enable EFA use
            env_vars["FI_PROVIDER"] = "efa"
        if instance_type in SM_EFA_RDMA_INSTANCES:
            # Use EFA's RDMA functionality for one-sided and two-sided transfer
            env_vars["FI_EFA_USE_DEVICE_RDMA"] = "1"
            env_vars["RDMAV_FORK_SAFE"] = "1"
        env_vars["NCCL_SOCKET_IFNAME"] = str(network_interface_name)
        env_vars["NCCL_PROTO"] = "simple"
    elif distribution and distribution == "mpirun":
        logger.info("Distribution: mpirun")

        env_vars["MASTER_ADDR"] = env_vars["SM_MASTER_ADDR"]
        env_vars["MASTER_PORT"] = str(env_vars["SM_MASTER_PORT"])

        host_list = [
            "{}:{}".format(host, int(env_vars["SM_NPROC_PER_NODE"])) for host in sorted_hosts
        ]
        env_vars["SM_HOSTS_LIST"] = ",".join(host_list)

        instance_type = env_vars["SM_CURRENT_INSTANCE_TYPE"]

        if instance_type in SM_EFA_NCCL_INSTANCES:
            env_vars["SM_FI_PROVIDER"] = "-x FI_PROVIDER=efa"
            env_vars["SM_NCCL_PROTO"] = "-x NCCL_PROTO=simple"
        else:
            env_vars["SM_FI_PROVIDER"] = ""
            env_vars["SM_NCCL_PROTO"] = ""

        if instance_type in SM_EFA_RDMA_INSTANCES:
            env_vars["SM_FI_EFA_USE_DEVICE_RDMA"] = "-x FI_EFA_USE_DEVICE_RDMA=1"
        else:
            env_vars["SM_FI_EFA_USE_DEVICE_RDMA"] = ""

    with open(output_file, "w") as f:
        for key, value in env_vars.items():
            f.write(f"export {key}='{safe_serialize(value)}'\n")

    logger.info("Environment Variables:")
    log_env_variables(env_vars_dict=env_vars)


def main(sys_args=None):
    """Entry point for bootstrap script"""

    exit_code = DEFAULT_FAILURE_CODE

    try:
        args = _parse_args(sys_args)

        logger.info("Arguments:")
        for arg in vars(args):
            logger.info("%s=%s", arg, getattr(args, arg))

        client_python_version = args.client_python_version
        client_sagemaker_pysdk_version = args.client_sagemaker_pysdk_version
        job_conda_env = args.job_conda_env
        pipeline_execution_id = args.pipeline_execution_id
        dependency_settings = _DependencySettings.from_string(args.dependency_settings)
        func_step_workspace = args.func_step_s3_dir
        distribution = args.distribution
        user_nproc_per_node = args.user_nproc_per_node

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

        if os.path.exists(RESOURCE_CONFIG):
            try:
                logger.info("Found %s", RESOURCE_CONFIG)
                with open(RESOURCE_CONFIG, "r") as f:
                    resource_config = json.load(f)
                set_env(
                    resource_config=resource_config,
                    distribution=distribution,
                    user_nproc_per_node=user_nproc_per_node,
                )
            except (json.JSONDecodeError, FileNotFoundError) as e:
                # Optionally, you might want to log this error
                logger.info("ERROR: Error processing %s: %s", RESOURCE_CONFIG, str(e))

        exit_code = SUCCESS_EXIT_CODE
    except Exception as e:  # pylint: disable=broad-except
        logger.exception("Error encountered while bootstrapping runtime environment: %s", e)

        _write_failure_reason_file(str(e))
    finally:
        sys.exit(exit_code)


if __name__ == "__main__":
    main(sys.argv[1:])