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
"""This module is used to define the environment variables for the training job container."""
from __future__ import absolute_import

from typing import Dict, Any
import multiprocessing
import subprocess
import json
import os
import sys
import logging

# Initialize logger
SM_LOG_LEVEL = os.environ.get("SM_LOG_LEVEL", 20)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(console_handler)
logger.setLevel(SM_LOG_LEVEL)

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
INPUT_DATA_CONFIG = f"{SM_INPUT_CONFIG_DIR}/inputdataconfig.json"
HYPERPARAMETERS_CONFIG = f"{SM_INPUT_CONFIG_DIR}/hyperparameters.json"

ENV_OUTPUT_FILE = "sm_training.env"


def num_cpus():
    """Return the number of CPUs available in the current container.

    Returns:
        int: Number of CPUs available in the current container.
    """
    return multiprocessing.cpu_count()


def num_gpus():
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


def num_neurons():
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
        logger.info(f"Found {neuron_cores} neurons on this instance")
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
                            If neuron installed then {}".format(
                        msg
                    )
                )
            except AttributeError:
                logger.info("No Neurons detected (normal if no neurons installed)")
        else:
            logger.info("No Neurons detected (normal if no neurons installed)")

        return 0


def set_env(
    resource_config: Dict[str, Any] = {},
    input_data_config: Dict[str, Any] = {},
    hyperparameters_config: Dict[str, Any] = {},
    output_file: str = "sm_training.env",
    write_to_etc: bool = False,
):
    """Set environment variables for the training job container.

    Args:
        resource_config (Dict[str, Any]): Resource configuration for the training job.
        input_data_config (Dict[str, Any]): Input data configuration for the training job.
        hyperparameters_config (Dict[str, Any]): Hyperparameters configuration for the training job.
        output_file (str): Output file to write the environment variables.
        write_to_etc (bool): Whether to write the environment variables to /etc/environment.
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
        "SM_LOG_LEVEL": SM_LOG_LEVEL,
        "SM_MASTER_ADDR": SM_MASTER_ADDR,
        "SM_MASTER_PORT": SM_MASTER_PORT,
    }

    # Data Channels
    channels = list(input_data_config.keys())
    for channel in channels:
        env_vars[f"SM_CHANNEL_{channel.upper()}"] = f"{SM_INPUT_DATA_DIR}/{channel}"
    env_vars["SM_CHANNELS"] = channels

    # Hyperparameters
    env_vars["SM_HPS"] = hyperparameters_config
    for key, value in hyperparameters_config.items():
        env_vars[f"SM_HP_{key.upper()}"] = value

    # Host Variables
    current_host = resource_config["current_host"]
    hosts = resource_config["hosts"]
    sorted_hosts = sorted(hosts)

    env_vars["SM_CURRENT_HOST"] = current_host
    env_vars["SM_HOSTS"] = sorted_hosts
    env_vars["SM_NETWORK_INTERFACE_NAME"] = resource_config["network_interface_name"]
    env_vars["SM_HOST_COUNT"] = len(sorted_hosts)
    env_vars["SM_CURRENT_HOST_RANK"] = sorted_hosts.index(current_host)

    env_vars["SM_NUM_CPUS"] = num_cpus()
    env_vars["SM_NUM_GPUS"] = num_gpus()
    env_vars["SM_NUM_NEURONS"] = num_neurons()

    # Misc.
    env_vars["SM_RESOURCE_CONFIG"] = resource_config
    env_vars["SM_INPUT_DATA_CONFIG"] = input_data_config

    # All Training Environment Variables
    env_vars["SM_TRAINING_ENV"] = {
        "channel_input_dirs": {
            channel: env_vars[f"SM_CHANNEL_{channel.upper()}"] for channel in channels
        },
        "current_host": env_vars["SM_CURRENT_HOST"],
        "hosts": env_vars["SM_HOSTS"],
        "master_addr": env_vars["SM_MASTER_ADDR"],
        "master_port": env_vars["SM_MASTER_PORT"],
        "hyperparameters": env_vars["SM_HPS"],
        "input_data_config": input_data_config,
        "input_config_dir": env_vars["SM_INPUT_CONFIG_DIR"],
        "input_data_dir": env_vars["SM_INPUT_DATA_DIR"],
        "input_dir": env_vars["SM_INPUT_DIR"],
        "job_name": os.environ["TRAINING_JOB_NAME"],
        "log_level": env_vars["SM_LOG_LEVEL"],
        "model_dir": env_vars["SM_MODEL_DIR"],
        "network_interface_name": env_vars["SM_NETWORK_INTERFACE_NAME"],
        "num_cpus": env_vars["SM_NUM_CPUS"],
        "num_gpus": env_vars["SM_NUM_GPUS"],
        "num_neurons": env_vars["SM_NUM_NEURONS"],
        "output_data_dir": env_vars["SM_OUTPUT_DATA_DIR"],
        "resource_config": env_vars["SM_RESOURCE_CONFIG"],
    }
    try:
        cur_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback to current working directory
        cur_dir = os.getcwd()
    with open(os.path.join(cur_dir, output_file), "w") as f:
        for key, value in env_vars.items():
            if isinstance(value, (list, dict)):
                f.write(f"export {key}='{json.dumps(value)}'\n")
            else:
                f.write(f"export {key}='{value}'\n")

    # Need to write to /etc/environment for MPI to work
    if write_to_etc:
        with open("/etc/environment", "a") as f:
            for key, value in env_vars.items():
                if isinstance(value, (list, dict)):
                    f.write(f"{key}='{json.dumps(value)}'\n")
                else:
                    f.write(f"{key}='{value}'\n")


if __name__ == "__main__":
    with open(RESOURCE_CONFIG, "r") as f:
        resource_config = json.load(f)
    with open(INPUT_DATA_CONFIG, "r") as f:
        input_data_config = json.load(f)
    with open(HYPERPARAMETERS_CONFIG, "r") as f:
        hyperparameters_config = json.load(f)

    set_env(
        resource_config=resource_config,
        input_data_config=input_data_config,
        hyperparameters_config=hyperparameters_config,
        output_file=ENV_OUTPUT_FILE,
        write_to_etc=True,
    )
