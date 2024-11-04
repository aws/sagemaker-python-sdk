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
"""This module provides utility functions for the container drivers."""
from __future__ import absolute_import

import os
import logging
import sys
import subprocess
import traceback
import json

from typing import List, Dict, Any, Tuple, IO

# Initialize logger
SM_LOG_LEVEL = os.environ.get("SM_LOG_LEVEL", 20)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(console_handler)
logger.setLevel(int(SM_LOG_LEVEL))

FAILURE_FILE = "/opt/ml/output/failure"
DEFAULT_FAILURE_MESSAGE = f"""
Training Execution failed.
For more details, see CloudWatch logs at 'aws/sagemaker/TrainingJobs'.
TrainingJob - {os.environ['TRAINING_JOB_NAME']}
"""

USER_CODE_PATH = "/opt/ml/input/data/code"
SOURCE_CODE_CONFIG_JSON = "/opt/ml/input/data/sm_code/sourcecodeconfig.json"

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


def write_failure_file(message: str = DEFAULT_FAILURE_MESSAGE):
    """Write a failure file with the message."""
    if not os.path.exists(FAILURE_FILE):
        with open(FAILURE_FILE, "w") as f:
            f.write(message)


def read_source_code_config_json(source_code_config_file: Dict[str, Any] = SOURCE_CODE_CONFIG_JSON):
    """Read the source code config json file."""
    with open(source_code_config_file, "r") as f:
        distribution_config = json.load(f)
    return distribution_config


def get_process_count(source_code_config: Dict[str, Any]) -> int:
    """Get the number of processes to run on each node in the training job."""
    if source_code_config.get("distribution", {}).get("process_count_per_node") is not None:
        return int(source_code_config["distribution"]["process_count_per_node"])
    if os.environ.get("SM_NUM_GPUS") is not None:
        return int(os.environ["SM_NUM_GPUS"])
    if os.environ.get("SM_NUM_NEURONS") is not None:
        return int(os.environ["SM_NUM_NEURONS"])
    return 1  # Default to 1 process per node


def get_python_executable() -> str:
    """Get the python executable path."""
    return sys.executable


def log_subprocess_output(pipe: IO[bytes]):
    """Log the output from the subprocess."""
    for line in iter(pipe.readline, b""):
        logger.info(line.decode("utf-8").strip())


def execute_commands(commands: List[str]) -> Tuple[int, str]:
    """Execute the provided commands and return exit code with failure traceback if any."""
    try:
        process = subprocess.Popen(
            commands,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with process.stdout:
            log_subprocess_output(process.stdout)
        exitcode = process.wait()
        if exitcode != 0:
            raise subprocess.CalledProcessError(exitcode, commands)
        return exitcode, ""
    except subprocess.CalledProcessError as e:
        # Capture the traceback in case of failure
        error_traceback = traceback.format_exc()
        print(f"Command failed with exit code {e.returncode}. Traceback: {error_traceback}")
        return e.returncode, error_traceback


def is_worker_node() -> bool:
    """Check if the current node is a worker node."""
    return os.environ.get("SM_CURRENT_HOST") != os.environ.get("SM_MASTER_ADDR")


def is_master_node() -> bool:
    """Check if the current node is the master node."""
    return os.environ.get("SM_CURRENT_HOST") == os.environ.get("SM_MASTER_ADDR")
