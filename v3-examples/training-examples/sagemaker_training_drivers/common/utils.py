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

from typing import List, Dict, Any, Tuple, IO, Optional

# Initialize logger
SM_LOG_LEVEL = os.environ.get("SM_LOG_LEVEL", 20)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(console_handler)
logger.setLevel(int(SM_LOG_LEVEL))

FAILURE_FILE = "/opt/ml/output/failure"
DEFAULT_FAILURE_MESSAGE = """
Training Execution failed.
For more details, see CloudWatch logs at 'aws/sagemaker/TrainingJobs'.
TrainingJob - {training_job_name}
"""

USER_CODE_PATH = "/opt/ml/input/data/code"
SOURCE_CODE_JSON = "/opt/ml/input/data/sm_drivers/sourcecode.json"
DISTRIBUTED_JSON = "/opt/ml/input/data/sm_drivers/distributed.json"

HYPERPARAMETERS_JSON = "/opt/ml/input/config/hyperparameters.json"

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


def write_failure_file(message: Optional[str] = None):
    """Write a failure file with the message."""
    if message is None:
        message = DEFAULT_FAILURE_MESSAGE.format(training_job_name=os.environ["TRAINING_JOB_NAME"])
    if not os.path.exists(FAILURE_FILE):
        with open(FAILURE_FILE, "w") as f:
            f.write(message)


def read_source_code_json(source_code_json: Dict[str, Any] = SOURCE_CODE_JSON):
    """Read the source code config json file."""
    try:
        with open(source_code_json, "r") as f:
            source_code_dict = json.load(f) or {}
    except FileNotFoundError:
        source_code_dict = {}
    return source_code_dict


def read_distributed_json(distributed_json: Dict[str, Any] = DISTRIBUTED_JSON):
    """Read the distribution config json file."""
    try:
        with open(distributed_json, "r") as f:
            distributed_dict = json.load(f) or {}
    except FileNotFoundError:
        distributed_dict = {}
    return distributed_dict


def read_hyperparameters_json(hyperparameters_json: Dict[str, Any] = HYPERPARAMETERS_JSON):
    """Read the hyperparameters config json file."""
    try:
        with open(hyperparameters_json, "r") as f:
            hyperparameters_dict = json.load(f) or {}
    except FileNotFoundError:
        hyperparameters_dict = {}
    return hyperparameters_dict


def get_process_count(process_count: Optional[int] = None) -> int:
    """Get the number of processes to run on each node in the training job."""
    return (
        process_count
        or int(os.environ.get("SM_NUM_GPUS", 0))
        or int(os.environ.get("SM_NUM_NEURONS", 0))
        or 1
    )


def hyperparameters_to_cli_args(hyperparameters: Dict[str, Any]) -> List[str]:
    """Convert the hyperparameters to CLI arguments."""
    cli_args = []
    for key, value in hyperparameters.items():
        value = safe_deserialize(value)
        cli_args.extend([f"--{key}", safe_serialize(value)])

    return cli_args


def safe_deserialize(data: Any) -> Any:
    """Safely deserialize data from a JSON string.

    This function handles the following cases:
    1. If `data` is not a string, it returns the input as-is.
    2. If `data` is a JSON-encoded string, it attempts to deserialize it using `json.loads()`.
    3. If `data` is a string but cannot be decoded as JSON, it returns the original string.

    Returns:
        Any: The deserialized data, or the original input if it cannot be JSON-decoded.
    """
    if not isinstance(data, str):
        return data

    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return data


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
