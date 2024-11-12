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
"""This module is the entry point for the Torchrun driver script."""
from __future__ import absolute_import

import os
import sys

from typing import List, Tuple

from utils import (
    logger,
    read_source_code_json,
    read_distributed_runner_json,
    read_hyperparameters_json,
    hyperparameters_to_cli_args,
    get_process_count,
    get_python_executable,
    execute_commands,
    write_failure_file,
    USER_CODE_PATH,
    SM_EFA_NCCL_INSTANCES,
    SM_EFA_RDMA_INSTANCES,
)


def pytorch_version() -> Tuple[int, int]:
    """Get the PyTorch version as a tuple of integers."""
    import torch

    return tuple(map(int, torch.__version__.split(".")[:2]))


def get_base_pytorch_command() -> List[str]:
    """Get the base Torch Distributed launcher to execute"""
    if pytorch_version() >= (1, 9):
        return ["torchrun"]
    return [f"{get_python_executable()}", "-m", "torch.distributed.launch"]


def setup_env():
    """Setup the environment variables for PyTorch distributed training"""
    instance_type = os.environ["SM_CURRENT_INSTANCE_TYPE"]
    network_interface_name = os.environ.get("SM_NETWORK_INTERFACE_NAME", "eth0")
    if instance_type in SM_EFA_NCCL_INSTANCES:
        # Enable EFA use
        os.environ["FI_PROVIDER"] = "efa"
    if instance_type in SM_EFA_RDMA_INSTANCES:
        # Use EFA's RDMA functionality for one-sided and two-sided transfer
        os.environ["FI_EFA_USE_DEVICE_RDMA"] = "1"
        os.environ["RDMAV_FORK_SAFE"] = "1"
    os.environ["NCCL_SOCKET_IFNAME"] = str(network_interface_name)
    os.environ["NCCL_PROTO"] = "simple"


def create_commands():
    """Create the Torch Distributed command to execute"""
    source_code = read_source_code_json()
    distribution = read_distributed_runner_json()
    hyperparameters = read_hyperparameters_json()

    process_count = get_process_count(distribution)
    host_count = int(os.environ["SM_HOST_COUNT"])

    torch_cmd = []
    if os.environ.get("RUN_NEURON_PARALLEL_COMPILE") == "1":
        torch_cmd.append("neuron_parallel_compile")

    torch_cmd.extend(get_base_pytorch_command())
    torch_cmd.extend(
        [
            f"--nnodes={host_count}",
            f"--nproc_per_node={process_count}",
        ]
    )

    # If more than one node is used, add node rank information
    if int(host_count) > 1:
        torch_cmd.extend(
            [
                f"--master_addr={os.environ['SM_MASTER_ADDR']}",
                f"--master_port={os.environ['SM_MASTER_PORT']}",
                f"--node_rank={os.environ['SM_CURRENT_HOST_RANK']}",
            ]
        )

    torch_cmd.extend([os.path.join(USER_CODE_PATH, source_code["entry_script"])])

    args = hyperparameters_to_cli_args(hyperparameters)
    torch_cmd += args

    return torch_cmd


def main():
    """Main function to execute the PyTorch distributed training script.

    This function sets some environment variables and executes the PyTorch
    distributed training script.

    Execution Lifecycle:
    1. Setup Environment Variables for PyTorch Distributed Training
    2. Create Torch Distributed Command
    3. Execute Torch Distributed Command with user script provided in `entry_script`
    4. Exit

    """
    setup_env()
    torch_cmd = create_commands()
    logger.info(f"Executing command: {' '.join(torch_cmd)}")
    exit_code, traceback = execute_commands(torch_cmd)
    if exit_code != 0:
        write_failure_file(traceback)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
