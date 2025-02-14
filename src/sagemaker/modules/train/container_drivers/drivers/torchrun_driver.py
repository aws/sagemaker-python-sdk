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
import json

from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.utils import (  # noqa: E402 # pylint: disable=C0413,E0611
    logger,
    hyperparameters_to_cli_args,
    get_process_count,
    get_python_executable,
    execute_commands,
    write_failure_file,
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
    entry_script = os.environ["SM_ENTRY_SCRIPT"]
    distributed_config = json.loads(os.environ["SM_DISTRIBUTED_CONFIG"])
    hyperparameters = json.loads(os.environ["SM_HPS"])

    process_count = int(distributed_config["process_count_per_node"] or 0)
    process_count = get_process_count(process_count)
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

    torch_cmd.extend([entry_script])

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
