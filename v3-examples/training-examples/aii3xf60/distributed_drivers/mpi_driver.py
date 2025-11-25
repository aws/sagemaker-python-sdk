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
"""This module is the entry point for the MPI driver script."""
from __future__ import absolute_import

import os
import sys
import json

from sagemaker.train.container_drivers.distributed_drivers.mpi_utils import (
    start_sshd_daemon,
    bootstrap_master_node,
    bootstrap_worker_node,
    get_mpirun_command,
    write_status_file_to_workers,
    write_env_vars_to_file,
)


from sagemaker.train.container_drivers.common.utils import (
    logger,
    hyperparameters_to_cli_args,
    get_process_count,
    execute_commands,
    write_failure_file,
)


def main():
    """Main function for the MPI driver script.

    The MPI Dirver is responsible for setting up the MPI environment,
    generating the correct mpi commands, and launching the MPI job.

    Execution Lifecycle:
    1. Setup General Environment Variables at /etc/environment
    2. Start SSHD Daemon
    3. Bootstrap Worker Nodes
       a. Wait to establish connection with Master Node
       b. Wait for Master Node to write status file
    4. Bootstrap Master Node
        a. Wait to establish connection with Worker Nodes
        b. Generate MPI Command
        c. Execute MPI Command with user script provided in `entry_script`
        d. Write status file to Worker Nodes
    5. Exit

    """
    entry_script = os.environ["SM_ENTRY_SCRIPT"]
    distributed_config = json.loads(os.environ["SM_DISTRIBUTED_CONFIG"])
    hyperparameters = json.loads(os.environ["SM_HPS"])

    sm_current_host = os.environ["SM_CURRENT_HOST"]
    sm_hosts = json.loads(os.environ["SM_HOSTS"])
    sm_master_addr = os.environ["SM_MASTER_ADDR"]

    write_env_vars_to_file()
    start_sshd_daemon()

    if sm_current_host != sm_master_addr:
        bootstrap_worker_node(sm_master_addr)
    else:
        worker_hosts = [host for host in sm_hosts if host != sm_master_addr]
        bootstrap_master_node(worker_hosts)

        host_list = json.loads(os.environ["SM_HOSTS"])
        host_count = int(os.environ["SM_HOST_COUNT"])
        process_count = int(distributed_config["process_count_per_node"] or 0)
        process_count = get_process_count(process_count)

        if process_count > 1:
            host_list = ["{}:{}".format(host, process_count) for host in host_list]

        mpi_command = get_mpirun_command(
            host_count=host_count,
            host_list=host_list,
            num_processes=process_count,
            additional_options=distributed_config["mpi_additional_options"] or [],
            entry_script_path=entry_script,
        )

        args = hyperparameters_to_cli_args(hyperparameters)
        mpi_command += args

        logger.info(f"Executing command: {' '.join(mpi_command)}")
        exit_code, error_traceback = execute_commands(mpi_command)
        write_status_file_to_workers(worker_hosts)

        if exit_code != 0:
            write_failure_file(error_traceback)
            sys.exit(exit_code)


if __name__ == "__main__":
    main()
