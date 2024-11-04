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
import json

from utils import (
    logger,
    read_source_code_config_json,
    get_process_count,
    execute_commands,
    write_failure_file,
    USER_CODE_PATH,
)
from mpi_utils import (
    start_sshd_daemon,
    bootstrap_master_node,
    bootstrap_worker_node,
    get_mpirun_command,
    write_status_file_to_workers,
    write_env_vars_to_file,
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
    source_code_config = read_source_code_config_json()
    distribution = source_code_config.get("distribution", {})
    sm_distributed_settings = distribution.get("smdistributed_settings", {})

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
        process_count = get_process_count(source_code_config)

        if process_count > 1:
            host_list = ["{}:{}".format(host, process_count) for host in host_list]

        mpi_command = get_mpirun_command(
            host_count=host_count,
            host_list=host_list,
            num_processes=process_count,
            smdataparallel_enabled=sm_distributed_settings.get("enable_dataparallel", False),
            smmodelparallel_enabled=sm_distributed_settings.get("enable_modelparallel", False),
            additional_options=distribution.get("mpi_additional_options", []),
            entry_script_path=os.path.join(USER_CODE_PATH, source_code_config["entry_script"]),
        )

        logger.info(f"Executing command: {mpi_command}")
        exit_code, error_traceback = execute_commands(mpi_command)
        if exit_code != 0:
            write_failure_file(error_traceback)

        write_status_file_to_workers(worker_hosts)


if __name__ == "__main__":
    main()
