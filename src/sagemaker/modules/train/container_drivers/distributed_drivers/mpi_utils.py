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
"""This module provides mpi related utility functions for the container drivers."""
from __future__ import absolute_import

import os
import sys
import subprocess
import time

from pathlib import Path
from typing import List

import paramiko

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.utils import (  # noqa: E402 # pylint: disable=C0413,E0611
    SM_EFA_NCCL_INSTANCES,
    SM_EFA_RDMA_INSTANCES,
    get_python_executable,
    logger,
)

FINISHED_STATUS_FILE = "/tmp/done.algo-1"
READY_FILE = "/tmp/ready.%s"
DEFAULT_SSH_PORT = 22


def _write_file_to_host(host: str, status_file: str) -> bool:
    """Write the a file to the provided host."""
    try:
        logger.info(f"Writing {status_file} to {host}")
        subprocess.run(
            ["ssh", host, "touch", f"{status_file}"],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("Finished writing status file")
        return True
    except subprocess.CalledProcessError:
        logger.info(f"Cannot connect to {host}")
        return False


def write_status_file_to_workers(worker_hosts: List[str], status_file: str = FINISHED_STATUS_FILE):
    """Write the status file to all worker nodes."""
    for worker in worker_hosts:
        retry = 0
        while not _write_file_to_host(worker, status_file):
            time.sleep(5)
            retry += 1
            if retry > 5:
                raise TimeoutError(f"Timed out waiting for {worker} to be reachable.")
            logger.info(f"Retrying to write status file to {worker}")


def _wait_for_status_file(status_file: str):
    """Wait for the status file to be created."""
    logger.info(f"Waiting for status file {status_file}")
    while not os.path.exists(status_file):
        time.sleep(30)
    logger.info(f"Found status file {status_file}")


def start_sshd_daemon():
    """Start the SSH daemon on the current node."""
    sshd_executable = "/usr/sbin/sshd"

    if not os.path.exists(sshd_executable):
        raise RuntimeError("SSH daemon not found.")

    # Start the sshd in daemon mode (-D)
    subprocess.Popen([sshd_executable, "-D"])
    logger.info("Started SSH daemon.")


class CustomHostKeyPolicy(paramiko.client.MissingHostKeyPolicy):
    """Class to handle host key policy for SageMaker distributed training SSH connections.

    Example:
    >>> client = paramiko.SSHClient()
    >>> client.set_missing_host_key_policy(CustomHostKeyPolicy())
    >>> # Will succeed for SageMaker algorithm containers
    >>> client.connect('algo-1234.internal')
    >>> # Will raise SSHException for other unknown hosts
    >>> client.connect('unknown-host')  # raises SSHException
    """

    def missing_host_key(self, client, hostname, key):
        """Accept host keys for algo-* hostnames, reject others.

        Args:
            client: The SSHClient instance
            hostname: The hostname attempting to connect
            key: The host key

        Raises:
            paramiko.SSHException: If hostname doesn't match algo-* pattern
        """
        if hostname.startswith("algo-"):
            client.get_host_keys().add(hostname, key.get_name(), key)
            return
        raise paramiko.SSHException(f"Unknown host key for {hostname}")


def _can_connect(host: str, port: int = DEFAULT_SSH_PORT) -> bool:
    """Check if the connection to the provided host and port is possible."""
    try:
        logger.debug("Testing connection to host %s", host)
        with paramiko.SSHClient() as client:
            client.load_system_host_keys()
            client.set_missing_host_key_policy(CustomHostKeyPolicy())
            client.connect(host, port=port)
            logger.info("Can connect to host %s", host)
            return True
    except Exception as e:  # pylint: disable=W0703
        logger.info("Cannot connect to host %s", host)
        logger.debug(f"Connection failed with exception: {e}")
        return False


def _wait_for_workers(worker_hosts: List[str], port: int = DEFAULT_SSH_PORT, timeout: int = 300):
    """Master node waits until it can connect to all worker nodes."""
    start_time = time.time()
    if not worker_hosts:
        logger.info("No worker nodes to connect to.")
        return

    while True:
        logger.info("Master is attempting to connect to all workers...")
        all_workers_connected = all(
            _can_connect(worker, port) and os.path.exists(READY_FILE % worker)
            for worker in worker_hosts
        )

        if all_workers_connected:
            logger.info("Master can connect to all worker nodes.")
            break
        if time.time() - start_time > timeout:
            raise TimeoutError("Timed out waiting for workers to be reachable.")

        time.sleep(5)  # Wait for 5 seconds before trying again


def _wait_for_master(master_host: str, port: int = DEFAULT_SSH_PORT, timeout: int = 300):
    """Worker nodes wait until they can connect to the master node."""
    start_time = time.time()
    while True:
        logger.info(f"Worker is attempting to connect to the master node {master_host}...")
        if _can_connect(master_host, port):
            logger.info(f"Worker can connect to master node {master_host}.")
            break
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timed out waiting for master {master_host} to be reachable.")

        time.sleep(5)  # Wait for 5 seconds before trying again


def bootstrap_worker_node(master_host: str, status_file: str = FINISHED_STATUS_FILE):
    """Bootstrap the worker nodes."""
    logger.info("Bootstrapping worker node...")
    _wait_for_master(master_host)
    _write_file_to_host(master_host, READY_FILE % os.environ["SM_CURRENT_HOST"])
    _wait_for_status_file(status_file)


def bootstrap_master_node(worker_hosts: List[str]):
    """Bootstrap the master node."""
    logger.info("Bootstrapping master node...")
    _wait_for_workers(worker_hosts)


def validate_smddprun() -> bool:
    """Whether smddprun is installed.

    Returns:
        bool: True if installed
    """
    try:
        output = subprocess.run(
            ["which", "smddprun"],
            capture_output=True,
            text=True,
            check=True,
        )
        return output.stdout != ""
    except subprocess.CalledProcessError:
        return False


def validate_smddpmprun() -> bool:
    """Whether smddpmprun is installed.

    Returns:
        bool: True if both are installed
    """
    try:
        output = subprocess.run(
            ["which", "smddpmprun"],
            capture_output=True,
            text=True,
            check=True,
        )
        return output.stdout != ""
    except subprocess.CalledProcessError:
        return False


def write_env_vars_to_file():
    """Write environment variables to /etc/environment file."""
    with open("/etc/environment", "a", encoding="utf-8") as f:
        for name in os.environ:
            f.write(f"{name}={os.environ.get(name)}\n")


def get_mpirun_command(
    host_count: int,
    host_list: List[str],
    num_processes: int,
    additional_options: List[str],
    entry_script_path: str,
):
    """Fetch mpi command"""
    network_interface_name = os.environ.get("SM_NETWORK_INTERFACE_NAME", "eth0")

    mpirun_command = [
        "mpirun",
        "--host",
        ",".join(host_list),
        "-np",
        str(num_processes),
        "--allow-run-as-root",
        "--tag-output",
        "-mca",
        "btl_tcp_if_include",
        network_interface_name,
        "-mca",
        "oob_tcp_if_include",
        network_interface_name,
        "-mca",
        "plm_rsh_no_tree_spawn",
        "1",
        "-mca",
        "pml",
        "ob1",
        "-mca",
        "btl",
        "^openib",
        "-mca",
        "orte_abort_on_non_zero_status",
        "1",
        "-mca",
        "btl_vader_single_copy_mechanism",
        "none",
        "-mca",
        "plm_rsh_num_concurrent",
        str(host_count),
        "-x",
        "NCCL_SOCKET_IFNAME=%s" % network_interface_name,
        "-x",
        "LD_LIBRARY_PATH",
        "-x",
        "PATH",
    ]

    if additional_options:
        mpirun_command.extend(additional_options)

    instance_type = os.environ["SM_CURRENT_INSTANCE_TYPE"]
    # EFA settings
    if instance_type in SM_EFA_NCCL_INSTANCES:
        mpirun_command.extend(["-x", "FI_PROVIDER=efa"])
        # Use simple protocol to handle the out-of-order data delivery from EFA
        mpirun_command.extend(["-x", "NCCL_PROTO=simple"])

    if instance_type in SM_EFA_RDMA_INSTANCES:
        # Use EFA's RDMA functionality for one-sided and two-sided transfer
        mpirun_command.extend(["-x", "FI_EFA_USE_DEVICE_RDMA=1"])

    for credential in [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ]:
        if credential in os.environ:
            mpirun_command.extend(["-x", credential])

    mpirun_command.extend([get_python_executable()])
    mpirun_command.extend(["-m", "mpi4py", entry_script_path])
    return mpirun_command
