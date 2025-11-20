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
"""An utils function for runtime environment. This must be kept independent of SageMaker PySDK"""
from __future__ import absolute_import

import argparse
import json
import os
import subprocess
import sys
import time
from typing import List

import paramiko

if __package__ is None or __package__ == "":
    from runtime_environment_manager import (
        get_logger,
    )
else:
    from sagemaker.train.remote_function.runtime_environment.runtime_environment_manager import (
        get_logger,
    )

SUCCESS_EXIT_CODE = 0
DEFAULT_FAILURE_CODE = 1

FINISHED_STATUS_FILE = "/tmp/done.algo-1"
READY_FILE = "/tmp/ready.%s"
DEFAULT_SSH_PORT = 22

FAILURE_REASON_PATH = "/opt/ml/output/failure"
FINISHED_STATUS_FILE = "/tmp/done.algo-1"

logger = get_logger()


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


def _parse_args(sys_args):
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_ended", type=str, default="0")
    args, _ = parser.parse_known_args(sys_args)
    return args


def _can_connect(host: str, port: int = DEFAULT_SSH_PORT) -> bool:
    """Check if the connection to the provided host and port is possible."""
    try:
        with paramiko.SSHClient() as client:
            client.load_system_host_keys()
            client.set_missing_host_key_policy(CustomHostKeyPolicy())
            client.connect(host, port=port)
            logger.info("Can connect to host %s", host)
            return True
    except Exception as e:  # pylint: disable=W0703
        logger.info("Cannot connect to host %s", host)
        logger.debug("Connection failed with exception: %s", e)
        return False


def _write_file_to_host(host: str, status_file: str) -> bool:
    """Write the a file to the provided host."""
    try:
        logger.info("Writing %s to %s", status_file, host)
        subprocess.run(
            ["ssh", host, "touch", f"{status_file}"],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("Finished writing status file")
        return True
    except subprocess.CalledProcessError:
        logger.info("Cannot connect to %s", host)
        return False


def _write_failure_reason_file(failure_msg):
    """Create a file 'failure' with failure reason written if bootstrap runtime env failed.

    See: https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html
    Args:
        failure_msg: The content of file to be written.
    """
    if not os.path.exists(FAILURE_REASON_PATH):
        with open(FAILURE_REASON_PATH, "w") as f:
            f.write("RuntimeEnvironmentError: " + failure_msg)


def _wait_for_master(master_host: str, port: int = DEFAULT_SSH_PORT, timeout: int = 300):
    """Worker nodes wait until they can connect to the master node."""
    start_time = time.time()
    while True:
        logger.info("Worker is attempting to connect to the master node %s...", master_host)
        if _can_connect(master_host, port):
            logger.info("Worker can connect to master node %s.", master_host)
            break
        if time.time() - start_time > timeout:
            raise TimeoutError("Timed out waiting for master %s to be reachable." % master_host)

        time.sleep(5)  # Wait for 5 seconds before trying again


def _wait_for_status_file(status_file: str):
    """Wait for the status file to be created."""
    logger.info("Waiting for status file %s", status_file)
    while not os.path.exists(status_file):
        time.sleep(30)
    logger.info("Found status file %s", status_file)


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


def bootstrap_master_node(worker_hosts: List[str]):
    """Bootstrap the master node."""
    logger.info("Bootstrapping master node...")
    _wait_for_workers(worker_hosts)


def bootstrap_worker_node(
    master_host: str, current_host: str, status_file: str = FINISHED_STATUS_FILE
):
    """Bootstrap the worker nodes."""
    logger.info("Bootstrapping worker node...")
    _wait_for_master(master_host)
    _write_file_to_host(master_host, READY_FILE % current_host)
    _wait_for_status_file(status_file)


def start_sshd_daemon():
    """Start the SSH daemon on the current node."""
    sshd_executable = "/usr/sbin/sshd"

    if not os.path.exists(sshd_executable):
        raise RuntimeError("SSH daemon not found.")

    # Start the sshd in daemon mode (-D)
    subprocess.Popen([sshd_executable, "-D"])
    logger.info("Started SSH daemon.")


def write_status_file_to_workers(worker_hosts: List[str], status_file: str = FINISHED_STATUS_FILE):
    """Write the status file to all worker nodes."""
    for worker in worker_hosts:
        retry = 0
        while not _write_file_to_host(worker, status_file):
            time.sleep(5)
            retry += 1
            if retry > 5:
                raise TimeoutError("Timed out waiting for %s to be reachable." % worker)
            logger.info("Retrying to write status file to %s", worker)


def main(sys_args=None):
    """Entry point for bootstrap script"""
    try:
        args = _parse_args(sys_args)

        job_ended = args.job_ended

        main_host = os.environ["SM_MASTER_ADDR"]
        current_host = os.environ["SM_CURRENT_HOST"]

        if job_ended == "0":
            logger.info("Job is running, bootstrapping nodes")

            start_sshd_daemon()

            if current_host != main_host:
                bootstrap_worker_node(main_host, current_host)
            else:
                sorted_hosts = json.loads(os.environ["SM_HOSTS"])
                worker_hosts = [host for host in sorted_hosts if host != main_host]

                bootstrap_master_node(worker_hosts)
        else:
            logger.info("Job ended, writing status file to workers")

            if current_host == main_host:
                sorted_hosts = json.loads(os.environ["SM_HOSTS"])
                worker_hosts = [host for host in sorted_hosts if host != main_host]

                write_status_file_to_workers(worker_hosts)
    except Exception as e:  # pylint: disable=broad-except
        logger.exception("Error encountered while bootstrapping runtime environment: %s", e)

        _write_failure_reason_file(str(e))

        sys.exit(DEFAULT_FAILURE_CODE)


if __name__ == "__main__":
    main(sys.argv[1:])