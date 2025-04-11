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
"""MPI Utils Unit Tests."""
from __future__ import absolute_import

import os
from mock import patch

import sagemaker.remote_function.runtime_environment.mpi_utils_remote as mpi_utils_remote  # noqa: E402


@patch.dict(
    os.environ,
    {
        "SM_MASTER_ADDR": "algo-1",
        "SM_CURRENT_HOST": "algo-1",
        "SM_HOSTS": '["algo-1", "algo-2"]',
    },
)
@patch("sagemaker.remote_function.runtime_environment.mpi_utils_remote.bootstrap_master_node")
@patch("sagemaker.remote_function.runtime_environment.mpi_utils_remote.bootstrap_worker_node")
@patch("sagemaker.remote_function.runtime_environment.mpi_utils_remote.start_sshd_daemon")
def test_mpi_utils_main_job_start(
    mock_start_sshd_daemon,
    mock_bootstrap_worker_node,
    mock_bootstrap_master_node,
):

    mpi_utils_remote.main()

    mock_start_sshd_daemon.assert_called_once()
    mock_bootstrap_worker_node.assert_not_called()
    mock_bootstrap_master_node.assert_called_once()


@patch.dict(
    os.environ,
    {
        "SM_MASTER_ADDR": "algo-1",
        "SM_CURRENT_HOST": "algo-2",
        "SM_HOSTS": '["algo-1", "algo-2"]',
    },
)
@patch("sagemaker.remote_function.runtime_environment.mpi_utils_remote.bootstrap_master_node")
@patch("sagemaker.remote_function.runtime_environment.mpi_utils_remote.bootstrap_worker_node")
@patch("sagemaker.remote_function.runtime_environment.mpi_utils_remote.start_sshd_daemon")
def test_mpi_utils_worker_job_start(
    mock_start_sshd_daemon,
    mock_bootstrap_worker_node,
    mock_bootstrap_master_node,
):

    mpi_utils_remote.main()

    mock_start_sshd_daemon.assert_called_once()
    mock_bootstrap_worker_node.assert_called_once()
    mock_bootstrap_master_node.assert_not_called()


@patch.dict(
    os.environ,
    {
        "SM_MASTER_ADDR": "algo-1",
        "SM_CURRENT_HOST": "algo-1",
        "SM_HOSTS": '["algo-1", "algo-2"]',
    },
)
@patch("sagemaker.remote_function.runtime_environment.mpi_utils_remote.bootstrap_master_node")
@patch("sagemaker.remote_function.runtime_environment.mpi_utils_remote.bootstrap_worker_node")
@patch("sagemaker.remote_function.runtime_environment.mpi_utils_remote.start_sshd_daemon")
@patch(
    "sagemaker.remote_function.runtime_environment.mpi_utils_remote.write_status_file_to_workers"
)
def test_mpi_utils_main_job_end(
    mock_write_status_file_to_workers,
    mock_start_sshd_daemon,
    mock_bootstrap_worker_node,
    mock_bootstrap_master_node,
):

    mpi_utils_remote.main(["--job_ended", "1"])

    mock_start_sshd_daemon.assert_not_called()
    mock_bootstrap_worker_node.assert_not_called()
    mock_bootstrap_master_node.assert_not_called()
    mock_write_status_file_to_workers.assert_called_once()


@patch.dict(
    os.environ,
    {
        "SM_MASTER_ADDR": "algo-1",
        "SM_CURRENT_HOST": "algo-2",
        "SM_HOSTS": '["algo-1", "algo-2"]',
    },
)
@patch("sagemaker.remote_function.runtime_environment.mpi_utils_remote.bootstrap_master_node")
@patch("sagemaker.remote_function.runtime_environment.mpi_utils_remote.bootstrap_worker_node")
@patch("sagemaker.remote_function.runtime_environment.mpi_utils_remote.start_sshd_daemon")
@patch(
    "sagemaker.remote_function.runtime_environment.mpi_utils_remote.write_status_file_to_workers"
)
def test_mpi_utils_worker_job_end(
    mock_write_status_file_to_workers,
    mock_start_sshd_daemon,
    mock_bootstrap_worker_node,
    mock_bootstrap_master_node,
):

    mpi_utils_remote.main(["--job_ended", "1"])

    mock_start_sshd_daemon.assert_not_called()
    mock_bootstrap_worker_node.assert_not_called()
    mock_bootstrap_master_node.assert_not_called()
    mock_write_status_file_to_workers.assert_not_called()
