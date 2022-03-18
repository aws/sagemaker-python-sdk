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
from __future__ import absolute_import

import os
import pytest
from mock import patch, Mock

import sagemaker.local.utils


@patch("sagemaker.local.utils.os.path")
@patch("sagemaker.local.utils.os")
def test_copy_directory_structure(m_os, m_os_path):
    m_os_path.exists.return_value = False
    sagemaker.local.utils.copy_directory_structure("/tmp/", "code/")
    m_os.makedirs.assert_called_with("/tmp/", "code/")


@patch("shutil.rmtree", Mock())
@patch("sagemaker.local.utils.recursive_copy")
def test_move_to_destination_local(recursive_copy):
    # local files will just be recursively copied
    # given absolute path
    sagemaker.local.utils.move_to_destination("/tmp/data", "file:///target/dir", "job", None)
    recursive_copy.assert_called_with("/tmp/data", "/target/dir")
    # given relative path
    sagemaker.local.utils.move_to_destination("/tmp/data", "file://root/target/dir", "job", None)
    recursive_copy.assert_called_with("/tmp/data", os.path.abspath("./root/target/dir"))


@patch("shutil.rmtree", Mock())
@patch("sagemaker.local.utils.recursive_copy")
def test_move_to_destination_s3(recursive_copy):
    sms = Mock()

    # without trailing slash in prefix
    sagemaker.local.utils.move_to_destination("/tmp/data", "s3://bucket/path", "job", sms)
    sms.upload_data.assert_called_with("/tmp/data", "bucket", "path/job")
    recursive_copy.assert_not_called()

    # with trailing slash in prefix
    sagemaker.local.utils.move_to_destination("/tmp/data", "s3://bucket/path/", "job", sms)
    sms.upload_data.assert_called_with("/tmp/data", "bucket", "path/job")

    # without path, with trailing slash
    sagemaker.local.utils.move_to_destination("/tmp/data", "s3://bucket/", "job", sms)
    sms.upload_data.assert_called_with("/tmp/data", "bucket", "job")

    # without path, without trailing slash
    sagemaker.local.utils.move_to_destination("/tmp/data", "s3://bucket", "job", sms)
    sms.upload_data.assert_called_with("/tmp/data", "bucket", "job")


def test_move_to_destination_illegal_destination():
    with pytest.raises(ValueError):
        sagemaker.local.utils.move_to_destination("/tmp/data", "ftp://ftp/in/2018", "job", None)


@patch("sagemaker.local.utils.os.path")
@patch("sagemaker.local.utils.copy_tree")
def test_recursive_copy(copy_tree, m_os_path):
    m_os_path.isdir.return_value = True
    sagemaker.local.utils.recursive_copy("source", "destination")
    copy_tree.assert_called_with("source", "destination")


@patch("sagemaker.local.utils.os")
@patch("sagemaker.local.utils.get_child_process_ids")
def test_kill_child_processes(m_get_child_process_ids, m_os):
    m_get_child_process_ids.return_value = ["child_pids"]
    sagemaker.local.utils.kill_child_processes("pid")
    m_os.kill.assert_called_with("child_pids", 15)


@patch("sagemaker.local.utils.subprocess")
def test_get_child_process_ids(m_subprocess):
    cmd = "pgrep -P pid".split()
    process_mock = Mock()
    attrs = {"communicate.return_value": (b"\n", False), "returncode": 0}
    process_mock.configure_mock(**attrs)
    m_subprocess.Popen.return_value = process_mock
    sagemaker.local.utils.get_child_process_ids("pid")
    m_subprocess.Popen.assert_called_with(cmd, stdout=m_subprocess.PIPE, stderr=m_subprocess.PIPE)


@patch("sagemaker.local.utils.subprocess")
def test_get_docker_host(m_subprocess):
    cmd = "docker context inspect".split()
    process_mock = Mock()
    endpoints = [
        {"test": "tcp://host:port", "result": "host"},
        {"test": "fd://something", "result": "localhost"},
        {"test": "unix://path/to/socket", "result": "localhost"},
        {"test": "npipe:////./pipe/foo", "result": "localhost"},
    ]
    for endpoint in endpoints:
        return_value = (
            '[\n{\n"Endpoints":{\n"docker":{\n"Host": "%s"}\n}\n}\n]\n' % endpoint["test"]
        )
        attrs = {"communicate.return_value": (return_value.encode("utf-8"), None), "returncode": 0}
        process_mock.configure_mock(**attrs)
        m_subprocess.Popen.return_value = process_mock
        host = sagemaker.local.utils.get_docker_host()
        m_subprocess.Popen.assert_called_with(
            cmd, stdout=m_subprocess.PIPE, stderr=m_subprocess.PIPE
        )
        assert host == endpoint["result"]
