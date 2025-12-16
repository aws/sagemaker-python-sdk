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
import errno
import pytest
import json
from mock import patch, Mock, mock_open

import sagemaker.local.utils
from sagemaker.session_settings import SessionSettings


@patch("sagemaker.local.utils.os.path")
@patch("sagemaker.local.utils.os")
def test_copy_directory_structure(m_os, m_os_path):
    m_os_path.join.return_value = "/tmp/code/"
    sagemaker.local.utils.copy_directory_structure("/tmp/", "code/")
    m_os.makedirs.assert_called_with("/tmp/code/", exist_ok=True)


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
    sms = Mock(
        settings=SessionSettings(),
    )

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


@patch("shutil.rmtree", Mock())
def test_move_to_destination_s3_with_prefix():
    sms = Mock(
        settings=SessionSettings(),
    )
    uri = sagemaker.local.utils.move_to_destination(
        "/tmp/data", "s3://bucket/path", "job", sms, "foo_prefix"
    )
    sms.upload_data.assert_called_with("/tmp/data", "bucket", "path/job/foo_prefix")
    assert uri == "s3://bucket/path/job/foo_prefix"


def test_move_to_destination_illegal_destination():
    with pytest.raises(ValueError):
        sagemaker.local.utils.move_to_destination("/tmp/data", "ftp://ftp/in/2018", "job", None)


@patch("sagemaker.local.utils.os.path")
@patch("sagemaker.local.utils.shutil.copytree")
def test_recursive_copy(copy_tree, m_os_path):
    m_os_path.isdir.return_value = True
    sagemaker.local.utils.recursive_copy("source", "destination")
    copy_tree.assert_called_with("source", "destination", dirs_exist_ok=True)


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


@patch("sagemaker.local.utils.subprocess")
def test_get_docker_host_rootless_docker(m_subprocess):
    """Test that rootless Docker is detected and returns fixed IP"""
    # Mock docker info process for rootless Docker
    info_process_mock = Mock()
    info_attrs = {"communicate.return_value": (b"Cgroup Driver: none", b""), "returncode": 0}
    info_process_mock.configure_mock(**info_attrs)
    m_subprocess.Popen.return_value = info_process_mock

    host = sagemaker.local.utils.get_docker_host()
    assert host == "172.17.0.1"

    # Verify docker info was called
    m_subprocess.Popen.assert_called_with(
        ["docker", "info"], stdout=m_subprocess.PIPE, stderr=m_subprocess.PIPE
    )


@patch("sagemaker.local.utils.subprocess")
def test_get_docker_host_traditional_docker(m_subprocess):
    """Test that traditional Docker falls back to existing logic"""
    scenarios = [
        {
            "docker_info": b"Cgroup Driver: cgroupfs",
            "context_host": "tcp://host:port",
            "result": "host",
        },
        {
            "docker_info": b"Cgroup Driver: cgroupfs",
            "context_host": "unix:///var/run/docker.sock",
            "result": "localhost",
        },
        {
            "docker_info": b"Cgroup Driver: cgroupfs",
            "context_host": "fd://something",
            "result": "localhost",
        },
    ]

    for scenario in scenarios:
        # Mock docker info process for traditional Docker
        info_process_mock = Mock()
        info_attrs = {"communicate.return_value": (scenario["docker_info"], b""), "returncode": 0}
        info_process_mock.configure_mock(**info_attrs)

        # Mock docker context inspect process
        context_return_value = (
            '[\n{\n"Endpoints":{\n"docker":{\n"Host": "%s"}\n}\n}\n]\n' % scenario["context_host"]
        )
        context_process_mock = Mock()
        context_attrs = {
            "communicate.return_value": (context_return_value.encode("utf-8"), None),
            "returncode": 0,
        }
        context_process_mock.configure_mock(**context_attrs)

        m_subprocess.Popen.side_effect = [info_process_mock, context_process_mock]

        host = sagemaker.local.utils.get_docker_host()
        assert host == scenario["result"]


@pytest.mark.parametrize(
    "json_path, expected",
    [
        ("Name", "John Doe"),
        ("Age", 31),
        ("Experiences[0].Company", "Foo Inc."),
        ("Experiences[0].Tenure", 5),
        ("Experiences[0].Projects[0]['XYZ project']", "Backend Rest Api development"),
        ("Experiences[0].Projects[1]['ABC project']", "Data migration"),
        ("Experiences[1].Company", "Bar Ltd."),
        ("Experiences[1].Tenure", 2),
    ],
)
def test_get_using_dot_notation(json_path, expected):
    resume = {
        "Name": "John Doe",
        "Age": 31,
        "Experiences": [
            {
                "Company": "Foo Inc.",
                "Role": "SDE",
                "Tenure": 5,
                "Projects": [
                    {"XYZ project": "Backend Rest Api development"},
                    {"ABC project": "Data migration"},
                ],
            },
            {"Company": "Bar Ltd.", "Role": "Web developer", "Tenure": 2},
        ],
    }
    actual = sagemaker.local.utils.get_using_dot_notation(resume, json_path)
    assert actual == expected


def test_get_using_dot_notation_type_error():
    with pytest.raises(ValueError):
        sagemaker.local.utils.get_using_dot_notation({"foo": "bar"}, "foo.test")


def test_get_using_dot_notation_key_error():
    with pytest.raises(ValueError):
        sagemaker.local.utils.get_using_dot_notation({"foo": {"bar": 1}}, "foo.test")


def test_get_using_dot_notation_index_error():
    with pytest.raises(ValueError):
        sagemaker.local.utils.get_using_dot_notation({"foo": ["bar"]}, "foo[1]")


def raise_os_error(args):
    err = OSError()
    err.errno = errno.EACCES
    raise err


@patch("shutil.rmtree", side_effect=raise_os_error)
@patch("sagemaker.local.utils.recursive_copy")
def test_move_to_destination_local_root_failure(recursive_copy, mock_rmtree):
    # This should not raise, in case root owns files, make sure it doesn't
    sagemaker.local.utils.move_to_destination("/tmp/data", "file:///target/dir/", "job", None)
    mock_rmtree.assert_called_once()
    recursive_copy.assert_called_with(
        "/tmp/data", os.path.abspath(os.path.join(os.sep, "target", "dir"))
    )


def test_check_for_studio_with_valid_request():
    metadata = {"AppType": "KernelGateway"}
    with patch("builtins.open", mock_open(read_data=json.dumps(metadata))):
        with patch("os.path.exists", return_value=True):
            is_studio = sagemaker.local.utils.check_for_studio()
            assert is_studio is True


def test_check_for_studio_with_invalid_request():
    metadata = {"AppType": "DUMMY"}
    with patch("builtins.open", mock_open(read_data=json.dumps(metadata))):
        with patch("os.path.exists", return_value=True):
            with pytest.raises(NotImplementedError):
                sagemaker.local.utils.check_for_studio()


def test_check_for_studio_without_app_type():
    metadata = {}
    with patch("builtins.open", mock_open(read_data=json.dumps(metadata))):
        with patch("os.path.exists", return_value=True):
            is_studio = sagemaker.local.utils.check_for_studio()
            assert is_studio is False
