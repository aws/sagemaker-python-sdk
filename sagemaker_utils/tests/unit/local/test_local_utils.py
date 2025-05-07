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
from sagemaker.utils.local.utils import copy_directory_structure, move_to_destination, recursive_copy, \
    kill_child_processes, get_child_process_ids, get_docker_host, get_using_dot_notation, check_for_studio

from sagemaker.utils.session_settings import SessionSettings


@patch("sagemaker.utils.local.utils.os.path")
@patch("sagemaker.utils.local.utils.os")
def test_copy_directory_structure(m_os, m_os_path):
    m_os_path.exists.return_value = False
    copy_directory_structure("/tmp/", "code/")
    m_os.makedirs.assert_called_with("/tmp/", "code/")


@patch("shutil.rmtree", Mock())
@patch("sagemaker.utils.local.utils.recursive_copy")
def test_move_to_destination_local(recursive_copy):
    # local files will just be recursively copied
    # given absolute path
    move_to_destination("/tmp/data", "file:///target/dir", "job", None)
    recursive_copy.assert_called_with("/tmp/data", "/target/dir")
    # given relative path
    move_to_destination("/tmp/data", "file://root/target/dir", "job", None)
    recursive_copy.assert_called_with("/tmp/data", os.path.abspath("./root/target/dir"))


@patch("shutil.rmtree", Mock())
@patch("sagemaker.utils.local.utils.recursive_copy")
def test_move_to_destination_s3(recursive_copy):
    sms = Mock(
        settings=SessionSettings(),
    )

    # without trailing slash in prefix
    move_to_destination("/tmp/data", "s3://bucket/path", "job", sms)
    sms.upload_data.assert_called_with("/tmp/data", "bucket", "path/job")
    recursive_copy.assert_not_called()

    # with trailing slash in prefix
    move_to_destination("/tmp/data", "s3://bucket/path/", "job", sms)
    sms.upload_data.assert_called_with("/tmp/data", "bucket", "path/job")

    # without path, with trailing slash
    move_to_destination("/tmp/data", "s3://bucket/", "job", sms)
    sms.upload_data.assert_called_with("/tmp/data", "bucket", "job")

    # without path, without trailing slash
    move_to_destination("/tmp/data", "s3://bucket", "job", sms)
    sms.upload_data.assert_called_with("/tmp/data", "bucket", "job")


@patch("shutil.rmtree", Mock())
def test_move_to_destination_s3_with_prefix():
    sms = Mock(
        settings=SessionSettings(),
    )
    uri = move_to_destination(
        "/tmp/data", "s3://bucket/path", "job", sms, "foo_prefix"
    )
    sms.upload_data.assert_called_with("/tmp/data", "bucket", "path/job/foo_prefix")
    assert uri == "s3://bucket/path/job/foo_prefix"


def test_move_to_destination_illegal_destination():
    with pytest.raises(ValueError):
        move_to_destination("/tmp/data", "ftp://ftp/in/2018", "job", None)


@patch("sagemaker.utils.local.utils.os.path")
@patch("sagemaker.utils.local.utils.shutil.copytree")
def test_recursive_copy(copy_tree, m_os_path):
    m_os_path.isdir.return_value = True
    recursive_copy("source", "destination")
    copy_tree.assert_called_with("source", "destination", dirs_exist_ok=True)


@patch("sagemaker.utils.local.utils.os")
@patch("sagemaker.utils.local.utils.get_child_process_ids")
def test_kill_child_processes(m_get_child_process_ids, m_os):
    m_get_child_process_ids.return_value = ["child_pids"]
    kill_child_processes("pid")
    m_os.kill.assert_called_with("child_pids", 15)


@patch("sagemaker.utils.local.utils.subprocess")
def test_get_child_process_ids(m_subprocess):
    cmd = "pgrep -P pid".split()
    process_mock = Mock()
    attrs = {"communicate.return_value": (b"\n", False), "returncode": 0}
    process_mock.configure_mock(**attrs)
    m_subprocess.Popen.return_value = process_mock
    get_child_process_ids("pid")
    m_subprocess.Popen.assert_called_with(cmd, stdout=m_subprocess.PIPE, stderr=m_subprocess.PIPE)


@patch("sagemaker.utils.local.utils.subprocess")
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
        host = get_docker_host()
        m_subprocess.Popen.assert_called_with(
            cmd, stdout=m_subprocess.PIPE, stderr=m_subprocess.PIPE
        )
        assert host == endpoint["result"]


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
    actual = get_using_dot_notation(resume, json_path)
    assert actual == expected


def test_get_using_dot_notation_type_error():
    with pytest.raises(ValueError):
        get_using_dot_notation({"foo": "bar"}, "foo.test")


def test_get_using_dot_notation_key_error():
    with pytest.raises(ValueError):
        get_using_dot_notation({"foo": {"bar": 1}}, "foo.test")


def test_get_using_dot_notation_index_error():
    with pytest.raises(ValueError):
        get_using_dot_notation({"foo": ["bar"]}, "foo[1]")


def raise_os_error(args):
    err = OSError()
    err.errno = errno.EACCES
    raise err


@patch("shutil.rmtree", side_effect=raise_os_error)
@patch("sagemaker.utils.local.utils.recursive_copy")
def test_move_to_destination_local_root_failure(recursive_copy, mock_rmtree):
    # This should not raise, in case root owns files, make sure it doesn't
    move_to_destination("/tmp/data", "file:///target/dir/", "job", None)
    mock_rmtree.assert_called_once()
    recursive_copy.assert_called_with(
        "/tmp/data", os.path.abspath(os.path.join(os.sep, "target", "dir"))
    )


def test_check_for_studio_with_valid_request():
    metadata = {"AppType": "KernelGateway"}
    with patch("builtins.open", mock_open(read_data=json.dumps(metadata))):
        with patch("os.path.exists", return_value=True):
            is_studio = check_for_studio()
            assert is_studio is True


def test_check_for_studio_with_invalid_request():
    metadata = {"AppType": "DUMMY"}
    with patch("builtins.open", mock_open(read_data=json.dumps(metadata))):
        with patch("os.path.exists", return_value=True):
            with pytest.raises(NotImplementedError):
                check_for_studio()


def test_check_for_studio_without_app_type():
    metadata = {}
    with patch("builtins.open", mock_open(read_data=json.dumps(metadata))):
        with patch("os.path.exists", return_value=True):
            is_studio = check_for_studio()
            assert is_studio is False
