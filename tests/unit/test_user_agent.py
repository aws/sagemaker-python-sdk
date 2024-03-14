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

import json
from mock import MagicMock, patch, mock_open


from sagemaker.user_agent import (
    SDK_PREFIX,
    SDK_VERSION,
    PYTHON_VERSION,
    OS_NAME_VERSION,
    NOTEBOOK_PREFIX,
    STUDIO_PREFIX,
    process_notebook_metadata_file,
    process_studio_metadata_file,
    determine_prefix,
    prepend_user_agent,
)


# Test process_notebook_metadata_file function
def test_process_notebook_metadata_file_exists(tmp_path):
    notebook_file = tmp_path / "sagemaker-notebook-instance-version.txt"
    notebook_file.write_text("instance_type")

    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=notebook_file.read_text())):
            assert process_notebook_metadata_file() == "instance_type"


def test_process_notebook_metadata_file_not_exists(tmp_path):
    with patch("os.path.exists", return_value=False):
        assert process_notebook_metadata_file() is None


# Test process_studio_metadata_file function
def test_process_studio_metadata_file_exists(tmp_path):
    studio_file = tmp_path / "resource-metadata.json"
    studio_file.write_text(json.dumps({"AppType": "studio_type"}))

    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=studio_file.read_text())):
            assert process_studio_metadata_file() == "studio_type"


def test_process_studio_metadata_file_not_exists(tmp_path):
    with patch("os.path.exists", return_value=False):
        assert process_studio_metadata_file() is None


# Test determine_prefix function
def test_determine_prefix_notebook_instance_type(monkeypatch):
    monkeypatch.setattr(
        "sagemaker.user_agent.process_notebook_metadata_file", lambda: "instance_type"
    )
    assert (
        determine_prefix()
        == f"{SDK_PREFIX}/{SDK_VERSION} {PYTHON_VERSION} {OS_NAME_VERSION} {NOTEBOOK_PREFIX}/instance_type"
    )


def test_determine_prefix_studio_app_type(monkeypatch):
    monkeypatch.setattr(
        "sagemaker.user_agent.process_studio_metadata_file", lambda: "studio_app_type"
    )
    assert (
        determine_prefix()
        == f"{SDK_PREFIX}/{SDK_VERSION} {PYTHON_VERSION} {OS_NAME_VERSION} {STUDIO_PREFIX}/studio_app_type"
    )


def test_determine_prefix_no_metadata(monkeypatch):
    monkeypatch.setattr("sagemaker.user_agent.process_notebook_metadata_file", lambda: None)
    monkeypatch.setattr("sagemaker.user_agent.process_studio_metadata_file", lambda: None)
    assert determine_prefix() == f"{SDK_PREFIX}/{SDK_VERSION} {PYTHON_VERSION} {OS_NAME_VERSION}"


# Test prepend_user_agent function
def test_prepend_user_agent_existing_user_agent(monkeypatch):
    client = MagicMock()
    client._client_config.user_agent = "existing_user_agent"
    monkeypatch.setattr("sagemaker.user_agent.determine_prefix", lambda _: "prefix")
    prepend_user_agent(client)
    assert client._client_config.user_agent == "prefix existing_user_agent"


def test_prepend_user_agent_no_user_agent(monkeypatch):
    client = MagicMock()
    client._client_config.user_agent = None
    monkeypatch.setattr("sagemaker.user_agent.determine_prefix", lambda _: "prefix")
    prepend_user_agent(client)
    assert client._client_config.user_agent == "prefix"
