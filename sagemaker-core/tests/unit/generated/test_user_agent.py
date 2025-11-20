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
from mock import patch, mock_open


from sagemaker.core.utils.user_agent import (
    SagemakerCore_PREFIX,
    SagemakerCore_VERSION,
    NOTEBOOK_PREFIX,
    STUDIO_PREFIX,
    process_notebook_metadata_file,
    process_studio_metadata_file,
    get_user_agent_extra_suffix,
)
from sagemaker.core.utils.user_agent import SagemakerCore_PREFIX


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


# Test get_user_agent_extra_suffix function
def test_get_user_agent_extra_suffix():
    assert get_user_agent_extra_suffix() == f"lib/{SagemakerCore_PREFIX}#{SagemakerCore_VERSION}"

    with patch(
        "sagemaker.core.utils.user_agent.process_notebook_metadata_file",
        return_value="instance_type",
    ):
        assert (
            get_user_agent_extra_suffix()
            == f"lib/{SagemakerCore_PREFIX}#{SagemakerCore_VERSION} md/{NOTEBOOK_PREFIX}#instance_type"
        )

    with patch(
        "sagemaker.core.utils.user_agent.process_studio_metadata_file", return_value="studio_type"
    ):
        assert (
            get_user_agent_extra_suffix()
            == f"lib/{SagemakerCore_PREFIX}#{SagemakerCore_VERSION} md/{STUDIO_PREFIX}#studio_type"
        )
