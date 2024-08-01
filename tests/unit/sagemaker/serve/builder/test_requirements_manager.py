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

import unittest
from unittest.mock import patch, call

from sagemaker.serve.builder.requirements_manager import RequirementsManager


class TestRequirementsManager(unittest.TestCase):

    @patch(
        "sagemaker.serve.builder.requirements_manager.RequirementsManager._update_conda_env_in_path"
    )
    @patch(
        "sagemaker.serve.builder.requirements_manager.RequirementsManager._install_requirements_txt"
    )
    @patch(
        "sagemaker.serve.builder.requirements_manager.RequirementsManager._capture_from_local_runtime"
    )
    def test_detect_file_exists(
        self,
        mock_capture_from_local_runtime,
        mock_install_requirements_txt,
        mock_update_conda_env_in_path,
    ) -> str:

        mock_capture_from_local_runtime.side_effect = lambda: ".txt"
        RequirementsManager().detect_file_exists()
        mock_install_requirements_txt.assert_called_once()

        mock_capture_from_local_runtime.side_effect = lambda: ".yml"
        RequirementsManager().detect_file_exists()
        mock_update_conda_env_in_path.assert_called_once()

    @patch(
        "sagemaker.serve.builder.requirements_manager.RequirementsManager._capture_from_local_runtime"
    )
    def test_detect_file_exists_fail(self, mock__capture_from_local_runtime) -> str:
        mock_dependencies = "mock.ini"
        mock__capture_from_local_runtime.side_effect = lambda: "invalid requirement"
        self.assertRaises(
            ValueError, lambda: RequirementsManager().detect_file_exists(mock_dependencies)
        )

    @patch("sagemaker.serve.builder.requirements_manager.logger")
    @patch("sagemaker.serve.builder.requirements_manager.subprocess")
    def test_install_requirements_txt(self, mock_subprocess, mock_logger):

        RequirementsManager()._install_requirements_txt()

        calls = [call("Running command to pip install"), call("Command ran successfully")]
        mock_logger.info.assert_has_calls(calls)
        mock_subprocess.run.assert_called_once_with(
            "pip install -r requirements.txt", shell=True, check=True
        )

    @patch("sagemaker.serve.builder.requirements_manager.logger")
    @patch("sagemaker.serve.builder.requirements_manager.subprocess")
    def test_update_conda_env_in_path(self, mock_subprocess, mock_logger):

        RequirementsManager()._update_conda_env_in_path()

        calls = [call("Updating conda env"), call("Conda env updated successfully")]
        mock_logger.info.assert_has_calls(calls)
        mock_subprocess.run.assert_called_once_with(
            "conda env update -f conda_in_process.yml", shell=True, check=True
        )
