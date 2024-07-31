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
import subprocess
from unittest.mock import patch, Mock

from sagemaker.serve.mode.in_process_mode import InProcessMode
from sagemaker.serve.builder.requirements_manager import RequirementsManager

class TestRequirementsManager(unittest.TestCase):

    def test_detect_file_exists_fail(self, mock_dependencies: str = None) -> str:
        mock_dependencies = "mock.ini"
        self.assertRaises(ValueError, RequirementsManager().detect_file_exists(mock_dependencies))

    @patch("sagemaker.serve.mode.in_process_mode.logger")
    @patch("sagemaker.session.Session")
    def test_install_requirements_txt(self, mock_logger):

        mock_logger.info.assert_called_once_with("Running command to pip install")

        mock_logger.info.assert_called_once_with("Command ran successfully")

    @patch("sagemaker.serve.mode.in_process_mode.logger")
    @patch("sagemaker.session.Session")
    def test_update_conda_env_in_path(self, mock_logger):

        mock_logger.info.assert_called_once_with("Updating conda env")


        # mock_multi_model_server_deep_ping = Mock()
        # mock_multi_model_server_deep_ping.side_effect = lambda *args, **kwargs: (
        #     True,
        # )

        # in_process_mode = InProcessMode(
        #     model_server=ModelServer.MMS,
        #     inference_spec=mock_inference_spec,
        #     schema_builder=SchemaBuilder(mock_sample_input, mock_sample_output),
        #     session=mock_session,
        #     model_path="model_path",
        # )

        # in_process_mode._multi_model_server_deep_ping = mock_multi_model_server_deep_ping

        # in_process_mode.create_server(predictor=mock_predictor)

        mock_logger.info.assert_called_once_with("Conda env updated successfully")
