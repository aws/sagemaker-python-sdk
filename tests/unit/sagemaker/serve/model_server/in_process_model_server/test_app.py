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
import pytest

from unittest.mock import patch, Mock
from sagemaker.serve.model_server.in_process_model_server.app import InProcessServer
from tests.integ.sagemaker.serve.constants import (
    PYTHON_VERSION_IS_NOT_310,
)

mock_model_id = "mock_model_id"


class TestAppInProcessServer(unittest.TestCase):
    @pytest.mark.skipif(
        PYTHON_VERSION_IS_NOT_310,
        reason="The goal of these tests are to test the serving components of our feature",
    )
    @patch("sagemaker.serve.model_server.in_process_model_server.app.threading")
    @patch("sagemaker.serve.spec.inference_spec.InferenceSpec")
    def test_in_process_server_init(self, mock_inference_spec, mock_threading):
        mock_generator = Mock()
        mock_generator.side_effect = None

        in_process_server = InProcessServer(inference_spec=mock_inference_spec)
        in_process_server._generator = mock_generator

    @pytest.mark.skipif(
        PYTHON_VERSION_IS_NOT_310,
        reason="The goal of these test are to test the serving components of our feature",
    )
    @patch("sagemaker.serve.model_server.in_process_model_server.app.logger")
    @patch("sagemaker.serve.model_server.in_process_model_server.app.threading")
    @patch("sagemaker.serve.spec.inference_spec.InferenceSpec")
    def test_start_server(self, mock_inference_spec, mock_threading, mock_logger):
        mock_generator = Mock()
        mock_generator.side_effect = None
        mock_thread = Mock()
        mock_threading.Thread.return_value = mock_thread

        in_process_server = InProcessServer(inference_spec=mock_inference_spec)
        in_process_server._generator = mock_generator

        in_process_server.start_server()

        mock_logger.info.assert_called()
        mock_thread.start.assert_called()

    @pytest.mark.skipif(
        PYTHON_VERSION_IS_NOT_310,
        reason="The goal of these test are to test the serving components of our feature",
    )
    @patch("sagemaker.serve.model_server.in_process_model_server.app.asyncio")
    @patch("sagemaker.serve.spec.inference_spec.InferenceSpec")
    def test_start_run_async_in_thread(self, mock_inference_spec, mock_asyncio):
        mock_inference_spec.load.side_effect = lambda *args, **kwargs: "Dummy load"

        mock_loop = Mock()
        mock_asyncio.new_event_loop.side_effect = lambda: mock_loop

        in_process_server = InProcessServer(inference_spec=mock_inference_spec)
        in_process_server._start_run_async_in_thread()

        mock_asyncio.set_event_loop.assert_called_once_with(mock_loop)
        mock_loop.run_until_complete.assert_called()

    @patch("sagemaker.serve.spec.inference_spec.InferenceSpec")
    async def test_serve(self, mock_inference_spec):
        mock_inference_spec.load.side_effect = lambda *args, **kwargs: "Dummy load"

        mock_server = Mock()

        in_process_server = InProcessServer(inference_spec=mock_inference_spec)
        in_process_server.server = mock_server

        await in_process_server._serve()

        mock_server.serve.assert_called()
