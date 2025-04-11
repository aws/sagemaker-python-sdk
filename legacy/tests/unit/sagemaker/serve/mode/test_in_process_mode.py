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
from unittest.mock import patch, Mock

from sagemaker.serve.mode.in_process_mode import InProcessMode
from sagemaker.serve import SchemaBuilder
from sagemaker.serve.utils.exceptions import InProcessDeepPingException


mock_prompt = "Hello, I'm a language model,"
mock_response = "Hello, I'm a language model, and I'm here to help you with your English."
mock_sample_input = {"inputs": mock_prompt, "parameters": {}}
mock_sample_output = [{"generated_text": mock_response}]
mock_model = "gpt2"


class TestInProcessMode(unittest.TestCase):

    @patch("sagemaker.serve.mode.in_process_mode.Path")
    @patch("sagemaker.serve.spec.inference_spec.InferenceSpec")
    @patch("sagemaker.session.Session")
    def test_load_happy_transformers(self, mock_session, mock_inference_spec, mock_path):
        mock_path.return_value.exists.side_effect = lambda *args, **kwargs: True
        mock_path.return_value.is_dir.side_effect = lambda *args, **kwargs: True

        mock_inference_spec.load.side_effect = lambda *args, **kwargs: "Dummy load"

        mock_schema_builder = SchemaBuilder(mock_sample_input, mock_sample_output)
        in_process_mode = InProcessMode(
            inference_spec=mock_inference_spec,
            model=mock_model,
            schema_builder=mock_schema_builder,
            session=mock_session,
            model_path="model_path",
            env_vars={"key": "val"},
        )

        res = in_process_mode.load(model_path="/tmp/model-builder/code/")

        self.assertEqual(res, "Dummy load")
        self.assertEqual(in_process_mode.inference_spec, mock_inference_spec)
        self.assertEqual(in_process_mode.schema_builder, mock_schema_builder)
        self.assertEqual(in_process_mode.model_path, "model_path")
        self.assertEqual(in_process_mode.env_vars, {"key": "val"})

    @patch("sagemaker.serve.mode.in_process_mode.Path")
    @patch("sagemaker.serve.spec.inference_spec.InferenceSpec")
    @patch("sagemaker.session.Session")
    def test_load_happy_djl_serving(self, mock_session, mock_inference_spec, mock_path):
        mock_path.return_value.exists.side_effect = lambda *args, **kwargs: True
        mock_path.return_value.is_dir.side_effect = lambda *args, **kwargs: True

        mock_inference_spec.load.side_effect = lambda *args, **kwargs: "Dummy load"

        mock_schema_builder = SchemaBuilder(mock_sample_input, mock_sample_output)
        in_process_mode = InProcessMode(
            inference_spec=mock_inference_spec,
            model=mock_model,
            schema_builder=mock_schema_builder,
            session=mock_session,
            model_path="model_path",
            env_vars={"key": "val"},
        )

        res = in_process_mode.load(model_path="/tmp/model-builder/code/")

        self.assertEqual(res, "Dummy load")
        self.assertEqual(in_process_mode.inference_spec, mock_inference_spec)
        self.assertEqual(in_process_mode.schema_builder, mock_schema_builder)
        self.assertEqual(in_process_mode.model_path, "model_path")
        self.assertEqual(in_process_mode.env_vars, {"key": "val"})

    @patch("sagemaker.serve.mode.in_process_mode.Path")
    @patch("sagemaker.serve.spec.inference_spec.InferenceSpec")
    @patch("sagemaker.session.Session")
    def test_load_ex(self, mock_session, mock_inference_spec, mock_path):
        mock_path.return_value.exists.side_effect = lambda *args, **kwargs: False
        mock_path.return_value.is_dir.side_effect = lambda *args, **kwargs: True

        mock_inference_spec.load.side_effect = lambda *args, **kwargs: "Dummy load"

        mock_schema_builder = SchemaBuilder(mock_sample_input, mock_sample_output)
        in_process_mode = InProcessMode(
            inference_spec=mock_inference_spec,
            model=mock_model,
            schema_builder=mock_schema_builder,
            session=mock_session,
            model_path="model_path",
        )

        self.assertRaises(ValueError, in_process_mode.load, "/tmp/model-builder/code/")

        mock_path.return_value.exists.side_effect = lambda *args, **kwargs: True
        mock_path.return_value.is_dir.side_effect = lambda *args, **kwargs: False

        mock_inference_spec.load.side_effect = lambda *args, **kwargs: "Dummy load"
        mock_schema_builder = SchemaBuilder(mock_sample_input, mock_sample_output)
        in_process_mode = InProcessMode(
            inference_spec=mock_inference_spec,
            model=mock_model,
            schema_builder=mock_schema_builder,
            session=mock_session,
            model_path="model_path",
        )

        self.assertRaises(ValueError, in_process_mode.load, "/tmp/model-builder/code/")

    @patch("sagemaker.serve.mode.in_process_mode.logger")
    @patch("sagemaker.base_predictor.PredictorBase")
    @patch("sagemaker.serve.spec.inference_spec.InferenceSpec")
    @patch("sagemaker.session.Session")
    def test_create_server_happy(
        self, mock_session, mock_inference_spec, mock_predictor, mock_logger
    ):
        mock_start_serving = Mock()
        mock_start_serving.side_effect = lambda *args, **kwargs: (
            True,
            None,
        )

        mock_response = "Fake response"
        mock_multi_model_server_deep_ping = Mock()
        mock_multi_model_server_deep_ping.side_effect = lambda *args, **kwargs: (
            True,
            mock_response,
        )

        in_process_mode = InProcessMode(
            inference_spec=mock_inference_spec,
            model=mock_model,
            schema_builder=SchemaBuilder(mock_sample_input, mock_sample_output),
            session=mock_session,
            model_path="model_path",
        )

        in_process_mode._deep_ping = mock_multi_model_server_deep_ping
        in_process_mode._start_serving = mock_start_serving

        in_process_mode.create_server(predictor=mock_predictor)

        mock_logger.info.assert_called_once_with("Waiting for fastapi server to start up...")
        mock_logger.debug.assert_called_once_with(
            "Ping health check has passed. Returned %s", str(mock_response)
        )

    @patch("sagemaker.base_predictor.PredictorBase")
    @patch("sagemaker.serve.spec.inference_spec.InferenceSpec")
    @patch("sagemaker.session.Session")
    def test_create_server_ex(
        self,
        mock_session,
        mock_inference_spec,
        mock_predictor,
    ):
        mock_start_serving = Mock()
        mock_start_serving.side_effect = lambda *args, **kwargs: (
            True,
            None,
        )

        mock_multi_model_server_deep_ping = Mock()
        mock_multi_model_server_deep_ping.side_effect = lambda *args, **kwargs: (
            False,
            None,
        )

        in_process_mode = InProcessMode(
            inference_spec=mock_inference_spec,
            model=mock_model,
            schema_builder=SchemaBuilder(mock_sample_input, mock_sample_output),
            session=mock_session,
            model_path="model_path",
        )

        in_process_mode._deep_ping = mock_multi_model_server_deep_ping
        in_process_mode._start_serving = mock_start_serving

        self.assertRaises(InProcessDeepPingException, in_process_mode.create_server, mock_predictor)

    @patch(
        "sagemaker.serve.model_server.in_process_model_server.in_process_server.InProcessServing._stop_serving"
    )
    @patch("sagemaker.serve.spec.inference_spec.InferenceSpec")
    @patch("sagemaker.session.Session")
    def test_destroy_server(
        self,
        mock_session,
        mock_inference_spec,
        mock_stop_serving,
    ):
        in_process_mode = InProcessMode(
            inference_spec=mock_inference_spec,
            model=mock_model,
            schema_builder=SchemaBuilder(mock_sample_input, mock_sample_output),
            session=mock_session,
            model_path="model_path",
        )

        in_process_mode.destroy_server()

        mock_stop_serving.assert_called()
