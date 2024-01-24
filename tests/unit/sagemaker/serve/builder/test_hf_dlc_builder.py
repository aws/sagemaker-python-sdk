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
from unittest.mock import MagicMock, patch

import unittest
from sagemaker.serve.builder.model_builder import ModelBuilder
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve import ModelServer

from sagemaker.serve.utils.predictors import HfDLCLocalModePredictor

mock_model_id = "TheBloke/Llama-2-7b-chat-fp16"
mock_t5_model_id = "google/flan-t5-xxl"
mock_prompt = "Hello, I'm a language model,"
mock_response = "Hello, I'm a language model, and I'm here to help you with your English."
mock_sample_input = {"inputs": mock_prompt, "parameters": {}}
mock_sample_output = [{"generated_text": mock_response}]
mock_expected_huggingfaceaccelerate_serving_properties = {
    "engine": "Python",
    "option.entryPoint": "inference.py",
    "option.model_id": "TheBloke/Llama-2-7b-chat-fp16",
    "option.tensor_parallel_degree": 4,
    "option.dtype": "fp16",
}

mock_schema_builder = MagicMock()
mock_schema_builder.sample_input = mock_sample_input
mock_schema_builder.sample_output = mock_sample_output

mock_schema_builder_invalid = MagicMock()
mock_schema_builder_invalid.sample_input = {"invalid": "format"}
mock_schema_builder_invalid.sample_output = mock_sample_output


class TestHFDlcBuilder(unittest.TestCase):

    @patch("sagemaker.serve.builder.hf_dlc_builder._capture_telemetry", side_effect=None)
    @patch("sagemaker.serve.builder.hf_dlc_builder._get_ram_usage_mb", return_value=1024)
    @patch("sagemaker.serve.builder.djl_builder._get_nb_instance", return_value="ml.g5.24xlarge")
    def test_build_deploy_for_hf_dlc_local_container(
            self,
            mock_get_nb_instance,
            mock_get_ram_usage_mb,
            mock_telemetry,
    ):
        builder = ModelBuilder(
            model=mock_model_id,
            schema_builder=mock_schema_builder,
            mode=Mode.LOCAL_CONTAINER,
            model_server=ModelServer.HF_DLC_SERVER,
        )

        builder._prepare_for_mode = MagicMock()
        builder._prepare_for_mode.side_effect = None

        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True

        assert builder.nb_instance_type == "ml.g5.24xlarge"

        builder.modes[str(Mode.LOCAL_CONTAINER)] = MagicMock()
        predictor = model.deploy(model_data_download_timeout=1800)

        assert builder.env_vars["MODEL_LOADING_TIMEOUT"] == "1800"
        assert isinstance(predictor, HfDLCLocalModePredictor)

        builder._original_deploy = MagicMock()
        builder._prepare_for_mode.return_value = (None, {})
        predictor = model.deploy(mode=Mode.SAGEMAKER_ENDPOINT, role="mock_role_arn")
        assert "HF_MODEL_ID" in model.env
        assert "HUGGING_FACE_HUB_TOKEN" in model.env
        assert isinstance(predictor, HfDLCLocalModePredictor)

        with self.assertRaises(ValueError) as _:
            model.deploy(mode=Mode.IN_PROCESS)
