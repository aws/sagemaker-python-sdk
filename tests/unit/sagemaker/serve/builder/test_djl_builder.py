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
from sagemaker.djl_inference.model import DJLModel
from sagemaker.serve.utils.exceptions import (
    LocalDeepPingException,
    LocalModelLoadException,
    LocalModelOutOfMemoryException,
    LocalModelInvocationException,
)
from sagemaker.serve.utils.predictors import DjlLocalModePredictor, InProcessModePredictor
from tests.unit.sagemaker.serve.constants import MOCK_IMAGE_CONFIG, MOCK_VPC_CONFIG

mock_model_id = "TheBloke/Llama-2-7b-chat-fp16"
mock_prompt = "Hello, I'm a language model,"
mock_response = "Hello, I'm a language model, and I'm here to help you with your English."
mock_sample_input = {"inputs": mock_prompt, "parameters": {}}
mock_sample_output = [{"generated_text": mock_response}]
mock_default_configs = {
    "HF_MODEL_ID": mock_model_id,
    "OPTION_ENGINE": "Python",
    "TENSOR_PARALLEL_DEGREE": "max",
    "OPTION_DTYPE": "bf16",
    "MODEL_LOADING_TIMEOUT": "1800",
}
mock_most_performant_serving_properties = {
    "OPTION_ENGINE": "Python",
    "HF_MODEL_ID": "TheBloke/Llama-2-7b-chat-fp16",
    "TENSOR_PARALLEL_DEGREE": "1",
    "OPTION_DTYPE": "bf16",
    "MODEL_LOADING_TIMEOUT": "1800",
}
mock_inference_spec = MagicMock()
mock_inference_spec.get_model = "TheBloke/Llama-2-7b-chat-fp16"

mock_schema_builder = MagicMock()
mock_schema_builder.sample_input = mock_sample_input
mock_schema_builder.sample_output = mock_sample_output

mock_schema_builder_invalid = MagicMock()
mock_schema_builder_invalid.sample_input = {"invalid": "format"}
mock_schema_builder_invalid.sample_output = mock_sample_output


class TestDjlBuilder(unittest.TestCase):
    @patch("sagemaker.serve.builder.djl_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=False,
    )
    @patch("sagemaker.serve.builder.djl_builder._get_ram_usage_mb", return_value=1024)
    @patch("sagemaker.serve.builder.djl_builder._get_nb_instance", return_value="ml.g5.24xlarge")
    @patch(
        "sagemaker.serve.builder.djl_builder._get_default_djl_configurations",
        return_value=(mock_default_configs, 128),
    )
    def test_build_deploy_for_djl_local_container(
        self,
        mock_default_djl_config,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_is_jumpstart_model,
        mock_telemetry,
    ):
        builder = ModelBuilder(
            model=mock_model_id,
            name="mock_model_name",
            schema_builder=mock_schema_builder,
            mode=Mode.LOCAL_CONTAINER,
            model_server=ModelServer.DJL_SERVING,
            image_config=MOCK_IMAGE_CONFIG,
            vpc_config=MOCK_VPC_CONFIG,
        )

        builder._prepare_for_mode = MagicMock()
        builder._prepare_for_mode.side_effect = None

        model = builder.build()
        assert model.name == "mock_model_name"

        builder.serve_settings.telemetry_opt_out = True

        assert isinstance(model, DJLModel)
        assert builder.schema_builder.sample_input["parameters"]["max_new_tokens"] == 128
        assert builder.nb_instance_type == "ml.g5.24xlarge"
        assert model.image_config == MOCK_IMAGE_CONFIG
        assert model.vpc_config == MOCK_VPC_CONFIG
        assert "lmi" in builder.image_uri

        builder.modes[str(Mode.LOCAL_CONTAINER)] = MagicMock()
        predictor = model.deploy(model_data_download_timeout=1800)

        assert builder.env_vars["MODEL_LOADING_TIMEOUT"] == "1800"
        assert isinstance(predictor, DjlLocalModePredictor)

        builder._original_deploy = MagicMock()
        builder._prepare_for_mode.return_value = (None, {})
        predictor = model.deploy(mode=Mode.SAGEMAKER_ENDPOINT, role="mock_role_arn")
        assert "TRANSFORMERS_OFFLINE" in model.env

        with self.assertRaises(ValueError) as _:
            model.deploy(mode=Mode.IN_PROCESS)

    @patch("sagemaker.serve.builder.djl_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=False,
    )
    @patch("sagemaker.serve.builder.djl_builder._get_ram_usage_mb", return_value=1024)
    @patch("sagemaker.serve.builder.djl_builder._get_nb_instance", return_value="ml.g5.24xlarge")
    @patch(
        "sagemaker.serve.builder.djl_builder._get_default_djl_configurations",
        return_value=(mock_default_configs, 128),
    )
    def test_build_deploy_for_djl_in_process(
        self,
        mock_default_djl_config,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_is_jumpstart_model,
        mock_telemetry,
    ):
        builder = ModelBuilder(
            model=mock_model_id,
            name="mock_model_name",
            schema_builder=mock_schema_builder,
            mode=Mode.IN_PROCESS,
            model_server=ModelServer.DJL_SERVING,
            image_config=MOCK_IMAGE_CONFIG,
            vpc_config=MOCK_VPC_CONFIG,
        )

        builder._prepare_for_mode = MagicMock()
        builder._prepare_for_mode.side_effect = None

        model = builder.build()
        assert model.name == "mock_model_name"

        builder.serve_settings.telemetry_opt_out = True

        assert isinstance(model, DJLModel)
        assert builder.schema_builder.sample_input["parameters"]["max_new_tokens"] == 128
        assert builder.nb_instance_type == "ml.g5.24xlarge"
        assert model.image_config == MOCK_IMAGE_CONFIG
        assert model.vpc_config == MOCK_VPC_CONFIG
        assert "lmi" in builder.image_uri

        builder.modes[str(Mode.IN_PROCESS)] = MagicMock()
        predictor = model.deploy(model_data_download_timeout=1800)

        assert builder.env_vars["MODEL_LOADING_TIMEOUT"] == "1800"
        assert isinstance(predictor, InProcessModePredictor)

        builder._original_deploy = MagicMock()
        builder._prepare_for_mode.return_value = (None, {})

    @patch("sagemaker.serve.builder.djl_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=False,
    )
    @patch("sagemaker.serve.builder.djl_builder._get_ram_usage_mb", return_value=1024)
    @patch("sagemaker.serve.builder.djl_builder._get_nb_instance", return_value="ml.g5.24xlarge")
    @patch(
        "sagemaker.serve.builder.djl_builder._get_admissible_tensor_parallel_degrees",
        return_value=[4, 2, 1],
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._serial_benchmark",
        side_effect=[(5.6, 5.6, 18), (5.4, 5.4, 20), (5.2, 5.2, 25)],
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._concurrent_benchmark",
        side_effect=[(0.03, 16), (0.10, 4), (0.15, 2)],
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._get_default_djl_configurations",
        return_value=(mock_default_configs, 128),
    )
    def test_tune_for_djl_local_container(
        self,
        mock_default_djl_config,
        mock_concurrent_benchmarks,
        mock_serial_benchmarks,
        mock_admissible_tensor_parallel_degrees,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_is_jumpstart_model,
        mock_telemetry,
    ):
        builder = ModelBuilder(
            model=mock_model_id,
            schema_builder=mock_schema_builder,
            mode=Mode.LOCAL_CONTAINER,
            model_server=ModelServer.DJL_SERVING,
        )
        builder._prepare_for_mode = MagicMock()
        builder._prepare_for_mode.side_effect = None
        builder._djl_model_builder_deploy_wrapper = MagicMock()

        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True
        tuned_model = model.tune()
        assert tuned_model.env == mock_most_performant_serving_properties

    @patch("sagemaker.serve.builder.djl_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=False,
    )
    @patch("sagemaker.serve.builder.djl_builder._get_ram_usage_mb", return_value=1024)
    @patch("sagemaker.serve.builder.djl_builder._get_nb_instance", return_value="ml.g5.24xlarge")
    @patch(
        "sagemaker.serve.builder.djl_builder._serial_benchmark",
        **{"return_value.raiseError.side_effect": LocalDeepPingException("mock_exception")},
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._get_admissible_tensor_parallel_degrees",
        return_value=[4],
    )
    @patch("sagemaker.serve.model_server.djl_serving.utils._get_available_gpus", return_value=None)
    def test_tune_for_djl_local_container_deep_ping_ex(
        self,
        mock_get_available_gpus,
        mock_get_admissible_tensor_parallel_degrees,
        mock_serial_benchmarks,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_is_jumpstart_model,
        mock_telemetry,
    ):
        builder = ModelBuilder(
            model=mock_model_id,
            schema_builder=mock_schema_builder,
            mode=Mode.LOCAL_CONTAINER,
            model_server=ModelServer.DJL_SERVING,
        )
        builder._prepare_for_mode = MagicMock()
        builder._prepare_for_mode.side_effect = None

        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True
        tuned_model = model.tune()
        assert tuned_model.env == mock_default_configs

    @patch("sagemaker.serve.builder.djl_builder._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.builder.djl_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=False,
    )
    @patch("sagemaker.serve.builder.djl_builder._get_ram_usage_mb", return_value=1024)
    @patch("sagemaker.serve.builder.djl_builder._get_nb_instance", return_value="ml.g5.24xlarge")
    @patch(
        "sagemaker.serve.builder.djl_builder._serial_benchmark",
        **{"return_value.raiseError.side_effect": LocalModelLoadException("mock_exception")},
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._get_admissible_tensor_parallel_degrees",
        return_value=[4],
    )
    @patch("sagemaker.serve.model_server.djl_serving.utils._get_available_gpus", return_value=None)
    def test_tune_for_djl_local_container_load_ex(
        self,
        mock_get_available_gpus,
        mock_get_admissible_tensor_parallel_degrees,
        mock_serial_benchmarks,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_is_jumpstart_model,
        mock_telemetry,
        mock_get_model_config_properties_from_hf,
    ):
        mock_get_model_config_properties_from_hf.return_value = {}

        builder = ModelBuilder(
            model=mock_model_id,
            schema_builder=mock_schema_builder,
            mode=Mode.LOCAL_CONTAINER,
            model_server=ModelServer.DJL_SERVING,
        )
        builder._prepare_for_mode = MagicMock()
        builder._prepare_for_mode.side_effect = None

        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True
        tuned_model = model.tune()
        assert tuned_model.env == mock_default_configs

    @patch("sagemaker.serve.builder.djl_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=False,
    )
    @patch("sagemaker.serve.builder.djl_builder._get_ram_usage_mb", return_value=1024)
    @patch("sagemaker.serve.builder.djl_builder._get_nb_instance", return_value="ml.g5.24xlarge")
    @patch(
        "sagemaker.serve.builder.djl_builder._serial_benchmark",
        **{"return_value.raiseError.side_effect": LocalModelOutOfMemoryException("mock_exception")},
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._get_admissible_tensor_parallel_degrees",
        return_value=[4],
    )
    @patch("sagemaker.serve.model_server.djl_serving.utils._get_available_gpus", return_value=None)
    def test_tune_for_djl_local_container_oom_ex(
        self,
        mock_get_available_gpus,
        mock_get_admissible_tensor_parallel_degrees,
        mock_serial_benchmarks,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_is_jumpstart_model,
        mock_telemetry,
    ):
        builder = ModelBuilder(
            model=mock_model_id,
            schema_builder=mock_schema_builder,
            mode=Mode.LOCAL_CONTAINER,
            model_server=ModelServer.DJL_SERVING,
        )
        builder._prepare_for_mode = MagicMock()
        builder._prepare_for_mode.side_effect = None

        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True
        tuned_model = model.tune()
        assert tuned_model.env == mock_default_configs

    @patch("sagemaker.serve.builder.djl_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=False,
    )
    @patch("sagemaker.serve.builder.djl_builder._get_ram_usage_mb", return_value=1024)
    @patch("sagemaker.serve.builder.djl_builder._get_nb_instance", return_value="ml.g5.24xlarge")
    @patch(
        "sagemaker.serve.builder.djl_builder._serial_benchmark",
        **{"return_value.raiseError.side_effect": LocalModelInvocationException("mock_exception")},
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._get_admissible_tensor_parallel_degrees",
        return_value=[4],
    )
    @patch("sagemaker.serve.model_server.djl_serving.utils._get_available_gpus", return_value=None)
    def test_tune_for_djl_local_container_invoke_ex(
        self,
        mock_get_available_gpus,
        mock_get_admissible_tensor_parallel_degrees,
        mock_serial_benchmarks,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_is_jumpstart_model,
        mock_telemetry,
    ):
        builder = ModelBuilder(
            model=mock_model_id,
            schema_builder=mock_schema_builder,
            mode=Mode.LOCAL_CONTAINER,
            model_server=ModelServer.DJL_SERVING,
        )
        builder._prepare_for_mode = MagicMock()
        builder._prepare_for_mode.side_effect = None

        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True
        tuned_model = model.tune()
        assert tuned_model.env == mock_default_configs

    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=False,
    )
    def test_sample_data_validations(self, mock_is_jumpstart_model):
        builder = ModelBuilder(
            model=mock_model_id,
            schema_builder=mock_schema_builder_invalid,
            mode=Mode.LOCAL_CONTAINER,
            model_server=ModelServer.DJL_SERVING,
        )

        with self.assertRaises(ValueError) as _:
            builder.build()
