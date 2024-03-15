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
from sagemaker.serve.utils.types import _DjlEngine
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve import ModelServer
from sagemaker.djl_inference.model import (
    DeepSpeedModel,
    FasterTransformerModel,
    HuggingFaceAccelerateModel,
)
from sagemaker.serve.utils.exceptions import (
    LocalDeepPingException,
    LocalModelLoadException,
    LocalModelOutOfMemoryException,
    LocalModelInvocationException,
)
from sagemaker.serve.utils.predictors import DjlLocalModePredictor
from tests.unit.sagemaker.serve.constants import MOCK_IMAGE_CONFIG, MOCK_VPC_CONFIG

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
mock_expected_deepspeed_serving_properties = {
    "engine": "DeepSpeed",
    "option.entryPoint": "inference.py",
    "option.model_id": "TheBloke/Llama-2-7b-chat-fp16",
    "option.tensor_parallel_degree": 4,
    "option.dtype": "fp16",
    "option.max_tokens": 256,
    "option.triangular_masking": True,
    "option.return_tuple": True,
}
mock_expected_fastertransformer_serving_properties = {
    "engine": "FasterTransformer",
    "option.entryPoint": "inference.py",
    "option.model_id": "google/flan-t5-xxl",
    "option.tensor_parallel_degree": 4,
    "option.dtype": "fp16",
}
mock_most_performant_serving_properties = {
    "engine": "Python",
    "option.entryPoint": "inference.py",
    "option.model_id": "TheBloke/Llama-2-7b-chat-fp16",
    "option.tensor_parallel_degree": 1,
    "option.dtype": "bf16",
}
mock_model_config_properties = {"model_type": "llama", "num_attention_heads": 32}
mock_model_config_properties_faster_transformer = {"model_type": "t5", "num_attention_heads": 32}
mock_set_serving_properties = (4, "fp16", 1, 256, 256)

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
    @patch(
        "sagemaker.serve.builder.djl_builder._auto_detect_engine",
        return_value=(_DjlEngine.HUGGINGFACE_ACCELERATE, mock_model_config_properties),
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._set_serve_properties",
        return_value=mock_set_serving_properties,
    )
    @patch("sagemaker.serve.builder.djl_builder.prepare_for_djl_serving", side_effect=None)
    @patch("sagemaker.serve.builder.djl_builder._get_ram_usage_mb", return_value=1024)
    @patch("sagemaker.serve.builder.djl_builder._get_nb_instance", return_value="ml.g5.24xlarge")
    def test_build_deploy_for_djl_local_container(
        self,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_prepare_for_djl_serving,
        mock_set_serving_properties,
        mock_auto_detect_engine,
        mock_is_jumpstart_model,
        mock_telemetry,
    ):
        builder = ModelBuilder(
            model=mock_model_id,
            schema_builder=mock_schema_builder,
            mode=Mode.LOCAL_CONTAINER,
            model_server=ModelServer.DJL_SERVING,
            image_config=MOCK_IMAGE_CONFIG,
            vpc_config=MOCK_VPC_CONFIG,
        )

        builder._prepare_for_mode = MagicMock()
        builder._prepare_for_mode.side_effect = None

        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True

        assert isinstance(model, HuggingFaceAccelerateModel)
        assert (
            model.generate_serving_properties()
            == mock_expected_huggingfaceaccelerate_serving_properties
        )
        assert builder._default_tensor_parallel_degree == 4
        assert builder._default_data_type == "fp16"
        assert builder._default_max_tokens == 256
        assert builder._default_max_new_tokens == 256
        assert builder.schema_builder.sample_input["parameters"]["max_new_tokens"] == 256
        assert builder.nb_instance_type == "ml.g5.24xlarge"
        assert model.image_config == MOCK_IMAGE_CONFIG
        assert model.vpc_config == MOCK_VPC_CONFIG
        assert "deepspeed" in builder.image_uri

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

    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=False,
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._auto_detect_engine",
        return_value=(
            _DjlEngine.FASTER_TRANSFORMER,
            mock_model_config_properties_faster_transformer,
        ),
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._set_serve_properties",
        return_value=mock_set_serving_properties,
    )
    @patch("sagemaker.serve.builder.djl_builder._get_nb_instance", return_value="ml.g5.24xlarge")
    def test_build_for_djl_local_container_faster_transformer(
        self,
        mock_get_nb_instance,
        mock_set_serving_properties,
        mock_auto_detect_engine,
        mock_is_jumpstart_model,
    ):
        builder = ModelBuilder(
            model=mock_t5_model_id,
            schema_builder=mock_schema_builder,
            mode=Mode.LOCAL_CONTAINER,
            model_server=ModelServer.DJL_SERVING,
            image_config=MOCK_IMAGE_CONFIG,
            vpc_config=MOCK_VPC_CONFIG,
        )
        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True

        assert isinstance(model, FasterTransformerModel)
        assert (
            model.generate_serving_properties()
            == mock_expected_fastertransformer_serving_properties
        )
        assert model.image_config == MOCK_IMAGE_CONFIG
        assert model.vpc_config == MOCK_VPC_CONFIG
        assert "fastertransformer" in builder.image_uri

    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=False,
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._auto_detect_engine",
        return_value=(_DjlEngine.DEEPSPEED, mock_model_config_properties),
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._set_serve_properties",
        return_value=mock_set_serving_properties,
    )
    @patch("sagemaker.serve.builder.djl_builder._get_nb_instance", return_value="ml.g5.24xlarge")
    def test_build_for_djl_local_container_deepspeed(
        self,
        mock_get_nb_instance,
        mock_set_serving_properties,
        mock_auto_detect_engine,
        mock_is_jumpstart_model,
    ):
        builder = ModelBuilder(
            model=mock_model_id,
            schema_builder=mock_schema_builder,
            mode=Mode.LOCAL_CONTAINER,
            model_server=ModelServer.DJL_SERVING,
            image_config=MOCK_IMAGE_CONFIG,
            vpc_config=MOCK_VPC_CONFIG,
        )
        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True

        assert isinstance(model, DeepSpeedModel)
        assert model.image_config == MOCK_IMAGE_CONFIG
        assert model.vpc_config == MOCK_VPC_CONFIG
        assert model.generate_serving_properties() == mock_expected_deepspeed_serving_properties
        assert "deepspeed" in builder.image_uri

    @patch("sagemaker.serve.builder.djl_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=False,
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._auto_detect_engine",
        return_value=(_DjlEngine.HUGGINGFACE_ACCELERATE, mock_model_config_properties),
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._set_serve_properties",
        return_value=mock_set_serving_properties,
    )
    @patch("sagemaker.serve.builder.djl_builder.prepare_for_djl_serving", side_effect=None)
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
    def test_tune_for_djl_local_container(
        self,
        mock_concurrent_benchmarks,
        mock_serial_benchmarks,
        mock_admissible_tensor_parallel_degrees,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_prepare_for_djl_serving,
        mock_set_serving_properties,
        mock_auto_detect_engine,
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
        assert tuned_model.generate_serving_properties() == mock_most_performant_serving_properties

    @patch("sagemaker.serve.builder.djl_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=False,
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._auto_detect_engine",
        return_value=(_DjlEngine.HUGGINGFACE_ACCELERATE, mock_model_config_properties),
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._set_serve_properties",
        return_value=mock_set_serving_properties,
    )
    @patch("sagemaker.serve.builder.djl_builder.prepare_for_djl_serving", side_effect=None)
    @patch("sagemaker.serve.builder.djl_builder._get_ram_usage_mb", return_value=1024)
    @patch("sagemaker.serve.builder.djl_builder._get_nb_instance", return_value="ml.g5.24xlarge")
    @patch(
        "sagemaker.serve.builder.djl_builder._serial_benchmark",
        **{"return_value.raiseError.side_effect": LocalDeepPingException("mock_exception")}
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._get_admissible_tensor_parallel_degrees",
        return_value=[4],
    )
    def test_tune_for_djl_local_container_deep_ping_ex(
        self,
        mock_get_admissible_tensor_parallel_degrees,
        mock_serial_benchmarks,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_prepare_for_djl_serving,
        mock_set_serving_properties,
        mock_auto_detect_engine,
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
        assert (
            tuned_model.generate_serving_properties()
            == mock_expected_huggingfaceaccelerate_serving_properties
        )

    @patch("sagemaker.serve.builder.djl_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=False,
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._auto_detect_engine",
        return_value=(_DjlEngine.HUGGINGFACE_ACCELERATE, mock_model_config_properties),
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._set_serve_properties",
        return_value=mock_set_serving_properties,
    )
    @patch("sagemaker.serve.builder.djl_builder.prepare_for_djl_serving", side_effect=None)
    @patch("sagemaker.serve.builder.djl_builder._get_ram_usage_mb", return_value=1024)
    @patch("sagemaker.serve.builder.djl_builder._get_nb_instance", return_value="ml.g5.24xlarge")
    @patch(
        "sagemaker.serve.builder.djl_builder._serial_benchmark",
        **{"return_value.raiseError.side_effect": LocalModelLoadException("mock_exception")}
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._get_admissible_tensor_parallel_degrees",
        return_value=[4],
    )
    def test_tune_for_djl_local_container_load_ex(
        self,
        mock_get_admissible_tensor_parallel_degrees,
        mock_serial_benchmarks,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_prepare_for_djl_serving,
        mock_set_serving_properties,
        mock_auto_detect_engine,
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
        assert (
            tuned_model.generate_serving_properties()
            == mock_expected_huggingfaceaccelerate_serving_properties
        )

    @patch("sagemaker.serve.builder.djl_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=False,
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._auto_detect_engine",
        return_value=(_DjlEngine.HUGGINGFACE_ACCELERATE, mock_model_config_properties),
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._set_serve_properties",
        return_value=mock_set_serving_properties,
    )
    @patch("sagemaker.serve.builder.djl_builder.prepare_for_djl_serving", side_effect=None)
    @patch("sagemaker.serve.builder.djl_builder._get_ram_usage_mb", return_value=1024)
    @patch("sagemaker.serve.builder.djl_builder._get_nb_instance", return_value="ml.g5.24xlarge")
    @patch(
        "sagemaker.serve.builder.djl_builder._serial_benchmark",
        **{"return_value.raiseError.side_effect": LocalModelOutOfMemoryException("mock_exception")}
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._get_admissible_tensor_parallel_degrees",
        return_value=[4],
    )
    def test_tune_for_djl_local_container_oom_ex(
        self,
        mock_get_admissible_tensor_parallel_degrees,
        mock_serial_benchmarks,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_prepare_for_djl_serving,
        mock_set_serving_properties,
        mock_auto_detect_engine,
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
        assert (
            tuned_model.generate_serving_properties()
            == mock_expected_huggingfaceaccelerate_serving_properties
        )

    @patch("sagemaker.serve.builder.djl_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=False,
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._auto_detect_engine",
        return_value=(_DjlEngine.HUGGINGFACE_ACCELERATE, mock_model_config_properties),
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._set_serve_properties",
        return_value=mock_set_serving_properties,
    )
    @patch("sagemaker.serve.builder.djl_builder.prepare_for_djl_serving", side_effect=None)
    @patch("sagemaker.serve.builder.djl_builder._get_ram_usage_mb", return_value=1024)
    @patch("sagemaker.serve.builder.djl_builder._get_nb_instance", return_value="ml.g5.24xlarge")
    @patch(
        "sagemaker.serve.builder.djl_builder._serial_benchmark",
        **{"return_value.raiseError.side_effect": LocalModelInvocationException("mock_exception")}
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._get_admissible_tensor_parallel_degrees",
        return_value=[4],
    )
    def test_tune_for_djl_local_container_invoke_ex(
        self,
        mock_get_admissible_tensor_parallel_degrees,
        mock_serial_benchmarks,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_prepare_for_djl_serving,
        mock_set_serving_properties,
        mock_auto_detect_engine,
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
        assert (
            tuned_model.generate_serving_properties()
            == mock_expected_huggingfaceaccelerate_serving_properties
        )

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
