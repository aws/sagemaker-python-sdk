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
from sagemaker.serve.utils.exceptions import (
    LocalDeepPingException,
    LocalModelLoadException,
    LocalModelOutOfMemoryException,
    LocalModelInvocationException,
)

mock_model_id = "huggingface-llm-amazon-falconlite"
mock_t5_model_id = "google/flan-t5-xxl"
mock_prompt = "Hello, I'm a language model,"
mock_response = "Hello, I'm a language model, and I'm here to help you with your English."
mock_sample_input = {"inputs": mock_prompt, "parameters": {}}
mock_sample_output = [{"generated_text": mock_response}]

mock_set_serving_properties = (4, "fp16", 1, 256, 256)

mock_tgi_most_performant_model_serving_properties = {
    "SAGEMAKER_PROGRAM": "inference.py",
    "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
    "SM_NUM_GPUS": "2",
}
mock_tgi_model_serving_properties = {
    "SAGEMAKER_PROGRAM": "inference.py",
    "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
    "SM_NUM_GPUS": "2",
}

mock_djl_most_performant_model_serving_properties = {
    "SAGEMAKER_PROGRAM": "inference.py",
    "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
    "OPTION_TENSOR_PARALLEL_DEGREE": "4",
}
mock_djl_model_serving_properties = {
    "SAGEMAKER_PROGRAM": "inference.py",
    "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
    "OPTION_TENSOR_PARALLEL_DEGREE": "4",
}

mock_schema_builder = MagicMock()
mock_schema_builder.sample_input = mock_sample_input
mock_schema_builder.sample_output = mock_sample_output

mock_tgi_image_uri = (
    "123456789712.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi"
    "-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
)
mock_djl_image_uri = (
    "123456789712.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.24.0-neuronx-sdk2.14.1"
)


class TestJumpStartBuilder(unittest.TestCase):
    @patch("sagemaker.serve.builder.jumpstart_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=True,
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._create_pre_trained_js_model",
        return_value=MagicMock(),
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.prepare_tgi_js_resources",
        return_value=({"model_type": "t5", "n_head": 71}, True),
    )
    @patch("sagemaker.serve.builder.jumpstart_builder._get_ram_usage_mb", return_value=1024)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder._get_nb_instance", return_value="ml.g5.24xlarge"
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder._get_admissible_tensor_parallel_degrees",
        return_value=[4, 2, 1],
    )
    @patch(
        "sagemaker.serve.utils.tuning._serial_benchmark",
        side_effect=[(5, 5, 25), (5.4, 5.4, 20), (5.2, 5.2, 15)],
    )
    @patch(
        "sagemaker.serve.utils.tuning._concurrent_benchmark",
        side_effect=[(0.9, 1), (0.10, 4), (0.13, 2)],
    )
    def test_tune_for_tgi_js_local_container(
        self,
        mock_concurrent_benchmarks,
        mock_serial_benchmarks,
        mock_admissible_tensor_parallel_degrees,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_prepare_for_tgi,
        mock_pre_trained_model,
        mock_is_jumpstart_model,
        mock_telemetry,
    ):
        builder = ModelBuilder(
            model="facebook/galactica-mock-model-id",
            schema_builder=mock_schema_builder,
            mode=Mode.LOCAL_CONTAINER,
        )

        mock_pre_trained_model.return_value.image_uri = mock_tgi_image_uri

        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True

        mock_pre_trained_model.return_value.env = mock_tgi_model_serving_properties

        tuned_model = model.tune()
        assert tuned_model.env == mock_tgi_most_performant_model_serving_properties

    @patch("sagemaker.serve.builder.jumpstart_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=True,
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._create_pre_trained_js_model",
        return_value=MagicMock(),
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.prepare_tgi_js_resources",
        return_value=({"model_type": "sharding_not_supported", "n_head": 71}, True),
    )
    @patch("sagemaker.serve.builder.jumpstart_builder._get_ram_usage_mb", return_value=1024)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder._get_nb_instance", return_value="ml.g5.24xlarge"
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder._get_admissible_tensor_parallel_degrees",
        return_value=[4, 2, 1],
    )
    @patch(
        "sagemaker.serve.utils.tuning._serial_benchmark",
        side_effect=[(5, 5, 25), (5.4, 5.4, 20), (5.2, 5.2, 15)],
    )
    @patch(
        "sagemaker.serve.utils.tuning._concurrent_benchmark",
        side_effect=[(0.9, 1), (0.10, 4), (0.13, 2)],
    )
    def test_tune_for_tgi_js_local_container_sharding_not_supported(
        self,
        mock_concurrent_benchmarks,
        mock_serial_benchmarks,
        mock_admissible_tensor_parallel_degrees,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_prepare_for_tgi,
        mock_pre_trained_model,
        mock_is_jumpstart_model,
        mock_telemetry,
    ):
        builder = ModelBuilder(
            model=mock_model_id, schema_builder=mock_schema_builder, mode=Mode.LOCAL_CONTAINER
        )

        mock_pre_trained_model.return_value.image_uri = mock_tgi_image_uri

        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True

        mock_pre_trained_model.return_value.env = mock_tgi_model_serving_properties

        tuned_model = model.tune()
        assert tuned_model.env == mock_tgi_most_performant_model_serving_properties

    @patch("sagemaker.serve.builder.jumpstart_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=True,
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._create_pre_trained_js_model",
        return_value=MagicMock(),
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.prepare_tgi_js_resources",
        return_value=({"model_type": "t5", "n_head": 71}, True),
    )
    @patch("sagemaker.serve.builder.jumpstart_builder._get_ram_usage_mb", return_value=1024)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder._get_nb_instance", return_value="ml.g5.24xlarge"
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder._get_admissible_tensor_parallel_degrees",
        return_value=[4, 2, 1],
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._serial_benchmark",
        **{"return_value.raiseError.side_effect": LocalDeepPingException("mock_exception")}
    )
    def test_tune_for_tgi_js_local_container_deep_ping_ex(
        self,
        mock_serial_benchmarks,
        mock_admissible_tensor_parallel_degrees,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_prepare_for_tgi,
        mock_pre_trained_model,
        mock_is_jumpstart_model,
        mock_telemetry,
    ):
        builder = ModelBuilder(
            model=mock_model_id, schema_builder=mock_schema_builder, mode=Mode.LOCAL_CONTAINER
        )

        mock_pre_trained_model.return_value.image_uri = mock_tgi_image_uri

        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True

        mock_pre_trained_model.return_value.env = mock_tgi_model_serving_properties

        tuned_model = model.tune()
        assert tuned_model.env == mock_tgi_model_serving_properties

    @patch("sagemaker.serve.builder.jumpstart_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=True,
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._create_pre_trained_js_model",
        return_value=MagicMock(),
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.prepare_tgi_js_resources",
        return_value=({"model_type": "RefinedWebModel", "n_head": 71}, True),
    )
    @patch("sagemaker.serve.builder.jumpstart_builder._get_ram_usage_mb", return_value=1024)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder._get_nb_instance", return_value="ml.g5.24xlarge"
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder._get_admissible_tensor_parallel_degrees",
        return_value=[4, 2, 1],
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._serial_benchmark",
        **{"return_value.raiseError.side_effect": LocalModelLoadException("mock_exception")}
    )
    def test_tune_for_tgi_js_local_container_load_ex(
        self,
        mock_serial_benchmarks,
        mock_admissible_tensor_parallel_degrees,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_prepare_for_tgi,
        mock_pre_trained_model,
        mock_is_jumpstart_model,
        mock_telemetry,
    ):
        builder = ModelBuilder(
            model=mock_model_id, schema_builder=mock_schema_builder, mode=Mode.LOCAL_CONTAINER
        )

        mock_pre_trained_model.return_value.image_uri = mock_tgi_image_uri

        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True

        mock_pre_trained_model.return_value.env = mock_tgi_model_serving_properties

        tuned_model = model.tune()
        assert tuned_model.env == mock_tgi_model_serving_properties

    @patch("sagemaker.serve.builder.jumpstart_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=True,
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._create_pre_trained_js_model",
        return_value=MagicMock(),
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.prepare_tgi_js_resources",
        return_value=({"n_head": 71}, True),
    )
    @patch("sagemaker.serve.builder.jumpstart_builder._get_ram_usage_mb", return_value=1024)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder._get_nb_instance", return_value="ml.g5.24xlarge"
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder._get_admissible_tensor_parallel_degrees",
        return_value=[4, 2, 1],
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._serial_benchmark",
        **{"return_value.raiseError.side_effect": LocalModelOutOfMemoryException("mock_exception")}
    )
    def test_tune_for_tgi_js_local_container_oom_ex(
        self,
        mock_serial_benchmarks,
        mock_admissible_tensor_parallel_degrees,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_prepare_for_tgi,
        mock_pre_trained_model,
        mock_is_jumpstart_model,
        mock_telemetry,
    ):
        builder = ModelBuilder(
            model=mock_model_id, schema_builder=mock_schema_builder, mode=Mode.LOCAL_CONTAINER
        )

        mock_pre_trained_model.return_value.image_uri = mock_tgi_image_uri

        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True

        mock_pre_trained_model.return_value.env = mock_tgi_model_serving_properties

        tuned_model = model.tune()
        assert tuned_model.env == mock_tgi_model_serving_properties

    @patch("sagemaker.serve.builder.jumpstart_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=True,
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._create_pre_trained_js_model",
        return_value=MagicMock(),
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.prepare_tgi_js_resources",
        return_value=({"model_type": "t5", "n_head": 71}, True),
    )
    @patch("sagemaker.serve.builder.jumpstart_builder._get_ram_usage_mb", return_value=1024)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder._get_nb_instance", return_value="ml.g5.24xlarge"
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder._get_admissible_tensor_parallel_degrees",
        return_value=[4, 2, 1],
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._serial_benchmark",
        **{"return_value.raiseError.side_effect": LocalModelInvocationException("mock_exception")}
    )
    def test_tune_for_tgi_js_local_container_invoke_ex(
        self,
        mock_serial_benchmarks,
        mock_admissible_tensor_parallel_degrees,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_prepare_for_tgi,
        mock_pre_trained_model,
        mock_is_jumpstart_model,
        mock_telemetry,
    ):
        builder = ModelBuilder(
            model=mock_model_id, schema_builder=mock_schema_builder, mode=Mode.LOCAL_CONTAINER
        )

        mock_pre_trained_model.return_value.image_uri = mock_tgi_image_uri

        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True

        mock_pre_trained_model.return_value.env = mock_tgi_model_serving_properties

        tuned_model = model.tune()
        assert tuned_model.env == mock_tgi_model_serving_properties

    @patch("sagemaker.serve.builder.jumpstart_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=True,
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._create_pre_trained_js_model",
        return_value=MagicMock(),
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.prepare_djl_js_resources",
        return_value=(
            mock_set_serving_properties,
            {"model_type": "t5", "n_head": 71},
            True,
        ),
    )
    @patch("sagemaker.serve.builder.jumpstart_builder._get_ram_usage_mb", return_value=1024)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder._get_nb_instance", return_value="ml.g5.24xlarge"
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder._get_admissible_tensor_parallel_degrees",
        return_value=[4, 2, 1],
    )
    @patch(
        "sagemaker.serve.utils.tuning._serial_benchmark",
        side_effect=[(5, 5, 25), (5.4, 5.4, 20), (5.2, 5.2, 15)],
    )
    @patch(
        "sagemaker.serve.utils.tuning._concurrent_benchmark",
        side_effect=[(0.9, 1), (0.10, 4), (0.13, 2)],
    )
    def test_tune_for_djl_js_local_container(
        self,
        mock_concurrent_benchmarks,
        mock_serial_benchmarks,
        mock_admissible_tensor_parallel_degrees,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_prepare_for_tgi,
        mock_pre_trained_model,
        mock_is_jumpstart_model,
        mock_telemetry,
    ):
        builder = ModelBuilder(
            model="facebook/galactica-mock",
            schema_builder=mock_schema_builder,
            mode=Mode.LOCAL_CONTAINER,
        )

        mock_pre_trained_model.return_value.image_uri = mock_djl_image_uri

        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True

        mock_pre_trained_model.return_value.env = mock_djl_model_serving_properties

        tuned_model = model.tune()
        assert tuned_model.env == mock_djl_most_performant_model_serving_properties

    @patch("sagemaker.serve.builder.jumpstart_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=True,
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._create_pre_trained_js_model",
        return_value=MagicMock(),
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.prepare_djl_js_resources",
        return_value=(
            mock_set_serving_properties,
            {"model_type": "RefinedWebModel", "n_head": 71},
            True,
        ),
    )
    @patch("sagemaker.serve.builder.jumpstart_builder._get_ram_usage_mb", return_value=1024)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder._get_nb_instance", return_value="ml.g5.24xlarge"
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder._get_admissible_tensor_parallel_degrees",
        return_value=[1],
    )
    @patch(
        "sagemaker.serve.builder.djl_builder._serial_benchmark",
        **{"return_value.raiseError.side_effect": LocalModelInvocationException("mock_exception")}
    )
    def test_tune_for_djl_js_local_container_invoke_ex(
        self,
        mock_serial_benchmarks,
        mock_admissible_tensor_parallel_degrees,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_prepare_for_tgi,
        mock_pre_trained_model,
        mock_is_jumpstart_model,
        mock_telemetry,
    ):
        builder = ModelBuilder(
            model=mock_model_id, schema_builder=mock_schema_builder, mode=Mode.LOCAL_CONTAINER
        )

        mock_pre_trained_model.return_value.image_uri = mock_djl_image_uri

        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True

        mock_pre_trained_model.return_value.env = mock_djl_model_serving_properties

        tuned_model = model.tune()
        assert tuned_model.env == mock_djl_model_serving_properties

    @patch("sagemaker.serve.builder.jumpstart_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=True,
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._create_pre_trained_js_model",
        return_value=MagicMock(),
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.prepare_tgi_js_resources",
        return_value=({"model_type": "t5", "n_head": 71}, True),
    )
    def test_tune_for_djl_js_endpoint_mode_ex(
        self,
        mock_prepare_for_tgi,
        mock_pre_trained_model,
        mock_is_jumpstart_model,
        mock_telemetry,
    ):
        builder = ModelBuilder(
            model=mock_model_id, schema_builder=mock_schema_builder, mode=Mode.SAGEMAKER_ENDPOINT
        )

        mock_pre_trained_model.return_value.image_uri = mock_djl_image_uri

        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True

        tuned_model = model.tune()
        assert tuned_model == model
