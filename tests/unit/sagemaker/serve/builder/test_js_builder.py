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
from unittest.mock import MagicMock, patch, Mock

import unittest

from sagemaker.enums import Tag
from sagemaker.serve import SchemaBuilder
from sagemaker.serve.builder.model_builder import ModelBuilder
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.utils.exceptions import (
    LocalDeepPingException,
    LocalModelLoadException,
    LocalModelOutOfMemoryException,
    LocalModelInvocationException,
)
from tests.unit.sagemaker.serve.constants import (
    DEPLOYMENT_CONFIGS,
    OPTIMIZED_DEPLOYMENT_CONFIG_WITH_GATED_DRAFT_MODEL,
    CAMEL_CASE_ADDTL_DRAFT_MODEL_DATA_SOURCES,
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
mock_invalid_image_uri = (
    "123456789712.dkr.ecr.us-west-2.amazonaws.com/invalid"
    "-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
)
mock_djl_image_uri = (
    "123456789712.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.24.0-neuronx-sdk2.14.1"
)

mock_model_data = {
    "S3DataSource": {
        "S3Uri": "s3://jumpstart-private-cache-prod-us-west-2/huggingface-llm/huggingface-llm-zephyr-7b-gemma"
        "/artifacts/inference-prepack/v1.0.0/",
        "S3DataType": "S3Prefix",
        "CompressionType": "None",
    }
}
mock_model_data_str = (
    "s3://jumpstart-private-cache-prod-us-west-2/huggingface-llm/huggingface-llm-zephyr-7b-gemma"
    "/artifacts/inference-prepack/v1.0.0/"
)

mock_optimization_job_response = {
    "OptimizationJobArn": "arn:aws:sagemaker:us-west-2:312206380606:optimization-job"
    "/modelbuilderjob-c9b28846f963497ca540010b2aa2ec8d",
    "OptimizationJobStatus": "COMPLETED",
    "OptimizationStartTime": "",
    "OptimizationEndTime": "",
    "CreationTime": "",
    "LastModifiedTime": "",
    "OptimizationJobName": "modelbuilderjob-c9b28846f963497ca540010b2aa2ec8d",
    "ModelSource": {
        "S3": {
            "S3Uri": "s3://jumpstart-private-cache-alpha-us-west-2/meta-textgeneration/"
            "meta-textgeneration-llama-3-8b-instruct/artifacts/inference-prepack/v1.1.0/"
        }
    },
    "OptimizationEnvironment": {
        "ENDPOINT_SERVER_TIMEOUT": "3600",
        "HF_MODEL_ID": "/opt/ml/model",
        "MODEL_CACHE_ROOT": "/opt/ml/model",
        "OPTION_DTYPE": "fp16",
        "OPTION_MAX_ROLLING_BATCH_SIZE": "4",
        "OPTION_NEURON_OPTIMIZE_LEVEL": "2",
        "OPTION_N_POSITIONS": "2048",
        "OPTION_ROLLING_BATCH": "auto",
        "OPTION_TENSOR_PARALLEL_DEGREE": "2",
        "SAGEMAKER_ENV": "1",
        "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
        "SAGEMAKER_PROGRAM": "inference.py",
    },
    "DeploymentInstanceType": "ml.inf2.48xlarge",
    "OptimizationConfigs": [
        {
            "ModelCompilationConfig": {
                "Image": "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-neuronx-sdk2.18.2",
                "OverrideEnvironment": {
                    "OPTION_DTYPE": "fp16",
                    "OPTION_MAX_ROLLING_BATCH_SIZE": "4",
                    "OPTION_NEURON_OPTIMIZE_LEVEL": "2",
                    "OPTION_N_POSITIONS": "2048",
                    "OPTION_ROLLING_BATCH": "auto",
                    "OPTION_TENSOR_PARALLEL_DEGREE": "2",
                },
            }
        }
    ],
    "OutputConfig": {
        "S3OutputLocation": "s3://dont-delete-ss-jarvis-integ-test-312206380606-us-west-2/"
        "code/a75a061aba764f2aa014042bcdc1464b/"
    },
    "OptimizationOutput": {
        "RecommendedInferenceImage": "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
        "djl-inference:0.28.0-neuronx-sdk2.18.2"
    },
    "RoleArn": "arn:aws:iam::312206380606:role/service-role/AmazonSageMaker-ExecutionRole-20230707T131628",
    "StoppingCondition": {"MaxRuntimeInSeconds": 36000},
    "ResponseMetadata": {
        "RequestId": "704c7bcd-41e2-4d73-8039-262ff6a3f38b",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "704c7bcd-41e2-4d73-8039-262ff6a3f38b",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "1787",
            "date": "Thu, 04 Jul 2024 16:55:50 GMT",
        },
        "RetryAttempts": 0,
    },
}


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
    def test__build_for_jumpstart_value_error(
        self,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_prepare_for_tgi,
        mock_pre_trained_model,
        mock_is_jumpstart_model,
        mock_telemetry,
    ):
        builder = ModelBuilder(
            model="facebook/invalid",
            schema_builder=mock_schema_builder,
            mode=Mode.LOCAL_CONTAINER,
        )

        mock_pre_trained_model.return_value.image_uri = mock_invalid_image_uri

        self.assertRaises(
            ValueError,
            lambda: builder.build(),
        )

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
        "sagemaker.serve.builder.jumpstart_builder.prepare_mms_js_resources",
        return_value=({"model_type": "t5", "n_head": 71}, True),
    )
    @patch("sagemaker.serve.builder.jumpstart_builder._get_ram_usage_mb", return_value=1024)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder._get_nb_instance", return_value="ml.g5.24xlarge"
    )
    def test__build_for_mms_jumpstart(
        self,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_prepare_for_mms,
        mock_pre_trained_model,
        mock_is_jumpstart_model,
        mock_telemetry,
    ):
        builder = ModelBuilder(
            model="facebook/galactica-mock-model-id",
            schema_builder=mock_schema_builder,
            mode=Mode.LOCAL_CONTAINER,
        )

        mock_pre_trained_model.return_value.image_uri = (
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface"
            "-pytorch-inference:2.1.0-transformers4.37.0-gpu-py310-cu118"
            "-ubuntu20.04"
        )

        builder.build()
        builder.serve_settings.telemetry_opt_out = True

        mock_prepare_for_mms.assert_called()

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
        **{"return_value.raiseError.side_effect": LocalDeepPingException("mock_exception")},
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
        **{"return_value.raiseError.side_effect": LocalModelLoadException("mock_exception")},
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
        **{"return_value.raiseError.side_effect": LocalModelOutOfMemoryException("mock_exception")},
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
        **{"return_value.raiseError.side_effect": LocalModelInvocationException("mock_exception")},
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
        **{"return_value.raiseError.side_effect": LocalModelInvocationException("mock_exception")},
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
    def test_js_gated_model_in_endpoint_mode(
        self,
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
            mode=Mode.SAGEMAKER_ENDPOINT,
        )

        mock_pre_trained_model.return_value.image_uri = mock_tgi_image_uri
        mock_pre_trained_model.return_value.model_data = mock_model_data

        model = builder.build()

        assert model is not None

    @patch("sagemaker.serve.builder.jumpstart_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=True,
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._create_pre_trained_js_model",
        return_value=MagicMock(),
    )
    def test_js_gated_model_in_local_mode(
        self,
        mock_pre_trained_model,
        mock_is_jumpstart_model,
        mock_telemetry,
    ):
        builder = ModelBuilder(
            model="huggingface-llm-zephyr-7b-gemma",
            schema_builder=mock_schema_builder,
            mode=Mode.LOCAL_CONTAINER,
        )

        mock_pre_trained_model.return_value.image_uri = mock_tgi_image_uri
        mock_pre_trained_model.return_value.model_data = mock_model_data_str

        self.assertRaisesRegex(
            ValueError,
            "JumpStart Gated Models are only supported in SAGEMAKER_ENDPOINT mode.",
            lambda: builder.build(),
        )

    @patch("sagemaker.serve.builder.jumpstart_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=True,
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._create_pre_trained_js_model",
        return_value=MagicMock(),
    )
    def test_js_gated_model_ex(
        self,
        mock_pre_trained_model,
        mock_is_jumpstart_model,
        mock_telemetry,
    ):
        builder = ModelBuilder(
            model="huggingface-llm-zephyr-7b-gemma",
            schema_builder=mock_schema_builder,
            mode=Mode.LOCAL_CONTAINER,
        )

        mock_pre_trained_model.return_value.image_uri = mock_tgi_image_uri
        mock_pre_trained_model.return_value.model_data = None

        self.assertRaises(
            ValueError,
            lambda: builder.build(),
        )

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
    def test_list_deployment_configs(
        self,
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
        )

        mock_pre_trained_model.return_value.image_uri = mock_tgi_image_uri
        mock_pre_trained_model.return_value.list_deployment_configs.side_effect = (
            lambda: DEPLOYMENT_CONFIGS
        )

        configs = builder.list_deployment_configs()

        self.assertEqual(configs, DEPLOYMENT_CONFIGS)

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
    def test_get_deployment_config(
        self,
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
        )

        mock_pre_trained_model.return_value.image_uri = mock_tgi_image_uri

        expected = DEPLOYMENT_CONFIGS[0]
        mock_pre_trained_model.return_value.deployment_config = expected

        self.assertEqual(builder.get_deployment_config(), expected)

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
    def test_set_deployment_config(
        self,
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
        )

        mock_pre_trained_model.return_value.image_uri = mock_tgi_image_uri

        builder.build()
        builder.set_deployment_config("config-1", "ml.g5.24xlarge")

        mock_pre_trained_model.return_value.set_deployment_config.assert_called_with(
            "config-1", "ml.g5.24xlarge"
        )

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
    def test_set_deployment_config_ex(
        self,
        mock_get_nb_instance,
        mock_get_ram_usage_mb,
        mock_prepare_for_tgi,
        mock_pre_trained_model,
        mock_is_jumpstart_model,
        mock_telemetry,
    ):
        mock_pre_trained_model.return_value.image_uri = mock_tgi_image_uri

        self.assertRaisesRegex(
            Exception,
            "Cannot set deployment config to an uninitialized model.",
            lambda: ModelBuilder(
                model="facebook/galactica-mock-model-id", schema_builder=mock_schema_builder
            ).set_deployment_config("config-2", "ml.g5.24xlarge"),
        )

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
    def test_display_benchmark_metrics(
        self,
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
        )

        mock_pre_trained_model.return_value.image_uri = mock_tgi_image_uri
        mock_pre_trained_model.return_value.list_deployment_configs.side_effect = (
            lambda: DEPLOYMENT_CONFIGS
        )

        builder.list_deployment_configs()

        builder.display_benchmark_metrics()

        mock_pre_trained_model.return_value.display_benchmark_metrics.assert_called_once()

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
    def test_display_benchmark_metrics_initial(
        self,
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
        )

        mock_pre_trained_model.return_value.image_uri = mock_tgi_image_uri
        mock_pre_trained_model.return_value.list_deployment_configs.side_effect = (
            lambda: DEPLOYMENT_CONFIGS
        )

        builder.display_benchmark_metrics()

        mock_pre_trained_model.return_value.display_benchmark_metrics.assert_called_once()

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
    def test_fine_tuned_model_with_fine_tuning_model_path(
        self,
        mock_prepare_for_tgi,
        mock_pre_trained_model,
        mock_is_jumpstart_model,
        mock_telemetry,
    ):
        mock_pre_trained_model.return_value.image_uri = mock_djl_image_uri
        mock_fine_tuning_model_path = "s3://test"

        sample_input = {
            "inputs": "The diamondback terrapin or simply terrapin is a species of turtle native to the brackish "
            "coastal tidal marshes of the",
            "parameters": {"max_new_tokens": 1024},
        }
        sample_output = [
            {
                "generated_text": "The diamondback terrapin or simply terrapin is a species of turtle native to the "
                "brackish coastal tidal marshes of the east coast."
            }
        ]
        builder = ModelBuilder(
            model="meta-textgeneration-llama-3-70b",
            schema_builder=SchemaBuilder(sample_input, sample_output),
            model_metadata={
                "FINE_TUNING_MODEL_PATH": mock_fine_tuning_model_path,
            },
        )
        model = builder.build()

        model.model_data["S3DataSource"].__setitem__.assert_called_with(
            "S3Uri", mock_fine_tuning_model_path
        )
        mock_pre_trained_model.return_value.add_tags.assert_called_with(
            {"Key": Tag.FINE_TUNING_MODEL_PATH, "Value": mock_fine_tuning_model_path}
        )

    @patch("sagemaker.serve.builder.jumpstart_builder._capture_telemetry", side_effect=None)
    @patch.object(ModelBuilder, "_get_serve_setting", autospec=True)
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
    def test_fine_tuned_model_with_fine_tuning_job_name(
        self,
        mock_prepare_for_tgi,
        mock_pre_trained_model,
        mock_is_jumpstart_model,
        mock_serve_settings,
        mock_telemetry,
    ):
        mock_fine_tuning_model_path = "s3://test"
        mock_sagemaker_session = Mock()
        mock_sagemaker_session.sagemaker_client.describe_training_job.return_value = {
            "ModelArtifacts": {
                "S3ModelArtifacts": mock_fine_tuning_model_path,
            }
        }
        mock_pre_trained_model.return_value.image_uri = mock_djl_image_uri
        mock_fine_tuning_job_name = "mock-job"

        sample_input = {
            "inputs": "The diamondback terrapin or simply terrapin is a species of turtle native to the brackish "
            "coastal tidal marshes of the",
            "parameters": {"max_new_tokens": 1024},
        }
        sample_output = [
            {
                "generated_text": "The diamondback terrapin or simply terrapin is a species of turtle native to the "
                "brackish coastal tidal marshes of the east coast."
            }
        ]
        builder = ModelBuilder(
            model="meta-textgeneration-llama-3-70b",
            schema_builder=SchemaBuilder(sample_input, sample_output),
            model_metadata={"FINE_TUNING_JOB_NAME": mock_fine_tuning_job_name},
            sagemaker_session=mock_sagemaker_session,
        )
        model = builder.build(sagemaker_session=mock_sagemaker_session)

        mock_sagemaker_session.sagemaker_client.describe_training_job.assert_called_once_with(
            TrainingJobName=mock_fine_tuning_job_name
        )

        model.model_data["S3DataSource"].__setitem__.assert_any_call(
            "S3Uri", mock_fine_tuning_model_path
        )
        mock_pre_trained_model.return_value.add_tags.assert_called_with(
            [
                {"key": Tag.FINE_TUNING_JOB_NAME, "value": mock_fine_tuning_job_name},
                {"key": Tag.FINE_TUNING_MODEL_PATH, "value": mock_fine_tuning_model_path},
            ]
        )

    @patch("sagemaker.serve.builder.jumpstart_builder._capture_telemetry", side_effect=None)
    @patch.object(ModelBuilder, "_get_serve_setting", autospec=True)
    def test_optimize_quantize_for_jumpstart(
        self,
        mock_serve_settings,
        mock_telemetry,
    ):
        mock_sagemaker_session = Mock()

        mock_pysdk_model = Mock()
        mock_pysdk_model.env = {"SAGEMAKER_ENV": "1"}
        mock_pysdk_model.model_data = mock_model_data
        mock_pysdk_model.image_uri = mock_tgi_image_uri
        mock_pysdk_model.list_deployment_configs.return_value = DEPLOYMENT_CONFIGS
        mock_pysdk_model.deployment_config = DEPLOYMENT_CONFIGS[0]

        sample_input = {
            "inputs": "The diamondback terrapin or simply terrapin is a species "
            "of turtle native to the brackish coastal tidal marshes of the",
            "parameters": {"max_new_tokens": 1024},
        }
        sample_output = [
            {
                "generated_text": "The diamondback terrapin or simply terrapin is a "
                "species of turtle native to the brackish coastal "
                "tidal marshes of the east coast."
            }
        ]

        model_builder = ModelBuilder(
            model="meta-textgeneration-llama-3-70b",
            schema_builder=SchemaBuilder(sample_input, sample_output),
            sagemaker_session=mock_sagemaker_session,
        )

        model_builder.pysdk_model = mock_pysdk_model

        out_put = model_builder._optimize_for_jumpstart(
            accept_eula=True,
            quantization_config={
                "OverrideEnvironment": {"OPTION_QUANTIZE": "awq"},
            },
            env_vars={
                "OPTION_TENSOR_PARALLEL_DEGREE": "1",
                "OPTION_MAX_ROLLING_BATCH_SIZE": "2",
            },
            output_path="s3://bucket/code/",
        )

        self.assertIsNotNone(out_put)

    @patch("sagemaker.serve.builder.jumpstart_builder._capture_telemetry", side_effect=None)
    @patch.object(ModelBuilder, "_get_serve_setting", autospec=True)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=True,
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._create_pre_trained_js_model",
        return_value=MagicMock(),
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder._jumpstart_speculative_decoding",
        return_value=True,
    )
    def test_jumpstart_model_provider_calls_jumpstart_speculative_decoding(
        self,
        mock_js_speculative_decoding,
        mock_pretrained_js_model,
        mock_is_js_model,
        mock_serve_settings,
        mock_capture_telemetry,
    ):
        mock_sagemaker_session = Mock()
        mock_pysdk_model = Mock()
        mock_pysdk_model.env = {"SAGEMAKER_ENV": "1"}
        mock_pysdk_model.model_data = mock_model_data
        mock_pysdk_model.image_uri = mock_tgi_image_uri
        mock_pysdk_model.list_deployment_configs.return_value = DEPLOYMENT_CONFIGS
        mock_pysdk_model.deployment_config = OPTIMIZED_DEPLOYMENT_CONFIG_WITH_GATED_DRAFT_MODEL
        mock_pysdk_model.additional_model_data_sources = CAMEL_CASE_ADDTL_DRAFT_MODEL_DATA_SOURCES

        sample_input = {
            "inputs": "The diamondback terrapin or simply terrapin is a species "
            "of turtle native to the brackish coastal tidal marshes of the",
            "parameters": {"max_new_tokens": 1024},
        }
        sample_output = [
            {
                "generated_text": "The diamondback terrapin or simply terrapin is a "
                "species of turtle native to the brackish coastal "
                "tidal marshes of the east coast."
            }
        ]

        model_builder = ModelBuilder(
            model="meta-textgeneration-llama-3-70b",
            schema_builder=SchemaBuilder(sample_input, sample_output),
            sagemaker_session=mock_sagemaker_session,
        )

        model_builder.pysdk_model = mock_pysdk_model

        model_builder._optimize_for_jumpstart(
            accept_eula=True,
            speculative_decoding_config={
                "ModelProvider": "JumpStart",
                "ModelID": "meta-textgeneration-llama-3-2-1b",
                "AcceptEula": False,
            },
        )

        mock_js_speculative_decoding.assert_called_once()

    @patch("sagemaker.serve.builder.jumpstart_builder._capture_telemetry", side_effect=None)
    @patch.object(ModelBuilder, "_get_serve_setting", autospec=True)
    def test_optimize_quantize_and_compile_for_jumpstart(
        self,
        mock_serve_settings,
        mock_telemetry,
    ):
        mock_sagemaker_session = Mock()
        mock_metadata_config = Mock()
        mock_metadata_config.resolved_config = {
            "supported_inference_instance_types": ["ml.inf2.48xlarge"],
            "hosting_neuron_model_id": "huggingface-llmneuron-mistral-7b",
        }

        mock_pysdk_model = Mock()
        mock_pysdk_model.env = {"SAGEMAKER_ENV": "1"}
        mock_pysdk_model.model_data = mock_model_data
        mock_pysdk_model.image_uri = mock_tgi_image_uri
        mock_pysdk_model.list_deployment_configs.return_value = DEPLOYMENT_CONFIGS
        mock_pysdk_model.deployment_config = DEPLOYMENT_CONFIGS[0]
        mock_pysdk_model.config_name = "config_name"
        mock_pysdk_model._metadata_configs = {"config_name": mock_metadata_config}

        sample_input = {
            "inputs": "The diamondback terrapin or simply terrapin is a species "
            "of turtle native to the brackish coastal tidal marshes of the",
            "parameters": {"max_new_tokens": 1024},
        }
        sample_output = [
            {
                "generated_text": "The diamondback terrapin or simply terrapin is a "
                "species of turtle native to the brackish coastal "
                "tidal marshes of the east coast."
            }
        ]

        model_builder = ModelBuilder(
            model="meta-textgeneration-llama-3-70b",
            schema_builder=SchemaBuilder(sample_input, sample_output),
            sagemaker_session=mock_sagemaker_session,
        )

        model_builder.pysdk_model = mock_pysdk_model

        out_put = model_builder._optimize_for_jumpstart(
            accept_eula=True,
            quantization_config={
                "OverrideEnvironment": {"OPTION_QUANTIZE": "awq"},
            },
            compilation_config={"OverrideEnvironment": {"OPTION_TENSOR_PARALLEL_DEGREE": "2"}},
            output_path="s3://bucket/code/",
        )

        self.assertIsNotNone(out_put)

    @patch("sagemaker.serve.builder.jumpstart_builder._capture_telemetry", side_effect=None)
    @patch.object(ModelBuilder, "_get_serve_setting", autospec=True)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_gated_model",
        return_value=True,
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=True,
    )
    @patch("sagemaker.serve.builder.jumpstart_builder.JumpStart._create_pre_trained_js_model")
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.prepare_tgi_js_resources",
        return_value=({"model_type": "t5", "n_head": 71}, True),
    )
    def test_optimize_compile_for_jumpstart_without_neuron_env(
        self,
        mock_prepare_for_tgi,
        mock_pre_trained_model,
        mock_is_jumpstart_model,
        mock_is_gated_model,
        mock_serve_settings,
        mock_telemetry,
    ):
        mock_sagemaker_session = Mock()
        mock_sagemaker_session.wait_for_optimization_job.side_effect = (
            lambda *args: mock_optimization_job_response
        )

        mock_pre_trained_model.return_value = MagicMock()
        mock_pre_trained_model.return_value.env = dict()
        mock_pre_trained_model.return_value.model_data = mock_model_data
        mock_pre_trained_model.return_value.image_uri = mock_tgi_image_uri
        mock_pre_trained_model.return_value.list_deployment_configs.return_value = (
            DEPLOYMENT_CONFIGS
        )
        mock_pre_trained_model.return_value.deployment_config = DEPLOYMENT_CONFIGS[0]
        mock_pre_trained_model.return_value._metadata_configs = None

        sample_input = {
            "inputs": "The diamondback terrapin or simply terrapin is a species "
            "of turtle native to the brackish coastal tidal marshes of the",
            "parameters": {"max_new_tokens": 1024},
        }
        sample_output = [
            {
                "generated_text": "The diamondback terrapin or simply terrapin is a "
                "species of turtle native to the brackish coastal "
                "tidal marshes of the east coast."
            }
        ]

        model_builder = ModelBuilder(
            model="meta-textgeneration-llama-3-70b",
            schema_builder=SchemaBuilder(sample_input, sample_output),
            sagemaker_session=mock_sagemaker_session,
        )

        optimized_model = model_builder.optimize(
            accept_eula=True,
            instance_type="ml.inf2.48xlarge",
            compilation_config={
                "OverrideEnvironment": {
                    "OPTION_TENSOR_PARALLEL_DEGREE": "2",
                    "OPTION_N_POSITIONS": "2048",
                    "OPTION_DTYPE": "fp16",
                    "OPTION_ROLLING_BATCH": "auto",
                    "OPTION_MAX_ROLLING_BATCH_SIZE": "4",
                    "OPTION_NEURON_OPTIMIZE_LEVEL": "2",
                }
            },
            output_path="s3://bucket/code/",
        )

        self.assertEqual(
            optimized_model.image_uri,
            mock_optimization_job_response["OptimizationOutput"]["RecommendedInferenceImage"],
        )
        self.assertEqual(
            optimized_model.model_data["S3DataSource"]["S3Uri"],
            mock_optimization_job_response["OutputConfig"]["S3OutputLocation"],
        )

    @patch("sagemaker.serve.builder.jumpstart_builder._capture_telemetry", side_effect=None)
    @patch.object(ModelBuilder, "_get_serve_setting", autospec=True)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_gated_model",
        return_value=True,
    )
    @patch("sagemaker.serve.builder.jumpstart_builder.JumpStartModel")
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=True,
    )
    @patch("sagemaker.serve.builder.jumpstart_builder.JumpStart._create_pre_trained_js_model")
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.prepare_tgi_js_resources",
        return_value=({"model_type": "t5", "n_head": 71}, True),
    )
    def test_optimize_compile_for_jumpstart_with_neuron_env(
        self,
        mock_prepare_for_tgi,
        mock_pre_trained_model,
        mock_is_jumpstart_model,
        mock_js_model,
        mock_is_gated_model,
        mock_serve_settings,
        mock_telemetry,
    ):
        mock_sagemaker_session = Mock()
        mock_metadata_config = Mock()
        mock_sagemaker_session.wait_for_optimization_job.side_effect = (
            lambda *args: mock_optimization_job_response
        )

        mock_metadata_config.resolved_config = {
            "supported_inference_instance_types": ["ml.inf2.48xlarge"],
            "hosting_neuron_model_id": "neuron_model_id",
        }

        mock_js_model.return_value = MagicMock()
        mock_js_model.return_value.env = dict()

        mock_pre_trained_model.return_value = MagicMock()
        mock_pre_trained_model.return_value.env = dict()
        mock_pre_trained_model.return_value.config_name = "config_name"
        mock_pre_trained_model.return_value.model_data = mock_model_data
        mock_pre_trained_model.return_value.image_uri = mock_tgi_image_uri
        mock_pre_trained_model.return_value.list_deployment_configs.return_value = (
            DEPLOYMENT_CONFIGS
        )
        mock_pre_trained_model.return_value.deployment_config = DEPLOYMENT_CONFIGS[0]
        mock_pre_trained_model.return_value._metadata_configs = {
            "config_name": mock_metadata_config
        }

        sample_input = {
            "inputs": "The diamondback terrapin or simply terrapin is a species "
            "of turtle native to the brackish coastal tidal marshes of the",
            "parameters": {"max_new_tokens": 1024},
        }
        sample_output = [
            {
                "generated_text": "The diamondback terrapin or simply terrapin is a "
                "species of turtle native to the brackish coastal "
                "tidal marshes of the east coast."
            }
        ]

        model_builder = ModelBuilder(
            model="meta-textgeneration-llama-3-70b",
            schema_builder=SchemaBuilder(sample_input, sample_output),
            sagemaker_session=mock_sagemaker_session,
        )

        optimized_model = model_builder.optimize(
            accept_eula=True,
            instance_type="ml.inf2.48xlarge",
            compilation_config={
                "OverrideEnvironment": {
                    "OPTION_TENSOR_PARALLEL_DEGREE": "2",
                    "OPTION_N_POSITIONS": "2048",
                    "OPTION_DTYPE": "fp16",
                    "OPTION_ROLLING_BATCH": "auto",
                    "OPTION_MAX_ROLLING_BATCH_SIZE": "4",
                    "OPTION_NEURON_OPTIMIZE_LEVEL": "2",
                }
            },
            output_path="s3://bucket/code/",
        )

        self.assertEqual(
            optimized_model.image_uri,
            mock_optimization_job_response["OptimizationOutput"]["RecommendedInferenceImage"],
        )
        self.assertEqual(
            optimized_model.model_data["S3DataSource"]["S3Uri"],
            mock_optimization_job_response["OutputConfig"]["S3OutputLocation"],
        )
        self.assertEqual(optimized_model.env["OPTION_TENSOR_PARALLEL_DEGREE"], "2")
        self.assertEqual(optimized_model.env["OPTION_N_POSITIONS"], "2048")
        self.assertEqual(optimized_model.env["OPTION_DTYPE"], "fp16")
        self.assertEqual(optimized_model.env["OPTION_ROLLING_BATCH"], "auto")
        self.assertEqual(optimized_model.env["OPTION_MAX_ROLLING_BATCH_SIZE"], "4")
        self.assertEqual(optimized_model.env["OPTION_NEURON_OPTIMIZE_LEVEL"], "2")

    @patch("sagemaker.serve.builder.jumpstart_builder._capture_telemetry", side_effect=None)
    @patch.object(ModelBuilder, "_get_serve_setting", autospec=True)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_gated_model",
        return_value=True,
    )
    @patch("sagemaker.serve.builder.jumpstart_builder.JumpStartModel")
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=True,
    )
    @patch("sagemaker.serve.builder.jumpstart_builder.JumpStart._create_pre_trained_js_model")
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.prepare_tgi_js_resources",
        return_value=({"model_type": "t5", "n_head": 71}, True),
    )
    def test_optimize_compile_for_jumpstart_without_compilation_config(
        self,
        mock_prepare_for_tgi,
        mock_pre_trained_model,
        mock_is_jumpstart_model,
        mock_js_model,
        mock_is_gated_model,
        mock_serve_settings,
        mock_telemetry,
    ):
        mock_sagemaker_session = Mock()
        mock_metadata_config = Mock()
        mock_sagemaker_session.wait_for_optimization_job.side_effect = (
            lambda *args: mock_optimization_job_response
        )

        mock_metadata_config.resolved_config = {
            "supported_inference_instance_types": ["ml.inf2.48xlarge"],
            "hosting_neuron_model_id": "huggingface-llmneuron-mistral-7b",
        }

        mock_js_model.return_value = MagicMock()
        mock_js_model.return_value.env = {
            "SAGEMAKER_PROGRAM": "inference.py",
            "ENDPOINT_SERVER_TIMEOUT": "3600",
            "MODEL_CACHE_ROOT": "/opt/ml/model",
            "SAGEMAKER_ENV": "1",
            "HF_MODEL_ID": "/opt/ml/model",
            "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
        }

        mock_pre_trained_model.return_value = MagicMock()
        mock_pre_trained_model.return_value.env = dict()
        mock_pre_trained_model.return_value.config_name = "config_name"
        mock_pre_trained_model.return_value.model_data = mock_model_data
        mock_pre_trained_model.return_value.image_uri = mock_tgi_image_uri
        mock_pre_trained_model.return_value.list_deployment_configs.return_value = (
            DEPLOYMENT_CONFIGS
        )
        mock_pre_trained_model.return_value.deployment_config = DEPLOYMENT_CONFIGS[0]
        mock_pre_trained_model.return_value._metadata_configs = {
            "config_name": mock_metadata_config
        }

        sample_input = {
            "inputs": "The diamondback terrapin or simply terrapin is a species "
            "of turtle native to the brackish coastal tidal marshes of the",
            "parameters": {"max_new_tokens": 1024},
        }
        sample_output = [
            {
                "generated_text": "The diamondback terrapin or simply terrapin is a "
                "species of turtle native to the brackish coastal "
                "tidal marshes of the east coast."
            }
        ]

        model_builder = ModelBuilder(
            model="meta-textgeneration-llama-3-70b",
            schema_builder=SchemaBuilder(sample_input, sample_output),
            sagemaker_session=mock_sagemaker_session,
        )

        optimized_model = model_builder.optimize(
            accept_eula=True,
            instance_type="ml.inf2.24xlarge",
            output_path="s3://bucket/code/",
        )

        self.assertEqual(
            optimized_model.image_uri,
            mock_optimization_job_response["OptimizationOutput"]["RecommendedInferenceImage"],
        )
        self.assertEqual(
            optimized_model.model_data["S3DataSource"]["S3Uri"],
            mock_optimization_job_response["OutputConfig"]["S3OutputLocation"],
        )
        self.assertEqual(optimized_model.env["SAGEMAKER_PROGRAM"], "inference.py")
        self.assertEqual(optimized_model.env["ENDPOINT_SERVER_TIMEOUT"], "3600")
        self.assertEqual(optimized_model.env["MODEL_CACHE_ROOT"], "/opt/ml/model")
        self.assertEqual(optimized_model.env["SAGEMAKER_ENV"], "1")
        self.assertEqual(optimized_model.env["HF_MODEL_ID"], "/opt/ml/model")
        self.assertEqual(optimized_model.env["SAGEMAKER_MODEL_SERVER_WORKERS"], "1")


class TestJumpStartModelBuilderOptimizationUseCases(unittest.TestCase):

    @patch("sagemaker.serve.builder.jumpstart_builder._capture_telemetry", side_effect=None)
    @patch.object(ModelBuilder, "_get_serve_setting", autospec=True)
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_gated_model",
        return_value=True,
    )
    @patch("sagemaker.serve.builder.jumpstart_builder.JumpStartModel")
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_jumpstart_model_id",
        return_value=True,
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_fine_tuned_model",
        return_value=False,
    )
    def test_optimize_on_js_model_should_ignore_pre_optimized_configurations(
        self,
        mock_is_fine_tuned,
        mock_is_jumpstart_model,
        mock_js_model,
        mock_is_gated_model,
        mock_serve_settings,
        mock_telemetry,
    ):
        mock_sagemaker_session = Mock()
        mock_sagemaker_session.wait_for_optimization_job.side_effect = (
            lambda *args: mock_optimization_job_response
        )

        mock_lmi_js_model = MagicMock()
        mock_lmi_js_model.image_uri = mock_djl_image_uri
        mock_lmi_js_model.env = {
            "SAGEMAKER_PROGRAM": "inference.py",
            "ENDPOINT_SERVER_TIMEOUT": "3600",
            "MODEL_CACHE_ROOT": "/opt/ml/model",
            "SAGEMAKER_ENV": "1",
            "HF_MODEL_ID": "/opt/ml/model",
            "OPTION_ENFORCE_EAGER": "true",
            "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            "OPTION_TENSOR_PARALLEL_DEGREE": "8",
        }

        mock_js_model.return_value = mock_lmi_js_model

        model_builder = ModelBuilder(
            model="meta-textgeneration-llama-3-1-70b-instruct",
            schema_builder=SchemaBuilder("test", "test"),
            sagemaker_session=mock_sagemaker_session,
        )

        optimized_model = model_builder.optimize(
            accept_eula=True,
            instance_type="ml.g5.24xlarge",
            quantization_config={
                "OverrideEnvironment": {
                    "OPTION_QUANTIZE": "fp8",
                    "OPTION_TENSOR_PARALLEL_DEGREE": "4",
                },
            },
            output_path="s3://bucket/code/",
        )

        assert mock_lmi_js_model.set_deployment_config.call_args_list[0].kwargs == {
            "instance_type": "ml.g5.24xlarge",
            "config_name": "lmi",
        }
        assert optimized_model.env == {
            "SAGEMAKER_PROGRAM": "inference.py",
            "ENDPOINT_SERVER_TIMEOUT": "3600",
            "MODEL_CACHE_ROOT": "/opt/ml/model",
            "SAGEMAKER_ENV": "1",
            "HF_MODEL_ID": "/opt/ml/model",
            "OPTION_ENFORCE_EAGER": "true",
            "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            "OPTION_TENSOR_PARALLEL_DEGREE": "4",  # should be overridden from 8 to 4
            "OPTION_QUANTIZE": "fp8",  # should be added to the env
        }
