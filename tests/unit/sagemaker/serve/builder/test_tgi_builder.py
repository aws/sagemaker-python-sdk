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

from unittest import TestCase
from unittest.mock import MagicMock, patch
from sagemaker.serve.builder.model_builder import ModelBuilder
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.utils.predictors import TgiLocalModePredictor

MOCK_MODEL_ID = "meta-llama/Meta-Llama-3-8B"
MOCK_PROMPT = "The man worked as a [MASK]."
MOCK_SAMPLE_INPUT = {"inputs": "Hello, I'm a language model", "parameters": {"max_new_tokens": 128}}
MOCK_SAMPLE_OUTPUT = [{"generated_text": "Hello, I'm a language modeler."}]
MOCK_SCHEMA_BUILDER = MagicMock()
MOCK_SCHEMA_BUILDER.sample_input = MOCK_SAMPLE_INPUT
MOCK_SCHEMA_BUILDER.sample_output = MOCK_SAMPLE_OUTPUT
MOCK_IMAGE_CONFIG = (
    "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
    "huggingface-pytorch-inference:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04-v1.0"
)
MOCK_MODEL_PATH = "mock model path"


class TestTGIBuilder(TestCase):
    @patch(
        "sagemaker.serve.builder.tgi_builder._get_nb_instance",
        return_value="ml.g5.24xlarge",
    )
    @patch("sagemaker.serve.builder.tgi_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.model_builder.get_huggingface_model_metadata",
        return_value={"pipeline_tag": "text-generation"},
    )
    @patch(
        "sagemaker.serve.builder.tgi_builder._get_model_config_properties_from_hf",
        return_value=({}, None),
    )
    @patch(
        "sagemaker.serve.builder.tgi_builder._get_default_tgi_configurations",
        return_value=({}, None),
    )
    def test_tgi_builder_sagemaker_endpoint_mode_no_s3_upload_success(
        self,
        mock_default_tgi_configurations,
        mock_hf_model_config,
        mock_hf_model_md,
        mock_get_nb_instance,
        mock_telemetry,
    ):
        # verify SAGEMAKER_ENDPOINT deploy
        builder = ModelBuilder(
            model=MOCK_MODEL_ID,
            name="mock_model_name",
            schema_builder=MOCK_SCHEMA_BUILDER,
            mode=Mode.SAGEMAKER_ENDPOINT,
        )

        builder._prepare_for_mode = MagicMock()
        builder._prepare_for_mode.return_value = (None, {})

        model = builder.build()
        assert model.name == "mock_model_name"

        builder.serve_settings.telemetry_opt_out = True
        builder._original_deploy = MagicMock()

        model.deploy(mode=Mode.SAGEMAKER_ENDPOINT, role="mock_role_arn")

        assert "HF_MODEL_ID" in model.env
        with self.assertRaises(ValueError) as _:
            model.deploy(mode=Mode.IN_PROCESS)
        builder._prepare_for_mode.assert_called_with()

    @patch(
        "sagemaker.serve.builder.tgi_builder._get_nb_instance",
        return_value="ml.g5.24xlarge",
    )
    @patch("sagemaker.serve.builder.tgi_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.model_builder.get_huggingface_model_metadata",
        return_value={"pipeline_tag": "text-generation"},
    )
    @patch(
        "sagemaker.serve.builder.tgi_builder._get_model_config_properties_from_hf",
        return_value=({}, None),
    )
    @patch(
        "sagemaker.serve.builder.tgi_builder._get_default_tgi_configurations",
        return_value=({}, None),
    )
    def test_tgi_builder_overwritten_deploy_from_local_container_to_sagemaker_endpoint_success(
        self,
        mock_default_tgi_configurations,
        mock_hf_model_config,
        mock_hf_model_md,
        mock_get_nb_instance,
        mock_telemetry,
    ):
        # verify LOCAL_CONTAINER deploy
        builder = ModelBuilder(
            model=MOCK_MODEL_ID,
            schema_builder=MOCK_SCHEMA_BUILDER,
            mode=Mode.LOCAL_CONTAINER,
            model_path=MOCK_MODEL_PATH,
        )

        builder._prepare_for_mode = MagicMock()
        builder._prepare_for_mode.side_effect = None
        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True
        builder.modes[str(Mode.LOCAL_CONTAINER)] = MagicMock()

        predictor = model.deploy(model_data_download_timeout=1800)

        assert builder.env_vars["MODEL_LOADING_TIMEOUT"] == "1800"
        assert isinstance(predictor, TgiLocalModePredictor)
        assert builder.nb_instance_type == "ml.g5.24xlarge"

        # verify SAGEMAKER_ENDPOINT overwritten deploy
        builder._original_deploy = MagicMock()
        builder._prepare_for_mode.return_value = (None, {})

        model.deploy(mode=Mode.SAGEMAKER_ENDPOINT, role="mock_role_arn")

        assert "HF_MODEL_ID" in model.env
        with self.assertRaises(ValueError) as _:
            model.deploy(mode=Mode.IN_PROCESS)
        builder._prepare_for_mode.call_args_list[1].assert_called_once_with(
            model_path=MOCK_MODEL_PATH, should_upload_artifacts=True
        )

    @patch(
        "sagemaker.serve.builder.tgi_builder._get_nb_instance",
        return_value="ml.g5.24xlarge",
    )
    @patch("sagemaker.serve.builder.tgi_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.model_builder.get_huggingface_model_metadata",
        return_value={"pipeline_tag": "text-generation"},
    )
    @patch(
        "sagemaker.serve.builder.tgi_builder._get_model_config_properties_from_hf",
        return_value=({}, None),
    )
    @patch(
        "sagemaker.serve.builder.tgi_builder._get_default_tgi_configurations",
        return_value=({}, None),
    )
    @patch("sagemaker.serve.builder.tgi_builder._is_optimized", return_value=True)
    def test_tgi_builder_optimized_sagemaker_endpoint_mode_no_s3_upload_success(
        self,
        mock_is_optimized,
        mock_default_tgi_configurations,
        mock_hf_model_config,
        mock_hf_model_md,
        mock_get_nb_instance,
        mock_telemetry,
    ):
        # verify LOCAL_CONTAINER deploy
        builder = ModelBuilder(
            model=MOCK_MODEL_ID,
            schema_builder=MOCK_SCHEMA_BUILDER,
            mode=Mode.LOCAL_CONTAINER,
            model_path=MOCK_MODEL_PATH,
        )

        builder._prepare_for_mode = MagicMock()
        builder._prepare_for_mode.side_effect = None
        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True
        builder.modes[str(Mode.LOCAL_CONTAINER)] = MagicMock()

        model.deploy(model_data_download_timeout=1800)

        # verify SAGEMAKER_ENDPOINT overwritten deploy
        builder._original_deploy = MagicMock()
        builder._prepare_for_mode.return_value = (None, {})

        model.deploy(mode=Mode.SAGEMAKER_ENDPOINT, role="mock_role_arn")

        # verify that if optimized, no s3 upload occurs
        builder._prepare_for_mode.assert_called_with()

    @patch(
        "sagemaker.serve.builder.tgi_builder._get_nb_instance",
        return_value="ml.g5.24xlarge",
    )
    @patch("sagemaker.serve.builder.tgi_builder._capture_telemetry", side_effect=None)
    @patch(
        "sagemaker.serve.builder.model_builder.get_huggingface_model_metadata",
        return_value={"pipeline_tag": "text-generation"},
    )
    @patch(
        "sagemaker.serve.builder.tgi_builder._get_model_config_properties_from_hf",
        return_value=({}, None),
    )
    @patch(
        "sagemaker.serve.builder.tgi_builder._get_default_tgi_configurations",
        return_value=({}, None),
    )
    @patch(
        "sagemaker.serve.builder.tgi_builder._get_admissible_tensor_parallel_degrees",
        return_value=[4, 8],
    )
    @patch("sagemaker.serve.builder.tgi_builder._get_admissible_dtypes", return_value=["fp16"])
    @patch("sagemaker.serve.builder.tgi_builder.datetime")
    @patch("sagemaker.serve.builder.tgi_builder.timedelta", return_value=1800)
    @patch("sagemaker.serve.builder.tgi_builder._serial_benchmark")
    @patch("sagemaker.serve.builder.tgi_builder._concurrent_benchmark")
    def test_tgi_builder_tune_success(
        self,
        mock_concurrent_benchmark,
        mock_serial_benchmark,
        mock_timedelta,
        mock_datetime,
        mock_get_admissible_dtypes,
        mock_get_admissible_tensor_parallel_degrees,
        mock_default_tgi_configurations,
        mock_hf_model_config,
        mock_hf_model_md,
        mock_get_nb_instance,
        mock_telemetry,
    ):
        # WHERE
        mock_datetime.now.side_effect = [0, 100, 200]
        mock_serial_benchmark.side_effect = [(1000, 10000, 10), (500, 5000, 50)]
        mock_concurrent_benchmark.side_effect = [(10, 10), (50, 5)]

        builder = ModelBuilder(
            model=MOCK_MODEL_ID,
            schema_builder=MOCK_SCHEMA_BUILDER,
            mode=Mode.LOCAL_CONTAINER,
            model_path=MOCK_MODEL_PATH,
        )
        builder._prepare_for_mode = MagicMock()
        builder._prepare_for_mode.side_effect = None

        model = builder.build()

        builder.serve_settings.telemetry_opt_out = True
        builder.modes[str(Mode.LOCAL_CONTAINER)] = MagicMock()
        builder.pysdk_model = MagicMock()

        # WHEN
        ret_new_model = model.tune(max_tuning_duration=1800)

        # THEN
        assert ret_new_model != model
        assert len(mock_datetime.now.call_args_list) == 3
        assert len(mock_serial_benchmark.call_args_list) == 2
        assert len(mock_concurrent_benchmark.call_args_list) == 2
        assert ret_new_model.env["NUM_SHARD"] == "8"
        assert ret_new_model.env["DTYPE"] == "fp16"
        assert ret_new_model.env["SHARDED"] == "true"
