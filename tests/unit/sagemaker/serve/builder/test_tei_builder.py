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
from tests.unit.sagemaker.serve.constants import MOCK_VPC_CONFIG

from sagemaker.serve.utils.predictors import TeiLocalModePredictor

MOCK_MODEL_ID = "bert-base-uncased"
MOCK_PROMPT = "The man worked as a [MASK]."
MOCK_SAMPLE_INPUT = {"inputs": MOCK_PROMPT}
MOCK_SAMPLE_OUTPUT = [
    {
        "score": 0.0974755585193634,
        "token": 10533,
        "token_str": "carpenter",
        "sequence": "the man worked as a carpenter.",
    },
    {
        "score": 0.052383411675691605,
        "token": 15610,
        "token_str": "waiter",
        "sequence": "the man worked as a waiter.",
    },
    {
        "score": 0.04962712526321411,
        "token": 13362,
        "token_str": "barber",
        "sequence": "the man worked as a barber.",
    },
    {
        "score": 0.0378861166536808,
        "token": 15893,
        "token_str": "mechanic",
        "sequence": "the man worked as a mechanic.",
    },
    {
        "score": 0.037680838257074356,
        "token": 18968,
        "token_str": "salesman",
        "sequence": "the man worked as a salesman.",
    },
]
MOCK_SCHEMA_BUILDER = MagicMock()
MOCK_SCHEMA_BUILDER.sample_input = MOCK_SAMPLE_INPUT
MOCK_SCHEMA_BUILDER.sample_output = MOCK_SAMPLE_OUTPUT
MOCK_IMAGE_CONFIG = (
    "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
    "huggingface-pytorch-inference:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04-v1.0"
)
MOCK_MODEL_PATH = "mock model path"


class TestTEIBuilder(unittest.TestCase):
    @patch(
        "sagemaker.serve.builder.tei_builder._get_nb_instance",
        return_value="ml.g5.24xlarge",
    )
    @patch("sagemaker.serve.builder.tei_builder._capture_telemetry", side_effect=None)
    def test_tei_builder_sagemaker_endpoint_mode_no_s3_upload_success(
        self,
        mock_get_nb_instance,
        mock_telemetry,
    ):
        # verify SAGEMAKER_ENDPOINT deploy
        builder = ModelBuilder(
            model=MOCK_MODEL_ID,
            name="mock_model_name",
            schema_builder=MOCK_SCHEMA_BUILDER,
            mode=Mode.SAGEMAKER_ENDPOINT,
            model_metadata={
                "HF_TASK": "sentence-similarity",
            },
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
        "sagemaker.serve.builder.tei_builder._get_nb_instance",
        return_value="ml.g5.24xlarge",
    )
    @patch("sagemaker.serve.builder.tei_builder._capture_telemetry", side_effect=None)
    def test_tei_builder_overwritten_deploy_from_local_container_to_sagemaker_endpoint_success(
        self,
        mock_get_nb_instance,
        mock_telemetry,
    ):
        # verify LOCAL_CONTAINER deploy
        builder = ModelBuilder(
            model=MOCK_MODEL_ID,
            schema_builder=MOCK_SCHEMA_BUILDER,
            mode=Mode.LOCAL_CONTAINER,
            vpc_config=MOCK_VPC_CONFIG,
            model_metadata={
                "HF_TASK": "sentence-similarity",
            },
            model_path=MOCK_MODEL_PATH,
        )

        builder._prepare_for_mode = MagicMock()
        builder._prepare_for_mode.side_effect = None
        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True
        builder.modes[str(Mode.LOCAL_CONTAINER)] = MagicMock()

        predictor = model.deploy(model_data_download_timeout=1800)

        assert model.vpc_config == MOCK_VPC_CONFIG
        assert builder.env_vars["MODEL_LOADING_TIMEOUT"] == "1800"
        assert isinstance(predictor, TeiLocalModePredictor)
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
        "sagemaker.serve.builder.tei_builder._get_nb_instance",
        return_value="ml.g5.24xlarge",
    )
    @patch("sagemaker.serve.builder.tei_builder._capture_telemetry", side_effect=None)
    @patch("sagemaker.serve.builder.tei_builder._is_optimized", return_value=True)
    def test_tei_builder_optimized_sagemaker_endpoint_mode_no_s3_upload_success(
        self,
        mock_is_optimized,
        mock_get_nb_instance,
        mock_telemetry,
    ):
        # verify LOCAL_CONTAINER deploy
        builder = ModelBuilder(
            model=MOCK_MODEL_ID,
            schema_builder=MOCK_SCHEMA_BUILDER,
            mode=Mode.LOCAL_CONTAINER,
            vpc_config=MOCK_VPC_CONFIG,
            model_metadata={
                "HF_TASK": "sentence-similarity",
            },
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
        "sagemaker.serve.builder.tei_builder._get_nb_instance",
        return_value="ml.g5.24xlarge",
    )
    @patch("sagemaker.serve.builder.tei_builder._capture_telemetry", side_effect=None)
    def test_tei_builder_image_uri_override_success(
        self,
        mock_get_nb_instance,
        mock_telemetry,
    ):
        builder = ModelBuilder(
            model=MOCK_MODEL_ID,
            schema_builder=MOCK_SCHEMA_BUILDER,
            mode=Mode.LOCAL_CONTAINER,
            image_uri=MOCK_IMAGE_CONFIG,
            model_metadata={
                "HF_TASK": "sentence-similarity",
            },
        )

        builder._prepare_for_mode = MagicMock()
        builder._prepare_for_mode.side_effect = None

        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True

        builder.modes[str(Mode.LOCAL_CONTAINER)] = MagicMock()
        predictor = model.deploy(model_data_download_timeout=1800)

        assert builder.image_uri == MOCK_IMAGE_CONFIG
        assert builder.env_vars["MODEL_LOADING_TIMEOUT"] == "1800"
        assert isinstance(predictor, TeiLocalModePredictor)

        assert builder.nb_instance_type == "ml.g5.24xlarge"

        builder._original_deploy = MagicMock()
        builder._prepare_for_mode.return_value = (None, {})
        predictor = model.deploy(mode=Mode.SAGEMAKER_ENDPOINT, role="mock_role_arn")
        assert "HF_MODEL_ID" in model.env

        with self.assertRaises(ValueError) as _:
            model.deploy(mode=Mode.IN_PROCESS)
