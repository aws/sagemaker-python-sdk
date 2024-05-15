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

from sagemaker.serve.utils.predictors import TransformersLocalModePredictor

mock_model_id = "bert-base-uncased"
mock_prompt = "The man worked as a [MASK]."
mock_sample_input = {"inputs": mock_prompt}
mock_sample_output = [
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
mock_schema_builder = MagicMock()
mock_schema_builder.sample_input = mock_sample_input
mock_schema_builder.sample_output = mock_sample_output
MOCK_IMAGE_CONFIG = (
    "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
    "huggingface-pytorch-inference:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04-v1.0"
)


class TestTransformersBuilder(unittest.TestCase):
    @patch(
        "sagemaker.serve.builder.transformers_builder._get_nb_instance",
        return_value="ml.g5.24xlarge",
    )
    @patch("sagemaker.serve.builder.transformers_builder._capture_telemetry", side_effect=None)
    def test_build_deploy_for_transformers_local_container_and_remote_container(
        self,
        mock_get_nb_instance,
        mock_telemetry,
    ):
        builder = ModelBuilder(
            model=mock_model_id,
            schema_builder=mock_schema_builder,
            mode=Mode.LOCAL_CONTAINER,
            vpc_config=MOCK_VPC_CONFIG,
        )

        builder._prepare_for_mode = MagicMock()
        builder._prepare_for_mode.side_effect = None

        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True

        builder.modes[str(Mode.LOCAL_CONTAINER)] = MagicMock()
        predictor = model.deploy(model_data_download_timeout=1800)

        assert model.vpc_config == MOCK_VPC_CONFIG
        assert builder.env_vars["MODEL_LOADING_TIMEOUT"] == "1800"
        assert isinstance(predictor, TransformersLocalModePredictor)

        assert builder.nb_instance_type == "ml.g5.24xlarge"

        builder._original_deploy = MagicMock()
        builder._prepare_for_mode.return_value = (None, {})
        predictor = model.deploy(mode=Mode.SAGEMAKER_ENDPOINT, role="mock_role_arn")
        assert "HF_MODEL_ID" in model.env

        with self.assertRaises(ValueError) as _:
            model.deploy(mode=Mode.IN_PROCESS)

    @patch(
        "sagemaker.serve.builder.transformers_builder._get_nb_instance",
        return_value="ml.g5.24xlarge",
    )
    @patch("sagemaker.serve.builder.transformers_builder._capture_telemetry", side_effect=None)
    def test_image_uri(
        self,
        mock_get_nb_instance,
        mock_telemetry,
    ):
        builder = ModelBuilder(
            model=mock_model_id,
            schema_builder=mock_schema_builder,
            mode=Mode.LOCAL_CONTAINER,
            image_uri=MOCK_IMAGE_CONFIG,
        )

        builder._prepare_for_mode = MagicMock()
        builder._prepare_for_mode.side_effect = None

        model = builder.build()
        builder.serve_settings.telemetry_opt_out = True

        builder.modes[str(Mode.LOCAL_CONTAINER)] = MagicMock()
        predictor = model.deploy(model_data_download_timeout=1800)

        assert builder.image_uri == MOCK_IMAGE_CONFIG
        assert builder.env_vars["MODEL_LOADING_TIMEOUT"] == "1800"
        assert isinstance(predictor, TransformersLocalModePredictor)

        assert builder.nb_instance_type == "ml.g5.24xlarge"

        builder._original_deploy = MagicMock()
        builder._prepare_for_mode.return_value = (None, {})
        predictor = model.deploy(mode=Mode.SAGEMAKER_ENDPOINT, role="mock_role_arn")
        assert "HF_MODEL_ID" in model.env

        with self.assertRaises(ValueError) as _:
            model.deploy(mode=Mode.IN_PROCESS)
