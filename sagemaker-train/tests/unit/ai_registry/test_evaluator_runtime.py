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
"""Tests for the Lambda runtime of ``Evaluator.create``.

Before this change ``EVALUATOR_DEFAULT_RUNTIME`` was ``python3.9``, ``Evaluator.create`` had
no ``runtime`` parameter, and ``CustomAgentLambda.create`` carried its own ``python3.12``
literal, so the two Lambda paths of the SDK deployed different runtimes.
"""

import pytest
from unittest.mock import patch, MagicMock

from sagemaker.ai_registry.evaluator import Evaluator, EvaluatorMethod
from sagemaker.ai_registry.air_constants import (
    EVALUATOR_DEFAULT_RUNTIME,
    RESPONSE_KEY_HUB_CONTENT_VERSION,
    RESPONSE_KEY_HUB_CONTENT_ARN,
    RESPONSE_KEY_CREATION_TIME,
    RESPONSE_KEY_LAST_MODIFIED_TIME,
    REWARD_FUNCTION,
)
from sagemaker.train.custom_agent_lambda import CustomAgentLambda

DUMMY_ROLE = "arn:aws:iam::123456789012:role/SageMakerRole"


def test_default_runtime_is_python312():
    assert EVALUATOR_DEFAULT_RUNTIME == "python3.12"


def test_custom_agent_lambda_shares_the_evaluator_runtime():
    defaults = CustomAgentLambda.create.__func__.__defaults__
    assert EVALUATOR_DEFAULT_RUNTIME in defaults


class TestEvaluatorRuntime:
    @pytest.fixture(autouse=True)
    def stub_aws_resolution(self):
        """Keep the tests offline, same stubs as ``test_evaluator.py``."""
        session = MagicMock()
        with (
            patch("sagemaker.ai_registry.evaluator.Session", return_value=session),
            patch("sagemaker.ai_registry.evaluator._get_current_domain_id", return_value=None),
            patch(
                "sagemaker.train.defaults.TrainDefaults.get_sagemaker_session",
                return_value=session,
            ),
            patch("sagemaker.train.defaults.TrainDefaults.get_role", return_value=DUMMY_ROLE),
            patch(
                "sagemaker.ai_registry.air_hub_entity.AIRHub.get_hub_name",
                return_value="test-hub",
            ),
        ):
            yield

    @pytest.fixture
    def lambda_client(self):
        with (
            patch("sagemaker.ai_registry.evaluator._get_default_bucket", return_value="bucket"),
            patch("sagemaker.ai_registry.evaluator.boto3") as mock_boto3,
            patch("sagemaker.ai_registry.evaluator.AIRHub") as mock_air_hub,
            patch("zipfile.ZipFile"),
        ):
            client = MagicMock()
            mock_boto3.client.return_value = client
            client.create_function.return_value = {"FunctionArn": "lambda-arn"}
            mock_air_hub.upload_to_s3.return_value = "s3://bucket/path"
            mock_air_hub.import_hub_content.return_value = {"HubContentArn": "test-arn"}
            mock_air_hub.describe_hub_content.return_value = {
                RESPONSE_KEY_HUB_CONTENT_VERSION: "1.0.0",
                RESPONSE_KEY_HUB_CONTENT_ARN: "test-arn",
                RESPONSE_KEY_CREATION_TIME: "2024-01-01",
                RESPONSE_KEY_LAST_MODIFIED_TIME: "2024-01-01",
            }
            yield client

    def test_create_from_file_uses_python312_by_default(self, lambda_client):
        evaluator = Evaluator.create(
            name="test-evaluator",
            source="/local/path/function.py",
            type=REWARD_FUNCTION,
            wait=False,
        )

        assert evaluator.method == EvaluatorMethod.BYOC
        call_kwargs = lambda_client.create_function.call_args.kwargs
        assert call_kwargs["Runtime"] == "python3.12"

    def test_runtime_reaches_the_lambda_create_call(self, lambda_client):
        Evaluator.create(
            name="test-evaluator",
            source="/local/path/function.py",
            type=REWARD_FUNCTION,
            wait=False,
            runtime="python3.13",
        )

        call_kwargs = lambda_client.create_function.call_args.kwargs
        assert call_kwargs["Runtime"] == "python3.13"
        assert call_kwargs["Handler"] == "lambda_function.lambda_handler"
