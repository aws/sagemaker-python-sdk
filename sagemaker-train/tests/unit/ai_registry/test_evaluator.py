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

import pytest
from unittest.mock import patch, MagicMock
import json

from sagemaker.ai_registry.evaluator import Evaluator, EvaluatorMethod
from sagemaker.ai_registry.air_constants import (
    RESPONSE_KEY_HUB_CONTENT_VERSION, RESPONSE_KEY_HUB_CONTENT_ARN,
    RESPONSE_KEY_CREATION_TIME, RESPONSE_KEY_LAST_MODIFIED_TIME, REWARD_FUNCTION
)


class TestEvaluator:
    @patch('sagemaker.ai_registry.evaluator.AIRHub')
    def test_create_with_lambda_arn(self, mock_air_hub):
        mock_air_hub.import_hub_content.return_value = {"HubContentArn": "test-arn"}
        mock_air_hub.describe_hub_content.return_value = {
            RESPONSE_KEY_HUB_CONTENT_VERSION: "1.0.0",
            RESPONSE_KEY_HUB_CONTENT_ARN: "test-arn",
            RESPONSE_KEY_CREATION_TIME: "2024-01-01",
            RESPONSE_KEY_LAST_MODIFIED_TIME: "2024-01-01"
        }
        
        evaluator = Evaluator.create(
            name="test-evaluator",
            source="arn:aws:lambda:us-west-2:123456789012:function:test",
            type=REWARD_FUNCTION,
            wait=False
        )
        
        assert evaluator.name == "test-evaluator"
        assert evaluator.version == "1.0.0"
        assert evaluator.method == EvaluatorMethod.LAMBDA
        mock_air_hub.import_hub_content.assert_called_once()

    @patch('sagemaker.ai_registry.evaluator.boto3')
    @patch('sagemaker.ai_registry.evaluator.AIRHub')
    def test_create_with_byoc(self, mock_air_hub, mock_boto3):
        mock_lambda_client = MagicMock()
        mock_boto3.client.return_value = mock_lambda_client
        mock_lambda_client.create_function.return_value = {"FunctionArn": "lambda-arn"}
        
        mock_air_hub.upload_to_s3.return_value = "s3://bucket/path"
        mock_air_hub.import_hub_content.return_value = {"HubContentArn": "test-arn"}
        mock_air_hub.describe_hub_content.return_value = {
            RESPONSE_KEY_HUB_CONTENT_VERSION: "1.0.0",
            RESPONSE_KEY_HUB_CONTENT_ARN: "test-arn",
            RESPONSE_KEY_CREATION_TIME: "2024-01-01",
            RESPONSE_KEY_LAST_MODIFIED_TIME: "2024-01-01"
        }
        
        with patch("builtins.open", MagicMock()), \
             patch("zipfile.ZipFile") as mock_zip, \
             patch("os.path.splitext", return_value=("function", ".py")), \
             patch("os.path.basename", return_value="function.py"):
            
            mock_zip_instance = MagicMock()
            mock_zip.return_value.__enter__.return_value = mock_zip_instance
            
            evaluator = Evaluator.create(
                name="test-evaluator",
                source="/local/path/function.py",
                type=REWARD_FUNCTION,
                wait=False
            )
        
        assert evaluator.method == EvaluatorMethod.BYOC
        mock_air_hub.upload_to_s3.assert_called_once()

    @patch('sagemaker.ai_registry.evaluator.AIRHub')
    def test_get_all(self, mock_air_hub):
        mock_air_hub.list_hub_content.return_value = {
            "items": [
                {
                    "HubContentName": "eval1",
                    "HubContentVersion": "1.0.0",
                    "HubContentArn": "arn1",
                    "HubContentStatus": "Available",
                    "HubContentDocument": json.dumps({
                        "JsonContent": json.dumps({"Reference": "ref1", "SubType": "AWS/Evaluator"})
                    }),
                    "HubContentSearchKeywords": ["method:lambda"],
                    "CreationTime": "2024-01-01",
                    "LastModifiedTime": "2024-01-01"
                }
            ],
            "next_token": None
        }
        
        evaluator_list = Evaluator.get_all()
        
        assert evaluator_list.next_token is None

    @patch('sagemaker.ai_registry.evaluator.AIRHub')
    def test_get_versions(self, mock_air_hub):
        mock_air_hub.list_hub_content_versions.return_value = [
            {"HubContentVersion": "1.0.0"},
            {"HubContentVersion": "2.0.0"}
        ]
        mock_air_hub.describe_hub_content.return_value = {
            "HubContentName": "test-eval",
            "HubContentArn": "test-arn",
            "HubContentVersion": "1.0.0",
            "HubContentStatus": "Available",
            "HubContentDocument": json.dumps({
                "SubType": "AWS/Evaluator",
                "JsonContent": json.dumps({"Reference": "ref"})
            }),
            "HubContentSearchKeywords": ["method:lambda"],
            "CreationTime": "2024-01-01",
            "LastModifiedTime": "2024-01-01"
        }
        
        evaluator = Evaluator("test", "1.0.0", "arn", "AWS/Evaluator", method=EvaluatorMethod.LAMBDA)
        versions = evaluator.get_versions()
        
        assert len(versions) == 2

    @patch('sagemaker.ai_registry.evaluator.Evaluator.create')
    def test_create_version_success(self, mock_create):
        mock_create.return_value = MagicMock()
        
        evaluator = Evaluator("test", "1.0.0", "arn", "AWS/Evaluator", method=EvaluatorMethod.LAMBDA, reference="lambda-arn")
        result = evaluator.create_version("arn:aws:lambda:us-west-2:123456789012:function:new")
        
        assert result is True
        mock_create.assert_called_once()

    @patch('sagemaker.ai_registry.evaluator.AIRHub')
    def test_create_version_failure(self, mock_air_hub):
        mock_air_hub.describe_hub_content.side_effect = Exception("Error")
        
        evaluator = Evaluator("test", "1.0.0", "arn", "RewardFunction", method=EvaluatorMethod.LAMBDA)
        
        with pytest.raises(RuntimeError, match="Failed to create new version: Error"):
            evaluator.create_version("arn:aws:lambda:us-west-2:123456789012:function:new")
