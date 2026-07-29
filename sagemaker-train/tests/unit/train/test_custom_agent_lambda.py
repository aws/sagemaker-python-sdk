"""Unit tests for CustomAgentLambda."""
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.train.custom_agent_lambda import CustomAgentLambda


MOCK_ROLE = "arn:aws:iam::123:role/test"
MOCK_ARN = "arn:aws:lambda:us-west-2:123:function:my-fn"


class TestCustomAgentLambdaCreate:
    @patch("sagemaker.train.custom_agent_lambda.boto3")
    @patch("sagemaker.train.defaults.TrainDefaults.get_role", return_value=MOCK_ROLE)
    def test_create_from_inline_code(self, mock_get_role, mock_boto3):
        mock_client = MagicMock()
        mock_client.create_function.return_value = {"FunctionArn": MOCK_ARN}
        mock_boto3.client.return_value = mock_client

        adapter = CustomAgentLambda.create(
            function_name="my-fn",
            source="def handler(event, ctx): return {}",
        )
        assert adapter.lambda_arn == MOCK_ARN
        call_kwargs = mock_client.create_function.call_args[1]
        assert "ZipFile" in call_kwargs["Code"]

    @patch("sagemaker.train.custom_agent_lambda.boto3")
    @patch("sagemaker.train.defaults.TrainDefaults.get_role", return_value=MOCK_ROLE)
    def test_create_from_local_file(self, mock_get_role, mock_boto3):
        mock_client = MagicMock()
        mock_client.create_function.return_value = {"FunctionArn": MOCK_ARN}
        mock_boto3.client.return_value = mock_client

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def handler(event, ctx): return {}")
            f.flush()
            adapter = CustomAgentLambda.create(function_name="my-fn", source=f.name)
        os.unlink(f.name)
        assert adapter.lambda_arn == MOCK_ARN
        call_kwargs = mock_client.create_function.call_args[1]
        assert "ZipFile" in call_kwargs["Code"]

    @patch("sagemaker.train.custom_agent_lambda.boto3")
    @patch("sagemaker.train.defaults.TrainDefaults.get_role", return_value=MOCK_ROLE)
    def test_create_from_s3_uri(self, mock_get_role, mock_boto3):
        mock_client = MagicMock()
        mock_client.create_function.return_value = {"FunctionArn": MOCK_ARN}
        mock_boto3.client.return_value = mock_client

        adapter = CustomAgentLambda.create(
            function_name="my-fn",
            source="s3://my-bucket/code/handler.zip",
        )
        assert adapter.lambda_arn == MOCK_ARN
        call_kwargs = mock_client.create_function.call_args[1]
        assert call_kwargs["Code"] == {
            "S3Bucket": "my-bucket",
            "S3Key": "code/handler.zip",
        }

    @patch("sagemaker.train.custom_agent_lambda.boto3")
    @patch("sagemaker.train.defaults.TrainDefaults.get_role", return_value=MOCK_ROLE)
    def test_create_generates_function_name_when_not_provided(self, mock_get_role, mock_boto3):
        mock_client = MagicMock()
        mock_client.create_function.return_value = {"FunctionArn": MOCK_ARN}
        mock_boto3.client.return_value = mock_client

        adapter = CustomAgentLambda.create(
            source="def handler(event, ctx): return {}",
        )
        assert adapter.lambda_arn == MOCK_ARN
        call_kwargs = mock_client.create_function.call_args[1]
        assert call_kwargs["FunctionName"].startswith("SageMaker-agent-adapter-")

    def test_create_empty_source_raises(self):
        with pytest.raises(ValueError, match="must be provided"):
            CustomAgentLambda.create(source="")

    def test_create_whitespace_source_raises(self):
        with pytest.raises(ValueError, match="must be provided"):
            CustomAgentLambda.create(source="   ")


class TestCustomAgentLambdaGet:
    @patch("sagemaker.train.custom_agent_lambda.boto3")
    def test_get_existing(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        adapter = CustomAgentLambda.get(MOCK_ARN)
        assert adapter.lambda_arn == MOCK_ARN
        mock_client.get_function.assert_called_once()

    def test_repr(self):
        adapter = CustomAgentLambda(MOCK_ARN)
        assert MOCK_ARN in repr(adapter)
