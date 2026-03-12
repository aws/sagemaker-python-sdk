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
"""Unit tests for sagemaker.core.lambda_helper module."""
from __future__ import absolute_import

import pytest
import os
import zipfile
from io import BytesIO
from unittest.mock import Mock, patch, MagicMock, call
from botocore.exceptions import ClientError

from sagemaker.core.lambda_helper import (
    Lambda,
    _get_s3_client,
    _get_lambda_client,
    _upload_to_s3,
    _zip_lambda_code,
)


class TestLambdaInit:
    """Test Lambda class initialization."""

    def test_lambda_init_with_function_arn(self):
        """Test initialization with function ARN."""
        lambda_obj = Lambda(
            function_arn="arn:aws:lambda:us-west-2:123456789012:function:my-function"
        )
        assert (
            lambda_obj.function_arn == "arn:aws:lambda:us-west-2:123456789012:function:my-function"
        )
        assert lambda_obj.function_name is None

    def test_lambda_init_with_function_name_and_required_params(self):
        """Test initialization with function name and required parameters."""
        lambda_obj = Lambda(
            function_name="my-function",
            execution_role_arn="arn:aws:iam::123456789012:role/my-role",
            script="/path/to/script.py",
            handler="script.handler",
        )
        assert lambda_obj.function_name == "my-function"
        assert lambda_obj.execution_role_arn == "arn:aws:iam::123456789012:role/my-role"
        assert lambda_obj.script == "/path/to/script.py"
        assert lambda_obj.handler == "script.handler"

    def test_lambda_init_missing_function_arn_and_name(self):
        """Test initialization fails without function ARN or name."""
        with pytest.raises(
            ValueError, match="Either function_arn or function_name must be provided"
        ):
            Lambda()

    def test_lambda_init_missing_execution_role(self):
        """Test initialization fails without execution role when creating new function."""
        with pytest.raises(ValueError, match="execution_role_arn must be provided"):
            Lambda(
                function_name="my-function", script="/path/to/script.py", handler="script.handler"
            )

    def test_lambda_init_missing_code(self):
        """Test initialization fails without code when creating new function."""
        with pytest.raises(ValueError, match="Either zipped_code_dir or script must be provided"):
            Lambda(
                function_name="my-function",
                execution_role_arn="arn:aws:iam::123456789012:role/my-role",
                handler="script.handler",
            )

    def test_lambda_init_both_script_and_zipped_code(self):
        """Test initialization fails with both script and zipped_code_dir."""
        with pytest.raises(ValueError, match="Provide either script or zipped_code_dir, not both"):
            Lambda(
                function_name="my-function",
                execution_role_arn="arn:aws:iam::123456789012:role/my-role",
                script="/path/to/script.py",
                zipped_code_dir="/path/to/code.zip",
                handler="script.handler",
            )

    def test_lambda_init_missing_handler(self):
        """Test initialization fails without handler."""
        with pytest.raises(ValueError, match="Lambda handler must be provided"):
            Lambda(
                function_name="my-function",
                execution_role_arn="arn:aws:iam::123456789012:role/my-role",
                script="/path/to/script.py",
            )

    def test_lambda_init_with_optional_params(self):
        """Test initialization with optional parameters."""
        lambda_obj = Lambda(
            function_name="my-function",
            execution_role_arn="arn:aws:iam::123456789012:role/my-role",
            script="/path/to/script.py",
            handler="script.handler",
            timeout=300,
            memory_size=512,
            runtime="python3.9",
            vpc_config={"SubnetIds": ["subnet-123"]},
            environment={"Variables": {"KEY": "value"}},
            layers=["arn:aws:lambda:us-west-2:123456789012:layer:my-layer:1"],
        )
        assert lambda_obj.timeout == 300
        assert lambda_obj.memory_size == 512
        assert lambda_obj.runtime == "python3.9"
        assert lambda_obj.vpc_config == {"SubnetIds": ["subnet-123"]}
        assert lambda_obj.environment == {"Variables": {"KEY": "value"}}
        assert lambda_obj.layers == ["arn:aws:lambda:us-west-2:123456789012:layer:my-layer:1"]


class TestLambdaCreate:
    """Test Lambda.create method."""

    @patch("sagemaker.core.lambda_helper._get_lambda_client")
    @patch("sagemaker.core.lambda_helper._zip_lambda_code")
    def test_create_with_script(self, mock_zip, mock_get_client):
        """Test creating Lambda function with script."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_zip.return_value = b"zipped_code"
        mock_client.create_function.return_value = {
            "FunctionArn": "arn:aws:lambda:us-west-2:123456789012:function:my-function"
        }

        lambda_obj = Lambda(
            function_name="my-function",
            execution_role_arn="arn:aws:iam::123456789012:role/my-role",
            script="/path/to/script.py",
            handler="script.handler",
        )
        result = lambda_obj.create()

        assert result["FunctionArn"] == "arn:aws:lambda:us-west-2:123456789012:function:my-function"
        mock_client.create_function.assert_called_once()
        call_args = mock_client.create_function.call_args[1]
        assert call_args["FunctionName"] == "my-function"
        assert call_args["Code"] == {"ZipFile": b"zipped_code"}

    @patch("sagemaker.core.lambda_helper._get_lambda_client")
    @patch("sagemaker.core.lambda_helper._get_s3_client")
    @patch("sagemaker.core.lambda_helper._upload_to_s3")
    @patch("sagemaker.core.lambda_helper.s3.determine_bucket_and_prefix")
    def test_create_with_zipped_code(
        self, mock_determine, mock_upload, mock_get_s3, mock_get_lambda
    ):
        """Test creating Lambda function with zipped code directory."""
        mock_lambda_client = Mock()
        mock_get_lambda.return_value = mock_lambda_client
        mock_determine.return_value = ("my-bucket", "prefix")
        mock_upload.return_value = "prefix/lambda/my-function/code"
        mock_lambda_client.create_function.return_value = {
            "FunctionArn": "arn:aws:lambda:us-west-2:123456789012:function:my-function"
        }

        lambda_obj = Lambda(
            function_name="my-function",
            execution_role_arn="arn:aws:iam::123456789012:role/my-role",
            zipped_code_dir="/path/to/code.zip",
            handler="script.handler",
        )
        result = lambda_obj.create()

        assert result["FunctionArn"] == "arn:aws:lambda:us-west-2:123456789012:function:my-function"
        call_args = mock_lambda_client.create_function.call_args[1]
        assert call_args["Code"] == {
            "S3Bucket": "my-bucket",
            "S3Key": "prefix/lambda/my-function/code",
        }

    @patch("sagemaker.core.lambda_helper._get_lambda_client")
    def test_create_without_function_name(self, mock_get_client):
        """Test create fails without function name."""
        lambda_obj = Lambda(
            function_arn="arn:aws:lambda:us-west-2:123456789012:function:my-function"
        )

        with pytest.raises(ValueError, match="FunctionName must be provided"):
            lambda_obj.create()

    @patch("sagemaker.core.lambda_helper._get_lambda_client")
    @patch("sagemaker.core.lambda_helper._zip_lambda_code")
    def test_create_with_client_error(self, mock_zip, mock_get_client):
        """Test create handles ClientError."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_zip.return_value = b"zipped_code"
        error = ClientError(
            {"Error": {"Code": "InvalidParameterValue", "Message": "Invalid parameter"}},
            "CreateFunction",
        )
        mock_client.create_function.side_effect = error

        lambda_obj = Lambda(
            function_name="my-function",
            execution_role_arn="arn:aws:iam::123456789012:role/my-role",
            script="/path/to/script.py",
            handler="script.handler",
        )

        with pytest.raises(ValueError):
            lambda_obj.create()


class TestLambdaUpdate:
    """Test Lambda.update method."""

    @patch("sagemaker.core.lambda_helper._get_lambda_client")
    @patch("sagemaker.core.lambda_helper._zip_lambda_code")
    def test_update_with_script(self, mock_zip, mock_get_client):
        """Test updating Lambda function with script."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_zip.return_value = b"zipped_code"
        mock_client.update_function_code.return_value = {
            "FunctionArn": "arn:aws:lambda:us-west-2:123456789012:function:my-function"
        }

        lambda_obj = Lambda(
            function_name="my-function",
            execution_role_arn="arn:aws:iam::123456789012:role/my-role",
            script="/path/to/script.py",
            handler="script.handler",
        )
        result = lambda_obj.update()

        assert result["FunctionArn"] == "arn:aws:lambda:us-west-2:123456789012:function:my-function"
        mock_client.update_function_code.assert_called_once()

    @patch("sagemaker.core.lambda_helper._get_lambda_client")
    @patch("sagemaker.core.lambda_helper._zip_lambda_code")
    def test_update_with_retry_on_resource_conflict(self, mock_zip, mock_get_client):
        """Test update retries on ResourceConflictException."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_zip.return_value = b"zipped_code"

        error = ClientError(
            {"Error": {"Code": "ResourceConflictException", "Message": "Resource in use"}},
            "UpdateFunctionCode",
        )
        mock_client.update_function_code.side_effect = [
            error,
            {"FunctionArn": "arn:aws:lambda:us-west-2:123456789012:function:my-function"},
        ]

        lambda_obj = Lambda(
            function_name="my-function",
            execution_role_arn="arn:aws:iam::123456789012:role/my-role",
            script="/path/to/script.py",
            handler="script.handler",
        )

        with patch("time.sleep"):
            result = lambda_obj.update()

        assert result["FunctionArn"] == "arn:aws:lambda:us-west-2:123456789012:function:my-function"
        assert mock_client.update_function_code.call_count == 2

    @patch("sagemaker.core.lambda_helper._get_lambda_client")
    @patch("sagemaker.core.lambda_helper._zip_lambda_code")
    def test_update_max_retries_exceeded(self, mock_zip, mock_get_client):
        """Test update fails after max retries."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_zip.return_value = b"zipped_code"

        error = ClientError(
            {"Error": {"Code": "ResourceConflictException", "Message": "Resource in use"}},
            "UpdateFunctionCode",
        )
        mock_client.update_function_code.side_effect = error

        lambda_obj = Lambda(
            function_name="my-function",
            execution_role_arn="arn:aws:iam::123456789012:role/my-role",
            script="/path/to/script.py",
            handler="script.handler",
        )

        with patch("time.sleep"):
            with pytest.raises(ValueError):
                lambda_obj.update()


class TestLambdaUpsert:
    """Test Lambda.upsert method."""

    @patch.object(Lambda, "create")
    def test_upsert_creates_new_function(self, mock_create):
        """Test upsert creates new function when it doesn't exist."""
        mock_create.return_value = {
            "FunctionArn": "arn:aws:lambda:us-west-2:123456789012:function:my-function"
        }

        lambda_obj = Lambda(
            function_name="my-function",
            execution_role_arn="arn:aws:iam::123456789012:role/my-role",
            script="/path/to/script.py",
            handler="script.handler",
        )
        result = lambda_obj.upsert()

        assert result["FunctionArn"] == "arn:aws:lambda:us-west-2:123456789012:function:my-function"
        mock_create.assert_called_once()

    @patch.object(Lambda, "create")
    @patch.object(Lambda, "update")
    def test_upsert_updates_existing_function(self, mock_update, mock_create):
        """Test upsert updates existing function."""
        mock_create.side_effect = ValueError("ResourceConflictException")
        mock_update.return_value = {
            "FunctionArn": "arn:aws:lambda:us-west-2:123456789012:function:my-function"
        }

        lambda_obj = Lambda(
            function_name="my-function",
            execution_role_arn="arn:aws:iam::123456789012:role/my-role",
            script="/path/to/script.py",
            handler="script.handler",
        )
        result = lambda_obj.upsert()

        assert result["FunctionArn"] == "arn:aws:lambda:us-west-2:123456789012:function:my-function"
        mock_update.assert_called_once()


class TestLambdaUpdateConfiguration:
    """Test Lambda.update_configuration method."""

    @patch("sagemaker.core.lambda_helper._get_lambda_client")
    def test_update_configuration_success(self, mock_get_client):
        """Test updating Lambda function configuration."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.update_function_configuration.return_value = {
            "FunctionArn": "arn:aws:lambda:us-west-2:123456789012:function:my-function",
            "Timeout": 300,
            "MemorySize": 256,
        }

        lambda_obj = Lambda(
            function_name="my-function",
            execution_role_arn="arn:aws:iam::123456789012:role/my-role",
            script="/path/to/script.py",
            handler="script.handler",
            timeout=300,
            memory_size=256,
            runtime="python3.12",
            environment={"Variables": {"KEY": "value"}},
            layers=["arn:aws:lambda:us-west-2:123456789012:layer:my-layer:1"],
        )
        result = lambda_obj.update_configuration()

        assert result["Timeout"] == 300
        assert result["MemorySize"] == 256
        mock_client.update_function_configuration.assert_called_once_with(
            FunctionName="my-function",
            Handler="script.handler",
            Runtime="python3.12",
            Role="arn:aws:iam::123456789012:role/my-role",
            Timeout=300,
            MemorySize=256,
            Environment={"Variables": {"KEY": "value"}},
            Layers=["arn:aws:lambda:us-west-2:123456789012:layer:my-layer:1"],
        )

    @patch("sagemaker.core.lambda_helper._get_lambda_client")
    def test_update_configuration_with_function_arn(self, mock_get_client):
        """Test updating configuration using function ARN."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.update_function_configuration.return_value = {"Timeout": 60}

        lambda_obj = Lambda(
            function_arn="arn:aws:lambda:us-west-2:123456789012:function:my-function",
            timeout=60,
        )
        result = lambda_obj.update_configuration()

        assert result["Timeout"] == 60
        call_kwargs = mock_client.update_function_configuration.call_args[1]
        assert call_kwargs["FunctionName"] == "arn:aws:lambda:us-west-2:123456789012:function:my-function"
        assert call_kwargs["Timeout"] == 60

    @patch("sagemaker.core.lambda_helper._get_lambda_client")
    def test_update_configuration_retry_on_resource_conflict(self, mock_get_client):
        """Test update_configuration retries on ResourceConflictException."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        error = ClientError(
            {"Error": {"Code": "ResourceConflictException", "Message": "Resource in use"}},
            "UpdateFunctionConfiguration",
        )
        mock_client.update_function_configuration.side_effect = [
            error,
            {"Timeout": 300},
        ]

        lambda_obj = Lambda(
            function_name="my-function",
            execution_role_arn="arn:aws:iam::123456789012:role/my-role",
            script="/path/to/script.py",
            handler="script.handler",
            timeout=300,
        )

        with patch("time.sleep"):
            result = lambda_obj.update_configuration()

        assert result["Timeout"] == 300
        assert mock_client.update_function_configuration.call_count == 2

    @patch("sagemaker.core.lambda_helper._get_lambda_client")
    def test_update_configuration_max_retries_exceeded(self, mock_get_client):
        """Test update_configuration fails after max retries."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        error = ClientError(
            {"Error": {"Code": "ResourceConflictException", "Message": "Resource in use"}},
            "UpdateFunctionConfiguration",
        )
        mock_client.update_function_configuration.side_effect = error

        lambda_obj = Lambda(
            function_name="my-function",
            execution_role_arn="arn:aws:iam::123456789012:role/my-role",
            script="/path/to/script.py",
            handler="script.handler",
        )

        with patch("time.sleep"):
            with pytest.raises(ValueError):
                lambda_obj.update_configuration()

    @patch("sagemaker.core.lambda_helper._get_lambda_client")
    def test_update_configuration_non_retryable_error(self, mock_get_client):
        """Test update_configuration raises on non-retryable errors."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        error = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "Function not found"}},
            "UpdateFunctionConfiguration",
        )
        mock_client.update_function_configuration.side_effect = error

        lambda_obj = Lambda(
            function_name="my-function",
            execution_role_arn="arn:aws:iam::123456789012:role/my-role",
            script="/path/to/script.py",
            handler="script.handler",
        )

        with pytest.raises(ValueError):
            lambda_obj.update_configuration()


class TestLambdaInvoke:
    """Test Lambda.invoke method."""

    @patch("sagemaker.core.lambda_helper._get_lambda_client")
    def test_invoke_success(self, mock_get_client):
        """Test successful Lambda invocation."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.invoke.return_value = {"StatusCode": 200, "Payload": Mock()}

        lambda_obj = Lambda(
            function_arn="arn:aws:lambda:us-west-2:123456789012:function:my-function"
        )
        result = lambda_obj.invoke()

        assert result["StatusCode"] == 200
        mock_client.invoke.assert_called_once_with(
            FunctionName="arn:aws:lambda:us-west-2:123456789012:function:my-function",
            InvocationType="RequestResponse",
        )

    @patch("sagemaker.core.lambda_helper._get_lambda_client")
    def test_invoke_with_client_error(self, mock_get_client):
        """Test invoke handles ClientError."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        error = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "Function not found"}},
            "Invoke",
        )
        mock_client.invoke.side_effect = error

        lambda_obj = Lambda(
            function_arn="arn:aws:lambda:us-west-2:123456789012:function:my-function"
        )

        with pytest.raises(ValueError):
            lambda_obj.invoke()


class TestLambdaDelete:
    """Test Lambda.delete method."""

    @patch("sagemaker.core.lambda_helper._get_lambda_client")
    def test_delete_success(self, mock_get_client):
        """Test successful Lambda deletion."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.delete_function.return_value = {}

        lambda_obj = Lambda(
            function_arn="arn:aws:lambda:us-west-2:123456789012:function:my-function"
        )
        result = lambda_obj.delete()

        assert result == {}
        mock_client.delete_function.assert_called_once_with(
            FunctionName="arn:aws:lambda:us-west-2:123456789012:function:my-function"
        )

    @patch("sagemaker.core.lambda_helper._get_lambda_client")
    def test_delete_with_client_error(self, mock_get_client):
        """Test delete handles ClientError."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        error = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "Function not found"}},
            "DeleteFunction",
        )
        mock_client.delete_function.side_effect = error

        lambda_obj = Lambda(
            function_arn="arn:aws:lambda:us-west-2:123456789012:function:my-function"
        )

        with pytest.raises(ValueError):
            lambda_obj.delete()


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_s3_client_with_existing_client(self):
        """Test getting S3 client when session already has one."""
        mock_session = Mock()
        mock_s3_client = Mock()
        mock_session.s3_client = mock_s3_client

        result = _get_s3_client(mock_session)

        assert result == mock_s3_client

    def test_get_s3_client_creates_new_client(self):
        """Test creating new S3 client."""
        mock_session = Mock()
        mock_session.s3_client = None
        mock_session.boto_region_name = "us-west-2"
        mock_boto_session = Mock()
        mock_session.boto_session = mock_boto_session
        mock_s3_client = Mock()
        mock_boto_session.client.return_value = mock_s3_client

        result = _get_s3_client(mock_session)

        assert result == mock_s3_client
        mock_boto_session.client.assert_called_once_with("s3", region_name="us-west-2")

    def test_get_lambda_client_with_existing_client(self):
        """Test getting Lambda client when session already has one."""
        mock_session = Mock()
        mock_lambda_client = Mock()
        mock_session.lambda_client = mock_lambda_client

        result = _get_lambda_client(mock_session)

        assert result == mock_lambda_client

    def test_get_lambda_client_creates_new_client(self):
        """Test creating new Lambda client."""
        mock_session = Mock()
        mock_session.lambda_client = None
        mock_session.boto_region_name = "us-west-2"
        mock_boto_session = Mock()
        mock_session.boto_session = mock_boto_session
        mock_lambda_client = Mock()
        mock_boto_session.client.return_value = mock_lambda_client

        result = _get_lambda_client(mock_session)

        assert result == mock_lambda_client
        mock_boto_session.client.assert_called_once_with("lambda", region_name="us-west-2")

    def test_upload_to_s3(self):
        """Test uploading file to S3."""
        mock_s3_client = Mock()

        result = _upload_to_s3(
            mock_s3_client, "my-function", "/path/to/code.zip", "my-bucket", "prefix"
        )

        assert result == "prefix/lambda/my-function/code"
        mock_s3_client.upload_file.assert_called_once_with(
            "/path/to/code.zip", "my-bucket", "prefix/lambda/my-function/code"
        )

    def test_zip_lambda_code(self, tmp_path):
        """Test zipping Lambda code."""
        # Create a temporary script file
        script_file = tmp_path / "test_script.py"
        script_file.write_text("print('Hello, Lambda!')")

        result = _zip_lambda_code(str(script_file))

        assert isinstance(result, bytes)

        # Verify the zip content
        buffer = BytesIO(result)
        with zipfile.ZipFile(buffer, "r") as z:
            assert "test_script.py" in z.namelist()
            content = z.read("test_script.py").decode("utf-8")
            assert content == "print('Hello, Lambda!')"
