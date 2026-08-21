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
import os
import threading
import unittest
from time import perf_counter
import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
import boto3
import sagemaker
from sagemaker.core.telemetry.constants import Feature, DEFAULT_AWS_REGION
from sagemaker.core.telemetry.attribution import _CREATED_BY_ENV_VAR
from sagemaker.core.telemetry.telemetry_logging import (
    _send_telemetry_request,
    _send_telemetry_request_sync,
    _telemetry_emitter,
    _construct_url,
    _get_accountId,
    _requests_helper,
    _get_region_or_default,
    _get_default_sagemaker_session,
    OS_NAME_VERSION,
    PYTHON_VERSION,
    TELEMETRY_REQUEST_TIMEOUT,
)
from sagemaker.core.user_agent import SDK_VERSION, process_studio_metadata_file

# Try to import sagemaker-serve exceptions, skip tests if not available
try:
    from sagemaker.serve.utils.exceptions import (
        ModelBuilderException,
        LocalModelOutOfMemoryException,
    )

    SAGEMAKER_SERVE_AVAILABLE = True
except ImportError:
    SAGEMAKER_SERVE_AVAILABLE = False

    # Create mock exceptions for type hints
    class ModelBuilderException(Exception):
        pass

    class LocalModelOutOfMemoryException(Exception):
        pass


MOCK_SESSION = Mock()
MOCK_EXCEPTION = LocalModelOutOfMemoryException("mock raise ex")
MOCK_FEATURE = Feature.SDK_DEFAULTS
MOCK_FUNC_NAME = "Mock.local_session.create_model"
MOCK_ENDPOINT_ARN = "arn:aws:sagemaker:us-west-2:123456789012:endpoint/test"


class LocalSagemakerClientMock:
    def __init__(self):
        self.sagemaker_session = MOCK_SESSION

    @_telemetry_emitter(MOCK_FEATURE, MOCK_FUNC_NAME)
    def mock_create_model(self, mock_exception_func=None):
        if mock_exception_func:
            mock_exception_func()


class TestTelemetryLogging(unittest.TestCase):
    @patch("sagemaker.core.telemetry.telemetry_logging._requests_helper")
    @patch("sagemaker.core.telemetry.telemetry_logging._get_accountId")
    def test_log_sucessfully(self, mock_get_accountId, mock_request_helper):
        """Test to check if the telemetry logging is successful"""
        MOCK_SESSION.boto_session.region_name = "us-west-2"
        mock_get_accountId.return_value = "testAccountId"
        _send_telemetry_request_sync("someStatus", "1", MOCK_SESSION)
        mock_request_helper.assert_called_with(
            "https://sm-pysdk-t-us-west-2.s3.us-west-2.amazonaws.com/"
            "telemetry?x-accountId=testAccountId&x-status=someStatus&x-feature=1",
            TELEMETRY_REQUEST_TIMEOUT,
        )

    @patch("sagemaker.core.telemetry.telemetry_logging._get_accountId")
    def test_log_handle_exception(self, mock_get_accountId):
        """Test to check if the exception is handled while logging telemetry"""
        mock_get_accountId.side_effect = Exception("Internal error")
        _send_telemetry_request_sync("someStatus", "1", MOCK_SESSION)
        self.assertRaises(Exception)

    @patch("sagemaker.core.telemetry.telemetry_logging._get_accountId")
    @patch("sagemaker.core.telemetry.telemetry_logging._get_region_or_default")
    def test_send_telemetry_request_success(self, mock_get_region, mock_get_accountId):
        """Test to check the _send_telemetry_request function with success status"""
        mock_get_accountId.return_value = "testAccountId"
        mock_get_region.return_value = "us-west-2"

        with patch(
            "sagemaker.core.telemetry.telemetry_logging._requests_helper"
        ) as mock_requests_helper:
            mock_requests_helper.return_value = None
            _send_telemetry_request_sync(1, [1, 2], MagicMock(), None, None, "extra_info")
            mock_requests_helper.assert_called_with(
                "https://sm-pysdk-t-us-west-2.s3.us-west-2.amazonaws.com/"
                "telemetry?x-accountId=testAccountId&x-status=1&x-feature=1,2&x-extra=extra_info",
                TELEMETRY_REQUEST_TIMEOUT,
            )

    @patch("sagemaker.core.telemetry.telemetry_logging._get_accountId")
    @patch("sagemaker.core.telemetry.telemetry_logging._get_region_or_default")
    def test_send_telemetry_request_failure(self, mock_get_region, mock_get_accountId):
        """Test to check the _send_telemetry_request function with failure status"""
        mock_get_accountId.return_value = "testAccountId"
        mock_get_region.return_value = "us-west-2"

        with patch(
            "sagemaker.core.telemetry.telemetry_logging._requests_helper"
        ) as mock_requests_helper:
            mock_requests_helper.return_value = None
            _send_telemetry_request_sync(
                0, [1, 2], MagicMock(), "failure_reason", "failure_type", "extra_info"
            )
            mock_requests_helper.assert_called_with(
                "https://sm-pysdk-t-us-west-2.s3.us-west-2.amazonaws.com/"
                "telemetry?x-accountId=testAccountId&x-status=0&x-feature=1,2"
                "&x-failureReason=failure_reason&x-failureType=failure_type&x-extra=extra_info",
                TELEMETRY_REQUEST_TIMEOUT,
            )

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_telemetry_emitter_decorator_no_call_when_disabled(
        self, mock_resolve_config, mock_send_telemetry_request
    ):
        """Test to check if the _telemetry_emitter decorator is not called when telemetry is disabled"""
        mock_resolve_config.return_value = True

        assert not mock_send_telemetry_request.called

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_telemetry_emitter_decorator_success(
        self, mock_resolve_config, mock_send_telemetry_request
    ):
        """Test to verify the _telemetry_emitter decorator with success status"""
        mock_resolve_config.return_value = False
        mock_local_client = LocalSagemakerClientMock()
        mock_local_client.sagemaker_session.endpoint_arn = MOCK_ENDPOINT_ARN
        mock_local_client.mock_create_model()
        app_type = process_studio_metadata_file()

        args = mock_send_telemetry_request.call_args.args
        latency = str(args[5]).split("latency=")[1]
        expected_extra_str = (
            f"{MOCK_FUNC_NAME}"
            f"&x-sdkVersion={SDK_VERSION}"
            f"&x-env={PYTHON_VERSION}"
            f"&x-sys={OS_NAME_VERSION}"
            f"&x-platform={app_type}"
            f"&x-endpointArn={MOCK_ENDPOINT_ARN}"
            f"&x-latency={latency}"
        )

        mock_send_telemetry_request.assert_called_once_with(
            1, [11, 12], MOCK_SESSION, None, None, expected_extra_str
        )

    @pytest.mark.skipif(not SAGEMAKER_SERVE_AVAILABLE, reason="Requires sagemaker-serve package")
    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_telemetry_emitter_decorator_handle_exception_success(
        self, mock_resolve_config, mock_send_telemetry_request
    ):
        """Test to verify the _telemetry_emitter decorator when function emits exception"""
        mock_resolve_config.return_value = False
        mock_local_client = LocalSagemakerClientMock()
        mock_local_client.sagemaker_session.endpoint_arn = MOCK_ENDPOINT_ARN
        app_type = process_studio_metadata_file()

        mock_exception = Mock()
        mock_exception_obj = MOCK_EXCEPTION
        mock_exception.side_effect = mock_exception_obj

        with self.assertRaises(ModelBuilderException) as _:
            mock_local_client.mock_create_model(mock_exception)

        args = mock_send_telemetry_request.call_args.args
        latency = str(args[5]).split("latency=")[1]
        expected_extra_str = (
            f"{MOCK_FUNC_NAME}"
            f"&x-sdkVersion={SDK_VERSION}"
            f"&x-env={PYTHON_VERSION}"
            f"&x-sys={OS_NAME_VERSION}"
            f"&x-platform={app_type}"
            f"&x-endpointArn={MOCK_ENDPOINT_ARN}"
            f"&x-latency={latency}"
        )

        mock_send_telemetry_request.assert_called_once_with(
            0,
            [11, 12],
            MOCK_SESSION,
            str(mock_exception_obj),
            mock_exception_obj.__class__.__name__,
            expected_extra_str,
        )

    def test_construct_url_with_failure_reason_and_extra_info(self):
        """Test to verify the _construct_url function with failure reason and extra info"""
        mock_accountId = "testAccountId"
        mock_status = 0
        mock_feature = "1,2"
        mock_failure_reason = str(MOCK_EXCEPTION)
        mock_failure_type = MOCK_EXCEPTION.__class__.__name__
        mock_extra_info = "mock_extra_info"
        mock_region = "us-west-2"

        resulted_url = _construct_url(
            accountId=mock_accountId,
            region=mock_region,
            status=mock_status,
            feature=mock_feature,
            failure_reason=mock_failure_reason,
            failure_type=mock_failure_type,
            extra_info=mock_extra_info,
        )

        expected_base_url = (
            f"https://sm-pysdk-t-{mock_region}.s3.{mock_region}.amazonaws.com/telemetry?"
            f"x-accountId={mock_accountId}"
            f"&x-status={mock_status}"
            f"&x-feature={mock_feature}"
            f"&x-failureReason={mock_failure_reason}"
            f"&x-failureType={mock_failure_type}"
            f"&x-extra={mock_extra_info}"
        )
        self.assertEqual(resulted_url, expected_base_url)

    @patch("sagemaker.core.telemetry.telemetry_logging.requests.get")
    def test_requests_helper_success(self, mock_requests_get):
        """Test to verify the _requests_helper function with success status"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests_get.return_value = mock_response
        url = "https://example.com"
        timeout = 10

        response = _requests_helper(url, timeout)

        # timeout must be a keyword argument: positionally it becomes `params`,
        # which leaves the request with no timeout at all.
        mock_requests_get.assert_called_once_with(url, timeout=timeout)
        self.assertEqual(response, mock_response)

    @patch("sagemaker.core.telemetry.telemetry_logging.requests.get")
    def test_requests_helper_exception(self, mock_requests_get):
        """Test to verify the _requests_helper function with exception"""
        mock_requests_get.side_effect = requests.exceptions.RequestException("Error making request")
        url = "https://example.com"
        timeout = 10

        response = _requests_helper(url, timeout)

        mock_requests_get.assert_called_once_with(url, timeout=timeout)
        self.assertIsNone(response)

    def test_get_accountId_success(self):
        """Test to verify the _get_accountId function with success status"""
        from sagemaker.core.helper.session_helper import Session

        boto_mock = MagicMock(name="boto_session")
        boto_mock.client("sts").get_caller_identity.return_value = {"Account": "testAccountId"}
        session = Session(boto_session=boto_mock)
        account_id = _get_accountId(session)

        self.assertEqual(account_id, "testAccountId")

    def test_get_accountId_exception(self):
        """Test to verify the _get_accountId function with exception"""
        from sagemaker.core.helper.session_helper import Session

        sts_client_mock = MagicMock()
        sts_client_mock.side_effect = Exception("Error creating STS client")
        boto_mock = MagicMock(name="boto_session")
        boto_mock.client("sts").get_caller_identity.return_value = sts_client_mock
        session = Session(boto_session=boto_mock)

        with pytest.raises(Exception) as exception:
            account_id = _get_accountId(session)
            assert account_id is None
            assert "Error creating STS client" in str(exception)

    def test_get_region_or_default_success(self):
        """Test to verify the _get_region_or_default function with success status"""
        mock_session = MagicMock()
        mock_session.boto_session = MagicMock(region_name="us-east-1")

        region = _get_region_or_default(mock_session)

        assert region == "us-east-1"

    def test_get_region_or_default_exception(self):
        """Test to verify the _get_region_or_default function with exception"""
        mock_session = MagicMock()
        mock_session.boto_session = MagicMock()
        mock_session.boto_session.region_name.side_effect = Exception("Error creating boto session")

        with pytest.raises(Exception) as exception:
            region = _get_region_or_default(mock_session)
            assert region == "us-west-2"
            assert "Error creating boto session" in str(exception)

    @patch.object(boto3.Session, "region_name", "us-west-2")
    def test_get_default_sagemaker_session(self):
        from sagemaker.core.helper.session_helper import Session

        sagemaker_session = _get_default_sagemaker_session()

        assert isinstance(sagemaker_session, Session) is True
        assert sagemaker_session.boto_session.region_name == "us-west-2"

    @patch.object(boto3.Session, "region_name", None)
    def test_get_default_sagemaker_session_with_no_region(self):
        with self.assertRaises(ValueError) as context:
            _get_default_sagemaker_session()

        assert "Must setup local AWS configuration with a region supported by SageMaker." in str(
            context.exception
        )

    @patch("sagemaker.core.telemetry.telemetry_logging._get_accountId")
    @patch("sagemaker.core.telemetry.telemetry_logging._get_region_or_default")
    def test_send_telemetry_request_valid_region(self, mock_get_region, mock_get_accountId):
        """Test to verify telemetry request is sent when region is valid"""
        mock_get_accountId.return_value = "testAccountId"
        mock_session = MagicMock()

        # Test with valid region
        mock_get_region.return_value = "us-east-1"
        with patch(
            "sagemaker.core.telemetry.telemetry_logging._requests_helper"
        ) as mock_requests_helper:
            _send_telemetry_request_sync(1, [1, 2], mock_session)
            # Assert telemetry request was sent
            mock_requests_helper.assert_called_once_with(
                "https://sm-pysdk-t-us-east-1.s3.us-east-1.amazonaws.com/telemetry?"
                "x-accountId=testAccountId&x-status=1&x-feature=1,2",
                TELEMETRY_REQUEST_TIMEOUT,
            )

    @patch("sagemaker.core.telemetry.telemetry_logging._get_accountId")
    @patch("sagemaker.core.telemetry.telemetry_logging._get_region_or_default")
    def test_send_telemetry_request_invalid_region(self, mock_get_region, mock_get_accountId):
        """Test to verify telemetry request is not sent when region is invalid"""
        mock_get_accountId.return_value = "testAccountId"
        mock_session = MagicMock()

        # Test with invalid region
        mock_get_region.return_value = "invalid-region"
        with patch(
            "sagemaker.core.telemetry.telemetry_logging._requests_helper"
        ) as mock_requests_helper:
            _send_telemetry_request_sync(1, [1, 2], mock_session)
            # Assert telemetry request was not sent
            mock_requests_helper.assert_not_called()

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_telemetry_emitter_with_created_by_env_var(
        self, mock_resolve_config, mock_send_telemetry_request
    ):
        """Test that x-createdBy is included when SAGEMAKER_PYSDK_CREATED_BY env var is set"""
        mock_resolve_config.return_value = False

        # Set environment variable
        os.environ[_CREATED_BY_ENV_VAR] = "awslabs/agent-plugins/sagemaker-ai"

        try:
            mock_local_client = LocalSagemakerClientMock()
            mock_local_client.mock_create_model()

            args = mock_send_telemetry_request.call_args.args
            extra_str = str(args[5])

            # Verify x-createdBy is in the extra string with URL encoding
            self.assertIn("x-createdBy=awslabs%2Fagent-plugins%2Fsagemaker-ai", extra_str)

            # Verify forward slashes are encoded as %2F
            self.assertNotIn("x-createdBy=awslabs/agent-plugins", extra_str)
        finally:
            # Clean up environment variable
            if _CREATED_BY_ENV_VAR in os.environ:
                del os.environ[_CREATED_BY_ENV_VAR]

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_telemetry_emitter_without_created_by_env_var(
        self, mock_resolve_config, mock_send_telemetry_request
    ):
        """Test that x-createdBy is NOT included when env var is not set"""
        mock_resolve_config.return_value = False

        # Ensure environment variable is not set
        if _CREATED_BY_ENV_VAR in os.environ:
            del os.environ[_CREATED_BY_ENV_VAR]

        mock_local_client = LocalSagemakerClientMock()
        mock_local_client.mock_create_model()

        args = mock_send_telemetry_request.call_args.args
        extra_str = str(args[5])

        # Verify x-createdBy is NOT in the extra string
        self.assertNotIn("x-createdBy", extra_str)

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_telemetry_emitter_created_by_with_special_chars(
        self, mock_resolve_config, mock_send_telemetry_request
    ):
        """Test that x-createdBy properly URL-encodes special characters"""
        mock_resolve_config.return_value = False

        # Set environment variable with special characters
        os.environ[_CREATED_BY_ENV_VAR] = "My App & Tools (v2.0)"

        try:
            mock_local_client = LocalSagemakerClientMock()
            mock_local_client.mock_create_model()

            args = mock_send_telemetry_request.call_args.args
            extra_str = str(args[5])

            # Verify special characters are URL-encoded
            self.assertIn("x-createdBy=My%20App%20%26%20Tools%20%28v2.0%29", extra_str)

            # Verify raw special characters are NOT in the URL
            self.assertNotIn("My App & Tools", extra_str)
            self.assertNotIn("(v2.0)", extra_str)
        finally:
            if _CREATED_BY_ENV_VAR in os.environ:
                del os.environ[_CREATED_BY_ENV_VAR]

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_telemetry_emitter_created_by_empty_string(
        self, mock_resolve_config, mock_send_telemetry_request
    ):
        """Test that x-createdBy is NOT included when env var is empty string"""
        mock_resolve_config.return_value = False

        # Set environment variable to empty string
        os.environ[_CREATED_BY_ENV_VAR] = ""

        try:
            mock_local_client = LocalSagemakerClientMock()
            mock_local_client.mock_create_model()

            args = mock_send_telemetry_request.call_args.args
            extra_str = str(args[5])

            # Verify x-createdBy is NOT added for empty string
            self.assertNotIn("x-createdBy", extra_str)
        finally:
            if _CREATED_BY_ENV_VAR in os.environ:
                del os.environ[_CREATED_BY_ENV_VAR]

    def test_construct_url_with_created_by(self):
        """Test URL construction includes x-createdBy in extra_info"""
        mock_accountId = "123456789012"
        mock_region = "us-west-2"
        mock_status = "1"
        mock_feature = "15"
        mock_extra_info = (
            "DataSet.create&x-sdkVersion=3.0&x-createdBy=awslabs%2Fagent-plugins%2Fsagemaker-ai"
        )

        url = _construct_url(
            accountId=mock_accountId,
            region=mock_region,
            status=mock_status,
            feature=mock_feature,
            failure_reason=None,
            failure_type=None,
            extra_info=mock_extra_info,
        )

        expected_url = (
            f"https://sm-pysdk-t-{mock_region}.s3.{mock_region}.amazonaws.com/telemetry?"
            f"x-accountId={mock_accountId}"
            f"&x-status={mock_status}"
            f"&x-feature={mock_feature}"
            f"&x-extra={mock_extra_info}"
        )

        self.assertEqual(url, expected_url)
        self.assertIn("x-createdBy=awslabs%2Fagent-plugins%2Fsagemaker-ai", url)


    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_telemetry_emitter_with_resource_arn(
        self, mock_resolve_config, mock_send_telemetry_request
    ):
        """Test that x-resourceArn is included when decorated function returns a TrainingJob."""
        mock_resolve_config.return_value = False

        mock_training_job = Mock()
        mock_training_job.__class__.__name__ = "TrainingJob"
        mock_training_job.training_job_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:training-job/my-job"
        )

        class TrainingJobReturningMock:
            def __init__(self):
                self.sagemaker_session = MOCK_SESSION

            @_telemetry_emitter(MOCK_FEATURE, MOCK_FUNC_NAME)
            def mock_train(self):
                return mock_training_job

        TrainingJobReturningMock().mock_train()

        args = mock_send_telemetry_request.call_args.args
        extra_str = str(args[5])
        self.assertIn(
            "x-resourceArn=arn:aws:sagemaker:us-west-2:123456789012:training-job/my-job",
            extra_str,
        )

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_telemetry_emitter_without_resource_arn(
        self, mock_resolve_config, mock_send_telemetry_request
    ):
        """Test that x-resourceArn is NOT included when response has no registered ARN."""
        mock_resolve_config.return_value = False

        mock_local_client = LocalSagemakerClientMock()
        mock_local_client.mock_create_model()

        args = mock_send_telemetry_request.call_args.args
        extra_str = str(args[5])
        self.assertNotIn("x-resourceArn", extra_str)

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_telemetry_emitter_appends_nova_sub_feature(
        self, mock_resolve_config, mock_send_telemetry_request
    ):
        """Test that MODEL_CUSTOMIZATION_NOVA (19) is appended when instance reports Nova model."""
        mock_resolve_config.return_value = False

        class NovaModelMock:
            def __init__(self):
                self.sagemaker_session = MOCK_SESSION

            def _is_nova_model_for_telemetry(self):
                return True

            @_telemetry_emitter(Feature.MODEL_CUSTOMIZATION, "NovaModelMock.train")
            def train(self):
                pass

        NovaModelMock().train()

        args = mock_send_telemetry_request.call_args.args
        feature_list = args[1]
        self.assertIn(15, feature_list)  # MODEL_CUSTOMIZATION
        self.assertIn(19, feature_list)  # MODEL_CUSTOMIZATION_NOVA
        self.assertNotIn(20, feature_list)  # MODEL_CUSTOMIZATION_OSS should NOT be present

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_telemetry_emitter_appends_oss_sub_feature(
        self, mock_resolve_config, mock_send_telemetry_request
    ):
        """Test that MODEL_CUSTOMIZATION_OSS (20) is appended when instance reports non-Nova model."""
        mock_resolve_config.return_value = False

        class OssModelMock:
            def __init__(self):
                self.sagemaker_session = MOCK_SESSION

            def _is_nova_model_for_telemetry(self):
                return False

            @_telemetry_emitter(Feature.MODEL_CUSTOMIZATION, "OssModelMock.train")
            def train(self):
                pass

        OssModelMock().train()

        args = mock_send_telemetry_request.call_args.args
        feature_list = args[1]
        self.assertIn(15, feature_list)  # MODEL_CUSTOMIZATION
        self.assertIn(20, feature_list)  # MODEL_CUSTOMIZATION_OSS
        self.assertNotIn(19, feature_list)  # MODEL_CUSTOMIZATION_NOVA should NOT be present

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_telemetry_emitter_no_sub_feature_without_detection_method(
        self, mock_resolve_config, mock_send_telemetry_request
    ):
        """Test that no NOVA/OSS sub-feature is appended when instance lacks detection method."""
        mock_resolve_config.return_value = False

        class NoDetectionMock:
            def __init__(self):
                self.sagemaker_session = MOCK_SESSION

            @_telemetry_emitter(Feature.MODEL_CUSTOMIZATION, "NoDetectionMock.do_work")
            def do_work(self):
                pass

        NoDetectionMock().do_work()

        args = mock_send_telemetry_request.call_args.args
        feature_list = args[1]
        self.assertIn(15, feature_list)  # MODEL_CUSTOMIZATION
        self.assertNotIn(19, feature_list)  # No NOVA
        self.assertNotIn(20, feature_list)  # No OSS

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_telemetry_emitter_handles_detection_method_exception(
        self, mock_resolve_config, mock_send_telemetry_request
    ):
        """Test that telemetry still works when _is_nova_model_for_telemetry raises an exception."""
        mock_resolve_config.return_value = False

        class BrokenDetectionMock:
            def __init__(self):
                self.sagemaker_session = MOCK_SESSION

            def _is_nova_model_for_telemetry(self):
                raise RuntimeError("detection failed")

            @_telemetry_emitter(Feature.MODEL_CUSTOMIZATION, "BrokenDetectionMock.train")
            def train(self):
                pass

        BrokenDetectionMock().train()

        args = mock_send_telemetry_request.call_args.args
        feature_list = args[1]
        self.assertIn(15, feature_list)  # MODEL_CUSTOMIZATION still present
        self.assertNotIn(19, feature_list)  # No NOVA (detection failed gracefully)
        self.assertNotIn(20, feature_list)  # No OSS

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_telemetry_opt_out_message_shown_only_once(
        self, mock_resolve_config, mock_send_telemetry_request
    ):
        """Test that the telemetry opt-out INFO message is logged only once per process."""
        import sagemaker.core.telemetry.telemetry_logging as telemetry_module

        mock_resolve_config.return_value = False
        # Reset the flag to simulate a fresh process
        telemetry_module._telemetry_msg_shown = False

        mock_local_client = LocalSagemakerClientMock()

        with patch.object(telemetry_module.logger, "info") as mock_logger_info:
            mock_local_client.mock_create_model()
            mock_local_client.mock_create_model()
            mock_local_client.mock_create_model()

            info_calls = [
                call for call in mock_logger_info.call_args_list
                if "telemetry" in str(call).lower() and "opt out" in str(call).lower()
            ]
            self.assertEqual(len(info_calls), 1, "Telemetry opt-out message should be logged exactly once")

        # Reset the flag for other tests
        telemetry_module._telemetry_msg_shown = False

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_telemetry_opt_out_message_not_shown_when_opted_out(
        self, mock_resolve_config, mock_send_telemetry_request
    ):
        """Test that the telemetry opt-out INFO message is not shown when user has opted out."""
        import sagemaker.core.telemetry.telemetry_logging as telemetry_module

        mock_resolve_config.return_value = True  # opted out
        # Reset the flag to simulate a fresh process
        telemetry_module._telemetry_msg_shown = False

        mock_local_client = LocalSagemakerClientMock()

        with patch.object(telemetry_module.logger, "info") as mock_logger_info:
            mock_local_client.mock_create_model()

            info_calls = [
                call for call in mock_logger_info.call_args_list
                if "telemetry" in str(call).lower() and "opt out" in str(call).lower()
            ]
            self.assertEqual(len(info_calls), 0, "Telemetry opt-out message should not appear when opted out")

        # Reset the flag for other tests
        telemetry_module._telemetry_msg_shown = False


class TestRequestsHelperTimeout(unittest.TestCase):
    """The telemetry GET must actually carry a timeout.

    `requests.get(url, params=None, **kwargs)` takes `params` second, so passing
    the timeout positionally appended it to the query string and left the
    request with no timeout, letting an unreachable telemetry endpoint block the
    caller indefinitely.
    """

    @patch("sagemaker.core.telemetry.telemetry_logging.requests.get")
    def test_timeout_passed_as_keyword_not_params(self, mock_requests_get):
        _requests_helper("https://example.com/telemetry?x-status=1", 2)

        _, kwargs = mock_requests_get.call_args
        self.assertEqual(kwargs["timeout"], 2)
        self.assertNotIn("params", kwargs)

    def test_timeout_reaches_prepared_request_not_the_url(self):
        """Guard against the regression at the layer where it was observable."""
        captured = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return None

        target = "sagemaker.core.telemetry.telemetry_logging.requests.get"
        with patch(target, side_effect=fake_get):
            _requests_helper("https://example.com/telemetry?x-status=1", 2)

        # The old code produced a URL ending in "&2" and no timeout kwarg.
        self.assertFalse(captured["url"].endswith("&2"))
        self.assertEqual(captured["kwargs"], {"timeout": 2})


class TestTelemetryIsNonBlocking(unittest.TestCase):
    """Telemetry must never add latency to the SDK call that triggered it.

    Regression guard: a Feature Store ingest returned in under a second
    server-side but the notebook cell took ~47 minutes, because each telemetry
    emission blocked on an endpoint the caller's VPC had no route to.
    """

    def setUp(self):
        import sagemaker.core.telemetry.telemetry_logging as telemetry_module

        self.telemetry_module = telemetry_module
        telemetry_module._in_flight_telemetry_requests = 0

    def tearDown(self):
        self.telemetry_module._in_flight_telemetry_requests = 0

    def test_send_returns_before_the_request_completes(self):
        release = threading.Event()
        entered = threading.Event()

        def blocking_send(*args, **kwargs):
            entered.set()
            release.wait(timeout=10)

        with patch.object(
            self.telemetry_module, "_send_telemetry_request_sync", side_effect=blocking_send
        ):
            start = perf_counter()
            thread = _send_telemetry_request(1, [1], MagicMock())
            elapsed = perf_counter() - start

            try:
                self.assertLess(elapsed, 1, "_send_telemetry_request blocked on the network call")
                self.assertTrue(entered.wait(timeout=5))
            finally:
                release.set()
                thread.join(timeout=5)

    def test_send_runs_on_a_daemon_thread(self):
        """Daemon threads are killed at exit, so a pending send cannot hang shutdown."""
        with patch.object(self.telemetry_module, "_send_telemetry_request_sync"):
            thread = _send_telemetry_request(1, [1], MagicMock())
            self.assertTrue(thread.daemon)
            thread.join(timeout=5)

    def test_send_forwards_all_arguments(self):
        session = MagicMock()
        with patch.object(self.telemetry_module, "_send_telemetry_request_sync") as mock_sync:
            thread = _send_telemetry_request(
                0, [1, 2], session, "failure_reason", "failure_type", "extra_info"
            )
            thread.join(timeout=5)

        mock_sync.assert_called_once_with(
            0, [1, 2], session, "failure_reason", "failure_type", "extra_info"
        )

    def test_thread_swallows_exceptions_and_releases_its_slot(self):
        """An exception in the thread has no caller to catch it, so it must not escape."""
        with patch.object(
            self.telemetry_module,
            "_send_telemetry_request_sync",
            side_effect=RuntimeError("boom"),
        ):
            thread = _send_telemetry_request(1, [1], MagicMock())
            thread.join(timeout=5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(self.telemetry_module._in_flight_telemetry_requests, 0)

    def test_events_are_dropped_when_too_many_are_in_flight(self):
        release = threading.Event()
        started = []

        def blocking_send(*args, **kwargs):
            started.append(1)
            release.wait(timeout=10)

        max_in_flight = self.telemetry_module.MAX_IN_FLIGHT_TELEMETRY_REQUESTS
        with patch.object(
            self.telemetry_module, "_send_telemetry_request_sync", side_effect=blocking_send
        ):
            threads = [_send_telemetry_request(1, [1], MagicMock()) for _ in range(max_in_flight)]
            try:
                self.assertTrue(all(t is not None for t in threads))
                # One more than the cap allows is dropped rather than spawning
                # an unbounded number of threads.
                self.assertIsNone(_send_telemetry_request(1, [1], MagicMock()))
            finally:
                release.set()
                for t in threads:
                    t.join(timeout=5)

        # Counter is released once the sends finish, so later events go through.
        self.assertEqual(self.telemetry_module._in_flight_telemetry_requests, 0)

    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_decorated_function_returns_without_waiting_for_telemetry(self, mock_resolve_config):
        mock_resolve_config.return_value = False
        release = threading.Event()

        def blocking_send(*args, **kwargs):
            release.wait(timeout=10)

        with patch.object(
            self.telemetry_module, "_send_telemetry_request_sync", side_effect=blocking_send
        ):
            try:
                start = perf_counter()
                LocalSagemakerClientMock().mock_create_model()
                elapsed = perf_counter() - start
                self.assertLess(elapsed, 1, "the decorated call waited on the telemetry request")
            finally:
                release.set()


class TestDefaultSessionRegion(unittest.TestCase):
    """The synthesized fallback session must use the caller's own region.

    Module-level functions such as `ingest_dataframe` have no session, so the
    decorator builds one. Hardcoding us-west-2 pointed telemetry at a region the
    caller may have no network route to.
    """

    @patch("sagemaker.core.telemetry.telemetry_logging.Session")
    @patch("sagemaker.core.telemetry.telemetry_logging.boto3.Session")
    def test_uses_region_resolved_from_environment(self, mock_boto_session, mock_session):
        mock_boto_session.return_value.region_name = "ca-central-1"

        _get_default_sagemaker_session()

        # Called with no region_name so boto3 resolves it from the environment
        # or the active profile, rather than being pinned to us-west-2.
        mock_boto_session.assert_called_once_with()
        mock_session.assert_called_once_with(boto_session=mock_boto_session.return_value)

    @patch("sagemaker.core.telemetry.telemetry_logging.Session")
    @patch("sagemaker.core.telemetry.telemetry_logging.boto3.Session")
    def test_falls_back_to_default_region_when_none_resolved(self, mock_boto_session, mock_session):
        mock_boto_session.return_value.region_name = None

        _get_default_sagemaker_session()

        # Session requires a region, so the default is still the last resort.
        self.assertEqual(
            mock_boto_session.call_args_list[-1],
            unittest.mock.call(region_name=DEFAULT_AWS_REGION),
        )
