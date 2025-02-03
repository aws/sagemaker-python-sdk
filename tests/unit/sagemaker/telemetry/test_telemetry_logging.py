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
import unittest
import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
import boto3
import sagemaker
from sagemaker.telemetry.constants import Feature
from sagemaker.telemetry.telemetry_logging import (
    _send_telemetry_request,
    _telemetry_emitter,
    _construct_url,
    _get_accountId,
    _requests_helper,
    _get_region_or_default,
    _get_default_sagemaker_session,
    OS_NAME_VERSION,
    PYTHON_VERSION,
)
from sagemaker.user_agent import SDK_VERSION, process_studio_metadata_file
from sagemaker.serve.utils.exceptions import ModelBuilderException, LocalModelOutOfMemoryException

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
    @patch("sagemaker.telemetry.telemetry_logging._requests_helper")
    @patch("sagemaker.telemetry.telemetry_logging._get_accountId")
    def test_log_sucessfully(self, mock_get_accountId, mock_request_helper):
        """Test to check if the telemetry logging is successful"""
        MOCK_SESSION.boto_session.region_name = "us-west-2"
        mock_get_accountId.return_value = "testAccountId"
        _send_telemetry_request("someStatus", "1", MOCK_SESSION)
        mock_request_helper.assert_called_with(
            "https://sm-pysdk-t-us-west-2.s3.us-west-2.amazonaws.com/"
            "telemetry?x-accountId=testAccountId&x-status=someStatus&x-feature=1",
            2,
        )

    @patch("sagemaker.telemetry.telemetry_logging._get_accountId")
    def test_log_handle_exception(self, mock_get_accountId):
        """Test to check if the exception is handled while logging telemetry"""
        mock_get_accountId.side_effect = Exception("Internal error")
        _send_telemetry_request("someStatus", "1", MOCK_SESSION)
        self.assertRaises(Exception)

    @patch("sagemaker.telemetry.telemetry_logging._get_accountId")
    @patch("sagemaker.telemetry.telemetry_logging._get_region_or_default")
    def test_send_telemetry_request_success(self, mock_get_region, mock_get_accountId):
        """Test to check the _send_telemetry_request function with success status"""
        mock_get_accountId.return_value = "testAccountId"
        mock_get_region.return_value = "us-west-2"

        with patch(
            "sagemaker.telemetry.telemetry_logging._requests_helper"
        ) as mock_requests_helper:
            mock_requests_helper.return_value = None
            _send_telemetry_request(1, [1, 2], MagicMock(), None, None, "extra_info")
            mock_requests_helper.assert_called_with(
                "https://sm-pysdk-t-us-west-2.s3.us-west-2.amazonaws.com/"
                "telemetry?x-accountId=testAccountId&x-status=1&x-feature=1,2&x-extra=extra_info",
                2,
            )

    @patch("sagemaker.telemetry.telemetry_logging._get_accountId")
    @patch("sagemaker.telemetry.telemetry_logging._get_region_or_default")
    def test_send_telemetry_request_failure(self, mock_get_region, mock_get_accountId):
        """Test to check the _send_telemetry_request function with failure status"""
        mock_get_accountId.return_value = "testAccountId"
        mock_get_region.return_value = "us-west-2"

        with patch(
            "sagemaker.telemetry.telemetry_logging._requests_helper"
        ) as mock_requests_helper:
            mock_requests_helper.return_value = None
            _send_telemetry_request(
                0, [1, 2], MagicMock(), "failure_reason", "failure_type", "extra_info"
            )
            mock_requests_helper.assert_called_with(
                "https://sm-pysdk-t-us-west-2.s3.us-west-2.amazonaws.com/"
                "telemetry?x-accountId=testAccountId&x-status=0&x-feature=1,2"
                "&x-failureReason=failure_reason&x-failureType=failure_type&x-extra=extra_info",
                2,
            )

    @patch("sagemaker.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.telemetry.telemetry_logging.resolve_value_from_config")
    def test_telemetry_emitter_decorator_no_call_when_disabled(
        self, mock_resolve_config, mock_send_telemetry_request
    ):
        """Test to check if the _telemetry_emitter decorator is not called when telemetry is disabled"""
        mock_resolve_config.return_value = True

        assert not mock_send_telemetry_request.called

    @patch("sagemaker.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.telemetry.telemetry_logging.resolve_value_from_config")
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
            1, [1, 2], MOCK_SESSION, None, None, expected_extra_str
        )

    @patch("sagemaker.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.telemetry.telemetry_logging.resolve_value_from_config")
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
            [1, 2],
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

    @patch("sagemaker.telemetry.telemetry_logging.requests.get")
    def test_requests_helper_success(self, mock_requests_get):
        """Test to verify the _requests_helper function with success status"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests_get.return_value = mock_response
        url = "https://example.com"
        timeout = 10

        response = _requests_helper(url, timeout)

        mock_requests_get.assert_called_once_with(url, timeout)
        self.assertEqual(response, mock_response)

    @patch("sagemaker.telemetry.telemetry_logging.requests.get")
    def test_requests_helper_exception(self, mock_requests_get):
        """Test to verify the _requests_helper function with exception"""
        mock_requests_get.side_effect = requests.exceptions.RequestException("Error making request")
        url = "https://example.com"
        timeout = 10

        response = _requests_helper(url, timeout)

        mock_requests_get.assert_called_once_with(url, timeout)
        self.assertIsNone(response)

    def test_get_accountId_success(self):
        """Test to verify the _get_accountId function with success status"""
        boto_mock = MagicMock(name="boto_session")
        boto_mock.client("sts").get_caller_identity.return_value = {"Account": "testAccountId"}
        session = sagemaker.Session(boto_session=boto_mock)
        account_id = _get_accountId(session)

        self.assertEqual(account_id, "testAccountId")

    def test_get_accountId_exception(self):
        """Test to verify the _get_accountId function with exception"""
        sts_client_mock = MagicMock()
        sts_client_mock.side_effect = Exception("Error creating STS client")
        boto_mock = MagicMock(name="boto_session")
        boto_mock.client("sts").get_caller_identity.return_value = sts_client_mock
        session = sagemaker.Session(boto_session=boto_mock)

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
        sagemaker_session = _get_default_sagemaker_session()

        assert isinstance(sagemaker_session, sagemaker.Session) is True
        assert sagemaker_session.boto_session.region_name == "us-west-2"

    @patch.object(boto3.Session, "region_name", None)
    def test_get_default_sagemaker_session_with_no_region(self):
        with self.assertRaises(ValueError) as context:
            _get_default_sagemaker_session()

        assert "Must setup local AWS configuration with a region supported by SageMaker." in str(
            context.exception
        )

    @patch("sagemaker.telemetry.telemetry_logging._get_accountId")
    @patch("sagemaker.telemetry.telemetry_logging._get_region_or_default")
    def test_send_telemetry_request_valid_region(self, mock_get_region, mock_get_accountId):
        """Test to verify telemetry request is sent when region is valid"""
        mock_get_accountId.return_value = "testAccountId"
        mock_session = MagicMock()

        # Test with valid region
        mock_get_region.return_value = "us-east-1"
        with patch(
            "sagemaker.telemetry.telemetry_logging._requests_helper"
        ) as mock_requests_helper:
            _send_telemetry_request(1, [1, 2], mock_session)
            # Assert telemetry request was sent
            mock_requests_helper.assert_called_once_with(
                "https://sm-pysdk-t-us-east-1.s3.us-east-1.amazonaws.com/telemetry?"
                "x-accountId=testAccountId&x-status=1&x-feature=1,2",
                2,
            )

    @patch("sagemaker.telemetry.telemetry_logging._get_accountId")
    @patch("sagemaker.telemetry.telemetry_logging._get_region_or_default")
    def test_send_telemetry_request_invalid_region(self, mock_get_region, mock_get_accountId):
        """Test to verify telemetry request is not sent when region is invalid"""
        mock_get_accountId.return_value = "testAccountId"
        mock_session = MagicMock()

        # Test with invalid region
        mock_get_region.return_value = "invalid-region"
        with patch(
            "sagemaker.telemetry.telemetry_logging._requests_helper"
        ) as mock_requests_helper:
            _send_telemetry_request(1, [1, 2], mock_session)
            # Assert telemetry request was not sent
            mock_requests_helper.assert_not_called()
