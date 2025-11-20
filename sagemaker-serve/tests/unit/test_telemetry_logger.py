"""
Unit tests for sagemaker.serve.utils.telemetry_logger module.

Tests telemetry collection and logging functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from time import perf_counter

from sagemaker.serve.utils.telemetry_logger import (
    _capture_telemetry,
    _send_telemetry,
    _construct_url,
    _requests_helper,
    _get_accountId,
    _get_region_or_default,
    _get_image_uri_option,
    MODE_TO_CODE,
    MODEL_SERVER_TO_CODE,
    MLFLOW_MODEL_PATH_CODE,
    MODEL_HUB_TO_CODE,
    SD_DRAFT_MODEL_SOURCE_TO_CODE,
)
from sagemaker.serve.utils.types import ModelServer, ImageUriOption, ModelHub
from sagemaker.serve.mode.function_pointers import Mode


class TestTelemetryConstants(unittest.TestCase):
    """Test telemetry constant mappings."""

    def test_mode_to_code_mapping(self):
        """Test MODE_TO_CODE has correct mappings."""
        self.assertEqual(MODE_TO_CODE[str(Mode.IN_PROCESS)], 1)
        self.assertEqual(MODE_TO_CODE[str(Mode.LOCAL_CONTAINER)], 2)
        self.assertEqual(MODE_TO_CODE[str(Mode.SAGEMAKER_ENDPOINT)], 3)

    def test_model_server_to_code_mapping(self):
        """Test MODEL_SERVER_TO_CODE has correct mappings."""
        self.assertEqual(MODEL_SERVER_TO_CODE[str(ModelServer.TORCHSERVE)], 1)
        self.assertEqual(MODEL_SERVER_TO_CODE[str(ModelServer.MMS)], 2)
        self.assertEqual(MODEL_SERVER_TO_CODE[str(ModelServer.TENSORFLOW_SERVING)], 3)
        self.assertEqual(MODEL_SERVER_TO_CODE[str(ModelServer.DJL_SERVING)], 4)
        self.assertEqual(MODEL_SERVER_TO_CODE[str(ModelServer.TRITON)], 5)
        self.assertEqual(MODEL_SERVER_TO_CODE[str(ModelServer.TGI)], 6)
        self.assertEqual(MODEL_SERVER_TO_CODE[str(ModelServer.TEI)], 7)
        self.assertEqual(MODEL_SERVER_TO_CODE[str(ModelServer.SMD)], 8)

    def test_model_hub_to_code_mapping(self):
        """Test MODEL_HUB_TO_CODE has correct mappings."""
        self.assertEqual(MODEL_HUB_TO_CODE[str(ModelHub.JUMPSTART)], 1)
        self.assertEqual(MODEL_HUB_TO_CODE[str(ModelHub.HUGGINGFACE)], 2)


class TestConstructUrl(unittest.TestCase):
    """Test _construct_url function."""

    def test_construct_url_basic(self):
        """Test constructing URL with basic parameters."""
        url = _construct_url(
            accountId="123456789012",
            mode="3",
            status="1",
            failure_reason=None,
            failure_type=None,
            extra_info=None,
            region="us-west-2"
        )
        
        self.assertIn("https://dev-exp-t-us-west-2.s3.us-west-2.amazonaws.com/telemetry", url)
        self.assertIn("x-accountId=123456789012", url)
        self.assertIn("x-mode=3", url)
        self.assertIn("x-status=1", url)

    def test_construct_url_with_failure(self):
        """Test constructing URL with failure information."""
        url = _construct_url(
            accountId="123456789012",
            mode="3",
            status="0",
            failure_reason="Test error",
            failure_type="ValueError",
            extra_info=None,
            region="us-east-1"
        )
        
        self.assertIn("x-status=0", url)
        self.assertIn("x-failureReason=Test error", url)
        self.assertIn("x-failureType=ValueError", url)

    def test_construct_url_with_extra_info(self):
        """Test constructing URL with extra information."""
        url = _construct_url(
            accountId="123456789012",
            mode="3",
            status="1",
            failure_reason=None,
            failure_type=None,
            extra_info="build&x-modelServer=1",
            region="us-west-2"
        )
        
        self.assertIn("x-extra=build&x-modelServer=1", url)

    def test_construct_url_different_regions(self):
        """Test constructing URL for different regions."""
        regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"]
        
        for region in regions:
            url = _construct_url(
                accountId="123456789012",
                mode="3",
                status="1",
                failure_reason=None,
                failure_type=None,
                extra_info=None,
                region=region
            )
            
            self.assertIn(f"dev-exp-t-{region}.s3.{region}.amazonaws.com", url)


class TestRequestsHelper(unittest.TestCase):
    """Test _requests_helper function."""

    @patch('sagemaker.serve.utils.telemetry_logger.requests.get')
    def test_requests_helper_success(self, mock_get):
        """Test successful request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = _requests_helper("https://example.com", 2)
        
        self.assertEqual(result, mock_response)
        mock_get.assert_called_once_with("https://example.com", 2)

    @patch('sagemaker.serve.utils.telemetry_logger.requests.get')
    def test_requests_helper_exception(self, mock_get):
        """Test request with exception."""
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")
        
        result = _requests_helper("https://example.com", 2)
        
        self.assertIsNone(result)


class TestGetAccountId(unittest.TestCase):
    """Test _get_accountId function."""

    def test_get_account_id_success(self):
        """Test getting account ID successfully."""
        mock_session = Mock()
        mock_sts = Mock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_session.boto_session.client.return_value = mock_sts
        
        account_id = _get_accountId(mock_session)
        
        self.assertEqual(account_id, "123456789012")
        mock_session.boto_session.client.assert_called_once_with("sts")

    def test_get_account_id_exception(self):
        """Test getting account ID with exception."""
        mock_session = Mock()
        mock_session.boto_session.client.side_effect = Exception("STS error")
        
        account_id = _get_accountId(mock_session)
        
        self.assertIsNone(account_id)


class TestGetRegionOrDefault(unittest.TestCase):
    """Test _get_region_or_default function."""

    def test_get_region_success(self):
        """Test getting region successfully."""
        mock_session = Mock()
        mock_session.boto_session.region_name = "us-east-1"
        
        region = _get_region_or_default(mock_session)
        
        self.assertEqual(region, "us-east-1")

    def test_get_region_exception_returns_default(self):
        """Test getting region with exception returns default."""
        mock_session = Mock()
        mock_session.boto_session.region_name = None
        # Simulate exception by making region_name raise
        type(mock_session.boto_session).region_name = property(
            lambda self: (_ for _ in ()).throw(Exception("No region"))
        )
        
        region = _get_region_or_default(mock_session)
        
        self.assertEqual(region, "us-west-2")


class TestGetImageUriOption(unittest.TestCase):
    """Test _get_image_uri_option function."""

    def test_get_image_uri_option_default(self):
        """Test getting image URI option for default image."""
        result = _get_image_uri_option(
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch:latest",
            is_custom_image=False
        )
        
        self.assertEqual(result, ImageUriOption.DEFAULT_IMAGE.value)

    @patch('sagemaker.serve.utils.telemetry_logger.is_1p_image_uri')
    def test_get_image_uri_option_custom_1p(self, mock_is_1p):
        """Test getting image URI option for custom 1P image."""
        mock_is_1p.return_value = True
        
        result = _get_image_uri_option(
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/custom:latest",
            is_custom_image=True
        )
        
        self.assertEqual(result, ImageUriOption.CUSTOM_1P_IMAGE.value)

    @patch('sagemaker.serve.utils.telemetry_logger.is_1p_image_uri')
    def test_get_image_uri_option_custom(self, mock_is_1p):
        """Test getting image URI option for custom image."""
        mock_is_1p.return_value = False
        
        result = _get_image_uri_option(
            "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-custom:latest",
            is_custom_image=True
        )
        
        self.assertEqual(result, ImageUriOption.CUSTOM_IMAGE.value)


class TestSendTelemetry(unittest.TestCase):
    """Test _send_telemetry function."""

    @patch('sagemaker.serve.utils.telemetry_logger._requests_helper')
    @patch('sagemaker.serve.utils.telemetry_logger._construct_url')
    @patch('sagemaker.serve.utils.telemetry_logger._get_region_or_default')
    @patch('sagemaker.serve.utils.telemetry_logger._get_accountId')
    def test_send_telemetry_success(
        self, mock_get_account, mock_get_region, mock_construct_url, mock_requests
    ):
        """Test sending telemetry successfully."""
        mock_get_account.return_value = "123456789012"
        mock_get_region.return_value = "us-west-2"
        mock_construct_url.return_value = "https://example.com/telemetry"
        mock_requests.return_value = Mock(status_code=200)
        
        mock_session = Mock()
        
        _send_telemetry(
            status="1",
            mode=3,
            session=mock_session,
            failure_reason=None,
            failure_type=None,
            extra_info="build"
        )
        
        mock_get_account.assert_called_once_with(mock_session)
        mock_get_region.assert_called_once_with(mock_session)
        mock_construct_url.assert_called_once()
        mock_requests.assert_called_once_with("https://example.com/telemetry", 2)

    @patch('sagemaker.serve.utils.telemetry_logger._get_accountId')
    def test_send_telemetry_exception_handled(self, mock_get_account):
        """Test that exceptions in send_telemetry are handled gracefully."""
        mock_get_account.side_effect = Exception("Network error")
        
        mock_session = Mock()
        
        # Should not raise exception
        _send_telemetry(
            status="1",
            mode=3,
            session=mock_session
        )


class TestCaptureTelemetryDecorator(unittest.TestCase):
    """Test _capture_telemetry decorator."""

    def test_capture_telemetry_success(self):
        """Test decorator with successful function execution."""
        @_capture_telemetry("test_func")
        def test_function(self):
            return "success"
        
        mock_self = Mock()
        mock_self.model_server = ModelServer.TORCHSERVE
        mock_self.image_uri = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch:latest"
        mock_self.mode = Mode.SAGEMAKER_ENDPOINT
        mock_self.sagemaker_session = Mock()
        mock_self.sagemaker_session.endpoint_arn = None
        mock_self.serve_settings = Mock()
        mock_self.serve_settings.telemetry_opt_out = True  # Opt out to avoid actual telemetry
        mock_self._is_custom_image_uri = False
        mock_self._is_mlflow_model = False
        mock_self.model_hub = False
        
        result = test_function(mock_self)
        
        self.assertEqual(result, "success")

    def test_capture_telemetry_with_exception(self):
        """Test decorator with function that raises generic exception."""
        @_capture_telemetry("test_func")
        def test_function(self):
            raise ValueError("Test error")
        
        mock_self = Mock()
        mock_self.model_server = ModelServer.TORCHSERVE
        mock_self.image_uri = None
        mock_self.mode = Mode.SAGEMAKER_ENDPOINT
        mock_self.sagemaker_session = Mock()
        mock_self.serve_settings = Mock()
        mock_self.serve_settings.telemetry_opt_out = True
        
        with self.assertRaises(ValueError):
            test_function(mock_self)

    @patch('sagemaker.serve.utils.telemetry_logger._send_telemetry')
    def test_capture_telemetry_sends_metrics(self, mock_send):
        """Test that decorator sends telemetry when not opted out."""
        @_capture_telemetry("test_func")
        def test_function(self):
            return "success"
        
        mock_self = Mock()
        mock_self.model_server = ModelServer.TORCHSERVE
        mock_self.image_uri = None
        mock_self.mode = Mode.SAGEMAKER_ENDPOINT
        mock_self.sagemaker_session = Mock()
        mock_self.serve_settings = Mock()
        mock_self.serve_settings.telemetry_opt_out = False  # Not opted out
        mock_self._is_custom_image_uri = False
        mock_self._is_mlflow_model = False
        mock_self.model_hub = False
        
        result = test_function(mock_self)
        
        self.assertEqual(result, "success")
        mock_send.assert_called_once()


if __name__ == '__main__':
    unittest.main()
