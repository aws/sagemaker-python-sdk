"""Additional unit tests for telemetry_logger.py to increase coverage."""

import unittest
from unittest.mock import Mock, patch


class TestConstructUrl(unittest.TestCase):
    """Test _construct_url function."""

    def test_construct_url_basic(self):
        """Test basic URL construction."""
        from sagemaker.serve.utils.telemetry_logger import _construct_url
        
        url = _construct_url("123456", "1", "1", None, None, None, "us-west-2")
        
        self.assertIn("x-accountId=123456", url)
        self.assertIn("x-mode=1", url)
        self.assertIn("x-status=1", url)

    def test_construct_url_with_failure(self):
        """Test URL construction with failure info."""
        from sagemaker.serve.utils.telemetry_logger import _construct_url
        
        url = _construct_url("123456", "1", "0", "Error message", "ValueError", None, "us-west-2")
        
        self.assertIn("x-failureReason=Error message", url)
        self.assertIn("x-failureType=ValueError", url)

    def test_construct_url_with_extra_info(self):
        """Test URL construction with extra info."""
        from sagemaker.serve.utils.telemetry_logger import _construct_url
        
        url = _construct_url("123456", "1", "1", None, None, "extra=data", "us-west-2")
        
        self.assertIn("x-extra=extra=data", url)


class TestRequestsHelper(unittest.TestCase):
    """Test _requests_helper function."""

    @patch('requests.get')
    def test_requests_helper_success(self, mock_get):
        """Test successful request."""
        from sagemaker.serve.utils.telemetry_logger import _requests_helper
        
        mock_response = Mock()
        mock_get.return_value = mock_response
        
        result = _requests_helper("http://example.com", 2)
        
        self.assertEqual(result, mock_response)

    @patch('requests.get')
    def test_requests_helper_exception(self, mock_get):
        """Test request with exception."""
        from sagemaker.serve.utils.telemetry_logger import _requests_helper
        import requests
        
        mock_get.side_effect = requests.exceptions.RequestException("Timeout")
        
        result = _requests_helper("http://example.com", 2)
        
        self.assertIsNone(result)


class TestGetAccountId(unittest.TestCase):
    """Test _get_accountId function."""

    def test_get_account_id_success(self):
        """Test successful account ID retrieval."""
        from sagemaker.serve.utils.telemetry_logger import _get_accountId
        
        mock_session = Mock()
        mock_sts = Mock()
        mock_session.boto_session.client.return_value = mock_sts
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        
        result = _get_accountId(mock_session)
        
        self.assertEqual(result, "123456789012")

    def test_get_account_id_exception(self):
        """Test account ID retrieval with exception."""
        from sagemaker.serve.utils.telemetry_logger import _get_accountId
        
        mock_session = Mock()
        mock_session.boto_session.client.side_effect = Exception("Error")
        
        result = _get_accountId(mock_session)
        
        self.assertIsNone(result)


class TestGetRegionOrDefault(unittest.TestCase):
    """Test _get_region_or_default function."""

    def test_get_region_success(self):
        """Test successful region retrieval."""
        from sagemaker.serve.utils.telemetry_logger import _get_region_or_default
        
        mock_session = Mock()
        mock_session.boto_session.region_name = "us-east-1"
        
        result = _get_region_or_default(mock_session)
        
        self.assertEqual(result, "us-east-1")

    def test_get_region_exception(self):
        """Test region retrieval with exception."""
        from sagemaker.serve.utils.telemetry_logger import _get_region_or_default
        
        mock_session = Mock()
        type(mock_session.boto_session).region_name = property(lambda self: (_ for _ in ()).throw(Exception("Error")))
        
        result = _get_region_or_default(mock_session)
        
        self.assertEqual(result, "us-west-2")


class TestGetImageUriOption(unittest.TestCase):
    """Test _get_image_uri_option function."""

    def test_get_image_uri_option_default(self):
        """Test default image option."""
        from sagemaker.serve.utils.telemetry_logger import _get_image_uri_option
        from sagemaker.serve.utils.types import ImageUriOption
        
        result = _get_image_uri_option("some-image:latest", False)
        
        self.assertEqual(result, ImageUriOption.DEFAULT_IMAGE.value)

    @patch('sagemaker.serve.utils.telemetry_logger.is_1p_image_uri')
    def test_get_image_uri_option_custom_1p(self, mock_is_1p):
        """Test custom 1P image option."""
        from sagemaker.serve.utils.telemetry_logger import _get_image_uri_option
        from sagemaker.serve.utils.types import ImageUriOption
        
        mock_is_1p.return_value = True
        
        result = _get_image_uri_option("763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch", True)
        
        self.assertEqual(result, ImageUriOption.CUSTOM_1P_IMAGE.value)

    @patch('sagemaker.serve.utils.telemetry_logger.is_1p_image_uri')
    def test_get_image_uri_option_custom(self, mock_is_1p):
        """Test custom image option."""
        from sagemaker.serve.utils.telemetry_logger import _get_image_uri_option
        from sagemaker.serve.utils.types import ImageUriOption
        
        mock_is_1p.return_value = False
        
        result = _get_image_uri_option("custom-registry.com/image:latest", True)
        
        self.assertEqual(result, ImageUriOption.CUSTOM_IMAGE.value)


if __name__ == "__main__":
    unittest.main()
