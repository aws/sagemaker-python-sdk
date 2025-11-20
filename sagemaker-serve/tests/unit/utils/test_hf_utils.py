"""Unit tests for sagemaker.serve.utils.hf_utils module."""
import unittest
from unittest.mock import Mock, patch, mock_open
import json
from urllib.error import HTTPError, URLError
from json import JSONDecodeError
from sagemaker.serve.utils.hf_utils import _get_model_config_properties_from_hf


class TestGetModelConfigPropertiesFromHf(unittest.TestCase):
    """Test cases for _get_model_config_properties_from_hf function."""

    @patch('urllib.request.urlopen')
    def test_get_model_config_success(self, mock_urlopen):
        """Test successful model config retrieval."""
        mock_config = {
            "model_type": "bert",
            "hidden_size": 768,
            "num_attention_heads": 12
        }
        mock_response = Mock()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_response.read.return_value = json.dumps(mock_config).encode()
        mock_urlopen.return_value = mock_response
        
        # Mock json.load to return our config
        with patch('json.load', return_value=mock_config):
            result = _get_model_config_properties_from_hf("bert-base-uncased")
        
        self.assertEqual(result, mock_config)
        self.assertEqual(result["model_type"], "bert")

    @patch('urllib.request.urlopen')
    @patch('urllib.request.Request')
    def test_get_model_config_with_token(self, mock_request, mock_urlopen):
        """Test model config retrieval with HF token."""
        mock_config = {"model_type": "gpt2"}
        mock_response = Mock()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        
        with patch('json.load', return_value=mock_config):
            result = _get_model_config_properties_from_hf(
                "gpt2",
                hf_hub_token="hf_test_token"
            )
        
        # Verify Request was called with authorization header
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        self.assertIn("Authorization", call_args[1]["headers"])
        self.assertEqual(result, mock_config)

    @patch('urllib.request.urlopen')
    def test_get_model_config_unauthorized_error(self, mock_urlopen):
        """Test handling of 401 Unauthorized error."""
        mock_urlopen.side_effect = HTTPError(
            "url", 401, "Unauthorized", {}, None
        )
        
        with self.assertRaises(ValueError) as context:
            _get_model_config_properties_from_hf("private-model")
        
        self.assertIn("gated/private", str(context.exception))
        self.assertIn("HUGGING_FACE_HUB_TOKEN", str(context.exception))

    @patch('urllib.request.urlopen')
    @patch('sagemaker.serve.utils.hf_utils.logger')
    def test_get_model_config_http_error(self, mock_logger, mock_urlopen):
        """Test handling of HTTP errors (non-401)."""
        mock_urlopen.side_effect = HTTPError(
            "url", 404, "Not Found", {}, None
        )
        
        with self.assertRaises(ValueError) as context:
            _get_model_config_properties_from_hf("non-existent-model")
        
        self.assertIn("Did not find a config.json", str(context.exception))
        mock_logger.warning.assert_called_once()

    @patch('urllib.request.urlopen')
    @patch('sagemaker.serve.utils.hf_utils.logger')
    def test_get_model_config_url_error(self, mock_logger, mock_urlopen):
        """Test handling of URL errors."""
        mock_urlopen.side_effect = URLError("Connection failed")
        
        with self.assertRaises(ValueError) as context:
            _get_model_config_properties_from_hf("model-id")
        
        self.assertIn("Did not find a config.json", str(context.exception))
        mock_logger.warning.assert_called_once()

    @patch('urllib.request.urlopen')
    @patch('sagemaker.serve.utils.hf_utils.logger')
    def test_get_model_config_timeout_error(self, mock_logger, mock_urlopen):
        """Test handling of timeout errors."""
        mock_urlopen.side_effect = TimeoutError("Request timed out")
        
        with self.assertRaises(ValueError) as context:
            _get_model_config_properties_from_hf("model-id")
        
        self.assertIn("Did not find a config.json", str(context.exception))
        mock_logger.warning.assert_called_once()

    @patch('urllib.request.urlopen')
    @patch('sagemaker.serve.utils.hf_utils.logger')
    def test_get_model_config_json_decode_error(self, mock_logger, mock_urlopen):
        """Test handling of JSON decode errors."""
        mock_response = Mock()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response
        
        with patch('json.load', side_effect=JSONDecodeError("msg", "doc", 0)):
            with self.assertRaises(ValueError) as context:
                _get_model_config_properties_from_hf("model-id")
        
        self.assertIn("Did not find a config.json", str(context.exception))
        mock_logger.warning.assert_called_once()

    @patch('urllib.request.urlopen')
    def test_get_model_config_url_format(self, mock_urlopen):
        """Test that correct URL is constructed."""
        mock_config = {"model_type": "test"}
        mock_response = Mock()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response
        
        with patch('json.load', return_value=mock_config):
            _get_model_config_properties_from_hf("org/model-name")
        
        # Verify the URL was constructed correctly
        expected_url = "https://huggingface.co/org/model-name/raw/main/config.json"
        mock_urlopen.assert_called_once()
        actual_url = mock_urlopen.call_args[0][0]
        self.assertEqual(actual_url, expected_url)


if __name__ == "__main__":
    unittest.main()
