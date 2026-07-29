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

        self.assertIn("Did not find any supported model config file", str(context.exception))
        self.assertEqual(mock_logger.warning.call_count, 3)

    @patch('urllib.request.urlopen')
    @patch('sagemaker.serve.utils.hf_utils.logger')
    def test_get_model_config_url_error(self, mock_logger, mock_urlopen):
        """Test handling of URL errors."""
        mock_urlopen.side_effect = URLError("Connection failed")
        
        with self.assertRaises(ValueError) as context:
            _get_model_config_properties_from_hf("model-id")

        self.assertIn("Did not find any supported model config file", str(context.exception))
        self.assertEqual(mock_logger.warning.call_count, 3)

    @patch('urllib.request.urlopen')
    @patch('sagemaker.serve.utils.hf_utils.logger')
    def test_get_model_config_timeout_error(self, mock_logger, mock_urlopen):
        """Test handling of timeout errors."""
        mock_urlopen.side_effect = TimeoutError("Request timed out")
        
        with self.assertRaises(ValueError) as context:
            _get_model_config_properties_from_hf("model-id")

        self.assertIn("Did not find any supported model config file", str(context.exception))
        self.assertEqual(mock_logger.warning.call_count, 3)

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

        self.assertIn("Did not find any supported model config file", str(context.exception))
        self.assertEqual(mock_logger.warning.call_count, 3)

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

    @patch("urllib.request.urlopen")
    def test_get_model_config_falls_back_to_model_index(self, mock_urlopen):
        """Test fallback to model_index.json when config.json is missing."""
        config_missing_error = HTTPError(
            "https://huggingface.co/org/model/raw/main/config.json", 404, "Not Found", {}, None
        )
        model_index_config = {"_class_name": "FluxPipeline", "_diffusers_version": "0.31.0"}

        mock_model_index_response = Mock()
        mock_model_index_response.__enter__ = Mock(return_value=mock_model_index_response)
        mock_model_index_response.__exit__ = Mock(return_value=False)

        def _urlopen_side_effect(request):
            url = request.full_url if hasattr(request, "full_url") else request
            if url.endswith("/config.json"):
                raise config_missing_error
            if url.endswith("/model_index.json"):
                return mock_model_index_response
            raise AssertionError(f"Unexpected URL called: {url}")

        mock_urlopen.side_effect = _urlopen_side_effect

        with patch("json.load", side_effect=[model_index_config]):
            result = _get_model_config_properties_from_hf("org/model-name")

        self.assertEqual(result, model_index_config)

    @patch("urllib.request.urlopen")
    @patch("sagemaker.serve.utils.hf_utils.logger")
    def test_get_model_config_dual_file_error_when_both_missing(self, mock_logger, mock_urlopen):
        """Test error when all known config files are missing."""
        mock_urlopen.side_effect = HTTPError("url", 404, "Not Found", {}, None)

        with self.assertRaises(ValueError) as context:
            _get_model_config_properties_from_hf("model-id")

        self.assertIn(
            "Expected one of: config.json, model_index.json, adapter_config.json",
            str(context.exception),
        )
        self.assertEqual(mock_urlopen.call_count, 3)
        self.assertEqual(mock_logger.warning.call_count, 3)

    @patch("urllib.request.urlopen")
    def test_get_model_config_falls_back_to_adapter_config(self, mock_urlopen):
        """Test fallback to adapter_config.json when config/model_index are missing."""
        config_missing_error = HTTPError(
            "https://huggingface.co/org/model/raw/main/config.json", 404, "Not Found", {}, None
        )
        model_index_missing_error = HTTPError(
            "https://huggingface.co/org/model/raw/main/model_index.json", 404, "Not Found", {}, None
        )
        adapter_config = {
            "base_model_name_or_path": "LiquidAI/LFM2.5-1.2B-Instruct",
            "peft_type": "LORA",
        }

        mock_adapter_response = Mock()
        mock_adapter_response.__enter__ = Mock(return_value=mock_adapter_response)
        mock_adapter_response.__exit__ = Mock(return_value=False)

        def _urlopen_side_effect(request):
            url = request.full_url if hasattr(request, "full_url") else request
            if url.endswith("/config.json"):
                raise config_missing_error
            if url.endswith("/model_index.json"):
                raise model_index_missing_error
            if url.endswith("/adapter_config.json"):
                return mock_adapter_response
            raise AssertionError(f"Unexpected URL called: {url}")

        mock_urlopen.side_effect = _urlopen_side_effect

        with patch("json.load", side_effect=[adapter_config]):
            result = _get_model_config_properties_from_hf("org/model-name")

        self.assertEqual(result, adapter_config)


if __name__ == "__main__":
    unittest.main()
