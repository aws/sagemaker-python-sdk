"""Unit tests for sagemaker.serve.utils.hf_utils module."""
import unittest
import os
import shutil
import sys
import tempfile
from unittest.mock import Mock, patch, mock_open
import json
from urllib.error import HTTPError, URLError
from json import JSONDecodeError
from sagemaker.serve.utils.hf_utils import (
    _get_model_config_properties_from_hf,
    download_huggingface_model,
)


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


class TestDownloadHuggingfaceModel(unittest.TestCase):
    """Test cases for the public download_huggingface_model helper."""

    def setUp(self):
        # download_huggingface_model imports huggingface_hub lazily; provide a
        # stub module so the tests do not require the real package installed.
        self._hf_stub = Mock()
        self._patcher = patch.dict(sys.modules, {"huggingface_hub": self._hf_stub})
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
        # Keep the suite side-effect free: never create real directories.
        makedirs_patcher = patch("sagemaker.serve.utils.hf_utils.os.makedirs")
        makedirs_patcher.start()
        self.addCleanup(makedirs_patcher.stop)
        self.local_dir = tempfile.mkdtemp(prefix="hf-test-")
        self.addCleanup(shutil.rmtree, self.local_dir, ignore_errors=True)

    def test_requires_local_dir_or_s3_uri(self):
        """Neither destination given is a caller error, before any download."""
        with self.assertRaises(ValueError) as context:
            download_huggingface_model("gpt2")
        self.assertIn("local_dir, s3_uri", str(context.exception))
        self._hf_stub.snapshot_download.assert_not_called()

    def test_downloads_to_local_dir_and_returns_it(self):
        """With only local_dir, it downloads there and returns the path."""
        result = download_huggingface_model("gpt2", local_dir=self.local_dir)
        self.assertEqual(result, self.local_dir)
        self.assertEqual(
            self._hf_stub.snapshot_download.call_args.kwargs["local_dir"], self.local_dir
        )

    def test_uploads_to_s3_and_returns_uri(self):
        """With s3_uri, it uploads the snapshot and returns the S3 URI."""
        with patch("sagemaker.core.s3.S3Uploader") as mock_uploader:
            mock_uploader.upload.return_value = "s3://bucket/prefix/gpt2"
            result = download_huggingface_model(
                "gpt2", local_dir=self.local_dir, s3_uri="s3://bucket/prefix"
            )
        self.assertEqual(result, "s3://bucket/prefix/gpt2")
        mock_uploader.upload.assert_called_once()
        self.assertEqual(
            mock_uploader.upload.call_args.kwargs["desired_s3_uri"], "s3://bucket/prefix"
        )

    def test_s3_only_uses_temp_dir_and_cleans_up(self):
        """With s3_uri and no local_dir, the snapshot stages in a temp dir that
        is removed after the upload (no snapshot left on the local volume)."""
        with patch("sagemaker.core.s3.S3Uploader") as mock_uploader:
            mock_uploader.upload.return_value = "s3://bucket/prefix/gpt2"
            result = download_huggingface_model("gpt2", s3_uri="s3://bucket/prefix")
        self.assertEqual(result, "s3://bucket/prefix/gpt2")
        staging_dir = self._hf_stub.snapshot_download.call_args.kwargs["local_dir"]
        self.assertFalse(os.path.exists(staging_dir))

    def test_forwards_hf_token_and_snapshot_passthroughs(self):
        """token / revision / patterns are passed through to snapshot_download."""
        download_huggingface_model(
            "gpt2",
            local_dir=self.local_dir,
            hf_hub_token="hf_tok",
            revision="v1.0",
            allow_patterns="*.safetensors",
            ignore_patterns="*.bin",
        )
        kwargs = self._hf_stub.snapshot_download.call_args.kwargs
        self.assertEqual(kwargs["token"], "hf_tok")
        self.assertEqual(kwargs["revision"], "v1.0")
        self.assertEqual(kwargs["allow_patterns"], "*.safetensors")
        self.assertEqual(kwargs["ignore_patterns"], "*.bin")

    def test_missing_huggingface_hub_raises_import_error(self):
        """The helper's own ImportError (with install guidance) is raised when
        huggingface_hub cannot be imported."""
        import builtins

        real_import = builtins.__import__

        def _raise_for_hf(name, *args, **kwargs):
            if name == "huggingface_hub":
                raise ImportError("No module named 'huggingface_hub'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_raise_for_hf), self.assertRaises(
            ImportError
        ) as context:
            download_huggingface_model("gpt2", local_dir=self.local_dir)
        self.assertIn("pip install huggingface_hub", str(context.exception))


if __name__ == "__main__":
    unittest.main()
