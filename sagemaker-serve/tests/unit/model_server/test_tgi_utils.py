"""Unit tests for TGI serving utils module."""
import unittest
from unittest.mock import Mock, patch


class TestTGIUtilsDataType(unittest.TestCase):
    """Test TGI utils data type functions."""

    def test_get_default_dtype(self):
        """Test _get_default_dtype returns bfloat16."""
        from sagemaker.serve.model_server.tgi.utils import _get_default_dtype
        
        result = _get_default_dtype()
        self.assertEqual(result, "bfloat16")

    def test_get_admissible_dtypes(self):
        """Test _get_admissible_dtypes returns list with bfloat16."""
        from sagemaker.serve.model_server.tgi.utils import _get_admissible_dtypes
        
        result = _get_admissible_dtypes()
        self.assertEqual(result, ["bfloat16"])


class TestTGIUtilsConfigurations(unittest.TestCase):
    """Test TGI utils configuration functions."""

    @patch('sagemaker.serve.model_server.tgi.utils._get_default_max_tokens')
    @patch('sagemaker.serve.model_server.tgi.utils._get_default_tensor_parallel_degree')
    def test_get_default_tgi_configurations_with_sharding(self, mock_parallel, mock_tokens):
        """Test TGI configurations with sharding enabled."""
        from sagemaker.serve.model_server.tgi.utils import _get_default_tgi_configurations
        
        mock_parallel.return_value = 4
        mock_tokens.return_value = (2048, 512)
        
        mock_schema_builder = Mock()
        mock_schema_builder.sample_input = {"inputs": "test"}
        mock_schema_builder.sample_output = [{"generated_text": "output"}]
        
        env, max_new_tokens = _get_default_tgi_configurations(
            "model-id",
            {"num_attention_heads": 32},
            mock_schema_builder
        )
        
        self.assertEqual(env["SHARDED"], "true")
        self.assertEqual(env["NUM_SHARD"], "4")
        self.assertEqual(env["DTYPE"], "bfloat16")
        self.assertEqual(max_new_tokens, 512)

    @patch('sagemaker.serve.model_server.tgi.utils._get_default_max_tokens')
    @patch('sagemaker.serve.model_server.tgi.utils._get_default_tensor_parallel_degree')
    def test_get_default_tgi_configurations_without_sharding(self, mock_parallel, mock_tokens):
        """Test TGI configurations with sharding disabled."""
        from sagemaker.serve.model_server.tgi.utils import _get_default_tgi_configurations
        
        mock_parallel.return_value = 1
        mock_tokens.return_value = (1024, 256)
        
        mock_schema_builder = Mock()
        mock_schema_builder.sample_input = {"inputs": "test"}
        mock_schema_builder.sample_output = [{"generated_text": "output"}]
        
        env, max_new_tokens = _get_default_tgi_configurations(
            "model-id",
            {"num_attention_heads": 12},
            mock_schema_builder
        )
        
        self.assertEqual(env["SHARDED"], "false")
        self.assertEqual(env["NUM_SHARD"], "1")
        self.assertEqual(env["DTYPE"], "bfloat16")
        self.assertEqual(max_new_tokens, 256)

    @patch('sagemaker.serve.model_server.tgi.utils._get_default_max_tokens')
    @patch('sagemaker.serve.model_server.tgi.utils._get_default_tensor_parallel_degree')
    def test_get_default_tgi_configurations_no_parallel_degree(self, mock_parallel, mock_tokens):
        """Test TGI configurations when parallel degree is None."""
        from sagemaker.serve.model_server.tgi.utils import _get_default_tgi_configurations
        
        mock_parallel.return_value = None
        mock_tokens.return_value = (1024, 256)
        
        mock_schema_builder = Mock()
        mock_schema_builder.sample_input = {"inputs": "test"}
        mock_schema_builder.sample_output = [{"generated_text": "output"}]
        
        env, max_new_tokens = _get_default_tgi_configurations(
            "model-id",
            {},
            mock_schema_builder
        )
        
        self.assertIsNone(env["SHARDED"])
        self.assertIsNone(env["NUM_SHARD"])
        self.assertEqual(env["DTYPE"], "bfloat16")
        self.assertEqual(max_new_tokens, 256)

    @patch('sagemaker.serve.model_server.tgi.utils._get_default_max_tokens')
    @patch('sagemaker.serve.model_server.tgi.utils._get_default_tensor_parallel_degree')
    def test_get_default_tgi_configurations_returns_tuple(self, mock_parallel, mock_tokens):
        """Test that function returns a tuple."""
        from sagemaker.serve.model_server.tgi.utils import _get_default_tgi_configurations
        
        mock_parallel.return_value = 2
        mock_tokens.return_value = (1024, 256)
        
        mock_schema_builder = Mock()
        mock_schema_builder.sample_input = {"inputs": "test"}
        mock_schema_builder.sample_output = [{"generated_text": "output"}]
        
        result = _get_default_tgi_configurations(
            "model-id",
            {"num_attention_heads": 16},
            mock_schema_builder
        )
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], dict)
        self.assertIsInstance(result[1], int)


if __name__ == "__main__":
    unittest.main()
