import unittest
from unittest.mock import patch
from sagemaker.serve.model_server.djl_serving.utils import (
    _get_default_tensor_parallel_degree,
    _get_default_data_type,
    _get_default_batch_size,
    _tokens_from_chars,
    _tokens_from_words,
    _set_tokens_to_tokens_threshold
)


class TestDJLUtils(unittest.TestCase):
    def test_get_default_data_type(self):
        result = _get_default_data_type()
        self.assertEqual(result, "bf16")

    def test_get_default_batch_size(self):
        result = _get_default_batch_size()
        self.assertEqual(result, 1)

    def test_tokens_from_chars(self):
        result = _tokens_from_chars("test")
        self.assertEqual(result, 1.0)

    def test_tokens_from_words(self):
        result = _tokens_from_words("hello world")
        self.assertEqual(result, 2)

    def test_set_tokens_to_tokens_threshold(self):
        self.assertEqual(_set_tokens_to_tokens_threshold(100), 128)
        self.assertEqual(_set_tokens_to_tokens_threshold(200), 256)
        self.assertEqual(_set_tokens_to_tokens_threshold(500), 512)

    @patch("sagemaker.serve.model_server.djl_serving.utils._get_available_gpus")
    def test_get_default_tensor_parallel_degree(self, mock_gpus):
        mock_gpus.return_value = [0, 1, 2, 3]
        hf_config = {"num_attention_heads": 12}
        result = _get_default_tensor_parallel_degree(hf_config)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
