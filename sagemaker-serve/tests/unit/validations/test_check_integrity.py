import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open
from sagemaker.serve.validations.check_integrity import (
    compute_hash,
    perform_integrity_check
)


class TestCheckIntegrity(unittest.TestCase):
    def test_compute_hash(self):
        buffer = b"test data"
        hash_value = compute_hash(buffer)
        self.assertIsInstance(hash_value, str)
        self.assertEqual(len(hash_value), 64)

    def test_compute_hash_consistency(self):
        buffer = b"test data"
        hash1 = compute_hash(buffer)
        hash2 = compute_hash(buffer)
        self.assertEqual(hash1, hash2)

    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data=b'{"sha256_hash": "test_hash"}')
    @patch("sagemaker.serve.validations.check_integrity._MetaData.from_json")
    def test_perform_integrity_check_failure(self, mock_metadata, mock_file, mock_exists):
        mock_exists.return_value = True
        mock_meta = type("obj", (object,), {"sha256_hash": "wrong_hash"})()
        mock_metadata.return_value = mock_meta
        
        with self.assertRaises(ValueError):
            perform_integrity_check(b"test", Path("/tmp/metadata.json"))


if __name__ == "__main__":
    unittest.main()
