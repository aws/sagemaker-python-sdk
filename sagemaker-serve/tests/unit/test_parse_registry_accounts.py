"""
Unit tests for sagemaker.serve.validations.parse_registry_accounts module.

Tests the extract_account_ids function that parses registry account IDs from JSON.
"""

import unittest
from unittest.mock import patch, mock_open, MagicMock
import json
import sys

# Mock os.listdir to prevent FileNotFoundError during module import
with patch('os.listdir', return_value=[]):
    from sagemaker.serve.validations.parse_registry_accounts import extract_account_ids


class TestExtractAccountIds(unittest.TestCase):
    """Test extract_account_ids function."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset the global account_ids set before each test
        import sagemaker.serve.validations.parse_registry_accounts as module
        module.account_ids = set()

    def test_extract_account_ids_from_simple_dict(self):
        """Test extracting account IDs from a simple dictionary with registries."""
        json_obj = {
            "registries": {
                "us-east-1": "123456789012",
                "us-west-2": "987654321098"
            }
        }
        
        import sagemaker.serve.validations.parse_registry_accounts as module
        extract_account_ids(json_obj)
        
        self.assertEqual(len(module.account_ids), 2)
        self.assertIn("123456789012", module.account_ids)
        self.assertIn("987654321098", module.account_ids)

    def test_extract_account_ids_from_nested_dict(self):
        """Test extracting account IDs from nested dictionary structure."""
        json_obj = {
            "versions": {
                "1.0": {
                    "registries": {
                        "us-east-1": "111111111111",
                        "eu-west-1": "222222222222"
                    }
                },
                "2.0": {
                    "registries": {
                        "us-east-1": "333333333333"
                    }
                }
            }
        }
        
        import sagemaker.serve.validations.parse_registry_accounts as module
        extract_account_ids(json_obj)
        
        self.assertEqual(len(module.account_ids), 3)
        self.assertIn("111111111111", module.account_ids)
        self.assertIn("222222222222", module.account_ids)
        self.assertIn("333333333333", module.account_ids)

    def test_extract_account_ids_from_list(self):
        """Test extracting account IDs when JSON contains lists."""
        json_obj = [
            {
                "registries": {
                    "us-east-1": "444444444444"
                }
            },
            {
                "registries": {
                    "us-west-2": "555555555555"
                }
            }
        ]
        
        import sagemaker.serve.validations.parse_registry_accounts as module
        extract_account_ids(json_obj)
        
        self.assertEqual(len(module.account_ids), 2)
        self.assertIn("444444444444", module.account_ids)
        self.assertIn("555555555555", module.account_ids)

    def test_extract_account_ids_with_no_registries(self):
        """Test that function handles JSON without registries key."""
        json_obj = {
            "versions": {
                "1.0": {
                    "image": "some-image:latest"
                }
            }
        }
        
        import sagemaker.serve.validations.parse_registry_accounts as module
        extract_account_ids(json_obj)
        
        self.assertEqual(len(module.account_ids), 0)

    def test_extract_account_ids_with_empty_registries(self):
        """Test extracting from empty registries dictionary."""
        json_obj = {
            "registries": {}
        }
        
        import sagemaker.serve.validations.parse_registry_accounts as module
        extract_account_ids(json_obj)
        
        self.assertEqual(len(module.account_ids), 0)

    def test_extract_account_ids_deduplicates(self):
        """Test that duplicate account IDs are deduplicated."""
        json_obj = {
            "version1": {
                "registries": {
                    "us-east-1": "123456789012",
                    "us-west-2": "123456789012"  # Duplicate
                }
            },
            "version2": {
                "registries": {
                    "eu-west-1": "123456789012"  # Duplicate again
                }
            }
        }
        
        import sagemaker.serve.validations.parse_registry_accounts as module
        extract_account_ids(json_obj)
        
        # Should only have one unique account ID
        self.assertEqual(len(module.account_ids), 1)
        self.assertIn("123456789012", module.account_ids)

    def test_extract_account_ids_with_mixed_structure(self):
        """Test extracting from complex mixed structure with lists and dicts."""
        json_obj = {
            "training": {
                "versions": {
                    "1.0": {
                        "registries": {
                            "us-east-1": "111111111111"
                        }
                    }
                }
            },
            "inference": {
                "versions": {
                    "2.0": {
                        "registries": {
                            "us-west-2": "222222222222"
                        }
                    }
                }
            }
        }
        
        import sagemaker.serve.validations.parse_registry_accounts as module
        extract_account_ids(json_obj)
        
        self.assertEqual(len(module.account_ids), 2)
        self.assertIn("111111111111", module.account_ids)
        self.assertIn("222222222222", module.account_ids)

    def test_extract_account_ids_with_non_dict_registries(self):
        """Test that function handles registries that is not a dict."""
        json_obj = {
            "registries": "not-a-dict"
        }
        
        import sagemaker.serve.validations.parse_registry_accounts as module
        # Should not raise an error, just skip
        extract_account_ids(json_obj)
        
        self.assertEqual(len(module.account_ids), 0)

    def test_extract_account_ids_with_deeply_nested_structure(self):
        """Test extracting from deeply nested structure."""
        json_obj = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "registries": {
                                "us-east-1": "999999999999"
                            }
                        }
                    }
                }
            }
        }
        
        import sagemaker.serve.validations.parse_registry_accounts as module
        extract_account_ids(json_obj)
        
        self.assertEqual(len(module.account_ids), 1)
        self.assertIn("999999999999", module.account_ids)

    def test_extract_account_ids_with_multiple_regions(self):
        """Test extracting account IDs from multiple regions."""
        json_obj = {
            "registries": {
                "us-east-1": "111111111111",
                "us-west-1": "222222222222",
                "us-west-2": "333333333333",
                "eu-west-1": "444444444444",
                "eu-central-1": "555555555555",
                "ap-southeast-1": "666666666666"
            }
        }
        
        import sagemaker.serve.validations.parse_registry_accounts as module
        extract_account_ids(json_obj)
        
        self.assertEqual(len(module.account_ids), 6)
        self.assertIn("111111111111", module.account_ids)
        self.assertIn("666666666666", module.account_ids)


class TestParseRegistryAccountsIntegration(unittest.TestCase):
    """Integration tests for the parse_registry_accounts module."""

    def test_module_has_account_ids_set(self):
        """Test that module has account_ids set defined."""
        import sagemaker.serve.validations.parse_registry_accounts as module
        self.assertTrue(hasattr(module, 'account_ids'))
        self.assertIsInstance(module.account_ids, set)

    def test_module_has_extract_function(self):
        """Test that module has extract_account_ids function."""
        import sagemaker.serve.validations.parse_registry_accounts as module
        self.assertTrue(hasattr(module, 'extract_account_ids'))
        self.assertTrue(callable(module.extract_account_ids))


if __name__ == '__main__':
    unittest.main()
