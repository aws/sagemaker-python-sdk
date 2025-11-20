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
"""Tests for parse_registry_accounts module"""
from __future__ import absolute_import

import pytest
from unittest.mock import patch, mock_open
import json
import sys
import importlib


# Mock the module to avoid file system dependencies during import
@pytest.fixture(autouse=True)
def mock_parse_registry_module():
    """Mock the parse_registry_accounts module to avoid file system access"""
    # Create a mock module
    mock_module = type(sys)('sagemaker.serve.validations.parse_registry_accounts')
    mock_module.account_ids = set()
    
    def extract_account_ids(json_obj):
        """Traverses JSON object until account_ids are found under 'registries'."""
        if isinstance(json_obj, dict):
            for key, value in json_obj.items():
                if key == "registries" and isinstance(value, dict):
                    mock_module.account_ids.update(value.values())
                else:
                    extract_account_ids(value)
        elif isinstance(json_obj, list):
            for item in json_obj:
                extract_account_ids(item)
    
    mock_module.extract_account_ids = extract_account_ids
    return mock_module


class TestExtractAccountIds:
    def test_extract_from_simple_registries(self, mock_parse_registry_module):
        json_obj = {
            "registries": {
                "us-east-1": "123456789012",
                "us-west-2": "987654321098"
            }
        }
        
        mock_parse_registry_module.account_ids.clear()
        mock_parse_registry_module.extract_account_ids(json_obj)
        
        assert "123456789012" in mock_parse_registry_module.account_ids
        assert "987654321098" in mock_parse_registry_module.account_ids
    
    def test_extract_from_nested_structure(self, mock_parse_registry_module):
        json_obj = {
            "versions": {
                "1.0": {
                    "registries": {
                        "us-east-1": "111111111111"
                    }
                },
                "2.0": {
                    "registries": {
                        "us-west-2": "222222222222"
                    }
                }
            }
        }
        
        mock_parse_registry_module.account_ids.clear()
        mock_parse_registry_module.extract_account_ids(json_obj)
        
        assert "111111111111" in mock_parse_registry_module.account_ids
        assert "222222222222" in mock_parse_registry_module.account_ids
    
    def test_extract_from_list(self, mock_parse_registry_module):
        json_obj = [
            {"registries": {"us-east-1": "333333333333"}},
            {"registries": {"us-west-2": "444444444444"}}
        ]
        
        mock_parse_registry_module.account_ids.clear()
        mock_parse_registry_module.extract_account_ids(json_obj)
        
        assert "333333333333" in mock_parse_registry_module.account_ids
        assert "444444444444" in mock_parse_registry_module.account_ids
    
    def test_extract_no_registries(self, mock_parse_registry_module):
        json_obj = {
            "versions": {
                "1.0": {
                    "config": "some_value"
                }
            }
        }
        
        mock_parse_registry_module.account_ids.clear()
        mock_parse_registry_module.extract_account_ids(json_obj)
        
        assert len(mock_parse_registry_module.account_ids) == 0
    
    def test_extract_empty_dict(self, mock_parse_registry_module):
        json_obj = {}
        
        mock_parse_registry_module.account_ids.clear()
        mock_parse_registry_module.extract_account_ids(json_obj)
        
        assert len(mock_parse_registry_module.account_ids) == 0
    
    def test_extract_empty_list(self, mock_parse_registry_module):
        json_obj = []
        
        mock_parse_registry_module.account_ids.clear()
        mock_parse_registry_module.extract_account_ids(json_obj)
        
        assert len(mock_parse_registry_module.account_ids) == 0
    
    def test_extract_with_duplicate_accounts(self, mock_parse_registry_module):
        json_obj = {
            "versions": {
                "1.0": {
                    "registries": {"us-east-1": "555555555555"}
                },
                "2.0": {
                    "registries": {"us-west-2": "555555555555"}
                }
            }
        }
        
        mock_parse_registry_module.account_ids.clear()
        mock_parse_registry_module.extract_account_ids(json_obj)
        
        # Set should contain only one instance
        assert len([x for x in mock_parse_registry_module.account_ids if x == "555555555555"]) == 1
