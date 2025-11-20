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
"""Tests for dependency_manager module"""
from __future__ import absolute_import

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import tempfile

from sagemaker.serve.detector.dependency_manager import (
    _parse_dependency_list,
    _is_valid_requirement_file,
    _process_custom_dependencies,
    _process_customer_provided_requirements,
)


class TestParseDependencyList:
    def test_parse_simple_packages(self):
        dependencies = ["numpy", "pandas", "scikit-learn"]
        result = _parse_dependency_list(dependencies)
        
        assert result == {"numpy": "", "pandas": "", "scikit-learn": ""}
    
    def test_parse_with_version_constraints(self):
        dependencies = [
            "numpy>=1.20.0",
            "pandas==1.3.0",
            "scikit-learn<1.0.0",
            "torch<=1.9.0"
        ]
        result = _parse_dependency_list(dependencies)
        
        assert result["numpy"] == ">=1.20.0"
        assert result["pandas"] == "==1.3.0"
        assert result["scikit-learn"] == "<1.0.0"
        assert result["torch"] == "<=1.9.0"
    
    def test_parse_with_multiple_constraints(self):
        dependencies = [
            "package>=1.0.0,<2.0.0",
            "another>=0.5,<=1.5"
        ]
        result = _parse_dependency_list(dependencies)
        
        assert result["package"] == ">=1.0.0,<2.0.0"
        assert result["another"] == ">=0.5,<=1.5"
    
    def test_parse_with_url(self):
        dependencies = [
            "package@https://github.com/user/repo/archive/main.zip"
        ]
        result = _parse_dependency_list(dependencies)
        
        assert "package" in result
        assert "https://github.com" in result["package"]
    
    def test_parse_with_comments(self):
        dependencies = [
            "# This is a comment",
            "numpy>=1.20.0",
            "# Another comment",
            "pandas"
        ]
        result = _parse_dependency_list(dependencies)
        
        assert "numpy" in result
        assert "pandas" in result
        assert len(result) == 2
    
    def test_parse_with_complex_versions(self):
        dependencies = [
            "package~=1.4.2",
            "another!=1.0.0",
        ]
        result = _parse_dependency_list(dependencies)
        
        assert result["package"] == "~=1.4.2"
        assert result["another"] == "!=1.0.0"
    
    def test_parse_empty_list(self):
        dependencies = []
        result = _parse_dependency_list(dependencies)
        
        assert result == {}
    
    def test_parse_with_dots_and_dashes(self):
        dependencies = [
            "my-package.name>=1.0",
            "another_package==2.0"
        ]
        result = _parse_dependency_list(dependencies)
        
        assert "my-package.name" in result
        assert "another_package" in result


class TestIsValidRequirementFile:
    def test_valid_txt_file(self):
        path = Path("requirements.txt")
        assert _is_valid_requirement_file(path) is True
    
    def test_invalid_extension(self):
        path = Path("requirements.json")
        assert _is_valid_requirement_file(path) is False
    
    def test_no_extension(self):
        path = Path("requirements")
        assert _is_valid_requirement_file(path) is False


class TestProcessCustomDependencies:
    def test_process_custom_dependencies(self):
        custom_deps = ["custom-package>=1.0", "another-package"]
        module_version_dict = {"existing": ">=2.0"}
        
        result = _process_custom_dependencies(custom_deps, module_version_dict)
        
        assert result["existing"] == ">=2.0"
        assert result["custom-package"] == ">=1.0"
        assert result["another-package"] == ""
    
    def test_process_custom_dependencies_override(self):
        custom_deps = ["existing>=3.0"]
        module_version_dict = {"existing": ">=2.0"}
        
        result = _process_custom_dependencies(custom_deps, module_version_dict)
        
        # Custom should override existing
        assert result["existing"] == ">=3.0"
    
    def test_process_empty_custom_dependencies(self):
        custom_deps = []
        module_version_dict = {"existing": ">=2.0"}
        
        result = _process_custom_dependencies(custom_deps, module_version_dict)
        
        assert result == {"existing": ">=2.0"}


class TestProcessCustomerProvidedRequirements:
    def test_process_valid_requirements_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("numpy>=1.20.0\n")
            f.write("pandas==1.3.0\n")
            temp_path = f.name
        
        try:
            module_version_dict = {"existing": ">=2.0"}
            result = _process_customer_provided_requirements(temp_path, module_version_dict)
            
            assert result["existing"] == ">=2.0"
            assert result["numpy"] == ">=1.20.0"
            assert result["pandas"] == "==1.3.0"
        finally:
            Path(temp_path).unlink()
    
    def test_process_nonexistent_file(self):
        requirements_file = "/nonexistent/requirements.txt"
        module_version_dict = {}
        
        with pytest.raises(Exception, match="doesn't exist"):
            _process_customer_provided_requirements(requirements_file, module_version_dict)
    
    def test_process_invalid_file_extension(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"key": "value"}')
            temp_path = f.name
        
        try:
            module_version_dict = {}
            with pytest.raises(Exception, match="doesn't exist"):
                _process_customer_provided_requirements(temp_path, module_version_dict)
        finally:
            Path(temp_path).unlink()
