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
"""Tests for custom_file_filter module."""
from __future__ import absolute_import

import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import pytest

from sagemaker.train.remote_function.custom_file_filter import (
    CustomFileFilter,
    resolve_custom_file_filter_from_config_file,
    copy_workdir,
)


class TestCustomFileFilter:
    """Test CustomFileFilter class."""

    def test_init_with_no_patterns(self):
        """Test initialization without ignore patterns."""
        filter_obj = CustomFileFilter()
        assert filter_obj.ignore_name_patterns == []
        assert filter_obj.workdir == os.getcwd()

    def test_init_with_patterns(self):
        """Test initialization with ignore patterns."""
        patterns = ["*.pyc", "__pycache__", "*.log"]
        filter_obj = CustomFileFilter(ignore_name_patterns=patterns)
        assert filter_obj.ignore_name_patterns == patterns

    def test_ignore_name_patterns_property(self):
        """Test ignore_name_patterns property."""
        patterns = ["*.txt", "temp*"]
        filter_obj = CustomFileFilter(ignore_name_patterns=patterns)
        assert filter_obj.ignore_name_patterns == patterns

    def test_workdir_property(self):
        """Test workdir property."""
        filter_obj = CustomFileFilter()
        assert filter_obj.workdir == os.getcwd()


class TestResolveCustomFileFilterFromConfigFile:
    """Test resolve_custom_file_filter_from_config_file function."""

    def test_returns_direct_input_when_provided_as_filter(self):
        """Test returns direct input when CustomFileFilter is provided."""
        filter_obj = CustomFileFilter(ignore_name_patterns=["*.pyc"])
        result = resolve_custom_file_filter_from_config_file(direct_input=filter_obj)
        assert result is filter_obj

    def test_returns_direct_input_when_provided_as_callable(self):
        """Test returns direct input when callable is provided."""
        def custom_filter(path, names):
            return []
        result = resolve_custom_file_filter_from_config_file(direct_input=custom_filter)
        assert result is custom_filter

    @patch("sagemaker.train.remote_function.custom_file_filter.resolve_value_from_config")
    def test_returns_none_when_no_config(self, mock_resolve):
        """Test returns None when no config is found."""
        mock_resolve.return_value = None
        result = resolve_custom_file_filter_from_config_file()
        assert result is None

    @patch("sagemaker.train.remote_function.custom_file_filter.resolve_value_from_config")
    def test_creates_filter_from_config(self, mock_resolve):
        """Test creates CustomFileFilter from config."""
        patterns = ["*.pyc", "*.log"]
        mock_resolve.return_value = patterns
        result = resolve_custom_file_filter_from_config_file()
        assert isinstance(result, CustomFileFilter)
        assert result.ignore_name_patterns == patterns

    @patch("sagemaker.train.remote_function.custom_file_filter.resolve_value_from_config")
    def test_passes_sagemaker_session_to_resolve(self, mock_resolve):
        """Test passes sagemaker_session to resolve_value_from_config."""
        mock_session = MagicMock()
        mock_resolve.return_value = None
        resolve_custom_file_filter_from_config_file(sagemaker_session=mock_session)
        mock_resolve.assert_called_once()
        assert mock_resolve.call_args[1]["sagemaker_session"] == mock_session


class TestCopyWorkdir:
    """Test copy_workdir function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_src = tempfile.mkdtemp()
        self.temp_dst = tempfile.mkdtemp()
        
        # Create test files
        with open(os.path.join(self.temp_src, "test.py"), "w") as f:
            f.write("print('test')")
        with open(os.path.join(self.temp_src, "test.txt"), "w") as f:
            f.write("text file")
        os.makedirs(os.path.join(self.temp_src, "__pycache__"))
        with open(os.path.join(self.temp_src, "__pycache__", "test.pyc"), "w") as f:
            f.write("compiled")

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_src):
            shutil.rmtree(self.temp_src)
        if os.path.exists(self.temp_dst):
            shutil.rmtree(self.temp_dst)

    @patch("os.getcwd")
    def test_copy_workdir_without_filter_only_python_files(self, mock_getcwd):
        """Test copy_workdir without filter copies only Python files."""
        mock_getcwd.return_value = self.temp_src
        dst = os.path.join(self.temp_dst, "output")
        
        copy_workdir(dst)
        
        assert os.path.exists(os.path.join(dst, "test.py"))
        assert not os.path.exists(os.path.join(dst, "test.txt"))
        assert not os.path.exists(os.path.join(dst, "__pycache__"))

    @patch("os.getcwd")
    def test_copy_workdir_with_callable_filter(self, mock_getcwd):
        """Test copy_workdir with callable filter."""
        mock_getcwd.return_value = self.temp_src
        dst = os.path.join(self.temp_dst, "output")
        
        def custom_filter(path, names):
            return ["test.txt"]
        
        copy_workdir(dst, custom_file_filter=custom_filter)
        
        assert os.path.exists(os.path.join(dst, "test.py"))
        assert not os.path.exists(os.path.join(dst, "test.txt"))

    def test_copy_workdir_with_custom_file_filter_object(self):
        """Test copy_workdir with CustomFileFilter object."""
        filter_obj = CustomFileFilter(ignore_name_patterns=["*.py"])
        filter_obj._workdir = self.temp_src
        dst = os.path.join(self.temp_dst, "output")
        
        copy_workdir(dst, custom_file_filter=filter_obj)
        
        assert not os.path.exists(os.path.join(dst, "test.py"))
        assert os.path.exists(os.path.join(dst, "test.txt"))

    def test_copy_workdir_with_pattern_matching(self):
        """Test copy_workdir with pattern matching in CustomFileFilter."""
        filter_obj = CustomFileFilter(ignore_name_patterns=["*.txt", "__pycache__"])
        filter_obj._workdir = self.temp_src
        dst = os.path.join(self.temp_dst, "output")
        
        copy_workdir(dst, custom_file_filter=filter_obj)
        
        assert os.path.exists(os.path.join(dst, "test.py"))
        assert not os.path.exists(os.path.join(dst, "test.txt"))
        assert not os.path.exists(os.path.join(dst, "__pycache__"))
