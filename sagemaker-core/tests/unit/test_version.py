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
"""Unit tests for sagemaker.core._version module."""
from __future__ import absolute_import

import os
import pytest
from unittest.mock import patch, mock_open


class TestVersion:
    """Test version loading functionality."""

    def test_version_file_read(self):
        """Test that version is read from VERSION file."""
        # Read the VERSION file directly to verify it exists and has content
        import os

        version_file_path = os.path.join(os.path.dirname(__file__), "..", "..", "VERSION")

        if os.path.exists(version_file_path):
            with open(version_file_path) as f:
                version = f.read().strip()
                assert len(version) > 0
                assert "." in version or version.isdigit()

    @patch("builtins.open", new_callable=mock_open, read_data="1.2.3\n")
    @patch("os.path.abspath")
    @patch("os.path.dirname")
    @patch("os.path.join")
    def test_version_file_parsing(self, mock_join, mock_dirname, mock_abspath, mock_file):
        """Test version file parsing with mocked file system."""
        mock_dirname.return_value = "/fake/path"
        mock_abspath.side_effect = lambda x: x
        mock_join.return_value = "/fake/VERSION"

        # Re-import to trigger the version loading with mocks
        import importlib
        from sagemaker.core import _version

        importlib.reload(_version)

        # Verify version was stripped of whitespace
        assert _version.__version__ == "1.2.3"

    def test_version_format(self):
        """Test that version follows semantic versioning format."""
        from sagemaker.core._version import __version__

        # Version should contain at least one dot (e.g., "1.0" or "1.0.0")
        assert "." in __version__ or __version__.isdigit()
