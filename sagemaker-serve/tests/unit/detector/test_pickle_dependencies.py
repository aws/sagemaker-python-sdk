"""Unit tests for sagemaker.serve.detector.pickle_dependencies module."""
import unittest
from unittest.mock import Mock, patch, mock_open, MagicMock
from pathlib import Path
import subprocess
import json
from sagemaker.serve.detector.pickle_dependencies import (
    batched,
    get_all_installed_packages,
)


class TestBatched(unittest.TestCase):
    """Test cases for batched function."""

    def test_batched_basic(self):
        """Test batched with basic input."""
        result = list(batched("ABCDEFG", 3))
        self.assertEqual(result, [("A", "B", "C"), ("D", "E", "F"), ("G",)])

    def test_batched_exact_division(self):
        """Test batched when length divides evenly."""
        result = list(batched([1, 2, 3, 4, 5, 6], 2))
        self.assertEqual(result, [(1, 2), (3, 4), (5, 6)])

    def test_batched_single_element(self):
        """Test batched with batch size of 1."""
        result = list(batched([1, 2, 3], 1))
        self.assertEqual(result, [(1,), (2,), (3,)])

    def test_batched_empty_iterable(self):
        """Test batched with empty iterable."""
        result = list(batched([], 3))
        self.assertEqual(result, [])

    def test_batched_invalid_n(self):
        """Test batched with invalid n value."""
        with self.assertRaises(ValueError):
            list(batched([1, 2, 3], 0))


class TestGetAllInstalledPackages(unittest.TestCase):
    """Test cases for get_all_installed_packages function."""

    @patch('subprocess.run')
    def test_get_all_installed_packages(self, mock_run):
        """Test getting all installed packages."""
        mock_packages = [
            {"name": "package1", "version": "1.0.0"},
            {"name": "package2", "version": "2.0.0"}
        ]
        mock_run.return_value = Mock(stdout=json.dumps(mock_packages).encode())
        
        result = get_all_installed_packages()
        
        self.assertEqual(result, mock_packages)
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_get_all_installed_packages_empty(self, mock_run):
        """Test getting installed packages when none exist."""
        mock_run.return_value = Mock(stdout=b"[]")
        
        result = get_all_installed_packages()
        
        self.assertEqual(result, [])


# Note: The following functions are complex and involve subprocess calls,
# file I/O, and sys.modules manipulation. They are better tested through
# integration tests rather than unit tests to avoid flaky mocks that can hang.
# 
# Functions not unit tested here (but covered by integration tests):
# - get_all_files_for_installed_packages_pip
# - get_all_files_for_installed_packages  
# - map_package_names_to_files
# - get_currently_used_packages
# - get_requirements_for_pkl_file
# - get_all_requirements


if __name__ == "__main__":
    unittest.main()
