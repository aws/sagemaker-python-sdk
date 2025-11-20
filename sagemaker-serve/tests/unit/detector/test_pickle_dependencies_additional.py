"""Additional unit tests for pickle_dependencies.py to increase coverage."""

import unittest
from unittest.mock import Mock, patch, mock_open
import tempfile
import os


class TestGetAllFilesForInstalledPackagesPip(unittest.TestCase):
    """Test get_all_files_for_installed_packages_pip function."""

    @unittest.skip("Complex subprocess mocking required")
    def test_get_all_files_for_installed_packages_pip(self):
        """Test get_all_files_for_installed_packages_pip."""
        pass


class TestGetAllFilesForInstalledPackages(unittest.TestCase):
    """Test get_all_files_for_installed_packages function."""

    @patch('sagemaker.serve.detector.pickle_dependencies.get_all_files_for_installed_packages_pip')
    def test_get_all_files_for_installed_packages(self, mock_get_files):
        """Test get_all_files_for_installed_packages."""
        from sagemaker.serve.detector.pickle_dependencies import get_all_files_for_installed_packages
        
        mock_get_files.return_value = [
            [b"Name: test-package\n", b"Location: /usr/lib\n", b"Files:\n", b"  file1.py\n"]
        ]
        
        result = get_all_files_for_installed_packages(["test-package"])
        
        self.assertIsInstance(result, dict)


class TestBatched(unittest.TestCase):
    """Test batched function."""

    def test_batched_normal(self):
        """Test batched with normal input."""
        from sagemaker.serve.detector.pickle_dependencies import batched
        
        result = list(batched("ABCDEFG", 3))
        
        self.assertEqual(result, [("A", "B", "C"), ("D", "E", "F"), ("G",)])

    def test_batched_invalid_n(self):
        """Test batched with invalid n."""
        from sagemaker.serve.detector.pickle_dependencies import batched
        
        with self.assertRaises(ValueError):
            list(batched("ABC", 0))


class TestGetAllInstalledPackages(unittest.TestCase):
    """Test get_all_installed_packages function."""

    @patch('subprocess.run')
    def test_get_all_installed_packages(self, mock_run):
        """Test get_all_installed_packages."""
        from sagemaker.serve.detector.pickle_dependencies import get_all_installed_packages
        
        mock_result = Mock()
        mock_result.stdout = b'[{"name": "package1", "version": "1.0.0"}]'
        mock_run.return_value = mock_result
        
        result = get_all_installed_packages()
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "package1")


class TestMapPackageNamesToFiles(unittest.TestCase):
    """Test map_package_names_to_files function."""

    @patch('sagemaker.serve.detector.pickle_dependencies.tqdm.tqdm')
    @patch('sagemaker.serve.detector.pickle_dependencies.get_all_files_for_installed_packages')
    def test_map_package_names_to_files(self, mock_get_files, mock_tqdm):
        """Test map_package_names_to_files."""
        from sagemaker.serve.detector.pickle_dependencies import map_package_names_to_files
        
        mock_get_files.return_value = {"package1": {"/path/file1.py"}}
        mock_pbar = Mock()
        mock_tqdm.return_value.__enter__.return_value = mock_pbar
        
        result = map_package_names_to_files(["package1", "package2"])
        
        self.assertIsInstance(result, dict)


class TestGetCurrentlyUsedPackages(unittest.TestCase):
    """Test get_currently_used_packages function."""
    pass


class TestGetRequirementsForPklFile(unittest.TestCase):
    """Test get_requirements_for_pkl_file function."""

    @patch('sagemaker.serve.detector.pickle_dependencies.get_currently_used_packages')
    @patch('sagemaker.serve.detector.pickle_dependencies.get_all_installed_packages')
    @patch('cloudpickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_get_requirements_for_pkl_file(self, mock_file, mock_load, mock_get_packages, mock_used_packages):
        """Test get_requirements_for_pkl_file."""
        from sagemaker.serve.detector.pickle_dependencies import get_requirements_for_pkl_file
        from pathlib import Path
        
        mock_get_packages.return_value = [
            {"name": "package1", "version": "1.0.0"},
            {"name": "boto3", "version": "1.20.0"}
        ]
        mock_used_packages.return_value = {"package1"}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "test.pkl"
            dest_path = Path(tmpdir) / "requirements.txt"
            
            get_requirements_for_pkl_file(pkl_path, dest_path)
            
            mock_load.assert_called_once()


class TestGetAllRequirements(unittest.TestCase):
    """Test get_all_requirements function."""

    @patch('sagemaker.serve.detector.pickle_dependencies.get_all_installed_packages')
    def test_get_all_requirements(self, mock_get_packages):
        """Test get_all_requirements."""
        from sagemaker.serve.detector.pickle_dependencies import get_all_requirements
        from pathlib import Path
        
        mock_get_packages.return_value = [
            {"name": "package1", "version": "1.0.0"},
            {"name": "package2", "version": "2.0.0"}
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            dest_path = Path(tmpdir) / "requirements.txt"
            
            get_all_requirements(dest_path)
            
            self.assertTrue(dest_path.exists())
            content = dest_path.read_text()
            self.assertIn("package1==1.0.0", content)
            self.assertIn("package2==2.0.0", content)


if __name__ == "__main__":
    unittest.main()
