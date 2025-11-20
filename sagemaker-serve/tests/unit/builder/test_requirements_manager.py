"""Unit tests for sagemaker.serve.builder.requirements_manager module."""
import unittest
from unittest.mock import Mock, patch, MagicMock
import os
from sagemaker.serve.builder.requirements_manager import RequirementsManager


class TestRequirementsManager(unittest.TestCase):
    """Test cases for RequirementsManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = RequirementsManager()

    @patch('subprocess.run')
    def test_install_requirements_txt(self, mock_run):
        """Test installing requirements from txt file."""
        self.manager._install_requirements_txt()
        
        mock_run.assert_called_once_with(
            "pip install -r in_process_requirements.txt",
            shell=True,
            check=True
        )

    @patch('subprocess.run')
    def test_update_conda_env_in_path(self, mock_run):
        """Test updating conda environment from yml file."""
        self.manager._update_conda_env_in_path()
        
        mock_run.assert_called_once_with(
            "conda env update -f conda_in_process.yml",
            shell=True,
            check=True
        )

    @patch.dict(os.environ, {'CONDA_DEFAULT_ENV': 'my-env'})
    def test_get_active_conda_env_name(self):
        """Test getting active conda environment name."""
        result = self.manager._get_active_conda_env_name()
        self.assertEqual(result, 'my-env')

    @patch.dict(os.environ, {}, clear=True)
    def test_get_active_conda_env_name_none(self):
        """Test getting conda env name when not set."""
        result = self.manager._get_active_conda_env_name()
        self.assertIsNone(result)

    @patch.dict(os.environ, {'CONDA_PREFIX': '/path/to/conda'})
    def test_get_active_conda_env_prefix(self):
        """Test getting active conda environment prefix."""
        result = self.manager._get_active_conda_env_prefix()
        self.assertEqual(result, '/path/to/conda')

    @patch.dict(os.environ, {}, clear=True)
    def test_get_active_conda_env_prefix_none(self):
        """Test getting conda prefix when not set."""
        result = self.manager._get_active_conda_env_prefix()
        self.assertIsNone(result)

    @patch.dict(os.environ, {}, clear=True)
    @patch('os.getcwd')
    def test_detect_conda_env_no_conda(self, mock_getcwd):
        """Test detecting dependencies when no conda env is active."""
        mock_getcwd.return_value = '/current/dir'
        
        result = self.manager._detect_conda_env_and_local_dependencies()
        
        expected_path = os.path.join('/current/dir', 'in_process_requirements.txt')
        self.assertEqual(result, expected_path)

    @patch.dict(os.environ, {'CONDA_DEFAULT_ENV': 'my-env'})
    @patch('os.getcwd')
    def test_detect_conda_env_with_conda(self, mock_getcwd):
        """Test detecting dependencies when conda env is active."""
        mock_getcwd.return_value = '/current/dir'
        
        result = self.manager._detect_conda_env_and_local_dependencies()
        
        expected_path = os.path.join('/current/dir', 'conda_in_process.yml')
        self.assertEqual(result, expected_path)

    @patch.dict(os.environ, {'CONDA_DEFAULT_ENV': 'base'})
    @patch('os.getcwd')
    @patch('sagemaker.serve.builder.requirements_manager.logger')
    def test_detect_conda_env_base_warning(self, mock_logger, mock_getcwd):
        """Test warning when using base conda environment."""
        mock_getcwd.return_value = '/current/dir'
        
        result = self.manager._detect_conda_env_and_local_dependencies()
        
        mock_logger.warning.assert_called_once()
        self.assertIn("base", mock_logger.warning.call_args[0][0])

    @patch.dict(os.environ, {'CONDA_PREFIX': '/conda/prefix'}, clear=True)
    @patch('os.getcwd')
    def test_detect_conda_env_with_prefix_only(self, mock_getcwd):
        """Test detecting dependencies with only conda prefix set."""
        mock_getcwd.return_value = '/current/dir'
        
        result = self.manager._detect_conda_env_and_local_dependencies()
        
        expected_path = os.path.join('/current/dir', 'conda_in_process.yml')
        self.assertEqual(result, expected_path)

    @patch('subprocess.run')
    def test_capture_and_install_txt_dependencies(self, mock_run):
        """Test capturing and installing txt dependencies."""
        self.manager.capture_and_install_dependencies("requirements.txt")
        
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_capture_and_install_yml_dependencies(self, mock_run):
        """Test capturing and installing yml dependencies."""
        self.manager.capture_and_install_dependencies("environment.yml")
        
        mock_run.assert_called_once()

    def test_capture_and_install_invalid_dependencies(self):
        """Test error handling for invalid dependency file."""
        with self.assertRaises(ValueError) as context:
            self.manager.capture_and_install_dependencies("invalid.json")
        
        self.assertIn("Invalid dependencies", str(context.exception))


if __name__ == "__main__":
    unittest.main()
