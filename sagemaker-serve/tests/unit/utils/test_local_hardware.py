"""Unit tests for sagemaker.serve.utils.local_hardware module."""
import unittest
from unittest.mock import Mock, patch, MagicMock
from sagemaker.serve.utils.local_hardware import (
    _get_ram_usage_mb,
    _get_gpu_info_fallback,
    hardware_lookup,
    fallback_gpu_resource_mapping,
)


class TestGetRamUsageMb(unittest.TestCase):
    """Test cases for _get_ram_usage_mb function."""

    @patch('psutil.virtual_memory')
    def test_get_ram_usage_mb(self, mock_virtual_memory):
        """Test RAM usage calculation."""
        # Mock virtual_memory to return a tuple where index 3 is used memory in bytes
        mock_virtual_memory.return_value = (
            16000000000,  # total
            8000000000,   # available
            50.0,         # percent
            8000000000,   # used (index 3)
            0, 0, 0, 0, 0, 0
        )
        
        result = _get_ram_usage_mb()
        
        # 8000000000 bytes / 1000000 = 8000 MB
        self.assertEqual(result, 8000.0)
        mock_virtual_memory.assert_called_once()


class TestHardwareLookup(unittest.TestCase):
    """Test cases for hardware lookup dictionaries."""

    def test_hardware_lookup_structure(self):
        """Test that hardware_lookup has expected GPU types."""
        expected_gpus = [
            "NVIDIA H100",
            "NVIDIA A100",
            "NVIDIA V100",
            "NVIDIA K80",
            "NVIDIA T4",
            "NVIDIA A10G"
        ]
        for gpu in expected_gpus:
            self.assertIn(gpu, hardware_lookup)

    def test_fallback_gpu_resource_mapping_has_common_instances(self):
        """Test that fallback mapping includes common instance types."""
        common_instances = [
            "ml.p3.2xlarge",
            "ml.g4dn.xlarge",
            "ml.g5.2xlarge"
        ]
        for instance in common_instances:
            self.assertIn(instance, fallback_gpu_resource_mapping)
            self.assertIsInstance(fallback_gpu_resource_mapping[instance], int)


class TestGetGpuInfoFallback(unittest.TestCase):
    """Test cases for _get_gpu_info_fallback function."""

    def test_get_gpu_info_fallback_valid_instance(self):
        """Test fallback GPU info for valid instance type."""
        result = _get_gpu_info_fallback("ml.p3.2xlarge")
        self.assertEqual(result, 1)

    def test_get_gpu_info_fallback_multi_gpu_instance(self):
        """Test fallback GPU info for multi-GPU instance."""
        result = _get_gpu_info_fallback("ml.p3.8xlarge")
        self.assertEqual(result, 4)

    def test_get_gpu_info_fallback_invalid_instance(self):
        """Test fallback GPU info raises error for invalid instance."""
        with self.assertRaises(ValueError) as context:
            _get_gpu_info_fallback("ml.invalid.instance")
        self.assertIn("not GPU enabled", str(context.exception))


class TestCheckDiskSpace(unittest.TestCase):
    """Test cases for _check_disk_space function."""

    @patch('shutil.disk_usage')
    @patch('sagemaker.serve.utils.local_hardware.logger')
    def test_check_disk_space_warning_threshold(self, mock_logger, mock_disk_usage):
        """Test disk space check triggers warning at 50% threshold."""
        from sagemaker.serve.utils.local_hardware import _check_disk_space
        
        # Mock disk usage: (total, used, free)
        mock_disk_usage.return_value = (1000000000, 600000000, 400000000)
        
        _check_disk_space("/some/path")
        
        mock_logger.warning.assert_called_once()
        self.assertIn("percent of disk space used", mock_logger.warning.call_args[0][0])

    @patch('shutil.disk_usage')
    @patch('sagemaker.serve.utils.local_hardware.logger')
    def test_check_disk_space_no_warning_below_threshold(self, mock_logger, mock_disk_usage):
        """Test disk space check doesn't warn below 50% threshold."""
        from sagemaker.serve.utils.local_hardware import _check_disk_space
        
        # Mock disk usage: (total, used, free)
        mock_disk_usage.return_value = (1000000000, 400000000, 600000000)
        
        _check_disk_space("/some/path")
        
        mock_logger.warning.assert_not_called()


# Note: The following functions involve subprocess calls, docker checks, or AWS API calls
# and are better tested through integration tests:
# - _get_available_gpus (subprocess nvidia-smi)
# - _get_nb_instance (depends on _get_available_gpus)
# - _check_docker_disk_usage (platform-specific docker paths)
# - _get_gpu_info (AWS EC2 API calls)


if __name__ == "__main__":
    unittest.main()
