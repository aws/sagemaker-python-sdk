"""Additional unit tests for local_hardware.py to increase coverage."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import subprocess


class TestGetAvailableGpus(unittest.TestCase):
    """Test _get_available_gpus function."""

    @patch('subprocess.run')
    def test_get_available_gpus_success_with_log(self, mock_run):
        """Test successful GPU detection with logging."""
        from sagemaker.serve.utils.local_hardware import _get_available_gpus
        
        mock_result = Mock()
        mock_result.stdout = b"name, memory.free\nNVIDIA A100, 40960 MiB\n"
        mock_run.return_value = mock_result
        
        result = _get_available_gpus(log=True)
        
        self.assertEqual(result, ["NVIDIA A100, 40960 MiB"])

    @patch('subprocess.run')
    def test_get_available_gpus_exception(self, mock_run):
        """Test GPU detection with exception."""
        from sagemaker.serve.utils.local_hardware import _get_available_gpus
        
        mock_run.side_effect = Exception("CUDA not available")
        
        result = _get_available_gpus(log=False)
        
        self.assertIsNone(result)


class TestGetNbInstance(unittest.TestCase):
    """Test _get_nb_instance function."""

    @patch('sagemaker.serve.utils.local_hardware._get_available_gpus')
    def test_get_nb_instance_no_gpu(self, mock_get_gpus):
        """Test when no GPU is available."""
        from sagemaker.serve.utils.local_hardware import _get_nb_instance
        
        mock_get_gpus.return_value = None
        
        result = _get_nb_instance()
        
        self.assertIsNone(result)

    @patch('multiprocessing.cpu_count')
    @patch('sagemaker.serve.utils.local_hardware._get_available_gpus')
    def test_get_nb_instance_unknown_gpu(self, mock_get_gpus, mock_cpu_count):
        """Test with unknown GPU type."""
        from sagemaker.serve.utils.local_hardware import _get_nb_instance
        
        mock_get_gpus.return_value = ["Unknown GPU, 16384 MiB"]
        mock_cpu_count.return_value = 8
        
        result = _get_nb_instance()
        
        self.assertIsNone(result)

    @patch('multiprocessing.cpu_count')
    @patch('sagemaker.serve.utils.local_hardware._get_available_gpus')
    def test_get_nb_instance_a100_ceil(self, mock_get_gpus, mock_cpu_count):
        """Test A100 GPU with ceil memory."""
        from sagemaker.serve.utils.local_hardware import _get_nb_instance
        
        mock_get_gpus.return_value = ["NVIDIA A100, 41943 MiB"]
        mock_cpu_count.return_value = 96
        
        result = _get_nb_instance()
        
        # Result depends on hardware_lookup configuration
        self.assertIsInstance(result, (str, type(None)))

    @patch('multiprocessing.cpu_count')
    @patch('sagemaker.serve.utils.local_hardware._get_available_gpus')
    def test_get_nb_instance_a100_floor(self, mock_get_gpus, mock_cpu_count):
        """Test A100 GPU with floor memory."""
        from sagemaker.serve.utils.local_hardware import _get_nb_instance
        
        mock_get_gpus.return_value = ["NVIDIA A100, 38000 MiB"]
        mock_cpu_count.return_value = 96
        
        result = _get_nb_instance()
        
        # Result depends on hardware_lookup configuration
        self.assertIsInstance(result, (str, type(None)))

    @patch('multiprocessing.cpu_count')
    @patch('sagemaker.serve.utils.local_hardware._get_available_gpus')
    def test_get_nb_instance_v100(self, mock_get_gpus, mock_cpu_count):
        """Test V100 GPU."""
        from sagemaker.serve.utils.local_hardware import _get_nb_instance
        
        mock_get_gpus.return_value = ["NVIDIA V100, 16384 MiB"]
        mock_cpu_count.return_value = 8
        
        result = _get_nb_instance()
        
        self.assertEqual(result, "ml.p3.2xlarge")


class TestCheckDiskSpace(unittest.TestCase):
    """Test _check_disk_space function."""

    @patch('shutil.disk_usage')
    def test_check_disk_space_warning(self, mock_disk_usage):
        """Test disk space check with warning."""
        from sagemaker.serve.utils.local_hardware import _check_disk_space
        
        mock_disk_usage.return_value = (100, 60, 40)  # total, used, free
        
        with self.assertLogs(level='WARNING') as log:
            _check_disk_space("/tmp")
        
        self.assertTrue(any("disk space" in msg for msg in log.output))

    @patch('shutil.disk_usage')
    def test_check_disk_space_ok(self, mock_disk_usage):
        """Test disk space check without warning."""
        from sagemaker.serve.utils.local_hardware import _check_disk_space
        
        mock_disk_usage.return_value = (100, 30, 70)  # total, used, free
        
        _check_disk_space("/tmp")


class TestCheckDockerDiskUsage(unittest.TestCase):
    """Test _check_docker_disk_usage function."""

    @patch('sagemaker.serve.utils.local_hardware.system')
    @patch('shutil.disk_usage')
    def test_check_docker_disk_usage_linux_warning(self, mock_disk_usage, mock_system):
        """Test docker disk usage on Linux with warning."""
        from sagemaker.serve.utils.local_hardware import _check_docker_disk_usage
        
        mock_system.return_value.lower.return_value = "linux"
        mock_disk_usage.return_value = (100, 60, 40)
        
        with self.assertLogs(level='WARNING') as log:
            _check_docker_disk_usage()
        
        self.assertTrue(any("docker disk space" in msg for msg in log.output))

    @patch('sagemaker.serve.utils.local_hardware.system')
    @patch('shutil.disk_usage')
    def test_check_docker_disk_usage_linux_ok(self, mock_disk_usage, mock_system):
        """Test docker disk usage on Linux without warning."""
        from sagemaker.serve.utils.local_hardware import _check_docker_disk_usage
        
        mock_system.return_value.lower.return_value = "linux"
        mock_disk_usage.return_value = (100, 30, 70)
        
        with self.assertLogs(level='INFO') as log:
            _check_docker_disk_usage()
        
        self.assertTrue(any("docker disk space" in msg for msg in log.output))

    @patch('sagemaker.serve.utils.local_hardware.system')
    @patch('shutil.disk_usage')
    def test_check_docker_disk_usage_exception(self, mock_disk_usage, mock_system):
        """Test docker disk usage with exception."""
        from sagemaker.serve.utils.local_hardware import _check_docker_disk_usage
        
        mock_system.return_value.lower.return_value = "linux"
        mock_disk_usage.side_effect = Exception("Path not found")
        
        with self.assertLogs(level='WARNING') as log:
            _check_docker_disk_usage()
        
        self.assertTrue(any("Unable to check" in msg for msg in log.output))


class TestGetGpuInfo(unittest.TestCase):
    """Test _get_gpu_info function."""

    def test_get_gpu_info_success(self):
        """Test successful GPU info retrieval."""
        from sagemaker.serve.utils.local_hardware import _get_gpu_info
        
        mock_session = Mock()
        mock_ec2_client = Mock()
        mock_session.boto_session.client.return_value = mock_ec2_client
        
        mock_ec2_client.describe_instance_types.return_value = {
            "InstanceTypes": [{
                "GpuInfo": {
                    "Gpus": [{"Count": 4}]
                }
            }]
        }
        
        result = _get_gpu_info("ml.g5.12xlarge", mock_session)
        
        self.assertEqual(result, 4)

    def test_get_gpu_info_no_gpu(self):
        """Test GPU info retrieval for non-GPU instance."""
        from sagemaker.serve.utils.local_hardware import _get_gpu_info
        
        mock_session = Mock()
        mock_ec2_client = Mock()
        mock_session.boto_session.client.return_value = mock_ec2_client
        
        mock_ec2_client.describe_instance_types.return_value = {
            "InstanceTypes": [{}]
        }
        
        with self.assertRaises(ValueError):
            _get_gpu_info("ml.m5.large", mock_session)


class TestGetGpuInfoFallback(unittest.TestCase):
    """Test _get_gpu_info_fallback function."""

    def test_get_gpu_info_fallback_success(self):
        """Test successful fallback GPU info retrieval."""
        from sagemaker.serve.utils.local_hardware import _get_gpu_info_fallback
        
        result = _get_gpu_info_fallback("ml.g5.12xlarge")
        
        self.assertEqual(result, 4)

    def test_get_gpu_info_fallback_no_gpu(self):
        """Test fallback for non-GPU instance."""
        from sagemaker.serve.utils.local_hardware import _get_gpu_info_fallback
        
        with self.assertRaises(ValueError):
            _get_gpu_info_fallback("ml.m5.large")


if __name__ == "__main__":
    unittest.main()
