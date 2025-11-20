"""Unit tests for sagemaker.serve.utils.hardware_detector module."""
import unittest
from unittest.mock import Mock, patch
from sagemaker.serve.utils.hardware_detector import (
    _format_instance_type,
    MIB_CONVERSION_FACTOR,
    MEMORY_BUFFER_MULTIPLIER,
)


class TestFormatInstanceType(unittest.TestCase):
    """Test cases for _format_instance_type function."""

    def test_format_sagemaker_instance_type(self):
        """Test formatting SageMaker instance type."""
        result = _format_instance_type("ml.p3.2xlarge")
        self.assertEqual(result, "p3.2xlarge")

    def test_format_ec2_instance_type(self):
        """Test formatting EC2 instance type (no ml prefix)."""
        result = _format_instance_type("p3.2xlarge")
        self.assertEqual(result, "p3.2xlarge")

    def test_format_instance_type_with_multiple_dots(self):
        """Test formatting instance type with multiple dots."""
        result = _format_instance_type("ml.g4dn.xlarge")
        self.assertEqual(result, "g4dn.xlarge")

    def test_format_instance_type_single_part(self):
        """Test formatting instance type with single part."""
        result = _format_instance_type("xlarge")
        self.assertEqual(result, "xlarge")


class TestConstants(unittest.TestCase):
    """Test cases for module constants."""

    def test_mib_conversion_factor(self):
        """Test MIB conversion factor is correct."""
        # 1 byte = 0.00000095367431640625 MiB
        self.assertAlmostEqual(MIB_CONVERSION_FACTOR, 0.00000095367431640625, places=20)

    def test_memory_buffer_multiplier(self):
        """Test memory buffer multiplier is 20%."""
        self.assertEqual(MEMORY_BUFFER_MULTIPLIER, 1.2)


class TestGetGpuInfoFallback(unittest.TestCase):
    """Test cases for _get_gpu_info_fallback function."""

    @patch('sagemaker.serve.utils.hardware_detector.instance_types_gpu_info')
    def test_get_gpu_info_fallback_valid_instance(self, mock_gpu_info):
        """Test fallback GPU info for valid instance type."""
        from sagemaker.serve.utils.hardware_detector import _get_gpu_info_fallback
        
        mock_gpu_info.retrieve.return_value = {
            "ml.p3.2xlarge": {
                "Count": 1,
                "TotalGpuMemoryInMiB": 16384
            }
        }
        
        result = _get_gpu_info_fallback("ml.p3.2xlarge", "us-west-2")
        
        self.assertEqual(result, (1, 16384))
        mock_gpu_info.retrieve.assert_called_once_with("us-west-2")

    @patch('sagemaker.serve.utils.hardware_detector.instance_types_gpu_info')
    def test_get_gpu_info_fallback_invalid_instance(self, mock_gpu_info):
        """Test fallback GPU info raises error for invalid instance."""
        from sagemaker.serve.utils.hardware_detector import _get_gpu_info_fallback
        
        mock_gpu_info.retrieve.return_value = {}
        
        with self.assertRaises(ValueError) as context:
            _get_gpu_info_fallback("ml.invalid.instance", "us-west-2")
        
        self.assertIn("not GPU enabled", str(context.exception))

    @patch('sagemaker.serve.utils.hardware_detector.instance_types_gpu_info')
    def test_get_gpu_info_fallback_multi_gpu(self, mock_gpu_info):
        """Test fallback GPU info for multi-GPU instance."""
        from sagemaker.serve.utils.hardware_detector import _get_gpu_info_fallback
        
        mock_gpu_info.retrieve.return_value = {
            "ml.p3.8xlarge": {
                "Count": 4,
                "TotalGpuMemoryInMiB": 65536
            }
        }
        
        result = _get_gpu_info_fallback("ml.p3.8xlarge", "us-east-1")
        
        self.assertEqual(result, (4, 65536))


# Note: _total_inference_model_size_mib requires the 'accelerate' package
# which is an optional dependency. This function is better tested through
# integration tests with the full HuggingFace extras installed.
# 
# Function not unit tested here (requires accelerate package):
# - _total_inference_model_size_mib


# Note: The following functions involve AWS EC2 API calls and are better tested
# through integration tests:
# - _get_gpu_info (AWS EC2 describe_instance_types API)


if __name__ == "__main__":
    unittest.main()
