"""Unit tests for sagemaker.serve.validations.check_image_and_hardware_type module."""
import unittest
from unittest.mock import patch
from sagemaker.serve.validations.check_image_and_hardware_type import (
    validate_image_uri_and_hardware,
    detect_hardware_type_of_instance,
    detect_triton_image_hardware_type,
    detect_torchserve_image_hardware_type,
    GPU_INSTANCE_FAMILIES,
    INF1_INSTANCE_FAMILIES,
    INF2_INSTANCE_FAMILIES,
    GRAVITON_INSTANCE_FAMILIES,
)
from sagemaker.serve.utils.types import ModelServer, HardwareType


class TestDetectHardwareTypeOfInstance(unittest.TestCase):
    """Test cases for detect_hardware_type_of_instance function."""

    def test_gpu_instance_types(self):
        """Test GPU instance type detection."""
        for family in GPU_INSTANCE_FAMILIES:
            instance_type = f"{family}.xlarge"
            result = detect_hardware_type_of_instance(instance_type)
            self.assertEqual(result, HardwareType.GPU)

    def test_inf1_instance_types(self):
        """Test Inferentia 1 instance type detection."""
        for family in INF1_INSTANCE_FAMILIES:
            instance_type = f"{family}.xlarge"
            result = detect_hardware_type_of_instance(instance_type)
            self.assertEqual(result, HardwareType.INFERENTIA_1)

    def test_inf2_instance_types(self):
        """Test Inferentia 2 instance type detection."""
        for family in INF2_INSTANCE_FAMILIES:
            instance_type = f"{family}.xlarge"
            result = detect_hardware_type_of_instance(instance_type)
            self.assertEqual(result, HardwareType.INFERENTIA_2)

    def test_graviton_instance_types(self):
        """Test Graviton instance type detection."""
        for family in GRAVITON_INSTANCE_FAMILIES:
            instance_type = f"{family}.xlarge"
            result = detect_hardware_type_of_instance(instance_type)
            self.assertEqual(result, HardwareType.GRAVITON)

    def test_cpu_instance_types(self):
        """Test CPU instance type detection."""
        cpu_instances = ["ml.m5.xlarge", "ml.c5.2xlarge", "ml.t3.medium"]
        for instance_type in cpu_instances:
            result = detect_hardware_type_of_instance(instance_type)
            self.assertEqual(result, HardwareType.CPU)


class TestDetectTritonImageHardwareType(unittest.TestCase):
    """Test cases for detect_triton_image_hardware_type function."""

    def test_cpu_image(self):
        """Test CPU Triton image detection."""
        image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/triton-cpu:latest"
        result = detect_triton_image_hardware_type(image_uri)
        self.assertEqual(result, HardwareType.CPU)

    def test_gpu_image(self):
        """Test GPU Triton image detection."""
        image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/triton-gpu:latest"
        result = detect_triton_image_hardware_type(image_uri)
        self.assertEqual(result, HardwareType.GPU)

    def test_default_gpu_image(self):
        """Test default Triton image (no cpu in name) is GPU."""
        image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/triton:latest"
        result = detect_triton_image_hardware_type(image_uri)
        self.assertEqual(result, HardwareType.GPU)


class TestDetectTorchserveImageHardwareType(unittest.TestCase):
    """Test cases for detect_torchserve_image_hardware_type function."""

    def test_neuronx_image(self):
        """Test Neuronx (Inferentia 2) image detection."""
        image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/pytorch-neuronx:latest"
        result = detect_torchserve_image_hardware_type(image_uri)
        self.assertEqual(result, HardwareType.INFERENTIA_2)

    def test_neuron_image(self):
        """Test Neuron (Inferentia 1) image detection."""
        image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/pytorch-neuron:latest"
        result = detect_torchserve_image_hardware_type(image_uri)
        self.assertEqual(result, HardwareType.INFERENTIA_1)

    def test_graviton_image(self):
        """Test Graviton image detection."""
        image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/pytorch-graviton:latest"
        result = detect_torchserve_image_hardware_type(image_uri)
        self.assertEqual(result, HardwareType.GRAVITON)

    def test_cpu_image(self):
        """Test CPU image detection."""
        image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/pytorch-cpu:latest"
        result = detect_torchserve_image_hardware_type(image_uri)
        self.assertEqual(result, HardwareType.CPU)

    def test_gpu_image(self):
        """Test GPU image detection (default)."""
        image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/pytorch:latest"
        result = detect_torchserve_image_hardware_type(image_uri)
        self.assertEqual(result, HardwareType.GPU)


class TestValidateImageUriAndHardware(unittest.TestCase):
    """Test cases for validate_image_uri_and_hardware function."""

    def test_xgboost_skips_validation(self):
        """Test that xgboost images skip validation."""
        # Should not raise any warnings
        result = validate_image_uri_and_hardware(
            "xgboost:latest",
            "ml.m5.xlarge",
            ModelServer.TORCHSERVE
        )
        self.assertIsNone(result)

    @patch('sagemaker.serve.validations.check_image_and_hardware_type.logger')
    def test_matching_hardware_types(self, mock_logger):
        """Test matching hardware types don't trigger warnings."""
        validate_image_uri_and_hardware(
            "pytorch-cpu:latest",
            "ml.m5.xlarge",
            ModelServer.TORCHSERVE
        )
        mock_logger.warning.assert_not_called()

    @patch('sagemaker.serve.validations.check_image_and_hardware_type.logger')
    def test_mismatched_hardware_types(self, mock_logger):
        """Test mismatched hardware types trigger warnings."""
        validate_image_uri_and_hardware(
            "pytorch-gpu:latest",
            "ml.m5.xlarge",  # CPU instance
            ModelServer.TORCHSERVE
        )
        mock_logger.warning.assert_called_once()

    @patch('sagemaker.serve.validations.check_image_and_hardware_type.logger')
    def test_triton_validation(self, mock_logger):
        """Test Triton image validation."""
        validate_image_uri_and_hardware(
            "triton-cpu:latest",
            "ml.m5.xlarge",
            ModelServer.TRITON
        )
        mock_logger.warning.assert_not_called()

    @patch('sagemaker.serve.validations.check_image_and_hardware_type.logger')
    def test_unsupported_model_server_skips_validation(self, mock_logger):
        """Test unsupported model servers skip validation."""
        validate_image_uri_and_hardware(
            "some-image:latest",
            "ml.m5.xlarge",
            ModelServer.DJL_SERVING
        )
        mock_logger.info.assert_called_once()
        mock_logger.warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
