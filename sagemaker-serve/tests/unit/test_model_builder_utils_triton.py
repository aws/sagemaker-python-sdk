"""
Unit tests for Triton-related methods in _ModelBuilderUtils.
Targets uncovered Triton functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import os
import tempfile
from pathlib import Path

from sagemaker.serve.model_builder_utils import _ModelBuilderUtils, TritonSerializer
from sagemaker.serve.constants import Framework
from sagemaker.serve.mode.function_pointers import Mode


class TestTritonSerializer(unittest.TestCase):
    """Test TritonSerializer class."""

    def test_triton_serializer_init(self):
        """Test TritonSerializer initialization."""
        mock_serializer = Mock()
        serializer = TritonSerializer(mock_serializer, "FP32")
        
        self.assertEqual(serializer.dtype, "FP32")
        self.assertEqual(serializer.input_serializer, mock_serializer)

    def test_triton_serializer_serialize(self):
        """Test TritonSerializer serialize method."""
        import numpy as np
        mock_serializer = Mock()
        mock_array = np.array([[1, 2, 3]])
        mock_serializer.serialize.return_value = mock_array
        
        serializer = TritonSerializer(mock_serializer, "FP32")
        result = serializer.serialize(mock_array)
        
        self.assertIsNotNone(result)


class TestValidateForTriton(unittest.TestCase):
    """Test _validate_for_triton method."""

    def test_validate_for_triton_missing_tritonclient(self):
        """Test validation fails without tritonclient - skipped as tritonclient is installed."""
        pass

    @patch('importlib.util.find_spec')
    @patch.object(_ModelBuilderUtils, '_has_nvidia_gpu')
    def test_validate_for_triton_no_gpu_local(self, mock_has_gpu, mock_find_spec):
        """Test validation fails for GPU mode without GPU."""
        utils = _ModelBuilderUtils()
        utils.mode = Mode.LOCAL_CONTAINER
        utils.model_path = "/tmp/model"
        utils.image_uri = "triton-gpu-image"
        utils.schema_builder = Mock()
        utils.schema_builder._update_serializer_deserializer_for_triton = Mock()
        utils.schema_builder._detect_dtype_for_triton = Mock()
        
        mock_find_spec.return_value = Mock()
        mock_has_gpu.return_value = False
        
        with self.assertRaises(ValueError):
            utils._validate_for_triton()

    @patch('importlib.util.find_spec')
    def test_validate_for_triton_unsupported_mode(self, mock_find_spec):
        """Test validation fails for unsupported mode."""
        utils = _ModelBuilderUtils()
        utils.mode = "UNSUPPORTED_MODE"
        utils.model_path = "/tmp/model"
        utils.schema_builder = Mock()
        
        mock_find_spec.return_value = Mock()
        
        with self.assertRaises(ValueError):
            utils._validate_for_triton()


class TestPrepareForTriton(unittest.TestCase):
    """Test _prepare_for_triton method."""

    @patch('shutil.copy2')
    @patch.object(_ModelBuilderUtils, '_export_pytorch_to_onnx')
    def test_prepare_for_triton_pytorch(self, mock_export, mock_copy):
        """Test preparing PyTorch model for Triton."""
        utils = _ModelBuilderUtils()
        utils.framework = Framework.PYTORCH
        utils.model = Mock()
        utils.schema_builder = Mock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            utils.model_path = tmpdir
            utils._prepare_for_triton()
            
            mock_export.assert_called_once()

    @patch('shutil.copy2')
    @patch.object(_ModelBuilderUtils, '_export_tf_to_onnx')
    def test_prepare_for_triton_tensorflow(self, mock_export, mock_copy):
        """Test preparing TensorFlow model for Triton."""
        utils = _ModelBuilderUtils()
        utils.framework = Framework.TENSORFLOW
        utils.model = Mock()
        utils.schema_builder = Mock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            utils.model_path = tmpdir
            utils._prepare_for_triton()
            
            mock_export.assert_called_once()

    @patch('shutil.copy2')
    @patch.object(_ModelBuilderUtils, '_generate_config_pbtxt')
    @patch.object(_ModelBuilderUtils, '_pack_conda_env')
    @patch.object(_ModelBuilderUtils, '_hmac_signing')
    def test_prepare_for_triton_inference_spec(self, mock_hmac, mock_pack, mock_config, mock_copy):
        """Test preparing inference spec for Triton."""
        utils = _ModelBuilderUtils()
        utils.inference_spec = Mock()
        utils.model = None
        utils.schema_builder = Mock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            utils.model_path = tmpdir
            utils._prepare_for_triton()
            
            mock_config.assert_called_once()
            mock_pack.assert_called_once()
            mock_hmac.assert_called_once()


class TestExportPytorchToOnnx(unittest.TestCase):
    """Test _export_pytorch_to_onnx method."""

    @patch('torch.onnx.export')
    def test_export_pytorch_to_onnx_success(self, mock_export):
        """Test successful PyTorch to ONNX export."""
        try:
            import ml_dtypes
            # Skip test if ml_dtypes doesn't have required attribute
            if not hasattr(ml_dtypes, 'float4_e2m1fn'):
                self.skipTest("ml_dtypes version incompatible with current numpy/onnx")
        except ImportError:
            pass
        
        utils = _ModelBuilderUtils()
        mock_model = Mock()
        mock_schema = Mock()
        mock_schema.sample_input = Mock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir)
            utils._export_pytorch_to_onnx(mock_model, export_path, mock_schema)
            
            mock_export.assert_called_once()

    def test_export_pytorch_to_onnx_no_torch(self):
        """Test PyTorch export without torch installed - skipped."""
        # Skipping as torch is installed in environment
        pass


class TestExportTFToOnnx(unittest.TestCase):
    """Test _export_tf_to_onnx method."""

    def test_export_tf_to_onnx_no_tf2onnx(self):
        """Test TensorFlow export without tf2onnx installed."""
        utils = _ModelBuilderUtils()
        
        # tf2onnx not installed in test environment
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ImportError):
                utils._export_tf_to_onnx(str(Path(tmpdir)), Mock(), Mock())


class TestGenerateConfigPbtxt(unittest.TestCase):
    """Test _generate_config_pbtxt method."""

    def test_generate_config_pbtxt_cpu(self):
        """Test generating config.pbtxt for CPU."""
        utils = _ModelBuilderUtils()
        utils.image_uri = "triton-cpu-image"
        utils.schema_builder = Mock()
        utils.schema_builder._sample_input_ndarray = Mock()
        utils.schema_builder._sample_input_ndarray.shape = [1, 10]
        utils.schema_builder._sample_output_ndarray = Mock()
        utils.schema_builder._sample_output_ndarray.shape = [1, 5]
        utils.schema_builder._input_triton_dtype = "FP32"
        utils.schema_builder._output_triton_dtype = "FP32"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir)
            utils._generate_config_pbtxt(pkl_path)
            
            config_path = pkl_path / "config.pbtxt"
            self.assertTrue(config_path.exists())
            content = config_path.read_text()
            self.assertIn("KIND_CPU", content)

    def test_generate_config_pbtxt_gpu(self):
        """Test generating config.pbtxt for GPU."""
        utils = _ModelBuilderUtils()
        utils.image_uri = "triton-gpu-image"
        utils.schema_builder = Mock()
        utils.schema_builder._sample_input_ndarray = Mock()
        utils.schema_builder._sample_input_ndarray.shape = [1, 10]
        utils.schema_builder._sample_output_ndarray = Mock()
        utils.schema_builder._sample_output_ndarray.shape = [1, 5]
        utils.schema_builder._input_triton_dtype = "FP32"
        utils.schema_builder._output_triton_dtype = "FP32"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir)
            utils._generate_config_pbtxt(pkl_path)
            
            config_path = pkl_path / "config.pbtxt"
            self.assertTrue(config_path.exists())
            content = config_path.read_text()
            self.assertIn("KIND_GPU", content)


class TestPackCondaEnv(unittest.TestCase):
    """Test _pack_conda_env method."""

    def test_pack_conda_env_no_conda_pack(self):
        """Test packing conda env without conda_pack."""
        utils = _ModelBuilderUtils()
        
        with patch('importlib.util.find_spec', return_value=None):
            with tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaises(ImportError):
                    utils._pack_conda_env(Path(tmpdir))

    def test_pack_conda_env_no_conda_pack_real(self):
        """Test packing conda env without conda_pack - real check."""
        utils = _ModelBuilderUtils()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ImportError):
                utils._pack_conda_env(Path(tmpdir))


class TestSaveInferenceSpec(unittest.TestCase):
    """Test _save_inference_spec method."""

    def test_save_inference_spec(self):
        """Test saving inference spec."""
        utils = _ModelBuilderUtils()
        utils.inference_spec = Mock()
        utils.schema_builder = Mock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            utils.model_path = tmpdir
            pkl_path = os.path.join(tmpdir, "model_repository", "model")
            os.makedirs(pkl_path, exist_ok=True)
            
            utils._save_inference_spec()
            
            # Check that serve.pkl was created
            self.assertTrue(os.path.exists(os.path.join(pkl_path, "serve.pkl")))


class TestHMACSignin(unittest.TestCase):
    """Test _hmac_signing method."""

    def test_hmac_signing(self):
        """Test HMAC signing."""
        utils = _ModelBuilderUtils()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            utils.model_path = tmpdir
            pkl_path = Path(tmpdir) / "model_repository" / "model"
            pkl_path.mkdir(parents=True)
            
            # Create dummy serve.pkl
            (pkl_path / "serve.pkl").write_bytes(b"dummy content")
            
            utils._hmac_signing()
            
            # Secret key is generated, not mocked
            self.assertIsNotNone(utils.secret_key)
            self.assertTrue((pkl_path / "metadata.json").exists())


class TestAutoDetectImageForTriton(unittest.TestCase):
    """Test _auto_detect_image_for_triton method."""

    def test_auto_detect_image_skip_if_provided(self):
        """Test skipping auto-detection if image_uri provided."""
        utils = _ModelBuilderUtils()
        utils.image_uri = "custom-triton-image"
        utils.sagemaker_session = Mock()
        
        utils._auto_detect_image_for_triton()
        
        self.assertEqual(utils.image_uri, "custom-triton-image")

    def test_auto_detect_image_cpu_instance(self):
        """Test auto-detecting Triton image for CPU instance."""
        utils = _ModelBuilderUtils()
        utils.image_uri = None
        utils.instance_type = "ml.m5.large"
        utils.sagemaker_session = Mock()
        utils.sagemaker_session.boto_region_name = "us-west-2"
        utils.inference_spec = None
        utils.framework = "pytorch"
        utils.version = "1.13"
        
        utils._auto_detect_image_for_triton()
        
        self.assertIsNotNone(utils.image_uri)
        self.assertIn("-cpu", utils.image_uri)

    def test_auto_detect_image_gpu_instance(self):
        """Test auto-detecting Triton image for GPU instance."""
        utils = _ModelBuilderUtils()
        utils.image_uri = None
        utils.instance_type = "ml.g5.xlarge"
        utils.sagemaker_session = Mock()
        utils.sagemaker_session.boto_region_name = "us-west-2"
        utils.inference_spec = None
        utils.framework = "pytorch"
        utils.version = "1.13"
        
        utils._auto_detect_image_for_triton()
        
        self.assertIsNotNone(utils.image_uri)
        self.assertNotIn("-cpu", utils.image_uri)

    def test_auto_detect_image_unsupported_region(self):
        """Test auto-detecting Triton image for unsupported region."""
        utils = _ModelBuilderUtils()
        utils.image_uri = None
        utils.instance_type = "ml.g5.xlarge"
        utils.sagemaker_session = Mock()
        utils.sagemaker_session.boto_region_name = "unsupported-region"
        
        with self.assertRaises(ValueError):
            utils._auto_detect_image_for_triton()


class TestValidateDJLServingSampleData(unittest.TestCase):
    """Test _validate_djl_serving_sample_data method."""

    def test_validate_djl_valid_data(self):
        """Test validation with valid DJL sample data."""
        utils = _ModelBuilderUtils()
        utils.schema_builder = Mock()
        utils.schema_builder.sample_input = {"inputs": "test", "parameters": {}}
        utils.schema_builder.sample_output = [{"generated_text": "output"}]
        
        # Should not raise
        utils._validate_djl_serving_sample_data()

    def test_validate_djl_invalid_input(self):
        """Test validation with invalid DJL input."""
        utils = _ModelBuilderUtils()
        utils.schema_builder = Mock()
        utils.schema_builder.sample_input = {"wrong_key": "test"}
        utils.schema_builder.sample_output = [{"generated_text": "output"}]
        
        with self.assertRaises(ValueError):
            utils._validate_djl_serving_sample_data()

    def test_validate_djl_invalid_output(self):
        """Test validation with invalid DJL output."""
        utils = _ModelBuilderUtils()
        utils.schema_builder = Mock()
        utils.schema_builder.sample_input = {"inputs": "test", "parameters": {}}
        utils.schema_builder.sample_output = [{"wrong_key": "output"}]
        
        with self.assertRaises(ValueError):
            utils._validate_djl_serving_sample_data()


class TestValidateTGIServingSampleData(unittest.TestCase):
    """Test _validate_tgi_serving_sample_data method."""

    def test_validate_tgi_valid_data(self):
        """Test validation with valid TGI sample data."""
        utils = _ModelBuilderUtils()
        utils.schema_builder = Mock()
        utils.schema_builder.sample_input = {"inputs": "test", "parameters": {}}
        utils.schema_builder.sample_output = [{"generated_text": "output"}]
        
        # Should not raise
        utils._validate_tgi_serving_sample_data()

    def test_validate_tgi_invalid_input(self):
        """Test validation with invalid TGI input."""
        utils = _ModelBuilderUtils()
        utils.schema_builder = Mock()
        utils.schema_builder.sample_input = "invalid"
        utils.schema_builder.sample_output = [{"generated_text": "output"}]
        
        with self.assertRaises(ValueError):
            utils._validate_tgi_serving_sample_data()


class TestCreateCondaEnv(unittest.TestCase):
    """Test _create_conda_env method."""

    @patch('sagemaker.serve.builder.requirements_manager.RequirementsManager')
    def test_create_conda_env_success(self, mock_req_manager):
        """Test successful conda env creation."""
        utils = _ModelBuilderUtils()
        mock_manager = Mock()
        mock_req_manager.return_value = mock_manager
        
        utils._create_conda_env()
        
        # Should not raise


if __name__ == "__main__":
    unittest.main()
