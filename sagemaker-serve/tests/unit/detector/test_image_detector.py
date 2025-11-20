"""
Unit tests for detector/image_detector.py module.
Tests framework detection, version casting, and container auto-detection.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from packaging import version as pkg_version
from sagemaker.serve.detector.image_detector import (
    auto_detect_container,
    _cast_to_compatible_version,
    _process_version,
    _later_version,
    _find_compatible_vs,
    _detect_framework_and_version,
    _get_model_base,
)


class TestGetModelBase(unittest.TestCase):
    """Test _get_model_base function."""

    def test_get_model_base_with_inheritance(self):
        """Test _get_model_base with a class that has inheritance."""
        class BaseModel:
            pass
        
        class MyModel(BaseModel):
            pass
        
        model = MyModel()
        result = _get_model_base(model)
        
        self.assertEqual(result, BaseModel)

    def test_get_model_base_without_inheritance(self):
        """Test _get_model_base with a class without inheritance."""
        class StandaloneModel:
            pass
        
        model = StandaloneModel()
        result = _get_model_base(model)
        
        # Should return the class itself when base is object
        self.assertEqual(result, StandaloneModel)

    def test_get_model_base_xgboost_special_case(self):
        """Test _get_model_base with XGBoost model."""
        class XGBModel:
            pass
        
        # Mock XGBoost module
        model = XGBModel()
        model.__class__.__module__ = 'xgboost.sklearn'
        
        result = _get_model_base(model)
        
        # Should return the class itself for XGBoost
        self.assertEqual(result, XGBModel)


class TestDetectFrameworkAndVersion(unittest.TestCase):
    """Test _detect_framework_and_version function."""

    def test_detect_pytorch(self):
        """Test detection of PyTorch framework."""
        import sys
        mock_torch = Mock()
        mock_torch.__version__ = "1.9.0+cpu"
        
        with patch.dict(sys.modules, {'torch': mock_torch}):
            fw, vs = _detect_framework_and_version("torch.nn.Module")
            
            self.assertEqual(fw, "pytorch")
            self.assertEqual(vs, "1.9.0")

    def test_detect_xgboost(self):
        """Test detection of XGBoost framework."""
        import sys
        mock_xgboost = Mock()
        mock_xgboost.__version__ = "1.5.0"
        
        with patch.dict(sys.modules, {'xgboost': mock_xgboost}):
            fw, vs = _detect_framework_and_version("xgboost.Booster")
            
            self.assertEqual(fw, "xgboost")
            self.assertEqual(vs, "1.5.0")

    def test_detect_tensorflow(self):
        """Test detection of TensorFlow framework."""
        import sys
        mock_tensorflow = Mock()
        mock_tensorflow.__version__ = "2.8.0"
        
        with patch.dict(sys.modules, {'tensorflow': mock_tensorflow}):
            fw, vs = _detect_framework_and_version("tensorflow.keras.Model")
            
            self.assertEqual(fw, "tensorflow")
            self.assertEqual(vs, "2.8.0")

    def test_detect_sklearn(self):
        """Test detection of scikit-learn framework."""
        import sys
        mock_sklearn = Mock()
        mock_sklearn.__version__ = "1.0.2"
        
        with patch.dict(sys.modules, {'sklearn': mock_sklearn}):
            fw, vs = _detect_framework_and_version("sklearn.linear_model.LogisticRegression")
            
            self.assertEqual(fw, "sklearn")
            self.assertEqual(vs, "1.0.2")

    def test_detect_unknown_framework(self):
        """Test detection with unknown framework."""
        with self.assertRaises(Exception) as context:
            _detect_framework_and_version("unknown.Model")
        
        self.assertIn("Unable to determine required container", str(context.exception))

    @patch('sagemaker.serve.detector.image_detector.logger')
    def test_detect_pytorch_import_error(self, mock_logger):
        """Test PyTorch detection when import fails."""
        import sys
        # Remove torch from sys.modules if it exists, then test import error
        with patch.dict(sys.modules, {'torch': None}):
            # This will trigger the ImportError in the try/except block
            fw, vs = _detect_framework_and_version("torch.nn.Module")
            
            self.assertEqual(fw, "pytorch")
            self.assertEqual(vs, "")
            mock_logger.warning.assert_called_once()


class TestProcessVersion(unittest.TestCase):
    """Test _process_version function."""

    def test_process_version_with_post(self):
        """Test _process_version with .post in version."""
        ver = pkg_version.parse("1.9.0.post1")
        result = _process_version(ver)
        
        self.assertEqual(result, "1.9.0-1")

    def test_process_version_without_post(self):
        """Test _process_version without .post."""
        ver = pkg_version.parse("1.9.0")
        result = _process_version(ver)
        
        self.assertEqual(result, "1.9.0")

    def test_process_version_none(self):
        """Test _process_version with None."""
        result = _process_version(None)
        
        self.assertIsNone(result)


class TestLaterVersion(unittest.TestCase):
    """Test _later_version function."""

    def test_later_version_true(self):
        """Test _later_version when current is later."""
        result = _later_version("1.9.2", "1.9.1")
        
        self.assertTrue(result)

    def test_later_version_false(self):
        """Test _later_version when current is earlier."""
        result = _later_version("1.9.1", "1.9.2")
        
        self.assertFalse(result)

    def test_later_version_equal(self):
        """Test _later_version when versions are equal."""
        result = _later_version("1.9.1", "1.9.1")
        
        self.assertFalse(result)

    def test_later_version_with_post(self):
        """Test _later_version with post versions."""
        result = _later_version("1.9-2", "1.9-1")
        
        self.assertTrue(result)


class TestFindCompatibleVs(unittest.TestCase):
    """Test _find_compatible_vs function."""

    def test_find_exact_match(self):
        """Test finding exact version match."""
        split_vs = [1, 9, 1]
        supported_vs = "1.9.1"
        
        upcast, downcast, found = _find_compatible_vs(split_vs, supported_vs)
        
        self.assertIsNone(upcast)
        self.assertIsNone(downcast)
        self.assertEqual(found, "1.9.1")

    def test_find_upcast_version(self):
        """Test finding upcast version."""
        split_vs = [1, 9, 1]
        supported_vs = "1.9.2"
        
        upcast, downcast, found = _find_compatible_vs(split_vs, supported_vs)
        
        self.assertEqual(upcast, "1.9.2")
        self.assertIsNone(downcast)
        self.assertIsNone(found)

    def test_find_downcast_version(self):
        """Test finding downcast version."""
        split_vs = [1, 9, 3]
        supported_vs = "1.9.2"
        
        upcast, downcast, found = _find_compatible_vs(split_vs, supported_vs)
        
        self.assertIsNone(upcast)
        self.assertEqual(downcast, "1.9.2")
        self.assertIsNone(found)

    def test_find_different_major_version(self):
        """Test with different major version."""
        split_vs = [2, 0, 0]
        supported_vs = "1.9.1"
        
        upcast, downcast, found = _find_compatible_vs(split_vs, supported_vs)
        
        self.assertIsNone(upcast)
        self.assertIsNone(downcast)
        self.assertIsNone(found)

    def test_find_with_post_version(self):
        """Test with post version format."""
        split_vs = [1, 9, 1]
        supported_vs = "1.9-1"
        
        upcast, downcast, found = _find_compatible_vs(split_vs, supported_vs)
        
        self.assertEqual(found, "1.9-1")


class TestCastToCompatibleVersion(unittest.TestCase):
    """Test _cast_to_compatible_version function."""

    @patch('sagemaker.serve.detector.image_detector.image_uris._config_for_framework_and_scope')
    def test_cast_exact_match(self, mock_config):
        """Test casting with exact version match."""
        mock_config.return_value = {
            'versions': {
                '1.9.0': {},
                '1.9.1': {},
                '1.10.0': {}
            }
        }
        
        result = _cast_to_compatible_version('pytorch', '1.9.1')
        
        self.assertIn('1.9.1', result)

    @patch('sagemaker.serve.detector.image_detector.image_uris._config_for_framework_and_scope')
    def test_cast_with_upcast(self, mock_config):
        """Test casting with upcast version."""
        mock_config.return_value = {
            'versions': {
                '1.9.0': {},
                '1.10.0': {},
                '1.11.0': {}
            }
        }
        
        result = _cast_to_compatible_version('pytorch', '1.9.5')
        
        # Should return downcast (1.9.0) and upcast (1.10.0)
        self.assertIsInstance(result, tuple)

    @patch('sagemaker.serve.detector.image_detector.image_uris._config_for_framework_and_scope')
    def test_cast_with_post_version(self, mock_config):
        """Test casting with .post version."""
        mock_config.return_value = {
            'versions': {
                '1.9.0': {},
                '1.9.0.post1': {},
                '1.10.0': {}
            }
        }
        
        result = _cast_to_compatible_version('pytorch', '1.9.0.post1')
        
        # Should handle post versions correctly
        self.assertIsInstance(result, tuple)


class TestAutoDetectContainer(unittest.TestCase):
    """Test auto_detect_container function."""

    def test_auto_detect_no_instance_type(self):
        """Test auto_detect_container without instance_type."""
        model = Mock()
        
        with self.assertRaises(ValueError) as context:
            auto_detect_container(model, "us-west-2", None)
        
        self.assertIn("Instance type is not specified", str(context.exception))

    @patch('sagemaker.serve.detector.image_detector._get_model_base')
    @patch('sagemaker.serve.detector.image_detector._detect_framework_and_version')
    @patch('sagemaker.serve.detector.image_detector._cast_to_compatible_version')
    @patch('sagemaker.serve.detector.image_detector.image_uris.retrieve')
    @patch('sagemaker.serve.detector.image_detector.platform.python_version_tuple')
    def test_auto_detect_pytorch_success(self, mock_py_tuple, mock_retrieve, mock_cast, mock_detect, mock_base):
        """Test successful PyTorch container detection."""
        mock_base.return_value = "torch.nn.Module"
        mock_detect.return_value = ("pytorch", "1.9.0")
        mock_cast.return_value = ("1.9.0", None, None)
        mock_py_tuple.return_value = ("3", "8", "10")
        mock_retrieve.return_value = "pytorch-inference:1.9.0-cpu-py38"
        
        model = Mock()
        dlc, fw, fw_version = auto_detect_container(model, "us-west-2", "ml.m5.large")
        
        self.assertEqual(dlc, "pytorch-inference:1.9.0-cpu-py38")
        self.assertEqual(fw, "pytorch")
        self.assertEqual(fw_version, "1.9.0")

    @patch('sagemaker.serve.detector.image_detector._get_model_base')
    @patch('sagemaker.serve.detector.image_detector._detect_framework_and_version')
    @patch('sagemaker.serve.detector.image_detector._cast_to_compatible_version')
    @patch('sagemaker.serve.detector.image_detector.image_uris.retrieve')
    @patch('sagemaker.serve.detector.image_detector.platform.python_version_tuple')
    def test_auto_detect_sklearn_uses_py3(self, mock_py_tuple, mock_retrieve, mock_cast, mock_detect, mock_base):
        """Test sklearn uses py3 instead of specific Python version."""
        mock_base.return_value = "sklearn.base.BaseEstimator"
        mock_detect.return_value = ("sklearn", "1.0.2")
        mock_cast.return_value = ("1.0.2", None, None)
        mock_py_tuple.return_value = ("3", "8", "10")
        mock_retrieve.return_value = "sklearn-inference:1.0.2-cpu-py3"
        
        model = Mock()
        dlc, fw, fw_version = auto_detect_container(model, "us-west-2", "ml.m5.large")
        
        # Verify sklearn uses py3
        mock_retrieve.assert_called_with(
            framework="sklearn",
            region="us-west-2",
            version="1.0.2",
            image_scope="inference",
            py_version="py3",
            instance_type="ml.m5.large"
        )

    @patch('sagemaker.serve.detector.image_detector._get_model_base')
    @patch('sagemaker.serve.detector.image_detector._detect_framework_and_version')
    @patch('sagemaker.serve.detector.image_detector._cast_to_compatible_version')
    @patch('sagemaker.serve.detector.image_detector.image_uris.retrieve')
    @patch('sagemaker.serve.detector.image_detector.platform.python_version_tuple')
    def test_auto_detect_fallback_to_latest(self, mock_py_tuple, mock_retrieve, mock_cast, mock_detect, mock_base):
        """Test fallback to latest version when requested version not available."""
        mock_base.return_value = "torch.nn.Module"
        mock_detect.return_value = ("pytorch", "1.9.0")
        mock_cast.return_value = ("1.9.0", None, None)
        mock_py_tuple.return_value = ("3", "8", "10")
        
        # First call fails, second succeeds with latest version
        mock_retrieve.side_effect = [
            ValueError("Version not found"),
            "pytorch-inference:1.10.0-cpu-py38"
        ]
        
        with patch('sagemaker.serve.detector.image_detector.image_uris._config_for_framework_and_scope') as mock_config:
            mock_config.return_value = {
                'versions': {
                    '1.8.0': {},
                    '1.9.0': {},
                    '1.10.0': {}
                }
            }
            
            model = Mock()
            dlc, fw, fw_version = auto_detect_container(model, "us-west-2", "ml.m5.large")
            
            self.assertEqual(dlc, "pytorch-inference:1.10.0-cpu-py38")

    @patch('sagemaker.serve.detector.image_detector._get_model_base')
    @patch('sagemaker.serve.detector.image_detector._detect_framework_and_version')
    @patch('sagemaker.serve.detector.image_detector._cast_to_compatible_version')
    @patch('sagemaker.serve.detector.image_detector.image_uris.retrieve')
    @patch('sagemaker.serve.detector.image_detector.platform.python_version_tuple')
    def test_auto_detect_no_compatible_version(self, mock_py_tuple, mock_retrieve, mock_cast, mock_detect, mock_base):
        """Test when no compatible DLC version is found."""
        mock_base.return_value = "torch.nn.Module"
        mock_detect.return_value = ("pytorch", "1.9.0")
        mock_cast.return_value = (None, None, None)
        mock_py_tuple.return_value = ("3", "8", "10")
        mock_retrieve.side_effect = ValueError("No version found")
        
        model = Mock()
        
        with self.assertRaises(ValueError) as context:
            auto_detect_container(model, "us-west-2", "ml.m5.large")
        
        self.assertIn("Unable to auto detect a DLC", str(context.exception))


if __name__ == "__main__":
    unittest.main()
