"""Unit tests for sagemaker.serve.marshalling.triton_translator module."""
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np


class TestNumpyTranslator(unittest.TestCase):
    """Test cases for NumpyTranslator class."""

    def setUp(self):
        """Set up test fixtures."""
        from sagemaker.serve.marshalling.triton_translator import NumpyTranslator
        self.translator = NumpyTranslator()

    def test_init_content_types(self):
        """Test NumpyTranslator initialization."""
        self.assertEqual(self.translator.CONTENT_TYPE, "application/x-npy")
        self.assertEqual(self.translator.ACCEPT, "application/x-npy")

    def test_serialize_returns_data_unchanged(self):
        """Test serialize returns data as-is."""
        data = np.array([1, 2, 3])
        result = self.translator.serialize(data)
        np.testing.assert_array_equal(result, data)

    def test_deserialize_returns_data_unchanged(self):
        """Test deserialize returns data as-is."""
        data = np.array([1, 2, 3])
        result = self.translator.deserialize(data)
        np.testing.assert_array_equal(result, data)

    def test_deserializer_raises_error(self):
        """Test _deserializer raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.translator._deserializer()
        self.assertIn("not meant to be invoked", str(context.exception))


class TestListTranslator(unittest.TestCase):
    """Test cases for ListTranslator class."""

    def setUp(self):
        """Set up test fixtures."""
        from sagemaker.serve.marshalling.triton_translator import ListTranslator
        self.translator = ListTranslator()

    def test_init_content_types(self):
        """Test ListTranslator initialization."""
        self.assertEqual(self.translator.CONTENT_TYPE, "application/list")
        self.assertEqual(self.translator.ACCEPT, "application/list")

    def test_serialize_list_to_numpy(self):
        """Test serializing list to numpy array."""
        data = [1, 2, 3, 4, 5]
        result = self.translator.serialize(data)
        
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array(data))

    def test_serialize_nested_list(self):
        """Test serializing nested list to numpy array."""
        data = [[1, 2], [3, 4]]
        result = self.translator.serialize(data)
        
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array(data))

    def test_serialize_with_mixed_types(self):
        """Test serialize handles mixed types (numpy converts to object array)."""
        # Numpy can actually handle mixed types by creating object arrays
        mixed_data = [1, "string", None]
        result = self.translator.serialize(mixed_data)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.object_)

    def test_deserialize_numpy_to_list(self):
        """Test deserializing numpy array to list."""
        data = np.array([1, 2, 3, 4, 5])
        result = self.translator.deserialize(data)
        
        self.assertIsInstance(result, list)
        self.assertEqual(result, [1, 2, 3, 4, 5])

    def test_deserialize_2d_numpy_to_list(self):
        """Test deserializing 2D numpy array to nested list."""
        data = np.array([[1, 2], [3, 4]])
        result = self.translator.deserialize(data)
        
        self.assertIsInstance(result, list)
        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_deserialize_invalid_data_raises_error(self):
        """Test deserialize raises error for invalid data."""
        invalid_data = "not a numpy array"
        
        with self.assertRaises(ValueError) as context:
            self.translator.deserialize(invalid_data)
        
        self.assertIn("Unable to convert", str(context.exception))

    def test_deserializer_raises_error(self):
        """Test _deserializer raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.translator._deserializer()
        self.assertIn("not meant to be invoked", str(context.exception))


class TestTorchTensorTranslator(unittest.TestCase):
    """Test cases for TorchTensorTranslator class."""

    @patch('torch.from_numpy')
    def test_init_content_types(self, mock_from_numpy):
        """Test TorchTensorTranslator initialization."""
        from sagemaker.serve.marshalling.triton_translator import TorchTensorTranslator
        
        translator = TorchTensorTranslator()
        
        self.assertEqual(translator.CONTENT_TYPE, "tensor/pt")
        self.assertEqual(translator.ACCEPT, "tensor/pt")

    @patch('torch.from_numpy')
    def test_serialize_torch_tensor_to_numpy(self, mock_from_numpy):
        """Test serializing torch tensor to numpy array."""
        from sagemaker.serve.marshalling.triton_translator import TorchTensorTranslator
        
        translator = TorchTensorTranslator()
        
        # Mock torch tensor
        mock_tensor = Mock()
        mock_numpy_array = np.array([1, 2, 3])
        mock_tensor.detach.return_value.numpy.return_value = mock_numpy_array
        
        result = translator.serialize(mock_tensor)
        
        np.testing.assert_array_equal(result, mock_numpy_array)
        mock_tensor.detach.assert_called_once()

    @patch('torch.from_numpy')
    def test_serialize_error_handling(self, mock_from_numpy):
        """Test serialize error handling."""
        from sagemaker.serve.marshalling.triton_translator import TorchTensorTranslator
        
        translator = TorchTensorTranslator()
        
        mock_tensor = Mock()
        mock_tensor.detach.side_effect = Exception("Test error")
        
        with self.assertRaises(ValueError) as context:
            translator.serialize(mock_tensor)
        
        self.assertIn("Unable to translate", str(context.exception))

    @patch('torch.from_numpy')
    def test_deserialize_numpy_to_torch(self, mock_from_numpy):
        """Test deserializing numpy array to torch tensor."""
        from sagemaker.serve.marshalling.triton_translator import TorchTensorTranslator
        
        mock_tensor = Mock()
        mock_from_numpy.return_value = mock_tensor
        
        translator = TorchTensorTranslator()
        
        data = np.array([1, 2, 3])
        result = translator.deserialize(data)
        
        self.assertEqual(result, mock_tensor)
        mock_from_numpy.assert_called_once_with(data)

    @patch('torch.from_numpy')
    def test_deserializer_raises_error(self, mock_from_numpy):
        """Test _deserializer raises ValueError."""
        from sagemaker.serve.marshalling.triton_translator import TorchTensorTranslator
        
        translator = TorchTensorTranslator()
        
        with self.assertRaises(ValueError) as context:
            translator._deserializer()
        self.assertIn("not meant to be invoked", str(context.exception))


class TestTensorflowTensorTranslator(unittest.TestCase):
    """Test cases for TensorflowTensorTranslator class."""

    @patch('tensorflow.convert_to_tensor')
    def test_init_content_types(self, mock_convert):
        """Test TensorflowTensorTranslator initialization."""
        from sagemaker.serve.marshalling.triton_translator import TensorflowTensorTranslator
        
        translator = TensorflowTensorTranslator()
        
        self.assertEqual(translator.CONTENT_TYPE, "tensor/tf")
        self.assertEqual(translator.ACCEPT, "tensor/tf")

    @patch('tensorflow.convert_to_tensor')
    def test_serialize_tf_tensor_to_numpy(self, mock_convert):
        """Test serializing TensorFlow tensor to numpy array."""
        from sagemaker.serve.marshalling.triton_translator import TensorflowTensorTranslator
        
        translator = TensorflowTensorTranslator()
        
        # Mock TF tensor
        mock_tensor = Mock()
        mock_numpy_array = np.array([1, 2, 3])
        mock_tensor.numpy.return_value = mock_numpy_array
        
        result = translator.serialize(mock_tensor)
        
        np.testing.assert_array_equal(result, mock_numpy_array)
        mock_tensor.numpy.assert_called_once()

    @patch('tensorflow.convert_to_tensor')
    def test_serialize_error_handling(self, mock_convert):
        """Test serialize error handling."""
        from sagemaker.serve.marshalling.triton_translator import TensorflowTensorTranslator
        
        translator = TensorflowTensorTranslator()
        
        mock_tensor = Mock()
        mock_tensor.numpy.side_effect = Exception("Test error")
        
        with self.assertRaises(ValueError) as context:
            translator.serialize(mock_tensor)
        
        self.assertIn("Unable to convert", str(context.exception))

    @patch('tensorflow.convert_to_tensor')
    def test_deserialize_numpy_to_tf(self, mock_convert):
        """Test deserializing numpy array to TensorFlow tensor."""
        from sagemaker.serve.marshalling.triton_translator import TensorflowTensorTranslator
        
        mock_tensor = Mock()
        mock_convert.return_value = mock_tensor
        
        translator = TensorflowTensorTranslator()
        
        data = np.array([1, 2, 3])
        result = translator.deserialize(data)
        
        self.assertEqual(result, mock_tensor)

    @patch('tensorflow.convert_to_tensor')
    def test_deserializer_raises_error(self, mock_convert):
        """Test _deserializer raises ValueError."""
        from sagemaker.serve.marshalling.triton_translator import TensorflowTensorTranslator
        
        translator = TensorflowTensorTranslator()
        
        with self.assertRaises(ValueError) as context:
            translator._deserializer()
        self.assertIn("not meant to be invoked", str(context.exception))


if __name__ == "__main__":
    unittest.main()
