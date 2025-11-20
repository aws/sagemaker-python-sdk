"""
Unit tests for tensorflow_serving inference.py module.
Simple tests that don't require module import.
"""

import unittest
from unittest.mock import Mock
import json
import io
import numpy as np
import pandas as pd


class TestTensorFlowServingInference(unittest.TestCase):
    """Test TensorFlow Serving inference module functions without importing the module."""

    def test_input_handler_logic(self):
        """Test input_handler logic."""
        def input_handler(data, context, schema_builder):
            read_data = data.read()
            if hasattr(schema_builder, "custom_input_translator"):
                deserialized_data = schema_builder.custom_input_translator.deserialize(
                    io.BytesIO(read_data), context.request_content_type
                )
            else:
                deserialized_data = schema_builder.input_deserializer.deserialize(
                    io.BytesIO(read_data), context.request_content_type
                )
            return json.dumps({"instances": deserialized_data})
        
        schema_builder = Mock()
        schema_builder.custom_input_translator = Mock()
        schema_builder.custom_input_translator.deserialize = Mock(return_value=[[1, 2, 3]])
        
        mock_data = Mock()
        mock_data.read = Mock(return_value=b'{"data": [1, 2, 3]}')
        
        mock_context = Mock()
        mock_context.request_content_type = "application/json"
        
        result = input_handler(mock_data, mock_context, schema_builder)
        
        expected = json.dumps({"instances": [[1, 2, 3]]})
        self.assertEqual(result, expected)

    def test_output_handler_logic(self):
        """Test output_handler logic."""
        def output_handler(data, context, schema_builder):
            if data.status_code != 200:
                raise ValueError(data.content.decode("utf-8"))
            
            response_content_type = context.accept_header
            prediction = data.content
            prediction_dict = json.loads(prediction.decode("utf-8"))
            
            if hasattr(schema_builder, "custom_output_translator"):
                return (
                    schema_builder.custom_output_translator.serialize(
                        prediction_dict["predictions"], response_content_type
                    ),
                    response_content_type,
                )
            else:
                return schema_builder.output_serializer.serialize(prediction_dict["predictions"]), response_content_type
        
        schema_builder = Mock()
        schema_builder.custom_output_translator = Mock()
        schema_builder.custom_output_translator.serialize = Mock(return_value=b'{"predictions": [0.1, 0.9]}')
        
        mock_data = Mock()
        mock_data.status_code = 200
        mock_data.content = json.dumps({"predictions": [0.1, 0.9]}).encode('utf-8')
        
        mock_context = Mock()
        mock_context.accept_header = "application/json"
        
        result, content_type = output_handler(mock_data, mock_context, schema_builder)
        
        self.assertEqual(result, b'{"predictions": [0.1, 0.9]}')
        self.assertEqual(content_type, "application/json")

    def test_convert_numpy_array_logic(self):
        """Test conversion of numpy array."""
        def _convert_for_serialization(deserialized_data):
            if isinstance(deserialized_data, np.ndarray):
                return deserialized_data.tolist()
            return deserialized_data
        
        data = np.array([[1, 2, 3], [4, 5, 6]])
        result = _convert_for_serialization(data)
        
        self.assertEqual(result, [[1, 2, 3], [4, 5, 6]])

    def test_convert_pandas_dataframe_logic(self):
        """Test conversion of pandas DataFrame."""
        def _convert_for_serialization(deserialized_data):
            if isinstance(deserialized_data, pd.DataFrame):
                return deserialized_data.to_dict(orient="list")
            return deserialized_data
        
        data = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        result = _convert_for_serialization(data)
        
        self.assertEqual(result, {'a': [1, 2], 'b': [3, 4]})

    def test_convert_pandas_series_logic(self):
        """Test conversion of pandas Series."""
        def _convert_for_serialization(deserialized_data):
            if isinstance(deserialized_data, pd.Series):
                return deserialized_data.tolist()
            return deserialized_data
        
        data = pd.Series([1, 2, 3, 4])
        result = _convert_for_serialization(data)
        
        self.assertEqual(result, [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
