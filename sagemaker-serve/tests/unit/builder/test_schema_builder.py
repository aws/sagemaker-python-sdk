import unittest
import numpy as np
from unittest.mock import Mock
from sagemaker.serve.builder.schema_builder import SchemaBuilder


class TestSchemaBuilder(unittest.TestCase):
    def test_numpy_input_output(self):
        sample_input = np.array([[1, 2, 3]])
        sample_output = np.array([[0.1, 0.9]])
        schema = SchemaBuilder(sample_input, sample_output)
        
        self.assertIsNotNone(schema.input_serializer)
        self.assertIsNotNone(schema.output_deserializer)

    def test_json_input_output(self):
        sample_input = {"inputs": "test"}
        sample_output = [{"result": "output"}]
        schema = SchemaBuilder(sample_input, sample_output)
        
        self.assertIsNotNone(schema.input_serializer)
        self.assertIsNotNone(schema.output_deserializer)

    def test_string_input_output(self):
        sample_input = "test input"
        sample_output = "test output"
        schema = SchemaBuilder(sample_input, sample_output)
        
        self.assertIsNotNone(schema.input_serializer)
        self.assertIsNotNone(schema.output_deserializer)

    def test_custom_translator(self):
        from sagemaker.serve.marshalling.custom_payload_translator import CustomPayloadTranslator
        
        class MockTranslator(CustomPayloadTranslator):
            def serialize_payload_to_bytes(self, payload):
                return b"serialized"
            
            def deserialize_payload_from_stream(self, stream):
                return "deserialized"
        
        translator = MockTranslator()
        schema = SchemaBuilder(
            sample_input="test",
            sample_output="output",
            input_translator=translator
        )
        
        self.assertTrue(hasattr(schema, "custom_input_translator"))

    def test_generate_marshalling_map(self):
        sample_input = {"inputs": "test"}
        sample_output = [{"result": "output"}]
        schema = SchemaBuilder(sample_input, sample_output)
        
        mapping = schema.generate_marshalling_map()
        self.assertIn("input_serializer", mapping)
        self.assertIn("output_deserializer", mapping)


if __name__ == "__main__":
    unittest.main()
