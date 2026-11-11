import json
import unittest
from io import BytesIO

import numpy as np
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
            sample_input="test", sample_output="output", input_translator=translator
        )

        self.assertTrue(hasattr(schema, "custom_input_translator"))

    def test_generate_marshalling_map(self):
        sample_input = {"inputs": "test"}
        sample_output = [{"result": "output"}]
        schema = SchemaBuilder(sample_input, sample_output)

        mapping = schema.generate_marshalling_map()
        self.assertIn("input_serializer", mapping)
        self.assertIn("output_deserializer", mapping)

    def test_numpy_input_deserializer_honors_json_content_type(self):
        schema = SchemaBuilder(np.array([[0]]), np.array([[0]]))
        stream = BytesIO(json.dumps([[1, 2], [3, 4]]).encode("utf-8"))

        result = schema.input_deserializer.deserialize(stream, content_type="application/json")

        np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4]]))

    def test_numpy_input_deserializer_honors_csv_content_type(self):
        schema = SchemaBuilder(np.array([[0]]), np.array([[0]]))
        stream = BytesIO(b"1,2\n3,4\n")

        result = schema.input_deserializer.deserialize(stream, content_type="text/csv")

        np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4]]))

    def test_numpy_output_deserializer_preserves_expected_accept_type(self):
        expected = np.array([[0.1, 0.9]])
        schema = SchemaBuilder(np.array([[0]]), expected)
        stream = BytesIO()
        np.save(stream, expected)
        stream.seek(0)

        result = schema.output_deserializer.deserialize(stream, content_type="text/html")

        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
