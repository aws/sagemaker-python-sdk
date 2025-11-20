import unittest
import io
from sagemaker.serve.marshalling.custom_payload_translator import CustomPayloadTranslator


class MockTranslator(CustomPayloadTranslator):
    def serialize_payload_to_bytes(self, payload):
        return str(payload).encode("utf-8")
    
    def deserialize_payload_from_stream(self, stream):
        return stream.read().decode("utf-8")


class TestCustomPayloadTranslator(unittest.TestCase):
    def test_init(self):
        translator = MockTranslator()
        self.assertEqual(translator.CONTENT_TYPE, "application/custom")
        self.assertEqual(translator.ACCEPT, "application/custom")

    def test_custom_content_types(self):
        translator = MockTranslator(content_type="text/plain", accept_type="text/html")
        self.assertEqual(translator.CONTENT_TYPE, "text/plain")
        self.assertEqual(translator.ACCEPT, "text/html")

    def test_serialize(self):
        translator = MockTranslator()
        result = translator.serialize("test")
        self.assertEqual(result, b"test")

    def test_deserialize(self):
        translator = MockTranslator()
        stream = io.BytesIO(b"test")
        result = translator.deserialize(stream)
        self.assertEqual(result, "test")


if __name__ == "__main__":
    unittest.main()
