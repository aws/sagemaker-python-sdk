import unittest
from unittest.mock import MagicMock

from sagemaker.exceptions import ModelStreamError, InternalStreamFailure
from sagemaker.iterators import LineIterator


class TestLineIterator(unittest.TestCase):
    def test_iteration_with_payload_parts(self):
        # Mocking the stream object
        self.stream = MagicMock()
        self.stream.__iter__.return_value = [
            {"PayloadPart": {"Bytes": b'{"outputs": [" a"]}\n'}},
            {"PayloadPart": {"Bytes": b'{"outputs": [" challenging"]}\n'}},
            {"PayloadPart": {"Bytes": b'{"outputs": [" problem"]}\n'}},
        ]
        self.iterator = LineIterator(self.stream)

        lines = list(self.iterator)
        expected_lines = [
            b'{"outputs": [" a"]}',
            b'{"outputs": [" challenging"]}',
            b'{"outputs": [" problem"]}',
        ]
        self.assertEqual(lines, expected_lines)

    def test_iteration_with_model_stream_error(self):
        # Mocking the stream object
        self.stream = MagicMock()
        self.stream.__iter__.return_value = [
            {"PayloadPart": {"Bytes": b'{"outputs": [" a"]}\n'}},
            {"PayloadPart": {"Bytes": b'{"outputs": [" challenging"]}\n'}},
            {"ModelStreamError": {"Message": "Error message", "ErrorCode": "500"}},
            {"PayloadPart": {"Bytes": b'{"outputs": [" problem"]}\n'}},
        ]
        self.iterator = LineIterator(self.stream)

        with self.assertRaises(ModelStreamError) as e:
            list(self.iterator)

        self.assertEqual(str(e.exception.message), "Error message")
        self.assertEqual(str(e.exception.code), "500")

    def test_iteration_with_internal_stream_failure(self):
        # Mocking the stream object
        self.stream = MagicMock()
        self.stream.__iter__.return_value = [
            {"PayloadPart": {"Bytes": b'{"outputs": [" a"]}\n'}},
            {"PayloadPart": {"Bytes": b'{"outputs": [" challenging"]}\n'}},
            {"InternalStreamFailure": {"Message": "Error internal stream failure"}},
            {"PayloadPart": {"Bytes": b'{"outputs": [" problem"]}\n'}},
        ]
        self.iterator = LineIterator(self.stream)

        with self.assertRaises(InternalStreamFailure) as e:
            list(self.iterator)

        self.assertEqual(str(e.exception.message), "Error internal stream failure")
