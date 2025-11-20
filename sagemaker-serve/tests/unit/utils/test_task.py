import unittest
from unittest.mock import patch, mock_open
from sagemaker.serve.utils.task import retrieve_local_schemas


class TestTask(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open, read_data='{"test-task": {"sample_inputs": {"properties": {"input": "test"}}, "sample_outputs": {"properties": {"output": "result"}}}}')
    def test_retrieve_local_schemas_success(self, mock_file):
        result = retrieve_local_schemas("test-task")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], {"input": "test"})
        self.assertEqual(result[1], {"output": "result"})

    @patch("builtins.open", new_callable=mock_open, read_data='{"other-task": {"sample_inputs": {"properties": {}}, "sample_outputs": {"properties": {}}}}')
    def test_retrieve_local_schemas_task_not_found(self, mock_file):
        with self.assertRaises(ValueError) as context:
            retrieve_local_schemas("non-existent-task")
        self.assertIn("Could not find", str(context.exception))

    @patch("builtins.open", side_effect=FileNotFoundError())
    def test_retrieve_local_schemas_file_not_found(self, mock_file):
        with self.assertRaises(ValueError) as context:
            retrieve_local_schemas("test-task")
        self.assertIn("Could not find tasks config file", str(context.exception))


if __name__ == "__main__":
    unittest.main()
