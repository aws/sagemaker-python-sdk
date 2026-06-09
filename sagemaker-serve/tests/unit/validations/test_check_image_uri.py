import unittest
from unittest.mock import Mock

from sagemaker.core.helper.pipeline_variable import PipelineVariable
from sagemaker.core.workflow.parameters import ParameterString
from sagemaker.serve.validations.check_image_uri import is_1p_image_uri, all_accounts


class TestCheckImageUri(unittest.TestCase):
    def test_is_1p_image_uri_true(self):
        # Use a known 1P account from the list
        image_uri = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch:latest"
        self.assertTrue(is_1p_image_uri(image_uri))

    def test_is_1p_image_uri_false(self):
        # Use a non-1P account
        image_uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com/custom:latest"
        self.assertFalse(is_1p_image_uri(image_uri))

    def test_is_1p_image_uri_another_1p_account(self):
        # Test with another known 1P account
        image_uri = "246618743249.dkr.ecr.us-west-2.amazonaws.com/tensorflow:latest"
        self.assertTrue(is_1p_image_uri(image_uri))

    def test_all_accounts_is_set(self):
        self.assertIsInstance(all_accounts, set)
        self.assertGreater(len(all_accounts), 0)

    def test_all_accounts_contains_known_accounts(self):
        # Verify some known AWS accounts are in the set
        self.assertIn("763104351884", all_accounts)
        self.assertIn("246618743249", all_accounts)

    def test_is_1p_image_uri_with_parameter_string_returns_false(self):
        # ParameterString should not raise TypeError and should return False
        param = ParameterString(
            "image_uri",
            default_value="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch:latest",
        )
        result = is_1p_image_uri(param)
        self.assertFalse(result)

    def test_is_1p_image_uri_with_pipeline_variable_returns_false(self):
        # A mock PipelineVariable instance should return False without raising
        mock_pipeline_var = Mock(spec=PipelineVariable)
        result = is_1p_image_uri(mock_pipeline_var)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
