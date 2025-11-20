"""Additional unit tests for async_inference_response.py to increase coverage."""

import unittest
from unittest.mock import Mock, patch
from botocore.exceptions import ClientError


class TestAsyncInferenceResponseGetResult(unittest.TestCase):
    """Test AsyncInferenceResponse get_result method."""

    def test_get_result_invalid_waiter_config(self):
        """Test get_result with invalid waiter_config."""
        from sagemaker.serve.async_inference.async_inference_response import AsyncInferenceResponse
        
        mock_predictor = Mock()
        response = AsyncInferenceResponse(mock_predictor, "s3://bucket/output", None)
        
        with self.assertRaises(ValueError) as context:
            response.get_result(waiter_config="invalid")
        
        self.assertIn("WaiterConfig", str(context.exception))

    @patch('sagemaker.serve.async_inference.async_inference_response.parse_s3_url')
    def test_get_result_from_s3_output_path_success(self, mock_parse):
        """Test _get_result_from_s3_output_path success."""
        from sagemaker.serve.async_inference.async_inference_response import AsyncInferenceResponse
        
        mock_predictor = Mock()
        mock_predictor.s3_client.get_object.return_value = {"Body": Mock()}
        mock_predictor.predictor._handle_response.return_value = "result"
        
        mock_parse.return_value = ("bucket", "key")
        
        response = AsyncInferenceResponse(mock_predictor, "s3://bucket/output", None)
        result = response._get_result_from_s3_output_path("s3://bucket/output")
        
        self.assertEqual(result, "result")

    @patch('sagemaker.serve.async_inference.async_inference_response.parse_s3_url')
    def test_get_result_from_s3_output_path_no_such_key(self, mock_parse):
        """Test _get_result_from_s3_output_path with NoSuchKey error."""
        from sagemaker.serve.async_inference.async_inference_response import AsyncInferenceResponse
        from sagemaker.core.exceptions import ObjectNotExistedError
        
        mock_predictor = Mock()
        error_response = {'Error': {'Code': 'NoSuchKey'}}
        mock_predictor.s3_client.get_object.side_effect = ClientError(error_response, 'GetObject')
        
        mock_parse.return_value = ("bucket", "key")
        
        response = AsyncInferenceResponse(mock_predictor, "s3://bucket/output", None)
        
        with self.assertRaises(ObjectNotExistedError):
            response._get_result_from_s3_output_path("s3://bucket/output")

    @patch('sagemaker.serve.async_inference.async_inference_response.parse_s3_url')
    def test_get_result_from_s3_output_path_unexpected_error(self, mock_parse):
        """Test _get_result_from_s3_output_path with unexpected error."""
        from sagemaker.serve.async_inference.async_inference_response import AsyncInferenceResponse
        from sagemaker.core.exceptions import UnexpectedClientError
        
        mock_predictor = Mock()
        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access denied'}}
        mock_predictor.s3_client.get_object.side_effect = ClientError(error_response, 'GetObject')
        
        mock_parse.return_value = ("bucket", "key")
        
        response = AsyncInferenceResponse(mock_predictor, "s3://bucket/output", None)
        
        with self.assertRaises(UnexpectedClientError):
            response._get_result_from_s3_output_path("s3://bucket/output")

    @patch('sagemaker.serve.async_inference.async_inference_response.parse_s3_url')
    def test_get_result_from_s3_output_failure_paths_with_failure(self, mock_parse):
        """Test _get_result_from_s3_output_failure_paths with failure."""
        from sagemaker.serve.async_inference.async_inference_response import AsyncInferenceResponse
        from sagemaker.core.exceptions import AsyncInferenceModelError
        
        mock_predictor = Mock()
        error_response = {'Error': {'Code': 'NoSuchKey'}}
        mock_predictor.s3_client.get_object.side_effect = [
            ClientError(error_response, 'GetObject'),
            {"Body": Mock()}
        ]
        mock_predictor.predictor._handle_response.return_value = "error message"
        
        mock_parse.side_effect = [("bucket", "key"), ("failure-bucket", "failure-key")]
        
        response = AsyncInferenceResponse(mock_predictor, "s3://bucket/output", "s3://bucket/failure")
        
        with self.assertRaises(AsyncInferenceModelError):
            response._get_result_from_s3_output_failure_paths("s3://bucket/output", "s3://bucket/failure")

    @patch('sagemaker.serve.async_inference.async_inference_response.parse_s3_url')
    def test_get_result_from_s3_output_failure_paths_still_running(self, mock_parse):
        """Test _get_result_from_s3_output_failure_paths when still running."""
        from sagemaker.serve.async_inference.async_inference_response import AsyncInferenceResponse
        from sagemaker.core.exceptions import ObjectNotExistedError
        
        mock_predictor = Mock()
        error_response = {'Error': {'Code': 'NoSuchKey'}}
        mock_predictor.s3_client.get_object.side_effect = ClientError(error_response, 'GetObject')
        
        mock_parse.return_value = ("bucket", "key")
        
        response = AsyncInferenceResponse(mock_predictor, "s3://bucket/output", "s3://bucket/failure")
        
        with self.assertRaises(ObjectNotExistedError):
            response._get_result_from_s3_output_failure_paths("s3://bucket/output", "s3://bucket/failure")


if __name__ == "__main__":
    unittest.main()
