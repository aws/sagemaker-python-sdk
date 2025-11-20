import unittest
from unittest.mock import Mock, patch
from botocore.exceptions import ClientError
from sagemaker.serve.async_inference.async_inference_response import AsyncInferenceResponse
from sagemaker.serve.async_inference import WaiterConfig
from sagemaker.core.exceptions import ObjectNotExistedError, UnexpectedClientError, AsyncInferenceModelError


class TestAsyncInferenceResponse(unittest.TestCase):
    def setUp(self):
        self.mock_predictor = Mock()
        self.mock_predictor.s3_client = Mock()
        self.mock_predictor.predictor = Mock()
        self.output_path = "s3://bucket/output/result.json"
        self.failure_path = "s3://bucket/failure/error.json"

    def test_init(self):
        response = AsyncInferenceResponse(self.mock_predictor, self.output_path, self.failure_path)
        self.assertEqual(response.output_path, self.output_path)
        self.assertEqual(response.failure_path, self.failure_path)
        self.assertIsNone(response._result)

    def test_get_result_without_waiter(self):
        response = AsyncInferenceResponse(self.mock_predictor, self.output_path, None)
        mock_s3_response = {"Body": Mock()}
        self.mock_predictor.s3_client.get_object.return_value = mock_s3_response
        self.mock_predictor.predictor._handle_response.return_value = "result"
        
        result = response.get_result()
        self.assertEqual(result, "result")

    def test_get_result_with_waiter(self):
        response = AsyncInferenceResponse(self.mock_predictor, self.output_path, self.failure_path)
        waiter_config = WaiterConfig(max_attempts=10, delay=5)
        self.mock_predictor._wait_for_output.return_value = "waiter_result"
        
        result = response.get_result(waiter_config)
        self.assertEqual(result, "waiter_result")

    def test_get_result_invalid_waiter_config(self):
        response = AsyncInferenceResponse(self.mock_predictor, self.output_path, None)
        with self.assertRaises(ValueError):
            response.get_result(waiter_config="invalid")

    def test_get_result_no_such_key(self):
        response = AsyncInferenceResponse(self.mock_predictor, self.output_path, None)
        error = ClientError({"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, "get_object")
        self.mock_predictor.s3_client.get_object.side_effect = error
        
        with self.assertRaises(ObjectNotExistedError):
            response.get_result()

    def test_get_result_with_failure_path(self):
        response = AsyncInferenceResponse(self.mock_predictor, self.output_path, self.failure_path)
        output_error = ClientError({"Error": {"Code": "NoSuchKey"}}, "get_object")
        failure_response = {"Body": Mock()}
        
        self.mock_predictor.s3_client.get_object.side_effect = [output_error, failure_response]
        self.mock_predictor.predictor._handle_response.return_value = "error message"
        
        with self.assertRaises(AsyncInferenceModelError):
            response.get_result()


class TestWaiterConfig(unittest.TestCase):
    def test_init_defaults(self):
        config = WaiterConfig()
        self.assertEqual(config.max_attempts, 60)
        self.assertEqual(config.delay, 15)

    def test_init_custom(self):
        config = WaiterConfig(max_attempts=100, delay=30)
        self.assertEqual(config.max_attempts, 100)
        self.assertEqual(config.delay, 30)

    def test_to_request_dict(self):
        config = WaiterConfig(max_attempts=50, delay=20)
        result = config._to_request_dict()
        self.assertEqual(result, {"Delay": 20, "MaxAttempts": 50})


if __name__ == "__main__":
    unittest.main()
