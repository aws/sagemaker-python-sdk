import unittest
from unittest.mock import Mock, patch
from sagemaker.serve.predictor_async import AsyncPredictor


class TestAsyncPredictor(unittest.TestCase):
    def setUp(self):
        self.mock_predictor = Mock()
        self.mock_predictor.endpoint_name = "test-endpoint"
        self.mock_predictor.sagemaker_session = Mock()
        self.mock_predictor.sagemaker_session.boto_session = Mock()
        self.mock_predictor.sagemaker_session.boto_region_name = "us-east-1"
        self.mock_predictor.sagemaker_session.s3_client = Mock()
        self.mock_predictor.serializer = Mock()
        self.mock_predictor.serializer.CONTENT_TYPE = "application/json"
        self.mock_predictor.deserializer = Mock()
        self.mock_predictor.accept = ["application/json"]

    def test_init(self):
        async_predictor = AsyncPredictor(self.mock_predictor, name="test")
        self.assertEqual(async_predictor.endpoint_name, "test-endpoint")
        self.assertEqual(async_predictor.name, "test")
        self.assertIsNotNone(async_predictor.s3_client)

    def test_predict_without_data_or_path_raises_error(self):
        async_predictor = AsyncPredictor(self.mock_predictor)
        with self.assertRaises(ValueError) as context:
            async_predictor.predict()
        self.assertIn("Please provide input data", str(context.exception))

    @patch.object(AsyncPredictor, "_upload_data_to_s3")
    @patch.object(AsyncPredictor, "_submit_async_request")
    @patch.object(AsyncPredictor, "_wait_for_output")
    def test_predict_with_data(self, mock_wait, mock_submit, mock_upload):
        mock_upload.return_value = "s3://bucket/input"
        mock_submit.return_value = {
            "OutputLocation": "s3://bucket/output",
            "FailureLocation": "s3://bucket/failure",
        }
        mock_wait.return_value = "result"

        async_predictor = AsyncPredictor(self.mock_predictor)
        result = async_predictor.predict(data="test_data")

        self.assertEqual(result, "result")
        mock_upload.assert_called_once()
        mock_submit.assert_called_once()
        mock_wait.assert_called_once()

    @patch.object(AsyncPredictor, "_submit_async_request")
    def test_predict_async_with_input_path(self, mock_submit):
        mock_submit.return_value = {
            "OutputLocation": "s3://bucket/output",
            "FailureLocation": "s3://bucket/failure",
        }

        async_predictor = AsyncPredictor(self.mock_predictor)
        response = async_predictor.predict_async(input_path="s3://bucket/input")

        self.assertIsNotNone(response)
        self.assertEqual(response.output_path, "s3://bucket/output")
        self.assertEqual(response.failure_path, "s3://bucket/failure")

    def test_create_request_args(self):
        async_predictor = AsyncPredictor(self.mock_predictor)
        args = async_predictor._create_request_args(
            input_path="s3://bucket/input", inference_id="test-id"
        )

        self.assertEqual(args["InputLocation"], "s3://bucket/input")
        self.assertEqual(args["EndpointName"], "test-endpoint")
        self.assertEqual(args["InferenceId"], "test-id")
        self.assertIn("Accept", args)

    @patch("sagemaker.serve.predictor_async.parse_s3_url")
    def test_upload_data_to_s3(self, mock_parse):
        mock_parse.return_value = ("bucket", "key")
        self.mock_predictor.serializer.serialize.return_value = b"serialized_data"

        async_predictor = AsyncPredictor(self.mock_predictor, name="test")
        async_predictor.sagemaker_session.default_bucket.return_value = "default-bucket"
        async_predictor.sagemaker_session.default_bucket_prefix = "prefix"

        result = async_predictor._upload_data_to_s3("test_data", "s3://bucket/key")

        self.assertEqual(result, "s3://bucket/key")
        async_predictor.s3_client.put_object.assert_called_once()

    def test_upload_data_to_s3_without_name_uses_endpoint_name(self):
        # Regression test for https://github.com/aws/sagemaker-python-sdk/issues/3210
        self.mock_predictor.serializer.serialize.return_value = b"serialized_data"

        async_predictor = AsyncPredictor(self.mock_predictor)
        async_predictor.sagemaker_session.default_bucket.return_value = "default-bucket"
        async_predictor.sagemaker_session.default_bucket_prefix = None

        result = async_predictor._upload_data_to_s3("test_data")

        self.assertIsNone(async_predictor.name)
        key = async_predictor.s3_client.put_object.call_args.kwargs["Key"]
        self.assertTrue(key.startswith("async-endpoint-inputs/test-endpoint-"))
        self.assertEqual(result, "s3://default-bucket/{}".format(key))

    def test_upload_data_to_s3_with_name_uses_name(self):
        self.mock_predictor.serializer.serialize.return_value = b"serialized_data"

        async_predictor = AsyncPredictor(self.mock_predictor, name="my-async")
        async_predictor.sagemaker_session.default_bucket.return_value = "default-bucket"
        async_predictor.sagemaker_session.default_bucket_prefix = None

        async_predictor._upload_data_to_s3("test_data")

        key = async_predictor.s3_client.put_object.call_args.kwargs["Key"]
        self.assertTrue(key.startswith("async-endpoint-inputs/my-async-"))

    @patch.object(AsyncPredictor, "_submit_async_request")
    def test_predict_async_with_data_and_no_name(self, mock_submit):
        # Regression test for https://github.com/aws/sagemaker-python-sdk/issues/3210
        mock_submit.return_value = {
            "OutputLocation": "s3://bucket/output",
            "FailureLocation": "s3://bucket/failure",
        }
        self.mock_predictor.serializer.serialize.return_value = b"serialized_data"

        async_predictor = AsyncPredictor(self.mock_predictor)
        async_predictor.sagemaker_session.default_bucket.return_value = "default-bucket"
        async_predictor.sagemaker_session.default_bucket_prefix = None

        response = async_predictor.predict_async(data="test_data")

        self.assertEqual(response.output_path, "s3://bucket/output")
        async_predictor.s3_client.put_object.assert_called_once()

    def test_delete_endpoint(self):
        async_predictor = AsyncPredictor(self.mock_predictor)
        async_predictor.delete_endpoint()
        self.mock_predictor.delete_endpoint.assert_called_once_with(True)

    def test_delete_model(self):
        async_predictor = AsyncPredictor(self.mock_predictor)
        async_predictor.delete_model()
        self.mock_predictor.delete_model.assert_called_once()

    def test_enable_data_capture(self):
        async_predictor = AsyncPredictor(self.mock_predictor)
        async_predictor.enable_data_capture()
        self.mock_predictor.enable_data_capture.assert_called_once()

    def test_disable_data_capture(self):
        async_predictor = AsyncPredictor(self.mock_predictor)
        async_predictor.disable_data_capture()
        self.mock_predictor.disable_data_capture.assert_called_once()


if __name__ == "__main__":
    unittest.main()
