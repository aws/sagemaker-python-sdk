import io
import unittest
from unittest.mock import Mock, patch

from sagemaker.core.deserializers import JSONDeserializer
from sagemaker.core.serializers import JSONSerializer
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


class _StubPredictor:
    """Minimal predictor exposing the attributes ``AsyncPredictor`` relies on."""

    def __init__(self, sagemaker_session, serializer, deserializer):
        self.endpoint_name = "test-endpoint"
        self.sagemaker_session = sagemaker_session
        self.serializer = serializer
        self.deserializer = deserializer

    @property
    def accept(self):
        return self.deserializer.ACCEPT

    def _handle_response(self, response):
        return self.deserializer.deserialize(response["Body"], response["ContentType"])


class TestAsyncPredictorSerializerOverrides(unittest.TestCase):
    """Overrides set on AsyncPredictor must reach the request and the result (issue #3100)."""

    def setUp(self):
        self.sagemaker_session = Mock()
        self.sagemaker_session.default_bucket.return_value = "bucket"
        self.sagemaker_session.default_bucket_prefix = None
        self.sagemaker_session.sagemaker_runtime_client.invoke_endpoint_async.return_value = {
            "OutputLocation": "s3://bucket/output",
        }
        self.sagemaker_session.s3_client.get_object.return_value = {
            "Body": io.BytesIO(b'{"result": [1, 2, 3]}'),
            "ContentType": "application/json",
        }
        default_deserializer = Mock()
        default_deserializer.ACCEPT = ("*/*",)
        self.predictor = _StubPredictor(self.sagemaker_session, Mock(), default_deserializer)

    def test_deserializer_override_is_used_for_accept_and_result(self):
        async_predictor = AsyncPredictor(self.predictor)

        async_predictor.deserializer = JSONDeserializer()

        self.assertIsInstance(self.predictor.deserializer, JSONDeserializer)

        result = async_predictor.predict(input_path="s3://bucket/input")

        _, kwargs = self.sagemaker_session.sagemaker_runtime_client.invoke_endpoint_async.call_args
        self.assertEqual(kwargs["Accept"], "application/json")
        self.assertEqual(result, {"result": [1, 2, 3]})

    def test_serializer_override_is_used_for_upload(self):
        async_predictor = AsyncPredictor(self.predictor, name="test")

        async_predictor.serializer = JSONSerializer()

        self.assertIsInstance(self.predictor.serializer, JSONSerializer)

        async_predictor.predict_async(data={"hi": "there"})

        _, kwargs = self.sagemaker_session.s3_client.put_object.call_args
        self.assertEqual(kwargs["Body"], '{"hi": "there"}')
        self.assertEqual(kwargs["ContentType"], "application/json")

    def test_reflects_serializers_set_on_wrapped_predictor(self):
        async_predictor = AsyncPredictor(self.predictor)

        serializer = JSONSerializer()
        deserializer = JSONDeserializer()
        self.predictor.serializer = serializer
        self.predictor.deserializer = deserializer

        self.assertIs(async_predictor.serializer, serializer)
        self.assertIs(async_predictor.deserializer, deserializer)


if __name__ == "__main__":
    unittest.main()
