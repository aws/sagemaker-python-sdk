from __future__ import absolute_import
import unittest

from mock.mock import patch
from mock import MagicMock

from tests.unit.sagemaker.jumpstart.utils import get_spec_from_base_spec
from botocore.client import ClientError


from sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub import JumpStartCuratedPublicHub
from sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub import PublicModelId

TEST_S3_BUCKET_ALREADY_EXISTS_RESPONSE = {
    "Error": {
        "Code": "BucketAlreadyExists",
        "Message": "Details/context around the exception or error",
    },
    "ResponseMetadata": {
        "RequestId": "1234567890ABCDEF",
        "HostId": "host ID data will appear here as a hash",
        "HTTPStatusCode": 400,
        "HTTPHeaders": {
            "x-amzn-requestid": "12345678-90AB-CDEF-1234567890A",
            "x-amz-id-2": "base64stringherenotarealamzid2value",
            "date": "Thu, 17 Jun 2021 16:08:34 GMT",
        },
        "RetryAttempts": 0,
    },
}

TEST_HUB_ALREADY_EXISTS_RESPONSE = {
    "Error": {
        "Code": "ResourceLimitExceeded",
        "Message": "Details/context around the exception or error",
    },
    "ResponseMetadata": {
        "RequestId": "1234567890ABCDEF",
        "HostId": "host ID data will appear here as a hash",
        "HTTPStatusCode": 400,
        "HTTPHeaders": {
            "x-amzn-requestid": "12345678-90AB-CDEF-1234567890A",
            "x-amz-id-2": "base64stringherenotarealamzid2value",
            "date": "Thu, 17 Jun 2021 16:08:34 GMT",
        },
        "RetryAttempts": 0,
    },
}

TEST_SERVICE_ERROR_RESPONSE = {
    "Error": {
        "Code": "SomeServiceException",
        "Message": "Details/context around the exception or error",
    },
    "ResponseMetadata": {
        "RequestId": "1234567890ABCDEF",
        "HostId": "host ID data will appear here as a hash",
        "HTTPStatusCode": 400,
        "HTTPHeaders": {
            "x-amzn-requestid": "12345678-90AB-CDEF-1234567890A",
            "x-amz-id-2": "base64stringherenotarealamzid2value",
            "date": "Thu, 17 Jun 2021 16:08:34 GMT",
        },
        "RetryAttempts": 0,
    },
}


class JumpStartCuratedPublicHubTest(unittest.TestCase):

    test_s3_prefix = "testCuratedHub"
    test_public_js_model = PublicModelId(id="pytorch", version="1.0.0")

    def setUp(self):
        self.test_curated_hub = JumpStartCuratedPublicHub(self.test_s3_prefix)


    """Creating S3 bucket tests"""

    @patch("sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub.JumpStartCuratedPublicHub._call_create_bucket")
    def test_get_or_create_s3_bucket_s3_does_not_exist_should_create_new(self, mock_create_bucket):
      self.test_curated_hub._get_or_create_s3_bucket("test_bucket")
      mock_create_bucket.assert_called_once_with("test_bucket")

    @patch("sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub.JumpStartCuratedPublicHub._call_create_bucket")
    def test_get_or_create_s3_bucket_s3_exists_should_noop(self, mock_create_bucket):
      mock_create_bucket.side_effect = ClientError(TEST_S3_BUCKET_ALREADY_EXISTS_RESPONSE, "test_operation")

      self.test_curated_hub._get_or_create_s3_bucket("test_bucket")

      mock_create_bucket.assert_called_once_with("test_bucket")

    @patch("sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub.JumpStartCuratedPublicHub._call_create_bucket")
    def test_get_or_create_s3_bucket_s3_fails_for_generic_reason_should_fail(self, mock_create_bucket):
      mock_create_bucket.side_effect = ClientError(TEST_SERVICE_ERROR_RESPONSE, "test_operation")

      with self.assertRaises(ClientError):
        self.test_curated_hub._get_or_create_s3_bucket("test_bucket")

      mock_create_bucket.assert_called_once_with("test_bucket")

    """Creating Curated Hub Tests"""

    @patch("sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub.JumpStartCuratedPublicHub._create_curated_hub")
    def test_get_or_create_curated_hub_does_not_exist_should_create_new(self, mock_create_curated_hub):
      self.test_curated_hub._get_or_create_curated_hub()
      mock_create_curated_hub.assert_called_once()
    
    @patch("sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub.JumpStartCuratedPublicHub._create_curated_hub")
    def test_get_or_create_curated_hub_hub_exists_should_noop(self, mock_create_curated_hub):
      mock_create_curated_hub.side_effect = ClientError(TEST_HUB_ALREADY_EXISTS_RESPONSE, "test_operation")

      self.test_curated_hub._get_or_create_curated_hub()

      mock_create_curated_hub.assert_called_once()

    @patch("sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub.JumpStartCuratedPublicHub._create_curated_hub")
    def test_get_or_create_curated_hub_fails_for_generic_reason_should_fail(self, mock_create_curated_hub):
      mock_create_curated_hub.side_effect = ClientError(TEST_SERVICE_ERROR_RESPONSE, "test_operation")

      with self.assertRaises(ClientError):
        self.test_curated_hub._get_or_create_curated_hub()

      mock_create_curated_hub.assert_called_once()

    """Creating Curated Hub Tests"""

    # @patch("boto3.client")
    # def test_get_or_create_s3_bucket_s3_exists_should_noop(self, mock_boto3_client):
    #    self.test_curated_hub.create()

    # @patch("boto3.client")
    # def test_create_curated_hub_valid_s3_prefix_should_succeed(self, mock_boto3_client):
    #    self.test_curated_hub.create()

    # def test_create_curated_hub_none_s3_prefix_should_fail(self):
    #    null_s3_prefix_curated_hub = JumpStartCuratedPublicHub(None)
    #    null_s3_prefix_curated_hub.create()


    """
    Successful tests
    """

    # @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    # def test_import_model(self, patched_get_model_specs):
    #     patched_get_model_specs.side_effect = get_spec_from_base_spec

    #     self.test_curated_hub._import_model(public_js_model=self.test_public_js_model)
    #     pass
