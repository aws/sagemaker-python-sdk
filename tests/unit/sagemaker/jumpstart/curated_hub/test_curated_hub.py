from __future__ import absolute_import
import unittest

from mock.mock import patch
import uuid


from sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub import JumpStartCuratedPublicHub
from sagemaker.jumpstart.curated_hub.utils import PublicModelId

TEST_S3_BUCKET_ALREADY_EXISTS_RESPONSE = {
    "Error": {
        "Code": "BucketAlreadyOwnedByYou",
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
        "Code": "ResourceInUse",
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

    test_hub_name = "test-curated-hub-chrstfu"
    test_preexisting_hub_name = "test_preexisting_hub"
    test_preexisting_bucket_name = "test_preexisting_bucket"
    test_region = "us-east-2"
    test_account_id = "123456789012"
    test_public_js_model = PublicModelId(id="autogluon-classification-ensemble", version="1.1.1")
    test_second_public_js_model = PublicModelId(id="catboost-classification-model", version="1.2.7")
    test_nonexistent_public_js_model = PublicModelId(id="fail", version="1.0.0")

    @patch("sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub.JumpStartCuratedPublicHub._init_clients")
    @patch("sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub.JumpStartCuratedPublicHub._get_curated_hub_and_curated_hub_s3_bucket_names")
    @patch("sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub.JumpStartCuratedPublicHub._get_studio_metadata")
    @patch("sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub.JumpStartCuratedPublicHub._get_account_id")
    def setUp(self, mock_account_id, mock_studio_metadata, mock_get_names, mock_init_clients):
        mock_account_id.return_value = self.test_account_id
        mock_studio_metadata.return_value = {}
        mock_get_names.return_value = self.test_preexisting_hub_name, self.test_preexisting_hub_name, True

        self.test_curated_hub = JumpStartCuratedPublicHub(self.test_hub_name, False, self.test_region)

    @patch("sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub.JumpStartCuratedPublicHub._get_preexisting_hub_and_s3_bucket_names")
    def test_get_curated_hub_and_curated_hub_s3_bucket_names_hub_does_not_exist_uses_input_values(self, mock_get_preexisting):
        mock_get_preexisting.return_value = None

        res_hub_name, res_hub_bucket_name, res_skip_create = self.test_curated_hub._get_curated_hub_and_curated_hub_s3_bucket_names(
            self.test_hub_name, False
        )

        self.assertFalse(res_skip_create)
        self.assertEqual(self.test_hub_name, res_hub_name)
        self.assertEqual(f"{self.test_hub_name}-{self.test_region}-{self.test_account_id}", res_hub_bucket_name)

    
