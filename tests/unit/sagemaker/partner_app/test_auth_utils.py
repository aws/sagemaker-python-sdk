from __future__ import absolute_import

import unittest
from unittest.mock import Mock, patch
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from hashlib import sha256

from sagemaker.partner_app.auth_utils import (
    PartnerAppAuthUtils,
    EMPTY_SHA256_HASH,
    UNSIGNED_PAYLOAD,
)


class TestPartnerAppAuthUtils(unittest.TestCase):
    def setUp(self):
        self.sigv4_mock = Mock(spec=SigV4Auth)
        self.app_arn = "arn:aws:sagemaker:us-west-2:123456789012:partner-app/abc123"
        self.url = "https://partner-app-abc123.us-west-2.amazonaws.com"
        self.method = "POST"
        self.headers = {"Authorization": "API_KEY", "Connection": "conn"}
        self.body = b'{"key": "value"}'  # Byte type body for hashing

    @patch("sagemaker.partner_app.auth_utils.AWSRequest")
    def test_get_signed_request_with_body(self, AWSRequestMock):
        aws_request_mock = Mock(spec=AWSRequest)
        AWSRequestMock.return_value = aws_request_mock

        expected_hash = sha256(self.body).hexdigest()
        # Authorization still has the original value as the sigv4 mock does not add this header
        expected_sign_headers = {
            "Authorization": "API_KEY",
            "X-Amz-Partner-App-Authorization": "API_KEY",
            "X-Mlapp-Sm-App-Server-Arn": self.app_arn,
            "X-Amz-Target": "SageMaker.CallPartnerAppApi",
            "X-Amz-Content-SHA256": expected_hash,
        }
        aws_request_mock.headers = expected_sign_headers

        # Mock the add_auth method on the SigV4Auth
        self.sigv4_mock.add_auth = Mock()

        url, signed_headers = PartnerAppAuthUtils.get_signed_request(
            self.sigv4_mock, self.app_arn, self.url, self.method, self.headers, self.body
        )

        # Assert X-Mlapp-Sm-App-Server-Arn header is correct
        self.assertEqual(signed_headers["X-Mlapp-Sm-App-Server-Arn"], self.app_arn)

        # Assert the Authorization header was moved to X-Amz-Partner-App-Authorization
        self.assertIn("X-Amz-Partner-App-Authorization", signed_headers)

        # Assert X-Amz-Content-SHA256 is set
        self.assertEqual(signed_headers["X-Amz-Content-SHA256"], expected_hash)

        # Assert the Connection header is reserved
        self.assertEqual(signed_headers["Connection"], "conn")

        # Assert AWSRequestMock was called
        AWSRequestMock.assert_called_once_with(
            method=self.method,
            url=self.url,
            headers=expected_sign_headers,
            data=self.body,
        )

    def test_get_signed_request_with_no_body(self):
        body = None
        url, signed_headers = PartnerAppAuthUtils.get_signed_request(
            self.sigv4_mock, self.app_arn, self.url, self.method, self.headers, body
        )

        # Assert X-Amz-Content-SHA256 is EMPTY_SHA256_HASH
        self.assertEqual(signed_headers["X-Amz-Content-SHA256"], EMPTY_SHA256_HASH)

    def test_get_signed_request_with_bytes_body(self):
        body = Mock()
        body.seek = Mock()
        body.tell = Mock(return_value=0)
        body.read = Mock(side_effect=[b"test", b""])

        url, signed_headers = PartnerAppAuthUtils.get_signed_request(
            self.sigv4_mock, self.app_arn, self.url, self.method, self.headers, body
        )

        # Verify the seek method was called
        body.seek.assert_called()

        # Calculate the expected checksum for the body
        checksum = sha256(b"test").hexdigest()

        # Assert X-Amz-Content-SHA256 is the calculated checksum
        self.assertEqual(signed_headers["X-Amz-Content-SHA256"], checksum)

    def test_get_body_header_unsigned_payload(self):
        body = {"key": "value"}

        result = PartnerAppAuthUtils.get_body_header(body)

        # Assert the result is UNSIGNED_PAYLOAD for unrecognized body type
        self.assertEqual(result, UNSIGNED_PAYLOAD)

    def test_get_body_header_empty_body(self):
        body = None

        result = PartnerAppAuthUtils.get_body_header(body)

        # Assert the result is EMPTY_SHA256_HASH for empty body
        self.assertEqual(result, EMPTY_SHA256_HASH)
