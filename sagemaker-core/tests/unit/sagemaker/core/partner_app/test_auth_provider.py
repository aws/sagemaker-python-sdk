from __future__ import absolute_import

import os
import unittest
from unittest.mock import patch, MagicMock
from requests import PreparedRequest
from sagemaker.core.partner_app.auth_provider import RequestsAuth, PartnerAppAuthProvider


class TestRequestsAuth(unittest.TestCase):

    @patch("sagemaker.core.partner_app.auth_provider.PartnerAppAuthUtils.get_signed_request")
    @patch("sagemaker.core.partner_app.auth_provider.SigV4Auth")
    def test_requests_auth_call(self, mock_sigv4_auth, mock_get_signed_request):
        # Prepare mock data
        mock_signed_url = "https://returned-url.test.com/"
        mock_signed_headers = {"Authorization": "SigV4", "x-amz-date": "20241016T120000Z"}
        mock_get_signed_request.return_value = (mock_signed_url, mock_signed_headers)

        # Create the objects needed for testing
        app_arn = "arn:aws:lambda:us-west-2:123456789012:sagemaker:test"
        under_test = RequestsAuth(sigv4=mock_sigv4_auth, app_arn=app_arn)

        # Create a prepared request object to simulate an actual request
        request = PreparedRequest()
        request.method = "GET"
        request_url = "https://test.com"
        request.url = request_url
        request_headers = {}
        request.headers = request_headers
        request.body = "{}"

        # Call the method under test
        updated_request = under_test(request)

        # Assertions to verify the behavior
        mock_get_signed_request.assert_called_once_with(
            sigv4=mock_sigv4_auth,
            app_arn=app_arn,
            url=request_url,
            method="GET",
            headers=request_headers,
            body=request.body,
        )

        self.assertEqual(updated_request.url, mock_signed_url)
        self.assertIn("Authorization", updated_request.headers)
        self.assertIn("x-amz-date", updated_request.headers)
        self.assertEqual(updated_request.headers["Authorization"], "SigV4")
        self.assertEqual(updated_request.headers["x-amz-date"], "20241016T120000Z")


class TestPartnerAppAuthProvider(unittest.TestCase):

    @patch("sagemaker.core.partner_app.auth_provider.boto3.Session")
    @patch("sagemaker.core.partner_app.auth_provider.SigV4Auth")
    @patch("sagemaker.core.partner_app.auth_provider.PartnerAppAuthUtils.get_signed_request")
    def test_get_signed_request(
        self, mock_get_signed_request, mock_sigv4auth_class, mock_boto3_session
    ):
        # Set up environment variable
        test_app_arn = "arn:aws-us-gov:sagemaker:us-west-2:123456789012:partner-app/my-app"
        os.environ["AWS_PARTNER_APP_ARN"] = test_app_arn

        # Mock the return value of boto3.Session().get_credentials()
        mock_credentials = MagicMock()
        mock_boto3_session.return_value.get_credentials.return_value = mock_credentials

        # Mock the SigV4Auth instance
        mock_sigv4auth_instance = MagicMock()
        mock_sigv4auth_class.return_value = mock_sigv4auth_instance

        # Initialize the PartnerAppAuthProvider class
        provider = PartnerAppAuthProvider()

        # Mock return value for get_signed_request
        mock_get_signed_request.return_value = {
            "Authorization": "SigV4",
            "x-amz-date": "20241016T120000Z",
        }

        # Call get_signed_request method
        signed_request = provider.get_signed_request(
            url="https://example.com",
            method="GET",
            headers={"Content-Type": "application/json"},
            body=None,
        )

        # Assert that the get_signed_request method was called with correct parameters
        mock_get_signed_request.assert_called_once_with(
            sigv4=mock_sigv4auth_instance,
            app_arn=test_app_arn,
            url="https://example.com",
            method="GET",
            headers={"Content-Type": "application/json"},
            body=None,
        )

        # Assert the response matches the mocked return value
        self.assertEqual(signed_request["Authorization"], "SigV4")
        self.assertEqual(signed_request["x-amz-date"], "20241016T120000Z")

    @patch("sagemaker.core.partner_app.auth_provider.SigV4Auth")
    def test_get_auth(self, mock_sigv4auth_class):
        # Set up environment variable
        os.environ["AWS_PARTNER_APP_ARN"] = (
            "arn:aws:sagemaker:us-west-2:123456789012:partner-app/app-abc"
        )

        # Mock the SigV4Auth instance
        mock_sigv4auth_instance = MagicMock()
        mock_sigv4auth_class.return_value = mock_sigv4auth_instance

        # Initialize the PartnerAppAuthProvider class
        provider = PartnerAppAuthProvider()

        # Call get_auth method
        auth_instance = provider.get_auth()

        # Assert that the returned object is a RequestsAuth instance
        self.assertIsInstance(auth_instance, RequestsAuth)

        # Assert that RequestsAuth was initialized with correct arguments
        self.assertEqual(auth_instance.sigv4, mock_sigv4auth_instance)
        self.assertEqual(auth_instance.app_arn, os.environ["AWS_PARTNER_APP_ARN"])

    def test_init_raises_value_error_with_missing_app_arn(self):
        # Remove the environment variable
        if "AWS_PARTNER_APP_ARN" in os.environ:
            del os.environ["AWS_PARTNER_APP_ARN"]

        # Ensure ValueError is raised when AWS_PARTNER_APP_ARN is not set
        with self.assertRaises(ValueError) as context:
            PartnerAppAuthProvider()

        self.assertIn(
            "Must specify the AWS_PARTNER_APP_ARN environment variable", str(context.exception)
        )

    def test_init_raises_value_error_with_invalid_app_arn(self):
        os.environ["AWS_PARTNER_APP_ARN"] = (
            "arn:aws:lambda:us-west-2:123456789012:function:my-function"
        )

        # Ensure ValueError is raised when AWS_PARTNER_APP_ARN is not set
        with self.assertRaises(ValueError) as context:
            PartnerAppAuthProvider()

        self.assertIn(
            "Must specify a valid AWS_PARTNER_APP_ARN environment variable", str(context.exception)
        )
