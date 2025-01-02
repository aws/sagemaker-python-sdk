# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

"""The SageMaker partner application SDK auth module"""
from __future__ import absolute_import

import os
import re
from typing import Dict, Tuple

import boto3
from botocore.auth import SigV4Auth
from botocore.credentials import Credentials
from requests.auth import AuthBase
from requests.models import PreparedRequest
from sagemaker.partner_app.auth_utils import PartnerAppAuthUtils

SERVICE_NAME = "sagemaker"
AWS_PARTNER_APP_ARN_REGEX = r"arn:aws[a-z\-]*:sagemaker:[a-z0-9\-]*:[0-9]{12}:partner-app\/.*"


class RequestsAuth(AuthBase):
    """Requests authentication class for SigV4 header generation.

    This class is used to generate the SigV4 header and add it to the request headers.
    """

    def __init__(self, sigv4: SigV4Auth, app_arn: str):
        """Initialize the RequestsAuth class.

        Args:
            sigv4 (SigV4Auth): SigV4Auth object
            app_arn (str): Application ARN
        """
        self.sigv4 = sigv4
        self.app_arn = app_arn

    def __call__(self, request: PreparedRequest) -> PreparedRequest:
        """Callback function to generate the SigV4 header and add it to the request headers.

        Args:
            request (PreparedRequest): PreparedRequest object

        Returns:
            PreparedRequest: PreparedRequest object with the SigV4 header added
        """
        url, signed_headers = PartnerAppAuthUtils.get_signed_request(
            sigv4=self.sigv4,
            app_arn=self.app_arn,
            url=request.url,
            method=request.method,
            headers=request.headers,
            body=request.body,
        )
        request.url = url
        request.headers.update(signed_headers)

        return request


class PartnerAppAuthProvider:
    """The SageMaker partner application SDK auth provider class"""

    def __init__(self, credentials: Credentials = None):
        """Initialize the PartnerAppAuthProvider class.

        Args:
            credentials (Credentials, optional): AWS credentials. Defaults to None.
        Raises:
            ValueError: If the AWS_PARTNER_APP_ARN environment variable is not set or is invalid.
        """
        self.app_arn = os.getenv("AWS_PARTNER_APP_ARN")
        if self.app_arn is None:
            raise ValueError("Must specify the AWS_PARTNER_APP_ARN environment variable")

        app_arn_regex_match = re.search(AWS_PARTNER_APP_ARN_REGEX, self.app_arn)
        if app_arn_regex_match is None:
            raise ValueError("Must specify a valid AWS_PARTNER_APP_ARN environment variable")

        split_arn = self.app_arn.split(":")
        self.region = split_arn[3]

        self.credentials = (
            credentials if credentials is not None else boto3.Session().get_credentials()
        )
        self.sigv4 = SigV4Auth(self.credentials, SERVICE_NAME, self.region)

    def get_signed_request(
        self, url: str, method: str, headers: dict, body: object
    ) -> Tuple[str, Dict[str, str]]:
        """Generate the SigV4 header and add it to the request headers.

        Args:
            url (str): Request URL
            method (str): HTTP method
            headers (dict): Request headers
            body (object): Request body

        Returns:
            tuple: (url, headers)
        """
        return PartnerAppAuthUtils.get_signed_request(
            sigv4=self.sigv4,
            app_arn=self.app_arn,
            url=url,
            method=method,
            headers=headers,
            body=body,
        )

    def get_auth(self) -> RequestsAuth:
        """Returns the callback class (RequestsAuth) used for generating the SigV4 header.

        Returns:
            RequestsAuth: Callback Object which will calculate the header just before
            request submission.
        """

        return RequestsAuth(self.sigv4, os.environ["AWS_PARTNER_APP_ARN"])
