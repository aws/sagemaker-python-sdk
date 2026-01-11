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

"""Partner App Auth Utils Module"""

from __future__ import absolute_import

from hashlib import sha256
import functools
from typing import Tuple, Dict

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

HEADER_CONNECTION = "Connection"
HEADER_X_AMZ_TARGET = "X-Amz-Target"
HEADER_AUTHORIZATION = "Authorization"
HEADER_PARTNER_APP_SERVER_ARN = "X-SageMaker-Partner-App-Server-Arn"
HEADER_PARTNER_APP_AUTHORIZATION = "X-Amz-Partner-App-Authorization"
HEADER_X_AMZ_CONTENT_SHA_256 = "X-Amz-Content-SHA256"
CALL_PARTNER_APP_API_ACTION = "SageMaker.CallPartnerAppApi"

PAYLOAD_BUFFER = 1024 * 1024
EMPTY_SHA256_HASH = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
UNSIGNED_PAYLOAD = "UNSIGNED-PAYLOAD"


class PartnerAppAuthUtils:
    """Partner App Auth Utils Class"""

    @staticmethod
    def get_signed_request(
        sigv4: SigV4Auth, app_arn: str, url: str, method: str, headers: dict, body: object
    ) -> Tuple[str, Dict[str, str]]:
        """Generate the SigV4 header and add it to the request headers.

        Args:
            sigv4 (SigV4Auth): SigV4Auth object
            app_arn (str): Application ARN
            url (str): Request URL
            method (str): HTTP method
            headers (dict): Request headers
            body (object): Request body
        Returns:
            tuple: (url, headers)
        """
        # Move API key to X-Amz-Partner-App-Authorization
        if HEADER_AUTHORIZATION in headers:
            headers[HEADER_PARTNER_APP_AUTHORIZATION] = headers[HEADER_AUTHORIZATION]

        # App Arn
        headers[HEADER_PARTNER_APP_SERVER_ARN] = app_arn

        # IAM Action
        headers[HEADER_X_AMZ_TARGET] = CALL_PARTNER_APP_API_ACTION

        # Body
        headers[HEADER_X_AMZ_CONTENT_SHA_256] = PartnerAppAuthUtils.get_body_header(body)

        # Connection header is excluded from server-side signature calculation
        connection_header = headers[HEADER_CONNECTION] if HEADER_CONNECTION in headers else None

        if HEADER_CONNECTION in headers:
            del headers[HEADER_CONNECTION]

        # Spaces are encoded as %20
        url = url.replace("+", "%20")

        # Calculate SigV4 header
        aws_request = AWSRequest(
            method=method,
            url=url,
            headers=headers,
            data=body,
        )
        sigv4.add_auth(aws_request)

        # Reassemble headers
        final_headers = dict(aws_request.headers.items())
        if connection_header is not None:
            final_headers[HEADER_CONNECTION] = connection_header

        return (url, final_headers)

    @staticmethod
    def get_body_header(body: object):
        """Calculate the body header for the SigV4 header.

        Args:
            body (object): Request body
        """
        if body and hasattr(body, "seek"):
            position = body.tell()
            read_chunksize = functools.partial(body.read, PAYLOAD_BUFFER)
            checksum = sha256()
            for chunk in iter(read_chunksize, b""):
                checksum.update(chunk)
            hex_checksum = checksum.hexdigest()
            body.seek(position)
            return hex_checksum

        if body and not isinstance(body, bytes):
            # Body is of a class we don't recognize, so don't sign the payload
            return UNSIGNED_PAYLOAD

        if body:
            # The request serialization has ensured that
            # request.body is a bytes() type.
            return sha256(body).hexdigest()

        # Body is None
        return EMPTY_SHA256_HASH
