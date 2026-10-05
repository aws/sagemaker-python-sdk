"""SageMaker Token Generator core signing logic.

Generates short-term bearer tokens for AWS SageMaker API authentication
using SigV4 signed pre-signed URLs.
"""

from __future__ import annotations

import base64

from botocore.auth import SigV4QueryAuth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

DEFAULT_HOST: str = "sagemaker.amazonaws.com"
DEFAULT_URL: str = f"https://{DEFAULT_HOST}/"
SERVICE_NAME: str = "sagemaker"
AUTH_PREFIX: str = "sagemaker-api-key-"
TOKEN_VERSION: str = "&Version=1"
TOKEN_DURATION: int = 43200  # 12 hours in seconds


def _generate_token(credentials: Credentials, region: str, expires: int) -> str:
    """Build a presigned bearer token.

    Args:
        credentials (Credentials): AWS credentials.
        region (str): AWS region.
        expires (int): Expiry time in seconds.

    Returns:
        str: A base64-encoded bearer token string.
    """
    request = AWSRequest(
        method="POST",
        url=DEFAULT_URL,
        headers={"host": DEFAULT_HOST},
        params={"Action": "CallWithBearerToken"},
    )

    # Take an atomic snapshot of the credentials before signing. botocore's
    # SigV4 signer reads access_key, token, and secret_key as three separate,
    # non-atomic property accesses. With refreshable credentials, a refresh that
    # lands mid-signature would mix key material from two credential versions and
    # produce a token that fails verification with InvalidSignature. Freezing
    # once (under botocore's refresh lock) guarantees all three reads come from a
    # single credential version. Static credentials also expose this method and
    # return a coherent snapshot, so this is safe for every credential type.
    if hasattr(credentials, "get_frozen_credentials"):
        credentials = credentials.get_frozen_credentials()

    auth = SigV4QueryAuth(credentials, SERVICE_NAME, region, expires=expires)
    auth.add_auth(request)

    presigned_url = request.url.replace("https://", "") + TOKEN_VERSION
    encoded_token = base64.b64encode(presigned_url.encode("utf-8")).decode("utf-8")

    return f"{AUTH_PREFIX}{encoded_token}"


class SageMakerTokenGenerator:
    """Generate short-lived AWS SageMaker bearer tokens."""

    def get_token(self, credentials: Credentials, region: str) -> str:
        """Generate a token using provided credentials and region.

        Args:
            credentials (Credentials): AWS credentials to sign the request.
            region (str): AWS region.

        Returns:
            str: A bearer token string.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not credentials:
            raise ValueError("Credentials cannot be None")
        if not region or not isinstance(region, str):
            raise ValueError("Region must be a non-empty string")

        return _generate_token(credentials, region, TOKEN_DURATION)
