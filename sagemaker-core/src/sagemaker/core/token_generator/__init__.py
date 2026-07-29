"""AWS SageMaker Token Generator.

A lightweight module for generating short-term bearer tokens for AWS SageMaker
API authentication. Provides the ``generate_token`` helper and the lower-level
``SageMakerTokenGenerator`` class.

Example::

    >>> from sagemaker.core.token_generator import generate_token
    >>> token = generate_token(region="us-east-1")
"""

from __future__ import annotations

import os
from datetime import timedelta

from botocore.credentials import CredentialProvider
from botocore.session import Session

from sagemaker.core.token_generator.token_generator import (
    TOKEN_DURATION,
    SageMakerTokenGenerator,
    _generate_token,
)

__all__ = ["SageMakerTokenGenerator", "generate_token"]


def generate_token(
    region: str | None = None,
    aws_credentials_provider: CredentialProvider | None = None,
    expiry: timedelta = timedelta(hours=12),
) -> str:
    """Generate a short-lived AWS SageMaker bearer token.

    Args:
        region (str): AWS region. Falls back to the ``AWS_REGION``
            environment variable when not provided.
        aws_credentials_provider (CredentialProvider): Optional credential
            provider. Uses the default AWS credential chain when omitted.
        expiry (timedelta): Token lifetime. Must be between 1 second and
            12 hours inclusive. Defaults to 12 hours.

    Returns:
        str: A bearer token string.

    Raises:
        ValueError: If *region* is missing or *expiry* is out of range.
        RuntimeError: If no valid AWS credentials are found.
    """
    region = region or os.environ.get("AWS_REGION")
    if not region:
        raise ValueError("Region must be provided or set via the AWS_REGION environment variable.")

    if expiry.total_seconds() <= 0 or expiry.total_seconds() > TOKEN_DURATION:
        raise ValueError(
            "Token expiry must be greater than zero and less than or equal to 12 hours"
        )

    credentials = (
        aws_credentials_provider.load() if aws_credentials_provider else Session().get_credentials()
    )

    if credentials is None:
        raise RuntimeError(
            "No AWS credentials found. Check your environment or credential provider."
        )

    return _generate_token(credentials, region, int(expiry.total_seconds()))
