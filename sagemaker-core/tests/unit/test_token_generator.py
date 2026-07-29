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
"""Unit tests for sagemaker.core.token_generator module."""

from __future__ import absolute_import

import base64
from datetime import timedelta

import pytest
from unittest.mock import Mock

from botocore.credentials import Credentials

from sagemaker.core.token_generator import generate_token, SageMakerTokenGenerator
from sagemaker.core.token_generator.token_generator import AUTH_PREFIX


class TestSageMakerTokenGenerator:
    """Tests for the SageMakerTokenGenerator class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test credentials and token generator instance."""
        self.token_generator = SageMakerTokenGenerator()
        self.credentials = Credentials(
            access_key="AKIAIOSFODNN7EXAMPLE",
            secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        )
        self.region = "us-west-2"

    def test_get_token_returns_non_null_token(self):
        """Test that get_token returns a non-null token."""
        token = self.token_generator.get_token(self.credentials, self.region)

        assert token is not None
        assert len(token) > 0

    def test_get_token_starts_with_correct_prefix(self):
        """Test that the token starts with the correct prefix."""
        token = self.token_generator.get_token(self.credentials, self.region)

        assert token.startswith(AUTH_PREFIX)

    def test_get_token_with_different_regions(self):
        """Test token generation with different regions."""
        regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-northeast-1"]

        for region in regions:
            token = self.token_generator.get_token(self.credentials, region)

            assert token is not None, f"Token should not be null for region: {region}"
            assert token.startswith(
                AUTH_PREFIX
            ), f"Token should start with the correct prefix for region: {region}"

    def test_get_token_is_base64_encoded(self):
        """Test that the token is properly Base64 encoded."""
        token = self.token_generator.get_token(self.credentials, self.region)

        token_without_prefix = token[len(AUTH_PREFIX) :]
        decoded = base64.b64decode(token_without_prefix)
        assert decoded is not None

    def test_get_token_contains_version_info(self):
        """Test that the decoded token contains version information."""
        token = self.token_generator.get_token(self.credentials, self.region)

        token_without_prefix = token[len(AUTH_PREFIX) :]
        decoded_string = base64.b64decode(token_without_prefix).decode("utf-8")
        assert "&Version=1" in decoded_string

    def test_get_token_different_credentials_produce_different_tokens(self):
        """Test that different credentials produce different tokens."""
        credentials1 = Credentials(
            access_key="AKIAIOSFODNN7EXAMPLE",
            secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        )
        credentials2 = Credentials(
            access_key="AKIAI44QH8DHBEXAMPLE",
            secret_key="je7MtGbClwBF/2Zp9Utk/h3yCo8nvbEXAMPLEKEY",
        )

        token1 = self.token_generator.get_token(credentials1, self.region)
        token2 = self.token_generator.get_token(credentials2, self.region)

        assert token1 != token2

    def test_get_token_with_session_token(self):
        """Test token generation with session token (temporary credentials)."""
        credentials_with_token = Credentials(
            access_key="AKIAIOSFODNN7EXAMPLE",
            secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            token="AQoDYXdzEJr...<remainder of security token>",
        )

        token = self.token_generator.get_token(credentials_with_token, self.region)

        assert token is not None
        assert token.startswith(AUTH_PREFIX)

    def test_get_token_no_credentials_raises_error(self):
        """Test that get_token raises ValueError when credentials are None."""
        with pytest.raises(ValueError, match="Credentials cannot be None"):
            self.token_generator.get_token(None, self.region)

    def test_get_token_no_region_raises_error(self):
        """Test that get_token raises ValueError when region is None or empty."""
        with pytest.raises(ValueError, match="Region must be a non-empty string"):
            self.token_generator.get_token(self.credentials, None)

        with pytest.raises(ValueError, match="Region must be a non-empty string"):
            self.token_generator.get_token(self.credentials, "")

    def test_get_token_contains_correct_expiry(self):
        """Test that the decoded token has the correct expiry duration (12 hours)."""
        token = self.token_generator.get_token(self.credentials, self.region)

        token_without_prefix = token[len(AUTH_PREFIX) :]
        decoded_string = base64.b64decode(token_without_prefix).decode("utf-8")
        assert "X-Amz-Expires=43200" in decoded_string

    def test_get_token_vs_generate_token_consistency(self):
        """Test that get_token and generate_token produce identical tokens for same inputs."""
        mock_provider = Mock()
        mock_provider.load.return_value = self.credentials

        token1 = self.token_generator.get_token(self.credentials, self.region)

        token2 = generate_token(
            region=self.region,
            aws_credentials_provider=mock_provider,
            expiry=timedelta(hours=12),
        )

        assert token1 == token2
        assert token1.startswith(AUTH_PREFIX)
        assert token2.startswith(AUTH_PREFIX)
        assert len(token1) == len(token2)

    def test_generate_token_with_custom_expiry_produces_different_token(self):
        """Test that different expiry durations produce different tokens."""
        mock_provider = Mock()
        mock_provider.load.return_value = self.credentials

        token_default = generate_token(
            region=self.region,
            aws_credentials_provider=mock_provider,
            expiry=timedelta(hours=12),
        )

        token_custom = generate_token(
            region=self.region,
            aws_credentials_provider=mock_provider,
            expiry=timedelta(hours=6),
        )

        assert token_default != token_custom
        assert token_default.startswith(AUTH_PREFIX)
        assert token_custom.startswith(AUTH_PREFIX)
