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

import threading
import urllib.parse
from datetime import datetime, timezone

import botocore.auth as botocore_auth
from botocore.auth import SigV4QueryAuth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials, ReadOnlyCredentials, RefreshableCredentials

from sagemaker.core.token_generator import generate_token, SageMakerTokenGenerator
from sagemaker.core.token_generator import token_generator as token_generator_module
from sagemaker.core.token_generator.token_generator import (
    AUTH_PREFIX,
    DEFAULT_HOST,
    DEFAULT_URL,
    SERVICE_NAME,
    _generate_token,
)


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


# Two internally-consistent credential versions. A generated token is corrupt if
# it mixes fields (access key / secret / session token) across these versions.
_KEYPAIRS = [
    ("AKIDVERSION0000000001", "secretVersion1zzzzzzzzzzzzzzzzzzzzzzzzzz", "sessionTokenV1AAAA"),
    ("AKIDVERSION0000000002", "secretVersion2yyyyyyyyyyyyyyyyyyyyyyyyyy", "sessionTokenV2BBBB"),
]
_AKID_TO_VERSION = {kp[0]: i for i, kp in enumerate(_KEYPAIRS)}
_TOKEN_TO_VERSION = {kp[2]: i for i, kp in enumerate(_KEYPAIRS)}
_AKID_TO_SECRET = {kp[0]: kp[1] for kp in _KEYPAIRS}
_REGION = "us-west-2"


def _make_rotate_on_read_credentials():
    """Build RefreshableCredentials that flip to the next version on every read.

    The expiry is always kept just inside botocore's advisory refresh window, so
    ``refresh_needed()`` is perpetually True and every access to ``access_key``,
    ``secret_key``, or ``token`` triggers a refresh. A single unfrozen SigV4
    signing pass therefore reads its three credential fields from three different
    versions -- deterministically corrupting the signature unless the signer
    snapshots the credentials first. This is the deterministic (100%-reliable)
    analogue of a concurrent credential-refresh race, kept fast for CI.
    """
    state = {"i": 0}
    lock = threading.Lock()

    def _refresh():
        with lock:
            state["i"] = (state["i"] + 1) % len(_KEYPAIRS)
            access_key, secret_key, session_token = _KEYPAIRS[state["i"]]
        # Expiry inside the advisory window keeps refresh_needed() True without
        # tripping the still-expired RuntimeError in botocore's refresh path.
        expiry = (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()
        return {
            "access_key": access_key,
            "secret_key": secret_key,
            "token": session_token,
            "expiry_time": expiry,
        }

    access_key, secret_key, session_token = _KEYPAIRS[0]
    return RefreshableCredentials(
        access_key=access_key,
        secret_key=secret_key,
        token=session_token,
        expiry_time=datetime.now(timezone.utc) + timedelta(seconds=60),
        refresh_using=_refresh,
        method="test-rotate-on-read",
    )


def _decode_query(token: str) -> dict:
    """Decode a bearer token back into its presigned-URL query parameters."""
    presigned = base64.b64decode(token[len(AUTH_PREFIX) :]).decode("utf-8")
    query = presigned.split("?", 1)[1]
    return {
        key: value[0] for key, value in urllib.parse.parse_qs(query, keep_blank_values=True).items()
    }


def _recompute_signature(akid: str, embedded_token: str, amzdate: str, expires: str, region: str):
    """Recompute the SigV4 signature for the exact request the token embeds.

    Signs an identical canonical request with a coherent static credential whose
    secret is the one that pairs with the embedded access key, pinning the clock
    to the token's own X-Amz-Date so the canonical request is byte-identical. If
    the token was signed with a secret from a different credential version, the
    recomputed signature will not match the embedded one.
    """
    static = Credentials(akid, _AKID_TO_SECRET[akid], embedded_token)
    fixed_dt = datetime.strptime(amzdate, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)

    original_now = botocore_auth.get_current_datetime
    botocore_auth.get_current_datetime = lambda: fixed_dt
    try:
        request = AWSRequest(
            method="POST",
            url=DEFAULT_URL,
            headers={"host": DEFAULT_HOST},
            params={"Action": "CallWithBearerToken"},
        )
        SigV4QueryAuth(static, SERVICE_NAME, region, expires=int(expires)).add_auth(request)
    finally:
        botocore_auth.get_current_datetime = original_now

    recomputed = {
        key: value[0]
        for key, value in urllib.parse.parse_qs(
            urllib.parse.urlsplit(request.url).query, keep_blank_values=True
        ).items()
    }
    return recomputed["X-Amz-Signature"]


def _token_is_corrupt(token: str) -> bool:
    """Return True if a generated token mixes credential versions.

    Two independent checks catch the non-atomic-read race: (1) the embedded
    access key and session token must belong to the same credential version, and
    (2) the embedded signature must match one recomputed with the secret paired
    to the embedded access key. A single credential refresh landing mid-signature
    trips one or both checks.
    """
    query = _decode_query(token)
    embedded_akid = query["X-Amz-Credential"].split("/", 1)[0]
    embedded_token = query.get("X-Amz-Security-Token")

    akid_version = _AKID_TO_VERSION.get(embedded_akid)
    token_version = _TOKEN_TO_VERSION.get(embedded_token)
    if akid_version is None or token_version is None or akid_version != token_version:
        return True
    if embedded_akid not in _AKID_TO_SECRET:
        return True

    scope_parts = query["X-Amz-Credential"].split("/")
    region = scope_parts[2] if len(scope_parts) > 2 else _REGION
    expected_signature = _recompute_signature(
        embedded_akid,
        embedded_token,
        query["X-Amz-Date"],
        query["X-Amz-Expires"],
        region,
    )
    return expected_signature != query["X-Amz-Signature"]


class TestGenerateTokenFreezesCredentials:
    """Regression tests for the credential-snapshot fix in ``_generate_token``.

    Without the ``get_frozen_credentials()`` snapshot, refreshable credentials
    that rotate mid-signature yield tokens that fail server-side verification
    with InvalidSignature. These tests reproduce that corruption deterministically
    and assert the fix eliminates it.
    """

    def test_static_credentials_are_never_flagged_corrupt(self):
        """Sanity check: coherent static credentials must produce coherent tokens."""
        access_key, secret_key, session_token = _KEYPAIRS[0]
        token = _generate_token(Credentials(access_key, secret_key, session_token), _REGION, 43200)
        assert not _token_is_corrupt(token)

    def test_rotating_credentials_produce_only_coherent_tokens(self):
        """Rotate-on-read credentials must still yield internally consistent tokens.

        With rotate-on-read credentials, an unfrozen signer deterministically mixes
        credential versions (100% corrupt). The frozen-snapshot fix must bring the
        corrupt count to zero.
        """
        corrupt = 0
        for _ in range(200):
            token = _generate_token(_make_rotate_on_read_credentials(), _REGION, 43200)
            if _token_is_corrupt(token):
                corrupt += 1
        assert corrupt == 0

    def test_rotating_credentials_are_coherent_under_concurrency(self):
        """A shared rotating credential hammered by many threads yields no corruption."""
        shared = _make_rotate_on_read_credentials()
        tokens = []
        tokens_lock = threading.Lock()

        def _worker():
            local = [_generate_token(shared, _REGION, 43200) for _ in range(40)]
            with tokens_lock:
                tokens.extend(local)

        threads = [threading.Thread(target=_worker) for _ in range(32)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        corrupt = sum(1 for token in tokens if _token_is_corrupt(token))
        assert corrupt == 0

    def test_signer_receives_single_consistent_credential_triple(self, monkeypatch):
        """The SigV4 signer must be constructed with one immutable credential snapshot.

        Captures the credentials object handed to ``SigV4QueryAuth`` and asserts it
        is a frozen ``ReadOnlyCredentials`` whose three fields form a single coherent
        version -- i.e. the signer never sees the live, rotating credential object.
        """
        captured = {}
        real_signer = token_generator_module.SigV4QueryAuth

        def _capturing_signer(credentials, *args, **kwargs):
            captured["credentials"] = credentials
            return real_signer(credentials, *args, **kwargs)

        monkeypatch.setattr(token_generator_module, "SigV4QueryAuth", _capturing_signer)

        _generate_token(_make_rotate_on_read_credentials(), _REGION, 43200)

        signed_with = captured["credentials"]
        assert isinstance(signed_with, ReadOnlyCredentials)
        # Reading the snapshot repeatedly must be stable, and its access key and
        # session token must belong to the same credential version.
        assert signed_with.access_key == signed_with.access_key
        assert _AKID_TO_VERSION[signed_with.access_key] == _TOKEN_TO_VERSION[signed_with.token]
        assert signed_with.secret_key == _AKID_TO_SECRET[signed_with.access_key]
