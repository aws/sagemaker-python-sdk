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
"""Unit tests for serialization security (HMAC + Secrets Manager + Parameter Store)."""
from __future__ import absolute_import

import hashlib
import hmac as hmac_module
import json
from unittest.mock import Mock, patch, MagicMock

import pytest

from sagemaker.core.remote_function.core.serialization import (
    _MetaData,
    _compute_hash,
    _compute_hmac,
    _extract_job_name_from_secret_arn,
    _get_or_create_hmac_secret,
    _get_hmac_key_from_secret,
    _store_secret_arn_in_parameter_store,
    _get_secret_arn_from_parameter_store,
    _validate_secret_arn,
    _perform_integrity_check,
    _upload_payload_and_metadata_to_s3,
    serialize_obj_to_s3,
    deserialize_obj_from_s3,
    serialize_func_to_s3,
    serialize_exception_to_s3,
    deserialize_func_from_s3,
    deserialize_exception_from_s3,
)
from sagemaker.core.remote_function.errors import DeserializationError


MOCK_JOB_NAME = "test-remote-function-job"
MOCK_SECRET_ARN = "arn:aws:secretsmanager:us-west-2:123456789012:secret:sagemaker/remote-function/test-remote-function-job/hmac-key-AbCdEf"
MOCK_HMAC_KEY = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
MOCK_ACCOUNT_ID = "123456789012"
MOCK_S3_URI = "s3://my-bucket/remote-function/test-remote-function-job/results"


def _mock_sagemaker_session(account_id=MOCK_ACCOUNT_ID):
    """Create a mock SageMaker session with Secrets Manager, SSM, and STS clients."""
    session = Mock()
    
    # Mock Secrets Manager client
    secrets_client = Mock()
    secrets_client.get_secret_value.return_value = {
        "ARN": MOCK_SECRET_ARN,
        "SecretString": MOCK_HMAC_KEY,
    }
    secrets_client.create_secret.return_value = {
        "ARN": MOCK_SECRET_ARN,
    }
    secrets_client.exceptions = Mock()
    secrets_client.exceptions.ResourceNotFoundException = type(
        "ResourceNotFoundException", (Exception,), {}
    )
    
    # Mock SSM client
    ssm_client = Mock()
    ssm_client.get_parameter.return_value = {
        "Parameter": {"Value": MOCK_SECRET_ARN}
    }
    ssm_client.exceptions = Mock()
    ssm_client.exceptions.ParameterNotFound = type(
        "ParameterNotFound", (Exception,), {}
    )
    
    # Mock STS client
    sts_client = Mock()
    sts_client.get_caller_identity.return_value = {"Account": account_id}
    
    def client_factory(service_name):
        if service_name == "secretsmanager":
            return secrets_client
        elif service_name == "ssm":
            return ssm_client
        elif service_name == "sts":
            return sts_client
        return Mock()
    
    session.boto_session.client = client_factory
    return session, secrets_client, ssm_client, sts_client


class TestMetaData:
    """Tests for _MetaData class."""

    def test_metadata_with_secret_arn(self):
        metadata = _MetaData(sha256_hash="abc123", secret_arn=MOCK_SECRET_ARN)
        json_bytes = metadata.to_json()
        parsed = _MetaData.from_json(json_bytes)
        
        assert parsed.sha256_hash == "abc123"
        assert parsed.secret_arn == MOCK_SECRET_ARN

    def test_metadata_without_secret_arn_legacy(self):
        metadata = _MetaData(sha256_hash="abc123")
        json_bytes = metadata.to_json()
        parsed = _MetaData.from_json(json_bytes)
        
        assert parsed.sha256_hash == "abc123"
        assert parsed.secret_arn is None

    def test_metadata_missing_hash_raises(self):
        with pytest.raises(DeserializationError, match="SHA256 hash"):
            _MetaData.from_json(json.dumps({"version": "2023-04-24", "serialization_module": "cloudpickle"}))

    def test_metadata_invalid_json_raises(self):
        with pytest.raises(DeserializationError, match="not a valid json"):
            _MetaData.from_json(b"not json")


class TestComputeHmac:
    """Tests for HMAC computation."""

    def test_compute_hmac(self):
        data = b"test data"
        key = "test-key"
        result = _compute_hmac(data, key)
        expected = hmac_module.new(key.encode(), msg=data, digestmod=hashlib.sha256).hexdigest()
        assert result == expected

    def test_compute_hmac_different_keys_produce_different_hashes(self):
        data = b"test data"
        hash1 = _compute_hmac(data, "key1")
        hash2 = _compute_hmac(data, "key2")
        assert hash1 != hash2

    def test_compute_hash_plain_sha256(self):
        data = b"test data"
        result = _compute_hash(data)
        expected = hashlib.sha256(data).hexdigest()
        assert result == expected


class TestGetOrCreateHmacSecret:
    """Tests for Secrets Manager integration."""

    def test_get_existing_secret(self):
        session, secrets_client, _, _ = _mock_sagemaker_session()
        
        arn, key = _get_or_create_hmac_secret(session, MOCK_JOB_NAME)
        
        assert arn == MOCK_SECRET_ARN
        assert key == MOCK_HMAC_KEY
        secrets_client.get_secret_value.assert_called_once_with(
            SecretId=f"sagemaker/remote-function/{MOCK_JOB_NAME}/hmac-key"
        )

    def test_create_new_secret_when_not_found(self):
        session, secrets_client, _, _ = _mock_sagemaker_session()
        
        # Simulate ResourceNotFoundException
        secrets_client.get_secret_value.side_effect = (
            secrets_client.exceptions.ResourceNotFoundException("not found")
        )
        
        arn, key = _get_or_create_hmac_secret(session, MOCK_JOB_NAME)
        
        assert arn == MOCK_SECRET_ARN
        assert len(key) == 64  # secrets.token_hex(32) produces 64 chars
        secrets_client.create_secret.assert_called_once()


class TestParameterStore:
    """Tests for Parameter Store trust anchor."""

    def test_store_secret_arn(self):
        session, _, ssm_client, _ = _mock_sagemaker_session()
        
        _store_secret_arn_in_parameter_store(session, MOCK_JOB_NAME, MOCK_SECRET_ARN)
        
        ssm_client.put_parameter.assert_called_once()
        call_kwargs = ssm_client.put_parameter.call_args[1]
        assert call_kwargs["Name"] == f"/sagemaker/remote-function/{MOCK_JOB_NAME}/secret-arn"
        assert call_kwargs["Value"] == MOCK_SECRET_ARN
        assert "Tags" in call_kwargs

    def test_get_secret_arn(self):
        session, _, ssm_client, _ = _mock_sagemaker_session()
        
        result = _get_secret_arn_from_parameter_store(session, MOCK_JOB_NAME)
        
        assert result == MOCK_SECRET_ARN
        ssm_client.get_parameter.assert_called_once_with(
            Name=f"/sagemaker/remote-function/{MOCK_JOB_NAME}/secret-arn"
        )

    def test_get_secret_arn_not_found_raises(self):
        session, _, ssm_client, _ = _mock_sagemaker_session()
        ssm_client.get_parameter.side_effect = (
            ssm_client.exceptions.ParameterNotFound("not found")
        )
        
        with pytest.raises(DeserializationError, match="Secret ARN not found"):
            _get_secret_arn_from_parameter_store(session, MOCK_JOB_NAME)


class TestValidateSecretArn:
    """Tests for secret ARN validation (Mitigations #1 and #3)."""

    def test_valid_secret_arn_passes(self):
        """Valid ARN in same account matching Parameter Store should pass."""
        session, _, _, _ = _mock_sagemaker_session()
        
        # Should not raise
        _validate_secret_arn(session, MOCK_SECRET_ARN)

    def test_cross_account_arn_rejected(self):
        """Mitigation #1: Secret ARN from different account should be rejected."""
        session, _, _, _ = _mock_sagemaker_session(account_id=MOCK_ACCOUNT_ID)
        
        attacker_arn = "arn:aws:secretsmanager:us-west-2:999999999999:secret:evil-secret"
        
        with pytest.raises(DeserializationError, match="same AWS account"):
            _validate_secret_arn(session, attacker_arn)

    def test_tampered_arn_rejected(self):
        """Mitigation #3: ARN not matching Parameter Store should be rejected."""
        session, _, ssm_client, _ = _mock_sagemaker_session()
        
        # Parameter Store returns the legitimate ARN
        ssm_client.get_parameter.return_value = {
            "Parameter": {"Value": MOCK_SECRET_ARN}
        }
        
        # Attacker's ARN (same account but different secret)
        tampered_arn = f"arn:aws:secretsmanager:us-west-2:{MOCK_ACCOUNT_ID}:secret:attacker-created-secret"
        
        with pytest.raises(DeserializationError, match="does not match expected format"):
            _validate_secret_arn(session, tampered_arn)

    def test_invalid_arn_format_rejected(self):
        """Malformed ARN should be rejected."""
        session, _, _, _ = _mock_sagemaker_session()
        
        with pytest.raises(DeserializationError, match="Invalid secret ARN format"):
            _validate_secret_arn(session, "not-an-arn")


class TestPerformIntegrityCheck:
    """Tests for integrity check with HMAC."""

    def test_hmac_integrity_check_passes(self):
        """Valid HMAC should pass integrity check."""
        session, _, _, _ = _mock_sagemaker_session()
        
        payload = b"test payload"
        expected_hmac = _compute_hmac(payload, MOCK_HMAC_KEY)
        
        # Should not raise
        _perform_integrity_check(
            expected_hash_value=expected_hmac,
            buffer=payload,
            sagemaker_session=session,
            secret_arn=MOCK_SECRET_ARN,
        )

    def test_hmac_integrity_check_fails_on_tampered_payload(self):
        """Tampered payload should fail HMAC check."""
        session, _, _, _ = _mock_sagemaker_session()
        
        original_payload = b"original payload"
        tampered_payload = b"tampered payload"
        expected_hmac = _compute_hmac(original_payload, MOCK_HMAC_KEY)
        
        with pytest.raises(DeserializationError, match="HMAC integrity check failed"):
            _perform_integrity_check(
                expected_hash_value=expected_hmac,
                buffer=tampered_payload,
                sagemaker_session=session,
                secret_arn=MOCK_SECRET_ARN,
            )

    def test_legacy_sha256_check_passes_with_warning(self):
        """Legacy SHA-256 check should pass with warning when no secret_arn."""
        payload = b"test payload"
        expected_hash = _compute_hash(payload)
        
        # Should not raise (legacy path)
        _perform_integrity_check(
            expected_hash_value=expected_hash,
            buffer=payload,
        )

    def test_legacy_sha256_check_fails_on_tampered_payload(self):
        """Legacy SHA-256 check should fail on tampered payload."""
        original_payload = b"original payload"
        tampered_payload = b"tampered payload"
        expected_hash = _compute_hash(original_payload)
        
        with pytest.raises(DeserializationError, match="Integrity check"):
            _perform_integrity_check(
                expected_hash_value=expected_hash,
                buffer=tampered_payload,
            )

    def test_hmac_check_requires_session(self):
        """HMAC check should require sagemaker_session."""
        with pytest.raises(DeserializationError, match="sagemaker_session is required"):
            _perform_integrity_check(
                expected_hash_value="hash",
                buffer=b"data",
                secret_arn=MOCK_SECRET_ARN,
            )

class TestAttackScenarios:
    """Tests simulating actual attack scenarios."""

    def test_attacker_replaces_payload_and_metadata_plain_hash(self):
        """Attacker replaces both files with plain SHA-256 - should fail HMAC check."""
        session, secrets_client, _, _ = _mock_sagemaker_session()
        
        # Attacker creates malicious payload
        malicious_payload = b"malicious code"
        
        # Attacker computes plain SHA-256 (not HMAC)
        plain_hash = hashlib.sha256(malicious_payload).hexdigest()
        
        # Attacker's HMAC won't match because they don't know the key
        with pytest.raises(DeserializationError, match="HMAC integrity check failed"):
            _perform_integrity_check(
                expected_hash_value=plain_hash,
                buffer=malicious_payload,
                sagemaker_session=session,
                secret_arn=MOCK_SECRET_ARN,
            )

    def test_attacker_points_to_cross_account_secret(self):
        """Attacker points to their own secret in different account - should be rejected."""
        session, _, _, _ = _mock_sagemaker_session()
        
        attacker_secret_arn = "arn:aws:secretsmanager:us-west-2:999999999999:secret:attacker-secret"
        
        with pytest.raises(DeserializationError, match="same AWS account"):
            _validate_secret_arn(session, attacker_secret_arn)

    def test_attacker_creates_secret_in_same_account(self):
        """Attacker creates secret in same account but ARN doesn't match Parameter Store."""
        session, _, ssm_client, _ = _mock_sagemaker_session()
        
        # Parameter Store has the legitimate ARN
        ssm_client.get_parameter.return_value = {
            "Parameter": {"Value": MOCK_SECRET_ARN}
        }
        
        # Attacker's secret in same account
        attacker_arn = f"arn:aws:secretsmanager:us-west-2:{MOCK_ACCOUNT_ID}:secret:sagemaker/remote-function/evil-job/hmac-key"
        
        with pytest.raises(DeserializationError, match="Secret ARN mismatch"):
            _validate_secret_arn(session, attacker_arn)
