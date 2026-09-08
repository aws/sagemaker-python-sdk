"""Validates the integrity of pickled file with SHA-256 hash.

Supports two modes:
- Plain SHA-256 (default): Used when no secret key is provided.
- HMAC-SHA256 (keyed): Used when a secret key is provided, for backward
  compatibility with older container images that perform HMAC-based checks.
"""

from __future__ import absolute_import
import hmac
import hashlib
import os
import secrets
from pathlib import Path

from sagemaker.core.remote_function.core.serialization import _MetaData

SAGEMAKER_SERVE_SECRET_KEY = "SAGEMAKER_SERVE_SECRET_KEY"


def generate_secret_key(nbytes: int = 32) -> str:
    """Generate a cryptographically secure secret key.

    Args:
        nbytes: Number of random bytes (the returned hex string will be
            twice this length). Defaults to 32 (256-bit key).

    Returns:
        A hex-encoded random string suitable for use as an HMAC key.
    """
    return secrets.token_hex(nbytes)


def compute_hash(buffer: bytes, secret_key: str = None) -> str:
    """Compute hash of the given buffer.

    When *secret_key* is provided the hash is an HMAC-SHA256 keyed digest;
    otherwise a plain SHA-256 digest is returned.

    Args:
        buffer: The bytes to hash.
        secret_key: Optional HMAC key. When ``None`` (default) a plain
            SHA-256 hash is computed.

    Returns:
        Hex-encoded hash string.
    """
    if secret_key:
        return hmac.new(secret_key.encode(), msg=buffer, digestmod=hashlib.sha256).hexdigest()
    return hashlib.sha256(buffer).hexdigest()


def perform_integrity_check(buffer: bytes, metadata_path: Path, secret_key: str = None):
    """Validates the integrity of bytes by comparing the hash value.

    Computes both the plain SHA-256 digest and (when a secret key is
    available) the HMAC-SHA256 digest, then checks whether the expected
    hash stored in *metadata_path* matches either one.  This provides
    backward compatibility between SDK versions that write plain hashes
    and container images that expect HMAC hashes (or vice-versa).

    Args:
        buffer: The serialized bytes to verify.
        metadata_path: Path to the ``metadata.json`` file containing the
            expected hash.
        secret_key: Optional HMAC key.  When ``None`` the function falls
            back to the ``SAGEMAKER_SERVE_SECRET_KEY`` environment variable.
    """
    if not Path.exists(metadata_path):
        raise ValueError("Path to metadata.json does not exist")

    with open(str(metadata_path), "rb") as md:
        expected_hash_value = _MetaData.from_json(md.read()).sha256_hash

    # Resolve secret key: explicit arg > environment variable > None
    effective_secret_key = secret_key or os.environ.get(SAGEMAKER_SERVE_SECRET_KEY)

    # Compute candidate digests
    plain_hash = hashlib.sha256(buffer).hexdigest()

    if hmac.compare_digest(expected_hash_value, plain_hash):
        return

    if effective_secret_key:
        hmac_hash = hmac.new(
            effective_secret_key.encode(), msg=buffer, digestmod=hashlib.sha256
        ).hexdigest()
        if hmac.compare_digest(expected_hash_value, hmac_hash):
            return

    raise ValueError("Integrity check for the serialized function or data failed.")
