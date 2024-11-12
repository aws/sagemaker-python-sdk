"""Validates the integrity of pickled file with HMAC signing."""

from __future__ import absolute_import
import secrets
import hmac
import hashlib
import os
from pathlib import Path

from sagemaker.remote_function.core.serialization import _MetaData


def generate_secret_key(nbytes: int = 32) -> str:
    """Generates secret key"""
    return secrets.token_hex(nbytes)


def compute_hash(buffer: bytes, secret_key: str) -> str:
    """Compute hash value using HMAC"""
    return hmac.new(secret_key.encode(), msg=buffer, digestmod=hashlib.sha256).hexdigest()


def perform_integrity_check(buffer: bytes, metadata_path: Path):
    """Validates the integrity of bytes by comparing the hash value"""
    secret_key = os.environ.get("SAGEMAKER_SERVE_SECRET_KEY")
    actual_hash_value = compute_hash(buffer=buffer, secret_key=secret_key)

    if not Path.exists(metadata_path):
        raise ValueError("Path to metadata.json does not exist")

    with open(str(metadata_path), "rb") as md:
        expected_hash_value = _MetaData.from_json(md.read()).sha256_hash

    if not hmac.compare_digest(expected_hash_value, actual_hash_value):
        raise ValueError("Integrity check for the serialized function or data failed.")
