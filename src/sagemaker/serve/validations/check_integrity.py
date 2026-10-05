"""Validates the integrity of pickled file with SHA-256 hash."""

from __future__ import absolute_import
import hmac
import hashlib
from pathlib import Path

from sagemaker.remote_function.core.serialization import _MetaData


def compute_hash(buffer: bytes) -> str:
    """Compute SHA-256 hash of the given buffer."""
    return hashlib.sha256(buffer).hexdigest()


def perform_integrity_check(buffer: bytes, metadata_path: Path):
    """Validates the integrity of bytes by comparing the hash value."""
    actual_hash_value = compute_hash(buffer=buffer)

    if not Path.exists(metadata_path):
        raise ValueError("Path to metadata.json does not exist")

    with open(str(metadata_path), "rb") as md:
        expected_hash_value = _MetaData.from_json(md.read()).sha256_hash

    if not hmac.compare_digest(expected_hash_value, actual_hash_value):
        raise ValueError("Integrity check for the serialized function or data failed.")
