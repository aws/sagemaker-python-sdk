"""Validates the integrity of pickled file with SHA256 hashing."""

from __future__ import absolute_import
import hashlib
from pathlib import Path

from sagemaker.core.remote_function.core.serialization import _MetaData


def compute_hash(buffer: bytes) -> str:
    """Compute SHA256 hash value of buffer.
    
    Args:
        buffer: Bytes to hash
        
    Returns:
        Hexadecimal SHA256 hash string
    """
    return hashlib.sha256(buffer).hexdigest()


def perform_integrity_check(buffer: bytes, metadata_path: Path):
    """Validates the integrity of bytes by comparing SHA256 hash.
    
    Args:
        buffer: Bytes to verify
        metadata_path: Path to metadata.json file
        
    Raises:
        ValueError: If metadata file doesn't exist or hash doesn't match
    """
    if not Path.exists(metadata_path):
        raise ValueError("Path to metadata.json does not exist")

    with open(str(metadata_path), "rb") as md:
        metadata = _MetaData.from_json(md.read())
        expected_hash_value = metadata.sha256_hash

    actual_hash_value = compute_hash(buffer=buffer)

    if expected_hash_value != actual_hash_value:
        raise ValueError("Integrity check for the serialized function or data failed.")
