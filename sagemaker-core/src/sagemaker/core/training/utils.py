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
"""Training utilities."""
from __future__ import absolute_import

import io
import json
import os
import tarfile
from typing import Any, Literal
from urllib.parse import urlparse
from sagemaker.core.utils.utils import Unassigned


def convert_unassigned_to_none(instance) -> Any:
    """Convert Unassigned values to None for any instance."""
    for name, value in instance.__dict__.items():
        if isinstance(value, Unassigned):
            setattr(instance, name, None)
    return instance


def _is_valid_path(path: str, path_type: Literal["File", "Directory", "Any"] = "Any") -> bool:
    """Check if the path is a valid local path.

    Args:
        path (str): Local path to validate
        path_type (Optional(Literal["File", "Directory", "Any"])): The type of the path to validate.
            Defaults to "Any".

    Returns:
        bool: True if the path is a valid local path, False otherwise
    """
    if not os.path.exists(path):
        return False

    if path_type == "File":
        return os.path.isfile(path)
    if path_type == "Directory":
        return os.path.isdir(path)

    return path_type == "Any"


def _is_valid_s3_uri(path: str, path_type: Literal["File", "Directory", "Any"] = "Any") -> bool:
    """Check if the path is a valid S3 URI.

    This method checks if the path is a valid S3 URI. If the path_type is specified,
    it will also check if the path is a file or a directory.
    This method does not check if the S3 bucket or object exists.

    Args:
        path (str): S3 URI to validate
        path_type (Optional(Literal["File", "Directory", "Any"])): The type of the path to validate.
            Defaults to "Any".

    Returns:
        bool: True if the path is a valid S3 URI, False otherwise
    """
    # Check if the path is a valid S3 URI
    if not path.startswith("s3://"):
        return False

    if path_type == "File":
        # If it's a file, it should not end with a slash
        return not path.endswith("/")
    if path_type == "Directory":
        # If it's a directory, it should end with a slash
        return path.endswith("/")

    return path_type == "Any"


_MANIFEST_CHECKPOINT_KEY = "checkpoint_s3_bucket"


def build_nova_hyperpod_manifest_s3_uri(s3_output_path: str, training_job_name: str) -> str:
    """Build the HyperPod manifest.json S3 URI for a Nova training job.

    HyperPod jobs write the manifest directly under the job directory:
    ``<s3_output_path>/<training_job_name>/manifest.json``.

    Args:
        s3_output_path: The training job's ``output_data_config.s3_output_path``.
        training_job_name: The training job name.

    Returns:
        Fully-qualified S3 URI to the job's manifest.json.
    """
    output_path = s3_output_path.rstrip("/")
    return f"{output_path}/{training_job_name}/manifest.json"


def build_nova_manifest_s3_uri(s3_output_path: str, training_job_name: str) -> str:
    """Build the serverless manifest.json S3 URI for a Nova training job.

    Serverless jobs write the manifest under a nested output directory:
    ``<s3_output_path>/<training_job_name>/output/output/manifest.json``.

    Args:
        s3_output_path: The training job's ``output_data_config.s3_output_path``.
        training_job_name: The training job name.

    Returns:
        Fully-qualified S3 URI to the job's manifest.json.
    """
    output_path = s3_output_path.rstrip("/")
    return f"{output_path}/{training_job_name}/output/output/manifest.json"


def build_nova_output_tar_gz_s3_uri(s3_output_path: str, training_job_name: str) -> str:
    """Build the output.tar.gz S3 URI for a Nova training job.

    Args:
        s3_output_path: The training job's ``output_data_config.s3_output_path``.
        training_job_name: The training job name.

    Returns:
        Fully-qualified S3 URI to the job's output.tar.gz.
    """
    output_path = s3_output_path.rstrip("/")
    return f"{output_path}/{training_job_name}/output/output.tar.gz"


def _split_s3_uri(s3_uri: str) -> tuple:
    """Split an S3 URI into (bucket, key)."""
    parsed = urlparse(s3_uri)
    return parsed.netloc, parsed.path.lstrip("/")


def read_nova_checkpoint_uri_from_manifest(s3_client, s3_uri: str) -> str:
    """Read the checkpoint URI from a raw manifest.json object in S3.

    Args:
        s3_client: A boto3 S3 client.
        s3_uri: S3 URI of the manifest.json object.

    Returns:
        The ``checkpoint_s3_bucket`` value from the manifest.

    Raises:
        ValueError: If the object is missing, unparseable, or lacks the key.
    """
    bucket, key = _split_s3_uri(s3_uri)
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        manifest = json.loads(response["Body"].read().decode("utf-8"))
    except s3_client.exceptions.NoSuchKey:
        raise ValueError(f"manifest.json not found at s3://{bucket}/{key}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse manifest.json: {e}")

    checkpoint_uri = manifest.get(_MANIFEST_CHECKPOINT_KEY)
    if not checkpoint_uri:
        raise ValueError(
            f"'{_MANIFEST_CHECKPOINT_KEY}' not found in manifest.json. "
            f"Available keys: {list(manifest.keys())}"
        )
    return checkpoint_uri


def _read_checkpoint_uri_from_tar_gz(s3_client, s3_uri: str) -> str:
    """Read the checkpoint URI from a manifest.json inside an output.tar.gz in S3.

    Args:
        s3_client: A boto3 S3 client.
        s3_uri: S3 URI of the output.tar.gz object.

    Returns:
        The ``checkpoint_s3_bucket`` value from the embedded manifest.

    Raises:
        ValueError: If the archive or manifest is missing or lacks the key.
    """
    bucket, key = _split_s3_uri(s3_uri)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    body = response["Body"].read()
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith("manifest.json"):
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            manifest = json.loads(extracted.read().decode("utf-8"))
            checkpoint_uri = manifest.get(_MANIFEST_CHECKPOINT_KEY)
            if checkpoint_uri:
                return checkpoint_uri

    raise ValueError(
        f"'{_MANIFEST_CHECKPOINT_KEY}' not found in manifest.json within "
        f"s3://{bucket}/{key}"
    )


def resolve_nova_checkpoint_uri(
    s3_client,
    s3_output_path: str,
    training_job_name: str,
) -> str:
    """Resolve the Nova checkpoint (escrow) URI from a training job's output.

    Reads ``checkpoint_s3_bucket`` from the job's manifest.json. The manifest is
    first looked up as a raw object, and if that fails, it falls back to the copy
    packaged inside ``output.tar.gz``.

    Args:
        s3_client: A boto3 S3 client.
        s3_output_path: The training job's ``output_data_config.s3_output_path``.
        training_job_name: The training job name.

    Returns:
        The checkpoint URI recorded in the manifest.

    Raises:
        ValueError: If the checkpoint URI cannot be resolved from any known
            output layout.
    """
    # Nova jobs write their manifest to different locations depending on the
    # training platform:
    #   HyperPod:   <output>/<job>/manifest.json
    #   Serverless: <output>/<job>/output/output/manifest.json
    #   Serverful:  <output>/<job>/output/output.tar.gz  (manifest is inside)
    # Try each in turn and surface every failure if none resolve, so the real
    # cause is not masked by a misleading message from the last attempt.
    hyperpod_manifest_uri = build_nova_hyperpod_manifest_s3_uri(
        s3_output_path, training_job_name
    )
    serverless_manifest_uri = build_nova_manifest_s3_uri(s3_output_path, training_job_name)
    tar_gz_uri = build_nova_output_tar_gz_s3_uri(s3_output_path, training_job_name)

    attempts = [
        ("HyperPod manifest.json", hyperpod_manifest_uri, read_nova_checkpoint_uri_from_manifest),
        ("serverless manifest.json", serverless_manifest_uri, read_nova_checkpoint_uri_from_manifest),
        ("serverful output.tar.gz", tar_gz_uri, _read_checkpoint_uri_from_tar_gz),
    ]

    errors = []
    for label, uri, reader in attempts:
        try:
            return reader(s3_client, uri)
        except Exception as error:  # noqa: PERF203 - each attempt may fail independently
            errors.append(f"{label} at {uri} failed: {error}")

    raise ValueError(
        "Could not resolve the Nova checkpoint URI from any known output layout. "
        + " ".join(errors)
    )
