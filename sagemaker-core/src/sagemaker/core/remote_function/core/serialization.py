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
"""SageMaker remote function data serializer/deserializer."""
from __future__ import absolute_import

import dataclasses
import json
import logging

import io

import sys
import hashlib
import hmac
import pickle
import secrets

from typing import Any, Callable, Union, Optional

import cloudpickle
from tblib import pickling_support

from sagemaker.core.remote_function.errors import (
    ServiceError,
    SerializationError,
    DeserializationError,
)
from sagemaker.core.s3 import S3Downloader, S3Uploader
from sagemaker.core.helper.session_helper import Session
from ._custom_dispatch_table import dispatch_table

# Note: do not use os.path.join for s3 uris, fails on windows

logger = logging.getLogger(__name__)


def _get_python_version():
    """Returns the current python version."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


@dataclasses.dataclass
class _MetaData:
    """Metadata about the serialized data or functions."""

    sha256_hash: str
    secret_arn: Optional[str] = None  # ARN to AWS Secrets Manager secret containing HMAC key
    version: str = "2023-04-24"
    python_version: str = _get_python_version()
    serialization_module: str = "cloudpickle"

    def to_json(self):
        """Converts metadata to json string."""
        return json.dumps(dataclasses.asdict(self)).encode()

    @staticmethod
    def from_json(s):
        """Converts json string to metadata object."""
        try:
            obj = json.loads(s)
        except json.decoder.JSONDecodeError:
            raise DeserializationError("Corrupt metadata file. It is not a valid json file.")

        sha256_hash = obj.get("sha256_hash")
        secret_arn = obj.get("secret_arn")  # May be None for legacy format
        metadata = _MetaData(sha256_hash=sha256_hash, secret_arn=secret_arn)
        metadata.version = obj.get("version")
        metadata.python_version = obj.get("python_version")
        metadata.serialization_module = obj.get("serialization_module")

        if not sha256_hash:
            raise DeserializationError(
                "Corrupt metadata file. SHA256 hash for the serialized data does not exist. "
                "Please make sure to install SageMaker SDK version >= 2.156.0 on the client side "
                "and try again."
            )

        if not (
            metadata.version == "2023-04-24" and metadata.serialization_module == "cloudpickle"
        ):
            raise DeserializationError(
                f"Corrupt metadata file. Serialization approach {s} is not supported."
            )

        return metadata


class CloudpickleSerializer:
    """Serializer using cloudpickle."""

    @staticmethod
    def serialize(obj: Any) -> bytes:
        """Serializes data object and uploads it to S3.

        Args:
            obj: object to be serialized and persisted
        Raises:
            SerializationError: when fail to serialize object to bytes.
        """
        try:
            io_buffer = io.BytesIO()
            custom_pickler = cloudpickle.CloudPickler(io_buffer)
            dt = pickle.Pickler.dispatch_table.__get__(custom_pickler)  # pylint: disable=no-member
            new_dt = dt.new_child(dispatch_table)
            pickle.Pickler.dispatch_table.__set__(  # pylint: disable=no-member
                custom_pickler, new_dt
            )
            custom_pickler.dump(obj)
            return io_buffer.getvalue()
        except Exception as e:
            if isinstance(
                e, NotImplementedError
            ) and "Instance of Run type is not allowed to be pickled." in str(e):
                raise SerializationError(
                    """You are trying to pass a sagemaker.experiments.run.Run object to
                       a remote function
                       or are trying to access a global sagemaker.experiments.run.Run object
                       from within the function. This is not supported.
                       You must use `load_run` to load an existing Run in the remote function
                       or instantiate a new Run in the function."""
                )

            raise SerializationError(
                "Error when serializing object of type [{}]: {}".format(type(obj).__name__, repr(e))
            ) from e

    @staticmethod
    def deserialize(s3_uri: str, bytes_to_deserialize: bytes) -> Any:
        """Downloads from S3 and then deserializes data objects.

        Args:
            s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
            bytes_to_deserialize: bytes to be deserialized.
        Returns :
            List of deserialized python objects.
        Raises:
            DeserializationError: when fail to serialize object to bytes.
        """

        try:
            return cloudpickle.loads(bytes_to_deserialize)
        except Exception as e:
            raise DeserializationError(
                "Error when deserializing bytes downloaded from {}: {}. "
                "NOTE: this may be caused by inconsistent sagemaker python sdk versions "
                "where remote function runs versus the one used on client side. "
                "If the sagemaker versions do not match, a warning message would "
                "be logged starting with 'Inconsistent sagemaker versions found'. "
                "Please check it to validate.".format(s3_uri, repr(e))
            ) from e


# TODO: use dask serializer in case dask distributed is installed in users' environment.
def serialize_func_to_s3(
    func: Callable, 
    sagemaker_session: Session, 
    s3_uri: str, 
    job_name: str,
    s3_kms_key: str = None
):
    """Serializes function and uploads it to S3.

    Args:
        func: function to be serialized and persisted
        sagemaker_session (sagemaker.core.helper.session.Session):
            The underlying Boto3 session which AWS service calls are delegated to.
        s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
        job_name (str): Remote function job name for secret management
        s3_kms_key (str): KMS key used to encrypt artifacts uploaded to S3.
    Raises:
        SerializationError: when fail to serialize function to bytes.
    """

    _upload_payload_and_metadata_to_s3(
        bytes_to_upload=CloudpickleSerializer.serialize(func),
        s3_uri=s3_uri,
        sagemaker_session=sagemaker_session,
        job_name=job_name,
        s3_kms_key=s3_kms_key,
    )


def deserialize_func_from_s3(sagemaker_session: Session, s3_uri: str) -> Callable:
    """Downloads from S3 and then deserializes data objects.

    This method downloads the serialized training job outputs to a temporary directory and
    then deserializes them using dask.

    Args:
        sagemaker_session (sagemaker.core.helper.session.Session):
            The underlying sagemaker session which AWS service calls are delegated to.
        s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
    Returns :
        The deserialized function.
    Raises:
        DeserializationError: when fail to serialize function to bytes.
    """
    metadata = _MetaData.from_json(
        _read_bytes_from_s3(f"{s3_uri}/metadata.json", sagemaker_session)
    )

    bytes_to_deserialize = _read_bytes_from_s3(f"{s3_uri}/payload.pkl", sagemaker_session)

    _perform_integrity_check(
        expected_hash_value=metadata.sha256_hash, 
        buffer=bytes_to_deserialize,
        sagemaker_session=sagemaker_session,
        secret_arn=metadata.secret_arn,
    )

    return CloudpickleSerializer.deserialize(f"{s3_uri}/payload.pkl", bytes_to_deserialize)


def serialize_obj_to_s3(
    obj: Any, 
    sagemaker_session: Session, 
    s3_uri: str, 
    job_name: str,
    s3_kms_key: str = None
):
    """Serializes data object and uploads it to S3.

    Args:
        obj: object to be serialized and persisted
        sagemaker_session (sagemaker.core.helper.session.Session):
            The underlying Boto3 session which AWS service calls are delegated to.
        s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
        job_name (str): Remote function job name for secret management
        s3_kms_key (str): KMS key used to encrypt artifacts uploaded to S3.
    Raises:
        SerializationError: when fail to serialize object to bytes.
    """

    _upload_payload_and_metadata_to_s3(
        bytes_to_upload=CloudpickleSerializer.serialize(obj),
        s3_uri=s3_uri,
        sagemaker_session=sagemaker_session,
        job_name=job_name,
        s3_kms_key=s3_kms_key,
    )


def json_serialize_obj_to_s3(
    obj: Any,
    json_key: str,
    sagemaker_session: Session,
    s3_uri: str,
    s3_kms_key: str = None,
):
    """Json serializes data object and uploads it to S3.

    If a function step's output is data referenced by other steps via JsonGet,
    its output should be json serialized and uploaded to S3.

    Args:
        obj: (Any) object to be serialized and persisted.
        json_key: (str) the json key pointing to function step output.
        sagemaker_session (sagemaker.core.helper.session.Session):
            The underlying Boto3 session which AWS service calls are delegated to.
        s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
        s3_kms_key (str): KMS key used to encrypt artifacts uploaded to S3.
    """
    json_serialized_result = {}
    try:
        to_dump = {json_key: obj, "Exception": None}
        json_serialized_result = json.dumps(to_dump)
    except TypeError as e:
        if "is not JSON serializable" in str(e):
            to_dump = {
                json_key: None,
                "Exception": f"The function return ({obj}) is not JSON serializable.",
            }
            json_serialized_result = json.dumps(to_dump)

    S3Uploader.upload_string_as_file_body(
        body=json_serialized_result,
        desired_s3_uri=s3_uri,
        sagemaker_session=sagemaker_session,
        kms_key=s3_kms_key,
    )


def deserialize_obj_from_s3(sagemaker_session: Session, s3_uri: str) -> Any:
    """Downloads from S3 and then deserializes data objects.

    Args:
        sagemaker_session (sagemaker.core.helper.session.Session):
            The underlying sagemaker session which AWS service calls are delegated to.
        s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
    Returns :
        Deserialized python objects.
    Raises:
        DeserializationError: when fail to serialize object to bytes.
    """

    metadata = _MetaData.from_json(
        _read_bytes_from_s3(f"{s3_uri}/metadata.json", sagemaker_session)
    )

    bytes_to_deserialize = _read_bytes_from_s3(f"{s3_uri}/payload.pkl", sagemaker_session)

    _perform_integrity_check(
        expected_hash_value=metadata.sha256_hash, 
        buffer=bytes_to_deserialize,
        sagemaker_session=sagemaker_session,
        secret_arn=metadata.secret_arn,
    )

    return CloudpickleSerializer.deserialize(f"{s3_uri}/payload.pkl", bytes_to_deserialize)


def serialize_exception_to_s3(
    exc: Exception, 
    sagemaker_session: Session, 
    s3_uri: str, 
    job_name: str,
    s3_kms_key: str = None
):
    """Serializes exception with traceback and uploads it to S3.

    Args:
        exc: Exception to be serialized and persisted
        sagemaker_session (sagemaker.core.helper.session.Session):
            The underlying Boto3 session which AWS service calls are delegated to.
        s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
        job_name (str): Remote function job name for secret management
        s3_kms_key (str): KMS key used to encrypt artifacts uploaded to S3.
    Raises:
        SerializationError: when fail to serialize object to bytes.
    """
    pickling_support.install()

    _upload_payload_and_metadata_to_s3(
        bytes_to_upload=CloudpickleSerializer.serialize(exc),
        s3_uri=s3_uri,
        sagemaker_session=sagemaker_session,
        job_name=job_name,
        s3_kms_key=s3_kms_key,
    )


def _upload_payload_and_metadata_to_s3(
    bytes_to_upload: Union[bytes, io.BytesIO],
    s3_uri: str,
    sagemaker_session: Session,
    job_name: str,
    s3_kms_key,
):
    """Uploads serialized payload and metadata to s3.

    Args:
        bytes_to_upload (bytes): Serialized bytes to upload.
        s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
        sagemaker_session (sagemaker.core.helper.session.Session):
            The underlying Boto3 session which AWS service calls are delegated to.
        job_name (str): Remote function job name for secret management
        s3_kms_key (str): KMS key used to encrypt artifacts uploaded to S3.
    """
    _upload_bytes_to_s3(bytes_to_upload, f"{s3_uri}/payload.pkl", s3_kms_key, sagemaker_session)

    # Get or create HMAC secret in Secrets Manager
    secret_arn, hmac_key = _get_or_create_hmac_secret(sagemaker_session, job_name)
    
    # Compute HMAC-SHA256 hash
    sha256_hash = _compute_hmac(bytes_to_upload, hmac_key)

    # Store secret ARN in Parameter Store as trust anchor (Mitigation #3)
    _store_secret_arn_in_parameter_store(sagemaker_session, job_name, secret_arn)

    _upload_bytes_to_s3(
        _MetaData(sha256_hash=sha256_hash, secret_arn=secret_arn).to_json(),
        f"{s3_uri}/metadata.json",
        s3_kms_key,
        sagemaker_session,
    )


def deserialize_exception_from_s3(sagemaker_session: Session, s3_uri: str) -> Any:
    """Downloads from S3 and then deserializes exception.

    Args:
        sagemaker_session (sagemaker.core.helper.session.Session):
            The underlying sagemaker session which AWS service calls are delegated to.
        s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
    Returns :
        Deserialized exception with traceback.
    Raises:
        DeserializationError: when fail to serialize object to bytes.
    """

    metadata = _MetaData.from_json(
        _read_bytes_from_s3(f"{s3_uri}/metadata.json", sagemaker_session)
    )

    bytes_to_deserialize = _read_bytes_from_s3(f"{s3_uri}/payload.pkl", sagemaker_session)

    _perform_integrity_check(
        expected_hash_value=metadata.sha256_hash, 
        buffer=bytes_to_deserialize,
        sagemaker_session=sagemaker_session,
        secret_arn=metadata.secret_arn,
    )

    return CloudpickleSerializer.deserialize(f"{s3_uri}/payload.pkl", bytes_to_deserialize)


def _upload_bytes_to_s3(b: Union[bytes, io.BytesIO], s3_uri, s3_kms_key, sagemaker_session):
    """Wrapping s3 uploading with exception translation for remote function."""
    try:
        S3Uploader.upload_bytes(b, s3_uri, kms_key=s3_kms_key, sagemaker_session=sagemaker_session)
    except Exception as e:
        raise ServiceError(
            "Failed to upload serialized bytes to {}: {}".format(s3_uri, repr(e))
        ) from e


def _read_bytes_from_s3(s3_uri, sagemaker_session):
    """Wrapping s3 downloading with exception translation for remote function."""
    try:
        return S3Downloader.read_bytes(s3_uri, sagemaker_session=sagemaker_session)
    except Exception as e:
        raise ServiceError(
            "Failed to read serialized bytes from {}: {}".format(s3_uri, repr(e))
        ) from e


def _compute_hash(buffer: bytes) -> str:
    """Compute the sha256 hash"""
    return hashlib.sha256(buffer).hexdigest()


def _get_or_create_hmac_secret(sagemaker_session: Session, job_name: str) -> tuple[str, str]:
    """Get or create HMAC key in AWS Secrets Manager.
    
    Args:
        sagemaker_session: SageMaker session
        job_name: Remote function job name
        
    Returns:
        Tuple of (secret_arn, hmac_key)
    """
    secret_name = f"sagemaker/remote-function/{job_name}/hmac-key"
    secrets_client = sagemaker_session.boto_session.client('secretsmanager')
    
    try:
        # Try to retrieve existing secret
        response = secrets_client.get_secret_value(SecretId=secret_name)
        return response['ARN'], response['SecretString']
    except secrets_client.exceptions.ResourceNotFoundException:
        # Create new secret
        hmac_key = secrets.token_hex(32)
        
        response = secrets_client.create_secret(
            Name=secret_name,
            SecretString=hmac_key,
            Description=f"HMAC key for SageMaker remote function job {job_name}",
            Tags=[
                {'Key': 'SageMaker:JobName', 'Value': job_name},
                {'Key': 'SageMaker:Purpose', 'Value': 'RemoteFunctionIntegrity'}
            ]
        )
        return response['ARN'], hmac_key


def _get_hmac_key_from_secret(sagemaker_session: Session, secret_arn: str) -> str:
    """Retrieve HMAC key from AWS Secrets Manager.
    
    Args:
        sagemaker_session: SageMaker session
        secret_arn: ARN of the secret containing HMAC key
        
    Returns:
        HMAC key string
    """
    secrets_client = sagemaker_session.boto_session.client('secretsmanager')
    response = secrets_client.get_secret_value(SecretId=secret_arn)
    return response['SecretString']


def _compute_hmac(buffer: bytes, hmac_key: str) -> str:
    """Compute HMAC-SHA256 hash.
    
    Args:
        buffer: Data to hash
        hmac_key: HMAC secret key
        
    Returns:
        HMAC-SHA256 hex digest
    """
    return hmac.new(hmac_key.encode(), msg=buffer, digestmod=hashlib.sha256).hexdigest()


def _store_secret_arn_in_parameter_store(
    sagemaker_session: Session, 
    job_name: str, 
    secret_arn: str
):
    """Store secret ARN in Parameter Store as trust anchor.
    
    Args:
        sagemaker_session: SageMaker session
        job_name: Remote function job name
        secret_arn: ARN of the secret to store
    """
    ssm_client = sagemaker_session.boto_session.client('ssm')
    parameter_name = f"/sagemaker/remote-function/{job_name}/secret-arn"
    
    try:
        ssm_client.put_parameter(
            Name=parameter_name,
            Value=secret_arn,
            Type="String",
            Description=f"Secret ARN for SageMaker remote function job {job_name}",
            Tags=[
                {'Key': 'SageMaker:JobName', 'Value': job_name},
                {'Key': 'SageMaker:Purpose', 'Value': 'RemoteFunctionIntegrity'}
            ]
        )
    except ssm_client.exceptions.ParameterAlreadyExists:
        ssm_client.put_parameter(
            Name=parameter_name,
            Value=secret_arn,
            Type="String",
            Overwrite=True,
        )


def _get_secret_arn_from_parameter_store(
    sagemaker_session: Session, 
    job_name: str
) -> str:
    """Retrieve secret ARN from Parameter Store.
    
    Args:
        sagemaker_session: SageMaker session
        job_name: Remote function job name
        
    Returns:
        Secret ARN string
        
    Raises:
        DeserializationError: If parameter not found
    """
    ssm_client = sagemaker_session.boto_session.client('ssm')
    parameter_name = f"/sagemaker/remote-function/{job_name}/secret-arn"
    
    try:
        response = ssm_client.get_parameter(Name=parameter_name)
        return response['Parameter']['Value']
    except ssm_client.exceptions.ParameterNotFound:
        raise DeserializationError(
            f"Secret ARN not found in Parameter Store for job {job_name}. "
            "This may indicate the job was not properly initialized or artifacts were tampered with."
        )


def _extract_job_name_from_secret_arn(secret_arn: str) -> str:
    """Extract job name from a Secrets Manager ARN.
    
    Secret name convention: sagemaker/remote-function/{job_name}/hmac-key
    ARN format: arn:aws:secretsmanager:region:account:secret:sagemaker/remote-function/{job_name}/hmac-key-XXXXXX
    
    Args:
        secret_arn: Full ARN of the secret
        
    Returns:
        Extracted job name
        
    Raises:
        DeserializationError: If ARN doesn't match expected format
    """
    # Length guard to prevent ReDoS on crafted inputs.
    # Real ARNs are ~165 chars (job names are max 63 chars per SageMaker API).
    MAX_SECRET_ARN_LENGTH = 256
    if len(secret_arn) > MAX_SECRET_ARN_LENGTH:
        raise DeserializationError(
            f"Secret ARN exceeds maximum length of {MAX_SECRET_ARN_LENGTH} characters"
        )

    import re
    # Use [^/]+ (non-greedy, no slashes) to prevent path-traversal in job name,
    # and anchor with $ to ensure the ARN ends with the expected suffix.
    match = re.search(
        r":secret:sagemaker/remote-function/([^/]+)/hmac-key-[A-Za-z0-9]{6}$", secret_arn
    )
    if not match:
        raise DeserializationError(
            f"Secret ARN does not match expected format "
            f"'sagemaker/remote-function/{{job_name}}/hmac-key-XXXXXX': {secret_arn}"
        )
    return match.group(1)


def _validate_secret_arn(
    sagemaker_session: Session,
    metadata_secret_arn: str,
):
    """Validate secret ARN from metadata against trusted sources.
    
    Implements two mitigations:
    1. Validate secret is in same AWS account
    2. Validate secret ARN matches Parameter Store (trust anchor)
    
    The job_name is derived from the secret ARN's naming convention, then
    independently validated against the SSM trust anchor.
    
    Args:
        sagemaker_session: SageMaker session
        metadata_secret_arn: Secret ARN from S3 metadata (untrusted)
        
    Raises:
        DeserializationError: If validation fails
    """
    # Mitigation #1: Validate same account
    sts_client = sagemaker_session.boto_session.client('sts')
    current_account_id = sts_client.get_caller_identity()['Account']
    
    # Parse account ID from ARN: arn:aws:secretsmanager:region:ACCOUNT_ID:secret:name
    arn_parts = metadata_secret_arn.split(":")
    if len(arn_parts) < 5:
        raise DeserializationError(f"Invalid secret ARN format: {metadata_secret_arn}")
    
    metadata_account_id = arn_parts[4]
    
    if metadata_account_id != current_account_id:
        raise DeserializationError(
            f"Secret must be in the same AWS account. "
            f"Expected account {current_account_id}, but got {metadata_account_id}. "
            "This may indicate a cross-account attack attempt."
        )
    
    # Mitigation #3: Validate against Parameter Store (trust anchor)
    job_name = _extract_job_name_from_secret_arn(metadata_secret_arn)
    expected_secret_arn = _get_secret_arn_from_parameter_store(sagemaker_session, job_name)
    
    if metadata_secret_arn != expected_secret_arn:
        raise DeserializationError(
            f"Secret ARN mismatch. Expected: {expected_secret_arn}, "
            f"Got: {metadata_secret_arn}. "
            "Possible tampering detected - metadata may have been modified."
        )


def _perform_integrity_check(
    expected_hash_value: str, 
    buffer: bytes,
    sagemaker_session: Optional[Session] = None,
    secret_arn: Optional[str] = None,
):
    """Performs integrity checks for serialized code/arguments uploaded to s3.

    Verifies whether the hash read from s3 matches the hash calculated
    during remote function execution.
    
    Args:
        expected_hash_value: Expected hash value from metadata
        buffer: Serialized data buffer
        sagemaker_session: SageMaker session (required for HMAC integrity check)
        secret_arn: ARN of secret containing HMAC key (required)
        
    Raises:
        DeserializationError: If integrity check fails or secret_arn is missing
    """
    if not secret_arn:
        raise DeserializationError(
            "Missing secret_arn in metadata. HMAC integrity check is required. "
            "Legacy SHA-256 integrity check is no longer supported due to security "
            "vulnerabilities. Please upgrade to the latest SDK version on both "
            "client and remote sides."
        )

    if not sagemaker_session:
        raise DeserializationError(
            "sagemaker_session is required for HMAC integrity check"
        )
    
    # Validate secret ARN (Mitigations #1 and #3)
    _validate_secret_arn(sagemaker_session, secret_arn)
    
    # Now safe to retrieve HMAC key
    hmac_key = _get_hmac_key_from_secret(sagemaker_session, secret_arn)
    actual_hash_value = _compute_hmac(buffer, hmac_key)
    
    if not hmac.compare_digest(expected_hash_value, actual_hash_value):
        raise DeserializationError(
            "HMAC integrity check failed. Serialized data may have been tampered with. "
            "Please restrict access to your S3 bucket."
        )
