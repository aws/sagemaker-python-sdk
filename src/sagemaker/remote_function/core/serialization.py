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
import os
import sys

import cloudpickle

from typing import Any, Callable
from sagemaker.remote_function.errors import ServiceError, SerializationError, DeserializationError
from sagemaker.s3 import S3Downloader, S3Uploader
from tblib import pickling_support


def _get_python_version():
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


@dataclasses.dataclass
class _MetaData:
    """Metadata about the serialized data or functions."""

    version: str = "2023-04-24"
    python_version: str = _get_python_version()
    serialization_module: str = "cloudpickle"

    def to_json(self):
        return json.dumps(dataclasses.asdict(self)).encode()

    @staticmethod
    def from_json(s):
        try:
            obj = json.loads(s)
        except json.decoder.JSONDecodeError:
            raise DeserializationError("Corrupt metadata file. It is not a valid json file.")

        metadata = _MetaData()
        metadata.version = obj.get("version")
        metadata.python_version = obj.get("python_version")
        metadata.serialization_module = obj.get("serialization_module")

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
    def serialize(obj: Any, sagemaker_session, s3_uri: str, s3_kms_key: str = None):
        """Serializes data object and uploads it to S3.

        Args:
            sagemaker_session (sagemaker.session.Session): The underlying Boto3 session which AWS service
                 calls are delegated to.
            s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
            s3_kms_key (str): KMS key used to encrypt artifacts uploaded to S3.
            obj: object to be serialized and persisted
        Raises:
            SerializationError: when fail to serialize object to bytes.
        """
        try:
            bytes_to_upload = cloudpickle.dumps(obj)
        except Exception as e:
            if isinstance(
                e, NotImplementedError
            ) and "Instance of Run type is not allowed to be pickled." in str(e):
                raise SerializationError(
                    """You are trying to pass a sagemaker.experiments.run.Run object to a remote function
                       or are trying to access a global sagemaker.experiments.run.Run object from within the function.
                       This is not supported. You must use `load_run` to load an existing Run in the remote function
                       or instantiate a new Run in the function."""
                ) from e

            raise SerializationError(
                "Error when serializing object of type [{}]: {}".format(type(obj).__name__, repr(e))
            ) from e

        _upload_bytes_to_s3(bytes_to_upload, s3_uri, s3_kms_key, sagemaker_session)

    @staticmethod
    def deserialize(sagemaker_session, s3_uri) -> Any:
        """Downloads from S3 and then deserializes data objects.

        Args:
            sagemaker_session (sagemaker.session.Session): The underlying sagemaker session which AWS service
                 calls are delegated to.
            s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
        Returns :
            List of deserialized python objects.
        Raises:
            DeserializationError: when fail to serialize object to bytes.
        """
        bytes_to_deserialize = _read_bytes_from_s3(s3_uri, sagemaker_session)

        try:
            return cloudpickle.loads(bytes_to_deserialize)
        except Exception as e:
            raise DeserializationError(
                "Error when deserializing bytes downloaded from {}: {}".format(s3_uri, repr(e))
            ) from e


# TODO: use dask serializer in case dask distributed is installed in users' environment.
def serialize_func_to_s3(func: Callable, sagemaker_session, s3_uri, s3_kms_key=None):
    """Serializes function and uploads it to S3.

    Args:
        sagemaker_session (sagemaker.session.Session): The underlying Boto3 session which AWS service
             calls are delegated to.
        s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
        s3_kms_key (str): KMS key used to encrypt artifacts uploaded to S3.
        func: function to be serialized and persisted
    Raises:
        SerializationError: when fail to serialize function to bytes.
    """

    _upload_bytes_to_s3(
        _MetaData().to_json(), os.path.join(s3_uri, "metadata.json"), s3_kms_key, sagemaker_session
    )
    CloudpickleSerializer.serialize(
        func, sagemaker_session, os.path.join(s3_uri, "payload.pkl"), s3_kms_key
    )


def deserialize_func_from_s3(sagemaker_session, s3_uri) -> Callable:
    """Downloads from S3 and then deserializes data objects.

    This method downloads the serialized training job outputs to a temporary directory and
    then deserializes them using dask.

    Args:
        sagemaker_session (sagemaker.session.Session): The underlying sagemaker session which AWS service
             calls are delegated to.
        s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
    Returns :
        The deserialized function.
    Raises:
        DeserializationError: when fail to serialize function to bytes.
    """
    _MetaData.from_json(
        _read_bytes_from_s3(os.path.join(s3_uri, "metadata.json"), sagemaker_session)
    )

    return CloudpickleSerializer.deserialize(sagemaker_session, os.path.join(s3_uri, "payload.pkl"))


def serialize_obj_to_s3(obj: Any, sagemaker_session, s3_uri: str, s3_kms_key: str = None):
    """Serializes data object and uploads it to S3.

    Args:
        sagemaker_session (sagemaker.session.Session): The underlying Boto3 session which AWS service
             calls are delegated to.
        s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
        s3_kms_key (str): KMS key used to encrypt artifacts uploaded to S3.
        obj: object to be serialized and persisted
    Raises:
        SerializationError: when fail to serialize object to bytes.
    """

    _upload_bytes_to_s3(
        _MetaData().to_json(), os.path.join(s3_uri, "metadata.json"), s3_kms_key, sagemaker_session
    )
    CloudpickleSerializer.serialize(
        obj, sagemaker_session, os.path.join(s3_uri, "payload.pkl"), s3_kms_key
    )


def deserialize_obj_from_s3(sagemaker_session, s3_uri) -> Any:
    """Downloads from S3 and then deserializes data objects.

    Args:
        sagemaker_session (sagemaker.session.Session): The underlying sagemaker session which AWS service
             calls are delegated to.
        s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
    Returns :
        Deserialized python objects.
    Raises:
        DeserializationError: when fail to serialize object to bytes.
    """

    _MetaData.from_json(
        _read_bytes_from_s3(os.path.join(s3_uri, "metadata.json"), sagemaker_session)
    )

    return CloudpickleSerializer.deserialize(sagemaker_session, os.path.join(s3_uri, "payload.pkl"))


def serialize_exception_to_s3(
    exc: Exception, sagemaker_session, s3_uri: str, s3_kms_key: str = None
):
    """Serializes exception with traceback and uploads it to S3.

    Args:
        sagemaker_session (sagemaker.session.Session): The underlying Boto3 session which AWS service
             calls are delegated to.
        s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
        s3_kms_key (str): KMS key used to encrypt artifacts uploaded to S3.
        exc: Exception to be serialized and persisted
    Raises:
        SerializationError: when fail to serialize object to bytes.
    """
    pickling_support.install()
    _upload_bytes_to_s3(
        _MetaData().to_json(), os.path.join(s3_uri, "metadata.json"), s3_kms_key, sagemaker_session
    )
    CloudpickleSerializer.serialize(
        exc, sagemaker_session, os.path.join(s3_uri, "payload.pkl"), s3_kms_key
    )


def deserialize_exception_from_s3(sagemaker_session, s3_uri) -> Any:
    """Downloads from S3 and then deserializes exception.

    Args:
        sagemaker_session (sagemaker.session.Session): The underlying sagemaker session which AWS service
             calls are delegated to.
        s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
    Returns :
        Deserialized exception with traceback.
    Raises:
        DeserializationError: when fail to serialize object to bytes.
    """

    _MetaData.from_json(
        _read_bytes_from_s3(os.path.join(s3_uri, "metadata.json"), sagemaker_session)
    )

    return CloudpickleSerializer.deserialize(sagemaker_session, os.path.join(s3_uri, "payload.pkl"))


def _upload_bytes_to_s3(bytes, s3_uri, s3_kms_key, sagemaker_session):
    """Wrapping s3 uploading with exception translation for remote function."""
    try:
        S3Uploader.upload_bytes(
            bytes, s3_uri, kms_key=s3_kms_key, sagemaker_session=sagemaker_session
        )
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
