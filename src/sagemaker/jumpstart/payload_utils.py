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
"""This module stores inference payload utilities for JumpStart models."""
from __future__ import absolute_import
import base64
import json
from typing import Any, Dict, List, Optional, Union
import re
import boto3

from sagemaker.jumpstart.accessors import JumpStartS3PayloadAccessor
from sagemaker.jumpstart.artifacts.payloads import _retrieve_example_payloads
from sagemaker.jumpstart.constants import (
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
)
from sagemaker.jumpstart.enums import JumpStartModelType, MIMEType
from sagemaker.jumpstart.types import JumpStartSerializablePayload
from sagemaker.jumpstart.utils import (
    get_jumpstart_content_bucket,
    get_region_fallback,
)
from sagemaker.session import Session


S3_BYTES_REGEX = r"^\$s3<(?P<s3_key>[a-zA-Z0-9-_/.]+)>$"
S3_B64_STR_REGEX = r"\$s3_b64<(?P<s3_key>[a-zA-Z0-9-_/.]+)>"


def _extract_field_from_json(
    json_input: dict,
    keys: List[str],
) -> Any:
    """Given a dictionary, returns value at specified keys.

    Raises:
        KeyError: If a key cannot be found in the json input.
    """
    curr_json = json_input
    for idx, key in enumerate(keys):
        if idx < len(keys) - 1:
            curr_json = curr_json[key]
            continue
        return curr_json[key]


def _construct_payload(
    prompt: str,
    model_id: str,
    model_version: str,
    region: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
    alias: Optional[str] = None,
) -> Optional[JumpStartSerializablePayload]:
    """Returns example payload from prompt.

    Args:
        prompt (str): String-valued prompt to embed in payload.
        model_id (str): JumpStart model ID of the JumpStart model for which to construct
            the payload.
        model_version (str): Version of the JumpStart model for which to retrieve the
            payload.
        region (Optional[str]): Region for which to retrieve the
            payload. (Default: None).
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions. If not
            specified, one is created using the default AWS configuration
            chain. (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
        model_type (JumpStartModelType): The type of the model, can be open weights model or
            proprietary model. (Default: JumpStartModelType.OPEN_WEIGHTS).
    Returns:
        Optional[JumpStartSerializablePayload]: serializable payload with prompt, or None if
            this feature is unavailable for the specified model.
    """
    payloads: Optional[Dict[str, JumpStartSerializablePayload]] = _retrieve_example_payloads(
        model_id=model_id,
        model_version=model_version,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
        model_type=model_type,
    )
    if payloads is None or len(payloads) == 0:
        return None

    payload_to_use: JumpStartSerializablePayload = (
        payloads[alias] if alias else list(payloads.values())[0]
    )

    prompt_key: Optional[str] = payload_to_use.prompt_key
    if prompt_key is None:
        return None

    payload_body = payload_to_use.body
    prompt_key_split = prompt_key.split(".")
    for idx, prompt_key in enumerate(prompt_key_split):
        if idx < len(prompt_key_split) - 1:
            payload_body = payload_body[prompt_key]
        else:
            payload_body[prompt_key] = prompt

    return payload_to_use


class PayloadSerializer:
    """Utility class for serializing payloads associated with JumpStart models.

    Many JumpStart models embed byte-streams into payloads corresponding to images, sounds,
    and other content types which require downloading from S3.
    """

    def __init__(
        self,
        bucket: Optional[str] = None,
        region: Optional[str] = None,
        s3_client: Optional[boto3.client] = None,
    ) -> None:
        """Initializes PayloadSerializer object."""
        self.bucket = bucket or get_jumpstart_content_bucket()
        self.region = region or get_region_fallback(
            s3_client=s3_client,
        )
        self.s3_client = s3_client

    def get_bytes_payload_with_s3_references(
        self,
        payload_str: str,
    ) -> bytes:
        """Returns bytes object corresponding to referenced S3 object.

        Raises:
            ValueError: If the raw bytes payload is not formatted correctly.
        """
        s3_keys = re.compile(S3_BYTES_REGEX).findall(payload_str)
        if len(s3_keys) != 1:
            raise ValueError("Invalid bytes payload.")

        s3_key = s3_keys[0]
        serialized_s3_object = JumpStartS3PayloadAccessor.get_object_cached(
            bucket=self.bucket, key=s3_key, region=self.region, s3_client=self.s3_client
        )

        return serialized_s3_object

    def embed_s3_references_in_str_payload(
        self,
        payload: str,
    ) -> str:
        """Inserts serialized S3 content into string payload.

        If no S3 content is embedded in payload, original string is returned.
        """
        return self._embed_s3_b64_references_in_str_payload(payload_body=payload)

    def _embed_s3_b64_references_in_str_payload(
        self,
        payload_body: str,
    ) -> str:
        """Performs base 64 encoding of payloads embedded in a payload.

        This is required so that byte-valued payloads can be transmitted efficiently
        as a utf-8 encoded string.
        """

        s3_keys = re.compile(S3_B64_STR_REGEX).findall(payload_body)
        for s3_key in s3_keys:
            b64_encoded_string = base64.b64encode(
                bytearray(
                    JumpStartS3PayloadAccessor.get_object_cached(
                        bucket=self.bucket, key=s3_key, region=self.region, s3_client=self.s3_client
                    )
                )
            ).decode()
            payload_body = payload_body.replace(f"$s3_b64<{s3_key}>", b64_encoded_string)
        return payload_body

    def embed_s3_references_in_json_payload(
        self, payload_body: Union[list, dict, str, int, float]
    ) -> Union[list, dict, str, int, float]:
        """Finds all S3 references in payload and embeds serialized S3 data.

        If no S3 references are found, the payload is returned un-modified.

        Raises:
            ValueError: If the payload has an unrecognized type.
        """
        if isinstance(payload_body, str):
            return self.embed_s3_references_in_str_payload(payload_body)
        if isinstance(payload_body, (float, int)):
            return payload_body
        if isinstance(payload_body, list):
            return [self.embed_s3_references_in_json_payload(item) for item in payload_body]
        if isinstance(payload_body, dict):
            return {
                key: self.embed_s3_references_in_json_payload(value)
                for key, value in payload_body.items()
            }
        raise ValueError(f"Payload has unrecognized type: {type(payload_body)}")

    def serialize(self, payload: JumpStartSerializablePayload) -> Union[str, bytes]:
        """Returns payload string or bytes that can be inputted to inference endpoint.

        Raises:
            ValueError: If the payload has an unrecognized type.
        """
        content_type = MIMEType.from_suffixed_type(payload.content_type)
        body = payload.body

        if content_type in {MIMEType.JSON, MIMEType.LIST_TEXT, MIMEType.X_TEXT}:
            body = self.embed_s3_references_in_json_payload(body)
        else:
            body = self.get_bytes_payload_with_s3_references(body)

        if isinstance(body, dict):
            body = json.dumps(body)
        elif not isinstance(body, str) and not isinstance(body, bytes):
            raise ValueError(f"Default payload '{body}' has unrecognized type: {type(body)}")

        return body
