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
"""This module contains constants for JumpStart."""
from __future__ import absolute_import
from __future__ import absolute_import
import logging
import os
from typing import Dict, Set, Type
import json
import boto3
from sagemaker.core.deserializers import BaseDeserializer, JSONDeserializer
from sagemaker.core.jumpstart.enums import (
    JumpStartScriptScope,
    SerializerType,
    DeserializerType,
    MIMEType,
    JumpStartModelType,
)
from sagemaker.core.jumpstart.types import JumpStartLaunchedRegionInfo, JumpStartS3FileType
from sagemaker.core.serializers import (
    BaseSerializer,
    CSVSerializer,
    DataSerializer,
    IdentitySerializer,
    JSONSerializer,
)
from sagemaker.core.helper.session_helper import Session


SAGEMAKER_PUBLIC_HUB = "SageMakerPublicHub"
DEFAULT_TRAINING_ENTRY_POINT = "transfer_learning.py"

JUMPSTART_LOGGER = logging.getLogger("sagemaker.jumpstart")

# disable logging if env var is set
JUMPSTART_LOGGER.addHandler(
    type(
        "",
        (logging.StreamHandler,),
        {
            "emit": lambda self, *args, **kwargs: (
                logging.StreamHandler.emit(self, *args, **kwargs)
                if not os.environ.get(ENV_VARIABLE_DISABLE_JUMPSTART_LOGGING)
                else None
            )
        },
    )()
)


_CURRENT_FILE_DIRECTORY_PATH = os.path.dirname(os.path.realpath(__file__))
REGION_CONFIG_JSON_FILENAME = "region_config.json"
REGION_CONFIG_JSON_FILEPATH = os.path.join(
    _CURRENT_FILE_DIRECTORY_PATH, REGION_CONFIG_JSON_FILENAME
)


def _load_region_config(filepath: str) -> Set[JumpStartLaunchedRegionInfo]:
    """Load the JumpStart region config from a JSON file."""
    debug_msg = f"Loading JumpStart region config from '{filepath}'."
    JUMPSTART_LOGGER.debug(debug_msg)
    try:
        with open(filepath) as f:
            config = json.load(f)

        return {
            JumpStartLaunchedRegionInfo(
                region_name=region,
                content_bucket=data["content_bucket"],
                gated_content_bucket=data.get("gated_content_bucket"),
                neo_content_bucket=data.get("neo_content_bucket"),
            )
            for region, data in config.items()
        }
    except Exception:  # pylint: disable=W0703
        JUMPSTART_LOGGER.error("Unable to load JumpStart region config.", exc_info=True)
        return set()


ENV_VARIABLE_DISABLE_JUMPSTART_LOGGING = "DISABLE_JUMPSTART_LOGGING"
ENV_VARIABLE_DISABLE_JUMPSTART_TELEMETRY = "DISABLE_JUMPSTART_TELEMETRY"

JUMPSTART_LAUNCHED_REGIONS: Set[JumpStartLaunchedRegionInfo] = _load_region_config(
    REGION_CONFIG_JSON_FILEPATH
)

JUMPSTART_REGION_NAME_TO_LAUNCHED_REGION_DICT = {
    region.region_name: region for region in JUMPSTART_LAUNCHED_REGIONS
}
JUMPSTART_REGION_NAME_SET = {region.region_name for region in JUMPSTART_LAUNCHED_REGIONS}

JUMPSTART_BUCKET_NAME_SET = {region.content_bucket for region in JUMPSTART_LAUNCHED_REGIONS}
JUMPSTART_GATED_BUCKET_NAME_SET = {
    region.gated_content_bucket
    for region in JUMPSTART_LAUNCHED_REGIONS
    if region.gated_content_bucket is not None
}

JUMPSTART_GATED_AND_PUBLIC_BUCKET_NAME_SET = JUMPSTART_BUCKET_NAME_SET.union(
    JUMPSTART_GATED_BUCKET_NAME_SET
)

JUMPSTART_DEFAULT_REGION_NAME = boto3.session.Session().region_name or "us-west-2"
NEO_DEFAULT_REGION_NAME = boto3.session.Session().region_name or "us-west-2"

JUMPSTART_MODEL_HUB_NAME = "SageMakerPublicHub"

JUMPSTART_DEFAULT_MANIFEST_FILE_S3_KEY = "models_manifest.json"
JUMPSTART_DEFAULT_PROPRIETARY_MANIFEST_KEY = "proprietary-sdk-manifest.json"

HUB_CONTENT_ARN_REGEX = r"arn:(.*?):sagemaker:(.*?):(.*?):hub-content/(.*?)/(.*?)/(.*?)/(.*?)$"
HUB_ARN_REGEX = r"arn:(.*?):sagemaker:(.*?):(.*?):hub/(.*?)$"

INFERENCE_ENTRY_POINT_SCRIPT_NAME = "inference.py"
TRAINING_ENTRY_POINT_SCRIPT_NAME = "transfer_learning.py"

SUPPORTED_JUMPSTART_SCOPES = set(scope.value for scope in JumpStartScriptScope)

ENV_VARIABLE_JUMPSTART_CONTENT_BUCKET_OVERRIDE = "AWS_JUMPSTART_CONTENT_BUCKET_OVERRIDE"
ENV_VARIABLE_JUMPSTART_GATED_CONTENT_BUCKET_OVERRIDE = "AWS_JUMPSTART_GATED_CONTENT_BUCKET_OVERRIDE"
ENV_VARIABLE_JUMPSTART_MODEL_ARTIFACT_BUCKET_OVERRIDE = "AWS_JUMPSTART_MODEL_BUCKET_OVERRIDE"
ENV_VARIABLE_JUMPSTART_SCRIPT_ARTIFACT_BUCKET_OVERRIDE = "AWS_JUMPSTART_SCRIPT_BUCKET_OVERRIDE"
ENV_VARIABLE_JUMPSTART_MANIFEST_LOCAL_ROOT_DIR_OVERRIDE = (
    "AWS_JUMPSTART_MANIFEST_LOCAL_ROOT_DIR_OVERRIDE"
)
ENV_VARIABLE_JUMPSTART_SPECS_LOCAL_ROOT_DIR_OVERRIDE = "AWS_JUMPSTART_SPECS_LOCAL_ROOT_DIR_OVERRIDE"
ENV_VARIABLE_NEO_CONTENT_BUCKET_OVERRIDE = "AWS_NEO_CONTENT_BUCKET_OVERRIDE"

JUMPSTART_RESOURCE_BASE_NAME = "sagemaker-jumpstart"

SAGEMAKER_GATED_MODEL_S3_URI_TRAINING_ENV_VAR_KEY = "SageMakerGatedModelS3Uri"

PROPRIETARY_MODEL_SPEC_PREFIX = "proprietary-models"
PROPRIETARY_MODEL_FILTER_NAME = "marketplace"

CONTENT_TYPE_TO_SERIALIZER_TYPE_MAP: Dict[MIMEType, SerializerType] = {
    MIMEType.X_IMAGE: SerializerType.RAW_BYTES,
    MIMEType.LIST_TEXT: SerializerType.JSON,
    MIMEType.X_TEXT: SerializerType.TEXT,
    MIMEType.JSON: SerializerType.JSON,
    MIMEType.CSV: SerializerType.CSV,
    MIMEType.WAV: SerializerType.RAW_BYTES,
}


ACCEPT_TYPE_TO_DESERIALIZER_TYPE_MAP: Dict[MIMEType, DeserializerType] = {
    MIMEType.JSON: DeserializerType.JSON,
}

SERIALIZER_TYPE_TO_CLASS_MAP: Dict[SerializerType, Type[BaseSerializer]] = {
    SerializerType.RAW_BYTES: DataSerializer,
    SerializerType.JSON: JSONSerializer,
    SerializerType.TEXT: IdentitySerializer,
    SerializerType.CSV: CSVSerializer,
}

DESERIALIZER_TYPE_TO_CLASS_MAP: Dict[DeserializerType, Type[BaseDeserializer]] = {
    DeserializerType.JSON: JSONDeserializer,
}

MODEL_TYPE_TO_MANIFEST_MAP: Dict[Type[JumpStartModelType], Type[JumpStartS3FileType]] = {
    JumpStartModelType.OPEN_WEIGHTS: JumpStartS3FileType.OPEN_WEIGHT_MANIFEST,
    JumpStartModelType.PROPRIETARY: JumpStartS3FileType.PROPRIETARY_MANIFEST,
}

MODEL_TYPE_TO_SPECS_MAP: Dict[Type[JumpStartModelType], Type[JumpStartS3FileType]] = {
    JumpStartModelType.OPEN_WEIGHTS: JumpStartS3FileType.OPEN_WEIGHT_SPECS,
    JumpStartModelType.PROPRIETARY: JumpStartS3FileType.PROPRIETARY_SPECS,
}

MODEL_ID_LIST_WEB_URL = "https://sagemaker.readthedocs.io/en/stable/doc_utils/pretrainedmodels.html"

try:
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION = Session(
        boto3.Session(region_name=JUMPSTART_DEFAULT_REGION_NAME)
    )
except Exception as e:  # pylint: disable=W0703
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION = None
    JUMPSTART_LOGGER.warning(
        "Unable to create default JumpStart SageMaker Session due to the following error: %s.",
        str(e),
    )

EXTRA_MODEL_ID_TAGS = ["sm-jumpstart-id", "sagemaker-studio:jumpstart-model-id"]
EXTRA_MODEL_VERSION_TAGS = [
    "sm-jumpstart-model-version",
    "sagemaker-studio:jumpstart-model-version",
]
