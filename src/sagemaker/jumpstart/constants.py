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
"""This module stores constants related to SageMaker JumpStart."""
from __future__ import absolute_import
import logging
import os
from typing import Dict, Set, Type
import boto3
from sagemaker.base_deserializers import BaseDeserializer, JSONDeserializer
from sagemaker.jumpstart.enums import (
    JumpStartScriptScope,
    SerializerType,
    DeserializerType,
    MIMEType,
    JumpStartModelType,
)
from sagemaker.jumpstart.types import JumpStartLaunchedRegionInfo, JumpStartS3FileType
from sagemaker.base_serializers import (
    BaseSerializer,
    CSVSerializer,
    DataSerializer,
    IdentitySerializer,
    JSONSerializer,
)
from sagemaker.session import Session


ENV_VARIABLE_DISABLE_JUMPSTART_LOGGING = "DISABLE_JUMPSTART_LOGGING"
ENV_VARIABLE_DISABLE_JUMPSTART_TELEMETRY = "DISABLE_JUMPSTART_TELEMETRY"

JUMPSTART_LAUNCHED_REGIONS: Set[JumpStartLaunchedRegionInfo] = set(
    [
        JumpStartLaunchedRegionInfo(
            region_name="us-west-2",
            content_bucket="jumpstart-cache-prod-us-west-2",
            gated_content_bucket="jumpstart-private-cache-prod-us-west-2",
            neo_content_bucket="sagemaker-sd-models-prod-us-west-2",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="us-east-1",
            content_bucket="jumpstart-cache-prod-us-east-1",
            gated_content_bucket="jumpstart-private-cache-prod-us-east-1",
            neo_content_bucket="sagemaker-sd-models-prod-us-east-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="us-east-2",
            content_bucket="jumpstart-cache-prod-us-east-2",
            gated_content_bucket="jumpstart-private-cache-prod-us-east-2",
            neo_content_bucket="sagemaker-sd-models-prod-us-east-2",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="eu-west-1",
            content_bucket="jumpstart-cache-prod-eu-west-1",
            gated_content_bucket="jumpstart-private-cache-prod-eu-west-1",
            neo_content_bucket="sagemaker-sd-models-prod-eu-west-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="eu-central-1",
            content_bucket="jumpstart-cache-prod-eu-central-1",
            gated_content_bucket="jumpstart-private-cache-prod-eu-central-1",
            neo_content_bucket="sagemaker-sd-models-prod-eu-central-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="eu-north-1",
            content_bucket="jumpstart-cache-prod-eu-north-1",
            gated_content_bucket="jumpstart-private-cache-prod-eu-north-1",
            neo_content_bucket="sagemaker-sd-models-prod-eu-north-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="me-south-1",
            content_bucket="jumpstart-cache-prod-me-south-1",
            gated_content_bucket="jumpstart-private-cache-prod-me-south-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="me-central-1",
            content_bucket="jumpstart-cache-prod-me-central-1",
            gated_content_bucket="jumpstart-private-cache-prod-me-central-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="ap-south-1",
            content_bucket="jumpstart-cache-prod-ap-south-1",
            gated_content_bucket="jumpstart-private-cache-prod-ap-south-1",
            neo_content_bucket="sagemaker-sd-models-prod-ap-south-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="eu-west-3",
            content_bucket="jumpstart-cache-prod-eu-west-3",
            gated_content_bucket="jumpstart-private-cache-prod-eu-west-3",
            neo_content_bucket="sagemaker-sd-models-prod-eu-west-3",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="af-south-1",
            content_bucket="jumpstart-cache-prod-af-south-1",
            gated_content_bucket="jumpstart-private-cache-prod-af-south-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="sa-east-1",
            content_bucket="jumpstart-cache-prod-sa-east-1",
            gated_content_bucket="jumpstart-private-cache-prod-sa-east-1",
            neo_content_bucket="sagemaker-sd-models-prod-sa-east-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="ap-east-1",
            content_bucket="jumpstart-cache-prod-ap-east-1",
            gated_content_bucket="jumpstart-private-cache-prod-ap-east-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="ap-northeast-2",
            content_bucket="jumpstart-cache-prod-ap-northeast-2",
            gated_content_bucket="jumpstart-private-cache-prod-ap-northeast-2",
            neo_content_bucket="sagemaker-sd-models-prod-ap-northeast-2",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="ap-northeast-3",
            content_bucket="jumpstart-cache-prod-ap-northeast-3",
            gated_content_bucket="jumpstart-private-cache-prod-ap-northeast-3",
            neo_content_bucket="sagemaker-sd-models-prod-ap-northeast-3",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="ap-southeast-3",
            content_bucket="jumpstart-cache-prod-ap-southeast-3",
            gated_content_bucket="jumpstart-private-cache-prod-ap-southeast-3",
            neo_content_bucket="sagemaker-sd-models-prod-ap-southeast-3",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="eu-west-2",
            content_bucket="jumpstart-cache-prod-eu-west-2",
            gated_content_bucket="jumpstart-private-cache-prod-eu-west-2",
            neo_content_bucket="sagemaker-sd-models-prod-eu-west-2",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="eu-south-1",
            content_bucket="jumpstart-cache-prod-eu-south-1",
            gated_content_bucket="jumpstart-private-cache-prod-eu-south-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="ap-northeast-1",
            content_bucket="jumpstart-cache-prod-ap-northeast-1",
            gated_content_bucket="jumpstart-private-cache-prod-ap-northeast-1",
            neo_content_bucket="sagemaker-sd-models-prod-ap-northeast-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="us-west-1",
            content_bucket="jumpstart-cache-prod-us-west-1",
            gated_content_bucket="jumpstart-private-cache-prod-us-west-1",
            neo_content_bucket="sagemaker-sd-models-prod-us-west-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="ap-southeast-1",
            content_bucket="jumpstart-cache-prod-ap-southeast-1",
            gated_content_bucket="jumpstart-private-cache-prod-ap-southeast-1",
            neo_content_bucket="sagemaker-sd-models-prod-ap-southeast-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="ap-southeast-2",
            content_bucket="jumpstart-cache-prod-ap-southeast-2",
            gated_content_bucket="jumpstart-private-cache-prod-ap-southeast-2",
            neo_content_bucket="sagemaker-sd-models-prod-ap-southeast-2",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="ca-central-1",
            content_bucket="jumpstart-cache-prod-ca-central-1",
            gated_content_bucket="jumpstart-private-cache-prod-ca-central-1",
            neo_content_bucket="sagemaker-sd-models-prod-ca-central-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="cn-north-1",
            content_bucket="jumpstart-cache-prod-cn-north-1",
        ),
        JumpStartLaunchedRegionInfo(
            region_name="il-central-1",
            content_bucket="jumpstart-cache-prod-il-central-1",
            gated_content_bucket="jumpstart-private-cache-prod-il-central-1",
        ),
    ]
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
