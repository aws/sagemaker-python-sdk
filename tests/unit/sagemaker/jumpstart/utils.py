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
from __future__ import absolute_import
import copy
from typing import List
import boto3

from sagemaker.jumpstart.cache import JumpStartModelsCache
from sagemaker.jumpstart.constants import (
    JUMPSTART_DEFAULT_REGION_NAME,
    JUMPSTART_LOGGER,
    JUMPSTART_REGION_NAME_SET,
)
from sagemaker.jumpstart.types import (
    JumpStartCachedS3ContentKey,
    JumpStartCachedS3ContentValue,
    JumpStartModelSpecs,
    JumpStartS3FileType,
    JumpStartModelHeader,
)
from sagemaker.jumpstart.enums import JumpStartModelType
from sagemaker.jumpstart.utils import get_formatted_manifest
from tests.unit.sagemaker.jumpstart.constants import (
    PROTOTYPICAL_MODEL_SPECS_DICT,
    BASE_MANIFEST,
    BASE_SPEC,
    BASE_PROPRIETARY_MANIFEST,
    BASE_PROPRIETARY_SPEC,
    BASE_HEADER,
    BASE_PROPRIETARY_HEADER,
    SPECIAL_MODEL_SPECS_DICT,
)


def get_header_from_base_header(
    _obj: JumpStartModelsCache = None,
    region: str = None,
    model_id: str = None,
    semantic_version_str: str = None,
    version: str = None,
    model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
) -> JumpStartModelHeader:

    if version and semantic_version_str:
        raise ValueError("Cannot specify both `version` and `semantic_version_str` fields.")

    if model_type == JumpStartModelType.PROPRIETARY:
        spec = copy.deepcopy(BASE_PROPRIETARY_HEADER)
        return JumpStartModelHeader(spec)

    if all(
        [
            "pytorch" not in model_id,
            "tensorflow" not in model_id,
            "huggingface" not in model_id,
            "mxnet" not in model_id,
            "xgboost" not in model_id,
            "catboost" not in model_id,
            "lightgbm" not in model_id,
            "sklearn" not in model_id,
        ]
    ):
        raise KeyError("Bad model ID")

    if region is not None and region not in JUMPSTART_REGION_NAME_SET:
        raise ValueError(
            f"Region name {region} not supported. Please use one of the supported regions in "
            f"{JUMPSTART_REGION_NAME_SET}"
        )

    spec = copy.deepcopy(BASE_HEADER)

    spec["version"] = version or semantic_version_str
    spec["model_id"] = model_id

    return JumpStartModelHeader(spec)


def get_prototype_manifest(
    region: str = JUMPSTART_DEFAULT_REGION_NAME,
    model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
) -> List[JumpStartModelHeader]:
    if model_type == JumpStartModelType.PROPRIETARY:
        return [JumpStartModelHeader(spec) for spec in BASE_PROPRIETARY_MANIFEST]
    return [
        get_header_from_base_header(region=region, model_id=model_id, version=version)
        for model_id in PROTOTYPICAL_MODEL_SPECS_DICT.keys()
        for version in ["1.0.0"]
    ]


def get_prototype_model_spec(
    region: str = None,
    model_id: str = None,
    version: str = None,
    s3_client: boto3.client = None,
    model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
) -> JumpStartModelSpecs:
    """This function mocks cache accessor functions. For this mock,
    we only retrieve model specs based on the model ID.
    """

    JUMPSTART_LOGGER.warning("some-logging-msg")
    specs = JumpStartModelSpecs(PROTOTYPICAL_MODEL_SPECS_DICT[model_id])
    return specs


def get_special_model_spec(
    region: str = None,
    model_id: str = None,
    version: str = None,
    s3_client: boto3.client = None,
    model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
) -> JumpStartModelSpecs:
    """This function mocks cache accessor functions. For this mock,
    we only retrieve model specs based on the model ID. This is reserved
    for special specs.
    """

    specs = JumpStartModelSpecs(SPECIAL_MODEL_SPECS_DICT[model_id])
    return specs


def get_special_model_spec_for_inference_component_based_endpoint(
    region: str = None,
    model_id: str = None,
    version: str = None,
    s3_client: boto3.client = None,
    model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
) -> JumpStartModelSpecs:
    """This function mocks cache accessor functions. For this mock,
    we only retrieve model specs based on the model ID and adding
    inference component based endpoint specific specification.
    This is reserved for special specs.
    """
    model_spec_dict = SPECIAL_MODEL_SPECS_DICT[model_id]
    model_spec_dict["hosting_resource_requirements"] = {
        "num_accelerators": 1,
        "min_memory_mb": 34360,
    }
    model_spec_dict["dynamic_container_deployment_supported"] = True
    specs = JumpStartModelSpecs(model_spec_dict)
    return specs


def get_spec_from_base_spec(
    _obj: JumpStartModelsCache = None,
    region: str = None,
    model_id: str = None,
    version_str: str = None,
    version: str = None,
    s3_client: boto3.client = None,
    model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
) -> JumpStartModelSpecs:

    if version and version_str:
        raise ValueError("Cannot specify both `version` and `semantic_version_str` fields.")

    if model_type == JumpStartModelType.PROPRIETARY:
        spec = copy.deepcopy(BASE_PROPRIETARY_SPEC)
        spec["version"] = version or version_str
        spec["model_id"] = model_id

        return JumpStartModelSpecs(spec)

    if all(
        [
            "pytorch" not in model_id,
            "tensorflow" not in model_id,
            "huggingface" not in model_id,
            "mxnet" not in model_id,
            "xgboost" not in model_id,
            "catboost" not in model_id,
            "lightgbm" not in model_id,
            "sklearn" not in model_id,
        ]
    ):
        raise KeyError("Bad model ID")

    if region is not None and region not in JUMPSTART_REGION_NAME_SET:
        raise ValueError(
            f"Region name {region} not supported. Please use one of the supported regions in "
            f"{JUMPSTART_REGION_NAME_SET}"
        )

    spec = copy.deepcopy(BASE_SPEC)

    spec["version"] = version or version_str
    spec["model_id"] = model_id

    return JumpStartModelSpecs(spec)


def patched_retrieval_function(
    _modelCacheObj: JumpStartModelsCache,
    key: JumpStartCachedS3ContentKey,
    value: JumpStartCachedS3ContentValue,
) -> JumpStartCachedS3ContentValue:

    filetype, s3_key = key.file_type, key.s3_key
    if filetype == JumpStartS3FileType.OPEN_WEIGHT_MANIFEST:

        return JumpStartCachedS3ContentValue(
            formatted_content=get_formatted_manifest(BASE_MANIFEST)
        )

    if filetype == JumpStartS3FileType.OPEN_WEIGHT_SPECS:
        _, model_id, specs_version = s3_key.split("/")
        version = specs_version.replace("specs_v", "").replace(".json", "")
        return JumpStartCachedS3ContentValue(
            formatted_content=get_spec_from_base_spec(model_id=model_id, version=version)
        )

    if filetype == JumpStartS3FileType.PROPRIETARY_MANIFEST:
        return JumpStartCachedS3ContentValue(
            formatted_content=get_formatted_manifest(BASE_PROPRIETARY_MANIFEST)
        )

    if filetype == JumpStartS3FileType.PROPRIETARY_SPECS:
        _, model_id, specs_version = s3_key.split("/")
        version = specs_version.replace("proprietary_specs_", "").replace(".json", "")
        return JumpStartCachedS3ContentValue(
            formatted_content=get_spec_from_base_spec(
                model_id=model_id,
                version=version,
                model_type=JumpStartModelType.PROPRIETARY,
            )
        )

    raise ValueError(f"Bad value for filetype: {filetype}")


def overwrite_dictionary(
    base_dictionary: dict,
    dictionary_with_overwrites: dict,
) -> dict:

    for key, value in dictionary_with_overwrites.items():

        if key in base_dictionary:
            base_dictionary_entry = base_dictionary[key]
            if isinstance(base_dictionary_entry, list):
                assert isinstance(value, list)
                value += base_dictionary_entry
            if isinstance(base_dictionary_entry, dict):
                assert isinstance(value, dict)
                value.update(base_dictionary_entry)

        base_dictionary[key] = value

    return base_dictionary
