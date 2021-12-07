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

from sagemaker.jumpstart.cache import JumpStartModelsCache
from sagemaker.jumpstart.types import (
    JumpStartCachedS3ContentKey,
    JumpStartCachedS3ContentValue,
    JumpStartModelSpecs,
    JumpStartS3FileType,
    JumpStartModelHeader,
)
from sagemaker.jumpstart.utils import get_formatted_manifest


BASE_SPEC = {
    "model_id": "pytorch-ic-mobilenet-v2",
    "version": "1.0.0",
    "min_sdk_version": "2.49.0",
    "training_supported": True,
    "incremental_training_supported": True,
    "hosting_ecr_specs": {
        "framework": "pytorch",
        "framework_version": "1.5.0",
        "py_version": "py3",
    },
    "training_ecr_specs": {
        "framework": "pytorch",
        "framework_version": "1.5.0",
        "py_version": "py3",
    },
    "hosting_artifact_key": "pytorch-infer/infer-pytorch-ic-mobilenet-v2.tar.gz",
    "training_artifact_key": "pytorch-training/train-pytorch-ic-mobilenet-v2.tar.gz",
    "hosting_script_key": "source-directory-tarballs/pytorch/inference/ic/v1.0.0/sourcedir.tar.gz",
    "training_script_key": "source-directory-tarballs/pytorch/transfer_learning/ic/v1.0.0/sourcedir.tar.gz",
    "hyperparameters": {
        "adam-learning-rate": {"type": "float", "default": 0.05, "min": 1e-08, "max": 1},
        "epochs": {"type": "int", "default": 3, "min": 1, "max": 1000},
        "batch-size": {"type": "int", "default": 4, "min": 1, "max": 1024},
    },
}

BASE_HEADER = {
    "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
    "version": "1.0.0",
    "min_version": "2.49.0",
    "spec_key": "community_models_specs/tensorflow-ic-imagenet"
    "-inception-v3-classification-4/specs_v1.0.0.json",
}

BASE_MANIFEST = [
    {
        "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
        "version": "1.0.0",
        "min_version": "2.49.0",
        "spec_key": "community_models_specs/tensorflow-ic-imagenet"
        "-inception-v3-classification-4/specs_v1.0.0.json",
    },
    {
        "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
        "version": "2.0.0",
        "min_version": "2.49.0",
        "spec_key": "community_models_specs/tensorflow-ic-imagenet"
        "-inception-v3-classification-4/specs_v2.0.0.json",
    },
    {
        "model_id": "pytorch-ic-imagenet-inception-v3-classification-4",
        "version": "1.0.0",
        "min_version": "2.49.0",
        "spec_key": "community_models_specs/pytorch-ic-"
        "imagenet-inception-v3-classification-4/specs_v1.0.0.json",
    },
    {
        "model_id": "pytorch-ic-imagenet-inception-v3-classification-4",
        "version": "2.0.0",
        "min_version": "2.49.0",
        "spec_key": "community_models_specs/pytorch-ic-imagenet-"
        "inception-v3-classification-4/specs_v2.0.0.json",
    },
    {
        "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
        "version": "3.0.0",
        "min_version": "4.49.0",
        "spec_key": "community_models_specs/tensorflow-ic-"
        "imagenet-inception-v3-classification-4/specs_v3.0.0.json",
    },
]


def get_header_from_base_header(
    region: str = None, model_id: str = None, version: str = None
) -> JumpStartModelHeader:

    if "pytorch" not in model_id and "tensorflow" not in model_id:
        raise KeyError("Bad model id")

    spec = copy.deepcopy(BASE_HEADER)

    spec["version"] = version
    spec["model_id"] = model_id

    return JumpStartModelHeader(spec)


def get_spec_from_base_spec(
    region: str = None, model_id: str = None, version: str = None
) -> JumpStartModelSpecs:

    if "pytorch" not in model_id and "tensorflow" not in model_id:
        raise KeyError("Bad model id")

    spec = copy.deepcopy(BASE_SPEC)

    spec["version"] = version
    spec["model_id"] = model_id

    return JumpStartModelSpecs(spec)


def patched_get_file_from_s3(
    _modelCacheObj: JumpStartModelsCache,
    key: JumpStartCachedS3ContentKey,
    value: JumpStartCachedS3ContentValue,
) -> JumpStartCachedS3ContentValue:

    filetype, s3_key = key.file_type, key.s3_key
    if filetype == JumpStartS3FileType.MANIFEST:

        return JumpStartCachedS3ContentValue(
            formatted_content=get_formatted_manifest(BASE_MANIFEST)
        )

    if filetype == JumpStartS3FileType.SPECS:
        _, model_id, specs_version = s3_key.split("/")
        version = specs_version.replace("specs_v", "").replace(".json", "")
        return JumpStartCachedS3ContentValue(
            formatted_content=get_spec_from_base_spec(model_id=model_id, version=version)
        )

    raise ValueError(f"Bad value for filetype: {filetype}")
