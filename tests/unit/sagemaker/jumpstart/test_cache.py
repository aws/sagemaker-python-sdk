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
import datetime
import io
import json
import botocore

from mock.mock import MagicMock
import pytest
from mock import patch

from sagemaker.jumpstart.cache import DEFAULT_MANIFEST_FILE_S3_KEY, JumpStartModelsCache
from sagemaker.jumpstart.types import (
    JumpStartCachedS3ContentKey,
    JumpStartCachedS3ContentValue,
    JumpStartModelHeader,
    JumpStartModelSpecs,
    JumpStartS3FileType,
    JumpStartVersionedModelId,
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
        "framework_version": "1.7.0",
        "py_version": "py3",
    },
    "training_ecr_specs": {
        "framework": "pytorch",
        "framework_version": "1.9.0",
        "py_version": "py3",
    },
    "hosting_artifact_uri": "pytorch-infer/infer-pytorch-ic-mobilenet-v2.tar.gz",
    "training_artifact_uri": "pytorch-training/train-pytorch-ic-mobilenet-v2.tar.gz",
    "hosting_script_uri": "source-directory-tarballs/pytorch/inference/ic/v1.0.0/sourcedir.tar.gz",
    "training_script_uri": "source-directory-tarballs/pytorch/transfer_learning/ic/v1.0.0/sourcedir.tar.gz",
    "hyperparameters": {
        "adam-learning-rate": {"type": "float", "default": 0.05, "min": 1e-08, "max": 1},
        "epochs": {"type": "int", "default": 3, "min": 1, "max": 1000},
        "batch-size": {"type": "int", "default": 4, "min": 1, "max": 1024},
    },
}


def get_spec_from_base_spec(model_id: str, version: str) -> JumpStartModelSpecs:
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
        manifest = [
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
        return JumpStartCachedS3ContentValue(
            formatted_file_content=get_formatted_manifest(manifest)
        )

    if filetype == JumpStartS3FileType.SPECS:
        _, model_id, specs_version = s3_key.split("/")
        version = specs_version.replace("specs_v", "").replace(".json", "")
        return JumpStartCachedS3ContentValue(
            formatted_file_content=get_spec_from_base_spec(model_id, version)
        )

    raise ValueError(f"Bad value for filetype: {filetype}")


@patch.object(JumpStartModelsCache, "_get_file_from_s3", patched_get_file_from_s3)
@patch("sagemaker.jumpstart.utils.get_sagemaker_version", lambda: "2.68.3")
def test_jumpstart_cache_get_header():

    cache = JumpStartModelsCache(bucket="some_bucket")

    assert (
        JumpStartModelHeader(
            {
                "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
                "version": "2.0.0",
                "min_version": "2.49.0",
                "spec_key": "community_models_specs/tensorflow-ic"
                "-imagenet-inception-v3-classification-4/specs_v2.0.0.json",
            }
        )
        == cache.get_header(model_id="tensorflow-ic-imagenet-inception-v3-classification-4")
    )

    # See if we can make the same query 2 times consecutively
    assert (
        JumpStartModelHeader(
            {
                "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
                "version": "2.0.0",
                "min_version": "2.49.0",
                "spec_key": "community_models_specs/tensorflow-ic"
                "-imagenet-inception-v3-classification-4/specs_v2.0.0.json",
            }
        )
        == cache.get_header(model_id="tensorflow-ic-imagenet-inception-v3-classification-4")
    )

    assert JumpStartModelHeader(
        {
            "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
            "version": "2.0.0",
            "min_version": "2.49.0",
            "spec_key": "community_models_specs/tensorflow-ic"
            "-imagenet-inception-v3-classification-4/specs_v2.0.0.json",
        }
    ) == cache.get_header(
        model_id="tensorflow-ic-imagenet-inception-v3-classification-4", semantic_version_str="*"
    )

    assert JumpStartModelHeader(
        {
            "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
            "version": "2.0.0",
            "min_version": "2.49.0",
            "spec_key": "community_models_specs/tensorflow-ic-"
            "imagenet-inception-v3-classification-4/specs_v2.0.0.json",
        }
    ) == cache.get_header(
        model_id="tensorflow-ic-imagenet-inception-v3-classification-4", semantic_version_str="2.*"
    )

    assert JumpStartModelHeader(
        {
            "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
            "version": "2.0.0",
            "min_version": "2.49.0",
            "spec_key": "community_models_specs/tensorflow-ic-"
            "imagenet-inception-v3-classification-4/specs_v2.0.0.json",
        }
    ) == cache.get_header(
        model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
        semantic_version_str="2.*.*",
    )

    assert JumpStartModelHeader(
        {
            "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
            "version": "2.0.0",
            "min_version": "2.49.0",
            "spec_key": "community_models_specs/tensorflow-ic-"
            "imagenet-inception-v3-classification-4/specs_v2.0.0.json",
        }
    ) == cache.get_header(
        model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
        semantic_version_str="2.0.0",
    )

    assert JumpStartModelHeader(
        {
            "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
            "version": "1.0.0",
            "min_version": "2.49.0",
            "spec_key": "community_models_specs/tensorflow-ic-"
            "imagenet-inception-v3-classification-4/specs_v1.0.0.json",
        }
    ) == cache.get_header(
        model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
        semantic_version_str="1.0.0",
    )

    assert JumpStartModelHeader(
        {
            "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
            "version": "1.0.0",
            "min_version": "2.49.0",
            "spec_key": "community_models_specs/tensorflow-ic-"
            "imagenet-inception-v3-classification-4/specs_v1.0.0.json",
        }
    ) == cache.get_header(
        model_id="tensorflow-ic-imagenet-inception-v3-classification-4", semantic_version_str="1.*"
    )

    assert JumpStartModelHeader(
        {
            "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
            "version": "1.0.0",
            "min_version": "2.49.0",
            "spec_key": "community_models_specs/tensorflow-ic-"
            "imagenet-inception-v3-classification-4/specs_v1.0.0.json",
        }
    ) == cache.get_header(
        model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
        semantic_version_str="1.*.*",
    )

    with pytest.raises(KeyError) as e:
        cache.get_header(
            model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
            semantic_version_str="3.*",
        )
    assert (
        "Unable to find model manifest for tensorflow-ic-imagenet-inception-v3-classification-4 "
        "with version 3.* compatible with your SageMaker version (2.68.3). Consider upgrading "
        "your SageMaker library to at least version 4.49.0 so you can use version 3.0.0 of "
        "tensorflow-ic-imagenet-inception-v3-classification-4." in str(e.value)
    )

    with pytest.raises(KeyError) as e:
        cache.get_header(
            model_id="pytorch-ic-imagenet-inception-v3-classification-4", semantic_version_str="3.*"
        )
    assert "Consider upgrading" not in str(e.value)

    with pytest.raises(ValueError):
        cache.get_header(
            model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
            semantic_version_str="BAD",
        )

    with pytest.raises(KeyError):
        cache.get_header(
            model_id="tensorflow-ic-imagenet-inception-v3-classification-4-bak",
        )


@patch("boto3.client")
def test_jumpstart_cache_handles_boto3_issues(mock_boto3_client):

    mock_boto3_client.return_value.get_object.side_effect = Exception()

    cache = JumpStartModelsCache(bucket="some_bucket")

    with pytest.raises(Exception):
        cache.get_header(
            model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
        )

    mock_boto3_client.return_value.reset_mock()

    mock_boto3_client.return_value.head_object.side_effect = Exception()

    cache = JumpStartModelsCache(bucket="some_bucket")

    with pytest.raises(Exception):
        cache.get_header(
            model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
        )


def test_jumpstart_cache_gets_cleared_when_params_are_set():
    cache = JumpStartModelsCache(bucket="some_bucket")
    cache.clear = MagicMock()
    cache.set_bucket("some_bucket")
    cache.clear.assert_called_once()
    cache.clear.reset_mock()
    cache.set_region("some_region")
    cache.clear.assert_called_once()
    cache.clear.reset_mock()
    cache.set_manifest_file_s3_key("some_key")
    cache.clear.assert_called_once()


def test_jumpstart_cache_accepts_input_parameters():

    region = "us-east-1"
    max_s3_cache_items = 1
    s3_cache_expiration_time = datetime.timedelta(weeks=2)
    max_semantic_version_cache_items = 3
    semantic_version_cache_expiration_time = datetime.timedelta(microseconds=4)
    bucket = "my-amazing-bucket"
    manifest_file_key = "some_s3_key"

    cache = JumpStartModelsCache(
        region=region,
        max_s3_cache_items=max_s3_cache_items,
        s3_cache_expiration_time=s3_cache_expiration_time,
        max_semantic_version_cache_items=max_semantic_version_cache_items,
        semantic_version_cache_expiration_time=semantic_version_cache_expiration_time,
        bucket=bucket,
        manifest_file_s3_key=manifest_file_key,
    )

    assert cache.get_manifest_file_s3_key() == manifest_file_key
    assert cache.get_region() == region
    assert cache.get_bucket() == bucket
    assert cache._s3_cache._max_cache_items == max_s3_cache_items
    assert cache._s3_cache._expiration_time == s3_cache_expiration_time
    assert (
        cache._model_id_semantic_version_manifest_key_cache._max_cache_items
        == max_semantic_version_cache_items
    )
    assert (
        cache._model_id_semantic_version_manifest_key_cache._expiration_time
        == semantic_version_cache_expiration_time
    )


@patch("boto3.client")
def test_jumpstart_cache_evaluates_md5_hash(mock_boto3_client):

    mock_json = json.dumps(
        [
            {
                "model_id": "pytorch-ic-imagenet-inception-v3-classification-4",
                "version": "2.0.0",
                "min_version": "2.49.0",
                "spec_key": "community_models_specs/pytorch-ic-"
                "imagenet-inception-v3-classification-4/specs_v2.0.0.json",
            }
        ]
    )

    bucket_name = "bucket_name"
    now = datetime.datetime.now()

    with patch("datetime.datetime") as mock_datetime:
        mock_datetime.now.return_value = now

        cache = JumpStartModelsCache(
            bucket=bucket_name, s3_cache_expiration_time=datetime.timedelta(hours=1)
        )

        mock_boto3_client.return_value.get_object.return_value = {
            "Body": botocore.response.StreamingBody(
                io.BytesIO(bytes(mock_json, "utf-8")), content_length=len(mock_json)
            )
        }
        mock_boto3_client.return_value.head_object.return_value = {"ETag": "hash1"}

        cache.get_header(model_id="pytorch-ic-imagenet-inception-v3-classification-4")

        # first time accessing cache should involve get_object and head_object
        mock_boto3_client.return_value.get_object.assert_called_with(
            Bucket=bucket_name, Key=DEFAULT_MANIFEST_FILE_S3_KEY
        )
        mock_boto3_client.return_value.head_object.assert_called_with(
            Bucket=bucket_name, Key=DEFAULT_MANIFEST_FILE_S3_KEY
        )

        mock_boto3_client.return_value.get_object.reset_mock()
        mock_boto3_client.return_value.head_object.reset_mock()

        # second time accessing cache should just involve head_object if hash hasn't changed
        mock_boto3_client.return_value.get_object.return_value = {
            "Body": botocore.response.StreamingBody(
                io.BytesIO(bytes(mock_json, "utf-8")), content_length=len(mock_json)
            )
        }
        mock_boto3_client.return_value.head_object.return_value = {"ETag": "hash1"}

        # invalidate cache
        mock_datetime.now.return_value += datetime.timedelta(hours=2)

        cache.get_header(model_id="pytorch-ic-imagenet-inception-v3-classification-4")

        mock_boto3_client.return_value.head_object.assert_called_with(
            Bucket=bucket_name, Key=DEFAULT_MANIFEST_FILE_S3_KEY
        )
        mock_boto3_client.return_value.get_object.assert_not_called()

        mock_boto3_client.return_value.get_object.reset_mock()
        mock_boto3_client.return_value.head_object.reset_mock()

        # third time accessing cache should involve head_object and get_object if hash has changed
        mock_boto3_client.return_value.head_object.return_value = {"ETag": "hash2"}
        mock_boto3_client.return_value.get_object.return_value = {
            "Body": botocore.response.StreamingBody(
                io.BytesIO(bytes(mock_json, "utf-8")), content_length=len(mock_json)
            )
        }

        # invalidate cache
        mock_datetime.now.return_value += datetime.timedelta(hours=2)

        cache.get_header(model_id="pytorch-ic-imagenet-inception-v3-classification-4")

        mock_boto3_client.return_value.get_object.assert_called_with(
            Bucket=bucket_name, Key=DEFAULT_MANIFEST_FILE_S3_KEY
        )
        mock_boto3_client.return_value.head_object.assert_called_with(
            Bucket=bucket_name, Key=DEFAULT_MANIFEST_FILE_S3_KEY
        )


@patch("boto3.client")
def test_jumpstart_cache_makes_correct_s3_calls(mock_boto3_client):

    # test get_header
    mock_json = json.dumps(
        [
            {
                "model_id": "pytorch-ic-imagenet-inception-v3-classification-4",
                "version": "2.0.0",
                "min_version": "2.49.0",
                "spec_key": "community_models_specs/pytorch-ic-"
                "imagenet-inception-v3-classification-4/specs_v2.0.0.json",
            }
        ]
    )
    mock_boto3_client.return_value.get_object.return_value = {
        "Body": botocore.response.StreamingBody(
            io.BytesIO(bytes(mock_json, "utf-8")), content_length=len(mock_json)
        )
    }

    mock_boto3_client.return_value.head_object.return_value = {"ETag": "some-hash"}

    bucket_name = "bucket_name"
    cache = JumpStartModelsCache(bucket=bucket_name)
    cache.get_header(model_id="pytorch-ic-imagenet-inception-v3-classification-4")

    mock_boto3_client.return_value.get_object.assert_called_with(
        Bucket=bucket_name, Key=DEFAULT_MANIFEST_FILE_S3_KEY
    )
    mock_boto3_client.return_value.head_object.assert_called_with(
        Bucket=bucket_name, Key=DEFAULT_MANIFEST_FILE_S3_KEY
    )

    # test get_specs. manifest already in cache, so only s3 call will be to get specs.
    mock_json = json.dumps(BASE_SPEC)

    mock_boto3_client.return_value.reset_mock()

    mock_boto3_client.return_value.get_object.return_value = {
        "Body": botocore.response.StreamingBody(
            io.BytesIO(bytes(mock_json, "utf-8")), content_length=len(mock_json)
        )
    }
    cache.get_specs(model_id="pytorch-ic-imagenet-inception-v3-classification-4")

    mock_boto3_client.return_value.get_object.assert_called_with(
        Bucket=bucket_name,
        Key="community_models_specs/pytorch-ic-imagenet-"
        "inception-v3-classification-4/specs_v2.0.0.json",
    )
    mock_boto3_client.return_value.head_object.assert_not_called()


@patch.object(JumpStartModelsCache, "_get_file_from_s3", patched_get_file_from_s3)
def test_jumpstart_cache_handles_bad_semantic_version_manifest_key_cache():
    cache = JumpStartModelsCache(bucket="some_bucket")

    cache.clear = MagicMock()
    cache._model_id_semantic_version_manifest_key_cache = MagicMock()
    cache._model_id_semantic_version_manifest_key_cache.get.side_effect = [
        JumpStartVersionedModelId(
            "tensorflow-ic-imagenet-inception-v3-classification-4", "999.0.0"
        ),
        JumpStartVersionedModelId("tensorflow-ic-imagenet-inception-v3-classification-4", "1.0.0"),
    ]

    assert JumpStartModelHeader(
        {
            "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
            "version": "1.0.0",
            "min_version": "2.49.0",
            "spec_key": "community_models_specs/tensorflow-ic-"
            "imagenet-inception-v3-classification-4/specs_v1.0.0.json",
        }
    ) == cache.get_header(
        model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
    )
    cache.clear.assert_called_once()
    cache.clear.reset_mock()

    cache._model_id_semantic_version_manifest_key_cache.get.side_effect = [
        JumpStartVersionedModelId(
            "tensorflow-ic-imagenet-inception-v3-classification-4", "999.0.0"
        ),
        JumpStartVersionedModelId(
            "tensorflow-ic-imagenet-inception-v3-classification-4", "987.0.0"
        ),
    ]
    with pytest.raises(KeyError):
        cache.get_header(
            model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
        )
    cache.clear.assert_called_once()


@patch.object(JumpStartModelsCache, "_get_file_from_s3", patched_get_file_from_s3)
@patch("sagemaker.jumpstart.utils.get_sagemaker_version", lambda: "2.68.3")
def test_jumpstart_cache_get_specs():
    cache = JumpStartModelsCache(bucket="some_bucket")

    model_id, version = "tensorflow-ic-imagenet-inception-v3-classification-4", "2.0.0"
    assert get_spec_from_base_spec(model_id, version) == cache.get_specs(
        model_id=model_id, semantic_version_str=version
    )

    model_id, version = "tensorflow-ic-imagenet-inception-v3-classification-4", "1.0.0"
    assert get_spec_from_base_spec(model_id, version) == cache.get_specs(
        model_id=model_id, semantic_version_str=version
    )

    model_id = "pytorch-ic-imagenet-inception-v3-classification-4"
    assert get_spec_from_base_spec(model_id, "1.0.0") == cache.get_specs(
        model_id=model_id, semantic_version_str="1.*"
    )

    with pytest.raises(KeyError):
        cache.get_specs(
            model_id=model_id + "bak",
        )

    with pytest.raises(KeyError):
        cache.get_specs(model_id=model_id, semantic_version_str="9.*")

    with pytest.raises(ValueError):
        cache.get_specs(model_id=model_id, semantic_version_str="BAD")
