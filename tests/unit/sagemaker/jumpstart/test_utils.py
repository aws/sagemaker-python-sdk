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
from mock.mock import Mock, patch
import pytest
import random
from sagemaker.jumpstart import utils
from sagemaker.jumpstart.constants import (
    JUMPSTART_BUCKET_NAME_SET,
    JUMPSTART_REGION_NAME_SET,
    JumpStartTag,
)
from sagemaker.jumpstart.types import JumpStartModelHeader, JumpStartVersionedModelId


def random_jumpstart_s3_uri(key):
    return f"s3://{random.choice(list(JUMPSTART_BUCKET_NAME_SET))}/{key}"


def test_get_jumpstart_content_bucket():
    bad_region = "bad_region"
    assert bad_region not in JUMPSTART_REGION_NAME_SET
    with pytest.raises(ValueError):
        utils.get_jumpstart_content_bucket(bad_region)


def test_get_jumpstart_launched_regions_message():

    with patch("sagemaker.jumpstart.constants.JUMPSTART_REGION_NAME_SET", {}):
        assert (
            utils.get_jumpstart_launched_regions_message()
            == "JumpStart is not available in any region."
        )

    with patch("sagemaker.jumpstart.constants.JUMPSTART_REGION_NAME_SET", {"some_region"}):
        assert (
            utils.get_jumpstart_launched_regions_message()
            == "JumpStart is available in some_region region."
        )

    with patch(
        "sagemaker.jumpstart.constants.JUMPSTART_REGION_NAME_SET", {"some_region1", "some_region2"}
    ):
        assert (
            utils.get_jumpstart_launched_regions_message()
            == "JumpStart is available in some_region1 and some_region2 regions."
        )

    with patch("sagemaker.jumpstart.constants.JUMPSTART_REGION_NAME_SET", {"a", "b", "c"}):
        assert (
            utils.get_jumpstart_launched_regions_message()
            == "JumpStart is available in a, b, and c regions."
        )


def test_get_formatted_manifest():
    mock_manifest = [
        {
            "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
            "version": "1.0.0",
            "min_version": "2.49.0",
            "spec_key": "community_models_specs/tensorflow-ic-imagenet-inception-v3-classification-4/specs_v1.0.0.json",
        },
    ]

    assert utils.get_formatted_manifest(mock_manifest) == {
        JumpStartVersionedModelId(
            "tensorflow-ic-imagenet-inception-v3-classification-4", "1.0.0"
        ): JumpStartModelHeader(mock_manifest[0])
    }

    assert utils.get_formatted_manifest([]) == {}


def test_parse_sagemaker_version():

    with patch("sagemaker.__version__", "1.2.3"):
        assert utils.parse_sagemaker_version() == "1.2.3"

    with patch("sagemaker.__version__", "1.2.3.3332j"):
        assert utils.parse_sagemaker_version() == "1.2.3"

    with patch("sagemaker.__version__", "1.2.3."):
        assert utils.parse_sagemaker_version() == "1.2.3"

    with pytest.raises(ValueError):
        with patch("sagemaker.__version__", "1.2.3dfsdfs"):
            utils.parse_sagemaker_version()

    with pytest.raises(RuntimeError):
        with patch("sagemaker.__version__", "1.2"):
            utils.parse_sagemaker_version()

    with pytest.raises(RuntimeError):
        with patch("sagemaker.__version__", "1"):
            utils.parse_sagemaker_version()

    with pytest.raises(RuntimeError):
        with patch("sagemaker.__version__", ""):
            utils.parse_sagemaker_version()

    with pytest.raises(RuntimeError):
        with patch("sagemaker.__version__", "1.2.3.4.5"):
            utils.parse_sagemaker_version()


@patch("sagemaker.jumpstart.utils.parse_sagemaker_version")
@patch("sagemaker.jumpstart.accessors.SageMakerSettings._parsed_sagemaker_version", "")
def test_get_sagemaker_version(patched_parse_sm_version: Mock):
    utils.get_sagemaker_version()
    utils.get_sagemaker_version()
    utils.get_sagemaker_version()
    assert patched_parse_sm_version.called_only_once()


def test_is_jumpstart_model_uri():

    assert not utils.is_jumpstart_model_uri("fdsfdsf")
    assert not utils.is_jumpstart_model_uri("s3://not-jumpstart-bucket/sdfsdfds")
    assert not utils.is_jumpstart_model_uri("some/actual/localfile")

    assert utils.is_jumpstart_model_uri(
        random_jumpstart_s3_uri("source_directory_tarballs/sourcedir.tar.gz")
    )
    assert utils.is_jumpstart_model_uri(random_jumpstart_s3_uri("random_key"))


def test_add_jumpstart_tags():
    tags = None
    inference_model_uri = "dfsdfsd"
    inference_script_uri = "dfsdfs"
    assert (
        utils.add_jumpstart_tags(
            tags=tags,
            inference_model_uri=inference_model_uri,
            inference_script_uri=inference_script_uri,
        )
        is None
    )

    tags = []
    inference_model_uri = "dfsdfsd"
    inference_script_uri = "dfsdfs"
    assert (
        utils.add_jumpstart_tags(
            tags=tags,
            inference_model_uri=inference_model_uri,
            inference_script_uri=inference_script_uri,
        )
        == []
    )

    tags = [{"some": "tag"}]
    inference_model_uri = "dfsdfsd"
    inference_script_uri = "dfsdfs"
    assert (
        utils.add_jumpstart_tags(
            tags=tags,
            inference_model_uri=inference_model_uri,
            inference_script_uri=inference_script_uri,
        )
        == [{"some": "tag"}]
    )

    tags = None
    inference_model_uri = random_jumpstart_s3_uri("random_key")
    inference_script_uri = "dfsdfs"
    assert (
        utils.add_jumpstart_tags(
            tags=tags,
            inference_model_uri=inference_model_uri,
            inference_script_uri=inference_script_uri,
        )
        == [{JumpStartTag.INFERENCE_MODEL_URI.value: inference_model_uri}]
    )

    tags = []
    inference_model_uri = random_jumpstart_s3_uri("random_key")
    inference_script_uri = "dfsdfs"
    assert (
        utils.add_jumpstart_tags(
            tags=tags,
            inference_model_uri=inference_model_uri,
            inference_script_uri=inference_script_uri,
        )
        == [{JumpStartTag.INFERENCE_MODEL_URI.value: inference_model_uri}]
    )

    tags = [{"some": "tag"}]
    inference_model_uri = random_jumpstart_s3_uri("random_key")
    inference_script_uri = "dfsdfs"
    assert utils.add_jumpstart_tags(
        tags=tags,
        inference_model_uri=inference_model_uri,
        inference_script_uri=inference_script_uri,
    ) == [
        {"some": "tag"},
        {JumpStartTag.INFERENCE_MODEL_URI.value: inference_model_uri},
    ]

    tags = None
    inference_script_uri = random_jumpstart_s3_uri("random_key")
    inference_model_uri = "dfsdfs"
    assert (
        utils.add_jumpstart_tags(
            tags=tags,
            inference_model_uri=inference_model_uri,
            inference_script_uri=inference_script_uri,
        )
        == [{JumpStartTag.INFERENCE_SCRIPT_URI.value: inference_script_uri}]
    )

    tags = []
    inference_script_uri = random_jumpstart_s3_uri("random_key")
    inference_model_uri = "dfsdfs"
    assert (
        utils.add_jumpstart_tags(
            tags=tags,
            inference_model_uri=inference_model_uri,
            inference_script_uri=inference_script_uri,
        )
        == [{JumpStartTag.INFERENCE_SCRIPT_URI.value: inference_script_uri}]
    )

    tags = [{"some": "tag"}]
    inference_script_uri = random_jumpstart_s3_uri("random_key")
    inference_model_uri = "dfsdfs"
    assert utils.add_jumpstart_tags(
        tags=tags,
        inference_model_uri=inference_model_uri,
        inference_script_uri=inference_script_uri,
    ) == [
        {"some": "tag"},
        {JumpStartTag.INFERENCE_SCRIPT_URI.value: inference_script_uri},
    ]

    tags = None
    inference_script_uri = random_jumpstart_s3_uri("random_key")
    inference_model_uri = random_jumpstart_s3_uri("random_key")
    assert utils.add_jumpstart_tags(
        tags=tags,
        inference_model_uri=inference_model_uri,
        inference_script_uri=inference_script_uri,
    ) == [
        {
            JumpStartTag.INFERENCE_MODEL_URI.value: inference_model_uri,
        },
        {JumpStartTag.INFERENCE_SCRIPT_URI.value: inference_script_uri},
    ]

    tags = []
    inference_script_uri = random_jumpstart_s3_uri("random_key")
    inference_model_uri = random_jumpstart_s3_uri("random_key")
    assert utils.add_jumpstart_tags(
        tags=tags,
        inference_model_uri=inference_model_uri,
        inference_script_uri=inference_script_uri,
    ) == [
        {
            JumpStartTag.INFERENCE_MODEL_URI.value: inference_model_uri,
        },
        {JumpStartTag.INFERENCE_SCRIPT_URI.value: inference_script_uri},
    ]

    tags = [{"some": "tag"}]
    inference_script_uri = random_jumpstart_s3_uri("random_key")
    inference_model_uri = random_jumpstart_s3_uri("random_key")
    assert utils.add_jumpstart_tags(
        tags=tags,
        inference_model_uri=inference_model_uri,
        inference_script_uri=inference_script_uri,
    ) == [
        {"some": "tag"},
        {
            JumpStartTag.INFERENCE_MODEL_URI.value: inference_model_uri,
        },
        {JumpStartTag.INFERENCE_SCRIPT_URI.value: inference_script_uri},
    ]
