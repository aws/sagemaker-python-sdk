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
"""Unit tests for ModelBuilder.register()."""

from unittest.mock import MagicMock, patch

import pytest

from sagemaker.serve.model_builder import ModelBuilder

ROLE_ARN = "arn:aws:iam::123456789012:role/SageMakerRole"
IMAGE_URI = "123456789012.dkr.ecr.us-west-2.amazonaws.com/inference:latest"
MODEL_PACKAGE_ARN = "arn:aws:sagemaker:us-west-2:123456789012:model-package/group/1"


@pytest.fixture
def model_builder():
    session = MagicMock(boto_region_name="us-west-2")
    # register() is telemetry-wrapped and reads sagemaker_config.
    session.sagemaker_config = {}
    return ModelBuilder(
        image_uri=IMAGE_URI,
        s3_model_data_url="s3://bucket/model.tar.gz",
        role_arn=ROLE_ARN,
        sagemaker_session=session,
        instance_type="ml.m5.large",
    )


def _register(model_builder, **kwargs):
    """Call register() and return the arguments it passes to create the model package."""
    with patch(
        "sagemaker.serve.model_builder.create_model_package_from_containers",
        return_value={"ModelPackageArn": MODEL_PACKAGE_ARN},
    ) as create_model_package, patch.object(
        model_builder, "_prepare_container_def", return_value={"Image": IMAGE_URI}
    ):
        arn = model_builder.register(model_package_group_name="group", **kwargs)
    assert arn == MODEL_PACKAGE_ARN
    return create_model_package.call_args.kwargs


@pytest.mark.parametrize(
    "types",
    [
        {},
        {"content_types": ["text/csv"]},
        {"response_types": ["application/json"]},
    ],
    ids=["neither", "content_types_only", "response_types_only"],
)
def test_register_without_content_or_response_types(model_builder, types):
    request_args = _register(model_builder, **types)

    for name in ("content_types", "response_types"):
        if name in types:
            assert request_args[name] == types[name]
        else:
            assert name not in request_args


def test_register_reuses_content_and_response_types_from_earlier_call(model_builder):
    _register(model_builder, content_types=["text/csv"], response_types=["application/json"])

    request_args = _register(model_builder)

    assert request_args["content_types"] == ["text/csv"]
    assert request_args["response_types"] == ["application/json"]
