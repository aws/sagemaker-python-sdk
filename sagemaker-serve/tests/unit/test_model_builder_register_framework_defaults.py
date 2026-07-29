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
"""Regression tests: ModelBuilder.register() must not require the user to set
``framework`` / ``framework_version`` manually.

GitHub issue #5391: register() raised
``AttributeError: 'ModelBuilder' object has no attribute 'framework'`` (and the
same for ``framework_version``) unless the user explicitly assigned them before
calling register(). These attributes are internal and should be initialized to
sensible defaults (None) at construction so register() works out of the box.

These tests lock in that a freshly constructed ModelBuilder already exposes both
attributes defaulted to None, and that register() reaches
``create_model_package_from_containers`` without an AttributeError when the user
never touches them.
"""
from __future__ import absolute_import

from unittest.mock import MagicMock, patch

import pytest


_ROLE = "arn:aws:iam::123456789012:role/SageMakerRole"


@pytest.fixture
def model_builder_cls():
    """Import ModelBuilder lazily so import issues stay scoped to this fixture."""
    from sagemaker.serve import ModelBuilder

    return ModelBuilder


def _make_builder(model_builder_cls):
    """Construct a ModelBuilder without ever setting framework attributes.

    The role resolver is patched so construction does no IAM work offline.
    """
    session = MagicMock(boto_region_name="us-west-2")
    # register() is telemetry-wrapped and reads sagemaker_config; a bare
    # MagicMock value fails config-schema validation, so pin it to an empty dict.
    session.sagemaker_config = {}
    with patch(
        "sagemaker.serve.model_builder.resolve_and_validate_role",
        return_value=_ROLE,
    ):
        return model_builder_cls(
            model="my_module.MyModelClass",
            sagemaker_session=session,
        )


def test_framework_attributes_default_to_none_on_construction(model_builder_cls):
    """A freshly built ModelBuilder exposes framework/framework_version = None.

    This is the root-cause guard for #5391: the attributes must exist without
    the user assigning them.
    """
    mb = _make_builder(model_builder_cls)

    assert hasattr(mb, "framework")
    assert hasattr(mb, "framework_version")
    assert mb.framework is None
    assert mb.framework_version is None


def test_register_does_not_require_user_set_framework_attributes(model_builder_cls):
    """register() must not raise AttributeError for framework/framework_version.

    We run the non-pipeline register() path (MagicMock session is not a
    PipelineSession) and only assert that the framework attributes flow through
    to update_container_with_inference_params without the user setting them.
    """
    mb = _make_builder(model_builder_cls)
    # Minimal state register() needs on the non-pipeline, versioned path.
    mb.image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/img:latest"
    mb.s3_model_data_url = "s3://bucket/model/"
    mb.content_types = ["application/json"]
    mb.response_types = ["application/json"]

    arn = "arn:aws:sagemaker:us-west-2:123456789012:model-package/g/1"

    captured = {}

    def _capture_params(**kwargs):
        # Record the framework values register() passed through.
        captured["framework"] = kwargs.get("framework")
        captured["framework_version"] = kwargs.get("framework_version")
        return kwargs.get("container_def")

    with patch(
        "sagemaker.serve.model_builder.create_model_package_from_containers",
        return_value={"ModelPackageArn": arn},
    ), patch("sagemaker.serve.model_builder.get_model_package_args", return_value={}), patch(
        "sagemaker.serve.model_builder.update_container_with_inference_params",
        side_effect=_capture_params,
    ), patch.object(
        mb, "_prepare_container_def", return_value={"Image": mb.image_uri}
    ):
        result = mb.register(model_package_group_name="g")

    assert result == arn
    # framework attributes were read (defaulted to None), not missing.
    assert captured["framework"] is None
    assert captured["framework_version"] is None
