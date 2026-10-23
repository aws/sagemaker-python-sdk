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
"""``ModelPackageGroup.get_or_create`` returns the existing group or creates it.

Before the change ``ModelPackageGroup`` had only the generated ``get`` and ``create``.
``DescribeModelPackageGroup`` reports a missing group as a ``ValidationException`` whose
message ends in ``does not exist``, so ten fine-tuning templates in the SageMaker agent
skills wrapped ``create`` in a ``try`` block and parsed ``e.response["Error"]["Code"]``
against ``("ResourceInUse", "ValidationException")`` to fall back to ``get``.
"""

from __future__ import absolute_import

from unittest.mock import MagicMock, patch

import boto3
import pytest
from botocore.exceptions import ClientError

from sagemaker.core.model_package_group import (
    ModelPackageGroupMixin,
    ModelPackageGroupSpec,
    is_not_found_error,
)
from sagemaker.core.resources import Base, ModelPackageGroup
from sagemaker.core.shapes import Tag

GROUP_NAME = "my-model-group"


def _client_error(code, message):
    return ClientError({"Error": {"Code": code, "Message": message}}, "DescribeModelPackageGroup")


def _not_found_error():
    return _client_error(
        "ValidationException",
        f"Model Package Group {GROUP_NAME} does not exist.",
    )


def _group(**overrides):
    kwargs = {
        "model_package_group_name": GROUP_NAME,
        "model_package_group_arn": f"arn:aws:sagemaker:us-west-2:123456789012:model-package-group/{GROUP_NAME}",
    }
    kwargs.update(overrides)
    return ModelPackageGroup(**kwargs)


def test_model_package_group_is_a_model_package_group_mixin():
    assert issubclass(ModelPackageGroup, ModelPackageGroupMixin)
    # The mixin adds behavior only: no model field, nothing in model_dump().
    assert "spec" not in ModelPackageGroup.model_fields
    assert "get_or_create" not in _group().model_dump()


def test_get_or_create_returns_the_existing_group_without_create():
    existing = _group()
    with (
        patch.object(ModelPackageGroup, "get", return_value=existing) as get,
        patch.object(ModelPackageGroup, "create") as create,
    ):
        result = ModelPackageGroup.get_or_create(
            GROUP_NAME, spec=ModelPackageGroupSpec(model_package_group_description="ignored")
        )

    assert result is existing
    get.assert_called_once_with(model_package_group_name=GROUP_NAME, session=None, region=None)
    create.assert_not_called()


def test_get_or_create_creates_the_group_when_describe_says_it_does_not_exist():
    created = _group()
    with (
        patch.object(ModelPackageGroup, "get", side_effect=_not_found_error()),
        patch.object(ModelPackageGroup, "create", return_value=created) as create,
    ):
        result = ModelPackageGroup.get_or_create(GROUP_NAME)

    assert result is created
    create.assert_called_once_with(model_package_group_name=GROUP_NAME, session=None, region=None)


def test_get_or_create_passes_the_spec_session_and_region_to_create():
    session = boto3.Session(
        aws_access_key_id="AKIDTEST", aws_secret_access_key="secret", region_name="us-west-2"
    )
    tags = [Tag(key="team", value="ml")]
    spec = ModelPackageGroupSpec(model_package_group_description="Fine-tuned models", tags=tags)
    with (
        patch.object(ModelPackageGroup, "get", side_effect=_not_found_error()) as get,
        patch.object(ModelPackageGroup, "create", return_value=_group()) as create,
    ):
        ModelPackageGroup.get_or_create(GROUP_NAME, spec=spec, session=session, region="eu-west-1")

    get.assert_called_once_with(
        model_package_group_name=GROUP_NAME, session=session, region="eu-west-1"
    )
    create.assert_called_once_with(
        model_package_group_name=GROUP_NAME,
        session=session,
        region="eu-west-1",
        model_package_group_description="Fine-tuned models",
        tags=tags,
    )


def test_get_or_create_creates_on_a_resource_not_found_code():
    error = _client_error("ResourceNotFoundException", "No such group")
    with (
        patch.object(ModelPackageGroup, "get", side_effect=error),
        patch.object(ModelPackageGroup, "create", return_value=_group()) as create,
    ):
        ModelPackageGroup.get_or_create(GROUP_NAME)

    create.assert_called_once()


def test_get_or_create_propagates_every_other_error():
    denied = _client_error("AccessDeniedException", "User is not authorized")
    with (
        patch.object(ModelPackageGroup, "get", side_effect=denied),
        patch.object(ModelPackageGroup, "create") as create,
    ):
        with pytest.raises(ClientError) as raised:
            ModelPackageGroup.get_or_create(GROUP_NAME)

    assert raised.value is denied
    create.assert_not_called()


def test_a_validation_exception_with_another_message_propagates():
    invalid = _client_error("ValidationException", "1 validation error detected: name too long")
    with (
        patch.object(ModelPackageGroup, "get", side_effect=invalid),
        patch.object(ModelPackageGroup, "create") as create,
    ):
        with pytest.raises(ClientError):
            ModelPackageGroup.get_or_create(GROUP_NAME)

    create.assert_not_called()


def test_get_or_create_end_to_end_through_the_generated_client_calls():
    client = MagicMock()
    client.describe_model_package_group.side_effect = [
        _not_found_error(),
        {
            "ModelPackageGroupName": GROUP_NAME,
            "ModelPackageGroupArn": f"arn:aws:sagemaker:us-west-2:123456789012:model-package-group/{GROUP_NAME}",
            "ModelPackageGroupStatus": "Completed",
        },
    ]
    # The generated get() and create() call Base.get_sagemaker_client, not cls.
    with patch.object(Base, "get_sagemaker_client", return_value=client):
        result = ModelPackageGroup.get_or_create(
            GROUP_NAME, spec=ModelPackageGroupSpec(model_package_group_description="desc")
        )

    client.create_model_package_group.assert_called_once_with(
        ModelPackageGroupName=GROUP_NAME, ModelPackageGroupDescription="desc"
    )
    assert client.describe_model_package_group.call_count == 2
    assert result.model_package_group_status == "Completed"


def test_spec_create_kwargs_drops_unset_fields():
    assert ModelPackageGroupSpec().create_kwargs() == {}
    assert ModelPackageGroupSpec(model_package_group_description="d").create_kwargs() == {
        "model_package_group_description": "d"
    }


@pytest.mark.parametrize(
    "code, message, expected",
    [
        ("ValidationException", "Model Package Group x does not exist.", True),
        ("ValidationException", "Could not find model package group x", True),
        ("ResourceNotFound", "Resource not found", True),
        ("ResourceNotFoundException", "Resource not found", True),
        ("ValidationException", "1 validation error detected", False),
        ("AccessDeniedException", "does not exist", False),
        ("ThrottlingException", "Rate exceeded", False),
    ],
)
def test_is_not_found_error(code, message, expected):
    assert is_not_found_error(_client_error(code, message)) is expected
