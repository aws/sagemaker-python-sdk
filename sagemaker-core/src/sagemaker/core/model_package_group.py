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
"""Hand-written behavior of the generated ``sagemaker.core.resources.ModelPackageGroup`` class.

``DescribeModelPackageGroup`` reports a missing group as a ``ValidationException`` whose
message ends in ``does not exist``, not as a ``ResourceNotFound`` error. So every caller
that wants "use the group, or create it" parsed ``ClientError`` codes and messages by hand.
``ModelPackageGroupMixin.get_or_create`` does that once. It returns the existing group, or
creates the group from a ``ModelPackageGroupSpec`` when the describe call says the group is
absent. Every other error propagates.
"""

from __future__ import absolute_import

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from boto3.session import Session
from botocore.exceptions import ClientError

from sagemaker.core.shapes import ManagedConfiguration, Tag

if TYPE_CHECKING:
    from sagemaker.core.resources import ModelPackageGroup

logger = logging.getLogger(__name__)

NOT_FOUND_ERROR_CODE_PREFIX = "ResourceNotFound"
VALIDATION_ERROR_CODE = "ValidationException"
NOT_FOUND_MESSAGE_MARKERS = ("does not exist", "not found", "could not find")


@dataclass(frozen=True)
class ModelPackageGroupSpec:
    """The create-only fields of a model package group.

    ``get_or_create`` passes the set fields to ``ModelPackageGroup.create`` when the group is
    absent, and ignores the spec when the group exists.
    """

    model_package_group_description: Optional[str] = None
    tags: Optional[List[Tag]] = None
    managed_configuration: Optional[ManagedConfiguration] = None

    def create_kwargs(self) -> Dict[str, Any]:
        """Return the set fields as keyword arguments of ``ModelPackageGroup.create``."""
        fields = {
            "model_package_group_description": self.model_package_group_description,
            "tags": self.tags,
            "managed_configuration": self.managed_configuration,
        }
        return {name: value for name, value in fields.items() if value is not None}


def is_not_found_error(error: ClientError) -> bool:
    """Return ``True`` when a describe ``ClientError`` means the resource does not exist.

    ``DescribeModelPackageGroup`` returns ``ValidationException`` with a message that ends in
    ``does not exist`` for a missing group. A code that starts with ``ResourceNotFound`` means
    the same on other APIs.
    """
    error_details = error.response.get("Error", {})
    code = error_details.get("Code", "")
    if code.startswith(NOT_FOUND_ERROR_CODE_PREFIX):
        return True
    message = error_details.get("Message", "").lower()
    return code == VALIDATION_ERROR_CODE and any(
        marker in message for marker in NOT_FOUND_MESSAGE_MARKERS
    )


class ModelPackageGroupMixin:
    """``get_or_create`` for the generated ``sagemaker.core.resources.ModelPackageGroup``.

    Mixed into the generated class. The mixin adds no model field and no state. It calls the
    generated ``get`` and ``create`` classmethods, so the returned group keeps the session and
    region it was loaded with.
    """

    @classmethod
    def get_or_create(
        cls,
        model_package_group_name: str,
        spec: Optional[ModelPackageGroupSpec] = None,
        session: Optional[Session] = None,
        region: Optional[str] = None,
    ) -> "ModelPackageGroup":
        """Return the model package group, or create it when the describe call says it is absent.

        Args:
            model_package_group_name: The name of the model group.
            spec: The description, tags, and managed configuration of a group this call
                creates. Ignored when the group exists.
            session: Boto3 session.
            region: Region name.

        Returns:
            The existing or the created ModelPackageGroup resource.

        Raises:
            botocore.exceptions.ClientError: Every describe or create error other than a
                missing group.
        """
        try:
            return cls.get(
                model_package_group_name=model_package_group_name, session=session, region=region
            )
        except ClientError as error:
            if not is_not_found_error(error):
                raise
        logger.info("Model package group %s does not exist. Creating it.", model_package_group_name)
        create_kwargs = (spec or ModelPackageGroupSpec()).create_kwargs()
        return cls.create(
            model_package_group_name=model_package_group_name,
            session=session,
            region=region,
            **create_kwargs,
        )
