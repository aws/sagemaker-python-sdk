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
"""SageMaker resource classes.

This module provides utilities that add the sagemaker_session property
required by the SageMaker MLOps pipeline components. The property is
applied to the V3 Model resource class so that ModelStep can access
sagemaker_session during pipeline composition.
"""
from __future__ import absolute_import

from typing import Any


def _sagemaker_session_getter(self) -> Any:
    """Get the SageMaker session associated with this model.

    Returns:
        Any: The SageMaker session (Session, PipelineSession,
            LocalSession, etc.), or None if not set.
    """
    return getattr(self, "_sagemaker_session", None)


def _sagemaker_session_setter(self, value: Any) -> None:
    """Set the SageMaker session for this model.

    Accepts any session-like object including Session, PipelineSession,
    LocalSession, or compatible mocks/stubs. This permissive approach
    avoids fragile type-checking that would break with mocks or
    alternative session implementations.

    Args:
        value: A SageMaker session instance (Session, PipelineSession,
            LocalSession, etc.), or None to clear the session.
    """
    self._sagemaker_session = value


# Property descriptor that can be applied to any class
sagemaker_session_property = property(
    fget=_sagemaker_session_getter,
    fset=_sagemaker_session_setter,
    doc="The SageMaker session associated with this model resource.",
)


def _has_sagemaker_session_property(cls: type) -> bool:
    """Check if a class already defines sagemaker_session as a property descriptor.

    Inspects the class MRO's __dict__ entries directly to correctly detect
    property descriptors. This avoids the pitfalls of using getattr() which
    can invoke descriptors or return inherited attribute values.

    Args:
        cls: The class to inspect.

    Returns:
        bool: True if sagemaker_session is already defined as a property
            in the class or any of its bases.
    """
    for klass in cls.__mro__:
        if "sagemaker_session" in klass.__dict__:
            if isinstance(klass.__dict__["sagemaker_session"], property):
                return True
    return False


def apply_sagemaker_session_property(cls: type) -> type:
    """Apply the sagemaker_session property to a class.

    This function adds the sagemaker_session property onto the given class.
    It inspects cls.__dict__ (and the MRO) to correctly detect whether a
    property descriptor already exists, avoiding issues with getattr-based
    detection that could incorrectly identify regular attributes or invoke
    descriptors.

    Note on import ordering: This function patches the class object in place.
    Any code that holds a reference to the class object (regardless of which
    module it was imported from) will see the patched property, since Python
    classes are mutable objects. However, if user code imports Model before
    this function is called, the Model class will not yet have the property.
    To ensure correct behavior, import Model from `sagemaker.core.resources`
    or `sagemaker.core.model_resource` which guarantee the patch is applied
    at import time.

    Args:
        cls: The class to patch.

    Returns:
        type: The patched class (same object, modified in place).
    """
    if not _has_sagemaker_session_property(cls):
        cls.sagemaker_session = sagemaker_session_property
    return cls
