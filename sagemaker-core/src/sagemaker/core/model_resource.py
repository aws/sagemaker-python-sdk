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
"""SageMaker Model resource with session support.

This module adds the sagemaker_session property to the V3 Model resource
class so that ModelStep can access sagemaker_session during pipeline
composition (particularly for repack steps).

The integration is done by monkey-patching the sagemaker_session property
onto the existing V3 Model class. This avoids creating a separate subclass
which could cause isinstance check failures depending on import path.

Import Ordering
---------------
Since this module patches the Model class object in place, the patch is
visible to all code that holds a reference to the same class object,
regardless of which module path was used to import it. However, to
guarantee the patch is applied before use:

- Import Model from ``sagemaker.core.resources`` (recommended) or
  ``sagemaker.core.model_resource``. Both modules ensure the patch is
  applied at import time.
- If importing directly from ``sagemaker.core.generated.resources``,
  ensure that ``sagemaker.core.model_resource`` has been imported first
  (e.g., by importing ``sagemaker.core.resources``).
"""
from __future__ import absolute_import

from sagemaker.core.session_mixin import apply_sagemaker_session_property

__all__ = ["Model"]

try:
    # Import the existing V3 Model resource class and patch it in place.
    # This ensures that regardless of import path (sagemaker.core.resources,
    # sagemaker.core.generated.resources, or sagemaker.core.model_resource),
    # the same Model class has the sagemaker_session property.
    from sagemaker.core.generated.resources import Model

    apply_sagemaker_session_property(Model)

except (ImportError, ModuleNotFoundError) as e:
    raise ImportError(
        "Failed to import Model from sagemaker.core.generated.resources. "
        "The V3 Model resource class is required for sagemaker_session "
        "property support (issue #5829). Please ensure the sagemaker-core "
        "package is properly installed. Original error: {}".format(e)
    ) from e
