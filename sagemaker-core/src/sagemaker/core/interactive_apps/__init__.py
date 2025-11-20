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
"""Classes for starting/accessing apps hosted on Amazon SageMaker Studio."""

from __future__ import absolute_import

from enum import Enum

from sagemaker.core.interactive_apps.base_interactive_app import (  # noqa: F401
    BaseInteractiveApp,
)
from sagemaker.core.interactive_apps.detail_profiler_app import (  # noqa: F401
    DetailProfilerApp,
)
from sagemaker.core.interactive_apps.tensorboard import (  # noqa: F401
    TensorBoardApp,
)


class SupportedInteractiveAppTypes(Enum):
    """SupportedInteractiveAppTypes indicates which apps are supported."""

    TENSORBOARD = 1


__all__ = [
    "BaseInteractiveApp",
    "DetailProfilerApp",
    "SupportedInteractiveAppTypes",
    "TensorBoardApp",
]
