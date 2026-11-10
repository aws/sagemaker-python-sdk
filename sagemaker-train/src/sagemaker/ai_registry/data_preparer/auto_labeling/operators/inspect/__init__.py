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
"""What is in my data, and can this be labeled?

A first-row read of each input, task detection, and the taxonomy gate.
"""

from __future__ import absolute_import

from sagemaker.ai_registry.data_preparer.auto_labeling.operators.inspect.inputs import (  # noqa: F401
    inspect,
    inspect_labeling_data,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.inspect.task import (  # noqa: F401
    detect,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.inspect.taxonomy import (  # noqa: F401
    gate,
    parse_label_set,
)

# The operators, then the helpers their callers need beside them, then the names a
# caller catches or compares against.
__all__ = [
    # operators
    "inspect",
    "inspect_labeling_data",
    "detect",
    "gate",
    # helpers
    "parse_label_set",
]
