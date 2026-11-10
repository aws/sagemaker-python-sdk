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
"""How good is it, and can I prove it?

Scoring a predictions file against gold, locally.
"""

from __future__ import absolute_import

from sagemaker.ai_registry.data_preparer.auto_labeling.operators.score.scorers import (  # noqa: F401
    score,
    parse_label_map,
    label_map_from_names,
    MAX_ROWS,
)

# The operators, then the helpers their callers need beside them, then the names a
# caller catches or compares against.
__all__ = [
    # operators
    "score",
    # helpers
    "parse_label_map",
    "label_map_from_names",
    # constants and exceptions
    "MAX_ROWS",
]
