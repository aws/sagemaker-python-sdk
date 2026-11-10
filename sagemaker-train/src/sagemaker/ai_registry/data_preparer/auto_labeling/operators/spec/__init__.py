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
"""What are the rules, and which version ships?

Spec induction, the judge rubric, selection, and adopting the in-job pick.

``select`` and ``adopt`` name the operators here; their modules are reached with
``importlib.import_module``.
"""

from __future__ import absolute_import

from sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec.induce import (  # noqa: F401
    induce_spec,
    draft_labeling_spec,
    RUBRIC_TASKS,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec.select import (  # noqa: F401
    select,
    select_labeling_spec,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec.adopt import (  # noqa: F401
    adopt,
    check_pick_folder,
    check_shippable,
    no_adopted_spec_message,
    PICK_FILES,
    AdoptError,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec.rubric import (  # noqa: F401
    rubric_gaps,
)

# The operators, then the helpers their callers need beside them, then the names a
# caller catches or compares against.
__all__ = [
    # operators
    "induce_spec",
    "draft_labeling_spec",
    "select",
    "select_labeling_spec",
    "adopt",
    "rubric_gaps",
    # helpers
    "check_pick_folder",
    "check_shippable",
    "no_adopted_spec_message",
    # constants and exceptions
    "RUBRIC_TASKS",
    "PICK_FILES",
    "AdoptError",
]
