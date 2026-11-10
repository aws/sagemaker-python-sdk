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
"""What am I measuring against, and is it still trustworthy?

The cold-start sample, the frozen split, folds, demos, and the people's labels.
"""

from __future__ import absolute_import

from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.cold_start import (  # noqa: F401
    draw_sample,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.split import (  # noqa: F401
    freeze,
    prepare_labeling_seed,
    guard_read,
    rehash,
    relabel,
    find_splits_dir,
    load_state,
    state_path,
    test_ids,
    test_read_event,
    SplitGuardError,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.folds import (  # noqa: F401
    build_kfold,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.demos import (  # noqa: F401
    select_demos,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.labels import (  # noqa: F401
    check_seed_append,
    merge_seed_labels,
    build_overlay,
    SeedAppendRefused,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.review_export import (  # noqa: F401
    flatten_export,
)

# The operators, then the helpers their callers need beside them, then the names a
# caller catches or compares against.
__all__ = [
    # operators
    "draw_sample",
    "freeze",
    "prepare_labeling_seed",
    "guard_read",
    "rehash",
    "relabel",
    "build_kfold",
    "select_demos",
    "check_seed_append",
    "merge_seed_labels",
    "build_overlay",
    "flatten_export",
    # helpers
    "find_splits_dir",
    "load_state",
    "state_path",
    "test_ids",
    "test_read_event",
    # constants and exceptions
    "SplitGuardError",
    "SeedAppendRefused",
]
