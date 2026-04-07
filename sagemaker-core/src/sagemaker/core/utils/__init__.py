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
"""Utilities package for SageMaker Core.

This package re-exports commonly used utility functions from common_utils
for backward compatibility and convenience.

Note: Uses lazy imports via __getattr__ to avoid circular import issues.
"""
from __future__ import absolute_import

# Public API surface: only non-private functions are exported via __all__.
# Private helpers (_get_resolved_path, _is_bad_path, _is_bad_link, _get_safe_members)
# are still importable directly but are not part of the public API.
__all__ = [
    "_save_model",
    "download_file_from_url",
    "custom_extractall_tarfile",
    "download_file",
    "download_folder",
    "create_tar_file",
    "repack_model",
    "name_from_image",
    "name_from_base",
    "unique_name_from_base",
    "unique_name_from_base_uuid4",
    "base_name_from_image",
    "base_from_name",
    "sagemaker_timestamp",
    "sagemaker_short_timestamp",
    "get_config_value",
]

# Internal helpers that are importable but not part of the public API
_INTERNAL_NAMES = [
    "_get_resolved_path",
    "_is_bad_path",
    "_is_bad_link",
    "_get_safe_members",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name in __all__ or name in _INTERNAL_NAMES:
        from sagemaker.core import common_utils

        return getattr(common_utils, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
