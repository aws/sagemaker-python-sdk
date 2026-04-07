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
"""Tests for pydantic compatibility check."""

import sys
from unittest import mock

import pytest


def test_check_pydantic_compatibility_passes_with_matching_versions():
    """Verify the check function does not raise when pydantic and pydantic-core are compatible."""
    from sagemaker.core._pydantic_compat import check_pydantic_compatibility

    # Should not raise any exception with the currently installed versions
    check_pydantic_compatibility()


def test_check_pydantic_compatibility_raises_on_system_error():
    """Mock pydantic import to raise SystemError and verify a clear ImportError is raised."""
    from sagemaker.core._pydantic_compat import check_pydantic_compatibility

    error_msg = (
        "The installed pydantic-core version (2.42.0) is incompatible "
        "with the current pydantic version, which requires 2.41.5."
    )

    with mock.patch.dict(sys.modules, {"pydantic": None}):
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == "pydantic":
                raise SystemError(error_msg)
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError) as exc_info:
                check_pydantic_compatibility()

            assert "incompatibility detected" in str(exc_info.value).lower() or \
                   "incompatible" in str(exc_info.value).lower()


def test_pydantic_import_error_message_contains_instructions():
    """Verify the error message includes pip install instructions."""
    from sagemaker.core._pydantic_compat import check_pydantic_compatibility

    error_msg = (
        "The installed pydantic-core version (2.42.0) is incompatible "
        "with the current pydantic version, which requires 2.41.5."
    )

    with mock.patch.dict(sys.modules, {"pydantic": None}):
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == "pydantic":
                raise SystemError(error_msg)
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError) as exc_info:
                check_pydantic_compatibility()

            error_str = str(exc_info.value)
            assert "pip install pydantic pydantic-core --force-reinstall" in error_str
