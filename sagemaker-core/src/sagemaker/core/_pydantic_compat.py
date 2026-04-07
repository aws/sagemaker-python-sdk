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
"""Pydantic compatibility check for sagemaker-core.

This module provides an early check for pydantic/pydantic-core version
compatibility to give users a clear error message with fix instructions
instead of a cryptic SystemError.
"""


def check_pydantic_compatibility():
    """Check that pydantic and pydantic-core versions are compatible.

    Raises:
        ImportError: If pydantic and pydantic-core versions are incompatible,
            with instructions on how to fix the issue.
    """
    try:
        import pydantic  # noqa: F401
    except SystemError as e:
        error_message = str(e)
        raise ImportError(
            f"Pydantic version incompatibility detected: {error_message}\n\n"
            "This typically happens when pydantic-core is upgraded independently "
            "of pydantic, causing a version mismatch.\n\n"
            "To fix this, run:\n"
            "    pip install pydantic pydantic-core --force-reinstall\n\n"
            "This will ensure both packages are installed at compatible versions."
        ) from e

    try:
        import pydantic_core  # noqa: F401
    except ImportError:
        # pydantic_core not installed separately is fine;
        # pydantic manages it as a dependency
        return

    # Additional version check: pydantic declares the exact pydantic-core
    # version it requires. Verify they match.
    try:
        pydantic_version = pydantic.VERSION
        pydantic_core_version = pydantic_core.VERSION

        # pydantic >= 2.x stores the required core version
        expected_core_version = getattr(pydantic, '__pydantic_core_version__', None)
        if expected_core_version is None:
            # Try alternative attribute name used in some pydantic versions
            expected_core_version = getattr(
                pydantic, '_internal', None
            ) and getattr(
                getattr(pydantic, '_internal', None),
                '_generate_schema',
                None,
            )
            # If we can't determine the expected version, skip the check
            return

        if pydantic_core_version != expected_core_version:
            raise ImportError(
                f"Pydantic/pydantic-core version mismatch detected: "
                f"pydantic {pydantic_version} requires pydantic-core=={expected_core_version}, "
                f"but pydantic-core {pydantic_core_version} is installed.\n\n"
                "To fix this, run:\n"
                "    pip install pydantic pydantic-core --force-reinstall\n\n"
                "This will ensure both packages are installed at compatible versions."
            )
    except (AttributeError, TypeError):
        # If we can't determine versions, skip the check
        # The SystemError catch above will handle the most common case
        pass
