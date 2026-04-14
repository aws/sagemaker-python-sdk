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

When pydantic-core is upgraded independently of pydantic (e.g., via
``pip install --force-reinstall --no-deps``), Python raises a ``SystemError``
at import time.  This module catches that error and converts it into a
helpful ``ImportError`` with remediation steps.
"""


def check_pydantic_compatibility():
    """Check that pydantic and pydantic-core versions are compatible.

    Pydantic internally requires an exact matching pydantic-core version
    (e.g., pydantic 2.11.5 requires pydantic-core==2.41.5).  If the two
    packages are out of sync, ``import pydantic`` raises a ``SystemError``.

    This function catches that ``SystemError`` and re-raises it as an
    ``ImportError`` with clear instructions on how to fix the issue.

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
