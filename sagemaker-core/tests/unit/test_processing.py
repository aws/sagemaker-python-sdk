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
"""Tests for sagemaker-core processing functionality.

Note: Tests for the FrameworkProcessor._package_code Windows PermissionError fix
(issue #5873) are located in tests/unit/test_processing_windows_fix.py since the
bug is in the main SageMaker SDK's sagemaker/processing.py module.
"""
import pytest


class TestProcessingPlaceholder:
    """Placeholder test class for sagemaker-core processing tests."""

    def test_placeholder(self):
        """Placeholder test to prevent empty test file warnings."""
        pass
