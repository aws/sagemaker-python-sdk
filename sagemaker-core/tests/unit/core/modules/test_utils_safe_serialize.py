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
"""Tests for safe_serialize in sagemaker.core.modules.utils with PipelineVariable support.

Verifies that safe_serialize correctly handles PipelineVariable objects
(e.g., ParameterInteger, ParameterString) by returning them as-is rather
than attempting str() conversion which would raise TypeError.

See: https://github.com/aws/sagemaker-python-sdk/issues/5504
"""
from __future__ import absolute_import

import pytest

from sagemaker.core.modules.utils import safe_serialize
from sagemaker.core.helper.pipeline_variable import PipelineVariable
from sagemaker.core.workflow.parameters import ParameterInteger, ParameterString


class TestSafeSerializeWithPipelineVariables:
    """Test safe_serialize handles PipelineVariable objects correctly."""

    def test_safe_serialize_with_parameter_integer(self):
        """ParameterInteger should be returned as-is (identity preserved)."""
        param = ParameterInteger(name="MaxDepth", default_value=5)
        result = safe_serialize(param)
        assert result is param
        assert isinstance(result, PipelineVariable)

    def test_safe_serialize_with_parameter_string(self):
        """ParameterString should be returned as-is (identity preserved)."""
        param = ParameterString(name="Algorithm", default_value="xgboost")
        result = safe_serialize(param)
        assert result is param
        assert isinstance(result, PipelineVariable)

    def test_safe_serialize_does_not_call_str_on_pipeline_variable(self):
        """Verify that PipelineVariable.__str__ is never invoked (would raise TypeError)."""
        param = ParameterInteger(name="TestParam", default_value=10)
        # This should NOT raise TypeError
        result = safe_serialize(param)
        assert result is param


class TestSafeSerializeBasicTypes:
    """Regression tests: verify basic types still work after PipelineVariable support."""

    def test_safe_serialize_with_string(self):
        """Strings should be returned as-is without JSON wrapping."""
        assert safe_serialize("hello") == "hello"

    def test_safe_serialize_with_int(self):
        """Integers should be JSON-serialized to string."""
        assert safe_serialize(42) == "42"

    def test_safe_serialize_with_dict(self):
        """Dicts should be JSON-serialized."""
        result = safe_serialize({"key": "val"})
        assert result == '{"key": "val"}'

    def test_safe_serialize_with_bool(self):
        """Booleans should be JSON-serialized."""
        assert safe_serialize(True) == "true"
        assert safe_serialize(False) == "false"

    def test_safe_serialize_with_none(self):
        """None should be JSON-serialized to 'null'."""
        assert safe_serialize(None) == "null"

    def test_safe_serialize_with_custom_object(self):
        """Custom objects should fall back to str()."""

        class CustomObj:
            def __str__(self):
                return "custom"

        assert safe_serialize(CustomObj()) == "custom"
