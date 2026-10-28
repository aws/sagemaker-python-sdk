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
"""Tests that public generate/evaluate/transform docstrings are autodoc-ready.

autodoc resolves class-level attribute access, which means _DualCallable.__get__
returns a functools.wraps-decorated function whose __doc__ is taken from the
underlying _impl.  These tests assert that the resolved docstring contains the
expected sections and keywords so the published API reference is complete.

generate and evaluate FAIL on the baseline (one-line impl docstrings); transform
passes already (full docstring on _transform_impl).
"""
from __future__ import absolute_import

import pytest

from sagemaker.ai_registry.data_preparer.preparer import DataPreparer

_MIN_DOCSTRING_LENGTH = 200


def _resolved_doc(attr_name: str) -> str:
    """Return the docstring that autodoc sees for DataPreparer.<attr_name>."""
    return getattr(DataPreparer, attr_name).__doc__ or ""


@pytest.mark.parametrize("method", ["generate", "evaluate", "transform"])
class TestPublicDocstringsComplete:
    """Each public primitive must have a full, autodoc-ready docstring."""

    def test_contains_args_section(self, method):
        doc = _resolved_doc(method)
        assert "Args:" in doc or "Args\n" in doc, (
            f"DataPreparer.{method}.__doc__ missing 'Args:' section"
        )

    def test_contains_returns_section(self, method):
        doc = _resolved_doc(method)
        assert "Returns:" in doc or "Returns\n" in doc, (
            f"DataPreparer.{method}.__doc__ missing 'Returns:' section"
        )

    def test_contains_raises_section(self, method):
        doc = _resolved_doc(method)
        assert "Raises:" in doc or "Raises\n" in doc, (
            f"DataPreparer.{method}.__doc__ missing 'Raises:' section"
        )

    def test_mentions_wait_parameter(self, method):
        doc = _resolved_doc(method)
        assert "wait" in doc, (
            f"DataPreparer.{method}.__doc__ does not mention the 'wait' parameter"
        )

    def test_docstring_length(self, method):
        doc = _resolved_doc(method)
        assert len(doc) >= _MIN_DOCSTRING_LENGTH, (
            f"DataPreparer.{method}.__doc__ is too short "
            f"({len(doc)} chars, expected >= {_MIN_DOCSTRING_LENGTH})"
        )
