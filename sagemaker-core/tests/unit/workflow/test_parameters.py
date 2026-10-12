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
"""Unit tests for pipeline parameters, focusing on hashability."""

from __future__ import absolute_import

import pytest

from sagemaker.core.workflow.parameters import (
    Parameter,
    ParameterBoolean,
    ParameterFloat,
    ParameterInteger,
    ParameterString,
    ParameterTypeEnum,
)


@pytest.fixture
def all_parameters():
    """Return one instance of each concrete parameter type."""
    return [
        ParameterString(name="MyString", default_value="foo"),
        ParameterInteger(name="MyInteger", default_value=1),
        ParameterFloat(name="MyFloat", default_value=1.0),
        ParameterBoolean(name="MyBoolean", default_value=True),
    ]


def test_all_parameters_are_hashable(all_parameters):
    """Every concrete parameter type must be hashable."""
    for param in all_parameters:
        assert isinstance(hash(param), int)


def test_base_parameter_is_hashable():
    """The base Parameter must be hashable."""
    param = Parameter(name="Base", parameter_type=ParameterTypeEnum.STRING)
    assert isinstance(hash(param), int)


def test_parameters_usable_as_set_members(all_parameters):
    """Parameters must be usable as set members without raising."""
    param_set = set(all_parameters)
    assert len(param_set) == len(all_parameters)
    for param in all_parameters:
        assert param in param_set


def test_parameters_usable_as_dict_keys(all_parameters):
    """Parameters must be usable as dict keys without raising."""
    mapping = {param: param.name for param in all_parameters}
    for param in all_parameters:
        assert mapping[param] == param.name


@pytest.mark.parametrize(
    "factory, kwargs",
    [
        (ParameterString, {"name": "P", "default_value": "v"}),
        (ParameterInteger, {"name": "P", "default_value": 3}),
        (ParameterFloat, {"name": "P", "default_value": 3.0}),
        (ParameterBoolean, {"name": "P", "default_value": False}),
    ],
)
def test_equal_parameters_hash_equal(factory, kwargs):
    """Equal parameters must hash to the same value (eq/hash invariant)."""
    first = factory(**kwargs)
    second = factory(**kwargs)
    assert first == second
    assert hash(first) == hash(second)


def test_different_type_parameters_are_distinct():
    """Parameters of the same name but different type are unequal and distinct.

    Note hash *inequality* is not part of the hashing contract (collisions are
    legal), so we assert on equality/distinct-membership rather than hashes.
    """
    integer_param = ParameterInteger(name="Shared")
    float_param = ParameterFloat(name="Shared")
    assert integer_param != float_param
    assert len({integer_param, float_param}) == 2


def test_parameter_string_dedupes_equal_instances():
    """ParameterString remains hashable and dedupes equal instances in a set."""
    first = ParameterString(name="Dup", default_value="x")
    second = ParameterString(name="Dup", default_value="x")
    assert len({first, second}) == 1


def test_parameter_string_hash_invariant_ignores_enum_values():
    """Equal ParameterStrings that differ only in enum_values must hash equal.

    ``enum_values`` is not part of attrs equality, so two such instances are
    equal; the old ``hash(tuple(self.to_request()))`` implementation broke the
    ``a == b`` implies ``hash(a) == hash(b)`` invariant here.
    """
    with_enum = ParameterString(name="P", enum_values=["a", "b"])
    without_enum = ParameterString(name="P")
    assert with_enum == without_enum
    assert hash(with_enum) == hash(without_enum)
