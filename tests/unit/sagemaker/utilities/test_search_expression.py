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
from __future__ import absolute_import

import pytest

from sagemaker.utilities.search_expression import (
    Filter,
    Operator,
    NestedFilter,
    SearchExpression,
    BooleanOperator,
)


def test_filters():
    search_filter = Filter(name="learning_rate", operator=Operator.EQUALS, value="0.1")

    assert {
        "Name": "learning_rate",
        "Operator": "Equals",
        "Value": "0.1",
    } == search_filter.to_boto()


def test_partial_filters():
    search_filter = Filter(name="learning_rate")

    assert {"Name": "learning_rate"} == search_filter.to_boto()


def test_nested_filters():
    search_filter = Filter(name="learning_rate", operator=Operator.EQUALS, value="0.1")
    filters = [search_filter]
    nested_filters = NestedFilter(property_name="hyper_param", filters=filters)

    assert {
        "Filters": [{"Name": "learning_rate", "Operator": "Equals", "Value": "0.1"}],
        "NestedPropertyName": "hyper_param",
    } == nested_filters.to_boto()


def test_search_expression():
    search_filter = Filter(name="learning_rate", operator=Operator.EQUALS, value="0.1")
    nested_filter = NestedFilter(property_name="hyper_param", filters=[search_filter])
    search_expression = SearchExpression(
        filters=[search_filter],
        nested_filters=[nested_filter],
        sub_expressions=[],
        boolean_operator=BooleanOperator.AND,
    )

    assert {
        "Filters": [{"Name": "learning_rate", "Operator": "Equals", "Value": "0.1"}],
        "NestedFilters": [
            {
                "Filters": [{"Name": "learning_rate", "Operator": "Equals", "Value": "0.1"}],
                "NestedPropertyName": "hyper_param",
            }
        ],
        "SubExpressions": [],
        "Operator": "And",
    } == search_expression.to_boto()


def test_illegal_search_expression():
    with pytest.raises(
        ValueError, match="You must specify at least one subexpression, filter, or nested filter"
    ):
        SearchExpression()
