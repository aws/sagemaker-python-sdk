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

from tests.integ.sagemaker.experiments.helpers import EXP_INTEG_TEST_NAME_PREFIX
from sagemaker.experiments.trial_component import _TrialComponent
from sagemaker.utilities.search_expression import Filter, Operator, SearchExpression, NestedFilter


def test_search(sagemaker_session):
    tc_names_searched = []
    search_filter = Filter(
        name="TrialComponentName", operator=Operator.CONTAINS, value=EXP_INTEG_TEST_NAME_PREFIX
    )
    search_expression = SearchExpression(filters=[search_filter])
    for tc in _TrialComponent.search(
        search_expression=search_expression, max_results=10, sagemaker_session=sagemaker_session
    ):
        tc_names_searched.append(tc.trial_component_name)

    assert len(tc_names_searched) > 0
    assert tc_names_searched


@pytest.mark.skip(reason="failed validation, need to wait for NestedFilter bug to be fixed")
def test_nested_search(sagemaker_session):
    tc_names_searched = []
    search_filter = Filter(
        name="TrialComponentName", operator=Operator.CONTAINS, value=EXP_INTEG_TEST_NAME_PREFIX
    )
    nested_filter = NestedFilter(property_name="TrialComponentName", filters=[search_filter])
    search_expression = SearchExpression(nested_filters=[nested_filter])
    for tc in _TrialComponent.search(
        search_expression=search_expression, max_results=10, sagemaker_session=sagemaker_session
    ):
        tc_names_searched.append(tc.trial_component_name)

    assert len(tc_names_searched) > 0
    assert tc_names_searched


def test_sub_expression(sagemaker_session):
    tc_names_searched = []
    search_filter = Filter(
        name="TrialComponentName", operator=Operator.CONTAINS, value=EXP_INTEG_TEST_NAME_PREFIX
    )
    sub_expression = SearchExpression(filters=[search_filter])
    search_expression = SearchExpression(sub_expressions=[sub_expression])
    for tc in _TrialComponent.search(
        search_expression=search_expression, max_results=10, sagemaker_session=sagemaker_session
    ):
        tc_names_searched.append(tc.trial_component_name)

    assert len(tc_names_searched) > 0
    assert tc_names_searched
