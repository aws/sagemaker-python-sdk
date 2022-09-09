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

import datetime

import pytest
from mock import Mock

from sagemaker.feature_store.feature_store import FeatureStore


@pytest.fixture
def sagemaker_session_mock():
    return Mock()


def test_list_feature_groups_with_no_filter(sagemaker_session_mock):
    feature_store = FeatureStore(sagemaker_session=sagemaker_session_mock)
    feature_store.list_feature_groups()
    sagemaker_session_mock.list_feature_groups.assert_called_with(
        name_contains=None,
        feature_group_status_equals=None,
        offline_store_status_equals=None,
        creation_time_after=None,
        creation_time_before=None,
        sort_order=None,
        sort_by=None,
        max_results=None,
        next_token=None,
    )


def test_list_feature_groups_with_all_filters(sagemaker_session_mock):
    feature_store = FeatureStore(sagemaker_session=sagemaker_session_mock)
    feature_store.list_feature_groups(
        name_contains="MyFeatureGroup",
        feature_group_status_equals="Created",
        offline_store_status_equals="Active",
        creation_time_after=datetime.datetime(2020, 12, 1),
        creation_time_before=datetime.datetime(2022, 7, 1),
        sort_order="Ascending",
        sort_by="Name",
        max_results=50,
        next_token="token",
    )
    sagemaker_session_mock.list_feature_groups.assert_called_with(
        name_contains="MyFeatureGroup",
        feature_group_status_equals="Created",
        offline_store_status_equals="Active",
        creation_time_after=datetime.datetime(2020, 12, 1),
        creation_time_before=datetime.datetime(2022, 7, 1),
        sort_order="Ascending",
        sort_by="Name",
        max_results=50,
        next_token="token",
    )
