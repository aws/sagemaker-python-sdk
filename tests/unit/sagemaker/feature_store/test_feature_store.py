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

import pandas as pd
import pytest
from mock import Mock

from sagemaker.feature_store.feature_store import FeatureStore

DATAFRAME = pd.DataFrame({"feature_1": [420, 380, 390], "feature_2": [50, 40, 45]})


@pytest.fixture
def sagemaker_session_mock():
    return Mock()


@pytest.fixture
def feature_group_mock():
    return Mock()


def test_minimal_create_dataset(sagemaker_session_mock, feature_group_mock):
    feature_store = FeatureStore(sagemaker_session=sagemaker_session_mock)
    dataset_builder = feature_store.create_dataset(
        base=feature_group_mock,
        output_path="file/to/path",
    )
    assert dataset_builder._sagemaker_session == sagemaker_session_mock
    assert dataset_builder._base == feature_group_mock
    assert dataset_builder._output_path == "file/to/path"


def test_complete_create_dataset(sagemaker_session_mock, feature_group_mock):
    feature_store = FeatureStore(sagemaker_session=sagemaker_session_mock)
    dataset_builder = feature_store.create_dataset(
        base=feature_group_mock,
        included_feature_names=["feature_1", "feature_2"],
        output_path="file/to/path",
        kms_key_id="kms-key-id",
    )
    assert dataset_builder._sagemaker_session == sagemaker_session_mock
    assert dataset_builder._base == feature_group_mock
    assert dataset_builder._included_feature_names == ["feature_1", "feature_2"]
    assert dataset_builder._output_path == "file/to/path"
    assert dataset_builder._kms_key_id == "kms-key-id"


def test_create_dataset_with_dataframe(sagemaker_session_mock):
    feature_store = FeatureStore(sagemaker_session=sagemaker_session_mock)
    dataset_builder = feature_store.create_dataset(
        base=DATAFRAME,
        record_identifier_feature_name="feature_1",
        event_time_identifier_feature_name="feature_2",
        included_feature_names=["feature_1", "feature_2"],
        output_path="file/to/path",
        kms_key_id="kms-key-id",
    )
    assert dataset_builder._sagemaker_session == sagemaker_session_mock
    assert dataset_builder._base.equals(DATAFRAME)
    assert dataset_builder._record_identifier_feature_name == "feature_1"
    assert dataset_builder._event_time_identifier_feature_name == "feature_2"
    assert dataset_builder._included_feature_names == ["feature_1", "feature_2"]
    assert dataset_builder._output_path == "file/to/path"
    assert dataset_builder._kms_key_id == "kms-key-id"


def test_create_dataset_with_dataframe_value_error(sagemaker_session_mock):
    feature_store = FeatureStore(sagemaker_session=sagemaker_session_mock)
    with pytest.raises(ValueError) as error:
        feature_store.create_dataset(
            base=DATAFRAME,
            included_feature_names=["feature_1", "feature_2"],
            output_path="file/to/path",
            kms_key_id="kms-key-id",
        )
    assert (
        "You must provide a record identifier feature name and an event time identifier feature "
        + "name if specify DataFrame as base."
        in str(error)
    )


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
