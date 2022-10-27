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

from sagemaker.feature_store.dataset_builder import DatasetBuilder


@pytest.fixture
def sagemaker_session_mock():
    return Mock()


@pytest.fixture
def feature_group_mock():
    return Mock()


def test_point_in_time_accurate_join(sagemaker_session_mock, feature_group_mock):
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=feature_group_mock,
        output_path="file/to/path",
    )
    dataset_builder.point_in_time_accurate_join()
    assert dataset_builder._point_in_time_accurate_join


def test_include_duplicated_records(sagemaker_session_mock, feature_group_mock):
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=feature_group_mock,
        output_path="file/to/path",
    )
    dataset_builder.include_duplicated_records()
    assert dataset_builder._include_duplicated_records


def test_include_deleted_records(sagemaker_session_mock, feature_group_mock):
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=feature_group_mock,
        output_path="file/to/path",
    )
    dataset_builder.include_deleted_records()
    assert dataset_builder._include_deleted_records


def test_with_number_of_recent_records_by_record_identifier(
    sagemaker_session_mock, feature_group_mock
):
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=feature_group_mock,
        output_path="file/to/path",
    )
    dataset_builder.with_number_of_recent_records_by_record_identifier(5)
    assert dataset_builder._number_of_recent_records == 5


def test_with_number_of_records_from_query_results(sagemaker_session_mock, feature_group_mock):
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=feature_group_mock,
        output_path="file/to/path",
    )
    dataset_builder.with_number_of_records_from_query_results(100)
    assert dataset_builder._number_of_records == 100


def test_as_of(sagemaker_session_mock, feature_group_mock):
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=feature_group_mock,
        output_path="file/to/path",
    )
    time = datetime.datetime.now()
    dataset_builder.as_of(time)
    assert dataset_builder._write_time_ending_timestamp == time


def test_with_event_time_range(sagemaker_session_mock, feature_group_mock):
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=feature_group_mock,
        output_path="file/to/path",
    )
    start = datetime.datetime.now()
    end = start + datetime.timedelta(minutes=1)
    dataset_builder.with_event_time_range(start, end)
    assert dataset_builder._event_time_starting_timestamp == start
    assert dataset_builder._event_time_ending_timestamp == end
