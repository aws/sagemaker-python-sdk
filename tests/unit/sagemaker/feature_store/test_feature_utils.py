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
# language governing permissions and limitations under the License.
"""Test for Feature Group Utils"""
from __future__ import absolute_import

import pandas as pd
import pytest
from mock import Mock

from sagemaker.feature_store.feature_utils import (
    _cast_object_to_string,
    prepare_fg_from_dataframe_or_file,
    get_feature_group_as_dataframe,
)
from sagemaker.feature_store.feature_definition import (
    FeatureTypeEnum,
)
from sagemaker.feature_store.feature_group import (
    FeatureGroup,
)


class PicklableMock(Mock):
    """Mock class use for tests"""

    def __reduce__(self):
        """Method from class Mock"""
        return (Mock, ())


@pytest.fixture
def sagemaker_session_mock():
    """Fixture Mock class"""
    mock = Mock()
    mock.sagemaker_config = None
    return mock


def test_convert_unsupported_types_to_supported(sagemaker_session_mock):
    feature_group = FeatureGroup(name="FailedGroup", sagemaker_session=sagemaker_session_mock)
    df = pd.DataFrame(
        {
            "float": pd.Series([2.0], dtype="float64"),
            "int": pd.Series([2], dtype="int64"),
            "object": pd.Series(["f1"], dtype="object"),
        }
    )
    # Converting object or O type to string
    df = _cast_object_to_string(data_frame=df)

    feature_definitions = feature_group.load_feature_definitions(data_frame=df)
    types = [fd.feature_type for fd in feature_definitions]

    assert types == [
        FeatureTypeEnum.FRACTIONAL,
        FeatureTypeEnum.INTEGRAL,
        FeatureTypeEnum.STRING,
    ]


def test_prepare_fg_from_dataframe(sagemaker_session_mock):
    very_long_name = "long" * 20
    df = pd.DataFrame(
        {
            "space feature": pd.Series([2.0], dtype="float64"),
            "dot.feature": pd.Series([2], dtype="int64"),
            very_long_name: pd.Series(["f1"], dtype="string"),
        }
    )

    feature_group = prepare_fg_from_dataframe_or_file(
        dataframe_or_path=df,
        session=sagemaker_session_mock,
        feature_group_name="testFG",
    )

    names = [fd.feature_name for fd in feature_group.feature_definitions]
    types = [fd.feature_type for fd in feature_group.feature_definitions]

    assert names == [
        "space_feature",
        "dotfeature",
        very_long_name[:62],
        "record_id",
        "data_as_of_date",
    ]
    assert types == [
        FeatureTypeEnum.FRACTIONAL,
        FeatureTypeEnum.INTEGRAL,
        FeatureTypeEnum.STRING,
        FeatureTypeEnum.INTEGRAL,
        FeatureTypeEnum.FRACTIONAL,
    ]


def test_get_fg_latest_without_eventid(sagemaker_session_mock):
    with pytest.raises(Exception):
        get_feature_group_as_dataframe(
            session=sagemaker_session_mock,
            feature_group_name="testFG",
            athena_bucket="s3://test",
            latest_ingestion=True,
            event_time_feature_name=None,
        )


def test_get_fg_without_sess_role_region(sagemaker_session_mock):
    with pytest.raises(Exception):
        get_feature_group_as_dataframe(
            session=None,
            region=None,
            role=None,
            feature_group_name="testFG",
            athena_bucket="s3://test",
            latest_ingestion=False,
        )
