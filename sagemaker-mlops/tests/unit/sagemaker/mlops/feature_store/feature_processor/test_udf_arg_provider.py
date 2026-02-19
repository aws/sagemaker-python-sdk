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
from __future__ import absolute_import

import pytest
import test_data_helpers as tdh
from mock import Mock, patch
from pyspark.sql import DataFrame, SparkSession

from sagemaker.mlops.feature_store.feature_processor._input_loader import InputLoader
from sagemaker.mlops.feature_store.feature_processor._params_loader import ParamsLoader
from sagemaker.mlops.feature_store.feature_processor._spark_factory import SparkSessionFactory
from sagemaker.mlops.feature_store.feature_processor._udf_arg_provider import SparkArgProvider
from sagemaker.mlops.feature_store.feature_processor._data_source import PySparkDataSource


@pytest.fixture
def params_loader():
    params_loader = Mock(ParamsLoader)
    params_loader.get_parameter_args = Mock(return_value={"params": {"key": "value"}})
    return params_loader


@pytest.fixture
def feature_group_as_spark_df():
    return Mock(DataFrame)


@pytest.fixture
def s3_uri_as_spark_df():
    return Mock(DataFrame)


@pytest.fixture
def base_data_source_as_spark_df():
    return Mock(DataFrame)


@pytest.fixture
def input_loader(feature_group_as_spark_df, s3_uri_as_spark_df):
    input_loader = Mock(InputLoader)
    input_loader.load_from_s3.return_value = s3_uri_as_spark_df
    input_loader.load_from_feature_group.return_value = feature_group_as_spark_df

    return input_loader


@pytest.fixture
def spark_session():
    return Mock(SparkSession)


@pytest.fixture
def spark_session_factory(spark_session):
    return Mock(SparkSessionFactory, spark_session=spark_session)


@pytest.fixture
def spark_arg_provider(params_loader, input_loader, spark_session_factory):
    return SparkArgProvider(params_loader, input_loader, spark_session_factory)


class MockDataSource(PySparkDataSource):

    data_source_unique_id = "test_id"
    data_source_name = "test_source"

    def read_data(self, spark, params) -> DataFrame:
        return Mock(DataFrame)


def test_provide_additional_kw_args(spark_arg_provider, spark_session):
    def udf(fg_input, s3_input, params, spark):
        return None

    additional_kw_args = spark_arg_provider.provide_additional_kwargs(udf)

    assert additional_kw_args.keys() == {"spark"}
    assert additional_kw_args["spark"] == spark_session


def test_not_provide_additional_kw_args(spark_arg_provider):
    def udf(input, params):
        return None

    additional_kw_args = spark_arg_provider.provide_additional_kwargs(udf)

    assert additional_kw_args == {}


def test_provide_params(spark_arg_provider, params_loader):
    fp_config = tdh.create_fp_config(
        inputs=[tdh.FEATURE_GROUP_DATA_SOURCE, tdh.S3_DATA_SOURCE],
        output=tdh.OUTPUT_FEATURE_GROUP_ARN,
    )

    def udf(fg_input, s3_input, params, spark):
        return None

    params = spark_arg_provider.provide_params_arg(udf, fp_config)

    params_loader.get_parameter_args.assert_called_with(fp_config)
    assert params == params_loader.get_parameter_args.return_value


def test_not_provide_params(spark_arg_provider, params_loader):
    fp_config = tdh.create_fp_config(
        inputs=[tdh.FEATURE_GROUP_DATA_SOURCE, tdh.S3_DATA_SOURCE],
        output=tdh.OUTPUT_FEATURE_GROUP_ARN,
    )

    def udf(fg_input, s3_input, spark):
        return None

    params = spark_arg_provider.provide_params_arg(udf, fp_config)

    assert params == {}


def test_provide_input_args_with_no_input(spark_arg_provider):
    fp_config = tdh.create_fp_config(inputs=[], output=tdh.OUTPUT_FEATURE_GROUP_ARN)

    def udf() -> DataFrame:
        return Mock(DataFrame)

    with pytest.raises(
        ValueError, match="Expected at least one input to the user defined function."
    ):
        spark_arg_provider.provide_input_args(udf, fp_config)


def test_provide_input_args_with_extra_udf_parameters(spark_arg_provider):
    fp_config = tdh.create_fp_config(
        inputs=[tdh.INPUT_S3_URI],
        output=tdh.OUTPUT_FEATURE_GROUP_ARN,
    )

    def udf(input_fg=None, input_s3_uri=None) -> DataFrame:
        return Mock(DataFrame)

    with pytest.raises(
        ValueError,
        match=r"The signature of the user defined function does not match the list of inputs requested."
        r" Expected 1 parameter\(s\).",
    ):
        spark_arg_provider.provide_input_args(udf, fp_config)


def test_provide_input_args_with_extra_fp_config_inputs(spark_arg_provider):
    fp_config = tdh.create_fp_config(
        inputs=[tdh.FEATURE_GROUP_DATA_SOURCE, tdh.S3_DATA_SOURCE],
        output=tdh.OUTPUT_FEATURE_GROUP_ARN,
    )

    def udf(input_fg=None) -> DataFrame:
        return Mock(DataFrame)

    with pytest.raises(
        ValueError,
        match=r"The signature of the user defined function does not match the list of inputs requested."
        r" Expected 2 parameter\(s\).",
    ):
        spark_arg_provider.provide_input_args(udf, fp_config)


def test_provide_input_args(
    spark_arg_provider,
    feature_group_as_spark_df,
    s3_uri_as_spark_df,
):
    fp_config = tdh.create_fp_config(
        inputs=[tdh.FEATURE_GROUP_DATA_SOURCE, tdh.S3_DATA_SOURCE],
        output=tdh.OUTPUT_FEATURE_GROUP_ARN,
    )

    def udf(input_fg=None, input_s3_uri=None) -> DataFrame:
        return Mock(DataFrame)

    inputs = spark_arg_provider.provide_input_args(udf, fp_config)

    assert inputs.keys() == {"input_fg", "input_s3_uri"}
    assert inputs["input_fg"] == feature_group_as_spark_df
    assert inputs["input_s3_uri"] == s3_uri_as_spark_df


def test_provide_input_args_with_reversed_inputs(
    spark_arg_provider,
    feature_group_as_spark_df,
    s3_uri_as_spark_df,
):
    fp_config = tdh.create_fp_config(
        inputs=[tdh.S3_DATA_SOURCE, tdh.FEATURE_GROUP_DATA_SOURCE],
        output=tdh.OUTPUT_FEATURE_GROUP_ARN,
    )

    def udf(input_fg=None, input_s3_uri=None) -> DataFrame:
        return Mock(DataFrame)

    inputs = spark_arg_provider.provide_input_args(udf, fp_config)

    assert inputs.keys() == {"input_fg", "input_s3_uri"}
    assert inputs["input_fg"] == s3_uri_as_spark_df
    assert inputs["input_s3_uri"] == feature_group_as_spark_df


def test_provide_input_args_with_optional_args_out_of_order(spark_arg_provider):
    fp_config = tdh.create_fp_config(
        inputs=[tdh.FEATURE_GROUP_DATA_SOURCE, tdh.S3_DATA_SOURCE],
        output=tdh.OUTPUT_FEATURE_GROUP_ARN,
    )

    def udf_spark_params(spark=None, params=None, input_fg=None, input_s3_uri=None) -> DataFrame:
        return Mock(DataFrame)

    def udf_params_spark(params=None, spark=None, input_fg=None, input_s3_uri=None) -> DataFrame:
        return Mock(DataFrame)

    def udf_spark(spark=None, input_fg=None, input_s3_uri=None) -> DataFrame:
        return Mock(DataFrame)

    def udf_params(params=None, input_fg=None, input_s3_uri=None) -> DataFrame:
        return Mock(DataFrame)

    for udf in [udf_spark_params, udf_params_spark, udf_spark, udf_params]:
        with pytest.raises(
            ValueError,
            match="Expected at least one input to the user defined function.",
        ):
            spark_arg_provider.provide_input_args(udf, fp_config)


def test_provide_input_args_with_optional_args(
    spark_arg_provider, feature_group_as_spark_df, s3_uri_as_spark_df
):
    fp_config = tdh.create_fp_config(
        inputs=[tdh.FEATURE_GROUP_DATA_SOURCE, tdh.S3_DATA_SOURCE],
        output=tdh.OUTPUT_FEATURE_GROUP_ARN,
    )

    def udf_all_optional(input_fg=None, input_s3_uri=None, params=None, spark=None) -> DataFrame:
        return Mock(DataFrame)

    def udf_no_optional(input_fg=None, input_s3_uri=None) -> DataFrame:
        return Mock(DataFrame)

    def udf_only_params(input_fg=None, input_s3_uri=None, params=None) -> DataFrame:
        return Mock(DataFrame)

    def udf_only_spark(input_fg=None, input_s3_uri=None, spark=None) -> DataFrame:
        return Mock(DataFrame)

    for udf in [udf_all_optional, udf_no_optional, udf_only_params, udf_only_spark]:
        inputs = spark_arg_provider.provide_input_args(udf, fp_config)

        assert inputs.keys() == {"input_fg", "input_s3_uri"}
        assert inputs["input_fg"] == feature_group_as_spark_df
        assert inputs["input_s3_uri"] == s3_uri_as_spark_df


def test_provide_input_arg_for_base_data_source(spark_arg_provider, params_loader, spark_session):
    fp_config = tdh.create_fp_config(inputs=[MockDataSource()], output=tdh.OUTPUT_FEATURE_GROUP_ARN)

    def udf(input_df) -> DataFrame:
        return input_df

    with patch.object(MockDataSource, "read_data", return_value=Mock(DataFrame)) as mock_read:
        spark_arg_provider.provide_input_args(udf, fp_config)
        mock_read.assert_called_with(spark=spark_session, params={"key": "value"})
        params_loader.get_parameter_args.assert_called_with(fp_config)
