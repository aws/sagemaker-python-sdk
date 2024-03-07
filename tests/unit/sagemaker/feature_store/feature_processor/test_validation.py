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

from typing import Callable
from pyspark.sql import DataFrame

import pytest

import test_data_helpers as tdh
from mock import Mock

from sagemaker.feature_store.feature_processor._validation import (
    SparkUDFSignatureValidator,
    Validator,
    ValidatorChain,
    BaseDataSourceValidator,
)
from sagemaker.feature_store.feature_processor._data_source import (
    BaseDataSource,
)


def test_validator_chain():
    fp_config = tdh.create_fp_config()
    udf = Mock(Callable)

    first_validator = Mock(Validator)
    second_validator = Mock(Validator)
    validator_chain = ValidatorChain([first_validator, second_validator])

    validator_chain.validate(udf, fp_config)

    first_validator.validate.assert_called_with(udf, fp_config)
    second_validator.validate.assert_called_with(udf, fp_config)


def test_validator_chain_validation_fails():
    fp_config = tdh.create_fp_config()
    udf = Mock(Callable)

    first_validator = Mock(validate=Mock(side_effect=ValueError()))
    second_validator = Mock(validate=Mock())
    validator_chain = ValidatorChain([first_validator, second_validator])

    with pytest.raises(ValueError):
        validator_chain.validate(udf, fp_config)


def test_spark_udf_signature_validator_valid():
    # One Input
    fp_config = tdh.create_fp_config(inputs=[tdh.FEATURE_GROUP_DATA_SOURCE])

    def one_data_source(fg_data_source, params, spark):
        return None

    SparkUDFSignatureValidator().validate(one_data_source, fp_config)

    # Two Inputs
    fp_config = tdh.create_fp_config(inputs=[tdh.FEATURE_GROUP_DATA_SOURCE, tdh.S3_DATA_SOURCE])

    def two_data_sources(fg_data_source, s3_data_source, params, spark):
        return None

    SparkUDFSignatureValidator().validate(two_data_sources, fp_config)

    # No Optional Args (params and spark)
    fp_config = tdh.create_fp_config(inputs=[tdh.FEATURE_GROUP_DATA_SOURCE, tdh.S3_DATA_SOURCE])

    def no_optional_args(fg_data_source, s3_data_source):
        return None

    SparkUDFSignatureValidator().validate(no_optional_args, fp_config)

    # Optional Args (no params)
    fp_config = tdh.create_fp_config(inputs=[tdh.FEATURE_GROUP_DATA_SOURCE, tdh.S3_DATA_SOURCE])

    def no_optional_params_arg(fg_data_source, s3_data_source, spark):
        return None

    SparkUDFSignatureValidator().validate(no_optional_params_arg, fp_config)

    # No Optional Args (no spark)
    fp_config = tdh.create_fp_config(inputs=[tdh.FEATURE_GROUP_DATA_SOURCE, tdh.S3_DATA_SOURCE])

    def no_optional_spark_arg(fg_data_source, s3_data_source, params):
        return None

    SparkUDFSignatureValidator().validate(no_optional_spark_arg, fp_config)


def test_spark_udf_signature_validator_udf_input_mismatch():
    fp_config = tdh.create_fp_config(inputs=[tdh.FEATURE_GROUP_DATA_SOURCE, tdh.S3_DATA_SOURCE])

    def one_input(one, params, spark):
        return None

    def three_inputs(one, two, three, params, spark):
        return None

    exception_string = (
        r"feature_processor expected a function with \(2\) parameter\(s\) before any"
        r" optional 'params' or 'spark' parameters for the \(2\) requested data source\(s\)\."
    )

    with pytest.raises(ValueError, match=exception_string):
        SparkUDFSignatureValidator().validate(one_input, fp_config)

    with pytest.raises(ValueError, match=exception_string):
        SparkUDFSignatureValidator().validate(three_inputs, fp_config)


def test_spark_udf_signature_validator_zero_input_params():
    def zero_inputs(params, spark):
        return None

    with pytest.raises(ValueError, match="feature_processor expects at least 1 input parameter."):
        fp_config = tdh.create_fp_config(inputs=[tdh.FEATURE_GROUP_DATA_SOURCE, tdh.S3_DATA_SOURCE])
        SparkUDFSignatureValidator().validate(zero_inputs, fp_config)


def test_spark_udf_signature_validator_udf_invalid_non_input_position():
    fp_config = tdh.create_fp_config(inputs=[tdh.FEATURE_GROUP_DATA_SOURCE, tdh.S3_DATA_SOURCE])
    with pytest.raises(
        ValueError,
        match="feature_processor expected the 'params' parameter to be the last or second last"
        " parameter after input parameters.",
    ):

        def invalid_params_position(params, fg_data_source, s3_data_source):
            return None

        SparkUDFSignatureValidator().validate(invalid_params_position, fp_config)

    with pytest.raises(
        ValueError,
        match="feature_processor expected the 'spark' parameter to be the last or second last"
        " parameter after input parameters.",
    ):

        def invalid_spark_position(spark, fg_data_source, s3_data_source):
            return None

        SparkUDFSignatureValidator().validate(invalid_spark_position, fp_config)


@pytest.mark.parametrize(
    "data_source_name, data_source_unique_id, error_pattern",
    [
        ("$_invalid_source", "unique_id", "data_source_name of input does not match pattern '.*'."),
        ("", "unique_id", "data_source_name of input does not match pattern '.*'."),
        (
            "source",
            tdh.DATA_SOURCE_UNIQUE_ID_TOO_LONG,
            "data_source_unique_id of input does not match pattern '.*'.",
        ),
        ("source", "", "data_source_unique_id of input does not match pattern '.*'."),
    ],
)
def test_spark_udf_signature_validator_udf_invalid_base_data_source(
    data_source_name, data_source_unique_id, error_pattern
):
    class TestInValidCustomDataSource(BaseDataSource):

        data_source_name = None
        data_source_unique_id = None

        def read_data(self, spark, params) -> DataFrame:
            return None

    test_data_source = TestInValidCustomDataSource()
    test_data_source.data_source_name = data_source_name
    test_data_source.data_source_unique_id = data_source_unique_id

    fp_config = tdh.create_fp_config(inputs=[test_data_source])

    def udf(input_data_source, params, spark):
        return None

    with pytest.raises(ValueError, match=error_pattern):
        BaseDataSourceValidator().validate(udf, fp_config)
