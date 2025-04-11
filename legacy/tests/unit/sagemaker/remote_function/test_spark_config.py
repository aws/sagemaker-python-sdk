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
from sagemaker.remote_function.spark_config import SparkConfig


def test_spark_config_default_value():
    spark_config = SparkConfig()

    assert spark_config.spark_event_logs_uri is None
    assert spark_config.submit_files is None
    assert spark_config.submit_jars is None
    assert spark_config.submit_py_files is None
    assert spark_config.configuration is None


def test_spark_config_with_invalid_spark_event_logs_uri():
    with pytest.raises(ValueError):
        SparkConfig(configuration={"invalid_key": "invalid_value"})


def test_spark_config_with_invalid_s3_uri():
    with pytest.raises(ValueError):
        SparkConfig(spark_event_logs_uri="invalid_s3_uri")


def test_spark_config_initialization():
    spark_config = SparkConfig(
        submit_files=["foo.txt"],
        submit_py_files=["bar.py"],
        submit_jars=["dummy.jar"],
        configuration={},
        spark_event_logs_uri="s3://event_logs_bucket",
    )

    assert spark_config.submit_files == ["foo.txt"]
    assert spark_config.submit_py_files == ["bar.py"]
    assert spark_config.submit_jars == ["dummy.jar"]
    assert spark_config.configuration == {}
    assert spark_config.spark_event_logs_uri == "s3://event_logs_bucket"
