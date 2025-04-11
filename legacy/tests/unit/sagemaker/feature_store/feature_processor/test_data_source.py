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

from pyspark.sql import DataFrame

from sagemaker.feature_store.feature_processor._data_source import PySparkDataSource


def test_pyspark_data_source():
    class TestDataSource(PySparkDataSource):

        data_source_unique_id = "test_unique_id"
        data_source_name = "test_source_name"

        def read_data(self, spark, params) -> DataFrame:
            return None

    test_data_source = TestDataSource()

    assert test_data_source.data_source_name == "test_source_name"
    assert test_data_source.data_source_unique_id == "test_unique_id"
    assert test_data_source.read_data(spark=None, params=None) is None
