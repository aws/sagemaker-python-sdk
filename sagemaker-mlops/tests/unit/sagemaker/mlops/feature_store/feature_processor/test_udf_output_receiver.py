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
from feature_store_pyspark.FeatureStoreManager import FeatureStoreManager
from mock import Mock
from py4j.protocol import Py4JJavaError
from pyspark.sql import DataFrame

from sagemaker.mlops.feature_store.feature_processor import IngestionError
from sagemaker.mlops.feature_store.feature_processor._spark_factory import (
    FeatureStoreManagerFactory,
)
from sagemaker.mlops.feature_store.feature_processor._udf_output_receiver import (
    SparkOutputReceiver,
)


@pytest.fixture
def df() -> Mock:
    return Mock(DataFrame)


@pytest.fixture
def feature_store_manager():
    return Mock(FeatureStoreManager)


@pytest.fixture
def feature_store_manager_factory(feature_store_manager):
    return Mock(FeatureStoreManagerFactory, feature_store_manager=feature_store_manager)


@pytest.fixture
def spark_output_receiver(feature_store_manager_factory):
    return SparkOutputReceiver(feature_store_manager_factory)


def test_ingest_udf_output_enable_ingestion_false(df, feature_store_manager, spark_output_receiver):
    fp_config = tdh.create_fp_config(enable_ingestion=False)
    spark_output_receiver.ingest_udf_output(df, fp_config)

    feature_store_manager.ingest_data.assert_not_called()


def test_ingest_udf_output(df, feature_store_manager, spark_output_receiver):
    fp_config = tdh.create_fp_config()
    spark_output_receiver.ingest_udf_output(df, fp_config)

    feature_store_manager.ingest_data.assert_called_with(
        input_data_frame=df,
        feature_group_arn=fp_config.output,
        target_stores=fp_config.target_stores,
    )


def test_ingest_udf_output_failed_records(df, feature_store_manager, spark_output_receiver):
    fp_config = tdh.create_fp_config()

    # Simulate streaming ingestion failure.
    mock_failed_records_df = Mock()
    mock_java_exception = Mock(_target_id="")
    mock_java_exception.getClass = Mock(
        return_value=Mock(getSimpleName=Mock(return_value="StreamIngestionFailureException"))
    )

    feature_store_manager.ingest_data.side_effect = Py4JJavaError(
        msg="", java_exception=mock_java_exception
    )
    feature_store_manager.get_failed_stream_ingestion_data_frame.return_value = (
        mock_failed_records_df
    )

    with pytest.raises(IngestionError):
        spark_output_receiver.ingest_udf_output(df, fp_config)

    mock_failed_records_df.show.assert_called_with(n=20, truncate=False)


def test_ingest_udf_output_all_py4j_error_raised(df, feature_store_manager, spark_output_receiver):
    fp_config = tdh.create_fp_config()

    # Simulate ingestion failure.
    mock_java_exception = Mock(_target_id="")
    mock_java_exception.getClass = Mock(
        return_value=Mock(getSimpleName=Mock(return_value="ValidationError"))
    )
    feature_store_manager.ingest_data.side_effect = Py4JJavaError(
        msg="", java_exception=mock_java_exception
    )

    with pytest.raises(Py4JJavaError):
        spark_output_receiver.ingest_udf_output(df, fp_config)
