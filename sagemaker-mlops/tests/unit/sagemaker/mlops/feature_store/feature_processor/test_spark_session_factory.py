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

import feature_store_pyspark
import pytest
from mock import Mock, patch, call

from sagemaker.mlops.feature_store.feature_processor._spark_factory import (
    FeatureStoreManagerFactory,
    SparkSessionFactory,
)


@pytest.fixture
def env_helper():
    return Mock(
        is_training_job=Mock(return_value=False),
        load_training_resource_config=Mock(return_value=None),
    )


def test_spark_session_factory_configuration():
    env_helper = Mock()
    spark_config = {"spark.test.key": "spark.test.value"}
    spark_session_factory = SparkSessionFactory(env_helper, spark_config)
    spark_configs = dict(spark_session_factory._get_spark_configs(is_training_job=False))
    jsc_hadoop_configs = dict(spark_session_factory._get_jsc_hadoop_configs())

    # General optimizations
    assert spark_configs.get("spark.hadoop.fs.s3a.aws.credentials.provider") == ",".join(
        [
            "com.amazonaws.auth.ContainerCredentialsProvider",
            "com.amazonaws.auth.profile.ProfileCredentialsProvider",
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
        ]
    )

    assert spark_configs.get("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version") == "2"
    assert (
        spark_configs.get("spark.hadoop.mapreduce.fileoutputcommitter.cleanup-failures.ignored")
        == "true"
    )
    assert spark_configs.get("spark.hadoop.parquet.enable.summary-metadata") == "false"

    assert spark_configs.get("spark.sql.parquet.mergeSchema") == "false"
    assert spark_configs.get("spark.sql.parquet.filterPushdown") == "true"
    assert spark_configs.get("spark.sql.hive.metastorePartitionPruning") == "true"

    assert spark_configs.get("spark.hadoop.fs.s3a.threads.max") == "500"
    assert spark_configs.get("spark.hadoop.fs.s3a.connection.maximum") == "500"
    assert spark_configs.get("spark.hadoop.fs.s3a.experimental.input.fadvise") == "normal"
    assert spark_configs.get("spark.hadoop.fs.s3a.block.size") == "128M"
    assert spark_configs.get("spark.hadoop.fs.s3a.fast.upload.buffer") == "disk"
    assert spark_configs.get("spark.hadoop.fs.trash.interval") == "0"
    assert spark_configs.get("spark.port.maxRetries") == "50"

    assert spark_configs.get("spark.test.key") == "spark.test.value"

    assert jsc_hadoop_configs.get("mapreduce.fileoutputcommitter.marksuccessfuljobs") == "false"

    # Verify configurations when not running on a training job
    assert ",".join(feature_store_pyspark.classpath_jars()) in spark_configs.get("spark.jars")
    assert ",".join(
        [
            "org.apache.hadoop:hadoop-aws:3.3.1",
            "org.apache.hadoop:hadoop-common:3.3.1",
        ]
    ) in spark_configs.get("spark.jars.packages")


def test_spark_session_factory_configuration_on_training_job():
    env_helper = Mock()
    spark_config = {"spark.test.key": "spark.test.value"}
    spark_session_factory = SparkSessionFactory(env_helper, spark_config)

    spark_config = spark_session_factory._get_spark_configs(is_training_job=True)
    assert dict(spark_config).get("spark.test.key") == "spark.test.value"

    assert all(tup[0] != "spark.jars" for tup in spark_config)
    assert all(tup[0] != "spark.jars.packages" for tup in spark_config)


@patch("pyspark.context.SparkContext.getOrCreate")
def test_spark_session_factory(mock_spark_context):
    env_helper = Mock()
    env_helper.get_instance_count.return_value = 1
    spark_session_factory = SparkSessionFactory(env_helper)

    spark_session_factory.spark_session

    _, _, kw_args = mock_spark_context.mock_calls[0]
    spark_conf = kw_args["conf"]

    mock_spark_context.assert_called_once()
    assert spark_conf.get("spark.master") == "local[*]"
    for cfg in spark_session_factory._get_spark_configs(True):
        assert spark_conf.get(cfg[0]) == cfg[1]


@patch("pyspark.context.SparkContext.getOrCreate")
def test_spark_session_factory_with_iceberg_config(mock_spark_context):
    mock_env_helper = Mock()
    mock_spark_context.side_effect = [Mock(), Mock()]

    spark_session_factory = SparkSessionFactory(mock_env_helper)

    spark_session = spark_session_factory.spark_session
    mock_conf = Mock()
    spark_session.conf = mock_conf

    spark_session_with_iceberg_config = spark_session_factory.get_spark_session_with_iceberg_config(
        "warehouse", "catalog"
    )

    assert spark_session is spark_session_with_iceberg_config
    expected_calls = [
        call.set(cfg[0], cfg[1])
        for cfg in spark_session_factory._get_iceberg_configs("warehouse", "catalog")
    ]

    mock_conf.assert_has_calls(expected_calls, any_order=False)


@patch("pyspark.context.SparkContext.getOrCreate")
def test_spark_session_factory_same_instance(mock_spark_context):
    mock_env_helper = Mock()
    mock_spark_context.side_effect = [Mock(), Mock()]

    spark_session_factory = SparkSessionFactory(mock_env_helper)

    a_reference = spark_session_factory.spark_session
    another_reference = spark_session_factory.spark_session

    assert a_reference is another_reference


@patch("feature_store_pyspark.FeatureStoreManager.FeatureStoreManager")
def test_feature_store_manager_same_instance(mock_feature_store_manager):
    mock_feature_store_manager.side_effect = [Mock(), Mock()]

    factory = FeatureStoreManagerFactory()

    assert factory.feature_store_manager is factory.feature_store_manager


def test_spark_session_factory_get_spark_session_with_iceberg_config(env_helper):
    spark_session_factory = SparkSessionFactory(env_helper)
    iceberg_configs = dict(spark_session_factory._get_iceberg_configs("s3://test/path", "Catalog"))

    assert (
        iceberg_configs.get("spark.sql.catalog.catalog")
        == "smfs.shaded.org.apache.iceberg.spark.SparkCatalog"
    )
    assert iceberg_configs.get("spark.sql.catalog.catalog.warehouse") == "s3://test/path"
    assert (
        iceberg_configs.get("spark.sql.catalog.catalog.catalog-impl")
        == "smfs.shaded.org.apache.iceberg.aws.glue.GlueCatalog"
    )
    assert (
        iceberg_configs.get("spark.sql.catalog.catalog.io-impl")
        == "smfs.shaded.org.apache.iceberg.aws.s3.S3FileIO"
    )
    assert iceberg_configs.get("spark.sql.catalog.catalog.glue.skip-name-validation") == "true"
