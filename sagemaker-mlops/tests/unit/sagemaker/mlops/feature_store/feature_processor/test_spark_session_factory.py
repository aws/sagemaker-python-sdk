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
import pyspark
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


@patch("feature_store_pyspark.classpath_jars", return_value=["/path/to/jar.jar"])
def test_spark_session_factory_configuration(mock_classpath_jars):
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
    assert ",".join(mock_classpath_jars.return_value) in spark_configs.get("spark.jars")
    from sagemaker.mlops.feature_store.feature_processor._spark_factory import _get_hadoop_version
    hadoop_version = _get_hadoop_version()
    assert ",".join(
        [
            f"org.apache.hadoop:hadoop-aws:{hadoop_version}",
            f"org.apache.hadoop:hadoop-common:{hadoop_version}",
        ]
    ) in spark_configs.get("spark.jars.packages")


@patch("feature_store_pyspark.classpath_jars", return_value=["/path/to/jar.jar"])
def test_spark_session_factory_configuration_on_training_job(mock_classpath_jars):
    env_helper = Mock()
    spark_config = {"spark.test.key": "spark.test.value"}
    spark_session_factory = SparkSessionFactory(env_helper, spark_config)

    spark_config = spark_session_factory._get_spark_configs(is_training_job=True)
    assert dict(spark_config).get("spark.test.key") == "spark.test.value"

    assert all(tup[0] != "spark.jars.packages" for tup in spark_config)

    # spark.jars should always be present (Feature Store JARs are always on the classpath)
    assert ",".join(mock_classpath_jars.return_value) in dict(spark_config).get("spark.jars")


@patch("feature_store_pyspark.classpath_jars", return_value=["/path/to/jar.jar"])
@patch("pyspark.context.SparkContext.getOrCreate")
def test_spark_session_factory(mock_spark_context, mock_classpath_jars):
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


@patch("feature_store_pyspark.classpath_jars", return_value=["/path/to/jar.jar"])
@patch("pyspark.context.SparkContext.getOrCreate")
def test_spark_session_factory_with_iceberg_config(mock_spark_context, mock_classpath_jars):
    mock_env_helper = Mock()
    mock_spark_context.side_effect = [Mock(), Mock()]

    spark_session_factory = SparkSessionFactory(mock_env_helper)

    spark_session = spark_session_factory.spark_session
    mock_conf = Mock()

    with patch.object(type(spark_session), "conf", new_callable=lambda: property(lambda self: mock_conf)):
        spark_session_with_iceberg_config = spark_session_factory.get_spark_session_with_iceberg_config(
            "warehouse", "catalog"
        )

        assert spark_session is spark_session_with_iceberg_config
        expected_calls = [
            call.set(cfg[0], cfg[1])
            for cfg in spark_session_factory._get_iceberg_configs("warehouse", "catalog")
        ]

        mock_conf.assert_has_calls(expected_calls, any_order=False)


@patch("feature_store_pyspark.classpath_jars", return_value=["/path/to/jar.jar"])
@patch("pyspark.context.SparkContext.getOrCreate")
def test_spark_session_factory_same_instance(mock_spark_context, mock_classpath_jars):
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


@pytest.mark.parametrize(
    "spark_version,expected_hadoop",
    [
        ("3.1.3", "3.2.0"),
        ("3.2.2", "3.3.1"),
        ("3.3.2", "3.3.2"),
        ("3.4.1", "3.3.4"),
        ("3.5.1", "3.3.4"),
    ],
)
def test_get_hadoop_version(spark_version, expected_hadoop):
    with patch.object(pyspark, "__version__", spark_version):
        from sagemaker.mlops.feature_store.feature_processor._spark_factory import _get_hadoop_version
        assert _get_hadoop_version() == expected_hadoop


def test_get_hadoop_version_unknown_falls_back():
    with patch.object(pyspark, "__version__", "3.6.0"):
        from sagemaker.mlops.feature_store.feature_processor._spark_factory import _get_hadoop_version
        assert _get_hadoop_version() == "3.3.4"


@patch("feature_store_pyspark.classpath_jars", return_value=["/path/to/jar.jar"])
def test_spark_configs_use_dynamic_hadoop_version(mock_classpath_jars):
    with patch.object(pyspark, "__version__", "3.5.1"):
        env_helper = Mock()
        factory = SparkSessionFactory(env_helper)
        configs = dict(factory._get_spark_configs(is_training_job=False))
        assert "org.apache.hadoop:hadoop-aws:3.3.4" in configs.get("spark.jars.packages")
        assert "org.apache.hadoop:hadoop-common:3.3.4" in configs.get("spark.jars.packages")


@patch("os.path.isdir", return_value=False)
def test_install_feature_store_jars_skips_when_no_target_dir(mock_isdir):
    SparkSessionFactory._install_feature_store_jars()
    mock_isdir.assert_called_once_with("/usr/lib/spark/jars")


@patch("shutil.copy")
@patch("os.path.exists", return_value=False)
@patch("os.path.isdir", return_value=True)
@patch("feature_store_pyspark.classpath_jars")
def test_install_feature_store_jars_copies_matching_jars(
    mock_classpath, mock_isdir, mock_exists, mock_copy
):
    mock_classpath.return_value = [
        "/path/to/jar-3.5-something.jar",
        "/path/to/jar-3.3-something.jar",
    ]
    with patch.object(pyspark, "__version__", "3.5.1"):
        SparkSessionFactory._install_feature_store_jars()
    mock_copy.assert_called_once_with(
        "/path/to/jar-3.5-something.jar",
        "/usr/lib/spark/jars/jar-3.5-something.jar",
    )