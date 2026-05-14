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
"""Integration tests for Spark multi-version compatibility."""
from __future__ import absolute_import

import pyspark
import pytest
from mock import Mock

from sagemaker.mlops.feature_store.feature_processor._spark_factory import (
    SparkSessionFactory,
    SPARK_TO_HADOOP_MAP,
    _get_hadoop_version,
)
from sagemaker.mlops.feature_store.feature_processor._image_resolver import (
    SPARK_IMAGE_SUPPORT_MATRIX,
    _get_spark_image_uri,
)


@pytest.mark.slow_test
def test_hadoop_version_resolves_for_installed_pyspark():
    """Verify that the installed PySpark version resolves to a known Hadoop version."""
    hadoop_version = _get_hadoop_version()
    spark_major_minor = ".".join(pyspark.__version__.split(".")[:2])

    if spark_major_minor in SPARK_TO_HADOOP_MAP:
        assert hadoop_version == SPARK_TO_HADOOP_MAP[spark_major_minor]
    else:
        # Unknown version falls back to latest
        assert hadoop_version == "3.3.4"


@pytest.mark.slow_test
def test_spark_session_factory_configs_include_dynamic_hadoop():
    """Verify SparkSessionFactory produces configs with the correct Hadoop Maven coordinates."""
    env_helper = Mock()
    factory = SparkSessionFactory(env_helper)
    configs = dict(factory._get_spark_configs(is_training_job=False))

    hadoop_version = _get_hadoop_version()
    packages = configs.get("spark.jars.packages", "")
    assert f"org.apache.hadoop:hadoop-aws:{hadoop_version}" in packages
    assert f"org.apache.hadoop:hadoop-common:{hadoop_version}" in packages


@pytest.mark.slow_test
def test_image_resolver_returns_uri_for_installed_pyspark():
    """Verify the image resolver returns a valid URI for the installed PySpark + Python version."""
    import sys

    spark_major_minor = ".".join(pyspark.__version__.split(".")[:2])
    py_version = f"py{sys.version_info[0]}{sys.version_info[1]}"

    supported_py = SPARK_IMAGE_SUPPORT_MATRIX.get(spark_major_minor)
    if supported_py is None or py_version not in supported_py:
        pytest.skip(
            f"Spark {spark_major_minor} + {py_version} not in support matrix; "
            f"skipping image resolver test"
        )

    session = Mock(boto_region_name="us-west-2")
    image_uri = _get_spark_image_uri(session)

    assert "sagemaker-spark-processing" in image_uri
    assert spark_major_minor in image_uri
