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

import sys

import pyspark
import pytest
from mock import Mock, patch

from sagemaker.mlops.feature_store.feature_processor._image_resolver import _get_spark_image_uri


@patch("sagemaker.mlops.feature_store.feature_processor._image_resolver.image_uris.retrieve")
def test_spark_33_py39(mock_retrieve):
    mock_retrieve.return_value = "123456.dkr.ecr.us-west-2.amazonaws.com/sagemaker-spark-processing:3.3-cpu-py39-v1"
    session = Mock(boto_region_name="us-west-2")
    with patch.object(pyspark, "__version__", "3.3.2"), \
         patch.object(sys, "version_info", (3, 9, 0)):
        result = _get_spark_image_uri(session)
    mock_retrieve.assert_called_once_with(
        framework="spark",
        region="us-west-2",
        version="3.3",
        py_version="py39",
        container_version="v1",
    )
    assert result == mock_retrieve.return_value


@patch("sagemaker.mlops.feature_store.feature_processor._image_resolver.image_uris.retrieve")
def test_spark_35_py39(mock_retrieve):
    mock_retrieve.return_value = "123456.dkr.ecr.us-west-2.amazonaws.com/sagemaker-spark-processing:3.5-cpu-py39-v1"
    session = Mock(boto_region_name="us-west-2")
    with patch.object(pyspark, "__version__", "3.5.1"), \
         patch.object(sys, "version_info", (3, 9, 0)):
        result = _get_spark_image_uri(session)
    mock_retrieve.assert_called_once_with(
        framework="spark",
        region="us-west-2",
        version="3.5",
        py_version="py39",
        container_version="v1",
    )
    assert result == mock_retrieve.return_value


@patch("sagemaker.mlops.feature_store.feature_processor._image_resolver.image_uris.retrieve")
def test_spark_35_py312(mock_retrieve):
    mock_retrieve.return_value = "123456.dkr.ecr.us-west-2.amazonaws.com/sagemaker-spark-processing:3.5-cpu-py312-v1"
    session = Mock(boto_region_name="us-west-2")
    with patch.object(pyspark, "__version__", "3.5.1"), \
         patch.object(sys, "version_info", (3, 12, 0)):
        result = _get_spark_image_uri(session)
    mock_retrieve.assert_called_once_with(
        framework="spark",
        region="us-west-2",
        version="3.5",
        py_version="py312",
        container_version="v1",
    )
    assert result == mock_retrieve.return_value


def test_spark_34_raises():
    session = Mock(boto_region_name="us-west-2")
    with patch.object(pyspark, "__version__", "3.4.1"), \
         patch.object(sys, "version_info", (3, 9, 0)):
        with pytest.raises(ValueError, match="No SageMaker Spark container image available for Spark 3.4"):
            _get_spark_image_uri(session)


def test_spark_35_py310_raises():
    session = Mock(boto_region_name="us-west-2")
    with patch.object(pyspark, "__version__", "3.5.1"), \
         patch.object(sys, "version_info", (3, 10, 0)):
        with pytest.raises(ValueError, match="SageMaker Spark 3.5 container images support"):
            _get_spark_image_uri(session)


def test_spark_33_py312_raises():
    session = Mock(boto_region_name="us-west-2")
    with patch.object(pyspark, "__version__", "3.3.2"), \
         patch.object(sys, "version_info", (3, 12, 0)):
        with pytest.raises(ValueError, match="SageMaker Spark 3.3 container images support"):
            _get_spark_image_uri(session)


def test_unknown_spark_version_raises():
    session = Mock(boto_region_name="us-west-2")
    with patch.object(pyspark, "__version__", "3.6.0"), \
         patch.object(sys, "version_info", (3, 9, 0)):
        with pytest.raises(ValueError, match="No SageMaker Spark container image available for Spark 3.6"):
            _get_spark_image_uri(session)
