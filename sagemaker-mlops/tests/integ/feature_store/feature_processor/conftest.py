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
"""Conftest for feature processor integration tests."""
import os
import tempfile

import pytest

from sagemaker.mlops.feature_store.feature_processor._spark_factory import SparkSessionFactory


@pytest.fixture(autouse=True, scope="session")
def isolate_ivy_cache():
    """Give each pytest-xdist worker its own Ivy cache to prevent concurrent cache corruption."""
    ivy_dir = os.path.join(tempfile.mkdtemp(), ".ivy2")
    original = SparkSessionFactory._get_spark_configs

    def _patched_get_spark_configs(self, is_training_job):
        configs = original(self, is_training_job)
        configs.append(("spark.jars.ivy", ivy_dir))
        return configs

    SparkSessionFactory._get_spark_configs = _patched_get_spark_configs
    yield
    SparkSessionFactory._get_spark_configs = original
