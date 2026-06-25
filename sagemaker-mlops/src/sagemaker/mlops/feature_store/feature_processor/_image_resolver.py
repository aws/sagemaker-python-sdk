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
"""Resolves SageMaker Spark container image URIs based on installed PySpark and Python versions."""
from __future__ import absolute_import

import sys

from sagemaker.core import image_uris

SPARK_IMAGE_SUPPORT_MATRIX = {
    "3.1": ["py37"],
    "3.2": ["py39"],
    "3.3": ["py39"],
    "3.5": ["py39", "py312"],
}


def _get_spark_image_uri(session):
    """Resolve the SageMaker Spark container image URI for the installed PySpark and Python versions.

    Args:
        session: SageMaker Session with boto_region_name attribute.

    Returns:
        str: The ECR image URI for the matching Spark container.

    Raises:
        ValueError: If the Spark/Python version combination is not supported.
    """
    import pyspark

    spark_version = ".".join(pyspark.__version__.split(".")[:2])
    py_version = f"py{sys.version_info[0]}{sys.version_info[1]}"

    supported_py = SPARK_IMAGE_SUPPORT_MATRIX.get(spark_version)
    if supported_py is None:
        supported = ", ".join(sorted(SPARK_IMAGE_SUPPORT_MATRIX.keys()))
        raise ValueError(
            f"No SageMaker Spark container image available for Spark {spark_version}. "
            f"Supported versions for remote execution: {supported}."
        )

    if py_version not in supported_py:
        raise ValueError(
            f"SageMaker Spark {spark_version} container images support "
            f"{', '.join(supported_py)}. Current Python version: {py_version}."
        )

    return image_uris.retrieve(
        framework="spark",
        region=session.boto_region_name,
        version=spark_version,
        py_version=py_version,
        container_version="v1",
    )
