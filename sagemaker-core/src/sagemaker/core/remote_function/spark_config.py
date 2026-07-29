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
"""This module is used to define the Spark job config to remote function."""
from __future__ import absolute_import

from typing import Optional, List, Dict, Union
import attr
from urllib.parse import urlparse
from sagemaker.core.workflow import is_pipeline_variable


def _validate_configuration(instance, attribute, configuration):
    # pylint: disable=unused-argument
    """This is the helper method to validate the spark configuration"""
    if configuration:
        SparkConfigUtils.validate_configuration(configuration=configuration)


def _validate_s3_uri(instance, attribute, s3_uri):
    # pylint: disable=unused-argument
    """This is the helper method to validate the s3 uri"""
    if s3_uri:
        SparkConfigUtils.validate_s3_uri(s3_uri)


@attr.s(frozen=True)
class SparkConfig:
    """This is the class to initialize the spark configurations for remote function

    Attributes:
        submit_jars (Optional[List[str]]): A list which contains paths to the jars which
            are going to be submitted to Spark job. The location can be a valid s3 uri or
            local path to the jar. Defaults to ``None``.
        submit_py_files (Optional[List[str]]): A list which contains paths to the python
            files which are going to be submitted to Spark job. The location can be a
            valid s3 uri or local path to the python file. Defaults to ``None``.
        submit_files (Optional[List[str]]): A list which contains paths to the files which
            are going to be submitted to Spark job. The location can be a valid s3 uri or
            local path to the python file. Defaults to ``None``.
        configuration (list[dict] or dict): Configuration for Hadoop, Spark, or Hive.
            List or dictionary of EMR-style classifications.
            https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-configure-apps.html
        spark_event_logs_s3_uri (str): S3 path where Spark application events will
            be published to.
    """

    submit_jars: Optional[List[str]] = attr.ib(default=None)
    submit_py_files: Optional[List[str]] = attr.ib(default=None)
    submit_files: Optional[List[str]] = attr.ib(default=None)
    configuration: Optional[Union[List[Dict], Dict]] = attr.ib(
        default=None, validator=_validate_configuration
    )
    spark_event_logs_uri: Optional[str] = attr.ib(default=None, validator=_validate_s3_uri)


class SparkConfigUtils:
    """Util class for spark configurations"""

    _valid_configuration_keys = ["Classification", "Properties", "Configurations"]
    _valid_configuration_classifications = [
        "core-site",
        "hadoop-env",
        "hadoop-log4j",
        "hive-env",
        "hive-log4j",
        "hive-exec-log4j",
        "hive-site",
        "spark-defaults",
        "spark-env",
        "spark-log4j",
        "spark-hive-site",
        "spark-metrics",
        "yarn-env",
        "yarn-site",
        "export",
    ]

    @staticmethod
    def validate_configuration(configuration: Dict):
        """Validates the user-provided Hadoop/Spark/Hive configuration.

        This ensures that the list or dictionary the user provides will serialize to
        JSON matching the schema of EMR's application configuration

        Args:
            configuration (Dict): A dict that contains the configuration overrides to
                the default values. For more information, please visit:
                https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-configure-apps.html
        """
        emr_configure_apps_url = (
            "https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-configure-apps.html"
        )
        if isinstance(configuration, dict):
            keys = configuration.keys()
            if "Classification" not in keys or "Properties" not in keys:
                raise ValueError(
                    f"Missing one or more required keys in configuration dictionary "
                    f"{configuration} Please see {emr_configure_apps_url} for more information"
                )

            for key in keys:
                if key not in SparkConfigUtils._valid_configuration_keys:
                    raise ValueError(
                        f"Invalid key: {key}. "
                        f"Must be one of {SparkConfigUtils._valid_configuration_keys}. "
                        f"Please see {emr_configure_apps_url} for more information."
                    )
                if key == "Classification":
                    if (
                        configuration[key]
                        not in SparkConfigUtils._valid_configuration_classifications
                    ):
                        raise ValueError(
                            f"Invalid classification: {key}. Must be one of "
                            f"{SparkConfigUtils._valid_configuration_classifications}"
                        )

        if isinstance(configuration, list):
            for item in configuration:
                SparkConfigUtils.validate_configuration(item)

    # TODO (guoqioa@): method only checks urlparse scheme, need to perform deep s3 validation
    @staticmethod
    def validate_s3_uri(spark_output_s3_path):
        """Validate whether the URI uses an S3 scheme.

        In the future, this validation will perform deeper S3 validation.

        Args:
            spark_output_s3_path (str): The URI of the Spark output S3 Path.
        """
        if is_pipeline_variable(spark_output_s3_path):
            return

        if urlparse(spark_output_s3_path).scheme != "s3":
            raise ValueError(
                f"Invalid s3 path: {spark_output_s3_path}. Please enter something like "
                "s3://bucket-name/folder-name"
            )
