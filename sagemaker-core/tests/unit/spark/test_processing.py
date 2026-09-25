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
from unittest.mock import Mock, patch

from sagemaker.core.spark.processing import PySparkProcessor, _SparkProcessorBase
from sagemaker.core.shapes import (
    ProcessingInput,
    ProcessingOutput,
    ProcessingS3Input,
    ProcessingS3Output,
)


@pytest.fixture
def mock_session():
    session = Mock()
    session.boto_session = Mock()
    session.boto_session.region_name = "us-west-2"
    session.sagemaker_client = Mock()
    session.default_bucket = Mock(return_value="test-bucket")
    session.default_bucket_prefix = "sagemaker"
    session.expand_role = Mock(side_effect=lambda x: x)
    session.sagemaker_config = {}
    return session


@pytest.fixture
def pyspark_processor(mock_session):
    return PySparkProcessor(
        role="arn:aws:iam::123456789012:role/SageMakerRole",
        image_uri="test-spark-image:latest",
        instance_count=1,
        instance_type="ml.m5.xlarge",
        sagemaker_session=mock_session,
    )


class TestStageSubmitDepsTypeValidation:
    """#3809: a non-list submit dep must raise a clear ValueError, not iterate a string."""

    def test_string_submit_deps_raises_clear_error(self, pyspark_processor):
        with pytest.raises(ValueError, match="submit_deps must be a list"):
            pyspark_processor._stage_submit_deps("s3://bucket/my.py", "py-files")

    def test_list_submit_deps_does_not_raise_type_error(self, pyspark_processor):
        # A list of S3 URIs should be accepted and produce no input channel.
        input_channel, spark_opt = pyspark_processor._stage_submit_deps(
            ["s3://bucket/my.py"], "py-files"
        )
        assert input_channel is None
        assert spark_opt == "s3://bucket/my.py"


class TestSparkEventLogsOutputV3Shape:
    """#6253: spark_event_logs_s3_uri output must use the V3 ProcessingOutput shape."""

    def test_extend_processing_args_builds_v3_output(self, pyspark_processor):
        _, outputs = pyspark_processor._extend_processing_args(
            [], [], spark_event_logs_s3_uri="s3://bucket/spark-events/"
        )
        assert outputs is not None and len(outputs) == 1
        output = outputs[0]
        assert isinstance(output, ProcessingOutput)
        assert output.output_name == "spark-event-logs"
        assert isinstance(output.s3_output, ProcessingS3Output)
        assert output.s3_output.s3_uri == "s3://bucket/spark-events/"
        assert (
            output.s3_output.local_path == _SparkProcessorBase._spark_event_log_default_local_path
        )
        assert output.s3_output.s3_upload_mode == "Continuous"


class TestStageSubmitDepsInputV3Shape:
    """#6252: a local submit dep must build the V3 ProcessingInput shape."""

    def test_local_dep_builds_v3_input(self, pyspark_processor):
        with (
            patch("sagemaker.core.spark.processing.os.path.isfile", return_value=True),
            patch("sagemaker.core.spark.processing.shutil.copy"),
            patch("sagemaker.core.spark.processing.os.listdir", return_value=["my.py"]),
            patch("sagemaker.core.spark.processing.S3Uploader.upload"),
        ):
            input_channel, spark_opt = pyspark_processor._stage_submit_deps(
                ["/local/path/my.py"], "py-files"
            )

        assert isinstance(input_channel, ProcessingInput)
        assert input_channel.input_name == "py-files"
        assert isinstance(input_channel.s3_input, ProcessingS3Input)
        assert input_channel.s3_input.s3_data_type == "S3Prefix"
        assert input_channel.s3_input.s3_input_mode == "File"
        # The spark-submit option points at the container-local mount path.
        assert input_channel.s3_input.local_path in spark_opt
