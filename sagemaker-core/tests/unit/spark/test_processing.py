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

from unittest.mock import Mock, patch

import pytest
import tempfile

from sagemaker.core.spark.processing import PySparkProcessor


@pytest.fixture
def mock_session():
    session = Mock()
    session.boto_session = Mock()
    session.boto_session.region_name = "us-west-2"
    session.boto_region_name = "us-west-2"
    session.sagemaker_client = Mock()
    session.default_bucket = Mock(return_value="test-bucket")
    session.default_bucket_prefix = "sagemaker"
    session.expand_role = Mock(side_effect=lambda x: x)
    session.sagemaker_config = {}
    return session


def _make_processor(mock_session):
    processor = PySparkProcessor(
        role="arn:aws:iam::123456789012:role/SageMakerRole",
        image_uri="test-image:latest",
        instance_count=1,
        instance_type="ml.m5.xlarge",
        sagemaker_session=mock_session,
    )
    processor._current_job_name = "test-job"
    return processor


class TestPySparkProcessorV3ProcessingInputs:
    def test_extend_processing_args_builds_v3_processing_output_for_spark_event_logs(
        self, mock_session
    ):
        processor = _make_processor(mock_session)

        _, extended_outputs = processor._extend_processing_args(
            inputs=None,
            outputs=None,
            spark_event_logs_s3_uri="s3://bucket/spark-logs",
        )

        assert len(extended_outputs) == 1
        output = extended_outputs[0]
        assert output.output_name == "spark-event-logs"
        assert output.s3_output.local_path == "/opt/ml/processing/spark-events/"
        assert output.s3_output.s3_uri == "s3://bucket/spark-logs"
        assert output.s3_output.s3_upload_mode == "Continuous"

    @patch("sagemaker.core.spark.processing.S3Uploader.upload_string_as_file_body")
    def test_stage_configuration_builds_v3_processing_input(self, mock_upload, mock_session):
        processor = _make_processor(mock_session)

        config_input = processor._stage_configuration(
            [{"Classification": "spark-defaults", "Properties": {"spark.app.name": "test"}}]
        )

        mock_upload.assert_called_once()
        assert config_input.input_name == processor._conf_container_input_name
        assert config_input.s3_input.s3_uri == (
            "s3://test-bucket/sagemaker/test-job/input/conf/configuration.json"
        )
        assert config_input.s3_input.local_path == "/opt/ml/processing/input/conf"
        assert config_input.s3_input.s3_data_type == "S3Prefix"

    @patch("sagemaker.core.spark.processing.S3Uploader.upload")
    def test_stage_submit_deps_builds_v3_processing_input_for_local_dependencies(
        self, mock_upload, mock_session, tmp_path
    ):
        processor = _make_processor(mock_session)
        dep_file = tmp_path / "dep.py"
        dep_file.write_text("print('dep')", encoding="utf-8")

        input_channel, spark_opt = processor._stage_submit_deps(
            [str(dep_file)], processor._submit_py_files_input_channel_name
        )

        mock_upload.assert_called_once()
        assert input_channel.input_name == processor._submit_py_files_input_channel_name
        assert input_channel.s3_input.s3_uri == "s3://test-bucket/sagemaker/test-job/input/py-files"
        assert input_channel.s3_input.local_path == "/opt/ml/processing/input/py-files"
        assert spark_opt == "/opt/ml/processing/input/py-files"
