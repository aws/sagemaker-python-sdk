# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the \"license\" file accompanying this file. This file is
# distributed on an \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock, mock_open
from sagemaker.core.processing import (
    Processor,
    ScriptProcessor,
    FrameworkProcessor,
    _processing_input_to_request_dict,
    _processing_output_to_request_dict,
    _get_process_request,
    logs_for_processing_job,
)
from sagemaker.core.shapes import (
    ProcessingInput,
    ProcessingOutput,
    ProcessingS3Input,
    ProcessingS3Output,
)
from sagemaker.core.network import NetworkConfig


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


class TestProcessorNormalizeArgs:
    def test_normalize_args_with_pipeline_variable_code(self, mock_session):
        from sagemaker.core.workflow.pipeline_context import PipelineSession
        from sagemaker.core.workflow import is_pipeline_variable

        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        code_var = Mock()
        with patch("sagemaker.core.processing.is_pipeline_variable", return_value=True):
            with pytest.raises(ValueError, match="code argument has to be a valid S3 URI"):
                processor._normalize_args(code=code_var)


class TestProcessorNormalizeInputs:
    def test_normalize_inputs_with_dataset_definition(self, mock_session):
        from sagemaker.core.shapes import DatasetDefinition, AthenaDatasetDefinition

        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )
        processor._current_job_name = "test-job"

        athena_def = AthenaDatasetDefinition(
            catalog="catalog",
            database="database",
            query_string="SELECT * FROM table",
            output_s3_uri="s3://bucket/output",
            output_format="PARQUET",
        )
        dataset_def = DatasetDefinition(athena_dataset_definition=athena_def)
        inputs = [ProcessingInput(input_name="data", dataset_definition=dataset_def)]

        result = processor._normalize_inputs(inputs)
        assert len(result) == 1
        assert result[0].dataset_definition == dataset_def

    def test_normalize_inputs_with_pipeline_variable_s3_uri(self, mock_session):
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )
        processor._current_job_name = "test-job"

        # Create a mock that will pass pydantic validation
        with patch("sagemaker.core.processing.is_pipeline_variable", return_value=True):
            s3_input = ProcessingS3Input(
                s3_uri="s3://bucket/input",
                local_path="/opt/ml/processing/input",
                s3_data_type="S3Prefix",
                s3_input_mode="File",
            )
            inputs = [ProcessingInput(input_name="input-1", s3_input=s3_input)]

            result = processor._normalize_inputs(inputs)
            assert len(result) == 1

    def test_normalize_inputs_with_pipeline_config(self, mock_session):
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )
        processor._current_job_name = "test-job"

        s3_input = ProcessingS3Input(
            s3_uri="/local/path",
            local_path="/opt/ml/processing/input",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
        )
        inputs = [ProcessingInput(input_name="input-1", s3_input=s3_input)]

        with patch("sagemaker.core.workflow.utilities._pipeline_config") as mock_config:
            mock_config.pipeline_name = "test-pipeline"
            mock_config.step_name = "test-step"
            with patch("sagemaker.core.s3.S3Uploader.upload", return_value="s3://bucket/uploaded"):
                result = processor._normalize_inputs(inputs)
                assert len(result) == 1

    def test_normalize_inputs_invalid_type(self, mock_session):
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )
        processor._current_job_name = "test-job"

        with pytest.raises(TypeError, match="must be provided as ProcessingInput objects"):
            processor._normalize_inputs(["invalid"])


class TestProcessorNormalizeOutputs:
    def test_normalize_outputs_with_pipeline_variable(self, mock_session):
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )
        processor._current_job_name = "test-job"

        with patch("sagemaker.core.processing.is_pipeline_variable", return_value=True):
            s3_output = ProcessingS3Output(
                s3_uri="s3://bucket/output",
                local_path="/opt/ml/processing/output",
                s3_upload_mode="EndOfJob",
            )
            outputs = [ProcessingOutput(output_name="output-1", s3_output=s3_output)]

            result = processor._normalize_outputs(outputs)
            assert len(result) == 1

    def test_normalize_outputs_with_pipeline_config(self, mock_session):
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )
        processor._current_job_name = "test-job"

        s3_output = ProcessingS3Output(
            s3_uri="/local/output",
            local_path="/opt/ml/processing/output",
            s3_upload_mode="EndOfJob",
        )
        outputs = [ProcessingOutput(output_name="output-1", s3_output=s3_output)]

        with patch("sagemaker.core.workflow.utilities._pipeline_config") as mock_config:
            mock_config.pipeline_name = "test-pipeline"
            mock_config.step_name = "test-step"
            result = processor._normalize_outputs(outputs)
            assert len(result) == 1

    def test_normalize_outputs_with_empty_bucket_prefix(self, mock_session):
        mock_session.default_bucket_prefix = None

        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )
        processor._current_job_name = "test-job"

        s3_output = ProcessingS3Output(
            s3_uri="/local/output",
            local_path="/opt/ml/processing/output",
            s3_upload_mode="EndOfJob",
        )
        outputs = [ProcessingOutput(output_name="output-1", s3_output=s3_output)]

        with patch("sagemaker.core.workflow.utilities._pipeline_config") as mock_config:
            mock_config.pipeline_name = "test-pipeline"
            mock_config.step_name = "test-step"
            result = processor._normalize_outputs(outputs)
            assert len(result) == 1

    def test_normalize_outputs_invalid_type(self, mock_session):
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )
        processor._current_job_name = "test-job"

        with pytest.raises(TypeError, match="must be provided as ProcessingOutput objects"):
            processor._normalize_outputs(["invalid"])


class TestProcessorStartNew:
    def test_start_new_with_pipeline_session(self, mock_session):
        from sagemaker.core.workflow.pipeline_context import PipelineSession

        pipeline_session = PipelineSession()
        pipeline_session.sagemaker_client = Mock()
        pipeline_session.default_bucket = Mock(return_value="test-bucket")
        pipeline_session.default_bucket_prefix = "sagemaker"
        pipeline_session.expand_role = Mock(side_effect=lambda x: x)
        pipeline_session.sagemaker_config = {}
        pipeline_session._intercept_create_request = Mock()

        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=pipeline_session,
        )

        with patch.object(
            processor,
            "_get_process_args",
            return_value={
                "job_name": "test-job",
                "inputs": [],
                "output_config": {"Outputs": []},
                "resources": {"ClusterConfig": {}},
                "stopping_condition": None,
                "app_specification": {},
                "environment": None,
                "network_config": None,
                "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
                "tags": [],
            },
        ):
            result = processor._start_new([], [], None)
            assert result is None


class TestProcessorGetProcessArgs:
    def test_get_process_args_with_stopping_condition(self, mock_session):
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            max_runtime_in_seconds=3600,
            sagemaker_session=mock_session,
        )
        processor._current_job_name = "test-job"

        args = processor._get_process_args([], [], None)
        assert args["stopping_condition"]["MaxRuntimeInSeconds"] == 3600

    def test_get_process_args_without_stopping_condition(self, mock_session):
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )
        processor._current_job_name = "test-job"

        args = processor._get_process_args([], [], None)
        assert args["stopping_condition"] is None

    def test_get_process_args_with_arguments(self, mock_session):
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )
        processor._current_job_name = "test-job"
        processor.arguments = ["--arg1", "value1"]

        args = processor._get_process_args([], [], None)
        assert args["app_specification"]["ContainerArguments"] == ["--arg1", "value1"]

    def test_get_process_args_with_entrypoint(self, mock_session):
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            entrypoint=["python", "script.py"],
            sagemaker_session=mock_session,
        )
        processor._current_job_name = "test-job"

        args = processor._get_process_args([], [], None)
        assert args["app_specification"]["ContainerEntrypoint"] == ["python", "script.py"]

    def test_get_process_args_with_network_config(self, mock_session):
        network_config = NetworkConfig(enable_network_isolation=True)

        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            network_config=network_config,
            sagemaker_session=mock_session,
        )
        processor._current_job_name = "test-job"

        args = processor._get_process_args([], [], None)
        assert args["network_config"] is not None


class TestScriptProcessor:
    def test_init_with_sklearn_image(self, mock_session):
        processor = ScriptProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="sklearn:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )
        assert processor.command == ["python3"]

    def test_get_user_code_name(self, mock_session):
        processor = ScriptProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            command=["python3"],
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        result = processor._get_user_code_name("s3://bucket/path/script.py")
        assert result == "script.py"

    def test_handle_user_code_url_s3(self, mock_session):
        processor = ScriptProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            command=["python3"],
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        result = processor._handle_user_code_url("s3://bucket/script.py")
        assert result == "s3://bucket/script.py"

    def test_handle_user_code_url_local_file(self, mock_session):
        processor = ScriptProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            command=["python3"],
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )
        processor._current_job_name = "test-job"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
            f.write("print('test')")
            temp_file = f.name

        try:
            with patch("sagemaker.core.s3.S3Uploader.upload", return_value="s3://bucket/script.py"):
                result = processor._handle_user_code_url(temp_file)
                assert result == "s3://bucket/script.py"
        finally:
            os.unlink(temp_file)

    def test_handle_user_code_url_file_not_found(self, mock_session):
        processor = ScriptProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            command=["python3"],
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        with pytest.raises(ValueError, match="wasn't found"):
            processor._handle_user_code_url("/nonexistent/file.py")

    def test_handle_user_code_url_directory(self, mock_session):
        processor = ScriptProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            command=["python3"],
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="must be a file"):
                processor._handle_user_code_url(tmpdir)

    def test_handle_user_code_url_invalid_scheme(self, mock_session):
        processor = ScriptProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            command=["python3"],
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        with pytest.raises(ValueError, match="url scheme .* is not recognized"):
            processor._handle_user_code_url("http://example.com/script.py")

    def test_upload_code_with_pipeline_config(self, mock_session):
        processor = ScriptProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            command=["python3"],
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )
        processor._current_job_name = "test-job"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
            f.write("print('test')")
            temp_file = f.name

        try:
            with patch("sagemaker.core.workflow.utilities._pipeline_config") as mock_config:
                mock_config.pipeline_name = "test-pipeline"
                mock_config.code_hash = "abc123"
                with patch("sagemaker.core.s3.S3Uploader.upload", return_value="s3://bucket/code"):
                    result = processor._upload_code(temp_file)
                    assert result == "s3://bucket/code"
        finally:
            os.unlink(temp_file)

    def test_convert_code_and_add_to_inputs(self, mock_session):
        processor = ScriptProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            command=["python3"],
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        inputs = []
        result = processor._convert_code_and_add_to_inputs(inputs, "s3://bucket/code.py")

        assert len(result) == 1
        assert result[0].input_name == "code"

    def test_set_entrypoint(self, mock_session):
        processor = ScriptProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            command=["python3"],
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        processor._set_entrypoint(["python3"], "script.py")
        assert processor.entrypoint[-1].endswith("script.py")


class TestFrameworkProcessor:
    def test_init_default_command(self, mock_session):
        processor = FrameworkProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )
        assert processor.command == ["python"]

    def test_init_with_code_location(self, mock_session):
        processor = FrameworkProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            code_location="s3://bucket/code/",
            sagemaker_session=mock_session,
        )
        assert processor.code_location == "s3://bucket/code"

    def test_patch_inputs_with_payload(self, mock_session):
        processor = FrameworkProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        inputs = []
        result = processor._patch_inputs_with_payload(inputs, "s3://bucket/code/sourcedir.tar.gz")

        assert len(result) == 1
        assert result[0].input_name == "code"

    def test_set_entrypoint_framework(self, mock_session):
        processor = FrameworkProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        processor._set_entrypoint(["python"], "runproc.sh")
        assert processor.entrypoint[0] == "/bin/bash"

    def test_generate_framework_script(self, mock_session):
        processor = FrameworkProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            command=["python3"],
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        script = processor._generate_framework_script("train.py")
        assert "#!/bin/bash" in script
        assert "train.py" in script
        assert "python3" in script

    def test_create_and_upload_runproc_with_pipeline(self, mock_session):
        processor = FrameworkProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        with patch("sagemaker.core.workflow.utilities._pipeline_config") as mock_config:
            mock_config.pipeline_name = "test-pipeline"
            with patch(
                "sagemaker.core.s3.S3Uploader.upload_string_as_file_body",
                return_value="s3://bucket/runproc.sh",
            ):
                result = processor._create_and_upload_runproc(
                    "train.py", None, "s3://bucket/runproc.sh"
                )
                assert result == "s3://bucket/runproc.sh"

    def test_create_and_upload_runproc_without_pipeline(self, mock_session):
        processor = FrameworkProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        with patch("sagemaker.core.workflow.utilities._pipeline_config", None):
            with patch(
                "sagemaker.core.s3.S3Uploader.upload_string_as_file_body",
                return_value="s3://bucket/runproc.sh",
            ):
                result = processor._create_and_upload_runproc(
                    "train.py", None, "s3://bucket/runproc.sh"
                )
                assert result == "s3://bucket/runproc.sh"


class TestHelperFunctions:
    def test_processing_input_to_request_dict(self):
        s3_input = ProcessingS3Input(
            s3_uri="s3://bucket/input",
            local_path="/opt/ml/processing/input",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
        )
        processing_input = ProcessingInput(input_name="data", s3_input=s3_input)

        result = _processing_input_to_request_dict(processing_input)

        assert result["InputName"] == "data"
        assert result["S3Input"]["S3Uri"] == "s3://bucket/input"

    def test_processing_output_to_request_dict(self):
        s3_output = ProcessingS3Output(
            s3_uri="s3://bucket/output",
            local_path="/opt/ml/processing/output",
            s3_upload_mode="EndOfJob",
        )
        processing_output = ProcessingOutput(output_name="results", s3_output=s3_output)

        result = _processing_output_to_request_dict(processing_output)

        assert result["OutputName"] == "results"
        assert result["S3Output"]["S3Uri"] == "s3://bucket/output"

    def test_get_process_request_minimal(self):
        result = _get_process_request(
            inputs=[],
            output_config={"Outputs": []},
            job_name="test-job",
            resources={"ClusterConfig": {}},
            stopping_condition=None,
            app_specification={"ImageUri": "test-image"},
            environment=None,
            network_config=None,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            tags=None,
        )

        assert result["ProcessingJobName"] == "test-job"
        assert result["RoleArn"] == "arn:aws:iam::123456789012:role/SageMakerRole"

    def test_get_process_request_with_all_params(self):
        result = _get_process_request(
            inputs=[{"InputName": "data"}],
            output_config={"Outputs": [{"OutputName": "results"}]},
            job_name="test-job",
            resources={"ClusterConfig": {}},
            stopping_condition={"MaxRuntimeInSeconds": 3600},
            app_specification={"ImageUri": "test-image"},
            environment={"KEY": "VALUE"},
            network_config={"EnableNetworkIsolation": True},
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            tags=[{"Key": "Project", "Value": "ML"}],
            experiment_config={"ExperimentName": "test-exp"},
        )

        assert result["ProcessingInputs"] == [{"InputName": "data"}]
        assert result["Environment"] == {"KEY": "VALUE"}
        assert result["ExperimentConfig"] == {"ExperimentName": "test-exp"}


class TestLogsForProcessingJob:
    def test_logs_for_processing_job(self, mock_session):
        with patch("sagemaker.core.processing._wait_until") as mock_wait:
            mock_wait.return_value = {"ProcessingJobStatus": "Completed"}

            with patch("sagemaker.core.processing._logs_init") as mock_logs_init:
                mock_logs_init.return_value = (1, [], {}, Mock(), "log-group", False, lambda x: x)

                with patch("sagemaker.core.processing._flush_log_streams"):
                    with patch("sagemaker.core.processing._get_initial_job_state") as mock_state:
                        from sagemaker.core.common_utils import LogState

                        mock_state.return_value = LogState.COMPLETE
                        logs_for_processing_job(mock_session, "test-job", wait=False, poll=1)


class TestProcessorStartNewWithSubmit:
    def test_start_new_submit_success(self, mock_session):
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )
        processor._current_job_name = "test-job"

        mock_session._intercept_create_request = Mock()

        with patch.object(
            processor,
            "_get_process_args",
            return_value={
                "job_name": "test-job",
                "inputs": [],
                "output_config": {"Outputs": []},
                "resources": {"ClusterConfig": {}},
                "stopping_condition": None,
                "app_specification": {"ImageUri": "test-image"},
                "environment": None,
                "network_config": None,
                "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
                "tags": [],
            },
        ):
            with patch("sagemaker.core.processing.serialize", return_value={}):
                with patch("sagemaker.core.processing.ProcessingJob") as mock_job:
                    with patch(
                        "sagemaker.core.utils.code_injection.codec.transform",
                        return_value={"processing_job_name": "test-job"},
                    ):
                        result = processor._start_new([], [], None)
                        assert result is not None

    def test_start_new_submit_failure(self, mock_session):
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )
        processor._current_job_name = "test-job"

        mock_session.sagemaker_client.create_processing_job = Mock(
            side_effect=Exception("API Error")
        )

        def intercept_func(request, submit_func, operation):
            if submit_func:
                submit_func(request)

        mock_session._intercept_create_request = intercept_func

        with patch.object(
            processor,
            "_get_process_args",
            return_value={
                "job_name": "test-job",
                "inputs": [],
                "output_config": {"Outputs": []},
                "resources": {"ClusterConfig": {}},
                "stopping_condition": None,
                "app_specification": {"ImageUri": "test-image"},
                "environment": None,
                "network_config": None,
                "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
                "tags": [],
            },
        ):
            with patch("sagemaker.core.processing.serialize", return_value={}):
                with pytest.raises(Exception, match="API Error"):
                    processor._start_new([], [], None)


class TestScriptProcessorRun:
    def test_run_with_wait(self, mock_session):
        processor = ScriptProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            command=["python3"],
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        mock_job = Mock()
        mock_job.wait = Mock()

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
            f.write("print('test')")
            temp_file = f.name

        try:
            with patch.object(processor, "_start_new", return_value=mock_job):
                with patch("os.path.isfile", return_value=True):
                    with patch(
                        "sagemaker.core.s3.S3Uploader.upload", return_value="s3://bucket/code.py"
                    ):
                        processor.run(code=temp_file, wait=True, logs=False)
                        mock_job.wait.assert_called_once()
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_run_without_wait(self, mock_session):
        processor = ScriptProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            command=["python3"],
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        mock_job = Mock()

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
            f.write("print('test')")
            temp_file = f.name

        try:
            with patch.object(processor, "_start_new", return_value=mock_job):
                with patch("os.path.isfile", return_value=True):
                    with patch(
                        "sagemaker.core.s3.S3Uploader.upload", return_value="s3://bucket/code.py"
                    ):
                        processor.run(code=temp_file, wait=False)
                        assert len(processor.jobs) == 1
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestFrameworkProcessorPackageCode:
    def test_package_code_with_source_dir(self, mock_session):
        processor = FrameworkProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create entry point file
            entry_point = os.path.join(tmpdir, "train.py")
            with open(entry_point, "w") as f:
                f.write("print('training')")

            result = processor._package_code(
                entry_point=entry_point,
                source_dir=tmpdir,
                requirements=None,
                job_name="test-job",
                kms_key=None,
            )
            # Check that result is an S3 URI
            assert result.startswith("s3://")
            assert "sourcedir.tar.gz" in result

    def test_package_code_without_source_dir(self, mock_session):
        processor = FrameworkProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            entry_point = os.path.join(tmpdir, "train.py")
            with open(entry_point, "w") as f:
                f.write("print('training')")

            result = processor._package_code(
                entry_point=entry_point,
                source_dir=None,
                requirements=None,
                job_name="test-job",
                kms_key=None,
            )
            # Check that result is an S3 URI
            assert result.startswith("s3://")
            assert "sourcedir.tar.gz" in result

    def test_package_code_source_dir_not_exists(self, mock_session):
        processor = FrameworkProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        with pytest.raises(ValueError, match="source_dir does not exist"):
            processor._package_code(
                entry_point="train.py",
                source_dir="/nonexistent/dir",
                requirements=None,
                job_name="test-job",
                kms_key=None,
            )


class TestFrameworkProcessorRun:
    def test_run_with_s3_code(self, mock_session):
        processor = FrameworkProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        mock_job = Mock()
        mock_job.wait = Mock()

        with patch.object(processor, "_start_new", return_value=mock_job):
            processor.run(code="s3://bucket/train.py", wait=False)
            assert processor.latest_job == mock_job

    def test_run_with_local_code(self, mock_session):
        processor = FrameworkProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        mock_job = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            entry_point = os.path.join(tmpdir, "train.py")
            with open(entry_point, "w") as f:
                f.write("print('training')")

            with patch.object(processor, "_start_new", return_value=mock_job):
                with patch.object(
                    processor, "_package_code", return_value="s3://bucket/code.tar.gz"
                ):
                    with patch(
                        "sagemaker.core.s3.S3Uploader.upload_string_as_file_body",
                        return_value="s3://bucket/runproc.sh",
                    ):
                        processor.run(code=entry_point, wait=False)
                        assert processor.latest_job == mock_job


class TestFrameworkProcessorPackAndUpload:
    def test_pack_and_upload_code_with_s3_uri(self, mock_session):
        processor = FrameworkProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        result_uri, result_inputs, result_job_name = processor._pack_and_upload_code(
            code="s3://bucket/train.py",
            source_dir=None,
            requirements=None,
            job_name=None,
            inputs=None,
            kms_key=None,
        )

        assert result_uri == "s3://bucket/train.py"

    def test_pack_and_upload_code_with_local_file(self, mock_session):
        processor = FrameworkProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            entry_point = os.path.join(tmpdir, "train.py")
            with open(entry_point, "w") as f:
                f.write("print('training')")

            with patch.object(
                processor, "_package_code", return_value="s3://bucket/code/sourcedir.tar.gz"
            ):
                with patch(
                    "sagemaker.core.s3.S3Uploader.upload_string_as_file_body",
                    return_value="s3://bucket/runproc.sh",
                ):
                    result_uri, result_inputs, result_job_name = processor._pack_and_upload_code(
                        code=entry_point,
                        source_dir=None,
                        requirements=None,
                        job_name=None,
                        inputs=None,
                        kms_key=None,
                    )

                    assert result_uri == "s3://bucket/runproc.sh"
                    assert len(result_inputs) == 1


class TestProcessingInputOutputHelpers:
    def test_processing_input_with_app_managed(self):
        s3_input = ProcessingS3Input(
            s3_uri="s3://bucket/input",
            local_path="/opt/ml/processing/input",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
        )
        processing_input = ProcessingInput(input_name="data", s3_input=s3_input, app_managed=True)

        result = _processing_input_to_request_dict(processing_input)

        assert result["AppManaged"] is True

    def test_processing_output_with_app_managed(self):
        s3_output = ProcessingS3Output(
            s3_uri="s3://bucket/output",
            local_path="/opt/ml/processing/output",
            s3_upload_mode="EndOfJob",
        )
        processing_output = ProcessingOutput(
            output_name="results", s3_output=s3_output, app_managed=True
        )

        result = _processing_output_to_request_dict(processing_output)

        assert result["AppManaged"] is True


class TestLogsForProcessingJobWait:
    def test_logs_for_processing_job_wait_true_completes(self, mock_session):
        # Test that logs_for_processing_job handles wait=True correctly
        # This is a simplified test that verifies the function can be called
        with patch("sagemaker.core.processing._wait_until") as mock_wait:
            mock_wait.return_value = {"ProcessingJobStatus": "Completed"}

            with patch("sagemaker.core.processing._logs_init") as mock_logs_init:
                mock_logs_init.return_value = (1, [], {}, Mock(), "log-group", False, lambda x: x)

                with patch("sagemaker.core.processing._flush_log_streams"):
                    with patch("sagemaker.core.processing._get_initial_job_state") as mock_state:
                        from sagemaker.core.common_utils import LogState

                        mock_state.return_value = LogState.COMPLETE

                        with patch("sagemaker.core.processing._check_job_status"):
                            # This should complete without errors
                            logs_for_processing_job(mock_session, "test-job", wait=True, poll=1)


class TestProcessorGenerateJobName:
    def test_generate_job_name_with_invalid_chars(self, mock_session):
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test/image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            base_job_name="my_job@name#test",
            sagemaker_session=mock_session,
        )

        result = processor._generate_current_job_name()

        # Should replace invalid characters with hyphens
        assert "@" not in result
        assert "#" not in result
        assert "_" not in result


class TestProcessorWithPipelineVariable:
    def test_get_process_args_with_pipeline_variable_role(self, mock_session):
        from sagemaker.core.workflow import is_pipeline_variable

        role_var = Mock()

        with patch("sagemaker.core.processing.is_pipeline_variable", return_value=True):
            processor = Processor(
                role=role_var,
                image_uri="test-image:latest",
                instance_count=1,
                instance_type="ml.m5.xlarge",
                sagemaker_session=mock_session,
            )
            processor._current_job_name = "test-job"

            args = processor._get_process_args([], [], None)
            assert args["role_arn"] == role_var


# Additional tests from test_processing_extended.py
class TestProcessorBasics:
    """Test cases for basic Processor functionality"""

    def test_init_with_minimal_params(self, mock_session):
        """Test initialization with minimal parameters"""
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        assert processor.role == "arn:aws:iam::123456789012:role/SageMakerRole"
        assert processor.image_uri == "test-image:latest"
        assert processor.instance_count == 1
        assert processor.instance_type == "ml.m5.xlarge"
        assert processor.volume_size_in_gb == 30

    def test_init_with_all_params(self, mock_session):
        """Test initialization with all parameters"""
        network_config = NetworkConfig(
            enable_network_isolation=True, security_group_ids=["sg-123"], subnets=["subnet-123"]
        )

        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=2,
            instance_type="ml.m5.2xlarge",
            entrypoint=["python", "script.py"],
            volume_size_in_gb=50,
            volume_kms_key="kms-key-123",
            output_kms_key="output-kms-key",
            max_runtime_in_seconds=7200,
            base_job_name="test-processor",
            sagemaker_session=mock_session,
            env={"KEY": "VALUE"},
            tags=[("Project", "ML")],
            network_config=network_config,
        )

        assert processor.instance_count == 2
        assert processor.volume_size_in_gb == 50
        assert processor.entrypoint == ["python", "script.py"]
        assert processor.env == {"KEY": "VALUE"}
        assert processor.network_config == network_config

    def test_init_without_role_raises_error(self, mock_session):
        """Test initialization without role raises ValueError"""
        with pytest.raises(ValueError, match="AWS IAM role is required"):
            Processor(
                image_uri="test-image:latest",
                instance_count=1,
                instance_type="ml.m5.xlarge",
                sagemaker_session=mock_session,
            )

    def test_init_with_local_instance_type(self):
        """Test initialization with local instance type"""
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="local",
        )

        from sagemaker.core.local.local_session import LocalSession

        assert isinstance(processor.sagemaker_session, LocalSession)

    def test_run_with_minimal_params(self, mock_session):
        """Test run method with minimal parameters"""
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        mock_job = Mock()
        mock_job.wait = Mock()

        with patch.object(processor, "_start_new", return_value=mock_job):
            processor.run(wait=False, logs=False)

        assert processor.latest_job == mock_job

    def test_run_with_logs_but_no_wait_raises_error(self, mock_session):
        """Test run with logs=True but wait=False raises ValueError"""
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        with pytest.raises(ValueError, match="Logs can only be shown if wait is set to True"):
            processor.run(wait=False, logs=True)

    def test_run_with_inputs_and_outputs(self, mock_session):
        """Test run method with inputs and outputs"""
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        inputs = [
            ProcessingInput(
                input_name="input-1",
                s3_input=ProcessingS3Input(
                    s3_uri="s3://bucket/input",
                    local_path="/opt/ml/processing/input",
                    s3_data_type="S3Prefix",
                    s3_input_mode="File",
                ),
            )
        ]

        outputs = [
            ProcessingOutput(
                output_name="output-1",
                s3_output=ProcessingS3Output(
                    s3_uri="s3://bucket/output",
                    local_path="/opt/ml/processing/output",
                    s3_upload_mode="EndOfJob",
                ),
            )
        ]

        mock_job = Mock()
        mock_job.wait = Mock()

        with patch.object(processor, "_start_new", return_value=mock_job):
            processor.run(inputs=inputs, outputs=outputs, wait=False, logs=False)

        assert processor.latest_job == mock_job

    def test_run_with_arguments(self, mock_session):
        """Test run method with arguments"""
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        arguments = ["--arg1", "value1", "--arg2", "value2"]

        mock_job = Mock()
        mock_job.wait = Mock()

        with patch.object(processor, "_start_new", return_value=mock_job):
            processor.run(arguments=arguments, wait=False, logs=False)

        assert processor.arguments == arguments

    def test_run_with_experiment_config(self, mock_session):
        """Test run method with experiment configuration"""
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        experiment_config = {"ExperimentName": "my-experiment", "TrialName": "my-trial"}

        mock_job = Mock()
        mock_job.wait = Mock()

        with patch.object(processor, "_start_new", return_value=mock_job):
            processor.run(experiment_config=experiment_config, wait=False, logs=False)


class TestProcessorJobTracking:
    """Test cases for Processor job tracking"""

    def test_jobs_list_updated_after_run(self, mock_session):
        """Test that jobs list is updated after run"""
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        mock_job = Mock()
        mock_job.wait = Mock()

        with patch.object(processor, "_start_new", return_value=mock_job):
            processor.run(wait=False, logs=False)

        assert len(processor.jobs) == 1
        assert processor.jobs[0] == mock_job

    def test_latest_job_updated_after_run(self, mock_session):
        """Test that latest_job is updated after run"""
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        mock_job1 = Mock()
        mock_job1.wait = Mock()
        mock_job2 = Mock()
        mock_job2.wait = Mock()

        with patch.object(processor, "_start_new", side_effect=[mock_job1, mock_job2]):
            processor.run(wait=False, logs=False)
            processor.run(wait=False, logs=False)

        assert processor.latest_job == mock_job2
        assert len(processor.jobs) == 2
