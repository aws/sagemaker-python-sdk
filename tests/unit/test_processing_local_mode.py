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

import os
import tempfile

import pytest
from unittest.mock import Mock, patch

from sagemaker.processing import (
    Processor,
    ScriptProcessor,
    FrameworkProcessor,
    ProcessingInput,
    ProcessingOutput,
    ProcessingS3Input,
    ProcessingS3Output,
    _processing_input_to_request_dict,
    _processing_output_to_request_dict,
    _get_process_request,
    logs_for_processing_job,
)
from sagemaker.network import NetworkConfig


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
    session.local_mode = False
    return session


class TestProcessorNormalizeArgs:
    def test_normalize_args_with_pipeline_variable_code(self, mock_session):
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        code_var = Mock()
        with patch("sagemaker.processing.is_pipeline_variable", return_value=True):
            with pytest.raises(ValueError, match="code argument has to be a valid S3 URI"):
                processor._normalize_args(code=code_var)


class TestProcessorNormalizeInputs:
    def test_normalize_inputs_with_dataset_definition(self, mock_session):
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )
        processor._current_job_name = "test-job"

        # Use a mock for dataset_definition since it's not part of the core fix
        dataset_def = Mock()
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

        with patch("sagemaker.processing.is_pipeline_variable", return_value=True):
            s3_input = ProcessingS3Input(
                s3_uri="s3://bucket/input",
                local_path="/opt/ml/processing/input",
                s3_data_type="S3Prefix",
                s3_input_mode="File",
            )
            inputs = [ProcessingInput(input_name="input-1", s3_input=s3_input)]

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

        with patch("sagemaker.processing.is_pipeline_variable", return_value=True):
            s3_output = ProcessingS3Output(
                s3_uri="s3://bucket/output",
                local_path="/opt/ml/processing/output",
                s3_upload_mode="EndOfJob",
            )
            outputs = [ProcessingOutput(output_name="output-1", s3_output=s3_output)]

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


class TestFileUriPreservedInLocalMode:
    """Tests that file:// URIs are preserved in local mode.

    **Validates: Requirements 1.1, 1.2, 2.1, 2.2**

    These tests verify the fix for issue #5562: file:// URIs should be
    preserved when the session is in local mode, rather than being replaced
    with auto-generated S3 paths.
    """

    @pytest.fixture
    def local_mock_session(self):
        session = Mock()
        session.boto_session = Mock()
        session.boto_session.region_name = "us-west-2"
        session.sagemaker_client = Mock()
        session.default_bucket = Mock(return_value="default-bucket")
        session.default_bucket_prefix = "prefix"
        session.expand_role = Mock(side_effect=lambda x: x)
        session.sagemaker_config = {}
        session.local_mode = True
        return session

    @pytest.mark.parametrize(
        "file_uri",
        [
            "file:///tmp/output",
            "file:///home/user/results",
            "file:///data/processed",
        ],
    )
    def test_normalize_outputs_preserves_file_uri_in_local_mode(self, local_mock_session, file_uri):
        """file:// URIs must be preserved when local_mode=True.

        The fix ensures that _normalize_outputs does not replace file:// URIs
        with s3:// paths when the session is in local mode.
        """
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="local",
            sagemaker_session=local_mock_session,
        )
        processor._current_job_name = "test-job"

        s3_output = ProcessingS3Output(
            s3_uri=file_uri,
            local_path="/opt/ml/processing/output",
            s3_upload_mode="EndOfJob",
        )
        outputs = [ProcessingOutput(output_name="my-output", s3_output=s3_output)]

        with patch("sagemaker.processing.workflow_utilities._pipeline_config", None):
            result = processor._normalize_outputs(outputs)

        assert len(result) == 1
        assert result[0].s3_output.s3_uri == file_uri, (
            f"Expected file:// URI to be preserved as '{file_uri}' in local mode, "
            f"but got '{result[0].s3_output.s3_uri}'"
        )


class TestPreservationNonLocalFileBehavior:
    """Preservation property tests: Non-local-file behavior must remain unchanged.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4**

    These tests capture baseline behavior. They MUST PASS on both
    unfixed and fixed code, confirming no regressions are introduced by the fix.
    """

    @pytest.fixture
    def session_local_mode_true(self):
        session = Mock()
        session.boto_session = Mock()
        session.boto_session.region_name = "us-west-2"
        session.sagemaker_client = Mock()
        session.default_bucket = Mock(return_value="default-bucket")
        session.default_bucket_prefix = "prefix"
        session.expand_role = Mock(side_effect=lambda x: x)
        session.sagemaker_config = {}
        session.local_mode = True
        return session

    @pytest.fixture
    def session_local_mode_false(self):
        session = Mock()
        session.boto_session = Mock()
        session.boto_session.region_name = "us-west-2"
        session.sagemaker_client = Mock()
        session.default_bucket = Mock(return_value="default-bucket")
        session.default_bucket_prefix = "prefix"
        session.expand_role = Mock(side_effect=lambda x: x)
        session.sagemaker_config = {}
        session.local_mode = False
        return session

    def _make_processor(self, session):
        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=session,
        )
        processor._current_job_name = "test-job"
        return processor

    # --- Requirement 3.1: S3 URIs pass through unchanged regardless of local_mode ---

    @pytest.mark.parametrize(
        "s3_uri,local_mode_fixture",
        [
            ("s3://my-bucket/path", "session_local_mode_true"),
            ("s3://my-bucket/path", "session_local_mode_false"),
            ("s3://another-bucket/deep/nested/path", "session_local_mode_true"),
            ("s3://another-bucket/deep/nested/path", "session_local_mode_false"),
        ],
    )
    def test_s3_uri_preserved_regardless_of_local_mode(self, s3_uri, local_mode_fixture, request):
        """S3 URIs must pass through unchanged regardless of local_mode setting.

        **Validates: Requirements 3.1**
        """
        session = request.getfixturevalue(local_mode_fixture)
        processor = self._make_processor(session)

        s3_output = ProcessingS3Output(
            s3_uri=s3_uri,
            local_path="/opt/ml/processing/output",
            s3_upload_mode="EndOfJob",
        )
        outputs = [ProcessingOutput(output_name="my-output", s3_output=s3_output)]

        with patch("sagemaker.processing.workflow_utilities._pipeline_config", None):
            result = processor._normalize_outputs(outputs)

        assert len(result) == 1
        assert result[0].s3_output.s3_uri == s3_uri

    # --- Requirement 3.2: Non-S3 URIs with local_mode=False replaced with S3 paths ---

    @pytest.mark.parametrize(
        "non_s3_uri",
        [
            "/local/output/path",
            "http://example.com/output",
            "ftp://server/output",
        ],
    )
    def test_non_s3_uri_replaced_when_not_local_mode(self, non_s3_uri, session_local_mode_false):
        """Non-S3 URIs in non-local sessions are replaced with auto-generated S3 paths.

        **Validates: Requirements 3.2**
        """
        processor = self._make_processor(session_local_mode_false)

        s3_output = ProcessingS3Output(
            s3_uri=non_s3_uri,
            local_path="/opt/ml/processing/output",
            s3_upload_mode="EndOfJob",
        )
        outputs = [ProcessingOutput(output_name="output-1", s3_output=s3_output)]

        with patch("sagemaker.processing.workflow_utilities._pipeline_config", None):
            result = processor._normalize_outputs(outputs)

        assert len(result) == 1
        assert result[0].s3_output.s3_uri.startswith("s3://default-bucket/")

    # --- Requirement 3.3: Pipeline variable URIs skip normalization ---

    def test_pipeline_variable_uri_skips_normalization(self, session_local_mode_false):
        """Pipeline variable URIs skip normalization entirely.

        **Validates: Requirements 3.3**
        """
        processor = self._make_processor(session_local_mode_false)

        s3_output = ProcessingS3Output(
            s3_uri="s3://bucket/output",
            local_path="/opt/ml/processing/output",
            s3_upload_mode="EndOfJob",
        )
        outputs = [ProcessingOutput(output_name="output-1", s3_output=s3_output)]

        with patch("sagemaker.processing.is_pipeline_variable", return_value=True):
            result = processor._normalize_outputs(outputs)

        assert len(result) == 1
        # Pipeline variable outputs are appended as-is without URI modification
        assert result[0].s3_output.s3_uri == "s3://bucket/output"

    # --- Requirement 3.4: Non-ProcessingOutput objects raise TypeError ---

    @pytest.mark.parametrize(
        "invalid_output",
        [
            ["a string"],
            [42],
            [{"key": "value"}],
        ],
    )
    def test_non_processing_output_raises_type_error(self, invalid_output, session_local_mode_false):
        """Non-ProcessingOutput objects must raise TypeError.

        **Validates: Requirements 3.4**
        """
        processor = self._make_processor(session_local_mode_false)

        with pytest.raises(TypeError, match="must be provided as ProcessingOutput objects"):
            processor._normalize_outputs(invalid_output)

    # --- Output name auto-generation ---

    def test_multiple_outputs_with_s3_uris_preserved(self, session_local_mode_false):
        """Multiple outputs with S3 URIs are all preserved unchanged.

        **Validates: Requirements 3.1, 3.2**
        """
        processor = self._make_processor(session_local_mode_false)

        outputs = [
            ProcessingOutput(
                output_name="first-output",
                s3_output=ProcessingS3Output(
                    s3_uri="s3://my-bucket/first",
                    local_path="/opt/ml/processing/output1",
                    s3_upload_mode="EndOfJob",
                ),
            ),
            ProcessingOutput(
                output_name="second-output",
                s3_output=ProcessingS3Output(
                    s3_uri="s3://my-bucket/second",
                    local_path="/opt/ml/processing/output2",
                    s3_upload_mode="EndOfJob",
                ),
            ),
        ]

        with patch("sagemaker.processing.workflow_utilities._pipeline_config", None):
            result = processor._normalize_outputs(outputs)

        assert len(result) == 2
        assert result[0].output_name == "first-output"
        assert result[1].output_name == "second-output"
        # S3 URIs should be preserved since they already have s3:// scheme
        assert result[0].s3_output.s3_uri == "s3://my-bucket/first"
        assert result[1].s3_output.s3_uri == "s3://my-bucket/second"


class TestProcessorLocalModeRole:
    """Tests for local mode role validation behavior.

    The implementation checks whether the session is in local mode (via
    session.local_mode attribute) OR the instance_type starts with 'local'.
    If either condition is true, role is not required.
    """

    def _make_local_mock_session(self):
        """Create a mock session that simulates local mode."""
        mock_local_session = Mock()
        mock_local_session.boto_session = Mock()
        mock_local_session.boto_session.region_name = "us-west-2"
        mock_local_session.sagemaker_client = Mock()
        mock_local_session.default_bucket = Mock(return_value="test-bucket")
        mock_local_session.default_bucket_prefix = "sagemaker"
        mock_local_session.expand_role = Mock(side_effect=lambda x: x)
        mock_local_session.sagemaker_config = {}
        mock_local_session.local_mode = True
        return mock_local_session

    def test_processor_init_without_role_in_local_mode_no_error(self):
        """Processor with instance_type='local' and no role should not raise."""
        mock_local_session = self._make_local_mock_session()

        processor = Processor(
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="local",
            sagemaker_session=mock_local_session,
        )
        assert processor.role is None
        assert processor.instance_type == "local"

    def test_processor_init_without_role_in_local_gpu_mode_no_error(self):
        """Processor with instance_type='local_gpu' and no role should not raise."""
        mock_local_session = self._make_local_mock_session()

        processor = Processor(
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="local_gpu",
            sagemaker_session=mock_local_session,
        )
        assert processor.role is None
        assert processor.instance_type == "local_gpu"

    def test_processor_init_without_role_with_local_session_no_error(self):
        """Processor with session.local_mode=True and no role should not raise.

        This tests the case where the session is in local mode (e.g., created
        externally as a LocalSession) but the instance_type is a cloud type.
        The _is_local_mode() helper checks session.local_mode in addition to
        instance_type, so role is not required when the session itself indicates
        local mode. This supports use cases where a LocalSession is passed in
        with a non-local instance_type for testing purposes.
        """
        mock_local_session = self._make_local_mock_session()

        processor = Processor(
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_local_session,
        )
        assert processor.role is None

    def test_processor_init_without_role_non_local_raises_error(self):
        """Processor with instance_type='ml.m5.xlarge' and no role should still raise ValueError."""
        mock_session = Mock()
        mock_session.boto_session = Mock()
        mock_session.boto_session.region_name = "us-west-2"
        mock_session.sagemaker_client = Mock()
        mock_session.default_bucket = Mock(return_value="test-bucket")
        mock_session.default_bucket_prefix = "sagemaker"
        mock_session.expand_role = Mock(side_effect=lambda x: x)
        mock_session.sagemaker_config = {}
        mock_session.local_mode = False

        with pytest.raises(ValueError, match="AWS IAM role is required"):
            Processor(
                image_uri="test-image:latest",
                instance_count=1,
                instance_type="ml.m5.xlarge",
                sagemaker_session=mock_session,
            )

    def test_processor_init_with_role_in_local_mode_still_works(self):
        """Processor with instance_type='local' and a valid role should still work fine."""
        mock_local_session = self._make_local_mock_session()

        processor = Processor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="local",
            sagemaker_session=mock_local_session,
        )
        assert processor.role == "arn:aws:iam::123456789012:role/SageMakerRole"
        assert processor.instance_type == "local"


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
            with patch("sagemaker.s3.S3Uploader.upload", return_value="s3://bucket/script.py"):
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
        assert "install_requirements.py" in script

    def test_create_and_upload_runproc(self, mock_session):
        processor = FrameworkProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        with patch(
            "sagemaker.s3.S3Uploader.upload_string_as_file_body",
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
        """Test initialization without role raises ValueError in non-local mode"""
        with pytest.raises(ValueError, match="AWS IAM role is required"):
            Processor(
                image_uri="test-image:latest",
                instance_count=1,
                instance_type="ml.m5.xlarge",
                sagemaker_session=mock_session,
            )

    def test_init_with_local_instance_type(self):
        """Test initialization with local instance type creates a LocalSession."""
        with patch("sagemaker.processing.LocalSession") as mock_local_session_cls:
            mock_local_session = Mock()
            mock_local_session.local_mode = True
            mock_local_session.boto_session = Mock()
            mock_local_session.boto_session.region_name = "us-west-2"
            mock_local_session.sagemaker_client = Mock()
            mock_local_session.default_bucket = Mock(return_value="test-bucket")
            mock_local_session.default_bucket_prefix = "sagemaker"
            mock_local_session.expand_role = Mock(side_effect=lambda x: x)
            mock_local_session.sagemaker_config = {}
            mock_local_session_cls.return_value = mock_local_session

            processor = Processor(
                role="arn:aws:iam::123456789012:role/SageMakerRole",
                image_uri="test-image:latest",
                instance_count=1,
                instance_type="local",
            )

            mock_local_session_cls.assert_called_once()
            assert processor.sagemaker_session == mock_local_session

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
