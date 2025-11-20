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
"""Tests for invoke_function module."""
from __future__ import absolute_import

import json
import pytest
from unittest.mock import patch, MagicMock, call

from sagemaker.train.remote_function.invoke_function import (
    _parse_args,
    _get_sagemaker_session,
    _load_run_object,
    _load_pipeline_context,
    _execute_remote_function,
    main,
    SUCCESS_EXIT_CODE,
)
from sagemaker.train.remote_function.job import KEY_EXPERIMENT_NAME, KEY_RUN_NAME


class TestParseArgs:
    """Test _parse_args function."""

    def test_parse_required_args(self):
        """Test parsing required arguments."""
        args = [
            "--region", "us-west-2",
            "--s3_base_uri", "s3://my-bucket/path",
        ]
        parsed = _parse_args(args)
        assert parsed.region == "us-west-2"
        assert parsed.s3_base_uri == "s3://my-bucket/path"

    def test_parse_all_args(self):
        """Test parsing all arguments."""
        args = [
            "--region", "us-east-1",
            "--s3_base_uri", "s3://bucket/path",
            "--s3_kms_key", "key-123",
            "--run_in_context", '{"experiment": "exp1"}',
            "--pipeline_step_name", "step1",
            "--pipeline_execution_id", "exec-123",
            "--property_references", "prop1", "val1", "prop2", "val2",
            "--serialize_output_to_json", "true",
            "--func_step_s3_dir", "s3://bucket/func",
        ]
        parsed = _parse_args(args)
        assert parsed.region == "us-east-1"
        assert parsed.s3_base_uri == "s3://bucket/path"
        assert parsed.s3_kms_key == "key-123"
        assert parsed.run_in_context == '{"experiment": "exp1"}'
        assert parsed.pipeline_step_name == "step1"
        assert parsed.pipeline_execution_id == "exec-123"
        assert parsed.property_references == ["prop1", "val1", "prop2", "val2"]
        assert parsed.serialize_output_to_json is True
        assert parsed.func_step_s3_dir == "s3://bucket/func"

    def test_parse_serialize_output_false(self):
        """Test parsing serialize_output_to_json as false."""
        args = [
            "--region", "us-west-2",
            "--s3_base_uri", "s3://bucket/path",
            "--serialize_output_to_json", "false",
        ]
        parsed = _parse_args(args)
        assert parsed.serialize_output_to_json is False

    def test_parse_default_values(self):
        """Test default values for optional arguments."""
        args = [
            "--region", "us-west-2",
            "--s3_base_uri", "s3://bucket/path",
        ]
        parsed = _parse_args(args)
        assert parsed.s3_kms_key is None
        assert parsed.run_in_context is None
        assert parsed.pipeline_step_name is None
        assert parsed.pipeline_execution_id is None
        assert parsed.property_references == []
        assert parsed.serialize_output_to_json is False
        assert parsed.func_step_s3_dir is None


class TestGetSagemakerSession:
    """Test _get_sagemaker_session function."""

    @patch("sagemaker.train.remote_function.invoke_function.boto3.session.Session")
    @patch("sagemaker.train.remote_function.invoke_function.Session")
    def test_creates_session_with_region(self, mock_session_class, mock_boto_session):
        """Test creates SageMaker session with correct region."""
        mock_boto = MagicMock()
        mock_boto_session.return_value = mock_boto
        
        _get_sagemaker_session("us-west-2")
        
        mock_boto_session.assert_called_once_with(region_name="us-west-2")
        mock_session_class.assert_called_once_with(boto_session=mock_boto)


class TestLoadRunObject:
    """Test _load_run_object function."""

    @patch("sagemaker.core.experiments.run.Run")
    def test_loads_run_from_json(self, mock_run_class):
        """Test loads Run object from JSON string."""
        run_dict = {
            KEY_EXPERIMENT_NAME: "my-experiment",
            KEY_RUN_NAME: "my-run",
        }
        run_json = json.dumps(run_dict)
        mock_session = MagicMock()
        
        _load_run_object(run_json, mock_session)
        
        mock_run_class.assert_called_once_with(
            experiment_name="my-experiment",
            run_name="my-run",
            sagemaker_session=mock_session,
        )


class TestLoadPipelineContext:
    """Test _load_pipeline_context function."""

    def test_loads_context_with_all_fields(self):
        """Test loads pipeline context with all fields."""
        args = MagicMock()
        args.pipeline_step_name = "step1"
        args.pipeline_execution_id = "exec-123"
        args.property_references = ["prop1", "val1", "prop2", "val2"]
        args.serialize_output_to_json = True
        args.func_step_s3_dir = "s3://bucket/func"
        
        context = _load_pipeline_context(args)
        
        assert context.step_name == "step1"
        assert context.execution_id == "exec-123"
        assert context.property_references == {"prop1": "val1", "prop2": "val2"}
        assert context.serialize_output_to_json is True
        assert context.func_step_s3_dir == "s3://bucket/func"

    def test_loads_context_with_empty_property_references(self):
        """Test loads pipeline context with empty property references."""
        args = MagicMock()
        args.pipeline_step_name = "step1"
        args.pipeline_execution_id = "exec-123"
        args.property_references = []
        args.serialize_output_to_json = False
        args.func_step_s3_dir = None
        
        context = _load_pipeline_context(args)
        
        assert context.property_references == {}


class TestExecuteRemoteFunction:
    """Test _execute_remote_function function."""

    @patch("sagemaker.core.remote_function.core.stored_function.StoredFunction")
    def test_executes_without_run_context(self, mock_stored_function_class):
        """Test executes stored function without run context."""
        mock_stored_func = MagicMock()
        mock_stored_function_class.return_value = mock_stored_func
        mock_session = MagicMock()
        mock_context = MagicMock()
        
        _execute_remote_function(
            sagemaker_session=mock_session,
            s3_base_uri="s3://bucket/path",
            s3_kms_key="key-123",
            run_in_context=None,
            hmac_key="hmac-key",
            context=mock_context,
        )
        
        mock_stored_function_class.assert_called_once_with(
            sagemaker_session=mock_session,
            s3_base_uri="s3://bucket/path",
            s3_kms_key="key-123",
            hmac_key="hmac-key",
            context=mock_context,
        )
        mock_stored_func.load_and_invoke.assert_called_once()

    @patch("sagemaker.train.remote_function.invoke_function._load_run_object")
    @patch("sagemaker.core.remote_function.core.stored_function.StoredFunction")
    def test_executes_with_run_context(self, mock_stored_function_class, mock_load_run):
        """Test executes stored function with run context."""
        mock_stored_func = MagicMock()
        mock_stored_function_class.return_value = mock_stored_func
        mock_run = MagicMock()
        mock_load_run.return_value = mock_run
        mock_session = MagicMock()
        mock_context = MagicMock()
        run_json = '{"experiment": "exp1"}'
        
        _execute_remote_function(
            sagemaker_session=mock_session,
            s3_base_uri="s3://bucket/path",
            s3_kms_key=None,
            run_in_context=run_json,
            hmac_key="hmac-key",
            context=mock_context,
        )
        
        # Verify run object was loaded and used as context manager
        mock_load_run.assert_called_once_with(run_json, mock_session)
        mock_run.__enter__.assert_called_once()
        mock_run.__exit__.assert_called_once()


class TestMain:
    """Test main function."""

    @patch("sagemaker.train.remote_function.invoke_function._execute_remote_function")
    @patch("sagemaker.train.remote_function.invoke_function._get_sagemaker_session")
    @patch("sagemaker.train.remote_function.invoke_function._load_pipeline_context")
    @patch("sagemaker.train.remote_function.invoke_function._parse_args")
    @patch.dict("os.environ", {"REMOTE_FUNCTION_SECRET_KEY": "test-key"})
    def test_main_success(self, mock_parse, mock_load_context, mock_get_session, mock_execute):
        """Test main function successful execution."""
        mock_args = MagicMock()
        mock_args.region = "us-west-2"
        mock_args.s3_base_uri = "s3://bucket/path"
        mock_args.s3_kms_key = None
        mock_args.run_in_context = None
        mock_parse.return_value = mock_args
        
        mock_context = MagicMock()
        mock_context.step_name = None
        mock_load_context.return_value = mock_context
        
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        
        with pytest.raises(SystemExit) as exc_info:
            main(["--region", "us-west-2", "--s3_base_uri", "s3://bucket/path"])
        
        assert exc_info.value.code == SUCCESS_EXIT_CODE
        mock_execute.assert_called_once()

    @patch("sagemaker.train.remote_function.invoke_function.handle_error")
    @patch("sagemaker.train.remote_function.invoke_function._execute_remote_function")
    @patch("sagemaker.train.remote_function.invoke_function._get_sagemaker_session")
    @patch("sagemaker.train.remote_function.invoke_function._load_pipeline_context")
    @patch("sagemaker.train.remote_function.invoke_function._parse_args")
    @patch.dict("os.environ", {"REMOTE_FUNCTION_SECRET_KEY": "test-key"})
    def test_main_handles_exception(
        self, mock_parse, mock_load_context, mock_get_session, mock_execute, mock_handle_error
    ):
        """Test main function handles exceptions."""
        mock_args = MagicMock()
        mock_args.region = "us-west-2"
        mock_args.s3_base_uri = "s3://bucket/path"
        mock_args.s3_kms_key = None
        mock_args.run_in_context = None
        mock_parse.return_value = mock_args
        
        mock_context = MagicMock()
        mock_context.step_name = None
        mock_load_context.return_value = mock_context
        
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        
        test_exception = Exception("Test error")
        mock_execute.side_effect = test_exception
        mock_handle_error.return_value = 1
        
        with pytest.raises(SystemExit) as exc_info:
            main(["--region", "us-west-2", "--s3_base_uri", "s3://bucket/path"])
        
        assert exc_info.value.code == 1
        mock_handle_error.assert_called_once()
