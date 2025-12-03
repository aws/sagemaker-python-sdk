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
"""Tests for show_results_utils module."""
from __future__ import absolute_import

import json
import pytest
from unittest.mock import patch, MagicMock, Mock, call
from io import BytesIO

from sagemaker.train.common_utils.show_results_utils import (
    _extract_training_job_name_from_steps,
    _extract_metrics_from_results,
    _show_benchmark_results,
    _display_metrics_tables,
    _parse_prompt,
    _parse_response,
    _format_score,
    _truncate_text,
    _download_llmaj_results_from_s3,
    _display_single_llmaj_evaluation,
    _show_llmaj_results,
)


# Test constants
DEFAULT_BUCKET = "test-bucket"
DEFAULT_PREFIX = "test-prefix"
DEFAULT_JOB_NAME = "test-job-123"
DEFAULT_S3_OUTPUT = f"s3://{DEFAULT_BUCKET}/{DEFAULT_PREFIX}"


@pytest.fixture
def mock_pipeline_execution():
    """Create a mock pipeline execution."""
    execution = MagicMock()
    execution.s3_output_path = DEFAULT_S3_OUTPUT
    execution._pipeline_execution = MagicMock()
    return execution


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client."""
    with patch('boto3.client') as mock_client:
        s3_mock = MagicMock()
        mock_client.return_value = s3_mock
        yield s3_mock


class TestExtractTrainingJobName:
    """Tests for _extract_training_job_name_from_steps function."""
    
    def test_extract_with_no_pipeline_execution(self):
        """Test extraction when pipeline execution is None."""
        execution = MagicMock()
        execution._pipeline_execution = None
        
        result = _extract_training_job_name_from_steps(execution)
        assert result is None
    
    def test_extract_with_custom_model_metrics_priority(self):
        """Test that EvaluateCustomModelMetrics has highest priority."""
        execution = MagicMock()
        
        # Create mock steps
        step1 = MagicMock()
        step1.step_name = 'EvaluateBaseModelMetrics'
        step1.metadata = MagicMock()
        step1.metadata.training_job = MagicMock()
        step1.metadata.training_job.arn = 'arn:aws:sagemaker:us-west-2:123:training-job/base-job'
        
        step2 = MagicMock()
        step2.step_name = 'EvaluateCustomModelMetrics'
        step2.metadata = MagicMock()
        step2.metadata.training_job = MagicMock()
        step2.metadata.training_job.arn = 'arn:aws:sagemaker:us-west-2:123:training-job/custom-job'
        
        execution._pipeline_execution.get_all_steps.return_value = iter([step1, step2])
        
        result = _extract_training_job_name_from_steps(execution)
        assert result == 'custom-job'
    
    def test_extract_with_base_model_metrics_priority(self):
        """Test that EvaluateBaseModelMetrics has second priority."""
        execution = MagicMock()
        
        step1 = MagicMock()
        step1.step_name = 'EvaluateOtherStep'
        step1.metadata = MagicMock()
        step1.metadata.training_job = MagicMock()
        step1.metadata.training_job.arn = 'arn:aws:sagemaker:us-west-2:123:training-job/other-job'
        
        step2 = MagicMock()
        step2.step_name = 'EvaluateBaseModelMetrics'
        step2.metadata = MagicMock()
        step2.metadata.training_job = MagicMock()
        step2.metadata.training_job.arn = 'arn:aws:sagemaker:us-west-2:123:training-job/base-job'
        
        execution._pipeline_execution.get_all_steps.return_value = iter([step1, step2])
        
        result = _extract_training_job_name_from_steps(execution)
        assert result == 'base-job'
    
    def test_extract_with_custom_pattern(self):
        """Test extraction with custom step name pattern."""
        execution = MagicMock()
        
        step = MagicMock()
        step.step_name = 'CustomEvaluateStep'
        step.metadata = MagicMock()
        step.metadata.training_job = MagicMock()
        step.metadata.training_job.arn = 'arn:aws:sagemaker:us-west-2:123:training-job/custom-job'
        
        execution._pipeline_execution.get_all_steps.return_value = iter([step])
        
        result = _extract_training_job_name_from_steps(execution, 'CustomEvaluate')
        assert result == 'custom-job'
    
    def test_extract_with_no_matching_steps(self):
        """Test extraction when no steps match the pattern."""
        execution = MagicMock()
        
        step = MagicMock()
        step.step_name = 'OtherStep'
        
        execution._pipeline_execution.get_all_steps.return_value = iter([step])
        
        result = _extract_training_job_name_from_steps(execution)
        assert result is None
    
    def test_extract_with_exception(self):
        """Test extraction handles exceptions gracefully."""
        execution = MagicMock()
        execution._pipeline_execution.get_all_steps.side_effect = Exception("Test error")
        
        result = _extract_training_job_name_from_steps(execution)
        assert result is None


class TestExtractMetricsFromResults:
    """Tests for _extract_metrics_from_results function."""
    
    def test_extract_from_all_key(self):
        """Test extracting metrics from standard 'all' key."""
        results_dict = {
            'results': {
                'all': {
                    'accuracy': 0.95,
                    'f1_score': 0.92
                }
            }
        }
        
        metrics = _extract_metrics_from_results(results_dict)
        assert metrics == {'accuracy': 0.95, 'f1_score': 0.92}
    
    def test_extract_from_custom_key(self):
        """Test extracting metrics from custom task key."""
        results_dict = {
            'results': {
                'custom|gen_qa_gen_qa|0': {
                    'accuracy': 0.88,
                    'precision': 0.90
                }
            }
        }
        
        metrics = _extract_metrics_from_results(results_dict)
        assert metrics == {'accuracy': 0.88, 'precision': 0.90}
    
    def test_extract_with_empty_results(self):
        """Test extracting from empty results."""
        results_dict = {'results': {}}
        
        metrics = _extract_metrics_from_results(results_dict)
        assert metrics == {}
    
    def test_extract_with_no_results_key(self):
        """Test extracting when results key is missing."""
        results_dict = {}
        
        metrics = _extract_metrics_from_results(results_dict)
        assert metrics == {}


class TestShowBenchmarkResults:
    """Tests for _show_benchmark_results function."""
    
    @patch('sagemaker.train.common_utils.show_results_utils._display_metrics_tables')
    @patch('sagemaker.train.common_utils.show_results_utils._extract_metrics_from_results')
    @patch('sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps')
    @patch('boto3.client')
    def test_show_results_with_custom_and_base(
        self, mock_boto_client, mock_extract_job, mock_extract_metrics, mock_display, mock_pipeline_execution
    ):
        """Test showing results with both custom and base models."""
        # Setup mocks
        s3_mock = MagicMock()
        mock_boto_client.return_value = s3_mock
        
        mock_extract_job.side_effect = ['custom-job', 'base-job']
        
        # Mock S3 list_objects_v2 - return different results for each call
        s3_mock.list_objects_v2.side_effect = [
            {
                'Contents': [
                    {'Key': f'{DEFAULT_PREFIX}/custom-job/output/output/results_test.json'}
                ]
            },
            {
                'Contents': [
                    {'Key': f'{DEFAULT_PREFIX}/base-job/output/output/results_test.json'}
                ]
            }
        ]
        
        # Mock S3 get_object - return different results for each call
        results_json = json.dumps({'results': {'all': {'accuracy': 0.95}}})
        s3_mock.get_object.side_effect = [
            {'Body': BytesIO(results_json.encode('utf-8'))},
            {'Body': BytesIO(results_json.encode('utf-8'))}
        ]
        
        mock_extract_metrics.return_value = {'accuracy': 0.95}
        
        # Execute
        _show_benchmark_results(mock_pipeline_execution)
        
        # Verify
        assert mock_extract_job.call_count == 2
        assert s3_mock.list_objects_v2.call_count == 2
        mock_display.assert_called_once()
    
    @patch('boto3.client')
    def test_show_results_no_s3_output_path(self, mock_boto_client, mock_pipeline_execution):
        """Test error when s3_output_path is not set."""
        mock_pipeline_execution.s3_output_path = None
        
        with pytest.raises(ValueError, match="Cannot download results"):
            _show_benchmark_results(mock_pipeline_execution)
    
    @patch('sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps')
    @patch('boto3.client')
    def test_show_results_no_job_names(self, mock_boto_client, mock_extract_job, mock_pipeline_execution):
        """Test error when no job names can be extracted."""
        mock_extract_job.return_value = None
        
        with pytest.raises(ValueError, match="Could not extract"):
            _show_benchmark_results(mock_pipeline_execution)
    
    @patch('sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps')
    @patch('boto3.client')
    def test_show_results_no_files_found(self, mock_boto_client, mock_extract_job, mock_pipeline_execution):
        """Test error when no results files found in S3."""
        s3_mock = MagicMock()
        mock_boto_client.return_value = s3_mock
        mock_extract_job.side_effect = ['custom-job', None]
        
        s3_mock.list_objects_v2.return_value = {}
        
        with pytest.raises(FileNotFoundError, match="No files found"):
            _show_benchmark_results(mock_pipeline_execution)


class TestDisplayMetricsTables:
    """Tests for _display_metrics_tables function."""
    
    @patch('rich.console.Console')
    def test_display_custom_metrics_only(self, mock_console_class):
        """Test displaying only custom model metrics."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        custom_metrics = {'accuracy': 0.95, 'f1_score': 0.92}
        s3_paths = {'custom': 's3://bucket/custom/', 'base': None}
        
        _display_metrics_tables(custom_metrics, None, s3_paths)
        
        # Verify console.print was called
        assert mock_console.print.call_count >= 2
    
    @patch('rich.console.Console')
    def test_display_both_metrics(self, mock_console_class):
        """Test displaying both custom and base metrics."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        custom_metrics = {'accuracy': 0.95}
        base_metrics = {'accuracy': 0.88}
        s3_paths = {'custom': 's3://bucket/custom/', 'base': 's3://bucket/base/'}
        
        _display_metrics_tables(custom_metrics, base_metrics, s3_paths)
        
        assert mock_console.print.call_count >= 3
    
    @patch('IPython.get_ipython')
    @patch('rich.console.Console')
    def test_display_in_jupyter(self, mock_console_class, mock_get_ipython):
        """Test displaying in Jupyter environment."""
        # Mock Jupyter environment
        mock_ipython = MagicMock()
        mock_ipython.config = {'IPKernelApp': {}}
        mock_get_ipython.return_value = mock_ipython
        
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        custom_metrics = {'accuracy': 0.95}
        s3_paths = {'custom': 's3://bucket/custom/', 'base': None}
        
        _display_metrics_tables(custom_metrics, None, s3_paths)
        
        # Verify Console was created with force_jupyter=True
        mock_console_class.assert_called_with(force_jupyter=True)


class TestLLMAJHelperFunctions:
    """Tests for LLM As Judge helper functions."""
    
    def test_parse_prompt_valid_json(self):
        """Test parsing valid prompt JSON."""
        prompt_str = "[{'role': 'user', 'content': 'Test prompt'}]"
        result = _parse_prompt(prompt_str)
        assert result == 'Test prompt'
    
    def test_parse_prompt_invalid_json(self):
        """Test parsing invalid prompt returns original."""
        prompt_str = "Invalid JSON"
        result = _parse_prompt(prompt_str)
        assert result == "Invalid JSON"
    
    def test_parse_response_valid_json(self):
        """Test parsing valid response JSON."""
        response_str = "['Test response']"
        result = _parse_response(response_str)
        assert result == 'Test response'
    
    def test_parse_response_invalid_json(self):
        """Test parsing invalid response returns original."""
        response_str = "Invalid JSON"
        result = _parse_response(response_str)
        assert result == "Invalid JSON"
    
    def test_format_score(self):
        """Test score formatting as percentage."""
        assert _format_score(0.8333) == '83.3%'
        assert _format_score(1.0) == '100.0%'
        assert _format_score(0.0) == '0.0%'
    
    def test_truncate_text_short(self):
        """Test truncating text shorter than max length."""
        text = "Short text"
        result = _truncate_text(text, 100)
        assert result == "Short text"
    
    def test_truncate_text_long(self):
        """Test truncating text longer than max length."""
        text = "A" * 150
        result = _truncate_text(text, 100)
        assert len(result) == 100
        assert result.endswith("...")


class TestDownloadLLMAJResults:
    """Tests for _download_llmaj_results_from_s3 function."""
    
    @patch('sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps')
    @patch('boto3.client')
    def test_download_results_success(self, mock_boto_client, mock_extract_job, mock_pipeline_execution):
        """Test successful download of LLMAJ results."""
        s3_mock = MagicMock()
        mock_boto_client.return_value = s3_mock
        mock_extract_job.return_value = 'test-job'
        
        # Mock finding bedrock job name
        s3_mock.list_objects_v2.side_effect = [
            {
                'Contents': [
                    {'Key': f'{DEFAULT_PREFIX}/test-job/output/output/bedrock-job/eval_results/bedrock_llm_judge_results.json'}
                ]
            },
            {
                'Contents': [
                    {'Key': f'{DEFAULT_PREFIX}/bedrock-job/models/output_output.jsonl'}
                ]
            }
        ]
        
        # Mock JSONL content
        jsonl_content = json.dumps({'inputRecord': {}, 'modelResponses': [], 'automatedEvaluationResult': {'scores': []}})
        s3_mock.get_object.return_value = {
            'Body': BytesIO(jsonl_content.encode('utf-8'))
        }
        
        results = _download_llmaj_results_from_s3(mock_pipeline_execution)
        
        assert len(results) == 1
        assert 'inputRecord' in results[0]
    
    @patch('boto3.client')
    def test_download_results_no_s3_path(self, mock_boto_client, mock_pipeline_execution):
        """Test error when s3_output_path is not set."""
        mock_pipeline_execution.s3_output_path = None
        
        with pytest.raises(ValueError, match="Cannot download results"):
            _download_llmaj_results_from_s3(mock_pipeline_execution)
    
    @patch('sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps')
    @patch('boto3.client')
    def test_download_results_no_job_name(self, mock_boto_client, mock_extract_job, mock_pipeline_execution):
        """Test error when job name cannot be extracted."""
        mock_extract_job.return_value = None
        
        with pytest.raises(ValueError, match="Could not extract training job name"):
            _download_llmaj_results_from_s3(mock_pipeline_execution)
    
    @patch('sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps')
    @patch('boto3.client')
    def test_download_results_no_jsonl_file(self, mock_boto_client, mock_extract_job, mock_pipeline_execution):
        """Test error when JSONL file not found."""
        s3_mock = MagicMock()
        mock_boto_client.return_value = s3_mock
        mock_extract_job.return_value = 'test-job'
        
        s3_mock.list_objects_v2.return_value = {
            'Contents': [
                {'Key': f'{DEFAULT_PREFIX}/test-job/other_file.txt'}
            ]
        }
        
        with pytest.raises(FileNotFoundError, match="No _output.jsonl file found"):
            _download_llmaj_results_from_s3(mock_pipeline_execution)


class TestDisplaySingleLLMAJEvaluation:
    """Tests for _display_single_llmaj_evaluation function."""
    
    def test_display_without_explanations(self):
        """Test displaying evaluation without explanations."""
        mock_console = MagicMock()
        
        result = {
            'inputRecord': {'prompt': "[{'role': 'user', 'content': 'Test'}]"},
            'modelResponses': [{'response': "['Response']"}],
            'automatedEvaluationResult': {
                'scores': [
                    {'metricName': 'accuracy', 'result': 0.95}
                ]
            }
        }
        
        _display_single_llmaj_evaluation(result, 0, 10, mock_console, show_explanations=False)
        
        assert mock_console.print.call_count >= 3
    
    def test_display_with_explanations(self):
        """Test displaying evaluation with explanations."""
        mock_console = MagicMock()
        
        result = {
            'inputRecord': {'prompt': "[{'role': 'user', 'content': 'Test'}]"},
            'modelResponses': [{'response': "['Response']"}],
            'automatedEvaluationResult': {
                'scores': [
                    {
                        'metricName': 'accuracy',
                        'result': 0.95,
                        'evaluatorDetails': [{'explanation': 'Good result'}]
                    }
                ]
            }
        }
        
        _display_single_llmaj_evaluation(result, 0, 10, mock_console, show_explanations=True)
        
        assert mock_console.print.call_count >= 3


class TestShowLLMAJResults:
    """Tests for _show_llmaj_results function."""
    
    @patch('sagemaker.train.common_utils.show_results_utils._download_llmaj_results_from_s3')
    @patch('sagemaker.train.common_utils.show_results_utils._display_single_llmaj_evaluation')
    @patch('sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps')
    @patch('rich.console.Console')
    def test_show_results_default_pagination(
        self, mock_console_class, mock_extract_job, mock_display_single, mock_download, mock_pipeline_execution
    ):
        """Test showing results with default pagination."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        mock_extract_job.return_value = 'test-job'
        
        # Mock 10 results
        mock_results = [{'inputRecord': {}, 'modelResponses': [], 'automatedEvaluationResult': {'scores': []}}] * 10
        mock_download.return_value = mock_results
        
        _show_llmaj_results(mock_pipeline_execution, limit=5, offset=0)
        
        # Should display 5 results
        assert mock_display_single.call_count == 5
    
    @patch('sagemaker.train.common_utils.show_results_utils._download_llmaj_results_from_s3')
    @patch('sagemaker.train.common_utils.show_results_utils._display_single_llmaj_evaluation')
    @patch('sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps')
    @patch('rich.console.Console')
    def test_show_results_with_offset(
        self, mock_console_class, mock_extract_job, mock_display_single, mock_download, mock_pipeline_execution
    ):
        """Test showing results with offset."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        mock_extract_job.return_value = 'test-job'
        
        mock_results = [{'inputRecord': {}, 'modelResponses': [], 'automatedEvaluationResult': {'scores': []}}] * 10
        mock_download.return_value = mock_results
        
        _show_llmaj_results(mock_pipeline_execution, limit=3, offset=5)
        
        # Should display 3 results starting from index 5
        assert mock_display_single.call_count == 3
    
    @patch('sagemaker.train.common_utils.show_results_utils._download_llmaj_results_from_s3')
    @patch('sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps')
    @patch('rich.console.Console')
    def test_show_results_offset_beyond_total(
        self, mock_console_class, mock_extract_job, mock_download, mock_pipeline_execution
    ):
        """Test showing results when offset is beyond total."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        mock_extract_job.return_value = 'test-job'
        
        mock_results = [{'inputRecord': {}, 'modelResponses': [], 'automatedEvaluationResult': {'scores': []}}] * 5
        mock_download.return_value = mock_results
        
        _show_llmaj_results(mock_pipeline_execution, limit=5, offset=10)
        
        # Should print warning message
        assert any('beyond total' in str(call) for call in mock_console.print.call_args_list)
    
    @patch('sagemaker.train.common_utils.show_results_utils._download_llmaj_results_from_s3')
    @patch('sagemaker.train.common_utils.show_results_utils._display_single_llmaj_evaluation')
    @patch('sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps')
    @patch('rich.console.Console')
    def test_show_results_all(
        self, mock_console_class, mock_extract_job, mock_display_single, mock_download, mock_pipeline_execution
    ):
        """Test showing all results with limit=None."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        mock_extract_job.return_value = 'test-job'
        
        mock_results = [{'inputRecord': {}, 'modelResponses': [], 'automatedEvaluationResult': {'scores': []}}] * 10
        mock_download.return_value = mock_results
        
        _show_llmaj_results(mock_pipeline_execution, limit=None, offset=0)
        
        # Should display all 10 results
        assert mock_display_single.call_count == 10
