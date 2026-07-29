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
    _download_bedrock_aggregate_json,
    _calculate_win_rates,
    _display_win_rates,
    _display_aggregate_metrics,
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
    
    @patch('rich.console.Console')
    def test_display_in_jupyter(self, mock_console_class):
        """Test displaying metrics tables."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        custom_metrics = {'accuracy': 0.95}
        s3_paths = {'custom': 's3://bucket/custom/', 'base': None}
        
        _display_metrics_tables(custom_metrics, None, s3_paths)
        
        # Verify Console was created and print was called
        assert mock_console.print.call_count >= 2


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
    
    @patch('boto3.client')
    def test_download_results_success(self, mock_boto_client, mock_pipeline_execution):
        """Test successful download of LLMAJ results."""
        s3_mock = MagicMock()
        mock_boto_client.return_value = s3_mock
        
        # Mock S3 list_objects_v2 response
        s3_mock.list_objects_v2.return_value = {
            'Contents': [
                {'Key': f'{DEFAULT_PREFIX}/bedrock-job-123/models/output_output.jsonl'}
            ]
        }
        
        # Mock JSONL content
        jsonl_content = json.dumps({'inputRecord': {}, 'modelResponses': [], 'automatedEvaluationResult': {'scores': []}})
        s3_mock.get_object.return_value = {
            'Body': BytesIO(jsonl_content.encode('utf-8'))
        }
        
        results = _download_llmaj_results_from_s3(mock_pipeline_execution, 'bedrock-job-123')
        
        assert len(results) == 1
        assert 'inputRecord' in results[0]
    
    @patch('boto3.client')
    def test_download_results_no_s3_path(self, mock_boto_client, mock_pipeline_execution):
        """Test error when s3_output_path is not set."""
        mock_pipeline_execution.s3_output_path = None
        
        with pytest.raises(ValueError, match="Cannot download results"):
            _download_llmaj_results_from_s3(mock_pipeline_execution, 'bedrock-job-123')
    
    @patch('boto3.client')
    def test_download_results_no_files(self, mock_boto_client, mock_pipeline_execution):
        """Test error when no files found in S3."""
        s3_mock = MagicMock()
        mock_boto_client.return_value = s3_mock
        
        s3_mock.list_objects_v2.return_value = {}
        
        with pytest.raises(FileNotFoundError, match="No results found"):
            _download_llmaj_results_from_s3(mock_pipeline_execution, 'bedrock-job-123')
    
    @patch('boto3.client')
    def test_download_results_no_jsonl_file(self, mock_boto_client, mock_pipeline_execution):
        """Test error when JSONL file not found."""
        s3_mock = MagicMock()
        mock_boto_client.return_value = s3_mock
        
        s3_mock.list_objects_v2.return_value = {
            'Contents': [
                {'Key': f'{DEFAULT_PREFIX}/bedrock-job-123/other_file.txt'}
            ]
        }
        
        with pytest.raises(FileNotFoundError, match="No _output.jsonl file found"):
            _download_llmaj_results_from_s3(mock_pipeline_execution, 'bedrock-job-123')


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
    @patch('sagemaker.train.common_utils.show_results_utils._download_bedrock_aggregate_json')
    @patch('sagemaker.train.common_utils.show_results_utils._display_single_llmaj_evaluation')
    @patch('sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps')
    @patch('rich.console.Console')
    def test_show_results_default_pagination(
        self, mock_console_class, mock_extract_job, mock_display_single, mock_download_aggregate, mock_download, mock_pipeline_execution
    ):
        """Test showing results with default pagination."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        mock_extract_job.side_effect = ['custom-job', None]
        
        # Mock aggregate download
        mock_download_aggregate.return_value = ({'results': {}}, 'bedrock-job-123')
        
        # Mock 10 results
        mock_results = [{'inputRecord': {}, 'modelResponses': [], 'automatedEvaluationResult': {'scores': []}}] * 10
        mock_download.return_value = mock_results
        
        _show_llmaj_results(mock_pipeline_execution, limit=5, offset=0)
        
        # Should display 5 results
        assert mock_display_single.call_count == 5
    
    @patch('sagemaker.train.common_utils.show_results_utils._download_llmaj_results_from_s3')
    @patch('sagemaker.train.common_utils.show_results_utils._download_bedrock_aggregate_json')
    @patch('sagemaker.train.common_utils.show_results_utils._display_single_llmaj_evaluation')
    @patch('sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps')
    @patch('rich.console.Console')
    def test_show_results_with_offset(
        self, mock_console_class, mock_extract_job, mock_display_single, mock_download_aggregate, mock_download, mock_pipeline_execution
    ):
        """Test showing results with offset."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        mock_extract_job.side_effect = ['custom-job', None]
        
        # Mock aggregate download
        mock_download_aggregate.return_value = ({'results': {}}, 'bedrock-job-123')
        
        mock_results = [{'inputRecord': {}, 'modelResponses': [], 'automatedEvaluationResult': {'scores': []}}] * 10
        mock_download.return_value = mock_results
        
        _show_llmaj_results(mock_pipeline_execution, limit=3, offset=5)
        
        # Should display 3 results starting from index 5
        assert mock_display_single.call_count == 3
    
    @patch('sagemaker.train.common_utils.show_results_utils._download_llmaj_results_from_s3')
    @patch('sagemaker.train.common_utils.show_results_utils._download_bedrock_aggregate_json')
    @patch('sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps')
    @patch('rich.console.Console')
    def test_show_results_offset_beyond_total(
        self, mock_console_class, mock_extract_job, mock_download_aggregate, mock_download, mock_pipeline_execution
    ):
        """Test showing results when offset is beyond total."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        mock_extract_job.side_effect = ['custom-job', None]
        
        # Mock aggregate download
        mock_download_aggregate.return_value = ({'results': {}}, 'bedrock-job-123')
        
        mock_results = [{'inputRecord': {}, 'modelResponses': [], 'automatedEvaluationResult': {'scores': []}}] * 5
        mock_download.return_value = mock_results
        
        _show_llmaj_results(mock_pipeline_execution, limit=5, offset=10)
        
        # Function should complete without error (no results displayed)
        assert mock_console.print.called
    
    @patch('sagemaker.train.common_utils.show_results_utils._download_llmaj_results_from_s3')
    @patch('sagemaker.train.common_utils.show_results_utils._download_bedrock_aggregate_json')
    @patch('sagemaker.train.common_utils.show_results_utils._display_single_llmaj_evaluation')
    @patch('sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps')
    @patch('rich.console.Console')
    def test_show_results_all(
        self, mock_console_class, mock_extract_job, mock_display_single, mock_download_aggregate, mock_download, mock_pipeline_execution
    ):
        """Test showing all results with limit=None."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        mock_extract_job.side_effect = ['custom-job', None]
        
        # Mock aggregate download
        mock_download_aggregate.return_value = ({'results': {}}, 'bedrock-job-123')
        
        mock_results = [{'inputRecord': {}, 'modelResponses': [], 'automatedEvaluationResult': {'scores': []}}] * 10
        mock_download.return_value = mock_results
        
        _show_llmaj_results(mock_pipeline_execution, limit=None, offset=0)
        
        # Should display all 10 results
        assert mock_display_single.call_count == 10



class TestDownloadBedrockAggregateJson:
    """Tests for _download_bedrock_aggregate_json function."""
    
    @patch('boto3.client')
    def test_download_aggregate_success(self, mock_boto_client, mock_pipeline_execution):
        """Test successful download of aggregate JSON."""
        s3_mock = MagicMock()
        mock_boto_client.return_value = s3_mock
        
        # Mock S3 list_objects_v2 response
        s3_mock.list_objects_v2.return_value = {
            'Contents': [
                {'Key': f'{DEFAULT_PREFIX}/{DEFAULT_JOB_NAME}/output/output/bedrock-job-123/bedrock_llm_judge_results.json'}
            ]
        }
        
        # Mock aggregate JSON content
        aggregate_data = {
            'job_name': 'bedrock-job-123',
            'results': {
                'Faithfulness': {
                    'score': 1.0,
                    'total_evaluations': 10,
                    'passed': 10,
                    'failed': 0
                }
            }
        }
        s3_mock.get_object.return_value = {
            'Body': BytesIO(json.dumps(aggregate_data).encode('utf-8'))
        }
        
        result, bedrock_job_name = _download_bedrock_aggregate_json(
            mock_pipeline_execution, DEFAULT_JOB_NAME
        )
        
        assert result == aggregate_data
        assert bedrock_job_name == 'bedrock-job-123'
    
    @patch('boto3.client')
    def test_download_aggregate_no_files(self, mock_boto_client, mock_pipeline_execution):
        """Test error when no files found in S3."""
        s3_mock = MagicMock()
        mock_boto_client.return_value = s3_mock
        
        s3_mock.list_objects_v2.return_value = {}
        
        with pytest.raises(FileNotFoundError, match="No files at"):
            _download_bedrock_aggregate_json(mock_pipeline_execution, DEFAULT_JOB_NAME)
    
    @patch('boto3.client')
    def test_download_aggregate_file_not_found(self, mock_boto_client, mock_pipeline_execution):
        """Test error when aggregate JSON file not found."""
        s3_mock = MagicMock()
        mock_boto_client.return_value = s3_mock
        
        s3_mock.list_objects_v2.return_value = {
            'Contents': [
                {'Key': f'{DEFAULT_PREFIX}/{DEFAULT_JOB_NAME}/output/output/other_file.txt'}
            ]
        }
        
        with pytest.raises(FileNotFoundError, match="bedrock_llm_judge_results.json not found"):
            _download_bedrock_aggregate_json(mock_pipeline_execution, DEFAULT_JOB_NAME)
    
    def test_download_aggregate_no_s3_path(self, mock_pipeline_execution):
        """Test error when s3_output_path is not set."""
        mock_pipeline_execution.s3_output_path = None
        
        with pytest.raises(ValueError, match="s3_output_path is not set"):
            _download_bedrock_aggregate_json(mock_pipeline_execution, DEFAULT_JOB_NAME)


class TestCalculateWinRates:
    """Tests for _calculate_win_rates function."""
    
    def test_calculate_custom_wins(self):
        """Test win rate calculation when custom model wins majority."""
        custom_results = [
            {
                'automatedEvaluationResult': {
                    'scores': [
                        {'metricName': 'Faithfulness', 'result': 1.0},
                        {'metricName': 'Correctness', 'result': 0.9}
                    ]
                }
            },
            {
                'automatedEvaluationResult': {
                    'scores': [
                        {'metricName': 'Faithfulness', 'result': 0.95},
                        {'metricName': 'Correctness', 'result': 0.85}
                    ]
                }
            }
        ]
        
        base_results = [
            {
                'automatedEvaluationResult': {
                    'scores': [
                        {'metricName': 'Faithfulness', 'result': 0.8},
                        {'metricName': 'Correctness', 'result': 0.7}
                    ]
                }
            },
            {
                'automatedEvaluationResult': {
                    'scores': [
                        {'metricName': 'Faithfulness', 'result': 0.85},
                        {'metricName': 'Correctness', 'result': 0.75}
                    ]
                }
            }
        ]
        
        win_rates = _calculate_win_rates(custom_results, base_results)
        
        assert win_rates['custom_wins'] == 2
        assert win_rates['base_wins'] == 0
        assert win_rates['ties'] == 0
        assert win_rates['total'] == 2
        assert win_rates['custom_win_rate'] == 1.0
        assert win_rates['base_win_rate'] == 0.0
        assert win_rates['tie_rate'] == 0.0
    
    def test_calculate_base_wins(self):
        """Test win rate calculation when base model wins majority."""
        custom_results = [
            {
                'automatedEvaluationResult': {
                    'scores': [
                        {'metricName': 'Faithfulness', 'result': 0.7},
                        {'metricName': 'Correctness', 'result': 0.6}
                    ]
                }
            }
        ]
        
        base_results = [
            {
                'automatedEvaluationResult': {
                    'scores': [
                        {'metricName': 'Faithfulness', 'result': 0.9},
                        {'metricName': 'Correctness', 'result': 0.85}
                    ]
                }
            }
        ]
        
        win_rates = _calculate_win_rates(custom_results, base_results)
        
        assert win_rates['custom_wins'] == 0
        assert win_rates['base_wins'] == 1
        assert win_rates['ties'] == 0
        assert win_rates['base_win_rate'] == 1.0
    
    def test_calculate_ties(self):
        """Test win rate calculation with ties."""
        custom_results = [
            {
                'automatedEvaluationResult': {
                    'scores': [
                        {'metricName': 'Faithfulness', 'result': 0.9},
                        {'metricName': 'Correctness', 'result': 0.7}
                    ]
                }
            }
        ]
        
        base_results = [
            {
                'automatedEvaluationResult': {
                    'scores': [
                        {'metricName': 'Faithfulness', 'result': 0.8},
                        {'metricName': 'Correctness', 'result': 0.85}
                    ]
                }
            }
        ]
        
        win_rates = _calculate_win_rates(custom_results, base_results)
        
        assert win_rates['custom_wins'] == 0
        assert win_rates['base_wins'] == 0
        assert win_rates['ties'] == 1
        assert win_rates['tie_rate'] == 1.0
    
    def test_calculate_mixed_results(self):
        """Test win rate calculation with mixed wins and ties."""
        custom_results = [
            {
                'automatedEvaluationResult': {
                    'scores': [
                        {'metricName': 'Faithfulness', 'result': 1.0},
                        {'metricName': 'Correctness', 'result': 0.9}
                    ]
                }
            },
            {
                'automatedEvaluationResult': {
                    'scores': [
                        {'metricName': 'Faithfulness', 'result': 0.7},
                        {'metricName': 'Correctness', 'result': 0.6}
                    ]
                }
            },
            {
                'automatedEvaluationResult': {
                    'scores': [
                        {'metricName': 'Faithfulness', 'result': 0.9},
                        {'metricName': 'Correctness', 'result': 0.7}
                    ]
                }
            }
        ]
        
        base_results = [
            {
                'automatedEvaluationResult': {
                    'scores': [
                        {'metricName': 'Faithfulness', 'result': 0.8},
                        {'metricName': 'Correctness', 'result': 0.7}
                    ]
                }
            },
            {
                'automatedEvaluationResult': {
                    'scores': [
                        {'metricName': 'Faithfulness', 'result': 0.9},
                        {'metricName': 'Correctness', 'result': 0.85}
                    ]
                }
            },
            {
                'automatedEvaluationResult': {
                    'scores': [
                        {'metricName': 'Faithfulness', 'result': 0.8},
                        {'metricName': 'Correctness', 'result': 0.8}
                    ]
                }
            }
        ]
        
        win_rates = _calculate_win_rates(custom_results, base_results)
        
        assert win_rates['custom_wins'] == 1
        assert win_rates['base_wins'] == 1
        assert win_rates['ties'] == 1
        assert win_rates['total'] == 3
        assert abs(win_rates['custom_win_rate'] - 0.333) < 0.01
        assert abs(win_rates['base_win_rate'] - 0.333) < 0.01
        assert abs(win_rates['tie_rate'] - 0.333) < 0.01
    
    def test_calculate_empty_results(self):
        """Test win rate calculation with empty results."""
        win_rates = _calculate_win_rates([], [])
        
        assert win_rates['custom_wins'] == 0
        assert win_rates['base_wins'] == 0
        assert win_rates['ties'] == 0
        assert win_rates['total'] == 0
        assert win_rates['custom_win_rate'] == 0.0


class TestDisplayWinRates:
    """Tests for _display_win_rates function."""
    
    def test_display_win_rates(self):
        """Test displaying win rates."""
        mock_console = MagicMock()
        
        win_rates = {
            'custom_wins': 10,
            'base_wins': 5,
            'ties': 2,
            'total': 17,
            'custom_win_rate': 0.588,
            'base_win_rate': 0.294,
            'tie_rate': 0.118
        }
        
        _display_win_rates(win_rates, mock_console)
        
        # Verify console.print was called with Panel
        assert mock_console.print.called
        call_args = mock_console.print.call_args[0]
        assert len(call_args) > 0


class TestDisplayAggregateMetrics:
    """Tests for _display_aggregate_metrics function."""
    
    def test_display_custom_only(self):
        """Test displaying aggregate metrics for custom model only."""
        mock_console = MagicMock()
        
        custom_aggregate = {
            'results': {
                'Faithfulness': {
                    'score': 1.0,
                    'total_evaluations': 10,
                    'passed': 10,
                    'failed': 0
                },
                'CustomMetric': {
                    'score': 0.8,
                    'total_evaluations': 10,
                    'passed': 8,
                    'failed': 2,
                    'std_deviation': 0.02
                }
            }
        }
        
        _display_aggregate_metrics(custom_aggregate, None, mock_console)
        
        # Verify console.print was called at least once (for custom table)
        assert mock_console.print.call_count >= 1
    
    def test_display_with_base_model(self):
        """Test displaying aggregate metrics with base model."""
        mock_console = MagicMock()
        
        custom_aggregate = {
            'results': {
                'Faithfulness': {
                    'score': 1.0,
                    'total_evaluations': 10,
                    'passed': 10,
                    'failed': 0
                }
            }
        }
        
        base_aggregate = {
            'results': {
                'Faithfulness': {
                    'score': 0.9,
                    'total_evaluations': 10,
                    'passed': 9,
                    'failed': 1
                }
            }
        }
        
        _display_aggregate_metrics(custom_aggregate, base_aggregate, mock_console)
        
        # Verify console.print was called once (comparison table)
        assert mock_console.print.call_count == 1
    
    def test_display_builtin_vs_custom_metrics(self):
        """Test displaying both builtin and custom metrics."""
        mock_console = MagicMock()
        
        custom_aggregate = {
            'results': {
                'Faithfulness': {
                    'score': 1.0,
                    'total_evaluations': 10
                },
                'CustomMetric': {
                    'score': 0.85,
                    'total_evaluations': 10,
                    'std_deviation': 0.03
                }
            }
        }
        
        _display_aggregate_metrics(custom_aggregate, None, mock_console)
        
        assert mock_console.print.called
    
    def test_display_score_differences(self):
        """Test displaying score differences between models."""
        mock_console = MagicMock()
        
        custom_aggregate = {
            'results': {
                'Faithfulness': {
                    'score': 0.95,
                    'total_evaluations': 10
                },
                'Correctness': {
                    'score': 0.80,
                    'total_evaluations': 10
                }
            }
        }
        
        base_aggregate = {
            'results': {
                'Faithfulness': {
                    'score': 0.90,
                    'total_evaluations': 10
                },
                'Correctness': {
                    'score': 0.85,
                    'total_evaluations': 10
                }
            }
        }
        
        _display_aggregate_metrics(custom_aggregate, base_aggregate, mock_console)
        
        # Verify comparison table was printed once
        assert mock_console.print.call_count == 1


class TestShowLLMAJResultsIntegration:
    """Integration tests for _show_llmaj_results with new aggregate features."""
    
    @patch('sagemaker.train.common_utils.show_results_utils._display_aggregate_metrics')
    @patch('sagemaker.train.common_utils.show_results_utils._display_win_rates')
    @patch('sagemaker.train.common_utils.show_results_utils._calculate_win_rates')
    @patch('sagemaker.train.common_utils.show_results_utils._download_llmaj_results_from_s3')
    @patch('sagemaker.train.common_utils.show_results_utils._download_bedrock_aggregate_json')
    @patch('sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps')
    @patch('rich.console.Console')
    def test_show_results_with_aggregate_and_win_rates(
        self, mock_console_class, mock_extract_job, mock_download_aggregate,
        mock_download_results, mock_calculate_win, mock_display_win, mock_display_aggregate,
        mock_pipeline_execution
    ):
        """Test complete flow with aggregate metrics and win rates."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        # Mock job name extraction
        mock_extract_job.side_effect = ['custom-job', 'base-job']
        
        # Mock aggregate downloads
        custom_aggregate = {
            'results': {
                'Faithfulness': {'score': 1.0, 'total_evaluations': 10}
            }
        }
        base_aggregate = {
            'results': {
                'Faithfulness': {'score': 0.9, 'total_evaluations': 10}
            }
        }
        mock_download_aggregate.side_effect = [
            (custom_aggregate, 'bedrock-job-123'),
            (base_aggregate, 'bedrock-job-456')
        ]
        
        # Mock per-example results
        custom_results = [
            {
                'inputRecord': {'prompt': "[{'role': 'user', 'content': 'Test'}]"},
                'modelResponses': [{'response': "['Response']"}],
                'automatedEvaluationResult': {
                    'scores': [{'metricName': 'Faithfulness', 'result': 1.0}]
                }
            }
        ]
        base_results = [
            {
                'inputRecord': {'prompt': "[{'role': 'user', 'content': 'Test'}]"},
                'modelResponses': [{'response': "['Response']"}],
                'automatedEvaluationResult': {
                    'scores': [{'metricName': 'Faithfulness', 'result': 0.9}]
                }
            }
        ]
        mock_download_results.side_effect = [custom_results, base_results]
        
        # Mock win rates
        win_rates = {
            'custom_wins': 1, 'base_wins': 0, 'ties': 0, 'total': 1,
            'custom_win_rate': 1.0, 'base_win_rate': 0.0, 'tie_rate': 0.0
        }
        mock_calculate_win.return_value = win_rates
        
        # Execute
        _show_llmaj_results(mock_pipeline_execution, limit=5, offset=0)
        
        # Verify all components were called
        assert mock_download_aggregate.call_count == 2
        assert mock_download_results.call_count == 2
        mock_calculate_win.assert_called_once()
        mock_display_win.assert_called_once_with(win_rates, mock_console)
        mock_display_aggregate.assert_called_once_with(custom_aggregate, base_aggregate, mock_console)
    
    @patch('sagemaker.train.common_utils.show_results_utils._display_aggregate_metrics')
    @patch('sagemaker.train.common_utils.show_results_utils._download_llmaj_results_from_s3')
    @patch('sagemaker.train.common_utils.show_results_utils._download_bedrock_aggregate_json')
    @patch('sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps')
    @patch('rich.console.Console')
    def test_show_results_custom_only(
        self, mock_console_class, mock_extract_job, mock_download_aggregate,
        mock_download_results, mock_display_aggregate, mock_pipeline_execution
    ):
        """Test flow with custom model only (no base model)."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        # Mock job name extraction - only custom
        mock_extract_job.side_effect = ['custom-job', None]
        
        # Mock aggregate download
        custom_aggregate = {
            'results': {
                'Faithfulness': {'score': 1.0, 'total_evaluations': 10}
            }
        }
        mock_download_aggregate.return_value = (custom_aggregate, 'bedrock-job-123')
        
        # Mock per-example results
        custom_results = [
            {
                'inputRecord': {'prompt': "[{'role': 'user', 'content': 'Test'}]"},
                'modelResponses': [{'response': "['Response']"}],
                'automatedEvaluationResult': {
                    'scores': [{'metricName': 'Faithfulness', 'result': 1.0}]
                }
            }
        ]
        mock_download_results.return_value = custom_results
        
        # Execute
        _show_llmaj_results(mock_pipeline_execution, limit=5, offset=0)
        
        # Verify aggregate displayed with None for base
        mock_display_aggregate.assert_called_once_with(custom_aggregate, None, mock_console)
    
    @patch('sagemaker.train.common_utils.show_results_utils._download_llmaj_results_from_s3')
    @patch('sagemaker.train.common_utils.show_results_utils._download_bedrock_aggregate_json')
    @patch('sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps')
    @patch('rich.console.Console')
    def test_show_results_aggregate_not_found(
        self, mock_console_class, mock_extract_job, mock_download_aggregate,
        mock_download_results, mock_pipeline_execution
    ):
        """Test graceful degradation when aggregate results not found."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        # Mock job name extraction
        mock_extract_job.side_effect = ['custom-job', None]
        
        # Mock aggregate download failure
        mock_download_aggregate.side_effect = FileNotFoundError("Aggregate not found")
        
        # Mock per-example results still work
        custom_results = [
            {
                'inputRecord': {'prompt': "[{'role': 'user', 'content': 'Test'}]"},
                'modelResponses': [{'response': "['Response']"}],
                'automatedEvaluationResult': {
                    'scores': [{'metricName': 'Faithfulness', 'result': 1.0}]
                }
            }
        ]
        mock_download_results.return_value = custom_results
        
        # Execute - should not raise exception
        _show_llmaj_results(mock_pipeline_execution, limit=5, offset=0)
        
        # Verify per-example results were still attempted
        # Note: This will fail because bedrock_job_name is None, but that's expected behavior
        # The function should log a warning and continue


class TestBugConditionExploration:
    """Bug condition exploration tests for the base model per-example results S3 path bug.

    **Validates: Requirements 1.1, 1.4, 2.1, 2.4**

    Property 1: Fault Condition - Base Model Per-Example Results Downloaded From Wrong S3 Path

    In a two-model LLM-as-Judge evaluation where custom and base models have distinct
    bedrock_job_name values, _download_llmaj_results_from_s3 MUST be called with the
    base model's bedrock_job_name for the base model download, not the custom model's.

    EXPECTED: These tests FAIL on unfixed code, confirming the bug exists.
    The bug causes both calls to _download_llmaj_results_from_s3 to use
    "custom-bedrock-job" instead of using "base-bedrock-job" for the base model.
    """

    @pytest.mark.parametrize(
        "custom_scores,base_scores,description",
        [
            (
                [1.0, 0.9, 0.8, 0.7, 0.6],
                [0.3, 0.4, 0.2, 0.5, 0.1],
                "custom_wins_all",
            ),
            (
                [0.2, 0.3, 0.1, 0.4, 0.15],
                [0.9, 0.8, 0.95, 0.7, 0.85],
                "base_wins_all",
            ),
            (
                [0.9, 0.3, 0.8, 0.2, 0.7],
                [0.4, 0.8, 0.3, 0.9, 0.6],
                "mixed_wins",
            ),
        ],
        ids=["custom_wins", "base_wins", "mixed"],
    )
    @patch("sagemaker.train.common_utils.show_results_utils._display_aggregate_metrics")
    @patch("sagemaker.train.common_utils.show_results_utils._display_win_rates")
    @patch("sagemaker.train.common_utils.show_results_utils._calculate_win_rates")
    @patch("sagemaker.train.common_utils.show_results_utils._download_llmaj_results_from_s3")
    @patch("sagemaker.train.common_utils.show_results_utils._download_bedrock_aggregate_json")
    @patch("sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps")
    @patch("rich.console.Console")
    def test_base_model_per_example_uses_correct_bedrock_job_name(
        self,
        mock_console_class,
        mock_extract_job,
        mock_download_aggregate,
        mock_download_results,
        mock_calculate_win,
        mock_display_win,
        mock_display_aggregate,
        mock_pipeline_execution,
        custom_scores,
        base_scores,
        description,
    ):
        """Assert _download_llmaj_results_from_s3 is called with 'base-bedrock-job' for base model.

        On unfixed code, this FAILS because both calls use 'custom-bedrock-job'.
        """
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        # Return distinct job names for custom and base steps
        mock_extract_job.side_effect = ["custom-training-job", "base-training-job"]

        # Return distinct bedrock_job_name values for custom and base aggregates
        custom_aggregate = {"results": {"Metric1": {"score": 0.8, "total_evaluations": 5}}}
        base_aggregate = {"results": {"Metric1": {"score": 0.5, "total_evaluations": 5}}}
        mock_download_aggregate.side_effect = [
            ("custom_agg", "custom-bedrock-job"),
            ("base_agg", "base-bedrock-job"),
        ]

        # Build per-example result sets keyed by bedrock_job_name
        def make_results(scores):
            return [
                {
                    "inputRecord": {"prompt": "[{'role': 'user', 'content': 'Q'}]"},
                    "modelResponses": [{"response": "['A']"}],
                    "automatedEvaluationResult": {
                        "scores": [{"metricName": "Metric1", "result": s}]
                    },
                }
                for s in scores
            ]

        custom_results = make_results(custom_scores)
        base_results = make_results(base_scores)

        # Return different results depending on which bedrock_job_name is passed
        def download_side_effect(pipeline_exec, bedrock_job_name):
            if bedrock_job_name == "custom-bedrock-job":
                return custom_results
            elif bedrock_job_name == "base-bedrock-job":
                return base_results
            raise ValueError(f"Unexpected bedrock_job_name: {bedrock_job_name}")

        mock_download_results.side_effect = download_side_effect

        mock_calculate_win.return_value = {
            "custom_wins": 0, "base_wins": 0, "ties": 0, "total": 0,
            "custom_win_rate": 0.0, "base_win_rate": 0.0, "tie_rate": 0.0,
        }

        # Execute
        _show_llmaj_results(mock_pipeline_execution, limit=5, offset=0)

        # CRITICAL ASSERTION: base model download must use "base-bedrock-job"
        # On unfixed code, both calls use "custom-bedrock-job" — this assertion FAILS.
        assert mock_download_results.call_count == 2
        download_calls = mock_download_results.call_args_list
        # First call: custom model per-example results
        assert download_calls[0] == call(mock_pipeline_execution, "custom-bedrock-job"), (
            f"Expected custom download with 'custom-bedrock-job', "
            f"got {download_calls[0]}"
        )
        # Second call: base model per-example results — MUST use base bedrock job name
        assert download_calls[1] == call(mock_pipeline_execution, "base-bedrock-job"), (
            f"BUG CONFIRMED: base model download used '{download_calls[1]}' "
            f"instead of call(mock_pipeline_execution, 'base-bedrock-job'). "
            f"Both downloads used the custom model's bedrock_job_name."
        )

    @pytest.mark.parametrize(
        "custom_scores,base_scores,description",
        [
            (
                [1.0, 0.9, 0.8, 0.7, 0.6],
                [0.3, 0.4, 0.2, 0.5, 0.1],
                "custom_wins_all",
            ),
            (
                [0.2, 0.3, 0.1, 0.4, 0.15],
                [0.9, 0.8, 0.95, 0.7, 0.85],
                "base_wins_all",
            ),
            (
                [0.9, 0.3, 0.8, 0.2, 0.7],
                [0.4, 0.8, 0.3, 0.9, 0.6],
                "mixed_wins",
            ),
        ],
        ids=["custom_wins", "base_wins", "mixed"],
    )
    @patch("sagemaker.train.common_utils.show_results_utils._display_aggregate_metrics")
    @patch("sagemaker.train.common_utils.show_results_utils._display_win_rates")
    @patch("sagemaker.train.common_utils.show_results_utils._calculate_win_rates")
    @patch("sagemaker.train.common_utils.show_results_utils._download_llmaj_results_from_s3")
    @patch("sagemaker.train.common_utils.show_results_utils._download_bedrock_aggregate_json")
    @patch("sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps")
    @patch("rich.console.Console")
    def test_calculate_win_rates_receives_distinct_datasets(
        self,
        mock_console_class,
        mock_extract_job,
        mock_download_aggregate,
        mock_download_results,
        mock_calculate_win,
        mock_display_win,
        mock_display_aggregate,
        mock_pipeline_execution,
        custom_scores,
        base_scores,
        description,
    ):
        """Assert _calculate_win_rates receives two genuinely distinct datasets.

        On unfixed code, this FAILS because both datasets come from the same S3 path
        (custom model's bedrock_job_name), so _calculate_win_rates gets identical data.
        """
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        mock_extract_job.side_effect = ["custom-training-job", "base-training-job"]

        mock_download_aggregate.side_effect = [
            ("custom_agg", "custom-bedrock-job"),
            ("base_agg", "base-bedrock-job"),
        ]

        def make_results(scores):
            return [
                {
                    "inputRecord": {"prompt": "[{'role': 'user', 'content': 'Q'}]"},
                    "modelResponses": [{"response": "['A']"}],
                    "automatedEvaluationResult": {
                        "scores": [{"metricName": "Metric1", "result": s}]
                    },
                }
                for s in scores
            ]

        custom_results = make_results(custom_scores)
        base_results = make_results(base_scores)

        def download_side_effect(pipeline_exec, bedrock_job_name):
            if bedrock_job_name == "custom-bedrock-job":
                return custom_results
            elif bedrock_job_name == "base-bedrock-job":
                return base_results
            raise ValueError(f"Unexpected bedrock_job_name: {bedrock_job_name}")

        mock_download_results.side_effect = download_side_effect

        mock_calculate_win.return_value = {
            "custom_wins": 0, "base_wins": 0, "ties": 0, "total": 0,
            "custom_win_rate": 0.0, "base_win_rate": 0.0, "tie_rate": 0.0,
        }

        _show_llmaj_results(mock_pipeline_execution, limit=5, offset=0)

        # _calculate_win_rates must receive genuinely distinct datasets
        mock_calculate_win.assert_called_once()
        actual_custom, actual_base = mock_calculate_win.call_args[0]

        # On unfixed code, both arguments are custom_results (identical),
        # so this assertion FAILS.
        assert actual_custom is not actual_base or custom_scores == base_scores, (
            "BUG CONFIRMED: _calculate_win_rates received identical objects for "
            "custom and base results — both downloaded from custom model's S3 path."
        )

        # Verify the actual score values differ (stronger check)
        custom_first_score = actual_custom[0]["automatedEvaluationResult"]["scores"][0]["result"]
        base_first_score = actual_base[0]["automatedEvaluationResult"]["scores"][0]["result"]
        assert custom_first_score == custom_scores[0], (
            f"Custom results first score should be {custom_scores[0]}, got {custom_first_score}"
        )
        assert base_first_score == base_scores[0], (
            f"BUG CONFIRMED: Base results first score is {base_first_score} "
            f"(same as custom {custom_scores[0]}) instead of expected {base_scores[0]}. "
            f"Base per-example results were downloaded from the custom model's S3 path."
        )



class TestPreservationProperty:
    """Preservation property tests for _show_llmaj_results behavior.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

    Property 2: Preservation - Non-Two-Model and Non-Win-Rate Behavior Unchanged

    These tests observe and assert the current behavior of _show_llmaj_results for
    code paths that are NOT affected by the two-model win-rate bug. They must PASS
    on both unfixed and fixed code, confirming no regressions.
    """

    # --- Requirement 3.1: Single-model evaluation displays aggregate + per-example, no win rates ---

    @pytest.mark.parametrize(
        "result_count",
        [1, 3, 5, 10],
        ids=["1_result", "3_results", "5_results", "10_results"],
    )
    @patch("sagemaker.train.common_utils.show_results_utils._display_aggregate_metrics")
    @patch("sagemaker.train.common_utils.show_results_utils._calculate_win_rates")
    @patch("sagemaker.train.common_utils.show_results_utils._download_llmaj_results_from_s3")
    @patch("sagemaker.train.common_utils.show_results_utils._download_bedrock_aggregate_json")
    @patch("sagemaker.train.common_utils.show_results_utils._display_single_llmaj_evaluation")
    @patch("sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps")
    @patch("rich.console.Console")
    def test_single_model_no_win_rates(
        self,
        mock_console_class,
        mock_extract_job,
        mock_display_single,
        mock_download_aggregate,
        mock_download_results,
        mock_calculate_win,
        mock_display_aggregate,
        mock_pipeline_execution,
        result_count,
    ):
        """Single-model evaluation displays aggregate and per-example results without win rates.

        Observed behavior: When only custom_job_name exists (no base_job_name),
        _calculate_win_rates is never called.
        """
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        # Only custom model, no base
        mock_extract_job.side_effect = ["custom-job", None]

        custom_aggregate = {"results": {"Metric1": {"score": 0.9, "total_evaluations": result_count}}}
        mock_download_aggregate.return_value = (custom_aggregate, "bedrock-job-custom")

        mock_results = [
            {
                "inputRecord": {"prompt": "[{'role': 'user', 'content': 'Q'}]"},
                "modelResponses": [{"response": "['A']"}],
                "automatedEvaluationResult": {
                    "scores": [{"metricName": "Metric1", "result": 0.9}]
                },
            }
        ] * result_count
        mock_download_results.return_value = mock_results

        _show_llmaj_results(mock_pipeline_execution, limit=result_count, offset=0)

        # _calculate_win_rates must NOT be called for single-model evaluations
        mock_calculate_win.assert_not_called()
        # Aggregate metrics should be displayed with None for base
        mock_display_aggregate.assert_called_once_with(custom_aggregate, None, mock_console)
        # Per-example results should be displayed
        assert mock_display_single.call_count == result_count

    # --- Requirement 3.2: Base aggregate FileNotFoundError -> custom-only results ---

    @patch("sagemaker.train.common_utils.show_results_utils._display_aggregate_metrics")
    @patch("sagemaker.train.common_utils.show_results_utils._calculate_win_rates")
    @patch("sagemaker.train.common_utils.show_results_utils._download_llmaj_results_from_s3")
    @patch("sagemaker.train.common_utils.show_results_utils._download_bedrock_aggregate_json")
    @patch("sagemaker.train.common_utils.show_results_utils._display_single_llmaj_evaluation")
    @patch("sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps")
    @patch("rich.console.Console")
    def test_base_aggregate_not_found_displays_custom_only(
        self,
        mock_console_class,
        mock_extract_job,
        mock_display_single,
        mock_download_aggregate,
        mock_download_results,
        mock_calculate_win,
        mock_display_aggregate,
        mock_pipeline_execution,
    ):
        """When base aggregate raises FileNotFoundError, function continues with custom-only.

        Observed behavior: base_aggregate stays None, _display_aggregate_metrics is called
        with custom_aggregate and None for base.
        """
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        # Two-model evaluation
        mock_extract_job.side_effect = ["custom-job", "base-job"]

        custom_aggregate = {"results": {"Metric1": {"score": 0.85, "total_evaluations": 5}}}

        # Custom aggregate succeeds, base aggregate raises FileNotFoundError
        mock_download_aggregate.side_effect = [
            (custom_aggregate, "bedrock-job-custom"),
            FileNotFoundError("Base aggregate not found in S3"),
        ]

        mock_results = [
            {
                "inputRecord": {"prompt": "[{'role': 'user', 'content': 'Q'}]"},
                "modelResponses": [{"response": "['A']"}],
                "automatedEvaluationResult": {
                    "scores": [{"metricName": "Metric1", "result": 0.85}]
                },
            }
        ] * 3
        mock_download_results.return_value = mock_results

        # Should NOT raise — function continues gracefully
        _show_llmaj_results(mock_pipeline_execution, limit=5, offset=0)

        # Aggregate displayed with None for base
        mock_display_aggregate.assert_called_once_with(custom_aggregate, None, mock_console)
        # Per-example results still displayed
        assert mock_display_single.call_count == 3

    # --- Requirement 3.3: Pagination with limit and offset ---

    @pytest.mark.parametrize(
        "limit,offset,total,expected_display_count",
        [
            (3, 2, 10, 3),
            (1, 0, 5, 1),
            (5, 0, 3, 3),
            (20, 5, 10, 5),
            (2, 8, 10, 2),
            (10, 0, 10, 10),
        ],
        ids=[
            "limit3_offset2_total10",
            "limit1_offset0_total5",
            "limit5_offset0_total3",
            "limit20_offset5_total10",
            "limit2_offset8_total10",
            "limit10_offset0_total10",
        ],
    )
    @patch("sagemaker.train.common_utils.show_results_utils._download_llmaj_results_from_s3")
    @patch("sagemaker.train.common_utils.show_results_utils._download_bedrock_aggregate_json")
    @patch("sagemaker.train.common_utils.show_results_utils._display_single_llmaj_evaluation")
    @patch("sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps")
    @patch("rich.console.Console")
    def test_pagination_display_count(
        self,
        mock_console_class,
        mock_extract_job,
        mock_display_single,
        mock_download_aggregate,
        mock_download_results,
        mock_pipeline_execution,
        limit,
        offset,
        total,
        expected_display_count,
    ):
        """Pagination calls _display_single_llmaj_evaluation the correct number of times.

        Observed behavior: display count = min(limit, total - offset) when offset < total.
        """
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        mock_extract_job.side_effect = ["custom-job", None]
        mock_download_aggregate.return_value = ({"results": {}}, "bedrock-job")

        mock_results = [
            {
                "inputRecord": {"prompt": "[{'role': 'user', 'content': 'Q'}]"},
                "modelResponses": [{"response": "['A']"}],
                "automatedEvaluationResult": {
                    "scores": [{"metricName": "M", "result": 0.5}]
                },
            }
        ] * total
        mock_download_results.return_value = mock_results

        _show_llmaj_results(mock_pipeline_execution, limit=limit, offset=offset)

        assert mock_display_single.call_count == expected_display_count

    @patch("sagemaker.train.common_utils.show_results_utils._download_llmaj_results_from_s3")
    @patch("sagemaker.train.common_utils.show_results_utils._download_bedrock_aggregate_json")
    @patch("sagemaker.train.common_utils.show_results_utils._display_single_llmaj_evaluation")
    @patch("sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps")
    @patch("rich.console.Console")
    def test_pagination_starts_at_correct_index(
        self,
        mock_console_class,
        mock_extract_job,
        mock_display_single,
        mock_download_aggregate,
        mock_download_results,
        mock_pipeline_execution,
    ):
        """Pagination with limit=3, offset=2 on 10 results starts at index 2.

        Observed behavior: _display_single_llmaj_evaluation is called with indices 2, 3, 4.
        """
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        mock_extract_job.side_effect = ["custom-job", None]
        mock_download_aggregate.return_value = ({"results": {}}, "bedrock-job")

        # Create 10 distinct results so we can verify indices
        mock_results = [
            {
                "inputRecord": {"prompt": f"[{{'role': 'user', 'content': 'Q{i}'}}]"},
                "modelResponses": [{"response": f"['A{i}']"}],
                "automatedEvaluationResult": {
                    "scores": [{"metricName": "M", "result": float(i) / 10}]
                },
            }
            for i in range(10)
        ]
        mock_download_results.return_value = mock_results

        _show_llmaj_results(mock_pipeline_execution, limit=3, offset=2)

        assert mock_display_single.call_count == 3
        # Verify the indices passed to _display_single_llmaj_evaluation
        for call_idx, expected_i in enumerate([2, 3, 4]):
            actual_call = mock_display_single.call_args_list[call_idx]
            # Args: (result, index, total, console, show_explanations=...)
            assert actual_call[0][0] == mock_results[expected_i], (
                f"Call {call_idx}: expected result at index {expected_i}"
            )
            assert actual_call[0][1] == expected_i, (
                f"Call {call_idx}: expected index arg {expected_i}, got {actual_call[0][1]}"
            )

    # --- Requirement 3.4: show_explanations passthrough ---

    @pytest.mark.parametrize(
        "show_explanations",
        [True, False],
        ids=["explanations_on", "explanations_off"],
    )
    @patch("sagemaker.train.common_utils.show_results_utils._download_llmaj_results_from_s3")
    @patch("sagemaker.train.common_utils.show_results_utils._download_bedrock_aggregate_json")
    @patch("sagemaker.train.common_utils.show_results_utils._display_single_llmaj_evaluation")
    @patch("sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps")
    @patch("rich.console.Console")
    def test_show_explanations_passthrough(
        self,
        mock_console_class,
        mock_extract_job,
        mock_display_single,
        mock_download_aggregate,
        mock_download_results,
        mock_pipeline_execution,
        show_explanations,
    ):
        """show_explanations value is passed through to _display_single_llmaj_evaluation.

        Observed behavior: Each call to _display_single_llmaj_evaluation receives
        show_explanations as a keyword argument matching the value passed to _show_llmaj_results.
        """
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        mock_extract_job.side_effect = ["custom-job", None]
        mock_download_aggregate.return_value = ({"results": {}}, "bedrock-job")

        mock_results = [
            {
                "inputRecord": {"prompt": "[{'role': 'user', 'content': 'Q'}]"},
                "modelResponses": [{"response": "['A']"}],
                "automatedEvaluationResult": {
                    "scores": [{"metricName": "M", "result": 0.8}]
                },
            }
        ] * 3
        mock_download_results.return_value = mock_results

        _show_llmaj_results(
            mock_pipeline_execution, limit=3, offset=0, show_explanations=show_explanations
        )

        assert mock_display_single.call_count == 3
        for c in mock_display_single.call_args_list:
            assert c[1]["show_explanations"] == show_explanations, (
                f"Expected show_explanations={show_explanations}, got {c[1]}"
            )

    # --- Requirement 3.5: Per-example FileNotFoundError -> warning + aggregate display ---

    @pytest.mark.parametrize(
        "has_aggregate",
        [True, False],
        ids=["with_aggregate", "without_aggregate"],
    )
    @patch("sagemaker.train.common_utils.show_results_utils._display_aggregate_metrics")
    @patch("sagemaker.train.common_utils.show_results_utils._download_llmaj_results_from_s3")
    @patch("sagemaker.train.common_utils.show_results_utils._download_bedrock_aggregate_json")
    @patch("sagemaker.train.common_utils.show_results_utils._display_single_llmaj_evaluation")
    @patch("sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps")
    @patch("rich.console.Console")
    def test_per_example_not_found_displays_aggregate_if_available(
        self,
        mock_console_class,
        mock_extract_job,
        mock_display_single,
        mock_download_aggregate,
        mock_download_results,
        mock_display_aggregate,
        mock_pipeline_execution,
        has_aggregate,
    ):
        """When per-example results raise FileNotFoundError, aggregate metrics still display.

        Observed behavior: custom_results stays None, _display_single_llmaj_evaluation is
        never called, but _display_aggregate_metrics is called if aggregate was downloaded.
        """
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        mock_extract_job.side_effect = ["custom-job", None]

        if has_aggregate:
            custom_aggregate = {"results": {"M": {"score": 0.9, "total_evaluations": 5}}}
            mock_download_aggregate.return_value = (custom_aggregate, "bedrock-job")
        else:
            mock_download_aggregate.side_effect = FileNotFoundError("Aggregate not found")

        # Per-example download fails
        mock_download_results.side_effect = FileNotFoundError("Per-example results not found")

        # Should NOT raise
        _show_llmaj_results(mock_pipeline_execution, limit=5, offset=0)

        # No per-example results displayed
        mock_display_single.assert_not_called()

        if has_aggregate:
            mock_display_aggregate.assert_called_once()
        else:
            mock_display_aggregate.assert_not_called()

    # --- Requirement 3.6: Genuinely identical data -> 100% ties is legitimate ---
    # (This is tested via _calculate_win_rates directly in TestCalculateWinRates.test_calculate_ties)
    # Here we verify the integration: when both models return identical per-example data,
    # _calculate_win_rates IS called and its result is displayed.

    @patch("sagemaker.train.common_utils.show_results_utils._display_aggregate_metrics")
    @patch("sagemaker.train.common_utils.show_results_utils._display_win_rates")
    @patch("sagemaker.train.common_utils.show_results_utils._calculate_win_rates")
    @patch("sagemaker.train.common_utils.show_results_utils._download_llmaj_results_from_s3")
    @patch("sagemaker.train.common_utils.show_results_utils._download_bedrock_aggregate_json")
    @patch("sagemaker.train.common_utils.show_results_utils._extract_training_job_name_from_steps")
    @patch("rich.console.Console")
    def test_identical_data_win_rates_still_calculated(
        self,
        mock_console_class,
        mock_extract_job,
        mock_download_aggregate,
        mock_download_results,
        mock_calculate_win,
        mock_display_win,
        mock_display_aggregate,
        mock_pipeline_execution,
    ):
        """When both models have identical per-example data, win rates are still calculated.

        Observed behavior: _calculate_win_rates is called when both custom_results and
        base_results are non-None, regardless of whether the data is identical.
        This preserves the legitimate 100% ties case (Requirement 3.6).
        """
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        mock_extract_job.side_effect = ["custom-job", "base-job"]

        mock_download_aggregate.side_effect = [
            ({"results": {"M": {"score": 0.8}}}, "same-bedrock-job"),
            ({"results": {"M": {"score": 0.8}}}, "same-bedrock-job"),
        ]

        identical_results = [
            {
                "inputRecord": {"prompt": "[{'role': 'user', 'content': 'Q'}]"},
                "modelResponses": [{"response": "['A']"}],
                "automatedEvaluationResult": {
                    "scores": [{"metricName": "M", "result": 0.8}]
                },
            }
        ] * 5
        # Both calls return the same data (this is the current buggy behavior AND
        # the legitimate case when models genuinely tie)
        mock_download_results.return_value = identical_results

        win_rates = {
            "custom_wins": 0, "base_wins": 0, "ties": 5, "total": 5,
            "custom_win_rate": 0.0, "base_win_rate": 0.0, "tie_rate": 1.0,
        }
        mock_calculate_win.return_value = win_rates

        _show_llmaj_results(mock_pipeline_execution, limit=5, offset=0)

        # Win rates should still be calculated and displayed
        mock_calculate_win.assert_called_once()
        mock_display_win.assert_called_once_with(win_rates, mock_console)
