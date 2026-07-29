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
"""Tests for recipe_utils module."""
from __future__ import absolute_import

import json
import pytest
from unittest.mock import patch, MagicMock, Mock
from io import BytesIO

from sagemaker.train.common_utils.recipe_utils import (
    _is_nova_model,
    _get_hub_content_metadata,
    _download_s3_json,
    _find_evaluation_recipe,
    _get_evaluation_override_params,
    _extract_eval_override_options,
)


class TestIsNovaModel:
    """Tests for _is_nova_model function."""
    
    def test_nova_model_lowercase(self):
        """Test detection of nova model with lowercase."""
        assert _is_nova_model("amazon-nova-pro") is True
    
    def test_nova_model_uppercase(self):
        """Test detection of nova model with uppercase."""
        assert _is_nova_model("AMAZON-NOVA-PRO") is True
    
    def test_nova_model_mixed_case(self):
        """Test detection of nova model with mixed case."""
        assert _is_nova_model("Amazon-Nova-Lite") is True
    
    def test_non_nova_model(self):
        """Test detection of non-nova model."""
        assert _is_nova_model("meta-textgeneration-llama-3-2-1b-instruct") is False
    
    def test_empty_string(self):
        """Test with empty string."""
        assert _is_nova_model("") is False


class TestGetHubContentMetadata:
    """Tests for _get_hub_content_metadata function."""
    
    @patch('sagemaker.train.common_utils.recipe_utils.HubContent')
    def test_get_metadata_success(self, mock_hub_content_class):
        """Test successful retrieval of hub content metadata."""
        # Mock HubContent.get
        mock_hub_content = MagicMock()
        mock_hub_content.hub_content_name = 'test-model'
        mock_hub_content.hub_content_arn = 'arn:aws:sagemaker:us-west-2:aws:hub-content/test'
        mock_hub_content.hub_content_document = '{"RecipeCollection": []}'
        mock_hub_content_class.get.return_value = mock_hub_content
        
        result = _get_hub_content_metadata(
            hub_name="SageMakerPublicHub",
            hub_content_name="test-model"
        )
        
        assert result['hub_content_name'] == 'test-model'
        assert isinstance(result['hub_content_document'], dict)
        assert 'RecipeCollection' in result['hub_content_document']
    
    @patch('sagemaker.train.common_utils.recipe_utils.HubContent')
    def test_get_metadata_with_invalid_json(self, mock_hub_content_class):
        """Test handling of invalid JSON in hub content document."""
        mock_hub_content = MagicMock()
        mock_hub_content.hub_content_name = 'test-model'
        mock_hub_content.hub_content_document = 'invalid json'
        mock_hub_content_class.get.return_value = mock_hub_content
        
        result = _get_hub_content_metadata(
            hub_name="SageMakerPublicHub",
            hub_content_name="test-model"
        )
        
        # Should leave as string if parsing fails
        assert result['hub_content_document'] == 'invalid json'
    
    @patch('sagemaker.train.common_utils.recipe_utils.HubContent')
    def test_get_metadata_with_region_and_session(self, mock_hub_content_class):
        """Test with custom region and session."""
        mock_hub_content = MagicMock()
        mock_hub_content.hub_content_name = 'test-model'
        mock_hub_content_class.get.return_value = mock_hub_content
        
        mock_session = MagicMock()
        
        _get_hub_content_metadata(
            hub_name="SageMakerPublicHub",
            hub_content_name="test-model",
            region="us-east-1",
            session=mock_session
        )
        
        mock_hub_content_class.get.assert_called_once_with(
            hub_name="SageMakerPublicHub",
            hub_content_type="Model",
            hub_content_name="test-model",
            region="us-east-1",
            session=mock_session
        )


class TestDownloadS3Json:
    """Tests for _download_s3_json function."""
    
    @patch('boto3.client')
    def test_download_success(self, mock_boto_client):
        """Test successful download of JSON from S3."""
        s3_mock = MagicMock()
        mock_boto_client.return_value = s3_mock
        
        test_data = {'key': 'value', 'number': 42}
        s3_mock.get_object.return_value = {
            'Body': BytesIO(json.dumps(test_data).encode('utf-8'))
        }
        
        result = _download_s3_json("s3://test-bucket/path/to/file.json")
        
        assert result == test_data
        s3_mock.get_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='path/to/file.json'
        )
    
    @patch('boto3.client')
    def test_download_with_region(self, mock_boto_client):
        """Test download with custom region."""
        s3_mock = MagicMock()
        mock_boto_client.return_value = s3_mock
        
        s3_mock.get_object.return_value = {
            'Body': BytesIO(b'{"test": true}')
        }
        
        _download_s3_json("s3://bucket/key.json", region="us-east-1")
        
        mock_boto_client.assert_called_once_with('s3', region_name="us-east-1")
    
    def test_download_invalid_uri(self):
        """Test error with invalid S3 URI."""
        with pytest.raises(ValueError, match="Invalid S3 URI"):
            _download_s3_json("http://bucket/key.json")
    
    @patch('boto3.client')
    def test_download_invalid_json(self, mock_boto_client):
        """Test error with invalid JSON content."""
        s3_mock = MagicMock()
        mock_boto_client.return_value = s3_mock
        
        s3_mock.get_object.return_value = {
            'Body': BytesIO(b'invalid json')
        }
        
        with pytest.raises(json.JSONDecodeError):
            _download_s3_json("s3://bucket/key.json")


class TestFindEvaluationRecipe:
    """Tests for _find_evaluation_recipe function."""
    
    def test_find_evaluation_recipe_basic(self):
        """Test finding basic evaluation recipe."""
        recipe_collection = [
            {'Type': 'Training', 'Name': 'train-recipe'},
            {'Type': 'Evaluation', 'Name': 'eval-recipe'},
            {'Type': 'Inference', 'Name': 'inference-recipe'}
        ]
        
        result = _find_evaluation_recipe(recipe_collection)
        
        assert result is not None
        assert result['Name'] == 'eval-recipe'
    
    def test_find_evaluation_recipe_with_type_filter(self):
        """Test finding evaluation recipe with evaluation type filter."""
        recipe_collection = [
            {
                'Type': 'Evaluation',
                'EvaluationType': 'LLMAsJudge',
                'Name': 'llmaj-recipe'
            },
            {
                'Type': 'Evaluation',
                'EvaluationType': 'DeterministicEvaluation',
                'Name': 'deterministic-recipe'
            }
        ]
        
        result = _find_evaluation_recipe(
            recipe_collection,
            evaluation_type='DeterministicEvaluation'
        )
        
        assert result is not None
        assert result['Name'] == 'deterministic-recipe'
    
    def test_find_evaluation_recipe_not_found(self):
        """Test when evaluation recipe is not found."""
        recipe_collection = [
            {'Type': 'Training', 'Name': 'train-recipe'},
            {'Type': 'Inference', 'Name': 'inference-recipe'}
        ]
        
        result = _find_evaluation_recipe(recipe_collection)
        
        assert result is None
    
    def test_find_evaluation_recipe_type_mismatch(self):
        """Test when evaluation type doesn't match."""
        recipe_collection = [
            {
                'Type': 'Evaluation',
                'EvaluationType': 'LLMAsJudge',
                'Name': 'llmaj-recipe'
            }
        ]
        
        result = _find_evaluation_recipe(
            recipe_collection,
            evaluation_type='DeterministicEvaluation'
        )
        
        assert result is None
    
    def test_find_evaluation_recipe_empty_collection(self):
        """Test with empty recipe collection."""
        result = _find_evaluation_recipe([])
        assert result is None


class TestGetEvaluationOverrideParams:
    """Tests for _get_evaluation_override_params function."""
    
    @patch('sagemaker.train.common_utils.recipe_utils._download_s3_json')
    @patch('sagemaker.train.common_utils.recipe_utils._get_hub_content_metadata')
    def test_get_override_params_success(self, mock_get_metadata, mock_download):
        """Test successful retrieval of override parameters."""
        # Mock hub content metadata
        mock_get_metadata.return_value = {
            'hub_content_document': {
                'RecipeCollection': [
                    {
                        'Type': 'Evaluation',
                        'EvaluationType': 'DeterministicEvaluation',
                        'Name': 'eval-recipe',
                        'SmtjOverrideParamsS3Uri': 's3://bucket/params.json'
                    }
                ]
            }
        }
        
        # Mock S3 download
        override_params = {
            'max_new_tokens': {'default': 8192, 'type': 'integer'},
            'temperature': {'default': 0, 'type': 'integer'}
        }
        mock_download.return_value = override_params
        
        result = _get_evaluation_override_params("test-model")
        
        assert result == override_params
        mock_download.assert_called_once_with('s3://bucket/params.json', region=None)
    
    @patch('sagemaker.train.common_utils.recipe_utils._get_hub_content_metadata')
    def test_get_override_params_no_recipes(self, mock_get_metadata):
        """Test error when no recipes found."""
        mock_get_metadata.return_value = {
            'hub_content_document': {
                'RecipeCollection': []
            }
        }
        
        with pytest.raises(ValueError, match="Unsupported Base Model"):
            _get_evaluation_override_params("test-model")
    
    @patch('sagemaker.train.common_utils.recipe_utils._get_hub_content_metadata')
    def test_get_override_params_no_evaluation_recipe(self, mock_get_metadata):
        """Test error when evaluation recipe not found."""
        mock_get_metadata.return_value = {
            'hub_content_document': {
                'RecipeCollection': [
                    {'Type': 'Training', 'Name': 'train-recipe'}
                ]
            }
        }
        
        with pytest.raises(ValueError, match="not supported for evaluation"):
            _get_evaluation_override_params("test-model")
    
    @patch('sagemaker.train.common_utils.recipe_utils._get_hub_content_metadata')
    def test_get_override_params_missing_s3_uri(self, mock_get_metadata):
        """Test error when SmtjOverrideParamsS3Uri is missing."""
        mock_get_metadata.return_value = {
            'hub_content_document': {
                'RecipeCollection': [
                    {
                        'Type': 'Evaluation',
                        'EvaluationType': 'DeterministicEvaluation',
                        'Name': 'eval-recipe'
                    }
                ]
            }
        }
        
        with pytest.raises(ValueError, match="missing required configuration parameters"):
            _get_evaluation_override_params("test-model")
    
    @patch('sagemaker.train.common_utils.recipe_utils._download_s3_json')
    @patch('sagemaker.train.common_utils.recipe_utils._get_hub_content_metadata')
    def test_get_override_params_with_custom_hub(self, mock_get_metadata, mock_download):
        """Test with custom hub name."""
        mock_get_metadata.return_value = {
            'hub_content_document': {
                'RecipeCollection': [
                    {
                        'Type': 'Evaluation',
                        'EvaluationType': 'DeterministicEvaluation',
                        'SmtjOverrideParamsS3Uri': 's3://bucket/params.json'
                    }
                ]
            }
        }
        mock_download.return_value = {}
        
        _get_evaluation_override_params(
            "test-model",
            hub_name="CustomHub"
        )
        
        mock_get_metadata.assert_called_once()
        call_args = mock_get_metadata.call_args
        assert call_args[1]['hub_name'] == "CustomHub"


class TestExtractEvalOverrideOptions:
    """Tests for _extract_eval_override_options function."""
    
    def test_extract_default_params(self):
        """Test extracting default parameter values."""
        override_params = {
            'max_new_tokens': {'default': 8192, 'type': 'integer'},
            'temperature': {'default': 0, 'type': 'integer'},
            'top_k': {'default': -1, 'type': 'integer'},
            'top_p': {'default': 1.0, 'type': 'float'}
        }
        
        result = _extract_eval_override_options(override_params)
        
        assert result['max_new_tokens'] == '8192'
        assert result['temperature'] == '0'
        assert result['top_k'] == '-1'
        assert result['top_p'] == '1.0'
    
    def test_extract_full_spec(self):
        """Test extracting full parameter specifications."""
        override_params = {
            'max_new_tokens': {
                'default': 8192,
                'type': 'integer',
                'min': 1,
                'max': 16384
            }
        }
        
        result = _extract_eval_override_options(
            override_params,
            return_full_spec=True
        )
        
        assert result['max_new_tokens']['default'] == 8192
        assert result['max_new_tokens']['type'] == 'integer'
        assert result['max_new_tokens']['min'] == 1
        assert result['max_new_tokens']['max'] == 16384
    
    def test_extract_custom_param_names(self):
        """Test extracting specific parameter names."""
        override_params = {
            'max_new_tokens': {'default': 8192},
            'temperature': {'default': 0},
            'custom_param': {'default': 'value'}
        }
        
        result = _extract_eval_override_options(
            override_params,
            param_names=['max_new_tokens', 'custom_param']
        )
        
        assert 'max_new_tokens' in result
        assert 'custom_param' in result
        assert 'temperature' not in result
    
    def test_extract_missing_params(self):
        """Test handling of missing parameters."""
        override_params = {
            'max_new_tokens': {'default': 8192}
        }
        
        result = _extract_eval_override_options(
            override_params,
            param_names=['max_new_tokens', 'missing_param']
        )
        
        assert 'max_new_tokens' in result
        assert 'missing_param' not in result
    
    def test_extract_param_without_default(self):
        """Test handling of parameters without default value."""
        override_params = {
            'max_new_tokens': {'default': 8192},
            'no_default': {'type': 'integer'}
        }
        
        result = _extract_eval_override_options(override_params)
        
        assert 'max_new_tokens' in result
        assert 'no_default' not in result
    
    def test_extract_empty_override_params(self):
        """Test with empty override parameters."""
        result = _extract_eval_override_options({})
        assert result == {}
