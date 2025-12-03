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

from unittest.mock import Mock

from sagemaker.core.drift_check_baselines import DriftCheckBaselines


def test_drift_check_baselines_initialization_empty():
    """Test DriftCheckBaselines initialization with no parameters."""
    baselines = DriftCheckBaselines()

    assert baselines.model_statistics is None
    assert baselines.model_constraints is None
    assert baselines.model_data_statistics is None
    assert baselines.model_data_constraints is None
    assert baselines.bias_config_file is None
    assert baselines.bias_pre_training_constraints is None
    assert baselines.bias_post_training_constraints is None
    assert baselines.explainability_constraints is None
    assert baselines.explainability_config_file is None


def test_drift_check_baselines_to_request_dict_empty():
    """Test _to_request_dict with no parameters returns empty dict."""
    baselines = DriftCheckBaselines()

    request_dict = baselines._to_request_dict()

    assert request_dict == {}


def test_drift_check_baselines_with_model_quality():
    """Test DriftCheckBaselines with model quality metrics."""
    mock_statistics = Mock()
    mock_statistics._to_request_dict.return_value = {"S3Uri": "s3://bucket/stats.json"}
    mock_constraints = Mock()
    mock_constraints._to_request_dict.return_value = {"S3Uri": "s3://bucket/constraints.json"}

    baselines = DriftCheckBaselines(
        model_statistics=mock_statistics, model_constraints=mock_constraints
    )

    request_dict = baselines._to_request_dict()

    assert "ModelQuality" in request_dict
    assert request_dict["ModelQuality"]["Statistics"] == {"S3Uri": "s3://bucket/stats.json"}
    assert request_dict["ModelQuality"]["Constraints"] == {"S3Uri": "s3://bucket/constraints.json"}


def test_drift_check_baselines_with_model_data_quality():
    """Test DriftCheckBaselines with model data quality metrics."""
    mock_statistics = Mock()
    mock_statistics._to_request_dict.return_value = {"S3Uri": "s3://bucket/data-stats.json"}
    mock_constraints = Mock()
    mock_constraints._to_request_dict.return_value = {"S3Uri": "s3://bucket/data-constraints.json"}

    baselines = DriftCheckBaselines(
        model_data_statistics=mock_statistics, model_data_constraints=mock_constraints
    )

    request_dict = baselines._to_request_dict()

    assert "ModelDataQuality" in request_dict
    assert request_dict["ModelDataQuality"]["Statistics"] == {
        "S3Uri": "s3://bucket/data-stats.json"
    }
    assert request_dict["ModelDataQuality"]["Constraints"] == {
        "S3Uri": "s3://bucket/data-constraints.json"
    }


def test_drift_check_baselines_with_bias():
    """Test DriftCheckBaselines with bias metrics."""
    mock_config = Mock()
    mock_config._to_request_dict.return_value = {"S3Uri": "s3://bucket/bias-config.json"}
    mock_pre_training = Mock()
    mock_pre_training._to_request_dict.return_value = {"S3Uri": "s3://bucket/pre-training.json"}
    mock_post_training = Mock()
    mock_post_training._to_request_dict.return_value = {"S3Uri": "s3://bucket/post-training.json"}

    baselines = DriftCheckBaselines(
        bias_config_file=mock_config,
        bias_pre_training_constraints=mock_pre_training,
        bias_post_training_constraints=mock_post_training,
    )

    request_dict = baselines._to_request_dict()

    assert "Bias" in request_dict
    assert request_dict["Bias"]["ConfigFile"] == {"S3Uri": "s3://bucket/bias-config.json"}
    assert request_dict["Bias"]["PreTrainingConstraints"] == {
        "S3Uri": "s3://bucket/pre-training.json"
    }
    assert request_dict["Bias"]["PostTrainingConstraints"] == {
        "S3Uri": "s3://bucket/post-training.json"
    }


def test_drift_check_baselines_with_explainability():
    """Test DriftCheckBaselines with explainability metrics."""
    mock_constraints = Mock()
    mock_constraints._to_request_dict.return_value = {
        "S3Uri": "s3://bucket/explain-constraints.json"
    }
    mock_config = Mock()
    mock_config._to_request_dict.return_value = {"S3Uri": "s3://bucket/explain-config.json"}

    baselines = DriftCheckBaselines(
        explainability_constraints=mock_constraints, explainability_config_file=mock_config
    )

    request_dict = baselines._to_request_dict()

    assert "Explainability" in request_dict
    assert request_dict["Explainability"]["Constraints"] == {
        "S3Uri": "s3://bucket/explain-constraints.json"
    }
    assert request_dict["Explainability"]["ConfigFile"] == {
        "S3Uri": "s3://bucket/explain-config.json"
    }


def test_drift_check_baselines_all_parameters():
    """Test DriftCheckBaselines with all parameters."""
    # Create all mocks
    mock_model_stats = Mock()
    mock_model_stats._to_request_dict.return_value = {"S3Uri": "s3://bucket/model-stats.json"}
    mock_model_constraints = Mock()
    mock_model_constraints._to_request_dict.return_value = {
        "S3Uri": "s3://bucket/model-constraints.json"
    }

    mock_data_stats = Mock()
    mock_data_stats._to_request_dict.return_value = {"S3Uri": "s3://bucket/data-stats.json"}
    mock_data_constraints = Mock()
    mock_data_constraints._to_request_dict.return_value = {
        "S3Uri": "s3://bucket/data-constraints.json"
    }

    mock_bias_config = Mock()
    mock_bias_config._to_request_dict.return_value = {"S3Uri": "s3://bucket/bias-config.json"}
    mock_bias_pre = Mock()
    mock_bias_pre._to_request_dict.return_value = {"S3Uri": "s3://bucket/bias-pre.json"}
    mock_bias_post = Mock()
    mock_bias_post._to_request_dict.return_value = {"S3Uri": "s3://bucket/bias-post.json"}

    mock_explain_constraints = Mock()
    mock_explain_constraints._to_request_dict.return_value = {
        "S3Uri": "s3://bucket/explain-constraints.json"
    }
    mock_explain_config = Mock()
    mock_explain_config._to_request_dict.return_value = {"S3Uri": "s3://bucket/explain-config.json"}

    baselines = DriftCheckBaselines(
        model_statistics=mock_model_stats,
        model_constraints=mock_model_constraints,
        model_data_statistics=mock_data_stats,
        model_data_constraints=mock_data_constraints,
        bias_config_file=mock_bias_config,
        bias_pre_training_constraints=mock_bias_pre,
        bias_post_training_constraints=mock_bias_post,
        explainability_constraints=mock_explain_constraints,
        explainability_config_file=mock_explain_config,
    )

    request_dict = baselines._to_request_dict()

    # Verify all sections are present
    assert "ModelQuality" in request_dict
    assert "ModelDataQuality" in request_dict
    assert "Bias" in request_dict
    assert "Explainability" in request_dict

    # Verify structure
    assert len(request_dict) == 4


def test_drift_check_baselines_partial_model_quality():
    """Test DriftCheckBaselines with only model statistics."""
    mock_statistics = Mock()
    mock_statistics._to_request_dict.return_value = {"S3Uri": "s3://bucket/stats.json"}

    baselines = DriftCheckBaselines(model_statistics=mock_statistics)

    request_dict = baselines._to_request_dict()

    assert "ModelQuality" in request_dict
    assert "Statistics" in request_dict["ModelQuality"]
    assert "Constraints" not in request_dict["ModelQuality"]


def test_drift_check_baselines_partial_bias():
    """Test DriftCheckBaselines with only bias config file."""
    mock_config = Mock()
    mock_config._to_request_dict.return_value = {"S3Uri": "s3://bucket/bias-config.json"}

    baselines = DriftCheckBaselines(bias_config_file=mock_config)

    request_dict = baselines._to_request_dict()

    assert "Bias" in request_dict
    assert "ConfigFile" in request_dict["Bias"]
    assert "PreTrainingConstraints" not in request_dict["Bias"]
    assert "PostTrainingConstraints" not in request_dict["Bias"]
