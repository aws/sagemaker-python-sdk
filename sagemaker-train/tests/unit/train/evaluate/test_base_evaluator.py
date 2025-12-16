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
"""BaseEvaluator Tests."""
from __future__ import absolute_import

import pytest
from unittest.mock import patch, MagicMock, Mock
from pydantic import ValidationError

from sagemaker.core.shapes import VpcConfig
from sagemaker.core.resources import ModelPackageGroup, Artifact
from sagemaker.core.shapes import ArtifactSource, ArtifactSourceType
from sagemaker.core.utils.utils import Unassigned
from sagemaker.train.base_trainer import BaseTrainer

from sagemaker.train.evaluate.base_evaluator import BaseEvaluator


# Test constants
DEFAULT_MODEL = "llama3-2-1b-instruct"
DEFAULT_S3_OUTPUT = "s3://my-bucket/outputs"
DEFAULT_MLFLOW_ARN = "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/my-server"
DEFAULT_REGION = "us-west-2"
DEFAULT_ROLE_ARN = "arn:aws:iam::123456789012:role/test-role"
DEFAULT_MODEL_PACKAGE_ARN = "arn:aws:sagemaker:us-west-2:123456789012:model-package/my-package/1"
DEFAULT_MODEL_PACKAGE_GROUP_ARN = "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/my-package"
DEFAULT_HUB_CONTENT_ARN = "arn:aws:sagemaker:us-west-2:aws:hub-content/HubName/Model/llama3/1"
DEFAULT_ARTIFACT_ARN = "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact"


@pytest.fixture
def mock_session():
    """Create a mock SageMaker session."""
    session = MagicMock()
    session.boto_region_name = DEFAULT_REGION
    session.boto_session = MagicMock()
    session.get_caller_identity_arn.return_value = DEFAULT_ROLE_ARN
    return session


@pytest.fixture
def mock_model_info():
    """Create a mock model info object."""
    info = MagicMock()
    info.base_model_name = "llama3-2-1b-instruct"
    info.base_model_arn = DEFAULT_HUB_CONTENT_ARN
    info.source_model_package_arn = None
    return info


@pytest.fixture
def mock_model_info_with_package():
    """Create a mock model info object with source model package."""
    info = MagicMock()
    info.base_model_name = "llama3-2-1b-instruct"
    info.base_model_arn = DEFAULT_HUB_CONTENT_ARN
    info.source_model_package_arn = DEFAULT_MODEL_PACKAGE_ARN
    return info


class TestBaseEvaluatorInit:
    """Tests for BaseEvaluator initialization and validation."""
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_basic_init_with_jumpstart_model(self, mock_resolve, mock_session, mock_model_info):
        """Test basic initialization with JumpStart model ID."""
        mock_resolve.return_value = mock_model_info
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        assert evaluator.model == DEFAULT_MODEL
        assert evaluator.s3_output_path == DEFAULT_S3_OUTPUT
        assert evaluator.mlflow_resource_arn == DEFAULT_MLFLOW_ARN
        assert evaluator.model_package_group == DEFAULT_MODEL_PACKAGE_GROUP_ARN
        assert evaluator.sagemaker_session == mock_session
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_init_with_model_package_arn(self, mock_resolve, mock_session, mock_model_info_with_package):
        """Test initialization with ModelPackage ARN."""
        mock_resolve.return_value = mock_model_info_with_package
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL_PACKAGE_ARN,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            sagemaker_session=mock_session,
        )
        
        assert evaluator.model == DEFAULT_MODEL_PACKAGE_ARN
        assert evaluator._source_model_package_arn == DEFAULT_MODEL_PACKAGE_ARN
    
    @patch("boto3.client")
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_init_without_session_creates_default(self, mock_resolve, mock_boto_client, mock_model_info):
        """Test that default session is created if not provided."""
        mock_resolve.return_value = mock_model_info
        mock_sm_client = MagicMock()
        mock_boto_client.return_value = mock_sm_client
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        )
        
        assert evaluator.sagemaker_session is not None
        mock_boto_client.assert_called_once()
    
    @patch("os.environ.get")
    @patch("boto3.client")
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_init_respects_region_env_var(self, mock_resolve, mock_boto_client, mock_env_get, mock_model_info):
        """Test that SAGEMAKER_REGION environment variable is respected."""
        mock_resolve.return_value = mock_model_info
        mock_env_get.side_effect = lambda key, default=None: "eu-west-1" if key == "SAGEMAKER_REGION" else None
        mock_sm_client = MagicMock()
        mock_boto_client.return_value = mock_sm_client
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        )
        
        assert evaluator.sagemaker_session is not None
        # Verify boto3 client was called with correct region
        call_args = mock_boto_client.call_args
        assert call_args[1]['region_name'] == 'eu-west-1'
    
    @patch("os.environ.get")
    @patch("boto3.client")
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_init_respects_endpoint_env_var(self, mock_resolve, mock_boto_client, mock_env_get, mock_model_info):
        """Test that SAGEMAKER_ENDPOINT environment variable is respected."""
        mock_resolve.return_value = mock_model_info
        test_endpoint = "https://custom-endpoint.amazonaws.com"
        
        def env_side_effect(key, default=None):
            if key == "SAGEMAKER_ENDPOINT":
                return test_endpoint
            return default
        
        mock_env_get.side_effect = env_side_effect
        mock_sm_client = MagicMock()
        mock_boto_client.return_value = mock_sm_client
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        )
        
        # Verify boto3 client was called with custom endpoint
        call_args = mock_boto_client.call_args
        assert call_args[1]['endpoint_url'] == test_endpoint


class TestMLFlowARNValidation:
    """Tests for MLflow ARN validation."""
    
    @pytest.mark.parametrize(
        "mlflow_arn,should_pass",
        [
            ("arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/my-server", True),
            ("arn:aws-cn:sagemaker:cn-north-1:123456789012:mlflow-tracking-server/my-server", True),
            ("arn:aws-us-gov:sagemaker:us-gov-west-1:123456789012:mlflow-tracking-server/my-server", True),
            ("arn:aws:sagemaker:eu-west-1:123456789012:mlflow-tracking-server/server-name-123", True),
            # New mlflow-app pattern tests
            ("arn:aws:sagemaker:us-west-2:052150106756:mlflow-app/app-4WENMECTTDVE", True),
            ("arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-ABC123XYZ", True),
            ("arn:aws-cn:sagemaker:cn-north-1:123456789012:mlflow-app/app-TEST123", True),
            ("arn:aws-us-gov:sagemaker:us-gov-west-1:123456789012:mlflow-app/app-GOV456", True),
            ("arn:aws:sagemaker:eu-west-1:123456789012:mlflow-app/app-name-with-hyphens", True),
            # Invalid patterns
            ("invalid-arn", False),
            ("arn:aws:sagemaker:us-west-2:123456789012:wrong-resource/my-server", False),
            ("arn:aws:sagemaker:us-west-2:invalid-account:mlflow-tracking-server/my-server", False),
            ("arn:aws:sagemaker:us-west-2:invalid-account:mlflow-app/app-123", False),
        ],
    )
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_mlflow_arn_validation(self, mock_resolve, mlflow_arn, should_pass, mock_session, mock_model_info):
        """Test MLflow ARN format validation."""
        mock_resolve.return_value = mock_model_info
        
        if should_pass:
            evaluator = BaseEvaluator(
                model=DEFAULT_MODEL,
                s3_output_path=DEFAULT_S3_OUTPUT,
                mlflow_resource_arn=mlflow_arn,
                model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
                sagemaker_session=mock_session,
            )
            assert evaluator.mlflow_resource_arn == mlflow_arn
        else:
            with pytest.raises(ValidationError):
                BaseEvaluator(
                    model=DEFAULT_MODEL,
                    s3_output_path=DEFAULT_S3_OUTPUT,
                    mlflow_resource_arn=mlflow_arn,
                    model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
                    sagemaker_session=mock_session,
                )
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    @patch("sagemaker.train.common_utils.finetune_utils._resolve_mlflow_resource_arn")
    def test_mlflow_arn_optional_with_resolution(self, mock_resolve_mlflow, mock_resolve, mock_session, mock_model_info):
        """Test that MLflow ARN is optional and gets resolved automatically."""
        mock_resolve.return_value = mock_model_info
        resolved_arn = "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/resolved-server"
        mock_resolve_mlflow.return_value = resolved_arn
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        assert evaluator.mlflow_resource_arn == resolved_arn
        mock_resolve_mlflow.assert_called_once_with(mock_session, None)
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    @patch("sagemaker.train.common_utils.finetune_utils._resolve_mlflow_resource_arn")
    def test_mlflow_arn_provided_skips_resolution(self, mock_resolve_mlflow, mock_resolve, mock_session, mock_model_info):
        """Test that provided MLflow ARN is used instead of resolution."""
        mock_resolve.return_value = mock_model_info
        provided_arn = "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/provided-server"
        resolved_arn = "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/resolved-server"
        mock_resolve_mlflow.return_value = provided_arn  # Should use provided, not resolve
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=provided_arn,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        assert evaluator.mlflow_resource_arn == provided_arn
        # Should still call resolution with the provided ARN
        mock_resolve_mlflow.assert_called_once_with(mock_session, provided_arn)
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    @patch("sagemaker.train.common_utils.finetune_utils._resolve_mlflow_resource_arn")
    def test_mlflow_arn_resolution_returns_none(self, mock_resolve_mlflow, mock_resolve, mock_session, mock_model_info):
        """Test that MLflow resolution can return None (disabled tracking)."""
        mock_resolve.return_value = mock_model_info
        mock_resolve_mlflow.return_value = None
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        assert evaluator.mlflow_resource_arn is None
        mock_resolve_mlflow.assert_called_once_with(mock_session, None)
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    @patch("sagemaker.train.common_utils.finetune_utils._resolve_mlflow_resource_arn")
    def test_mlflow_arn_resolution_with_exception(self, mock_resolve_mlflow, mock_resolve, mock_session, mock_model_info):
        """Test that MLflow resolution exceptions are handled gracefully by returning None."""
        mock_resolve.return_value = mock_model_info
        # _resolve_mlflow_resource_arn handles exceptions internally and returns None
        mock_resolve_mlflow.return_value = None
        
        # Should still create evaluator, with MLflow ARN as None
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        # Exception in resolution is handled internally by _resolve_mlflow_resource_arn
        # which returns None
        assert evaluator.mlflow_resource_arn is None
        mock_resolve_mlflow.assert_called_once_with(mock_session, None)


class TestModelPackageGroupValidation:
    """Tests for model_package_group validation and resolution."""
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_model_package_group_arn_valid(self, mock_resolve, mock_session, mock_model_info):
        """Test valid model package group ARN."""
        mock_resolve.return_value = mock_model_info
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        assert evaluator.model_package_group == DEFAULT_MODEL_PACKAGE_GROUP_ARN
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    @patch("sagemaker.core.resources.ModelPackageGroup.get")
    def test_model_package_group_name_resolution(self, mock_mpg_get, mock_resolve, mock_session, mock_model_info):
        """Test model package group name resolution to ARN."""
        mock_resolve.return_value = mock_model_info
        
        # Mock ModelPackageGroup.get to return an object with ARN
        mock_mpg = MagicMock()
        mock_mpg.model_package_group_arn = DEFAULT_MODEL_PACKAGE_GROUP_ARN
        mock_mpg_get.return_value = mock_mpg
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group="my-package",
            sagemaker_session=mock_session,
            region=DEFAULT_REGION,
        )
        
        assert evaluator.model_package_group == DEFAULT_MODEL_PACKAGE_GROUP_ARN
        mock_mpg_get.assert_called_once_with(
            model_package_group_name="my-package",
            region=DEFAULT_REGION,
        )
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_model_package_group_object_resolution(self, mock_resolve, mock_session, mock_model_info):
        """Test ModelPackageGroup object resolution to ARN."""
        mock_resolve.return_value = mock_model_info
        
        mock_mpg = MagicMock()
        mock_mpg.model_package_group_arn = DEFAULT_MODEL_PACKAGE_GROUP_ARN
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=mock_mpg,
            sagemaker_session=mock_session,
        )
        
        assert evaluator.model_package_group == DEFAULT_MODEL_PACKAGE_GROUP_ARN
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    @patch("sagemaker.core.resources.ModelPackageGroup.get")
    def test_model_package_group_name_not_found(self, mock_mpg_get, mock_resolve, mock_session, mock_model_info):
        """Test model package group name that doesn't exist."""
        mock_resolve.return_value = mock_model_info
        mock_mpg_get.side_effect = Exception("Model package group not found")
        
        with pytest.raises(ValidationError, match="Failed to resolve model package group name"):
            BaseEvaluator(
                model=DEFAULT_MODEL,
                s3_output_path=DEFAULT_S3_OUTPUT,
                mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
                model_package_group="non-existent-package",
                sagemaker_session=mock_session,
                region=DEFAULT_REGION,
            )
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_model_package_group_invalid_type(self, mock_resolve, mock_session, mock_model_info):
        """Test invalid model_package_group type."""
        mock_resolve.return_value = mock_model_info
        
        with pytest.raises(ValidationError):
            BaseEvaluator(
                model=DEFAULT_MODEL,
                s3_output_path=DEFAULT_S3_OUTPUT,
                mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
                model_package_group=12345,  # Invalid type
                sagemaker_session=mock_session,
            )


class TestModelResolution:
    """Tests for model resolution and validation."""
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_model_resolution_jumpstart(self, mock_resolve, mock_session, mock_model_info):
        """Test model resolution for JumpStart model."""
        mock_resolve.return_value = mock_model_info
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        assert evaluator._base_model_name == "llama3-2-1b-instruct"
        assert evaluator._base_model_arn == DEFAULT_HUB_CONTENT_ARN
        assert evaluator._source_model_package_arn is None
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_model_resolution_model_package(self, mock_resolve, mock_session, mock_model_info_with_package):
        """Test model resolution for ModelPackage."""
        mock_resolve.return_value = mock_model_info_with_package
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL_PACKAGE_ARN,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            sagemaker_session=mock_session,
        )
        
        assert evaluator._base_model_name == "llama3-2-1b-instruct"
        assert evaluator._base_model_arn == DEFAULT_HUB_CONTENT_ARN
        assert evaluator._source_model_package_arn == DEFAULT_MODEL_PACKAGE_ARN
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_model_resolution_non_hub_content_fails(self, mock_resolve, mock_session):
        """Test that non-hub-content base models fail validation for ModelPackages."""
        mock_info = MagicMock()
        mock_info.base_model_name = "custom-model"
        mock_info.base_model_arn = "arn:aws:sagemaker:us-west-2:123456789012:model/custom-model"
        mock_info.source_model_package_arn = DEFAULT_MODEL_PACKAGE_ARN
        mock_resolve.return_value = mock_info
        
        with pytest.raises(ValidationError, match="Base model is not supported"):
            BaseEvaluator(
                model=DEFAULT_MODEL_PACKAGE_ARN,
                s3_output_path=DEFAULT_S3_OUTPUT,
                mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
                sagemaker_session=mock_session,
            )
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_model_resolution_failure(self, mock_resolve, mock_session):
        """Test model resolution failure."""
        mock_resolve.side_effect = Exception("Failed to resolve model")
        
        with pytest.raises(ValidationError, match="Failed to resolve model"):
            BaseEvaluator(
                model="invalid-model",
                s3_output_path=DEFAULT_S3_OUTPUT,
                mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
                model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
                sagemaker_session=mock_session,
            )


class TestBaseEvalNameGeneration:
    """Tests for base_eval_name generation."""
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_custom_eval_name(self, mock_resolve, mock_session, mock_model_info):
        """Test custom eval name is used."""
        mock_resolve.return_value = mock_model_info
        custom_name = "my-custom-eval"
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            base_eval_name=custom_name,
            sagemaker_session=mock_session,
        )
        
        assert evaluator.base_eval_name == custom_name
    
    @patch("uuid.uuid4")
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_auto_generated_eval_name(self, mock_resolve, mock_uuid, mock_session, mock_model_info):
        """Test auto-generated eval name format."""
        mock_resolve.return_value = mock_model_info
        mock_uuid.return_value = MagicMock(__str__=lambda self: "12345678-1234-5678-1234-567812345678")
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        # Should be format: eval-{model_name}-{uuid}
        assert evaluator.base_eval_name.startswith("eval-llama3")
        assert evaluator.base_eval_name.endswith("12345678")
    
    @patch("uuid.uuid4")
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_eval_name_sanitization(self, mock_resolve, mock_uuid, mock_session):
        """Test eval name sanitization for special characters."""
        mock_info = MagicMock()
        mock_info.base_model_name = "model@name#with$special%chars"
        mock_info.base_model_arn = DEFAULT_HUB_CONTENT_ARN
        mock_info.source_model_package_arn = None
        mock_resolve.return_value = mock_info
        
        mock_uuid.return_value = MagicMock(__str__=lambda self: "12345678-1234-5678-1234-567812345678")
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        # Special characters should be replaced with hyphens
        assert "@" not in evaluator.base_eval_name
        assert "#" not in evaluator.base_eval_name
        assert "$" not in evaluator.base_eval_name


class TestModelPackageGroupInference:
    """Tests for model package group ARN inference."""
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_infer_model_package_group_arn(self, mock_resolve, mock_session, mock_model_info_with_package):
        """Test inferring model package group ARN from model package ARN."""
        mock_resolve.return_value = mock_model_info_with_package
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL_PACKAGE_ARN,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            sagemaker_session=mock_session,
        )
        
        inferred_arn = evaluator._infer_model_package_group_arn()
        assert inferred_arn == DEFAULT_MODEL_PACKAGE_GROUP_ARN
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_infer_model_package_group_arn_no_source(self, mock_resolve, mock_session, mock_model_info):
        """Test inferring returns None when no source model package."""
        mock_resolve.return_value = mock_model_info
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        inferred_arn = evaluator._infer_model_package_group_arn()
        assert inferred_arn is None
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_get_model_package_group_arn_provided(self, mock_resolve, mock_session, mock_model_info):
        """Test getting model package group ARN when provided."""
        mock_resolve.return_value = mock_model_info
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        arn = evaluator._get_model_package_group_arn()
        assert arn == DEFAULT_MODEL_PACKAGE_GROUP_ARN
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_get_model_package_group_arn_inferred(self, mock_resolve, mock_session, mock_model_info_with_package):
        """Test getting model package group ARN when inferred."""
        mock_resolve.return_value = mock_model_info_with_package
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL_PACKAGE_ARN,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            sagemaker_session=mock_session,
        )
        
        arn = evaluator._get_model_package_group_arn()
        assert arn == DEFAULT_MODEL_PACKAGE_GROUP_ARN
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_get_model_package_group_arn_missing_returns_none(self, mock_resolve, mock_session, mock_model_info):
        """Test that missing model_package_group returns None for JumpStart models."""
        mock_resolve.return_value = mock_model_info
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            sagemaker_session=mock_session,
        )
        
        # Should return None for JumpStart models without user-provided model_package_group
        result = evaluator._get_model_package_group_arn()
        assert result is None


class TestArtifactManagement:
    """Tests for artifact creation and management."""
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    @patch("sagemaker.core.resources.Artifact.get_all")
    def test_get_existing_artifact(self, mock_get_all, mock_resolve, mock_session, mock_model_info):
        """Test getting existing artifact."""
        mock_resolve.return_value = mock_model_info
        
        # Mock artifact iterator
        mock_artifact = MagicMock()
        mock_artifact.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_get_all.return_value = iter([mock_artifact])
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        artifact_arn = evaluator._get_or_create_artifact_arn(DEFAULT_HUB_CONTENT_ARN, DEFAULT_REGION)
        assert artifact_arn == DEFAULT_ARTIFACT_ARN
        mock_get_all.assert_called_once_with(
            source_uri=DEFAULT_HUB_CONTENT_ARN,
            region=DEFAULT_REGION,
        )
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    @patch("sagemaker.core.resources.Artifact.get_all")
    @patch("sagemaker.core.resources.Artifact.create")
    def test_create_new_artifact_for_hub_content(self, mock_create, mock_get_all, mock_resolve, mock_session, mock_model_info):
        """Test creating new artifact for hub content."""
        mock_resolve.return_value = mock_model_info
        
        # Mock no existing artifacts
        mock_get_all.return_value = iter([])
        
        # Mock artifact creation
        mock_artifact = MagicMock()
        mock_artifact.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_create.return_value = mock_artifact
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        artifact_arn = evaluator._get_or_create_artifact_arn(DEFAULT_HUB_CONTENT_ARN, DEFAULT_REGION)
        assert artifact_arn == DEFAULT_ARTIFACT_ARN
        mock_create.assert_called_once()
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    @patch("sagemaker.core.resources.Artifact.get_all")
    @patch("sagemaker.core.resources.Artifact.create")
    def test_create_new_artifact_for_model_package(self, mock_create, mock_get_all, mock_resolve, mock_session, mock_model_info_with_package):
        """Test creating new artifact for model package."""
        mock_resolve.return_value = mock_model_info_with_package
        
        # Mock no existing artifacts
        mock_get_all.return_value = iter([])
        
        # Mock artifact creation
        mock_artifact = MagicMock()
        mock_artifact.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_create.return_value = mock_artifact
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL_PACKAGE_ARN,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            sagemaker_session=mock_session,
        )
        
        artifact_arn = evaluator._get_or_create_artifact_arn(DEFAULT_MODEL_PACKAGE_ARN, DEFAULT_REGION)
        assert artifact_arn == DEFAULT_ARTIFACT_ARN
        mock_create.assert_called_once()
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    @patch("sagemaker.core.resources.Artifact.get_all")
    @patch("sagemaker.core.resources.Artifact.create")
    def test_artifact_creation_failure(self, mock_create, mock_get_all, mock_resolve, mock_session, mock_model_info):
        """Test artifact creation failure."""
        mock_resolve.return_value = mock_model_info
        
        # Mock no existing artifacts
        mock_get_all.return_value = iter([])
        
        # Mock artifact creation failure
        mock_create.side_effect = Exception("Artifact creation failed")
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        with pytest.raises(RuntimeError, match="Failed to create artifact"):
            evaluator._get_or_create_artifact_arn(DEFAULT_HUB_CONTENT_ARN, DEFAULT_REGION)


class TestAWSExecutionContext:
    """Tests for AWS execution context retrieval."""
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_get_aws_execution_context(self, mock_resolve, mock_session, mock_model_info):
        """Test getting AWS execution context."""
        mock_resolve.return_value = mock_model_info
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
            region=DEFAULT_REGION,
        )
        
        context = evaluator._get_aws_execution_context()
        
        assert context['role_arn'] == DEFAULT_ROLE_ARN
        assert context['region'] == DEFAULT_REGION
        assert context['account_id'] == '123456789012'
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_get_aws_execution_context_without_region(self, mock_resolve, mock_session, mock_model_info):
        """Test getting AWS execution context without explicit region."""
        mock_resolve.return_value = mock_model_info
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        context = evaluator._get_aws_execution_context()
        
        assert context['role_arn'] == DEFAULT_ROLE_ARN
        assert context['region'] == DEFAULT_REGION  # From mock_session
        assert context['account_id'] == '123456789012'


class TestTemplateRendering:
    """Tests for template selection and rendering."""
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_select_template_base_only(self, mock_resolve, mock_session, mock_model_info):
        """Test template selection for JumpStart model."""
        mock_resolve.return_value = mock_model_info
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        base_template = "base_template"
        full_template = "full_template"
        
        selected = evaluator._select_template(base_template, full_template)
        assert selected == base_template
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_select_template_full(self, mock_resolve, mock_session, mock_model_info_with_package):
        """Test template selection for ModelPackage."""
        mock_resolve.return_value = mock_model_info_with_package
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL_PACKAGE_ARN,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            sagemaker_session=mock_session,
        )
        
        base_template = "base_template"
        full_template = "full_template"
        
        selected = evaluator._select_template(base_template, full_template)
        assert selected == full_template
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_add_vpc_and_kms_to_context(self, mock_resolve, mock_session, mock_model_info):
        """Test adding VPC and KMS to context."""
        mock_resolve.return_value = mock_model_info
        
        vpc_config = VpcConfig(
            security_group_ids=["sg-12345"],
            subnets=["subnet-12345", "subnet-67890"]
        )
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            networking=vpc_config,
            kms_key_id="arn:aws:kms:us-west-2:123456789012:key/12345",
            sagemaker_session=mock_session,
        )
        
        context = {}
        context = evaluator._add_vpc_and_kms_to_context(context)
        
        assert context['vpc_config'] is True
        assert context['vpc_security_group_ids'] == ["sg-12345"]
        assert context['vpc_subnets'] == ["subnet-12345", "subnet-67890"]
        assert context['kms_key_id'] == "arn:aws:kms:us-west-2:123456789012:key/12345"
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_add_vpc_and_kms_to_context_none(self, mock_resolve, mock_session, mock_model_info):
        """Test adding VPC and KMS to context when not provided."""
        mock_resolve.return_value = mock_model_info
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        context = {}
        context = evaluator._add_vpc_and_kms_to_context(context)
        
        assert 'vpc_config' not in context
        assert 'kms_key_id' not in context
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_render_pipeline_definition(self, mock_resolve, mock_session, mock_model_info):
        """Test rendering pipeline definition."""
        mock_resolve.return_value = mock_model_info
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        template_str = "Role: {{ role_arn }}, Output: {{ s3_output_path }}"
        context = {
            'role_arn': DEFAULT_ROLE_ARN,
            's3_output_path': DEFAULT_S3_OUTPUT,
        }
        
        rendered = evaluator._render_pipeline_definition(template_str, context)
        assert rendered == f"Role: {DEFAULT_ROLE_ARN}, Output: {DEFAULT_S3_OUTPUT}"


class TestBaseTemplateContext:
    """Tests for base template context building."""
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_get_base_template_context(self, mock_resolve, mock_session, mock_model_info):
        """Test building base template context."""
        mock_resolve.return_value = mock_model_info
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            mlflow_experiment_name="my-experiment",
            mlflow_run_name="my-run",
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        context = evaluator._get_base_template_context(
            role_arn=DEFAULT_ROLE_ARN,
            region=DEFAULT_REGION,
            account_id="123456789012",
            model_package_group_arn=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            resolved_model_artifact_arn=DEFAULT_ARTIFACT_ARN,
        )
        
        assert context['role_arn'] == DEFAULT_ROLE_ARN
        assert context['mlflow_resource_arn'] == DEFAULT_MLFLOW_ARN
        assert context['mlflow_experiment_name'] == "my-experiment"
        assert context['mlflow_run_name'] == "my-run"
        assert context['model_package_group_arn'] == DEFAULT_MODEL_PACKAGE_GROUP_ARN
        assert context['base_model_arn'] == DEFAULT_HUB_CONTENT_ARN
        assert context['s3_output_path'] == DEFAULT_S3_OUTPUT
        assert context['dataset_artifact_arn'] == DEFAULT_ARTIFACT_ARN
        assert 'action_arn_prefix' in context


class TestResolveModelArtifacts:
    """Tests for model artifacts resolution."""
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    @patch("sagemaker.core.resources.Artifact.get_all")
    def test_resolve_model_artifacts_jumpstart(self, mock_get_all, mock_resolve, mock_session, mock_model_info):
        """Test resolving model artifacts for JumpStart model."""
        mock_resolve.return_value = mock_model_info
        
        mock_artifact = MagicMock()
        mock_artifact.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_get_all.return_value = iter([mock_artifact])
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        artifacts = evaluator._resolve_model_artifacts(DEFAULT_REGION)
        
        assert artifacts['artifact_source_uri'] == DEFAULT_HUB_CONTENT_ARN
        assert artifacts['resolved_model_artifact_arn'] == DEFAULT_ARTIFACT_ARN
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    @patch("sagemaker.core.resources.Artifact.get_all")
    def test_resolve_model_artifacts_model_package(self, mock_get_all, mock_resolve, mock_session, mock_model_info_with_package):
        """Test resolving model artifacts for ModelPackage."""
        mock_resolve.return_value = mock_model_info_with_package
        
        mock_artifact = MagicMock()
        mock_artifact.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_get_all.return_value = iter([mock_artifact])
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL_PACKAGE_ARN,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            sagemaker_session=mock_session,
        )
        
        artifacts = evaluator._resolve_model_artifacts(DEFAULT_REGION)
        
        # Should prefer model package ARN
        assert artifacts['artifact_source_uri'] == DEFAULT_MODEL_PACKAGE_ARN
        assert artifacts['resolved_model_artifact_arn'] == DEFAULT_ARTIFACT_ARN


class TestOptionalFields:
    """Tests for optional fields."""
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_optional_mlflow_fields(self, mock_resolve, mock_session, mock_model_info):
        """Test optional MLflow fields default to None."""
        mock_resolve.return_value = mock_model_info
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        assert evaluator.mlflow_experiment_name is None
        assert evaluator.mlflow_run_name is None
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_optional_networking_and_kms(self, mock_resolve, mock_session, mock_model_info):
        """Test optional networking and KMS fields default to None."""
        mock_resolve.return_value = mock_model_info
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        assert evaluator.networking is None
        assert evaluator.kms_key_id is None


class TestEvaluateMethod:
    """Tests for evaluate method."""
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_evaluate_not_implemented(self, mock_resolve, mock_session, mock_model_info):
        """Test that evaluate raises NotImplementedError."""
        mock_resolve.return_value = mock_model_info
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        with pytest.raises(NotImplementedError, match="Subclasses must implement evaluate method"):
            evaluator.evaluate()


class TestGPTOSSModelValidation:
    """Tests for GPT OSS model validation."""
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_gpt_oss_20b_model_blocked(self, mock_resolve, mock_session):
        """Test that GPT OSS 20B model is blocked from evaluation."""
        mock_info = MagicMock()
        mock_info.base_model_name = "openai-reasoning-gpt-oss-20b"
        mock_info.base_model_arn = DEFAULT_HUB_CONTENT_ARN
        mock_info.source_model_package_arn = None
        mock_resolve.return_value = mock_info
        
        with pytest.raises(ValidationError, match="Evaluation is currently not supported for models created from GPT OSS 20B base model"):
            BaseEvaluator(
                model="openai-reasoning-gpt-oss-20b",
                s3_output_path=DEFAULT_S3_OUTPUT,
                mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
                model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
                sagemaker_session=mock_session,
            )
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_gpt_oss_120b_model_blocked(self, mock_resolve, mock_session):
        """Test that GPT OSS 120B model is blocked from evaluation."""
        mock_info = MagicMock()
        mock_info.base_model_name = "openai-reasoning-gpt-oss-120b"
        mock_info.base_model_arn = DEFAULT_HUB_CONTENT_ARN
        mock_info.source_model_package_arn = None
        mock_resolve.return_value = mock_info
        
        with pytest.raises(ValidationError, match="Evaluation is currently not supported for models created from GPT OSS 20B base model"):
            BaseEvaluator(
                model="openai-reasoning-gpt-oss-120b",
                s3_output_path=DEFAULT_S3_OUTPUT,
                mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
                model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
                sagemaker_session=mock_session,
            )
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_non_gpt_oss_model_allowed(self, mock_resolve, mock_session, mock_model_info):
        """Test that non-GPT OSS models are allowed."""
        mock_resolve.return_value = mock_model_info
        
        # Should not raise an error
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        assert evaluator.model == DEFAULT_MODEL


class TestDatasetValidation:
    """Tests for dataset validation using _validate_and_resolve_dataset."""
    
    def test_validate_dataset_s3_uri_valid(self):
        """Test validation of valid S3 URI."""
        dataset_uri = "s3://my-bucket/path/to/dataset.jsonl"
        result = BaseEvaluator._validate_and_resolve_dataset(dataset_uri)
        assert result == dataset_uri
    
    def test_validate_dataset_hub_content_arn_valid(self):
        """Test validation of valid hub-content DataSet ARN."""
        dataset_arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/AIRegistry/DataSet/my-dataset/1.0"
        result = BaseEvaluator._validate_and_resolve_dataset(dataset_arn)
        assert result == dataset_arn
    
    def test_validate_dataset_hub_content_arn_cn_partition(self):
        """Test validation of hub-content DataSet ARN with aws-cn partition."""
        dataset_arn = "arn:aws-cn:sagemaker:cn-north-1:123456789012:hub-content/CustomHub/DataSet/dataset/2.0"
        result = BaseEvaluator._validate_and_resolve_dataset(dataset_arn)
        assert result == dataset_arn
    
    def test_validate_dataset_hub_content_arn_custom_hub(self):
        """Test validation of hub-content DataSet ARN with custom hub name."""
        dataset_arn = "arn:aws:sagemaker:us-west-2:123456789012:hub-content/MyCustomHub-123/DataSet/test-data/3.5"
        result = BaseEvaluator._validate_and_resolve_dataset(dataset_arn)
        assert result == dataset_arn
    
    def test_validate_dataset_object_with_arn(self):
        """Test validation of DataSet object with arn attribute."""
        mock_dataset = MagicMock()
        mock_dataset.arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/AIRegistry/DataSet/my-dataset/1.0"
        result = BaseEvaluator._validate_and_resolve_dataset(mock_dataset)
        assert result == mock_dataset.arn
    
    def test_validate_dataset_invalid_type(self):
        """Test validation fails for invalid dataset type."""
        with pytest.raises(ValueError, match="Dataset must be a string"):
            BaseEvaluator._validate_and_resolve_dataset(12345)
    
    def test_validate_dataset_invalid_arn_format(self):
        """Test validation fails for invalid ARN format."""
        invalid_arn = "arn:aws:s3:::my-bucket/data"
        with pytest.raises(ValueError, match="Invalid dataset format"):
            BaseEvaluator._validate_and_resolve_dataset(invalid_arn)
    
    def test_validate_dataset_invalid_string(self):
        """Test validation fails for non-S3, non-ARN string."""
        invalid_str = "/local/path/to/dataset.jsonl"
        with pytest.raises(ValueError, match="Invalid dataset format"):
            BaseEvaluator._validate_and_resolve_dataset(invalid_str)
    
    def test_validate_dataset_error_message_contains_examples(self):
        """Test validation error message contains helpful examples."""
        with pytest.raises(ValueError) as exc_info:
            BaseEvaluator._validate_and_resolve_dataset("invalid-dataset")
        
        error_msg = str(exc_info.value)
        assert "arn:*:hub-content/*/DataSet/*" in error_msg
        assert "s3://*" in error_msg
        assert "Example" in error_msg


class TestModelPackageGroupRefactored:
    """Tests for refactored _get_model_package_group_arn method."""
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_get_mpg_arn_user_provided_for_jumpstart(self, mock_resolve, mock_session, mock_model_info):
        """Test that user-provided model_package_group is used for JumpStart model."""
        mock_resolve.return_value = mock_model_info
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        result = evaluator._get_model_package_group_arn()
        assert result == DEFAULT_MODEL_PACKAGE_GROUP_ARN
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_get_mpg_arn_user_provided_for_model_package(self, mock_resolve, mock_session, mock_model_info_with_package):
        """Test that user-provided model_package_group is used even when using ModelPackage."""
        mock_resolve.return_value = mock_model_info_with_package
        
        user_provided_arn = "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/user-provided"
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL_PACKAGE_ARN,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=user_provided_arn,
            sagemaker_session=mock_session,
        )
        
        result = evaluator._get_model_package_group_arn()
        assert result == user_provided_arn
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_get_mpg_arn_inferred_for_model_package(self, mock_resolve, mock_session, mock_model_info_with_package):
        """Test that model_package_group is inferred from ModelPackage when not provided."""
        mock_resolve.return_value = mock_model_info_with_package
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL_PACKAGE_ARN,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            sagemaker_session=mock_session,
        )
        
        result = evaluator._get_model_package_group_arn()
        assert result == DEFAULT_MODEL_PACKAGE_GROUP_ARN
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_get_mpg_arn_returns_none_for_jumpstart(self, mock_resolve, mock_session, mock_model_info):
        """Test that model_package_group returns None for JumpStart model when not provided."""
        mock_resolve.return_value = mock_model_info
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            sagemaker_session=mock_session,
        )
        
        result = evaluator._get_model_package_group_arn()
        assert result is None
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_get_mpg_arn_fails_for_model_package_inference_failure(self, mock_resolve, mock_session):
        """Test that error is raised when ModelPackage ARN inference fails."""
        mock_info = MagicMock()
        mock_info.base_model_name = "test-model"
        mock_info.base_model_arn = DEFAULT_HUB_CONTENT_ARN
        mock_info.source_model_package_arn = "invalid-format-arn"
        mock_resolve.return_value = mock_info
        
        evaluator = BaseEvaluator(
            model="invalid-format-arn",
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            sagemaker_session=mock_session,
        )
        
        with pytest.raises(ValueError, match="Could not infer model_package_group"):
            evaluator._get_model_package_group_arn()


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_very_long_model_name(self, mock_resolve, mock_session):
        """Test handling of very long model names."""
        mock_info = MagicMock()
        mock_info.base_model_name = "a" * 300  # Very long name
        mock_info.base_model_arn = DEFAULT_HUB_CONTENT_ARN
        mock_info.source_model_package_arn = None
        mock_resolve.return_value = mock_info
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )
        
        # Base eval name should be truncated to stay under 256 chars
        assert len(evaluator.base_eval_name) <= 256
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_invalid_model_package_arn_format(self, mock_resolve, mock_session, mock_model_info_with_package):
        """Test handling of invalid model package ARN format."""
        # Use a model info with invalid format ARN
        mock_info = MagicMock()
        mock_info.base_model_name = "test-model"
        mock_info.base_model_arn = DEFAULT_HUB_CONTENT_ARN
        mock_info.source_model_package_arn = "invalid-arn-format"
        mock_resolve.return_value = mock_info
        
        evaluator = BaseEvaluator(
            model="invalid-arn-format",
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            sagemaker_session=mock_session,
        )
        
        # Should return None for invalid format
        inferred = evaluator._infer_model_package_group_arn()
        assert inferred is None
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_with_all_optional_params(self, mock_resolve, mock_session, mock_model_info):
        """Test initialization with all optional parameters."""
        mock_resolve.return_value = mock_model_info
        
        vpc_config = VpcConfig(
            security_group_ids=["sg-12345"],
            subnets=["subnet-12345"]
        )
        
        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            mlflow_experiment_name="test-experiment",
            mlflow_run_name="test-run",
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            base_eval_name="custom-eval",
            networking=vpc_config,
            kms_key_id="arn:aws:kms:us-west-2:123456789012:key/12345",
            region=DEFAULT_REGION,
            sagemaker_session=mock_session,
        )
        
        assert evaluator.model == DEFAULT_MODEL
        assert evaluator.s3_output_path == DEFAULT_S3_OUTPUT
        assert evaluator.mlflow_resource_arn == DEFAULT_MLFLOW_ARN
        assert evaluator.mlflow_experiment_name == "test-experiment"
        assert evaluator.mlflow_run_name == "test-run"
        assert evaluator.model_package_group == DEFAULT_MODEL_PACKAGE_GROUP_ARN
        assert evaluator.base_eval_name == "custom-eval"
        assert evaluator.networking == vpc_config
        assert evaluator.kms_key_id == "arn:aws:kms:us-west-2:123456789012:key/12345"
        assert evaluator.region == DEFAULT_REGION


class TestBaseTrainerHandling:
    """Tests for BaseTrainer model handling."""
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_base_trainer_with_valid_training_job(self, mock_resolve, mock_session, mock_model_info_with_package):
        """Test BaseTrainer with valid completed training job."""
        mock_resolve.return_value = mock_model_info_with_package
        
        # Create mock BaseTrainer with completed training job
        mock_trainer = MagicMock(spec=BaseTrainer)
        mock_training_job = MagicMock()
        mock_training_job.output_model_package_arn = DEFAULT_MODEL_PACKAGE_ARN
        mock_trainer._latest_training_job = mock_training_job
        
        evaluator = BaseEvaluator(
            model=mock_trainer,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            sagemaker_session=mock_session,
        )
        
        # Verify model resolution was called with the training job's model package ARN
        mock_resolve.assert_called_once_with(
            base_model=DEFAULT_MODEL_PACKAGE_ARN,
            sagemaker_session=mock_session
        )
        assert evaluator.model == mock_trainer
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_base_trainer_with_unassigned_arn(self, mock_resolve, mock_session):
        """Test BaseTrainer with Unassigned output_model_package_arn raises error."""
        # Create mock BaseTrainer with Unassigned ARN
        mock_trainer = MagicMock(spec=BaseTrainer)
        mock_training_job = MagicMock()
        mock_training_job.output_model_package_arn = Unassigned()
        mock_trainer._latest_training_job = mock_training_job
        
        with pytest.raises(ValidationError, match="BaseTrainer must have completed training job"):
            BaseEvaluator(
                model=mock_trainer,
                s3_output_path=DEFAULT_S3_OUTPUT,
                mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
                sagemaker_session=mock_session,
            )
    
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_base_trainer_without_training_job(self, mock_resolve, mock_session):
        """Test BaseTrainer without _latest_training_job falls through to normal processing."""
        # Create mock BaseTrainer without _latest_training_job attribute
        mock_trainer = MagicMock()
        mock_trainer.__class__.__name__ = 'BaseTrainer'
        # Don't set _latest_training_job attribute at all
        
        # This should fail during model resolution, not in BaseTrainer handling
        with pytest.raises(ValidationError, match="Failed to resolve model"):
            BaseEvaluator(
                model=mock_trainer,
                s3_output_path=DEFAULT_S3_OUTPUT,
                mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
                sagemaker_session=mock_session,
            )
    
    def test_base_trainer_without_output_model_package_arn_attribute(self, mock_session):
        """Test BaseTrainer with training job but missing output_model_package_arn attribute."""
        
        # Create a custom class that doesn't have output_model_package_arn
        class MockTrainingJobWithoutArn:
            pass
        
        # Create mock BaseTrainer with _latest_training_job but no output_model_package_arn
        mock_trainer = MagicMock()
        mock_trainer.__class__.__name__ = 'BaseTrainer'
        mock_trainer._latest_training_job = MockTrainingJobWithoutArn()
        
        with pytest.raises(ValidationError, match="BaseTrainer must have completed training job"):
            BaseEvaluator(
                model=mock_trainer,
                s3_output_path=DEFAULT_S3_OUTPUT,
                mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
                sagemaker_session=mock_session,
            )
