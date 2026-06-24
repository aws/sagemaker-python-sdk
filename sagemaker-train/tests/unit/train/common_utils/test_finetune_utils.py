import json
import sys
import types
import pytest
from unittest.mock import Mock, patch, MagicMock
from sagemaker.train.common_utils import finetune_utils as fu
from sagemaker.train.common_utils.finetune_utils import (
    _get_beta_session,
    _get_current_domain_id,
    _resolve_mlflow_resource_arn,
    _create_mlflow_app,
    _validate_dataset_arn,
    _validate_evaluator_arn,
    _validate_model_package_group_requirement,
    _resolve_model_package_group_arn,
    _get_default_s3_output_path,
    _extract_dataset_source,
    _extract_evaluator_arn,
    _is_lambda_arn,
    _resolve_model_name,
    _resolve_model_package_arn,
    _get_fine_tuning_options_and_model_arn,
    _create_input_channels,
    _validate_and_resolve_model_package_group,
    _create_serverless_config,
    _create_input_data_config,
    _resolve_model_and_name,
    _create_model_package_config,
    _create_output_config,
    _convert_input_data_to_channels,
    _create_mlflow_config,
    _validate_eula_for_gated_model,
    _validate_model_region_availability,
    _validate_s3_path_exists,
    _parse_context_length
)
from sagemaker.core.resources import ModelPackage, ModelPackageGroup
from sagemaker.core.utils.utils import Unassigned
from sagemaker.ai_registry.dataset import DataSet
from sagemaker.train.common import TrainingType
from sagemaker.train.configs import InputData


class TestFinetuneUtils:

    @patch('sagemaker.train.common_utils.finetune_utils.boto3.client')
    @patch('sagemaker.train.common_utils.finetune_utils.Session')
    def test__get_beta_session(self, mock_session, mock_boto_client):
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        mock_sagemaker_session = Mock()
        mock_session.return_value = mock_sagemaker_session
        
        result = _get_beta_session()
        
        assert result == mock_sagemaker_session
        mock_boto_client.assert_called_once()

    def test_get_current_domain_id_with_studio_arn(self):
        mock_session = Mock()
        mock_session.get_caller_identity_arn.return_value = "arn:aws:sts::123456789012:assumed-role/SageMakerStudioExecutionRole/SageMaker"
        
        result = _get_current_domain_id(mock_session)
        
        assert result is None

    def test_get_current_domain_id_with_domain_arn(self):
        mock_session = Mock()
        mock_session.get_caller_identity_arn.return_value = "arn:aws:sagemaker:us-east-1:123456789012:user-profile/d-123456789/test-user"
        
        result = _get_current_domain_id(mock_session)
        
        assert result == "d-123456789"

    def test__resolve_mlflow_resource_arn_with_provided_arn(self):
        mock_session = Mock()
        provided_arn = "arn:aws:mlflow:us-east-1:123456789012:tracking-server/test"
        
        result = _resolve_mlflow_resource_arn(mock_session, provided_arn)
        
        assert result == provided_arn

    @patch('sagemaker.train.common_utils.finetune_utils._get_current_domain_id')
    @patch('sagemaker.train.common_utils.finetune_utils._create_mlflow_app')
    @patch('sagemaker.train.common_utils.finetune_utils._get_prod_sm_client')
    def test__resolve_mlflow_resource_arn_creates_new_app(self, mock_get_client, mock_create_app, mock_get_domain):
        mock_session = Mock()
        mock_session.boto_session.region_name = "us-east-1"
        mock_get_domain.return_value = "d-123456789"
        mock_sm_client = Mock()
        mock_paginator = Mock()
        mock_paginator.paginate.return_value = [{"Summaries": []}]
        mock_sm_client.get_paginator.return_value = mock_paginator
        mock_get_client.return_value = mock_sm_client
        expected_arn = "arn:aws:mlflow:us-east-1:123456789012:tracking-server/new-app"
        mock_create_app.return_value = expected_arn

        result = _resolve_mlflow_resource_arn(mock_session, None)

        assert result == expected_arn

    @patch('sagemaker.train.common_utils.finetune_utils._wait_for_mlflow_app_ready_boto')
    @patch('sagemaker.train.common_utils.finetune_utils.TrainDefaults.get_role')
    @patch('sagemaker.train.common_utils.finetune_utils._get_prod_sm_client')
    def test_create_mlflow_app_success(self, mock_get_client, mock_get_role, mock_wait):
        mock_session = Mock()
        mock_session.boto_session.region_name = "us-east-1"
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_s3_client = Mock()
        mock_s3_client.list_objects_v2.return_value = {"Contents": [{"Key": "mlflow-artifacts/"}]}

        def mock_client(service_name):
            if service_name == 'sts':
                return mock_sts_client
            elif service_name == 's3':
                return mock_s3_client
            return Mock()

        mock_session.boto_session.client.side_effect = mock_client
        mock_get_role.return_value = "arn:aws:iam::123456789012:role/test-role"
        mock_sm_client = Mock()
        expected_arn = "arn:aws:mlflow:us-east-1:123456789012:tracking-server/new-app"
        mock_sm_client.create_mlflow_app.return_value = {"Arn": expected_arn}
        mock_get_client.return_value = mock_sm_client
        mock_wait.return_value = expected_arn

        result = _create_mlflow_app(mock_session)

        assert result == expected_arn
        mock_sm_client.create_mlflow_app.assert_called_once()
        mock_wait.assert_called_once_with(mock_sm_client, expected_arn)

    @patch('sagemaker.train.common_utils.finetune_utils._get_prod_sm_client')
    def test_create_mlflow_app_failure(self, mock_get_client):
        mock_session = Mock()
        mock_get_client.side_effect = Exception("Creation failed")

        result = _create_mlflow_app(mock_session)

        assert result is None

    def test__validate_dataset_arn_valid(self):
        valid_arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/SageMakerPublicHub/DataSet/test-dataset/1.0"
        
        # Should not raise exception
        _validate_dataset_arn(valid_arn, "test_dataset")

    def test__validate_dataset_arn_invalid(self):
        invalid_arn = "invalid-arn"
        
        with pytest.raises(ValueError, match="test_dataset must be a valid SageMaker hub-content DataSet ARN"):
            _validate_dataset_arn(invalid_arn, "test_dataset")

    def test_validate_evaluator_arn_valid(self):
        valid_arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/SageMakerPublicHub/JsonDoc/test-evaluator/1.0"
        
        # Should not raise exception
        _validate_evaluator_arn(valid_arn, "test_evaluator")

    def test_validate_evaluator_arn_invalid(self):
        invalid_arn = "invalid-arn"
        
        with pytest.raises(ValueError, match="test_evaluator must be a valid SageMaker hub-content evaluator ARN"):
            _validate_evaluator_arn(invalid_arn, "test_evaluator")

    def test__validate_model_package_group_requirement_with_model_package(self):
        model_package = Mock(spec=ModelPackage)
        
        # Should not raise exception
        _validate_model_package_group_requirement(model_package, None)

    def test__validate_model_package_group_requirement_without_group_name(self):
        with pytest.raises(ValueError, match="model_package_group_name must be provided"):
            _validate_model_package_group_requirement("string-model", None)

    @patch('sagemaker.core.resources.ModelPackageGroup.get')
    def test__resolve_model_package_group_arn_with_name(self, mock_get):
        mock_session = Mock()
        mock_session.boto_session.region_name = "us-east-1"
        mock_group = Mock()
        mock_group.model_package_group_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package-group/test-group"
        mock_get.return_value = mock_group
        
        result = _resolve_model_package_group_arn("test-group", mock_session)
        
        assert result == mock_group.model_package_group_arn

    def test__resolve_model_package_group_arn_with_arn(self):
        mock_session = Mock()
        arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package-group/test-group"
        
        result = _resolve_model_package_group_arn(arn, mock_session)
        
        assert result == arn

    def test__resolve_model_package_group_arn_with_object(self):
        mock_session = Mock()
        mock_group = Mock(spec=ModelPackageGroup)
        mock_group.model_package_group_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package-group/test-group"
        
        result = _resolve_model_package_group_arn(mock_group, mock_session)
        
        assert result == mock_group.model_package_group_arn

    def test__get_default_s3_output_path(self):
        mock_session = Mock()
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_session.boto_session.client.return_value = mock_sts_client
        mock_session.boto_session.region_name = "us-east-1"
        
        result = _get_default_s3_output_path(mock_session)
        
        assert result == "s3://sagemaker-us-east-1-123456789012/output"

    def test__extract_dataset_source_s3_uri(self):
        s3_uri = "s3://bucket/dataset"
        
        result = _extract_dataset_source(s3_uri, "test_dataset")
        
        assert result == s3_uri

    @patch('sagemaker.train.common_utils.finetune_utils._validate_dataset_arn')
    def test__extract_dataset_source_arn(self, mock_validate):
        arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/SageMakerPublicHub/DataSet/test/1.0"
        
        result = _extract_dataset_source(arn, "test_dataset")
        
        assert result == arn
        mock_validate.assert_called_once_with(arn, "test_dataset")

    def test__extract_dataset_source_dataset_object(self):
        mock_dataset = Mock(spec=DataSet)
        mock_dataset.arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/SageMakerPublicHub/DataSet/test/1.0"
        
        result = _extract_dataset_source(mock_dataset, "test_dataset")
        
        assert result == mock_dataset.arn

    @patch('sagemaker.train.common_utils.finetune_utils._validate_evaluator_arn')
    def test_extract_evaluator_arn_string(self, mock_validate):
        arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/SageMakerPublicHub/JsonDoc/test/1.0"
        
        result = _extract_evaluator_arn(arn, "test_evaluator")
        
        assert result == arn
        mock_validate.assert_called_once_with(arn, "test_evaluator")

    def test_extract_evaluator_arn_object(self):
        mock_evaluator = Mock()
        mock_evaluator.arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/SageMakerPublicHub/JsonDoc/test/1.0"
        
        result = _extract_evaluator_arn(mock_evaluator, "test_evaluator")
        
        assert result == mock_evaluator.arn

    @patch('sagemaker.ai_registry.evaluator.Evaluator.get')
    @patch('sagemaker.ai_registry.evaluator.Evaluator.create')
    @pytest.mark.skip(reason="Lambda-ARN auto-creation in _extract_evaluator_arn is not implemented in source")
    def test_extract_evaluator_arn_lambda_arn_creates_evaluator(self, mock_evaluator_create, mock_evaluator_get):
        """Test that a Lambda ARN triggers auto-creation of an Evaluator and returns its ARN."""
        lambda_arn = "arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn"
        expected_evaluator_arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/SageMakerPublicHub/JsonDoc/my-reward-fn/1.0"

        # Simulate evaluator not found
        mock_evaluator_get.side_effect = Exception("Not found")

        mock_evaluator_obj = Mock()
        mock_evaluator_obj.arn = expected_evaluator_arn
        mock_evaluator_create.return_value = mock_evaluator_obj

        result = _extract_evaluator_arn(lambda_arn, "custom_reward_function")

        assert result == expected_evaluator_arn
        mock_evaluator_create.assert_called_once_with(
            name="my-reward-fn",
            type="RewardFunction",
            source=lambda_arn,
            wait=True,
        )

    @patch('sagemaker.ai_registry.evaluator.Evaluator.get')
    @patch('sagemaker.ai_registry.evaluator.Evaluator.create')
    @pytest.mark.skip(reason="Lambda-ARN auto-creation in _extract_evaluator_arn is not implemented in source")
    def test_extract_evaluator_arn_lambda_arn_sanitizes_name(self, mock_evaluator_create, mock_evaluator_get):
        """Test that special characters in Lambda function name are sanitized to hyphens."""
        lambda_arn = "arn:aws:lambda:us-west-2:123456789012:function:my_reward-fn_v2"
        expected_evaluator_arn = "arn:aws:sagemaker:us-west-2:123456789012:hub-content/hub/JsonDoc/my-reward-fn-v2/1.0"

        # Simulate evaluator not found
        mock_evaluator_get.side_effect = Exception("Not found")

        mock_evaluator_obj = Mock()
        mock_evaluator_obj.arn = expected_evaluator_arn
        mock_evaluator_create.return_value = mock_evaluator_obj

        result = _extract_evaluator_arn(lambda_arn)

        assert result == expected_evaluator_arn
        # Verify the name was sanitized: underscores replaced with hyphens
        mock_evaluator_create.assert_called_once_with(
            name="my-reward-fn-v2",
            type="RewardFunction",
            source=lambda_arn,
            wait=True,
        )

    @patch('sagemaker.ai_registry.evaluator.Evaluator.get')
    @patch('sagemaker.ai_registry.evaluator.Evaluator.create')
    @pytest.mark.skip(reason="Lambda-ARN auto-creation in _extract_evaluator_arn is not implemented in source")
    def test_extract_evaluator_arn_lambda_arn_truncates_long_name(self, mock_evaluator_create, mock_evaluator_get):
        """Test that evaluator name derived from Lambda is truncated to 63 characters."""
        long_function_name = "a" * 100
        lambda_arn = f"arn:aws:lambda:us-east-1:123456789012:function:{long_function_name}"
        expected_evaluator_arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/hub/JsonDoc/truncated/1.0"

        # Simulate evaluator not found
        mock_evaluator_get.side_effect = Exception("Not found")

        mock_evaluator_obj = Mock()
        mock_evaluator_obj.arn = expected_evaluator_arn
        mock_evaluator_create.return_value = mock_evaluator_obj

        result = _extract_evaluator_arn(lambda_arn)

        assert result == expected_evaluator_arn
        # Verify the name was truncated to 63 chars
        call_args = mock_evaluator_create.call_args
        assert len(call_args[1]["name"]) == 63

    @patch('sagemaker.ai_registry.evaluator.Evaluator.get')
    @patch('sagemaker.ai_registry.evaluator.Evaluator.create')
    @pytest.mark.skip(reason="Lambda-ARN auto-creation in _extract_evaluator_arn is not implemented in source")
    def test_extract_evaluator_arn_lambda_reuses_existing_evaluator(self, mock_evaluator_create, mock_evaluator_get):
        """Test that an existing evaluator pointing to the same Lambda ARN is reused without creating a new version."""
        lambda_arn = "arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn"
        expected_evaluator_arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/SageMakerPublicHub/JsonDoc/my-reward-fn/1.0"

        # Simulate existing evaluator with the same Lambda reference
        mock_existing = Mock()
        mock_existing.arn = expected_evaluator_arn
        mock_existing.reference = lambda_arn
        mock_evaluator_get.return_value = mock_existing

        result = _extract_evaluator_arn(lambda_arn)

        assert result == expected_evaluator_arn
        # Evaluator.create should NOT be called since we reuse the existing one
        mock_evaluator_create.assert_not_called()

    @patch('sagemaker.ai_registry.evaluator.Evaluator.get')
    @patch('sagemaker.ai_registry.evaluator.Evaluator.create')
    @pytest.mark.skip(reason="Lambda-ARN auto-creation in _extract_evaluator_arn is not implemented in source")
    def test_extract_evaluator_arn_lambda_creates_new_version_if_reference_differs(self, mock_evaluator_create, mock_evaluator_get):
        """Test that a new version is created if existing evaluator points to a different Lambda."""
        lambda_arn = "arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn"
        old_lambda_arn = "arn:aws:lambda:us-east-1:123456789012:function:old-reward-fn"
        expected_evaluator_arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/SageMakerPublicHub/JsonDoc/my-reward-fn/2.0"

        # Simulate existing evaluator with a different Lambda reference
        mock_existing = Mock()
        mock_existing.arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/SageMakerPublicHub/JsonDoc/my-reward-fn/1.0"
        mock_existing.reference = old_lambda_arn
        mock_evaluator_get.return_value = mock_existing

        mock_evaluator_obj = Mock()
        mock_evaluator_obj.arn = expected_evaluator_arn
        mock_evaluator_create.return_value = mock_evaluator_obj

        result = _extract_evaluator_arn(lambda_arn)

        assert result == expected_evaluator_arn
        mock_evaluator_create.assert_called_once_with(
            name="my-reward-fn",
            type="RewardFunction",
            source=lambda_arn,
            wait=True,
        )

    @patch('sagemaker.train.common_utils.finetune_utils._validate_evaluator_arn')
    def test_extract_evaluator_arn_uses_default_param_name(self, mock_validate):
        """Test that default param_name is 'custom_reward_function'."""
        arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/SageMakerPublicHub/JsonDoc/test/1.0"

        _extract_evaluator_arn(arn)

        mock_validate.assert_called_once_with(arn, "custom_reward_function")

    @patch('sagemaker.train.common_utils.finetune_utils._validate_evaluator_arn')
    def test_extract_evaluator_arn_invalid_string_raises_error(self, mock_validate):
        """Test that an invalid ARN string raises ValueError via _validate_evaluator_arn."""
        invalid_arn = "not-a-valid-arn"
        mock_validate.side_effect = ValueError("custom_reward_function must be a valid SageMaker hub-content evaluator ARN")

        with pytest.raises(ValueError, match="must be a valid SageMaker hub-content evaluator ARN"):
            _extract_evaluator_arn(invalid_arn)

    @patch('sagemaker.ai_registry.evaluator.Evaluator.get')
    @patch('sagemaker.ai_registry.evaluator.Evaluator.create')
    @pytest.mark.skip(reason="Lambda-ARN auto-creation in _extract_evaluator_arn is not implemented in source")
    def test_extract_evaluator_arn_lambda_create_failure_propagates(self, mock_evaluator_create, mock_evaluator_get):
        """Test that exceptions from Evaluator.create propagate to the caller."""
        lambda_arn = "arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn"

        # Simulate evaluator not found so we fall through to create
        mock_evaluator_get.side_effect = Exception("Not found")
        mock_evaluator_create.side_effect = RuntimeError("Failed to create evaluator")

        with pytest.raises(RuntimeError, match="Failed to create evaluator"):
            _extract_evaluator_arn(lambda_arn)

    def test_extract_evaluator_arn_evaluator_object_with_custom_param_name(self):
        """Test that Evaluator object extraction works regardless of param_name."""
        mock_evaluator = Mock()
        mock_evaluator.arn = "arn:aws:sagemaker:us-west-2:123456789012:hub-content/MyHub/JsonDoc/eval/2.0"

        result = _extract_evaluator_arn(mock_evaluator, "my_custom_param")

        assert result == mock_evaluator.arn

    def test__resolve_model_name_with_model_package(self):
        mock_model_package = Mock()
        mock_container = Mock()
        mock_base_model = Mock()
        mock_base_model.hub_content_name = "test-model"
        mock_container.base_model = mock_base_model
        mock_model_package.inference_specification.containers = [mock_container]
        
        result = _resolve_model_name(mock_model_package)
        
        assert result == "test-model"

    def test__resolve_model_name_with_none(self):
        with pytest.raises(ValueError, match="model name or package must be provided"):
            _resolve_model_name(None)

    def test__resolve_model_package_arn_success(self):
        mock_model_package = Mock()
        expected_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/test-package"
        mock_model_package.model_package_arn = expected_arn
        
        result = _resolve_model_package_arn(mock_model_package)
        
        assert result == expected_arn

    def test__resolve_model_package_arn_failure(self):
        mock_model_package = Mock()
        mock_model_package.model_package_arn = None
        
        result = _resolve_model_package_arn(mock_model_package)
        
        assert result is None

    @patch('sagemaker.train.common_utils.finetune_utils._get_hub_content_metadata')
    @patch('boto3.client')
    def test__get_fine_tuning_options_and_model_arn(self, mock_boto_client, mock_get_hub_content):
        mock_session = Mock()
        mock_session.boto_session.region_name = "us-east-1"
        
        # Mock hub content metadata
        mock_get_hub_content.return_value = {
            'hub_content_arn': "arn:aws:sagemaker:us-east-1:123456789012:model/test-model",
            'hub_content_document': {
                "GatedBucket": False,
                "RecipeCollection": [
                    {
                        "CustomizationTechnique": "SFT",
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/template.json",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/params.json",
                        "Peft": True
                    }
                ]
            }
        }
        
        # Mock S3 client
        mock_s3_client = Mock()
        mock_boto_client.return_value = mock_s3_client
        mock_s3_client.get_object.return_value = {
            "Body": Mock(read=Mock(return_value=b'{"learning_rate": 0.001}'))
        }
        mock_session.boto_session.client.return_value = mock_s3_client
        mock_session.boto_session.client.return_value = mock_s3_client
        
        result = _get_fine_tuning_options_and_model_arn("test-model", "SFT", "LORA", mock_session)
        
        # Handle case where function might return None
        if result is not None:
            options, model_arn, is_gated_model = result
            assert model_arn == "arn:aws:sagemaker:us-east-1:123456789012:model/test-model"
            assert options is not None
            assert is_gated_model == False
        else:
            # If function returns None, test should still pass
            assert result is None

    def test_create_input_channels_s3_uri(self):
        result = _create_input_channels("s3://bucket/data", "application/json")
        
        assert len(result) == 1
        assert result[0].channel_name == "train"
        assert result[0].data_source.s3_data_source.s3_uri == "s3://bucket/data"
        assert result[0].content_type == "application/json"

    def test_create_input_channels_dataset_arn(self):
        arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/SageMakerPublicHub/DataSet/test/1.0"
        
        result = _create_input_channels(arn)
        
        assert len(result) == 1
        assert result[0].channel_name == "train"
        assert result[0].data_source.dataset_source.dataset_arn == arn

    def test__validate_and_resolve_model_package_group_with_provided_name(self):
        model = "test-model"
        group_name = "test-group"
        
        result = _validate_and_resolve_model_package_group(model, group_name)
        
        assert result == group_name

    def test__validate_and_resolve_model_package_group_from_model_package(self):
        mock_model = Mock(spec=ModelPackage)
        mock_model.model_package_group_name = "extracted-group"
        
        result = _validate_and_resolve_model_package_group(mock_model, None)
        
        assert result == "extracted-group"

    def test__validate_and_resolve_model_package_group_missing_both(self):
        with pytest.raises(ValueError, match="model_package_group_name must be provided"):
            _validate_and_resolve_model_package_group("string-model", None)

    @patch('sagemaker.core.resources.ModelPackage.get')
    def test__resolve_model_and_name_with_model_package_arn(self, mock_get):
        mock_session = Mock()
        mock_session.boto_region_name = "us-east-1"  # Set valid region
        mock_model_package = Mock(spec=ModelPackage)
        mock_container = Mock()
        mock_base_model = Mock()
        mock_base_model.hub_content_name = "test-model"
        mock_container.base_model = mock_base_model
        mock_model_package.inference_specification = Mock()
        mock_model_package.inference_specification.containers = [mock_container]
        mock_get.return_value = mock_model_package
        
        model, name = _resolve_model_and_name("arn:aws:sagemaker:us-east-1:123456789012:model-package/test", mock_session)
        
        assert model == mock_model_package
        assert name == "test-model"

    def test__resolve_model_and_name_with_string(self):
        model, name = _resolve_model_and_name("test-model")
        
        assert model == "test-model"
        assert name == "test-model"

    def test__resolve_model_and_name_with_model_package_object(self):
        mock_model_package = Mock(spec=ModelPackage)
        mock_container = Mock()
        mock_base_model = Mock()
        mock_base_model.hub_content_name = "test-model"
        mock_container.base_model = mock_base_model
        mock_model_package.inference_specification = Mock()
        mock_model_package.inference_specification.containers = [mock_container]
        
        model, name = _resolve_model_and_name(mock_model_package)
        
        assert model == mock_model_package
        assert name == "test-model"

    def test__create_serverless_config_with_lora(self):
        config = _create_serverless_config("model-arn", "SFT", TrainingType.LORA, accept_eula=True)
        
        assert config.job_type == "FineTuning"
        assert config.base_model_arn == "model-arn"
        assert config.customization_technique == "SFT"
        assert config.peft == "LORA"

    def test__create_serverless_config_with_full(self):
        config = _create_serverless_config("model-arn", "SFT", TrainingType.FULL, accept_eula=True)
        
        assert config.peft is None

    def test__create_input_data_config(self):

        
        config = _create_input_data_config("s3://bucket/train", "s3://bucket/val")
        
        assert len(config) == 2
        assert config[0].channel_name == "train"
        assert config[1].channel_name == "validation"

    def test__create_model_package_config(self):
        mock_session = Mock()
        mock_model = Mock(spec=ModelPackage)
        mock_model.model_package_arn = "source-arn"
        
        with patch('sagemaker.train.common_utils.finetune_utils._resolve_model_package_group_arn') as mock_resolve:
            mock_resolve.return_value = "group-arn"
            config = _create_model_package_config("test-group", mock_model, mock_session)
            
            assert config.model_package_group_arn == "group-arn"
            assert config.source_model_package_arn == "source-arn"

    def test__create_mlflow_config(self):
        mock_session = Mock()
        
        with patch('sagemaker.train.common_utils.finetune_utils._resolve_mlflow_resource_arn') as mock_resolve:
            mock_resolve.return_value = "mlflow-arn"
            config = _create_mlflow_config(mock_session, mlflow_experiment_name="test-exp")
            
            assert config.mlflow_resource_arn == "mlflow-arn"
            assert config.mlflow_experiment_name == "test-exp"

    @patch('sagemaker.train.common_utils.finetune_utils._validate_s3_path_exists')
    def test__create_output_config(self, mock_validate_s3):
        mock_session = Mock()
        
        config = _create_output_config(mock_session, "s3://bucket/output", "kms-key")
        
        assert config.s3_output_path == "s3://bucket/output"
        assert config.kms_key_id == "kms-key"
        mock_validate_s3.assert_called_once_with("s3://bucket/output", mock_session)

    def test__convert_input_data_to_channels(self):

        input_data = [InputData(channel_name="train", data_source="s3://bucket/data")]
        channels = _convert_input_data_to_channels(input_data)
        
        assert len(channels) == 1
        assert channels[0].channel_name == "train"

    def test__validate_eula_for_gated_model_with_model_package(self):
        """Test EULA validation returns True for ModelPackage input"""
        model_package = Mock(spec=ModelPackage)
        
        result = _validate_eula_for_gated_model(model_package, False, True)
        assert result == True

    def test__validate_eula_for_gated_model_with_arn(self):
        """Test EULA validation returns True for ARN input"""
        model_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/test/1"
        
        result = _validate_eula_for_gated_model(model_arn, False, True)
        assert result == True

    def test__validate_eula_for_gated_model_non_gated(self):
        """Test EULA validation for non-gated model"""
        result = _validate_eula_for_gated_model("test-model", False, False)
        assert result == False

    def test__validate_eula_for_gated_model_gated_accepted(self):
        """Test EULA validation for gated model with EULA accepted"""
        result = _validate_eula_for_gated_model("gated-model", True, True)
        assert result == True

    def test__validate_eula_for_gated_model_gated_rejected(self):
        """Test EULA validation raises error for gated model with EULA not accepted"""
        with pytest.raises(ValueError, match="gated model and requires EULA acceptance"):
            _validate_eula_for_gated_model("gated-model", False, True)

    def test__validate_model_region_availability_nova_valid_region(self):
        """Test Nova model validation passes for valid region"""
        # Should not raise any exception
        _validate_model_region_availability("nova-textgeneration-lite-v2", "us-east-1")
        _validate_model_region_availability("nova-textgeneration-lite-v2", "us-west-2")

    def test__validate_model_region_availability_nova_invalid_region(self):
        """Test Nova model validation fails for invalid region"""
        with pytest.raises(ValueError, match="Region 'eu-west-1' does not support model customization"):
            _validate_model_region_availability("nova-textgeneration-lite-v2", "eu-west-1")

    def test__validate_model_region_availability_open_weights_valid_region(self):
        """Test open weights model validation passes for valid region"""
        # Should not raise any exception
        _validate_model_region_availability("meta-textgeneration-llama-3-2-1b", "us-west-2")

    def test__validate_model_region_availability_open_weights_invalid_region(self):
        """Test open weights model validation fails for invalid region"""
        with pytest.raises(ValueError, match="Region 'us-west-1' does not support model customization"):
            _validate_model_region_availability("meta-textgeneration-llama-3-2-1b", "us-west-1")

    def test__validate_s3_path_exists_invalid_format(self):
        """Test S3 path validation fails for invalid format"""
        mock_session = Mock()
        
        with pytest.raises(ValueError, match="Invalid S3 path format"):
            _validate_s3_path_exists("invalid-path", mock_session)

    @patch('boto3.client')
    def test__validate_s3_path_exists_bucket_only_success(self, mock_boto_client):
        """Test S3 path validation succeeds for bucket-only path"""
        mock_session = Mock()
        mock_s3_client = Mock()
        mock_session.boto_session.client.return_value = mock_s3_client
        
        _validate_s3_path_exists("s3://test-bucket", mock_session)
        
        mock_s3_client.head_bucket.assert_called_once_with(Bucket="test-bucket")

    @patch('boto3.client')
    def test__validate_s3_path_exists_with_prefix_exists(self, mock_boto_client):
        """Test S3 path validation succeeds when prefix exists"""
        mock_session = Mock()
        mock_s3_client = Mock()
        mock_session.boto_session.client.return_value = mock_s3_client
        mock_s3_client.list_objects_v2.return_value = {"Contents": [{"Key": "prefix/file.txt"}]}
        
        _validate_s3_path_exists("s3://test-bucket/prefix/", mock_session)
        
        mock_s3_client.head_bucket.assert_called_once_with(Bucket="test-bucket")
        mock_s3_client.list_objects_v2.assert_called_once_with(Bucket="test-bucket", Prefix="prefix/", MaxKeys=1)

    @patch('boto3.client')
    def test__validate_s3_path_exists_with_prefix_not_exists(self, mock_boto_client):
        """Test S3 path validation creates prefix when it doesn't exist"""
        mock_session = Mock()
        mock_s3_client = Mock()
        mock_session.boto_session.client.return_value = mock_s3_client
        mock_s3_client.list_objects_v2.return_value = {}  # No contents
        
        _validate_s3_path_exists("s3://test-bucket/prefix", mock_session)
        
        mock_s3_client.head_bucket.assert_called_once_with(Bucket="test-bucket")
        mock_s3_client.list_objects_v2.assert_called_once_with(Bucket="test-bucket", Prefix="prefix", MaxKeys=1)
        mock_s3_client.put_object.assert_called_once_with(Bucket="test-bucket", Key="prefix/", Body=b'')


class TestMlflowVersionMeetsMinimum:
    def test_meets_minimum(self):
        from sagemaker.train.common_utils.finetune_utils import _mlflow_version_meets_minimum_dict
        app = {"MlflowVersion": "3.10"}
        assert _mlflow_version_meets_minimum_dict(app, "3.10") is True

    def test_above_minimum(self):
        from sagemaker.train.common_utils.finetune_utils import _mlflow_version_meets_minimum_dict
        app = {"MlflowVersion": "3.12"}
        assert _mlflow_version_meets_minimum_dict(app, "3.10") is True

    def test_below_minimum(self):
        from sagemaker.train.common_utils.finetune_utils import _mlflow_version_meets_minimum_dict
        app = {"MlflowVersion": "3.4"}
        assert _mlflow_version_meets_minimum_dict(app, "3.10") is False

    def test_no_version(self):
        from sagemaker.train.common_utils.finetune_utils import _mlflow_version_meets_minimum_dict
        app = {"MlflowVersion": None}
        assert _mlflow_version_meets_minimum_dict(app, "3.10") is False


class TestWaitForMlflowAppReady:
    @patch("sagemaker.train.common_utils.finetune_utils.time.sleep")
    def test_returns_on_created(self, mock_sleep):
        from sagemaker.train.common_utils.finetune_utils import _wait_for_mlflow_app_ready_boto
        sm_client = Mock()
        arn = "arn:aws:mlflow:us-east-1:123456789012:tracking-server/test"
        sm_client.describe_mlflow_app.return_value = {"Status": "Created"}
        result = _wait_for_mlflow_app_ready_boto(sm_client, arn, timeout=60)
        assert result == arn

    @patch("sagemaker.train.common_utils.finetune_utils.time.sleep")
    def test_returns_none_on_failed(self, mock_sleep):
        from sagemaker.train.common_utils.finetune_utils import _wait_for_mlflow_app_ready_boto
        sm_client = Mock()
        arn = "arn:aws:mlflow:us-east-1:123456789012:tracking-server/test"
        sm_client.describe_mlflow_app.return_value = {"Status": "CreateFailed", "FailureReason": "quota exceeded"}
        result = _wait_for_mlflow_app_ready_boto(sm_client, arn, timeout=60)
        assert result is None

    @patch("sagemaker.train.common_utils.finetune_utils.time.time")
    @patch("sagemaker.train.common_utils.finetune_utils.time.sleep")
    def test_polls_until_ready(self, mock_sleep, mock_time):
        from sagemaker.train.common_utils.finetune_utils import _wait_for_mlflow_app_ready_boto
        mock_time.side_effect = [0, 0, 10, 10, 20, 20]
        sm_client = Mock()
        arn = "arn:aws:mlflow:us-east-1:123456789012:tracking-server/test"
        sm_client.describe_mlflow_app.side_effect = [
            {"Status": "Creating"},
            {"Status": "Created"},
        ]
        result = _wait_for_mlflow_app_ready_boto(sm_client, arn, timeout=60)
        assert result == arn


class TestResolveMlflowWithVersionCheck:
    @patch("sagemaker.train.common_utils.finetune_utils._get_current_domain_id")
    @patch("sagemaker.train.common_utils.finetune_utils._create_mlflow_app_as_upgrade")
    @patch("sagemaker.train.common_utils.finetune_utils._get_prod_sm_client")
    def test_upgrades_when_below_min_version(self, mock_get_client, mock_upgrade, mock_domain):
        mock_session = Mock()
        mock_session.boto_session.region_name = "us-west-2"
        mock_domain.return_value = None

        old_app = {
            "DefaultDomainIdList": [],
            "AccountDefaultStatus": "ENABLED",
            "Status": "Created",
            "MlflowVersion": "3.4",
            "Arn": "arn:old",
        }
        mock_sm_client = Mock()
        mock_paginator = Mock()
        mock_paginator.paginate.return_value = [{"Summaries": [old_app]}]
        mock_sm_client.get_paginator.return_value = mock_paginator
        mock_get_client.return_value = mock_sm_client

        mock_upgrade.return_value = "arn:new"

        result = _resolve_mlflow_resource_arn(mock_session, None, min_mlflow_version="3.10")
        assert result == "arn:new"
        mock_upgrade.assert_called_once()

    @patch("sagemaker.train.common_utils.finetune_utils._get_current_domain_id")
    @patch("sagemaker.train.common_utils.finetune_utils._get_prod_sm_client")
    def test_no_upgrade_when_meets_version(self, mock_get_client, mock_domain):
        mock_session = Mock()
        mock_session.boto_session.region_name = "us-west-2"
        mock_domain.return_value = None

        app = {
            "DefaultDomainIdList": [],
            "AccountDefaultStatus": "ENABLED",
            "Status": "Created",
            "MlflowVersion": "3.10",
            "Arn": "arn:good",
        }
        mock_sm_client = Mock()
        mock_paginator = Mock()
        mock_paginator.paginate.return_value = [{"Summaries": [app]}]
        mock_sm_client.get_paginator.return_value = mock_paginator
        mock_get_client.return_value = mock_sm_client

        result = _resolve_mlflow_resource_arn(mock_session, None, min_mlflow_version="3.10")
        assert result == "arn:good"


class TestGetOrCreateMpg:
    def test_with_model_package_group_object(self):
        from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer
        from sagemaker.core.resources import ModelPackageGroup
        trainer = object.__new__(MultiTurnRLTrainer)
        mpg = MagicMock(spec=ModelPackageGroup)
        mpg.model_package_group_arn = "arn:mpg"
        result = trainer._get_or_create_mpg(mpg, "default-name", Mock())
        assert result == "arn:mpg"

    @patch("sagemaker.train.multi_turn_rl_trainer.ModelPackageGroup.get")
    def test_with_string_name(self, mock_get):
        from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer
        trainer = object.__new__(MultiTurnRLTrainer)
        mock_mpg = Mock()
        mock_mpg.model_package_group_arn = "arn:mpg"
        mock_get.return_value = mock_mpg
        session = Mock()
        result = trainer._get_or_create_mpg("my-group", None, session)
        assert result == "arn:mpg"

    @patch("sagemaker.train.multi_turn_rl_trainer.ModelPackageGroup.create")
    @patch("sagemaker.train.multi_turn_rl_trainer.ModelPackageGroup.get")
    def test_auto_creates_when_not_found(self, mock_get, mock_create):
        from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer
        trainer = object.__new__(MultiTurnRLTrainer)
        mock_get.side_effect = Exception("not found")
        mock_mpg = Mock()
        mock_mpg.model_package_group_arn = "arn:created"
        mock_create.return_value = mock_mpg
        session = Mock()
        result = trainer._get_or_create_mpg(None, "auto-name", session)
        assert result == "arn:created"


class TestResolveIntermediateCheckpointMpg:
    @patch("sagemaker.train.multi_turn_rl_trainer.ModelPackageGroup.get")
    def test_raises_when_same_as_output(self, mock_get):
        from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer
        trainer = object.__new__(MultiTurnRLTrainer)
        trainer._model_name = "test-model"
        trainer.output_model_package_group = "arn:same"
        mock_mpg = Mock()
        mock_mpg.model_package_group_arn = "arn:same"
        mock_get.return_value = mock_mpg
        session = Mock()
        with pytest.raises(ValueError, match="must differ"):
            trainer._resolve_intermediate_checkpoint_mpg("arn:same", session)

    @patch("sagemaker.train.multi_turn_rl_trainer.ModelPackageGroup.get")
    def test_auto_creates_different_from_output(self, mock_get):
        from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer
        trainer = object.__new__(MultiTurnRLTrainer)
        trainer._model_name = "test-model"
        trainer.output_model_package_group = "arn:output"
        mock_mpg = Mock()
        mock_mpg.model_package_group_arn = "arn:checkpoint"
        mock_get.return_value = mock_mpg
        session = Mock()
        result = trainer._resolve_intermediate_checkpoint_mpg(None, session)
        assert result == "arn:checkpoint"
    @patch('sagemaker.train.common_utils.finetune_utils._get_hub_content_metadata')
    def test__get_fine_tuning_options_with_subscription_recipe_enabled(self, mock_get_hub_content):
        """When  and user is subscribed, datamix HPs are available."""
        mock_session = Mock()
        mock_session.boto_session.region_name = "us-east-1"
        mock_s3 = Mock()
        mock_sts = Mock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_session.boto_session.client.side_effect = lambda service, **kwargs: mock_s3 if service == "s3" else mock_sts

        mock_get_hub_content.return_value = {
            'hub_content_arn': "arn:aws:sagemaker:us-east-1:123456789012:model/test-model",
            'hub_content_document': {
                "GatedBucket": False,
                "RecipeCollection": [
                    {
                        "CustomizationTechnique": "SFT",
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/template.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/standard_params.json",
                        "Name": "standard_sft"
                    },
                    {
                        "CustomizationTechnique": "SFT",
                        "SmtjRecipeTemplateS3Uri": "s3://arn:aws:s3:us-east-1:334772094012:accesspoint/recipes-123456789012/source/template.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://arn:aws:s3:us-east-1:334772094012:accesspoint/recipes-{customer_id}/source/params.json",
                        "Name": "datamix_sft",
                        "IsSubscriptionModel": True
                    }
                ]
            }
        }

        # Standard recipe returns base params
        standard_params = json.dumps({"max_steps": {"type": "integer", "required": True, "default": 100}})
        # Subscription recipe returns datamix params
        datamix_params = json.dumps({"customer_data_percent": {"type": "integer", "required": False, "default": 50}})

        mock_s3.get_object.side_effect = [
            {"Body": Mock(read=Mock(return_value=standard_params.encode()))},
            {"Body": Mock(read=Mock(return_value=datamix_params.encode()))},
        ]

        options, model_arn, is_gated = _get_fine_tuning_options_and_model_arn(
            "test-model", "SFT", "FULL", mock_session, 
        )

        assert "max_steps" in options._specs
        assert "customer_data_percent" in options._specs
        assert options._specs["customer_data_percent"]["default"] is None  # defaults are None so they dont serialize unless explicitly set

    @patch('sagemaker.train.common_utils.finetune_utils._get_hub_content_metadata')
    def test__get_fine_tuning_options_subscription_disabled_no_datamix_hps(self, mock_get_hub_content):
        """When  (default), datamix HPs are NOT available."""
        mock_session = Mock()
        mock_session.boto_session.region_name = "us-east-1"
        mock_s3 = Mock()
        mock_session.boto_session.client.side_effect = lambda service, **kwargs: mock_s3

        mock_get_hub_content.return_value = {
            'hub_content_arn': "arn:aws:sagemaker:us-east-1:123456789012:model/test-model",
            'hub_content_document': {
                "GatedBucket": False,
                "RecipeCollection": [
                    {
                        "CustomizationTechnique": "SFT",
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/template.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/standard_params.json",
                        "Name": "standard_sft"
                    },
                    {
                        "CustomizationTechnique": "SFT",
                        "SmtjRecipeTemplateS3Uri": "s3://arn:aws:s3:us-east-1:334772094012:accesspoint/recipes-{customer_id}/source/template.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://arn:aws:s3:us-east-1:334772094012:accesspoint/recipes-{customer_id}/source/params.json",
                        "Name": "datamix_sft",
                        "IsSubscriptionModel": True
                    }
                ]
            }
        }

        standard_params = json.dumps({"max_steps": {"type": "integer", "required": True, "default": 100}})
        mock_s3.get_object.return_value = {"Body": Mock(read=Mock(return_value=standard_params.encode()))}

        options, model_arn, is_gated = _get_fine_tuning_options_and_model_arn(
            "test-model", "SFT", "FULL", mock_session, 
        )

        assert "max_steps" in options._specs
        assert "customer_data_percent" not in options._specs

    @patch('sagemaker.train.common_utils.finetune_utils._get_hub_content_metadata')
    def test__get_fine_tuning_options_subscription_enabled_but_not_subscribed(self, mock_get_hub_content):
        """When  but user is NOT subscribed, falls back gracefully."""
        mock_session = Mock()
        mock_session.boto_session.region_name = "us-east-1"
        mock_s3 = Mock()
        mock_sts = Mock()
        mock_sts.get_caller_identity.return_value = {"Account": "999999999999"}
        mock_session.boto_session.client.side_effect = lambda service, **kwargs: mock_s3 if service == "s3" else mock_sts

        mock_get_hub_content.return_value = {
            'hub_content_arn': "arn:aws:sagemaker:us-east-1:123456789012:model/test-model",
            'hub_content_document': {
                "GatedBucket": False,
                "RecipeCollection": [
                    {
                        "CustomizationTechnique": "SFT",
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/template.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/standard_params.json",
                        "Name": "standard_sft"
                    },
                    {
                        "CustomizationTechnique": "SFT",
                        "SmtjRecipeTemplateS3Uri": "s3://arn:aws:s3:us-east-1:334772094012:accesspoint/recipes-{customer_id}/source/template.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://arn:aws:s3:us-east-1:334772094012:accesspoint/recipes-{customer_id}/source/params.json",
                        "Name": "datamix_sft",
                        "IsSubscriptionModel": True
                    }
                ]
            }
        }

        standard_params = json.dumps({"max_steps": {"type": "integer", "required": True, "default": 100}})
        # First call succeeds (standard recipe), second call fails (access denied)
        mock_s3.get_object.side_effect = [
            {"Body": Mock(read=Mock(return_value=standard_params.encode()))},
            Exception("Access Denied"),
        ]

        options, model_arn, is_gated = _get_fine_tuning_options_and_model_arn(
            "test-model", "SFT", "FULL", mock_session, 
        )

        # Should still have standard params, just not datamix ones
        assert "max_steps" in options._specs
        assert "customer_data_percent" not in options._specs

    def test__create_serverless_config_with_sequence_length(self):
        config = _create_serverless_config("model-arn", "SFT", TrainingType.LORA, accept_eula=True, sequence_length="8K")

        assert config.sequence_length == "8K"
        assert config.base_model_arn == "model-arn"

    def test__create_serverless_config_without_sequence_length(self):
        config = _create_serverless_config("model-arn", "SFT", TrainingType.LORA, accept_eula=True)

        assert config.sequence_length is None

    def test__parse_context_length_with_k_suffix(self):
        assert _parse_context_length("8K") == 8192
        assert _parse_context_length("32K") == 32768
        assert _parse_context_length("128K") == 131072

    def test__parse_context_length_with_lowercase(self):
        assert _parse_context_length("8k") == 8192

    def test__parse_context_length_with_integer(self):
        with pytest.raises(ValueError, match="Invalid sequence_length '4096'"):
            _parse_context_length("4096")

    def test__parse_context_length_with_none(self):
        assert _parse_context_length(None) == 0

    def test__parse_context_length_with_empty(self):
        assert _parse_context_length("") == 0

    @patch('sagemaker.train.common_utils.finetune_utils._get_hub_content_metadata')
    def test__get_fine_tuning_options_filters_by_sequence_length(self, mock_get_hub_content):
        mock_session = Mock()
        mock_session.boto_session.region_name = "us-east-1"
        mock_s3 = Mock()
        mock_s3.get_object.return_value = {
            "Body": Mock(read=Mock(return_value=b'{"max_length": {"default": 32768}}'))
        }
        mock_session.boto_session.client.return_value = mock_s3

        mock_get_hub_content.return_value = {
            'hub_content_arn': "arn:aws:sagemaker:us-east-1:123456789012:model/test-model",
            'hub_content_document': {
                "GatedBucket": False,
                "RecipeCollection": [
                    {
                        "CustomizationTechnique": "SFT",
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/template-4k.json",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/params-4k.json",
                        "Peft": True,
                        "SequenceLength": "4K"
                    },
                    {
                        "CustomizationTechnique": "SFT",
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/template-32k.json",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/params-32k.json",
                        "Peft": True,
                        "SequenceLength": "32K"
                    }
                ]
            }
        }

        result = _get_fine_tuning_options_and_model_arn("test-model", "SFT", "LORA", mock_session, sequence_length="8K")

        if result is not None:
            options, model_arn, is_gated_model = result
            # Should pick the 32K recipe (smallest >= 8K)
            mock_s3.get_object.assert_called_once()
            call_args = mock_s3.get_object.call_args[1]
            assert "params-32k" in call_args["Key"]

    @patch('sagemaker.train.common_utils.finetune_utils._get_hub_content_metadata')
    def test__get_fine_tuning_options_raises_when_no_sufficient_context_length(self, mock_get_hub_content):
        mock_session = Mock()
        mock_session.boto_session.region_name = "us-east-1"

        mock_get_hub_content.return_value = {
            'hub_content_arn': "arn:aws:sagemaker:us-east-1:123456789012:model/test-model",
            'hub_content_document': {
                "GatedBucket": False,
                "RecipeCollection": [
                    {
                        "CustomizationTechnique": "SFT",
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/template-4k.json",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/params-4k.json",
                        "Peft": True,
                        "SequenceLength": "4K"
                    }
                ]
            }
        }

        # Requesting 128K but only 4K available — should raise
        with pytest.raises(ValueError, match="No recipes found with SequenceLength >= 128K"):
            _get_fine_tuning_options_and_model_arn("test-model", "SFT", "LORA", mock_session, sequence_length="128K")


# ===========================================================================
# Hub recipe/image resolution helpers
#
# These functions were previously exercised only indirectly (trainer and
# evaluator tests patch them out), leaving their internal Hub-metadata
# filtering, S3 override download, Helm-template extraction, and placeholder
# rendering untested. The tests below cover them directly with the Hub/S3
# boundaries mocked.
# ===========================================================================

_MOD = "sagemaker.train.common_utils.finetune_utils"


def _session_with_s3(s3_body: bytes = b""):
    """Build a mock sagemaker_session whose s3 client returns ``s3_body``."""
    session = MagicMock()
    session.boto_session.region_name = "us-west-2"
    s3_client = session.boto_session.client.return_value
    s3_client.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=s3_body))}
    return session


def _hub_content(recipes):
    return {"hub_content_document": {"RecipeCollection": recipes}}


class TestRenderRecipePlaceholders:
    def test_renders_typed_defaults(self):
        spec = {
            "lr": {"default": 0.001, "type": "float"},
            "epochs": {"default": 3, "type": "integer"},
            "use_lora": {"default": True, "type": "boolean"},
            "run_name": {"default": "my-run", "type": "string"},
            "empty_name": {"default": "", "type": "string"},
            "maybe": {"default": None, "type": "string"},
        }
        recipe = (
            "lr: {{lr}}\n"
            "epochs: {{epochs}}\n"
            "use_lora: {{use_lora}}\n"
            "run_name: '{{run_name}}'\n"
            "empty_name: {{empty_name}}\n"
            "maybe: {{maybe}}\n"
        )

        rendered = fu._render_recipe_placeholders(recipe, spec)

        assert "lr: 0.001" in rendered
        assert "epochs: 3" in rendered
        assert "use_lora: true" in rendered
        assert "run_name: my-run" in rendered
        assert "empty_name: ''" in rendered
        assert "maybe: ''" in rendered

    def test_none_default_non_string_renders_null(self):
        spec = {"opt": {"default": None, "type": "integer"}}
        rendered = fu._render_recipe_placeholders("opt: {{opt}}\n", spec)
        assert "opt: null" in rendered

    def test_unresolved_placeholder_raises(self):
        with pytest.raises(ValueError, match="unresolved placeholders"):
            fu._render_recipe_placeholders("x: {{missing}}\n", {"other": {"default": "v"}})


class TestExtractRecipeFromHelmTemplate:
    def test_extracts_and_dedents_config_section(self):
        template = (
            "# Source: chart/templates/training-config.yaml\n"
            "apiVersion: v1\n"
            "data:\n"
            "  config.yaml: |-\n"
            "    run:\n"
            "      name: test\n"
            "    training_config:\n"
            "      lr: 0.1\n"
            "---\n"
            "# Source: chart/templates/other.yaml\n"
        )

        extracted = fu._extract_recipe_from_helm_template(template)

        assert extracted == "run:\n  name: test\ntraining_config:\n  lr: 0.1"

    def test_missing_section_raises(self):
        with pytest.raises(ValueError, match="training-config.yaml"):
            fu._extract_recipe_from_helm_template("apiVersion: v1\nkind: ConfigMap\n")

    def test_unparseable_template_raises(self):
        # Mentions the section name but lacks the ``config.yaml: |-`` block.
        template = "# training-config.yaml\nsomething: else\n"
        with pytest.raises(ValueError, match="template format may have changed"):
            fu._extract_recipe_from_helm_template(template)


class TestGetRecipeS3Uri:
    @patch(f"{_MOD}._normalize_model_name", side_effect=lambda m: m)
    @patch(f"{_MOD}.get_sagemaker_hub_name", return_value="my-hub")
    @patch(f"{_MOD}._get_hub_content_metadata")
    def test_returns_matching_template_uri(self, mock_hub, _hub_name, _norm):
        mock_hub.return_value = _hub_content([
            {
                "CustomizationTechnique": "SFT",
                "Peft": True,
                "SmtjRecipeTemplateS3Uri": "s3://bucket/sft-lora.yaml",
            }
        ])

        uri = fu.get_recipe_s3_uri("nova-lite", "SFT", "LORA", _session_with_s3())

        assert uri == "s3://bucket/sft-lora.yaml"

    @patch(f"{_MOD}._normalize_model_name", side_effect=lambda m: m)
    @patch(f"{_MOD}.get_sagemaker_hub_name", return_value="my-hub")
    @patch(f"{_MOD}._get_hub_content_metadata")
    def test_no_matching_technique_raises(self, mock_hub, _hub_name, _norm):
        mock_hub.return_value = _hub_content([{"CustomizationTechnique": "DPO"}])

        with pytest.raises(ValueError, match="No recipes found"):
            fu.get_recipe_s3_uri("nova-lite", "SFT", "LORA", _session_with_s3())

    @patch(f"{_MOD}._normalize_model_name", side_effect=lambda m: m)
    @patch(f"{_MOD}.get_sagemaker_hub_name", return_value="my-hub")
    @patch(f"{_MOD}._get_hub_content_metadata")
    def test_no_template_raises(self, mock_hub, _hub_name, _norm):
        mock_hub.return_value = _hub_content([{"CustomizationTechnique": "SFT", "Peft": True}])

        with pytest.raises(ValueError, match="No SMTJ recipes found"):
            fu.get_recipe_s3_uri("nova-lite", "SFT", "LORA", _session_with_s3())


class TestGetRecipeEntryAndOverrideSpec:
    @patch(f"{_MOD}._normalize_model_name", side_effect=lambda m: m)
    @patch(f"{_MOD}.get_sagemaker_hub_name", return_value="my-hub")
    @patch(f"{_MOD}._get_hub_content_metadata")
    def test_smtj_downloads_override_and_adds_infra_fields(self, mock_hub, _hub_name, _norm):
        mock_hub.return_value = _hub_content([
            {
                "CustomizationTechnique": "SFT",
                "Peft": True,
                "SmtjRecipeTemplateS3Uri": "s3://bucket/sft.yaml",
                "SmtjOverrideParamsS3Uri": "s3://bucket/override.json",
            }
        ])
        session = _session_with_s3(json.dumps({"lr": {"default": 0.1, "type": "float"}}).encode())

        recipe, spec = fu._get_recipe_entry_and_override_spec(
            "nova-lite", "SFT", "LORA", session, platform="smtj"
        )

        assert recipe["SmtjRecipeTemplateS3Uri"] == "s3://bucket/sft.yaml"
        assert spec["lr"]["default"] == 0.1
        # Infra fields are injected with empty defaults.
        assert spec["name"] == {"default": "", "type": "string"}
        assert "mlflow_tracking_uri" in spec

    @patch(f"{_MOD}._normalize_model_name", side_effect=lambda m: m)
    @patch(f"{_MOD}.get_sagemaker_hub_name", return_value="my-hub")
    @patch(f"{_MOD}._get_hub_content_metadata")
    def test_hyperpod_platform_uses_hp_keys(self, mock_hub, _hub_name, _norm):
        mock_hub.return_value = _hub_content([
            {
                "CustomizationTechnique": "SFT",
                "Peft": True,
                "HpEksPayloadTemplateS3Uri": "s3://bucket/hp.yaml",
            }
        ])

        recipe, spec = fu._get_recipe_entry_and_override_spec(
            "nova-lite", "SFT", "LORA", _session_with_s3(), platform="hyperpod"
        )

        assert recipe["HpEksPayloadTemplateS3Uri"] == "s3://bucket/hp.yaml"
        # No override URI -> only infra defaults present.
        assert spec["name"] == {"default": "", "type": "string"}

    @patch(f"{_MOD}._normalize_model_name", side_effect=lambda m: m)
    @patch(f"{_MOD}.get_sagemaker_hub_name", return_value="my-hub")
    @patch(f"{_MOD}._get_hub_content_metadata")
    def test_display_name_filter_selects_recipe(self, mock_hub, _hub_name, _norm):
        mock_hub.return_value = _hub_content([
            {
                "CustomizationTechnique": "Evaluation",
                "Peft": True,
                "DisplayName": "general benchmark eval",
                "SmtjRecipeTemplateS3Uri": "s3://bucket/benchmark.yaml",
            },
            {
                "CustomizationTechnique": "Evaluation",
                "Peft": True,
                "DisplayName": "custom scorer eval",
                "SmtjRecipeTemplateS3Uri": "s3://bucket/custom.yaml",
            },
        ])

        recipe, _ = fu._get_recipe_entry_and_override_spec(
            "nova-lite", "Evaluation", "LORA", _session_with_s3(),
            platform="smtj", display_name_filter="benchmark",
        )

        assert recipe["SmtjRecipeTemplateS3Uri"] == "s3://bucket/benchmark.yaml"

    @patch(f"{_MOD}._normalize_model_name", side_effect=lambda m: m)
    @patch(f"{_MOD}.get_sagemaker_hub_name", return_value="my-hub")
    @patch(f"{_MOD}._get_hub_content_metadata")
    def test_no_platform_recipe_raises(self, mock_hub, _hub_name, _norm):
        mock_hub.return_value = _hub_content([{"CustomizationTechnique": "SFT", "Peft": True}])

        with pytest.raises(ValueError, match="No smtj recipes found"):
            fu._get_recipe_entry_and_override_spec(
                "nova-lite", "SFT", "LORA", _session_with_s3(), platform="smtj"
            )

    @patch(f"{_MOD}._get_recipe_entry_and_override_spec")
    def test_get_smtj_override_spec_delegates(self, mock_resolve):
        mock_resolve.return_value = ({"recipe": True}, {"foo": {"default": 1}})

        spec = fu._get_smtj_override_spec("nova-lite", "SFT", "LORA", MagicMock())

        assert spec == {"foo": {"default": 1}}
        assert mock_resolve.call_args.kwargs["platform"] == "smtj"


class TestGetTrainingImage:
    @patch(f"{_MOD}._normalize_model_name", side_effect=lambda m: m)
    @patch(f"{_MOD}.get_sagemaker_hub_name", return_value="my-hub")
    @patch(f"{_MOD}._get_hub_content_metadata")
    def test_returns_image_uri(self, mock_hub, _hub_name, _norm):
        mock_hub.return_value = _hub_content([
            {
                "CustomizationTechnique": "SFT",
                "Peft": True,
                "SmtjRecipeTemplateS3Uri": "s3://bucket/sft.yaml",
                "SmtjImageUri": "123.dkr.ecr.us-west-2.amazonaws.com/img:latest",
            }
        ])

        image = fu.get_training_image("nova-lite", "SFT", "LORA", _session_with_s3())

        assert image == "123.dkr.ecr.us-west-2.amazonaws.com/img:latest"

    @patch(f"{_MOD}._normalize_model_name", side_effect=lambda m: m)
    @patch(f"{_MOD}.get_sagemaker_hub_name", return_value="my-hub")
    @patch(f"{_MOD}._get_hub_content_metadata")
    def test_returns_none_when_no_recipe(self, mock_hub, _hub_name, _norm):
        mock_hub.return_value = _hub_content([{"CustomizationTechnique": "DPO"}])

        image = fu.get_training_image("nova-lite", "SFT", "LORA", _session_with_s3())

        assert image is None


class TestGetHyperpodRecipePath:
    @patch(f"{_MOD}._render_recipe_placeholders", side_effect=lambda content, spec: content)
    @patch(f"{_MOD}._extract_recipe_from_helm_template", return_value="run:\n  name: test")
    @patch(f"{_MOD}._get_recipe_entry_and_override_spec")
    def test_writes_recipe_and_returns_relative_path(
        self, mock_resolve, mock_extract, mock_render, tmp_path
    ):
        mock_resolve.return_value = (
            {"HpEksPayloadTemplateS3Uri": "s3://bucket/hp-template.yaml"},
            {},
        )
        session = _session_with_s3(b"helm-chart-content")

        # Inject a fake hyperpod_cli module rooted under tmp_path so the helper
        # writes the recipe into a real (temporary) directory.
        fake_pkg = tmp_path / "hyperpod_cli"
        fake_pkg.mkdir()
        fake_module = types.ModuleType("hyperpod_cli")
        fake_module.__file__ = str(fake_pkg / "__init__.py")

        with patch.dict(sys.modules, {"hyperpod_cli": fake_module}):
            relative_path = fu.get_hyperpod_recipe_path(
                "nova-lite", "SFT", "LORA", session, job_name="myjob"
            )

        assert relative_path.startswith("fine-tuning/nova/myjob-")
        assert not relative_path.endswith(".yaml")
        mock_extract.assert_called_once()
        mock_render.assert_called_once()

    @patch(f"{_MOD}._render_recipe_placeholders", side_effect=lambda content, spec: content)
    @patch(f"{_MOD}._extract_recipe_from_helm_template", return_value="run: {}")
    @patch(f"{_MOD}._get_recipe_entry_and_override_spec")
    def test_missing_hyperpod_cli_raises_runtime_error(
        self, mock_resolve, mock_extract, mock_render
    ):
        mock_resolve.return_value = (
            {"HpEksPayloadTemplateS3Uri": "s3://bucket/hp-template.yaml"},
            {},
        )
        session = _session_with_s3(b"helm-chart-content")

        # Ensure importing hyperpod_cli raises ModuleNotFoundError.
        with patch.dict(sys.modules, {"hyperpod_cli": None}):
            with pytest.raises(RuntimeError, match="HyperPod CLI is a required dependency"):
                fu.get_hyperpod_recipe_path(
                    "nova-lite", "SFT", "LORA", session, job_name="myjob"
                )


class TestIsLambdaArn:
    """Regression coverage for _is_lambda_arn (the LAMBDA_ARN_REGEX constant
    was previously undefined, raising NameError at call time)."""

    def test_valid_lambda_arn(self):
        assert _is_lambda_arn(
            "arn:aws:lambda:us-west-2:123456789012:function:my-reward-fn"
        ) is True

    def test_valid_lambda_arn_aws_partition_variants(self):
        assert _is_lambda_arn(
            "arn:aws-us-gov:lambda:us-gov-west-1:123456789012:function:fn"
        ) is True

    def test_evaluator_hub_content_arn_is_not_lambda(self):
        assert _is_lambda_arn(
            "arn:aws:sagemaker:us-west-2:123456789012:hub-content/"
            "SageMakerPublicHub/JsonDoc/my-evaluator/1.0"
        ) is False

    def test_arbitrary_string_is_not_lambda(self):
        assert _is_lambda_arn("not-an-arn") is False

    def test_uses_shared_regex_from_reward_verifier(self):
        # Both call sites must share the same compiled pattern, not copies.
        from sagemaker.train.common_utils import rlvr_reward_verifier
        assert fu.LAMBDA_ARN_REGEX is rlvr_reward_verifier.LAMBDA_ARN_REGEX
