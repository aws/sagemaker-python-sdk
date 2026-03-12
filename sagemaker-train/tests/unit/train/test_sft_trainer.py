import pytest
from unittest.mock import Mock, patch, MagicMock
from sagemaker.train.sft_trainer import SFTTrainer
from sagemaker.train.common import TrainingType
from sagemaker.core.resources import ModelPackage


class TestSFTTrainer:
    
    @pytest.fixture
    def mock_session(self):
        session = Mock()
        session.region_name = "us-east-1"
        return session

    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    def test_init_with_defaults(self, mock_finetuning_options, mock_validate_group, mock_session):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        trainer = SFTTrainer(model="test-model", model_package_group="test-group")
        assert trainer.training_type == TrainingType.LORA
        assert trainer.model == "test-model"

    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    def test_init_with_full_training_type(self, mock_finetuning_options, mock_validate_group, mock_session):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        trainer = SFTTrainer(model="test-model", training_type=TrainingType.FULL, model_package_group="test-group")
        assert trainer.training_type == TrainingType.FULL

    @patch('sagemaker.train.common_utils.finetune_utils._get_beta_session')
    @patch('sagemaker.train.sft_trainer._resolve_model_and_name')
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.sft_trainer.TrainDefaults.get_role')
    @patch('sagemaker.train.sft_trainer.TrainDefaults.get_sagemaker_session')
    @patch('sagemaker.train.sft_trainer._get_unique_name')
    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.sft_trainer._create_input_data_config')
    @patch('sagemaker.train.sft_trainer._convert_input_data_to_channels')
    @patch('sagemaker.train.sft_trainer._create_output_config')
    @patch('sagemaker.train.sft_trainer._create_mlflow_config')
    @patch('sagemaker.train.sft_trainer._create_model_package_config')
    @patch('sagemaker.core.resources.TrainingJob.create')
    def test_peft_value_for_lora_training(self, mock_training_job_create, mock_model_package_config, mock_mlflow_config, mock_output_config, mock_convert_channels, mock_input_config, mock_validate_group, mock_unique_name, mock_get_sagemaker_session, mock_get_role, 
                                        mock_get_options, mock_resolve_model, mock_get_session):
        # Mock all utility functions
        mock_validate_group.return_value = "test-group"
        mock_resolve_model.return_value = ("test-model", "test-model")
        mock_get_session.return_value = Mock()
        mock_get_sagemaker_session.return_value = Mock()
        
        mock_fine_tuning_options = Mock()
        mock_fine_tuning_options.to_dict.return_value = {"learning_rate": "0.001"}
        mock_get_options.return_value = (mock_fine_tuning_options, "model-arn", False)
        
        mock_get_role.return_value = "test-role"
        mock_unique_name.return_value = "test-job-name"
        
        mock_input_config.return_value = [Mock()]
        mock_convert_channels.return_value = [Mock()]
        mock_output_config.return_value = Mock()
        mock_mlflow_config.return_value = Mock()
        mock_model_package_config.return_value = Mock()
        
        mock_training_job = Mock()
        mock_training_job.arn = "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        mock_training_job.wait = Mock()
        mock_training_job_create.return_value = mock_training_job
        
        trainer = SFTTrainer(model="test-model", training_type=TrainingType.LORA, model_package_group="test-group", training_dataset="s3://bucket/train")
        trainer.train(wait=False)
        
        assert mock_training_job_create.called

    @patch('sagemaker.train.common_utils.finetune_utils._get_beta_session')
    @patch('sagemaker.train.sft_trainer._resolve_model_and_name')
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.sft_trainer.TrainDefaults.get_role')
    @patch('sagemaker.train.sft_trainer.TrainDefaults.get_sagemaker_session')
    @patch('sagemaker.train.sft_trainer._get_unique_name')
    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.sft_trainer._create_input_data_config')
    @patch('sagemaker.train.sft_trainer._convert_input_data_to_channels')
    @patch('sagemaker.train.sft_trainer._create_output_config')
    @patch('sagemaker.train.sft_trainer._create_mlflow_config')
    @patch('sagemaker.train.sft_trainer._create_model_package_config')
    @patch('sagemaker.core.resources.TrainingJob.create')
    def test_peft_value_for_full_training(self, mock_training_job_create, mock_model_package_config, mock_mlflow_config, mock_output_config, mock_convert_channels, mock_input_config, mock_validate_group, mock_unique_name, mock_get_sagemaker_session, mock_get_role,
                                        mock_get_options, mock_resolve_model, mock_get_session):
        # Mock all utility functions
        mock_validate_group.return_value = "test-group"
        mock_resolve_model.return_value = ("test-model", "test-model")
        mock_get_session.return_value = Mock()
        mock_get_sagemaker_session.return_value = Mock()
        
        mock_fine_tuning_options = Mock()
        mock_fine_tuning_options.to_dict.return_value = {"learning_rate": "0.001"}
        mock_get_options.return_value = (mock_fine_tuning_options, "model-arn", False)
        
        mock_get_role.return_value = "test-role"
        mock_unique_name.return_value = "test-job-name"
        
        mock_input_config.return_value = [Mock()]
        mock_convert_channels.return_value = [Mock()]
        mock_output_config.return_value = Mock()
        mock_mlflow_config.return_value = Mock()
        mock_model_package_config.return_value = Mock()
        
        mock_training_job = Mock()
        mock_training_job.arn = "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        mock_training_job.wait = Mock()
        mock_training_job_create.return_value = mock_training_job
        
        trainer = SFTTrainer(model="test-model", training_type=TrainingType.FULL, model_package_group="test-group", training_dataset="s3://bucket/train")
        trainer.train(wait=False)
        
        assert mock_training_job_create.called

    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    def test_training_type_string_value(self, mock_finetuning_options, mock_validate_group, mock_session):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        trainer = SFTTrainer(model="test-model", training_type="CUSTOM", model_package_group="test-group")
        assert trainer.training_type == "CUSTOM"

    @patch('sagemaker.train.sft_trainer._resolve_model_and_name')
    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    def test_model_package_input(self, mock_finetuning_options, mock_validate_group, mock_resolve_model):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        
        model_package = Mock(spec=ModelPackage)
        model_package.inference_specification = Mock()
        
        # Make _resolve_model_and_name return the same model_package object
        mock_resolve_model.return_value = (model_package, "test-model")
        
        trainer = SFTTrainer(model=model_package)
        assert trainer.model == model_package

    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    def test_init_with_datasets(self, mock_finetuning_options, mock_validate_group, mock_session):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        trainer = SFTTrainer(
            model="test-model",
            model_package_group="test-group",
            training_dataset="s3://bucket/train",
            validation_dataset="s3://bucket/val"
        )
        assert trainer.training_dataset == "s3://bucket/train"
        assert trainer.validation_dataset == "s3://bucket/val"

    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    def test_init_with_mlflow_config(self, mock_finetuning_options, mock_validate_group, mock_session):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        trainer = SFTTrainer(
            model="test-model",
            model_package_group="test-group",
            mlflow_resource_arn="arn:aws:mlflow:us-east-1:123456789012:tracking-server/test",
            mlflow_experiment_name="test-experiment",
            mlflow_run_name="test-run"
        )
        assert trainer.mlflow_resource_arn == "arn:aws:mlflow:us-east-1:123456789012:tracking-server/test"
        assert trainer.mlflow_experiment_name == "test-experiment"
        assert trainer.mlflow_run_name == "test-run"

    @patch('sagemaker.train.common_utils.finetune_utils._get_beta_session')
    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    def test_fit_without_datasets_raises_error(self, mock_finetuning_options, mock_validate_group, mock_get_session):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        mock_get_session.return_value = Mock()
        trainer = SFTTrainer(model="test-model", model_package_group="test-group")
        
        with pytest.raises(Exception):
            trainer.train(wait=False)

    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group')
    def test_model_package_group_handling(self, mock_validate_group, mock_get_options):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_get_options.return_value = (mock_hyperparams, "model-arn", False)
        
        trainer = SFTTrainer(
            model="test-model",
            model_package_group="test-group"
        )
        assert trainer.model_package_group == "test-group"

    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    def test_s3_output_path_configuration(self, mock_finetuning_options, mock_validate_group, mock_session):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        trainer = SFTTrainer(
            model="test-model",
            model_package_group="test-group",
            s3_output_path="s3://bucket/output"
        )
        assert trainer.s3_output_path == "s3://bucket/output"

    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    def test_gated_model_eula_validation(self, mock_finetuning_options, mock_validate_group, mock_session):
        """Test EULA validation for gated models"""
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", True)  # is_gated_model=True
        
        # Should raise error when accept_eula=False for gated model
        with pytest.raises(ValueError, match="gated model and requires EULA acceptance"):
            SFTTrainer(model="gated-model", model_package_group="test-group", accept_eula=False)
        
        # Should work when accept_eula=True for gated model
        trainer = SFTTrainer(model="gated-model", model_package_group="test-group", accept_eula=True)
        assert trainer.accept_eula == True


    @patch('sagemaker.train.sft_trainer._resolve_model_and_name')
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.sft_trainer.TrainDefaults.get_role')
    @patch('sagemaker.train.sft_trainer.TrainDefaults.get_sagemaker_session')
    @patch('sagemaker.train.sft_trainer._get_unique_name')
    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.sft_trainer._create_input_data_config')
    @patch('sagemaker.train.sft_trainer._convert_input_data_to_channels')
    @patch('sagemaker.train.sft_trainer._create_output_config')
    @patch('sagemaker.train.sft_trainer._create_mlflow_config')
    @patch('sagemaker.train.sft_trainer._create_model_package_config')
    @patch('sagemaker.core.resources.TrainingJob.create')
    def test_train_with_tags(self, mock_training_job_create, mock_model_package_config, 
                            mock_mlflow_config, mock_output_config, mock_convert_channels, mock_input_config, 
                            mock_validate_group, mock_unique_name, mock_get_sagemaker_session, mock_get_role, 
                            mock_get_options, mock_resolve_model):
        mock_validate_group.return_value = "test-group"
        mock_resolve_model.return_value = ("test-model", "test-model")
        mock_get_sagemaker_session.return_value = Mock()
        mock_fine_tuning_options = Mock()
        mock_fine_tuning_options.to_dict.return_value = {"learning_rate": "0.001"}
        mock_get_options.return_value = (mock_fine_tuning_options, "model-arn", False)
        mock_get_role.return_value = "test-role"
        mock_unique_name.return_value = "test-job-name"
        mock_input_config.return_value = [Mock()]
        mock_convert_channels.return_value = [Mock()]
        mock_output_config.return_value = Mock()
        mock_mlflow_config.return_value = Mock()
        mock_model_package_config.return_value = Mock()
        mock_training_job = Mock()
        mock_training_job.arn = "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        mock_training_job.wait = Mock()
        mock_training_job_create.return_value = mock_training_job
        
        trainer = SFTTrainer(model="test-model", model_package_group="test-group", training_dataset="s3://bucket/train")
        trainer.train(wait=False)
        
        mock_training_job_create.assert_called_once()
        call_kwargs = mock_training_job_create.call_args[1]
        assert call_kwargs["tags"] == [
            {"key": "sagemaker-studio:jumpstart-model-id", "value": "test-model"},
            {"key": "sagemaker-studio:jumpstart-hub-name", "value": "SageMakerPublicHub"}
        ]

    def test_process_hyperparameters_removes_constructor_handled_keys(self):
        """Test that _process_hyperparameters removes keys handled by constructor inputs."""
        # Create mock hyperparameters with all possible keys
        mock_hyperparams = Mock()
        mock_hyperparams._specs = {
            'data_path': 'test_data_path',
            'output_path': 'test_output_path', 
            'training_data_name': 'test_training_data_name',
            'validation_data_name': 'test_validation_data_name',
            'validation_data_path': 'test_validation_data_path',
            'other_param': 'should_remain'
        }
        
        # Add attributes to mock
        mock_hyperparams.data_path = 'test_data_path'
        mock_hyperparams.output_path = 'test_output_path'
        mock_hyperparams.training_data_name = 'test_training_data_name'
        mock_hyperparams.validation_data_name = 'test_validation_data_name'
        mock_hyperparams.validation_data_path = 'test_validation_data_path'
        
        # Create trainer instance with mock hyperparameters
        trainer = SFTTrainer.__new__(SFTTrainer)
        trainer.hyperparameters = mock_hyperparams
        
        # Call the method
        trainer._process_hyperparameters()
        
        # Verify attributes were removed
        assert not hasattr(mock_hyperparams, 'data_path')
        assert not hasattr(mock_hyperparams, 'output_path')
        assert not hasattr(mock_hyperparams, 'training_data_name')
        assert not hasattr(mock_hyperparams, 'validation_data_name')
        assert not hasattr(mock_hyperparams, 'validation_data_path')
        
        # Verify _specs were updated
        assert 'data_path' not in mock_hyperparams._specs
        assert 'output_path' not in mock_hyperparams._specs
        assert 'training_data_name' not in mock_hyperparams._specs
        assert 'validation_data_name' not in mock_hyperparams._specs
        assert 'validation_data_path' not in mock_hyperparams._specs
        assert 'other_param' in mock_hyperparams._specs

    def test_process_hyperparameters_handles_missing_attributes(self):
        """Test that _process_hyperparameters handles missing attributes gracefully."""
        # Create mock hyperparameters with only some keys
        mock_hyperparams = Mock()
        mock_hyperparams._specs = {
            'data_path': 'test_data_path',
            'other_param': 'should_remain'
        }
        mock_hyperparams.data_path = 'test_data_path'
        
        # Create trainer instance
        trainer = SFTTrainer.__new__(SFTTrainer)
        trainer.hyperparameters = mock_hyperparams
        
        # Call the method
        trainer._process_hyperparameters()
        
        # Verify only existing attributes were processed
        assert not hasattr(mock_hyperparams, 'data_path')
        assert 'data_path' not in mock_hyperparams._specs
        assert 'other_param' in mock_hyperparams._specs

    def test_process_hyperparameters_with_none_hyperparameters(self):
        """Test that _process_hyperparameters handles None hyperparameters."""
        trainer = SFTTrainer.__new__(SFTTrainer)
        trainer.hyperparameters = None
        
        # Should not raise an exception
        trainer._process_hyperparameters()

    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    def test_accepts_stopping_condition(self, mock_finetuning, mock_validate):
        """Test SFTTrainer accepts stopping_condition parameter."""
        from sagemaker.train.configs import StoppingCondition
        
        mock_validate.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning.return_value = (mock_hyperparams, "model-arn", False)
        
        stopping_condition = StoppingCondition(max_runtime_in_seconds=7200)
        trainer = SFTTrainer(
            model="test-model",
            model_package_group="test-group",
            stopping_condition=stopping_condition
        )
        
        assert trainer.stopping_condition == stopping_condition
        assert trainer.stopping_condition.max_runtime_in_seconds == 7200

    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    def test_default_stopping_condition_is_none(self, mock_finetuning, mock_validate):
        """Test SFTTrainer defaults stopping_condition to None."""
        mock_validate.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning.return_value = (mock_hyperparams, "model-arn", False)
        
        trainer = SFTTrainer(model="test-model", model_package_group="test-group")
        assert trainer.stopping_condition is None
