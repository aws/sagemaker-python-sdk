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

    @patch('sagemaker.train.common_utils.trainer_wait.wait')
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
    def test_train_passes_wait_timeout(self, mock_training_job_create, mock_model_package_config,
                                       mock_mlflow_config, mock_output_config, mock_convert_channels,
                                       mock_input_config, mock_validate_group, mock_unique_name,
                                       mock_get_sagemaker_session, mock_get_role, mock_get_options,
                                       mock_resolve_model, mock_get_session, mock_wait):
        """Test that wait_timeout is passed to _wait as timeout kwarg."""
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
        mock_training_job_create.return_value = mock_training_job

        trainer = SFTTrainer(model="test-model", model_package_group="test-group", training_dataset="s3://bucket/train")
        trainer.train(wait=True, wait_timeout=600)

        mock_wait.assert_called_once_with(mock_training_job, timeout=600, poll=5)

    @patch('sagemaker.train.common_utils.trainer_wait.wait')
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
    def test_train_without_wait_timeout_uses_default(self, mock_training_job_create, mock_model_package_config,
                                                      mock_mlflow_config, mock_output_config, mock_convert_channels,
                                                      mock_input_config, mock_validate_group, mock_unique_name,
                                                      mock_get_sagemaker_session, mock_get_role, mock_get_options,
                                                      mock_resolve_model, mock_get_session, mock_wait):
        """Test that _wait is called without timeout kwarg when wait_timeout is None."""
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
        mock_training_job_create.return_value = mock_training_job

        trainer = SFTTrainer(model="test-model", model_package_group="test-group", training_dataset="s3://bucket/train")
        trainer.train(wait=True)

        mock_wait.assert_called_once_with(mock_training_job, poll=5)

    @patch('sagemaker.train.common_utils.trainer_wait.wait')
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
    def test_train_wait_false_skips_wait(self, mock_training_job_create, mock_model_package_config,
                                         mock_mlflow_config, mock_output_config, mock_convert_channels,
                                         mock_input_config, mock_validate_group, mock_unique_name,
                                         mock_get_sagemaker_session, mock_get_role, mock_get_options,
                                         mock_resolve_model, mock_get_session, mock_wait):
        """Test that _wait is not called when wait=False."""
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
        mock_training_job_create.return_value = mock_training_job

        trainer = SFTTrainer(model="test-model", model_package_group="test-group", training_dataset="s3://bucket/train")
        trainer.train(wait=False, wait_timeout=600)

        mock_wait.assert_not_called()


class TestSFTTrainerComputeDispatch:
    """Tests for compute dispatch in SFTTrainer."""

    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.sft_trainer._resolve_model_and_name')
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    def _make_trainer(self, mock_opts, mock_resolve, mock_validate, compute=None):
        from sagemaker.train.sft_trainer import SFTTrainer
        from sagemaker.core.training.configs import Compute, HyperPodCompute
        mock_resolve.return_value = ("model", "nova-textgeneration-lite-v2")
        mock_validate.return_value = "group"
        mock_hp = Mock()
        mock_hp.to_dict.return_value = {}
        mock_opts.return_value = (mock_hp, "arn", False)
        return SFTTrainer(model="amazon.nova-lite-v2", compute=compute, model_package_group="grp")

    def test_rejects_invalid_compute_type(self):
        with pytest.raises(TypeError, match="Compute or HyperPodCompute"):
            self._make_trainer(compute="invalid")

    def test_accepts_none_compute(self):
        trainer = self._make_trainer(compute=None)
        assert trainer.compute is None

    def test_accepts_compute_instance(self):
        from sagemaker.core.training.configs import Compute
        compute = Compute(instance_type="ml.p5.48xlarge", instance_count=4)
        trainer = self._make_trainer(compute=compute)
        assert trainer.compute is compute

    def test_accepts_hyperpod_compute(self):
        from sagemaker.core.training.configs import HyperPodCompute
        compute = HyperPodCompute(cluster_name="my-cluster", instance_type="ml.p5.48xlarge")
        trainer = self._make_trainer(compute=compute)
        assert trainer.compute is compute

    def test_none_routes_to_serverless(self):
        trainer = self._make_trainer(compute=None)
        # The serverless path is inlined in train(); verify routing by ensuring
        # neither compute-backed method is called and the serverless branch is
        # entered (it begins by resolving the SageMaker session).
        with patch.object(trainer, '_train_serverful_smtj') as mock_smtj, \
             patch.object(trainer, '_train_hyperpod') as mock_hp, \
             patch(
                 'sagemaker.train.defaults.TrainDefaults.get_sagemaker_session',
                 side_effect=RuntimeError('serverless-path-reached'),
             ):
            with pytest.raises(RuntimeError, match='serverless-path-reached'):
                trainer.train(training_dataset="s3://bucket/data.jsonl", wait=False)
            mock_smtj.assert_not_called()
            mock_hp.assert_not_called()

    def test_compute_routes_to_smtj(self):
        from sagemaker.core.training.configs import Compute
        compute = Compute(instance_type="ml.p5.48xlarge", instance_count=4)
        trainer = self._make_trainer(compute=compute)
        with patch.object(trainer, '_train_serverful_smtj', return_value=Mock()) as mock_smtj:
            trainer.train(training_dataset="s3://bucket/data.jsonl", wait=False)
            mock_smtj.assert_called_once()

    def test_hyperpod_routes_to_hyperpod(self):
        from sagemaker.core.training.configs import HyperPodCompute
        compute = HyperPodCompute(cluster_name="my-cluster", instance_type="ml.p5.48xlarge")
        trainer = self._make_trainer(compute=compute)
        with patch.object(trainer, '_train_hyperpod', return_value="job-name") as mock_hp:
            trainer.train(training_dataset="s3://bucket/data.jsonl", wait=False)
            mock_hp.assert_called_once()


class TestSFTTrainerDataMixingIntegration:
    """Unit tests for SFTTrainer data mixing integration."""

    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    def test_sft_trainer_accepts_data_mixing_config(self, mock_finetuning_options, mock_validate_group):
        """Test SFTTrainer constructor accepts data_mixing_config parameter."""
        from sagemaker.train.data_mixing_config import DataMixingConfig

        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)

        config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages={"code": 60.0, "math": 40.0},
        )
        trainer = SFTTrainer(
            model="nova-textgeneration-micro",
            model_package_group="test-group",
            data_mixing_config=config,
        )
        assert trainer.data_mixing_config is config

    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    def test_sft_trainer_data_mixing_config_defaults_to_none(self, mock_finetuning_options, mock_validate_group):
        """Test SFTTrainer defaults data_mixing_config to None when not provided."""
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)

        trainer = SFTTrainer(model="test-model", model_package_group="test-group")
        assert trainer.data_mixing_config is None

    @patch('sagemaker.train.sft_trainer._validate_hyperparameter_values')
    @patch('sagemaker.train.sft_trainer.resolve_datamix_recipe')
    @patch('sagemaker.train.sft_trainer._resolve_model_and_name')
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.sft_trainer.TrainDefaults.get_role')
    @patch('sagemaker.train.sft_trainer.TrainDefaults.get_sagemaker_session')
    @patch('sagemaker.train.sft_trainer._get_unique_name')
    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.sft_trainer._create_input_data_config')
    @patch('sagemaker.train.sft_trainer._convert_input_data_to_channels')
    @patch('sagemaker.train.sft_trainer._create_output_config')
    @patch('sagemaker.train.sft_trainer._create_serverless_config')
    @patch('sagemaker.train.sft_trainer._create_mlflow_config')
    @patch('sagemaker.train.sft_trainer._create_model_package_config')
    @patch('sagemaker.core.resources.TrainingJob.create')
    def test_train_includes_serialized_data_mixing_config_in_hyperparameters(
        self,
        mock_training_job_create,
        mock_model_package_config,
        mock_mlflow_config,
        mock_serverless_config,
        mock_output_config,
        mock_convert_channels,
        mock_input_config,
        mock_validate_group,
        mock_unique_name,
        mock_get_sagemaker_session,
        mock_get_role,
        mock_get_options,
        mock_resolve_model,
        mock_resolve_datamix_recipe,
        mock_validate_hp_values,
    ):
        """Test train() includes serialized data mixing config in hyperparameters."""
        from sagemaker.train.data_mixing_config import DataMixingConfig

        mock_validate_group.return_value = "test-group"
        mock_resolve_model.return_value = ("nova-textgeneration-micro", "nova-textgeneration-micro")

        mock_session = Mock()
        mock_session.boto_session = Mock()
        mock_session.boto_session.region_name = "us-east-1"
        mock_get_sagemaker_session.return_value = mock_session

        mock_fine_tuning_options = Mock()
        mock_fine_tuning_options.to_dict.return_value = {"learning_rate": "0.001"}
        mock_get_options.return_value = (mock_fine_tuning_options, "model-arn", False)

        mock_get_role.return_value = "test-role"
        mock_unique_name.return_value = "test-job-name"
        mock_input_config.return_value = [Mock()]
        mock_convert_channels.return_value = [Mock()]
        mock_output_config.return_value = Mock()
        # serverless_config=None means platform is SMHP
        mock_serverless_config.return_value = None
        mock_mlflow_config.return_value = Mock()
        mock_model_package_config.return_value = Mock()

        # Mock resolve_datamix_recipe to return recipe categories
        mock_resolve_datamix_recipe.return_value = {"code": 50.0, "math": 50.0}

        mock_training_job = Mock()
        mock_training_job.arn = "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        mock_training_job_create.return_value = mock_training_job

        config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages={"code": 60.0, "math": 40.0},
        )
        trainer = SFTTrainer(
            model="nova-textgeneration-micro",
            model_package_group="test-group",
            training_dataset="s3://bucket/train",
            data_mixing_config=config,
        )
        trainer.train(wait=False)

        mock_training_job_create.assert_called_once()
        call_kwargs = mock_training_job_create.call_args[1]
        hyper_params = call_kwargs["hyper_parameters"]
        assert hyper_params["customer_data_percent"] == "50"
        assert hyper_params["nova_code_percent"] == "60"
        assert hyper_params["nova_math_percent"] == "40"

    @patch('sagemaker.train.sft_trainer._resolve_model_and_name')
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.sft_trainer.TrainDefaults.get_role')
    @patch('sagemaker.train.sft_trainer.TrainDefaults.get_sagemaker_session')
    @patch('sagemaker.train.sft_trainer._get_unique_name')
    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.sft_trainer._create_input_data_config')
    @patch('sagemaker.train.sft_trainer._convert_input_data_to_channels')
    @patch('sagemaker.train.sft_trainer._create_output_config')
    @patch('sagemaker.train.sft_trainer._create_serverless_config')
    @patch('sagemaker.train.sft_trainer._create_mlflow_config')
    @patch('sagemaker.train.sft_trainer._create_model_package_config')
    def test_train_raises_valueerror_for_non_nova_model(
        self,
        mock_model_package_config,
        mock_mlflow_config,
        mock_serverless_config,
        mock_output_config,
        mock_convert_channels,
        mock_input_config,
        mock_validate_group,
        mock_unique_name,
        mock_get_sagemaker_session,
        mock_get_role,
        mock_get_options,
        mock_resolve_model,
    ):
        """Test train() raises ValueError when data_mixing_config is used with non-Nova model."""
        from sagemaker.train.data_mixing_config import DataMixingConfig

        mock_validate_group.return_value = "test-group"
        # Using a non-Nova model name
        mock_resolve_model.return_value = ("meta-llama-3", "meta-llama-3")

        mock_session = Mock()
        mock_session.boto_session = Mock()
        mock_session.boto_session.region_name = "us-east-1"
        mock_get_sagemaker_session.return_value = mock_session

        mock_fine_tuning_options = Mock()
        mock_fine_tuning_options.to_dict.return_value = {"learning_rate": "0.001"}
        mock_get_options.return_value = (mock_fine_tuning_options, "model-arn", False)

        mock_get_role.return_value = "test-role"
        mock_unique_name.return_value = "test-job-name"
        mock_input_config.return_value = [Mock()]
        mock_convert_channels.return_value = [Mock()]
        mock_output_config.return_value = Mock()
        # serverless_config=None means platform is SMHP (passes platform check)
        mock_serverless_config.return_value = None
        mock_mlflow_config.return_value = Mock()
        mock_model_package_config.return_value = Mock()

        config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages={"code": 60.0, "math": 40.0},
        )
        trainer = SFTTrainer(
            model="meta-llama-3",
            model_package_group="test-group",
            training_dataset="s3://bucket/train",
            data_mixing_config=config,
        )

        with pytest.raises(ValueError, match="Data mixing is only supported for Nova models"):
            trainer.train(wait=False)

    @patch('sagemaker.train.sft_trainer._resolve_model_and_name')
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.sft_trainer.TrainDefaults.get_role')
    @patch('sagemaker.train.sft_trainer.TrainDefaults.get_sagemaker_session')
    @patch('sagemaker.train.sft_trainer._get_unique_name')
    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.sft_trainer._create_input_data_config')
    @patch('sagemaker.train.sft_trainer._convert_input_data_to_channels')
    @patch('sagemaker.train.sft_trainer._create_output_config')
    @patch('sagemaker.train.sft_trainer._create_serverless_config')
    @patch('sagemaker.train.sft_trainer._create_mlflow_config')
    @patch('sagemaker.train.sft_trainer._create_model_package_config')
    @patch('sagemaker.core.resources.TrainingJob.create')
    def test_train_without_data_mixing_config_omits_data_mixing_from_request(
        self,
        mock_training_job_create,
        mock_model_package_config,
        mock_mlflow_config,
        mock_serverless_config,
        mock_output_config,
        mock_convert_channels,
        mock_input_config,
        mock_validate_group,
        mock_unique_name,
        mock_get_sagemaker_session,
        mock_get_role,
        mock_get_options,
        mock_resolve_model,
    ):
        """Test train() without data_mixing_config does not include data mixing in hyperparameters."""
        mock_validate_group.return_value = "test-group"
        mock_resolve_model.return_value = ("test-model", "test-model")

        mock_session = Mock()
        mock_session.boto_session = Mock()
        mock_session.boto_session.region_name = "us-east-1"
        mock_get_sagemaker_session.return_value = mock_session

        mock_fine_tuning_options = Mock()
        mock_fine_tuning_options.to_dict.return_value = {"learning_rate": "0.001"}
        mock_get_options.return_value = (mock_fine_tuning_options, "model-arn", False)

        mock_get_role.return_value = "test-role"
        mock_unique_name.return_value = "test-job-name"
        mock_input_config.return_value = [Mock()]
        mock_convert_channels.return_value = [Mock()]
        mock_output_config.return_value = Mock()
        mock_serverless_config.return_value = None
        mock_mlflow_config.return_value = Mock()
        mock_model_package_config.return_value = Mock()

        mock_training_job = Mock()
        mock_training_job.arn = "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        mock_training_job_create.return_value = mock_training_job

        trainer = SFTTrainer(
            model="test-model",
            model_package_group="test-group",
            training_dataset="s3://bucket/train",
        )
        trainer.train(wait=False)

        mock_training_job_create.assert_called_once()
        call_kwargs = mock_training_job_create.call_args[1]
        hyper_params = call_kwargs["hyper_parameters"]
        assert "customer_data_percent" not in hyper_params


class TestSFTTrainerHyperPodDatamixOrchestration:
    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.sft_trainer._resolve_model_and_name')
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    def _make_trainer(self, mock_opts, mock_resolve, mock_validate, training_type=TrainingType.LORA, data_mixing_config=None, training_image=None):
        """Helper to construct an SFTTrainer with HyperPodCompute."""
        from sagemaker.core.training.configs import HyperPodCompute
        from sagemaker.train.data_mixing_config import DataMixingConfig

        mock_resolve.return_value = ("amazon.nova-lite-v2", "nova-textgeneration-lite-v2")
        mock_validate.return_value = "test-group"
        mock_hp = Mock()
        mock_hp.to_dict.return_value = {}
        mock_opts.return_value = (mock_hp, "model-arn", False)

        compute = HyperPodCompute(cluster_name="my-cluster", instance_type="ml.p5.48xlarge")

        if data_mixing_config is None:
            data_mixing_config = DataMixingConfig(
                customer_data_percent=50.0,
                nova_data_percentages={"code": 60.0, "math": 40.0},
            )

        trainer = SFTTrainer(
            model="amazon.nova-lite-v2",
            training_type=training_type,
            model_package_group="test-group",
            compute=compute,
            data_mixing_config=data_mixing_config,
            training_dataset="s3://bucket/train",
        )
        if training_image is not None:
            trainer.training_image = training_image
        return trainer

    @patch('sagemaker.train.sft_trainer.build_hyperpod_datamix_recipe_from_context')
    @patch('sagemaker.train.sft_trainer.validate_data_mixing_categories')
    @patch('sagemaker.train.sft_trainer.resolve_hyperpod_datamix_context')
    @patch('sagemaker.train.sft_trainer.validate_data_mixing_model')
    def test_full_orchestration_call_order(
        self,
        mock_validate_model,
        mock_resolve,
        mock_validate_categories,
        mock_build,
    ):
        """Test that validate_data_mixing_model → resolve → validate_categories → build are called in order."""
        from sagemaker.train.data_mixing_config import DataMixingConfig

        # Setup mocks
        mock_context = Mock()
        mock_context.categories = {"code": 50.0, "math": 50.0}
        mock_resolve.return_value = mock_context

        mock_validated_config = Mock(spec=DataMixingConfig)
        mock_validate_categories.return_value = mock_validated_config

        mock_build.return_value = ("recipes/nova_sft_lora", "123456.dkr.ecr.us-east-1.amazonaws.com/image:latest")

        trainer = self._make_trainer()

        # Track call order
        call_order = []
        mock_validate_model.side_effect = lambda *args, **kwargs: call_order.append("validate_model")
        mock_resolve.side_effect = lambda *args, **kwargs: (call_order.append("resolve"), mock_context)[1]
        mock_validate_categories.side_effect = lambda *args, **kwargs: (call_order.append("validate_categories"), mock_validated_config)[1]
        mock_build.side_effect = lambda *args, **kwargs: (call_order.append("build"), ("recipes/nova_sft_lora", "123456.dkr.ecr.us-east-1.amazonaws.com/image:latest"))[1]

        with patch.object(trainer, '_train_hyperpod', return_value=Mock()):
            trainer.train(wait=False)

        assert call_order == ["validate_model", "resolve", "validate_categories", "build"]

    @patch('sagemaker.train.sft_trainer.build_hyperpod_datamix_recipe_from_context')
    @patch('sagemaker.train.sft_trainer.validate_data_mixing_categories')
    @patch('sagemaker.train.sft_trainer.resolve_hyperpod_datamix_context')
    @patch('sagemaker.train.sft_trainer.validate_data_mixing_model')
    def test_customization_technique_sft_passed_to_resolve(
        self,
        mock_validate_model,
        mock_resolve,
        mock_validate_categories,
        mock_build,
    ):
        """Test that customization_technique='SFT' is passed to resolve_hyperpod_datamix_context."""
        from sagemaker.train.data_mixing_config import DataMixingConfig

        mock_context = Mock()
        mock_context.categories = {"code": 50.0, "math": 50.0}
        mock_resolve.return_value = mock_context
        mock_validate_categories.return_value = Mock(spec=DataMixingConfig)
        mock_build.return_value = ("recipes/nova_sft_lora", None)

        trainer = self._make_trainer()

        with patch.object(trainer, '_train_hyperpod', return_value=Mock()):
            trainer.train(wait=False)

        mock_resolve.assert_called_once()
        call_kwargs = mock_resolve.call_args[1]
        assert call_kwargs["customization_technique"] == "SFT"

    @patch('sagemaker.train.sft_trainer.build_hyperpod_datamix_recipe_from_context')
    @patch('sagemaker.train.sft_trainer.validate_data_mixing_categories')
    @patch('sagemaker.train.sft_trainer.resolve_hyperpod_datamix_context')
    @patch('sagemaker.train.sft_trainer.validate_data_mixing_model')
    def test_training_type_lora_passed_to_resolve(
        self,
        mock_validate_model,
        mock_resolve,
        mock_validate_categories,
        mock_build,
    ):
        """Test that training_type='LORA' is derived from self.training_type=TrainingType.LORA."""
        from sagemaker.train.data_mixing_config import DataMixingConfig

        mock_context = Mock()
        mock_context.categories = {"code": 50.0, "math": 50.0}
        mock_resolve.return_value = mock_context
        mock_validate_categories.return_value = Mock(spec=DataMixingConfig)
        mock_build.return_value = ("recipes/nova_sft_lora", None)

        trainer = self._make_trainer(training_type=TrainingType.LORA)

        with patch.object(trainer, '_train_hyperpod', return_value=Mock()):
            trainer.train(wait=False)

        call_kwargs = mock_resolve.call_args[1]
        assert call_kwargs["training_type"] == "LORA"

    @patch('sagemaker.train.sft_trainer.build_hyperpod_datamix_recipe_from_context')
    @patch('sagemaker.train.sft_trainer.validate_data_mixing_categories')
    @patch('sagemaker.train.sft_trainer.resolve_hyperpod_datamix_context')
    @patch('sagemaker.train.sft_trainer.validate_data_mixing_model')
    def test_training_type_full_passed_to_resolve(
        self,
        mock_validate_model,
        mock_resolve,
        mock_validate_categories,
        mock_build,
    ):
        """Test that training_type='FULL' is derived from self.training_type=TrainingType.FULL."""
        from sagemaker.train.data_mixing_config import DataMixingConfig

        mock_context = Mock()
        mock_context.categories = {"code": 50.0, "math": 50.0}
        mock_resolve.return_value = mock_context
        mock_validate_categories.return_value = Mock(spec=DataMixingConfig)
        mock_build.return_value = ("recipes/nova_sft_full", None)

        trainer = self._make_trainer(training_type=TrainingType.FULL)

        with patch.object(trainer, '_train_hyperpod', return_value=Mock()):
            trainer.train(wait=False)

        call_kwargs = mock_resolve.call_args[1]
        assert call_kwargs["training_type"] == "FULL"

    @patch('sagemaker.train.sft_trainer.build_hyperpod_datamix_recipe_from_context')
    @patch('sagemaker.train.sft_trainer.validate_data_mixing_categories')
    @patch('sagemaker.train.sft_trainer.resolve_hyperpod_datamix_context')
    @patch('sagemaker.train.sft_trainer.validate_data_mixing_model')
    def test_recipe_path_set_from_build_result(
        self,
        mock_validate_model,
        mock_resolve,
        mock_validate_categories,
        mock_build,
    ):
        """Test that self._recipe_path is set to the relative_recipe_path from build."""
        from sagemaker.train.data_mixing_config import DataMixingConfig

        mock_context = Mock()
        mock_context.categories = {"code": 50.0, "math": 50.0}
        mock_resolve.return_value = mock_context
        mock_validate_categories.return_value = Mock(spec=DataMixingConfig)
        mock_build.return_value = ("recipes/nova_pro_sft_lora_datamix", "ecr-image:latest")

        trainer = self._make_trainer()

        with patch.object(trainer, '_train_hyperpod', return_value=Mock()):
            trainer.train(wait=False)

        assert trainer._recipe_path == "recipes/nova_pro_sft_lora_datamix"

    @patch('sagemaker.train.sft_trainer.build_hyperpod_datamix_recipe_from_context')
    @patch('sagemaker.train.sft_trainer.validate_data_mixing_categories')
    @patch('sagemaker.train.sft_trainer.resolve_hyperpod_datamix_context')
    @patch('sagemaker.train.sft_trainer.validate_data_mixing_model')
    def test_training_image_set_when_image_uri_not_none_and_not_already_set(
        self,
        mock_validate_model,
        mock_resolve,
        mock_validate_categories,
        mock_build,
    ):
        """Test that self.training_image is set when image_uri is not None and not already set."""
        from sagemaker.train.data_mixing_config import DataMixingConfig

        mock_context = Mock()
        mock_context.categories = {"code": 50.0, "math": 50.0}
        mock_resolve.return_value = mock_context
        mock_validate_categories.return_value = Mock(spec=DataMixingConfig)
        expected_image = "123456.dkr.ecr.us-east-1.amazonaws.com/training:v1"
        mock_build.return_value = ("recipes/nova_sft_lora", expected_image)

        # training_image is None by default (not already set)
        trainer = self._make_trainer(training_image=None)

        with patch.object(trainer, '_train_hyperpod', return_value=Mock()):
            trainer.train(wait=False)

        assert trainer.training_image == expected_image

    @patch('sagemaker.train.sft_trainer.build_hyperpod_datamix_recipe_from_context')
    @patch('sagemaker.train.sft_trainer.validate_data_mixing_categories')
    @patch('sagemaker.train.sft_trainer.resolve_hyperpod_datamix_context')
    @patch('sagemaker.train.sft_trainer.validate_data_mixing_model')
    def test_training_image_not_overwritten_when_already_set(
        self,
        mock_validate_model,
        mock_resolve,
        mock_validate_categories,
        mock_build,
    ):
        """Test that self.training_image is not overwritten if already set."""
        from sagemaker.train.data_mixing_config import DataMixingConfig

        mock_context = Mock()
        mock_context.categories = {"code": 50.0, "math": 50.0}
        mock_resolve.return_value = mock_context
        mock_validate_categories.return_value = Mock(spec=DataMixingConfig)
        mock_build.return_value = ("recipes/nova_sft_lora", "new-image-from-helm:latest")

        existing_image = "user-provided-image:v1"
        trainer = self._make_trainer(training_image=existing_image)

        with patch.object(trainer, '_train_hyperpod', return_value=Mock()):
            trainer.train(wait=False)

        # Should retain the existing user-provided image, not overwrite
        assert trainer.training_image == existing_image

    @patch('sagemaker.train.sft_trainer.build_hyperpod_datamix_recipe_from_context')
    @patch('sagemaker.train.sft_trainer.validate_data_mixing_categories')
    @patch('sagemaker.train.sft_trainer.resolve_hyperpod_datamix_context')
    @patch('sagemaker.train.sft_trainer.validate_data_mixing_model')
    def test_training_image_not_set_when_image_uri_is_none(
        self,
        mock_validate_model,
        mock_resolve,
        mock_validate_categories,
        mock_build,
    ):
        """Test that self.training_image is not set when image_uri from build is None."""
        from sagemaker.train.data_mixing_config import DataMixingConfig

        mock_context = Mock()
        mock_context.categories = {"code": 50.0, "math": 50.0}
        mock_resolve.return_value = mock_context
        mock_validate_categories.return_value = Mock(spec=DataMixingConfig)
        mock_build.return_value = ("recipes/nova_sft_lora", None)

        trainer = self._make_trainer(training_image=None)

        with patch.object(trainer, '_train_hyperpod', return_value=Mock()):
            trainer.train(wait=False)

        # training_image should remain None since image_uri from build was None
        assert trainer.training_image is None


class TestSFTTrainerSmtjS3DataType:
    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.sft_trainer._resolve_model_and_name')
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    def _make_trainer(self, mock_opts, mock_resolve, mock_validate, model_name="nova-textgeneration-lite", compute=None):
        from sagemaker.train.sft_trainer import SFTTrainer
        from sagemaker.core.training.configs import Compute
        mock_resolve.return_value = ("model", model_name)
        mock_validate.return_value = "group"
        mock_hp = Mock()
        mock_hp.to_dict.return_value = {}
        mock_hp._user_set = []
        mock_opts.return_value = (mock_hp, "arn", False)
        if compute is None:
            compute = Compute(instance_type="ml.p5.48xlarge", instance_count=4)
        return SFTTrainer(model=model_name, compute=compute, model_package_group="grp")

    def _run_smtj_and_capture_input_config(self, model_name):
        trainer = self._make_trainer(model_name=model_name)

        mock_model_trainer_cls = Mock()
        mock_model_trainer_cls.from_recipe.return_value = Mock()

        mock_open = MagicMock()
        mock_open.return_value.__enter__ = Mock(return_value=MagicMock(read=Mock(return_value="dummy: recipe")))
        mock_open.return_value.__exit__ = Mock(return_value=False)

        mock_s3_client = Mock()
        mock_s3_client.download_file = Mock()

        mock_session = Mock()
        mock_session.boto_session.region_name = "us-east-1"
        mock_session.boto_session.client.return_value = mock_s3_client

        with patch('sagemaker.train.defaults.TrainDefaults') as mock_defaults, \
             patch('sagemaker.train.common_utils.finetune_utils.get_recipe_s3_uri', return_value="s3://bucket/recipe.yaml"), \
             patch('sagemaker.train.common_utils.finetune_utils.get_training_image', return_value="img:latest"), \
             patch('sagemaker.train.common_utils.finetune_utils._get_smtj_override_spec', return_value={}), \
             patch('sagemaker.train.common_utils.finetune_utils._render_recipe_placeholders', return_value="content"), \
             patch('tempfile.NamedTemporaryFile') as mock_tmp, \
             patch('sagemaker.train.base_trainer.open', mock_open, create=True), \
             patch.dict('sys.modules', {'sagemaker.train.model_trainer': Mock(ModelTrainer=mock_model_trainer_cls)}):

            mock_defaults.get_sagemaker_session.return_value = mock_session
            mock_defaults.get_role.return_value = "arn:aws:iam::123456789012:role/test"
            mock_tmp.return_value.name = "/tmp/mock_recipe.yaml"

            trainer._train_serverful_smtj(training_dataset="s3://bucket/train.jsonl", wait=False)

        return mock_model_trainer_cls.from_recipe.call_args[1]["input_data_config"]

    def test_nova_sft_uses_converse_s3_data_type(self):
        input_data_config = self._run_smtj_and_capture_input_config("nova-textgeneration-lite")
        assert input_data_config is not None
        assert input_data_config[0].data_source.s3_data_type == "Converse"

    def test_oss_model_uses_s3prefix_data_type(self):
        input_data_config = self._run_smtj_and_capture_input_config("meta-textgeneration-llama-3-2-1b-instruct")
        assert input_data_config is not None
        assert input_data_config[0].data_source.s3_data_type == "S3Prefix"


class TestSFTTrainerBaseModelName:
    """Tests for base_model_name param and iterative training with S3 checkpoints."""

    @patch('sagemaker.train.sft_trainer._validate_eula_for_gated_model', return_value=False)
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group', return_value="my-group")
    @patch('sagemaker.train.sft_trainer._resolve_model_and_name', return_value=("model_obj", "nova-textgeneration-lite-v2"))
    def test_s3_model_with_base_model_name(self, mock_resolve, mock_validate_group, mock_get_options, mock_eula):
        """When model is S3 URI with base_model_name, model_source is set."""
        from sagemaker.core.training.configs import HyperPodCompute

        mock_hp = Mock()
        mock_hp.to_dict.return_value = {}
        mock_hp._specs = {}
        mock_hp._user_set = None
        mock_get_options.return_value = (mock_hp, "model-arn", False)

        trainer = SFTTrainer(
            model="s3://bucket/checkpoint/step_10",
            base_model_name="amazon.nova-2-lite-v1",
            compute=HyperPodCompute(cluster_name="my-cluster", node_count=4),
            training_dataset="s3://bucket/train.jsonl",
        )

        assert trainer.model_source == "s3://bucket/checkpoint/step_10"
        assert trainer._model_name == "nova-textgeneration-lite-v2"

    @patch('sagemaker.train.sft_trainer._validate_eula_for_gated_model', return_value=False)
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group', return_value="my-group")
    @patch('sagemaker.train.sft_trainer._resolve_model_and_name', return_value=("model_obj", "nova-textgeneration-lite-v2"))
    def test_s3_model_without_base_model_name_raises(self, mock_resolve, mock_validate_group, mock_get_options, mock_eula):
        """When model is S3 URI without base_model_name, ValueError is raised."""
        from sagemaker.core.training.configs import HyperPodCompute

        mock_hp = Mock()
        mock_hp.to_dict.return_value = {}
        mock_get_options.return_value = (mock_hp, "model-arn", False)

        with pytest.raises(ValueError, match="base_model_name is required"):
            SFTTrainer(
                model="s3://bucket/checkpoint/step_10",
                compute=HyperPodCompute(cluster_name="my-cluster", node_count=4),
                training_dataset="s3://bucket/train.jsonl",
            )

    @patch('sagemaker.train.sft_trainer._validate_eula_for_gated_model', return_value=False)
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group', return_value="my-group")
    @patch('sagemaker.train.sft_trainer._resolve_model_and_name', return_value=("model_obj", "nova-textgeneration-lite-v2"))
    def test_s3_model_without_compute_raises(self, mock_resolve, mock_validate_group, mock_get_options, mock_eula):
        """When model is S3 URI without compute, ValueError is raised."""
        mock_hp = Mock()
        mock_hp.to_dict.return_value = {}
        mock_get_options.return_value = (mock_hp, "model-arn", False)

        with pytest.raises(ValueError, match="only supported with HyperPodCompute"):
            SFTTrainer(
                model="s3://bucket/checkpoint/step_10",
                base_model_name="amazon.nova-2-lite-v1",
                training_dataset="s3://bucket/train.jsonl",
            )

    @patch('sagemaker.train.sft_trainer._validate_eula_for_gated_model', return_value=False)
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group', return_value="my-group")
    @patch('sagemaker.train.sft_trainer._resolve_model_and_name', return_value=("model_obj", "nova-textgeneration-lite-v2"))
    def test_normal_model_name_sets_no_model_source(self, mock_resolve, mock_validate_group, mock_get_options, mock_eula):
        """When model is a name (not S3), model_source is None."""
        mock_hp = Mock()
        mock_hp.to_dict.return_value = {}
        mock_hp._specs = {}
        mock_hp._user_set = None
        mock_get_options.return_value = (mock_hp, "model-arn", False)

        trainer = SFTTrainer(
            model="amazon.nova-2-lite-v1",
            model_package_group="my-group",
            training_dataset="s3://bucket/train.jsonl",
        )

        assert trainer.model_source is None

    @patch('sagemaker.train.sft_trainer._validate_eula_for_gated_model', return_value=False)
    @patch('sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.sft_trainer._validate_and_resolve_model_package_group', return_value="my-group")
    @patch('sagemaker.train.sft_trainer._resolve_model_and_name', return_value=("model_obj", "nova-textgeneration-lite-v2"))
    def test_disable_output_compression_stored(self, mock_resolve, mock_validate_group, mock_get_options, mock_eula):
        """disable_output_compression is stored on the trainer."""
        mock_hp = Mock()
        mock_hp.to_dict.return_value = {}
        mock_hp._specs = {}
        mock_hp._user_set = None
        mock_get_options.return_value = (mock_hp, "model-arn", False)

        trainer = SFTTrainer(
            model="amazon.nova-2-lite-v1",
            model_package_group="my-group",
            disable_output_compression=True,
        )

        assert trainer.disable_output_compression is True
