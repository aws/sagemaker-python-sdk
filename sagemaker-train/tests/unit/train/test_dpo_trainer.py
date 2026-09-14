import pytest
from unittest.mock import Mock, patch
from sagemaker.train.dpo_trainer import DPOTrainer
from sagemaker.train.common import TrainingType
from sagemaker.core.resources import ModelPackage


class TestDPOTrainer:
    
    @pytest.fixture
    def mock_session(self):
        session = Mock()
        session.region_name = "us-east-1"
        return session

    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    def test_init_with_defaults(self, mock_finetuning_options, mock_validate_group, mock_session):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        trainer = DPOTrainer(model="test-model", model_package_group="test-group")
        assert trainer.training_type == TrainingType.LORA
        assert trainer.model == "test-model"

    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    def test_init_with_full_training_type(self, mock_finetuning_options, mock_validate_group, mock_session):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        trainer = DPOTrainer(model="test-model", training_type=TrainingType.FULL, model_package_group="test-group")
        assert trainer.training_type == TrainingType.FULL

    @patch('sagemaker.train.dpo_trainer._resolve_model_and_name')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.dpo_trainer.TrainDefaults.get_role')
    @patch('sagemaker.train.dpo_trainer.TrainDefaults.get_sagemaker_session')
    @patch('sagemaker.train.dpo_trainer._get_unique_name')
    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._create_input_data_config')
    @patch('sagemaker.train.dpo_trainer._convert_input_data_to_channels')
    @patch('sagemaker.train.dpo_trainer._create_output_config')
    @patch('sagemaker.train.dpo_trainer._create_serverless_config')
    @patch('sagemaker.train.dpo_trainer._create_mlflow_config')
    @patch('sagemaker.train.dpo_trainer._create_model_package_config')
    @patch('sagemaker.core.resources.TrainingJob.create')
    def test_train_with_lora(self, mock_training_job_create, mock_model_package_config, mock_mlflow_config, 
                            mock_serverless_config, mock_output_config, mock_convert_channels, mock_input_config, 
                            mock_validate_group, mock_unique_name, mock_get_sagemaker_session, mock_get_role, 
                            mock_get_options, mock_resolve_model):
        mock_validate_group.return_value = "test-group"
        mock_resolve_model.return_value = ("test-model", "test-model")
        
        mock_fine_tuning_options = Mock()
        mock_fine_tuning_options.to_dict.return_value = {"learning_rate": "0.001"}
        mock_get_options.return_value = (mock_fine_tuning_options, "model-arn", False)
        
        mock_get_role.return_value = "test-role"
        mock_unique_name.return_value = "test-job-name"
        mock_get_sagemaker_session.return_value = Mock(sagemaker_config={})
        
        mock_input_config.return_value = [Mock()]
        mock_convert_channels.return_value = [Mock()]
        mock_output_config.return_value = Mock()
        mock_serverless_config.return_value = Mock()
        mock_mlflow_config.return_value = Mock()
        mock_model_package_config.return_value = Mock()
        
        mock_training_job = Mock()
        mock_training_job.arn = "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        mock_training_job.wait = Mock()
        mock_training_job_create.return_value = mock_training_job
        
        trainer = DPOTrainer(model="test-model", training_type=TrainingType.LORA, model_package_group="test-group", training_dataset="s3://bucket/train")
        trainer.train(wait=False)
        
        assert mock_training_job_create.called

    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    def test_training_type_string_value(self, mock_finetuning_options, mock_validate_group, mock_session):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        trainer = DPOTrainer(model="test-model", training_type="CUSTOM", model_package_group="test-group")
        assert trainer.training_type == "CUSTOM"

    @patch('sagemaker.train.dpo_trainer._resolve_model_and_name')
    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    def test_model_package_input(self, mock_finetuning_options, mock_validate_group, mock_resolve_model):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        
        model_package = Mock(spec=ModelPackage)
        model_package.inference_specification = Mock()
        
        mock_resolve_model.return_value = (model_package, "test-model")
        
        trainer = DPOTrainer(model=model_package)
        assert trainer.model == model_package

    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    def test_init_with_datasets(self, mock_finetuning_options, mock_validate_group, mock_session):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        trainer = DPOTrainer(
            model="test-model",
            model_package_group="test-group",
            training_dataset="s3://bucket/train",
            validation_dataset="s3://bucket/val"
        )
        assert trainer.training_dataset == "s3://bucket/train"
        assert trainer.validation_dataset == "s3://bucket/val"

    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    def test_init_with_mlflow_config(self, mock_finetuning_options, mock_validate_group, mock_session):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        trainer = DPOTrainer(
            model="test-model",
            model_package_group="test-group",
            mlflow_resource_arn="arn:aws:mlflow:us-east-1:123456789012:tracking-server/test",
            mlflow_experiment_name="test-experiment",
            mlflow_run_name="test-run"
        )
        assert trainer.mlflow_resource_arn == "arn:aws:mlflow:us-east-1:123456789012:tracking-server/test"
        assert trainer.mlflow_experiment_name == "test-experiment"
        assert trainer.mlflow_run_name == "test-run"

    @patch('sagemaker.train.dpo_trainer._resolve_model_and_name')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.dpo_trainer.TrainDefaults.get_role')
    @patch('sagemaker.train.dpo_trainer.TrainDefaults.get_sagemaker_session')
    @patch('sagemaker.train.dpo_trainer._get_unique_name')
    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._create_input_data_config')
    @patch('sagemaker.train.dpo_trainer._convert_input_data_to_channels')
    @patch('sagemaker.train.dpo_trainer._create_output_config')
    @patch('sagemaker.train.dpo_trainer._create_serverless_config')
    @patch('sagemaker.train.dpo_trainer._create_mlflow_config')
    @patch('sagemaker.train.dpo_trainer._create_model_package_config')
    @patch('sagemaker.core.resources.TrainingJob.create')
    def test_train_with_full_training(self, mock_training_job_create, mock_model_package_config, mock_mlflow_config,
                                     mock_serverless_config, mock_output_config, mock_convert_channels, mock_input_config,
                                     mock_validate_group, mock_unique_name, mock_get_sagemaker_session, mock_get_role,
                                     mock_get_options, mock_resolve_model):
        mock_validate_group.return_value = "test-group"
        mock_resolve_model.return_value = ("test-model", "test-model")
        
        mock_fine_tuning_options = Mock()
        mock_fine_tuning_options.to_dict.return_value = {"learning_rate": "0.001"}
        mock_get_options.return_value = (mock_fine_tuning_options, "model-arn", False)
        
        mock_get_role.return_value = "test-role"
        mock_unique_name.return_value = "test-job-name"
        mock_get_sagemaker_session.return_value = Mock(sagemaker_config={})
        
        mock_input_config.return_value = [Mock()]
        mock_convert_channels.return_value = [Mock()]
        mock_output_config.return_value = Mock()
        mock_serverless_config.return_value = Mock()
        mock_mlflow_config.return_value = Mock()
        mock_model_package_config.return_value = Mock()
        
        mock_training_job = Mock()
        mock_training_job.arn = "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        mock_training_job.wait = Mock()
        mock_training_job_create.return_value = mock_training_job
        
        trainer = DPOTrainer(model="test-model", training_type=TrainingType.FULL, model_package_group="test-group", training_dataset="s3://bucket/train")
        trainer.train(wait=False)
        
        assert mock_training_job_create.called

    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    def test_fit_without_datasets_raises_error(self, mock_finetuning_options, mock_validate_group):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        trainer = DPOTrainer(model="test-model", model_package_group="test-group")
        
        with pytest.raises(Exception):
            trainer.train(wait=False)

    @patch('sagemaker.train.common_utils.finetune_utils._resolve_model_name')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    def test_model_package_group_handling(self, mock_validate_group, mock_get_options, mock_resolve_model):
        mock_validate_group.return_value = "test-group"
        mock_resolve_model.return_value = "resolved-model"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_get_options.return_value = (mock_hyperparams, "model-arn", False)
        
        trainer = DPOTrainer(
            model="test-model",
            model_package_group="test-group"
        )
        assert trainer.model_package_group == "test-group"

    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    def test_s3_output_path_configuration(self, mock_finetuning_options, mock_validate_group, mock_session):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        trainer = DPOTrainer(
            model="test-model",
            model_package_group="test-group",
            s3_output_path="s3://bucket/output"
        )
        assert trainer.s3_output_path == "s3://bucket/output"

    @patch('sagemaker.train.dpo_trainer._resolve_model_and_name')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.dpo_trainer.TrainDefaults.get_role')
    @patch('sagemaker.train.dpo_trainer.TrainDefaults.get_sagemaker_session')
    @patch('sagemaker.train.dpo_trainer._get_unique_name')
    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._create_input_data_config')
    @patch('sagemaker.train.dpo_trainer._convert_input_data_to_channels')
    @patch('sagemaker.train.dpo_trainer._create_output_config')
    @patch('sagemaker.train.dpo_trainer._create_serverless_config')
    @patch('sagemaker.train.dpo_trainer._create_mlflow_config')
    @patch('sagemaker.train.dpo_trainer._create_model_package_config')
    @patch('sagemaker.core.resources.TrainingJob.create')
    def test_train_with_tags(self, mock_training_job_create, mock_model_package_config, 
                            mock_mlflow_config, mock_serverless_config, mock_output_config, mock_convert_channels, 
                            mock_input_config, mock_validate_group, mock_unique_name, mock_get_sagemaker_session, 
                            mock_get_role, mock_get_options, mock_resolve_model):
        mock_validate_group.return_value = "test-group"
        mock_resolve_model.return_value = ("test-model", "test-model")
        mock_fine_tuning_options = Mock()
        mock_fine_tuning_options.to_dict.return_value = {"learning_rate": "0.001"}
        mock_get_options.return_value = (mock_fine_tuning_options, "model-arn", False)
        mock_get_role.return_value = "test-role"
        mock_unique_name.return_value = "test-job-name"
        mock_get_sagemaker_session.return_value = Mock(sagemaker_config={})
        mock_input_config.return_value = [Mock()]
        mock_convert_channels.return_value = [Mock()]
        mock_output_config.return_value = Mock()
        mock_serverless_config.return_value = Mock()
        mock_mlflow_config.return_value = Mock()
        mock_model_package_config.return_value = Mock()
        mock_training_job = Mock()
        mock_training_job.arn = "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        mock_training_job.wait = Mock()
        mock_training_job_create.return_value = mock_training_job
        
        trainer = DPOTrainer(model="test-model", model_package_group="test-group", training_dataset="s3://bucket/train")
        trainer.train(wait=False)
        
        mock_training_job_create.assert_called_once()
        call_kwargs = mock_training_job_create.call_args[1]
        assert call_kwargs["tags"] == [
            {"key": "sagemaker-sdk:jumpstart-model-id", "value": "test-model"},
            {"key": "sagemaker-sdk:jumpstart-hub-name", "value": "SageMakerPublicHub"}
        ]

    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    def test_gated_model_eula_validation(self, mock_finetuning_options, mock_validate_group, mock_session):
        """Test EULA validation for gated models"""
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", True)  # is_gated_model=True
        
        # Should raise error when accept_eula=False for gated model
        with pytest.raises(ValueError, match="gated model and requires EULA acceptance"):
            DPOTrainer(model="gated-model", model_package_group="test-group", accept_eula=False)
        
        # Should work when accept_eula=True for gated model
        trainer = DPOTrainer(model="gated-model", model_package_group="test-group", accept_eula=True)
        assert trainer.accept_eula == True

    def test_process_hyperparameters_removes_constructor_handled_keys(self):
        """Test that _process_hyperparameters removes keys handled by constructor inputs."""
        # Create mock hyperparameters with all possible keys
        mock_hyperparams = Mock()
        mock_hyperparams._specs = {
            'data_path': 'test_data_path',
            'output_path': 'test_output_path', 
            'training_data_name': 'test_training_data_name',
            'validation_data_name': 'test_validation_data_name',
            'other_param': 'should_remain'
        }
        
        # Add attributes to mock
        mock_hyperparams.data_path = 'test_data_path'
        mock_hyperparams.output_path = 'test_output_path'
        mock_hyperparams.training_data_name = 'test_training_data_name'
        mock_hyperparams.validation_data_name = 'test_validation_data_name'
        
        # Create trainer instance with mock hyperparameters
        trainer = DPOTrainer.__new__(DPOTrainer)
        trainer.hyperparameters = mock_hyperparams
        
        # Call the method
        trainer._process_hyperparameters()
        
        # Verify attributes were removed
        assert not hasattr(mock_hyperparams, 'data_path')
        assert not hasattr(mock_hyperparams, 'output_path')
        assert not hasattr(mock_hyperparams, 'training_data_name')
        assert not hasattr(mock_hyperparams, 'validation_data_name')
        
        # Verify _specs were updated
        assert 'data_path' not in mock_hyperparams._specs
        assert 'output_path' not in mock_hyperparams._specs
        assert 'training_data_name' not in mock_hyperparams._specs
        assert 'validation_data_name' not in mock_hyperparams._specs
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
        trainer = DPOTrainer.__new__(DPOTrainer)
        trainer.hyperparameters = mock_hyperparams
        
        # Call the method
        trainer._process_hyperparameters()
        
        # Verify only existing attributes were processed
        assert not hasattr(mock_hyperparams, 'data_path')
        assert 'data_path' not in mock_hyperparams._specs
        assert 'other_param' in mock_hyperparams._specs

    def test_process_hyperparameters_with_none_hyperparameters(self):
        """Test that _process_hyperparameters handles None hyperparameters."""
        trainer = DPOTrainer.__new__(DPOTrainer)
        trainer.hyperparameters = None
        
        # Should not raise an exception
        trainer._process_hyperparameters()

    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    def test_accepts_stopping_condition(self, mock_finetuning, mock_validate):
        """Test DPOTrainer accepts stopping_condition parameter."""
        from sagemaker.train.configs import StoppingCondition
        
        mock_validate.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning.return_value = (mock_hyperparams, "model-arn", False)
        
        stopping_condition = StoppingCondition(max_runtime_in_seconds=14400)
        trainer = DPOTrainer(
            model="test-model",
            model_package_group="test-group",
            stopping_condition=stopping_condition
        )
        
        assert trainer.stopping_condition == stopping_condition
        assert trainer.stopping_condition.max_runtime_in_seconds == 14400

    @patch('sagemaker.train.common_utils.trainer_wait.wait')
    @patch('sagemaker.train.dpo_trainer._resolve_model_and_name')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.dpo_trainer.TrainDefaults.get_role')
    @patch('sagemaker.train.dpo_trainer.TrainDefaults.get_sagemaker_session')
    @patch('sagemaker.train.dpo_trainer._get_unique_name')
    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._create_input_data_config')
    @patch('sagemaker.train.dpo_trainer._convert_input_data_to_channels')
    @patch('sagemaker.train.dpo_trainer._create_output_config')
    @patch('sagemaker.train.dpo_trainer._create_serverless_config')
    @patch('sagemaker.train.dpo_trainer._create_mlflow_config')
    @patch('sagemaker.train.dpo_trainer._create_model_package_config')
    @patch('sagemaker.core.resources.TrainingJob.create')
    def test_train_passes_wait_timeout(self, mock_training_job_create, mock_model_package_config,
                                       mock_mlflow_config, mock_serverless_config, mock_output_config,
                                       mock_convert_channels, mock_input_config, mock_validate_group,
                                       mock_unique_name, mock_get_sagemaker_session, mock_get_role,
                                       mock_get_options, mock_resolve_model, mock_wait):
        """Test that wait_timeout is passed to _wait as timeout kwarg."""
        mock_validate_group.return_value = "test-group"
        mock_resolve_model.return_value = ("test-model", "test-model")
        mock_get_sagemaker_session.return_value = Mock(sagemaker_config={})
        mock_fine_tuning_options = Mock()
        mock_fine_tuning_options.to_dict.return_value = {"learning_rate": "0.001"}
        mock_get_options.return_value = (mock_fine_tuning_options, "model-arn", False)
        mock_get_role.return_value = "test-role"
        mock_unique_name.return_value = "test-job-name"
        mock_input_config.return_value = [Mock()]
        mock_convert_channels.return_value = [Mock()]
        mock_output_config.return_value = Mock()
        mock_serverless_config.return_value = Mock()
        mock_mlflow_config.return_value = Mock()
        mock_model_package_config.return_value = Mock()
        mock_training_job = Mock()
        mock_training_job.arn = "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        mock_training_job_create.return_value = mock_training_job

        trainer = DPOTrainer(model="test-model", model_package_group="test-group", training_dataset="s3://bucket/train")
        trainer.train(wait=True, wait_timeout=600)

        mock_wait.assert_called_once_with(mock_training_job, timeout=600, poll=5)

    @patch('sagemaker.train.common_utils.trainer_wait.wait')
    @patch('sagemaker.train.dpo_trainer._resolve_model_and_name')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.dpo_trainer.TrainDefaults.get_role')
    @patch('sagemaker.train.dpo_trainer.TrainDefaults.get_sagemaker_session')
    @patch('sagemaker.train.dpo_trainer._get_unique_name')
    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._create_input_data_config')
    @patch('sagemaker.train.dpo_trainer._convert_input_data_to_channels')
    @patch('sagemaker.train.dpo_trainer._create_output_config')
    @patch('sagemaker.train.dpo_trainer._create_serverless_config')
    @patch('sagemaker.train.dpo_trainer._create_mlflow_config')
    @patch('sagemaker.train.dpo_trainer._create_model_package_config')
    @patch('sagemaker.core.resources.TrainingJob.create')
    def test_train_without_wait_timeout_uses_default(self, mock_training_job_create, mock_model_package_config,
                                                      mock_mlflow_config, mock_serverless_config, mock_output_config,
                                                      mock_convert_channels, mock_input_config, mock_validate_group,
                                                      mock_unique_name, mock_get_sagemaker_session, mock_get_role,
                                                      mock_get_options, mock_resolve_model, mock_wait):
        """Test that _wait is called without timeout kwarg when wait_timeout is None."""
        mock_validate_group.return_value = "test-group"
        mock_resolve_model.return_value = ("test-model", "test-model")
        mock_get_sagemaker_session.return_value = Mock(sagemaker_config={})
        mock_fine_tuning_options = Mock()
        mock_fine_tuning_options.to_dict.return_value = {"learning_rate": "0.001"}
        mock_get_options.return_value = (mock_fine_tuning_options, "model-arn", False)
        mock_get_role.return_value = "test-role"
        mock_unique_name.return_value = "test-job-name"
        mock_input_config.return_value = [Mock()]
        mock_convert_channels.return_value = [Mock()]
        mock_output_config.return_value = Mock()
        mock_serverless_config.return_value = Mock()
        mock_mlflow_config.return_value = Mock()
        mock_model_package_config.return_value = Mock()
        mock_training_job = Mock()
        mock_training_job.arn = "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        mock_training_job_create.return_value = mock_training_job

        trainer = DPOTrainer(model="test-model", model_package_group="test-group", training_dataset="s3://bucket/train")
        trainer.train(wait=True)

        mock_wait.assert_called_once_with(mock_training_job, poll=5)

    @patch('sagemaker.train.common_utils.trainer_wait.wait')
    @patch('sagemaker.train.dpo_trainer._resolve_model_and_name')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.dpo_trainer.TrainDefaults.get_role')
    @patch('sagemaker.train.dpo_trainer.TrainDefaults.get_sagemaker_session')
    @patch('sagemaker.train.dpo_trainer._get_unique_name')
    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._create_input_data_config')
    @patch('sagemaker.train.dpo_trainer._convert_input_data_to_channels')
    @patch('sagemaker.train.dpo_trainer._create_output_config')
    @patch('sagemaker.train.dpo_trainer._create_serverless_config')
    @patch('sagemaker.train.dpo_trainer._create_mlflow_config')
    @patch('sagemaker.train.dpo_trainer._create_model_package_config')
    @patch('sagemaker.core.resources.TrainingJob.create')
    def test_train_wait_false_skips_wait(self, mock_training_job_create, mock_model_package_config,
                                         mock_mlflow_config, mock_serverless_config, mock_output_config,
                                         mock_convert_channels, mock_input_config, mock_validate_group,
                                         mock_unique_name, mock_get_sagemaker_session, mock_get_role,
                                         mock_get_options, mock_resolve_model, mock_wait):
        """Test that _wait is not called when wait=False."""
        mock_validate_group.return_value = "test-group"
        mock_resolve_model.return_value = ("test-model", "test-model")
        mock_get_sagemaker_session.return_value = Mock(sagemaker_config={})
        mock_fine_tuning_options = Mock()
        mock_fine_tuning_options.to_dict.return_value = {"learning_rate": "0.001"}
        mock_get_options.return_value = (mock_fine_tuning_options, "model-arn", False)
        mock_get_role.return_value = "test-role"
        mock_unique_name.return_value = "test-job-name"
        mock_input_config.return_value = [Mock()]
        mock_convert_channels.return_value = [Mock()]
        mock_output_config.return_value = Mock()
        mock_serverless_config.return_value = Mock()
        mock_mlflow_config.return_value = Mock()
        mock_model_package_config.return_value = Mock()
        mock_training_job = Mock()
        mock_training_job.arn = "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        mock_training_job_create.return_value = mock_training_job

        trainer = DPOTrainer(model="test-model", model_package_group="test-group", training_dataset="s3://bucket/train")
        trainer.train(wait=False, wait_timeout=600)

        mock_wait.assert_not_called()


    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    def test_init_sequence_length_default_none(self, mock_finetuning_options, mock_validate_group):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        trainer = DPOTrainer(model="test-model", model_package_group="test-group")
        assert trainer.sequence_length is None

    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    def test_init_with_sequence_length(self, mock_finetuning_options, mock_validate_group):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        trainer = DPOTrainer(model="test-model", model_package_group="test-group", sequence_length="8K")
        assert trainer.sequence_length == "8K"

    @patch('sagemaker.train.dpo_trainer._resolve_model_and_name')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.dpo_trainer.TrainDefaults.get_role')
    @patch('sagemaker.train.dpo_trainer.TrainDefaults.get_sagemaker_session')
    @patch('sagemaker.train.dpo_trainer._get_unique_name')
    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._create_input_data_config')
    @patch('sagemaker.train.dpo_trainer._convert_input_data_to_channels')
    @patch('sagemaker.train.dpo_trainer._create_output_config')
    @patch('sagemaker.train.dpo_trainer._create_serverless_config')
    @patch('sagemaker.train.dpo_trainer._create_mlflow_config')
    @patch('sagemaker.train.dpo_trainer._create_model_package_config')
    @patch('sagemaker.core.resources.TrainingJob.create')
    def test_train_passes_sequence_length_to_serverless_config(self, mock_training_job_create,
            mock_model_package_config, mock_mlflow_config, mock_serverless_config,
            mock_output_config, mock_convert_channels, mock_input_config,
            mock_validate_group, mock_unique_name, mock_get_sagemaker_session,
            mock_get_role, mock_get_options, mock_resolve_model):
        mock_validate_group.return_value = "test-group"
        mock_resolve_model.return_value = ("test-model", "test-model")
        mock_get_sagemaker_session.return_value = Mock(sagemaker_config={})
        mock_fine_tuning_options = Mock()
        mock_fine_tuning_options.to_dict.return_value = {}
        mock_get_options.return_value = (mock_fine_tuning_options, "model-arn", False)
        mock_get_role.return_value = "test-role"
        mock_unique_name.return_value = "test-job-name"
        mock_input_config.return_value = [Mock()]
        mock_convert_channels.return_value = [Mock()]
        mock_output_config.return_value = Mock()
        mock_serverless_config.return_value = Mock()
        mock_mlflow_config.return_value = Mock()
        mock_model_package_config.return_value = Mock()
        mock_training_job = Mock()
        mock_training_job_create.return_value = mock_training_job

        trainer = DPOTrainer(model="test-model", model_package_group="test-group",
                            training_dataset="s3://bucket/train", sequence_length="16K")
        trainer.train(wait=False)

        mock_serverless_config.assert_called_once()
        call_kwargs = mock_serverless_config.call_args[1]
        assert call_kwargs["sequence_length"] == "16K"


class TestDPOTrainerComputeDispatch:
    """Tests for compute dispatch in DPOTrainer."""

    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._resolve_model_and_name')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    def _make_trainer(self, mock_opts, mock_resolve, mock_validate, compute=None):
        from sagemaker.core.training.configs import Compute, HyperPodCompute
        mock_resolve.return_value = ("model", "nova-textgeneration-lite-v2")
        mock_validate.return_value = "group"
        mock_hp = Mock()
        mock_hp.to_dict.return_value = {}
        mock_opts.return_value = (mock_hp, "arn", False)
        return DPOTrainer(model="amazon.nova-lite-v2", compute=compute, model_package_group="grp")

    def test_rejects_invalid_compute_type(self):
        from sagemaker.core.training.configs import Compute, HyperPodCompute
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


class TestDPOTrainerBaseModelName:
    """Tests for base_model_name param and iterative training."""

    @patch('sagemaker.train.dpo_trainer._validate_eula_for_gated_model', return_value=False)
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group', return_value="my-group")
    @patch('sagemaker.train.dpo_trainer._resolve_model_and_name', return_value=("model_obj", "nova-textgeneration-lite-v2"))
    def test_s3_model_with_base_model_name(self, mock_resolve, mock_validate_group, mock_get_options, mock_eula):
        from sagemaker.core.training.configs import HyperPodCompute

        mock_hp = Mock()
        mock_hp.to_dict.return_value = {}
        mock_hp._specs = {}
        mock_hp._user_set = None
        mock_get_options.return_value = (mock_hp, "model-arn", False)

        trainer = DPOTrainer(
            model="s3://bucket/checkpoint/step_10",
            base_model_name="amazon.nova-2-lite-v1",
            compute=HyperPodCompute(cluster_name="my-cluster", node_count=4),
            training_dataset="s3://bucket/train.jsonl",
        )

        assert trainer.model_source == "s3://bucket/checkpoint/step_10"
        assert trainer._model_name == "nova-textgeneration-lite-v2"

    @patch('sagemaker.train.dpo_trainer._validate_eula_for_gated_model', return_value=False)
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group', return_value="my-group")
    @patch('sagemaker.train.dpo_trainer._resolve_model_and_name', return_value=("model_obj", "nova-textgeneration-lite-v2"))
    def test_s3_model_without_base_model_name_raises(self, mock_resolve, mock_validate_group, mock_get_options, mock_eula):
        from sagemaker.core.training.configs import HyperPodCompute

        mock_hp = Mock()
        mock_hp.to_dict.return_value = {}
        mock_get_options.return_value = (mock_hp, "model-arn", False)

        with pytest.raises(ValueError, match="base_model_name is required"):
            DPOTrainer(
                model="s3://bucket/checkpoint/step_10",
                compute=HyperPodCompute(cluster_name="my-cluster", node_count=4),
                training_dataset="s3://bucket/train.jsonl",
            )


class TestDPOTrainerDryRun:
    """Tests for DPOTrainer.train(dry_run=True)."""

    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.dpo_trainer.TrainDefaults.get_role')
    @patch('sagemaker.train.dpo_trainer.TrainDefaults.get_sagemaker_session')
    @patch('sagemaker.train.dpo_trainer._get_unique_name')
    @patch('sagemaker.train.dpo_trainer._create_input_data_config')
    @patch('sagemaker.train.dpo_trainer._convert_input_data_to_channels')
    @patch('sagemaker.train.dpo_trainer._create_output_config')
    @patch('sagemaker.train.dpo_trainer._create_serverless_config')
    @patch('sagemaker.train.dpo_trainer._create_mlflow_config')
    @patch('sagemaker.train.dpo_trainer._create_model_package_config')
    @patch('sagemaker.train.dpo_trainer._validate_hyperparameter_values')
    @patch('sagemaker.core.resources.TrainingJob.create')
    @patch('sagemaker.train.common_utils.data_utils.validate_data_path_exists')
    def test_dry_run_returns_none_without_submitting(
        self, mock_validate_s3, mock_create, mock_validate_hp, mock_model_pkg,
        mock_mlflow, mock_serverless, mock_output, mock_channels, mock_input,
        mock_name, mock_session, mock_role, mock_options, mock_group,
    ):
        mock_group.return_value = "test-group"
        mock_hp = Mock()
        mock_hp.to_dict.return_value = {}
        mock_hp._specs = {}
        mock_options.return_value = (mock_hp, "model-arn", False)

        sess = Mock()
        sess.boto_session.region_name = "us-east-1"
        sess.boto_region_name = "us-east-1"
        sess.sagemaker_config = {}
        mock_session.return_value = sess
        mock_role.return_value = "test-role"
        mock_name.return_value = "job-name"
        mock_input.return_value = [Mock()]
        mock_channels.return_value = [Mock()]
        mock_output.return_value = Mock()
        mock_serverless.return_value = Mock()
        mock_mlflow.return_value = Mock()
        mock_model_pkg.return_value = Mock()

        trainer = DPOTrainer(
            model="test-model", model_package_group="test-group",
            training_dataset="s3://bucket/train.jsonl",
        )
        trainer.train(dry_run=True)

        mock_create.assert_not_called()
        mock_role.assert_called_once()
        mock_validate_hp.assert_called_once()


class TestDPOTrainerListSupportedModels:

    @patch("sagemaker.train.common_utils.recipe_utils._list_hub_models_by_recipe")
    def test_list_supported_models(self, mock_list):
        mock_list.return_value = ["meta-llama/Llama-3"]
        result = DPOTrainer.list_supported_models()
        assert result == ["meta-llama/Llama-3"]
        mock_list.assert_called_once_with(
            recipe_type="FineTuning", technique="DPO", session=None
        )

class TestDPOTrainerPipelineSession:
    """Test DPOTrainer behavior when PipelineSession is used.

    Ref: https://github.com/aws/sagemaker-python-sdk/issues/6163
    """

    @patch('sagemaker.train.dpo_trainer._create_model_package_config')
    @patch('sagemaker.train.dpo_trainer._create_mlflow_config')
    @patch('sagemaker.train.dpo_trainer._create_output_config')
    @patch('sagemaker.train.dpo_trainer._create_serverless_config')
    @patch('sagemaker.train.dpo_trainer._convert_input_data_to_channels')
    @patch('sagemaker.train.dpo_trainer._create_input_data_config')
    @patch('sagemaker.train.dpo_trainer._get_unique_name')
    @patch('sagemaker.train.dpo_trainer.TrainDefaults.get_role')
    @patch('sagemaker.train.dpo_trainer.TrainDefaults.get_sagemaker_session')
    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.dpo_trainer._resolve_model_and_name')
    @patch('sagemaker.train.common_utils.finetune_utils._get_beta_session')
    @patch('sagemaker.core.resources.TrainingJob.create')
    def test_train_with_pipeline_session_does_not_launch_job(
        self, mock_training_job_create, mock_beta_session, mock_resolve_model,
        mock_finetuning_options, mock_validate_group, mock_get_session, mock_get_role,
        mock_unique_name, mock_input_config, mock_convert_channels,
        mock_serverless_config, mock_output_config, mock_mlflow_config, mock_model_package_config,
    ):
        """When PipelineSession is passed, _intercept_create_request traps the args."""
        from sagemaker.train.dpo_trainer import DPOTrainer
        from sagemaker.core.workflow.pipeline_context import PipelineSession, _JobStepArguments

        pipeline_session = Mock(spec=PipelineSession)
        pipeline_session.boto_session = Mock()
        pipeline_session.boto_session.region_name = "us-west-2"

        step_args = _JobStepArguments("train", {"training_job_name": "test-dpo-job-001"})
        pipeline_session._intercept_create_request.return_value = None
        pipeline_session.context = step_args
        mock_get_session.return_value = pipeline_session

        mock_resolve_model.return_value = ("test-model", "resolved-model-name")
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {"param1": "value1"}
        mock_hyperparams._specs = {"param1": {"type": "string"}}
        mock_hyperparams._user_set = set()
        mock_finetuning_options.return_value = (mock_hyperparams, "arn:aws:sagemaker:us-west-2:123456789012:model/test", False)
        mock_validate_group.return_value = "test-group"
        mock_get_role.return_value = "arn:aws:iam::123456789012:role/Role"
        mock_unique_name.return_value = "test-dpo-job-001"
        mock_input_config.return_value = {"train": "s3://bucket/data"}
        mock_convert_channels.return_value = [{"ChannelName": "train"}]
        mock_serverless_config.return_value = {"BaseModelArn": "arn:model"}
        mock_output_config.return_value = {"S3OutputPath": "s3://bucket/output"}
        mock_mlflow_config.return_value = None
        mock_model_package_config.return_value = None
        mock_beta_session.return_value = pipeline_session

        trainer = DPOTrainer(model="test-model", training_dataset="s3://bucket/data", model_package_group="test-group", sagemaker_session=pipeline_session)
        trainer._model_arn = "arn:aws:sagemaker:us-west-2:123456789012:model/test"
        trainer._model_name = "test-model"
        trainer.accept_eula = True
        trainer.hyperparameters = mock_hyperparams

        result = trainer.train()

        from sagemaker.core.workflow.pipeline_context import _StepArguments
        # @runnable_by_pipeline intercepts and returns _StepArguments
        assert isinstance(result, _StepArguments)
        assert result.caller_name == "train"
        assert result.func is not None
        assert result.func_args[0] is trainer
        mock_training_job_create.assert_not_called()

    @patch('sagemaker.train.dpo_trainer._create_model_package_config')
    @patch('sagemaker.train.dpo_trainer._create_mlflow_config')
    @patch('sagemaker.train.dpo_trainer._create_output_config')
    @patch('sagemaker.train.dpo_trainer._create_serverless_config')
    @patch('sagemaker.train.dpo_trainer._convert_input_data_to_channels')
    @patch('sagemaker.train.dpo_trainer._create_input_data_config')
    @patch('sagemaker.train.dpo_trainer._get_unique_name')
    @patch('sagemaker.train.dpo_trainer.TrainDefaults.get_role')
    @patch('sagemaker.train.dpo_trainer.TrainDefaults.get_sagemaker_session')
    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.dpo_trainer._resolve_model_and_name')
    @patch('sagemaker.train.common_utils.finetune_utils._get_beta_session')
    @patch('sagemaker.core.resources.TrainingJob.create')
    def test_train_pipeline_session_produces_valid_step_arguments(
        self, mock_training_job_create, mock_beta_session, mock_resolve_model,
        mock_finetuning_options, mock_validate_group, mock_get_session, mock_get_role,
        mock_unique_name, mock_input_config, mock_convert_channels,
        mock_serverless_config, mock_output_config, mock_mlflow_config, mock_model_package_config,
    ):
        """TrainingStep.arguments produces valid PascalCase dict."""
        from sagemaker.train.dpo_trainer import DPOTrainer
        from sagemaker.core.workflow.pipeline_context import PipelineSession, _StepArguments

        # Avoid depending on sagemaker-mlops (the dependency direction is
        # sagemaker-mlops -> sagemaker-train). TrainingStep.arguments internally
        # calls execute_job_functions and reads pipeline_session.context.args.
        from sagemaker.core.workflow.utilities import execute_job_functions

        pipeline_session = PipelineSession.__new__(PipelineSession)
        pipeline_session._context = None
        pipeline_session.boto_session = Mock()
        pipeline_session.boto_session.region_name = "us-west-2"
        mock_get_session.return_value = pipeline_session

        mock_resolve_model.return_value = ("test-model", "test")
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {"lr": "0.001"}
        mock_hyperparams._specs = {"lr": {"type": "string"}}
        mock_hyperparams._user_set = set()
        mock_finetuning_options.return_value = (mock_hyperparams, "arn:model", False)
        mock_validate_group.return_value = "grp"
        mock_get_role.return_value = "arn:aws:iam::123:role/Role"
        mock_unique_name.return_value = "test-job"
        mock_input_config.return_value = [{"DataSource": {"S3DataSource": {"S3Uri": "s3://data"}}}]
        mock_convert_channels.return_value = [{"ChannelName": "train"}]
        mock_serverless_config.return_value = {"BaseModelArn": "arn:model", "JobType": "FineTuning"}
        mock_output_config.return_value = {"S3OutputPath": "s3://output"}
        mock_mlflow_config.return_value = None
        mock_model_package_config.return_value = None
        mock_beta_session.return_value = pipeline_session

        trainer = DPOTrainer(model="test-model", training_dataset="s3://bucket/data", model_package_group="grp", sagemaker_session=pipeline_session)
        trainer._model_arn = "arn:model"
        trainer._model_name = "test-model"
        trainer.accept_eula = True
        trainer.hyperparameters = mock_hyperparams

        result = trainer.train()
        execute_job_functions(result)
        arguments = pipeline_session.context.args

        assert isinstance(arguments, dict)
        assert "session" not in arguments
        assert "region" not in arguments
        tags = arguments.get("Tags", [])
        for t in tags:
            assert "Key" in t and "Value" in t

    @patch('sagemaker.train.dpo_trainer._create_model_package_config')
    @patch('sagemaker.train.dpo_trainer._create_mlflow_config')
    @patch('sagemaker.train.dpo_trainer._create_output_config')
    @patch('sagemaker.train.dpo_trainer._create_serverless_config')
    @patch('sagemaker.train.dpo_trainer._convert_input_data_to_channels')
    @patch('sagemaker.train.dpo_trainer._create_input_data_config')
    @patch('sagemaker.train.dpo_trainer._get_unique_name')
    @patch('sagemaker.train.dpo_trainer.TrainDefaults.get_role')
    @patch('sagemaker.train.dpo_trainer.TrainDefaults.get_sagemaker_session')
    @patch('sagemaker.train.dpo_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.dpo_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.dpo_trainer._resolve_model_and_name')
    @patch('sagemaker.train.common_utils.finetune_utils._get_beta_session')
    @patch('sagemaker.train.common_utils.data_utils.validate_data_path_exists')
    @patch('sagemaker.core.resources.TrainingJob.create')
    def test_train_without_pipeline_session_launches_job(
        self, mock_training_job_create, mock_validate_path, mock_beta_session,
        mock_resolve_model, mock_finetuning_options, mock_validate_group,
        mock_get_session, mock_get_role, mock_unique_name, mock_input_config,
        mock_convert_channels, mock_serverless_config, mock_output_config,
        mock_mlflow_config, mock_model_package_config,
    ):
        """Regular Session launches job normally."""
        from sagemaker.train.dpo_trainer import DPOTrainer

        regular_session = Mock()
        regular_session.boto_session = Mock()
        regular_session.boto_session.region_name = "us-west-2"
        regular_session.sagemaker_config = {}
        mock_get_session.return_value = regular_session

        mock_resolve_model.return_value = ("test-model", "test")
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {"lr": "0.001"}
        mock_hyperparams._specs = {"lr": {"type": "string"}}
        mock_hyperparams._user_set = set()
        mock_finetuning_options.return_value = (mock_hyperparams, "arn:model", False)
        mock_validate_group.return_value = "grp"
        mock_get_role.return_value = "arn:aws:iam::123:role/Role"
        mock_unique_name.return_value = "test-job"
        mock_input_config.return_value = {}
        mock_convert_channels.return_value = []
        mock_serverless_config.return_value = {"BaseModelArn": "arn:model"}
        mock_output_config.return_value = {"S3OutputPath": "s3://output"}
        mock_mlflow_config.return_value = None
        mock_model_package_config.return_value = None
        mock_beta_session.return_value = regular_session
        mock_training_job = Mock()
        mock_training_job_create.return_value = mock_training_job

        trainer = DPOTrainer(model="test-model", training_dataset="s3://bucket/data", model_package_group="grp", sagemaker_session=regular_session)
        trainer._model_arn = "arn:model"
        trainer._model_name = "test-model"
        trainer.accept_eula = True
        trainer.hyperparameters = mock_hyperparams

        result = trainer.train(wait=False)
        mock_training_job_create.assert_called_once()
        assert result == mock_training_job
