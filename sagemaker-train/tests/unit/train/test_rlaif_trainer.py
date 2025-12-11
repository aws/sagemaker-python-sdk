import pytest
from unittest.mock import Mock, patch
from sagemaker.train.rlaif_trainer import RLAIFTrainer
from sagemaker.train.common import TrainingType
from sagemaker.core.resources import ModelPackage


class TestRLAIFTrainer:
    
    @pytest.fixture
    def mock_session(self):
        session = Mock()
        session.region_name = "us-east-1"
        return session

    @patch('sagemaker.train.rlaif_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.rlaif_trainer._get_fine_tuning_options_and_model_arn')
    def test_init_with_defaults(self, mock_finetuning_options, mock_validate_group, mock_session):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        trainer = RLAIFTrainer(model="test-model", model_package_group="test-group")
        assert trainer.training_type == TrainingType.LORA
        assert trainer.model == "test-model"

    @patch('sagemaker.train.rlaif_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.rlaif_trainer._get_fine_tuning_options_and_model_arn')
    def test_init_with_full_training_type(self, mock_finetuning_options, mock_validate_group, mock_session):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        trainer = RLAIFTrainer(model="test-model", training_type=TrainingType.FULL, model_package_group="test-group")
        assert trainer.training_type == TrainingType.FULL

    @patch('sagemaker.train.common_utils.finetune_utils._get_beta_session')
    @patch('sagemaker.train.rlaif_trainer._resolve_model_and_name')
    @patch('sagemaker.train.rlaif_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.rlaif_trainer.TrainDefaults.get_role')
    @patch('sagemaker.train.rlaif_trainer.TrainDefaults.get_sagemaker_session')
    @patch('sagemaker.train.rlaif_trainer._get_unique_name')
    @patch('sagemaker.train.rlaif_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.rlaif_trainer._create_input_data_config')
    @patch('sagemaker.train.rlaif_trainer._convert_input_data_to_channels')
    @patch('sagemaker.train.rlaif_trainer._create_output_config')
    @patch('sagemaker.train.rlaif_trainer._create_mlflow_config')
    @patch('sagemaker.train.rlaif_trainer._create_model_package_config')
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
        
        trainer = RLAIFTrainer(model="test-model", training_type=TrainingType.LORA, model_package_group="test-group", training_dataset="s3://bucket/train")
        trainer.train(wait=False)
        
        assert mock_training_job_create.called

    @patch('sagemaker.train.common_utils.finetune_utils._get_beta_session')
    @patch('sagemaker.train.rlaif_trainer._resolve_model_and_name')
    @patch('sagemaker.train.rlaif_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.rlaif_trainer.TrainDefaults.get_role')
    @patch('sagemaker.train.rlaif_trainer.TrainDefaults.get_sagemaker_session')
    @patch('sagemaker.train.rlaif_trainer._get_unique_name')
    @patch('sagemaker.train.rlaif_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.rlaif_trainer._create_input_data_config')
    @patch('sagemaker.train.rlaif_trainer._convert_input_data_to_channels')
    @patch('sagemaker.train.rlaif_trainer._create_output_config')
    @patch('sagemaker.train.rlaif_trainer._create_mlflow_config')
    @patch('sagemaker.train.rlaif_trainer._create_model_package_config')
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
        
        trainer = RLAIFTrainer(model="test-model", training_type=TrainingType.FULL, model_package_group="test-group", training_dataset="s3://bucket/train")
        trainer.train(wait=False)
        
        assert mock_training_job_create.called

    @patch('sagemaker.train.rlaif_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.rlaif_trainer._get_fine_tuning_options_and_model_arn')
    def test_training_type_string_value(self, mock_finetuning_options, mock_validate_group, mock_session):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        trainer = RLAIFTrainer(model="test-model", training_type="CUSTOM", model_package_group="test-group")
        assert trainer.training_type == "CUSTOM"

    @patch('sagemaker.train.common_utils.finetune_utils._get_beta_session')
    @patch('sagemaker.train.rlaif_trainer._resolve_model_and_name')
    @patch('sagemaker.train.rlaif_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.rlaif_trainer._get_fine_tuning_options_and_model_arn')
    def test_model_package_input(self, mock_finetuning_options, mock_validate_group, mock_resolve_model, mock_get_session):
        mock_validate_group.return_value = "test-group"
        mock_get_session.return_value = Mock()
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        
        model_package = Mock(spec=ModelPackage)
        model_package.inference_specification = Mock()
        
        # Make _resolve_model_and_name return the same model_package object
        mock_resolve_model.return_value = (model_package, "test-model")
        
        trainer = RLAIFTrainer(model=model_package)
        assert trainer.model == model_package

    @patch('sagemaker.train.rlaif_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.rlaif_trainer._get_fine_tuning_options_and_model_arn')
    def test_init_with_datasets(self, mock_finetuning_options, mock_validate_group, mock_session):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        trainer = RLAIFTrainer(
            model="test-model",
            model_package_group="test-group",
            training_dataset="s3://bucket/train",
            validation_dataset="s3://bucket/val"
        )
        assert trainer.training_dataset == "s3://bucket/train"
        assert trainer.validation_dataset == "s3://bucket/val"

    @patch('sagemaker.train.rlaif_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.rlaif_trainer._get_fine_tuning_options_and_model_arn')
    def test_init_with_mlflow_config(self, mock_finetuning_options, mock_validate_group, mock_session):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        trainer = RLAIFTrainer(
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
    @patch('sagemaker.train.rlaif_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.rlaif_trainer._get_fine_tuning_options_and_model_arn')
    def test_train_without_datasets_raises_error(self, mock_finetuning_options, mock_validate_group, mock_get_session):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        mock_get_session.return_value = Mock()
        trainer = RLAIFTrainer(model="test-model", model_package_group="test-group")
        
        with pytest.raises(Exception):
            trainer.train(wait=False)

    @patch('sagemaker.train.common_utils.finetune_utils._get_beta_session')
    @patch('sagemaker.train.common_utils.finetune_utils._resolve_model_name')
    @patch('sagemaker.train.rlaif_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.rlaif_trainer._validate_and_resolve_model_package_group')
    def test_model_package_group_handling(self, mock_validate_group, mock_get_options, mock_resolve_model, mock_get_session):
        mock_validate_group.return_value = "test-group"
        mock_get_session.return_value = Mock()
        mock_resolve_model.return_value = "resolved-model"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_get_options.return_value = (mock_hyperparams, "model-arn", False)
        
        trainer = RLAIFTrainer(
            model="test-model",
            model_package_group="test-group"
        )
        assert trainer.model_package_group == "test-group"

    @patch('sagemaker.train.rlaif_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.rlaif_trainer._get_fine_tuning_options_and_model_arn')
    def test_s3_output_path_configuration(self, mock_finetuning_options, mock_validate_group, mock_session):
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", False)
        trainer = RLAIFTrainer(
            model="test-model",
            model_package_group="test-group",
            s3_output_path="s3://bucket/output"
        )
        assert trainer.s3_output_path == "s3://bucket/output"

    @patch('sagemaker.train.common_utils.finetune_utils._get_beta_session')
    @patch('sagemaker.train.rlaif_trainer._resolve_model_and_name')
    @patch('sagemaker.train.rlaif_trainer._get_fine_tuning_options_and_model_arn')
    @patch('sagemaker.train.rlaif_trainer.TrainDefaults.get_role')
    @patch('sagemaker.train.rlaif_trainer.TrainDefaults.get_sagemaker_session')
    @patch('sagemaker.train.rlaif_trainer._get_unique_name')
    @patch('sagemaker.train.rlaif_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.rlaif_trainer._create_input_data_config')
    @patch('sagemaker.train.rlaif_trainer._convert_input_data_to_channels')
    @patch('sagemaker.train.rlaif_trainer._create_output_config')
    @patch('sagemaker.train.rlaif_trainer._create_mlflow_config')
    @patch('sagemaker.train.rlaif_trainer._create_model_package_config')
    @patch('sagemaker.core.resources.TrainingJob.create')
    def test_train_with_tags(self, mock_training_job_create, mock_model_package_config, 
                            mock_mlflow_config, mock_output_config, mock_convert_channels, mock_input_config, 
                            mock_validate_group, mock_unique_name, mock_get_sagemaker_session, mock_get_role, 
                            mock_get_options, mock_resolve_model, mock_get_session):
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
        
        trainer = RLAIFTrainer(model="test-model", model_package_group="test-group", training_dataset="s3://bucket/train")
        trainer.train(wait=False)
        
        mock_training_job_create.assert_called_once()
        call_kwargs = mock_training_job_create.call_args[1]
        assert call_kwargs["tags"] == [
            {"key": "sagemaker-studio:jumpstart-model-id", "value": "test-model"},
            {"key": "sagemaker-studio:jumpstart-hub-name", "value": "SageMakerPublicHub"}
        ]

    @patch('sagemaker.train.rlaif_trainer._validate_and_resolve_model_package_group')
    @patch('sagemaker.train.rlaif_trainer._get_fine_tuning_options_and_model_arn')
    def test_gated_model_eula_validation(self, mock_finetuning_options, mock_validate_group, mock_session):
        """Test EULA validation for gated models"""
        mock_validate_group.return_value = "test-group"
        mock_hyperparams = Mock()
        mock_hyperparams.to_dict.return_value = {}
        mock_finetuning_options.return_value = (mock_hyperparams, "model-arn", True)  # is_gated_model=True
        
        # Should raise error when accept_eula=False for gated model
        with pytest.raises(ValueError, match="gated model and requires EULA acceptance"):
            RLAIFTrainer(model="gated-model", model_package_group="test-group", accept_eula=False)
        
        # Should work when accept_eula=True for gated model
        trainer = RLAIFTrainer(model="gated-model", model_package_group="test-group", accept_eula=True)
        assert trainer.accept_eula == True

    def test_process_hyperparameters_removes_constructor_handled_keys(self):
        """Test that _process_hyperparameters removes keys handled by constructor inputs."""
        # Create mock hyperparameters with all possible keys
        mock_hyperparams = Mock()
        mock_hyperparams._specs = {
            'output_path': 'test_output_path',
            'data_path': 'test_data_path',
            'validation_data_path': 'test_validation_data_path',
            'other_param': 'should_remain'
        }
        
        # Add attributes to mock
        mock_hyperparams.output_path = 'test_output_path'
        mock_hyperparams.data_path = 'test_data_path'
        mock_hyperparams.validation_data_path = 'test_validation_data_path'
        
        # Create trainer instance with mock hyperparameters
        trainer = RLAIFTrainer.__new__(RLAIFTrainer)
        trainer.hyperparameters = mock_hyperparams
        trainer.reward_model_id = "test-reward-model"
        
        # Call the method
        trainer._process_hyperparameters()
        
        # Verify attributes were removed
        assert not hasattr(mock_hyperparams, 'output_path')
        assert not hasattr(mock_hyperparams, 'data_path')
        assert not hasattr(mock_hyperparams, 'validation_data_path')
        
        # Verify _specs were updated
        assert 'output_path' not in mock_hyperparams._specs
        assert 'data_path' not in mock_hyperparams._specs
        assert 'validation_data_path' not in mock_hyperparams._specs
        assert 'other_param' in mock_hyperparams._specs
        
        # Verify judge_model_id was set
        assert mock_hyperparams.judge_model_id == "bedrock/test-reward-model"

    def test_process_hyperparameters_updates_judge_model_id(self):
        """Test that _process_hyperparameters updates judge_model_id when reward_model_id is provided."""
        # Use a simple object instead of Mock to allow proper attribute assignment
        class MockHyperparams:
            def __init__(self):
                self._specs = {'some_param': 'value'}  # Non-empty specs
        
        mock_hyperparams = MockHyperparams()
        
        trainer = RLAIFTrainer.__new__(RLAIFTrainer)
        trainer.hyperparameters = mock_hyperparams
        trainer.reward_model_id = "my-reward-model"
        
        trainer._process_hyperparameters()
        
        assert hasattr(mock_hyperparams, 'judge_model_id')
        assert mock_hyperparams.judge_model_id == "bedrock/my-reward-model"

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
        trainer = RLAIFTrainer.__new__(RLAIFTrainer)
        trainer.hyperparameters = mock_hyperparams
        trainer.reward_model_id = None
        
        # Call the method
        trainer._process_hyperparameters()
        
        # Verify only existing attributes were processed
        assert not hasattr(mock_hyperparams, 'data_path')
        assert 'data_path' not in mock_hyperparams._specs
        assert 'other_param' in mock_hyperparams._specs

    def test_process_hyperparameters_with_none_hyperparameters(self):
        """Test that _process_hyperparameters handles None hyperparameters."""
        trainer = RLAIFTrainer.__new__(RLAIFTrainer)
        trainer.hyperparameters = None
        
        # Should not raise an exception
        trainer._process_hyperparameters()

    def test_process_hyperparameters_early_return_on_none(self):
        """Test that _process_hyperparameters returns early when hyperparameters is None."""
        trainer = RLAIFTrainer.__new__(RLAIFTrainer)
        trainer.hyperparameters = None
        trainer.reward_model_id = "test-model"
        
        # Should return early and not attempt to set judge_model_id
        trainer._process_hyperparameters()
        
        # No exception should be raised

    def test_update_judge_prompt_template_direct_with_matching_template(self):
        """Test _update_judge_prompt_template_direct with matching template."""
        mock_hyperparams = Mock()
        mock_hyperparams._specs = {
            'judge_prompt_template': {
                'enum': ['templates/summarize.jinja', 'templates/helpfulness.jinja']
            }
        }
        
        trainer = RLAIFTrainer.__new__(RLAIFTrainer)
        trainer.hyperparameters = mock_hyperparams
        
        trainer._update_judge_prompt_template_direct("Builtin.summarize")
        
        assert mock_hyperparams.judge_prompt_template == 'templates/summarize.jinja'

    def test_update_judge_prompt_template_direct_with_no_enum(self):
        """Test _update_judge_prompt_template_direct when no enum is available."""
        mock_hyperparams = Mock()
        mock_hyperparams._specs = {'judge_prompt_template': {}}
        mock_hyperparams.judge_prompt_template = 'current_template.jinja'
        
        trainer = RLAIFTrainer.__new__(RLAIFTrainer)
        trainer.hyperparameters = mock_hyperparams
        
        trainer._update_judge_prompt_template_direct("Builtin.current_template")
        
        assert mock_hyperparams.judge_prompt_template == 'current_template.jinja'

    def test_update_judge_prompt_template_direct_no_matching_template(self):
        """Test _update_judge_prompt_template_direct raises error for non-matching template."""
        mock_hyperparams = Mock()
        mock_hyperparams._specs = {
            'judge_prompt_template': {
                'enum': ['templates/summarize.jinja', 'templates/helpfulness.jinja']
            }
        }
        
        trainer = RLAIFTrainer.__new__(RLAIFTrainer)
        trainer.hyperparameters = mock_hyperparams
        
        with pytest.raises(ValueError, match="Selected reward function option 'Builtin.nonexistent' is not available"):
            trainer._update_judge_prompt_template_direct("Builtin.nonexistent")

    def test_update_judge_prompt_template_direct_early_return(self):
        """Test _update_judge_prompt_template_direct returns early when no templates available."""
        mock_hyperparams = Mock()
        mock_hyperparams._specs = {'judge_prompt_template': {}}
        mock_hyperparams.judge_prompt_template = None
        
        trainer = RLAIFTrainer.__new__(RLAIFTrainer)
        trainer.hyperparameters = mock_hyperparams
        
        # Should return early without error
        trainer._update_judge_prompt_template_direct("Builtin.anything")

    def test_process_non_builtin_reward_prompt_removes_judge_template(self):
        """Test _process_non_builtin_reward_prompt removes judge_prompt_template."""
        mock_hyperparams = Mock()
        mock_hyperparams._specs = {'judge_prompt_template': 'template.jinja'}
        mock_hyperparams.judge_prompt_template = 'template.jinja'
        
        trainer = RLAIFTrainer.__new__(RLAIFTrainer)
        trainer.hyperparameters = mock_hyperparams
        trainer.reward_prompt = "arn:aws:sagemaker:us-east-1:123456789012:evaluator/test"
        
        with patch('sagemaker.train.rlaif_trainer._extract_evaluator_arn') as mock_extract:
            mock_extract.return_value = "test-arn"
            trainer._process_non_builtin_reward_prompt()
        
        assert not hasattr(mock_hyperparams, 'judge_prompt_template')
        assert 'judge_prompt_template' not in mock_hyperparams._specs
        assert trainer._evaluator_arn == "test-arn"

    def test_process_non_builtin_reward_prompt_with_hub_content(self):
        """Test _process_non_builtin_reward_prompt with hub content name."""
        mock_hyperparams = Mock()
        mock_hyperparams._specs = {}
        
        trainer = RLAIFTrainer.__new__(RLAIFTrainer)
        trainer.hyperparameters = mock_hyperparams
        trainer.reward_prompt = "custom-prompt-name"
        trainer.sagemaker_session = None
        
        with patch('sagemaker.train.rlaif_trainer.TrainDefaults.get_sagemaker_session') as mock_session, \
             patch('sagemaker.train.rlaif_trainer._get_hub_content_metadata') as mock_hub:
            mock_session.return_value = Mock(boto_session=Mock(region_name="us-west-2"))
            mock_hub.return_value = Mock(hub_content_arn="hub-content-arn")
            
            trainer._process_non_builtin_reward_prompt()
        
        assert trainer._evaluator_arn == "hub-content-arn"

    def test_process_non_builtin_reward_prompt_hub_content_error(self):
        """Test _process_non_builtin_reward_prompt raises error for invalid hub content."""
        mock_hyperparams = Mock()
        mock_hyperparams._specs = {}
        
        trainer = RLAIFTrainer.__new__(RLAIFTrainer)
        trainer.hyperparameters = mock_hyperparams
        trainer.reward_prompt = "invalid-prompt"
        trainer.sagemaker_session = None
        
        with patch('sagemaker.train.rlaif_trainer.TrainDefaults.get_sagemaker_session') as mock_session, \
             patch('sagemaker.train.rlaif_trainer._get_hub_content_metadata') as mock_hub:
            mock_session.return_value = Mock(boto_session=Mock(region_name="us-west-2"))
            mock_hub.side_effect = Exception("Not found")
            
            with pytest.raises(ValueError, match="Custom prompt 'invalid-prompt' not found in HubContent"):
                trainer._process_non_builtin_reward_prompt()

    def test_validate_reward_model_id_valid_models(self):
        """Test _validate_reward_model_id with valid model IDs."""
        trainer = RLAIFTrainer.__new__(RLAIFTrainer)
        
        valid_models = [
            "openai.gpt-oss-120b-1:0",
            "openai.gpt-oss-20b-1:0", 
            "qwen.qwen3-32b-v1:0",
            "qwen.qwen3-coder-30b-a3b-v1:0"
        ]
        
        for model_id in valid_models:
            result = trainer._validate_reward_model_id(model_id)
            assert result == model_id

    def test_validate_reward_model_id_invalid_model(self):
        """Test _validate_reward_model_id raises error for invalid model ID."""
        trainer = RLAIFTrainer.__new__(RLAIFTrainer)
        
        with pytest.raises(ValueError, match="Invalid reward_model_id 'invalid-model-id'"):
            trainer._validate_reward_model_id("invalid-model-id")

    def test_validate_reward_model_id_none_model(self):
        """Test _validate_reward_model_id handles None model ID."""
        trainer = RLAIFTrainer.__new__(RLAIFTrainer)
        
        result = trainer._validate_reward_model_id(None)
        assert result is None
