"""Unit tests for MultiTurnRLTrainer."""
import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from sagemaker.ai_registry.dataset import DataSet
from sagemaker.core.resources import ModelPackage, MlflowApp
from sagemaker.train.custom_agent_lambda import CustomAgentLambda
from sagemaker.train.multi_turn_rl_trainer import (
    MultiTurnRLTrainer,
    BEDROCK_AGENT_CORE_ARN_PATTERN,
    LAMBDA_ARN_PATTERN,
    S3_URI_PATTERN,
    AGENT_RUNTIME_ID_PATTERN,
    JOB_CATEGORY,
    JOB_CONFIG_SCHEMA_VERSION,
    # SUPPORTED_BASE_MODELS,
    # _resolve_base_model_name,
    _resolve_agent_runtime_arn,
    _list_all_mtrl_models,
)


BEDROCK_AGENT_ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/AGENTID123"
LAMBDA_ARN = "arn:aws:lambda:us-west-2:123456789012:function:my-adapter"
MODEL_ARN = "arn:aws:sagemaker:us-west-2:aws:hub-content/SageMakerPublicHub/Model/test-model"
MPG_ARN = "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/my-group"
MLFLOW_ARN = "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/my-server"
S3_OUTPUT = "s3://my-bucket/output/"
S3_DATA = "s3://my-bucket/data/prompts.jsonl"
DATASET_ARN = "arn:aws:sagemaker:us-west-2:123456789012:hub-content/SageMakerPublicHub/Dataset/my-ds"


class TestARNPatterns:
    def test_bedrock_agent_arn(self):
        assert BEDROCK_AGENT_CORE_ARN_PATTERN.match(BEDROCK_AGENT_ARN)
        assert not BEDROCK_AGENT_CORE_ARN_PATTERN.match(LAMBDA_ARN)

    def test_lambda_arn(self):
        assert LAMBDA_ARN_PATTERN.match(LAMBDA_ARN)
        assert not LAMBDA_ARN_PATTERN.match(BEDROCK_AGENT_ARN)

    def test_s3_uri(self):
        assert S3_URI_PATTERN.match("s3://bucket/key")
        assert not S3_URI_PATTERN.match("https://bucket/key")


class TestBaseModelMap:
    # def test_resolve_well_known_name(self):
    #     assert _resolve_base_model_name("Qwen/Qwen3-32B") == "huggingface-reasoning-qwen3-32b"

    # def test_resolve_all_well_known_names(self):
    #     for name, expected in SUPPORTED_BASE_MODELS.items():
    #         assert _resolve_base_model_name(name) == expected

    # def test_resolve_hub_content_name_passthrough(self):
    #     assert _resolve_base_model_name("huggingface-reasoning-qwen3-32b") == "huggingface-reasoning-qwen3-32b"

    # def test_resolve_unknown_name_warns(self):
    #     import logging
    #     with patch("sagemaker.train.multi_turn_rl_trainer.logger") as mock_logger:
    #         result = _resolve_base_model_name("some-custom-model")
    #         assert result == "some-custom-model"
    #         mock_logger.warning.assert_called_once()

    @patch("sagemaker.train.multi_turn_rl_trainer._list_all_mtrl_models")
    def test_list_supported_models(self, mock_list):
        mock_list.return_value = ["Qwen/Qwen3-32B", "meta-llama/Llama-3"]
        result = MultiTurnRLTrainer.list_supported_models()
        assert isinstance(result, list)
        assert "Qwen/Qwen3-32B" in result


class TestValidation:
    def test_invalid_agent_config_raises(self):
        with pytest.raises(ValueError, match="Invalid agent_env"):
            MultiTurnRLTrainer._validate_agent_config("not-an-arn")

    def test_valid_bedrock_agent_config(self):
        MultiTurnRLTrainer._validate_agent_config(BEDROCK_AGENT_ARN)

    def test_valid_lambda_agent_config(self):
        MultiTurnRLTrainer._validate_agent_config(LAMBDA_ARN)

    def test_valid_adapter_agent_config(self):
        adapter = CustomAgentLambda(lambda_arn=LAMBDA_ARN)
        MultiTurnRLTrainer._validate_agent_config(adapter)

    def test_invalid_networking_empty_sg(self):
        vpc = MagicMock()
        vpc.security_group_ids = []
        vpc.subnets = ["subnet-123"]
        with pytest.raises(ValueError, match="security_group_ids"):
            MultiTurnRLTrainer._validate_networking(vpc)

    def test_networking_none_ok(self):
        MultiTurnRLTrainer._validate_networking(None)


class TestJobConfigDocument:
    """Test _build_job_config_document and its helpers."""

    def _make_trainer(self, agent_config=BEDROCK_AGENT_ARN, **overrides):
        """Create a trainer with mocked internals for config doc testing."""
        trainer = object.__new__(MultiTurnRLTrainer)
        trainer.agent_env = agent_config
        trainer.bedrock_agentcore_qualifier = overrides.get("bedrock_agentcore_qualifier", "DEFAULT")
        trainer.s3_output_path = S3_OUTPUT
        trainer.output_model_package_group = MPG_ARN
        trainer.intermediate_checkpoint_model_package_group = overrides.get("intermediate_checkpoint_model_package_group", "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/default-ckpt-mpg")
        trainer.mlflow_app_arn = MLFLOW_ARN
        trainer.mlflow_experiment_name = overrides.get("mlflow_experiment_name")
        trainer.mlflow_run_name = overrides.get("mlflow_run_name")
        trainer.accept_eula = True
        trainer.kms_key_arn = overrides.get("kms_key_arn")
        trainer.networking = overrides.get("networking")
        trainer.model = overrides.get("model", "test-model-id")
        trainer.validation_dataset = overrides.get("validation_dataset")
        trainer._model_arn = overrides.get("model_arn", MODEL_ARN)
        trainer.training_dataset = overrides.get("training_dataset", S3_DATA)
        trainer._final_hyperparameters = overrides.get("hyperparameters", {})
        trainer._hp_defaults = {}
        return trainer

    def test_bedrock_agent_config(self):
        trainer = self._make_trainer()
        doc = json.loads(
            trainer._build_job_config_document()
        )
        agent = doc["AgentConfig"]
        assert agent["BedrockAgentCoreConfig"]["AgentRuntimeArn"] == BEDROCK_AGENT_ARN
        assert agent["BedrockAgentCoreConfig"]["Qualifier"] == "DEFAULT"

    def test_bedrock_agent_with_qualifier(self):
        trainer = self._make_trainer(bedrock_agentcore_qualifier="CUSTOM")
        doc = json.loads(
            trainer._build_job_config_document()
        )
        agent = doc["AgentConfig"]
        assert agent["BedrockAgentCoreConfig"]["AgentRuntimeArn"] == BEDROCK_AGENT_ARN
        assert agent["BedrockAgentCoreConfig"]["Qualifier"] == "CUSTOM"

    def test_lambda_agent_config(self):
        trainer = self._make_trainer(agent_config=LAMBDA_ARN)
        doc = json.loads(
            trainer._build_job_config_document()
        )
        assert doc["AgentConfig"]["CustomAgentLambdaConfig"]["LambdaArn"] == LAMBDA_ARN

    def test_adapter_agent_config(self):
        adapter = CustomAgentLambda(lambda_arn=LAMBDA_ARN)
        trainer = self._make_trainer(agent_config=adapter)
        doc = json.loads(
            trainer._build_job_config_document()
        )
        assert doc["AgentConfig"]["CustomAgentLambdaConfig"]["LambdaArn"] == LAMBDA_ARN

    def test_s3_input_data(self):
        trainer = self._make_trainer()
        doc = json.loads(
            trainer._build_job_config_document()
        )
        channel = doc["InputDataConfig"][0]
        assert channel["ChannelName"] == "train"
        assert channel["DataSource"]["S3DataSource"]["S3Uri"] == S3_DATA

    def test_dataset_arn_input_data(self):
        trainer = self._make_trainer(training_dataset=DATASET_ARN)
        doc = json.loads(
            trainer._build_job_config_document()
        )
        channel = doc["InputDataConfig"][0]
        assert channel["DataSource"]["DatasetSource"]["DatasetArn"] == DATASET_ARN

    def test_dataset_object_input_data(self):
        ds = MagicMock(spec=DataSet)
        ds.arn = DATASET_ARN
        trainer = self._make_trainer(training_dataset=ds)
        doc = json.loads(
            trainer._build_job_config_document()
        )
        assert doc["InputDataConfig"][0]["DataSource"]["DatasetSource"]["DatasetArn"] == DATASET_ARN

    def test_output_data_config(self):
        trainer = self._make_trainer()
        doc = json.loads(
            trainer._build_job_config_document()
        )
        assert doc["OutputDataConfig"]["S3OutputPath"] == S3_OUTPUT
        assert "KmsKeyId" not in doc["OutputDataConfig"]

    def test_output_data_config_with_kms(self):
        trainer = self._make_trainer(kms_key_arn="arn:kms:key")
        doc = json.loads(
            trainer._build_job_config_document()
        )
        assert doc["OutputDataConfig"]["KmsKeyArn"] == "arn:kms:key"

    def test_training_config(self):
        trainer = self._make_trainer(hyperparameters={"lr": "0.001"})
        doc = json.loads(
            trainer._build_job_config_document()
        )
        tc = doc["TrainingConfig"]
        assert tc["BaseModelArn"] == MODEL_ARN
        assert tc["AcceptEula"] is True
        assert tc["HyperParameters"] == {"lr": "0.001"}
        assert tc["MlflowConfig"]["MlflowResourceArn"] == MLFLOW_ARN

    def test_mlflow_optional_fields(self):
        trainer = self._make_trainer(
            mlflow_experiment_name="exp1", mlflow_run_name="run1"
        )
        doc = json.loads(
            trainer._build_job_config_document()
        )
        mlflow = doc["TrainingConfig"]["MlflowConfig"]
        assert mlflow["MlflowExperimentName"] == "exp1"
        assert mlflow["MlflowRunName"] == "run1"

    def test_mlflow_optional_fields_omitted(self):
        trainer = self._make_trainer()
        doc = json.loads(
            trainer._build_job_config_document()
        )
        mlflow = doc["TrainingConfig"]["MlflowConfig"]
        assert "MlflowExperimentName" not in mlflow
        assert "MlflowRunName" not in mlflow

    def test_mlflow_app_object(self):
        app = MagicMock(spec=MlflowApp)
        app.arn = MLFLOW_ARN
        trainer = self._make_trainer()
        trainer.mlflow_app_arn = app
        doc = json.loads(
            trainer._build_job_config_document()
        )
        assert doc["TrainingConfig"]["MlflowConfig"]["MlflowResourceArn"] == MLFLOW_ARN

    def test_vpc_config_included(self):
        vpc = MagicMock()
        vpc.security_group_ids = ["sg-123"]
        vpc.subnets = ["subnet-456"]
        trainer = self._make_trainer(networking=vpc)
        doc = json.loads(
            trainer._build_job_config_document()
        )
        assert doc["VpcConfig"]["SecurityGroupIds"] == ["sg-123"]
        assert doc["VpcConfig"]["Subnets"] == ["subnet-456"]

    def test_vpc_config_omitted(self):
        trainer = self._make_trainer()
        doc = json.loads(
            trainer._build_job_config_document()
        )
        assert "VpcConfig" not in doc

    def test_source_model_package_arn_from_model_package(self):
        mock_mp = MagicMock(spec=ModelPackage)
        mock_mp.model_package_arn = "arn:src:pkg"
        trainer = self._make_trainer(model=mock_mp)
        doc = json.loads(
            trainer._build_job_config_document()
        )
        assert doc["ModelPackageConfig"]["InputModelPackageArn"] == "arn:src:pkg"

    def test_source_model_package_arn_absent_for_string_model(self):
        trainer = self._make_trainer(model="some-model-id")
        doc = json.loads(
            trainer._build_job_config_document()
        )
        assert "InputModelPackageArn" not in doc["ModelPackageConfig"]

    def test_intermediate_checkpoint_mpg_included(self):
        arn = "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/ckpt-grp"
        trainer = self._make_trainer(intermediate_checkpoint_model_package_group=arn)
        doc = json.loads(trainer._build_job_config_document())
        assert doc["ModelPackageConfig"]["IntermediateCheckpointModelPackageGroupArn"] == arn

    def test_intermediate_checkpoint_mpg_always_present(self):
        trainer = self._make_trainer()
        doc = json.loads(trainer._build_job_config_document())
        assert "IntermediateCheckpointModelPackageGroupArn" in doc["ModelPackageConfig"]

    def test_round_trip_serialization(self):
        trainer = self._make_trainer(hyperparameters={"k": "v"})
        doc_str = trainer._build_job_config_document()
        doc = json.loads(doc_str)
        assert json.dumps(doc)  # re-serializable
        assert doc["TrainingConfig"]["HyperParameters"]["k"] == "v"

    def test_validation_dataset_s3(self):
        val_s3 = "s3://my-bucket/val/data.jsonl"
        trainer = self._make_trainer(validation_dataset=val_s3)
        doc = json.loads(
            trainer._build_job_config_document()
        )
        channels = doc["InputDataConfig"]
        assert len(channels) == 2
        assert channels[0]["ChannelName"] == "train"
        assert channels[1]["ChannelName"] == "validation"
        assert channels[1]["DataSource"]["S3DataSource"]["S3Uri"] == val_s3

    def test_validation_dataset_arn(self):
        trainer = self._make_trainer(validation_dataset=DATASET_ARN)
        doc = json.loads(
            trainer._build_job_config_document()
        )
        channels = doc["InputDataConfig"]
        assert len(channels) == 2
        assert channels[1]["ChannelName"] == "validation"
        assert channels[1]["DataSource"]["DatasetSource"]["DatasetArn"] == DATASET_ARN

    def test_validation_dataset_object(self):
        ds = MagicMock(spec=DataSet)
        ds.arn = DATASET_ARN
        trainer = self._make_trainer(validation_dataset=ds)
        doc = json.loads(
            trainer._build_job_config_document()
        )
        channels = doc["InputDataConfig"]
        assert len(channels) == 2
        assert channels[1]["DataSource"]["DatasetSource"]["DatasetArn"] == DATASET_ARN

    def test_no_validation_dataset(self):
        trainer = self._make_trainer()
        doc = json.loads(
            trainer._build_job_config_document()
        )
        assert len(doc["InputDataConfig"]) == 1


class TestMlflowConfigNone:
    """Test _build_mlflow_config when mlflow_app_arn is None."""

    def _make_trainer(self, **overrides):
        trainer = object.__new__(MultiTurnRLTrainer)
        trainer.agent_env = BEDROCK_AGENT_ARN
        trainer.bedrock_agentcore_qualifier = "DEFAULT"
        trainer.s3_output_path = S3_OUTPUT
        trainer.output_model_package_group = MPG_ARN
        trainer.intermediate_checkpoint_model_package_group = "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/default-ckpt-mpg"
        trainer.mlflow_app_arn = overrides.get("mlflow_app_arn")
        trainer.mlflow_experiment_name = overrides.get("mlflow_experiment_name")
        trainer.mlflow_run_name = overrides.get("mlflow_run_name")
        trainer.accept_eula = True
        trainer.kms_key_arn = None
        trainer.networking = None
        trainer.model = "test-model-id"
        trainer.validation_dataset = None
        trainer._model_arn = MODEL_ARN
        trainer.training_dataset = S3_DATA
        trainer._final_hyperparameters = {}
        trainer._hp_defaults = {}
        trainer.sagemaker_session = None
        return trainer

    @patch("sagemaker.train.multi_turn_rl_trainer._resolve_mlflow_resource_arn", return_value=None)
    @patch("sagemaker.train.multi_turn_rl_trainer.TrainDefaults")
    def test_mlflow_config_none_when_no_arn_resolved(self, mock_defaults, mock_resolve):
        trainer = self._make_trainer()
        result = trainer._build_mlflow_config()
        assert result is None

    @patch("sagemaker.train.multi_turn_rl_trainer._resolve_mlflow_resource_arn", return_value=MLFLOW_ARN)
    @patch("sagemaker.train.multi_turn_rl_trainer.TrainDefaults")
    def test_mlflow_config_resolved_from_prod(self, mock_defaults, mock_resolve):
        trainer = self._make_trainer()
        result = trainer._build_mlflow_config()
        assert result["MlflowResourceArn"] == MLFLOW_ARN

    def test_mlflow_config_explicit_arn(self):
        trainer = self._make_trainer(mlflow_app_arn=MLFLOW_ARN)
        result = trainer._build_mlflow_config()
        assert result["MlflowResourceArn"] == MLFLOW_ARN

    def test_mlflow_config_omitted_from_training_config(self):
        trainer = self._make_trainer()
        trainer.mlflow_app_arn = None
        with patch("sagemaker.train.multi_turn_rl_trainer._resolve_mlflow_resource_arn", return_value=None), \
             patch("sagemaker.train.multi_turn_rl_trainer.TrainDefaults"):
            doc = json.loads(trainer._build_job_config_document())
        assert "MlflowConfig" not in doc["TrainingConfig"]


class TestResolveModelPackageGroup:
    """Test _resolve_model_package_group."""

    def _make_trainer(self):
        trainer = object.__new__(MultiTurnRLTrainer)
        trainer._model_name = "test-model"
        trainer.sagemaker_session = None
        return trainer

    def _mock_session(self):
        session = MagicMock()
        session.boto_session = MagicMock()
        session.boto_session.region_name = "us-west-2"
        return session

    @patch("sagemaker.train.multi_turn_rl_trainer.ModelPackageGroup.get")
    def test_string_name_calls_get(self, mock_get):
        mock_mpg = MagicMock()
        mock_mpg.model_package_group_arn = MPG_ARN
        mock_get.return_value = mock_mpg

        trainer = self._make_trainer()
        result = trainer._resolve_model_package_group("model", "my-group", self._mock_session())
        assert result == MPG_ARN
        mock_get.assert_called_once()

    @patch("sagemaker.train.multi_turn_rl_trainer.ModelPackageGroup.get")
    def test_arn_string_calls_get(self, mock_get):
        mock_mpg = MagicMock()
        mock_mpg.model_package_group_arn = MPG_ARN
        mock_get.return_value = mock_mpg

        trainer = self._make_trainer()
        result = trainer._resolve_model_package_group("model", MPG_ARN, self._mock_session())
        assert result == MPG_ARN
        mock_get.assert_called_once()

    def test_mpg_object_returns_arn(self):
        from sagemaker.core.resources import ModelPackageGroup as MPG
        mock_mpg = MagicMock(spec=MPG)
        mock_mpg.model_package_group_arn = MPG_ARN

        trainer = self._make_trainer()
        result = trainer._resolve_model_package_group("model", mock_mpg, self._mock_session())
        assert result == MPG_ARN

    @patch("sagemaker.train.multi_turn_rl_trainer.ModelPackageGroup.get")
    def test_none_with_model_package_derives(self, mock_get):
        mock_model = MagicMock(spec=ModelPackage)
        mock_model.model_package_group_name = "derived-group"
        mock_mpg = MagicMock()
        mock_mpg.model_package_group_arn = MPG_ARN
        mock_get.return_value = mock_mpg

        trainer = self._make_trainer()
        result = trainer._resolve_model_package_group(mock_model, None, self._mock_session())
        assert result == MPG_ARN

    @patch("sagemaker.train.multi_turn_rl_trainer.ModelPackageGroup.create")
    @patch("sagemaker.train.multi_turn_rl_trainer.ModelPackageGroup.get")
    def test_none_auto_creates_on_miss(self, mock_get, mock_create):
        mock_get.side_effect = Exception("does not exist")
        mock_mpg = MagicMock()
        mock_mpg.model_package_group_arn = "arn:aws:sagemaker:us-west-2:123:model-package-group/test-model-mtrl-mpg"
        mock_create.return_value = mock_mpg

        trainer = self._make_trainer()
        result = trainer._resolve_model_package_group("test-model", None, self._mock_session())
        assert "test-model-mtrl-mpg" in result
        mock_create.assert_called_once()

    @patch("sagemaker.train.multi_turn_rl_trainer.ModelPackageGroup.get")
    def test_none_reuses_existing(self, mock_get):
        mock_mpg = MagicMock()
        mock_mpg.model_package_group_arn = "arn:aws:sagemaker:us-west-2:123:model-package-group/test-model-mtrl-mpg"
        mock_get.return_value = mock_mpg

        trainer = self._make_trainer()
        result = trainer._resolve_model_package_group("test-model", None, self._mock_session())
        assert "test-model-mtrl-mpg" in result

    @patch("sagemaker.train.multi_turn_rl_trainer.ModelPackageGroup.create")
    @patch("sagemaker.train.multi_turn_rl_trainer.ModelPackageGroup.get")
    def test_none_raises_on_create_failure(self, mock_get, mock_create):
        mock_get.side_effect = Exception("does not exist")
        mock_create.side_effect = Exception("permission denied")

        trainer = self._make_trainer()
        with pytest.raises(ValueError, match="Failed to create"):
            trainer._resolve_model_package_group("test-model", None, self._mock_session())


    @patch("sagemaker.train.multi_turn_rl_trainer.ModelPackageGroup.create")
    @patch("sagemaker.train.multi_turn_rl_trainer.ModelPackageGroup.get")
    def test_nova_model_creates_restricted_mpg(self, mock_get, mock_create):
        mock_get.side_effect = Exception("does not exist")
        mock_mpg = MagicMock()
        mock_mpg.model_package_group_arn = "arn:aws:sagemaker:us-west-2:123:model-package-group/amazon-nova-pro-mtrl-mpg"
        mock_create.return_value = mock_mpg

        trainer = self._make_trainer()
        trainer._model_name = "amazon-nova-pro"
        result = trainer._resolve_model_package_group("amazon-nova-pro", None, self._mock_session())
        assert "amazon-nova-pro-mtrl-mpg" in result
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["managed_configuration"].managed_storage_type == "Restricted"


class TestAgentRuntimeIdPattern:
    def test_valid_runtime_id(self):
        assert AGENT_RUNTIME_ID_PATTERN.match("myRuntime-aBcDeFgHiJ")

    def test_runtime_id_must_start_with_letter(self):
        assert not AGENT_RUNTIME_ID_PATTERN.match("1badStart-aBcDeFgHiJ")

    def test_arn_does_not_match(self):
        assert not AGENT_RUNTIME_ID_PATTERN.match(BEDROCK_AGENT_ARN)

    def test_lambda_arn_does_not_match(self):
        assert not AGENT_RUNTIME_ID_PATTERN.match(LAMBDA_ARN)


class TestResolveAgentRuntimeArn:
    @patch("sagemaker.train.multi_turn_rl_trainer.boto3.Session")
    def test_resolves_id_to_arn(self, mock_session_cls):
        mock_client = MagicMock()
        mock_client.get_agent_runtime.return_value = {
            "agentRuntimeArn": BEDROCK_AGENT_ARN,
        }
        mock_session_cls.return_value.client.return_value = mock_client

        result = _resolve_agent_runtime_arn("myRuntime-aBcDeFgHiJ")
        assert result == BEDROCK_AGENT_ARN
        mock_client.get_agent_runtime.assert_called_once_with(
            agentRuntimeId="myRuntime-aBcDeFgHiJ"
        )

    @patch("sagemaker.train.multi_turn_rl_trainer.boto3.Session")
    def test_raises_on_missing_arn(self, mock_session_cls):
        mock_client = MagicMock()
        mock_client.get_agent_runtime.return_value = {}
        mock_session_cls.return_value.client.return_value = mock_client

        with pytest.raises(ValueError, match="returned no ARN"):
            _resolve_agent_runtime_arn("myRuntime-aBcDeFgHiJ")

    @patch("sagemaker.train.multi_turn_rl_trainer.boto3.Session")
    def test_raises_on_api_error(self, mock_session_cls):
        mock_client = MagicMock()
        mock_client.get_agent_runtime.side_effect = Exception("not found")
        mock_session_cls.return_value.client.return_value = mock_client

        with pytest.raises(ValueError, match="Failed to resolve"):
            _resolve_agent_runtime_arn("myRuntime-aBcDeFgHiJ")


class TestValidationAgentRuntimeId:
    def test_valid_runtime_id_accepted(self):
        MultiTurnRLTrainer._validate_agent_config("myRuntime-aBcDeFgHiJ")

    def test_invalid_string_rejected(self):
        with pytest.raises(ValueError, match="Invalid agent_env"):
            MultiTurnRLTrainer._validate_agent_config("not-valid")


class TestListSupportedModels:
    @patch("sagemaker.train.multi_turn_rl_trainer._list_all_mtrl_models")
    def test_returns_models(self, mock_list):
        mock_list.return_value = ["model-a", "model-b"]
        result = MultiTurnRLTrainer.list_supported_models()
        assert result == ["model-a", "model-b"]
        mock_list.assert_called_once_with(session=None)

    @patch("sagemaker.train.multi_turn_rl_trainer._list_all_mtrl_models")
    def test_passes_session(self, mock_list):
        mock_list.return_value = []
        mock_session = MagicMock()
        MultiTurnRLTrainer.list_supported_models(session=mock_session)
        mock_list.assert_called_once_with(session=mock_session)


class TestListHubModelsByRecipe:
    """Tests for _list_hub_models_by_recipe in recipe_utils."""

    @patch("sagemaker.train.common_utils.recipe_utils.boto3.Session")
    def test_finds_mtrl_models(self, mock_session_cls):
        mock_client = MagicMock()
        mock_session_cls.return_value.client.return_value = mock_client

        mock_client.list_hub_contents.return_value = {
            "HubContentSummaries": [
                {
                    "HubContentName": "model-with-mtrl",
                    "HubContentSearchKeywords": [
                        "@recipe:finetuning_mtrl_lora",
                        "@framework:huggingface",
                    ],
                },
                {
                    "HubContentName": "model-without-mtrl",
                    "HubContentSearchKeywords": [
                        "@recipe:finetuning_sft_lora",
                    ],
                },
            ],
        }

        from sagemaker.train.common_utils.recipe_utils import _list_hub_models_by_recipe
        result = _list_hub_models_by_recipe(recipe_type="FineTuning", technique="MTRL")
        assert result == ["model-with-mtrl"]
        mock_client.describe_hub_content.assert_not_called()

    @patch("sagemaker.train.common_utils.recipe_utils.boto3.Session")
    def test_finds_evaluation_models(self, mock_session_cls):
        mock_client = MagicMock()
        mock_session_cls.return_value.client.return_value = mock_client

        mock_client.list_hub_contents.return_value = {
            "HubContentSummaries": [
                {
                    "HubContentName": "model-eval",
                    "HubContentSearchKeywords": [
                        "@recipe:evaluation_mtrlevaluation_deterministicevaluation",
                    ],
                },
            ],
        }

        from sagemaker.train.common_utils.recipe_utils import _list_hub_models_by_recipe
        result = _list_hub_models_by_recipe(recipe_type="Evaluation", technique="MTRLEvaluation")
        assert result == ["model-eval"]

    @patch("sagemaker.train.common_utils.recipe_utils.boto3.Session")
    def test_paginates(self, mock_session_cls):
        mock_client = MagicMock()
        mock_session_cls.return_value.client.return_value = mock_client

        mock_client.list_hub_contents.side_effect = [
            {
                "HubContentSummaries": [
                    {
                        "HubContentName": "model-a",
                        "HubContentSearchKeywords": ["@recipe:finetuning_mtrl_lora"],
                    },
                ],
                "NextToken": "tok",
            },
            {
                "HubContentSummaries": [
                    {
                        "HubContentName": "model-b",
                        "HubContentSearchKeywords": ["@recipe:finetuning_mtrl_lora"],
                    },
                ],
            },
        ]

        from sagemaker.train.common_utils.recipe_utils import _list_hub_models_by_recipe
        result = _list_hub_models_by_recipe(recipe_type="FineTuning", technique="MTRL")
        assert result == ["model-a", "model-b"]
        assert mock_client.list_hub_contents.call_count == 2

    @patch("sagemaker.train.common_utils.recipe_utils.boto3.Session")
    def test_no_keywords_skips_model(self, mock_session_cls):
        mock_client = MagicMock()
        mock_session_cls.return_value.client.return_value = mock_client

        mock_client.list_hub_contents.return_value = {
            "HubContentSummaries": [
                {"HubContentName": "model-no-keywords"},
            ],
        }

        from sagemaker.train.common_utils.recipe_utils import _list_hub_models_by_recipe
        result = _list_hub_models_by_recipe(recipe_type="FineTuning", technique="MTRL")
        assert result == []

    def test_invalid_recipe_type_raises(self):
        from sagemaker.train.common_utils.recipe_utils import _list_hub_models_by_recipe
        with pytest.raises(ValueError, match="recipe_type must be"):
            _list_hub_models_by_recipe(recipe_type="Invalid", technique="MTRL")


class TestListAgentRuntimes:
    @patch("sagemaker.train.multi_turn_rl_trainer.boto3.Session")
    def test_lists_runtimes(self, mock_session_cls):
        mock_client = MagicMock()
        mock_session_cls.return_value.client.return_value = mock_client

        mock_client.list_agent_runtimes.return_value = {
            "agentRuntimes": [
                {
                    "agentRuntimeArn": "arn:aws:bedrock-agentcore:us-west-2:123:runtime/rt1",
                    "agentRuntimeId": "myAgent-aBcDeFgHiJ",
                    "agentRuntimeName": "mtrl-agent",
                    "status": "READY",
                },
                {
                    "agentRuntimeArn": "arn:aws:bedrock-agentcore:us-west-2:123:runtime/rt2",
                    "agentRuntimeId": "other-aBcDeFgHiJ",
                    "agentRuntimeName": "other-agent",
                    "status": "READY",
                },
            ],
        }

        result = MultiTurnRLTrainer.list_bedrock_agentcore_runtimes()
        assert len(result) == 2
        assert result[0]["name"] == "mtrl-agent"
        assert result[1]["name"] == "other-agent"

    @patch("sagemaker.train.multi_turn_rl_trainer.boto3.Session")
    def test_paginates(self, mock_session_cls):
        mock_client = MagicMock()
        mock_session_cls.return_value.client.return_value = mock_client

        mock_client.list_agent_runtimes.side_effect = [
            {
                "agentRuntimes": [
                    {"agentRuntimeArn": "arn1", "agentRuntimeId": "a-aBcDeFgHiJ",
                     "agentRuntimeName": "a", "status": "READY"},
                ],
                "nextToken": "tok",
            },
            {
                "agentRuntimes": [
                    {"agentRuntimeArn": "arn2", "agentRuntimeId": "b-aBcDeFgHiJ",
                     "agentRuntimeName": "b", "status": "READY"},
                ],
            },
        ]

        result = MultiTurnRLTrainer.list_bedrock_agentcore_runtimes()
        assert len(result) == 2
        assert mock_client.list_agent_runtimes.call_count == 2
