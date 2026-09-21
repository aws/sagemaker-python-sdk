"""Unit tests for MultiTurnRLTrainer."""
import json
import warnings
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from sagemaker.ai_registry.dataset import DataSet
from sagemaker.core.resources import Base, Job, ModelPackage, MlflowApp
from sagemaker.core.shapes import Tag
from sagemaker.core.workflow.execution_variables import ExecutionVariable
from sagemaker.core.workflow.functions import Join
from sagemaker.core.workflow.parameters import ParameterString
from sagemaker.core.workflow.pipeline_context import (
    PipelineSession,
    _JobStepArguments,
    _StepArguments,
    retrieve_caller_name,
)
from sagemaker.core.workflow.utilities import execute_job_functions
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

    @patch("sagemaker.train.common_utils.recipe_utils.boto3.Session")
    def test_finds_models_with_bare_keyword_no_strategy(self, mock_session_cls):
        """Techniques whose recipes carry no strategy suffix are tagged with the
        bare ``@recipe:finetuning_{technique}`` keyword (e.g. CPT). The matcher
        must find these, not just ``{base}_{strategy}`` forms."""
        mock_client = MagicMock()
        mock_session_cls.return_value.client.return_value = mock_client

        mock_client.list_hub_contents.return_value = {
            "HubContentSummaries": [
                {
                    "HubContentName": "model-cpt-bare",
                    "HubContentSearchKeywords": ["@recipe:finetuning_cpt"],
                },
                {
                    "HubContentName": "model-cpt-suffixed",
                    "HubContentSearchKeywords": ["@recipe:finetuning_cpt_full"],
                },
            ],
        }

        from sagemaker.train.common_utils.recipe_utils import _list_hub_models_by_recipe
        result = _list_hub_models_by_recipe(recipe_type="FineTuning", technique="CPT")
        assert result == ["model-cpt-bare", "model-cpt-suffixed"]

    @patch("sagemaker.train.common_utils.recipe_utils.boto3.Session")
    def test_does_not_match_technique_sharing_a_prefix(self, mock_session_cls):
        """A shorter technique must not match a longer one that merely shares its
        prefix (e.g. ``rl`` must not match ``rlvr``)."""
        mock_client = MagicMock()
        mock_session_cls.return_value.client.return_value = mock_client

        mock_client.list_hub_contents.return_value = {
            "HubContentSummaries": [
                {
                    "HubContentName": "model-rlvr",
                    "HubContentSearchKeywords": ["@recipe:finetuning_rlvr_lora"],
                },
            ],
        }

        from sagemaker.train.common_utils.recipe_utils import _list_hub_models_by_recipe
        result = _list_hub_models_by_recipe(recipe_type="FineTuning", technique="rl")
        assert result == []


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


class TestDryRun:
    """Test dry_run=True skips job submission and MLflow creation."""

    def _make_trainer(self):
        """Create a trainer with mocked internals for dry_run testing."""
        trainer = object.__new__(MultiTurnRLTrainer)
        trainer.agent_env = BEDROCK_AGENT_ARN
        trainer.bedrock_agentcore_qualifier = "DEFAULT"
        trainer.s3_output_path = S3_OUTPUT
        trainer.output_model_package_group = MPG_ARN
        trainer.intermediate_checkpoint_model_package_group = "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/ckpt-mpg"
        trainer.mlflow_app_arn = None  # Force MLflow resolution
        trainer.mlflow_experiment_name = None
        trainer.mlflow_run_name = None
        trainer.accept_eula = True
        trainer.kms_key_arn = None
        trainer.networking = None
        trainer.model = "test-model-id"
        trainer.validation_dataset = None
        trainer._model_arn = MODEL_ARN
        trainer._model_name = "test-model"
        trainer.training_dataset = S3_DATA
        trainer.hyperparameters = MagicMock()
        trainer.hyperparameters.to_dict.return_value = {}
        trainer.hyperparameters._specs = {}
        trainer._hp_defaults = {}
        trainer._final_hyperparameters = {}
        mock_session = MagicMock()
        mock_session.sagemaker_config = {"SchemaVersion": "1.0"}
        mock_session.boto_session.region_name = "us-west-2"
        trainer.sagemaker_session = mock_session
        trainer.role = "arn:aws:iam::123456789012:role/TestRole"
        trainer.base_job_name = "test-mtrl"
        trainer._recipe_path = None
        trainer._overrides = None
        trainer._resolved_recipe_cache = None
        return trainer

    @patch("sagemaker.train.multi_turn_rl_trainer._resolve_mlflow_resource_arn")
    @patch("sagemaker.train.multi_turn_rl_trainer.Job")
    @patch("sagemaker.train.multi_turn_rl_trainer.TrainDefaults.get_role")
    def test_dry_run_skips_job_creation(self, mock_get_role, mock_job_cls, mock_resolve_mlflow):
        """dry_run=True returns None without calling Job.create."""
        mock_resolve_mlflow.return_value = None
        mock_get_role.return_value = "arn:aws:iam::123456789012:role/TestRole"
        trainer = self._make_trainer()

        result = trainer.train(dry_run=True)

        assert result is None
        mock_job_cls.create.assert_not_called()

    @patch("sagemaker.train.multi_turn_rl_trainer._resolve_mlflow_resource_arn")
    @patch("sagemaker.train.multi_turn_rl_trainer.Job")
    @patch("sagemaker.train.multi_turn_rl_trainer.TrainDefaults.get_role")
    def test_dry_run_passes_flag_to_mlflow_resolver(self, mock_get_role, mock_job_cls, mock_resolve_mlflow):
        """dry_run=True is forwarded to _resolve_mlflow_resource_arn."""
        mock_resolve_mlflow.return_value = None
        mock_get_role.return_value = "arn:aws:iam::123456789012:role/TestRole"
        trainer = self._make_trainer()

        trainer.train(dry_run=True)

        # Verify dry_run=True was passed
        call_kwargs = mock_resolve_mlflow.call_args[1]
        assert call_kwargs["dry_run"] is True


PINNED_JOB_NAME = "test-model-mtrl-1757000000-abc123"
ROLE_ARN = "arn:aws:iam::123456789012:role/SageMakerRole"


class _Hyperparameters:
    """Minimal stand-in for the hyperparameters object `train()` snapshots."""

    def __init__(self, values=None):
        self._values = values or {}

    def to_dict(self):
        return dict(self._values)


class TestCaptureTagCoercion:
    """The capture path must emit the same wire tags as `Job.create`.

    The capture path never reaches `Job.create`, so `train()` coerces dict-form
    tags through the `Tag` model before `serialize`, which is the same parse
    `Job.create` applies. Population: `_get_jumpstart_tags` lowercase dicts and
    `BaseTrainer.tags` `Tag` objects.
    """

    @staticmethod
    def _capture_wire_tags(tags):
        """What the capture path sends: coerce to `Tag`, then `serialize`."""
        from sagemaker.core.utils.utils import serialize

        return serialize([Tag(**tag) if isinstance(tag, dict) else tag for tag in tags])

    @staticmethod
    def _job_create_wire_tags(tags):
        """What `Job.create` really sends for `tags`, measured not derived."""
        mock_client = MagicMock()
        with patch.object(Base, "get_sagemaker_client", return_value=mock_client):
            try:
                Job.create(
                    job_name=PINNED_JOB_NAME,
                    role_arn=ROLE_ARN,
                    job_category=JOB_CATEGORY,
                    job_config_schema_version=JOB_CONFIG_SCHEMA_VERSION,
                    job_config_document="{}",
                    tags=tags,
                )
            except Exception:
                # Constructing the `Job` resource from a MagicMock response fails after
                # the call. The call itself is what is under test.
                pass
        assert mock_client.create_job.call_args is not None, "create_job was not reached"
        return mock_client.create_job.call_args.kwargs["Tags"]

    def test_matches_job_create_for_lowercase_dicts(self):
        tags = [{"key": "sagemaker-sdk:jumpstart-model-id", "value": "test-model"}]
        assert self._capture_wire_tags(tags) == self._job_create_wire_tags(tags)

    def test_matches_job_create_for_tag_objects(self):
        tags = [Tag(key="Project", value="beta")]
        assert self._capture_wire_tags(tags) == self._job_create_wire_tags(tags)

    def test_matches_job_create_for_both_forms_in_one_list(self):
        tags = [
            {"key": "sagemaker-sdk:jumpstart-model-id", "value": "test-model"},
            Tag(key="Project", value="beta"),
        ]
        assert self._capture_wire_tags(tags) == self._job_create_wire_tags(tags)

    def test_empty_list_is_preserved_not_dropped(self):
        """`Job.create` sends `Tags: []` rather than omitting the key."""
        assert self._capture_wire_tags([]) == self._job_create_wire_tags([]) == []


class TestPipelineCapture:
    """Under a `PipelineSession`, `train()` must capture rather than submit.

    The producer half of composing a `JobStep` over `CreateJob`: declare a caller name
    the resolver recognises, assemble the request `Job.create` would have assembled,
    and hand it to the session instead of the service.
    """

    @pytest.fixture(autouse=True)
    def _skip_role_validation(self, monkeypatch):
        """`TrainDefaults.get_role` validates even an explicit role against live IAM."""
        monkeypatch.setattr(
            "sagemaker.train.defaults.resolve_and_validate_role",
            lambda provided_role=None, **kwargs: provided_role or ROLE_ARN,
        )

    @staticmethod
    def _pipeline_session():
        """A real `PipelineSession` with only its outbound edges stubbed.

        `_intercept_create_request` is deliberately NOT mocked: these tests assert on
        the `_JobStepArguments` it really builds.
        """
        session = PipelineSession()
        session.sagemaker_client = MagicMock()
        session.sagemaker_config = {}
        return session

    @staticmethod
    def _direct_session():
        """A non-pipeline session stub.

        `sagemaker_config` is a real dict because `_telemetry_emitter` resolves the
        opt-out flag through it and jsonschema rejects a MagicMock there.
        """
        session = MagicMock()
        session.sagemaker_config = {}
        return session

    @staticmethod
    def _make_trainer(sagemaker_session=None, **overrides):
        """A trainer with `__init__`'s resolution already done, as the other suites do."""
        trainer = object.__new__(MultiTurnRLTrainer)
        trainer.agent_env = BEDROCK_AGENT_ARN
        trainer.bedrock_agentcore_qualifier = "DEFAULT"
        trainer.s3_output_path = overrides.get("s3_output_path", S3_OUTPUT)
        trainer.output_model_package_group = MPG_ARN
        trainer.intermediate_checkpoint_model_package_group = (
            "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/default-ckpt-mpg"
        )
        trainer.mlflow_app_arn = MLFLOW_ARN
        trainer.mlflow_experiment_name = None
        trainer.mlflow_run_name = None
        trainer.accept_eula = True
        trainer.kms_key_arn = overrides.get("kms_key_arn")
        trainer.networking = None
        trainer.model = "test-model-id"
        trainer.validation_dataset = None
        trainer._model_arn = MODEL_ARN
        trainer.training_dataset = overrides.get("training_dataset", S3_DATA)
        trainer._hp_defaults = {}
        trainer.hyperparameters = _Hyperparameters(overrides.get("hyperparameters"))
        trainer._model_name = "test-model"
        trainer.base_job_name = "test-model-mtrl"
        trainer.role = ROLE_ARN
        trainer.tags = overrides.get("tags")
        trainer.sagemaker_session = sagemaker_session
        trainer._latest_job = overrides.get("latest_job")
        trainer._recipe_path = None
        trainer._overrides = None
        trainer._resolved_recipe_cache = None
        return trainer

    @staticmethod
    def _resolve(value):
        """Resolve an encoded document to text, the way the service would."""
        if isinstance(value, str):
            return value
        if isinstance(value, Join):
            return value.on.join(TestPipelineCapture._resolve(item) for item in value.values)
        if isinstance(value, ExecutionVariable):
            return "EXEC-ID"
        if isinstance(value, ParameterString):
            return "RESOLVED-PARAM"
        raise AssertionError("unexpected value in encoded document: %r" % (value,))

    @staticmethod
    def _pinned_name():
        return patch(
            "sagemaker.train.multi_turn_rl_trainer._get_unique_name",
            return_value=PINNED_JOB_NAME,
        )

    def _capture(self, trainer, session):
        """Drive the decorator, then replay the captured call as a step would."""
        with self._pinned_name():
            step_args = trainer.train()
            assert isinstance(step_args, _StepArguments)
            execute_job_functions(step_args)
        return session.context

    # --- the declared caller name ---------------------------------------------

    def test_declares_the_create_job_caller_name(self):
        """`JobStep`'s `expected_caller={"create_job"}` guard accepts only this value."""
        assert MultiTurnRLTrainer._pipeline_caller_name == "create_job"

    def test_caller_name_is_a_plain_string_on_the_class(self):
        assert isinstance(MultiTurnRLTrainer.__dict__["_pipeline_caller_name"], str)

    def test_resolver_returns_the_declared_name(self):
        """Nothing else in the resolver produces "create_job".

        The duck-typed branches return run/train/transform/tune, so without this
        declaration `JobStep`'s guard is unreachable from any producer.
        """
        assert retrieve_caller_name(self._make_trainer()) == "create_job"

    def test_resolver_does_not_mistake_the_trainer_for_a_model_trainer(self):
        trainer = self._make_trainer()
        # The trainer carries BaseTrainer's attributes the train duck-typing keys on,
        # so the declared name winning here is the resolution ORDER under test.
        assert retrieve_caller_name(trainer) != "train"

    def test_declaration_wins_over_the_model_trainer_branch(self):
        """Why the resolver checks the declaration first, for this class specifically.

        Load-bearing now, not hypothetically. This class has a `train()` method and, by
        subclassing `BaseTrainer`, already carries `input_data_config`, which is one of
        the two markers the CreateTrainingJob branch accepts, so it already matches that
        branch structurally. The declaration is the only thing keeping it out. Removing
        it does not yield `None`, it resolves "train" and composes a `TrainingStep` over
        `CreateJob` arguments.

        Setting `training_image` here adds the branch's other marker, so the instance
        carries both of them rather than only the inherited one. The declaration still
        wins, which is what pins the ordering.
        """
        trainer = self._make_trainer()
        trainer.training_image = "123456789012.dkr.ecr.us-west-2.amazonaws.com/img:latest"
        assert retrieve_caller_name(trainer) == "create_job"

    # --- capture instead of submission ----------------------------------------

    def test_train_returns_step_arguments_and_submits_nothing(self):
        session = self._pipeline_session()
        trainer = self._make_trainer(sagemaker_session=session)
        with self._pinned_name(), patch(
            "sagemaker.train.multi_turn_rl_trainer.Job.create"
        ) as mock_create:
            step_args = trainer.train()
        assert isinstance(step_args, _StepArguments)
        assert step_args.caller_name == "create_job"
        mock_create.assert_not_called()

    def test_executing_the_captured_call_populates_the_session_context(self):
        session = self._pipeline_session()
        trainer = self._make_trainer(sagemaker_session=session)
        with patch("sagemaker.train.multi_turn_rl_trainer.Job.create") as mock_create:
            context = self._capture(trainer, session)
        mock_create.assert_not_called()
        assert isinstance(context, _JobStepArguments)
        assert context.caller_name == "create_job"

    def test_captured_request_carries_the_create_job_envelope(self):
        session = self._pipeline_session()
        trainer = self._make_trainer(sagemaker_session=session)
        args = self._capture(trainer, session).args
        assert set(args) == {
            "JobName",
            "RoleArn",
            "JobCategory",
            "JobConfigSchemaVersion",
            "JobConfigDocument",
            "Tags",
        }
        assert args["JobCategory"] == JOB_CATEGORY
        assert args["JobConfigSchemaVersion"] == JOB_CONFIG_SCHEMA_VERSION
        assert args["RoleArn"] == ROLE_ARN

    def test_session_and_region_are_not_request_members(self):
        """They are `Job.create`'s client-resolution arguments, not `CreateJob` keys."""
        session = self._pipeline_session()
        trainer = self._make_trainer(sagemaker_session=session)
        args = self._capture(trainer, session).args
        assert "session" not in args
        assert "region" not in args

    def test_job_name_is_left_in_the_captured_request(self):
        """Popping it would make `trim_request_dict`'s custom-prefix branch unreachable.

        That branch is `if job_key in request_dict:`, so a producer that pops the key
        silently gives a user who opted into `use_custom_job_prefix` nothing at all.
        The upstream finetune trainers pop it; this does not.
        """
        session = self._pipeline_session()
        trainer = self._make_trainer(sagemaker_session=session)
        assert self._capture(trainer, session).args["JobName"] == PINNED_JOB_NAME

    def test_capture_leaves_latest_job_untouched(self):
        """The early return skips the assignment, so a prior handle is not clobbered."""
        previous = MagicMock()
        session = self._pipeline_session()
        trainer = self._make_trainer(sagemaker_session=session, latest_job=previous)
        self._capture(trainer, session)
        assert trainer._latest_job is previous

    def test_output_model_package_arn_reports_none_after_capture(self):
        """The only property reading `_latest_job` already guards on None."""
        session = self._pipeline_session()
        trainer = self._make_trainer(sagemaker_session=session)
        self._capture(trainer, session)
        assert trainer._latest_job is None
        assert trainer.output_model_package_arn is None

    def test_wait_is_overridden_and_announced(self):
        """`wait` is meaningless under a pipeline session, and is not silently dropped.

        `runnable_by_pipeline` forces it to False and warns before the body runs, so
        the override is announced by the framework rather than absorbed here.
        """
        session = self._pipeline_session()
        trainer = self._make_trainer(sagemaker_session=session)
        with self._pinned_name(), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            step_args = trainer.train(wait=True)
        assert step_args.func_kwargs.get("wait") is False
        assert any(
            "No Wait" in str(w.message) for w in caught
        ), [str(w.message) for w in caught]

    # --- parity with the direct submission path -------------------------------

    def _direct_create_kwargs(self, trainer):
        """Run the direct path and record what it hands to `Job.create`."""
        recorded = {}

        def _record(**kwargs):
            recorded.update(kwargs)
            return MagicMock()

        with self._pinned_name(), patch(
            "sagemaker.train.multi_turn_rl_trainer.Job.create", side_effect=_record
        ), patch(
            "sagemaker.train.multi_turn_rl_trainer.AgentRFTJob.from_job",
            return_value=MagicMock(),
        ):
            trainer.train(wait=False)
        return recorded

    @staticmethod
    def _job_create_wire_request(create_kwargs, job_config_document):
        """Push `create_kwargs` through the real `Job.create` and capture the wire dict.

        `session` and `region` are dropped: they select the client rather than form part
        of the request, which is exactly why the captured request omits them. The
        document is substituted so the comparison isolates the envelope from the one
        deliberate difference between the routes (see the document tests below).
        """
        replay = {k: v for k, v in create_kwargs.items() if k not in ("session", "region")}
        replay["job_config_document"] = job_config_document
        mock_client = MagicMock()
        with patch.object(Base, "get_sagemaker_client", return_value=mock_client):
            try:
                Job.create(**replay)
            except Exception:
                pass
        assert mock_client.create_job.call_args is not None, "create_job was not reached"
        return mock_client.create_job.call_args.kwargs

    def test_captured_request_matches_what_job_create_would_send(self):
        """Criterion: the hand-built dict equals the resource layer's own output.

        Demonstrated by running both routes and pushing the direct route's arguments
        through the real `Job.create`, `populate_chained_attributes` and `serialize`,
        rather than by reasoning about what they do.
        """
        tags = [Tag(key="Project", value="beta")]
        direct = self._direct_create_kwargs(
            self._make_trainer(sagemaker_session=self._direct_session(), tags=tags)
        )
        assert set(direct) == {
            "job_name",
            "job_category",
            "role_arn",
            "job_config_schema_version",
            "job_config_document",
            "tags",
            "session",
            "region",
        }

        session = self._pipeline_session()
        captured = dict(self._capture(
            self._make_trainer(sagemaker_session=session, tags=tags), session
        ).args)
        # The document is captured as the raw config dict for JobStep to scope and
        # encode; compare it to the direct route's config, and the rest to the wire.
        document = captured.pop("JobConfigDocument")
        assert document == json.loads(direct["job_config_document"])
        wire = self._job_create_wire_request(direct, direct["job_config_document"])
        wire.pop("JobConfigDocument")
        assert captured == wire

    def test_captured_request_matches_job_create_with_both_tag_forms(self):
        """The studio dicts and a user `Tag` object converge on the same wire form."""
        tags = [Tag(key="Project", value="beta"), {"key": "Team", "value": "verse"}]
        direct = self._direct_create_kwargs(
            self._make_trainer(sagemaker_session=self._direct_session(), tags=tags)
        )
        session = self._pipeline_session()
        captured = dict(self._capture(
            self._make_trainer(sagemaker_session=session, tags=tags), session
        ).args)
        document = captured.pop("JobConfigDocument")
        assert document == json.loads(direct["job_config_document"])
        wire = self._job_create_wire_request(direct, direct["job_config_document"])
        wire.pop("JobConfigDocument")
        assert captured == wire
        assert captured["Tags"] == [
            {"Key": "sagemaker-sdk:jumpstart-model-id", "Value": "test-model"},
            {"Key": "sagemaker-sdk:jumpstart-hub-name", "Value": "SageMakerPublicHub"},
            {"Key": "Project", "Value": "beta"},
            {"Key": "Team", "Value": "verse"},
        ]

    def test_tags_are_always_present_in_the_captured_request(self):
        """`Job.create` passes `tags` unconditionally, and `serialize` keeps `[]`."""
        session = self._pipeline_session()
        trainer = self._make_trainer(sagemaker_session=session, tags=None)
        args = self._capture(trainer, session).args
        # The two JumpStart tags are always present, so this is never the empty case,
        # but the key is unconditional either way.
        assert "Tags" in args
        assert {t["Key"] for t in args["Tags"]} == {
            "sagemaker-sdk:jumpstart-model-id",
            "sagemaker-sdk:jumpstart-hub-name",
        }

    # --- the JobConfigDocument mechanism --------------------------------------

    def test_document_describes_the_same_config_on_both_routes(self):
        """The capture route hands JobStep the raw dict the direct route serializes."""
        session = self._pipeline_session()
        trainer = self._make_trainer(sagemaker_session=session)
        captured = self._capture(trainer, session).args["JobConfigDocument"]
        direct = self._make_trainer(sagemaker_session=self._direct_session())._build_job_config_document()
        assert isinstance(captured, dict)
        assert captured == json.loads(direct)

    def test_direct_document_is_byte_identical_to_the_previous_behaviour(self):
        """Criterion: the direct path is unchanged, indentation included."""
        trainer = self._make_trainer(sagemaker_session=self._direct_session())
        assert trainer._build_job_config_document() == json.dumps(
            trainer._build_job_config(), indent=2
        )

    def test_pipeline_variable_in_the_job_config_survives_capture(self):
        """`serialize` keeps a `PipelineVariable` intact inside the captured dict,
        so JobStep can encode it into the definition."""
        session = self._pipeline_session()
        trainer = self._make_trainer(
            sagemaker_session=session, s3_output_path=ParameterString(name="OutputPath")
        )
        document = self._capture(trainer, session).args["JobConfigDocument"]
        assert isinstance(document, dict)
        assert isinstance(document["OutputDataConfig"]["S3OutputPath"], ParameterString)

    def test_pipeline_variable_document_would_be_a_type_error_unencoded(self):
        """Locks the reason the encoder is needed rather than assuming it."""
        trainer = self._make_trainer(
            sagemaker_session=self._direct_session(), s3_output_path=ParameterString(name="OutputPath")
        )
        with pytest.raises(TypeError, match="not JSON serializable"):
            trainer._build_job_config_document()

    # --- the direct path is unchanged -----------------------------------------

    def test_direct_path_still_submits_and_returns_a_job(self):
        """Criterion: a normal session behaves exactly as before."""
        job_handle = MagicMock()
        trainer = self._make_trainer(sagemaker_session=self._direct_session())
        with self._pinned_name(), patch(
            "sagemaker.train.multi_turn_rl_trainer.Job.create"
        ) as mock_create, patch(
            "sagemaker.train.multi_turn_rl_trainer.AgentRFTJob.from_job",
            return_value=job_handle,
        ):
            returned = trainer.train(wait=False)
        mock_create.assert_called_once()
        assert returned is job_handle
        assert trainer._latest_job is job_handle
        job_handle.wait.assert_not_called()

    def test_direct_path_honours_wait(self):
        job_handle = MagicMock()
        trainer = self._make_trainer(sagemaker_session=self._direct_session())
        with self._pinned_name(), patch(
            "sagemaker.train.multi_turn_rl_trainer.Job.create"
        ), patch(
            "sagemaker.train.multi_turn_rl_trainer.AgentRFTJob.from_job",
            return_value=job_handle,
        ):
            trainer.train(wait=True)
        job_handle.wait.assert_called_once()

    def test_direct_path_forwards_tags_without_normalisation(self):
        """Unchanged: `Job.create` performs the coercion on this route."""
        tags = [Tag(key="Project", value="beta")]
        direct = self._direct_create_kwargs(
            self._make_trainer(sagemaker_session=self._direct_session(), tags=tags)
        )
        assert direct["tags"][-1] is tags[0]

    def test_dry_run_is_overridden_and_announced_under_a_pipeline_session(self, caplog):
        """`dry_run` cannot validate here -- its path encodes the document eagerly.

        The capture still happens (nothing is submitted either way) and the
        override is announced rather than silent, like the `wait` override.
        """
        session = self._pipeline_session()
        trainer = self._make_trainer(sagemaker_session=session)
        with self._pinned_name(), caplog.at_level("WARNING"):
            step_args = trainer.train(dry_run=True)
            execute_job_functions(step_args)
        assert session.context is not None, "dry_run suppressed the capture"
        assert any("dry_run is ignored" in r.message for r in caplog.records)
