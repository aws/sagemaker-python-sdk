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
import time
from typing import Optional, Set
from unittest import mock
import unittest
from inspect import signature
from mock import Mock

import pytest
from sagemaker.async_inference.async_inference_config import AsyncInferenceConfig

from sagemaker.debugger.profiler_config import ProfilerConfig
from sagemaker.estimator import Estimator
from sagemaker.instance_group import InstanceGroup
from sagemaker.jumpstart.artifacts.environment_variables import (
    _retrieve_default_environment_variables,
)
from sagemaker.jumpstart.artifacts.hyperparameters import _retrieve_default_hyperparameters
from sagemaker.jumpstart.artifacts.metric_definitions import (
    _retrieve_default_training_metric_definitions,
)
from sagemaker.jumpstart.constants import (
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    JUMPSTART_DEFAULT_REGION_NAME,
)
from sagemaker.jumpstart.enums import JumpStartScriptScope, JumpStartTag, JumpStartModelType

from sagemaker.jumpstart.estimator import JumpStartEstimator

from sagemaker.jumpstart.utils import get_jumpstart_content_bucket
from sagemaker.session import Session
from sagemaker.session_settings import SessionSettings
from tests.integ.sagemaker.jumpstart.utils import get_training_dataset_for_model_and_version
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from tests.unit.sagemaker.jumpstart.utils import (
    get_special_model_spec,
    overwrite_dictionary,
)
import boto3


execution_role = "fake role! do not use!"
region = "us-west-2"
sagemaker_session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION
sagemaker_session.get_caller_identity_arn = lambda: execution_role
default_predictor = Predictor("eiifccreeeiuchhnehtlbdecgeeelgjccjvvbbcncnhv", sagemaker_session)
default_predictor_with_presets = Predictor(
    "eiifccreeeiuihlrblivhchuefdckrluliilctfjgknk", sagemaker_session
)


class EstimatorTest(unittest.TestCase):
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_LOGGER")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_LOGGER")
    @mock.patch("sagemaker.utils.sagemaker_timestamp")
    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
    @mock.patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.fit")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_non_prepacked(
        self,
        mock_estimator_deploy: mock.Mock,
        mock_estimator_fit: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_get_model_type: mock.Mock,
        mock_session_estimator: mock.Mock,
        mock_session_model: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
        mock_sagemaker_timestamp: mock.Mock,
        mock_jumpstart_model_factory_logger: mock.Mock,
        mock_jumpstart_estimator_factory_logger: mock.Mock,
    ):
        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        mock_sagemaker_timestamp.return_value = "9876"

        mock_estimator_deploy.return_value = default_predictor

        model_id, model_version = "js-trainable-model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_get_model_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        mock_session_estimator.return_value = sagemaker_session
        mock_session_model.return_value = sagemaker_session

        estimator = JumpStartEstimator(
            model_id=model_id,
        )
        mock_jumpstart_estimator_factory_logger.info.assert_called_once_with(
            "No instance type selected for training job. Defaulting to %s.", "ml.p3.2xlarge"
        )

        mock_estimator_init.assert_called_once_with(
            instance_type="ml.p3.2xlarge",
            instance_count=1,
            image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/autogluon-training:0.4.3-gpu-py38",
            model_uri="s3://jumpstart-cache-prod-us-west-2/autogluon-training/train-autogluon-"
            "classification-ensemble.tar.gz",
            source_dir="s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/autogluon/"
            "transfer_learning/classification/v1.0.2/sourcedir.tar.gz",
            entry_point="transfer_learning.py",
            hyperparameters={
                "eval_metric": "auto",
                "presets": "medium_quality",
                "auto_stack": "False",
                "num_bag_folds": "0",
                "num_bag_sets": "1",
                "num_stack_levels": "0",
                "refit_full": "False",
                "set_best_to_refit_full": "False",
                "save_space": "False",
                "verbosity": "2",
            },
            role=execution_role,
            encrypt_inter_container_traffic=True,
            sagemaker_session=sagemaker_session,
            enable_network_isolation=False,
            tags=[
                {"Key": JumpStartTag.MODEL_ID, "Value": "js-trainable-model"},
                {"Key": JumpStartTag.MODEL_VERSION, "Value": "1.1.1"},
            ],
        )

        channels = {
            "training": f"s3://{get_jumpstart_content_bucket(region)}/"
            f"{get_training_dataset_for_model_and_version(model_id, model_version)}",
        }

        mock_jumpstart_estimator_factory_logger.info.reset_mock()
        estimator.fit(channels)
        mock_jumpstart_estimator_factory_logger.info.assert_not_called()

        mock_estimator_fit.assert_called_once_with(
            inputs=channels, wait=True, job_name="blahblahblah-9876"
        )

        mock_jumpstart_model_factory_logger.info.reset_mock()
        mock_jumpstart_estimator_factory_logger.info.reset_mock()
        estimator.deploy()
        mock_jumpstart_model_factory_logger.info.assert_called_once_with(
            "No instance type selected for inference hosting endpoint. Defaulting to %s.",
            "ml.p2.xlarge",
        )
        mock_jumpstart_estimator_factory_logger.info.assert_not_called()

        mock_estimator_deploy.assert_called_once_with(
            instance_type="ml.p2.xlarge",
            initial_instance_count=1,
            image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/autogluon-inference:0.4.3-gpu-py38",
            source_dir="s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/autogluon/"
            "inference/classification/v1.0.0/sourcedir.tar.gz",
            entry_point="inference.py",
            env={
                "SAGEMAKER_PROGRAM": "inference.py",
                "ENDPOINT_SERVER_TIMEOUT": "3600",
                "MODEL_CACHE_ROOT": "/opt/ml/model",
                "SAGEMAKER_ENV": "1",
                "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            },
            predictor_cls=Predictor,
            role=execution_role,
            wait=True,
            use_compiled_model=False,
            enable_network_isolation=False,
            model_name="blahblahblah-9876",
            endpoint_name="blahblahblah-9876",
            tags=[
                {"Key": JumpStartTag.MODEL_ID, "Value": "js-trainable-model"},
                {"Key": JumpStartTag.MODEL_VERSION, "Value": "1.1.1"},
            ],
        )

    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.fit")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_prepacked(
        self,
        mock_estimator_deploy: mock.Mock,
        mock_estimator_fit: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_session_estimator: mock.Mock,
        mock_session_model: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):
        mock_estimator_deploy.return_value = default_predictor

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "js-trainable-model-prepacked", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session_estimator.return_value = sagemaker_session
        mock_session_model.return_value = sagemaker_session

        estimator = JumpStartEstimator(
            model_id=model_id,
        )

        mock_estimator_init.assert_called_once_with(
            instance_type="ml.p3.16xlarge",
            instance_count=1,
            image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:1.10.2"
            "-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
            model_uri="s3://jumpstart-cache-prod-us-west-2/huggingface-training/train-huggingface"
            "-text2text-flan-t5-base.tar.gz",
            source_dir="s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/huggingface/"
            "transfer_learning/text2text/prepack/v1.0.1/sourcedir.tar.gz",
            entry_point="transfer_learning.py",
            hyperparameters={
                "epochs": "1",
                "seed": "42",
                "batch_size": "64",
                "learning_rate": "0.0001",
                "validation_split_ratio": "0.05",
                "train_data_split_seed": "0",
            },
            metric_definitions=[
                {"Name": "huggingface-text2text:eval-loss", "Regex": "'eval_loss': ([0-9\\.]+)"}
            ],
            role=execution_role,
            encrypt_inter_container_traffic=False,
            sagemaker_session=sagemaker_session,
            enable_network_isolation=False,
            tags=[
                {"Key": JumpStartTag.MODEL_ID, "Value": "js-trainable-model-prepacked"},
                {"Key": JumpStartTag.MODEL_VERSION, "Value": "1.2.0"},
            ],
        )

        channels = {
            "training": f"s3://{get_jumpstart_content_bucket(region)}/"
            f"some-training-dataset-doesn't-matter",
        }

        estimator.fit(channels)

        mock_estimator_fit.assert_called_once_with(inputs=channels, wait=True)

        estimator.deploy()

        mock_estimator_deploy.assert_called_once_with(
            instance_type="ml.g5.xlarge",
            initial_instance_count=1,
            image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:"
            "1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
            env={
                "SAGEMAKER_PROGRAM": "inference.py",
                "ENDPOINT_SERVER_TIMEOUT": "3600",
                "MODEL_CACHE_ROOT": "/opt/ml/model",
                "SAGEMAKER_ENV": "1",
                "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            },
            predictor_cls=Predictor,
            role=execution_role,
            wait=True,
            use_compiled_model=False,
            enable_network_isolation=False,
            tags=[
                {"Key": JumpStartTag.MODEL_ID, "Value": "js-trainable-model-prepacked"},
                {"Key": JumpStartTag.MODEL_VERSION, "Value": "1.2.0"},
            ],
        )

    @mock.patch("sagemaker.utils.sagemaker_timestamp")
    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.fit")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_gated_model_s3_uri(
        self,
        mock_estimator_deploy: mock.Mock,
        mock_estimator_fit: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_session_estimator: mock.Mock,
        mock_session_model: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
        mock_timestamp: mock.Mock,
    ):
        mock_estimator_deploy.return_value = default_predictor

        mock_timestamp.return_value = "8675309"

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "js-gated-artifact-trainable-model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session_estimator.return_value = sagemaker_session
        mock_session_model.return_value = sagemaker_session

        with pytest.raises(ValueError) as e:
            JumpStartEstimator(
                model_id=model_id,
                environment={
                    "accept_eula": "false",
                    "what am i": "doing",
                    "SageMakerGatedModelS3Uri": "none of your business",
                },
            )
        assert str(e.value) == (
            "Need to define ‘accept_eula'='true' within Environment. "
            "Model 'meta-textgeneration-llama-2-7b-f' requires accepting end-user "
            "license agreement (EULA). See "
            "https://jumpstart-cache-prod-us-west-2.s3.us-west-2.amazonaws.com/fmhMetadata/eula/llamaEula.txt"
            " for terms of use."
        )

        mock_estimator_init.reset_mock()

        estimator = JumpStartEstimator(model_id=model_id, environment={"accept_eula": "true"})

        mock_estimator_init.assert_called_once_with(
            instance_type="ml.p3.2xlarge",
            instance_count=1,
            image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117",
            source_dir="s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/"
            "meta/transfer_learning/textgeneration/v1.0.0/sourcedir.tar.gz",
            entry_point="transfer_learning.py",
            role=execution_role,
            sagemaker_session=sagemaker_session,
            max_run=360000,
            enable_network_isolation=True,
            encrypt_inter_container_traffic=True,
            environment={
                "accept_eula": "true",
                "SageMakerGatedModelS3Uri": "s3://jumpstart-cache-alpha-us-west-2/dummy.tar.gz",
            },
            tags=[
                {
                    "Key": "sagemaker-sdk:jumpstart-model-id",
                    "Value": "js-gated-artifact-trainable-model",
                },
                {"Key": "sagemaker-sdk:jumpstart-model-version", "Value": "2.0.0"},
            ],
        )

        channels = {
            "training": f"s3://{get_jumpstart_content_bucket(region)}/"
            f"some-training-dataset-doesn't-matter",
        }

        estimator.fit(channels)

        mock_estimator_fit.assert_called_once_with(
            inputs=channels, wait=True, job_name="meta-textgeneration-llama-2-7b-f-8675309"
        )

        estimator.deploy()

        mock_estimator_deploy.assert_called_once_with(
            instance_type="ml.g5.2xlarge",
            initial_instance_count=1,
            predictor_cls=Predictor,
            endpoint_name="meta-textgeneration-llama-2-7b-f-8675309",
            image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117",
            wait=True,
            model_data_download_timeout=3600,
            container_startup_health_check_timeout=3600,
            role=execution_role,
            enable_network_isolation=True,
            model_name="meta-textgeneration-llama-2-7b-f-8675309",
            use_compiled_model=False,
            tags=[
                {
                    "Key": "sagemaker-sdk:jumpstart-model-id",
                    "Value": "js-gated-artifact-trainable-model",
                },
                {"Key": "sagemaker-sdk:jumpstart-model-version", "Value": "2.0.0"},
            ],
        )

    @mock.patch(
        "sagemaker.jumpstart.artifacts.environment_variables.get_jumpstart_gated_content_bucket"
    )
    @mock.patch("sagemaker.utils.sagemaker_timestamp")
    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.fit")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_gated_model_non_model_package_s3_uri(
        self,
        mock_estimator_deploy: mock.Mock,
        mock_estimator_fit: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_session_estimator: mock.Mock,
        mock_session_model: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
        mock_timestamp: mock.Mock,
        mock_get_jumpstart_gated_content_bucket: mock.Mock,
    ):
        mock_estimator_deploy.return_value = default_predictor

        mock_get_jumpstart_gated_content_bucket.return_value = "top-secret-private-models-bucket"
        mock_timestamp.return_value = "8675309"

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "js-gated-artifact-non-model-package-trainable-model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session_estimator.return_value = sagemaker_session
        mock_session_model.return_value = sagemaker_session

        estimator = JumpStartEstimator(model_id=model_id, environment={"accept_eula": True})

        mock_estimator_init.assert_called_once_with(
            instance_type="ml.g5.12xlarge",
            instance_count=1,
            image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pyt"
            "orch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04",
            source_dir="s3://jumpstart-cache-prod-us-west-2/source-d"
            "irectory-tarballs/meta/transfer_learning/textgeneration/prepack/v1.0.1/sourcedir.tar.gz",
            entry_point="transfer_learning.py",
            hyperparameters={
                "int8_quantization": "False",
                "enable_fsdp": "True",
                "epoch": "5",
                "learning_rate": "0.0001",
                "lora_r": "8",
                "lora_alpha": "32",
                "lora_dropout": "0.05",
                "instruction_tuned": "False",
                "chat_dataset": "False",
                "add_input_output_demarcation_key": "True",
                "per_device_train_batch_size": "4",
                "per_device_eval_batch_size": "1",
                "max_train_samples": "-1",
                "max_val_samples": "-1",
                "seed": "10",
                "max_input_length": "-1",
                "validation_split_ratio": "0.2",
                "train_data_split_seed": "0",
                "preprocessing_num_workers": "None",
            },
            metric_definitions=[
                {
                    "Name": "huggingface-textgeneration:eval-loss",
                    "Regex": "eval_epoch_loss=tensor\\(([0-9\\.]+)",
                },
                {
                    "Name": "huggingface-textgeneration:eval-ppl",
                    "Regex": "eval_ppl=tensor\\(([0-9\\.]+)",
                },
                {
                    "Name": "huggingface-textgeneration:train-loss",
                    "Regex": "train_epoch_loss=([0-9\\.]+)",
                },
            ],
            role="fake role! do not use!",
            max_run=360000,
            sagemaker_session=sagemaker_session,
            tags=[
                {
                    "Key": "sagemaker-sdk:jumpstart-model-id",
                    "Value": "js-gated-artifact-non-model-package-trainable-model",
                },
                {"Key": "sagemaker-sdk:jumpstart-model-version", "Value": "3.0.0"},
            ],
            encrypt_inter_container_traffic=True,
            enable_network_isolation=True,
            environment={
                "SELF_DESTRUCT": "true",
                "accept_eula": True,
                "SageMakerGatedModelS3Uri": "s3://top-secret-private-"
                "models-bucket/meta-training/train-meta-textgeneration-llama-2-7b.tar.gz",
            },
        )

        channels = {
            "training": f"s3://{get_jumpstart_content_bucket(region)}/"
            f"some-training-dataset-doesn't-matter",
        }

        estimator.fit(channels)

        mock_estimator_fit.assert_called_once_with(
            inputs=channels, wait=True, job_name="meta-textgeneration-llama-2-7b-8675309"
        )

        estimator.deploy()

        mock_estimator_deploy.assert_called_once_with(
            instance_type="ml.g5.2xlarge",
            initial_instance_count=1,
            image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytor"
            "ch-tgi-inference:2.0.1-tgi1.1.0-gpu-py39-cu118-ubuntu20.04",
            env={
                "SAGEMAKER_PROGRAM": "inference.py",
                "ENDPOINT_SERVER_TIMEOUT": "3600",
                "MODEL_CACHE_ROOT": "/opt/ml/model",
                "SAGEMAKER_ENV": "1",
                "HF_MODEL_ID": "/opt/ml/model",
                "MAX_INPUT_LENGTH": "4095",
                "MAX_TOTAL_TOKENS": "4096",
                "SM_NUM_GPUS": "1",
                "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            },
            predictor_cls=Predictor,
            endpoint_name="meta-textgeneration-llama-2-7b-8675309",
            tags=[
                {
                    "Key": "sagemaker-sdk:jumpstart-model-id",
                    "Value": "js-gated-artifact-non-model-package-trainable-model",
                },
                {"Key": "sagemaker-sdk:jumpstart-model-version", "Value": "3.0.0"},
            ],
            wait=True,
            model_data_download_timeout=1200,
            container_startup_health_check_timeout=1200,
            role="fake role! do not use!",
            enable_network_isolation=True,
            model_name="meta-textgeneration-llama-2-7b-8675309",
            use_compiled_model=False,
        )

    @mock.patch("sagemaker.utils.sagemaker_timestamp")
    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.fit")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_jumpstart_model_package_artifact_s3_uri_unsupported_region(
        self,
        mock_estimator_deploy: mock.Mock,
        mock_estimator_fit: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_session_estimator: mock.Mock,
        mock_session_model: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
        mock_timestamp: mock.Mock,
    ):
        mock_estimator_deploy.return_value = default_predictor

        mock_timestamp.return_value = "8675309"

        model_id, _ = "js-gated-artifact-trainable-model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session_estimator.return_value = sagemaker_session
        mock_session_model.return_value = sagemaker_session

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        with pytest.raises(ValueError) as e:
            JumpStartEstimator(model_id=model_id, region="eu-north-1")

        assert (
            str(e.value) == "Model package artifact s3 uri for 'js-gated-artifact-trainable-model' "
            "not supported in eu-north-1. Please try one of the following regions: "
            "us-west-2, us-east-1, eu-west-1, ap-southeast-1."
        )

    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.fit")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_deprecated(
        self,
        mock_estimator_deploy: mock.Mock,
        mock_estimator_fit: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "deprecated_model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        with pytest.raises(ValueError):
            JumpStartEstimator(
                model_id=model_id,
            )

        channels = {
            "training": f"s3://{get_jumpstart_content_bucket(region)}/"
            f"some-training-dataset-doesn't-matter",
        }

        JumpStartEstimator(model_id=model_id, tolerate_deprecated_model=True).fit(channels).deploy()

    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.fit")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_vulnerable(
        self,
        mock_estimator_deploy: mock.Mock,
        mock_estimator_fit: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):
        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS
        model_id, _ = "vulnerable_model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        with pytest.raises(ValueError):
            JumpStartEstimator(
                model_id=model_id,
            )

        channels = {
            "training": f"s3://{get_jumpstart_content_bucket(region)}/"
            f"some-training-dataset-doesn't-matter",
        }

        JumpStartEstimator(model_id=model_id, tolerate_vulnerable_model=True).fit(channels).deploy()

    def test_estimator_use_kwargs(self):

        all_init_kwargs_used = {
            "image_uri": "blah1",
            "role": "str = None",
            "instance_count": 1,
            "instance_type": "ml.p2.xlarge",
            "keep_alive_period_in_seconds": 1,
            "volume_size": 30,
            "volume_kms_key": "Optional[Union[str, PipelineVariable]] = None",
            "max_run": 24 * 60 * 60,
            "input_mode": "File",
            "output_path": "Optional[Union[str, PipelineVariable]] = None",
            "output_kms_key": "Optional[Union[str, PipelineVariable]] = None",
            "base_job_name": "Optional[str] = None",
            "sagemaker_session": DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
            "hyperparameters": {"hyp1": "val1"},
            "tags": [],
            "subnets": ["1", "2"],
            "security_group_ids": ["1", "2"],
            "model_uri": "Optional[str] = None",
            "model_channel_name": "Union[str, PipelineVariable] = model",
            "metric_definitions": [{"1": "hum"}],
            "encrypt_inter_container_traffic": True,
            "use_spot_instances": False,
            "max_wait": 4,
            "checkpoint_s3_uri": "43",
            "checkpoint_local_path": "Optional[Union[str, PipelineVariable]] = None",
            "enable_network_isolation": True,
            "rules": ["RuleBase()"],
            "debugger_hook_config": True,
            "tensorboard_output_config": "TensorBoardOutputConfig()",
            "enable_sagemaker_metrics": True,
            "profiler_config": ProfilerConfig(),
            "disable_profiler": False,
            "environment": {"1": "2"},
            "max_retry_attempts": 4,
            "source_dir": "blah",
            "git_config": {"1", "3"},
            "container_log_level": 4,
            "code_location": "Optional[str] = None",
            "entry_point": "Optional[Union[str, PipelineVariable]] = None",
            "dependencies": ["Optional[List[str]] = None"],
            "instance_groups": [InstanceGroup()],
            "training_repository_access_mode": "Optional[Union[str, PipelineVariable]] = None",
            "training_repository_credentials_provider_arn": "Optional[Union[str, PipelineVariable]] = None",
        }
        all_fit_kwargs_used = {
            "inputs": {"hello": "world"},
            "wait": True,
            "logs": "All",
            "job_name": "none_of_your_business",
            "experiment_config": {"1": "2"},
        }
        all_deploy_kwargs_used = {
            "initial_instance_count": 88,
            "instance_type": "ml.p2.xlarge",
            "serializer": "BaseSerializer()",
            "deserializer": "BaseDeserializer()",
            "accelerator_type": "None",
            "endpoint_name": "None",
            "tags": [],
            "kms_key": "None",
            "wait": True,
            "data_capture_config": "None",
            "async_inference_config": "None",
            "serverless_inference_config": "None",
            "volume_size": 3,
            "model_data_download_timeout": 4,
            "container_startup_health_check_timeout": 2,
            "inference_recommendation_id": "None",
            "explainer_config": "None",
        }

        self.evaluate_estimator_workflow_with_kwargs(
            init_kwargs=all_init_kwargs_used,
            fit_kwargs=all_fit_kwargs_used,
            deploy_kwargs=all_deploy_kwargs_used,
        )

    @mock.patch("sagemaker.jumpstart.factory.estimator.hyperparameters_utils.retrieve_default")
    @mock.patch("sagemaker.jumpstart.factory.estimator.metric_definitions_utils.retrieve_default")
    @mock.patch("sagemaker.jumpstart.factory.estimator.environment_variables.retrieve_default")
    @mock.patch("sagemaker.utils.sagemaker_timestamp")
    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.fit")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def evaluate_estimator_workflow_with_kwargs(
        self,
        mock_estimator_deploy: mock.Mock,
        mock_estimator_fit: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_session_estimator: mock.Mock,
        mock_session_model: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
        mock_timestamp: mock.Mock,
        mock_retrieve_default_environment_variables: mock.Mock,
        mock_retrieve_metric_definitions: mock.Mock,
        mock_retrieve_hyperparameters: mock.Mock,
        init_kwargs: Optional[dict] = None,
        fit_kwargs: Optional[dict] = None,
        deploy_kwargs: Optional[dict] = None,
    ):

        mock_retrieve_default_environment_variables.side_effect = (
            _retrieve_default_environment_variables
        )

        mock_retrieve_metric_definitions.side_effect = _retrieve_default_training_metric_definitions

        mock_retrieve_hyperparameters.side_effect = _retrieve_default_hyperparameters

        if init_kwargs is None:
            init_kwargs = {}

        if fit_kwargs is None:
            fit_kwargs = {}

        if deploy_kwargs is None:
            deploy_kwargs = {}

        mock_timestamp.return_value = "1234"

        mock_estimator_deploy.return_value = default_predictor

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, model_version = "js-trainable-model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session_estimator.return_value = sagemaker_session
        mock_session_model.return_value = sagemaker_session

        estimator = JumpStartEstimator(
            model_id=model_id,
            tolerate_deprecated_model=True,
            tolerate_vulnerable_model=True,
            region=region,
            **init_kwargs,
        )

        expected_init_kwargs = overwrite_dictionary(
            {
                "instance_type": "ml.p3.2xlarge",
                "instance_count": 1,
                "image_uri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/autogluon-training:0.4.3-gpu-py38",
                "model_uri": "s3://jumpstart-cache-prod-us-west-2/autogluon-training/train-autogluon-"
                "classification-ensemble.tar.gz",
                "source_dir": "s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/autogluon/"
                "transfer_learning/classification/v1.0.2/sourcedir.tar.gz",
                "entry_point": "transfer_learning.py",
                "hyperparameters": {
                    "eval_metric": "auto",
                    "presets": "medium_quality",
                    "auto_stack": "False",
                    "num_bag_folds": "0",
                    "num_bag_sets": "1",
                    "num_stack_levels": "0",
                    "refit_full": "False",
                    "set_best_to_refit_full": "False",
                    "save_space": "False",
                    "verbosity": "2",
                },
                "role": execution_role,
                "encrypt_inter_container_traffic": True,
                "sagemaker_session": sagemaker_session,
                "enable_network_isolation": False,
                "tags": [
                    {"Key": "sagemaker-sdk:jumpstart-model-id", "Value": "js-trainable-model"},
                    {"Key": "sagemaker-sdk:jumpstart-model-version", "Value": "1.1.1"},
                ],
            },
            init_kwargs,
        )

        mock_estimator_init.assert_called_once_with(**expected_init_kwargs)

        channels = {
            "training": f"s3://{get_jumpstart_content_bucket(region)}/"
            f"{get_training_dataset_for_model_and_version(model_id, model_version)}",
        }

        expected_fit_kwargs = overwrite_dictionary(
            {"inputs": channels, "wait": True, "job_name": "none_of_your_business"}, fit_kwargs
        )

        estimator.fit(**expected_fit_kwargs)

        mock_estimator_fit.assert_called_once_with(**expected_fit_kwargs)

        mock_retrieve_default_environment_variables.assert_called_once()
        mock_retrieve_metric_definitions.assert_called_once()
        mock_retrieve_hyperparameters.assert_called_once()
        estimator.deploy(**deploy_kwargs)

        expected_deploy_kwargs = overwrite_dictionary(
            {
                "instance_type": "ml.p2.xlarge",
                "initial_instance_count": 1,
                "image_uri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/autogluon-inference:0.4.3-gpu-py38",
                "source_dir": "s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/autogluon/"
                "inference/classification/v1.0.0/sourcedir.tar.gz",
                "entry_point": "inference.py",
                "env": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "ENDPOINT_SERVER_TIMEOUT": "3600",
                    "MODEL_CACHE_ROOT": "/opt/ml/model",
                    "SAGEMAKER_ENV": "1",
                    "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
                },
                "predictor_cls": Predictor,
                "role": init_kwargs["role"],
                "enable_network_isolation": False,
                "use_compiled_model": False,
                "model_name": "blahblahblah-1234",
                "endpoint_name": "blahblahblah-1234",
                "tags": [
                    {"Key": "sagemaker-sdk:jumpstart-model-id", "Value": "js-trainable-model"},
                    {"Key": "sagemaker-sdk:jumpstart-model-version", "Value": "1.1.1"},
                ],
            },
            deploy_kwargs,
        )

        mock_estimator_deploy.assert_called_once_with(**expected_deploy_kwargs)

    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_jumpstart_estimator_tags_disabled(
        self,
        mock_get_model_specs: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "js-trainable-model-prepacked", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        settings = SessionSettings(include_jumpstart_tags=False)

        mock_session = mock.MagicMock(
            sagemaker_config={}, boto_region_name="us-west-2", settings=settings
        )

        estimator = JumpStartEstimator(
            model_id=model_id,
            sagemaker_session=mock_session,
            tags=[{"Key": "blah", "Value": "blahagain"}],
        )

        channels = {
            "training": f"s3://{get_jumpstart_content_bucket(region)}/"
            f"some-training-dataset-doesn't-matter",
        }

        estimator.fit(channels)

        self.assertEqual(
            mock_session.train.call_args[1]["tags"],
            [{"Key": "blah", "Value": "blahagain"}],
        )

    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_jumpstart_estimator_tags(
        self,
        mock_get_model_specs: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "js-trainable-model-prepacked", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session = mock.MagicMock(sagemaker_config={}, boto_region_name="us-west-2")

        estimator = JumpStartEstimator(
            model_id=model_id,
            sagemaker_session=mock_session,
            tags=[{"Key": "blah", "Value": "blahagain"}],
        )

        channels = {
            "training": f"s3://{get_jumpstart_content_bucket(region)}/"
            f"some-training-dataset-doesn't-matter",
        }

        estimator.fit(channels)

        js_tags = [
            {"Key": "sagemaker-sdk:jumpstart-model-id", "Value": "js-trainable-model-prepacked"},
            {"Key": "sagemaker-sdk:jumpstart-model-version", "Value": "1.2.0"},
        ]

        self.assertEqual(
            mock_session.train.call_args[1]["tags"],
            [{"Key": "blah", "Value": "blahagain"}] + js_tags,
        )

    @mock.patch("sagemaker.jumpstart.estimator.JumpStartEstimator._attach")
    @mock.patch("sagemaker.jumpstart.estimator.get_model_id_version_from_training_job")
    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_jumpstart_estimator_attach_no_model_id_happy_case(
        self,
        mock_get_model_specs: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
        get_model_id_version_from_training_job: mock.Mock,
        mock_attach: mock.Mock,
    ):

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        get_model_id_version_from_training_job.return_value = (
            "js-trainable-model-prepacked",
            "1.0.0",
        )

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session = mock.MagicMock(sagemaker_config={}, boto_region_name="us-west-2")

        JumpStartEstimator.attach(
            training_job_name="some-training-job-name", sagemaker_session=mock_session
        )

        get_model_id_version_from_training_job.assert_called_once_with(
            training_job_name="some-training-job-name",
            sagemaker_session=mock_session,
        )

        mock_attach.assert_called_once_with(
            training_job_name="some-training-job-name",
            sagemaker_session=mock_session,
            model_channel_name="model",
            additional_kwargs={
                "model_id": "js-trainable-model-prepacked",
                "model_version": "1.0.0",
            },
        )

    @mock.patch("sagemaker.jumpstart.estimator.JumpStartEstimator._attach")
    @mock.patch("sagemaker.jumpstart.estimator.get_model_id_version_from_training_job")
    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_jumpstart_estimator_attach_no_model_id_sad_case(
        self,
        mock_get_model_specs: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
        get_model_id_version_from_training_job: mock.Mock,
        mock_attach: mock.Mock,
    ):

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        get_model_id_version_from_training_job.side_effect = ValueError()

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session = mock.MagicMock(sagemaker_config={}, boto_region_name="us-west-2")

        with pytest.raises(ValueError):
            JumpStartEstimator.attach(
                training_job_name="some-training-job-name", sagemaker_session=mock_session
            )

        get_model_id_version_from_training_job.assert_called_once_with(
            training_job_name="some-training-job-name",
            sagemaker_session=mock_session,
        )

        mock_attach.assert_not_called()

    def test_jumpstart_estimator_kwargs_match_parent_class(self):

        """If you add arguments to <Estimator constructor>, this test will fail.
        Please add the new argument to the skip set below,
        and reach out to JumpStart team."""

        init_args_to_skip: Set[str] = set(["kwargs"])
        fit_args_to_skip: Set[str] = set()
        deploy_args_to_skip: Set[str] = set(["kwargs"])

        parent_class_init = Estimator.__init__
        parent_class_init_args = set(signature(parent_class_init).parameters.keys())

        js_class_init = JumpStartEstimator.__init__
        js_class_init_args = set(signature(js_class_init).parameters.keys())
        assert js_class_init_args - parent_class_init_args == {
            "model_id",
            "model_version",
            "region",
            "tolerate_vulnerable_model",
            "tolerate_deprecated_model",
        }
        assert parent_class_init_args - js_class_init_args == init_args_to_skip

        parent_class_fit = Estimator.fit
        parent_class_fit_args = set(signature(parent_class_fit).parameters.keys())

        js_class_fit = JumpStartEstimator.fit
        js_class_fit_args = set(signature(js_class_fit).parameters.keys())

        assert js_class_fit_args - parent_class_fit_args == set()
        assert parent_class_fit_args - js_class_fit_args == fit_args_to_skip

        model_class_init = Model.__init__
        model_class_init_args = set(signature(model_class_init).parameters.keys())

        parent_class_deploy = Estimator.deploy
        parent_class_deploy_args = set(signature(parent_class_deploy).parameters.keys())

        js_class_deploy = JumpStartEstimator.deploy
        js_class_deploy_args = set(signature(js_class_deploy).parameters.keys())

        assert js_class_deploy_args - parent_class_deploy_args == model_class_init_args - {
            "model_data",
            "self",
            "name",
            "resources",
        }
        assert parent_class_deploy_args - js_class_deploy_args == deploy_args_to_skip

    @mock.patch("sagemaker.jumpstart.estimator.get_init_kwargs")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    def test_validate_model_id_and_get_type(
        self,
        mock_validate_model_id_and_get_type: mock.Mock,
        mock_init: mock.Mock,
        mock_get_init_kwargs: mock.Mock,
    ):
        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS
        JumpStartEstimator(model_id="valid_model_id")

        mock_validate_model_id_and_get_type.return_value = False
        with pytest.raises(ValueError):
            JumpStartEstimator(model_id="invalid_model_id")

    @mock.patch("sagemaker.jumpstart.estimator.get_default_predictor")
    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.fit")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_no_predictor_returns_default_predictor(
        self,
        mock_estimator_deploy: mock.Mock,
        mock_estimator_fit: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_session_estimator: mock.Mock,
        mock_session_model: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
        mock_get_default_predictor: mock.Mock,
    ):
        mock_estimator_deploy.return_value = default_predictor

        mock_get_default_predictor.return_value = default_predictor_with_presets

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "js-trainable-model-prepacked", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session_estimator.return_value = sagemaker_session
        mock_session_model.return_value = sagemaker_session

        estimator = JumpStartEstimator(
            model_id=model_id,
        )

        channels = {
            "training": f"s3://{get_jumpstart_content_bucket(region)}/"
            f"some-training-dataset-doesn't-matter",
        }

        estimator.fit(channels)

        predictor = estimator.deploy()

        mock_get_default_predictor.assert_called_once_with(
            predictor=default_predictor,
            model_id=model_id,
            model_version="*",
            region=region,
            tolerate_deprecated_model=False,
            tolerate_vulnerable_model=False,
            sagemaker_session=estimator.sagemaker_session,
        )
        self.assertEqual(type(predictor), Predictor)
        self.assertEqual(predictor, default_predictor_with_presets)

    @mock.patch("sagemaker.jumpstart.estimator.get_default_predictor")
    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.fit")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_no_predictor_yes_async_inference_config(
        self,
        mock_estimator_deploy: mock.Mock,
        mock_estimator_fit: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_session_estimator: mock.Mock,
        mock_session_model: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
        mock_get_default_predictor: mock.Mock,
    ):
        mock_estimator_deploy.return_value = default_predictor

        mock_get_default_predictor.return_value = default_predictor_with_presets

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "js-trainable-model-prepacked", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session_estimator.return_value = sagemaker_session
        mock_session_model.return_value = sagemaker_session

        estimator = JumpStartEstimator(
            model_id=model_id,
        )

        channels = {
            "training": f"s3://{get_jumpstart_content_bucket(region)}/"
            f"some-training-dataset-doesn't-matter",
        }

        estimator.fit(channels)

        predictor = estimator.deploy(async_inference_config=AsyncInferenceConfig())

        mock_get_default_predictor.assert_not_called()
        self.assertEqual(type(predictor), Predictor)

    @mock.patch("sagemaker.jumpstart.estimator.get_default_predictor")
    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.fit")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_yes_predictor_returns_unmodified_predictor(
        self,
        mock_estimator_deploy: mock.Mock,
        mock_estimator_fit: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_session_estimator: mock.Mock,
        mock_session_model: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
        mock_get_default_predictor: mock.Mock,
    ):
        mock_estimator_deploy.return_value = default_predictor

        mock_get_default_predictor.return_value = default_predictor_with_presets

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "js-trainable-model-prepacked", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session_estimator.return_value = sagemaker_session
        mock_session_model.return_value = sagemaker_session

        estimator = JumpStartEstimator(
            model_id=model_id,
        )

        channels = {
            "training": f"s3://{get_jumpstart_content_bucket(region)}/"
            f"some-training-dataset-doesn't-matter",
        }

        estimator.fit(channels)

        predictor = estimator.deploy(predictor_cls=Predictor)

        mock_get_default_predictor.assert_not_called()
        self.assertEqual(type(predictor), Predictor)
        self.assertEqual(predictor, default_predictor)

    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.factory.estimator._model_supports_incremental_training")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_LOGGER.warning")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.fit")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_incremental_training_with_unsupported_model_logs_warning(
        self,
        mock_estimator_deploy: mock.Mock,
        mock_estimator_fit: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_session_estimator: mock.Mock,
        mock_session_model: mock.Mock,
        mock_logger_warning: mock.Mock,
        mock_supports_incremental_training: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):
        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        mock_estimator_deploy.return_value = default_predictor

        model_id = "js-trainable-model"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session_estimator.return_value = sagemaker_session
        mock_session_model.return_value = sagemaker_session

        mock_supports_incremental_training.return_value = False

        JumpStartEstimator(
            model_id=model_id,
            model_uri="some-weird-model-uri",
        )

        mock_logger_warning.assert_called_once_with(
            "'%s' does not support incremental training but is being trained with non-default model artifact.",
            model_id,
        )
        mock_supports_incremental_training.assert_called_once_with(
            model_id=model_id,
            model_version="*",
            region=region,
            tolerate_deprecated_model=False,
            tolerate_vulnerable_model=False,
            sagemaker_session=sagemaker_session,
        )

    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.factory.estimator._model_supports_incremental_training")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_LOGGER.warning")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.fit")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_incremental_training_with_supported_model_doesnt_log_warning(
        self,
        mock_estimator_deploy: mock.Mock,
        mock_estimator_fit: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_session_estimator: mock.Mock,
        mock_session_model: mock.Mock,
        mock_logger_warning: mock.Mock,
        mock_supports_incremental_training: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):
        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        mock_estimator_deploy.return_value = default_predictor

        model_id = "js-trainable-model"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session_estimator.return_value = sagemaker_session
        mock_session_model.return_value = sagemaker_session

        mock_supports_incremental_training.return_value = True

        JumpStartEstimator(
            model_id=model_id,
            model_uri="some-weird-model-uri",
        )

        mock_logger_warning.assert_not_called()
        mock_supports_incremental_training.assert_called_once_with(
            model_id=model_id,
            model_version="*",
            region=region,
            tolerate_deprecated_model=False,
            tolerate_vulnerable_model=False,
            sagemaker_session=sagemaker_session,
        )

    @mock.patch("sagemaker.utils.sagemaker_timestamp")
    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.fit")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_estimator_sets_different_inference_instance_depending_on_training_instance(
        self,
        mock_estimator_deploy: mock.Mock,
        mock_estimator_fit: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_session_estimator: mock.Mock,
        mock_session_model: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
        mock_sagemaker_timestamp: mock.Mock,
    ):
        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        mock_sagemaker_timestamp.return_value = "3456"

        mock_estimator_deploy.return_value = default_predictor

        model_id = "inference-instance-types-variant-model"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session_estimator.return_value = sagemaker_session
        mock_session_model.return_value = sagemaker_session

        estimator = JumpStartEstimator(
            model_id=model_id, image_uri="blah", instance_type="ml.trn1.xlarge"
        )
        estimator.deploy(image_uri="blah")
        assert mock_estimator_deploy.call_args[1]["instance_type"] == "ml.inf1.xlarge"
        mock_estimator_deploy.reset_mock()

        estimator = JumpStartEstimator(
            model_id=model_id, image_uri="blah", instance_type="ml.p2.xlarge"
        )
        estimator.deploy(image_uri="blah")
        assert mock_estimator_deploy.call_args[1]["instance_type"] == "ml.p2.xlarge"
        mock_estimator_deploy.reset_mock()

        estimator = JumpStartEstimator(
            model_id=model_id, image_uri="blah", instance_type="ml.p2.12xlarge"
        )
        estimator.deploy(image_uri="blah")
        assert mock_estimator_deploy.call_args[1]["instance_type"] == "ml.p5.xlarge"
        mock_estimator_deploy.reset_mock()

        estimator = JumpStartEstimator(
            model_id=model_id, image_uri="blah", instance_type="ml.blah.xblah"
        )
        estimator.deploy(image_uri="blah")
        assert mock_estimator_deploy.call_args[1]["instance_type"] == "ml.p4de.24xlarge"

    @mock.patch("sagemaker.utils.sagemaker_timestamp")
    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.fit")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_training_passes_role_to_deploy(
        self,
        mock_estimator_deploy: mock.Mock,
        mock_estimator_fit: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_session_estimator: mock.Mock,
        mock_session_model: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
        mock_sagemaker_timestamp: mock.Mock,
    ):
        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        mock_sagemaker_timestamp.return_value = "3456"

        mock_estimator_deploy.return_value = default_predictor

        model_id, model_version = "js-trainable-model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session_estimator.return_value = sagemaker_session
        mock_session_model.return_value = sagemaker_session

        mock_role = f"mock-role+{time.time()}"

        estimator = JumpStartEstimator(
            model_id=model_id,
            role=mock_role,
        )

        channels = {
            "training": f"s3://{get_jumpstart_content_bucket(region)}/"
            f"{get_training_dataset_for_model_and_version(model_id, model_version)}",
        }

        estimator.fit(channels)

        estimator.deploy()

        mock_estimator_deploy.assert_called_once_with(
            instance_type="ml.p2.xlarge",
            initial_instance_count=1,
            image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/autogluon-inference:0.4.3-gpu-py38",
            source_dir="s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/"
            "autogluon/inference/classification/v1.0.0/sourcedir.tar.gz",
            entry_point="inference.py",
            env={
                "SAGEMAKER_PROGRAM": "inference.py",
                "ENDPOINT_SERVER_TIMEOUT": "3600",
                "MODEL_CACHE_ROOT": "/opt/ml/model",
                "SAGEMAKER_ENV": "1",
                "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            },
            predictor_cls=Predictor,
            wait=True,
            use_compiled_model=False,
            role=mock_role,
            enable_network_isolation=False,
            model_name="blahblahblah-3456",
            endpoint_name="blahblahblah-3456",
            tags=[
                {"Key": JumpStartTag.MODEL_ID, "Value": "js-trainable-model"},
                {"Key": JumpStartTag.MODEL_VERSION, "Value": "1.1.1"},
            ],
        )

    @mock.patch("sagemaker.utils.sagemaker_timestamp")
    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch(
        "sagemaker.jumpstart.factory.model.DEFAULT_JUMPSTART_SAGEMAKER_SESSION", sagemaker_session
    )
    @mock.patch(
        "sagemaker.jumpstart.factory.estimator.DEFAULT_JUMPSTART_SAGEMAKER_SESSION",
        sagemaker_session,
    )
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.fit")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_training_passes_session_to_deploy(
        self,
        mock_estimator_deploy: mock.Mock,
        mock_estimator_fit: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
        mock_sagemaker_timestamp: mock.Mock,
    ):
        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        mock_sagemaker_timestamp.return_value = "3456"

        mock_estimator_deploy.return_value = default_predictor

        model_id, model_version = "js-trainable-model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_role = f"dsfsdfsd{time.time()}"
        region = "us-west-2"
        mock_sagemaker_session = mock.MagicMock(sagemaker_config={}, boto_region_name=region)
        mock_sagemaker_session.get_caller_identity_arn = lambda: mock_role

        estimator = JumpStartEstimator(
            model_id=model_id,
            sagemaker_session=mock_sagemaker_session,
        )

        channels = {
            "training": f"s3://{get_jumpstart_content_bucket(region)}/"
            f"{get_training_dataset_for_model_and_version(model_id, model_version)}",
        }

        estimator.fit(channels)

        estimator.deploy()

        mock_estimator_deploy.assert_called_once_with(
            instance_type="ml.p2.xlarge",
            initial_instance_count=1,
            image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/autogluon-inference:0.4.3-gpu-py38",
            source_dir="s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/"
            "autogluon/inference/classification/v1.0.0/sourcedir.tar.gz",
            entry_point="inference.py",
            env={
                "SAGEMAKER_PROGRAM": "inference.py",
                "ENDPOINT_SERVER_TIMEOUT": "3600",
                "MODEL_CACHE_ROOT": "/opt/ml/model",
                "SAGEMAKER_ENV": "1",
                "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            },
            predictor_cls=Predictor,
            wait=True,
            use_compiled_model=False,
            role=mock_role,
            enable_network_isolation=False,
            model_name="blahblahblah-3456",
            endpoint_name="blahblahblah-3456",
            tags=[
                {"Key": JumpStartTag.MODEL_ID, "Value": "js-trainable-model"},
                {"Key": JumpStartTag.MODEL_VERSION, "Value": "1.1.1"},
            ],
        )

    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.factory.estimator._retrieve_estimator_init_kwargs")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.estimator.JumpStartModelsAccessor.reset_cache")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_model_id_not_found_refeshes_cache_training(
        self,
        mock_reset_cache: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_retrieve_kwargs: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_estimator_deploy: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):
        mock_estimator_deploy.return_value = default_predictor

        mock_validate_model_id_and_get_type.side_effect = [False, False]

        model_id, _ = "js-trainable-model", "*"

        mock_retrieve_kwargs.return_value = {}

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        with pytest.raises(ValueError):
            JumpStartEstimator(
                model_id=model_id,
            )

        mock_reset_cache.assert_called_once_with()
        mock_validate_model_id_and_get_type.assert_has_calls(
            calls=[
                mock.call(
                    model_id="js-trainable-model",
                    model_version=None,
                    region=None,
                    script=JumpStartScriptScope.TRAINING,
                    sagemaker_session=None,
                ),
                mock.call(
                    model_id="js-trainable-model",
                    model_version=None,
                    region=None,
                    script=JumpStartScriptScope.TRAINING,
                    sagemaker_session=None,
                ),
            ]
        )

        mock_validate_model_id_and_get_type.reset_mock()
        mock_reset_cache.reset_mock()

        mock_validate_model_id_and_get_type.side_effect = [False, True]
        JumpStartEstimator(
            model_id=model_id,
        )

        mock_reset_cache.assert_called_once_with()
        mock_validate_model_id_and_get_type.assert_has_calls(
            calls=[
                mock.call(
                    model_id="js-trainable-model",
                    model_version=None,
                    region=None,
                    script=JumpStartScriptScope.TRAINING,
                    sagemaker_session=None,
                ),
                mock.call(
                    model_id="js-trainable-model",
                    model_version=None,
                    region=None,
                    script=JumpStartScriptScope.TRAINING,
                    sagemaker_session=None,
                ),
            ]
        )

    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_model_artifact_variant_estimator(
        self,
        mock_estimator_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "model-artifact-variant-model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        # this instance type has a special model artifact
        JumpStartEstimator(model_id=model_id, instance_type="ml.p2.xlarge")

        mock_estimator_init.assert_called_once_with(
            instance_type="ml.p2.xlarge",
            instance_count=1,
            image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:1.13.1"
            "-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
            model_uri="s3://jumpstart-cache-prod-us-west-2/hello-mars-1",
            source_dir="s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/"
            "pytorch/transfer_learning/ic/v1.0.0/sourcedir.tar.gz",
            entry_point="transfer_learning.py",
            hyperparameters={"epochs": "3", "adam-learning-rate": "0.05", "batch-size": "4"},
            metric_definitions=[
                {"Regex": "val_accuracy: ([0-9\\.]+)", "Name": "pytorch-ic:val-accuracy"}
            ],
            role=execution_role,
            sagemaker_session=sagemaker_session,
            enable_network_isolation=False,
            encrypt_inter_container_traffic=True,
            volume_size=456,
            tags=[
                {"Key": JumpStartTag.MODEL_ID, "Value": "model-artifact-variant-model"},
                {"Key": JumpStartTag.MODEL_VERSION, "Value": "1.0.0"},
            ],
        )

        mock_estimator_init.reset_mock()

        JumpStartEstimator(model_id=model_id, instance_type="ml.p99.xlarge")

        mock_estimator_init.assert_called_once_with(
            instance_type="ml.p99.xlarge",
            instance_count=1,
            image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.5.0-gpu-py3",
            model_uri="s3://jumpstart-cache-prod-us-west-2/pytorch-training/train-pytorch-ic-mobilenet-v2.tar.gz",
            source_dir="s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/pytorch/"
            "transfer_learning/ic/v1.0.0/sourcedir.tar.gz",
            entry_point="transfer_learning.py",
            hyperparameters={"epochs": "3", "adam-learning-rate": "0.05", "batch-size": "4"},
            metric_definitions=[
                {"Regex": "val_accuracy: ([0-9\\.]+)", "Name": "pytorch-ic:val-accuracy"}
            ],
            role=execution_role,
            sagemaker_session=sagemaker_session,
            enable_network_isolation=False,
            encrypt_inter_container_traffic=True,
            volume_size=456,
            tags=[
                {
                    "Key": "sagemaker-sdk:jumpstart-model-id",
                    "Value": "model-artifact-variant-model",
                },
                {"Key": "sagemaker-sdk:jumpstart-model-version", "Value": "1.0.0"},
            ],
        )

    @mock.patch("sagemaker.jumpstart.estimator.get_default_predictor")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.fit")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_jumpstart_estimator_session(
        self,
        mock_get_model_specs: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
        mock_deploy,
        mock_fit,
        mock_init,
        get_default_predictor,
    ):

        mock_validate_model_id_and_get_type.return_value = True

        model_id, _ = "js-trainable-model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        region = "eu-west-1"  # some non-default region

        if region == JUMPSTART_DEFAULT_REGION_NAME:
            region = "us-west-2"

        session = Session(boto_session=boto3.session.Session(region_name=region))

        assert session.boto_region_name != JUMPSTART_DEFAULT_REGION_NAME

        session.get_caller_identity_arn = Mock(return_value="blah")

        estimator = JumpStartEstimator(model_id=model_id, sagemaker_session=session)
        estimator.fit()

        estimator.deploy()

        assert len(mock_get_model_specs.call_args_list) > 1

        regions = {call[1]["region"] for call in mock_get_model_specs.call_args_list}

        assert len(regions) == 1
        assert list(regions)[0] == region

        s3_clients = {call[1]["s3_client"] for call in mock_get_model_specs.call_args_list}
        assert len(s3_clients) == 1
        assert list(s3_clients)[0] == session.s3_client


def test_jumpstart_estimator_requires_model_id():
    with pytest.raises(ValueError):
        JumpStartEstimator()
