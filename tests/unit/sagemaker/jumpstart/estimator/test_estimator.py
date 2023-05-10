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

import pytest

from sagemaker.debugger.profiler_config import ProfilerConfig
from sagemaker.estimator import Estimator
from sagemaker.instance_group import InstanceGroup

from sagemaker.jumpstart.estimator import JumpStartEstimator

from sagemaker.jumpstart.utils import get_jumpstart_content_bucket
from tests.integ.sagemaker.jumpstart.utils import get_training_dataset_for_model_and_version
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.session import Session
from tests.unit.sagemaker.jumpstart.utils import (
    get_special_model_spec,
    overwrite_dictionary,
)


execution_role = "fake role! do not use!"
region = "us-west-2"
sagemaker_session = Session()
sagemaker_session.get_caller_identity_arn = lambda: execution_role
default_predictor = Predictor("eiifccreeeiuchhnehtlbdecgeeelgjccjvvbbcncnhv", sagemaker_session)
default_predictor_with_presets = Predictor(
    "eiifccreeeiuihlrblivhchuefdckrluliilctfjgknk", sagemaker_session
)


class EstimatorTest(unittest.TestCase):
    @mock.patch("sagemaker.utils.sagemaker_timestamp")
    @mock.patch("sagemaker.jumpstart.estimator.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
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
        mock_session_estimator: mock.Mock,
        mock_session_model: mock.Mock,
        mock_is_valid_model_id: mock.Mock,
        mock_sagemaker_timestamp: mock.Mock,
    ):
        mock_is_valid_model_id.return_value = True

        mock_sagemaker_timestamp.return_value = "9876"

        mock_estimator_deploy.return_value = default_predictor

        model_id, model_version = "js-trainable-model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session_estimator.return_value = sagemaker_session
        mock_session_model.return_value = sagemaker_session

        estimator = JumpStartEstimator(
            model_id=model_id,
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
        )

        channels = {
            "training": f"s3://{get_jumpstart_content_bucket(region)}/"
            f"{get_training_dataset_for_model_and_version(model_id, model_version)}",
        }

        estimator.fit(channels)

        mock_estimator_fit.assert_called_once_with(
            inputs=channels, wait=True, job_name="blahblahblah-9876"
        )

        estimator.deploy()

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
        )

    @mock.patch("sagemaker.jumpstart.estimator.is_valid_model_id")
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
        mock_is_valid_model_id: mock.Mock,
    ):
        mock_estimator_deploy.return_value = default_predictor

        mock_is_valid_model_id.return_value = True

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
        )

    @mock.patch("sagemaker.jumpstart.estimator.is_valid_model_id")
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
        mock_is_valid_model_id: mock.Mock,
    ):

        mock_is_valid_model_id.return_value = True

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

    @mock.patch("sagemaker.jumpstart.estimator.is_valid_model_id")
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
        mock_is_valid_model_id: mock.Mock,
    ):
        mock_is_valid_model_id.return_value = True
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
            "sagemaker_session": Session(),
            "hyperparameters": {"hyp1": "val1"},
            "tags": [{"1": "hum"}],
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
            "tags": ["None"],
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

    @mock.patch("sagemaker.utils.sagemaker_timestamp")
    @mock.patch("sagemaker.jumpstart.estimator.is_valid_model_id")
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
        mock_is_valid_model_id: mock.Mock,
        mock_timestamp: mock.Mock,
        init_kwargs: Optional[dict] = None,
        fit_kwargs: Optional[dict] = None,
        deploy_kwargs: Optional[dict] = None,
    ):

        if init_kwargs is None:
            init_kwargs = {}

        if fit_kwargs is None:
            fit_kwargs = {}

        if deploy_kwargs is None:
            deploy_kwargs = {}

        mock_timestamp.return_value = "1234"

        mock_estimator_deploy.return_value = default_predictor

        mock_is_valid_model_id.return_value = True

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
            },
            deploy_kwargs,
        )

        mock_estimator_deploy.assert_called_once_with(**expected_deploy_kwargs)

    def test_jumpstart_estimator_kwargs_match_parent_class(self):

        """If you add arguments to <Estimator constructor>, this test will fail.
        Please add the new argument to the skip set below,
        and cut a ticket sev-3 to JumpStart team: AWS > SageMaker > JumpStart"""

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
        }
        assert parent_class_deploy_args - js_class_deploy_args == deploy_args_to_skip

    @mock.patch("sagemaker.jumpstart.estimator.get_init_kwargs")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.estimator.is_valid_model_id")
    def test_is_valid_model_id(
        self,
        mock_is_valid_model_id: mock.Mock,
        mock_init: mock.Mock,
        mock_get_init_kwargs: mock.Mock,
    ):
        mock_is_valid_model_id.return_value = True
        JumpStartEstimator(model_id="valid_model_id")

        mock_is_valid_model_id.return_value = False
        with pytest.raises(ValueError):
            JumpStartEstimator(model_id="invalid_model_id")

    @mock.patch("sagemaker.jumpstart.estimator.get_default_predictor")
    @mock.patch("sagemaker.jumpstart.estimator.is_valid_model_id")
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
        mock_is_valid_model_id: mock.Mock,
        mock_get_default_predictor: mock.Mock,
    ):
        mock_estimator_deploy.return_value = default_predictor

        mock_get_default_predictor.return_value = default_predictor_with_presets

        mock_is_valid_model_id.return_value = True

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
        )
        self.assertEqual(type(predictor), Predictor)
        self.assertEqual(predictor, default_predictor_with_presets)

    @mock.patch("sagemaker.jumpstart.estimator.get_default_predictor")
    @mock.patch("sagemaker.jumpstart.estimator.is_valid_model_id")
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
        mock_is_valid_model_id: mock.Mock,
        mock_get_default_predictor: mock.Mock,
    ):
        mock_estimator_deploy.return_value = default_predictor

        mock_get_default_predictor.return_value = default_predictor_with_presets

        mock_is_valid_model_id.return_value = True

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

    @mock.patch("sagemaker.jumpstart.estimator.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.factory.estimator._model_supports_incremental_training")
    @mock.patch("sagemaker.jumpstart.factory.estimator.logger.warning")
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
        mock_is_valid_model_id: mock.Mock,
    ):
        mock_is_valid_model_id.return_value = True

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
        )

    @mock.patch("sagemaker.jumpstart.estimator.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.factory.estimator._model_supports_incremental_training")
    @mock.patch("sagemaker.jumpstart.factory.estimator.logger.warning")
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
        mock_is_valid_model_id: mock.Mock,
    ):
        mock_is_valid_model_id.return_value = True

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
        )

    @mock.patch("sagemaker.utils.sagemaker_timestamp")
    @mock.patch("sagemaker.jumpstart.estimator.is_valid_model_id")
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
        mock_is_valid_model_id: mock.Mock,
        mock_sagemaker_timestamp: mock.Mock,
    ):
        mock_is_valid_model_id.return_value = True

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
        )

    @mock.patch("sagemaker.utils.sagemaker_timestamp")
    @mock.patch("sagemaker.jumpstart.estimator.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
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
        mock_session_estimator: mock.Mock,
        mock_session_model: mock.Mock,
        mock_is_valid_model_id: mock.Mock,
        mock_sagemaker_timestamp: mock.Mock,
    ):
        mock_is_valid_model_id.return_value = True

        mock_sagemaker_timestamp.return_value = "3456"

        mock_estimator_deploy.return_value = default_predictor

        model_id, model_version = "js-trainable-model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session_estimator.return_value = sagemaker_session
        mock_session_model.return_value = sagemaker_session

        mock_role = f"dsfsdfsd{time.time()}"
        mock_sagemaker_session = Session()
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
        )


def test_jumpstart_estimator_requires_model_id():
    with pytest.raises(ValueError):
        JumpStartEstimator()
