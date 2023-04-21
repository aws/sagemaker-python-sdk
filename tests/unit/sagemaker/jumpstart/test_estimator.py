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
from unittest import mock
import unittest

from sagemaker.jumpstart.estimator import JumpStartEstimator

from sagemaker.jumpstart.predictor import JumpStartPredictor
from sagemaker.jumpstart.utils import get_jumpstart_content_bucket
from tests.integ.sagemaker.jumpstart.utils import get_training_dataset_for_model_and_version
from tests.unit.sagemaker.jumpstart.utils import get_special_model_spec


class EstimatorTest(unittest.TestCase):

    execution_role = "fake role! do not use!"
    region = "us-west-2"

    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.fit")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.factory.estimator.get_execution_role")
    @mock.patch("sagemaker.jumpstart.factory.model.get_execution_role")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_non_prepacked(
        self,
        mock_get_execution_role_model: mock.Mock,
        mock_get_execution_role_estimator: mock.Mock,
        mock_estimator_deploy: mock.Mock,
        mock_estimator_fit: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
    ):
        model_id, model_version = "js-trainable-model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_get_execution_role_model.return_value = self.execution_role
        mock_get_execution_role_estimator.return_value = self.execution_role

        estimator = JumpStartEstimator(
            model_id=model_id,
        )

        mock_estimator_init.assert_called_once_with(
            model_id="js-trainable-model",
            model_version="*",
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
            role=self.execution_role,
            encrypt_inter_container_traffic=True,
        )

        channels = {
            "training": f"s3://{get_jumpstart_content_bucket(self.region)}/"
            f"{get_training_dataset_for_model_and_version(model_id, model_version)}",
        }

        estimator.fit(channels)

        mock_estimator_fit.assert_called_once_with(inputs=channels)

        estimator.deploy()

        mock_estimator_deploy.assert_called_once_with(
            model_id="js-trainable-model",
            model_version="*",
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
            predictor_cls=JumpStartPredictor,
            role=self.execution_role,
        )

    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.fit")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.factory.estimator.get_execution_role")
    @mock.patch("sagemaker.jumpstart.factory.model.get_execution_role")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_prepacked(
        self,
        mock_get_execution_role_model: mock.Mock,
        mock_get_execution_role_estimator: mock.Mock,
        mock_estimator_deploy: mock.Mock,
        mock_estimator_fit: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
    ):
        model_id, _ = "js-trainable-model-prepacked", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_get_execution_role_model.return_value = self.execution_role
        mock_get_execution_role_estimator.return_value = self.execution_role

        estimator = JumpStartEstimator(
            model_id=model_id,
        )

        mock_estimator_init.assert_called_once_with(
            model_id="js-trainable-model-prepacked",
            model_version="*",
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
            role=self.execution_role,
            encrypt_inter_container_traffic=False,
        )

        channels = {
            "training": f"s3://{get_jumpstart_content_bucket(self.region)}/"
            f"some-training-dataset-doesn't-matter",
        }

        estimator.fit(channels)

        mock_estimator_fit.assert_called_once_with(inputs=channels)

        estimator.deploy()

        mock_estimator_deploy.assert_called_once_with(
            model_id="js-trainable-model-prepacked",
            model_version="*",
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
            predictor_cls=JumpStartPredictor,
            role=self.execution_role,
        )
