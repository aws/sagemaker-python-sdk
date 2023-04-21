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

from sagemaker.jumpstart.model import JumpStartModel

from sagemaker.jumpstart.predictor import JumpStartPredictor
from tests.unit.sagemaker.jumpstart.utils import get_special_model_spec


class ModelTest(unittest.TestCase):

    execution_role = "fake role! do not use!"
    region = "us-west-2"

    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.model.Model.deploy")
    @mock.patch("sagemaker.jumpstart.factory.model.get_execution_role")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_non_prepacked(
        self,
        mock_get_execution_role: mock.Mock,
        mock_model_deploy: mock.Mock,
        mock_model_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
    ):
        model_id, _ = "js-trainable-model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_get_execution_role.return_value = self.execution_role

        model = JumpStartModel(
            model_id=model_id,
        )

        mock_model_init.assert_called_once_with(
            model_id="js-trainable-model",
            model_version="*",
            image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/"
            "autogluon-inference:0.4.3-gpu-py38",
            model_data="s3://jumpstart-cache-prod-us-west-2/autogluon-infer/"
            "v1.1.0/infer-autogluon-classification-ensemble.tar.gz",
            source_dir="s3://jumpstart-cache-prod-us-west-2/source-directory-"
            "tarballs/autogluon/inference/classification/v1.0.0/sourcedir.tar.gz",
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

        model.deploy()

        mock_model_deploy.assert_called_once_with(
            initial_instance_count=1, instance_type="ml.p2.xlarge"
        )

    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.model.Model.deploy")
    @mock.patch("sagemaker.jumpstart.factory.model.get_execution_role")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_prepacked(
        self,
        mock_get_execution_role: mock.Mock,
        mock_model_deploy: mock.Mock,
        mock_model_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
    ):
        model_id, _ = "js-model-class-model-prepacked", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_get_execution_role.return_value = self.execution_role

        model = JumpStartModel(
            model_id=model_id,
        )

        mock_model_init.assert_called_once_with(
            model_id="js-model-class-model-prepacked",
            model_version="*",
            image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:"
            "1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
            model_data="s3://jumpstart-cache-prod-us-west-2/huggingface-infer/prepack/"
            "v1.0.0/infer-prepack-huggingface-txt2img-conflictx-complex-lineart.tar.gz",
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

        model.deploy()

        mock_model_deploy.assert_called_once_with(
            initial_instance_count=1, instance_type="ml.p3.2xlarge"
        )
