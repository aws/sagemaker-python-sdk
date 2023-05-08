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
from inspect import signature
from typing import Optional, Set
from unittest import mock
import unittest
import pytest

from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.session import Session

from tests.unit.sagemaker.jumpstart.utils import get_special_model_spec, overwrite_dictionary


execution_role = "fake role! do not use!"
region = "us-west-2"
sagemaker_session = Session()
sagemaker_session.get_caller_identity_arn = lambda: execution_role
default_predictor = Predictor("blah", sagemaker_session)
default_predictor_with_presets = Predictor(
    "eiifccreeeiuihlrblivhchuefdckrluliilctfjgknk", sagemaker_session
)


class ModelTest(unittest.TestCase):
    @mock.patch("sagemaker.utils.sagemaker_timestamp")
    @mock.patch("sagemaker.jumpstart.model.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.model.Model.deploy")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_non_prepacked(
        self,
        mock_model_deploy: mock.Mock,
        mock_model_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_is_valid_model_id: mock.Mock,
        mock_sagemaker_timestamp: mock.Mock,
    ):
        mock_model_deploy.return_value = default_predictor

        mock_sagemaker_timestamp.return_value = "7777"

        mock_is_valid_model_id.return_value = True
        model_id, _ = "js-trainable-model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        model = JumpStartModel(
            model_id=model_id,
        )

        mock_model_init.assert_called_once_with(
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
            predictor_cls=Predictor,
            role=execution_role,
            sagemaker_session=sagemaker_session,
            enable_network_isolation=False,
            name="blahblahblah-7777",
        )

        model.deploy()

        mock_model_deploy.assert_called_once_with(
            initial_instance_count=1,
            instance_type="ml.p2.xlarge",
            wait=True,
            endpoint_name="blahblahblah-7777",
        )

    @mock.patch("sagemaker.jumpstart.model.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.model.Model.deploy")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_prepacked(
        self,
        mock_model_deploy: mock.Mock,
        mock_model_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_is_valid_model_id: mock.Mock,
    ):
        mock_model_deploy.return_value = default_predictor

        mock_is_valid_model_id.return_value = True

        model_id, _ = "js-model-class-model-prepacked", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        model = JumpStartModel(
            model_id=model_id,
        )

        mock_model_init.assert_called_once_with(
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
            predictor_cls=Predictor,
            role=execution_role,
            sagemaker_session=sagemaker_session,
            enable_network_isolation=False,
        )

        model.deploy()

        mock_model_deploy.assert_called_once_with(
            initial_instance_count=1,
            instance_type="ml.p3.2xlarge",
            wait=True,
        )

    @mock.patch("sagemaker.jumpstart.model.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.model.Model.deploy")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_deprecated(
        self,
        mock_model_deploy: mock.Mock,
        mock_model_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_is_valid_model_id: mock.Mock,
    ):
        mock_model_deploy.return_value = default_predictor

        mock_is_valid_model_id.return_value = True

        model_id, _ = "deprecated_model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        with pytest.raises(ValueError):
            JumpStartModel(
                model_id=model_id,
            )

        JumpStartModel(model_id=model_id, tolerate_deprecated_model=True).deploy()

    @mock.patch("sagemaker.jumpstart.model.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.model.Model.deploy")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_vulnerable(
        self,
        mock_model_deploy: mock.Mock,
        mock_model_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_is_valid_model_id: mock.Mock,
    ):
        mock_is_valid_model_id.return_value = True

        mock_model_deploy.return_value = default_predictor

        model_id, _ = "vulnerable_model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        with pytest.raises(ValueError):
            JumpStartModel(
                model_id=model_id,
            )

        JumpStartModel(model_id=model_id, tolerate_vulnerable_model=True).deploy()

    def test_model_use_kwargs(self):

        all_init_kwargs_used = {
            "image_uri": "Union[str, PipelineVariable]",
            "model_data": "Optional[Union[str, PipelineVariable]]",
            "role": "Optional[str] = None",
            "predictor_cls": Predictor,
            "env": {"1": 4},
            "name": "Optional[str] = None",
            "vpc_config": {"dsfsfs": "dfsfsd"},
            "enable_network_isolation": True,
            "model_kms_key": "Optional[str] = None",
            "image_config": {"s": "sd"},
            "source_dir": "Optional[str] = None",
            "code_location": "Optional[str] = None",
            "entry_point": "Optional[str] = None",
            "container_log_level": 83,
            "dependencies": ["help"],
            "git_config": {"dsfsd": "fsfs"},
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

        self.evaluate_model_workflow_with_kwargs(
            init_kwargs=all_init_kwargs_used,
            deploy_kwargs=all_deploy_kwargs_used,
        )

    @mock.patch("sagemaker.jumpstart.model.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.model.Model.deploy")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def evaluate_model_workflow_with_kwargs(
        self,
        mock_model_deploy: mock.Mock,
        mock_model_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_is_valid_model_id: mock.Mock,
        init_kwargs: Optional[dict] = None,
        deploy_kwargs: Optional[dict] = None,
    ):

        mock_model_deploy.return_value = default_predictor

        mock_is_valid_model_id.return_value = True

        mock_session.return_value = sagemaker_session

        if init_kwargs is None:
            init_kwargs = {}

        if deploy_kwargs is None:
            deploy_kwargs = {}

        model_id, _ = "js-model-class-model-prepacked", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        model = JumpStartModel(
            model_id=model_id,
            **init_kwargs,
        )

        expected_init_kwargs = overwrite_dictionary(
            {
                "image_uri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:"
                "1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                "model_data": "s3://jumpstart-cache-prod-us-west-2/huggingface-infer/prepack/"
                "v1.0.0/infer-prepack-huggingface-txt2img-conflictx-complex-lineart.tar.gz",
                "env": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "ENDPOINT_SERVER_TIMEOUT": "3600",
                    "MODEL_CACHE_ROOT": "/opt/ml/model",
                    "SAGEMAKER_ENV": "1",
                    "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
                },
                "predictor_cls": Predictor,
                "role": execution_role,
                "sagemaker_session": sagemaker_session,
                "enable_network_isolation": False,
            },
            init_kwargs,
        )

        mock_model_init.assert_called_once_with(**expected_init_kwargs)

        model.deploy(**deploy_kwargs)

        expected_deploy_kwargs = overwrite_dictionary(
            {"initial_instance_count": 1, "instance_type": "ml.p3.2xlarge"}, deploy_kwargs
        )

        mock_model_deploy.assert_called_once_with(**expected_deploy_kwargs)

    def test_jumpstart_model_kwargs_match_parent_class(self):

        """If you add arguments to <Model constructor>, this test will fail.
        Please add the new argument to the skip set below,
        and cut a ticket sev-3 to JumpStart team: AWS > SageMaker > JumpStart"""

        init_args_to_skip: Set[str] = set()
        deploy_args_to_skip: Set[str] = set(["kwargs"])

        parent_class_init = Model.__init__
        parent_class_init_args = set(signature(parent_class_init).parameters.keys())

        js_class_init = JumpStartModel.__init__
        js_class_init_args = set(signature(js_class_init).parameters.keys())

        assert js_class_init_args - parent_class_init_args == {
            "model_id",
            "model_version",
            "region",
            "tolerate_vulnerable_model",
            "tolerate_deprecated_model",
            "instance_type",
        }
        assert parent_class_init_args - js_class_init_args == init_args_to_skip

        parent_class_deploy = Model.deploy
        parent_class_deploy_args = set(signature(parent_class_deploy).parameters.keys())

        js_class_deploy = JumpStartModel.deploy
        js_class_deploy_args = set(signature(js_class_deploy).parameters.keys())

        assert js_class_deploy_args - parent_class_deploy_args == set()
        assert parent_class_deploy_args - js_class_deploy_args == deploy_args_to_skip

    @mock.patch("sagemaker.jumpstart.model.get_init_kwargs")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.model.is_valid_model_id")
    def test_is_valid_model_id(
        self,
        mock_is_valid_model_id: mock.Mock,
        mock_init: mock.Mock,
        mock_get_init_kwargs: mock.Mock,
    ):
        mock_is_valid_model_id.return_value = True
        JumpStartModel(model_id="valid_model_id")

        mock_is_valid_model_id.return_value = False
        with pytest.raises(ValueError):
            JumpStartModel(model_id="invalid_model_id")

    @mock.patch("sagemaker.jumpstart.model.get_default_predictor")
    @mock.patch("sagemaker.jumpstart.model.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.model.Model.deploy")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_no_predictor_returns_default_predictor(
        self,
        mock_model_deploy: mock.Mock,
        mock_model_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_is_valid_model_id: mock.Mock,
        mock_get_default_predictor: mock.Mock,
    ):
        mock_get_default_predictor.return_value = default_predictor_with_presets

        mock_model_deploy.return_value = default_predictor

        mock_is_valid_model_id.return_value = True

        model_id, _ = "js-model-class-model-prepacked", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        model = JumpStartModel(
            model_id=model_id,
        )

        predictor = model.deploy()

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

    @mock.patch("sagemaker.jumpstart.model.get_default_predictor")
    @mock.patch("sagemaker.jumpstart.model.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.model.Model.deploy")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_yes_predictor_returns_default_predictor(
        self,
        mock_model_deploy: mock.Mock,
        mock_model_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_is_valid_model_id: mock.Mock,
        mock_get_default_predictor: mock.Mock,
    ):
        mock_get_default_predictor.return_value = default_predictor_with_presets

        mock_model_deploy.return_value = default_predictor

        mock_is_valid_model_id.return_value = True

        model_id, _ = "js-model-class-model-prepacked", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        model = JumpStartModel(model_id=model_id, predictor_cls=Predictor)

        predictor = model.deploy()

        mock_get_default_predictor.assert_not_called()
        self.assertEqual(type(predictor), Predictor)
        self.assertEqual(predictor, default_predictor)


def test_jumpstart_model_requires_model_id():
    with pytest.raises(ValueError):
        JumpStartModel()
