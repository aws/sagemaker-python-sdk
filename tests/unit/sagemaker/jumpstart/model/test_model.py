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
from mock import MagicMock
import pytest
from sagemaker.async_inference.async_inference_config import AsyncInferenceConfig
from sagemaker.jumpstart.artifacts.environment_variables import (
    _retrieve_default_environment_variables,
)
from sagemaker.jumpstart.constants import DEFAULT_JUMPSTART_SAGEMAKER_SESSION
from sagemaker.jumpstart.enums import JumpStartScriptScope, JumpStartTag

from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.session_settings import SessionSettings
from sagemaker.enums import EndpointType
from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements

from tests.unit.sagemaker.jumpstart.utils import (
    get_special_model_spec,
    overwrite_dictionary,
    get_special_model_spec_for_inference_component_based_endpoint,
)

execution_role = "fake role! do not use!"
region = "us-west-2"
sagemaker_session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION
sagemaker_session.get_caller_identity_arn = lambda: execution_role
default_predictor = Predictor("blah", sagemaker_session)
default_predictor_with_presets = Predictor(
    "eiifccreeeiuihlrblivhchuefdckrluliilctfjgknk", sagemaker_session
)


class ModelTest(unittest.TestCase):

    mock_session_empty_config = MagicMock(sagemaker_config={})

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
            tags=[
                {"Key": JumpStartTag.MODEL_ID, "Value": "js-trainable-model"},
                {"Key": JumpStartTag.MODEL_VERSION, "Value": "1.1.1"},
            ],
            endpoint_logging=False,
        )

    @mock.patch("sagemaker.utils.sagemaker_timestamp")
    @mock.patch("sagemaker.jumpstart.model.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.model.Model.deploy")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_non_prepacked_inference_component_based_endpoint(
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

        mock_get_model_specs.side_effect = (
            get_special_model_spec_for_inference_component_based_endpoint
        )

        mock_session.return_value = sagemaker_session

        model = JumpStartModel(
            model_id=model_id,
        )

        resource_requirements = ResourceRequirements(
            requests={
                "num_accelerators": 1,
                "memory": 34360,
            },
            limits={},
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
            resources=resource_requirements,
        )

        model.deploy(endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED)

        mock_model_deploy.assert_called_once_with(
            initial_instance_count=1,
            instance_type="ml.p2.xlarge",
            wait=True,
            endpoint_name="blahblahblah-7777",
            tags=[
                {"Key": JumpStartTag.MODEL_ID, "Value": "js-trainable-model"},
                {"Key": JumpStartTag.MODEL_VERSION, "Value": "1.1.1"},
            ],
            endpoint_logging=False,
            resources=resource_requirements,
            endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
        )

    @mock.patch("sagemaker.utils.sagemaker_timestamp")
    @mock.patch("sagemaker.jumpstart.model.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.model.Model.deploy")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_non_prepacked_inference_component_based_endpoint_no_default_pass_custom_resources(
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

        custom_resource_requirements = ResourceRequirements(
            requests={
                "num_accelerators": 2,
                "memory": 20480,
            },
            limits={},
        )

        model.deploy(
            endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
            resources=custom_resource_requirements,
        )

        mock_model_deploy.assert_called_once_with(
            initial_instance_count=1,
            instance_type="ml.p3.2xlarge",
            wait=True,
            tags=[
                {"Key": JumpStartTag.MODEL_ID, "Value": "js-model-class-model-prepacked"},
                {"Key": JumpStartTag.MODEL_VERSION, "Value": "1.1.0"},
            ],
            endpoint_logging=False,
            resources=custom_resource_requirements,
            endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
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
            tags=[
                {"Key": JumpStartTag.MODEL_ID, "Value": "js-model-class-model-prepacked"},
                {"Key": JumpStartTag.MODEL_VERSION, "Value": "1.1.0"},
            ],
            endpoint_logging=False,
        )

    @mock.patch("sagemaker.model.LOGGER.warning")
    @mock.patch("sagemaker.utils.sagemaker_timestamp")
    @mock.patch("sagemaker.session.Session.endpoint_from_production_variants")
    @mock.patch("sagemaker.session.Session.create_model")
    @mock.patch("sagemaker.jumpstart.model.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_no_compiled_model_warning_log_js_models(
        self,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_is_valid_model_id: mock.Mock,
        mock_create_model: mock.Mock,
        mock_endpoint_from_production_variants: mock.Mock,
        mock_timestamp: mock.Mock,
        mock_warning: mock.Mock(),
    ):

        mock_timestamp.return_value = "1234"

        mock_is_valid_model_id.return_value = True

        model_id, _ = "gated_llama_neuron_model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        model = JumpStartModel(
            model_id=model_id,
        )

        model.deploy(accept_eula=True)

        mock_warning.assert_not_called()

    @mock.patch("sagemaker.utils.sagemaker_timestamp")
    @mock.patch("sagemaker.session.Session.endpoint_from_production_variants")
    @mock.patch("sagemaker.session.Session.create_model")
    @mock.patch("sagemaker.jumpstart.model.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_eula_gated_conditional_s3_prefix_metadata_model(
        self,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_is_valid_model_id: mock.Mock,
        mock_create_model: mock.Mock,
        mock_endpoint_from_production_variants: mock.Mock,
        mock_timestamp: mock.Mock,
    ):

        mock_timestamp.return_value = "1234"

        mock_is_valid_model_id.return_value = True

        model_id, _ = "gated_variant-model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        model = JumpStartModel(
            model_id=model_id,
        )

        model.deploy(accept_eula=True, instance_type="ml.p2.xlarge")

        mock_create_model.assert_called_once_with(
            name="dfsdfsds-1234",
            role="fake role! do not use!",
            container_defs={
                "Image": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-"
                "inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "ENDPOINT_SERVER_TIMEOUT": "3600",
                    "MODEL_CACHE_ROOT": "/opt/ml/model",
                    "SAGEMAKER_ENV": "1",
                    "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
                },
                "ModelDataSource": {
                    "S3DataSource": {
                        "S3Uri": "s3://jumpstart-private-cache-prod-us-west-2/some-instance-specific/model/prefix/",
                        "S3DataType": "S3Prefix",
                        "CompressionType": "None",
                        "ModelAccessConfig": {"AcceptEula": True},
                    }
                },
            },
            vpc_config=None,
            enable_network_isolation=True,
            tags=[
                {"Key": "sagemaker-sdk:jumpstart-model-id", "Value": "gated_variant-model"},
                {"Key": "sagemaker-sdk:jumpstart-model-version", "Value": "1.0.0"},
            ],
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
            "endpoint_logging": False,
        }

        self.evaluate_model_workflow_with_kwargs(
            init_kwargs=all_init_kwargs_used,
            deploy_kwargs=all_deploy_kwargs_used,
        )

    @mock.patch("sagemaker.jumpstart.factory.model.environment_variables.retrieve_default")
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
        mock_retrieve_environment_variables: mock.Mock,
        init_kwargs: Optional[dict] = None,
        deploy_kwargs: Optional[dict] = None,
    ):

        mock_retrieve_environment_variables.side_effect = _retrieve_default_environment_variables

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

        mock_retrieve_environment_variables.assert_called_once()

        expected_deploy_kwargs = overwrite_dictionary(
            {
                "initial_instance_count": 1,
                "instance_type": "ml.p3.2xlarge",
                "tags": [
                    {"Key": JumpStartTag.MODEL_ID, "Value": "js-model-class-model-prepacked"},
                    {"Key": JumpStartTag.MODEL_VERSION, "Value": "1.1.0"},
                ],
            },
            deploy_kwargs,
        )

        mock_model_deploy.assert_called_once_with(**expected_deploy_kwargs)

    def test_jumpstart_model_kwargs_match_parent_class(self):
        """If you add arguments to <Model constructor>, this test will fail.
        Please add the new argument to the skip set below,
        and reach out to JumpStart team."""

        init_args_to_skip: Set[str] = set([])
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
            "model_package_arn",
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
            sagemaker_session=model.sagemaker_session,
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
    def test_no_predictor_yes_async_inference_config(
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

        model.deploy(async_inference_config=AsyncInferenceConfig())

        mock_get_default_predictor.assert_not_called()

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

    @mock.patch("sagemaker.jumpstart.model.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.factory.model._retrieve_model_init_kwargs")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.model.JumpStartModelsAccessor.reset_cache")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_model_id_not_found_refeshes_cach_inference(
        self,
        mock_reset_cache: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_retrieve_kwargs: mock.Mock,
        mock_model_init: mock.Mock,
        mock_is_valid_model_id: mock.Mock,
    ):

        mock_is_valid_model_id.side_effect = [False, False]

        model_id, _ = "js-trainable-model", "*"

        mock_retrieve_kwargs.return_value = {}

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        with pytest.raises(ValueError):
            JumpStartModel(
                model_id=model_id,
            )

        mock_reset_cache.assert_called_once_with()
        mock_is_valid_model_id.assert_has_calls(
            calls=[
                mock.call(
                    model_id="js-trainable-model",
                    model_version=None,
                    region=None,
                    script=JumpStartScriptScope.INFERENCE,
                    sagemaker_session=None,
                ),
                mock.call(
                    model_id="js-trainable-model",
                    model_version=None,
                    region=None,
                    script=JumpStartScriptScope.INFERENCE,
                    sagemaker_session=None,
                ),
            ]
        )

        mock_is_valid_model_id.reset_mock()
        mock_reset_cache.reset_mock()

        mock_is_valid_model_id.side_effect = [False, True]
        JumpStartModel(
            model_id=model_id,
        )

        mock_reset_cache.assert_called_once_with()
        mock_is_valid_model_id.assert_has_calls(
            calls=[
                mock.call(
                    model_id="js-trainable-model",
                    model_version=None,
                    region=None,
                    script=JumpStartScriptScope.INFERENCE,
                    sagemaker_session=None,
                ),
                mock.call(
                    model_id="js-trainable-model",
                    model_version=None,
                    region=None,
                    script=JumpStartScriptScope.INFERENCE,
                    sagemaker_session=None,
                ),
            ]
        )

    @mock.patch("sagemaker.jumpstart.model.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_jumpstart_model_tags(
        self,
        mock_get_model_specs: mock.Mock,
        mock_is_valid_model_id: mock.Mock,
    ):

        mock_is_valid_model_id.return_value = True

        model_id, _ = "env-var-variant-model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session = MagicMock(sagemaker_config={})

        model = JumpStartModel(model_id=model_id, sagemaker_session=mock_session)

        model.deploy(tags=[{"Key": "blah", "Value": "blahagain"}])

        js_tags = [
            {"Key": "sagemaker-sdk:jumpstart-model-id", "Value": "env-var-variant-model"},
            {"Key": "sagemaker-sdk:jumpstart-model-version", "Value": "1.0.0"},
        ]

        self.assertEqual(
            mock_session.create_model.call_args[1]["tags"],
            [{"Key": "blah", "Value": "blahagain"}] + js_tags,
        )

        self.assertEqual(
            mock_session.endpoint_from_production_variants.call_args[1]["tags"],
            [{"Key": "blah", "Value": "blahagain"}] + js_tags,
        )

    @mock.patch("sagemaker.jumpstart.model.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_jumpstart_model_tags_disabled(
        self,
        mock_get_model_specs: mock.Mock,
        mock_is_valid_model_id: mock.Mock,
    ):

        mock_is_valid_model_id.return_value = True

        model_id, _ = "env-var-variant-model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        settings = SessionSettings(include_jumpstart_tags=False)
        mock_session = MagicMock(sagemaker_config={}, settings=settings)

        model = JumpStartModel(model_id=model_id, sagemaker_session=mock_session)

        model.deploy(tags=[{"Key": "blah", "Value": "blahagain"}])

        self.assertEqual(
            mock_session.create_model.call_args[1]["tags"],
            [{"Key": "blah", "Value": "blahagain"}],
        )

        self.assertEqual(
            mock_session.endpoint_from_production_variants.call_args[1]["tags"],
            [{"Key": "blah", "Value": "blahagain"}],
        )

    @mock.patch("sagemaker.jumpstart.model.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_jumpstart_model_package_arn(
        self,
        mock_get_model_specs: mock.Mock,
        mock_is_valid_model_id: mock.Mock,
    ):

        mock_is_valid_model_id.return_value = True

        model_id, _ = "js-model-package-arn", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session = MagicMock(sagemaker_config={})

        model = JumpStartModel(model_id=model_id, sagemaker_session=mock_session)

        tag = {"Key": "foo", "Value": "bar"}
        tags = [tag]

        model.deploy(tags=tags)

        self.assertEqual(
            mock_session.create_model.call_args[0][2],
            {
                "ModelPackageName": "arn:aws:sagemaker:us-west-2:594846645681:model-package"
                "/llama2-7b-f-e46eb8a833643ed58aaccd81498972c3"
            },
        )

        self.assertIn(tag, mock_session.create_model.call_args[1]["tags"])

    @mock.patch("sagemaker.jumpstart.model.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_jumpstart_model_package_arn_override(
        self,
        mock_get_model_specs: mock.Mock,
        mock_is_valid_model_id: mock.Mock,
    ):

        mock_is_valid_model_id.return_value = True

        # arbitrary model without model packarn arn
        model_id, _ = "js-trainable-model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session = MagicMock(sagemaker_config={})

        model_package_arn = (
            "arn:aws:sagemaker:us-west-2:867530986753:model-package/"
            "llama2-ynnej-f-e46eb8a833643ed58aaccd81498972c3"
        )
        model = JumpStartModel(
            model_id=model_id, model_package_arn=model_package_arn, sagemaker_session=mock_session
        )

        model.deploy()

        self.assertEqual(
            mock_session.create_model.call_args[0][2],
            {
                "ModelPackageName": model_package_arn,
                "Environment": {
                    "ENDPOINT_SERVER_TIMEOUT": "3600",
                    "MODEL_CACHE_ROOT": "/opt/ml/model",
                    "SAGEMAKER_ENV": "1",
                    "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
                    "SAGEMAKER_PROGRAM": "inference.py",
                },
            },
        )

    @mock.patch("sagemaker.jumpstart.model.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_jumpstart_model_package_arn_unsupported_region(
        self,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_is_valid_model_id: mock.Mock,
    ):

        mock_is_valid_model_id.return_value = True

        model_id, _ = "js-model-package-arn", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = MagicMock(sagemaker_config={})

        with pytest.raises(ValueError) as e:
            JumpStartModel(model_id=model_id, region="us-east-2")
        assert (
            str(e.value) == "Model package arn for 'js-model-package-arn' not supported in "
            "us-east-2. Please try one of the following regions: us-west-2, us-east-1."
        )

    @mock.patch("sagemaker.utils.sagemaker_timestamp")
    @mock.patch("sagemaker.jumpstart.model.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.model.Model.deploy")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_LOGGER.info")
    def test_model_data_s3_prefix_override(
        self,
        mock_js_info_logger: mock.Mock,
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

        JumpStartModel(model_id=model_id, model_data="s3://some-bucket/path/to/prefix/")

        mock_model_init.assert_called_once_with(
            image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/"
            "autogluon-inference:0.4.3-gpu-py38",
            model_data={
                "S3DataSource": {
                    "S3Uri": "s3://some-bucket/path/to/prefix/",
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            },
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

        mock_js_info_logger.assert_called_with(
            "S3 prefix model_data detected for JumpStartModel: '%s'. "
            "Converting to S3DataSource dictionary: '%s'.",
            "s3://some-bucket/path/to/prefix/",
            '{"S3DataSource": {"S3Uri": "s3://some-bucket/path/to/prefix/", '
            '"S3DataType": "S3Prefix", "CompressionType": "None"}}',
        )

    @mock.patch("sagemaker.jumpstart.model.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.model.Model.deploy")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_LOGGER.info")
    def test_model_data_s3_prefix_model(
        self,
        mock_js_info_logger: mock.Mock,
        mock_model_deploy: mock.Mock,
        mock_model_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_is_valid_model_id: mock.Mock,
    ):
        mock_model_deploy.return_value = default_predictor

        mock_is_valid_model_id.return_value = True
        model_id, _ = "model_data_s3_prefix_model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        JumpStartModel(model_id=model_id, instance_type="ml.p2.xlarge")

        mock_model_init.assert_called_once_with(
            image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.12.0-gpu-py38",
            model_data={
                "S3DataSource": {
                    "S3Uri": "s3://jumpstart-cache-prod-us-west-2/huggingface-infer/prepack/v1.0.1/",
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            },
            predictor_cls=Predictor,
            role=execution_role,
            sagemaker_session=sagemaker_session,
            enable_network_isolation=False,
        )

        mock_js_info_logger.assert_not_called()

    @mock.patch("sagemaker.jumpstart.model.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.model.Model.deploy")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_LOGGER.info")
    def test_model_artifact_variant_model(
        self,
        mock_js_info_logger: mock.Mock,
        mock_model_deploy: mock.Mock,
        mock_model_init: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_is_valid_model_id: mock.Mock,
    ):
        mock_model_deploy.return_value = default_predictor

        mock_is_valid_model_id.return_value = True
        model_id, _ = "model-artifact-variant-model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        # this instance type has a special model artifact
        JumpStartModel(model_id=model_id, instance_type="ml.p2.xlarge")

        mock_model_init.assert_called_once_with(
            image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-"
            "inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
            model_data="s3://jumpstart-cache-prod-us-west-2/hello-world-1",
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
            enable_network_isolation=True,
        )

        mock_model_init.reset_mock()

        JumpStartModel(model_id=model_id, instance_type="ml.p99.xlarge")

        mock_model_init.assert_called_once_with(
            image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.5.0-gpu-py3",
            model_data="s3://jumpstart-cache-prod-us-west-2/basfsdfssf",
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
            enable_network_isolation=True,
        )

    @mock.patch("sagemaker.jumpstart.model.is_valid_model_id")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.model.Model.deploy")
    @mock.patch("sagemaker.jumpstart.model.Model.register")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_model_registry_accept_and_response_types(
        self,
        mock_model_register: mock.Mock,
        mock_model_deploy: mock.Mock,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_is_valid_model_id: mock.Mock,
    ):
        mock_model_deploy.return_value = default_predictor

        mock_is_valid_model_id.return_value = True
        model_id, _ = "model_data_s3_prefix_model", "*"

        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        model = JumpStartModel(model_id=model_id, instance_type="ml.p2.xlarge")

        model.register()

        mock_model_register.assert_called_once_with(
            content_types=["application/x-text"],
            response_types=["application/json;verbose", "application/json"],
        )


def test_jumpstart_model_requires_model_id():
    with pytest.raises(ValueError):
        JumpStartModel()
