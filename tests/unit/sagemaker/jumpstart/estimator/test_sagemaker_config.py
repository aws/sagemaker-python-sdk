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
import random
from unittest import mock
import unittest
from sagemaker.base_predictor import Predictor
from sagemaker.config.config_schema import (
    MODEL_ENABLE_NETWORK_ISOLATION_PATH,
    MODEL_EXECUTION_ROLE_ARN_PATH,
    TRAINING_JOB_ENABLE_NETWORK_ISOLATION_PATH,
    TRAINING_JOB_INTER_CONTAINER_ENCRYPTION_PATH,
    TRAINING_JOB_ROLE_ARN_PATH,
)
from sagemaker.jumpstart.constants import DEFAULT_JUMPSTART_SAGEMAKER_SESSION

from sagemaker.jumpstart.estimator import JumpStartEstimator
from sagemaker.jumpstart.enums import JumpStartModelType
from sagemaker.session import Session
from sagemaker.utils import resolve_value_from_config

from tests.unit.sagemaker.jumpstart.utils import get_special_model_spec


execution_role = "fake role! do not use!"
region = "us-west-2"
sagemaker_session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION
sagemaker_session.get_caller_identity_arn = lambda: execution_role
default_predictor = Predictor("eiifccreeeiujigjjdfgiujrcibigckbtregvkjeurru", sagemaker_session)


override_role = "asdfjkl;"
override_enable_network_isolation = random.choice([True, False])
override_encrypt_inter_container_traffic = random.choice([True, False])
override_inference_role = "eiifccreeeiunctrbkvbdrjbnelvuuunktbnbkdukklb"
override_inference_enable_network_isolation = random.choice([True, False])


config_role = "this is your security compliant role"
config_enable_network_isolation = random.choice([True, False])
config_intercontainer_encryption = random.choice([True, False])
config_inference_enable_network_isolation = random.choice([True, False])
config_inference_role = "this idsfass your security compliant role"


metadata_enable_network_isolation = random.choice([True, False])
metadata_intercontainer_encryption = random.choice([True, False])
metadata_inference_enable_network_isolation = random.choice([True, False])
metadata_inference_role = "th1234567iant role"


def config_value_impl(sagemaker_session: Session, config_path: str, sagemaker_config: dict):
    if config_path == TRAINING_JOB_ENABLE_NETWORK_ISOLATION_PATH:
        return config_enable_network_isolation

    if config_path == TRAINING_JOB_ROLE_ARN_PATH:
        return config_role

    if config_path == TRAINING_JOB_INTER_CONTAINER_ENCRYPTION_PATH:
        return config_intercontainer_encryption

    if config_path == MODEL_EXECUTION_ROLE_ARN_PATH:
        return config_inference_role

    if config_path == MODEL_ENABLE_NETWORK_ISOLATION_PATH:
        return config_inference_enable_network_isolation

    raise AssertionError(f"Bad config path: {config_path}")


class IntelligentDefaultsEstimatorTest(unittest.TestCase):
    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.factory.model._retrieve_model_init_kwargs")
    @mock.patch("sagemaker.jumpstart.factory.estimator._retrieve_estimator_init_kwargs")
    @mock.patch("sagemaker.utils.get_sagemaker_config_value")
    @mock.patch("sagemaker.jumpstart.utils.resolve_value_from_config")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_without_arg_overwrites_without_kwarg_collisions_with_config(
        self,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_resolve_value_from_config: mock.Mock,
        mock_get_sagemaker_config_value: mock.Mock,
        mock_retrieve_estimator_init_kwargs: mock.Mock,
        mock_retrieve_model_init_kwargs: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_estimator_deploy: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):

        mock_estimator_deploy.return_value = default_predictor

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "js-trainable-model", "*"

        mock_retrieve_estimator_init_kwargs.return_value = {}

        mock_get_sagemaker_config_value.side_effect = config_value_impl

        mock_resolve_value_from_config.side_effect = resolve_value_from_config
        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        estimator = JumpStartEstimator(
            model_id=model_id,
        )

        mock_retrieve_model_init_kwargs.return_value = {}

        self.assertEqual(mock_get_sagemaker_config_value.call_count, 1)
        self.assertEqual(mock_estimator_init.call_args[1].get("role"), config_role)
        assert "enable_network_isolation" not in mock_estimator_init.call_args[1]
        assert "encrypt_inter_container_traffic" not in mock_estimator_init.call_args[1]

        estimator.deploy()

        self.assertEqual(mock_get_sagemaker_config_value.call_count, 3)

        self.assertEqual(mock_estimator_deploy.call_args[1].get("role"), config_inference_role)

        assert "enable_network_isolation" not in mock_estimator_deploy.call_args[1]

    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.factory.model._retrieve_model_init_kwargs")
    @mock.patch("sagemaker.jumpstart.factory.estimator._retrieve_estimator_init_kwargs")
    @mock.patch("sagemaker.utils.get_sagemaker_config_value")
    @mock.patch("sagemaker.jumpstart.utils.resolve_value_from_config")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_without_arg_overwrites_with_kwarg_collisions_with_config(
        self,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_resolve_value_from_config: mock.Mock,
        mock_get_sagemaker_config_value: mock.Mock,
        mock_retrieve_kwargs: mock.Mock,
        mock_model_retrieve_kwargs: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_estimator_deploy: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):

        mock_estimator_deploy.return_value = default_predictor

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "js-trainable-model", "*"

        mock_retrieve_kwargs.return_value = {
            "enable_network_isolation": metadata_enable_network_isolation,
            "encrypt_inter_container_traffic": metadata_intercontainer_encryption,
        }

        mock_get_sagemaker_config_value.side_effect = config_value_impl

        mock_resolve_value_from_config.side_effect = resolve_value_from_config
        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        estimator = JumpStartEstimator(
            model_id=model_id,
        )

        self.assertEqual(mock_get_sagemaker_config_value.call_count, 3)
        self.assertEqual(mock_estimator_init.call_args[1].get("role"), config_role)
        self.assertEqual(
            mock_estimator_init.call_args[1].get("enable_network_isolation"),
            config_enable_network_isolation,
        )
        self.assertEqual(
            mock_estimator_init.call_args[1].get("encrypt_inter_container_traffic"),
            config_intercontainer_encryption,
        )

        mock_model_retrieve_kwargs.side_effect = [
            {
                "enable_network_isolation": metadata_inference_enable_network_isolation,
            },
        ]

        estimator.deploy()

        self.assertEqual(mock_get_sagemaker_config_value.call_count, 6)

        self.assertEqual(mock_estimator_deploy.call_args[1].get("role"), config_inference_role)

        self.assertEqual(
            mock_estimator_deploy.call_args[1].get("enable_network_isolation"),
            config_inference_enable_network_isolation,
        )

    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.factory.model._retrieve_model_init_kwargs")
    @mock.patch("sagemaker.jumpstart.factory.estimator._retrieve_estimator_init_kwargs")
    @mock.patch("sagemaker.utils.get_sagemaker_config_value")
    @mock.patch("sagemaker.jumpstart.utils.resolve_value_from_config")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_with_arg_overwrites_with_kwarg_collisions_with_config(
        self,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_resolve_value_from_config: mock.Mock,
        mock_get_sagemaker_config_value: mock.Mock,
        mock_retrieve_kwargs: mock.Mock,
        mock_model_retrieve_kwargs: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_estimator_deploy: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):

        mock_estimator_deploy.return_value = default_predictor

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "js-trainable-model", "*"

        mock_retrieve_kwargs.return_value = {
            "enable_network_isolation": metadata_enable_network_isolation,
            "encrypt_inter_container_traffic": metadata_intercontainer_encryption,
        }

        mock_get_sagemaker_config_value.side_effect = config_value_impl

        mock_resolve_value_from_config.side_effect = resolve_value_from_config
        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        estimator = JumpStartEstimator(
            model_id=model_id,
            role=override_role,
            enable_network_isolation=override_enable_network_isolation,
            encrypt_inter_container_traffic=override_encrypt_inter_container_traffic,
        )

        self.assertEqual(mock_get_sagemaker_config_value.call_count, 1)
        self.assertEqual(mock_estimator_init.call_args[1].get("role"), override_role)
        self.assertEqual(
            mock_estimator_init.call_args[1].get("enable_network_isolation"),
            override_enable_network_isolation,
        )
        self.assertEqual(
            mock_estimator_init.call_args[1].get("encrypt_inter_container_traffic"),
            override_encrypt_inter_container_traffic,
        )

        mock_model_retrieve_kwargs.side_effect = [
            {
                "enable_network_isolation": metadata_inference_enable_network_isolation,
            },
        ]

        mock_inference_override_role = "fsdfsdf"
        estimator.deploy(
            role=mock_inference_override_role,
            enable_network_isolation=override_inference_enable_network_isolation,
        )

        self.assertEqual(mock_get_sagemaker_config_value.call_count, 3)

        self.assertEqual(
            mock_estimator_deploy.call_args[1].get("role"), mock_inference_override_role
        )

        self.assertEqual(
            mock_estimator_deploy.call_args[1].get("enable_network_isolation"),
            override_inference_enable_network_isolation,
        )

    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.factory.model._retrieve_model_init_kwargs")
    @mock.patch("sagemaker.jumpstart.factory.estimator._retrieve_estimator_init_kwargs")
    @mock.patch("sagemaker.utils.get_sagemaker_config_value")
    @mock.patch("sagemaker.jumpstart.utils.resolve_value_from_config")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_with_arg_overwrites_without_kwarg_collisions_with_config(
        self,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_resolve_value_from_config: mock.Mock,
        mock_get_sagemaker_config_value: mock.Mock,
        mock_retrieve_kwargs: mock.Mock,
        mock_model_retrieve_kwargs: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_estimator_deploy: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):

        mock_estimator_deploy.return_value = default_predictor

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "js-trainable-model", "*"

        mock_retrieve_kwargs.return_value = {}

        mock_get_sagemaker_config_value.side_effect = config_value_impl

        mock_resolve_value_from_config.side_effect = resolve_value_from_config
        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        estimator = JumpStartEstimator(
            model_id=model_id,
            role=override_role,
            enable_network_isolation=override_enable_network_isolation,
            encrypt_inter_container_traffic=override_encrypt_inter_container_traffic,
        )

        self.assertEqual(mock_get_sagemaker_config_value.call_count, 1)
        self.assertEqual(mock_estimator_init.call_args[1].get("role"), override_role)
        self.assertEqual(
            mock_estimator_init.call_args[1].get("enable_network_isolation"),
            override_enable_network_isolation,
        )
        self.assertEqual(
            mock_estimator_init.call_args[1].get("encrypt_inter_container_traffic"),
            override_encrypt_inter_container_traffic,
        )

        mock_model_retrieve_kwargs.return_value = {}

        mock_inference_override_role = "fsdfsdf"
        estimator.deploy(
            role=mock_inference_override_role,
            enable_network_isolation=override_inference_enable_network_isolation,
        )

        self.assertEqual(mock_get_sagemaker_config_value.call_count, 3)

        self.assertEqual(
            mock_estimator_deploy.call_args[1].get("role"), mock_inference_override_role
        )

        self.assertEqual(
            mock_estimator_deploy.call_args[1].get("enable_network_isolation"),
            override_inference_enable_network_isolation,
        )

    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.session.Session.get_caller_identity_arn")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.factory.model._retrieve_model_init_kwargs")
    @mock.patch("sagemaker.jumpstart.factory.estimator._retrieve_estimator_init_kwargs")
    @mock.patch("sagemaker.utils.get_sagemaker_config_value")
    @mock.patch("sagemaker.jumpstart.utils.resolve_value_from_config")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_without_arg_overwrites_without_kwarg_collisions_without_config(
        self,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_resolve_value_from_config: mock.Mock,
        mock_get_sagemaker_config_value: mock.Mock,
        mock_retrieve_kwargs: mock.Mock,
        mock_retrieve_model_init_kwargs: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_estimator_deploy: mock.Mock,
        mock_get_caller_identity_arn: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):

        mock_estimator_deploy.return_value = default_predictor

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "js-trainable-model", "*"

        mock_get_caller_identity_arn.return_value = execution_role

        mock_retrieve_kwargs.return_value = {}

        mock_get_sagemaker_config_value.return_value = None

        mock_resolve_value_from_config.side_effect = resolve_value_from_config
        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        estimator = JumpStartEstimator(
            model_id=model_id,
        )

        self.assertEqual(mock_get_sagemaker_config_value.call_count, 1)
        self.assertEqual(mock_estimator_init.call_args[1].get("role"), execution_role)
        assert "enable_network_isolation" not in mock_estimator_init.call_args[1]
        assert "encrypt_inter_container_traffic" not in mock_estimator_init.call_args[1]

        estimator.deploy()

        mock_retrieve_model_init_kwargs.return_value = {}

        self.assertEqual(mock_get_sagemaker_config_value.call_count, 3)

        self.assertEqual(mock_estimator_deploy.call_args[1].get("role"), execution_role)

        assert "enable_network_isolation" not in mock_estimator_deploy.call_args[1]

    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.session.Session.get_caller_identity_arn")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.factory.model._retrieve_model_init_kwargs")
    @mock.patch("sagemaker.jumpstart.factory.estimator._retrieve_estimator_init_kwargs")
    @mock.patch("sagemaker.utils.get_sagemaker_config_value")
    @mock.patch("sagemaker.jumpstart.utils.resolve_value_from_config")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_without_arg_overwrites_with_kwarg_collisions_without_config(
        self,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_resolve_value_from_config: mock.Mock,
        mock_get_sagemaker_config_value: mock.Mock,
        mock_retrieve_kwargs: mock.Mock,
        mock_model_retrieve_kwargs: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_estimator_deploy: mock.Mock,
        mock_get_caller_identity_arn: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):

        mock_estimator_deploy.return_value = default_predictor

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        mock_get_caller_identity_arn.return_value = execution_role
        model_id, _ = "js-trainable-model", "*"

        mock_retrieve_kwargs.return_value = {
            "enable_network_isolation": metadata_enable_network_isolation,
            "encrypt_inter_container_traffic": metadata_intercontainer_encryption,
        }

        mock_get_sagemaker_config_value.return_value = None

        mock_resolve_value_from_config.side_effect = resolve_value_from_config
        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        estimator = JumpStartEstimator(
            model_id=model_id,
        )

        self.assertEqual(mock_get_sagemaker_config_value.call_count, 3)
        self.assertEqual(mock_estimator_init.call_args[1].get("role"), execution_role)
        self.assertEqual(
            mock_estimator_init.call_args[1].get("enable_network_isolation"),
            metadata_enable_network_isolation,
        )
        self.assertEqual(
            mock_estimator_init.call_args[1].get("encrypt_inter_container_traffic"),
            metadata_intercontainer_encryption,
        )

        mock_model_retrieve_kwargs.return_value = {
            "enable_network_isolation": metadata_inference_enable_network_isolation,
        }

        estimator.deploy()

        self.assertEqual(mock_get_sagemaker_config_value.call_count, 6)

        self.assertEqual(mock_estimator_deploy.call_args[1].get("role"), execution_role)

        self.assertEqual(
            mock_estimator_deploy.call_args[1].get("enable_network_isolation"),
            metadata_inference_enable_network_isolation,
        )

    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.factory.model._retrieve_model_init_kwargs")
    @mock.patch("sagemaker.jumpstart.factory.estimator._retrieve_estimator_init_kwargs")
    @mock.patch("sagemaker.utils.get_sagemaker_config_value")
    @mock.patch("sagemaker.jumpstart.utils.resolve_value_from_config")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_with_arg_overwrites_with_kwarg_collisions_without_config(
        self,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_resolve_value_from_config: mock.Mock,
        mock_get_sagemaker_config_value: mock.Mock,
        mock_retrieve_kwargs: mock.Mock,
        mock_model_retrieve_kwargs: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_estimator_deploy: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):
        mock_estimator_deploy.return_value = default_predictor

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "js-trainable-model", "*"

        mock_retrieve_kwargs.return_value = {
            "enable_network_isolation": metadata_enable_network_isolation,
            "encrypt_inter_container_traffic": metadata_intercontainer_encryption,
        }

        mock_get_sagemaker_config_value.return_value = None

        mock_resolve_value_from_config.side_effect = resolve_value_from_config
        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        estimator = JumpStartEstimator(
            model_id=model_id,
            role=override_role,
            enable_network_isolation=override_enable_network_isolation,
            encrypt_inter_container_traffic=override_encrypt_inter_container_traffic,
        )

        self.assertEqual(mock_get_sagemaker_config_value.call_count, 1)
        self.assertEqual(mock_estimator_init.call_args[1].get("role"), override_role)
        self.assertEqual(
            mock_estimator_init.call_args[1].get("enable_network_isolation"),
            override_enable_network_isolation,
        )
        self.assertEqual(
            mock_estimator_init.call_args[1].get("encrypt_inter_container_traffic"),
            override_encrypt_inter_container_traffic,
        )

        mock_model_retrieve_kwargs.return_value = {
            "enable_network_isolation": metadata_inference_enable_network_isolation,
        }

        estimator.deploy(
            role=override_inference_role,
            enable_network_isolation=override_inference_enable_network_isolation,
        )

        self.assertEqual(mock_get_sagemaker_config_value.call_count, 3)

        self.assertEqual(mock_estimator_deploy.call_args[1].get("role"), override_inference_role)

        self.assertEqual(
            mock_estimator_deploy.call_args[1].get("enable_network_isolation"),
            override_inference_enable_network_isolation,
        )

    @mock.patch("sagemaker.jumpstart.estimator.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.deploy")
    @mock.patch("sagemaker.jumpstart.estimator.Estimator.__init__")
    @mock.patch("sagemaker.jumpstart.factory.estimator._retrieve_estimator_init_kwargs")
    @mock.patch("sagemaker.utils.get_sagemaker_config_value")
    @mock.patch("sagemaker.jumpstart.utils.resolve_value_from_config")
    @mock.patch("sagemaker.jumpstart.factory.estimator.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_with_arg_overwrites_without_kwarg_collisions_without_config(
        self,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_resolve_value_from_config: mock.Mock,
        mock_get_sagemaker_config_value: mock.Mock,
        mock_retrieve_kwargs: mock.Mock,
        mock_estimator_init: mock.Mock,
        mock_estimator_deploy: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):
        mock_estimator_deploy.return_value = default_predictor

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "js-trainable-model", "*"

        mock_retrieve_kwargs.return_value = {}

        mock_get_sagemaker_config_value.return_value = None

        mock_resolve_value_from_config.side_effect = resolve_value_from_config
        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        estimator = JumpStartEstimator(
            model_id=model_id,
            role=override_role,
            enable_network_isolation=override_enable_network_isolation,
            encrypt_inter_container_traffic=override_encrypt_inter_container_traffic,
        )
        self.assertEqual(mock_get_sagemaker_config_value.call_count, 1)
        self.assertEqual(mock_estimator_init.call_args[1].get("role"), override_role)
        self.assertEqual(
            mock_estimator_init.call_args[1].get("enable_network_isolation"),
            override_enable_network_isolation,
        )
        self.assertEqual(
            mock_estimator_init.call_args[1].get("encrypt_inter_container_traffic"),
            override_encrypt_inter_container_traffic,
        )

        estimator.deploy(
            role=override_inference_role,
            enable_network_isolation=override_enable_network_isolation,
        )

        self.assertEqual(mock_get_sagemaker_config_value.call_count, 3)

        self.assertEqual(mock_estimator_deploy.call_args[1].get("role"), override_inference_role)

        self.assertEqual(
            mock_estimator_deploy.call_args[1].get("enable_network_isolation"),
            override_enable_network_isolation,
        )
