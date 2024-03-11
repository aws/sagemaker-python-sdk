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
from sagemaker.config.config_schema import (
    MODEL_ENABLE_NETWORK_ISOLATION_PATH,
    MODEL_EXECUTION_ROLE_ARN_PATH,
)
from sagemaker.jumpstart.constants import DEFAULT_JUMPSTART_SAGEMAKER_SESSION

from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.jumpstart.enums import JumpStartModelType
from sagemaker.session import Session
from sagemaker.utils import resolve_value_from_config

from tests.unit.sagemaker.jumpstart.utils import get_special_model_spec


execution_role = "fake role! do not use!"
region = "us-west-2"
sagemaker_session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION
sagemaker_session.get_caller_identity_arn = lambda: execution_role

override_role = "fdsfsdfs"
override_enable_network_isolation = random.choice([True, False])


config_role = "this is your security compliant role"
config_enable_network_isolation = random.choice([True, False])


metadata_enable_network_isolation = random.choice([True, False])


def config_value_impl(sagemaker_session: Session, config_path: str, sagemaker_config: dict):
    if config_path == MODEL_EXECUTION_ROLE_ARN_PATH:
        return config_role

    if config_path == MODEL_ENABLE_NETWORK_ISOLATION_PATH:
        return config_enable_network_isolation

    raise AssertionError(f"Bad config path: {config_path}")


class IntelligentDefaultsModelTest(unittest.TestCase):

    execution_role = "fake role! do not use!"
    region = "us-west-2"
    sagemaker_session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION

    @mock.patch("sagemaker.jumpstart.model.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.factory.model._retrieve_model_init_kwargs")
    @mock.patch("sagemaker.utils.get_sagemaker_config_value")
    @mock.patch("sagemaker.jumpstart.utils.resolve_value_from_config")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_without_arg_overwrites_without_kwarg_collisions_with_config(
        self,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_resolve_value_from_config: mock.Mock,
        mock_get_sagemaker_config_value: mock.Mock,
        mock_retrieve_kwargs: mock.Mock,
        mock_model_init: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS
        model_id, _ = "js-trainable-model", "*"

        mock_retrieve_kwargs.return_value = {}

        mock_get_sagemaker_config_value.side_effect = config_value_impl

        mock_resolve_value_from_config.side_effect = resolve_value_from_config
        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        JumpStartModel(
            model_id=model_id,
        )

        self.assertEquals(mock_get_sagemaker_config_value.call_count, 1)

        self.assertEquals(mock_model_init.call_args[1].get("role"), config_role)

        assert "enable_network_isolation" not in mock_model_init.call_args[1]

    @mock.patch("sagemaker.jumpstart.model.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.factory.model._retrieve_model_init_kwargs")
    @mock.patch("sagemaker.utils.get_sagemaker_config_value")
    @mock.patch("sagemaker.jumpstart.utils.resolve_value_from_config")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_all_arg_overwrites_without_kwarg_collisions_with_config(
        self,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_resolve_value_from_config: mock.Mock,
        mock_get_sagemaker_config_value: mock.Mock,
        mock_retrieve_kwargs: mock.Mock,
        mock_model_init: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "js-trainable-model", "*"

        mock_retrieve_kwargs.return_value = {}

        mock_get_sagemaker_config_value.side_effect = config_value_impl

        mock_resolve_value_from_config.side_effect = resolve_value_from_config
        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        JumpStartModel(
            model_id=model_id,
            enable_network_isolation=override_enable_network_isolation,
            role=override_role,
        )

        self.assertEquals(mock_get_sagemaker_config_value.call_count, 1)

        self.assertEquals(mock_model_init.call_args[1].get("role"), override_role)
        self.assertEquals(
            mock_model_init.call_args[1].get("enable_network_isolation"),
            override_enable_network_isolation,
        )

    @mock.patch("sagemaker.jumpstart.model.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.factory.model._retrieve_model_init_kwargs")
    @mock.patch("sagemaker.utils.get_sagemaker_config_value")
    @mock.patch("sagemaker.jumpstart.utils.resolve_value_from_config")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_without_arg_overwrites_all_kwarg_collisions_with_config(
        self,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_resolve_value_from_config: mock.Mock,
        mock_get_sagemaker_config_value: mock.Mock,
        mock_retrieve_kwargs: mock.Mock,
        mock_model_init: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "js-trainable-model", "*"

        mock_retrieve_kwargs.return_value = {
            "enable_network_isolation": metadata_enable_network_isolation,
        }

        mock_get_sagemaker_config_value.side_effect = config_value_impl

        mock_resolve_value_from_config.side_effect = resolve_value_from_config
        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        JumpStartModel(
            model_id=model_id,
        )

        self.assertEquals(mock_get_sagemaker_config_value.call_count, 2)

        self.assertEquals(mock_model_init.call_args[1].get("role"), config_role)
        self.assertEquals(
            mock_model_init.call_args[1].get("enable_network_isolation"),
            config_enable_network_isolation,
        )

    @mock.patch("sagemaker.jumpstart.model.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.factory.model._retrieve_model_init_kwargs")
    @mock.patch("sagemaker.utils.get_sagemaker_config_value")
    @mock.patch("sagemaker.jumpstart.utils.resolve_value_from_config")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_with_arg_overwrites_all_kwarg_collisions_with_config(
        self,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_resolve_value_from_config: mock.Mock,
        mock_get_sagemaker_config_value: mock.Mock,
        mock_retrieve_kwargs: mock.Mock,
        mock_model_init: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "js-trainable-model", "*"

        mock_retrieve_kwargs.return_value = {
            "enable_network_isolation": metadata_enable_network_isolation,
        }

        mock_get_sagemaker_config_value.side_effect = config_value_impl

        mock_resolve_value_from_config.side_effect = resolve_value_from_config
        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        JumpStartModel(
            model_id=model_id,
            role=override_role,
            enable_network_isolation=override_enable_network_isolation,
        )

        self.assertEquals(mock_get_sagemaker_config_value.call_count, 1)

        self.assertEquals(mock_model_init.call_args[1].get("role"), override_role)
        self.assertEquals(
            mock_model_init.call_args[1].get("enable_network_isolation"),
            override_enable_network_isolation,
        )

    @mock.patch("sagemaker.jumpstart.model.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.factory.model._retrieve_model_init_kwargs")
    @mock.patch("sagemaker.utils.get_sagemaker_config_value")
    @mock.patch("sagemaker.jumpstart.utils.resolve_value_from_config")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_without_arg_overwrites_all_kwarg_collisions_without_config(
        self,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_resolve_value_from_config: mock.Mock,
        mock_get_sagemaker_config_value: mock.Mock,
        mock_retrieve_kwargs: mock.Mock,
        mock_model_init: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "js-trainable-model", "*"

        mock_retrieve_kwargs.return_value = {
            "enable_network_isolation": metadata_enable_network_isolation,
        }

        mock_get_sagemaker_config_value.return_value = None

        mock_resolve_value_from_config.side_effect = resolve_value_from_config
        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        JumpStartModel(
            model_id=model_id,
        )

        self.assertEquals(mock_get_sagemaker_config_value.call_count, 2)

        self.assertEquals(mock_model_init.call_args[1].get("role"), execution_role)
        self.assertEquals(
            mock_model_init.call_args[1].get("enable_network_isolation"),
            metadata_enable_network_isolation,
        )

    @mock.patch("sagemaker.jumpstart.model.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.factory.model._retrieve_model_init_kwargs")
    @mock.patch("sagemaker.utils.get_sagemaker_config_value")
    @mock.patch("sagemaker.jumpstart.utils.resolve_value_from_config")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_with_arg_overwrites_all_kwarg_collisions_without_config(
        self,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_resolve_value_from_config: mock.Mock,
        mock_get_sagemaker_config_value: mock.Mock,
        mock_retrieve_kwargs: mock.Mock,
        mock_model_init: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):
        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "js-trainable-model", "*"

        mock_retrieve_kwargs.return_value = {
            "enable_network_isolation": metadata_enable_network_isolation,
        }

        mock_get_sagemaker_config_value.return_value = None

        mock_resolve_value_from_config.side_effect = resolve_value_from_config
        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        JumpStartModel(
            model_id=model_id,
            role=override_role,
            enable_network_isolation=override_enable_network_isolation,
        )

        self.assertEquals(mock_get_sagemaker_config_value.call_count, 1)

        self.assertEquals(mock_model_init.call_args[1].get("role"), override_role)
        self.assertEquals(
            mock_model_init.call_args[1].get("enable_network_isolation"),
            override_enable_network_isolation,
        )

    @mock.patch("sagemaker.jumpstart.model.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.factory.model._retrieve_model_init_kwargs")
    @mock.patch("sagemaker.utils.get_sagemaker_config_value")
    @mock.patch("sagemaker.jumpstart.utils.resolve_value_from_config")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_without_arg_overwrites_without_kwarg_collisions_without_config(
        self,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_resolve_value_from_config: mock.Mock,
        mock_get_sagemaker_config_value: mock.Mock,
        mock_retrieve_kwargs: mock.Mock,
        mock_model_init: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "js-trainable-model", "*"

        mock_retrieve_kwargs.return_value = {}

        mock_get_sagemaker_config_value.return_value = None

        mock_resolve_value_from_config.side_effect = resolve_value_from_config
        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        JumpStartModel(
            model_id=model_id,
        )

        self.assertEquals(mock_get_sagemaker_config_value.call_count, 1)

        self.assertEquals(mock_model_init.call_args[1].get("role"), execution_role)
        assert "enable_network_isolation" not in mock_model_init.call_args[1]

    @mock.patch("sagemaker.jumpstart.model.validate_model_id_and_get_type")
    @mock.patch("sagemaker.jumpstart.model.Model.__init__")
    @mock.patch("sagemaker.jumpstart.factory.model._retrieve_model_init_kwargs")
    @mock.patch("sagemaker.utils.get_sagemaker_config_value")
    @mock.patch("sagemaker.jumpstart.utils.resolve_value_from_config")
    @mock.patch("sagemaker.jumpstart.factory.model.Session")
    @mock.patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    @mock.patch("sagemaker.jumpstart.factory.model.JUMPSTART_DEFAULT_REGION_NAME", region)
    def test_with_arg_overwrites_without_kwarg_collisions_without_config(
        self,
        mock_get_model_specs: mock.Mock,
        mock_session: mock.Mock,
        mock_resolve_value_from_config: mock.Mock,
        mock_get_sagemaker_config_value: mock.Mock,
        mock_retrieve_kwargs: mock.Mock,
        mock_model_init: mock.Mock,
        mock_validate_model_id_and_get_type: mock.Mock,
    ):

        mock_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

        model_id, _ = "js-trainable-model", "*"

        mock_retrieve_kwargs.return_value = {}

        mock_get_sagemaker_config_value.return_value = None

        mock_resolve_value_from_config.side_effect = resolve_value_from_config
        mock_get_model_specs.side_effect = get_special_model_spec

        mock_session.return_value = sagemaker_session

        JumpStartModel(
            model_id=model_id,
            role=override_role,
            enable_network_isolation=override_enable_network_isolation,
        )

        self.assertEquals(mock_get_sagemaker_config_value.call_count, 1)

        self.assertEquals(mock_model_init.call_args[1].get("role"), override_role)
        self.assertEquals(
            mock_model_init.call_args[1].get("enable_network_isolation"),
            override_enable_network_isolation,
        )
