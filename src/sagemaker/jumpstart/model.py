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
"""This module stores JumpStart implementation of Model class."""

from __future__ import absolute_import

from copy import deepcopy
from typing import Any, Dict, Optional
from sagemaker import environment_variables, image_uris, instance_types, model_uris, script_uris
from sagemaker.jumpstart.artifacts import _model_supports_prepacked_inference
from sagemaker.jumpstart.constants import (
    INFERENCE_ENTRY_POINT_SCRIPT_NAME,
    JUMPSTART_DEFAULT_REGION_NAME,
)
from sagemaker.jumpstart.utils import update_dict_if_key_not_present
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.session import Session


class JumpStartModel(Model):
    """JumpStartModel class.

    This class sets defaults based on the model id and version.
    """

    def __init__(
        self,
        ############## ▼ args unique to JumpStartModel ▼ ##############
        model_id: str,
        model_version: Optional[str] = "*",
        instance_type: Optional[str] = None,
        region: Optional[str] = JUMPSTART_DEFAULT_REGION_NAME,
        ############## ▲ args unique to JumpStartModel ▲ ##############
        ############## ▼ args passed to base Model class ▼ ##############
        image_uri: Optional[str] = None,
        model_uri: Optional[str] = None,
        script_uri: Optional[str] = None,
        entry_point: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        predictor_cls: Predictor = "JumpStartPredictor",
        **kwargs,
        ############## ▲ args passed to base Model class ▲ ##############
    ):

        self.model_id = model_id
        self.model_version = model_version
        self.kwargs_for_base_model_class = deepcopy(kwargs)

        self.instance_type = instance_type or instance_types.retrieve_default(
            region=region, model_id=model_id, model_version=model_version
        )

        self.kwargs_for_base_model_class["model_id"] = model_id
        self.kwargs_for_base_model_class["model_version"] = model_version

        self.kwargs_for_base_model_class["predictor_cls"] = predictor_cls

        self.kwargs_for_base_model_class["image_uri"] = image_uri or image_uris.retrieve(
            region=None,
            framework=None,
            image_scope="inference",
            model_id=model_id,
            model_version=model_version,
            instance_type=self.instance_type,
        )

        self.kwargs_for_base_model_class["model_uri"] = model_uri or model_uris.retrieve(
            model_scope="inference",
            model_id=model_id,
            model_version=model_version,
        )

        if not _model_supports_prepacked_inference(
            model_id=model_id, model_version=model_version, region=region
        ):
            self.kwargs_for_base_model_class["script_uri"] = script_uri or script_uris.retrieve(
                script_scope="inference",
                model_id=model_id,
                model_version=model_version,
            )
            self.kwargs_for_base_model_class["entry_point"] = (
                entry_point or INFERENCE_ENTRY_POINT_SCRIPT_NAME
            )

        extra_env_vars = environment_variables.retrieve_default(
            region=region, model_id=model_id, model_version=model_version
        )

        curr_env_vars = env or {}
        new_env_vars = deepcopy(curr_env_vars)

        for key, value in extra_env_vars:
            new_env_vars = update_dict_if_key_not_present(
                new_env_vars,
                key,
                value,
            )

        if new_env_vars == {}:
            new_env_vars = None

        self.kwargs_for_base_model_class["env"] = new_env_vars

        # model_kwargs_to_add = _retrieve_kwargs(model_id=model_id, model_version=model_version, region=region)
        model_kwargs_to_add = {}

        new_kwargs_for_base_model_class = deepcopy(self.kwargs_for_base_model_class)
        for key, value in model_kwargs_to_add:
            new_kwargs_for_base_model_class = update_dict_if_key_not_present(
                new_kwargs_for_base_model_class,
                key,
                value,
            )

        self.kwargs_for_base_model_class = new_kwargs_for_base_model_class

        super(Model, self).__init__(**self.kwargs_for_base_model_class)

    @staticmethod
    def _update_dict_if_key_not_present(
        dict_to_update: dict, key_to_add: Any, value_to_add: Any
    ) -> dict:
        if key_to_add not in dict_to_update:
            dict_to_update[key_to_add] = value_to_add

        return dict_to_update

    def deploy(self, **kwargs) -> callable[str, Session]:

        kwargs = update_dict_if_key_not_present(kwargs, "initial_instance_count", 1)
        kwargs = update_dict_if_key_not_present(kwargs, "instance_type", self.instance_type)

        return super(Model, self).deploy(**kwargs)
