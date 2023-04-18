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
"""This module stores JumpStart implementation of Estimator class."""
from __future__ import absolute_import


from copy import deepcopy
from typing import Any, Optional
from sagemaker import (
    hyperparameters,
    image_uris,
    instance_types,
    metric_definitions,
    model_uris,
    script_uris,
)
from sagemaker.jumpstart.constants import JUMPSTART_DEFAULT_REGION_NAME
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.jumpstart.utils import update_dict_if_key_not_present
from sagemaker.model import Estimator


class JumpStartEstimator(Estimator):
    """JumpStartEstimator class.

    This class sets defaults based on the model id and version.
    """

    def __init__(
        self,
        model_id: str,
        model_version: Optional[str] = "*",
        region: Optional[str] = JUMPSTART_DEFAULT_REGION_NAME,
        kwargs_for_base_estimator_class: dict = {},
    ):
        self.model_id = model_id
        self.model_version = model_version
        self.kwargs_for_base_estimator_class = deepcopy(kwargs_for_base_estimator_class)

        self.kwargs_for_base_estimator_class = update_dict_if_key_not_present(
            self.kwargs_for_base_estimator_class,
            "image_uri",
            image_uris.retrieve(
                region=None,
                framework=None,
                image_scope="training",
                model_id=model_id,
                model_version=model_version,
                instance_type=self.instance_type,
            ),
        )

        self.kwargs_for_base_estimator_class = update_dict_if_key_not_present(
            self.kwargs_for_base_estimator_class,
            "model_uri",
            model_uris.retrieve(
                script_scope=JumpStartScriptScope.TRAINING,
                model_id=model_id,
                model_version=model_version,
            ),
        )

        self.kwargs_for_base_estimator_class = update_dict_if_key_not_present(
            self.kwargs_for_base_estimator_class,
            "script_uri",
            script_uris.retrieve(
                script_scope=JumpStartScriptScope.TRAINING,
                model_id=model_id,
                model_version=model_version,
            ),
        )

        default_hyperparameters = hyperparameters.retrieve_default(
            region=region, model_id=model_id, model_version=model_version
        )

        curr_hyperparameters = self.kwargs_for_base_estimator_class.get("hyperparameters", {})
        new_hyperparameters = deepcopy(curr_hyperparameters)

        for key, value in default_hyperparameters:
            new_hyperparameters = update_dict_if_key_not_present(
                new_hyperparameters,
                key,
                value,
            )

        if new_hyperparameters == {}:
            new_hyperparameters = None

        self.kwargs_for_base_estimator_class["hyperparameters"] = new_hyperparameters

        default_metric_definitions = metric_definitions.retrieve_default(
            region=region, model_id=model_id, model_version=model_version
        )

        curr_metric_definitions = self.kwargs_for_base_estimator_class.get("metric_definitions", [])
        new_metric_definitions = deepcopy(curr_metric_definitions)

        for metric_definition in default_metric_definitions:
            if metric_definition["Name"] not in [
                definition["Name"] for definition in new_metric_definitions
            ]:
                new_metric_definitions.append(metric_definition)

        if new_metric_definitions == []:
            new_metric_definitions = None

        self.kwargs_for_base_estimator_class["metric_definitions"] = new_metric_definitions

        # estimator_kwargs_to_add = _retrieve_kwargs(model_id=model_id, model_version=model_version, region=region)
        estimator_kwargs_to_add = {}

        new_kwargs_for_base_estimator_class = deepcopy(self.kwargs_for_base_estimator_class)
        for key, value in estimator_kwargs_to_add:
            new_kwargs_for_base_estimator_class = update_dict_if_key_not_present(
                new_kwargs_for_base_estimator_class,
                key,
                value,
            )

        self.kwargs_for_base_estimator_class = new_kwargs_for_base_estimator_class

        self.kwargs_for_base_estimator_class["model_id"] = model_id
        self.kwargs_for_base_estimator_class["model_version"] = model_version

        # self.kwargs_for_base_estimator_class = update_dict_if_key_not_present(
        #     self.kwargs_for_base_estimator_class,
        #     "predictor_cls",
        #     JumpStartPredictor,
        # )

        self.kwargs_for_base_estimator_class = update_dict_if_key_not_present(
            self.kwargs_for_base_estimator_class, "instance_count", 1
        )
        self.kwargs_for_base_estimator_class = update_dict_if_key_not_present(
            self.kwargs_for_base_estimator_class,
            "instance_type",
            instance_types.retrieve_default(
                region=region, model_id=model_id, model_version=model_version
            ),
        )

        super(Estimator, self).__init__(**self.kwargs_for_base_estimator_class)

    @staticmethod
    def _update_dict_if_key_not_present(
        dict_to_update: dict, key_to_add: Any, value_to_add: Any
    ) -> dict:
        if key_to_add not in dict_to_update:
            dict_to_update[key_to_add] = value_to_add

        return dict_to_update

    def fit(self, *largs, **kwargs) -> None:

        return super(Estimator, self).fit(*largs, **kwargs)
