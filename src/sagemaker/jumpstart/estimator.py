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


from typing import Dict, List, Optional

from sagemaker.estimator import Estimator

from sagemaker.jumpstart.factory.estimator import get_deploy_kwargs, get_fit_kwargs, get_init_kwargs


from sagemaker.predictor import Predictor


class JumpStartEstimator(Estimator):
    """JumpStartEstimator class.

    This class sets defaults based on the model id and version.
    """

    def __init__(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        instance_type: Optional[str] = None,
        instance_count: Optional[int] = None,
        region: Optional[str] = None,
        image_uri: Optional[str] = None,
        model_uri: Optional[str] = None,
        source_dir: Optional[str] = None,
        entry_point: Optional[str] = None,
        hyperparameters: Optional[dict] = None,
        metric_definitions: Optional[List[dict]] = None,
        **kwargs,
    ):
        estimator_init_kwargs = get_init_kwargs(
            model_id=model_id,
            model_version=model_version,
            instance_type=instance_type,
            instance_count=instance_count,
            region=region,
            image_uri=image_uri,
            model_uri=model_uri,
            source_dir=source_dir,
            entry_point=entry_point,
            hyperparameters=hyperparameters,
            metric_definitions=metric_definitions,
            kwargs=kwargs,
        )

        self.model_id = estimator_init_kwargs.model_id
        self.model_version = estimator_init_kwargs.model_version
        self.instance_type = estimator_init_kwargs.instance_type
        self.instance_count = estimator_init_kwargs.instance_count
        self.region = estimator_init_kwargs.region

        super(JumpStartEstimator, self).__init__(**estimator_init_kwargs.to_kwargs_dict())

    def fit(self, *largs, **kwargs) -> None:

        estimator_fit_kwargs = get_fit_kwargs(
            model_id=self.model_id,
            model_version=self.model_version,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            region=self.region,
            kwargs=kwargs,
        )

        return super(JumpStartEstimator, self).fit(*largs, **estimator_fit_kwargs.to_kwargs_dict())

    def deploy(
        self,
        image_uri: Optional[str] = None,
        source_dir: Optional[str] = None,
        entry_point: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        predictor_cls: Optional[Predictor] = None,
        initial_instance_count: Optional[int] = None,
        instance_type: Optional[str] = None,
        **kwargs,
    ) -> None:

        estimator_deploy_kwargs = get_deploy_kwargs(
            model_id=self.model_id,
            model_version=self.model_version,
            instance_type=instance_type,
            initial_instance_count=initial_instance_count,
            region=self.region,
            image_uri=image_uri,
            source_dir=source_dir,
            entry_point=entry_point,
            env=env,
            predictor_cls=predictor_cls,
            kwargs=kwargs,
        )

        return super(JumpStartEstimator, self).deploy(**estimator_deploy_kwargs.to_kwargs_dict())
