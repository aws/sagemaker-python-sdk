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

from typing import Dict, Optional
from sagemaker.jumpstart.factory.model import get_deploy_kwargs, get_init_kwargs
from sagemaker.model import Model
from sagemaker.predictor import Predictor, PredictorBase


class JumpStartModel(Model):
    """JumpStartModel class.

    This class sets defaults based on the model id and version.
    """

    def __init__(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        instance_type: Optional[str] = None,
        region: Optional[str] = None,
        image_uri: Optional[str] = None,
        model_data: Optional[str] = None,
        source_dir: Optional[str] = None,
        entry_point: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        predictor_cls: Optional[Predictor] = None,
        **kwargs,
    ):

        model_init_kwargs = get_init_kwargs(
            model_id=model_id,
            model_from_estimator=False,
            model_version=model_version,
            instance_type=instance_type,
            region=region,
            image_uri=image_uri,
            model_data=model_data,
            source_dir=source_dir,
            entry_point=entry_point,
            env=env,
            predictor_cls=predictor_cls,
            kwargs=kwargs,
        )

        self.model_id = model_init_kwargs.model_id
        self.model_version = model_init_kwargs.model_version
        self.instance_type = model_init_kwargs.instance_type
        self.region = model_init_kwargs.region

        super(JumpStartModel, self).__init__(**model_init_kwargs.to_kwargs_dict())

    def deploy(
        self,
        initial_instance_count: Optional[int] = None,
        instance_type: Optional[str] = None,
        **kwargs,
    ) -> PredictorBase:

        deploy_kwargs = get_deploy_kwargs(
            model_id=self.model_id,
            model_version=self.model_version,
            region=self.region,
            initial_instance_count=initial_instance_count,
            instance_type=instance_type or self.instance_type,
            kwargs=kwargs,
        )
        return super(JumpStartModel, self).deploy(**deploy_kwargs.to_kwargs_dict())
