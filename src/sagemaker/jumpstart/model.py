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
import logging

from typing import Dict, List, Optional, Union
from sagemaker.async_inference.async_inference_config import AsyncInferenceConfig
from sagemaker.base_deserializers import BaseDeserializer
from sagemaker.base_serializers import BaseSerializer
from sagemaker.explainer.explainer_config import ExplainerConfig
from sagemaker.jumpstart.factory.model import get_deploy_kwargs, get_init_kwargs
from sagemaker.model import Model
from sagemaker.model_monitor.data_capture_config import DataCaptureConfig
from sagemaker.predictor import PredictorBase
from sagemaker.serverless.serverless_inference_config import ServerlessInferenceConfig
from sagemaker.session import Session
from sagemaker.workflow.entities import PipelineVariable

logger = logging.getLogger(__name__)


class JumpStartModel(Model):
    """JumpStartModel class.

    This class sets defaults based on the model id and version.
    """

    def __init__(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        region: Optional[str] = None,
        instance_type: Optional[str] = None,
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        model_data: Optional[Union[str, PipelineVariable]] = None,
        role: Optional[str] = None,
        predictor_cls: Optional[callable] = None,
        env: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        name: Optional[str] = None,
        vpc_config: Optional[Dict[str, List[Union[str, PipelineVariable]]]] = None,
        sagemaker_session: Optional[Session] = None,
        enable_network_isolation: Union[bool, PipelineVariable] = None,
        model_kms_key: Optional[str] = None,
        image_config: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        source_dir: Optional[str] = None,
        code_location: Optional[str] = None,
        entry_point: Optional[str] = None,
        container_log_level: Optional[Union[int, PipelineVariable]] = None,
        dependencies: Optional[List[str]] = None,
        git_config: Optional[Dict[str, str]] = None,
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
            role=role,
            name=name,
            vpc_config=vpc_config,
            sagemaker_session=sagemaker_session,
            enable_network_isolation=enable_network_isolation,
            model_kms_key=model_kms_key,
            image_config=image_config,
            code_location=code_location,
            container_log_level=container_log_level,
            dependencies=dependencies,
            git_config=git_config,
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
        serializer: Optional[BaseSerializer] = None,
        deserializer: Optional[BaseDeserializer] = None,
        accelerator_type: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        tags: List[Dict[str, str]] = None,
        kms_key: Optional[str] = None,
        wait: Optional[bool] = None,
        data_capture_config: Optional[DataCaptureConfig] = None,
        async_inference_config: Optional[AsyncInferenceConfig] = None,
        serverless_inference_config: Optional[ServerlessInferenceConfig] = None,
        volume_size: Optional[int] = None,
        model_data_download_timeout: Optional[int] = None,
        container_startup_health_check_timeout: Optional[int] = None,
        inference_recommendation_id: Optional[str] = None,
        explainer_config: Optional[ExplainerConfig] = None,
    ) -> PredictorBase:
        """Creates endpoint by calling base Model class `deploy` method."""

        deploy_kwargs = get_deploy_kwargs(
            model_id=self.model_id,
            model_version=self.model_version,
            region=self.region,
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            serializer=serializer,
            deserializer=deserializer,
            accelerator_type=accelerator_type,
            endpoint_name=endpoint_name,
            tags=tags,
            kms_key=kms_key,
            wait=wait,
            data_capture_config=data_capture_config,
            async_inference_config=async_inference_config,
            serverless_inference_config=serverless_inference_config,
            volume_size=volume_size,
            model_data_download_timeout=model_data_download_timeout,
            container_startup_health_check_timeout=container_startup_health_check_timeout,
            inference_recommendation_id=inference_recommendation_id,
            explainer_config=explainer_config,
        )

        logger.info(  # pylint: disable=W1203
            f"Creating SageMaker Hosting endpoint for {self.model_id}. "
            f"Provisioning {deploy_kwargs.instance_type} instance in {deploy_kwargs.region}."
        )
        return super(JumpStartModel, self).deploy(**deploy_kwargs.to_kwargs_dict())
