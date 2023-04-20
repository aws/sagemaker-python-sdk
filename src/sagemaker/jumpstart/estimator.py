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


from typing import Dict, List, Optional, Union
from sagemaker import session
from sagemaker.async_inference.async_inference_config import AsyncInferenceConfig
from sagemaker.base_deserializers import BaseDeserializer
from sagemaker.base_serializers import BaseSerializer
from sagemaker.debugger.debugger import DebuggerHookConfig, RuleBase, TensorBoardOutputConfig
from sagemaker.debugger.profiler_config import ProfilerConfig

from sagemaker.estimator import Estimator
from sagemaker.explainer.explainer_config import ExplainerConfig
from sagemaker.inputs import FileSystemInput, TrainingInput
from sagemaker.instance_group import InstanceGroup

from sagemaker.jumpstart.factory.estimator import get_deploy_kwargs, get_fit_kwargs, get_init_kwargs
from sagemaker.model_monitor.data_capture_config import DataCaptureConfig


from sagemaker.serverless.serverless_inference_config import ServerlessInferenceConfig
from sagemaker.workflow.entities import PipelineVariable


class JumpStartEstimator(Estimator):
    """JumpStartEstimator class.

    This class sets defaults based on the model id and version.
    """

    def __init__(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        region: Optional[str] = None,
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        role: Optional[str] = None,
        instance_count: Optional[Union[int, PipelineVariable]] = None,
        instance_type: Optional[Union[str, PipelineVariable]] = None,
        keep_alive_period_in_seconds: Optional[Union[int, PipelineVariable]] = None,
        volume_size: Optional[Union[int, PipelineVariable]] = None,
        volume_kms_key: Optional[Union[str, PipelineVariable]] = None,
        max_run: Optional[Union[int, PipelineVariable]] = None,
        input_mode: Optional[Union[str, PipelineVariable]] = None,
        output_path: Optional[Union[str, PipelineVariable]] = None,
        output_kms_key: Optional[Union[str, PipelineVariable]] = None,
        base_job_name: Optional[str] = None,
        sagemaker_session: Optional[session.Session] = None,
        hyperparameters: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        tags: Optional[List[Dict[str, Union[str, PipelineVariable]]]] = None,
        subnets: Optional[List[Union[str, PipelineVariable]]] = None,
        security_group_ids: Optional[List[Union[str, PipelineVariable]]] = None,
        model_uri: Optional[str] = None,
        model_channel_name: Optional[Union[str, PipelineVariable]] = None,
        metric_definitions: Optional[List[Dict[str, Union[str, PipelineVariable]]]] = None,
        encrypt_inter_container_traffic: Union[bool, PipelineVariable] = None,
        use_spot_instances: Optional[Union[bool, PipelineVariable]] = None,
        max_wait: Optional[Union[int, PipelineVariable]] = None,
        checkpoint_s3_uri: Optional[Union[str, PipelineVariable]] = None,
        checkpoint_local_path: Optional[Union[str, PipelineVariable]] = None,
        enable_network_isolation: Union[bool, PipelineVariable] = None,
        rules: Optional[List[RuleBase]] = None,
        debugger_hook_config: Optional[Union[DebuggerHookConfig, bool]] = None,
        tensorboard_output_config: Optional[TensorBoardOutputConfig] = None,
        enable_sagemaker_metrics: Optional[Union[bool, PipelineVariable]] = None,
        profiler_config: Optional[ProfilerConfig] = None,
        disable_profiler: Optional[bool] = None,
        environment: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        max_retry_attempts: Optional[Union[int, PipelineVariable]] = None,
        source_dir: Optional[Union[str, PipelineVariable]] = None,
        git_config: Optional[Dict[str, str]] = None,
        container_log_level: Optional[Union[int, PipelineVariable]] = None,
        code_location: Optional[str] = None,
        entry_point: Optional[Union[str, PipelineVariable]] = None,
        dependencies: Optional[List[str]] = None,
        instance_groups: Optional[List[InstanceGroup]] = None,
        training_repository_access_mode: Optional[Union[str, PipelineVariable]] = None,
        training_repository_credentials_provider_arn: Optional[Union[str, PipelineVariable]] = None,
    ):
        estimator_init_kwargs = get_init_kwargs(
            model_id=model_id,
            model_version=model_version,
            role=role,
            region=region,
            instance_count=instance_count,
            instance_type=instance_type,
            keep_alive_period_in_seconds=keep_alive_period_in_seconds,
            volume_size=volume_size,
            volume_kms_key=volume_kms_key,
            max_run=max_run,
            input_mode=input_mode,
            output_path=output_path,
            output_kms_key=output_kms_key,
            base_job_name=base_job_name,
            sagemaker_session=sagemaker_session,
            tags=tags,
            subnets=subnets,
            security_group_ids=security_group_ids,
            model_uri=model_uri,
            model_channel_name=model_channel_name,
            metric_definitions=metric_definitions,
            encrypt_inter_container_traffic=encrypt_inter_container_traffic,
            use_spot_instances=use_spot_instances,
            max_wait=max_wait,
            checkpoint_s3_uri=checkpoint_s3_uri,
            checkpoint_local_path=checkpoint_local_path,
            rules=rules,
            debugger_hook_config=debugger_hook_config,
            tensorboard_output_config=tensorboard_output_config,
            enable_sagemaker_metrics=enable_sagemaker_metrics,
            enable_network_isolation=enable_network_isolation,
            profiler_config=profiler_config,
            disable_profiler=disable_profiler,
            environment=environment,
            max_retry_attempts=max_retry_attempts,
            source_dir=source_dir,
            git_config=git_config,
            hyperparameters=hyperparameters,
            container_log_level=container_log_level,
            code_location=code_location,
            entry_point=entry_point,
            dependencies=dependencies,
            instance_groups=instance_groups,
            training_repository_access_mode=training_repository_access_mode,
            training_repository_credentials_provider_arn=(
                training_repository_credentials_provider_arn
            ),
            image_uri=image_uri,
        )

        self.model_id = estimator_init_kwargs.model_id
        self.model_version = estimator_init_kwargs.model_version
        self.instance_type = estimator_init_kwargs.instance_type
        self.instance_count = estimator_init_kwargs.instance_count
        self.region = estimator_init_kwargs.region

        super(JumpStartEstimator, self).__init__(**estimator_init_kwargs.to_kwargs_dict())

    def fit(
        self,
        inputs: Optional[Union[str, Dict, TrainingInput, FileSystemInput]] = None,
        wait: Optional[bool] = None,
        logs: Optional[str] = None,
        job_name: Optional[str] = None,
        experiment_config: Optional[Dict[str, str]] = None,
    ) -> None:
        """Start training job by calling base Estimator class `fit` method"""

        estimator_fit_kwargs = get_fit_kwargs(
            model_id=self.model_id,
            model_version=self.model_version,
            region=self.region,
            inputs=inputs,
            wait=wait,
            logs=logs,
            job_name=job_name,
            experiment_config=experiment_config,
        )

        return super(JumpStartEstimator, self).fit(**estimator_fit_kwargs.to_kwargs_dict())

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
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        model_data: Optional[Union[str, PipelineVariable]] = None,
        role: Optional[str] = None,
        predictor_cls: Optional[callable] = None,
        env: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        name: Optional[str] = None,
        vpc_config: Optional[Dict[str, List[Union[str, PipelineVariable]]]] = None,
        sagemaker_session: Optional[session.Session] = None,
        enable_network_isolation: Union[bool, PipelineVariable] = None,
        model_kms_key: Optional[str] = None,
        image_config: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        source_dir: Optional[str] = None,
        code_location: Optional[str] = None,
        entry_point: Optional[str] = None,
        container_log_level: Optional[Union[int, PipelineVariable]] = None,
        dependencies: Optional[List[str]] = None,
        git_config: Optional[Dict[str, str]] = None,
    ) -> None:
        """Creates endpoint from training job by calling base Estimator class `deploy` method."""

        estimator_deploy_kwargs = get_deploy_kwargs(
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
            image_uri=image_uri,
            model_data=model_data,
            role=role,
            predictor_cls=predictor_cls,
            env=env,
            name=name,
            vpc_config=vpc_config,
            sagemaker_session=sagemaker_session,
            enable_network_isolation=enable_network_isolation,
            model_kms_key=model_kms_key,
            image_config=image_config,
            source_dir=source_dir,
            code_location=code_location,
            entry_point=entry_point,
            container_log_level=container_log_level,
            dependencies=dependencies,
            git_config=git_config,
        )

        return super(JumpStartEstimator, self).deploy(**estimator_deploy_kwargs.to_kwargs_dict())
