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
"""ImageSpec class module."""
from __future__ import absolute_import

from typing import Optional

from sagemaker import image_uris, Session
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.training_compiler.config import TrainingCompilerConfig


class ImageSpec:
    """ImageSpec class to get image URI for a specific framework version."""

    def __init__(
        self,
        framework_name: str,
        version: str,
        image_scope: Optional[str] = None,
        instance_type: Optional[str] = None,
        py_version: Optional[str] = None,
        region: Optional[str] = "us-west-2",
        accelerator_type: Optional[str] = None,
        container_version: Optional[str] = None,
        distribution: Optional[dict] = None,
        base_framework_version: Optional[str] = None,
        training_compiler_config: Optional[TrainingCompilerConfig] = None,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        hub_arn: Optional[str] = None,
        tolerate_vulnerable_model: Optional[bool] = False,
        tolerate_deprecated_model: Optional[bool] = False,
        sdk_version: Optional[str] = None,
        inference_tool: Optional[str] = None,
        serverless_inference_config: Optional[ServerlessInferenceConfig] = None,
        config_name: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
    ):
        self.framework_name = framework_name
        self.version = version
        self.image_scope = image_scope
        self.instance_type = instance_type
        self.py_version = py_version
        self.region = region
        self.accelerator_type = accelerator_type
        self.container_version = container_version
        self.distribution = distribution
        self.base_framework_version = base_framework_version
        self.training_compiler_config = training_compiler_config
        self.model_id = model_id
        self.model_version = model_version
        self.hub_arn = hub_arn
        self.tolerate_vulnerable_model = tolerate_vulnerable_model
        self.tolerate_deprecated_model = tolerate_deprecated_model
        self.sdk_version = sdk_version
        self.inference_tool = inference_tool
        self.serverless_inference_config = serverless_inference_config
        self.config_name = config_name
        self.sagemaker_session = sagemaker_session

    def get_image_uri(
        self, image_scope: Optional[str] = None, instance_type: Optional[str] = None
    ) -> str:
        """Get image URI for a specific framework version."""

        self.image_scope = image_scope or self.image_scope
        self.instance_type = instance_type or self.instance_type
        return image_uris.retrieve(
            framework=self.framework_name,
            image_scope=self.image_scope,
            instance_type=self.instance_type,
            py_version=self.py_version,
            region=self.region,
            version=self.version,
            accelerator_type=self.accelerator_type,
            container_version=self.container_version,
            distribution=self.distribution,
            base_framework_version=self.base_framework_version,
            training_compiler_config=self.training_compiler_config,
            model_id=self.model_id,
            model_version=self.model_version,
            hub_arn=self.hub_arn,
            tolerate_vulnerable_model=self.tolerate_vulnerable_model,
            tolerate_deprecated_model=self.tolerate_deprecated_model,
            sdk_version=self.sdk_version,
            inference_tool=self.inference_tool,
            serverless_inference_config=self.serverless_inference_config,
            config_name=self.config_name,
            sagemaker_session=self.sagemaker_session,
        )
