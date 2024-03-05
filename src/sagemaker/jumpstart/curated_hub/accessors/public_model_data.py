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
"""This module accessors for the SageMaker JumpStart Public Hub."""
from __future__ import absolute_import
from typing import Dict, Any
from sagemaker import model_uris, script_uris
from sagemaker.jumpstart.curated_hub.accessors.fileinfo import HubContentDependencyType
from sagemaker.jumpstart.curated_hub.utils import (
    get_model_framework,
)
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.jumpstart.types import JumpStartModelSpecs
from sagemaker.jumpstart.utils import get_jumpstart_content_bucket
from sagemaker.jumpstart.curated_hub.accessors.objectlocation import (
    S3ObjectLocation,
    create_s3_object_reference_from_uri,
)


class PublicModelDataAccessor:
    """Accessor class for JumpStart model data s3 locations."""

    def __init__(
        self,
        region: str,
        model_specs: JumpStartModelSpecs,
        studio_specs: Dict[str, Dict[str, Any]],
    ):
        self._region = region
        self._bucket = get_jumpstart_content_bucket(region)
        self.model_specs = model_specs
        self.studio_specs = studio_specs  # Necessary for SDK - Studio metadata drift

    def get_s3_reference(self, dependency_type: HubContentDependencyType):
        """Retrieves S3 reference given a HubContentDependencyType."""
        return getattr(self, dependency_type.value)

    @property
    def inference_artifact_s3_reference(self):
        """Retrieves s3 reference for model inference artifact"""
        return create_s3_object_reference_from_uri(
            self._jumpstart_artifact_s3_uri(JumpStartScriptScope.INFERENCE)
        )

    @property
    def training_artifact_s3_reference(self):
        """Retrieves s3 reference for model training artifact"""
        return create_s3_object_reference_from_uri(
            self._jumpstart_artifact_s3_uri(JumpStartScriptScope.TRAINING)
        )

    @property
    def inference_script_s3_reference(self):
        """Retrieves s3 reference for model inference script"""
        return create_s3_object_reference_from_uri(
            self._jumpstart_script_s3_uri(JumpStartScriptScope.INFERENCE)
        )

    @property
    def training_script_s3_reference(self):
        """Retrieves s3 reference for model training script"""
        return create_s3_object_reference_from_uri(
            self._jumpstart_script_s3_uri(JumpStartScriptScope.TRAINING)
        )

    @property
    def default_training_dataset_s3_reference(self):
        """Retrieves s3 reference for s3 directory containing model training datasets"""
        return S3ObjectLocation(self._get_bucket_name(), self.__get_training_dataset_prefix())

    @property
    def demo_notebook_s3_reference(self):
        """Retrieves s3 reference for model demo jupyter notebook"""
        framework = get_model_framework(self.model_specs)
        key = f"{framework}-notebooks/{self.model_specs.model_id}-inference.ipynb"
        return S3ObjectLocation(self._get_bucket_name(), key)
    
    @property
    def markdown_s3_reference(self):
        """Retrieves s3 reference for model markdown"""
        framework = get_model_framework(self.model_specs)
        key = f"{framework}-metadata/{self.model_specs.model_id}.md"
        return S3ObjectLocation(self._get_bucket_name(), key)
    def _get_bucket_name(self) -> str:
        """Retrieves s3 bucket"""
        return self._bucket

    def __get_training_dataset_prefix(self) -> str:
        """Retrieves training dataset location"""
        return self.studio_specs["defaultDataKey"]

    def _jumpstart_script_s3_uri(self, model_scope: str) -> str:
        """Retrieves JumpStart script s3 location"""
        return script_uris.retrieve(
            region=self._region,
            model_id=self.model_specs.model_id,
            model_version=self.model_specs.version,
            script_scope=model_scope,
            tolerate_vulnerable_model=True,
            tolerate_deprecated_model=True,
        )

    def _jumpstart_artifact_s3_uri(self, model_scope: str) -> str:
        """Retrieves JumpStart artifact s3 location"""
        return model_uris.retrieve(
            region=self._region,
            model_id=self.model_specs.model_id,
            model_version=self.model_specs.version,
            model_scope=model_scope,
            tolerate_vulnerable_model=True,
            tolerate_deprecated_model=True,
        )
