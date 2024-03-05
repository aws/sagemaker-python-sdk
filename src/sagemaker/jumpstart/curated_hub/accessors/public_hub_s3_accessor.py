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
from sagemaker.jumpstart.curated_hub.utils import (
    get_model_framework,
)
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.jumpstart.types import JumpStartModelSpecs
from sagemaker.jumpstart.utils import get_jumpstart_content_bucket
from sagemaker.jumpstart.curated_hub.accessors.s3_object_reference import (
    S3ObjectLocation,
    create_s3_object_reference_from_uri,
)
from sagemaker.jumpstart.curated_hub.accessors.model_dependency_s3_accessor import (
    ModelDependencyS3Accessor,
)

class PublicHubS3Accessor(ModelDependencyS3Accessor):
    """Helper class to access Public Hub s3 bucket"""

    def __init__(self, region: str, studio_metadata_map: Dict[str, Dict[str, Any]]):
        self._region = region
        self._bucket = get_jumpstart_content_bucket(region)
        self._studio_metadata_map = studio_metadata_map  # Necessary for SDK - Studio metadata drift

    def get_bucket_name(self) -> str:
        """Retrieves s3 bucket"""
        return self._bucket
    
    def get_uncompressed_inference_artifact_s3_reference(self, model_specs: JumpStartModelSpecs
    ) -> bool:
        """Retrieves s3 reference for uncompressed model inference artifact."""
        return self.get_inference_artifact_s3_reference(model_specs)

    def get_inference_artifact_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectLocation:
        """Retrieves s3 reference for model inference artifact"""
        return create_s3_object_reference_from_uri(
            self._jumpstart_artifact_s3_uri(JumpStartScriptScope.INFERENCE, model_specs)
        )
    
    def get_uncompressed_training_artifact_s3_reference(self, model_specs: JumpStartModelSpecs
    ) -> bool:
        """Retrieves s3 reference for uncompressed model training artifact."""
        return self.get_training_artifact_s3_reference(model_specs)

    def get_training_artifact_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectLocation:
        """Retrieves s3 reference for model training artifact"""
        return create_s3_object_reference_from_uri(
            self._jumpstart_artifact_s3_uri(JumpStartScriptScope.TRAINING, model_specs)
        )

    def get_inference_script_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectLocation:
        """Retrieves s3 reference for model inference script"""
        return create_s3_object_reference_from_uri(
            self._jumpstart_script_s3_uri(JumpStartScriptScope.INFERENCE, model_specs)
        )

    def get_training_script_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectLocation:
        """Retrieves s3 reference for model training script"""
        return create_s3_object_reference_from_uri(
            self._jumpstart_script_s3_uri(JumpStartScriptScope.TRAINING, model_specs)
        )

    def get_default_training_dataset_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectLocation:
        """Retrieves s3 reference for s3 directory containing model training datasets"""
        return S3ObjectLocation(
            self.get_bucket_name(), self._get_training_dataset_prefix(model_specs)
        )

    def _get_training_dataset_prefix(self, model_specs: JumpStartModelSpecs) -> str:
        """Retrieves training dataset location"""
        studio_model_metadata = self._studio_metadata_map[model_specs.model_id]
        return studio_model_metadata["defaultDataKey"]

    def get_demo_notebook_s3_reference(self, model_specs: JumpStartModelSpecs) -> S3ObjectLocation:
        """Retrieves s3 reference for model demo jupyter notebook"""
        framework = get_model_framework(model_specs)
        key = f"{framework}-notebooks/{model_specs.model_id}-inference.ipynb"
        return S3ObjectLocation(self.get_bucket_name(), key)

    def get_markdown_s3_reference(self, model_specs: JumpStartModelSpecs) -> S3ObjectLocation:
        """Retrieves s3 reference for model markdown"""
        framework = get_model_framework(model_specs)
        key = f"{framework}-metadata/{model_specs.model_id}.md"
        return S3ObjectLocation(self.get_bucket_name(), key)

    def _jumpstart_script_s3_uri(self, model_scope: str, model_specs: JumpStartModelSpecs) -> str:
        """Retrieves JumpStart script s3 location"""
        return script_uris.retrieve(
            region=self._region,
            model_id=model_specs.model_id,
            model_version=model_specs.version,
            script_scope=model_scope,
            tolerate_vulnerable_model=True,
            tolerate_deprecated_model=True,
        )

    def _jumpstart_artifact_s3_uri(self, model_scope: str, model_specs: JumpStartModelSpecs) -> str:
        """Retrieves JumpStart artifact s3 location"""
        return model_uris.retrieve(
            region=self._region,
            model_id=model_specs.model_id,
            model_version=model_specs.version,
            model_scope=model_scope,
            tolerate_vulnerable_model=True,
            tolerate_deprecated_model=True,
        )
