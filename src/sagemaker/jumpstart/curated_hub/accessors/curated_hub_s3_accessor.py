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
"""This module accessors for the SageMaker JumpStart Curated Hub."""
from __future__ import absolute_import
from sagemaker.jumpstart.curated_hub.utils import (
    get_model_framework,
)
from sagemaker.jumpstart.types import JumpStartModelSpecs
from sagemaker.jumpstart.curated_hub.utils import (
    get_studio_model_metadata_map_from_region,
)
from sagemaker.jumpstart.curated_hub.accessors.s3_object_reference import (
    S3ObjectReference,
    create_s3_object_reference_from_bucket_and_key,
)
from sagemaker.jumpstart.curated_hub.accessors.model_dependency_s3_accessor import ModoelDependencyS3Accessor


class CuratedHubS3Accessor(ModoelDependencyS3Accessor):
    """Helper class to access Curated Hub s3 bucket"""

    def __init__(self, region: str, bucket: str):
        self._region = region
        self._bucket = bucket
        self._studio_metadata_map = get_studio_model_metadata_map_from_region(
            region
        )  # Necessary for SDK - Studio metadata drift

    def get_bucket(self) -> str:
        """Retrieves s3 bucket"""
        return self._bucket

    def get_inference_artifact_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectReference:
        """Retrieves s3 reference for model inference artifact"""
        return create_s3_object_reference_from_bucket_and_key(
            self.get_bucket(), f"{model_specs.model_id}/{model_specs.version}/infer.tar.gz"
        )

    def get_inference_script_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectReference:
        """Retrieves s3 reference for model traiing script"""
        return create_s3_object_reference_from_bucket_and_key(
            self.get_bucket(), f"{model_specs.model_id}/{model_specs.version}/sourcedir.tar.gz"
        )

    def get_training_artifact_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectReference:
        """Retrieves s3 reference for model training artifact"""
        return create_s3_object_reference_from_bucket_and_key(
            self.get_bucket(), f"{model_specs.model_id}/{model_specs.version}/train.tar.gz"
        )

    def get_training_script_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectReference:
        """Retrieves s3 reference for model training script"""
        return create_s3_object_reference_from_bucket_and_key(
            self.get_bucket(),
            f"{model_specs.model_id}/{model_specs.version}/training/sourcedir.tar.gz",
        )

    def get_demo_notebook_s3_reference(self, model_specs: JumpStartModelSpecs) -> S3ObjectReference:
        """Retrieves s3 reference for model jupyter notebook"""
        return create_s3_object_reference_from_bucket_and_key(
            self.get_bucket(), f"{model_specs.model_id}/{model_specs.version}/demo-notebook.ipynb"
        )

    def get_default_training_dataset_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectReference:
        """Retrieves s3 reference for s3 directory containing training datasets"""
        return create_s3_object_reference_from_bucket_and_key(
            self.get_bucket(), self._get_training_dataset_prefix(model_specs)
        )

    def _get_training_dataset_prefix(
        self, model_specs: JumpStartModelSpecs
    ) -> str:  # Studio expects the same format as public hub bucket
        """Retrieves training dataset"""
        studio_model_metadata = self._studio_metadata_map[model_specs.model_id]  # TODO: verify this
        return studio_model_metadata["defaultDataKey"]

    def get_markdown_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectReference:  # Studio expects the same format as public hub bucket
        """Retrieves s3 reference for model markdown file"""
        framework = get_model_framework(model_specs)
        key = f"{framework}-metadata/{model_specs.model_id}.md"
        return create_s3_object_reference_from_bucket_and_key(self.get_bucket(), key)
