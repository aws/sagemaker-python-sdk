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
import uuid
from typing import Dict, Any
from sagemaker.jumpstart.curated_hub.utils import (
    get_model_framework,
)
from sagemaker.jumpstart.types import JumpStartModelSpecs
from sagemaker.jumpstart.curated_hub.accessors.s3_object_reference import (
    S3ObjectLocation,
)
from sagemaker.jumpstart.curated_hub.accessors.constants import (
    PRIVATE_MODEL_TRAINING_ARTIFACT_TARBALL_S3_SUFFIX,
    PRIVATE_MODEL_TRAINING_SCRIPT_S3_SUFFIX,
    PRIVATE_MODEL_HOSTING_ARTIFACT_S3_TARBALL_SUFFIX,
    PRIVATE_MODEL_HOSTING_SCRIPT_S3_SUFFIX,
    PRIVATE_MODEL_INFERENCE_NOTEBOOK_S3_SUFFIX,
)
from sagemaker.jumpstart.curated_hub.accessors.model_dependency_s3_accessor import (
    ModelDependencyS3Accessor,
)

DISAMBIGUATE_SUFFIX = uuid.uuid4()


class CuratedHubS3Accessor(ModelDependencyS3Accessor):
    """Helper class to access Curated Hub s3 bucket"""

    def __init__(
        self,
        region: str,
        bucket: str,
        studio_metadata_map: Dict[str, Dict[str, Any]],
        base_s3_key: str = "",
    ):
        self._region = region
        self._hub_s3_config = S3ObjectLocation(bucket=bucket, key=base_s3_key)
        self._studio_metadata_map = studio_metadata_map  # Necessary for SDK - Studio metadata drift

    def get_bucket_name(self) -> str:
        """Retrieves s3 bucket name."""
        return self._hub_s3_config.bucket

    def get_inference_artifact_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectLocation:
        """Retrieves s3 reference for model inference artifact."""
        return S3ObjectLocation(
            self.get_bucket_name(),
            (
                f"{self._get_unique_s3_key_prefix(model_specs)}/"
                f"{PRIVATE_MODEL_HOSTING_ARTIFACT_S3_TARBALL_SUFFIX}"
            ),
        )

    def get_inference_script_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectLocation:
        """Retrieves s3 reference for model traiing script."""
        return S3ObjectLocation(
            self.get_bucket_name(),
            f"{self._get_unique_s3_key_prefix(model_specs)}"
            f"/{PRIVATE_MODEL_HOSTING_SCRIPT_S3_SUFFIX}",
        )

    def get_training_artifact_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectLocation:
        """Retrieves s3 reference for model training artifact."""
        return S3ObjectLocation(
            self.get_bucket_name(),
            (
                f"{self._get_unique_s3_key_prefix(model_specs)}/"
                f"{PRIVATE_MODEL_TRAINING_ARTIFACT_TARBALL_S3_SUFFIX}"
            ),
        )

    def get_training_script_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectLocation:
        """Retrieves s3 reference for model training script."""
        return S3ObjectLocation(
            self.get_bucket_name(),
            f"{self._get_unique_s3_key_prefix(model_specs)}"
            f"/{PRIVATE_MODEL_TRAINING_SCRIPT_S3_SUFFIX}",
        )

    def get_demo_notebook_s3_reference(self, model_specs: JumpStartModelSpecs) -> S3ObjectLocation:
        """Retrieves s3 reference for model jupyter notebook."""
        return S3ObjectLocation(
            self.get_bucket_name(),
            (
                f"{self._get_unique_s3_key_prefix(model_specs)}/"
                f"{PRIVATE_MODEL_INFERENCE_NOTEBOOK_S3_SUFFIX}"
            ),
        )

    def get_default_training_dataset_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectLocation:
        """Retrieves s3 reference for s3 directory containing training datasets."""
        return S3ObjectLocation(
            self.get_bucket_name(), 
            (
                f"{self._get_unique_s3_key_prefix(model_specs)}/"
                f"{self._get_training_dataset_prefix(model_specs)}"
            ),
        )

    def _get_unique_s3_key_prefix(self, model_specs: JumpStartModelSpecs) -> str:
        """Creates a unique s3 key prefix based off S3 storage config."""
        key = (
            f"{self._hub_s3_config.key}/Model/"
            f"{model_specs.model_id}-{DISAMBIGUATE_SUFFIX}/{model_specs.version}"
        )
        return key.lstrip("/")

    def _get_training_dataset_prefix(self, model_specs: JumpStartModelSpecs) -> str:
        """Retrieves s3 prefix for the training dataset."""
        studio_model_metadata = self._studio_metadata_map[model_specs.model_id]
        return studio_model_metadata["defaultDataKey"]

    def get_markdown_s3_reference(self, model_specs: JumpStartModelSpecs) -> S3ObjectLocation:
        """Retrieves s3 reference for model markdown file."""
        return S3ObjectLocation(
            self.get_bucket_name(), f"{self._get_unique_s3_key_prefix(model_specs)}/markdown.md"
        )
