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
from sagemaker.jumpstart.types import JumpStartModelSpecs
from sagemaker.jumpstart.curated_hub.accessors.s3_object_reference import (
    S3ObjectReference,
)


class ModoelDependencyS3Accessor:
    """Interface class to access JumpStart s3 buckets"""

    def get_bucket(self) -> str:
        """Retrieves s3 bucket"""
        raise Exception("Not implemented")

    def get_inference_artifact_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectReference:
        """Retrieves s3 reference for model inference artifact"""
        raise Exception("Not implemented")

    def get_training_artifact_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectReference:
        """Retrieves s3 reference for model training artifact"""
        raise Exception("Not implemented")

    def get_inference_script_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectReference:
        """Retrieves s3 reference for model inference script"""
        raise Exception("Not implemented")

    def get_training_script_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectReference:
        """Retrieves s3 reference for model training script"""
        raise Exception("Not implemented")

    def get_default_training_dataset_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectReference:
        """Retrieves s3 reference for s3 directory containing training datasets"""
        raise Exception("Not implemented")

    def get_demo_notebook_s3_reference(self, model_specs: JumpStartModelSpecs) -> S3ObjectReference:
        """Retrieves s3 reference for demo jupyter notebook"""
        raise Exception("Not implemented")

    def get_markdown_s3_reference(self, model_specs: JumpStartModelSpecs) -> S3ObjectReference:
        """Retrieves s3 reference for model markdown"""
        raise Exception("Not implemented")
