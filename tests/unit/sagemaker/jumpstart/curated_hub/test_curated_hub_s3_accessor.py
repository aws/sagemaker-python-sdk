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
"""This module contains utilities related to SageMaker JumpStart."""
from __future__ import absolute_import
import unittest

from sagemaker.jumpstart.curated_hub.accessors.curated_hub_s3_accessor import (
    CuratedHubS3Accessor,
)
from sagemaker.jumpstart.types import JumpStartModelSpecs
from tests.unit.sagemaker.jumpstart.constants import BASE_SPEC
from mock.mock import patch
from sagemaker.jumpstart.curated_hub.accessors.constants import (
    PRIVATE_MODEL_HOSTING_ARTIFACT_TARBALL_SUFFIX,
)

class CuratedHubS3AccessorTest(unittest.TestCase):
    @patch("sagemaker.jumpstart.curated_hub.accessors.curated_hub_s3_accessor.get_studio_model_metadata_map_from_region")
    def test_no_prefix(self, mock_studio_metadata):
        test_bucket = "test_bucket"
        test_key_prefix = "test_key_prefix"
        test_hub_accessor = CuratedHubS3Accessor(
            "us-west-2",
            test_bucket
        )

        test_specs = JumpStartModelSpecs(BASE_SPEC)
        key = f"Model/{test_specs.model_id}/{test_specs.version}/{PRIVATE_MODEL_HOSTING_ARTIFACT_TARBALL_SUFFIX}"

        test_reference = test_hub_accessor.get_inference_artifact_s3_reference(test_specs)
        self.assertEqual(test_bucket, test_reference.bucket)
        self.assertIn("Model", test_reference.key)
        self.assertIn(test_specs.model_id, test_reference.key)
        self.assertIn(test_specs.version, test_reference.key)
        self.assertIn(PRIVATE_MODEL_HOSTING_ARTIFACT_TARBALL_SUFFIX, test_reference.key)
        self.assertNotIn(test_key_prefix, test_reference.key)

    @patch("sagemaker.jumpstart.curated_hub.accessors.curated_hub_s3_accessor.get_studio_model_metadata_map_from_region")
    def test_with_prefix(self, mock_studio_metadata):
        test_bucket = "test_bucket"
        test_key_prefix = "test_key_prefix"
        test_hub_accessor = CuratedHubS3Accessor(
            "us-west-2",
            test_bucket,
            test_key_prefix
        )

        test_specs = JumpStartModelSpecs(BASE_SPEC)
        key = f"Model/{test_specs.model_id}/{test_specs.version}/{PRIVATE_MODEL_HOSTING_ARTIFACT_TARBALL_SUFFIX}"

        test_reference = test_hub_accessor.get_inference_artifact_s3_reference(test_specs)
        self.assertEqual(test_bucket, test_reference.bucket)
        self.assertIn(f"{test_key_prefix}/Model", test_reference.key)
        self.assertIn(test_specs.model_id, test_reference.key)
        self.assertIn(test_specs.version, test_reference.key)
        self.assertIn(PRIVATE_MODEL_HOSTING_ARTIFACT_TARBALL_SUFFIX, test_reference.key)
