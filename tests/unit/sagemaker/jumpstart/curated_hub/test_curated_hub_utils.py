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

from mock.mock import Mock

from sagemaker.jumpstart.curated_hub.utils import (
    list_objects_by_prefix,
    to_s3_folder_prefix,
    convert_s3_key_to_new_prefix,
    base_framework,
    get_model_framework,
)


class CuratedHubUtilsTest(unittest.TestCase):
    def test_list_objects_by_prefix_invalid_bucket_name_fails(self):
        with self.assertRaises(ValueError):
            list_objects_by_prefix(None, "prefix", None)

    def test_list_objects_by_prefix_invalid_prefix_fails(self):
        with self.assertRaises(ValueError):
            list_objects_by_prefix("bucket", None, None)

    def test_list_objects_by_prefix_no_content_empty(self):
        mock_s3_client = Mock()
        contents = {"Contents": ["hello"]}
        mock_s3_client.list_objects_v2.return_value = contents

        res = list_objects_by_prefix("bucket", "prefix", mock_s3_client)

        self.assertEqual(res, contents["Contents"])

    def test_to_s3_folder_prefix_removes_strings(self):
        input = "////test"
        output = "test/"

        self.assertEqual(to_s3_folder_prefix(input), output)

    def test_convert_s3_key_to_new_prefix(self):
        src = "src"
        dest = "dst"
        input = f"{src}/test"
        output = f"{dest}/test"

        self.assertEqual(convert_s3_key_to_new_prefix(input, src, dest), output)

    def test_base_framework(self):
        model_specs = Mock()
        model_specs.hosting_ecr_specs.framework = "huggingface"
        model_specs.hosting_ecr_specs.framework_version = "version"

        self.assertEqual(
            base_framework(model_specs), f"pytorch{model_specs.hosting_ecr_specs.framework_version}"
        )

    def test_get_model_framework(self):
        model_specs = Mock()
        model_specs.model_id = "test-model-id"

        self.assertEqual(get_model_framework(model_specs), "test")
