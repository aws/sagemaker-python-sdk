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

from mock.mock import patch, Mock

from sagemaker.jumpstart.curated_hub.jumpstart_curated_hub import JumpStartCuratedHub
from sagemaker.jumpstart.curated_hub.utils import PublicHubModel
from sagemaker.jumpstart.curated_hub.hub_model_specs.hub_model_specs import Dependency
from botocore.client import ClientError


TEST_S3_BUCKET_ALREADY_EXISTS_RESPONSE = {
    "Error": {
        "Code": "BucketAlreadyOwnedByYou",
    }
}

TEST_HUB_DOES_NOT_EXIST_RESPONSE = {
    "Error": {
        "Code": "ResourceNotFound",
    }
}

TEST_HUB_ALREADY_EXISTS_RESPONSE = {
    "Error": {
        "Code": "ResourceInUse",
    }
}

TEST_SERVICE_ERROR_RESPONSE = {
    "Error": {
        "Code": "SomeServiceException",
    }
}

TEST_HUB_NAME = "test-curated-hub-chrstfu"
TEST_REGION = "us-east-2"

TEST_PREEXISTING_HUB_NAME = "test_preexisting_hub"
TEST_PREEXISTING_BUCKET_NAME = "test_preexisting_bucket"
TEST_PREEXISTING_S3_KEY_PREFIX = "test_prefix"

TEST_HUB_CONTENT_ALREADY_IN_HUB_ID = "test_hub_content_already_in_hub"


def _mock_describe_version(mock_spec):
    if mock_spec.model_id == TEST_HUB_CONTENT_ALREADY_IN_HUB_ID:
        return
    raise ClientError(error_response=TEST_HUB_DOES_NOT_EXIST_RESPONSE, operation_name="blah")


class JumpStartCuratedHubTest(unittest.TestCase):

    test_account_id = "123456789012"

    test_public_js_model = PublicHubModel(id="autogluon-classification-ensemble", version="1.1.1")
    test_second_public_js_model = PublicHubModel(
        id="catboost-classification-model", version="1.2.7"
    )
    test_nonexistent_public_js_model = PublicHubModel(id="fail", version="1.0.0")

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub.JumpStartCuratedHub._init_dependencies"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub.JumpStartCuratedHub._get_sm_client"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub.get_studio_model_metadata_map_from_region"
    )
    def setUp(self, mock_metadata_map, mock_get_sm_client, mock_init_deps):
        mock_sm_client = Mock()
        mock_sm_client.describe_hub.return_value = {
            "S3StorageConfig": {
                "S3OutputPath": f"s3://{TEST_PREEXISTING_BUCKET_NAME}/{TEST_PREEXISTING_S3_KEY_PREFIX}"
            }
        }
        mock_get_sm_client.return_value = mock_sm_client

        self.test_curated_hub = JumpStartCuratedHub(TEST_HUB_NAME, TEST_REGION)

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub.JumpStartCuratedHub._init_dependencies"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub.JumpStartCuratedHub._get_sm_client"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub.get_studio_model_metadata_map_from_region"
    )
    def test_init_hub_exist_should_use_old_s3_config(
        self, mock_metadata_map, mock_get_sm_client, mock_init_deps
    ):
        mock_sm_client = Mock()
        mock_sm_client.describe_hub.return_value = {
            "S3StorageConfig": {
                "S3OutputPath": f"s3://{TEST_PREEXISTING_BUCKET_NAME}/{TEST_PREEXISTING_S3_KEY_PREFIX}"
            }
        }
        mock_get_sm_client.return_value = mock_sm_client

        curated_hub = JumpStartCuratedHub(TEST_HUB_NAME, TEST_REGION)

        self.assertEquals(curated_hub.curated_hub_s3_bucket_name, TEST_PREEXISTING_BUCKET_NAME)
        self.assertEquals(curated_hub.curated_hub_s3_key_prefix, TEST_PREEXISTING_S3_KEY_PREFIX)

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub.JumpStartCuratedHub._init_dependencies"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub.JumpStartCuratedHub._get_sm_client"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub.get_studio_model_metadata_map_from_region"
    )
    def test_init_s3_config_no_key_use_old_s3_config(
        self, mock_metadata_map, mock_get_sm_client, mock_init_deps
    ):
        mock_sm_client = Mock()
        mock_sm_client.describe_hub.return_value = {
            "S3StorageConfig": {"S3OutputPath": f"s3://{TEST_PREEXISTING_BUCKET_NAME}"}
        }
        mock_get_sm_client.return_value = mock_sm_client

        curated_hub = JumpStartCuratedHub(TEST_HUB_NAME, TEST_REGION)

        self.assertEquals(curated_hub.curated_hub_s3_bucket_name, TEST_PREEXISTING_BUCKET_NAME)
        self.assertEquals(curated_hub.curated_hub_s3_key_prefix, "")

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub.JumpStartCuratedHub._create_unique_s3_bucket_name"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub.JumpStartCuratedHub._init_dependencies"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub.JumpStartCuratedHub._get_sm_client"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub.get_studio_model_metadata_map_from_region"
    )
    def test_init_hub_not_exist_create_new_config(
        self, mock_metadata_map, mock_get_sm_client, mock_init_deps, mock_create_unique
    ):
        mock_new_bucket_name = "new_bucket_name"
        mock_create_unique.return_value = mock_new_bucket_name
        mock_sm_client = Mock()
        mock_sm_client.describe_hub.side_effect = ClientError(
            error_response=TEST_HUB_DOES_NOT_EXIST_RESPONSE, operation_name="blah"
        )
        mock_get_sm_client.return_value = mock_sm_client

        curated_hub = JumpStartCuratedHub(TEST_HUB_NAME, TEST_REGION)

        self.assertEquals(curated_hub.curated_hub_s3_bucket_name, mock_new_bucket_name)
        self.assertEquals(curated_hub.curated_hub_s3_key_prefix, "")

    def test_create_import_into_preexisting_true_and_hub_exists_should_succeed(self):
        mock_s3_client = Mock()
        mock_s3_client.create_bucket.side_effect = ClientError(
            error_response=TEST_S3_BUCKET_ALREADY_EXISTS_RESPONSE, operation_name="blah"
        )
        self.test_curated_hub._s3_client = mock_s3_client

        mock_curated_hub_client = Mock()
        mock_curated_hub_client.create_hub.side_effect = ClientError(
            error_response=TEST_HUB_ALREADY_EXISTS_RESPONSE, operation_name="blah"
        )
        self.test_curated_hub._curated_hub_client = mock_curated_hub_client

        self.test_curated_hub.create(import_into_preexisting=True)

        mock_s3_client.create_bucket.assert_called_with(
            Bucket=self.test_curated_hub.curated_hub_s3_bucket_name,
            CreateBucketConfiguration={"LocationConstraint": self.test_curated_hub._region},
        )
        mock_curated_hub_client.create_hub.assert_called_with(
            TEST_HUB_NAME, self.test_curated_hub.curated_hub_s3_bucket_name
        )

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub.JumpStartCuratedHub._init_dependencies"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub.JumpStartCuratedHub._get_sm_client"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub.get_studio_model_metadata_map_from_region"
    )
    def test_create_us_east_1_should_have_no_location_constraint(
        self, mock_metadata_map, mock_get_sm_client, mock_init_deps
    ):
        mock_sm_client = Mock()
        mock_sm_client.describe_hub.return_value = {
            "S3StorageConfig": {
                "S3OutputPath": f"s3://{TEST_PREEXISTING_BUCKET_NAME}/{TEST_PREEXISTING_S3_KEY_PREFIX}"
            }
        }
        mock_get_sm_client.return_value = mock_sm_client

        self.test_curated_hub = JumpStartCuratedHub(TEST_HUB_NAME, "us-east-1")

        mock_s3_client = Mock()
        mock_s3_client.create_bucket.side_effect = ClientError(
            error_response=TEST_S3_BUCKET_ALREADY_EXISTS_RESPONSE, operation_name="blah"
        )
        self.test_curated_hub._s3_client = mock_s3_client

        mock_curated_hub_client = Mock()
        mock_curated_hub_client.create_hub.side_effect = ClientError(
            error_response=TEST_HUB_ALREADY_EXISTS_RESPONSE, operation_name="blah"
        )
        self.test_curated_hub._curated_hub_client = mock_curated_hub_client

        self.test_curated_hub.create(import_into_preexisting=True)

        mock_s3_client.create_bucket.assert_called_with(
            Bucket=self.test_curated_hub.curated_hub_s3_bucket_name,
            CreateBucketConfiguration=None,
        )
        mock_curated_hub_client.create_hub.assert_called_with(
            TEST_HUB_NAME, self.test_curated_hub.curated_hub_s3_bucket_name
        )

    def test_create_import_into_preexisting_false_and_hub_exists_should_throw_exception(self):
        mock_s3_client = Mock()
        mock_s3_client.create_bucket.side_effect = ClientError(
            error_response=TEST_S3_BUCKET_ALREADY_EXISTS_RESPONSE, operation_name="blah"
        )
        self.test_curated_hub._s3_client = mock_s3_client

        with self.assertRaises(ClientError):
            self.test_curated_hub.create(import_into_preexisting=False)

        mock_s3_client.create_bucket.assert_called_with(
            Bucket=self.test_curated_hub.curated_hub_s3_bucket_name,
            CreateBucketConfiguration={"LocationConstraint": self.test_curated_hub._region},
        )

    def test_create_no_preexisting_hub_should_succeed(self):
        mock_s3_client = Mock()
        self.test_curated_hub._s3_client = mock_s3_client

        mock_curated_hub_client = Mock()
        self.test_curated_hub._curated_hub_client = mock_curated_hub_client

        self.test_curated_hub.create(import_into_preexisting=False)

        mock_s3_client.create_bucket.assert_called_with(
            Bucket=self.test_curated_hub.curated_hub_s3_bucket_name,
            CreateBucketConfiguration={"LocationConstraint": self.test_curated_hub._region},
        )
        mock_curated_hub_client.create_hub.assert_called_with(
            TEST_HUB_NAME, self.test_curated_hub.curated_hub_s3_bucket_name
        )

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub.JumpStartCuratedHub._import_models"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub.JumpStartCuratedHub._get_model_specs_for_list"
    )
    def test_sync_filters_existing_models(self, mock_get_model_specs_for_list, mock_import_models):
        mock_curated_hub_client = Mock()
        mock_curated_hub_client.describe_model_version.side_effect = _mock_describe_version
        self.test_curated_hub._curated_hub_client = mock_curated_hub_client

        mock_spec_1 = Mock()
        mock_spec_1.model_id = TEST_HUB_CONTENT_ALREADY_IN_HUB_ID
        mock_spec_2 = Mock()
        mock_spec_2.model_id = "blah"
        mock_get_model_specs_for_list.return_value = [mock_spec_1, mock_spec_2]

        # Passing in empty array since mock_get_model_specs_for_list is already patched
        self.test_curated_hub.sync([])

        mock_import_models.assert_called_with([mock_spec_2])

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub.JumpStartCuratedHub._import_models"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub.JumpStartCuratedHub._get_model_specs_for_list"
    )
    def test_sync_force_update_updates_all_models(
        self, mock_get_model_specs_for_list, mock_import_models
    ):
        mock_curated_hub_client = Mock()
        mock_curated_hub_client.describe_model_version.side_effect = _mock_describe_version
        self.test_curated_hub._curated_hub_client = mock_curated_hub_client

        mock_spec_1 = Mock()
        mock_spec_1.model_id = TEST_HUB_CONTENT_ALREADY_IN_HUB_ID
        mock_spec_2 = Mock()
        mock_spec_2.model_id = "blah"
        mock_get_model_specs_for_list.return_value = [mock_spec_1, mock_spec_2]

        # Passing in empty array since mock_get_model_specs_for_list is already patched
        self.test_curated_hub.sync([], force_update=True)

        mock_import_models.assert_called_with([mock_spec_1, mock_spec_2])

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub."
        + "JumpStartCuratedHub._delete_model_dependencies_no_content_noop"
    )
    def test_delete_model_from_curated_hub_deletes_dependencies_true_deletes_dependencies(
        self, mock_delete_model_deps
    ):
        mock_hub_client = Mock()
        self.test_curated_hub._curated_hub_client = mock_hub_client

        self.test_curated_hub._delete_model_from_curated_hub("test_spec", True)

        mock_delete_model_deps.assert_called_once_with("test_spec")
        mock_hub_client.delete_version_of_model.assert_not_called()
        mock_hub_client.delete_all_versions_of_model.assert_called_once_with("test_spec")

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub."
        + "JumpStartCuratedHub._delete_model_dependencies_no_content_noop"
    )
    def test_delete_model_from_curated_hub_deletes_dependencies_false_keeps_dependencies(
        self, mock_delete_model_deps
    ):
        mock_hub_client = Mock()
        self.test_curated_hub._curated_hub_client = mock_hub_client

        self.test_curated_hub._delete_model_from_curated_hub("test_spec", True, False)

        mock_delete_model_deps.assert_not_called()
        mock_hub_client.delete_version_of_model.assert_not_called()
        mock_hub_client.delete_all_versions_of_model.assert_called_once_with("test_spec")

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub."
        + "JumpStartCuratedHub._delete_model_dependencies_no_content_noop"
    )
    def test_delete_model_from_curated_hub_deletes_delete_single_version(
        self, mock_delete_model_deps
    ):
        mock_hub_client = Mock()
        self.test_curated_hub._curated_hub_client = mock_hub_client
        test_spec = Mock()
        test_spec.model_id = "model_id"
        test_spec.version = "version"

        self.test_curated_hub._delete_model_from_curated_hub(test_spec, False)

        mock_delete_model_deps.assert_called_once_with(test_spec)
        mock_hub_client.delete_version_of_model.assert_called_once_with(
            test_spec.model_id, test_spec.version
        )
        mock_hub_client.delete_all_versions_of_model.assert_not_called()

    @patch("json.loads")
    def test_get_hub_content_dependencies_from_model_document(self, mock_json_loads):
        mock_json_loads.return_value = {
            "Dependencies": [
                {
                    "DependencyOriginPath": "s3://test_origin_1",
                    "DependencyCopyPath": "s3://test_copy/test_copy_key_1",
                    "DependencyType": "Model",
                },
                {
                    "DependencyOriginPath": "s3://test_origin_2",
                    "DependencyCopyPath": "s3://test_copy/test_copy_key_2",
                    "DependencyType": "Model",
                },
            ]
        }

        res = self.test_curated_hub._get_hub_content_dependencies_from_model_document("blah")

        expected = [
            Dependency(
                DependencyOriginPath="s3://test_origin_1",
                DependencyCopyPath="s3://test_copy/test_copy_key_1",
                DependencyType="Model",
            ),
            Dependency(
                DependencyOriginPath="s3://test_origin_2",
                DependencyCopyPath="s3://test_copy/test_copy_key_2",
                DependencyType="Model",
            ),
        ]
        self.assertEqual(expected, res)

    def test_format_dependency_dst_uris_for_delete_objects_if_not_dict_append(self):
        test_dependency = Dependency(
            DependencyOriginPath="s3://test_origin",
            DependencyCopyPath="s3://test_copy/test_copy_key",
            DependencyType="Model",
        )

        keys = self.test_curated_hub._format_dependency_dst_uris_for_delete_objects(test_dependency)

        expected = [{"Key": "test_copy_key"}]
        self.assertEqual(expected, keys)

    @patch("sagemaker.jumpstart.curated_hub.jumpstart_curated_hub.find_objects_under_prefix")
    def test_format_dependency_dst_uris_for_delete_objects_if_dict_append_dict_keys(
        self, mock_find_objects
    ):
        test_dependency = Dependency(
            DependencyOriginPath="s3://test_origin",
            DependencyCopyPath="s3://test_copy/test_dict_key/",
            DependencyType="Model",
        )
        mock_find_objects.return_value = ["test_directory_key_1", "test_directory_key_2"]

        keys = self.test_curated_hub._format_dependency_dst_uris_for_delete_objects(test_dependency)

        expected = [{"Key": "test_directory_key_1"}, {"Key": "test_directory_key_2"}]
        self.assertEqual(expected, keys)

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub."
        + "JumpStartCuratedHub._format_dependency_dst_uris_for_delete_objects"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub."
        + "JumpStartCuratedHub._get_hub_content_dependencies_from_model_document"
    )
    def test_delete_model_dependencies_no_content_noop_delete_no_error_passes(
        self, mock_get_deps, mock_format_deps
    ):
        mock_s3_client = Mock()
        self.test_curated_hub._s3_client = mock_s3_client
        mock_hub_client = Mock()
        self.test_curated_hub._curated_hub_client = mock_hub_client
        mock_format_deps.return_value = []
        mock_s3_client.delete_objects.return_value = {}
        mock_hub_client.describe_model_version.return_value = {"HubContentDocument": "mock"}

        test_spec = Mock()
        test_spec.model_id = "test_model_id"
        test_spec.version = "test_model_version"

        self.test_curated_hub._delete_model_dependencies_no_content_noop(test_spec)

        mock_hub_client.describe_model_version.assert_called_once()
        mock_s3_client.delete_objects.assert_called_once()
        mock_get_deps.assert_called_once()

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub."
        + "JumpStartCuratedHub._get_hub_content_dependencies_from_model_document"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_hub."
        + "JumpStartCuratedHub._format_dependency_dst_uris_for_delete_objects"
    )
    def test_delete_model_dependencies_no_content_noop_delete_error_throws_error(
        self, mock_get_deps, mock_format_deps
    ):
        mock_s3_client = Mock()
        self.test_curated_hub._s3_client = mock_s3_client
        mock_sm_client = Mock()
        self.test_curated_hub._sm_client = mock_sm_client
        mock_format_deps.return_value = []
        mock_s3_client.delete_objects.return_value = {"Errors": ["test_error"]}
        mock_sm_client.describe_model_version.return_value = {"HubContentDocument": "mock"}

        test_spec = Mock()
        test_spec.model_id = "test_model_id"
        test_spec.version = "test_model_version"

        with self.assertRaises(Exception):
            self.test_curated_hub._delete_model_dependencies_no_content_noop(test_spec)
