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

from sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub import JumpStartCuratedPublicHub
from sagemaker.jumpstart.curated_hub.utils import PublicModelId
from sagemaker.jumpstart.curated_hub.hub_model_specs.hub_model_specs import Dependency

TEST_S3_BUCKET_ALREADY_EXISTS_RESPONSE = {
    "Error": {
        "Code": "BucketAlreadyOwnedByYou",
        "Message": "Details/context around the exception or error",
    },
    "ResponseMetadata": {
        "RequestId": "1234567890ABCDEF",
        "HostId": "host ID data will appear here as a hash",
        "HTTPStatusCode": 400,
        "HTTPHeaders": {
            "x-amzn-requestid": "12345678-90AB-CDEF-1234567890A",
            "x-amz-id-2": "base64stringherenotarealamzid2value",
            "date": "Thu, 17 Jun 2021 16:08:34 GMT",
        },
        "RetryAttempts": 0,
    },
}

TEST_HUB_ALREADY_EXISTS_RESPONSE = {
    "Error": {
        "Code": "ResourceInUse",
        "Message": "Details/context around the exception or error",
    },
    "ResponseMetadata": {
        "RequestId": "1234567890ABCDEF",
        "HostId": "host ID data will appear here as a hash",
        "HTTPStatusCode": 400,
        "HTTPHeaders": {
            "x-amzn-requestid": "12345678-90AB-CDEF-1234567890A",
            "x-amz-id-2": "base64stringherenotarealamzid2value",
            "date": "Thu, 17 Jun 2021 16:08:34 GMT",
        },
        "RetryAttempts": 0,
    },
}

TEST_SERVICE_ERROR_RESPONSE = {
    "Error": {
        "Code": "SomeServiceException",
        "Message": "Details/context around the exception or error",
    },
    "ResponseMetadata": {
        "RequestId": "1234567890ABCDEF",
        "HostId": "host ID data will appear here as a hash",
        "HTTPStatusCode": 400,
        "HTTPHeaders": {
            "x-amzn-requestid": "12345678-90AB-CDEF-1234567890A",
            "x-amz-id-2": "base64stringherenotarealamzid2value",
            "date": "Thu, 17 Jun 2021 16:08:34 GMT",
        },
        "RetryAttempts": 0,
    },
}


class JumpStartCuratedPublicHubTest(unittest.TestCase):

    test_hub_name = "test-curated-hub-chrstfu"
    test_preexisting_hub_name = "test_preexisting_hub"
    test_preexisting_bucket_name = "test_preexisting_bucket"
    test_region = "us-east-2"
    test_account_id = "123456789012"

    test_public_js_model = PublicModelId(id="autogluon-classification-ensemble", version="1.1.1")
    test_second_public_js_model = PublicModelId(id="catboost-classification-model", version="1.2.7")
    test_nonexistent_public_js_model = PublicModelId(id="fail", version="1.0.0")

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub.JumpStartCuratedPublicHub._init_clients"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub."
        + "JumpStartCuratedPublicHub._get_curated_hub_and_curated_hub_s3_bucket_names"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub.get_studio_model_metadata_map_from_region"
    )
    def setUp(self, mock_studio_metadata, mock_get_names, mock_init_clients):
        mock_studio_metadata.return_value = {}
        mock_get_names.return_value = self.test_preexisting_hub_name, self.test_preexisting_hub_name

        self.test_curated_hub = JumpStartCuratedPublicHub(
            self.test_hub_name, False, self.test_region
        )

        self.assertTrue(self.test_curated_hub._should_skip_create())

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub."
        + "JumpStartCuratedPublicHub._get_preexisting_hub_and_s3_bucket_names"
    )
    def test_get_curated_hub_and_curated_hub_s3_bucket_names_hub_does_not_exist_uses_input_values(
        self, mock_get_preexisting
    ):
        mock_get_preexisting.return_value = None

        (
            res_hub_name,
            res_hub_bucket_name,
        ) = self.test_curated_hub._get_curated_hub_and_curated_hub_s3_bucket_names(
            self.test_hub_name, False
        )

        self.assertEqual(self.test_hub_name, res_hub_name)
        self.assertIn(f"{self.test_hub_name}-{self.test_region}", res_hub_bucket_name)

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub."
        + "JumpStartCuratedPublicHub._get_preexisting_hub_and_s3_bucket_names"
    )
    def test_get_curated_hub_and_curated_hub_s3_bucket_names_hub_does_exist_uses_preexisting_values(
        self, mock_get_preexisting
    ):
        mock_get_preexisting.return_value = (
            self.test_preexisting_hub_name,
            self.test_preexisting_bucket_name,
        )

        (
            res_hub_name,
            res_hub_bucket_name,
        ) = self.test_curated_hub._get_curated_hub_and_curated_hub_s3_bucket_names(
            self.test_hub_name, True
        )

        self.assertEqual(self.test_preexisting_hub_name, res_hub_name)
        self.assertEqual(self.test_preexisting_bucket_name, res_hub_bucket_name)

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub."
        + "JumpStartCuratedPublicHub._get_preexisting_hub_and_s3_bucket_names"
    )
    def test_get_curated_hub_and_curated_hub_s3_bucket_names_hub_does_exist_import_to_existing_hub_false_throws_error(
        self, mock_get_preexisting
    ):
        mock_get_preexisting.return_value = (
            self.test_preexisting_hub_name,
            self.test_preexisting_bucket_name,
        )

        with self.assertRaises(Exception) as context:
            self.test_curated_hub._get_curated_hub_and_curated_hub_s3_bucket_names(
                self.test_hub_name, False
            )

        error_msg = (
            f"Hub with name {self.test_preexisting_hub_name} detected on account. "
            "The limit of hubs per account is 1. If you wish to use this hub as the curated hub, "
            "please set the flag `import_to_preexisting_hub` to True."
        )
        self.assertEqual(error_msg, str(context.exception))

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub."
        + "JumpStartCuratedPublicHub._create_hub_and_hub_bucket"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub.JumpStartCuratedPublicHub._should_skip_create"
    )
    def test_create_if_skip_create_true_does_not_create(
        self, mock_skip_create, mock_create_hub_and_hub_bucket
    ):
        mock_skip_create.return_value = True

        self.test_curated_hub.create()

        mock_create_hub_and_hub_bucket.assert_not_called()

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub."
        + "JumpStartCuratedPublicHub._create_hub_and_hub_bucket"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub.JumpStartCuratedPublicHub._should_skip_create"
    )
    def test_create_if_skip_create_false_does_create(
        self, mock_skip_create, mock_create_hub_and_hub_bucket
    ):
        mock_skip_create.return_value = False

        self.test_curated_hub.create()

        mock_create_hub_and_hub_bucket.assert_called_once()

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub.JumpStartCuratedPublicHub._get_model_specs"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub.JumpStartCuratedPublicHub._import_models"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub.JumpStartCuratedPublicHub._model_needs_update"
    )
    def test_sync_filters_models_that_dont_need_update(
        self, mock_need_update, mock_import_models, mock_model_specs
    ):
        test_list = ["test_specs_1", "test_specs_2"]
        mock_model_specs.return_value = test_list
        mock_need_update.side_effect = self._mock_should_update_model

        self.test_curated_hub.sync(test_list)

        mock_import_models.assert_called_with(["test_specs_1"])

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub.JumpStartCuratedPublicHub._get_model_specs"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub.JumpStartCuratedPublicHub._import_models"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub.JumpStartCuratedPublicHub._model_needs_update"
    )
    def test_sync_force_update_true_updates_all_models(
        self, mock_need_update, test_import_models, mock_model_specs
    ):
        test_list = ["test_specs_1", "test_specs_2"]
        mock_model_specs.return_value = test_list
        mock_need_update.side_effect = self._mock_should_update_model

        self.test_curated_hub.sync(test_list, True)

        test_import_models.assert_called_with(test_list)

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub."
        + "JumpStartCuratedPublicHub._delete_model_dependencies_no_content_noop"
    )
    def test_delete_model_from_curated_hub_deletes_dependencies_true_deletes_dependecnies(
        self, mock_delete_model_deps
    ):
        mock_hub_client = Mock()
        self.test_curated_hub._curated_hub_client = mock_hub_client

        self.test_curated_hub._delete_model_from_curated_hub("test_spec", True)

        mock_delete_model_deps.assert_called_once_with("test_spec")
        mock_hub_client.delete_version_of_model.assert_not_called()
        mock_hub_client.delete_all_versions_of_model.assert_called_once_with("test_spec")

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub."
        + "JumpStartCuratedPublicHub._delete_model_dependencies_no_content_noop"
    )
    def test_delete_model_from_curated_hub_deletes_dependencies_false_keeps_dependecnies(
        self, mock_delete_model_deps
    ):
        mock_hub_client = Mock()
        self.test_curated_hub._curated_hub_client = mock_hub_client

        self.test_curated_hub._delete_model_from_curated_hub("test_spec", True, False)

        mock_delete_model_deps.assert_not_called()
        mock_hub_client.delete_version_of_model.assert_not_called()
        mock_hub_client.delete_all_versions_of_model.assert_called_once_with("test_spec")

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub."
        + "JumpStartCuratedPublicHub._delete_model_dependencies_no_content_noop"
    )
    def test_delete_model_from_curated_hub_deletes_delete_single_vesion(
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

    @patch("sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub.find_objects_under_prefix")
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
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub."
        + "JumpStartCuratedPublicHub._format_dependency_dst_uris_for_delete_objects"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub."
        + "JumpStartCuratedPublicHub._get_hub_content_dependencies_from_model_document"
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
        mock_hub_client.desribe_model.return_value = {"HubContentDocument": "mock"}

        test_spec = Mock()
        test_spec.model_id = "test_model_id"
        test_spec.version = "test_model_version"

        self.test_curated_hub._delete_model_dependencies_no_content_noop(test_spec)

        mock_hub_client.desribe_model.assert_called_once()
        mock_s3_client.delete_objects.assert_called_once()
        mock_get_deps.assert_called_once()

    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub."
        + "JumpStartCuratedPublicHub._get_hub_content_dependencies_from_model_document"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub."
        + "JumpStartCuratedPublicHub._format_dependency_dst_uris_for_delete_objects"
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
        mock_sm_client.describe_hub_content.return_value = {"HubContentDocument": "mock"}

        test_spec = Mock()
        test_spec.model_id = "test_model_id"
        test_spec.version = "test_model_version"

        with self.assertRaises(Exception):
            self.test_curated_hub._delete_model_dependencies_no_content_noop(test_spec)

    def _mock_should_update_model(self, model_id: str):
        return model_id == "test_specs_1"
