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
from __future__ import absolute_import

from unittest import TestCase
from unittest.mock import Mock, PropertyMock, patch, mock_open

from sagemaker.serve.model_server.djl_serving.prepare import (
    _copy_jumpstart_artifacts,
    _create_dir_structure,
    _extract_js_resource,
)
from tests.unit.sagemaker.serve.model_server.constants import (
    MOCK_JUMPSTART_ID,
    MOCK_TMP_DIR,
    MOCK_COMPRESSED_MODEL_DATA_STR,
    MOCK_UNCOMPRESSED_MODEL_DATA_STR,
    MOCK_UNCOMPRESSED_MODEL_DATA_STR_FOR_DICT,
    MOCK_UNCOMPRESSED_MODEL_DATA_DICT,
    MOCK_INVALID_MODEL_DATA_DICT,
)

MOCK_DJL_JUMPSTART_GLOBED_RESOURCES = ["./config.json"]


class DjlPrepareTests(TestCase):
    @patch("sagemaker.serve.model_server.djl_serving.prepare._check_disk_space")
    @patch("sagemaker.serve.model_server.djl_serving.prepare._check_docker_disk_usage")
    @patch("sagemaker.serve.model_server.djl_serving.prepare.Path")
    def test_create_dir_structure_from_new(self, mock_path, mock_disk_usage, mock_disk_space):
        mock_model_path = Mock()
        mock_model_path.exists.return_value = False
        mock_code_dir = Mock()
        mock_model_path.joinpath.return_value = mock_code_dir
        mock_path.return_value = mock_model_path

        ret_model_path, ret_code_dir = _create_dir_structure(mock_model_path)

        mock_model_path.mkdir.assert_called_once_with(parents=True)
        mock_model_path.joinpath.assert_called_once_with("code")
        mock_code_dir.mkdir.assert_called_once_with(exist_ok=True, parents=True)
        mock_disk_space.assert_called_once_with(mock_model_path)
        mock_disk_usage.assert_called_once()

        self.assertEqual(ret_model_path, mock_model_path)
        self.assertEqual(ret_code_dir, mock_code_dir)

    @patch("sagemaker.serve.model_server.djl_serving.prepare.Path")
    def test_create_dir_structure_invalid_path(self, mock_path):
        mock_model_path = Mock()
        mock_model_path.exists.return_value = True
        mock_model_path.is_dir.return_value = False
        mock_path.return_value = mock_model_path

        with self.assertRaises(ValueError) as context:
            _create_dir_structure(mock_model_path)

        self.assertEqual("model_dir is not a valid directory", str(context.exception))

    @patch("sagemaker.serve.model_server.djl_serving.prepare.S3Downloader")
    @patch("builtins.open", new_callable=mock_open, read_data="data")
    @patch("json.load", return_value={})
    def test_prepare_djl_js_resources_for_jumpstart_uncompressed_str(
        self,
        mock_load,
        mock_open,
        mock_s3_downloader,
    ):
        mock_code_dir = Mock()
        mock_s3_downloader_obj = Mock()
        mock_s3_downloader.return_value = mock_s3_downloader_obj

        _copy_jumpstart_artifacts(
            MOCK_UNCOMPRESSED_MODEL_DATA_STR, MOCK_JUMPSTART_ID, mock_code_dir
        )

        mock_s3_downloader_obj.download.assert_called_once_with(
            MOCK_UNCOMPRESSED_MODEL_DATA_STR, mock_code_dir
        )

    @patch("sagemaker.serve.model_server.djl_serving.prepare.S3Downloader")
    @patch("builtins.open", new_callable=mock_open, read_data="data")
    @patch("json.load", return_value={})
    def test_prepare_djl_js_resources_for_jumpstart_uncompressed_dict(
        self,
        mock_load,
        mock_open,
        mock_s3_downloader,
    ):
        mock_code_dir = Mock()
        mock_s3_downloader_obj = Mock()
        mock_s3_downloader.return_value = mock_s3_downloader_obj

        _copy_jumpstart_artifacts(
            MOCK_UNCOMPRESSED_MODEL_DATA_DICT, MOCK_JUMPSTART_ID, mock_code_dir
        )

        mock_s3_downloader_obj.download.assert_called_once_with(
            MOCK_UNCOMPRESSED_MODEL_DATA_STR_FOR_DICT, mock_code_dir
        )

    @patch("sagemaker.serve.model_server.djl_serving.prepare.S3Downloader")
    @patch("builtins.open", new_callable=mock_open, read_data="data")
    @patch("json.load", return_value={})
    def test_prepare_djl_js_resources_for_jumpstart_invalid_model_data(
        self,
        mock_load,
        mock_open,
        mock_s3_downloader,
    ):
        mock_code_dir = Mock()
        mock_s3_downloader_obj = Mock()
        mock_s3_downloader.return_value = mock_s3_downloader_obj

        with self.assertRaises(ValueError) as context:
            _copy_jumpstart_artifacts(
                MOCK_INVALID_MODEL_DATA_DICT, MOCK_JUMPSTART_ID, mock_code_dir
            )

        self.assertTrue(
            "JumpStart model data compression format is unsupported" in str(context.exception)
        )

    @patch("sagemaker.serve.model_server.djl_serving.prepare.S3Downloader")
    @patch("sagemaker.serve.model_server.djl_serving.prepare._extract_js_resource")
    @patch("sagemaker.serve.model_server.djl_serving.prepare._tmpdir")
    @patch("builtins.open", new_callable=mock_open, read_data="data")
    @patch("json.load", return_value={})
    def test_prepare_djl_js_resources_for_jumpstart_compressed_str(
        self,
        mock_load,
        mock_open,
        mock_tmpdir,
        mock_extract_js_resource,
        mock_s3_downloader,
    ):
        mock_code_dir = Mock()

        mock_s3_downloader_obj = Mock()
        mock_s3_downloader.return_value = mock_s3_downloader_obj

        mock_tmpdir_obj = Mock()
        mock_js_dir = Mock()
        mock_js_dir.return_value = MOCK_TMP_DIR
        type(mock_tmpdir_obj).__enter__ = PropertyMock(return_value=mock_js_dir)
        type(mock_tmpdir_obj).__exit__ = PropertyMock(return_value=Mock())
        mock_tmpdir.return_value = mock_tmpdir_obj

        _copy_jumpstart_artifacts(MOCK_COMPRESSED_MODEL_DATA_STR, MOCK_JUMPSTART_ID, mock_code_dir)

        mock_s3_downloader_obj.download.assert_called_once_with(
            MOCK_COMPRESSED_MODEL_DATA_STR, MOCK_TMP_DIR
        )
        mock_extract_js_resource.assert_called_once_with(
            MOCK_TMP_DIR, mock_code_dir, MOCK_JUMPSTART_ID
        )

    @patch("sagemaker.serve.model_server.djl_serving.prepare.Path")
    @patch("sagemaker.serve.model_server.djl_serving.prepare.tarfile")
    def test_extract_js_resources_success(self, mock_tarfile, mock_path):
        mock_path_obj = Mock()
        mock_path_obj.joinpath.return_value = Mock()
        mock_path.return_value = mock_path_obj

        mock_tar_obj = Mock()
        mock_enter = Mock()
        mock_resource_obj = Mock()
        mock_enter.return_value = mock_resource_obj
        type(mock_tar_obj).__enter__ = PropertyMock(return_value=mock_enter)
        type(mock_tar_obj).__exit__ = PropertyMock(return_value=Mock())
        mock_tarfile.open.return_value = mock_tar_obj

        js_model_dir = ""
        code_dir = Mock()
        _extract_js_resource(js_model_dir, code_dir, MOCK_JUMPSTART_ID)

        mock_path.assert_called_once_with(js_model_dir)
        mock_path_obj.joinpath.assert_called_once_with(f"infer-prepack-{MOCK_JUMPSTART_ID}.tar.gz")
        mock_resource_obj.extractall.assert_called_once_with(path=code_dir, filter="data")
