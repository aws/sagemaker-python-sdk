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
from unittest.mock import Mock, PropertyMock, patch, mock_open, call

from sagemaker.serve.model_server.djl_serving.prepare import (
    _copy_jumpstart_artifacts,
    _create_dir_structure,
    _move_to_code_dir,
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

MOCK_DJL_JUMPSTART_GLOBED_RESOURCES = ["./inference.py", "./serving.properties", "./config.json"]


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

        self.assertEquals(ret_model_path, mock_model_path)
        self.assertEquals(ret_code_dir, mock_code_dir)

    @patch("sagemaker.serve.model_server.djl_serving.prepare.Path")
    def test_create_dir_structure_invalid_path(self, mock_path):
        mock_model_path = Mock()
        mock_model_path.exists.return_value = True
        mock_model_path.is_dir.return_value = False
        mock_path.return_value = mock_model_path

        with self.assertRaises(ValueError) as context:
            _create_dir_structure(mock_model_path)

        self.assertEquals("model_dir is not a valid directory", str(context.exception))

    @patch("sagemaker.serve.model_server.djl_serving.prepare.S3Downloader")
    @patch("sagemaker.serve.model_server.djl_serving.prepare._tmpdir")
    @patch(
        "sagemaker.serve.model_server.djl_serving.prepare._read_existing_serving_properties",
        return_value={},
    )
    @patch("sagemaker.serve.model_server.djl_serving.prepare._move_to_code_dir")
    @patch("builtins.open", new_callable=mock_open, read_data="data")
    @patch("json.load", return_value={})
    def test_prepare_djl_js_resources_for_jumpstart_uncompressed_str(
        self,
        mock_load,
        mock_open,
        mock_move_to_code_dir,
        mock_existing_props,
        mock_tmpdir,
        mock_s3_downloader,
    ):
        mock_code_dir = Mock()
        mock_config_json_file = Mock()
        mock_config_json_file.is_file.return_value = True
        mock_code_dir.joinpath.return_value = mock_config_json_file

        mock_s3_downloader_obj = Mock()
        mock_s3_downloader.return_value = mock_s3_downloader_obj

        mock_tmpdir_obj = Mock()
        mock_js_dir = Mock()
        mock_js_dir.return_value = MOCK_TMP_DIR
        type(mock_tmpdir_obj).__enter__ = PropertyMock(return_value=mock_js_dir)
        type(mock_tmpdir_obj).__exit__ = PropertyMock(return_value=Mock())
        mock_tmpdir.return_value = mock_tmpdir_obj

        existing_properties, hf_model_config, success = _copy_jumpstart_artifacts(
            MOCK_UNCOMPRESSED_MODEL_DATA_STR, MOCK_JUMPSTART_ID, mock_code_dir
        )

        mock_s3_downloader_obj.download.assert_called_once_with(
            MOCK_UNCOMPRESSED_MODEL_DATA_STR, MOCK_TMP_DIR
        )
        mock_move_to_code_dir.assert_called_once_with(MOCK_TMP_DIR, mock_code_dir)
        mock_code_dir.joinpath.assert_called_once_with("config.json")
        self.assertEqual(existing_properties, {})
        self.assertEqual(hf_model_config, {})
        self.assertEqual(success, True)

    @patch("sagemaker.serve.model_server.djl_serving.prepare.S3Downloader")
    @patch("sagemaker.serve.model_server.djl_serving.prepare._tmpdir")
    @patch(
        "sagemaker.serve.model_server.djl_serving.prepare._read_existing_serving_properties",
        return_value={},
    )
    @patch("sagemaker.serve.model_server.djl_serving.prepare._move_to_code_dir")
    @patch("builtins.open", new_callable=mock_open, read_data="data")
    @patch("json.load", return_value={})
    def test_prepare_djl_js_resources_for_jumpstart_uncompressed_dict(
        self,
        mock_load,
        mock_open,
        mock_move_to_code_dir,
        mock_existing_props,
        mock_tmpdir,
        mock_s3_downloader,
    ):
        mock_code_dir = Mock()
        mock_config_json_file = Mock()
        mock_config_json_file.is_file.return_value = True
        mock_code_dir.joinpath.return_value = mock_config_json_file

        mock_s3_downloader_obj = Mock()
        mock_s3_downloader.return_value = mock_s3_downloader_obj

        mock_tmpdir_obj = Mock()
        mock_js_dir = Mock()
        mock_js_dir.return_value = MOCK_TMP_DIR
        type(mock_tmpdir_obj).__enter__ = PropertyMock(return_value=mock_js_dir)
        type(mock_tmpdir_obj).__exit__ = PropertyMock(return_value=Mock())
        mock_tmpdir.return_value = mock_tmpdir_obj

        existing_properties, hf_model_config, success = _copy_jumpstart_artifacts(
            MOCK_UNCOMPRESSED_MODEL_DATA_DICT, MOCK_JUMPSTART_ID, mock_code_dir
        )

        mock_s3_downloader_obj.download.assert_called_once_with(
            MOCK_UNCOMPRESSED_MODEL_DATA_STR_FOR_DICT, MOCK_TMP_DIR
        )
        mock_move_to_code_dir.assert_called_once_with(MOCK_TMP_DIR, mock_code_dir)
        mock_code_dir.joinpath.assert_called_once_with("config.json")
        self.assertEqual(existing_properties, {})
        self.assertEqual(hf_model_config, {})
        self.assertEqual(success, True)

    @patch("sagemaker.serve.model_server.djl_serving.prepare._tmpdir")
    @patch("sagemaker.serve.model_server.djl_serving.prepare._move_to_code_dir")
    def test_prepare_djl_js_resources_for_jumpstart_invalid_model_data(
        self, mock_move_to_code_dir, mock_tmpdir
    ):
        mock_code_dir = Mock()
        mock_tmpdir_obj = Mock()
        type(mock_tmpdir_obj).__enter__ = PropertyMock(return_value=Mock())
        type(mock_tmpdir_obj).__exit__ = PropertyMock(return_value=Mock())
        mock_tmpdir.return_value = mock_tmpdir_obj

        with self.assertRaises(ValueError) as context:
            _copy_jumpstart_artifacts(
                MOCK_INVALID_MODEL_DATA_DICT, MOCK_JUMPSTART_ID, mock_code_dir
            )

        assert not mock_move_to_code_dir.called
        self.assertTrue(
            "JumpStart model data compression format is unsupported" in str(context.exception)
        )

    @patch("sagemaker.serve.model_server.djl_serving.prepare.S3Downloader")
    @patch("sagemaker.serve.model_server.djl_serving.prepare._extract_js_resource")
    @patch("sagemaker.serve.model_server.djl_serving.prepare._tmpdir")
    @patch(
        "sagemaker.serve.model_server.djl_serving.prepare._read_existing_serving_properties",
        return_value={},
    )
    @patch("sagemaker.serve.model_server.djl_serving.prepare._move_to_code_dir")
    @patch("builtins.open", new_callable=mock_open, read_data="data")
    @patch("json.load", return_value={})
    def test_prepare_djl_js_resources_for_jumpstart_compressed_str(
        self,
        mock_load,
        mock_open,
        mock_move_to_code_dir,
        mock_existing_props,
        mock_tmpdir,
        mock_extract_js_resource,
        mock_s3_downloader,
    ):
        mock_code_dir = Mock()
        mock_config_json_file = Mock()
        mock_config_json_file.is_file.return_value = True
        mock_code_dir.joinpath.return_value = mock_config_json_file

        mock_s3_downloader_obj = Mock()
        mock_s3_downloader.return_value = mock_s3_downloader_obj

        mock_tmpdir_obj = Mock()
        mock_js_dir = Mock()
        mock_js_dir.return_value = MOCK_TMP_DIR
        type(mock_tmpdir_obj).__enter__ = PropertyMock(return_value=mock_js_dir)
        type(mock_tmpdir_obj).__exit__ = PropertyMock(return_value=Mock())
        mock_tmpdir.return_value = mock_tmpdir_obj

        existing_properties, hf_model_config, success = _copy_jumpstart_artifacts(
            MOCK_COMPRESSED_MODEL_DATA_STR, MOCK_JUMPSTART_ID, mock_code_dir
        )

        mock_s3_downloader_obj.download.assert_called_once_with(
            MOCK_COMPRESSED_MODEL_DATA_STR, MOCK_TMP_DIR
        )
        mock_extract_js_resource.assert_called_with(MOCK_TMP_DIR, MOCK_JUMPSTART_ID)
        mock_move_to_code_dir.assert_called_once_with(MOCK_TMP_DIR, mock_code_dir)
        mock_code_dir.joinpath.assert_called_once_with("config.json")
        self.assertEqual(existing_properties, {})
        self.assertEqual(hf_model_config, {})
        self.assertEqual(success, True)

    @patch("sagemaker.serve.model_server.djl_serving.prepare.Path")
    @patch("sagemaker.serve.model_server.djl_serving.prepare.shutil")
    def test_move_to_code_dir_success(self, mock_shutil, mock_path):
        mock_path_obj = Mock()
        mock_js_model_resources = Mock()
        mock_js_model_resources.glob.return_value = MOCK_DJL_JUMPSTART_GLOBED_RESOURCES
        mock_path_obj.joinpath.return_value = mock_js_model_resources
        mock_path.return_value = mock_path_obj

        mock_js_model_dir = ""
        mock_code_dir = Mock()
        _move_to_code_dir(mock_js_model_dir, mock_code_dir)

        mock_path_obj.joinpath.assert_called_once_with("model")

        expected_moves = [
            call("./inference.py", mock_code_dir),
            call("./serving.properties", mock_code_dir),
            call("./config.json", mock_code_dir),
        ]
        mock_shutil.move.assert_has_calls(expected_moves)

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
        _extract_js_resource(js_model_dir, MOCK_JUMPSTART_ID)

        mock_path.assert_called_once_with(js_model_dir)
        mock_path_obj.joinpath.assert_called_once_with(f"infer-prepack-{MOCK_JUMPSTART_ID}.tar.gz")
        mock_resource_obj.extractall.assert_called_once_with(path=js_model_dir, filter="data")
