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

from pathlib import PosixPath
import platform
import numpy as np

from unittest import TestCase
from unittest.mock import Mock, PropertyMock, patch, mock_open

from sagemaker.serve.model_server.djl_serving.server import (
    LocalDJLServing,
)
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

CPU_TF_IMAGE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-inference:2.0.0-transformers4.28.1-cpu-py310-ubuntu20.04"
)
MODEL_PATH = "model_path"
MODEL_REPO = f"{MODEL_PATH}/1"
ENV_VAR = {"KEY": "VALUE"}
PAYLOAD = np.random.rand(3, 4).astype(dtype=np.float32)
DTYPE = "TYPE_FP32"
SECRET_KEY = "secret_key"
INFER_RESPONSE = {"outputs": [{"name": "output_name"}]}


class DjlPrepareTests(TestCase):
    def test_start_invoke_destroy_local_djl_server(self):
        mock_container = Mock()
        mock_docker_client = Mock()
        mock_docker_client.containers.run.return_value = mock_container

        local_djl_server = LocalDJLServing()
        mock_schema_builder = Mock()
        mock_schema_builder.input_serializer.serialize.return_value = PAYLOAD
        local_djl_server.schema_builder = mock_schema_builder

        local_djl_server._start_serving(
            client=mock_docker_client,
            model_path=MODEL_PATH,
            secret_key=SECRET_KEY,
            env_vars=ENV_VAR,
            image=CPU_TF_IMAGE,
        )

        mock_docker_client.containers.run.assert_called_once_with(
            CPU_TF_IMAGE,
            "serve",
            network_mode="host",
            detach=True,
            auto_remove=True,
            volumes={PosixPath("model_path/code"): {"bind": "/opt/ml/model/", "mode": "rw"}},
            environment={
                "KEY": "VALUE",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SERVE_SECRET_KEY": "secret_key",
                "LOCAL_PYTHON": platform.python_version(),
            },
        )

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
