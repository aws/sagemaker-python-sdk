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
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from sagemaker.serve.model_server.multi_model_server.prepare import _create_dir_structure

from sagemaker.serve.model_server.multi_model_server.server import (
    LocalMultiModelServer,
)

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


class MultiModelServerPrepareTests(TestCase):
    def test_start_invoke_destroy_local_multi_model_server(self):
        mock_container = Mock()
        mock_docker_client = Mock()
        mock_docker_client.containers.run.return_value = mock_container

        local_multi_model_server = LocalMultiModelServer()
        mock_schema_builder = Mock()
        mock_schema_builder.input_serializer.serialize.return_value = PAYLOAD
        local_multi_model_server.schema_builder = mock_schema_builder

        local_multi_model_server._start_serving(
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

    @patch("sagemaker.serve.model_server.multi_model_server.prepare._check_disk_space")
    @patch("sagemaker.serve.model_server.multi_model_server.prepare._check_docker_disk_usage")
    @patch("sagemaker.serve.model_server.multi_model_server.prepare.Path")
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

    @patch("sagemaker.serve.model_server.multi_model_server.prepare.Path")
    def test_create_dir_structure_invalid_path(self, mock_path):
        mock_model_path = Mock()
        mock_model_path.exists.return_value = True
        mock_model_path.is_dir.return_value = False
        mock_path.return_value = mock_model_path

        with self.assertRaises(ValueError) as context:
            _create_dir_structure(mock_model_path)

        self.assertEquals("model_dir is not a valid directory", str(context.exception))
