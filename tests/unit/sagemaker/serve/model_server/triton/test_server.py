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
from unittest.mock import Mock, patch, ANY

import numpy as np
from docker.types import DeviceRequest
from sagemaker.serve.model_server.triton.server import LocalTritonServer, SageMakerTritonServer

GPU_TRITON_IMAGE = "301217895009.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tritonserver:23.02-py3"
CPU_TRITON_IMAGE = (
    "301217895009.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tritonserver:23.02-py3-cpu"
)
MODEL_PATH = "model_path"
MODEL_REPO = f"{MODEL_PATH}/model_repository"
ENV_VAR = {"KEY": "VALUE"}
_SHM_SIZE = "2G"
PAYLOAD = np.random.rand(3, 4).astype(dtype=np.float32)
S3_URI = "s3://mock_model_data_uri"
DTYPE = "TYPE_FP32"
SECRET_KEY = "secret_key"


INFER_RESPONSE = {"outputs": [{"name": "output_name"}]}


class TritonServerTests(TestCase):
    @patch("sagemaker.serve.model_server.triton.server.importlib")
    def test_start_invoke_destroy_local_triton_server_gpu(self, mock_importlib):
        mock_triton_client = Mock()
        mock_importlib.import_module.side_effect = lambda module_name: (
            mock_triton_client if module_name == "tritonclient.http" else None
        )

        mock_container = Mock()
        mock_docker_client = Mock()
        mock_docker_client.containers.run.return_value = mock_container

        # Launch container in GPU mode

        local_triton_server = LocalTritonServer()
        mock_schema_builder = Mock()
        mock_schema_builder.input_serializer.serialize.return_value = PAYLOAD
        mock_schema_builder._input_triton_dtype = DTYPE
        mock_schema_builder._output_triton_dtype = DTYPE
        local_triton_server.schema_builder = mock_schema_builder

        local_triton_server._start_triton_server(
            docker_client=mock_docker_client,
            model_path=MODEL_PATH,
            image_uri=GPU_TRITON_IMAGE,
            env_vars=ENV_VAR,
        )

        mock_docker_client.containers.run.assert_called_once_with(
            image=GPU_TRITON_IMAGE,
            command=["tritonserver", "--model-repository=/models"],
            shm_size=_SHM_SIZE,
            device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
            network_mode="host",
            detach=True,
            auto_remove=True,
            volumes={MODEL_REPO: {"bind": "/models", "mode": "rw"}},
            environment=ENV_VAR,
        )

        # Try to ping container
        mock_client = Mock()
        mock_response = Mock()
        mock_client.infer.return_value = mock_response
        mock_response.get_response.return_value = INFER_RESPONSE
        mock_triton_client.InferenceServerClient.return_value = mock_client

        mock_request = Mock()
        mock_triton_client.InferInput.return_value = mock_request

        local_triton_server._invoke_triton_server(payload=PAYLOAD)

        mock_triton_client.InferenceServerClient.assert_called_once_with(url="localhost:8000")
        mock_triton_client.InferInput.assert_called_once_with(
            "input_1", PAYLOAD.shape, datatype="FP32"
        )
        mock_request.set_data_from_numpy.assert_called_once_with(PAYLOAD, binary_data=True)
        mock_client.infer.assert_called_with(model_name="model", inputs=[mock_request])

    @patch("sagemaker.serve.model_server.triton.server.importlib")
    def test_start_invoke_destroy_local_triton_server_cpu(self, mock_importlib):
        mock_triton_client = Mock()
        mock_importlib.import_module.side_effect = lambda module_name: (
            mock_triton_client if module_name == "tritonclient.http" else None
        )

        mock_container = Mock()
        mock_docker_client = Mock()
        mock_docker_client.containers.run.return_value = mock_container

        # Launch container in CPU mode

        local_triton_server = LocalTritonServer()
        mock_schema_builder = Mock()
        mock_schema_builder.input_serializer.serialize.return_value = PAYLOAD
        mock_schema_builder._input_triton_dtype = DTYPE
        local_triton_server.schema_builder = mock_schema_builder

        local_triton_server._start_triton_server(
            docker_client=mock_docker_client,
            model_path=MODEL_PATH,
            image_uri=CPU_TRITON_IMAGE,
            env_vars=ENV_VAR,
        )

        mock_docker_client.containers.run.assert_called_once_with(
            image=CPU_TRITON_IMAGE,
            command=["tritonserver", "--model-repository=/models"],
            shm_size=_SHM_SIZE,
            network_mode="host",
            detach=True,
            auto_remove=True,
            volumes={MODEL_REPO: {"bind": "/models", "mode": "rw"}},
            environment=ENV_VAR,
        )

        # Try to ping container
        mock_client = Mock()
        mock_response = Mock()
        mock_client.infer.return_value = mock_response
        mock_response.get_response.return_value = INFER_RESPONSE
        mock_triton_client.InferenceServerClient.return_value = mock_client

        mock_request = Mock()
        mock_triton_client.InferInput.return_value = mock_request

        local_triton_server._invoke_triton_server(payload=PAYLOAD)

        mock_triton_client.InferenceServerClient.assert_called_once_with(url="localhost:8000")
        mock_triton_client.InferInput.assert_called_once_with(
            "input_1", PAYLOAD.shape, datatype="FP32"
        )
        mock_request.set_data_from_numpy.assert_called_once_with(PAYLOAD, binary_data=True)
        mock_client.infer.assert_called_with(model_name="model", inputs=[mock_request])

    @patch("sagemaker.serve.model_server.triton.server.platform")
    @patch("sagemaker.serve.model_server.triton.server.upload")
    def test_upload_artifacts_sagemaker_triton_server(self, mock_upload, mock_platform):
        mock_session = Mock()
        mock_platform.python_version.return_value = "3.8"
        mock_upload.side_effect = lambda session, repo, bucket, prefix: (
            S3_URI
            if session == mock_session
            and repo == MODEL_PATH + "/model_repository"
            and bucket == "mock_model_data_uri"
            else None
        )

        s3_upload_path, env_vars = SageMakerTritonServer()._upload_triton_artifacts(
            model_path=MODEL_PATH,
            sagemaker_session=mock_session,
            s3_model_data_url=S3_URI,
            image=GPU_TRITON_IMAGE,
            should_upload_artifacts=True,
        )

        mock_upload.assert_called_once_with(mock_session, MODEL_REPO, "mock_model_data_uri", ANY)
        self.assertEqual(s3_upload_path, S3_URI)
        self.assertEqual(env_vars.get("SAGEMAKER_TRITON_DEFAULT_MODEL_NAME"), "model")
        self.assertEqual(env_vars.get("TRITON_MODEL_DIR"), "/opt/ml/model/model")
        self.assertNotIn("SAGEMAKER_SERVE_SECRET_KEY", env_vars)
        self.assertEqual(env_vars.get("LOCAL_PYTHON"), "3.8")
