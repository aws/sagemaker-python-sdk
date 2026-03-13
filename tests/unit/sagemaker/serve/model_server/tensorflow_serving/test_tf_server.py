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
from unittest.mock import Mock, patch, ANY

import numpy as np

from sagemaker.serve.model_server.tensorflow_serving.server import (
    LocalTensorflowServing,
    SageMakerTensorflowServing,
)

CPU_TF_IMAGE = "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:2.14.1-cpu"
MODEL_PATH = "model_path"
MODEL_REPO = f"{MODEL_PATH}/1"
ENV_VAR = {"KEY": "VALUE"}
_SHM_SIZE = "2G"
PAYLOAD = np.random.rand(3, 4).astype(dtype=np.float32)
S3_URI = "s3://mock_model_data_uri"
DTYPE = "TYPE_FP32"
SECRET_KEY = "secret_key"

INFER_RESPONSE = {"outputs": [{"name": "output_name"}]}


class TensorflowservingServerTests(TestCase):
    def test_start_invoke_destroy_local_tensorflow_serving_server(self):
        mock_container = Mock()
        mock_docker_client = Mock()
        mock_docker_client.containers.run.return_value = mock_container

        local_tensorflow_server = LocalTensorflowServing()
        mock_schema_builder = Mock()
        mock_schema_builder.input_serializer.serialize.return_value = PAYLOAD
        local_tensorflow_server.schema_builder = mock_schema_builder

        local_tensorflow_server._start_tensorflow_serving(
            client=mock_docker_client,
            model_path=MODEL_PATH,
            env_vars=ENV_VAR,
            image=CPU_TF_IMAGE,
        )

        mock_docker_client.containers.run.assert_called_once_with(
            CPU_TF_IMAGE,
            "serve",
            detach=True,
            auto_remove=True,
            network_mode="host",
            volumes={PosixPath("model_path"): {"bind": "/opt/ml/model", "mode": "rw"}},
            environment={
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                "SAGEMAKER_PROGRAM": "inference.py",
                "LOCAL_PYTHON": platform.python_version(),
                "KEY": "VALUE",
            },
        )

    @patch("sagemaker.serve.model_server.tensorflow_serving.server.platform")
    @patch("sagemaker.serve.model_server.tensorflow_serving.server.upload")
    def test_upload_artifacts_sagemaker_triton_server(self, mock_upload, mock_platform):
        mock_session = Mock()
        mock_platform.python_version.return_value = "3.8"
        mock_upload.side_effect = lambda session, repo, bucket, prefix: (
            S3_URI
            if session == mock_session and repo == MODEL_PATH and bucket == "mock_model_data_uri"
            else None
        )

        (
            s3_upload_path,
            env_vars,
        ) = SageMakerTensorflowServing()._upload_tensorflow_serving_artifacts(
            model_path=MODEL_PATH,
            sagemaker_session=mock_session,
            s3_model_data_url=S3_URI,
            image=CPU_TF_IMAGE,
            should_upload_artifacts=True,
        )

        mock_upload.assert_called_once_with(mock_session, MODEL_PATH, "mock_model_data_uri", ANY)
        self.assertEqual(s3_upload_path, S3_URI)
        self.assertNotIn("SAGEMAKER_SERVE_SECRET_KEY", env_vars)
        self.assertEqual(env_vars.get("LOCAL_PYTHON"), "3.8")
