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
from unittest import TestCase
from unittest.mock import Mock, patch

from docker.types import DeviceRequest
from sagemaker.serve.model_server.tei.server import LocalTeiServing, SageMakerTeiServing
from sagemaker.serve.utils.exceptions import LocalModelInvocationException

TEI_IMAGE = (
    "246618743249.dkr.ecr.us-west-2.amazonaws.com/tei:2.0.1-tei1.2.3-gpu-py310-cu122-ubuntu22.04"
)
MODEL_PATH = "model_path"
ENV_VAR = {"KEY": "VALUE"}
PAYLOAD = {
    "inputs": {
        "sourceSentence": "How cute your dog is!",
        "sentences": ["The mitochondria is the powerhouse of the cell.", "Your dog is so cute."],
    }
}
S3_URI = "s3://mock_model_data_uri"
SECRET_KEY = "secret_key"
INFER_RESPONSE = []


class TeiServerTests(TestCase):
    @patch("sagemaker.serve.model_server.tei.server.requests")
    def test_start_invoke_destroy_local_tei_server(self, mock_requests):
        mock_container = Mock()
        mock_docker_client = Mock()
        mock_docker_client.containers.run.return_value = mock_container

        local_tei_server = LocalTeiServing()
        mock_schema_builder = Mock()
        mock_schema_builder.input_serializer.serialize.return_value = PAYLOAD
        local_tei_server.schema_builder = mock_schema_builder

        local_tei_server._start_tei_serving(
            client=mock_docker_client,
            model_path=MODEL_PATH,
            secret_key=SECRET_KEY,
            image=TEI_IMAGE,
            env_vars=ENV_VAR,
        )

        mock_docker_client.containers.run.assert_called_once_with(
            TEI_IMAGE,
            shm_size="2G",
            device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
            network_mode="host",
            detach=True,
            auto_remove=True,
            volumes={PosixPath("model_path/code"): {"bind": "/opt/ml/model/", "mode": "rw"}},
            environment={
                "TRANSFORMERS_CACHE": "/opt/ml/model/",
                "HF_HOME": "/opt/ml/model/",
                "HUGGINGFACE_HUB_CACHE": "/opt/ml/model/",
                "KEY": "VALUE",
                "SAGEMAKER_SERVE_SECRET_KEY": "secret_key",
            },
        )

        mock_response = Mock()
        mock_requests.post.side_effect = lambda *args, **kwargs: mock_response
        mock_response.content = INFER_RESPONSE

        res = local_tei_server._invoke_tei_serving(
            request=PAYLOAD, content_type="application/json", accept="application/json"
        )

        self.assertEqual(res, INFER_RESPONSE)

    def test_tei_deep_ping(self):
        mock_predictor = Mock()
        mock_response = Mock()
        mock_schema_builder = Mock()

        mock_predictor.predict.side_effect = lambda *args, **kwargs: mock_response
        mock_schema_builder.sample_input = PAYLOAD

        local_tei_server = LocalTeiServing()
        local_tei_server.schema_builder = mock_schema_builder
        res = local_tei_server._tei_deep_ping(mock_predictor)

        self.assertEqual(res, (True, mock_response))

    def test_tei_deep_ping_invoke_ex(self):
        mock_predictor = Mock()
        mock_schema_builder = Mock()

        mock_predictor.predict.side_effect = lambda *args, **kwargs: exec(
            'raise(ValueError("422 Client Error: Unprocessable Entity for url:"))'
        )
        mock_schema_builder.sample_input = PAYLOAD

        local_tei_server = LocalTeiServing()
        local_tei_server.schema_builder = mock_schema_builder

        self.assertRaises(
            LocalModelInvocationException, lambda: local_tei_server._tei_deep_ping(mock_predictor)
        )

    def test_tei_deep_ping_ex(self):
        mock_predictor = Mock()

        mock_predictor.predict.side_effect = lambda *args, **kwargs: Exception()

        local_tei_server = LocalTeiServing()
        res = local_tei_server._tei_deep_ping(mock_predictor)

        self.assertEqual(res, (False, None))

    @patch("sagemaker.serve.model_server.tei.server.S3Uploader")
    def test_upload_artifacts_sagemaker_tei_server(self, mock_uploader):
        mock_session = Mock()
        mock_uploader.upload.side_effect = (
            lambda *args, **kwargs: "s3://sagemaker-us-west-2-123456789123/tei-2024-05-20-16-05-36-027/code"
        )

        s3_upload_path, env_vars = SageMakerTeiServing()._upload_tei_artifacts(
            model_path=MODEL_PATH,
            sagemaker_session=mock_session,
            s3_model_data_url=S3_URI,
            image=TEI_IMAGE,
            should_upload_artifacts=True,
        )

        mock_uploader.upload.assert_called_once()
        self.assertEqual(
            s3_upload_path,
            {
                "S3DataSource": {
                    "CompressionType": "None",
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://sagemaker-us-west-2-123456789123/tei-2024-05-20-16-05-36-027/code/",
                }
            },
        )
        self.assertIsNotNone(env_vars)
