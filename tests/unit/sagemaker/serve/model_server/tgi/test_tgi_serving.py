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
from unittest.mock import Mock, patch
from pathlib import Path
from sagemaker.serve.model_server.tgi.server import LocalTgiServing, SageMakerTgiServing

MOCK_IMAGE = "mock image"
MOCK_MODEL_PATH = "mock model path"
MOCK_SECRET_KEY = "mock secret key"
MOCK_ENV_VARS = {"mock key": "mock value"}
MOCK_SAGEMAKER_SESSION = Mock()
MOCK_S3_MODEL_DATA_URL = "mock s3 path"
MOCK_MODEL_DATA_URL = "mock model data url"

EXPECTED_MODE_DIR_BINDING = "/opt/ml/model/"
EXPECTED_SHM_SIZE = "2G"
EXPECTED_UPDATED_ENV_VARS = {
    "HF_HOME": "/opt/ml/model/",
    "HUGGINGFACE_HUB_CACHE": "/opt/ml/model/",
    "mock key": "mock value",
}
EXPECTED_MODEL_DATA = {
    "S3DataSource": {
        "CompressionType": "None",
        "S3DataType": "S3Prefix",
        "S3Uri": MOCK_MODEL_DATA_URL + "/",
    }
}


class TestLocalTgiServing(TestCase):
    def test_tgi_serving_runs_container_non_jumpstart_success(self):
        # WHERE
        mock_container_client = Mock()
        mock_container = Mock()
        mock_container_client.containers.run.return_value = mock_container
        localTgiServing = LocalTgiServing()

        # WHEN
        localTgiServing._start_tgi_serving(
            mock_container_client,
            MOCK_IMAGE,
            MOCK_MODEL_PATH,
            MOCK_ENV_VARS,
            False,
        )

        # THEN
        mock_container_client.containers.run.assert_called_once_with(
            MOCK_IMAGE,
            shm_size=EXPECTED_SHM_SIZE,
            device_requests=[
                {
                    "Driver": "",
                    "Count": -1,
                    "DeviceIDs": [],
                    "Capabilities": [["gpu"]],
                    "Options": {},
                }
            ],
            network_mode="host",
            detach=True,
            auto_remove=True,
            volumes={
                Path(MOCK_MODEL_PATH).joinpath("code"): {
                    "bind": EXPECTED_MODE_DIR_BINDING,
                    "mode": "rw",
                }
            },
            environment=EXPECTED_UPDATED_ENV_VARS,
        )
        assert localTgiServing.container == mock_container

    def test_tgi_serving_runs_container_jumpstart_success(self):
        # WHERE
        mock_container_client = Mock()
        mock_container = Mock()
        mock_container_client.containers.run.return_value = mock_container
        localTgiServing = LocalTgiServing()

        # WHEN
        localTgiServing._start_tgi_serving(
            mock_container_client, MOCK_IMAGE, MOCK_MODEL_PATH, MOCK_ENV_VARS, True
        )

        # THEN
        mock_container_client.containers.run.assert_called_once_with(
            MOCK_IMAGE,
            ["--model-id", EXPECTED_MODE_DIR_BINDING],
            shm_size=EXPECTED_SHM_SIZE,
            device_requests=[
                {
                    "Driver": "",
                    "Count": -1,
                    "DeviceIDs": [],
                    "Capabilities": [["gpu"]],
                    "Options": {},
                }
            ],
            network_mode="host",
            detach=True,
            auto_remove=True,
            volumes={
                Path(MOCK_MODEL_PATH).joinpath("code"): {
                    "bind": EXPECTED_MODE_DIR_BINDING,
                    "mode": "rw",
                }
            },
            environment=MOCK_ENV_VARS,
        )
        assert localTgiServing.container == mock_container


class TestSageMakerTgiServing(TestCase):

    @patch("sagemaker.serve.model_server.tgi.server._is_s3_uri")
    @patch("sagemaker.serve.model_server.tgi.server.parse_s3_url")
    @patch("sagemaker.serve.model_server.tgi.server.fw_utils")
    @patch("sagemaker.serve.model_server.tgi.server.determine_bucket_and_prefix")
    @patch("sagemaker.serve.model_server.tgi.server.s3_path_join")
    @patch("sagemaker.serve.model_server.tgi.server.S3Uploader")
    def test_tgi_serving_upload_tgi_artifacts_s3_url_passed_success(
        self,
        mock_s3_uploader,
        mock_s3_path_join,
        mock_determine_bucket_and_prefix,
        mock_fw_utils,
        mock_parse_s3_url,
        mock_is_s3_uri,
    ):
        # WHERE
        mock_is_s3_uri.return_value = False
        mock_parse_s3_url.return_value = ("mock_bucket_1", "mock_prefix_1")
        mock_fw_utils.model_code_key_prefix.return_value = "mock_code_key_prefix"
        mock_determine_bucket_and_prefix.return_value = ("mock_bucket_2", "mock_prefix_2")
        mock_s3_path_join.return_value = "mock_s3_location"
        mock_s3_uploader.upload.return_value = MOCK_MODEL_DATA_URL

        sagemakerTgiServing = SageMakerTgiServing()

        # WHEN
        ret_model_data, ret_env_vars = sagemakerTgiServing._upload_tgi_artifacts(
            MOCK_MODEL_PATH,
            MOCK_SAGEMAKER_SESSION,
            False,
            MOCK_S3_MODEL_DATA_URL,
            MOCK_IMAGE,
            MOCK_ENV_VARS,
            True,
        )

        # THEN
        mock_is_s3_uri.assert_called_once_with(MOCK_MODEL_PATH)
        mock_parse_s3_url.assert_called_once_with(url=MOCK_S3_MODEL_DATA_URL)
        mock_fw_utils.model_code_key_prefix.assert_called_once_with(
            "mock_prefix_1", None, MOCK_IMAGE
        )
        mock_determine_bucket_and_prefix.assert_called_once_with(
            bucket="mock_bucket_1",
            key_prefix="mock_code_key_prefix",
            sagemaker_session=MOCK_SAGEMAKER_SESSION,
        )
        mock_s3_path_join.assert_called_once_with("s3://", "mock_bucket_2", "mock_prefix_2", "code")
        mock_s3_uploader.upload.assert_called_once_with(
            f"{MOCK_MODEL_PATH}/code", "mock_s3_location", None, MOCK_SAGEMAKER_SESSION
        )
        assert ret_model_data == EXPECTED_MODEL_DATA
        assert ret_env_vars == EXPECTED_UPDATED_ENV_VARS

    @patch("sagemaker.serve.model_server.tgi.server._is_s3_uri")
    @patch("sagemaker.serve.model_server.tgi.server.parse_s3_url")
    @patch("sagemaker.serve.model_server.tgi.server.fw_utils")
    @patch("sagemaker.serve.model_server.tgi.server.determine_bucket_and_prefix")
    @patch("sagemaker.serve.model_server.tgi.server.s3_path_join")
    @patch("sagemaker.serve.model_server.tgi.server.S3Uploader")
    def test_tgi_serving_upload_tgi_artifacts_jumpstart_success(
        self,
        mock_s3_uploader,
        mock_s3_path_join,
        mock_determine_bucket_and_prefix,
        mock_fw_utils,
        mock_parse_s3_url,
        mock_is_s3_uri,
    ):
        # WHERE
        mock_is_s3_uri.return_value = False
        mock_parse_s3_url.return_value = ("mock_bucket_1", "mock_prefix_1")
        mock_fw_utils.model_code_key_prefix.return_value = "mock_code_key_prefix"
        mock_determine_bucket_and_prefix.return_value = ("mock_bucket_2", "mock_prefix_2")
        mock_s3_path_join.return_value = "mock_s3_location"
        mock_s3_uploader.upload.return_value = MOCK_MODEL_DATA_URL

        sagemakerTgiServing = SageMakerTgiServing()

        # WHEN
        ret_model_data, ret_env_vars = sagemakerTgiServing._upload_tgi_artifacts(
            MOCK_MODEL_PATH,
            MOCK_SAGEMAKER_SESSION,
            True,
            MOCK_S3_MODEL_DATA_URL,
            MOCK_IMAGE,
            MOCK_ENV_VARS,
            True,
        )

        # THEN
        mock_is_s3_uri.assert_called_once_with(MOCK_MODEL_PATH)
        mock_parse_s3_url.assert_called_once_with(url=MOCK_S3_MODEL_DATA_URL)
        mock_fw_utils.model_code_key_prefix.assert_called_once_with(
            "mock_prefix_1", None, MOCK_IMAGE
        )
        mock_determine_bucket_and_prefix.assert_called_once_with(
            bucket="mock_bucket_1",
            key_prefix="mock_code_key_prefix",
            sagemaker_session=MOCK_SAGEMAKER_SESSION,
        )
        mock_s3_path_join.assert_called_once_with("s3://", "mock_bucket_2", "mock_prefix_2", "code")
        mock_s3_uploader.upload.assert_called_once_with(
            f"{MOCK_MODEL_PATH}/code", "mock_s3_location", None, MOCK_SAGEMAKER_SESSION
        )
        assert ret_model_data == EXPECTED_MODEL_DATA
        assert ret_env_vars == {}

    @patch("sagemaker.serve.model_server.tgi.server._is_s3_uri")
    @patch("sagemaker.serve.model_server.tgi.server.parse_s3_url")
    @patch("sagemaker.serve.model_server.tgi.server.fw_utils")
    @patch("sagemaker.serve.model_server.tgi.server.determine_bucket_and_prefix")
    @patch("sagemaker.serve.model_server.tgi.server.s3_path_join")
    @patch("sagemaker.serve.model_server.tgi.server.S3Uploader")
    def test_tgi_serving_upload_tgi_artifacts(
        self,
        mock_s3_uploader,
        mock_s3_path_join,
        mock_determine_bucket_and_prefix,
        mock_fw_utils,
        mock_parse_s3_url,
        mock_is_s3_uri,
    ):
        # WHERE
        mock_is_s3_uri.return_value = True

        sagemakerTgiServing = SageMakerTgiServing()

        # WHEN
        ret_model_data, ret_env_vars = sagemakerTgiServing._upload_tgi_artifacts(
            MOCK_MODEL_PATH,
            MOCK_SAGEMAKER_SESSION,
            False,
            MOCK_S3_MODEL_DATA_URL,
            MOCK_IMAGE,
            MOCK_ENV_VARS,
            True,
        )

        # THEN
        mock_is_s3_uri.assert_called_once_with(MOCK_MODEL_PATH)
        assert not mock_parse_s3_url.called
        assert not mock_fw_utils.model_code_key_prefix.called
        assert not mock_determine_bucket_and_prefix.called
        assert not mock_s3_path_join.called
        assert not mock_s3_uploader.upload.called
        assert ret_model_data == {
            "S3DataSource": {
                "CompressionType": "None",
                "S3DataType": "S3Prefix",
                "S3Uri": MOCK_MODEL_PATH + "/",
            }
        }
        assert ret_env_vars == EXPECTED_UPDATED_ENV_VARS
