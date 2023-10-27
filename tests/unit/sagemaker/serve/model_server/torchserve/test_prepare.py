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
from unittest.mock import Mock, patch, mock_open

from sagemaker.serve.model_server.torchserve.prepare import prepare_for_torchserve

MODEL_PATH = "/path/to/your/model/dir"
SHARED_LIBS = ["/path/to/shared/libs"]
DEPENDENCIES = "dependencies"
INFERENCE_SPEC = Mock()
INFERENCE_SPEC.prepare = Mock(return_value=None)

SECRET_KEY = "secret-key"

mock_session = Mock()


class PrepareForTorchServeTests(TestCase):
    @patch("builtins.open", new_callable=mock_open, read_data=b"{}")
    @patch("sagemaker.serve.model_server.torchserve.prepare.prepare_wheel")
    @patch("sagemaker.serve.model_server.torchserve.prepare._MetaData")
    @patch("sagemaker.serve.model_server.torchserve.prepare.compute_hash")
    @patch("sagemaker.serve.model_server.torchserve.prepare.generate_secret_key")
    @patch("sagemaker.serve.model_server.torchserve.prepare.capture_dependencies")
    @patch("sagemaker.serve.model_server.torchserve.prepare.shutil")
    @patch("sagemaker.serve.model_server.torchserve.prepare.Path")
    def test_prepare_happy(
        self,
        mock_path,
        mock_shutil,
        mock_capture_dependencies,
        mock_generate_secret_key,
        mock_compute_hash,
        mock_metadata,
        mock_prepare_whl,
        mock_open,
    ):

        mock_path_instance = mock_path.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.joinpath.return_value = Mock()

        mock_generate_secret_key.return_value = SECRET_KEY

        secret_key = prepare_for_torchserve(
            model_path=MODEL_PATH,
            shared_libs=SHARED_LIBS,
            dependencies=DEPENDENCIES,
            session=mock_session,
            inference_spec=INFERENCE_SPEC,
        )

        mock_path_instance.mkdir.assert_not_called()
        INFERENCE_SPEC.prepare.assert_called_once()
        self.assertEquals(secret_key, SECRET_KEY)
