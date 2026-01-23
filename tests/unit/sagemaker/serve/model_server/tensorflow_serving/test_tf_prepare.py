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
import pytest

from sagemaker.serve.model_server.tensorflow_serving.prepare import prepare_for_tf_serving

MODEL_PATH = "/path/to/your/model/dir"
SHARED_LIBS = ["/path/to/shared/libs"]
DEPENDENCIES = {"dependencies": "requirements.txt"}
INFERENCE_SPEC = Mock()
IMAGE_URI = "mock_image_uri"
XGB_1P_IMAGE_URI = "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow:1.7-1"
INFERENCE_SPEC.prepare = Mock(return_value=None)

SECRET_KEY = "secret-key"

mock_session = Mock()


class PrepareForTensorflowServingTests(TestCase):
    def setUp(self):
        INFERENCE_SPEC.reset_mock()

    @patch("builtins.open", new_callable=mock_open, read_data=b"{}")
    @patch("sagemaker.serve.model_server.tensorflow_serving.prepare._move_contents")
    @patch(
        "sagemaker.serve.model_server.tensorflow_serving.prepare."
        "_get_saved_model_path_for_tensorflow_and_keras_flavor"
    )
    @patch("sagemaker.serve.model_server.tensorflow_serving.prepare._MetaData")
    @patch("sagemaker.serve.model_server.tensorflow_serving.prepare.compute_hash")
    @patch("sagemaker.serve.model_server.tensorflow_serving.prepare.capture_dependencies")
    @patch("sagemaker.serve.model_server.tensorflow_serving.prepare.shutil")
    @patch("sagemaker.serve.model_server.tensorflow_serving.prepare.Path")
    def test_prepare_happy(
        self,
        mock_path,
        mock_shutil,
        mock_capture_dependencies,
        mock_compute_hash,
        mock_metadata,
        mock_get_saved_model_path,
        mock_move_contents,
        mock_open,
    ):

        mock_path_instance = mock_path.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.joinpath.return_value = Mock()
        mock_get_saved_model_path.return_value = MODEL_PATH + "/1/"

        secret_key = prepare_for_tf_serving(
            model_path=MODEL_PATH,
            shared_libs=SHARED_LIBS,
            dependencies=DEPENDENCIES,
        )

        mock_path_instance.mkdir.assert_not_called()
        self.assertEqual(secret_key, "")

    @patch("builtins.open", new_callable=mock_open, read_data=b"{}")
    @patch("sagemaker.serve.model_server.tensorflow_serving.prepare._move_contents")
    @patch(
        "sagemaker.serve.model_server.tensorflow_serving.prepare."
        "_get_saved_model_path_for_tensorflow_and_keras_flavor"
    )
    @patch("sagemaker.serve.model_server.tensorflow_serving.prepare._MetaData")
    @patch("sagemaker.serve.model_server.tensorflow_serving.prepare.compute_hash")
    @patch("sagemaker.serve.model_server.tensorflow_serving.prepare.capture_dependencies")
    @patch("sagemaker.serve.model_server.tensorflow_serving.prepare.shutil")
    @patch("sagemaker.serve.model_server.tensorflow_serving.prepare.Path")
    def test_prepare_saved_model_not_found(
        self,
        mock_path,
        mock_shutil,
        mock_capture_dependencies,
        mock_compute_hash,
        mock_metadata,
        mock_get_saved_model_path,
        mock_move_contents,
        mock_open,
    ):

        mock_path_instance = mock_path.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.joinpath.return_value = Mock()
        mock_get_saved_model_path.return_value = None

        with pytest.raises(
            ValueError, match="SavedModel is not found for Tensorflow or Keras flavor."
        ):
            prepare_for_tf_serving(
                model_path=MODEL_PATH,
                shared_libs=SHARED_LIBS,
                dependencies=DEPENDENCIES,
            )
