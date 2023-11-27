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

from sagemaker.serve.model_server.triton.triton_builder import Triton
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.mode.function_pointers import Mode
import torch

TRITON_IMAGE = "301217895009.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tritonserver:23.02-py3"
MODEL_PATH = "/path/to/working/dir"
S3_UPLOAD_PATH = "s3://path/to/bucket"
ENV_VAR = {"KEY": "VALUE"}
ROLE_ARN = "ROLE_ARN"
pt_schema_builder = SchemaBuilder(sample_input=torch.rand(3, 4), sample_output=torch.rand(1, 10))

MOCK_SERVE_SETTINGS = Mock()
MOCK_SESSION = Mock()
MOCK_MODES = Mock()
MOCK_DEPLOY_WRAPPER = Mock()


class pytorch:
    pass


MOCK_PT_MODEL = Mock(spec=pytorch)
MOCK_TF_MODEL = Mock()


class TritonBuilderTests(TestCase):
    def setUp(self):

        # Prepare the _ServeSettings class
        MOCK_SERVE_SETTINGS.role_arn = ROLE_ARN

        # Prepare the triton builder class
        self.triton_builder = Triton()

        self.triton_builder.model = MOCK_PT_MODEL
        self.triton_builder.image_uri = TRITON_IMAGE
        self.triton_builder.mode = Mode.LOCAL_CONTAINER
        self.triton_builder.schema_builder = pt_schema_builder
        self.triton_builder.model_path = MODEL_PATH
        self.triton_builder.s3_upload_path = S3_UPLOAD_PATH
        self.triton_builder.serve_settings = MOCK_SERVE_SETTINGS
        self.triton_builder.env_vars = ENV_VAR
        self.triton_builder.sagemaker_session = MOCK_SESSION
        self.triton_builder.modes = MOCK_MODES
        self.triton_builder._model_builder_deploy_wrapper = MOCK_DEPLOY_WRAPPER
        self.triton_builder.inference_spec = None

        mock_export = Mock()
        self.triton_builder._export_pytorch_to_onnx = mock_export
        self.triton_builder._export_tf_to_onnx = mock_export

    @patch("sagemaker.serve.model_server.triton.triton_builder.Model")
    @patch("sagemaker.serve.model_server.triton.triton_builder.Path")
    @patch("sagemaker.serve.model_server.triton.triton_builder._get_available_gpus")
    @patch("sagemaker.serve.model_server.triton.triton_builder._detect_framework_and_version")
    def test_build_for_triton_pt(self, mock_detect_fw, mock_get_gpus, mock_path, mock_model):
        mock_model_path = Mock()
        mock_path.return_value = mock_model_path
        mock_model_path.exists.return_value = True
        mock_model_path.is_dir.return_value = True
        mock_model_path.joinpath.return_value = mock_model_path

        mock_detect_fw.return_value = ("pytorch", "2.0.1")

        self.triton_builder._build_for_triton()

        self.triton_builder._export_pytorch_to_onnx.assert_called_once_with(
            export_path=mock_model_path, model=MOCK_PT_MODEL, schema_builder=pt_schema_builder
        )

        mock_model.assert_called_with(
            image_uri=TRITON_IMAGE,
            model_data=S3_UPLOAD_PATH,
            role=ROLE_ARN,
            env=ENV_VAR,
            sagemaker_session=MOCK_SESSION,
            predictor_cls=self.triton_builder._get_triton_predictor,
        )

    @patch("sagemaker.serve.model_server.triton.triton_builder.Model")
    @patch("sagemaker.serve.model_server.triton.triton_builder.Path")
    @patch("sagemaker.serve.model_server.triton.triton_builder._get_available_gpus")
    @patch("sagemaker.serve.model_server.triton.triton_builder._detect_framework_and_version")
    def test_build_for_triton_tf(self, mock_detect_fw, mock_get_gpus, mock_path, mock_model):
        mock_model_path = Mock()
        mock_path.return_value = mock_model_path
        mock_model_path.exists.return_value = True
        mock_model_path.is_dir.return_value = True
        mock_model_path.joinpath.return_value = mock_model_path

        mock_detect_fw.return_value = ("tensorflow", "2.0.1")

        self.triton_builder.model = MOCK_TF_MODEL
        self.triton_builder._build_for_triton()

        self.triton_builder._export_tf_to_onnx.assert_called_once_with(
            export_path=mock_model_path, model=MOCK_TF_MODEL, schema_builder=pt_schema_builder
        )

        mock_model.assert_called_with(
            image_uri=TRITON_IMAGE,
            model_data=S3_UPLOAD_PATH,
            role=ROLE_ARN,
            env=ENV_VAR,
            sagemaker_session=MOCK_SESSION,
            predictor_cls=self.triton_builder._get_triton_predictor,
        )
