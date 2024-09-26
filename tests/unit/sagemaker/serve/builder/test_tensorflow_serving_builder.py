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
from unittest.mock import MagicMock, patch

import unittest
from pathlib import Path

from sagemaker.serve import ModelBuilder, ModelServer


class TestTransformersBuilder(unittest.TestCase):
    def setUp(self):
        self.instance = ModelBuilder()
        self.instance.model_server = ModelServer.TENSORFLOW_SERVING
        self.instance.model_path = "/fake/model/path"
        self.instance.image_uri = "fake_image_uri"
        self.instance.s3_upload_path = "s3://bucket/path"
        self.instance.serve_settings = MagicMock(role_arn="fake_role_arn")
        self.instance.schema_builder = MagicMock()
        self.instance.env_vars = {}
        self.instance.sagemaker_session = MagicMock()
        self.instance.image_config = {}
        self.instance.vpc_config = {}
        self.instance.modes = {}
        self.instance.name = "model-name-mock-uuid-hex"

    @patch("os.makedirs")
    @patch("os.path.exists")
    @patch("sagemaker.serve.builder.tf_serving_builder.save_pkl")
    def test_save_schema_builder(self, mock_save_pkl, mock_exists, mock_makedirs):
        mock_exists.return_value = False
        self.instance._save_schema_builder()
        mock_makedirs.assert_called_once_with(self.instance.model_path)
        code_path = Path(self.instance.model_path).joinpath("code")
        mock_save_pkl.assert_called_once_with(code_path, self.instance.schema_builder)

    @patch("sagemaker.serve.builder.tf_serving_builder.TensorflowServing._get_client_translators")
    @patch("sagemaker.serve.builder.tf_serving_builder.TensorFlowPredictor")
    def test_get_tensorflow_predictor(self, mock_predictor, mock_get_marshaller):
        endpoint_name = "test_endpoint"
        predictor = self.instance._get_tensorflow_predictor(
            endpoint_name, self.instance.sagemaker_session
        )
        mock_predictor.assert_called_once_with(
            endpoint_name=endpoint_name,
            sagemaker_session=self.instance.sagemaker_session,
            serializer=self.instance.schema_builder.custom_input_translator,
            deserializer=self.instance.schema_builder.custom_output_translator,
        )
        self.assertEqual(predictor, mock_predictor.return_value)

    @patch("sagemaker.serve.builder.tf_serving_builder.TensorFlowModel")
    def test_create_tensorflow_model(self, mock_model):
        model = self.instance._create_tensorflow_model()
        mock_model.assert_called_once_with(
            image_uri=self.instance.image_uri,
            image_config=self.instance.image_config,
            vpc_config=self.instance.vpc_config,
            model_data=self.instance.s3_upload_path,
            role=self.instance.serve_settings.role_arn,
            env=self.instance.env_vars,
            sagemaker_session=self.instance.sagemaker_session,
            predictor_cls=self.instance._get_tensorflow_predictor,
            name="model-name-mock-uuid-hex",
        )
        self.assertEqual(model, mock_model.return_value)
