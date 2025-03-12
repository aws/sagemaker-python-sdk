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

from unittest.mock import MagicMock, patch, Mock, mock_open, ANY

import unittest
from pathlib import Path
from copy import deepcopy

import deepdiff
import pytest
from sagemaker.enums import EndpointType

from sagemaker.async_inference.async_inference_config import AsyncInferenceConfig
from sagemaker.batch_inference.batch_transform_inference_config import BatchTransformInferenceConfig

from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements

from sagemaker.serverless.serverless_inference_config import ServerlessInferenceConfig

from sagemaker.model import Model

from sagemaker.serve import SchemaBuilder
from sagemaker.serve.builder.model_builder import ModelBuilder
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.model_format.mlflow.constants import MLFLOW_TRACKING_ARN
from sagemaker.serve.utils import task
from sagemaker.serve.utils.exceptions import TaskNotFoundException
from sagemaker.serve.utils.predictors import TensorflowServingLocalPredictor
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.validations.optimization import _validate_optimization_configuration
from tests.unit.sagemaker.serve.constants import MOCK_IMAGE_CONFIG, MOCK_VPC_CONFIG

schema_builder = MagicMock()
mock_inference_spec = Mock()
mock_fw_model = Mock()
module = "module"
class_name = "class_name"
mock_fw_model.__class__.__module__ = module
mock_fw_model.__class__.__name__ = class_name
MODEL_PATH = "test_path"
CODE_PATH = Path(MODEL_PATH + "/code")
ENV_VAR_PAIR = [("some key", "some value")]
model_data = MagicMock()
framework = "framework"
version = "version"
ENV_VARS = {"some key": "some value", "MODEL_CLASS_NAME": f"{module}.{class_name}"}
ENV_VARS_INF_SPEC = {"some key": "some value"}
INSTANCE_GPU_INFO = (2, 8)

mock_image_uri = "abcd/efghijk"
mock_1p_dlc_image_uri = "763104351884.dkr.ecr.us-east-1.amazonaws.com"
mock_role_arn = "arn:aws:iam::123456789012:role/SageMakerRole"
mock_s3_model_data_url = "sample s3 data url"
mock_secret_key = "mock_secret_key"
mock_instance_type = "mock instance type"

supported_model_servers = {
    ModelServer.TORCHSERVE,
    ModelServer.TRITON,
    ModelServer.DJL_SERVING,
    ModelServer.TENSORFLOW_SERVING,
    ModelServer.MMS,
    ModelServer.TGI,
    ModelServer.TEI,
}

mock_session = MagicMock()

RESOURCE_REQUIREMENTS = ResourceRequirements(
    requests={
        "num_cpus": 0.5,
        "memory": 512,
        "copies": 2,
    },
    limits={},
)


class TestModelBuilder(unittest.TestCase):
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_validation_cannot_set_both_model_and_inference_spec(self, mock_serveSettings):
        builder = ModelBuilder(inference_spec="some value", model=Mock(spec=object))
        self.assertRaisesRegex(
            Exception,
            "Can only set one of the following: model, inference_spec.",
            builder.build,
            Mode.SAGEMAKER_ENDPOINT,
            mock_role_arn,
            mock_session,
        )

    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_validation_unsupported_model_server_type(self, mock_serveSettings):
        builder = ModelBuilder(model_server="invalid_model_server")
        self.assertRaisesRegex(
            Exception,
            "%s is not supported yet! Supported model servers: %s"
            % (builder.model_server, supported_model_servers),
            builder.build,
            Mode.SAGEMAKER_ENDPOINT,
            mock_role_arn,
            mock_session,
        )

    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_validation_model_server_not_set_with_image_uri(self, mock_serveSettings):
        builder = ModelBuilder(image_uri="image_uri")
        self.assertRaisesRegex(
            Exception,
            "Model_server must be set when non-first-party image_uri is set. "
            + "Supported model servers: %s" % supported_model_servers,
            builder.build,
            Mode.SAGEMAKER_ENDPOINT,
            mock_role_arn,
            mock_session,
        )

    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_save_model_throw_exception_when_none_of_model_and_inference_spec_is_set(
        self, mock_serveSettings
    ):
        builder = ModelBuilder(inference_spec=None, model=None)
        self.assertRaisesRegex(
            Exception,
            "Cannot detect required model or inference spec",
            builder.build,
            Mode.SAGEMAKER_ENDPOINT,
            mock_role_arn,
            mock_session,
        )

    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_djl")
    def test_model_server_override_djl_with_model(self, mock_build_for_djl, mock_serve_settings):
        mock_setting_object = mock_serve_settings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        builder = ModelBuilder(model_server=ModelServer.DJL_SERVING, model="gpt_llm_burt")
        builder.build(sagemaker_session=mock_session)

        mock_build_for_djl.assert_called_once()

    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_model_server_override_djl_without_model_or_mlflow(self, mock_serve_settings):
        builder = ModelBuilder(
            model_server=ModelServer.DJL_SERVING, model=None, inference_spec=None
        )
        self.assertRaisesRegex(
            Exception,
            "Missing required parameter `model` or 'ml_flow' path",
            builder.build,
            Mode.SAGEMAKER_ENDPOINT,
            mock_role_arn,
            mock_session,
        )

    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_torchserve")
    def test_model_server_override_torchserve_with_model(
        self, mock_build_for_ts, mock_serve_settings
    ):
        mock_setting_object = mock_serve_settings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        builder = ModelBuilder(model_server=ModelServer.TORCHSERVE, model="gpt_llm_burt")
        builder.build(sagemaker_session=mock_session)

        mock_build_for_ts.assert_called_once()

    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_model_server_override_torchserve_without_model_or_mlflow(self, mock_serve_settings):
        builder = ModelBuilder(model_server=ModelServer.TORCHSERVE)
        self.assertRaisesRegex(
            Exception,
            "Missing required parameter `model` or 'ml_flow' path",
            builder.build,
            Mode.SAGEMAKER_ENDPOINT,
            mock_role_arn,
            mock_session,
        )

    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_triton")
    def test_model_server_override_triton_with_model(self, mock_build_for_ts, mock_serve_settings):
        mock_setting_object = mock_serve_settings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        builder = ModelBuilder(model_server=ModelServer.TRITON, model="gpt_llm_burt")
        builder.build(sagemaker_session=mock_session)

        mock_build_for_ts.assert_called_once()

    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_tensorflow_serving")
    def test_model_server_override_tensor_with_model(self, mock_build_for_ts, mock_serve_settings):
        mock_setting_object = mock_serve_settings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        builder = ModelBuilder(model_server=ModelServer.TENSORFLOW_SERVING, model="gpt_llm_burt")
        builder.build(sagemaker_session=mock_session)

        mock_build_for_ts.assert_called_once()

    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_tei")
    def test_model_server_override_tei_with_model(self, mock_build_for_ts, mock_serve_settings):
        mock_setting_object = mock_serve_settings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        builder = ModelBuilder(model_server=ModelServer.TEI, model="gpt_llm_burt")
        builder.build(sagemaker_session=mock_session)

        mock_build_for_ts.assert_called_once()

    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_tgi")
    def test_model_server_override_tgi_with_model(self, mock_build_for_ts, mock_serve_settings):
        mock_setting_object = mock_serve_settings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        builder = ModelBuilder(model_server=ModelServer.TGI, model="gpt_llm_burt")
        builder.build(sagemaker_session=mock_session)

        mock_build_for_ts.assert_called_once()

    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_transformers")
    def test_model_server_override_transformers_with_model(
        self, mock_build_for_ts, mock_serve_settings
    ):
        mock_setting_object = mock_serve_settings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        builder = ModelBuilder(model_server=ModelServer.MMS, model="gpt_llm_burt")
        builder.build(sagemaker_session=mock_session)

        mock_build_for_ts.assert_called_once()

    @patch("os.makedirs", Mock())
    @patch(
        "sagemaker.serve.builder.model_builder.ModelBuilder._is_jumpstart_model_id",
        return_value=False,
    )
    @patch("sagemaker.serve.builder.model_builder._detect_framework_and_version")
    @patch("sagemaker.serve.builder.model_builder.prepare_for_torchserve")
    @patch("sagemaker.serve.builder.model_builder.save_pkl")
    @patch("sagemaker.serve.builder.model_builder.auto_detect_container")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    @patch("sagemaker.serve.builder.model_builder.SageMakerEndpointMode")
    @patch("sagemaker.serve.builder.model_builder.Model")
    @patch("os.path.exists")
    def test_build_happy_path_with_sagemaker_endpoint_mode_and_byoc(
        self,
        mock_path_exists,
        mock_sdk_model,
        mock_sageMakerEndpointMode,
        mock_serveSettings,
        mock_detect_container,
        mock_save_pkl,
        mock_prepare_for_torchserve,
        mock_detect_fw_version,
        mock_is_jumpstart_model_id,
    ):
        # setup mocks
        mock_detect_container.side_effect = lambda model, region, instance_type: (
            mock_image_uri
            if model == mock_fw_model
            and region == mock_session.boto_region_name
            and instance_type == "ml.c5.xlarge"
            else None
        )

        mock_detect_fw_version.return_value = framework, version

        mock_prepare_for_torchserve.side_effect = (
            lambda model_path, shared_libs, dependencies, session, image_uri, inference_spec: (
                mock_secret_key
                if model_path == MODEL_PATH
                and shared_libs == []
                and dependencies == {"auto": False}
                and session == session
                and image_uri == mock_image_uri
                and inference_spec is None
                else None
            )
        )

        # Mock _ServeSettings
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_path_exists.side_effect = lambda path: True if path == "test_path" else False

        mock_mode = Mock()
        mock_sageMakerEndpointMode.side_effect = lambda inference_spec, model_server: (
            mock_mode if inference_spec is None and model_server == ModelServer.TORCHSERVE else None
        )
        mock_mode.prepare.side_effect = lambda model_path, secret_key, s3_model_data_url, sagemaker_session, image_uri, jumpstart, **kwargs: (  # noqa E501
            (
                model_data,
                ENV_VAR_PAIR,
            )
            if model_path == MODEL_PATH
            and secret_key == mock_secret_key
            and s3_model_data_url == mock_s3_model_data_url
            and sagemaker_session == mock_session
            and image_uri == mock_image_uri
            else None
        )

        mock_model_obj = Mock()
        mock_sdk_model.side_effect = lambda image_uri, image_config, vpc_config, model_data, role, env, sagemaker_session, predictor_cls, name: (  # noqa E501
            mock_model_obj
            if image_uri == mock_image_uri
            and image_config == MOCK_IMAGE_CONFIG
            and vpc_config == MOCK_VPC_CONFIG
            and model_data == model_data
            and role == mock_role_arn
            and env == ENV_VARS
            and sagemaker_session == mock_session
            and "model-name-" in name
            else None
        )

        mock_session.sagemaker_client._user_agent_creator.to_string = lambda: "sample agent"

        # run
        builder = ModelBuilder(
            model_path=MODEL_PATH,
            schema_builder=schema_builder,
            model=mock_fw_model,
            model_server=ModelServer.TORCHSERVE,
            image_uri=mock_image_uri,
            image_config=MOCK_IMAGE_CONFIG,
            vpc_config=MOCK_VPC_CONFIG,
        )
        build_result = builder.build(sagemaker_session=mock_session)

        # assert auto-detection was skipped
        mock_detect_container.assert_not_called()

        # assert model returned by builder is expected
        self.assertEqual(mock_model_obj, build_result)
        self.assertEqual(build_result.mode, Mode.SAGEMAKER_ENDPOINT)
        self.assertEqual(build_result.modes, {str(Mode.SAGEMAKER_ENDPOINT): mock_mode})
        self.assertEqual(build_result.serve_settings, mock_setting_object)

    @patch("os.makedirs", Mock())
    @patch(
        "sagemaker.serve.builder.model_builder.ModelBuilder._is_jumpstart_model_id",
        return_value=False,
    )
    @patch("sagemaker.serve.builder.model_builder._detect_framework_and_version")
    @patch("sagemaker.serve.builder.model_builder.prepare_for_torchserve")
    @patch("sagemaker.serve.builder.model_builder.save_pkl")
    @patch("sagemaker.serve.builder.model_builder.auto_detect_container")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    @patch("sagemaker.serve.builder.model_builder.SageMakerEndpointMode")
    @patch("sagemaker.serve.builder.model_builder.Model")
    @patch("os.path.exists")
    def test_build_happy_path_with_sagemaker_endpoint_mode_and_1p_dlc_as_byoc(
        self,
        mock_path_exists,
        mock_sdk_model,
        mock_sageMakerEndpointMode,
        mock_serveSettings,
        mock_detect_container,
        mock_save_pkl,
        mock_prepare_for_torchserve,
        mock_detect_fw_version,
        mock_is_jumpstart_model_id,
    ):
        # setup mocks
        mock_detect_container.side_effect = lambda model, region, instance_type: (
            mock_1p_dlc_image_uri
            if model == mock_fw_model
            and region == mock_session.boto_region_name
            and instance_type == "ml.c5.xlarge"
            else None
        )
        mock_detect_fw_version.return_value = framework, version

        mock_prepare_for_torchserve.side_effect = (
            lambda model_path, shared_libs, dependencies, session, image_uri, inference_spec: (
                mock_secret_key
                if model_path == MODEL_PATH
                and shared_libs == []
                and dependencies == {"auto": False}
                and session == session
                and image_uri == mock_1p_dlc_image_uri
                and inference_spec is None
                else None
            )
        )

        # Mock _ServeSettings
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_path_exists.side_effect = lambda path: True if path == "test_path" else False

        mock_mode = Mock()
        mock_sageMakerEndpointMode.side_effect = lambda inference_spec, model_server: (
            mock_mode if inference_spec is None and model_server == ModelServer.TORCHSERVE else None
        )
        mock_mode.prepare.side_effect = lambda model_path, secret_key, s3_model_data_url, sagemaker_session, image_uri, jumpstart, **kwargs: (  # noqa E501
            (
                model_data,
                ENV_VAR_PAIR,
            )
            if model_path == MODEL_PATH
            and secret_key == mock_secret_key
            and s3_model_data_url == mock_s3_model_data_url
            and sagemaker_session == mock_session
            and image_uri == mock_1p_dlc_image_uri
            else None
        )

        mock_model_obj = Mock()
        mock_sdk_model.side_effect = lambda image_uri, image_config, vpc_config, model_data, role, env, sagemaker_session, predictor_cls, name: (  # noqa E501
            mock_model_obj
            if image_uri == mock_1p_dlc_image_uri
            and model_data == model_data
            and role == mock_role_arn
            and env == ENV_VARS
            and sagemaker_session == mock_session
            and "model-name-" in name
            else None
        )

        mock_session.sagemaker_client._user_agent_creator.to_string = lambda: "sample agent"

        # run
        builder = ModelBuilder(
            model_path=MODEL_PATH,
            schema_builder=schema_builder,
            model=mock_fw_model,
            model_server=ModelServer.TORCHSERVE,
            image_uri=mock_1p_dlc_image_uri,
        )
        build_result = builder.build(sagemaker_session=mock_session)

        # assert auto-detection was skipped
        mock_detect_container.assert_not_called()

        # assert model returned by builder is expected
        self.assertEqual(mock_model_obj, build_result)
        self.assertEqual(build_result.mode, Mode.SAGEMAKER_ENDPOINT)
        self.assertEqual(build_result.modes, {str(Mode.SAGEMAKER_ENDPOINT): mock_mode})
        self.assertEqual(build_result.serve_settings, mock_setting_object)

    @patch("os.makedirs", Mock())
    @patch("sagemaker.serve.builder.model_builder._detect_framework_and_version")
    @patch("sagemaker.serve.builder.model_builder.prepare_for_torchserve")
    @patch("sagemaker.serve.builder.model_builder.save_pkl")
    @patch("sagemaker.serve.builder.model_builder.auto_detect_container")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    @patch("sagemaker.serve.builder.model_builder.SageMakerEndpointMode")
    @patch("sagemaker.serve.builder.model_builder.Model")
    @patch("os.path.exists")
    def test_build_happy_path_with_sagemaker_endpoint_mode_and_inference_spec(
        self,
        mock_path_exists,
        mock_sdk_model,
        mock_sageMakerEndpointMode,
        mock_serveSettings,
        mock_detect_container,
        mock_save_pkl,
        mock_prepare_for_torchserve,
        mock_detect_fw_version,
    ):
        # setup mocks
        mock_native_model = Mock()
        mock_inference_spec.load = lambda model_path: (
            mock_native_model if model_path == MODEL_PATH else None
        )

        mock_detect_fw_version.return_value = framework, version

        mock_detect_container.side_effect = lambda model, region, instance_type: (
            mock_image_uri
            if model == mock_native_model
            and region == mock_session.boto_region_name
            and instance_type == "ml.c5.xlarge"
            else None
        )

        mock_prepare_for_torchserve.side_effect = (
            lambda model_path, shared_libs, dependencies, session, image_uri, inference_spec: (
                mock_secret_key
                if model_path == MODEL_PATH
                and shared_libs == []
                and dependencies == {"auto": False}
                and session == mock_session
                and image_uri == mock_image_uri
                and inference_spec == mock_inference_spec
                else None
            )
        )

        # Mock _ServeSettings
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_path_exists.side_effect = lambda path: True if path == "test_path" else False

        mock_mode = Mock()
        mock_sageMakerEndpointMode.side_effect = lambda inference_spec, model_server: (
            mock_mode
            if inference_spec == mock_inference_spec and model_server == ModelServer.TORCHSERVE
            else None
        )
        mock_mode.prepare.side_effect = lambda model_path, secret_key, s3_model_data_url, sagemaker_session, image_uri, jumpstart, **kwargs: (  # noqa E501
            (
                model_data,
                ENV_VAR_PAIR,
            )
            if model_path == MODEL_PATH
            and secret_key == mock_secret_key
            and s3_model_data_url == mock_s3_model_data_url
            and sagemaker_session == mock_session
            and image_uri == mock_image_uri
            else None
        )

        mock_model_obj = Mock()
        mock_sdk_model.side_effect = lambda image_uri, image_config, vpc_config, model_data, role, env, sagemaker_session, predictor_cls, name: (  # noqa E501
            mock_model_obj
            if image_uri == mock_image_uri
            and model_data == model_data
            and role == mock_role_arn
            and env == ENV_VARS_INF_SPEC
            and sagemaker_session == mock_session
            and "model-name-" in name
            else None
        )

        # run
        builder = ModelBuilder(
            model_path=MODEL_PATH,
            schema_builder=schema_builder,
            inference_spec=mock_inference_spec,
        )
        build_result = builder.build(sagemaker_session=mock_session)

        # assert pkl file was saved
        mock_save_pkl.assert_called_once_with(CODE_PATH, (mock_inference_spec, schema_builder))

        # assert model returned by builder is expected
        self.assertEqual(mock_model_obj, build_result)
        self.assertEqual(build_result.mode, Mode.SAGEMAKER_ENDPOINT)
        self.assertEqual(build_result.modes, {str(Mode.SAGEMAKER_ENDPOINT): mock_mode})
        self.assertEqual(build_result.serve_settings, mock_setting_object)

    @patch("os.makedirs", Mock())
    @patch(
        "sagemaker.serve.builder.model_builder.ModelBuilder._is_jumpstart_model_id",
        return_value=False,
    )
    @patch("sagemaker.serve.builder.model_builder._detect_framework_and_version")
    @patch("sagemaker.serve.builder.model_builder.prepare_for_torchserve")
    @patch("sagemaker.serve.builder.model_builder.save_pkl")
    @patch("sagemaker.serve.builder.model_builder.auto_detect_container")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    @patch("sagemaker.serve.builder.model_builder.SageMakerEndpointMode")
    @patch("sagemaker.serve.builder.model_builder.Model")
    @patch("os.path.exists")
    def test_build_happy_path_with_sagemakerEndpoint_mode_and_model(
        self,
        mock_path_exists,
        mock_sdk_model,
        mock_sageMakerEndpointMode,
        mock_serveSettings,
        mock_detect_container,
        mock_save_pkl,
        mock_prepare_for_torchserve,
        mock_detect_fw_version,
        mock_is_jumpstart_model_id,
    ):
        # setup mocks
        mock_detect_container.side_effect = lambda model, region, instance_type: (
            mock_image_uri
            if model == mock_fw_model
            and region == mock_session.boto_region_name
            and instance_type == "ml.c5.xlarge"
            else None
        )

        mock_detect_fw_version.return_value = framework, version

        mock_prepare_for_torchserve.side_effect = (
            lambda model_path, shared_libs, dependencies, session, image_uri, inference_spec: (
                mock_secret_key
                if model_path == MODEL_PATH
                and shared_libs == []
                and dependencies == {"auto": False}
                and session == session
                and image_uri == mock_image_uri
                and inference_spec is None
                else None
            )
        )

        # Mock _ServeSettings
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_path_exists.side_effect = lambda path: True if path == "test_path" else False

        mock_mode = Mock()
        mock_sageMakerEndpointMode.side_effect = lambda inference_spec, model_server: (
            mock_mode if inference_spec is None and model_server == ModelServer.TORCHSERVE else None
        )
        mock_mode.prepare.side_effect = lambda model_path, secret_key, s3_model_data_url, sagemaker_session, image_uri, jumpstart, **kwargs: (  # noqa E501
            (
                model_data,
                ENV_VAR_PAIR,
            )
            if model_path == MODEL_PATH
            and secret_key == mock_secret_key
            and s3_model_data_url == mock_s3_model_data_url
            and sagemaker_session == mock_session
            and image_uri == mock_image_uri
            else None
        )

        mock_model_obj = Mock()
        mock_sdk_model.side_effect = lambda image_uri, image_config, vpc_config, model_data, role, env, sagemaker_session, predictor_cls, name: (  # noqa E501
            mock_model_obj
            if image_uri == mock_image_uri
            and model_data == model_data
            and role == mock_role_arn
            and env == ENV_VARS
            and sagemaker_session == mock_session
            and "model-name-" in name
            else None
        )

        mock_session.sagemaker_client._user_agent_creator.to_string = lambda: "sample agent"

        # run
        builder = ModelBuilder(
            model_path=MODEL_PATH,
            schema_builder=schema_builder,
            model=mock_fw_model,
        )
        build_result = builder.build(sagemaker_session=mock_session)

        # assert pkl file was saved
        mock_save_pkl.assert_called_once_with(CODE_PATH, (mock_fw_model, schema_builder))

        # assert model returned by builder is expected
        self.assertEqual(mock_model_obj, build_result)
        self.assertEqual(build_result.mode, Mode.SAGEMAKER_ENDPOINT)
        self.assertEqual(build_result.modes, {str(Mode.SAGEMAKER_ENDPOINT): mock_mode})
        self.assertEqual(build_result.serve_settings, mock_setting_object)

        # assert user agent
        user_agent = builder.sagemaker_session.sagemaker_client._user_agent_creator.to_string()
        self.assertEqual("sample agent ModelBuilder", user_agent)

    @patch("os.makedirs", Mock())
    @patch(
        "sagemaker.serve.builder.model_builder.ModelBuilder._is_jumpstart_model_id",
        return_value=False,
    )
    @patch("sagemaker.serve.builder.model_builder.save_xgboost")
    @patch("sagemaker.serve.builder.model_builder._detect_framework_and_version")
    @patch("sagemaker.serve.builder.model_builder.prepare_for_torchserve")
    @patch("sagemaker.serve.builder.model_builder.save_pkl")
    @patch("sagemaker.serve.builder.model_builder.auto_detect_container")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    @patch("sagemaker.serve.builder.model_builder.SageMakerEndpointMode")
    @patch("sagemaker.serve.builder.model_builder.Model")
    @patch("os.path.exists")
    def test_build_happy_path_with_sagemakerEndpoint_mode_and_xgboost_model(
        self,
        mock_path_exists,
        mock_sdk_model,
        mock_sageMakerEndpointMode,
        mock_serveSettings,
        mock_detect_container,
        mock_save_pkl,
        mock_prepare_for_torchserve,
        mock_detect_fw_version,
        mock_save_xgb,
        mock_is_jumpstart_model_id,
    ):
        # setup mocks
        mock_detect_container.side_effect = lambda model, region, instance_type: (
            mock_image_uri
            if model == mock_fw_model
            and region == mock_session.boto_region_name
            and instance_type == "ml.c5.xlarge"
            else None
        )

        mock_detect_fw_version.return_value = "xgboost", version

        mock_prepare_for_torchserve.side_effect = (
            lambda model_path, shared_libs, dependencies, session, image_uri, inference_spec: (
                mock_secret_key
                if model_path == MODEL_PATH
                and shared_libs == []
                and dependencies == {"auto": False}
                and session == session
                and image_uri == mock_image_uri
                and inference_spec is None
                else None
            )
        )

        # Mock _ServeSettings
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_path_exists.side_effect = lambda path: True if path == "test_path" else False

        mock_mode = Mock()
        mock_sageMakerEndpointMode.side_effect = lambda inference_spec, model_server: (
            mock_mode if inference_spec is None and model_server == ModelServer.TORCHSERVE else None
        )
        mock_mode.prepare.side_effect = lambda model_path, secret_key, s3_model_data_url, sagemaker_session, image_uri, jumpstart, **kwargs: (  # noqa E501
            (
                model_data,
                ENV_VAR_PAIR,
            )
            if model_path == MODEL_PATH
            and secret_key == mock_secret_key
            and s3_model_data_url == mock_s3_model_data_url
            and sagemaker_session == mock_session
            and image_uri == mock_image_uri
            else None
        )

        mock_model_obj = Mock()
        mock_sdk_model.side_effect = lambda image_uri, image_config, vpc_config, model_data, role, env, sagemaker_session, predictor_cls, name: (  # noqa E501
            mock_model_obj
            if image_uri == mock_image_uri
            and model_data == model_data
            and role == mock_role_arn
            and env == ENV_VARS
            and sagemaker_session == mock_session
            and "model-name-" in name
            else None
        )

        mock_session.sagemaker_client._user_agent_creator.to_string = lambda: "sample agent"

        # run
        builder = ModelBuilder(
            model_path=MODEL_PATH,
            schema_builder=schema_builder,
            model=mock_fw_model,
        )
        build_result = builder.build(sagemaker_session=mock_session)

        # assert xgboost model is saved
        mock_save_xgb.assert_called_once_with(CODE_PATH, mock_fw_model)

        # assert pkl file was saved
        mock_save_pkl.assert_called_once_with(CODE_PATH, ("xgboost", schema_builder))

        # assert model returned by builder is expected
        self.assertEqual(mock_model_obj, build_result)
        self.assertEqual(build_result.mode, Mode.SAGEMAKER_ENDPOINT)
        self.assertEqual(build_result.modes, {str(Mode.SAGEMAKER_ENDPOINT): mock_mode})
        self.assertEqual(build_result.serve_settings, mock_setting_object)

        # assert user agent
        user_agent = builder.sagemaker_session.sagemaker_client._user_agent_creator.to_string()
        self.assertEqual("sample agent ModelBuilder", user_agent)

    @patch("os.makedirs", Mock())
    @patch("sagemaker.serve.builder.model_builder.prepare_for_torchserve")
    @patch("sagemaker.serve.builder.model_builder.save_pkl")
    @patch("sagemaker.serve.builder.model_builder.auto_detect_container")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    @patch("sagemaker.serve.builder.model_builder.LocalContainerMode")
    @patch("sagemaker.serve.builder.model_builder.Model")
    @patch("os.path.exists")
    def test_build_happy_path_with_local_container_mode(
        self,
        mock_path_exists,
        mock_sdk_model,
        mock_localContainerMode,
        mock_serveSettings,
        mock_detect_container,
        mock_save_pkl,
        mock_prepare_for_torchserve,
    ):
        # setup mocks
        mock_native_model = Mock()
        mock_inference_spec.load = lambda model_path: (
            mock_native_model if model_path == MODEL_PATH else None
        )

        mock_detect_container.side_effect = lambda model, region, instance_type: (
            mock_image_uri
            if model == mock_native_model
            and region == mock_session.boto_region_name
            and instance_type == "ml.c5.xlarge"
            else None
        )

        mock_prepare_for_torchserve.side_effect = (
            lambda model_path, shared_libs, dependencies, session, image_uri, inference_spec: (
                mock_secret_key
                if model_path == MODEL_PATH
                and shared_libs == []
                and dependencies == {"auto": False}
                and session == mock_session
                and image_uri == mock_image_uri
                and inference_spec == mock_inference_spec
                else None
            )
        )

        # Mock _ServeSettings
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_path_exists.side_effect = lambda path: True if path == "test_path" else False

        mock_mode = Mock()
        mock_localContainerMode.side_effect = (
            lambda inference_spec, schema_builder, session, model_path, env_vars, model_server: (
                mock_mode
                if inference_spec == mock_inference_spec
                and schema_builder == schema_builder
                and model_server == ModelServer.TORCHSERVE
                and session == mock_session
                and model_path == MODEL_PATH
                and env_vars == {}
                and model_server == ModelServer.TORCHSERVE
                else None
            )
        )
        mock_mode.prepare.side_effect = lambda: None

        mock_model_obj = Mock()
        mock_sdk_model.side_effect = lambda image_uri, image_config, vpc_config, model_data, role, env, sagemaker_session, predictor_cls, name: (  # noqa E501
            mock_model_obj
            if image_uri == mock_image_uri
            and model_data is None
            and role == mock_role_arn
            and env == {}
            and sagemaker_session == mock_session
            and "model-name-" in name
            else None
        )

        # run
        builder = ModelBuilder(
            model_path=MODEL_PATH,
            schema_builder=schema_builder,
            inference_spec=mock_inference_spec,
            mode=Mode.LOCAL_CONTAINER,
        )
        build_result = builder.build(sagemaker_session=mock_session)

        # assert pkl file was saved
        mock_save_pkl.assert_called_once_with(CODE_PATH, (mock_inference_spec, schema_builder))

        # assert model returned by builder is expected
        self.assertEqual(mock_model_obj, build_result)
        self.assertEqual(build_result.mode, Mode.LOCAL_CONTAINER)
        self.assertEqual(build_result.serve_settings, mock_setting_object)

    @patch("os.makedirs", Mock())
    @patch("sagemaker.serve.builder.model_builder._detect_framework_and_version")
    @patch("sagemaker.serve.builder.model_builder.prepare_for_torchserve")
    @patch("sagemaker.serve.builder.model_builder.save_pkl")
    @patch("sagemaker.serve.builder.model_builder.auto_detect_container")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    @patch("sagemaker.serve.builder.model_builder.SageMakerEndpointMode")
    @patch("sagemaker.serve.builder.model_builder.LocalContainerMode")
    @patch("sagemaker.serve.builder.model_builder.Model")
    @patch("os.path.exists")
    def test_build_happy_path_with_localContainer_mode_overwritten_with_sagemaker_mode(
        self,
        mock_path_exists,
        mock_sdk_model,
        mock_localContainerMode,
        mock_sageMakerEndpointMode,
        mock_serveSettings,
        mock_detect_container,
        mock_save_pkl,
        mock_prepare_for_torchserve,
        mock_detect_fw_version,
    ):
        # setup mocks
        mock_native_model = Mock()
        mock_inference_spec.load = lambda model_path: (
            mock_native_model if model_path == MODEL_PATH else None
        )

        mock_detect_fw_version.return_value = framework, version

        mock_detect_container.side_effect = lambda model, region, instance_type: (
            mock_image_uri
            if model == mock_native_model
            and region == mock_session.boto_region_name
            and instance_type == "ml.c5.xlarge"
            else None
        )

        mock_prepare_for_torchserve.side_effect = (
            lambda model_path, shared_libs, dependencies, session, image_uri, inference_spec: (
                mock_secret_key
                if model_path == MODEL_PATH
                and shared_libs == []
                and dependencies == {"auto": False}
                and session == mock_session
                and image_uri == mock_image_uri
                and inference_spec == mock_inference_spec
                else None
            )
        )

        # Mock _ServeSettings
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_path_exists.side_effect = lambda path: True if path == "test_path" else False

        mock_lc_mode = Mock()
        mock_localContainerMode.side_effect = (
            lambda inference_spec, schema_builder, session, model_path, env_vars, model_server: (
                mock_lc_mode
                if inference_spec == mock_inference_spec
                and schema_builder == schema_builder
                and model_server == ModelServer.TORCHSERVE
                and session == mock_session
                and model_path == MODEL_PATH
                and env_vars == {}
                and model_server == ModelServer.TORCHSERVE
                else None
            )
        )
        mock_lc_mode.prepare.side_effect = lambda: None

        mock_sagemaker_endpoint_mode = Mock()
        mock_sageMakerEndpointMode.side_effect = lambda inference_spec, model_server: (
            mock_sagemaker_endpoint_mode
            if inference_spec == mock_inference_spec and model_server == ModelServer.TORCHSERVE
            else None
        )
        mock_sagemaker_endpoint_mode.prepare.side_effect = lambda model_path, secret_key, s3_model_data_url, sagemaker_session, image_uri, jumpstart, **kwargs: (  # noqa E501
            (
                model_data,
                ENV_VAR_PAIR,
            )
            if model_path == MODEL_PATH
            and secret_key == mock_secret_key
            and s3_model_data_url == mock_s3_model_data_url
            and sagemaker_session == mock_session
            and image_uri == mock_image_uri
            else None
        )

        mock_model_obj = Mock()
        mock_sdk_model.side_effect = lambda image_uri, image_config, vpc_config, model_data, role, env, sagemaker_session, predictor_cls, name: (  # noqa E501
            mock_model_obj
            if image_uri == mock_image_uri
            and model_data is None
            and role == mock_role_arn
            and env == {}
            and sagemaker_session == mock_session
            and "model-name-" in name
            else None
        )

        # run
        builder = ModelBuilder(
            model_path=MODEL_PATH,
            schema_builder=schema_builder,
            inference_spec=mock_inference_spec,
            mode=Mode.LOCAL_CONTAINER,
        )

        build_result = builder.build(sagemaker_session=mock_session)

        # assert pkl file was saved
        mock_save_pkl.assert_called_once_with(CODE_PATH, (mock_inference_spec, schema_builder))

        # assert model returned by builder is expected
        self.assertEqual(mock_model_obj, build_result)
        self.assertEqual(build_result.mode, Mode.LOCAL_CONTAINER)
        self.assertEqual(build_result.serve_settings, mock_setting_object)

        mock_predictor = Mock()

        builder._original_deploy = Mock()
        builder._original_deploy.side_effect = lambda *args, **kwargs: (
            mock_predictor
            if kwargs.get("initial_instance_count") == 1
            and kwargs.get("instance_type") == mock_instance_type
            else None
        )

        build_result.deploy(
            initial_instance_count=1, instance_type=mock_instance_type, mode=Mode.SAGEMAKER_ENDPOINT
        )

        self.assertEqual(builder.mode, Mode.SAGEMAKER_ENDPOINT)
        self.assertEqual(build_result.mode, Mode.SAGEMAKER_ENDPOINT)
        self.assertEqual(build_result.model_data, model_data)

        build_result.env.update.assert_called_once_with(ENV_VAR_PAIR)
        mock_sageMakerEndpointMode.assert_called_once_with(
            inference_spec=mock_inference_spec, model_server=ModelServer.TORCHSERVE
        )

    @patch("os.makedirs", Mock())
    @patch(
        "sagemaker.serve.builder.model_builder.ModelBuilder._is_jumpstart_model_id",
        return_value=False,
    )
    @patch("sagemaker.serve.builder.model_builder._detect_framework_and_version")
    @patch("sagemaker.serve.builder.model_builder.prepare_for_torchserve")
    @patch("sagemaker.serve.builder.model_builder.save_pkl")
    @patch("sagemaker.serve.builder.model_builder.auto_detect_container")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    @patch("sagemaker.serve.builder.model_builder.SageMakerEndpointMode")
    @patch("sagemaker.serve.builder.model_builder.LocalContainerMode")
    @patch("sagemaker.serve.builder.model_builder.Model")
    @patch("os.path.exists")
    def test_build_happy_path_with_sagemaker_endpoint_mode_overwritten_with_local_container(
        self,
        mock_path_exists,
        mock_sdk_model,
        mock_localContainerMode,
        mock_sageMakerEndpointMode,
        mock_serveSettings,
        mock_detect_container,
        mock_save_pkl,
        mock_prepare_for_torchserve,
        mock_detect_fw_version,
        mock_is_jumpstart_model_id,
    ):
        # setup mocks
        mock_detect_fw_version.return_value = framework, version

        mock_detect_container.side_effect = lambda model, region, instance_type: (
            mock_image_uri
            if model == mock_fw_model
            and region == mock_session.boto_region_name
            and instance_type == "ml.c5.xlarge"
            else None
        )

        mock_prepare_for_torchserve.side_effect = (
            lambda model_path, shared_libs, dependencies, image_uri, session, inference_spec: (
                mock_secret_key
                if model_path == MODEL_PATH
                and shared_libs == []
                and dependencies == {"auto": False}
                and session == mock_session
                and inference_spec is None
                and image_uri == mock_image_uri
                else None
            )
        )

        # Mock _ServeSettings
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_path_exists.side_effect = lambda path: True if path == "test_path" else False

        mock_mode = Mock()
        mock_sageMakerEndpointMode.side_effect = lambda inference_spec, model_server: (
            mock_mode if inference_spec is None and model_server == ModelServer.TORCHSERVE else None
        )
        mock_mode.prepare.side_effect = lambda model_path, secret_key, s3_model_data_url, sagemaker_session, image_uri, jumpstart, **kwargs: (  # noqa E501
            (
                model_data,
                ENV_VAR_PAIR,
            )
            if model_path == MODEL_PATH
            and secret_key == mock_secret_key
            and s3_model_data_url == mock_s3_model_data_url
            and sagemaker_session == mock_session
            and image_uri == mock_image_uri
            else None
        )

        mock_lc_mode = Mock()
        mock_localContainerMode.side_effect = (
            lambda inference_spec, schema_builder, session, model_path, env_vars, model_server: (
                mock_lc_mode
                if inference_spec is None
                and schema_builder == schema_builder
                and model_server == ModelServer.TORCHSERVE
                and session == mock_session
                and model_path == MODEL_PATH
                and env_vars == ENV_VARS
                else None
            )
        )
        mock_lc_mode.prepare.side_effect = lambda: None
        mock_lc_mode.create_server.side_effect = (
            lambda image_uri, container_timeout_seconds, secret_key, predictor: (
                None
                if image_uri == mock_image_uri
                and secret_key == mock_secret_key
                and container_timeout_seconds == 60
                else None
            )
        )

        mock_model_obj = Mock()
        mock_sdk_model.side_effect = lambda image_uri, image_config, vpc_config, model_data, role, env, sagemaker_session, predictor_cls, name: (  # noqa E501
            mock_model_obj
            if image_uri == mock_image_uri
            and model_data == model_data
            and role == mock_role_arn
            and env == ENV_VARS
            and sagemaker_session == mock_session
            and "model-name-" in name
            else None
        )

        # run
        builder = ModelBuilder(
            model_path=MODEL_PATH,
            schema_builder=schema_builder,
            model=mock_fw_model,
        )
        build_result = builder.build(sagemaker_session=mock_session)

        # assert pkl file was saved
        mock_save_pkl.assert_called_once_with(CODE_PATH, (mock_fw_model, schema_builder))

        # assert model returned by builder is expected
        self.assertEqual(mock_model_obj, build_result)
        self.assertEqual(build_result.mode, Mode.SAGEMAKER_ENDPOINT)
        self.assertEqual(build_result.modes, {str(Mode.SAGEMAKER_ENDPOINT): mock_mode})
        self.assertEqual(build_result.serve_settings, mock_setting_object)

        build_result.deploy(mode=Mode.LOCAL_CONTAINER)

        self.assertEqual(builder.mode, Mode.LOCAL_CONTAINER)

    @patch("sagemaker.serve.builder.tgi_builder.HuggingFaceModel")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.serve.utils.hf_utils.urllib")
    @patch("sagemaker.serve.utils.hf_utils.json")
    @patch("sagemaker.huggingface.llm_utils.urllib")
    @patch("sagemaker.huggingface.llm_utils.json")
    @patch("sagemaker.model_uris.retrieve")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_build_happy_path_when_schema_builder_not_present(
        self,
        mock_serveSettings,
        mock_model_uris_retrieve,
        mock_llm_utils_json,
        mock_llm_utils_urllib,
        mock_model_json,
        mock_model_urllib,
        mock_image_uris_retrieve,
        mock_hf_model,
    ):
        # Setup mocks

        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        # HF Pipeline Tag
        mock_model_uris_retrieve.side_effect = KeyError
        mock_llm_utils_json.load.return_value = {"pipeline_tag": "text-generation"}
        mock_llm_utils_urllib.request.Request.side_effect = Mock()

        # HF Model config
        mock_model_json.load.return_value = {"some": "config"}
        mock_model_urllib.request.Request.side_effect = Mock()

        mock_image_uris_retrieve.return_value = "https://some-image-uri"

        model_builder = ModelBuilder(model="meta-llama/Llama-2-7b-hf")
        model_builder.build(sagemaker_session=mock_session)

        self.assertIsNotNone(model_builder.schema_builder)
        sample_inputs, sample_outputs = task.retrieve_local_schemas("text-generation")
        self.assertEqual(
            sample_inputs["inputs"], model_builder.schema_builder.sample_input["inputs"]
        )
        self.assertEqual(sample_outputs, model_builder.schema_builder.sample_output)

    @patch("sagemaker.serve.builder.tgi_builder.HuggingFaceModel")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.serve.utils.hf_utils.urllib")
    @patch("sagemaker.serve.utils.hf_utils.json")
    @patch("sagemaker.huggingface.llm_utils.urllib")
    @patch("sagemaker.huggingface.llm_utils.json")
    @patch("sagemaker.model_uris.retrieve")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_build_negative_path_when_schema_builder_not_present(
        self,
        mock_serveSettings,
        mock_model_uris_retrieve,
        mock_llm_utils_json,
        mock_llm_utils_urllib,
        mock_model_json,
        mock_model_urllib,
        mock_image_uris_retrieve,
        mock_hf_model,
    ):
        # Setup mocks

        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        # HF Pipeline Tag
        mock_model_uris_retrieve.side_effect = KeyError
        mock_llm_utils_json.load.return_value = {"pipeline_tag": "unsupported-task"}
        mock_llm_utils_urllib.request.Request.side_effect = Mock()

        # HF Model config
        mock_model_json.load.return_value = {"some": "config"}
        mock_model_urllib.request.Request.side_effect = Mock()

        mock_image_uris_retrieve.return_value = "https://some-image-uri"

        model_builder = ModelBuilder(model="CompVis/stable-diffusion-v1-4")

        self.assertRaisesRegex(
            TaskNotFoundException,
            "Error Message: HuggingFace Schema builder samples for unsupported-task could not be found locally or via "
            "remote.",
            lambda: model_builder.build(sagemaker_session=mock_session),
        )

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_transformers", Mock())
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._can_fit_on_single_gpu")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.serve.utils.hf_utils.urllib")
    @patch("sagemaker.serve.utils.hf_utils.json")
    @patch("sagemaker.huggingface.llm_utils.urllib")
    @patch("sagemaker.huggingface.llm_utils.json")
    @patch("sagemaker.model_uris.retrieve")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_build_can_fit_on_single_gpu(
        self,
        mock_serveSettings,
        mock_model_uris_retrieve,
        mock_llm_utils_json,
        mock_llm_utils_urllib,
        mock_model_json,
        mock_model_urllib,
        mock_image_uris_retrieve,
        mock_can_fit_on_single_gpu,
    ):
        # Setup mocks
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        # HF Pipeline Tag
        mock_model_uris_retrieve.side_effect = KeyError
        mock_llm_utils_json.load.return_value = {"pipeline_tag": "text-classification"}
        mock_llm_utils_urllib.request.Request.side_effect = Mock()

        # HF Model config
        mock_model_json.load.return_value = {"some": "config"}
        mock_model_urllib.request.Request.side_effect = Mock()

        mock_image_uris_retrieve.return_value = "https://some-image-uri"

        model_builder = ModelBuilder(model="meta-llama/Llama-2-7b-hf")
        model_builder.build(sagemaker_session=mock_session)

        mock_can_fit_on_single_gpu.assert_called_once()

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_transformers")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._can_fit_on_single_gpu")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.serve.utils.hf_utils.urllib")
    @patch("sagemaker.serve.utils.hf_utils.json")
    @patch("sagemaker.huggingface.llm_utils.urllib")
    @patch("sagemaker.huggingface.llm_utils.json")
    @patch("sagemaker.model_uris.retrieve")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_build_for_transformers_happy_case(
        self,
        mock_serveSettings,
        mock_model_uris_retrieve,
        mock_llm_utils_json,
        mock_llm_utils_urllib,
        mock_model_json,
        mock_model_urllib,
        mock_image_uris_retrieve,
        mock_can_fit_on_single_gpu,
        mock_build_for_transformers,
    ):
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_model_uris_retrieve.side_effect = KeyError
        mock_llm_utils_json.load.return_value = {"pipeline_tag": "text-classification"}
        mock_llm_utils_urllib.request.Request.side_effect = Mock()

        mock_model_json.load.return_value = {"some": "config"}
        mock_model_urllib.request.Request.side_effect = Mock()

        mock_image_uris_retrieve.return_value = "https://some-image-uri"
        mock_can_fit_on_single_gpu.return_value = True

        model_builder = ModelBuilder(model="stable-diffusion")
        model_builder.build(sagemaker_session=mock_session)

        mock_build_for_transformers.assert_called_once()

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_transformers")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._try_fetch_gpu_info")
    @patch("sagemaker.serve.builder.model_builder._total_inference_model_size_mib")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.serve.utils.hf_utils.urllib")
    @patch("sagemaker.serve.utils.hf_utils.json")
    @patch("sagemaker.huggingface.llm_utils.urllib")
    @patch("sagemaker.huggingface.llm_utils.json")
    @patch("sagemaker.model_uris.retrieve")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_build_for_transformers_happy_case_with_values(
        self,
        mock_serveSettings,
        mock_model_uris_retrieve,
        mock_llm_utils_json,
        mock_llm_utils_urllib,
        mock_model_json,
        mock_model_urllib,
        mock_image_uris_retrieve,
        mock_total_inference_model_size_mib,
        mock_try_fetch_gpu_info,
        mock_build_for_transformers,
    ):
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_model_uris_retrieve.side_effect = KeyError
        mock_llm_utils_json.load.return_value = {"pipeline_tag": "text-classification"}
        mock_llm_utils_urllib.request.Request.side_effect = Mock()

        mock_model_json.load.return_value = {"some": "config"}
        mock_model_urllib.request.Request.side_effect = Mock()
        mock_try_fetch_gpu_info.return_value = 2
        mock_total_inference_model_size_mib.return_value = 2

        mock_image_uris_retrieve.return_value = "https://some-image-uri"

        model_builder = ModelBuilder(model="stable-diffusion")
        model_builder.build(sagemaker_session=mock_session)

        mock_build_for_transformers.assert_called_once()

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_transformers", Mock())
    @patch("sagemaker.serve.builder.model_builder._get_gpu_info")
    @patch("sagemaker.serve.builder.model_builder._total_inference_model_size_mib")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.serve.utils.hf_utils.urllib")
    @patch("sagemaker.serve.utils.hf_utils.json")
    @patch("sagemaker.huggingface.llm_utils.urllib")
    @patch("sagemaker.huggingface.llm_utils.json")
    @patch("sagemaker.model_uris.retrieve")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_build_for_transformers_happy_case_with_valid_gpu_info(
        self,
        mock_serveSettings,
        mock_model_uris_retrieve,
        mock_llm_utils_json,
        mock_llm_utils_urllib,
        mock_model_json,
        mock_model_urllib,
        mock_image_uris_retrieve,
        mock_total_inference_model_size_mib,
        mock_try_fetch_gpu_info,
    ):
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_model_uris_retrieve.side_effect = KeyError
        mock_llm_utils_json.load.return_value = {"pipeline_tag": "text-classification"}
        mock_llm_utils_urllib.request.Request.side_effect = Mock()

        mock_model_json.load.return_value = {"some": "config"}
        mock_model_urllib.request.Request.side_effect = Mock()
        mock_try_fetch_gpu_info.return_value = INSTANCE_GPU_INFO
        mock_total_inference_model_size_mib.return_value = 10_000

        mock_image_uris_retrieve.return_value = "https://some-image-uri"

        model_builder = ModelBuilder(model="stable-diffusion")
        model_builder.build(sagemaker_session=mock_session)
        self.assertEqual(
            model_builder._try_fetch_gpu_info(), INSTANCE_GPU_INFO[1] / INSTANCE_GPU_INFO[0]
        )
        self.assertEqual(model_builder._can_fit_on_single_gpu(), False)

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_transformers", Mock())
    @patch("sagemaker.serve.builder.model_builder._get_gpu_info")
    @patch("sagemaker.serve.builder.model_builder._get_gpu_info_fallback")
    @patch("sagemaker.serve.builder.model_builder._total_inference_model_size_mib")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.serve.utils.hf_utils.urllib")
    @patch("sagemaker.serve.utils.hf_utils.json")
    @patch("sagemaker.huggingface.llm_utils.urllib")
    @patch("sagemaker.huggingface.llm_utils.json")
    @patch("sagemaker.model_uris.retrieve")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_build_for_transformers_happy_case_with_valid_gpu_fallback(
        self,
        mock_serveSettings,
        mock_model_uris_retrieve,
        mock_llm_utils_json,
        mock_llm_utils_urllib,
        mock_model_json,
        mock_model_urllib,
        mock_image_uris_retrieve,
        mock_total_inference_model_size_mib,
        mock_gpu_fallback,
        mock_try_fetch_gpu_info,
    ):
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_model_uris_retrieve.side_effect = KeyError
        mock_llm_utils_json.load.return_value = {"pipeline_tag": "text-classification"}
        mock_llm_utils_urllib.request.Request.side_effect = Mock()

        mock_model_json.load.return_value = {"some": "config"}
        mock_model_urllib.request.Request.side_effect = Mock()
        mock_try_fetch_gpu_info.side_effect = ValueError
        mock_gpu_fallback.return_value = INSTANCE_GPU_INFO
        mock_total_inference_model_size_mib.return_value = (
            INSTANCE_GPU_INFO[1] / INSTANCE_GPU_INFO[0] - 1
        )

        mock_image_uris_retrieve.return_value = "https://some-image-uri"

        model_builder = ModelBuilder(
            model="stable-diffusion",
            sagemaker_session=mock_session,
            instance_type=mock_instance_type,
        )
        self.assertEqual(
            model_builder._try_fetch_gpu_info(), INSTANCE_GPU_INFO[1] / INSTANCE_GPU_INFO[0]
        )
        self.assertEqual(model_builder._can_fit_on_single_gpu(), True)

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_transformers")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._can_fit_on_single_gpu")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.serve.utils.hf_utils.urllib")
    @patch("sagemaker.serve.utils.hf_utils.json")
    @patch("sagemaker.huggingface.llm_utils.urllib")
    @patch("sagemaker.huggingface.llm_utils.json")
    @patch("sagemaker.model_uris.retrieve")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_build_fallback_to_transformers(
        self,
        mock_serveSettings,
        mock_model_uris_retrieve,
        mock_llm_utils_json,
        mock_llm_utils_urllib,
        mock_model_json,
        mock_model_urllib,
        mock_image_uris_retrieve,
        mock_can_fit_on_single_gpu,
        mock_build_for_transformers,
    ):
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_model_uris_retrieve.side_effect = KeyError
        mock_llm_utils_json.load.return_value = {"pipeline_tag": "text-classification"}
        mock_llm_utils_urllib.request.Request.side_effect = Mock()

        mock_model_json.load.return_value = {"some": "config"}
        mock_model_urllib.request.Request.side_effect = Mock()
        mock_build_for_transformers.side_effect = Mock()

        mock_image_uris_retrieve.return_value = "https://some-image-uri"
        mock_can_fit_on_single_gpu.return_value = False

        model_builder = ModelBuilder(model="gpt_llm_burt")
        model_builder.build(sagemaker_session=mock_session)

        mock_build_for_transformers.assert_called_once()

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_tgi")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.serve.utils.hf_utils.urllib")
    @patch("sagemaker.serve.utils.hf_utils.json")
    @patch("sagemaker.huggingface.llm_utils.urllib")
    @patch("sagemaker.huggingface.llm_utils.json")
    @patch("sagemaker.model_uris.retrieve")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_text_generation(
        self,
        mock_serveSettings,
        mock_model_uris_retrieve,
        mock_llm_utils_json,
        mock_llm_utils_urllib,
        mock_model_json,
        mock_model_urllib,
        mock_image_uris_retrieve,
        mock_build_for_tgi,
    ):
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_model_uris_retrieve.side_effect = KeyError
        mock_llm_utils_json.load.return_value = {"pipeline_tag": "text-generation"}
        mock_llm_utils_urllib.request.Request.side_effect = Mock()

        mock_model_json.load.return_value = {"some": "config"}
        mock_model_urllib.request.Request.side_effect = Mock()
        mock_build_for_tgi.side_effect = Mock()

        mock_image_uris_retrieve.return_value = "https://some-image-uri"

        model_builder = ModelBuilder(model="bloom-560m")
        model_builder.build(sagemaker_session=mock_session)

        mock_build_for_tgi.assert_called_once()

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_tei")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.serve.utils.hf_utils.urllib")
    @patch("sagemaker.serve.utils.hf_utils.json")
    @patch("sagemaker.huggingface.llm_utils.urllib")
    @patch("sagemaker.huggingface.llm_utils.json")
    @patch("sagemaker.model_uris.retrieve")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_sentence_similarity(
        self,
        mock_serveSettings,
        mock_model_uris_retrieve,
        mock_llm_utils_json,
        mock_llm_utils_urllib,
        mock_model_json,
        mock_model_urllib,
        mock_image_uris_retrieve,
        mock_build_for_tei,
    ):
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_model_uris_retrieve.side_effect = KeyError
        mock_llm_utils_json.load.return_value = {"pipeline_tag": "sentence-similarity"}
        mock_llm_utils_urllib.request.Request.side_effect = Mock()

        mock_model_json.load.return_value = {"some": "config"}
        mock_model_urllib.request.Request.side_effect = Mock()
        mock_build_for_tei.side_effect = Mock()

        mock_image_uris_retrieve.return_value = "https://some-image-uri"

        model_builder = ModelBuilder(model="bloom-560m", schema_builder=schema_builder)
        model_builder.build(sagemaker_session=mock_session)

        mock_build_for_tei.assert_called_once()

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_transformers", Mock())
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._try_fetch_gpu_info")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.serve.utils.hf_utils.urllib")
    @patch("sagemaker.serve.utils.hf_utils.json")
    @patch("sagemaker.huggingface.llm_utils.urllib")
    @patch("sagemaker.huggingface.llm_utils.json")
    @patch("sagemaker.model_uris.retrieve")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_try_fetch_gpu_info_throws(
        self,
        mock_serveSettings,
        mock_model_uris_retrieve,
        mock_llm_utils_json,
        mock_llm_utils_urllib,
        mock_model_json,
        mock_model_urllib,
        mock_image_uris_retrieve,
        mock_can_fit_on_single_gpu,
    ):
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_model_uris_retrieve.side_effect = KeyError
        mock_llm_utils_json.load.return_value = {"pipeline_tag": "text-classification"}
        mock_llm_utils_urllib.request.Request.side_effect = Mock()

        mock_model_json.load.return_value = {"some": "config"}
        mock_model_urllib.request.Request.side_effect = Mock()

        mock_image_uris_retrieve.return_value = "https://some-image-uri"
        mock_can_fit_on_single_gpu.side_effect = ValueError

        model_builder = ModelBuilder(model="gpt_llm_burt")
        model_builder.build(sagemaker_session=mock_session)

        self.assertEqual(model_builder._can_fit_on_single_gpu(), False)

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_transformers", Mock())
    @patch("sagemaker.serve.builder.model_builder._total_inference_model_size_mib")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.serve.utils.hf_utils.urllib")
    @patch("sagemaker.serve.utils.hf_utils.json")
    @patch("sagemaker.huggingface.llm_utils.urllib")
    @patch("sagemaker.huggingface.llm_utils.json")
    @patch("sagemaker.model_uris.retrieve")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_total_inference_model_size_mib_throws(
        self,
        mock_serveSettings,
        mock_model_uris_retrieve,
        mock_llm_utils_json,
        mock_llm_utils_urllib,
        mock_model_json,
        mock_model_urllib,
        mock_image_uris_retrieve,
        mock_total_inference_model_size_mib,
    ):
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_model_uris_retrieve.side_effect = KeyError
        mock_llm_utils_json.load.return_value = {"pipeline_tag": "text-classification"}
        mock_llm_utils_urllib.request.Request.side_effect = Mock()

        mock_model_json.load.return_value = {"some": "config"}
        mock_model_urllib.request.Request.side_effect = Mock()

        mock_image_uris_retrieve.return_value = "https://some-image-uri"
        mock_total_inference_model_size_mib.side_effect = ValueError

        model_builder = ModelBuilder(model="gpt_llm_burt")
        model_builder.build(sagemaker_session=mock_session)

        self.assertEqual(model_builder._can_fit_on_single_gpu(), False)

    @patch("sagemaker.serve.builder.tgi_builder.HuggingFaceModel")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.serve.utils.hf_utils.urllib")
    @patch("sagemaker.serve.utils.hf_utils.json")
    @patch("sagemaker.huggingface.llm_utils.urllib")
    @patch("sagemaker.huggingface.llm_utils.json")
    @patch("sagemaker.model_uris.retrieve")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_build_happy_path_override_with_task_provided(
        self,
        mock_serveSettings,
        mock_model_uris_retrieve,
        mock_llm_utils_json,
        mock_llm_utils_urllib,
        mock_model_json,
        mock_model_urllib,
        mock_image_uris_retrieve,
        mock_hf_model,
    ):
        # Setup mocks

        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        # HF Pipeline Tag
        mock_model_uris_retrieve.side_effect = KeyError
        mock_llm_utils_json.load.return_value = {"pipeline_tag": "fill-mask"}
        mock_llm_utils_urllib.request.Request.side_effect = Mock()

        # HF Model config
        mock_model_json.load.return_value = {"some": "config"}
        mock_model_urllib.request.Request.side_effect = Mock()

        mock_image_uris_retrieve.return_value = "https://some-image-uri"

        model_builder = ModelBuilder(
            model="bert-base-uncased", model_metadata={"HF_TASK": "text-generation"}
        )
        model_builder.build(sagemaker_session=mock_session)

        self.assertIsNotNone(model_builder.schema_builder)
        sample_inputs, sample_outputs = task.retrieve_local_schemas("text-generation")
        self.assertEqual(
            sample_inputs["inputs"], model_builder.schema_builder.sample_input["inputs"]
        )
        self.assertEqual(sample_outputs, model_builder.schema_builder.sample_output)

    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.serve.utils.hf_utils.urllib")
    @patch("sagemaker.serve.utils.hf_utils.json")
    @patch("sagemaker.huggingface.llm_utils.urllib")
    @patch("sagemaker.huggingface.llm_utils.json")
    @patch("sagemaker.model_uris.retrieve")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_build_task_override_with_invalid_task_provided(
        self,
        mock_serveSettings,
        mock_model_uris_retrieve,
        mock_llm_utils_json,
        mock_llm_utils_urllib,
        mock_model_json,
        mock_model_urllib,
        mock_image_uris_retrieve,
    ):
        # Setup mocks

        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        # HF Pipeline Tag
        mock_model_uris_retrieve.side_effect = KeyError
        mock_llm_utils_json.load.return_value = {"pipeline_tag": "fill-mask"}
        mock_llm_utils_urllib.request.Request.side_effect = Mock()

        # HF Model config
        mock_model_json.load.return_value = {"some": "config"}
        mock_model_urllib.request.Request.side_effect = Mock()

        mock_image_uris_retrieve.return_value = "https://some-image-uri"
        model_ids_with_invalid_task = {
            "bert-base-uncased": "invalid-task",
            "bert-large-uncased-whole-word-masking-finetuned-squad": "",
        }
        for model_id in model_ids_with_invalid_task:
            provided_task = model_ids_with_invalid_task[model_id]
            model_builder = ModelBuilder(model=model_id, model_metadata={"HF_TASK": provided_task})

            self.assertRaisesRegex(
                TaskNotFoundException,
                f"Error Message: HuggingFace Schema builder samples for {provided_task} could not be found locally or "
                f"via remote.",
                lambda: model_builder.build(sagemaker_session=mock_session),
            )

    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.model_uris.retrieve")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_build_task_override_with_invalid_model_provided(
        self,
        mock_serveSettings,
        mock_model_uris_retrieve,
        mock_image_uris_retrieve,
    ):
        # Setup mocks

        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        # HF Pipeline Tag
        mock_model_uris_retrieve.side_effect = KeyError

        mock_image_uris_retrieve.return_value = "https://some-image-uri"
        invalid_model_id = ""
        provided_task = "fill-mask"

        model_builder = ModelBuilder(
            model=invalid_model_id, model_metadata={"HF_TASK": provided_task}
        )
        with self.assertRaises(Exception):
            model_builder.build(sagemaker_session=mock_session)

    @patch("os.makedirs", Mock())
    @patch("sagemaker.serve.builder.model_builder._maintain_lineage_tracking_for_mlflow_model")
    @patch("sagemaker.serve.builder.model_builder._detect_framework_and_version")
    @patch("sagemaker.serve.builder.model_builder.prepare_for_torchserve")
    @patch("sagemaker.serve.builder.model_builder.save_pkl")
    @patch("sagemaker.serve.builder.model_builder._copy_directory_contents")
    @patch("sagemaker.serve.builder.model_builder._generate_mlflow_artifact_path")
    @patch("sagemaker.serve.builder.model_builder._get_all_flavor_metadata")
    @patch("sagemaker.serve.builder.model_builder._select_container_for_mlflow_model")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    @patch("sagemaker.serve.builder.model_builder.SageMakerEndpointMode")
    @patch("sagemaker.serve.builder.model_builder.Model")
    @patch("builtins.open", new_callable=mock_open, read_data="data")
    @patch("os.path.isfile", return_value=True)
    @patch("os.path.exists")
    def test_build_mlflow_model_local_input_happy(
        self,
        mock_path_exists,
        mock_is_file,
        mock_open,
        mock_sdk_model,
        mock_sageMakerEndpointMode,
        mock_serveSettings,
        mock_detect_container,
        mock_get_all_flavor_metadata,
        mock_generate_mlflow_artifact_path,
        mock_copy_directory_contents,
        mock_save_pkl,
        mock_prepare_for_torchserve,
        mock_detect_fw_version,
        mock_lineage_tracking,
    ):
        # setup mocks

        mock_detect_container.return_value = mock_image_uri
        mock_get_all_flavor_metadata.return_value = {"sklearn": "some_data"}
        mock_generate_mlflow_artifact_path.return_value = "some_path"

        mock_prepare_for_torchserve.return_value = mock_secret_key

        # Mock _ServeSettings
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_path_exists.side_effect = lambda path: True if path == "test_path" else False

        mock_mode = Mock()
        mock_sageMakerEndpointMode.side_effect = lambda inference_spec, model_server: (
            mock_mode if inference_spec is None and model_server == ModelServer.TORCHSERVE else None
        )
        mock_mode.prepare.return_value = (
            model_data,
            ENV_VAR_PAIR,
        )

        updated_env_var = deepcopy(ENV_VARS)
        updated_env_var.update({"MLFLOW_MODEL_FLAVOR": "sklearn"})
        mock_model_obj = Mock()
        mock_sdk_model.return_value = mock_model_obj

        mock_session.sagemaker_client._user_agent_creator.to_string = lambda: "sample agent"

        # run
        builder = ModelBuilder(
            schema_builder=schema_builder, model_metadata={"MLFLOW_MODEL_PATH": MODEL_PATH}
        )
        build_result = builder.build(sagemaker_session=mock_session)

        # assert model returned by builder is expected
        self.assertEqual(mock_model_obj, build_result)
        self.assertEqual(build_result.mode, Mode.SAGEMAKER_ENDPOINT)
        self.assertEqual(build_result.modes, {str(Mode.SAGEMAKER_ENDPOINT): mock_mode})
        self.assertEqual(build_result.serve_settings, mock_setting_object)
        self.assertEqual(builder.env_vars["MLFLOW_MODEL_FLAVOR"], "sklearn")

        build_result.deploy(
            initial_instance_count=1, instance_type=mock_instance_type, mode=Mode.SAGEMAKER_ENDPOINT
        )
        mock_lineage_tracking.assert_called_once()

    @patch("os.makedirs", Mock())
    @patch("sagemaker.serve.builder.model_builder._detect_framework_and_version")
    @patch("sagemaker.serve.builder.model_builder.prepare_for_torchserve")
    @patch("sagemaker.serve.builder.model_builder.save_pkl")
    @patch("sagemaker.serve.builder.model_builder._copy_directory_contents")
    @patch("sagemaker.serve.builder.model_builder._generate_mlflow_artifact_path")
    @patch("sagemaker.serve.builder.model_builder._get_all_flavor_metadata")
    @patch("sagemaker.serve.builder.model_builder._select_container_for_mlflow_model")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    @patch("sagemaker.serve.builder.model_builder.SageMakerEndpointMode")
    @patch("sagemaker.serve.builder.model_builder.Model")
    @patch("builtins.open", new_callable=mock_open, read_data="data")
    @patch("os.path.isfile", return_value=True)
    @patch("os.path.exists")
    def test_build_mlflow_model_local_input_happy_flavor_server_mismatch(
        self,
        mock_path_exists,
        mock_is_file,
        mock_open,
        mock_sdk_model,
        mock_sageMakerEndpointMode,
        mock_serveSettings,
        mock_detect_container,
        mock_get_all_flavor_metadata,
        mock_generate_mlflow_artifact_path,
        mock_copy_directory_contents,
        mock_save_pkl,
        mock_prepare_for_torchserve,
        mock_detect_fw_version,
    ):
        # setup mocks

        mock_detect_container.return_value = mock_image_uri
        mock_get_all_flavor_metadata.return_value = {"sklearn": "some_data"}
        mock_generate_mlflow_artifact_path.return_value = "some_path"

        mock_prepare_for_torchserve.return_value = mock_secret_key

        # Mock _ServeSettings
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_path_exists.side_effect = lambda path: True if path == "test_path" else False

        mock_mode = Mock()
        mock_sageMakerEndpointMode.side_effect = lambda inference_spec, model_server: (
            mock_mode if inference_spec is None and model_server == ModelServer.TORCHSERVE else None
        )
        mock_mode.prepare.return_value = (
            model_data,
            ENV_VAR_PAIR,
        )

        updated_env_var = deepcopy(ENV_VARS)
        updated_env_var.update({"MLFLOW_MODEL_FLAVOR": "sklearn"})
        mock_model_obj = Mock()
        mock_sdk_model.return_value = mock_model_obj

        mock_session.sagemaker_client._user_agent_creator.to_string = lambda: "sample agent"

        # run
        builder = ModelBuilder(
            schema_builder=schema_builder,
            model_metadata={"MLFLOW_MODEL_PATH": MODEL_PATH},
            model_server=ModelServer.TENSORFLOW_SERVING,
        )
        with self.assertRaises(ValueError):
            builder.build(
                Mode.SAGEMAKER_ENDPOINT,
                mock_role_arn,
                mock_session,
            )

    @patch("os.makedirs", Mock())
    @patch("sagemaker.serve.builder.model_builder.S3Downloader.list")
    @patch("sagemaker.serve.builder.model_builder._detect_framework_and_version")
    @patch("sagemaker.serve.builder.model_builder.prepare_for_torchserve")
    @patch("sagemaker.serve.builder.model_builder.save_pkl")
    @patch("sagemaker.serve.builder.model_builder._download_s3_artifacts")
    @patch("sagemaker.serve.builder.model_builder._generate_mlflow_artifact_path")
    @patch("sagemaker.serve.builder.model_builder._get_all_flavor_metadata")
    @patch("sagemaker.serve.builder.model_builder._select_container_for_mlflow_model")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    @patch("sagemaker.serve.builder.model_builder.SageMakerEndpointMode")
    @patch("sagemaker.serve.builder.model_builder.Model")
    @patch("builtins.open", new_callable=mock_open, read_data="data")
    @patch("os.path.exists")
    def test_build_mlflow_model_s3_input_happy(
        self,
        mock_path_exists,
        mock_open,
        mock_sdk_model,
        mock_sageMakerEndpointMode,
        mock_serveSettings,
        mock_detect_container,
        mock_get_all_flavor_metadata,
        mock_generate_mlflow_artifact_path,
        mock_download_s3_artifacts,
        mock_save_pkl,
        mock_prepare_for_torchserve,
        mock_detect_fw_version,
        mock_s3_downloader,
    ):
        # setup mocks
        mock_s3_downloader.return_value = ["s3://some_path/MLmodel"]

        mock_detect_container.return_value = mock_image_uri
        mock_get_all_flavor_metadata.return_value = {"sklearn": "some_data"}
        mock_generate_mlflow_artifact_path.return_value = "some_path"

        mock_prepare_for_torchserve.return_value = mock_secret_key

        # Mock _ServeSettings
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_path_exists.side_effect = lambda path: True if path == "test_path" else False

        mock_mode = Mock()
        mock_sageMakerEndpointMode.side_effect = lambda inference_spec, model_server: (
            mock_mode if inference_spec is None and model_server == ModelServer.TORCHSERVE else None
        )
        mock_mode.prepare.return_value = (
            model_data,
            ENV_VAR_PAIR,
        )

        updated_env_var = deepcopy(ENV_VARS)
        updated_env_var.update({"MLFLOW_MODEL_FLAVOR": "sklearn"})
        mock_model_obj = Mock()
        mock_sdk_model.return_value = mock_model_obj

        mock_session.sagemaker_client._user_agent_creator.to_string = lambda: "sample agent"

        # run
        builder = ModelBuilder(
            schema_builder=schema_builder, model_metadata={"MLFLOW_MODEL_PATH": "s3://test_path/"}
        )
        build_result = builder.build(sagemaker_session=mock_session)

        # assert model returned by builder is expected
        self.assertEqual(mock_model_obj, build_result)
        self.assertEqual(build_result.mode, Mode.SAGEMAKER_ENDPOINT)
        self.assertEqual(build_result.modes, {str(Mode.SAGEMAKER_ENDPOINT): mock_mode})
        self.assertEqual(build_result.serve_settings, mock_setting_object)
        self.assertEqual(builder.env_vars["MLFLOW_MODEL_FLAVOR"], "sklearn")

    @patch("os.makedirs", Mock())
    @patch("sagemaker.serve.builder.model_builder.S3Downloader.list")
    @patch("sagemaker.serve.builder.model_builder._detect_framework_and_version")
    @patch("sagemaker.serve.builder.model_builder.prepare_for_torchserve")
    @patch("sagemaker.serve.builder.model_builder.save_pkl")
    @patch("sagemaker.serve.builder.model_builder._download_s3_artifacts")
    @patch("sagemaker.serve.builder.model_builder._generate_mlflow_artifact_path")
    @patch("sagemaker.serve.builder.model_builder._get_all_flavor_metadata")
    @patch("sagemaker.serve.builder.model_builder._select_container_for_mlflow_model")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    @patch("sagemaker.serve.builder.model_builder.SageMakerEndpointMode")
    @patch("sagemaker.serve.builder.model_builder.Model")
    @patch("builtins.open", new_callable=mock_open, read_data="data")
    @patch("os.path.exists")
    def test_build_mlflow_model_s3_input_non_mlflow_case(
        self,
        mock_path_exists,
        mock_open,
        mock_sdk_model,
        mock_sageMakerEndpointMode,
        mock_serveSettings,
        mock_detect_container,
        mock_get_all_flavor_metadata,
        mock_generate_mlflow_artifact_path,
        mock_download_s3_artifacts,
        mock_save_pkl,
        mock_prepare_for_torchserve,
        mock_detect_fw_version,
        mock_s3_downloader,
    ):
        # setup mocks
        mock_s3_downloader.return_value = []
        mock_detect_container.return_value = mock_image_uri
        mock_get_all_flavor_metadata.return_value = {"sklearn": "some_data"}
        mock_generate_mlflow_artifact_path.return_value = "some_path"

        mock_prepare_for_torchserve.return_value = mock_secret_key

        # Mock _ServeSettings
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_path_exists.side_effect = lambda path: True if path == "test_path" else False

        mock_mode = Mock()
        mock_sageMakerEndpointMode.side_effect = lambda inference_spec, model_server: (
            mock_mode if inference_spec is None and model_server == ModelServer.TORCHSERVE else None
        )
        mock_mode.prepare.return_value = (
            model_data,
            ENV_VAR_PAIR,
        )

        updated_env_var = deepcopy(ENV_VARS)
        updated_env_var.update({"MLFLOW_MODEL_FLAVOR": "sklearn"})
        mock_model_obj = Mock()
        mock_sdk_model.return_value = mock_model_obj

        mock_session.sagemaker_client._user_agent_creator.to_string = lambda: "sample agent"

        # run
        builder = ModelBuilder(
            schema_builder=schema_builder, model_metadata={"MLFLOW_MODEL_PATH": "s3://test_path/"}
        )

        self.assertRaisesRegex(
            Exception,
            "Cannot detect required model or inference spec",
            builder.build,
            Mode.SAGEMAKER_ENDPOINT,
            mock_role_arn,
            mock_session,
        )

    @patch("os.makedirs", Mock())
    @patch("sagemaker.serve.builder.model_builder._maintain_lineage_tracking_for_mlflow_model")
    @patch("sagemaker.serve.builder.tf_serving_builder.prepare_for_tf_serving")
    @patch("sagemaker.serve.builder.model_builder.S3Downloader.list")
    @patch("sagemaker.serve.builder.model_builder._detect_framework_and_version")
    @patch("sagemaker.serve.builder.tf_serving_builder.save_pkl")
    @patch("sagemaker.serve.builder.model_builder._download_s3_artifacts")
    @patch("sagemaker.serve.builder.model_builder._generate_mlflow_artifact_path")
    @patch("sagemaker.serve.builder.model_builder._get_all_flavor_metadata")
    @patch("sagemaker.serve.builder.model_builder._select_container_for_mlflow_model")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    @patch("sagemaker.serve.builder.model_builder.SageMakerEndpointMode")
    @patch("sagemaker.serve.builder.tf_serving_builder.TensorFlowModel")
    @patch("builtins.open", new_callable=mock_open, read_data="data")
    @patch("os.path.exists")
    def test_build_mlflow_model_s3_input_tensorflow_serving_happy(
        self,
        mock_path_exists,
        mock_open,
        mock_sdk_model,
        mock_sageMakerEndpointMode,
        mock_serveSettings,
        mock_detect_container,
        mock_get_all_flavor_metadata,
        mock_generate_mlflow_artifact_path,
        mock_download_s3_artifacts,
        mock_save_pkl,
        mock_detect_fw_version,
        mock_s3_downloader,
        mock_prepare_for_tf_serving,
        mock_lineage_tracking,
    ):
        # setup mocks
        mock_s3_downloader.return_value = ["s3://some_path/MLmodel"]

        mock_detect_container.return_value = mock_image_uri
        mock_get_all_flavor_metadata.return_value = {"tensorflow": "some_data"}
        mock_generate_mlflow_artifact_path.return_value = "some_path"

        mock_prepare_for_tf_serving.return_value = mock_secret_key

        # Mock _ServeSettings
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_path_exists.side_effect = lambda path: True if path == "test_path" else False

        mock_mode = Mock()
        mock_sageMakerEndpointMode.side_effect = lambda inference_spec, model_server: (
            mock_mode
            if inference_spec is None and model_server == ModelServer.TENSORFLOW_SERVING
            else None
        )
        mock_mode.prepare.return_value = (
            model_data,
            ENV_VAR_PAIR,
        )

        updated_env_var = deepcopy(ENV_VARS)
        updated_env_var.update({"MLFLOW_MODEL_FLAVOR": "tensorflow"})
        mock_model_obj = Mock()
        mock_sdk_model.return_value = mock_model_obj

        mock_session.sagemaker_client._user_agent_creator.to_string = lambda: "sample agent"

        # run
        builder = ModelBuilder(
            schema_builder=schema_builder, model_metadata={"MLFLOW_MODEL_PATH": "s3://test_path/"}
        )
        build_result = builder.build(sagemaker_session=mock_session)
        self.assertEqual(mock_model_obj, build_result)
        self.assertEqual(build_result.mode, Mode.SAGEMAKER_ENDPOINT)
        self.assertEqual(build_result.modes, {str(Mode.SAGEMAKER_ENDPOINT): mock_mode})
        self.assertEqual(build_result.serve_settings, mock_setting_object)
        self.assertEqual(builder.env_vars["MLFLOW_MODEL_FLAVOR"], "tensorflow")

        build_result.deploy(
            initial_instance_count=1, instance_type=mock_instance_type, mode=Mode.SAGEMAKER_ENDPOINT
        )
        mock_lineage_tracking.assert_called_once()

    @patch("os.makedirs", Mock())
    @patch("sagemaker.serve.builder.tf_serving_builder.prepare_for_tf_serving")
    @patch("sagemaker.serve.builder.model_builder.S3Downloader.list")
    @patch("sagemaker.serve.builder.model_builder._detect_framework_and_version")
    @patch("sagemaker.serve.builder.tf_serving_builder.save_pkl")
    @patch("sagemaker.serve.builder.model_builder._download_s3_artifacts")
    @patch("sagemaker.serve.builder.model_builder._generate_mlflow_artifact_path")
    @patch("sagemaker.serve.builder.model_builder._get_all_flavor_metadata")
    @patch("sagemaker.serve.builder.model_builder._select_container_for_mlflow_model")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    @patch("sagemaker.serve.builder.model_builder.LocalContainerMode")
    @patch("sagemaker.serve.builder.tf_serving_builder.TensorFlowModel")
    @patch("builtins.open", new_callable=mock_open, read_data="data")
    @patch("os.path.exists")
    def test_build_mlflow_model_s3_input_tensorflow_serving_local_mode_happy(
        self,
        mock_path_exists,
        mock_open,
        mock_sdk_model,
        mock_local_container_mode,
        mock_serveSettings,
        mock_detect_container,
        mock_get_all_flavor_metadata,
        mock_generate_mlflow_artifact_path,
        mock_download_s3_artifacts,
        mock_save_pkl,
        mock_detect_fw_version,
        mock_s3_downloader,
        mock_prepare_for_tf_serving,
    ):
        # setup mocks
        mock_s3_downloader.return_value = ["s3://some_path/MLmodel"]

        mock_detect_container.return_value = mock_image_uri
        mock_get_all_flavor_metadata.return_value = {"tensorflow": "some_data"}
        mock_generate_mlflow_artifact_path.return_value = "some_path"

        mock_prepare_for_tf_serving.return_value = mock_secret_key

        # Mock _ServeSettings
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_path_exists.side_effect = lambda path: True if path == "test_path" else False

        mock_mode = Mock()
        mock_mode.prepare.side_effect = lambda: None
        mock_local_container_mode.return_value = mock_mode
        mock_mode.prepare.return_value = (
            model_data,
            ENV_VAR_PAIR,
        )

        updated_env_var = deepcopy(ENV_VARS)
        updated_env_var.update({"MLFLOW_MODEL_FLAVOR": "tensorflow"})
        mock_model_obj = Mock()
        mock_sdk_model.return_value = mock_model_obj

        mock_session.sagemaker_client._user_agent_creator.to_string = lambda: "sample agent"

        # run
        builder = ModelBuilder(
            mode=Mode.LOCAL_CONTAINER,
            schema_builder=schema_builder,
            model_metadata={"MLFLOW_MODEL_PATH": "s3://test_path/"},
        )
        build_result = builder.build(sagemaker_session=mock_session)
        self.assertEqual(mock_model_obj, build_result)
        self.assertEqual(build_result.mode, Mode.LOCAL_CONTAINER)
        self.assertEqual(build_result.modes, {str(Mode.LOCAL_CONTAINER): mock_mode})
        self.assertEqual(build_result.serve_settings, mock_setting_object)
        self.assertEqual(builder.env_vars["MLFLOW_MODEL_FLAVOR"], "tensorflow")

        predictor = build_result.deploy()
        assert isinstance(predictor, TensorflowServingLocalPredictor)

    @patch("os.makedirs", Mock())
    @patch(
        "sagemaker.serve.builder.model_builder.ModelBuilder._is_jumpstart_model_id",
        return_value=False,
    )
    @patch("sagemaker.serve.builder.tf_serving_builder.prepare_for_tf_serving")
    @patch("sagemaker.serve.builder.model_builder.S3Downloader.list")
    @patch("sagemaker.serve.builder.model_builder._detect_framework_and_version")
    @patch("sagemaker.serve.builder.model_builder.save_pkl")
    @patch("sagemaker.serve.builder.model_builder._download_s3_artifacts")
    @patch("sagemaker.serve.builder.model_builder._generate_mlflow_artifact_path")
    @patch("sagemaker.serve.builder.model_builder._get_all_flavor_metadata")
    @patch("sagemaker.serve.builder.model_builder._select_container_for_mlflow_model")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    @patch("sagemaker.serve.builder.model_builder.SageMakerEndpointMode")
    @patch("sagemaker.serve.builder.tf_serving_builder.TensorFlowModel")
    @patch("builtins.open", new_callable=mock_open, read_data="data")
    @patch("os.path.exists")
    def test_build_tensorflow_serving_non_mlflow_case(
        self,
        mock_path_exists,
        mock_open,
        mock_sdk_model,
        mock_sageMakerEndpointMode,
        mock_serveSettings,
        mock_detect_container,
        mock_get_all_flavor_metadata,
        mock_generate_mlflow_artifact_path,
        mock_download_s3_artifacts,
        mock_save_pkl,
        mock_detect_fw_version,
        mock_s3_downloader,
        mock_prepare_for_tf_serving,
        mock_is_jumpstart_model_id,
    ):
        mock_s3_downloader.return_value = []
        mock_detect_container.return_value = mock_image_uri
        mock_get_all_flavor_metadata.return_value = {"tensorflow": "some_data"}
        mock_generate_mlflow_artifact_path.return_value = "some_path"

        mock_prepare_for_tf_serving.return_value = mock_secret_key

        # Mock _ServeSettings
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_path_exists.side_effect = lambda path: True if path == "test_path" else False

        mock_mode = Mock()
        mock_sageMakerEndpointMode.side_effect = lambda inference_spec, model_server: (
            mock_mode
            if inference_spec is None and model_server == ModelServer.TENSORFLOW_SERVING
            else None
        )
        mock_mode.prepare.return_value = (
            model_data,
            ENV_VAR_PAIR,
        )

        updated_env_var = deepcopy(ENV_VARS)
        updated_env_var.update({"MLFLOW_MODEL_FLAVOR": "tensorflow"})
        mock_model_obj = Mock()
        mock_sdk_model.return_value = mock_model_obj

        mock_session.sagemaker_client._user_agent_creator.to_string = lambda: "sample agent"

        # run
        builder = ModelBuilder(
            model=mock_fw_model,
            schema_builder=schema_builder,
            model_server=ModelServer.TENSORFLOW_SERVING,
        )

        self.assertRaisesRegex(
            Exception,
            "Tensorflow Serving is currently only supported for mlflow models.",
            builder.build,
            Mode.SAGEMAKER_ENDPOINT,
            mock_role_arn,
            mock_session,
        )

    @patch.object(ModelBuilder, "_prepare_for_mode")
    @patch.object(ModelBuilder, "_build_for_djl")
    @patch.object(ModelBuilder, "_is_jumpstart_model_id", return_value=False)
    @patch.object(ModelBuilder, "_get_serve_setting", autospec=True)
    @patch("sagemaker.serve.utils.telemetry_logger._send_telemetry")
    def test_optimize(
        self,
        mock_send_telemetry,
        mock_get_serve_setting,
        mock_is_jumpstart_model_id,
        mock_build_for_djl,
        mock_prepare_for_mode,
    ):
        mock_sagemaker_session = Mock()

        mock_settings = Mock()
        mock_settings.telemetry_opt_out = False
        mock_get_serve_setting.return_value = mock_settings

        pysdk_model = Mock()
        pysdk_model.env = {"key": "val"}
        pysdk_model.add_tags.side_effect = lambda *arg, **kwargs: None

        mock_build_for_djl.side_effect = lambda **kwargs: pysdk_model
        mock_prepare_for_mode.side_effect = lambda *args, **kwargs: (
            {
                "S3DataSource": {
                    "S3Uri": "s3://uri",
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            },
            {"key": "val"},
        )

        builder = ModelBuilder(
            schema_builder=SchemaBuilder(
                sample_input={"inputs": "Hello", "parameters": {}},
                sample_output=[{"generated_text": "Hello"}],
            ),
            model="meta-llama/Meta-Llama-3-8B",
            sagemaker_session=mock_sagemaker_session,
            env_vars={"HF_TOKEN": "token"},
            model_metadata={"CUSTOM_MODEL_PATH": "/tmp/modelbuilders/code"},
        )
        builder.pysdk_model = pysdk_model

        job_name = "my-optimization-job"
        instance_type = "ml.g5.24xlarge"
        output_path = "s3://my-bucket/output"
        quantization_config = {
            "Image": "quantization-image-uri",
            "OverrideEnvironment": {"OPTION_QUANTIZE": "awq"},
        }
        env_vars = {"Var1": "value", "Var2": "value"}
        kms_key = "arn:aws:kms:us-west-2:123456789012:key/my-key-id"
        max_runtime_in_sec = 3600
        tags = [
            {"Key": "Project", "Value": "my-project"},
            {"Key": "Environment", "Value": "production"},
        ]
        vpc_config = {
            "SecurityGroupIds": ["sg-01234567890abcdef", "sg-fedcba9876543210"],
            "Subnets": ["subnet-01234567", "subnet-89abcdef"],
        }

        mock_sagemaker_session.wait_for_optimization_job.side_effect = lambda *args, **kwargs: {
            "OptimizationJobArn": "arn:aws:sagemaker:us-west-2:123456789012:optimization-job/my-optimization-job",
            "OptimizationJobName": "my-optimization-job",
        }

        builder.optimize(
            instance_type=instance_type,
            output_path=output_path,
            role_arn=mock_role_arn,
            job_name=job_name,
            quantization_config=quantization_config,
            env_vars=env_vars,
            kms_key=kms_key,
            max_runtime_in_sec=max_runtime_in_sec,
            tags=tags,
            vpc_config=vpc_config,
        )

        self.assertEqual(builder.env_vars["HUGGING_FACE_HUB_TOKEN"], "token")
        self.assertEqual(builder.model_server, ModelServer.DJL_SERVING)

        assert mock_send_telemetry.call_count == 2
        mock_sagemaker_session.sagemaker_client.create_optimization_job.assert_called_once_with(
            OptimizationJobName="my-optimization-job",
            DeploymentInstanceType="ml.g5.24xlarge",
            RoleArn="arn:aws:iam::123456789012:role/SageMakerRole",
            OptimizationEnvironment={"Var1": "value", "Var2": "value"},
            ModelSource={"S3": {"S3Uri": "s3://uri"}},
            OptimizationConfigs=[
                {
                    "ModelQuantizationConfig": {
                        "Image": "quantization-image-uri",
                        "OverrideEnvironment": {"OPTION_QUANTIZE": "awq"},
                    }
                }
            ],
            OutputConfig={
                "S3OutputLocation": "s3://my-bucket/output",
                "KmsKeyId": "arn:aws:kms:us-west-2:123456789012:key/my-key-id",
            },
            StoppingCondition={"MaxRuntimeInSeconds": 3600},
            Tags=[
                {"Key": "Project", "Value": "my-project"},
                {"Key": "Environment", "Value": "production"},
            ],
            VpcConfig={
                "SecurityGroupIds": ["sg-01234567890abcdef", "sg-fedcba9876543210"],
                "Subnets": ["subnet-01234567", "subnet-89abcdef"],
            },
        )

    def test_handle_mlflow_input_without_mlflow_model_path(self):
        builder = ModelBuilder(model_metadata={})
        assert not builder._has_mlflow_arguments()

    @patch("importlib.util.find_spec")
    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.get_run")
    @patch.object(ModelBuilder, "_mlflow_metadata_exists", autospec=True)
    @patch.object(ModelBuilder, "_initialize_for_mlflow", autospec=True)
    @patch("sagemaker.serve.builder.model_builder._download_s3_artifacts")
    @patch("sagemaker.serve.builder.model_builder._validate_input_for_mlflow")
    def test_handle_mlflow_input_run_id(
        self,
        mock_validate,
        mock_s3_downloader,
        mock_initialize,
        mock_check_mlflow_model,
        mock_get_run,
        mock_set_tracking_uri,
        mock_find_spec,
    ):
        mock_find_spec.return_value = True
        mock_run_info = Mock()
        mock_run_info.info.artifact_uri = "s3://bucket/path"
        mock_get_run.return_value = mock_run_info
        mock_check_mlflow_model.return_value = True
        mock_s3_downloader.return_value = ["s3://some_path/MLmodel"]

        builder = ModelBuilder(
            model_metadata={
                "MLFLOW_MODEL_PATH": "runs:/runid/mlflow-path",
                "MLFLOW_TRACKING_ARN": "arn:aws:sagemaker:us-west-2:000000000000:mlflow-tracking-server/test",
            }
        )
        builder._handle_mlflow_input()
        mock_initialize.assert_called_once_with(builder, "s3://bucket/path/mlflow-path")

    @patch("importlib.util.find_spec")
    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.MlflowClient.get_model_version")
    @patch.object(ModelBuilder, "_mlflow_metadata_exists", autospec=True)
    @patch.object(ModelBuilder, "_initialize_for_mlflow", autospec=True)
    @patch("sagemaker.serve.builder.model_builder._download_s3_artifacts")
    @patch("sagemaker.serve.builder.model_builder._validate_input_for_mlflow")
    def test_handle_mlflow_input_registry_path_with_model_version(
        self,
        mock_validate,
        mock_s3_downloader,
        mock_initialize,
        mock_check_mlflow_model,
        mock_get_model_version,
        mock_set_tracking_uri,
        mock_find_spec,
    ):
        mock_find_spec.return_value = True
        mock_registry_path = Mock()
        mock_registry_path.source = "s3://bucket/path/"
        mock_get_model_version.return_value = mock_registry_path
        mock_check_mlflow_model.return_value = True
        mock_s3_downloader.return_value = ["s3://some_path/MLmodel"]

        builder = ModelBuilder(
            model_metadata={
                "MLFLOW_MODEL_PATH": "models:/model-name/1",
                "MLFLOW_TRACKING_ARN": "arn:aws:sagemaker:us-west-2:000000000000:mlflow-tracking-server/test",
            }
        )
        builder._handle_mlflow_input()
        mock_initialize.assert_called_once_with(builder, "s3://bucket/path/")

    @patch("importlib.util.find_spec")
    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.MlflowClient.get_model_version_by_alias")
    @patch.object(ModelBuilder, "_mlflow_metadata_exists", autospec=True)
    @patch.object(ModelBuilder, "_initialize_for_mlflow", autospec=True)
    @patch("sagemaker.serve.builder.model_builder._download_s3_artifacts")
    @patch("sagemaker.serve.builder.model_builder._validate_input_for_mlflow")
    def test_handle_mlflow_input_registry_path_with_model_alias(
        self,
        mock_validate,
        mock_s3_downloader,
        mock_initialize,
        mock_check_mlflow_model,
        mock_get_model_version_by_alias,
        mock_set_tracking_uri,
        mock_find_spec,
    ):
        mock_find_spec.return_value = True
        mock_registry_path = Mock()
        mock_registry_path.source = "s3://bucket/path"
        mock_get_model_version_by_alias.return_value = mock_registry_path
        mock_check_mlflow_model.return_value = True
        mock_s3_downloader.return_value = ["s3://some_path/MLmodel"]

        builder = ModelBuilder(
            model_metadata={
                "MLFLOW_MODEL_PATH": "models:/model-name@production",
                "MLFLOW_TRACKING_ARN": "arn:aws:sagemaker:us-west-2:000000000000:mlflow-tracking-server/test",
            }
        )
        builder._handle_mlflow_input()
        mock_initialize.assert_called_once_with(builder, "s3://bucket/path/")

    @patch("mlflow.MlflowClient.get_model_version")
    @patch.object(ModelBuilder, "_mlflow_metadata_exists", autospec=True)
    @patch.object(ModelBuilder, "_initialize_for_mlflow", autospec=True)
    @patch("sagemaker.serve.builder.model_builder._download_s3_artifacts")
    @patch("sagemaker.serve.builder.model_builder._validate_input_for_mlflow")
    def test_handle_mlflow_input_registry_path_missing_tracking_server_arn(
        self,
        mock_validate,
        mock_s3_downloader,
        mock_initialize,
        mock_check_mlflow_model,
        mock_get_model_version,
    ):
        mock_registry_path = Mock()
        mock_registry_path.source = "s3://bucket/path"
        mock_get_model_version.return_value = mock_registry_path
        mock_check_mlflow_model.return_value = True
        mock_s3_downloader.return_value = ["s3://some_path/MLmodel"]

        builder = ModelBuilder(
            model_metadata={
                "MLFLOW_MODEL_PATH": "models:/model-name/1",
            }
        )
        self.assertRaisesRegex(
            Exception,
            "%s is not provided in ModelMetadata or through set_tracking_arn "
            "but MLflow model path was provided." % MLFLOW_TRACKING_ARN,
            builder._handle_mlflow_input,
        )

    @patch.object(ModelBuilder, "_mlflow_metadata_exists", autospec=True)
    @patch.object(ModelBuilder, "_initialize_for_mlflow", autospec=True)
    @patch("sagemaker.serve.builder.model_builder._download_s3_artifacts")
    @patch("sagemaker.serve.builder.model_builder._validate_input_for_mlflow")
    def test_handle_mlflow_input_model_package_arn(
        self, mock_validate, mock_s3_downloader, mock_initialize, mock_check_mlflow_model
    ):
        mock_check_mlflow_model.return_value = True
        mock_s3_downloader.return_value = ["s3://some_path/MLmodel"]
        mock_model_package = {"SourceUri": "s3://bucket/path"}
        mock_session.sagemaker_client.describe_model_package.return_value = mock_model_package

        builder = ModelBuilder(
            model_metadata={
                "MLFLOW_MODEL_PATH": "arn:aws:sagemaker:us-west-2:000000000000:model-package/test",
                "MLFLOW_TRACKING_ARN": "arn:aws:sagemaker:us-west-2:000000000000:mlflow-tracking-server/test",
            },
            sagemaker_session=mock_session,
        )
        builder._handle_mlflow_input()
        mock_initialize.assert_called_once_with(builder, "s3://bucket/path")

    @patch("importlib.util.find_spec", Mock(return_value=True))
    @patch("mlflow.set_tracking_uri")
    def test_set_tracking_arn_success(self, mock_set_tracking_uri):
        builder = ModelBuilder(
            model_metadata={
                "MLFLOW_MODEL_PATH": "arn:aws:sagemaker:us-west-2:000000000000:model-package/test",
            }
        )
        tracking_arn = "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/test"
        builder.set_tracking_arn(tracking_arn)
        mock_set_tracking_uri.assert_called_once_with(tracking_arn)
        assert builder.model_metadata[MLFLOW_TRACKING_ARN] == tracking_arn

    @patch("importlib.util.find_spec", Mock(return_value=False))
    def test_set_tracking_arn_mlflow_not_installed(self):
        builder = ModelBuilder(
            model_metadata={
                "MLFLOW_MODEL_PATH": "arn:aws:sagemaker:us-west-2:000000000000:model-package/test",
            }
        )
        tracking_arn = "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/test"
        self.assertRaisesRegex(
            ImportError,
            "Unable to import sagemaker_mlflow, check if sagemaker_mlflow is installed",
            builder.set_tracking_arn,
            tracking_arn,
        )

    @patch.object(ModelBuilder, "_get_serve_setting", autospec=True)
    def test_optimize_local_mode(self, mock_get_serve_setting):
        model_builder = ModelBuilder(
            model="meta-textgeneration-llama-3-70b", mode=Mode.LOCAL_CONTAINER
        )

        self.assertRaisesRegex(
            ValueError,
            "Model optimization is only supported in Sagemaker Endpoint Mode.",
            lambda: model_builder.optimize(
                instance_type="ml.g5.24xlarge",
                quantization_config={"OverrideEnvironment": {"OPTION_QUANTIZE": "awq"}},
            ),
        )

    @patch.object(ModelBuilder, "_prepare_for_mode")
    @patch.object(ModelBuilder, "_get_serve_setting", autospec=True)
    def test_optimize_for_hf_with_both_quantization_and_compilation(
        self,
        mock_get_serve_setting,
        mock_prepare_for_mode,
    ):
        mock_prepare_for_mode.side_effect = lambda *args, **kwargs: (
            {
                "S3DataSource": {
                    "CompressionType": "None",
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://bucket/code/code/",
                }
            },
            {"DTYPE": "bfloat16"},
        )

        mock_pysdk_model = Mock()
        mock_pysdk_model.model_data = None
        mock_pysdk_model.env = {"HF_MODEL_ID": "meta-llama/Meta-Llama-3-8B-Instruc"}

        model_builder = ModelBuilder(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            env_vars={"HF_TOKEN": "token"},
            model_metadata={
                "CUSTOM_MODEL_PATH": "s3://bucket/path/",
            },
            role_arn="role-arn",
            instance_type="ml.g5.2xlarge",
        )

        model_builder.pysdk_model = mock_pysdk_model

        out_put = model_builder._optimize_for_hf(
            job_name="job_name-123",
            quantization_config={
                "OverrideEnvironment": {"OPTION_QUANTIZE": "awq"},
            },
            compilation_config={"OverrideEnvironment": {"OPTION_TENSOR_PARALLEL_DEGREE": "2"}},
            output_path="s3://bucket/code/",
        )

        self.assertEqual(model_builder.env_vars["HF_TOKEN"], "token")
        self.assertEqual(model_builder.role_arn, "role-arn")
        self.assertEqual(model_builder.instance_type, "ml.g5.2xlarge")
        self.assertEqual(model_builder.pysdk_model.env["OPTION_QUANTIZE"], "awq")
        self.assertEqual(model_builder.pysdk_model.env["OPTION_TENSOR_PARALLEL_DEGREE"], "2")
        self.assertEqual(
            out_put,
            {
                "OptimizationJobName": "job_name-123",
                "DeploymentInstanceType": "ml.g5.2xlarge",
                "RoleArn": "role-arn",
                "ModelSource": {"S3": {"S3Uri": "s3://bucket/code/code/"}},
                "OptimizationConfigs": [
                    {
                        "ModelQuantizationConfig": {
                            "OverrideEnvironment": {"OPTION_QUANTIZE": "awq"}
                        }
                    },
                    {
                        "ModelCompilationConfig": {
                            "OverrideEnvironment": {"OPTION_TENSOR_PARALLEL_DEGREE": "2"}
                        }
                    },
                ],
                "OutputConfig": {"S3OutputLocation": "s3://bucket/code/"},
            },
        )

    @patch.object(ModelBuilder, "_get_serve_setting", autospec=True)
    def test_optimize_exclusive_sharding(self, mock_get_serve_setting):
        mock_sagemaker_session = Mock()
        model_builder = ModelBuilder(
            model="meta-textgeneration-llama-3-70b",
            sagemaker_session=mock_sagemaker_session,
        )

        self.assertRaisesRegex(
            ValueError,
            "Optimizations that use Compilation and Sharding are not supported for GPU instances.",
            lambda: model_builder.optimize(
                instance_type="ml.g5.24xlarge",
                quantization_config={"OverrideEnvironment": {"OPTION_QUANTIZE": "awq"}},
                compilation_config={"OverrideEnvironment": {"OPTION_QUANTIZE": "awq"}},
                sharding_config={"OverrideEnvironment": {"OPTION_QUANTIZE": "awq"}},
            ),
        )

    @patch.object(ModelBuilder, "_get_serve_setting", autospec=True)
    def test_optimize_exclusive_sharding_args(self, mock_get_serve_setting):
        mock_sagemaker_session = Mock()
        model_builder = ModelBuilder(
            model="meta-textgeneration-llama-3-70b",
            sagemaker_session=mock_sagemaker_session,
        )

        self.assertRaisesRegex(
            ValueError,
            "OPTION_TENSOR_PARALLEL_DEGREE is a required environment variable with sharding config.",
            lambda: model_builder.optimize(
                instance_type="ml.g5.24xlarge",
                sharding_config={"OverrideEnvironment": {"OPTION_QUANTIZE": "awq"}},
            ),
        )

    @patch.object(ModelBuilder, "_prepare_for_mode")
    @patch.object(ModelBuilder, "_get_serve_setting", autospec=True)
    def test_optimize_for_hf_with_custom_s3_path(
        self,
        mock_get_serve_setting,
        mock_prepare_for_mode,
    ):
        mock_prepare_for_mode.side_effect = lambda *args, **kwargs: (
            {
                "S3DataSource": {
                    "CompressionType": "None",
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://bucket/code/code/",
                }
            },
            {"DTYPE": "bfloat16"},
        )

        mock_pysdk_model = Mock()
        mock_pysdk_model.model_data = None
        mock_pysdk_model.env = {"HF_MODEL_ID": "meta-llama/Meta-Llama-3-8B-Instruc"}

        model_builder = ModelBuilder(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            env_vars={"HF_TOKEN": "token"},
            model_metadata={
                "CUSTOM_MODEL_PATH": "s3://bucket/path/",
            },
            role_arn="role-arn",
            instance_type="ml.g5.2xlarge",
        )

        model_builder.pysdk_model = mock_pysdk_model

        out_put = model_builder._optimize_for_hf(
            job_name="job_name-123",
            quantization_config={
                "OverrideEnvironment": {"OPTION_QUANTIZE": "awq"},
            },
            output_path="s3://bucket/code/",
        )

        self.assertEqual(model_builder.env_vars["HF_TOKEN"], "token")
        self.assertEqual(model_builder.role_arn, "role-arn")
        self.assertEqual(model_builder.instance_type, "ml.g5.2xlarge")
        self.assertEqual(model_builder.pysdk_model.env["OPTION_QUANTIZE"], "awq")
        self.assertEqual(
            out_put,
            {
                "OptimizationJobName": "job_name-123",
                "DeploymentInstanceType": "ml.g5.2xlarge",
                "RoleArn": "role-arn",
                "ModelSource": {"S3": {"S3Uri": "s3://bucket/code/code/"}},
                "OptimizationConfigs": [
                    {"ModelQuantizationConfig": {"OverrideEnvironment": {"OPTION_QUANTIZE": "awq"}}}
                ],
                "OutputConfig": {"S3OutputLocation": "s3://bucket/code/"},
            },
        )

    @patch(
        "sagemaker.serve.builder.model_builder.download_huggingface_model_metadata", autospec=True
    )
    @patch.object(ModelBuilder, "_prepare_for_mode")
    @patch.object(ModelBuilder, "_get_serve_setting", autospec=True)
    def test_optimize_for_hf_without_custom_s3_path(
        self,
        mock_get_serve_setting,
        mock_prepare_for_mode,
        mock_download_huggingface_model_metadata,
    ):
        mock_prepare_for_mode.side_effect = lambda *args, **kwargs: (
            {
                "S3DataSource": {
                    "CompressionType": "None",
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://bucket/code/code/",
                }
            },
            {"DTYPE": "bfloat16"},
        )

        mock_pysdk_model = Mock()
        mock_pysdk_model.model_data = None
        mock_pysdk_model.env = {"HF_MODEL_ID": "meta-llama/Meta-Llama-3-8B-Instruc"}

        model_builder = ModelBuilder(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            env_vars={"HUGGING_FACE_HUB_TOKEN": "token"},
            role_arn="role-arn",
            instance_type="ml.g5.2xlarge",
        )

        model_builder.pysdk_model = mock_pysdk_model

        out_put = model_builder._optimize_for_hf(
            job_name="job_name-123",
            quantization_config={
                "OverrideEnvironment": {"OPTION_QUANTIZE": "awq"},
            },
            output_path="s3://bucket/code/",
        )

        self.assertEqual(model_builder.role_arn, "role-arn")
        self.assertEqual(model_builder.instance_type, "ml.g5.2xlarge")
        self.assertEqual(model_builder.pysdk_model.env["OPTION_QUANTIZE"], "awq")
        self.assertEqual(
            out_put,
            {
                "OptimizationJobName": "job_name-123",
                "DeploymentInstanceType": "ml.g5.2xlarge",
                "RoleArn": "role-arn",
                "ModelSource": {"S3": {"S3Uri": "s3://bucket/code/code/"}},
                "OptimizationConfigs": [
                    {"ModelQuantizationConfig": {"OverrideEnvironment": {"OPTION_QUANTIZE": "awq"}}}
                ],
                "OutputConfig": {"S3OutputLocation": "s3://bucket/code/"},
            },
        )

    def test_deploy_invalid_inputs(self):
        model_builder = ModelBuilder(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            env_vars={"HUGGING_FACE_HUB_TOKEN": "token"},
            role_arn="role-arn",
            instance_type="ml.g5.2xlarge",
        )
        inputs = {"endpoint_name": "endpoint-001"}

        try:
            model_builder.deploy(**inputs)
        except ValueError as e:
            assert "Model Needs to be built before deploying" in str(e)

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._is_jumpstart_model_id")
    def test_display_benchmark_metrics_non_string_model(self, mock_is_jumpstart):
        """Test that ValueError is raised when model is not a string"""
        builder = ModelBuilder(model=Mock())  # Non-string model

        self.assertRaisesRegex(
            ValueError,
            "Benchmarking is only supported for JumpStart or HuggingFace models",
            builder.display_benchmark_metrics,
        )
        mock_is_jumpstart.assert_not_called()

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._is_jumpstart_model_id")
    @patch("sagemaker.serve.builder.jumpstart_builder.JumpStart.display_benchmark_metrics")
    def test_display_benchmark_metrics_jumpstart_model(
        self, mock_display_benchmark_metrics, mock_is_jumpstart
    ):
        """Test successful execution for jumpstart model"""
        mock_is_jumpstart.return_value = True

        builder = ModelBuilder(model="jumpstart-model-id")
        builder.display_benchmark_metrics()

        mock_is_jumpstart.assert_called_once()
        mock_display_benchmark_metrics.assert_called_once()

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._is_jumpstart_model_id")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._use_jumpstart_equivalent")
    @patch("sagemaker.serve.builder.jumpstart_builder.JumpStart.display_benchmark_metrics")
    def test_display_benchmark_metrics_with_jumpstart_equivalent(
        self, mock_display_benchmark_metrics, mock_has_equivalent, mock_is_jumpstart
    ):
        """Test successful execution for model with jumpstart equivalent"""
        mock_is_jumpstart.return_value = False
        mock_has_equivalent.return_value = True

        builder = ModelBuilder(model="hf-model-id")
        builder.display_benchmark_metrics()

        mock_is_jumpstart.assert_called_once()
        mock_has_equivalent.assert_called_once()
        mock_display_benchmark_metrics.assert_called_once()

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._is_jumpstart_model_id")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._use_jumpstart_equivalent")
    def test_display_benchmark_metrics_unsupported_model(
        self, mock_has_equivalent, mock_is_jumpstart
    ):
        """Test that ValueError is raised for unsupported models"""
        mock_is_jumpstart.return_value = False
        mock_has_equivalent.return_value = False

        builder = ModelBuilder(model="huggingface-model-id")

        self.assertRaisesRegex(
            ValueError,
            "This model does not have benchmark metrics yet",
            builder.display_benchmark_metrics,
        )

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._is_jumpstart_model_id")
    def test_get_deployment_config_non_string_model(self, mock_is_jumpstart):
        """Test that ValueError is raised when model is not a string"""
        builder = ModelBuilder(model=Mock())  # Non-string model

        self.assertRaisesRegex(
            ValueError,
            "Deployment config is only supported for JumpStart or HuggingFace models",
            builder.get_deployment_config,
        )
        mock_is_jumpstart.assert_not_called()

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._is_jumpstart_model_id")
    @patch("sagemaker.serve.builder.jumpstart_builder.JumpStart.get_deployment_config")
    def test_get_deployment_config_jumpstart_model(
        self, mock_get_deployment_config, mock_is_jumpstart
    ):
        """Test successful execution for jumpstart model"""
        mock_is_jumpstart.return_value = True

        builder = ModelBuilder(model="jumpstart-model-id")
        builder.get_deployment_config()

        mock_is_jumpstart.assert_called_once()
        mock_get_deployment_config.assert_called_once()

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._is_jumpstart_model_id")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._use_jumpstart_equivalent")
    @patch("sagemaker.serve.builder.jumpstart_builder.JumpStart.get_deployment_config")
    def test_get_deployment_config_with_jumpstart_equivalent(
        self, mock_get_deployment_config, mock_has_equivalent, mock_is_jumpstart
    ):
        """Test successful execution for model with jumpstart equivalent"""
        mock_is_jumpstart.return_value = False
        mock_has_equivalent.return_value = True

        builder = ModelBuilder(model="hf-model-id")
        builder.get_deployment_config()

        mock_is_jumpstart.assert_called_once()
        mock_has_equivalent.assert_called_once()
        mock_get_deployment_config.assert_called_once()

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._is_jumpstart_model_id")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._use_jumpstart_equivalent")
    def test_get_deployment_config_unsupported_model(self, mock_has_equivalent, mock_is_jumpstart):
        """Test that ValueError is raised for unsupported models"""
        mock_is_jumpstart.return_value = False
        mock_has_equivalent.return_value = False

        builder = ModelBuilder(model="huggingface-model-id")

        self.assertRaisesRegex(
            ValueError,
            "This model does not have any deployment config yet",
            builder.get_deployment_config,
        )

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._is_jumpstart_model_id")
    def test_list_deployment_configs_non_string_model(self, mock_is_jumpstart):
        """Test that ValueError is raised when model is not a string"""
        builder = ModelBuilder(model=Mock())  # Non-string model

        self.assertRaisesRegex(
            ValueError,
            "Deployment config is only supported for JumpStart or HuggingFace models",
            builder.list_deployment_configs,
        )
        mock_is_jumpstart.assert_not_called()

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._is_jumpstart_model_id")
    @patch("sagemaker.serve.builder.jumpstart_builder.JumpStart.list_deployment_configs")
    def test_list_deployment_configs_jumpstart_model(
        self, mock_list_deployment_configs, mock_is_jumpstart
    ):
        """Test successful execution for jumpstart model"""
        mock_is_jumpstart.return_value = True

        builder = ModelBuilder(model="jumpstart-model-id")
        builder.list_deployment_configs()

        mock_is_jumpstart.assert_called_once()
        mock_list_deployment_configs.assert_called_once()

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._is_jumpstart_model_id")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._use_jumpstart_equivalent")
    @patch("sagemaker.serve.builder.jumpstart_builder.JumpStart.list_deployment_configs")
    def test_list_deployment_configs_with_jumpstart_equivalent(
        self, mock_list_deployment_configs, mock_has_equivalent, mock_is_jumpstart
    ):
        """Test successful execution for model with jumpstart equivalent"""
        mock_is_jumpstart.return_value = False
        mock_has_equivalent.return_value = True

        builder = ModelBuilder(model="hf-model-id")
        builder.list_deployment_configs()

        mock_is_jumpstart.assert_called_once()
        mock_has_equivalent.assert_called_once()
        mock_list_deployment_configs.assert_called_once()

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._is_jumpstart_model_id")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._use_jumpstart_equivalent")
    def test_list_deployment_configs_unsupported_model(
        self, mock_has_equivalent, mock_is_jumpstart
    ):
        """Test that ValueError is raised for unsupported models"""
        mock_is_jumpstart.return_value = False
        mock_has_equivalent.return_value = False

        builder = ModelBuilder(model="huggingface-model-id")

        self.assertRaisesRegex(
            ValueError,
            "This model does not have any deployment config yet",
            builder.list_deployment_configs,
        )

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._is_jumpstart_model_id")
    def test_set_deployment_config_non_string_model(self, mock_is_jumpstart):
        """Test that ValueError is raised when model is not a string"""
        builder = ModelBuilder(model=Mock())  # Non-string model
        instance_type = "ml.g5.xlarge"
        config_name = "config-name"
        self.assertRaisesRegex(
            ValueError,
            "Deployment config is only supported for JumpStart or HuggingFace models",
            builder.set_deployment_config,
            config_name,
            instance_type,
        )
        mock_is_jumpstart.assert_not_called()

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._is_jumpstart_model_id")
    @patch("sagemaker.serve.builder.jumpstart_builder.JumpStart.set_deployment_config")
    def test_set_deployment_config_jumpstart_model(
        self, mock_set_deployment_config, mock_is_jumpstart
    ):
        """Test successful execution for jumpstart model"""
        mock_is_jumpstart.return_value = True
        instance_type = "ml.g5.xlarge"
        config_name = "config-name"

        builder = ModelBuilder(model="jumpstart-model-id")
        builder.set_deployment_config(config_name, instance_type)

        mock_is_jumpstart.assert_called_once()
        mock_set_deployment_config.assert_called_once()

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._is_jumpstart_model_id")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._use_jumpstart_equivalent")
    @patch("sagemaker.serve.builder.jumpstart_builder.JumpStart.set_deployment_config")
    def test_set_deployment_config_with_jumpstart_equivalent(
        self, mock_set_deployment_config, mock_has_equivalent, mock_is_jumpstart
    ):
        """Test successful execution for model with jumpstart equivalent"""
        mock_is_jumpstart.return_value = False
        mock_has_equivalent.return_value = True
        instance_type = "ml.g5.xlarge"
        config_name = "config-name"

        builder = ModelBuilder(model="hf-model-id")
        builder.set_deployment_config(config_name, instance_type)

        mock_is_jumpstart.assert_called_once()
        mock_has_equivalent.assert_called_once()
        mock_set_deployment_config.assert_called_once()

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._is_jumpstart_model_id")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._use_jumpstart_equivalent")
    def test_set_deployment_config_unsupported_model(self, mock_has_equivalent, mock_is_jumpstart):
        """Test that ValueError is raised for unsupported models"""
        mock_is_jumpstart.return_value = False
        mock_has_equivalent.return_value = False
        instance_type = "ml.g5.xlarge"
        config_name = "config-name"

        builder = ModelBuilder(model="huggingface-model-id")

        self.assertRaisesRegex(
            ValueError,
            f"The deployment config {config_name} cannot be set on this model",
            builder.set_deployment_config,
            config_name,
            instance_type,
        )

    @patch(
        "sagemaker.serve.builder.model_builder.ModelBuilder._retrieve_hugging_face_model_mapping"
    )
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._is_gated_model")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_jumpstart")
    def test_use_jumpstart_equivalent_return_true(
        self, mock_build_for_jumpstart, mock_is_gated_model, mock_retrieve_mapping
    ):
        """Test that _use_jumpstart_equivalent returns True when equivalent exists"""
        mock_retrieve_mapping.return_value = {
            "HuggingFaceH4/zephyr-7b-beta": {
                "jumpstart-model-id": "js-model",
                "jumpstart-model-version": "1.0.0",
                "hf-model-repo-sha": None,
            }
        }
        mock_is_gated_model.return_value = False

        builder = ModelBuilder(model="HuggingFaceH4/zephyr-7b-beta")

        self.assertTrue(builder._use_jumpstart_equivalent())

    @patch(
        "sagemaker.serve.builder.model_builder.ModelBuilder._retrieve_hugging_face_model_mapping"
    )
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._is_gated_model")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_jumpstart")
    def test_use_jumpstart_equivalent_return_true_with_schema_builder(
        self, mock_build_for_jumpstart, mock_is_gated_model, mock_retrieve_mapping
    ):
        """Test that _use_jumpstart_equivalent returns True when equivalent exists"""
        mock_retrieve_mapping.return_value = {
            "HuggingFaceH4/zephyr-7b-beta": {
                "jumpstart-model-id": "js-model",
                "jumpstart-model-version": "1.0.0",
                "hf-model-repo-sha": None,
            }
        }
        mock_is_gated_model.return_value = False

        builder = ModelBuilder(model="HuggingFaceH4/zephyr-7b-beta", sagemaker_session=mock_session)

        self.assertTrue(builder._use_jumpstart_equivalent())
        self.assertIsNotNone(builder.schema_builder)
        inputs, outputs = task.retrieve_local_schemas("text-generation")
        self.assertEqual(builder.schema_builder.sample_input["inputs"], inputs["inputs"])
        self.assertEqual(builder.schema_builder.sample_output, outputs)

    @patch(
        "sagemaker.serve.builder.model_builder.ModelBuilder._retrieve_hugging_face_model_mapping"
    )
    def test_use_jumpstart_equivalent_return_false(self, mock_retrieve_mapping):
        """Test that _use_jumpstart_equivalent returns false when equivalent doesn't exist"""
        mock_retrieve_mapping.return_value = {
            "hf-model-id": {
                "jumpstart-model-id": "js-model",
                "jumpstart-model-version": "1.0.0",
                "hf-model-repo-sha": None,
            }
        }

        builder = ModelBuilder(model="model-id")

        self.assertFalse(builder._use_jumpstart_equivalent())

    def test_use_jumpstart_equivalent_return_false_with_env_vars(self):
        """Test that _use_jumpstart_equivalent returns false when env_vars is provided"""
        builder = ModelBuilder(model="model-id", env_vars={"mock-key": "mock-value"})

        self.assertFalse(builder._use_jumpstart_equivalent())

    def test_use_jumpstart_equivalent_return_false_with_image_uri(self):
        """Test that _use_jumpstart_equivalent returns false when image_uri is provided"""
        builder = ModelBuilder(model="model-id", image_uri="mock-uri")

        self.assertFalse(builder._use_jumpstart_equivalent())

    @patch("sagemaker.serve.builder.model_builder.JumpStartS3PayloadAccessor")
    @patch("sagemaker.serve.builder.model_builder.get_jumpstart_content_bucket")
    def test_retrieve_hugging_face_model_mapping(self, mock_content_bucket, mock_payload_accessor):
        """Test that _retrieve_hugging_face_model_mapping returns the correct mapping"""
        mock_get_object = Mock()
        mock_get_object.return_value = (
            '{"js-model-id": {"hf-model-id": "hf-model", "jumpstart-model-version": "1.0.0"}}'
        )
        mock_payload_accessor.get_object_cached = mock_get_object
        expected_mapping = {
            "hf-model": {
                "jumpstart-model-id": "js-model-id",
                "jumpstart-model-version": "1.0.0",
                "hf-model-repo-sha": None,
                "merged-at": None,
            }
        }

        builder = ModelBuilder(model="hf-model", sagemaker_session=mock_session)

        self.assertEqual(builder._retrieve_hugging_face_model_mapping(), expected_mapping)

    @patch.object(ModelBuilder, "_prepare_for_mode")
    @patch.object(ModelBuilder, "_get_serve_setting", autospec=True)
    def test_optimize_with_gpu_instance_and_llama_3_1_and_compilation(
        self,
        mock_get_serve_setting,
        mock_prepare_for_mode,
    ):
        mock_prepare_for_mode.side_effect = lambda *args, **kwargs: (
            {
                "S3DataSource": {
                    "CompressionType": "None",
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://bucket/code/code/",
                }
            },
            {"DTYPE": "bfloat16"},
        )

        mock_pysdk_model = Mock()
        mock_pysdk_model.model_data = None
        mock_pysdk_model.env = {"HF_MODEL_ID": "meta-llama/Meta-Llama-3-2-8B-Instruct"}

        sample_input = {"inputs": "dummy prompt", "parameters": {}}

        sample_output = [{"generated_text": "dummy response"}]

        dummy_schema_builder = SchemaBuilder(sample_input, sample_output)

        model_builder = ModelBuilder(
            model="meta-llama/Meta-Llama-3-2-8B-Instruct",
            schema_builder=dummy_schema_builder,
            env_vars={"HF_TOKEN": "token"},
            model_metadata={
                "CUSTOM_MODEL_PATH": "s3://bucket/path/",
            },
            role_arn="role-arn",
            instance_type="ml.g5.2xlarge",
        )

        model_builder.pysdk_model = mock_pysdk_model

        self.assertRaisesRegex(
            ValueError,
            "Compilation is not supported for models greater than Llama-3.0 with a GPU instance.",
            lambda: model_builder.optimize(
                job_name="job_name-123",
                instance_type="ml.g5.24xlarge",
                compilation_config={"OverrideEnvironment": {"OPTION_TENSOR_PARALLEL_DEGREE": "2"}},
                output_path="s3://bucket/code/",
            ),
        )

    @patch.object(ModelBuilder, "_prepare_for_mode")
    @patch.object(ModelBuilder, "_get_serve_setting", autospec=True)
    def test_optimize_with_gpu_instance_and_compilation_with_speculative_decoding(
        self,
        mock_get_serve_setting,
        mock_prepare_for_mode,
    ):
        mock_prepare_for_mode.side_effect = lambda *args, **kwargs: (
            {
                "S3DataSource": {
                    "CompressionType": "None",
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://bucket/code/code/",
                }
            },
            {"DTYPE": "bfloat16"},
        )

        mock_pysdk_model = Mock()
        mock_pysdk_model.model_data = None
        mock_pysdk_model.env = {"HF_MODEL_ID": "modelid"}

        sample_input = {"inputs": "dummy prompt", "parameters": {}}

        sample_output = [{"generated_text": "dummy response"}]

        dummy_schema_builder = SchemaBuilder(sample_input, sample_output)

        model_builder = ModelBuilder(
            model="modelid",
            schema_builder=dummy_schema_builder,
            env_vars={"HF_TOKEN": "token"},
            model_metadata={
                "CUSTOM_MODEL_PATH": "s3://bucket/path/",
            },
            role_arn="role-arn",
            instance_type="ml.g5.2xlarge",
        )

        model_builder.pysdk_model = mock_pysdk_model

        self.assertRaisesRegex(
            ValueError,
            "Optimizations that use Compilation and Speculative Decoding are not supported for GPU instances.",
            lambda: model_builder.optimize(
                job_name="job_name-123",
                instance_type="ml.g5.24xlarge",
                speculative_decoding_config={
                    "ModelProvider": "custom",
                    "ModelSource": "s3://data-source",
                },
                compilation_config={"OverrideEnvironment": {"OPTION_TENSOR_PARALLEL_DEGREE": "2"}},
                output_path="s3://bucket/code/",
            ),
        )


class TestModelBuilderOptimizationSharding(unittest.TestCase):
    @patch.object(ModelBuilder, "_prepare_for_mode")
    @patch.object(ModelBuilder, "_build_for_djl")
    @patch.object(ModelBuilder, "_is_jumpstart_model_id", return_value=False)
    @patch.object(ModelBuilder, "_get_serve_setting", autospec=True)
    @patch("sagemaker.serve.utils.telemetry_logger._send_telemetry")
    def test_optimize_sharding_with_env_vars(
        self,
        mock_send_telemetry,
        mock_get_serve_setting,
        mock_is_jumpstart_model_id,
        mock_build_for_djl,
        mock_prepare_for_mode,
    ):
        mock_sagemaker_session = Mock()

        mock_settings = Mock()
        mock_settings.telemetry_opt_out = False
        mock_get_serve_setting.return_value = mock_settings

        pysdk_model = Mock()
        pysdk_model.env = {"key": "val"}
        pysdk_model.add_tags.side_effect = lambda *arg, **kwargs: None

        mock_build_for_djl.side_effect = lambda **kwargs: pysdk_model
        mock_prepare_for_mode.side_effect = lambda *args, **kwargs: (
            {
                "S3DataSource": {
                    "S3Uri": "s3://uri",
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            },
            {"key": "val"},
        )

        builder = ModelBuilder(
            schema_builder=SchemaBuilder(
                sample_input={"inputs": "Hello", "parameters": {}},
                sample_output=[{"generated_text": "Hello"}],
            ),
            model="meta-llama/Meta-Llama-3-8B",
            sagemaker_session=mock_sagemaker_session,
            env_vars={"HF_TOKEN": "token"},
            model_metadata={"CUSTOM_MODEL_PATH": "/tmp/modelbuilders/code"},
        )
        builder.pysdk_model = pysdk_model

        job_name = "my-optimization-job"
        instance_type = "ml.g5.24xlarge"
        output_path = "s3://my-bucket/output"
        sharding_config = {"key": "value"}
        env_vars = {"OPTION_TENSOR_PARALLEL_DEGREE": "1"}
        kms_key = "arn:aws:kms:us-west-2:123456789012:key/my-key-id"
        max_runtime_in_sec = 3600
        tags = [
            {"Key": "Project", "Value": "my-project"},
            {"Key": "Environment", "Value": "production"},
        ]
        vpc_config = {
            "SecurityGroupIds": ["sg-01234567890abcdef", "sg-fedcba9876543210"],
            "Subnets": ["subnet-01234567", "subnet-89abcdef"],
        }

        mock_sagemaker_session.wait_for_optimization_job.side_effect = lambda *args, **kwargs: {
            "OptimizationJobArn": "arn:aws:sagemaker:us-west-2:123456789012:optimization-job/my-optimization-job",
            "OptimizationJobName": "my-optimization-job",
        }

        # With override
        builder.optimize(
            instance_type=instance_type,
            output_path=output_path,
            role_arn=mock_role_arn,
            job_name=job_name,
            sharding_config=sharding_config,
            env_vars=env_vars,
            kms_key=kms_key,
            max_runtime_in_sec=max_runtime_in_sec,
            tags=tags,
            vpc_config=vpc_config,
        )

        self.assertEqual(builder.env_vars["HUGGING_FACE_HUB_TOKEN"], "token")
        self.assertEqual(builder.model_server, ModelServer.DJL_SERVING)

        assert mock_send_telemetry.call_count == 2
        mock_sagemaker_session.sagemaker_client.create_optimization_job.assert_called_once_with(
            OptimizationJobName="my-optimization-job",
            DeploymentInstanceType="ml.g5.24xlarge",
            RoleArn="arn:aws:iam::123456789012:role/SageMakerRole",
            OptimizationEnvironment={"OPTION_TENSOR_PARALLEL_DEGREE": "1"},
            ModelSource={"S3": {"S3Uri": "s3://uri"}},
            OptimizationConfigs=[{"ModelShardingConfig": {"key": "value"}}],
            OutputConfig={
                "S3OutputLocation": "s3://my-bucket/output",
                "KmsKeyId": "arn:aws:kms:us-west-2:123456789012:key/my-key-id",
            },
            StoppingCondition={"MaxRuntimeInSeconds": 3600},
            Tags=[
                {"Key": "Project", "Value": "my-project"},
                {"Key": "Environment", "Value": "production"},
            ],
            VpcConfig={
                "SecurityGroupIds": ["sg-01234567890abcdef", "sg-fedcba9876543210"],
                "Subnets": ["subnet-01234567", "subnet-89abcdef"],
            },
        )

    @patch.object(ModelBuilder, "_prepare_for_mode")
    @patch.object(ModelBuilder, "_build_for_djl")
    @patch.object(ModelBuilder, "_is_jumpstart_model_id", return_value=False)
    @patch.object(ModelBuilder, "_get_serve_setting", autospec=True)
    @patch("sagemaker.serve.utils.telemetry_logger._send_telemetry")
    def test_optimize_sharding_with_override_and_env_var(
        self,
        mock_send_telemetry,
        mock_get_serve_setting,
        mock_is_jumpstart_model_id,
        mock_build_for_djl,
        mock_prepare_for_mode,
    ):
        mock_sagemaker_session = Mock()

        mock_settings = Mock()
        mock_settings.telemetry_opt_out = False
        mock_get_serve_setting.return_value = mock_settings

        pysdk_model = Mock()
        pysdk_model.env = {"key": "val"}
        pysdk_model.add_tags.side_effect = lambda *arg, **kwargs: None

        mock_build_for_djl.side_effect = lambda **kwargs: pysdk_model
        mock_prepare_for_mode.side_effect = lambda *args, **kwargs: (
            {
                "S3DataSource": {
                    "S3Uri": "s3://uri",
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            },
            {"key": "val"},
        )

        builder = ModelBuilder(
            schema_builder=SchemaBuilder(
                sample_input={"inputs": "Hello", "parameters": {}},
                sample_output=[{"generated_text": "Hello"}],
            ),
            model="meta-llama/Meta-Llama-3-8B",
            sagemaker_session=mock_sagemaker_session,
            env_vars={"HF_TOKEN": "token"},
            model_metadata={"CUSTOM_MODEL_PATH": "/tmp/modelbuilders/code"},
        )
        builder.pysdk_model = pysdk_model

        job_name = "my-optimization-job"
        instance_type = "ml.g5.24xlarge"
        output_path = "s3://my-bucket/output"
        sharding_config = {"OverrideEnvironment": {"OPTION_TENSOR_PARALLEL_DEGREE": "1"}}
        env_vars = {"OPTION_TENSOR_PARALLEL_DEGREE": "1"}
        kms_key = "arn:aws:kms:us-west-2:123456789012:key/my-key-id"
        max_runtime_in_sec = 3600
        tags = [
            {"Key": "Project", "Value": "my-project"},
            {"Key": "Environment", "Value": "production"},
        ]
        vpc_config = {
            "SecurityGroupIds": ["sg-01234567890abcdef", "sg-fedcba9876543210"],
            "Subnets": ["subnet-01234567", "subnet-89abcdef"],
        }

        mock_sagemaker_session.wait_for_optimization_job.side_effect = lambda *args, **kwargs: {
            "OptimizationJobArn": "arn:aws:sagemaker:us-west-2:123456789012:optimization-job/my-optimization-job",
            "OptimizationJobName": "my-optimization-job",
        }

        # With override
        builder.optimize(
            instance_type=instance_type,
            output_path=output_path,
            role_arn=mock_role_arn,
            job_name=job_name,
            sharding_config=sharding_config,
            env_vars=env_vars,
            kms_key=kms_key,
            max_runtime_in_sec=max_runtime_in_sec,
            tags=tags,
            vpc_config=vpc_config,
        )

        self.assertEqual(builder.env_vars["HUGGING_FACE_HUB_TOKEN"], "token")
        self.assertEqual(builder.model_server, ModelServer.DJL_SERVING)

        assert mock_send_telemetry.call_count == 2
        mock_sagemaker_session.sagemaker_client.create_optimization_job.assert_called_once_with(
            OptimizationJobName="my-optimization-job",
            DeploymentInstanceType="ml.g5.24xlarge",
            RoleArn="arn:aws:iam::123456789012:role/SageMakerRole",
            OptimizationEnvironment={"OPTION_TENSOR_PARALLEL_DEGREE": "1"},
            ModelSource={"S3": {"S3Uri": "s3://uri"}},
            OptimizationConfigs=[
                {
                    "ModelShardingConfig": {
                        "OverrideEnvironment": {"OPTION_TENSOR_PARALLEL_DEGREE": "1"}
                    }
                }
            ],
            OutputConfig={
                "S3OutputLocation": "s3://my-bucket/output",
                "KmsKeyId": "arn:aws:kms:us-west-2:123456789012:key/my-key-id",
            },
            StoppingCondition={"MaxRuntimeInSeconds": 3600},
            Tags=[
                {"Key": "Project", "Value": "my-project"},
                {"Key": "Environment", "Value": "production"},
            ],
            VpcConfig={
                "SecurityGroupIds": ["sg-01234567890abcdef", "sg-fedcba9876543210"],
                "Subnets": ["subnet-01234567", "subnet-89abcdef"],
            },
        )

    @patch.object(ModelBuilder, "_prepare_for_mode")
    @patch.object(ModelBuilder, "_build_for_djl")
    @patch.object(ModelBuilder, "_is_jumpstart_model_id", return_value=False)
    @patch.object(ModelBuilder, "_get_serve_setting", autospec=True)
    @patch("sagemaker.serve.utils.telemetry_logger._send_telemetry")
    def test_optimize_sharding_with_override(
        self,
        mock_send_telemetry,
        mock_get_serve_setting,
        mock_is_jumpstart_model_id,
        mock_build_for_djl,
        mock_prepare_for_mode,
    ):
        mock_sagemaker_session = Mock()

        mock_settings = Mock()
        mock_settings.telemetry_opt_out = False
        mock_get_serve_setting.return_value = mock_settings

        pysdk_model = Mock()
        pysdk_model.env = {"key": "val"}
        pysdk_model.add_tags.side_effect = lambda *arg, **kwargs: None

        mock_build_for_djl.side_effect = lambda **kwargs: pysdk_model
        mock_prepare_for_mode.side_effect = lambda *args, **kwargs: (
            {
                "S3DataSource": {
                    "S3Uri": "s3://uri",
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            },
            {"key": "val"},
        )

        builder = ModelBuilder(
            schema_builder=SchemaBuilder(
                sample_input={"inputs": "Hello", "parameters": {}},
                sample_output=[{"generated_text": "Hello"}],
            ),
            model="meta-llama/Meta-Llama-3-8B",
            sagemaker_session=mock_sagemaker_session,
            env_vars={"HF_TOKEN": "token"},
            model_metadata={"CUSTOM_MODEL_PATH": "/tmp/modelbuilders/code"},
        )
        builder.pysdk_model = pysdk_model

        job_name = "my-optimization-job"
        instance_type = "ml.g5.24xlarge"
        output_path = "s3://my-bucket/output"
        sharding_config = {"OverrideEnvironment": {"OPTION_TENSOR_PARALLEL_DEGREE": "1"}}
        env_vars = {"Var1": "value", "Var2": "value"}
        kms_key = "arn:aws:kms:us-west-2:123456789012:key/my-key-id"
        max_runtime_in_sec = 3600
        tags = [
            {"Key": "Project", "Value": "my-project"},
            {"Key": "Environment", "Value": "production"},
        ]
        vpc_config = {
            "SecurityGroupIds": ["sg-01234567890abcdef", "sg-fedcba9876543210"],
            "Subnets": ["subnet-01234567", "subnet-89abcdef"],
        }

        mock_sagemaker_session.wait_for_optimization_job.side_effect = lambda *args, **kwargs: {
            "OptimizationJobArn": "arn:aws:sagemaker:us-west-2:123456789012:optimization-job/my-optimization-job",
            "OptimizationJobName": "my-optimization-job",
        }

        # With override
        builder.optimize(
            instance_type=instance_type,
            output_path=output_path,
            role_arn=mock_role_arn,
            job_name=job_name,
            sharding_config=sharding_config,
            env_vars=env_vars,
            kms_key=kms_key,
            max_runtime_in_sec=max_runtime_in_sec,
            tags=tags,
            vpc_config=vpc_config,
        )

        self.assertEqual(builder.env_vars["HUGGING_FACE_HUB_TOKEN"], "token")
        self.assertEqual(builder.model_server, ModelServer.DJL_SERVING)

        assert mock_send_telemetry.call_count == 2
        mock_sagemaker_session.sagemaker_client.create_optimization_job.assert_called_once_with(
            OptimizationJobName="my-optimization-job",
            DeploymentInstanceType="ml.g5.24xlarge",
            RoleArn="arn:aws:iam::123456789012:role/SageMakerRole",
            OptimizationEnvironment={"Var1": "value", "Var2": "value"},
            ModelSource={"S3": {"S3Uri": "s3://uri"}},
            OptimizationConfigs=[
                {
                    "ModelShardingConfig": {
                        "OverrideEnvironment": {"OPTION_TENSOR_PARALLEL_DEGREE": "1"}
                    }
                }
            ],
            OutputConfig={
                "S3OutputLocation": "s3://my-bucket/output",
                "KmsKeyId": "arn:aws:kms:us-west-2:123456789012:key/my-key-id",
            },
            StoppingCondition={"MaxRuntimeInSeconds": 3600},
            Tags=[
                {"Key": "Project", "Value": "my-project"},
                {"Key": "Environment", "Value": "production"},
            ],
            VpcConfig={
                "SecurityGroupIds": ["sg-01234567890abcdef", "sg-fedcba9876543210"],
                "Subnets": ["subnet-01234567", "subnet-89abcdef"],
            },
        )

        # squeeze in some validations
        with self.assertRaises(ValueError):
            builder.enable_network_isolation = True
            builder.optimize(sharding_config={})

    @patch.object(ModelBuilder, "_prepare_for_mode")
    @patch.object(ModelBuilder, "_build_for_jumpstart")
    @patch.object(ModelBuilder, "_is_jumpstart_model_id", return_value=True)
    @patch.object(ModelBuilder, "_get_serve_setting", autospec=True)
    @patch("sagemaker.serve.utils.telemetry_logger._send_telemetry")
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._is_gated_model", return_value=False
    )
    @patch(
        "sagemaker.serve.builder.jumpstart_builder.JumpStart._find_compatible_deployment_config",
        return_value=Mock(),
    )
    def test_optimize_sharding_with_override_for_js(
        self,
        mock_find_compatible_deployment_config,
        mock_is_gated_model,
        mock_send_telemetry,
        mock_get_serve_setting,
        mock_is_jumpstart_model_id,
        mock_build_for_jumpstart,
        mock_prepare_for_mode,
    ):
        mock_sagemaker_session = Mock()

        mock_settings = Mock()
        mock_settings.telemetry_opt_out = False
        mock_get_serve_setting.return_value = mock_settings

        pysdk_model = Mock()
        pysdk_model.env = {"key": "val"}
        pysdk_model._enable_network_isolation = True
        pysdk_model.add_tags.side_effect = lambda *arg, **kwargs: None
        pysdk_model.init_kwargs = {
            "image_uri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.29.0-lmi11.0.0-cu124"
        }

        mock_build_for_jumpstart.side_effect = lambda **kwargs: pysdk_model
        mock_prepare_for_mode.side_effect = lambda *args, **kwargs: (
            {
                "S3DataSource": {
                    "S3Uri": "s3://uri",
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            },
            {"key": "val"},
        )

        builder = ModelBuilder(
            schema_builder=SchemaBuilder(
                sample_input={"inputs": "Hello", "parameters": {}},
                sample_output=[{"generated_text": "Hello"}],
            ),
            model="meta-llama/Meta-Llama-3-8B",
            sagemaker_session=mock_sagemaker_session,
            env_vars={"HF_TOKEN": "token"},
            model_metadata={"CUSTOM_MODEL_PATH": "/tmp/modelbuilders/code"},
        )
        builder.pysdk_model = pysdk_model

        job_name = "my-optimization-job"
        instance_type = "ml.g5.24xlarge"
        output_path = "s3://my-bucket/output"
        sharding_config = {"OverrideEnvironment": {"OPTION_TENSOR_PARALLEL_DEGREE": "1"}}
        env_vars = {"Var1": "value", "Var2": "value"}
        kms_key = "arn:aws:kms:us-west-2:123456789012:key/my-key-id"
        max_runtime_in_sec = 3600
        tags = [
            {"Key": "Project", "Value": "my-project"},
            {"Key": "Environment", "Value": "production"},
        ]
        vpc_config = {
            "SecurityGroupIds": ["sg-01234567890abcdef", "sg-fedcba9876543210"],
            "Subnets": ["subnet-01234567", "subnet-89abcdef"],
        }

        mock_sagemaker_session.wait_for_optimization_job.side_effect = lambda *args, **kwargs: {
            "OptimizationJobArn": "arn:aws:sagemaker:us-west-2:123456789012:optimization-job/my-optimization-job",
            "OptimizationJobName": "my-optimization-job",
        }

        # With override
        model = builder.optimize(
            instance_type=instance_type,
            output_path=output_path,
            role_arn=mock_role_arn,
            job_name=job_name,
            sharding_config=sharding_config,
            env_vars=env_vars,
            kms_key=kms_key,
            max_runtime_in_sec=max_runtime_in_sec,
            tags=tags,
            vpc_config=vpc_config,
        )

        self.assertEqual(builder.env_vars["HUGGING_FACE_HUB_TOKEN"], "token")

        assert mock_send_telemetry.call_count == 2
        mock_sagemaker_session.sagemaker_client.create_optimization_job.assert_called_once_with(
            OptimizationJobName="my-optimization-job",
            ModelSource={"S3": {"S3Uri": ANY}},
            DeploymentInstanceType="ml.g5.24xlarge",
            OptimizationConfigs=[
                {
                    "ModelShardingConfig": {
                        "Image": "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.31.0-lmi13.0.0-cu124",
                        "OverrideEnvironment": {"OPTION_TENSOR_PARALLEL_DEGREE": "1"},
                    },
                }
            ],
            OutputConfig={
                "S3OutputLocation": "s3://my-bucket/output",
                "KmsKeyId": "arn:aws:kms:us-west-2:123456789012:key/my-key-id",
            },
            RoleArn="arn:aws:iam::123456789012:role/SageMakerRole",
            OptimizationEnvironment={
                "key": "val",
                "Var1": "value",
                "Var2": "value",
                "OPTION_TENSOR_PARALLEL_DEGREE": "1",
            },
            StoppingCondition={"MaxRuntimeInSeconds": 3600},
            Tags=[
                {"Key": "Project", "Value": "my-project"},
                {"Key": "Environment", "Value": "production"},
            ],
            VpcConfig={
                "SecurityGroupIds": ["sg-01234567890abcdef", "sg-fedcba9876543210"],
                "Subnets": ["subnet-01234567", "subnet-89abcdef"],
            },
        )

        assert not model._enable_network_isolation

    def test_model_sharding_with_eni_fails(self):
        test_model = Model(role="mock role")
        test_model._is_sharded_model = True
        test_model._enable_network_isolation = True
        self.assertRaisesRegex(
            ValueError,
            (
                "EnableNetworkIsolation cannot be set to True since "
                "SageMaker Fast Model Loading of model requires network access."
            ),
            lambda: test_model.deploy(initial_instance_count=1, instance_type="ml.g5.24xlarge"),
        )


class TestModelBuilderOptimizeValidations(unittest.TestCase):

    def test_corner_cases_throw_errors(self):
        self.assertRaisesRegex(
            ValueError,
            "Optimizations that uses None instance type are not currently supported",
            lambda: _validate_optimization_configuration(
                is_jumpstart=False,
                sharding_config={"key": "value"},
                instance_type=None,
                quantization_config=None,
                speculative_decoding_config=None,
                compilation_config=None,
            ),
        )

        self.assertRaisesRegex(
            ValueError,
            (
                "Optimizations that provide no optimization configs "
                "are currently not support on both GPU and Neuron instances."
            ),
            lambda: _validate_optimization_configuration(
                is_jumpstart=False,
                instance_type="ml.g5.24xlarge",
                quantization_config=None,
                speculative_decoding_config=None,
                compilation_config=None,
                sharding_config=None,
            ),
        )

        _validate_optimization_configuration(
            is_jumpstart=True,
            instance_type="ml.inf2.xlarge",
            quantization_config=None,
            speculative_decoding_config=None,
            compilation_config=None,
            sharding_config=None,
        )

    def test_trt_and_vllm_configurations_throw_errors_for_rule_set(self):
        # Quantization:smoothquant without compilation
        self.assertRaisesRegex(
            ValueError,
            "Optimizations that use Quantization:smoothquant must be provided with Compilation for GPU instances.",
            lambda: _validate_optimization_configuration(
                is_jumpstart=False,
                instance_type="ml.g5.24xlarge",
                quantization_config={
                    "OverrideEnvironment": {"OPTION_QUANTIZE": "smoothquant"},
                },
                sharding_config=None,
                speculative_decoding_config=None,
                compilation_config=None,
            ),
        )

        # Invalid quantization technique
        self.assertRaisesRegex(
            ValueError,
            "Optimizations that use Quantization:test are not supported for GPU instances.",
            lambda: _validate_optimization_configuration(
                is_jumpstart=False,
                instance_type="ml.g5.24xlarge",
                quantization_config={
                    "OverrideEnvironment": {"OPTION_QUANTIZE": "test"},
                },
                sharding_config=None,
                speculative_decoding_config=None,
                compilation_config=None,
            ),
        )

    def test_neuron_configurations_throw_errors_for_rule_set(self):
        self.assertRaisesRegex(
            ValueError,
            "Optimizations that use Speculative Decoding are not supported on Neuron instances.",
            lambda: _validate_optimization_configuration(
                is_jumpstart=False,
                instance_type="ml.inf2.xlarge",
                quantization_config=None,
                speculative_decoding_config={"key": "value"},
                compilation_config=None,
                sharding_config=None,
            ),
        )

        self.assertRaisesRegex(
            ValueError,
            "Optimizations that use Sharding are not supported on Neuron instances.",
            lambda: _validate_optimization_configuration(
                is_jumpstart=False,
                instance_type="ml.inf2.xlarge",
                quantization_config=None,
                speculative_decoding_config=None,
                compilation_config=None,
                sharding_config={"key": "value"},
            ),
        )

    def test_trt_configurations_rule_set(self):
        # Can be compiled with quantization
        _validate_optimization_configuration(
            is_jumpstart=False,
            instance_type="ml.g5.24xlarge",
            quantization_config={
                "OverrideEnvironment": {"OPTION_QUANTIZE": "smoothquant"},
            },
            sharding_config=None,
            speculative_decoding_config=None,
            compilation_config={"key": "value"},
        ),

        # Can be just compiled
        _validate_optimization_configuration(
            is_jumpstart=False,
            instance_type="ml.g5.24xlarge",
            quantization_config=None,
            sharding_config=None,
            speculative_decoding_config=None,
            compilation_config={"key": "value"},
        )

        # Can be just compiled with empty dict
        _validate_optimization_configuration(
            is_jumpstart=False,
            instance_type="ml.g5.24xlarge",
            quantization_config=None,
            sharding_config=None,
            speculative_decoding_config=None,
            compilation_config={},
        )

    def test_vllm_configurations_rule_set(self):
        # Can use speculative decoding
        _validate_optimization_configuration(
            is_jumpstart=False,
            instance_type="ml.g5.24xlarge",
            quantization_config=None,
            sharding_config=None,
            speculative_decoding_config={"key": "value"},
            compilation_config=None,
        )

        # Can be quantized
        _validate_optimization_configuration(
            is_jumpstart=False,
            instance_type="ml.g5.24xlarge",
            quantization_config={
                "OverrideEnvironment": {"OPTION_QUANTIZE": "awq"},
            },
            sharding_config=None,
            speculative_decoding_config=None,
            compilation_config=None,
        )

        # Can be sharded
        _validate_optimization_configuration(
            is_jumpstart=False,
            instance_type="ml.g5.24xlarge",
            quantization_config=None,
            sharding_config={"key": "value"},
            speculative_decoding_config=None,
            compilation_config=None,
        )

    def test_neuron_configurations_rule_set(self):
        # Can be compiled
        _validate_optimization_configuration(
            is_jumpstart=False,
            instance_type="ml.inf2.xlarge",
            quantization_config=None,
            sharding_config=None,
            speculative_decoding_config=None,
            compilation_config={"key": "value"},
        )

        # Can be compiled with empty dict
        _validate_optimization_configuration(
            is_jumpstart=False,
            instance_type="ml.inf2.xlarge",
            quantization_config=None,
            sharding_config=None,
            speculative_decoding_config=None,
            compilation_config={},
        )


@pytest.mark.parametrize(
    "test_case",
    [
        # Real-time deployment without update
        {
            "input_args": {"endpoint_name": "test"},
            "call_params": {
                "instance_type": "ml.g5.2xlarge",
                "initial_instance_count": 1,
                "endpoint_name": "test",
                "update_endpoint": False,
            },
        },
        # Real-time deployment with update
        {
            "input_args": {
                "endpoint_name": "existing-endpoint",
                "update_endpoint": True,
            },
            "call_params": {
                "instance_type": "ml.g5.2xlarge",
                "initial_instance_count": 1,
                "endpoint_name": "existing-endpoint",
                "update_endpoint": True,
            },
        },
        # Serverless deployment without update
        {
            "input_args": {
                "endpoint_name": "test",
                "inference_config": ServerlessInferenceConfig(),
            },
            "call_params": {
                "serverless_inference_config": ServerlessInferenceConfig(),
                "endpoint_name": "test",
                "update_endpoint": False,
            },
        },
        # Serverless deployment with update
        {
            "input_args": {
                "endpoint_name": "existing-endpoint",
                "inference_config": ServerlessInferenceConfig(),
                "update_endpoint": True,
            },
            "call_params": {
                "serverless_inference_config": ServerlessInferenceConfig(),
                "endpoint_name": "existing-endpoint",
                "update_endpoint": True,
            },
        },
        # Async deployment without update
        {
            "input_args": {
                "endpoint_name": "test",
                "inference_config": AsyncInferenceConfig(output_path="op-path"),
            },
            "call_params": {
                "async_inference_config": AsyncInferenceConfig(output_path="op-path"),
                "instance_type": "ml.g5.2xlarge",
                "initial_instance_count": 1,
                "endpoint_name": "test",
                "update_endpoint": False,
            },
        },
        # Async deployment with update
        {
            "input_args": {
                "endpoint_name": "existing-endpoint",
                "inference_config": AsyncInferenceConfig(output_path="op-path"),
                "update_endpoint": True,
            },
            "call_params": {
                "async_inference_config": AsyncInferenceConfig(output_path="op-path"),
                "instance_type": "ml.g5.2xlarge",
                "initial_instance_count": 1,
                "endpoint_name": "existing-endpoint",
                "update_endpoint": True,
            },
        },
        # Multi-Model deployment (update_endpoint not supported)
        {
            "input_args": {
                "endpoint_name": "test",
                "inference_config": RESOURCE_REQUIREMENTS,
            },
            "call_params": {
                "resources": RESOURCE_REQUIREMENTS,
                "role": "role-arn",
                "initial_instance_count": 1,
                "instance_type": "ml.g5.2xlarge",
                "mode": Mode.SAGEMAKER_ENDPOINT,
                "endpoint_type": EndpointType.INFERENCE_COMPONENT_BASED,
                "update_endpoint": False,
            },
        },
        # Batch transform
        {
            "input_args": {
                "inference_config": BatchTransformInferenceConfig(
                    instance_count=1,
                    instance_type="ml.m5.large",
                    output_path="op-path"
                )
            },
            "call_params": {
                "instance_count": 1,
                "instance_type": "ml.m5.large",
                "output_path": "op-path",
            },
            "id": "Batch",
        },
    ],
    ids=[
        "Real Time",
        "Real Time Update",
        "Serverless",
        "Serverless Update",
        "Async",
        "Async Update",
        "Multi-Model",
        "Batch",
    ],
)
@patch("sagemaker.serve.builder.model_builder.unique_name_from_base")
def test_deploy(mock_unique_name_from_base, test_case):
    mock_unique_name_from_base.return_value = "test"
    model: Model = MagicMock()
    model_builder = ModelBuilder(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        env_vars={"HUGGING_FACE_HUB_TOKEN": "token"},
        role_arn="role-arn",
        instance_type="ml.g5.2xlarge",
    )
    setattr(model_builder, "built_model", model)

    model_builder.deploy(**test_case["input_args"])

    if "id" in test_case and test_case["id"] == "Batch":
        args, kwargs = model.transformer.call_args_list[0]
    else:
        args, kwargs = model.deploy.call_args_list[0]

    diff = deepdiff.DeepDiff(kwargs, test_case["call_params"])
    assert diff == {}


def test_deploy_multi_model_update_error():
    model_builder = ModelBuilder(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        env_vars={"HUGGING_FACE_HUB_TOKEN": "token"},
        role_arn="role-arn",
        instance_type="ml.g5.2xlarge",
    )
    setattr(model_builder, "built_model", MagicMock())

    with pytest.raises(ValueError, match="Currently update_endpoint is supported for single model endpoints"):
        model_builder.deploy(
            endpoint_name="test",
            inference_config=RESOURCE_REQUIREMENTS,
            update_endpoint=True
        )
