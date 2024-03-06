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
from unittest.mock import MagicMock, patch, Mock

import unittest
from pathlib import Path

from sagemaker.serve.builder.model_builder import ModelBuilder
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.utils import task
from sagemaker.serve.utils.exceptions import TaskNotFoundException
from sagemaker.serve.utils.types import ModelServer
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
mock_role_arn = "sample role arn"
mock_s3_model_data_url = "sample s3 data url"
mock_secret_key = "mock_secret_key"
mock_instance_type = "mock instance type"

supported_model_server = {
    ModelServer.TORCHSERVE,
    ModelServer.TRITON,
    ModelServer.DJL_SERVING,
}

MIB_CONVERSION_FACTOR = 0.00000095367431640625
MEMORY_BUFFER_MULTIPLIER = 1.2  # 20% buffer

mock_session = MagicMock()


class TestModelBuilder(unittest.TestCase):
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_validation_in_progress_mode_not_supported(self, mock_serveSettings):
        builder = ModelBuilder()
        self.assertRaisesRegex(
            Exception,
            "IN_PROCESS mode is not supported yet!",
            builder.build,
            Mode.IN_PROCESS,
            mock_role_arn,
            mock_session,
        )

    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_validation_cannot_set_both_model_and_inference_spec(self, mock_serveSettings):
        builder = ModelBuilder(inference_spec="some value", model=Mock(spec=object))
        self.assertRaisesRegex(
            Exception,
            "Cannot have both the Model and Inference spec in the builder",
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
            % (builder.model_server, supported_model_server),
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
            + "Supported model servers: %s" % supported_model_server,
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

    @patch("os.makedirs", Mock())
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
    ):
        # setup mocks
        mock_detect_container.side_effect = (
            lambda model, region, instance_type: mock_image_uri
            if model == mock_fw_model
            and region == mock_session.boto_region_name
            and instance_type == "ml.c5.xlarge"
            else None
        )

        mock_detect_fw_version.return_value = framework, version

        mock_prepare_for_torchserve.side_effect = (
            lambda model_path, shared_libs, dependencies, session, image_uri, inference_spec: mock_secret_key
            if model_path == MODEL_PATH
            and shared_libs == []
            and dependencies == {"auto": False}
            and session == session
            and image_uri == mock_image_uri
            and inference_spec is None
            else None
        )

        # Mock _ServeSettings
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_path_exists.side_effect = lambda path: True if path == "test_path" else False

        mock_mode = Mock()
        mock_sageMakerEndpointMode.side_effect = (
            lambda inference_spec, model_server: mock_mode
            if inference_spec is None and model_server == ModelServer.TORCHSERVE
            else None
        )
        mock_mode.prepare.side_effect = (
            lambda model_path, secret_key, s3_model_data_url, sagemaker_session, image_uri, jumpstart: (
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
        mock_sdk_model.side_effect = (
            lambda image_uri, image_config, vpc_config, model_data, role, env, sagemaker_session, predictor_cls: mock_model_obj  # noqa E501
            if image_uri == mock_image_uri
            and image_config == MOCK_IMAGE_CONFIG
            and vpc_config == MOCK_VPC_CONFIG
            and model_data == model_data
            and role == mock_role_arn
            and env == ENV_VARS
            and sagemaker_session == mock_session
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
    ):
        # setup mocks
        mock_detect_container.side_effect = (
            lambda model, region, instance_type: mock_1p_dlc_image_uri
            if model == mock_fw_model
            and region == mock_session.boto_region_name
            and instance_type == "ml.c5.xlarge"
            else None
        )
        mock_detect_fw_version.return_value = framework, version

        mock_prepare_for_torchserve.side_effect = (
            lambda model_path, shared_libs, dependencies, session, image_uri, inference_spec: mock_secret_key
            if model_path == MODEL_PATH
            and shared_libs == []
            and dependencies == {"auto": False}
            and session == session
            and image_uri == mock_1p_dlc_image_uri
            and inference_spec is None
            else None
        )

        # Mock _ServeSettings
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_path_exists.side_effect = lambda path: True if path == "test_path" else False

        mock_mode = Mock()
        mock_sageMakerEndpointMode.side_effect = (
            lambda inference_spec, model_server: mock_mode
            if inference_spec is None and model_server == ModelServer.TORCHSERVE
            else None
        )
        mock_mode.prepare.side_effect = (
            lambda model_path, secret_key, s3_model_data_url, sagemaker_session, image_uri, jumpstart: (
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
        mock_sdk_model.side_effect = (
            lambda image_uri, image_config, vpc_config, model_data, role, env, sagemaker_session, predictor_cls: mock_model_obj  # noqa E501
            if image_uri == mock_1p_dlc_image_uri
            and model_data == model_data
            and role == mock_role_arn
            and env == ENV_VARS
            and sagemaker_session == mock_session
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
        mock_inference_spec.load = (
            lambda model_path: mock_native_model if model_path == MODEL_PATH else None
        )

        mock_detect_fw_version.return_value = framework, version

        mock_detect_container.side_effect = (
            lambda model, region, instance_type: mock_image_uri
            if model == mock_native_model
            and region == mock_session.boto_region_name
            and instance_type == "ml.c5.xlarge"
            else None
        )

        mock_prepare_for_torchserve.side_effect = (
            lambda model_path, shared_libs, dependencies, session, image_uri, inference_spec: mock_secret_key
            if model_path == MODEL_PATH
            and shared_libs == []
            and dependencies == {"auto": False}
            and session == mock_session
            and image_uri == mock_image_uri
            and inference_spec == mock_inference_spec
            else None
        )

        # Mock _ServeSettings
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_path_exists.side_effect = lambda path: True if path == "test_path" else False

        mock_mode = Mock()
        mock_sageMakerEndpointMode.side_effect = (
            lambda inference_spec, model_server: mock_mode
            if inference_spec == mock_inference_spec and model_server == ModelServer.TORCHSERVE
            else None
        )
        mock_mode.prepare.side_effect = (
            lambda model_path, secret_key, s3_model_data_url, sagemaker_session, image_uri, jumpstart: (
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
        mock_sdk_model.side_effect = (
            lambda image_uri, image_config, vpc_config, model_data, role, env, sagemaker_session, predictor_cls: mock_model_obj  # noqa E501
            if image_uri == mock_image_uri
            and model_data == model_data
            and role == mock_role_arn
            and env == ENV_VARS_INF_SPEC
            and sagemaker_session == mock_session
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
    ):
        # setup mocks
        mock_detect_container.side_effect = (
            lambda model, region, instance_type: mock_image_uri
            if model == mock_fw_model
            and region == mock_session.boto_region_name
            and instance_type == "ml.c5.xlarge"
            else None
        )

        mock_detect_fw_version.return_value = framework, version

        mock_prepare_for_torchserve.side_effect = (
            lambda model_path, shared_libs, dependencies, session, image_uri, inference_spec: mock_secret_key
            if model_path == MODEL_PATH
            and shared_libs == []
            and dependencies == {"auto": False}
            and session == session
            and image_uri == mock_image_uri
            and inference_spec is None
            else None
        )

        # Mock _ServeSettings
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_path_exists.side_effect = lambda path: True if path == "test_path" else False

        mock_mode = Mock()
        mock_sageMakerEndpointMode.side_effect = (
            lambda inference_spec, model_server: mock_mode
            if inference_spec is None and model_server == ModelServer.TORCHSERVE
            else None
        )
        mock_mode.prepare.side_effect = (
            lambda model_path, secret_key, s3_model_data_url, sagemaker_session, image_uri, jumpstart: (
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
        mock_sdk_model.side_effect = (
            lambda image_uri, image_config, vpc_config, model_data, role, env, sagemaker_session, predictor_cls: mock_model_obj  # noqa E501
            if image_uri == mock_image_uri
            and model_data == model_data
            and role == mock_role_arn
            and env == ENV_VARS
            and sagemaker_session == mock_session
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
    ):
        # setup mocks
        mock_detect_container.side_effect = (
            lambda model, region, instance_type: mock_image_uri
            if model == mock_fw_model
            and region == mock_session.boto_region_name
            and instance_type == "ml.c5.xlarge"
            else None
        )

        mock_detect_fw_version.return_value = "xgboost", version

        mock_prepare_for_torchserve.side_effect = (
            lambda model_path, shared_libs, dependencies, session, image_uri, inference_spec: mock_secret_key
            if model_path == MODEL_PATH
            and shared_libs == []
            and dependencies == {"auto": False}
            and session == session
            and image_uri == mock_image_uri
            and inference_spec is None
            else None
        )

        # Mock _ServeSettings
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_path_exists.side_effect = lambda path: True if path == "test_path" else False

        mock_mode = Mock()
        mock_sageMakerEndpointMode.side_effect = (
            lambda inference_spec, model_server: mock_mode
            if inference_spec is None and model_server == ModelServer.TORCHSERVE
            else None
        )
        mock_mode.prepare.side_effect = (
            lambda model_path, secret_key, s3_model_data_url, sagemaker_session, image_uri, jumpstart: (
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
        mock_sdk_model.side_effect = (
            lambda image_uri, image_config, vpc_config, model_data, role, env, sagemaker_session, predictor_cls: mock_model_obj  # noqa E501
            if image_uri == mock_image_uri
            and model_data == model_data
            and role == mock_role_arn
            and env == ENV_VARS
            and sagemaker_session == mock_session
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
        mock_inference_spec.load = (
            lambda model_path: mock_native_model if model_path == MODEL_PATH else None
        )

        mock_detect_container.side_effect = (
            lambda model, region, instance_type: mock_image_uri
            if model == mock_native_model
            and region == mock_session.boto_region_name
            and instance_type == "ml.c5.xlarge"
            else None
        )

        mock_prepare_for_torchserve.side_effect = (
            lambda model_path, shared_libs, dependencies, session, image_uri, inference_spec: mock_secret_key
            if model_path == MODEL_PATH
            and shared_libs == []
            and dependencies == {"auto": False}
            and session == mock_session
            and image_uri == mock_image_uri
            and inference_spec == mock_inference_spec
            else None
        )

        # Mock _ServeSettings
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_path_exists.side_effect = lambda path: True if path == "test_path" else False

        mock_mode = Mock()
        mock_localContainerMode.side_effect = (
            lambda inference_spec, schema_builder, session, model_path, env_vars, model_server: mock_mode
            if inference_spec == mock_inference_spec
            and schema_builder == schema_builder
            and model_server == ModelServer.TORCHSERVE
            and session == mock_session
            and model_path == MODEL_PATH
            and env_vars == {}
            and model_server == ModelServer.TORCHSERVE
            else None
        )
        mock_mode.prepare.side_effect = lambda: None

        mock_model_obj = Mock()
        mock_sdk_model.side_effect = (
            lambda image_uri, image_config, vpc_config, model_data, role, env, sagemaker_session, predictor_cls: mock_model_obj  # noqa E501
            if image_uri == mock_image_uri
            and model_data is None
            and role == mock_role_arn
            and env == {}
            and sagemaker_session == mock_session
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
        mock_inference_spec.load = (
            lambda model_path: mock_native_model if model_path == MODEL_PATH else None
        )

        mock_detect_fw_version.return_value = framework, version

        mock_detect_container.side_effect = (
            lambda model, region, instance_type: mock_image_uri
            if model == mock_native_model
            and region == mock_session.boto_region_name
            and instance_type == "ml.c5.xlarge"
            else None
        )

        mock_prepare_for_torchserve.side_effect = (
            lambda model_path, shared_libs, dependencies, session, image_uri, inference_spec: mock_secret_key
            if model_path == MODEL_PATH
            and shared_libs == []
            and dependencies == {"auto": False}
            and session == mock_session
            and image_uri == mock_image_uri
            and inference_spec == mock_inference_spec
            else None
        )

        # Mock _ServeSettings
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_path_exists.side_effect = lambda path: True if path == "test_path" else False

        mock_lc_mode = Mock()
        mock_localContainerMode.side_effect = (
            lambda inference_spec, schema_builder, session, model_path, env_vars, model_server: mock_lc_mode
            if inference_spec == mock_inference_spec
            and schema_builder == schema_builder
            and model_server == ModelServer.TORCHSERVE
            and session == mock_session
            and model_path == MODEL_PATH
            and env_vars == {}
            and model_server == ModelServer.TORCHSERVE
            else None
        )
        mock_lc_mode.prepare.side_effect = lambda: None

        mock_sagemaker_endpoint_mode = Mock()
        mock_sageMakerEndpointMode.side_effect = (
            lambda inference_spec, model_server: mock_sagemaker_endpoint_mode
            if inference_spec == mock_inference_spec and model_server == ModelServer.TORCHSERVE
            else None
        )
        mock_sagemaker_endpoint_mode.prepare.side_effect = (
            lambda model_path, secret_key, s3_model_data_url, sagemaker_session, image_uri, jumpstart: (
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
        mock_sdk_model.side_effect = (
            lambda image_uri, image_config, vpc_config, model_data, role, env, sagemaker_session, predictor_cls: mock_model_obj  # noqa E501
            if image_uri == mock_image_uri
            and model_data is None
            and role == mock_role_arn
            and env == {}
            and sagemaker_session == mock_session
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
        builder._original_deploy.side_effect = (
            lambda *args, **kwargs: mock_predictor
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
    ):
        # setup mocks
        mock_detect_fw_version.return_value = framework, version

        mock_detect_container.side_effect = (
            lambda model, region, instance_type: mock_image_uri
            if model == mock_fw_model
            and region == mock_session.boto_region_name
            and instance_type == "ml.c5.xlarge"
            else None
        )

        mock_prepare_for_torchserve.side_effect = (
            lambda model_path, shared_libs, dependencies, image_uri, session, inference_spec: mock_secret_key
            if model_path == MODEL_PATH
            and shared_libs == []
            and dependencies == {"auto": False}
            and session == mock_session
            and inference_spec is None
            and image_uri == mock_image_uri
            else None
        )

        # Mock _ServeSettings
        mock_setting_object = mock_serveSettings.return_value
        mock_setting_object.role_arn = mock_role_arn
        mock_setting_object.s3_model_data_url = mock_s3_model_data_url

        mock_path_exists.side_effect = lambda path: True if path == "test_path" else False

        mock_mode = Mock()
        mock_sageMakerEndpointMode.side_effect = (
            lambda inference_spec, model_server: mock_mode
            if inference_spec is None and model_server == ModelServer.TORCHSERVE
            else None
        )
        mock_mode.prepare.side_effect = (
            lambda model_path, secret_key, s3_model_data_url, sagemaker_session, image_uri, jumpstart: (
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
            lambda inference_spec, schema_builder, session, model_path, env_vars, model_server: mock_lc_mode
            if inference_spec is None
            and schema_builder == schema_builder
            and model_server == ModelServer.TORCHSERVE
            and session == mock_session
            and model_path == MODEL_PATH
            and env_vars == ENV_VARS
            else None
        )
        mock_lc_mode.prepare.side_effect = lambda: None
        mock_lc_mode.create_server.side_effect = (
            lambda image_uri, container_timeout_seconds, secret_key, predictor: None
            if image_uri == mock_image_uri
            and secret_key == mock_secret_key
            and container_timeout_seconds == 60
            else None
        )

        mock_model_obj = Mock()
        mock_sdk_model.side_effect = (
            lambda image_uri, image_config, vpc_config, model_data, role, env, sagemaker_session, predictor_cls: mock_model_obj  # noqa E501
            if image_uri == mock_image_uri
            and model_data == model_data
            and role == mock_role_arn
            and env == ENV_VARS
            and sagemaker_session == mock_session
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
    @patch("sagemaker.djl_inference.model.urllib")
    @patch("sagemaker.djl_inference.model.json")
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
    @patch("sagemaker.djl_inference.model.urllib")
    @patch("sagemaker.djl_inference.model.json")
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
        mock_llm_utils_json.load.return_value = {"pipeline_tag": "text-to-image"}
        mock_llm_utils_urllib.request.Request.side_effect = Mock()

        # HF Model config
        mock_model_json.load.return_value = {"some": "config"}
        mock_model_urllib.request.Request.side_effect = Mock()

        mock_image_uris_retrieve.return_value = "https://some-image-uri"

        model_builder = ModelBuilder(model="CompVis/stable-diffusion-v1-4")

        self.assertRaisesRegexp(
            TaskNotFoundException,
            "Error Message: Schema builder for text-to-image could not be found.",
            lambda: model_builder.build(sagemaker_session=mock_session),
        )

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_transformers", Mock())
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._can_fit_on_single_gpu")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.djl_inference.model.urllib")
    @patch("sagemaker.djl_inference.model.json")
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

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_djl")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._can_fit_on_single_gpu")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.djl_inference.model.urllib")
    @patch("sagemaker.djl_inference.model.json")
    @patch("sagemaker.huggingface.llm_utils.urllib")
    @patch("sagemaker.huggingface.llm_utils.json")
    @patch("sagemaker.model_uris.retrieve")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_build_is_deepspeed_model(
        self,
        mock_serveSettings,
        mock_model_uris_retrieve,
        mock_llm_utils_json,
        mock_llm_utils_urllib,
        mock_model_json,
        mock_model_urllib,
        mock_image_uris_retrieve,
        mock_can_fit_on_single_gpu,
        mock_build_for_djl,
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
        mock_can_fit_on_single_gpu.return_value = False

        model_builder = ModelBuilder(model="stable-diffusion")
        model_builder.build(sagemaker_session=mock_session)

        mock_build_for_djl.assert_called_once()

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_transformers")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._can_fit_on_single_gpu")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.djl_inference.model.urllib")
    @patch("sagemaker.djl_inference.model.json")
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
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._total_inference_model_size_mib")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.djl_inference.model.urllib")
    @patch("sagemaker.djl_inference.model.json")
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

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_djl", Mock())
    @patch("sagemaker.serve.builder.model_builder._get_gpu_info")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._total_inference_model_size_mib")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.djl_inference.model.urllib")
    @patch("sagemaker.djl_inference.model.json")
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
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._total_inference_model_size_mib")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.djl_inference.model.urllib")
    @patch("sagemaker.djl_inference.model.json")
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

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_transformers", Mock())
    @patch("sagemaker.serve.builder.model_builder.estimate_command_parser")
    @patch("sagemaker.serve.builder.model_builder.gather_data")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.djl_inference.model.urllib")
    @patch("sagemaker.djl_inference.model.json")
    @patch("sagemaker.huggingface.llm_utils.urllib")
    @patch("sagemaker.huggingface.llm_utils.json")
    @patch("sagemaker.model_uris.retrieve")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_build_for_transformers_happy_case_hugging_face_responses(
        self,
        mock_serveSettings,
        mock_model_uris_retrieve,
        mock_llm_utils_json,
        mock_llm_utils_urllib,
        mock_model_json,
        mock_model_urllib,
        mock_image_uris_retrieve,
        mock_gather_data,
        mock_parser,
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

        mock_parser.return_value = Mock()
        mock_gather_data.return_value = [[1, 1, 1, 1]]
        product = MIB_CONVERSION_FACTOR * 1 * MEMORY_BUFFER_MULTIPLIER

        model_builder = ModelBuilder(
            model="stable-diffusion",
            sagemaker_session=mock_session,
            instance_type=mock_instance_type,
        )
        self.assertEqual(model_builder._total_inference_model_size_mib(), product)

        mock_parser.return_value = Mock()
        mock_gather_data.return_value = None
        model_builder = ModelBuilder(
            model="stable-diffusion",
            sagemaker_session=mock_session,
            instance_type=mock_instance_type,
        )
        with self.assertRaises(ValueError) as _:
            model_builder._total_inference_model_size_mib()

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_djl")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._can_fit_on_single_gpu")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.djl_inference.model.urllib")
    @patch("sagemaker.djl_inference.model.json")
    @patch("sagemaker.huggingface.llm_utils.urllib")
    @patch("sagemaker.huggingface.llm_utils.json")
    @patch("sagemaker.model_uris.retrieve")
    @patch("sagemaker.serve.builder.model_builder._ServeSettings")
    def test_build_is_fast_transformers_model(
        self,
        mock_serveSettings,
        mock_model_uris_retrieve,
        mock_llm_utils_json,
        mock_llm_utils_urllib,
        mock_model_json,
        mock_model_urllib,
        mock_image_uris_retrieve,
        mock_can_fit_on_single_gpu,
        mock_build_for_djl,
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
        mock_can_fit_on_single_gpu.return_value = False

        model_builder = ModelBuilder(model="gpt_neo")
        model_builder.build(sagemaker_session=mock_session)

        mock_build_for_djl.assert_called_once()

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_transformers")
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._can_fit_on_single_gpu")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.djl_inference.model.urllib")
    @patch("sagemaker.djl_inference.model.json")
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
    @patch("sagemaker.djl_inference.model.urllib")
    @patch("sagemaker.djl_inference.model.json")
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

    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._build_for_transformers", Mock())
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._try_fetch_gpu_info")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.djl_inference.model.urllib")
    @patch("sagemaker.djl_inference.model.json")
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
    @patch("sagemaker.serve.builder.model_builder.ModelBuilder._total_inference_model_size_mib")
    @patch("sagemaker.image_uris.retrieve")
    @patch("sagemaker.djl_inference.model.urllib")
    @patch("sagemaker.djl_inference.model.json")
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
