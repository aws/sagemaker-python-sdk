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

import copy

import pytest
from mock import Mock, patch

import sagemaker
from sagemaker.model import Model
from sagemaker.async_inference import AsyncInferenceConfig
from sagemaker.serverless import ServerlessInferenceConfig

MODEL_DATA = "s3://bucket/model.tar.gz"
MODEL_IMAGE = "mi"
TIMESTAMP = "2020-07-02-20-10-30-288"
MODEL_NAME = "{}-{}".format(MODEL_IMAGE, TIMESTAMP)
ENDPOINT_NAME = "endpoint-{}".format(TIMESTAMP)

ACCELERATOR_TYPE = "ml.eia.medium"
INSTANCE_COUNT = 2
INSTANCE_TYPE = "ml.c4.4xlarge"
ROLE = "some-role"

BASE_PRODUCTION_VARIANT = {
    "ModelName": MODEL_NAME,
    "InstanceType": INSTANCE_TYPE,
    "InitialInstanceCount": INSTANCE_COUNT,
    "VariantName": "AllTraffic",
    "InitialVariantWeight": 1,
}


@pytest.fixture
def sagemaker_session():
    return Mock()


@patch("sagemaker.production_variant")
@patch("sagemaker.model.Model.prepare_container_def")
@patch("sagemaker.utils.name_from_base", return_value=MODEL_NAME)
def test_deploy(name_from_base, prepare_container_def, production_variant, sagemaker_session):
    production_variant.return_value = BASE_PRODUCTION_VARIANT

    container_def = {"Image": MODEL_IMAGE, "Environment": {}, "ModelDataUrl": MODEL_DATA}
    prepare_container_def.return_value = container_def

    model = Model(MODEL_IMAGE, MODEL_DATA, role=ROLE, sagemaker_session=sagemaker_session)
    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=INSTANCE_COUNT)

    name_from_base.assert_called_with(MODEL_IMAGE)
    assert 2 == name_from_base.call_count

    prepare_container_def.assert_called_with(
        INSTANCE_TYPE, accelerator_type=None, serverless_inference_config=None
    )
    production_variant.assert_called_with(
        MODEL_NAME,
        INSTANCE_TYPE,
        INSTANCE_COUNT,
        accelerator_type=None,
        serverless_inference_config=None,
        volume_size=None,
        model_data_download_timeout=None,
        container_startup_health_check_timeout=None,
    )

    sagemaker_session.create_model.assert_called_with(
        name=MODEL_NAME,
        role=ROLE,
        container_defs=container_def,
        vpc_config=None,
        enable_network_isolation=False,
        tags=None,
    )

    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=MODEL_NAME,
        production_variants=[BASE_PRODUCTION_VARIANT],
        tags=None,
        kms_key=None,
        wait=True,
        data_capture_config_dict=None,
        async_inference_config_dict=None,
    )


@patch("sagemaker.utils.name_from_base", return_value=ENDPOINT_NAME)
@patch("sagemaker.model.Model._create_sagemaker_model")
@patch("sagemaker.production_variant")
def test_deploy_accelerator_type(
    production_variant, create_sagemaker_model, name_from_base, sagemaker_session
):
    model = Model(
        MODEL_IMAGE, MODEL_DATA, role=ROLE, name=MODEL_NAME, sagemaker_session=sagemaker_session
    )

    production_variant_result = copy.deepcopy(BASE_PRODUCTION_VARIANT)
    production_variant_result["AcceleratorType"] = ACCELERATOR_TYPE
    production_variant.return_value = production_variant_result

    model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
        accelerator_type=ACCELERATOR_TYPE,
    )

    create_sagemaker_model.assert_called_with(INSTANCE_TYPE, ACCELERATOR_TYPE, None, None)
    production_variant.assert_called_with(
        MODEL_NAME,
        INSTANCE_TYPE,
        INSTANCE_COUNT,
        accelerator_type=ACCELERATOR_TYPE,
        serverless_inference_config=None,
        volume_size=None,
        model_data_download_timeout=None,
        container_startup_health_check_timeout=None,
    )

    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=ENDPOINT_NAME,
        production_variants=[production_variant_result],
        tags=None,
        kms_key=None,
        wait=True,
        data_capture_config_dict=None,
        async_inference_config_dict=None,
    )


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
def test_deploy_endpoint_name(sagemaker_session):
    model = Model(MODEL_IMAGE, MODEL_DATA, role=ROLE, sagemaker_session=sagemaker_session)

    endpoint_name = "blah"
    model.deploy(
        endpoint_name=endpoint_name,
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
    )

    assert endpoint_name == model.endpoint_name
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=endpoint_name,
        production_variants=[BASE_PRODUCTION_VARIANT],
        tags=None,
        kms_key=None,
        wait=True,
        data_capture_config_dict=None,
        async_inference_config_dict=None,
    )


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.utils.name_from_base")
@patch("sagemaker.utils.base_from_name")
@patch("sagemaker.production_variant")
def test_deploy_generates_endpoint_name_each_time_from_model_name(
    production_variant, base_from_name, name_from_base, sagemaker_session
):
    model = Model(
        MODEL_IMAGE, MODEL_DATA, name=MODEL_NAME, role=ROLE, sagemaker_session=sagemaker_session
    )

    model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
    )
    model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
    )

    base_from_name.assert_called_with(MODEL_NAME)
    name_from_base.assert_called_with(base_from_name.return_value)
    assert 2 == name_from_base.call_count


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.utils.name_from_base")
@patch("sagemaker.utils.base_from_name")
@patch("sagemaker.production_variant")
def test_deploy_generates_endpoint_name_each_time_from_base_name(
    production_variant, base_from_name, name_from_base, sagemaker_session
):
    model = Model(MODEL_IMAGE, MODEL_DATA, role=ROLE, sagemaker_session=sagemaker_session)

    base_name = "foo"
    model._base_name = base_name

    model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
    )
    model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
    )

    base_from_name.assert_not_called()
    name_from_base.assert_called_with(base_name)
    assert 2 == name_from_base.call_count


@patch("sagemaker.utils.name_from_base", return_value=ENDPOINT_NAME)
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
@patch("sagemaker.model.Model._create_sagemaker_model")
def test_deploy_tags(create_sagemaker_model, production_variant, name_from_base, sagemaker_session):
    model = Model(
        MODEL_IMAGE, MODEL_DATA, role=ROLE, name=MODEL_NAME, sagemaker_session=sagemaker_session
    )

    tags = [{"Key": "ModelName", "Value": "TestModel"}]
    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=INSTANCE_COUNT, tags=tags)

    create_sagemaker_model.assert_called_with(INSTANCE_TYPE, None, tags, None)
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=ENDPOINT_NAME,
        production_variants=[BASE_PRODUCTION_VARIANT],
        tags=tags,
        kms_key=None,
        wait=True,
        data_capture_config_dict=None,
        async_inference_config_dict=None,
    )


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.utils.name_from_base", return_value=ENDPOINT_NAME)
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
def test_deploy_kms_key(production_variant, name_from_base, sagemaker_session):
    model = Model(
        MODEL_IMAGE, MODEL_DATA, role=ROLE, name=MODEL_NAME, sagemaker_session=sagemaker_session
    )

    key = "some-key-arn"
    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=INSTANCE_COUNT, kms_key=key)

    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=ENDPOINT_NAME,
        production_variants=[BASE_PRODUCTION_VARIANT],
        tags=None,
        kms_key=key,
        wait=True,
        data_capture_config_dict=None,
        async_inference_config_dict=None,
    )


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.utils.name_from_base", return_value=ENDPOINT_NAME)
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
def test_deploy_async(production_variant, name_from_base, sagemaker_session):
    model = Model(
        MODEL_IMAGE, MODEL_DATA, role=ROLE, name=MODEL_NAME, sagemaker_session=sagemaker_session
    )

    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=INSTANCE_COUNT, wait=False)

    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=ENDPOINT_NAME,
        production_variants=[BASE_PRODUCTION_VARIANT],
        tags=None,
        kms_key=None,
        wait=False,
        data_capture_config_dict=None,
        async_inference_config_dict=None,
    )


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.utils.name_from_base", return_value=ENDPOINT_NAME)
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
def test_deploy_data_capture_config(production_variant, name_from_base, sagemaker_session):
    model = Model(
        MODEL_IMAGE, MODEL_DATA, role=ROLE, name=MODEL_NAME, sagemaker_session=sagemaker_session
    )

    data_capture_config = Mock()
    data_capture_config_dict = {"EnableCapture": True}
    data_capture_config._to_request_dict.return_value = data_capture_config_dict
    model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
        data_capture_config=data_capture_config,
    )

    data_capture_config._to_request_dict.assert_called_with()
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=ENDPOINT_NAME,
        production_variants=[BASE_PRODUCTION_VARIANT],
        tags=None,
        kms_key=None,
        wait=True,
        data_capture_config_dict=data_capture_config_dict,
        async_inference_config_dict=None,
    )


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.utils.name_from_base", return_value=ENDPOINT_NAME)
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
def test_deploy_async_inference(production_variant, name_from_base, sagemaker_session):
    model = Model(
        MODEL_IMAGE, MODEL_DATA, role=ROLE, name=MODEL_NAME, sagemaker_session=sagemaker_session
    )

    async_inference_config = AsyncInferenceConfig(output_path="s3://some-path")
    async_inference_config_dict = {
        "OutputConfig": {
            "S3OutputPath": "s3://some-path",
        },
    }

    model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
        async_inference_config=async_inference_config,
    )

    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=ENDPOINT_NAME,
        production_variants=[BASE_PRODUCTION_VARIANT],
        tags=None,
        kms_key=None,
        wait=True,
        data_capture_config_dict=None,
        async_inference_config_dict=async_inference_config_dict,
    )


@patch("sagemaker.utils.name_from_base", return_value=ENDPOINT_NAME)
@patch("sagemaker.model.Model._create_sagemaker_model")
@patch("sagemaker.production_variant")
def test_deploy_serverless_inference(production_variant, create_sagemaker_model, sagemaker_session):
    model = Model(
        MODEL_IMAGE, MODEL_DATA, role=ROLE, name=MODEL_NAME, sagemaker_session=sagemaker_session
    )

    production_variant_result = copy.deepcopy(BASE_PRODUCTION_VARIANT)
    production_variant.return_value = production_variant_result

    serverless_inference_config = ServerlessInferenceConfig()
    serverless_inference_config_dict = {
        "MemorySizeInMB": 2048,
        "MaxConcurrency": 5,
    }

    model.deploy(
        serverless_inference_config=serverless_inference_config,
    )

    create_sagemaker_model.assert_called_with(None, None, None, serverless_inference_config)
    production_variant.assert_called_with(
        MODEL_NAME,
        None,
        None,
        accelerator_type=None,
        serverless_inference_config=serverless_inference_config_dict,
        volume_size=None,
        model_data_download_timeout=None,
        container_startup_health_check_timeout=None,
    )

    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=ENDPOINT_NAME,
        production_variants=[production_variant_result],
        tags=None,
        kms_key=None,
        wait=True,
        data_capture_config_dict=None,
        async_inference_config_dict=None,
    )


def test_deploy_wrong_inference_type(sagemaker_session):
    model = Model(MODEL_IMAGE, MODEL_DATA, role=ROLE)

    bad_args = (
        {"instance_type": INSTANCE_TYPE},
        {"initial_instance_count": INSTANCE_COUNT},
        {"instance_type": None, "initial_instance_count": None},
    )
    for args in bad_args:
        with pytest.raises(
            ValueError,
            match="Must specify instance type and instance count unless using serverless inference",
        ):
            model.deploy(args)


def test_deploy_wrong_serverless_config(sagemaker_session):
    model = Model(MODEL_IMAGE, MODEL_DATA, role=ROLE)
    with pytest.raises(
        ValueError,
        match="serverless_inference_config needs to be a ServerlessInferenceConfig object",
    ):
        model.deploy(serverless_inference_config={})


@patch("sagemaker.session.Session")
@patch("sagemaker.local.LocalSession")
def test_deploy_creates_correct_session(local_session, session):
    # We expect a LocalSession when deploying to instance_type = 'local'
    model = Model(MODEL_IMAGE, MODEL_DATA, role=ROLE)
    model.deploy(endpoint_name="blah", instance_type="local", initial_instance_count=1)
    assert model.sagemaker_session == local_session.return_value

    # We expect a real Session when deploying to instance_type != local/local_gpu
    model = Model(MODEL_IMAGE, MODEL_DATA, role=ROLE)
    model.deploy(
        endpoint_name="remote_endpoint", instance_type="ml.m4.4xlarge", initial_instance_count=2
    )
    assert model.sagemaker_session == session.return_value


def test_deploy_no_role(sagemaker_session):
    model = Model(MODEL_IMAGE, MODEL_DATA, sagemaker_session=sagemaker_session)

    with pytest.raises(ValueError, match="Role can not be null for deploying a model"):
        model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=INSTANCE_COUNT)


def test_deploy_wrong_async_inferenc_config(sagemaker_session):
    model = Model(MODEL_IMAGE, MODEL_DATA, sagemaker_session=sagemaker_session, role=ROLE)

    with pytest.raises(
        ValueError, match="async_inference_config needs to be a AsyncInferenceConfig object"
    ):
        model.deploy(
            instance_type=INSTANCE_TYPE,
            initial_instance_count=INSTANCE_COUNT,
            async_inference_config={},
        )


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.predictor.Predictor._get_endpoint_config_name", Mock())
@patch("sagemaker.predictor.Predictor._get_model_names", Mock())
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
def test_deploy_predictor_cls(production_variant, sagemaker_session):
    model = Model(
        MODEL_IMAGE,
        MODEL_DATA,
        role=ROLE,
        name=MODEL_NAME,
        predictor_cls=sagemaker.predictor.Predictor,
        sagemaker_session=sagemaker_session,
    )

    endpoint_name = "foo"
    predictor = model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
        endpoint_name=endpoint_name,
    )

    assert isinstance(predictor, sagemaker.predictor.Predictor)
    assert predictor.endpoint_name == endpoint_name
    assert predictor.sagemaker_session == sagemaker_session

    endpoint_name_async = "foo-async"
    predictor_async = model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
        endpoint_name=endpoint_name_async,
        async_inference_config=AsyncInferenceConfig(),
    )

    assert isinstance(predictor_async, sagemaker.predictor_async.AsyncPredictor)
    assert predictor_async.name == model.name
    assert predictor_async.endpoint_name == endpoint_name_async
    assert predictor_async.sagemaker_session == sagemaker_session


@patch("sagemaker.production_variant")
@patch("sagemaker.model.Model.prepare_container_def")
@patch("sagemaker.utils.name_from_base", return_value=MODEL_NAME)
def test_deploy_customized_volume_size_and_timeout(
    name_from_base, prepare_container_def, production_variant, sagemaker_session
):
    volume_size_gb = 256
    model_data_download_timeout_sec = 1800
    startup_health_check_timeout_sec = 1800

    production_variant_result = copy.deepcopy(BASE_PRODUCTION_VARIANT)
    production_variant_result.update(
        {
            "VolumeSizeInGB": volume_size_gb,
            "ModelDataDownloadTimeoutInSeconds": model_data_download_timeout_sec,
            "ContainerStartupHealthCheckTimeoutInSeconds": startup_health_check_timeout_sec,
        }
    )
    production_variant.return_value = production_variant_result

    container_def = {"Image": MODEL_IMAGE, "Environment": {}, "ModelDataUrl": MODEL_DATA}
    prepare_container_def.return_value = container_def

    model = Model(MODEL_IMAGE, MODEL_DATA, role=ROLE, sagemaker_session=sagemaker_session)
    model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
        volume_size=volume_size_gb,
        model_data_download_timeout=model_data_download_timeout_sec,
        container_startup_health_check_timeout=startup_health_check_timeout_sec,
    )

    name_from_base.assert_called_with(MODEL_IMAGE)
    assert 2 == name_from_base.call_count

    prepare_container_def.assert_called_with(
        INSTANCE_TYPE, accelerator_type=None, serverless_inference_config=None
    )
    production_variant.assert_called_with(
        MODEL_NAME,
        INSTANCE_TYPE,
        INSTANCE_COUNT,
        accelerator_type=None,
        serverless_inference_config=None,
        volume_size=volume_size_gb,
        model_data_download_timeout=model_data_download_timeout_sec,
        container_startup_health_check_timeout=startup_health_check_timeout_sec,
    )

    sagemaker_session.create_model.assert_called_with(
        name=MODEL_NAME,
        role=ROLE,
        container_defs=container_def,
        vpc_config=None,
        enable_network_isolation=False,
        tags=None,
    )

    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=MODEL_NAME,
        production_variants=[production_variant_result],
        tags=None,
        kms_key=None,
        wait=True,
        data_capture_config_dict=None,
        async_inference_config_dict=None,
    )
