# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

MODEL_DATA = "s3://bucket/model.tar.gz"
MODEL_IMAGE = "mi"
TIMESTAMP = "2017-10-10-14-14-15"
MODEL_NAME = "{}-{}".format(MODEL_IMAGE, TIMESTAMP)

INSTANCE_COUNT = 2
INSTANCE_TYPE = "c4.4xlarge"
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
@patch("sagemaker.utils.name_from_image")
def test_deploy(name_from_image, prepare_container_def, production_variant, sagemaker_session):
    name_from_image.return_value = MODEL_NAME

    container_def = {"Image": MODEL_IMAGE, "Environment": {}, "ModelDataUrl": MODEL_DATA}
    prepare_container_def.return_value = container_def

    production_variant.return_value = BASE_PRODUCTION_VARIANT

    model = Model(MODEL_DATA, MODEL_IMAGE, role=ROLE, sagemaker_session=sagemaker_session)
    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=INSTANCE_COUNT)

    name_from_image.assert_called_with(MODEL_IMAGE)
    prepare_container_def.assert_called_with(INSTANCE_TYPE, accelerator_type=None)
    production_variant.assert_called_with(
        MODEL_NAME, INSTANCE_TYPE, INSTANCE_COUNT, accelerator_type=None
    )

    sagemaker_session.create_model.assert_called_with(
        MODEL_NAME, ROLE, container_def, vpc_config=None, enable_network_isolation=False, tags=None
    )

    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=MODEL_NAME,
        production_variants=[BASE_PRODUCTION_VARIANT],
        tags=None,
        kms_key=None,
        wait=True,
        data_capture_config_dict=None,
    )


@patch("sagemaker.model.Model._create_sagemaker_model")
@patch("sagemaker.production_variant")
def test_deploy_accelerator_type(production_variant, create_sagemaker_model, sagemaker_session):
    model = Model(
        MODEL_DATA, MODEL_IMAGE, role=ROLE, name=MODEL_NAME, sagemaker_session=sagemaker_session
    )

    accelerator_type = "ml.eia.medium"

    production_variant_result = copy.deepcopy(BASE_PRODUCTION_VARIANT)
    production_variant_result["AcceleratorType"] = accelerator_type
    production_variant.return_value = production_variant_result

    model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
        accelerator_type=accelerator_type,
    )

    create_sagemaker_model.assert_called_with(INSTANCE_TYPE, accelerator_type, None)
    production_variant.assert_called_with(
        MODEL_NAME, INSTANCE_TYPE, INSTANCE_COUNT, accelerator_type=accelerator_type
    )

    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=MODEL_NAME,
        production_variants=[production_variant_result],
        tags=None,
        kms_key=None,
        wait=True,
        data_capture_config_dict=None,
    )


@patch("sagemaker.utils.name_from_image", Mock())
@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
def test_deploy_endpoint_name(sagemaker_session):
    model = Model(MODEL_DATA, MODEL_IMAGE, role=ROLE, sagemaker_session=sagemaker_session)

    endpoint_name = "blah"
    model.deploy(
        endpoint_name=endpoint_name,
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
    )

    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=endpoint_name,
        production_variants=[BASE_PRODUCTION_VARIANT],
        tags=None,
        kms_key=None,
        wait=True,
        data_capture_config_dict=None,
    )


@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
@patch("sagemaker.model.Model._create_sagemaker_model")
def test_deploy_tags(create_sagemaker_model, production_variant, sagemaker_session):
    model = Model(
        MODEL_DATA, MODEL_IMAGE, role=ROLE, name=MODEL_NAME, sagemaker_session=sagemaker_session
    )

    tags = [{"Key": "ModelName", "Value": "TestModel"}]
    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=INSTANCE_COUNT, tags=tags)

    create_sagemaker_model.assert_called_with(INSTANCE_TYPE, None, tags)
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=MODEL_NAME,
        production_variants=[BASE_PRODUCTION_VARIANT],
        tags=tags,
        kms_key=None,
        wait=True,
        data_capture_config_dict=None,
    )


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
def test_deploy_kms_key(production_variant, sagemaker_session):
    model = Model(
        MODEL_DATA, MODEL_IMAGE, role=ROLE, name=MODEL_NAME, sagemaker_session=sagemaker_session
    )

    key = "some-key-arn"
    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=INSTANCE_COUNT, kms_key=key)

    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=MODEL_NAME,
        production_variants=[BASE_PRODUCTION_VARIANT],
        tags=None,
        kms_key=key,
        wait=True,
        data_capture_config_dict=None,
    )


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
def test_deploy_async(production_variant, sagemaker_session):
    model = Model(
        MODEL_DATA, MODEL_IMAGE, role=ROLE, name=MODEL_NAME, sagemaker_session=sagemaker_session
    )

    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=INSTANCE_COUNT, wait=False)

    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=MODEL_NAME,
        production_variants=[BASE_PRODUCTION_VARIANT],
        tags=None,
        kms_key=None,
        wait=False,
        data_capture_config_dict=None,
    )


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
def test_deploy_data_capture_config(production_variant, sagemaker_session):
    model = Model(
        MODEL_DATA, MODEL_IMAGE, role=ROLE, name=MODEL_NAME, sagemaker_session=sagemaker_session
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
        name=MODEL_NAME,
        production_variants=[BASE_PRODUCTION_VARIANT],
        tags=None,
        kms_key=None,
        wait=True,
        data_capture_config_dict=data_capture_config_dict,
    )


@patch("sagemaker.session.Session")
@patch("sagemaker.local.LocalSession")
def test_deploy_creates_correct_session(local_session, session, tmpdir):
    # We expect a LocalSession when deploying to instance_type = 'local'
    model = Model(MODEL_DATA, MODEL_IMAGE, role=ROLE)
    model.deploy(endpoint_name="blah", instance_type="local", initial_instance_count=1)
    assert model.sagemaker_session == local_session.return_value

    # We expect a real Session when deploying to instance_type != local/local_gpu
    model = Model(MODEL_DATA, MODEL_IMAGE, role=ROLE)
    model.deploy(
        endpoint_name="remote_endpoint", instance_type="ml.m4.4xlarge", initial_instance_count=2
    )
    assert model.sagemaker_session == session.return_value


def test_deploy_no_role(sagemaker_session):
    model = Model(MODEL_DATA, MODEL_IMAGE, sagemaker_session=sagemaker_session)

    with pytest.raises(ValueError, match="Role can not be null for deploying a model"):
        model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=INSTANCE_COUNT)


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.predictor.RealTimePredictor._get_endpoint_config_name", Mock())
@patch("sagemaker.predictor.RealTimePredictor._get_model_names", Mock())
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
def test_deploy_predictor_cls(production_variant, sagemaker_session):
    model = Model(
        MODEL_DATA,
        MODEL_IMAGE,
        role=ROLE,
        name=MODEL_NAME,
        predictor_cls=sagemaker.predictor.RealTimePredictor,
        sagemaker_session=sagemaker_session,
    )

    endpoint_name = "foo"
    predictor = model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
        endpoint_name=endpoint_name,
    )

    assert isinstance(predictor, sagemaker.predictor.RealTimePredictor)
    assert predictor.endpoint == endpoint_name
    assert predictor.sagemaker_session == sagemaker_session


@patch("sagemaker.model.Model._create_sagemaker_model")
def test_model_create_transformer(create_sagemaker_model, sagemaker_session):
    model_name = "auto-generated-model"
    model = Model(MODEL_DATA, MODEL_IMAGE, name=model_name, sagemaker_session=sagemaker_session)

    instance_type = "ml.m4.xlarge"
    transformer = model.transformer(instance_count=1, instance_type=instance_type)

    create_sagemaker_model.assert_called_with(instance_type, tags=None)

    assert isinstance(transformer, sagemaker.transformer.Transformer)
    assert transformer.model_name == model_name
    assert transformer.instance_type == instance_type
    assert transformer.instance_count == 1
    assert transformer.sagemaker_session == sagemaker_session
    assert transformer.base_transform_job_name == model_name

    assert transformer.strategy is None
    assert transformer.env is None
    assert transformer.output_path is None
    assert transformer.output_kms_key is None
    assert transformer.accept is None
    assert transformer.assemble_with is None
    assert transformer.volume_kms_key is None
    assert transformer.max_concurrent_transforms is None
    assert transformer.max_payload is None
    assert transformer.tags is None


@patch("sagemaker.model.Model._create_sagemaker_model")
def test_model_create_transformer_optional_params(create_sagemaker_model, sagemaker_session):
    model = Model(MODEL_DATA, MODEL_IMAGE, sagemaker_session=sagemaker_session)

    instance_type = "ml.m4.xlarge"
    strategy = "MultiRecord"
    assemble_with = "Line"
    output_path = "s3://bucket/path"
    kms_key = "key"
    accept = "text/csv"
    env = {"test": True}
    max_concurrent_transforms = 1
    max_payload = 6
    tags = [{"Key": "k", "Value": "v"}]

    transformer = model.transformer(
        instance_count=1,
        instance_type=instance_type,
        strategy=strategy,
        assemble_with=assemble_with,
        output_path=output_path,
        output_kms_key=kms_key,
        accept=accept,
        env=env,
        max_concurrent_transforms=max_concurrent_transforms,
        max_payload=max_payload,
        tags=tags,
        volume_kms_key=kms_key,
    )

    create_sagemaker_model.assert_called_with(instance_type, tags=tags)

    assert isinstance(transformer, sagemaker.transformer.Transformer)
    assert transformer.strategy == strategy
    assert transformer.assemble_with == assemble_with
    assert transformer.output_path == output_path
    assert transformer.output_kms_key == kms_key
    assert transformer.accept == accept
    assert transformer.max_concurrent_transforms == max_concurrent_transforms
    assert transformer.max_payload == max_payload
    assert transformer.env == env
    assert transformer.tags == tags
    assert transformer.volume_kms_key == kms_key


@patch("sagemaker.model.Model._create_sagemaker_model")
def test_model_create_transformer_network_isolation(create_sagemaker_model, sagemaker_session):
    model = Model(
        MODEL_DATA, MODEL_IMAGE, sagemaker_session=sagemaker_session, enable_network_isolation=True
    )

    transformer = model.transformer(1, "ml.m4.xlarge", env={"should_be": "overwritten"})
    assert transformer.env is None


@patch("sagemaker.session.Session")
@patch("sagemaker.local.LocalSession")
@patch("sagemaker.fw_utils.tar_and_upload_dir", Mock())
def test_transformer_creates_correct_session(local_session, session):
    model = Model(MODEL_DATA, MODEL_IMAGE, sagemaker_session=None)
    transformer = model.transformer(instance_count=1, instance_type="local")
    assert model.sagemaker_session == local_session.return_value
    assert transformer.sagemaker_session == local_session.return_value

    model = Model(MODEL_DATA, MODEL_IMAGE, sagemaker_session=None)
    transformer = model.transformer(instance_count=1, instance_type="ml.m5.xlarge")
    assert model.sagemaker_session == session.return_value
    assert transformer.sagemaker_session == session.return_value
