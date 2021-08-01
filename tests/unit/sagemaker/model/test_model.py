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

import pytest
from mock import Mock, patch

import sagemaker
from sagemaker.model import Model

MODEL_DATA = "s3://bucket/model.tar.gz"
MODEL_IMAGE = "mi"
TIMESTAMP = "2017-10-10-14-14-15"
MODEL_NAME = "{}-{}".format(MODEL_IMAGE, TIMESTAMP)

INSTANCE_COUNT = 2
INSTANCE_TYPE = "ml.c4.4xlarge"
ROLE = "some-role"


@pytest.fixture
def sagemaker_session():
    return Mock()


def test_prepare_container_def_with_model_data():
    model = Model(MODEL_IMAGE)
    container_def = model.prepare_container_def(INSTANCE_TYPE, "ml.eia.medium")

    expected = {"Image": MODEL_IMAGE, "Environment": {}}
    assert expected == container_def


def test_prepare_container_def_with_model_data_and_env():
    env = {"FOO": "BAR"}
    model = Model(MODEL_IMAGE, MODEL_DATA, env=env)

    expected = {"Image": MODEL_IMAGE, "Environment": env, "ModelDataUrl": MODEL_DATA}

    container_def = model.prepare_container_def(INSTANCE_TYPE, "ml.eia.medium")
    assert expected == container_def

    container_def = model.prepare_container_def()
    assert expected == container_def


def test_prepare_container_def_with_image_config():
    image_config = {"RepositoryAccessMode": "Vpc"}
    model = Model(MODEL_IMAGE, image_config=image_config)

    expected = {
        "Image": MODEL_IMAGE,
        "ImageConfig": {"RepositoryAccessMode": "Vpc"},
        "Environment": {},
    }

    container_def = model.prepare_container_def()
    assert expected == container_def


def test_model_enable_network_isolation():
    model = Model(MODEL_IMAGE, MODEL_DATA)
    assert model.enable_network_isolation() is False

    model = Model(MODEL_IMAGE, MODEL_DATA, enable_network_isolation=True)
    assert model.enable_network_isolation()


@patch("sagemaker.model.Model.prepare_container_def")
def test_create_sagemaker_model(prepare_container_def, sagemaker_session):
    container_def = {"Image": MODEL_IMAGE, "Environment": {}, "ModelDataUrl": MODEL_DATA}
    prepare_container_def.return_value = container_def

    model = Model(MODEL_DATA, MODEL_IMAGE, name=MODEL_NAME, sagemaker_session=sagemaker_session)
    model._create_sagemaker_model()

    prepare_container_def.assert_called_with(None, accelerator_type=None)
    sagemaker_session.create_model.assert_called_with(
        MODEL_NAME, None, container_def, vpc_config=None, enable_network_isolation=False, tags=None
    )


@patch("sagemaker.model.Model.prepare_container_def")
def test_create_sagemaker_model_instance_type(prepare_container_def, sagemaker_session):
    model = Model(MODEL_DATA, MODEL_IMAGE, name=MODEL_NAME, sagemaker_session=sagemaker_session)
    model._create_sagemaker_model(INSTANCE_TYPE)

    prepare_container_def.assert_called_with(INSTANCE_TYPE, accelerator_type=None)


@patch("sagemaker.model.Model.prepare_container_def")
def test_create_sagemaker_model_accelerator_type(prepare_container_def, sagemaker_session):
    model = Model(MODEL_IMAGE, MODEL_DATA, name=MODEL_NAME, sagemaker_session=sagemaker_session)

    accelerator_type = "ml.eia.medium"
    model._create_sagemaker_model(INSTANCE_TYPE, accelerator_type=accelerator_type)

    prepare_container_def.assert_called_with(INSTANCE_TYPE, accelerator_type=accelerator_type)


@patch("sagemaker.model.Model.prepare_container_def")
def test_create_sagemaker_model_tags(prepare_container_def, sagemaker_session):
    container_def = {"Image": MODEL_IMAGE, "Environment": {}, "ModelDataUrl": MODEL_DATA}
    prepare_container_def.return_value = container_def

    model = Model(MODEL_IMAGE, MODEL_DATA, name=MODEL_NAME, sagemaker_session=sagemaker_session)

    tags = {"Key": "foo", "Value": "bar"}
    model._create_sagemaker_model(INSTANCE_TYPE, tags=tags)

    sagemaker_session.create_model.assert_called_with(
        MODEL_NAME, None, container_def, vpc_config=None, enable_network_isolation=False, tags=tags
    )


@patch("sagemaker.model.Model.prepare_container_def")
@patch("sagemaker.utils.name_from_base")
@patch("sagemaker.utils.base_name_from_image")
def test_create_sagemaker_model_optional_model_params(
    base_name_from_image, name_from_base, prepare_container_def, sagemaker_session
):
    container_def = {"Image": MODEL_IMAGE, "Environment": {}, "ModelDataUrl": MODEL_DATA}
    prepare_container_def.return_value = container_def

    vpc_config = {"Subnets": ["123"], "SecurityGroupIds": ["456", "789"]}

    model = Model(
        MODEL_IMAGE,
        MODEL_DATA,
        name=MODEL_NAME,
        role=ROLE,
        vpc_config=vpc_config,
        enable_network_isolation=True,
        sagemaker_session=sagemaker_session,
    )
    model._create_sagemaker_model(INSTANCE_TYPE)

    base_name_from_image.assert_not_called()
    name_from_base.assert_not_called()

    sagemaker_session.create_model.assert_called_with(
        MODEL_NAME,
        ROLE,
        container_def,
        vpc_config=vpc_config,
        enable_network_isolation=True,
        tags=None,
    )


@patch("sagemaker.model.Model.prepare_container_def")
@patch("sagemaker.utils.name_from_base", return_value=MODEL_NAME)
@patch("sagemaker.utils.base_name_from_image")
def test_create_sagemaker_model_generates_model_name(
    base_name_from_image, name_from_base, prepare_container_def, sagemaker_session
):
    container_def = {"Image": MODEL_IMAGE, "Environment": {}, "ModelDataUrl": MODEL_DATA}
    prepare_container_def.return_value = container_def

    model = Model(
        MODEL_IMAGE,
        MODEL_DATA,
        sagemaker_session=sagemaker_session,
    )
    model._create_sagemaker_model(INSTANCE_TYPE)

    base_name_from_image.assert_called_with(MODEL_IMAGE)
    name_from_base.assert_called_with(base_name_from_image.return_value)

    sagemaker_session.create_model.assert_called_with(
        MODEL_NAME,
        None,
        container_def,
        vpc_config=None,
        enable_network_isolation=False,
        tags=None,
    )


@patch("sagemaker.model.Model.prepare_container_def")
@patch("sagemaker.utils.name_from_base", return_value=MODEL_NAME)
@patch("sagemaker.utils.base_name_from_image")
def test_create_sagemaker_model_generates_model_name_each_time(
    base_name_from_image, name_from_base, prepare_container_def, sagemaker_session
):
    container_def = {"Image": MODEL_IMAGE, "Environment": {}, "ModelDataUrl": MODEL_DATA}
    prepare_container_def.return_value = container_def

    model = Model(
        MODEL_IMAGE,
        MODEL_DATA,
        sagemaker_session=sagemaker_session,
    )
    model._create_sagemaker_model(INSTANCE_TYPE)
    model._create_sagemaker_model(INSTANCE_TYPE)

    base_name_from_image.assert_called_once_with(MODEL_IMAGE)
    name_from_base.assert_called_with(base_name_from_image.return_value)
    assert 2 == name_from_base.call_count


@patch("sagemaker.session.Session")
@patch("sagemaker.local.LocalSession")
def test_create_sagemaker_model_creates_correct_session(local_session, session):
    model = Model(MODEL_IMAGE, MODEL_DATA)
    model._create_sagemaker_model("local")
    assert model.sagemaker_session == local_session.return_value

    model = Model(MODEL_IMAGE, MODEL_DATA)
    model._create_sagemaker_model("ml.m5.xlarge")
    assert model.sagemaker_session == session.return_value


@patch("sagemaker.model.Model._create_sagemaker_model")
def test_model_create_transformer(create_sagemaker_model, sagemaker_session):
    model_name = "auto-generated-model"
    model = Model(MODEL_IMAGE, MODEL_DATA, name=model_name, sagemaker_session=sagemaker_session)

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
    model = Model(MODEL_IMAGE, MODEL_DATA, sagemaker_session=sagemaker_session)

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


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
def test_model_create_transformer_network_isolation(sagemaker_session):
    model = Model(
        MODEL_IMAGE, MODEL_DATA, sagemaker_session=sagemaker_session, enable_network_isolation=True
    )

    transformer = model.transformer(1, "ml.m4.xlarge", env={"should_be": "overwritten"})
    assert transformer.env is None


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
def test_model_create_transformer_base_name(sagemaker_session):
    model = Model(MODEL_IMAGE, MODEL_DATA, sagemaker_session=sagemaker_session)

    base_name = "foo"
    model._base_name = base_name

    transformer = model.transformer(1, "ml.m4.xlarge")
    assert base_name == transformer.base_transform_job_name


@patch("sagemaker.session.Session")
@patch("sagemaker.local.LocalSession")
def test_transformer_creates_correct_session(local_session, session):
    model = Model(MODEL_IMAGE, MODEL_DATA, sagemaker_session=None)
    transformer = model.transformer(instance_count=1, instance_type="local")
    assert model.sagemaker_session == local_session.return_value
    assert transformer.sagemaker_session == local_session.return_value

    model = Model(MODEL_IMAGE, MODEL_DATA, sagemaker_session=None)
    transformer = model.transformer(instance_count=1, instance_type="ml.m5.xlarge")
    assert model.sagemaker_session == session.return_value
    assert transformer.sagemaker_session == session.return_value


def test_delete_model(sagemaker_session):
    model = Model(MODEL_IMAGE, MODEL_DATA, name=MODEL_NAME, sagemaker_session=sagemaker_session)

    model.delete_model()
    sagemaker_session.delete_model.assert_called_with(model.name)


def test_delete_model_no_name(sagemaker_session):
    model = Model(MODEL_IMAGE, MODEL_DATA, sagemaker_session=sagemaker_session)

    with pytest.raises(
        ValueError, match="The SageMaker model must be created first before attempting to delete."
    ):
        model.delete_model()
    sagemaker_session.delete_model.assert_not_called()
