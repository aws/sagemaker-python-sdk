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

from mock import Mock, patch

import sagemaker
from sagemaker.model import Model

MODEL_DATA = "s3://bucket/model.tar.gz"
MODEL_IMAGE = "mi"


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
