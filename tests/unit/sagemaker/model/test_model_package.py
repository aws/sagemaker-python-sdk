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
from sagemaker.model import ModelPackage

DESCRIBE_MODEL_PACKAGE_RESPONSE = {
    "InferenceSpecification": {
        "SupportedResponseMIMETypes": ["text"],
        "SupportedContentTypes": ["text/csv"],
        "SupportedTransformInstanceTypes": ["ml.m4.xlarge", "ml.m4.2xlarge"],
        "Containers": [
            {
                "Image": "1.dkr.ecr.us-east-2.amazonaws.com/decision-trees-sample:latest",
                "ImageDigest": "sha256:1234556789",
                "ModelDataUrl": "s3://bucket/output/model.tar.gz",
            }
        ],
        "SupportedRealtimeInferenceInstanceTypes": ["ml.m4.xlarge", "ml.m4.2xlarge"],
    },
    "ModelPackageDescription": "Model Package created from training with "
    "arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
    "CreationTime": 1542752036.687,
    "ModelPackageArn": "arn:aws:sagemaker:us-east-2:123:model-package/mp-scikit-decision-trees",
    "ModelPackageStatusDetails": {"ValidationStatuses": [], "ImageScanStatuses": []},
    "SourceAlgorithmSpecification": {
        "SourceAlgorithms": [
            {
                "ModelDataUrl": "s3://bucket/output/model.tar.gz",
                "AlgorithmName": "arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
            }
        ]
    },
    "ModelPackageStatus": "Completed",
    "ModelPackageName": "mp-scikit-decision-trees-1542410022-2018-11-20-22-13-56-502",
    "CertifyForMarketplace": False,
}


@pytest.fixture
def sagemaker_session():
    session = Mock(
        default_bucket_prefix=None,
    )
    session.sagemaker_client.describe_model_package = Mock(
        return_value=DESCRIBE_MODEL_PACKAGE_RESPONSE
    )
    # For tests which doesn't verify config file injection, operate with empty config
    session.sagemaker_config = {}
    return session


def test_model_package_enable_network_isolation_with_no_product_id(sagemaker_session):
    model_package = ModelPackage(
        role="role", model_package_arn="my-model-package", sagemaker_session=sagemaker_session
    )
    assert model_package.enable_network_isolation() is False


def test_model_package_enable_network_isolation_with_product_id(sagemaker_session):
    model_package_response = copy.deepcopy(DESCRIBE_MODEL_PACKAGE_RESPONSE)
    model_package_response["InferenceSpecification"]["Containers"].append(
        {
            "Image": "1.dkr.ecr.us-east-2.amazonaws.com/some-container:latest",
            "ModelDataUrl": "s3://bucket/output/model.tar.gz",
            "ProductId": "some-product-id",
        }
    )
    sagemaker_session.sagemaker_client.describe_model_package = Mock(
        return_value=model_package_response
    )

    model_package = ModelPackage(
        role="role", model_package_arn="my-model-package", sagemaker_session=sagemaker_session
    )
    assert model_package.enable_network_isolation() is True


@patch("sagemaker.utils.name_from_base")
def test_create_sagemaker_model_uses_model_name(name_from_base, sagemaker_session):
    model_name = "my-model"
    model_package_name = "my-model-package"

    model_package = ModelPackage(
        role="role",
        name=model_name,
        model_package_arn=model_package_name,
        sagemaker_session=sagemaker_session,
    )

    model_package._create_sagemaker_model()

    assert model_name == model_package.name
    name_from_base.assert_not_called()

    sagemaker_session.create_model.assert_called_with(
        model_name,
        "role",
        {"ModelPackageName": model_package_name},
        vpc_config=None,
        enable_network_isolation=False,
    )


def test_create_sagemaker_model_include_environment_variable(sagemaker_session):
    model_name = "my-model"
    model_package_name = "my-model-package"
    env_key = "env_key"
    env_value = "env_value"
    environment = {env_key: env_value}

    model_package = ModelPackage(
        role="role",
        name=model_name,
        model_package_arn=model_package_name,
        env=environment,
        sagemaker_session=sagemaker_session,
    )

    model_package._create_sagemaker_model()

    sagemaker_session.create_model.assert_called_with(
        model_name,
        "role",
        {"ModelPackageName": model_package_name, "Environment": environment},
        vpc_config=None,
        enable_network_isolation=False,
    )


@patch("sagemaker.utils.name_from_base")
def test_create_sagemaker_model_generates_model_name(name_from_base, sagemaker_session):
    model_package_name = "my-model-package"

    model_package = ModelPackage(
        role="role", model_package_arn=model_package_name, sagemaker_session=sagemaker_session
    )

    model_package._create_sagemaker_model()

    name_from_base.assert_called_with(model_package_name)
    assert name_from_base.return_value == model_package.name


@patch("sagemaker.utils.name_from_base")
def test_create_sagemaker_model_generates_model_name_each_time(name_from_base, sagemaker_session):
    model_package_name = "my-model-package"

    model_package = ModelPackage(
        role="role", model_package_arn=model_package_name, sagemaker_session=sagemaker_session
    )

    model_package._create_sagemaker_model()
    model_package._create_sagemaker_model()

    name_from_base.assert_called_with(model_package_name)
    assert 2 == name_from_base.call_count


@patch("sagemaker.model.ModelPackage._create_sagemaker_model", Mock())
def test_model_package_create_transformer(sagemaker_session):
    model_package = ModelPackage(
        role="role", model_package_arn="my-model-package", sagemaker_session=sagemaker_session
    )
    model_package.name = "auto-generated-model"
    transformer = model_package.transformer(
        instance_count=1, instance_type="ml.m4.xlarge", env={"test": True}
    )
    assert isinstance(transformer, sagemaker.transformer.Transformer)
    assert transformer.model_name == "auto-generated-model"
    assert transformer.instance_type == "ml.m4.xlarge"
    assert transformer.env == {"test": True}


@patch("sagemaker.model.ModelPackage._create_sagemaker_model", Mock())
def test_model_package_create_transformer_with_product_id(sagemaker_session):
    model_package_response = copy.deepcopy(DESCRIBE_MODEL_PACKAGE_RESPONSE)
    model_package_response["InferenceSpecification"]["Containers"].append(
        {
            "Image": "1.dkr.ecr.us-east-2.amazonaws.com/some-container:latest",
            "ModelDataUrl": "s3://bucket/output/model.tar.gz",
            "ProductId": "some-product-id",
        }
    )
    sagemaker_session.sagemaker_client.describe_model_package = Mock(
        return_value=model_package_response
    )

    model_package = ModelPackage(
        role="role", model_package_arn="my-model-package", sagemaker_session=sagemaker_session
    )
    model_package.name = "auto-generated-model"
    transformer = model_package.transformer(
        instance_count=1, instance_type="ml.m4.xlarge", env={"test": True}
    )
    assert isinstance(transformer, sagemaker.transformer.Transformer)
    assert transformer.model_name == "auto-generated-model"
    assert transformer.instance_type == "ml.m4.xlarge"
    assert transformer.env is None
