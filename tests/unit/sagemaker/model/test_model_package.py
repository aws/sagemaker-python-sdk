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
from sagemaker.model_card.model_card import ModelCard, ModelOverview
from sagemaker.model_card.schema_constraints import ModelApprovalStatusEnum, ModelCardStatusEnum

MODEL_PACKAGE_VERSIONED_ARN = (
    "arn:aws:sagemaker:us-west-2:001234567890:model-package/testmodelgroup/1"
)

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
    "ModelApprovalStatus": "PendingManualApproval",
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
    "ModelCard": {
        "ModelCardStatus": "Draft",
        "ModelCardContent": '{"model_overview": {"model_creator": "updatedCreator", "model_artifact": []}}',
    },
}

MODEL_DATA = {
    "S3DataSource": {
        "S3Uri": "s3://bucket/model/prefix/",
        "S3DataType": "S3Prefix",
        "CompressionType": "None",
    }
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
        tags=None,
    )


@pytest.mark.parametrize(
    "model_package_arn",
    [
        "arn:aws:sagemaker:us-east-2:123:model-package/my-model-package-arn",
        "arn:aws:sagemaker:us-east-2:123:model-package/my-model-package-arn/12",
    ],
)
@patch("sagemaker.utils.name_from_base")
def test_create_sagemaker_model_uses_model_package_arn(
    name_from_base, sagemaker_session, model_package_arn
):
    model_name = "my-model"

    model_package = ModelPackage(
        role="role",
        name=model_name,
        model_package_arn=model_package_arn,
        sagemaker_session=sagemaker_session,
    )

    model_package._create_sagemaker_model()

    assert model_name == model_package.name
    name_from_base.assert_not_called()

    sagemaker_session.create_model.assert_called_with(
        model_name,
        "role",
        {"ModelPackageName": model_package_arn},
        vpc_config=None,
        enable_network_isolation=False,
        tags=None,
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
        tags=None,
    )


def test_create_sagemaker_model_include_tags(sagemaker_session):
    model_name = "my-model"
    model_package_name = "my-model-package"
    env_key = "env_key"
    env_value = "env_value"
    environment = {env_key: env_value}
    tags = [{"Key": "foo", "Value": "bar"}]

    model_package = ModelPackage(
        role="role",
        name=model_name,
        model_package_arn=model_package_name,
        env=environment,
        sagemaker_session=sagemaker_session,
    )

    model_package.deploy(tags=tags, instance_type="ml.p2.xlarge", initial_instance_count=1)

    sagemaker_session.create_model.assert_called_with(
        model_name,
        "role",
        {"ModelPackageName": model_package_name, "Environment": environment},
        vpc_config=None,
        enable_network_isolation=False,
        tags=tags,
    )


def test_model_package_model_data_source_supported(sagemaker_session):
    model_data_source = {
        "S3DataSource": {
            "S3Uri": "s3://bucket/model/prefix/",
            "S3DataType": "S3Prefix",
            "CompressionType": "None",
        }
    }
    model_package = ModelPackage(
        role="role",
        model_package_arn="my-model-package",
        model_data=model_data_source,
        sagemaker_session=sagemaker_session,
    )
    assert model_package.model_data == model_package.model_data


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


@patch("sagemaker.model.ModelPackage.update_approval_status")
def test_model_package_auto_approve_on_deploy(update_approval_status, sagemaker_session):
    tags = [{"Key": "foo", "Value": "bar"}]
    model_package = ModelPackage(
        role="role",
        model_package_arn=MODEL_PACKAGE_VERSIONED_ARN,
        sagemaker_session=sagemaker_session,
    )
    model_package.deploy(tags=tags, instance_type="ml.p2.xlarge", initial_instance_count=1)

    assert (
        update_approval_status.call_args_list[0][1]["approval_status"]
        == ModelApprovalStatusEnum.APPROVED
    )


def test_update_customer_metadata(sagemaker_session):
    model_package = ModelPackage(
        role="role",
        model_package_arn=MODEL_PACKAGE_VERSIONED_ARN,
        sagemaker_session=sagemaker_session,
    )

    customer_metadata_to_update = {
        "Key": "Value",
    }
    model_package.update_customer_metadata(customer_metadata_properties=customer_metadata_to_update)

    sagemaker_session.sagemaker_client.update_model_package.assert_called_with(
        ModelPackageArn=MODEL_PACKAGE_VERSIONED_ARN,
        CustomerMetadataProperties=customer_metadata_to_update,
    )


def test_remove_customer_metadata(sagemaker_session):
    model_package = ModelPackage(
        role="role",
        model_package_arn=MODEL_PACKAGE_VERSIONED_ARN,
        sagemaker_session=sagemaker_session,
    )

    customer_metadata_to_remove = ["Key"]

    model_package.remove_customer_metadata_properties(
        customer_metadata_properties_to_remove=customer_metadata_to_remove
    )

    sagemaker_session.sagemaker_client.update_model_package.assert_called_with(
        ModelPackageArn=MODEL_PACKAGE_VERSIONED_ARN,
        CustomerMetadataPropertiesToRemove=customer_metadata_to_remove,
    )


def test_add_inference_specification(sagemaker_session):
    model_package = ModelPackage(
        role="role",
        model_package_arn=MODEL_PACKAGE_VERSIONED_ARN,
        sagemaker_session=sagemaker_session,
    )

    image_uris = ["image_uri"]

    containers = [{"Image": "image_uri"}]

    try:
        model_package.add_inference_specification(
            image_uris=image_uris, name="Inference", containers=containers
        )
    except ValueError as ve:
        assert "Cannot have both containers and image_uris." in str(ve)

    try:
        model_package.add_inference_specification(name="Inference")
    except ValueError as ve:
        assert "Should have either containers or image_uris for inference." in str(ve)

    model_package.add_inference_specification(image_uris=image_uris, name="Inference")

    sagemaker_session.sagemaker_client.update_model_package.assert_called_with(
        ModelPackageArn=MODEL_PACKAGE_VERSIONED_ARN,
        AdditionalInferenceSpecificationsToAdd=[
            {
                "Containers": [{"Image": "image_uri"}],
                "Name": "Inference",
            }
        ],
    )


def test_update_inference_specification(sagemaker_session):
    model_package = ModelPackage(
        role="role",
        model_package_arn=MODEL_PACKAGE_VERSIONED_ARN,
        sagemaker_session=sagemaker_session,
    )

    image_uris = ["image_uri"]

    containers = [{"Image": "image_uri"}]

    try:
        model_package.update_inference_specification(image_uris=image_uris, containers=containers)
    except ValueError as ve:
        assert "Should have either containers or image_uris for inference." in str(ve)

    try:
        model_package.update_inference_specification()
    except ValueError as ve:
        assert "Should have either containers or image_uris for inference." in str(ve)

    model_package.update_inference_specification(image_uris=image_uris)

    sagemaker_session.sagemaker_client.update_model_package.assert_called_with(
        ModelPackageArn=MODEL_PACKAGE_VERSIONED_ARN,
        InferenceSpecification={
            "Containers": [{"Image": "image_uri"}],
        },
    )


def test_update_source_uri(sagemaker_session):
    source_uri = "dummy_source_uri"
    model_package = ModelPackage(
        role="role",
        model_package_arn=MODEL_PACKAGE_VERSIONED_ARN,
        sagemaker_session=sagemaker_session,
    )
    model_package.update_source_uri(source_uri=source_uri)
    sagemaker_session.sagemaker_client.update_model_package.assert_called_with(
        ModelPackageArn=MODEL_PACKAGE_VERSIONED_ARN, SourceUri=source_uri
    )


def test_update_model_card(sagemaker_session):
    model_package_response = copy.deepcopy(DESCRIBE_MODEL_PACKAGE_RESPONSE)

    sagemaker_session.sagemaker_client.describe_model_package = Mock(
        return_value=model_package_response
    )
    model_package = ModelPackage(
        role="role",
        model_package_arn=MODEL_PACKAGE_VERSIONED_ARN,
        sagemaker_session=sagemaker_session,
    )

    update_my_card = ModelCard(
        name="UpdateTestName",
        sagemaker_session=sagemaker_session,
        status=ModelCardStatusEnum.PENDING_REVIEW,
    )
    model_package.update_model_card(update_my_card)
    update_my_card_req = update_my_card._create_request_args()
    del update_my_card_req["ModelCardName"]
    del update_my_card_req["Content"]
    sagemaker_session.sagemaker_client.update_model_package.assert_called_with(
        ModelPackageArn=MODEL_PACKAGE_VERSIONED_ARN, ModelCard=update_my_card_req
    )

    model_overview = ModelOverview(
        model_creator="UpdatedNewCreator",
    )
    update_my_card_1 = ModelCard(
        name="UpdateTestName",
        sagemaker_session=sagemaker_session,
        status=ModelCardStatusEnum.DRAFT,
        model_overview=model_overview,
    )
    model_package.update_model_card(update_my_card_1)
    update_my_card_req_1 = update_my_card_1._create_request_args()
    del update_my_card_req_1["ModelCardName"]
    del update_my_card_req_1["ModelCardStatus"]
    update_my_card_req_1["ModelCardContent"] = update_my_card_req_1["Content"]
    del update_my_card_req_1["Content"]
    sagemaker_session.sagemaker_client.update_model_package.assert_called_with(
        ModelPackageArn=MODEL_PACKAGE_VERSIONED_ARN, ModelCard=update_my_card_req_1
    )
