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

import os
from sagemaker.model_card.schema_constraints import ModelApprovalStatusEnum
from sagemaker.utils import unique_name_from_base
from tests.integ import DATA_DIR
from sagemaker.xgboost import XGBoostModel
from sagemaker import image_uris
from sagemaker.session import get_execution_role
from sagemaker.model import ModelPackage

_XGBOOST_PATH = os.path.join(DATA_DIR, "xgboost_abalone")


def test_update_approval_model_package(sagemaker_session):

    model_group_name = unique_name_from_base("test-model-group")

    sagemaker_session.sagemaker_client.create_model_package_group(
        ModelPackageGroupName=model_group_name
    )

    xgb_model_data_s3 = sagemaker_session.upload_data(
        path=os.path.join(_XGBOOST_PATH, "xgb_model.tar.gz"),
        key_prefix="integ-test-data/xgboost/model",
    )
    model = XGBoostModel(
        model_data=xgb_model_data_s3, framework_version="1.3-1", sagemaker_session=sagemaker_session
    )

    model_package = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_group_name,
    )

    model_package.update_approval_status(
        approval_status=ModelApprovalStatusEnum.APPROVED, approval_description="dummy"
    )

    desc_model_package = sagemaker_session.sagemaker_client.describe_model_package(
        ModelPackageName=model_package.model_package_arn
    )
    assert desc_model_package["ModelApprovalStatus"] == ModelApprovalStatusEnum.APPROVED
    assert desc_model_package["ApprovalDescription"] == "dummy"

    sagemaker_session.sagemaker_client.delete_model_package(
        ModelPackageName=model_package.model_package_arn
    )
    sagemaker_session.sagemaker_client.delete_model_package_group(
        ModelPackageGroupName=model_group_name
    )


def test_inference_specification_addition(sagemaker_session):

    model_group_name = unique_name_from_base("test-model-group")

    sagemaker_session.sagemaker_client.create_model_package_group(
        ModelPackageGroupName=model_group_name
    )

    xgb_model_data_s3 = sagemaker_session.upload_data(
        path=os.path.join(_XGBOOST_PATH, "xgb_model.tar.gz"),
        key_prefix="integ-test-data/xgboost/model",
    )
    model = XGBoostModel(
        model_data=xgb_model_data_s3, framework_version="1.3-1", sagemaker_session=sagemaker_session
    )

    model_package = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_group_name,
    )

    xgb_image = image_uris.retrieve(
        "xgboost", sagemaker_session.boto_region_name, version="1", image_scope="inference"
    )
    model_package.add_inference_specification(image_uris=[xgb_image], name="Inference")
    desc_model_package = sagemaker_session.sagemaker_client.describe_model_package(
        ModelPackageName=model_package.model_package_arn
    )
    assert len(desc_model_package["AdditionalInferenceSpecifications"]) == 1
    assert desc_model_package["AdditionalInferenceSpecifications"][0]["Name"] == "Inference"

    sagemaker_session.sagemaker_client.delete_model_package(
        ModelPackageName=model_package.model_package_arn
    )
    sagemaker_session.sagemaker_client.delete_model_package_group(
        ModelPackageGroupName=model_group_name
    )


def test_update_inference_specification(sagemaker_session):
    model_group_name = unique_name_from_base("test-model-group")
    source_uri = "dummy source uri"

    sagemaker_session.sagemaker_client.create_model_package_group(
        ModelPackageGroupName=model_group_name
    )

    model_package = sagemaker_session.sagemaker_client.create_model_package(
        ModelPackageGroupName=model_group_name, SourceUri=source_uri
    )

    mp = ModelPackage(
        role=get_execution_role(sagemaker_session),
        model_package_arn=model_package["ModelPackageArn"],
        sagemaker_session=sagemaker_session,
    )

    xgb_image = image_uris.retrieve(
        "xgboost", sagemaker_session.boto_region_name, version="1", image_scope="inference"
    )

    mp.update_inference_specification(image_uris=[xgb_image])

    desc_model_package = sagemaker_session.sagemaker_client.describe_model_package(
        ModelPackageName=model_package["ModelPackageArn"]
    )

    sagemaker_session.sagemaker_client.delete_model_package(
        ModelPackageName=model_package["ModelPackageArn"]
    )
    sagemaker_session.sagemaker_client.delete_model_package_group(
        ModelPackageGroupName=model_group_name
    )

    assert len(desc_model_package["InferenceSpecification"]["Containers"]) == 1
    assert desc_model_package["InferenceSpecification"]["Containers"][0]["Image"] == xgb_image


def test_update_source_uri(sagemaker_session):
    model_group_name = unique_name_from_base("test-model-group")
    source_uri = "dummy source uri"

    sagemaker_session.sagemaker_client.create_model_package_group(
        ModelPackageGroupName=model_group_name
    )

    xgb_model_data_s3 = sagemaker_session.upload_data(
        path=os.path.join(_XGBOOST_PATH, "xgb_model.tar.gz"),
        key_prefix="integ-test-data/xgboost/model",
    )
    model = XGBoostModel(
        model_data=xgb_model_data_s3, framework_version="1.3-1", sagemaker_session=sagemaker_session
    )

    model_package = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_group_name,
    )

    desc_model_package = sagemaker_session.sagemaker_client.describe_model_package(
        ModelPackageName=model_package.model_package_arn
    )

    model_package.update_source_uri(source_uri=source_uri)
    desc_model_package = sagemaker_session.sagemaker_client.describe_model_package(
        ModelPackageName=model_package.model_package_arn
    )

    assert desc_model_package["SourceUri"] == source_uri


def test_clone_model_package_using_source_uri(sagemaker_session):
    model_group_name = unique_name_from_base("test-model-group")

    sagemaker_session.sagemaker_client.create_model_package_group(
        ModelPackageGroupName=model_group_name
    )

    xgb_model_data_s3 = sagemaker_session.upload_data(
        path=os.path.join(_XGBOOST_PATH, "xgb_model.tar.gz"),
        key_prefix="integ-test-data/xgboost/model",
    )
    model = XGBoostModel(
        model_data=xgb_model_data_s3, framework_version="1.3-1", sagemaker_session=sagemaker_session
    )

    model_package = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_group_name,
        source_uri="dummy-source-uri",
    )

    desc_model_package = sagemaker_session.sagemaker_client.describe_model_package(
        ModelPackageName=model_package.model_package_arn
    )

    model2 = XGBoostModel(
        model_data=xgb_model_data_s3, framework_version="1.3-1", sagemaker_session=sagemaker_session
    )
    cloned_model_package = model2.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_group_name,
        source_uri=model_package.model_package_arn,
    )

    desc_cloned_model_package = sagemaker_session.sagemaker_client.describe_model_package(
        ModelPackageName=cloned_model_package.model_package_arn
    )

    sagemaker_session.sagemaker_client.delete_model_package(
        ModelPackageName=model_package.model_package_arn
    )
    sagemaker_session.sagemaker_client.delete_model_package(
        ModelPackageName=cloned_model_package.model_package_arn
    )
    sagemaker_session.sagemaker_client.delete_model_package_group(
        ModelPackageGroupName=model_group_name
    )

    assert len(desc_cloned_model_package["InferenceSpecification"]["Containers"]) == len(
        desc_model_package["InferenceSpecification"]["Containers"]
    )
    assert len(
        desc_cloned_model_package["InferenceSpecification"]["SupportedTransformInstanceTypes"]
    ) == len(desc_model_package["InferenceSpecification"]["SupportedTransformInstanceTypes"])
    assert len(
        desc_cloned_model_package["InferenceSpecification"][
            "SupportedRealtimeInferenceInstanceTypes"
        ]
    ) == len(
        desc_model_package["InferenceSpecification"]["SupportedRealtimeInferenceInstanceTypes"]
    )
    assert len(desc_cloned_model_package["InferenceSpecification"]["SupportedContentTypes"]) == len(
        desc_model_package["InferenceSpecification"]["SupportedContentTypes"]
    )
    assert len(
        desc_cloned_model_package["InferenceSpecification"]["SupportedResponseMIMETypes"]
    ) == len(desc_model_package["InferenceSpecification"]["SupportedResponseMIMETypes"])
    assert desc_cloned_model_package["SourceUri"] == model_package.model_package_arn


def test_register_model_using_source_uri(sagemaker_session):
    model_name = unique_name_from_base("test-model")
    model_group_name = unique_name_from_base("test-model-group")

    sagemaker_session.sagemaker_client.create_model_package_group(
        ModelPackageGroupName=model_group_name
    )

    xgb_model_data_s3 = sagemaker_session.upload_data(
        path=os.path.join(_XGBOOST_PATH, "xgb_model.tar.gz"),
        key_prefix="integ-test-data/xgboost/model",
    )

    model = XGBoostModel(
        model_data=xgb_model_data_s3,
        framework_version="1.3-1",
        sagemaker_session=sagemaker_session,
        role=get_execution_role(sagemaker_session),
    )

    model.name = model_name
    model.create()
    desc_model = sagemaker_session.sagemaker_client.describe_model(ModelName=model_name)

    model = XGBoostModel(
        model_data=xgb_model_data_s3,
        framework_version="1.3-1",
        sagemaker_session=sagemaker_session,
        role=get_execution_role(sagemaker_session),
    )
    registered_model_package = model.register(
        inference_instances=["ml.m5.xlarge"],
        model_package_group_name=model_group_name,
        source_uri=desc_model["ModelArn"],
    )

    desc_registered_model_package = sagemaker_session.sagemaker_client.describe_model_package(
        ModelPackageName=registered_model_package.model_package_arn
    )

    sagemaker_session.sagemaker_client.delete_model(ModelName=model_name)
    sagemaker_session.sagemaker_client.delete_model_package(
        ModelPackageName=registered_model_package.model_package_arn
    )
    sagemaker_session.sagemaker_client.delete_model_package_group(
        ModelPackageGroupName=model_group_name
    )

    assert desc_registered_model_package["SourceUri"] == desc_model["ModelArn"]
    assert "InferenceSpecification" in desc_registered_model_package
    assert desc_registered_model_package["InferenceSpecification"] is not None
