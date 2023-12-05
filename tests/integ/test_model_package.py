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