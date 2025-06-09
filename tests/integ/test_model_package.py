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

import json
import os
from sagemaker.model_card.model_card import (
    AdditionalInformation,
    BusinessDetails,
    IntendedUses,
    ModelCard,
    ModelOverview,
    ModelPackageModelCard,
)
from sagemaker.model_card.schema_constraints import ModelApprovalStatusEnum, ModelCardStatusEnum
from sagemaker.utils import unique_name_from_base
from tests.integ import DATA_DIR
from sagemaker.xgboost import XGBoostModel
from sagemaker import image_uris
from sagemaker.session import get_execution_role
from sagemaker.model import ModelPackage
from sagemaker.model_life_cycle import ModelLifeCycle

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


def test_update_model_life_cycle_model_package(sagemaker_session):

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

    create_model_life_cycle = ModelLifeCycle(
        stage="Development",
        stage_status="In-Progress",
        stage_description="Development In Progress",
    )
    model_package = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_group_name,
        model_life_cycle=create_model_life_cycle,
    )

    desc_model_package = sagemaker_session.sagemaker_client.describe_model_package(
        ModelPackageName=model_package.model_package_arn
    )

    assert desc_model_package["ModelLifeCycle"] == create_model_life_cycle

    update_model_life_cycle = ModelLifeCycle(
        stage="Staging",
        stage_status="In-Progress",
        stage_description="Sending for Staging Verification",
    )
    update_model_life_cycle_req = update_model_life_cycle._to_request_dict()

    model_package.update_model_life_cycle(update_model_life_cycle_req)

    desc_model_package = sagemaker_session.sagemaker_client.describe_model_package(
        ModelPackageName=model_package.model_package_arn
    )
    assert desc_model_package["ModelLifeCycle"] == update_model_life_cycle

    sagemaker_session.sagemaker_client.delete_model_package(
        ModelPackageName=model_package.model_package_arn
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


def test_update_model_card_with_model_card_object(sagemaker_session):
    model_group_name = unique_name_from_base("test-model-group")
    intended_uses = IntendedUses(
        purpose_of_model="Test model card.",
        intended_uses="Not used except this test.",
        factors_affecting_model_efficiency="No.",
        risk_rating="Low",
        explanations_for_risk_rating="Just an example.",
    )
    business_details = BusinessDetails(
        business_problem="The business problem that your model is used to solve.",
        business_stakeholders="The stakeholders who have the interest in the business that your model is used for.",
        line_of_business="Services that the business is offering.",
    )
    additional_information = AdditionalInformation(
        ethical_considerations="Your model ethical consideration.",
        caveats_and_recommendations="Your model's caveats and recommendations.",
        custom_details={"custom details1": "details value"},
    )

    model_overview = ModelOverview(model_creator="TestCreator")

    my_card = ModelCard(
        name="TestName",
        sagemaker_session=sagemaker_session,
        status=ModelCardStatusEnum.DRAFT,
        model_overview=model_overview,
        intended_uses=intended_uses,
        business_details=business_details,
        additional_information=additional_information,
    )

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
        model_card=my_card,
    )

    desc_model_package = sagemaker_session.sagemaker_client.describe_model_package(
        ModelPackageName=model_package.model_package_arn
    )

    updated_model_overview = ModelOverview(model_creator="updatedCreator")
    updated_intended_uses = IntendedUses(
        purpose_of_model="Updated Test model card.",
    )
    updated_my_card = ModelCard(
        name="TestName",
        sagemaker_session=sagemaker_session,
        model_overview=updated_model_overview,
        intended_uses=updated_intended_uses,
    )
    model_package.update_model_card(updated_my_card)
    desc_model_package = sagemaker_session.sagemaker_client.describe_model_package(
        ModelPackageName=model_package.model_package_arn
    )

    model_card_content = json.loads(desc_model_package["ModelCard"]["ModelCardContent"])
    assert model_card_content["intended_uses"]["purpose_of_model"] == "Updated Test model card."
    assert model_card_content["model_overview"]["model_creator"] == "updatedCreator"
    updated_my_card_status = ModelCard(
        name="TestName",
        sagemaker_session=sagemaker_session,
        status=ModelCardStatusEnum.PENDING_REVIEW,
    )
    model_package.update_model_card(updated_my_card_status)
    desc_model_package = sagemaker_session.sagemaker_client.describe_model_package(
        ModelPackageName=model_package.model_package_arn
    )

    model_card_content = json.loads(desc_model_package["ModelCard"]["ModelCardContent"])
    assert desc_model_package["ModelCard"]["ModelCardStatus"] == ModelCardStatusEnum.PENDING_REVIEW


def test_update_model_card_with_model_card_json(sagemaker_session):
    model_group_name = unique_name_from_base("test-model-group")
    model_card_content = {
        "model_overview": {
            "model_creator": "TestCreator",
        },
        "intended_uses": {
            "purpose_of_model": "Test model card.",
            "intended_uses": "Not used except this test.",
            "factors_affecting_model_efficiency": "No.",
            "risk_rating": "Low",
            "explanations_for_risk_rating": "Just an example.",
        },
        "business_details": {
            "business_problem": "The business problem that your model is used to solve.",
            "business_stakeholders": "The stakeholders who have the interest in the business.",
            "line_of_business": "Services that the business is offering.",
        },
        "evaluation_details": [
            {
                "name": "Example evaluation job",
                "evaluation_observation": "Evaluation observations.",
                "metric_groups": [
                    {
                        "name": "binary classification metrics",
                        "metric_data": [{"name": "accuracy", "type": "number", "value": 0.5}],
                    }
                ],
            }
        ],
        "additional_information": {
            "ethical_considerations": "Your model ethical consideration.",
            "caveats_and_recommendations": 'Your model"s caveats and recommendations.',
            "custom_details": {"custom details1": "details value"},
        },
    }
    my_card = ModelPackageModelCard(
        model_card_status=ModelCardStatusEnum.DRAFT, model_card_content=model_card_content
    )

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
        model_card=my_card,
    )

    desc_model_package = sagemaker_session.sagemaker_client.describe_model_package(
        ModelPackageName=model_package.model_package_arn
    )

    updated_model_card_content = {
        "model_overview": {
            "model_creator": "updatedCreator",
        },
        "intended_uses": {
            "purpose_of_model": "Updated Test model card.",
            "intended_uses": "Not used except this test.",
            "factors_affecting_model_efficiency": "No.",
            "risk_rating": "Low",
            "explanations_for_risk_rating": "Just an example.",
        },
        "business_details": {
            "business_problem": "The business problem that your model is used to solve.",
            "business_stakeholders": "The stakeholders who have the interest in the business.",
            "line_of_business": "Services that the business is offering.",
        },
        "evaluation_details": [
            {
                "name": "Example evaluation job",
                "evaluation_observation": "Evaluation observations.",
                "metric_groups": [
                    {
                        "name": "binary classification metrics",
                        "metric_data": [{"name": "accuracy", "type": "number", "value": 0.5}],
                    }
                ],
            }
        ],
        "additional_information": {
            "ethical_considerations": "Your model ethical consideration.",
            "caveats_and_recommendations": 'Your model"s caveats and recommendations.',
            "custom_details": {"custom details1": "details value"},
        },
    }
    updated_my_card = ModelPackageModelCard(
        model_card_status=ModelCardStatusEnum.DRAFT, model_card_content=updated_model_card_content
    )
    model_package.update_model_card(updated_my_card)
    desc_model_package = sagemaker_session.sagemaker_client.describe_model_package(
        ModelPackageName=model_package.model_package_arn
    )

    model_card_content = json.loads(desc_model_package["ModelCard"]["ModelCardContent"])
    assert model_card_content["intended_uses"]["purpose_of_model"] == "Updated Test model card."
    assert model_card_content["model_overview"]["model_creator"] == "updatedCreator"
    updated_my_card_status = ModelPackageModelCard(
        model_card_status=ModelCardStatusEnum.PENDING_REVIEW,
    )
    model_package.update_model_card(updated_my_card_status)
    desc_model_package = sagemaker_session.sagemaker_client.describe_model_package(
        ModelPackageName=model_package.model_package_arn
    )

    assert desc_model_package["ModelCard"]["ModelCardStatus"] == ModelCardStatusEnum.PENDING_REVIEW


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
