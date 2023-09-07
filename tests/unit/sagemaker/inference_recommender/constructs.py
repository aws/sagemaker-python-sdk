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
"""This module stores constructs related to SageMaker Inference Recommender."""
from __future__ import absolute_import
from tests.unit.sagemaker.inference_recommender.constants import (
    IR_JOB_NAME,
    IR_RECOMMENDATION_BASE,
    IR_ROLE_ARN,
    IR_FRAMEWORK,
    IR_SAMPLE_PAYLOAD_URL,
    IR_SUPPORTED_CONTENT_TYPES,
    IR_FRAMEWORK_VERSION,
    IR_NEAREST_MODEL_NAME,
    IR_SUPPORTED_INSTANCE_TYPES,
    IR_MODEL_PACKAGE_VERSION_ARN,
    IR_MODEL_NAME,
    IR_CONTAINER_CONFIG,
    MODEL_CONFIG_WITH_COMPILATION_JOB_NAME,
    MODEL_CONFIG_WITH_INFERENCE_SPEC,
)


def create_inference_recommendations_job_default_base_response():
    return {
        "JobName": IR_JOB_NAME,
        "JobType": "Default",
        "RoleArn": IR_ROLE_ARN,
        "InputConfig": {
            "ContainerConfig": {
                "Domain": "MACHINE_LEARNING",
                "Task": "OTHER",
                "Framework": IR_FRAMEWORK,
                "PayloadConfig": {
                    "SamplePayloadUrl": IR_SAMPLE_PAYLOAD_URL,
                    "SupportedContentTypes": IR_SUPPORTED_CONTENT_TYPES,
                },
                "FrameworkVersion": IR_FRAMEWORK_VERSION,
                "NearestModelName": IR_NEAREST_MODEL_NAME,
                "SupportedInstanceTypes": IR_SUPPORTED_INSTANCE_TYPES,
            },
        },
        "JobDescription": "Inference Recommendations Job created with Python SDK",
    }


def create_inference_recommendations_job_default_with_model_package_arn():
    base_job_response = create_inference_recommendations_job_default_base_response()
    base_job_response["InputConfig"]["ModelPackageVersionArn"] = IR_MODEL_PACKAGE_VERSION_ARN
    base_job_response["InferenceRecommendations"] = [IR_RECOMMENDATION_BASE]

    return base_job_response


def create_inference_recommendations_job_default_with_model_package_arn_and_compilation():
    base_job_response = create_inference_recommendations_job_default_with_model_package_arn()
    base_job_response["InferenceRecommendations"][0][
        "ModelConfiguration"
    ] = MODEL_CONFIG_WITH_INFERENCE_SPEC
    return base_job_response


def create_inference_recommendations_job_default_with_model_name():
    base_job_response = create_inference_recommendations_job_default_base_response()
    base_job_response["InputConfig"]["ModelName"] = IR_MODEL_NAME
    base_job_response["InputConfig"]["ContainerConfig"] = IR_CONTAINER_CONFIG
    base_job_response["InferenceRecommendations"] = [IR_RECOMMENDATION_BASE]

    return base_job_response


def create_inference_recommendations_job_default_with_model_name_and_compilation():
    base_job_response = create_inference_recommendations_job_default_with_model_name()
    base_job_response["InferenceRecommendations"][0][
        "ModelConfiguration"
    ] = MODEL_CONFIG_WITH_COMPILATION_JOB_NAME
    return base_job_response
