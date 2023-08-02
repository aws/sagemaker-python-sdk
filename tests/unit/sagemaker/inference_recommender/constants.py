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
"""This module stores constants related to SageMaker Inference Recommender."""
from __future__ import absolute_import

IR_MODEL_DATA = "s3://bucket/ir-model.tar.gz"
IR_COMPILATION_MODEL_DATA = "s3://bucket/ir-neo-model.tar.gz"
IR_IMAGE = "123.dkr.ecr-us-west-2.amazonaws.com/sagemake-container:1.0"
IR_COMPILATION_IMAGE = "123.dkr.ecr-us-west-2.amazonaws.com/sagemake-neo-container:1.0"
IR_ENV = {"TS_DEFAULT_WORKERS_PER_MODEL": "4"}

IR_JOB_NAME = "SMPYTHONSDK-1234567891"
IR_ROLE_ARN = "arn:aws:iam::123456789123:role/service-role/AmazonSageMaker-ExecutionRole-UnitTest"
IR_SAMPLE_PAYLOAD_URL = "s3://sagemaker-us-west-2-123456789123/payload/payload.tar.gz"
IR_SUPPORTED_CONTENT_TYPES = ["text/csv"]
IR_MODEL_PACKAGE_VERSION_ARN = (
    "arn:aws:sagemaker:us-west-2:123456789123:model-package/unit-test-package-version/1"
)
IR_INFERENCE_SPEC_NAME = "neo-00000011-1222"
IR_COMPILATION_JOB_NAME = "neo-00011122-2333"
IR_MODEL_NAME = "ir-model-name"
IR_NEAREST_MODEL_NAME = "xgboost"
IR_FRAMEWORK = "XGBOOST"
IR_FRAMEWORK_VERSION = "1.2.0"
IR_NEAREST_MODEL_NAME = "xgboost"
IR_SUPPORTED_INSTANCE_TYPES = ["ml.c5.xlarge", "ml.c5.2xlarge"]

INVALID_RECOMMENDATION_ID = "ir-job6ab0ff22"
NOT_EXISTED_RECOMMENDATION_ID = IR_JOB_NAME + "/ad3ec9ee"
NOT_EXISTED_MODEL_RECOMMENDATION_ID = IR_MODEL_NAME + "/ad3ec9ee"
RECOMMENDATION_ID = IR_JOB_NAME + "/5bcee92e"
MODEL_RECOMMENDATION_ID = IR_MODEL_NAME + "/v0KObO5d"
MODEL_RECOMMENDATION_ENV = {"TS_DEFAULT_WORKERS_PER_MODEL": "4"}

IR_CONTAINER_CONFIG = {
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
}

MODEL_CONFIG_WITH_ENV = {
    "EnvironmentParameters": [
        {"Key": "TS_DEFAULT_WORKERS_PER_MODEL", "ValueType": "string", "Value": "4"}
    ]
}

MODEL_CONFIG_WITH_INFERENCE_SPEC = {
    "InferenceSpecificationName": IR_INFERENCE_SPEC_NAME,
}

MODEL_CONFIG_WITH_COMPILATION_JOB_NAME = {"CompilationJobName": IR_COMPILATION_JOB_NAME}

IR_RECOMMENDATION_BASE = {
    "RecommendationId": RECOMMENDATION_ID,
    "Metrics": {
        "CostPerHour": 0.7360000014305115,
        "CostPerInference": 7.456940238625975e-06,
        "MaxInvocations": 1645,
        "ModelLatency": 171,
    },
    "EndpointConfiguration": {
        "EndpointName": "sm-endpoint-name",
        "VariantName": "variant-name",
        "InstanceType": "ml.g4dn.xlarge",
        "InitialInstanceCount": 1,
    },
    "ModelConfiguration": {
        "EnvironmentParameters": [
            {"Key": "TS_DEFAULT_WORKERS_PER_MODEL", "ValueType": "string", "Value": "4"}
        ]
    },
}

DESCRIBE_MODEL_RESPONSE = {
    "ModelName": IR_MODEL_NAME,
    "CreationTime": 1542752036.687,
    "ExecutionRoleArn": "arn:aws:iam::111111111111:role/IrRole",
    "ModelArn": "arn:aws:sagemaker:us-east-2:123:model-package/ir-model",
    "PrimaryContainer": {
        "Environment": {"SAGEMAKER_REGION": "us-west-2"},
        "Image": IR_IMAGE,
        "ModelDataUrl": IR_MODEL_DATA,
    },
    "DeploymentRecommendation": {
        "RecommendationStatus": "COMPLETED",
        "RealTimeInferenceRecommendations": [
            {
                "RecommendationId": MODEL_RECOMMENDATION_ID,
                "InstanceType": "ml.g4dn.2xlarge",
                "Environment": MODEL_RECOMMENDATION_ENV,
            },
            {
                "RecommendationId": "test-model-name/d248qVYU",
                "InstanceType": "ml.c6i.large",
                "Environment": {},
            },
        ],
    },
}

DESCRIBE_MODEL_PACKAGE_RESPONSE = {
    "AdditionalInferenceSpecificationDefinition": {
        "SupportedResponseMIMETypes": ["text"],
        "SupportedContentTypes": ["text/csv"],
        "Containers": [
            {
                "Image": IR_COMPILATION_IMAGE,
                "ImageDigest": "sha256:1234556789",
                "ModelDataUrl": IR_COMPILATION_MODEL_DATA,
            }
        ],
        "SupportedRealtimeInferenceInstanceTypes": IR_SUPPORTED_INSTANCE_TYPES,
    },
    "InferenceSpecification": {
        "SupportedResponseMIMETypes": ["text"],
        "SupportedContentTypes": ["text/csv"],
        "Containers": [
            {
                "Image": IR_IMAGE,
                "ImageDigest": "sha256:1234556789",
                "ModelDataUrl": IR_MODEL_DATA,
            }
        ],
        "SupportedRealtimeInferenceInstanceTypes": IR_SUPPORTED_INSTANCE_TYPES,
    },
    "ModelPackageDescription": "Model Package created for deploying recommendation id test.",
    "CreationTime": 1542752036.687,
    "ModelPackageArn": "arn:aws:sagemaker:us-east-2:123:model-package/ir-model-package",
    "ModelPackageStatus": "Completed",
    "ModelPackageName": "ir-model-package",
}

DESCRIBE_COMPILATION_JOB_RESPONSE = {
    "CompilationJobStatus": "Completed",
    "ModelArtifacts": {"S3ModelArtifacts": IR_COMPILATION_MODEL_DATA},
    "InferenceImage": IR_COMPILATION_IMAGE,
}

IR_CONTAINER_DEF = {
    "Image": IR_IMAGE,
    "Environment": IR_ENV,
    "ModelDataUrl": IR_MODEL_DATA,
}

DEPLOYMENT_RECOMMENDATION_CONTAINER_DEF = {
    "Image": IR_IMAGE,
    "Environment": MODEL_RECOMMENDATION_ENV,
    "ModelDataUrl": IR_MODEL_DATA,
}

IR_COMPILATION_CONTAINER_DEF = {
    "Image": IR_COMPILATION_IMAGE,
    "Environment": {},
    "ModelDataUrl": IR_COMPILATION_MODEL_DATA,
}

IR_MODEL_PACKAGE_CONTAINER_DEF = {
    "ModelPackageName": IR_MODEL_PACKAGE_VERSION_ARN,
    "Environment": IR_ENV,
}

IR_COMPILATION_MODEL_PACKAGE_CONTAINER_DEF = {
    "ModelPackageName": IR_MODEL_PACKAGE_VERSION_ARN,
}
