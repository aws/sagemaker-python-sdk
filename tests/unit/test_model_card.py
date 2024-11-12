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

import io
import os
import json
import datetime
import re
import logging
import pytest
from mock import patch, Mock
from botocore.exceptions import ClientError
import botocore.response

from sagemaker.model_card import schema_constraints
from sagemaker.model_card import (
    Environment,
    ModelOverview,
    IntendedUses,
    BusinessDetails,
    ObjectiveFunction,
    TrainingMetric,
    HyperParameter,
    Metric,
    TrainingDetails,
    MetricGroup,
    EvaluationJob,
    AdditionalInformation,
    ModelCard,
    Function,
    TrainingJobDetails,
    ModelPackage,
)
from sagemaker.model_card.model_card import (
    ModelCardExportJob,
    ModelPackageCreator,
    SourceAlgorithm,
    Container,
    InferenceSpecification,
)
from sagemaker.model_card.helpers import (
    _MaxSizeArray,
    _IsList,
    _OneOf,
    _IsModelCardObject,
    _JSONEncoder,
    _hash_content_str,
    _DefaultToRequestDict,
    _SkipEncodingDecoding,
)
from sagemaker.model_card.evaluation_metric_parsers import (
    EvaluationMetricTypeEnum,
    EVALUATION_METRIC_PARSERS,
)


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "model_card")
# model details arguments
MODEL_ID = "test_model_id"
MODEL_NAME = "test_model_name"
MODEL_DESCRIPTION = "this is a test model"
MODEL_CARD_STATUS = schema_constraints.ModelCardStatusEnum.DRAFT
MODEL_VERSION = 1
PROBLEM_TYPE = "binary classification"
ALGORITHM_TYPE = "xgboost"
MODEL_CREATOR = "null tester"
MODEL_OWNER = "null owner"
MODEL_ARTIFACT = ["s3://1", "s3://2"]
MODEL_IMAGE = "test model container image"
INFERENCE_ENVRIONMENT = Environment(container_image=[MODEL_IMAGE])

# model package details arguments
MODEL_PACKAGE_ARN = "arn:aws:sagemaker:us-west-2:001234567890:model-package/testmodelgroup/1"
MODEL_PACKAGE_DESCRIPTION = "this is test model package"
MODEL_PACKAGE_STATUS = "Pending"
MODEL_APPROVAL_STATUS = "PendingManualApproval"
APPROVAL_DESCRIPTION = "approval_description"
MODEL_PACKAGE_GROUP_NAME = "testmodelgroup"
MODEL_PACKAGE_NAME = "model_package_name"
MODEL_PACKAGE_VERSION = 1
DOMAIN = "domain"
TASK = "task"
USER_PROFILE_NAME = "test-user"
CREATED_BY = ModelPackageCreator(USER_PROFILE_NAME)
ALGORITHM_NAME = "test-algorithm-arn"
MODEL_DATA_URL = "s3://test"
SOURCE_ALGORITHMS = [SourceAlgorithm(ALGORITHM_NAME, MODEL_DATA_URL)]
NEAREST_MODEL_NAME = "test-model"
CONTAINERS = [Container(MODEL_IMAGE, MODEL_DATA_URL, NEAREST_MODEL_NAME)]
INFERENCE_SPECIFICATION = InferenceSpecification(CONTAINERS)
CLARIFY_BIAS_JSON_PATH = os.path.join(DATA_DIR, "evaluation_metrics/clarify_bias.json")
MODEL_METRICS = {
    "Bias": {"Report": {"ContentType": "application/json", "S3Uri": CLARIFY_BIAS_JSON_PATH}}
}

# intended uses auguments
PURPOSE_OF_MODEL = "mock model for testing"
INTENDED_USES = "this model card is used for development testing"
FACTORS_AFFECTING_MODEL_EFFICIENCY = "a bad factor"
RISK_RATING = schema_constraints.RiskRatingEnum.LOW
EXPLANATIONS_FOR_RISK_RATING = "ramdomly the first example"

# business details auguments
BUSINESS_PROBLEM = "mock model for business problem testing"
BUSINESS_STAKEHOLDERS = "business stakeholders testing"
LINE_OF_BUSINESS = "how many business models"

# training details arguments
OBJECITVE_FUNCTION_FUNC = schema_constraints.ObjectiveFunctionEnum.MINIMIZE
OBJECTIVE_FUNCTION_FACET = schema_constraints.FacetEnum.LOSS
OBJECTIVE_FUNCTION_CONDITION = "only under test condition"
OBJECTIVE_FUNCTION_NOTES = "test objective function"
TRAINING_OBSERVATIONS = "the trainnig look great!"
TRAINING_ARN = "test training job arn"
TRAINING_DATASETS = ["s3://1", "s3://2"]
TRAINING_IMAGE = "training environment container image"
TRAINING_ENVIRONMENT = Environment(container_image=[TRAINING_IMAGE])
TRAINING_METRICS = [TrainingMetric(name="binary_f_beta", value=0.965, notes="example")]
USER_METRIC_NAME = "test_metric"
USER_METRIC = TrainingMetric(name=USER_METRIC_NAME, value=1)
USER_PROVIDED_TRAINING_METRICS = [USER_METRIC]
HYPER_PARAMETER = [HyperParameter(name="binary_f_beta", value=0.965)]
USER_PARAMETER_NAME = "test_parameter"
USER_PARAMETER = HyperParameter(name=USER_PARAMETER_NAME, value=1)
USER_PROVIDED_HYPER_PARAMETER = [USER_PARAMETER]

# evaluation job arguments
EVALUATION_JOB_NAME = "evaluation job 1"
EVALUATION_OBSERVATION = "evaluation looks good"
EVALUATION_DATASETS = ["s3://3", "s3://4"]
EVALUATION_METADATA = {"key": "value"}
metric_example = Metric(
    name="test_evaluation_metric",
    type=schema_constraints.MetricTypeEnum.MATRIX,
    value=[[1, 2, 3], [4, 5, 6]],
    x_axis_name="x",
    y_axis_name="y",
)
metric_group_example = MetricGroup(
    name="first evaluation result",
    metric_data=[metric_example],
)
EVALUATION_METRIC_GROUPS = [metric_group_example]

# addtional information arguments
ETHICAL_CONSIDERATIONS = "there is no ethical consideration for this model card"
CAVEATS_AND_RECOMMENDATIONS = "attention: this is a pure test model card"
CUSTOM_DETAILS = {"custom details1": "details value"}
MODEL_CARD_NAME = "test_model_card"
MODEL_CARD_NAME_FOR_CARRY_OVER_ADDITIONAL_CONTENT = (
    "test_model_card_for_carry_over_additional_content"
)
MODEL_CARD_ARN = "test_model_card_arn"
MODEL_CARD_ARN_FOR_CARRY_OVER_ADDITIONAL_CONTENT = (
    "test_model_card_arn_for_carry_over_additional_content"
)
MODEL_CARD_VERSION = 1
CREATE_MODEL_CARD_RETURN_EXAMPLE = {
    "ModelCardArn": MODEL_CARD_ARN,
    "ResponseMetadata": {
        "RequestId": "c6d4f483-b4e1-4c41-ab51-7f9990137184",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "c6d4f483-b4e1-4c41-ab51-7f9990137184",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "75",
            "date": "Tue, 13 Sep 2022 14:59:41 GMT",
        },
        "RetryAttempts": 0,
    },
}
LOAD_MODEL_CARD_EXMPLE = {
    "ModelCardArn": MODEL_CARD_ARN,
    "ModelCardName": MODEL_CARD_NAME,
    "ModelCardVersion": MODEL_CARD_VERSION,
    "Content": "{}",
    "ModelCardStatus": MODEL_CARD_STATUS,
    "CreationTime": datetime.datetime(2022, 9, 17, 17, 15, 45, 672000),
    "CreatedBy": {},
    "LastModifiedTime": datetime.datetime(2022, 9, 17, 17, 15, 45, 672000),
    "LastModifiedBy": {},
    "ResponseMetadata": {
        "RequestId": "7f317f47-a1e5-45dc-975a-fa4d9df81365",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "7f317f47-a1e5-45dc-975a-fa4d9df81365",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "2429",
            "date": "Mon, 19 Sep 2022 21:09:05 GMT",
        },
        "RetryAttempts": 0,
    },
}

# sample response of model card with model package information in it
MODEL_CARD_WITH_MODEL_PACKAGE_MOCK_RESPONSE = {
    "ModelCardArn": MODEL_CARD_ARN,
    "ModelCardName": MODEL_CARD_NAME,
    "ModelCardVersion": MODEL_CARD_VERSION,
    "Content": json.dumps(
        {
            "model_overview": {"model_id": MODEL_PACKAGE_ARN},
            "model_package_details": {
                "model_package_arn": MODEL_PACKAGE_ARN,
                "model_package_name": MODEL_PACKAGE_NAME,
                "model_package_group_name": MODEL_PACKAGE_GROUP_NAME,
                "model_package_version": MODEL_PACKAGE_VERSION,
                "model_package_description": MODEL_PACKAGE_DESCRIPTION,
                "inference_specification": {
                    "containers": [
                        {
                            "image": MODEL_IMAGE,
                            "model_data_url": MODEL_DATA_URL,
                            "nearest_model_name": NEAREST_MODEL_NAME,
                        }
                    ]
                },
                "model_package_status": MODEL_PACKAGE_STATUS,
                "model_approval_status": MODEL_APPROVAL_STATUS,
                "approval_description": APPROVAL_DESCRIPTION,
                "created_by": {
                    "user_profile_name": USER_PROFILE_NAME,
                },
                "domain": DOMAIN,
                "task": TASK,
                "source_algorithms": [
                    {"algorithm_name": ALGORITHM_NAME, "model_data_url": MODEL_DATA_URL}
                ],
            },
        }
    ),
    "ModelCardStatus": MODEL_CARD_STATUS,
    "CreationTime": datetime.datetime(2022, 9, 17, 17, 15, 45, 672000),
    "CreatedBy": {},
    "LastModifiedTime": datetime.datetime(2022, 9, 17, 17, 15, 45, 672000),
    "LastModifiedBy": {},
    "ResponseMetadata": {
        "RequestId": "7f317f47-a1e5-45dc-975a-fa4d9df81365",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "7f317f47-a1e5-45dc-975a-fa4d9df81365",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "2429",
            "date": "Mon, 19 Sep 2022 21:09:05 GMT",
        },
        "RetryAttempts": 0,
    },
}

SIMPLE_MODEL_CARD_ARN = "simple_test_model_card"
SIMPLE_MODEL_CARD_NAME = "simple_test_model_card_arn"
SIMPLE_MODEL_CARD_VERSION = 1
SIMPLE_MODEL_CARD_STATUS = schema_constraints.ModelCardStatusEnum.DRAFT
CREATE_SIMPLE_MODEL_CARD_RETURN_EXAMPLE = {
    "ModelCardArn": SIMPLE_MODEL_CARD_ARN,
    "ResponseMetadata": {
        "RequestId": "e91f9a8f-2a1d-4ea8-a0dd-5a360d1865d5",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "e91f9a8f-2a1d-4ea8-a0dd-5a360d1865d5",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "75",
            "date": "Tue, 14 Sep 2022 14:59:41 GMT",
        },
        "RetryAttempts": 0,
    },
}
LOAD_SIMPLE_MODEL_CARD_EXMPLE = {
    "ModelCardArn": SIMPLE_MODEL_CARD_ARN,
    "ModelCardName": SIMPLE_MODEL_CARD_NAME,
    "ModelCardVersion": SIMPLE_MODEL_CARD_VERSION,
    "Content": json.dumps(
        {
            "additional_information": {
                "ethical_considerations": ETHICAL_CONSIDERATIONS,
                "caveats_and_recommendations": CAVEATS_AND_RECOMMENDATIONS,
                "custom_details": CUSTOM_DETAILS,
            }
        }
    ),
    "ModelCardStatus": SIMPLE_MODEL_CARD_STATUS,
    "CreationTime": datetime.datetime(2022, 9, 17, 17, 15, 45, 672000),
    "CreatedBy": {},
    "LastModifiedTime": datetime.datetime(2022, 9, 17, 17, 15, 45, 672000),
    "LastModifiedBy": {},
    "ResponseMetadata": {
        "RequestId": "7f317f47-a1e5-45dc-975a-fa4d9df81365",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "7f317f47-a1e5-45dc-975a-fa4d9df81365",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "2429",
            "date": "Mon, 19 Sep 2022 21:09:05 GMT",
        },
        "RetryAttempts": 0,
    },
}
UPDATE_SIMPLE_MODEL_CARD_EXAMPLE = {
    "ModelCardArn": SIMPLE_MODEL_CARD_ARN,
    "ResponseMetadata": {
        "RequestId": "47a11228-7059-42c1-b7db-76fa103106f7",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "47a11228-7059-42c1-b7db-76fa103106f7",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "77",
            "date": "Mon, 19 Sep 2022 21:35:17 GMT",
        },
        "RetryAttempts": 0,
    },
}
DELETE_SIMPLE_MODEL_CARD_RETURN_EXAMPLE = {
    "ResponseMetadata": {
        "RequestId": "7f4404e1-5a6d-4113-9ff0-9af220d7206f",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "7f4404e1-5a6d-4113-9ff0-9af220d7206f",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "0",
            "date": "Mon, 19 Sep 2022 21:28:12 GMT",
        },
        "RetryAttempts": 0,
    }
}
GENERAL_CLIENT_ERROR = ClientError(
    error_response={
        "Error": {"Code": "ConflictException", "Message": ""},
        "ResponseMetadata": {"MaxAttemptsReached": True, "RetryAttempts": 4},
    },
    operation_name="CreateModelCard",
)

# autodiscovery
DESCRIBE_MODEL_EXAMPLE = {
    "ModelName": MODEL_NAME,
    "PrimaryContainer": {
        "Image": MODEL_IMAGE,
        "Mode": "SingleModel",
        "ModelDataUrl": MODEL_ARTIFACT[0],
    },
    "ExecutionRoleArn": "test execution role arn",
    "CreationTime": datetime.datetime(2022, 9, 20, 13, 4, 9, 134000),
    "ModelArn": MODEL_ID,
    "EnableNetworkIsolation": False,
    "ResponseMetadata": {
        "RequestId": "43ff67e3-2ae5-479e-bed0-9caccc9e62f0",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "43ff67e3-2ae5-479e-bed0-9caccc9e62f0",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "590",
            "date": "Tue, 20 Sep 2022 19:55:36 GMT",
        },
        "RetryAttempts": 0,
    },
}

DESCRIBE_MODEL_PACKAGE_EXAMPLE = {
    "ModelPackageArn": MODEL_PACKAGE_ARN,
    "ModelPackageName": MODEL_PACKAGE_NAME,
    "ModelPackageGroupName": MODEL_PACKAGE_GROUP_NAME,
    "ModelPackageVersion": MODEL_PACKAGE_VERSION,
    "ModelPackageDescription": MODEL_PACKAGE_DESCRIPTION,
    "CreationTime": datetime.datetime(2022, 9, 20, 13, 4, 9, 134000),
    "InferenceSpecification": {
        "Containers": [
            {
                "Image": MODEL_IMAGE,
                "ImageDigest": "sha256:4814427c3e0a6cf99e637704da3ada04219ac7cd5727ff62284153761d36d7d3",
                "ModelDataUrl": MODEL_DATA_URL,
                "NearestModelName": NEAREST_MODEL_NAME,
            }
        ],
        "SupportedContentTypes": [],
        "SupportedResponseMIMETypes": [],
    },
    "ModelPackageStatus": MODEL_PACKAGE_STATUS,
    "ModelApprovalStatus": MODEL_APPROVAL_STATUS,
    "CreatedBy": {
        "UserProfileArn": "arn:aws:sagemaker:us-west-2:001234567890:user-profile/d-crvaptvnkhbq/test",
        "UserProfileName": USER_PROFILE_NAME,
        "DomainId": "d-crvaptvnkhbq",
    },
    "Domain": DOMAIN,
    "Task": TASK,
    "ResponseMetadata": {
        "RequestId": "43ff67e3-2ae5-479e-bed0-9caccc9e62f0",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "43ff67e3-2ae5-479e-bed0-9caccc9e62f0",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "590",
            "date": "Tue, 20 Sep 2022 19:55:36 GMT",
        },
        "RetryAttempts": 0,
    },
}

SEARCH_MODEL_CARD_WITH_MODEL_ID_EXAMPLE = {
    "Results": [
        {
            "ModelCard": {
                "ModelCardArn": MODEL_CARD_ARN,
                "ModelCardName": MODEL_CARD_NAME,
                "ModelCardVersion": MODEL_CARD_VERSION,
                "Content": json.dumps(
                    {
                        "intended_uses": {"risk_rating": RISK_RATING},
                        "model_overview": {
                            "model_id": MODEL_ID,
                            "model_name": MODEL_NAME,
                        },
                    }
                ),
                "ModelCardStatus": SIMPLE_MODEL_CARD_STATUS,
                "CreationTime": datetime.datetime(2022, 9, 15, 14, 26, 38),
                "CreatedBy": {},
                "LastModifiedTime": datetime.datetime(2022, 9, 15, 14, 26, 38),
                "LastModifiedBy": {},
                "Tags": [{"Key": "aws:tag:domain", "Value": "beta"}],
                "ModelId": MODEL_ID,
                "RiskRating": RISK_RATING,
            }
        }
    ],
    "ResponseMetadata": {
        "RequestId": "d8f68dea-8452-47e6-bb70-e543f7e1c0ba",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "d8f68dea-8452-47e6-bb70-e543f7e1c0ba",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "514",
            "date": "Tue, 27 Sep 2022 16:29:58 GMT",
        },
        "RetryAttempts": 0,
    },
}

SEARCH_MODEL_CARD_WITH_MODEL_ID_EMPTY_EXAMPLE = {
    "Results": [],
    "ResponseMetadata": {
        "RequestId": "d8f68dea-8452-47e6-bb70-e543f7e1c0ba",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "d8f68dea-8452-47e6-bb70-e543f7e1c0ba",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "514",
            "date": "Tue, 27 Sep 2022 16:29:58 GMT",
        },
        "RetryAttempts": 0,
    },
}

MISSING_MODEL_CLIENT_ERROR = ClientError(
    error_response={
        "Error": {
            "Code": "BadRequest",
            "Message": f"Could not find model {MODEL_NAME}",
        },
        "ResponseMetadata": {"MaxAttemptsReached": True, "RetryAttempts": 4},
    },
    operation_name="DescribeModel",
)

SEARCH_IAM_PERMISSION_CLIENT_ERROR = ClientError(
    error_response={
        "Error": {
            "Code": "AccessDeniedException",
            "Message": "An error occurred (AccessDenied) when calling the Search operation",
        },
        "ResponseMetadata": {"MaxAttemptsReached": True, "RetryAttempts": 4},
    },
    operation_name="Search",
)

MISSING_MODEL_PACKAGE_CLIENT_ERROR = ClientError(
    error_response={
        "Error": {
            "Code": "BadRequest",
            "Message": f"Could not find model package {MODEL_PACKAGE_ARN}",
        },
        "ResponseMetadata": {"MaxAttemptsReached": True, "RetryAttempts": 4},
    },
    operation_name="DescribeModelPackage",
)

TRAINING_JOB_NAME = MODEL_NAME
TRAINING_JOB_ARN = "test training job id"
SEARCH_TRAINING_JOB_EXAMPLE = {
    "Results": [
        {
            "TrainingJob": {
                "TrainingJobName": TRAINING_JOB_NAME,
                "TrainingJobArn": TRAINING_JOB_ARN,
                "ModelArtifacts": {"S3ModelArtifacts": "s3://example"},
                "TrainingJobOutput": {"S3TrainingJobOutput": "s3://example"},
                "TrainingJobStatus": "Completed",
                "SecondaryStatus": "Completed",
                "AlgorithmSpecification": {
                    "TrainingImage": TRAINING_IMAGE,
                    "TrainingInputMode": "File",
                    "MetricDefinitions": [],
                    "EnableSageMakerMetricsTimeSeries": False,
                },
                "FinalMetricDataList": [
                    {
                        "MetricName": "train:binary_f_beta",
                        "Value": 0.9652714133262634,
                        "Timestamp": datetime.datetime(2022, 9, 5, 19, 18, 42),
                    },
                    {
                        "MetricName": "train:progress",
                        "Value": 100.0,
                        "Timestamp": datetime.datetime(2022, 9, 5, 19, 18, 40),
                    },
                ],
                "HyperParameters": {
                    "_kfold": "5",
                    "_tuning_objective_metric": "validation:accuracy",
                    "alpha": "0.0037170512924477993",
                    "colsample_bytree": "0.7476726040667319",
                    "eta": "0.011391935592233605",
                    "eval_metric": "accuracy,f1,balanced_accuracy,precision_macro,recall_macro,mlogloss",
                    "gamma": "1.8903517751689445",
                    "lambda": "0.5098604662224621",
                    "max_depth": "3",
                    "min_child_weight": "5.081388147234708e-06",
                    "num_class": "28",
                    "num_round": "165",
                    "objective": "multi:softprob",
                    "subsample": "0.8828549481113146",
                },
                "CreatedBy": {},
            },
        }
    ],
    "ResponseMetadata": {
        "RequestId": "b43aacee-c846-4ca6-b5c0-d9413cebda33",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "b43aacee-c846-4ca6-b5c0-d9413cebda33",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "6199",
            "date": "Tue, 20 Sep 2022 19:59:50 GMT",
        },
        "RetryAttempts": 0,
    },
}

SEARCH_TRAINING_JOB_AUTOPILOT_EXAMPLE = {
    "Results": [
        {
            "TrainingJob": {
                "TrainingJobName": TRAINING_JOB_NAME,
                "TrainingJobArn": TRAINING_JOB_ARN,
                "ModelArtifacts": {"S3ModelArtifacts": "s3://example"},
                "TrainingJobOutput": {"S3TrainingJobOutput": "s3://example"},
                "TrainingJobStatus": "Completed",
                "SecondaryStatus": "Completed",
                "HyperParameters": {
                    "processor_module": "candidate_data_processors.dpp2",
                    "sagemaker_program": "candidate_data_processors.trainer",
                    "sagemaker_submit_directory": "/opt/ml/input/data/code",
                },
                "AlgorithmSpecification": {
                    "TrainingImage": TRAINING_IMAGE,
                    "TrainingInputMode": "File",
                    "MetricDefinitions": [],
                    "EnableSageMakerMetricsTimeSeries": False,
                },
                "CreatedBy": {},
            }
        }
    ],
    "ResponseMetadata": {
        "RequestId": "b43aacee-c846-4ca6-b5c0-d9413cebda33",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "b43aacee-c846-4ca6-b5c0-d9413cebda33",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "6199",
            "date": "Tue, 20 Sep 2022 19:59:50 GMT",
        },
        "RetryAttempts": 0,
    },
}

DESCRIBE_TRAINING_JOB_EXAMPLE = SEARCH_TRAINING_JOB_EXAMPLE["Results"][0]["TrainingJob"]
MISSING_TRAINING_JOB_CLIENT_ERROR = ClientError(
    error_response={
        "Error": {"Code": "BadRequest", "Message": "Requested resource not found."},
        "ResponseMetadata": {"MaxAttemptsReached": True, "RetryAttempts": 4},
    },
    operation_name="DescribeModel",
)

S3_MISSING_KEY_CLIENT_ERROR = ClientError(
    error_response={"Error": {"Code": "NoSchKey", "Message": "The specified key does not exist."}},
    operation_name="GetObject",
)

# Export model card
EXPORT_JOB_NAME = f"{MODEL_CARD_NAME}export"
EXPORT_JOB_ARN = f"arn:aws:sagemaker:us-west-2:1234567890:model-card/{MODEL_CARD_NAME}/export-job/{EXPORT_JOB_NAME}"
S3_URL = "s3://test"
CREATE_EXPORT_MODEL_CARD_EXAMPLE = {
    "ModelCardExportJobArn": EXPORT_JOB_ARN,
    "ResponseMetadata": {
        "RequestId": "511c724c-848e-4a3f-a0ef-cc9e774b391a",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "511c724c-848e-4a3f-a0ef-cc9e774b391a",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "110",
            "date": "Tue, 27 Sep 2022 18:31:54 GMT",
        },
        "RetryAttempts": 0,
    },
}
DESCRIBE_MODEL_CARD_EXPORT_JOB_IN_PROGRESS_EXAMPLE = {
    "ModelCardExportJobName": EXPORT_JOB_NAME,
    "ModelCardExportJobArn": EXPORT_JOB_ARN,
    "Status": "InProgress",
    "ModelCardName": MODEL_CARD_NAME,
    "ModelCardVersion": MODEL_CARD_VERSION,
    "OutputConfig": {"S3OutputPath": S3_URL},
    "CreatedAt": datetime.datetime(2022, 9, 27, 13, 31, 55, 544000),
    "LastModifiedAt": datetime.datetime(2022, 9, 27, 13, 31, 56, 748000),
    "ResponseMetadata": {
        "RequestId": "fcb81056-04f6-4b41-873e-1fa63eb19b9a",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "fcb81056-04f6-4b41-873e-1fa63eb19b9a",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "337",
            "date": "Tue, 27 Sep 2022 18:53:21 GMT",
        },
        "RetryAttempts": 0,
    },
}
DESCRIBE_MODEL_CARD_EXPORT_JOB_COMPLETED_EXAMPLE = {
    "ModelCardExportJobName": EXPORT_JOB_NAME,
    "ModelCardExportJobArn": EXPORT_JOB_ARN,
    "Status": "Completed",
    "ModelCardName": MODEL_CARD_NAME,
    "ModelCardVersion": MODEL_CARD_VERSION,
    "OutputConfig": {"S3OutputPath": S3_URL},
    "ExportArtifacts": {"S3ExportArtifacts": f"{S3_URL}/{MODEL_CARD_NAME}/{EXPORT_JOB_NAME}.pdf"},
    "CreatedAt": datetime.datetime(2022, 9, 27, 13, 31, 55, 544000),
    "LastModifiedAt": datetime.datetime(2022, 9, 27, 13, 31, 56, 748000),
    "ResponseMetadata": {
        "RequestId": "fcb81056-04f6-4b41-873e-1fa63eb19b9a",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "fcb81056-04f6-4b41-873e-1fa63eb19b9a",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "337",
            "date": "Tue, 27 Sep 2022 18:53:21 GMT",
        },
        "RetryAttempts": 0,
    },
}
MODEL_CARD_EXPORT_JOB_FAILURE_REASON = "example failure reason"
DESCRIBE_MODEL_CARD_EXPORT_JOB_FAILED_EXAMPLE = {
    "ModelCardExportJobName": EXPORT_JOB_NAME,
    "ModelCardExportJobArn": EXPORT_JOB_ARN,
    "Status": "Failed",
    "ModelCardName": MODEL_CARD_NAME,
    "ModelCardVersion": MODEL_CARD_VERSION,
    "OutputConfig": {"S3OutputPath": S3_URL},
    "CreatedAt": datetime.datetime(2022, 9, 27, 13, 31, 55, 544000),
    "LastModifiedAt": datetime.datetime(2022, 9, 27, 13, 31, 56, 748000),
    "FailureReason": MODEL_CARD_EXPORT_JOB_FAILURE_REASON,
    "ResponseMetadata": {
        "RequestId": "fcb81056-04f6-4b41-873e-1fa63eb19b9a",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "fcb81056-04f6-4b41-873e-1fa63eb19b9a",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "337",
            "date": "Tue, 27 Sep 2022 18:53:21 GMT",
        },
        "RetryAttempts": 0,
    },
}
LIST_MODEL_CARD_EXPORT_JOB_EXAMPLE = {
    "ModelCardExportJobSummaries": [
        {
            "ModelCardExportJobName": EXPORT_JOB_NAME,
            "ModelCardExportJobArn": EXPORT_JOB_ARN,
            "Status": "Completed",
            "ModelCardName": MODEL_CARD_NAME,
            "ModelCardVersion": 0,
            "CreatedAt": datetime.datetime(2022, 9, 27, 16, 2, 46, 496000),
            "LastModifiedAt": datetime.datetime(2022, 9, 27, 16, 2, 52, 867000),
        }
    ],
    "ResponseMetadata": {
        "RequestId": "dfeddeda-83a5-4973-b39a-c50c162e7f6f",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "dfeddeda-83a5-4973-b39a-c50c162e7f6f",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "1442",
            "date": "Thu, 29 Sep 2022 18:36:25 GMT",
        },
        "RetryAttempts": 0,
    },
}
LIST_MODEL_CARD_VERSION_HISTORY_EXAMPLE = {
    "ModelCardVersionSummaryList": [
        {
            "ModelCardName": MODEL_CARD_NAME,
            "ModelCardArn": MODEL_CARD_ARN,
            "ModelCardStatus": MODEL_CARD_STATUS,
            "ModelCardVersion": MODEL_CARD_VERSION,
            "CreationTime": datetime.datetime(2022, 9, 27, 9, 8, 42, 164000),
            "LastModifiedTime": datetime.datetime(2022, 9, 27, 9, 8, 42, 164000),
        }
    ],
    "ResponseMetadata": {
        "RequestId": "abe7468d-6092-47f2-8c73-f627508ba407",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "abe7468d-6092-47f2-8c73-f627508ba407",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "251",
            "date": "Thu, 29 Sep 2022 19:37:38 GMT",
        },
        "RetryAttempts": 0,
    },
}
RESPONSE_CONTENT_EXAMPLE = {
    "intended_uses": {"purpose_of_model": PURPOSE_OF_MODEL},
    "business_details": {
        "business_problem": BUSINESS_PROBLEM,
        "business_stakeholders": BUSINESS_STAKEHOLDERS,
        "line_of_business": LINE_OF_BUSINESS,
    },
    "additional_information": {
        "ethical_considerations": ETHICAL_CONSIDERATIONS,
        "caveats_and_recommendations": CAVEATS_AND_RECOMMENDATIONS,
        "custom_details": CUSTOM_DETAILS,
    },
}
ORIGINAL_MOCK_STRING = "Original mock string."
SEARCH_LATEST_MODEL_CARD_EXAMPLE = {
    "Results": [
        {
            "ModelCard": {
                "ModelCardArn": MODEL_CARD_ARN_FOR_CARRY_OVER_ADDITIONAL_CONTENT,
                "ModelCardName": MODEL_CARD_NAME_FOR_CARRY_OVER_ADDITIONAL_CONTENT,
                "ModelCardVersion": MODEL_CARD_VERSION,
                "Content": {
                    "model_overview": {"model_id": MODEL_PACKAGE_ARN},
                    "intended_uses": {
                        "purpose_of_model": PURPOSE_OF_MODEL,
                        "risk_rating": RISK_RATING,
                        "factors_affecting_model_efficiency": FACTORS_AFFECTING_MODEL_EFFICIENCY,
                    },
                    "business_details": {
                        "business_problem": BUSINESS_PROBLEM,
                        "business_stakeholders": BUSINESS_STAKEHOLDERS,
                        "line_of_business": LINE_OF_BUSINESS,
                    },
                    "additional_information": {
                        "ethical_considerations": ETHICAL_CONSIDERATIONS,
                        "caveats_and_recommendations": CAVEATS_AND_RECOMMENDATIONS,
                        "custom_details": CUSTOM_DETAILS,
                    },
                },
                "ModelCardStatus": MODEL_CARD_STATUS,
                "CreationTime": datetime.datetime(2023, 4, 4, 15, 30, 3),
                "CreatedBy": {},
                "LastModifiedTime": datetime.datetime(2023, 4, 4, 15, 30, 3),
                "LastModifiedBy": {},
                "Tags": [],
                "ModelId": MODEL_PACKAGE_ARN,
            }
        },
    ],
    "ResponseMetadata": {
        "RequestId": "678b50e9-23a5-4ed4-a530-e0635e0fcffd",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "678b50e9-23a5-4ed4-a530-e0635e0fcffd",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "1523",
            "date": "Wed, 19 Apr 2023 03:32:28 GMT",
        },
        "RetryAttempts": 0,
    },
}
SEARCH_LATEST_MODEL_CARD_WITH_EMPTY_RESULT_EXAMPLE = {
    "Results": [],
    "ResponseMetadata": {
        "RequestId": "678b50e9-23a5-4ed4-a530-e0635e0fcffd",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "678b50e9-23a5-4ed4-a530-e0635e0fcffd",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "1523",
            "date": "Wed, 19 Apr 2023 03:32:28 GMT",
        },
        "RetryAttempts": 0,
    },
}
CONTENT_FROM_DESCRIBE_MODEL_CARD = {
    "model_overview": {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
    },
    "intended_uses": {
        "purpose_of_model": PURPOSE_OF_MODEL,
        "intended_uses": INTENDED_USES,
        "factors_affecting_model_efficiency": FACTORS_AFFECTING_MODEL_EFFICIENCY,
        "risk_rating": RISK_RATING,
        "explanations_for_risk_rating": EXPLANATIONS_FOR_RISK_RATING,
    },
    "business_details": {
        "business_problem": BUSINESS_PROBLEM,
        "business_stakeholders": BUSINESS_STAKEHOLDERS,
        "line_of_business": LINE_OF_BUSINESS,
    },
    "additional_information": {
        "ethical_considerations": ETHICAL_CONSIDERATIONS,
        "caveats_and_recommendations": CAVEATS_AND_RECOMMENDATIONS,
        "custom_details": CUSTOM_DETAILS,
    },
    "model_package_details": {
        "model_package_arn": MODEL_PACKAGE_ARN,
        "model_package_name": MODEL_PACKAGE_NAME,
        "model_package_group_name": MODEL_PACKAGE_GROUP_NAME,
        "model_package_version": MODEL_PACKAGE_VERSION,
        "model_package_description": MODEL_PACKAGE_DESCRIPTION,
        "inference_specification": {
            "containers": [
                {
                    "image": MODEL_IMAGE,
                    "model_data_url": MODEL_DATA_URL,
                    "nearest_model_name": NEAREST_MODEL_NAME,
                }
            ]
        },
        "model_package_status": MODEL_PACKAGE_STATUS,
        "model_approval_status": MODEL_APPROVAL_STATUS,
        "approval_description": APPROVAL_DESCRIPTION,
        "created_by": {
            "user_profile_name": USER_PROFILE_NAME,
        },
        "domain": DOMAIN,
        "task": TASK,
        "source_algorithms": [{"algorithm_name": ALGORITHM_NAME, "model_data_url": MODEL_DATA_URL}],
    },
}
DESCRIBE_MODEL_CARD_WITH_ADDITONAL_CONTENT = {
    "ModelCardArn": MODEL_CARD_ARN_FOR_CARRY_OVER_ADDITIONAL_CONTENT,
    "ModelCardName": MODEL_CARD_NAME_FOR_CARRY_OVER_ADDITIONAL_CONTENT,
    "ModelCardVersion": MODEL_CARD_VERSION,
    "Content": json.dumps(CONTENT_FROM_DESCRIBE_MODEL_CARD),
    "ModelCardStatus": MODEL_CARD_STATUS,
    "CreationTime": datetime.datetime(2022, 9, 17, 17, 15, 45, 672000),
    "CreatedBy": {},
    "LastModifiedTime": datetime.datetime(2022, 9, 17, 17, 15, 45, 672000),
    "LastModifiedBy": {},
    "ResponseMetadata": {
        "RequestId": "7f317f47-a1e5-45dc-975a-fa4d9df81365",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "7f317f47-a1e5-45dc-975a-fa4d9df81365",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "2429",
            "date": "Mon, 19 Sep 2022 21:09:05 GMT",
        },
        "RetryAttempts": 0,
    },
}


@pytest.fixture(name="model_overview_example")
def fixture_model_overview_example():
    """Example model overview instance."""
    test_example = ModelOverview(
        model_id=MODEL_ID,
        model_name=MODEL_NAME,
        model_description=MODEL_DESCRIPTION,
        model_version=MODEL_VERSION,
        problem_type=PROBLEM_TYPE,
        algorithm_type=ALGORITHM_TYPE,
        model_creator=MODEL_CREATOR,
        model_owner=MODEL_OWNER,
        model_artifact=MODEL_ARTIFACT,
        inference_environment=INFERENCE_ENVRIONMENT,
    )
    return test_example


@pytest.fixture(name="model_package_example")
def fixture_model_package_example():
    """Example ModelPackage instance"""
    test_example = ModelPackage(
        model_package_arn=MODEL_PACKAGE_ARN,
        model_package_description=MODEL_PACKAGE_DESCRIPTION,
        model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
        model_package_name=MODEL_PACKAGE_NAME,
        model_approval_status=MODEL_APPROVAL_STATUS,
        model_package_status=MODEL_PACKAGE_STATUS,
        model_package_version=MODEL_PACKAGE_VERSION,
        approval_description=APPROVAL_DESCRIPTION,
        domain=DOMAIN,
        task=TASK,
        created_by=CREATED_BY,
        source_algorithms=SOURCE_ALGORITHMS,
        inference_specification=INFERENCE_SPECIFICATION,
    )
    return test_example


@pytest.fixture(name="intended_uses_example")
def fixture_fixture_intended_uses_example():
    """Example intended uses instance."""
    test_example = IntendedUses(
        purpose_of_model=PURPOSE_OF_MODEL,
        intended_uses=INTENDED_USES,
        factors_affecting_model_efficiency=FACTORS_AFFECTING_MODEL_EFFICIENCY,
        risk_rating=RISK_RATING,
        explanations_for_risk_rating=EXPLANATIONS_FOR_RISK_RATING,
    )
    return test_example


@pytest.fixture(name="business_details_example")
def fixture_fixture_business_details_example():
    """Example business details instance."""
    test_example = BusinessDetails(
        business_problem=BUSINESS_PROBLEM,
        business_stakeholders=BUSINESS_STAKEHOLDERS,
        line_of_business=LINE_OF_BUSINESS,
    )
    return test_example


@pytest.fixture(name="training_details_example")
def fixture_fixture_training_details_example():
    """Example training details instance."""
    test_example = TrainingDetails(
        objective_function=ObjectiveFunction(
            function=Function(
                function=OBJECITVE_FUNCTION_FUNC,
                facet=OBJECTIVE_FUNCTION_FACET,
                condition=OBJECTIVE_FUNCTION_CONDITION,
            ),
            notes=OBJECTIVE_FUNCTION_NOTES,
        ),
        training_observations=TRAINING_OBSERVATIONS,
        training_job_details=TrainingJobDetails(
            training_arn=TRAINING_ARN,
            training_datasets=TRAINING_DATASETS,
            training_environment=TRAINING_ENVIRONMENT,
            training_metrics=TRAINING_METRICS,
            hyper_parameters=HYPER_PARAMETER,
        ),
    )
    return test_example


@pytest.fixture(name="evaluation_details_example")
def fixture_evaluation_details_example():
    """Example evaluation details instance."""
    test_example = [
        EvaluationJob(
            name=EVALUATION_JOB_NAME,
            evaluation_observation=EVALUATION_OBSERVATION,
            datasets=EVALUATION_DATASETS,
            metadata=EVALUATION_METADATA,
            metric_groups=EVALUATION_METRIC_GROUPS,
        )
    ]
    return test_example


@pytest.fixture(name="additional_information_example")
def fixture_additional_information_example():
    """Example additional information instance."""
    test_example = AdditionalInformation(
        ethical_considerations=ETHICAL_CONSIDERATIONS,
        caveats_and_recommendations=CAVEATS_AND_RECOMMENDATIONS,
        custom_details=CUSTOM_DETAILS,
    )
    return test_example


@patch("sagemaker.Session")
def test_create_model_card(
    session,
    model_overview_example,
    intended_uses_example,
    business_details_example,
    training_details_example,
    evaluation_details_example,
    additional_information_example,
):
    session.sagemaker_client.create_model_card = Mock(return_value=CREATE_MODEL_CARD_RETURN_EXAMPLE)
    session.sagemaker_client.describe_model_card = Mock(return_value=LOAD_MODEL_CARD_EXMPLE)

    card = ModelCard(
        name=MODEL_CARD_NAME,
        status=MODEL_CARD_STATUS,
        model_overview=model_overview_example,
        intended_uses=intended_uses_example,
        business_details=business_details_example,
        training_details=training_details_example,
        evaluation_details=evaluation_details_example,
        additional_information=additional_information_example,
        sagemaker_session=session,
    )

    card.create()

    assert card.arn == MODEL_CARD_ARN


@patch("sagemaker.Session")
def test_create_model_card_with_model_package(
    session, model_package_example, training_details_example, caplog
):
    session.sagemaker_client.create_model_card = Mock(return_value=CREATE_MODEL_CARD_RETURN_EXAMPLE)
    session.sagemaker_client.describe_model_card = Mock(
        return_value=MODEL_CARD_WITH_MODEL_PACKAGE_MOCK_RESPONSE
    )

    session.sagemaker_client.search.side_effect = [
        SEARCH_TRAINING_JOB_EXAMPLE,
        SEARCH_LATEST_MODEL_CARD_WITH_EMPTY_RESULT_EXAMPLE,
        SEARCH_LATEST_MODEL_CARD_WITH_EMPTY_RESULT_EXAMPLE,
    ]

    card = ModelCard(
        name=MODEL_CARD_NAME,
        status=MODEL_CARD_STATUS,
        model_package_details=model_package_example,
        sagemaker_session=session,
    )

    card.create()

    assert card.arn == MODEL_CARD_ARN
    assert card.status == MODEL_CARD_STATUS
    assert card.model_package_details.model_package_arn == MODEL_PACKAGE_ARN
    assert card.model_package_details.model_approval_status == MODEL_APPROVAL_STATUS
    assert card.model_package_details.created_by.user_profile_name == USER_PROFILE_NAME

    # testing with existing training details
    with caplog.at_level(logging.INFO):
        ModelCard(
            name=MODEL_CARD_NAME,
            status=MODEL_CARD_STATUS,
            training_details=training_details_example,
            model_package_details=model_package_example,
            sagemaker_session=session,
        )
        assert (
            "Skipping training details auto discovery. "
            "Training details already exists for this model card."
        ) in caplog.text


@patch("sagemaker.Session")
def test_create_model_card_with_multiple_models(
    session, model_package_example, model_overview_example
):

    card = ModelCard(
        name=MODEL_CARD_NAME,
        status=MODEL_CARD_STATUS,
        model_overview=model_overview_example,
        sagemaker_session=session,
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The model card has already been associated with a model with model Id {MODEL_ID}"  # noqa E501  # pylint: disable=c0301
        ),
    ):
        card.model_package_details = model_package_example


@patch("sagemaker.Session")
def test_create_model_card_duplicate(session):
    session.sagemaker_client.create_model_card.side_effect = [
        CREATE_MODEL_CARD_RETURN_EXAMPLE,
        GENERAL_CLIENT_ERROR,
    ]

    session.sagemaker_client.describe_model_card.side_effect = [
        LOAD_MODEL_CARD_EXMPLE,
    ]

    card1 = ModelCard(name=MODEL_CARD_NAME, sagemaker_session=session)
    card1.create()
    assert card1.arn == MODEL_CARD_ARN

    with pytest.raises(ClientError):
        card2 = ModelCard(name=MODEL_CARD_NAME, sagemaker_session=session)
        card2.create()


@patch("sagemaker.Session")
def test_create_multiple_model_cards_with_same_model(session, model_overview_example):
    session.sagemaker_client.create_model_card.side_effect = [
        CREATE_SIMPLE_MODEL_CARD_RETURN_EXAMPLE,
        GENERAL_CLIENT_ERROR,
    ]
    session.sagemaker_client.describe_model_card.side_effect = [
        LOAD_MODEL_CARD_EXMPLE,
    ]

    card1 = ModelCard(
        name="test1", model_overview=model_overview_example, sagemaker_session=session
    )
    card1.create()
    assert card1.arn == MODEL_CARD_ARN

    with pytest.raises(ClientError):
        card2 = ModelCard(
            name="test2",
            model_overview=model_overview_example,
            sagemaker_session=session,
        )
        card2.create()


@patch("sagemaker.Session")
def test_create_model_card_with_too_long_string(session, model_overview_example):
    too_long_string_client_error = ClientError(
        error_response={
            "Error": {
                "Code": "ValidationException",
                "Message": (
                    "The document is not valid evaluation_details.0.evaluation_job_arn: "
                    "String length must be less than or equal to 1024"
                ),
            },
            "ResponseMetadata": {"MaxAttemptsReached": False},
        },
        operation_name="CreateModelCard",
    )

    session.sagemaker_client.create_model_card.side_effect = too_long_string_client_error

    with pytest.raises(ClientError):
        model_overview_example.name = "x" * 1025
        card = ModelCard(
            name=MODEL_CARD_NAME,
            model_overview=model_overview_example,
            sagemaker_session=session,
        )
        card.create()


@patch("sagemaker.Session")
def test_carry_over_additional_content_from_model_package_group(session, model_package_example):
    session.sagemaker_client.describe_model_card = Mock(
        return_value=DESCRIBE_MODEL_CARD_WITH_ADDITONAL_CONTENT
    )

    session.sagemaker_client.search.side_effect = [
        SEARCH_TRAINING_JOB_EXAMPLE,
        SEARCH_LATEST_MODEL_CARD_EXAMPLE,
    ]

    mc = ModelCard(
        name=MODEL_CARD_NAME,
        status=MODEL_CARD_STATUS,
        sagemaker_session=session,
        model_package_details=model_package_example,
        business_details={
            "business_problem": ORIGINAL_MOCK_STRING,
            "business_stakeholders": ORIGINAL_MOCK_STRING,
        },
    )

    assert mc.intended_uses.purpose_of_model == PURPOSE_OF_MODEL
    assert mc.intended_uses.risk_rating == RISK_RATING
    assert mc.intended_uses.factors_affecting_model_efficiency == FACTORS_AFFECTING_MODEL_EFFICIENCY

    assert mc.business_details.business_problem == ORIGINAL_MOCK_STRING
    assert mc.business_details.business_stakeholders == ORIGINAL_MOCK_STRING

    assert mc.additional_information.ethical_considerations == ETHICAL_CONSIDERATIONS
    assert mc.additional_information.caveats_and_recommendations == CAVEATS_AND_RECOMMENDATIONS
    assert mc.additional_information.custom_details == CUSTOM_DETAILS


@pytest.mark.skip(
    "temporary skip until error pattern is updated for py311 number|MetricTypeEnum.NUMBER"
)
def test_metric_type_value_mismatch():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "One of type [<class 'int'>, <class 'float'>] is expected for metric type number"
        ),
    ):
        Metric(
            name="test_training_metric",
            type=schema_constraints.MetricTypeEnum.NUMBER,
            value="123",
        )


def test_max_size_list():
    assert _MaxSizeArray(5, int, [1, 2, 3])

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Provided item type is <class 'str'> and Expected the item type is <class 'int'>"
        ),
    ):
        _MaxSizeArray(5, int, ["123"])

    with pytest.raises(ValueError, match=re.escape("Data size 6 exceed the maximum size of 5")):
        _MaxSizeArray(5, int, [1, 2, 3, 4, 5, 6])

    with pytest.raises(ValueError, match=re.escape("Max size has to be positive integer")):
        _MaxSizeArray(-1, int, [1, 2, 3, 4, 5, 6])

    with pytest.raises(ValueError, match=re.escape("Item type has to be a class")):
        _MaxSizeArray(5, 1, [1, 2, 3, 4, 5, 6])

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Provided item type is <class 'str'> and Expected the item type is <class 'int'>"
        ),
    ):
        array = _MaxSizeArray(5, int, [1])
        array.append("a")

    with pytest.raises(ValueError, match=re.escape("Exceed the maximum size of 5")):
        array = _MaxSizeArray(5, int, [1, 2, 3, 4, 5])
        array.append(6)


def test_enumerator_attributes():
    assert ModelCard(name=MODEL_CARD_NAME, status="Draft")
    assert ModelCard(name=MODEL_CARD_NAME, status=schema_constraints.ModelCardStatusEnum.DRAFT)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Expected NewStatus to be one of ['Approved', 'Archived', 'Draft', 'PendingReview']"
        ),
    ):
        ModelCard(name=MODEL_CARD_NAME, status="NewStatus")

    with pytest.raises(
        ValueError,
        match=re.escape(
            (
                "Expected MetricTypeEnum.BOOLEAN to be one of ['ModelCardStatusEnum.APPROVED', "
                "'ModelCardStatusEnum.ARCHIVED', 'ModelCardStatusEnum.DRAFT', "
                "'ModelCardStatusEnum.PENDING_REVIEW']"
            )
        ),
    ):
        ModelCard(name=MODEL_CARD_NAME, status=schema_constraints.MetricTypeEnum.BOOLEAN)


def test_is_list_descriptor():
    class ExampleClass:  # pylint: disable=C0115
        attr1 = _IsList(item_type=str)

        def __init__(self, attr1):  # pylint: disable=C0116
            self.attr1 = attr1

    assert ExampleClass(["1", "2", "3"])

    with pytest.raises(
        ValueError,
        match=re.escape("Please assign a list to attr1"),
    ):
        ExampleClass(attr1="test1")


def test_one_of_descriptor():
    class ExampleClass:  # pylint: disable=C0115
        attr1 = _OneOf(schema_constraints.ModelCardStatusEnum)

        def __init__(self, attr1):  # pylint: disable=C0116
            self.attr1 = attr1

    assert ExampleClass(schema_constraints.ModelCardStatusEnum.DRAFT)

    with pytest.raises(
        ValueError,
        match=re.escape(
            (
                "Expected test_model_card_status to be one of "
                "['Approved', 'Archived', 'Draft', 'PendingReview']"
            )
        ),
    ):
        ExampleClass(attr1="test_model_card_status")

    with pytest.raises(
        ValueError,
        match=re.escape(
            (
                "Expected MetricTypeEnum.BAR_CHART to be one of "
                "['ModelCardStatusEnum.APPROVED', 'ModelCardStatusEnum.ARCHIVED', "
                "'ModelCardStatusEnum.DRAFT', 'ModelCardStatusEnum.PENDING_REVIEW']"
            )
        ),
    ):
        ExampleClass(attr1=schema_constraints.MetricTypeEnum.BAR_CHART)


def test_skip_encoding_descriptor():
    class ExampleClass:
        attr1 = _SkipEncodingDecoding(dict)

        def __init__(self, attr1):
            self.attr1 = attr1

    assert ExampleClass({"test": 1})

    with pytest.raises(
        ValueError,
        match=re.escape("Please assign a <class 'dict'> to attr1"),
    ):
        ExampleClass(attr1="test1")


def test_is_model_card_object_descriptor():
    class ExampleClass:  # pylint: disable=C0115
        attr1 = _IsModelCardObject(ModelOverview)

        def __init__(self, attr1):  # pylint: disable=C0116
            self.attr1 = attr1

    assert ExampleClass(ModelOverview())

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Expected <class 'sagemaker.model_card.model_card.IntendedUses'> instance to be of class ModelOverview"  # noqa E501  # pylint: disable=c0301
        ),
    ):
        ExampleClass(attr1=IntendedUses())

    # decode object from json data
    assert ExampleClass({"model_name": "test"})

    with pytest.raises(TypeError) as exception:
        ExampleClass({"test_attr": "test_val"})
    assert exception.value.args[0].endswith(
        "__init__() got an unexpected keyword argument 'test_attr'"
    )


@patch("sagemaker.Session")
def test_load_model_card(session):
    session.sagemaker_client.describe_model_card = Mock(return_value=LOAD_SIMPLE_MODEL_CARD_EXMPLE)
    card = ModelCard.load(name=SIMPLE_MODEL_CARD_NAME, sagemaker_session=session)

    assert card.status == SIMPLE_MODEL_CARD_STATUS
    assert not card.model_overview
    assert card.additional_information


@patch("sagemaker.Session")
def test_update_model_card(
    session,
    additional_information_example,
):
    session.sagemaker_client.create_model_card = Mock(
        return_value=CREATE_SIMPLE_MODEL_CARD_RETURN_EXAMPLE
    )
    session.sagemaker_client.describe_model_card = Mock(return_value=LOAD_SIMPLE_MODEL_CARD_EXMPLE)
    session.sagemaker_client.update_model_card = Mock(return_value=UPDATE_SIMPLE_MODEL_CARD_EXAMPLE)

    card = ModelCard(
        name=SIMPLE_MODEL_CARD_NAME,
        status=schema_constraints.ModelCardStatusEnum.DRAFT,
        additional_information=additional_information_example,
        sagemaker_session=session,
    )

    card.create()

    card.status = schema_constraints.ModelCardStatusEnum.APPROVED
    card.additional_information.ethical_considerations = "new ethical considerations"
    assert card.update()


@patch("sagemaker.Session")
def test_delete_model_card(session):
    session.sagemaker_client.describe_model_card = Mock(return_value=LOAD_SIMPLE_MODEL_CARD_EXMPLE)
    session.sagemaker_client.delete_model_card = Mock(
        return_value=DELETE_SIMPLE_MODEL_CARD_RETURN_EXAMPLE
    )

    card = ModelCard.load(name=SIMPLE_MODEL_CARD_NAME, sagemaker_session=session)

    assert card.arn == SIMPLE_MODEL_CARD_ARN
    assert card.delete()


def test_model_card_encoder():
    class ExampleClass(object):
        """Example class"""

        attr1 = _OneOf(schema_constraints.MetricTypeEnum)

        def __init__(self, attr1: schema_constraints.MetricTypeEnum):
            """Initialize an example class"""
            self.attr1 = attr1

        def _to_request_dict(self):
            """example method to return the _to_request_dict"""
            return {"attr1": self.attr1, "attr2": "test"}

    my_object = ExampleClass(attr1=schema_constraints.MetricTypeEnum.LINEAR_GRAPH)

    assert (
        json.dumps(my_object, cls=_JSONEncoder, sort_keys=True)
        == '{"attr1": "linear_graph", "attr2": "test"}'
    )


def test_model_card_encoder_with_skip_encoding():
    class ExampleClass(_DefaultToRequestDict):
        """Example class"""

        attr1 = _OneOf(schema_constraints.MetricTypeEnum)
        attr2 = _SkipEncodingDecoding(dict)

        def __init__(self, attr1: schema_constraints.MetricTypeEnum, attr2: dict):
            """Initialize an example class"""
            self.attr1 = attr1
            self.attr2 = attr2

    my_object = ExampleClass(
        attr1=schema_constraints.MetricTypeEnum.LINEAR_GRAPH, attr2={"test": 1}
    )

    assert json.dumps(my_object, cls=_JSONEncoder, sort_keys=True) == '{"attr1": "linear_graph"}'


def test_hash_content_str():
    content1 = json.dumps({"key": "value"})
    content2 = json.dumps({"key": "value2"})
    content3 = json.dumps({"key": "value"})

    assert _hash_content_str(content1) == _hash_content_str(content3)
    assert _hash_content_str(content1) != _hash_content_str(content2)


@patch("sagemaker.Session")
def test_model_details_autodiscovery(session):
    session.sagemaker_client.describe_model.side_effect = [
        DESCRIBE_MODEL_EXAMPLE,
        DESCRIBE_MODEL_EXAMPLE,
        MISSING_MODEL_CLIENT_ERROR,
    ]

    session.sagemaker_client.search.side_effect = [
        SEARCH_MODEL_CARD_WITH_MODEL_ID_EMPTY_EXAMPLE,
        SEARCH_MODEL_CARD_WITH_MODEL_ID_EXAMPLE,
    ]

    model = ModelOverview.from_model_name(MODEL_NAME, sagemaker_session=session)
    assert model.model_name == MODEL_NAME
    assert model.model_id == MODEL_ID
    assert model.inference_environment.container_image == [MODEL_IMAGE]
    assert model.model_artifact == [MODEL_ARTIFACT[0]]

    with pytest.raises(
        ValueError,
        match=re.escape(f"The model has been associated with {[MODEL_CARD_NAME]} model cards."),
    ):
        ModelOverview.from_model_name(MODEL_NAME, sagemaker_session=session)

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Model details for model {MODEL_NAME} could not be found. Make sure the model name is valid."  # noqa E501  # pylint: disable=c0301
        ),
    ):
        ModelOverview.from_model_name(MODEL_NAME, sagemaker_session=session)


@patch("sagemaker.Session")
def test_model_package_autodiscovery(session, model_overview_example, training_details_example):
    session.sagemaker_client.describe_model_package.side_effect = [
        DESCRIBE_MODEL_PACKAGE_EXAMPLE,
        DESCRIBE_MODEL_PACKAGE_EXAMPLE,
        MISSING_MODEL_PACKAGE_CLIENT_ERROR,
        DESCRIBE_MODEL_PACKAGE_EXAMPLE,
        DESCRIBE_MODEL_PACKAGE_EXAMPLE,
    ]

    session.sagemaker_client.search.side_effect = [
        SEARCH_MODEL_CARD_WITH_MODEL_ID_EMPTY_EXAMPLE,
        SEARCH_MODEL_CARD_WITH_MODEL_ID_EXAMPLE,
        SEARCH_IAM_PERMISSION_CLIENT_ERROR,
        SEARCH_MODEL_CARD_WITH_MODEL_ID_EMPTY_EXAMPLE,
    ]

    model_package_details = ModelPackage.from_model_package_arn(
        MODEL_PACKAGE_ARN, sagemaker_session=session
    )
    assert model_package_details.model_package_arn == MODEL_PACKAGE_ARN
    assert model_package_details.model_package_group_name == MODEL_PACKAGE_GROUP_NAME
    assert (
        model_package_details.inference_specification.containers[0].model_data_url == MODEL_DATA_URL
    )
    assert model_package_details.created_by.user_profile_name == USER_PROFILE_NAME

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The model package has already been associated with {[MODEL_CARD_NAME]} model cards."
        ),
    ):
        ModelPackage.from_model_package_arn(MODEL_PACKAGE_ARN, sagemaker_session=session)

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Model package details for {MODEL_PACKAGE_ARN} could not be found. Make sure the model package name or ARN is valid."  # noqa E501  # pylint: disable=c0301
        ),
    ):
        ModelPackage.from_model_package_arn(MODEL_PACKAGE_ARN, sagemaker_session=session)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Received AccessDeniedException while calling SageMaker Search operation "
            "on resource ModelCard. This could mean the IAM role does not "
            "have the resource permissions, in which case please add resource access "
            "and retry. For cases where the role has tag based resource policy, "
            "continuing to wait for tag propagation.."
        ),
    ):
        ModelPackage.from_model_package_arn(MODEL_PACKAGE_ARN, sagemaker_session=session)


@patch("sagemaker.Session")
def test_training_details_autodiscovery_from_model_overview(
    session, model_overview_example, caplog
):
    session.sagemaker_client.search.side_effect = [
        SEARCH_TRAINING_JOB_EXAMPLE,
        SEARCH_IAM_PERMISSION_CLIENT_ERROR,
    ]

    TrainingDetails.from_model_overview(
        model_overview=model_overview_example, sagemaker_session=session
    )
    assert (
        "TrainingJobDetails auto-discovery failed. "
        "There are 2 associated training jobs. "
        "Further clarification is required. "
        "You could use TrainingDetails.training_job_name after "
        "which training job to use is decided."
    ) in caplog.text

    model_overview_example.model_artifact = [MODEL_ARTIFACT[0]]
    training_details = TrainingDetails.from_model_overview(
        model_overview=model_overview_example, sagemaker_session=session
    )
    assert training_details.training_job_details.training_arn == TRAINING_JOB_ARN
    assert len(training_details.training_job_details.training_metrics) == len(
        SEARCH_TRAINING_JOB_EXAMPLE["Results"][0]["TrainingJob"]["FinalMetricDataList"]
    )
    assert len(training_details.training_job_details.hyper_parameters) == len(
        SEARCH_TRAINING_JOB_EXAMPLE["Results"][0]["TrainingJob"]["HyperParameters"]
    )
    assert training_details.training_job_details.training_environment.container_image == [
        TRAINING_IMAGE
    ]

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Received AccessDeniedException while calling SageMaker Search operation "
            "on resource TrainingJob. This could mean the IAM role does not "
            "have the resource permissions, in which case please add resource access "
            "and retry. For cases where the role has tag based resource policy, "
            "continuing to wait for tag propagation.."
        ),
    ):
        TrainingDetails.from_model_overview(
            model_overview=model_overview_example, sagemaker_session=session
        )

    model_overview_example.model_artifact = []
    TrainingDetails.from_model_overview(
        model_overview=model_overview_example, sagemaker_session=session
    )
    assert (
        "TrainingJobDetails auto-discovery failed. "
        "No associated training job. "
        "Please create one from scratch with TrainingJobDetails "
        "or use from_training_job_name() instead."
    ) in caplog.text


@patch("sagemaker.Session")
def test_training_details_autodiscovery_from_model_package_details(
    session, model_package_example, caplog
):
    session.sagemaker_client.search.side_effect = [
        SEARCH_TRAINING_JOB_EXAMPLE,
    ]

    training_details = model_package_example.discover_training_details(sagemaker_session=session)
    assert training_details.training_job_details.training_arn == TRAINING_JOB_ARN
    assert len(training_details.training_job_details.training_metrics) == len(
        SEARCH_TRAINING_JOB_EXAMPLE["Results"][0]["TrainingJob"]["FinalMetricDataList"]
    )
    assert len(training_details.training_job_details.hyper_parameters) == len(
        SEARCH_TRAINING_JOB_EXAMPLE["Results"][0]["TrainingJob"]["HyperParameters"]
    )
    assert training_details.training_job_details.training_environment.container_image == [
        TRAINING_IMAGE
    ]

    model_package_example.inference_specification.containers = []
    model_package_example.discover_training_details(sagemaker_session=session)
    assert (
        "TrainingJobDetails auto-discovery failed. "
        "No associated training job. "
        "Please create one from scratch with TrainingJobDetails "
        "or use from_training_job_name() instead."
    ) in caplog.text

    model_package_example.inference_specification = None
    with caplog.at_level(logging.INFO):
        model_package_example.discover_training_details(sagemaker_session=session)
        assert (
            "TrainingJobDetails auto-discovery was unsuccessful. "
            "No inference specification found for the given model package."
            "Please create one from scratch with TrainingJobDetails "
            "or use from_training_job_name() instead."
        ) in caplog.text


@patch("sagemaker.Session")
def test_evaluation_details_autodiscovery_from_model_package_details(
    session, model_package_example, caplog
):
    with open(CLARIFY_BIAS_JSON_PATH, "r", encoding="utf-8") as istr:
        data = json.dumps(json.load(istr))
        response = {
            "Body": botocore.response.StreamingBody(
                io.BytesIO(bytes(data, "utf-8")), content_length=len(data)
            ),
            "ContentType": "application/json",
        }
    session.boto_session.client.return_value.get_object.side_effect = [
        response,
    ]

    with caplog.at_level(logging.INFO):
        evaluation_details = model_package_example.discover_evaluation_details(
            sagemaker_session=session
        )
        assert (
            "Evaluation details auto-discovery was unsuccessful. "
            "ModelMetrics was not found in the given model package. "
            "Please create one from scratch with EvaluationJob."
        ) in caplog.text

    model_package_example.model_metrics = MODEL_METRICS
    evaluation_details = model_package_example.discover_evaluation_details(
        sagemaker_session=session
    )

    assert len(evaluation_details[0].metric_groups) == 3


@patch("sagemaker.Session")
def test_training_details_autodiscovery_from_model_overview_autopilot(
    session, model_overview_example, caplog
):
    session.sagemaker_client.search.side_effect = [
        SEARCH_TRAINING_JOB_AUTOPILOT_EXAMPLE,
    ]

    model_overview_example.model_artifact = [MODEL_ARTIFACT[0]]
    training_details = TrainingDetails.from_model_overview(
        model_overview=model_overview_example, sagemaker_session=session
    )

    # MetricDefinitions is empty
    assert len(training_details.training_job_details.training_metrics) == 0
    # HyperParameters have 3 keys
    assert len(training_details.training_job_details.hyper_parameters) == 3


@patch("sagemaker.Session")
def test_training_details_autodiscovery_from_job_name(session):
    session.sagemaker_client.describe_training_job.side_effect = [
        DESCRIBE_TRAINING_JOB_EXAMPLE,
        MISSING_TRAINING_JOB_CLIENT_ERROR,
    ]

    training_details = TrainingDetails.from_training_job_name(
        training_job_name=TRAINING_JOB_NAME, sagemaker_session=session
    )
    assert training_details.training_job_details.training_arn == TRAINING_JOB_ARN
    assert len(training_details.training_job_details.training_metrics) == len(
        SEARCH_TRAINING_JOB_EXAMPLE["Results"][0]["TrainingJob"]["FinalMetricDataList"]
    )
    assert len(training_details.training_job_details.hyper_parameters) == len(
        SEARCH_TRAINING_JOB_EXAMPLE["Results"][0]["TrainingJob"]["HyperParameters"]
    )
    assert training_details.training_job_details.training_environment.container_image == [
        TRAINING_IMAGE
    ]

    with pytest.raises(
        ValueError,
        match=re.escape(
            (
                "Training job details could not be found. "
                "Make sure the training job name is valid."
            )
        ),
    ):
        TrainingDetails.from_training_job_name(
            training_job_name=TRAINING_JOB_NAME, sagemaker_session=session
        )


def test_add_user_provided_training_metrics(training_details_example):
    assert len(training_details_example.training_job_details.user_provided_training_metrics) == 0
    training_details_example.add_metric(USER_METRIC)
    assert len(training_details_example.training_job_details.user_provided_training_metrics) == 1
    assert (
        training_details_example.training_job_details.user_provided_training_metrics[0].name
        == USER_METRIC_NAME
    )


def test_add_user_provided_hyper_parameters(training_details_example):
    assert len(training_details_example.training_job_details.user_provided_hyper_parameters) == 0
    training_details_example.add_parameter(USER_PARAMETER)
    assert len(training_details_example.training_job_details.user_provided_hyper_parameters) == 1
    assert (
        training_details_example.training_job_details.user_provided_hyper_parameters[0].name
        == USER_PARAMETER_NAME
    )


def test_add_evaluation_metrics_manually():
    evaluation_job = EvaluationJob(name=EVALUATION_JOB_NAME)

    metric_group = evaluation_job.add_metric_group("test_metric")
    assert len(evaluation_job.metric_groups) == 1
    evaluation_job.get_metric_group("test_metric").add_metric(metric_example)
    assert len(metric_group.metric_data) == 1

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Provided item type is <class 'dict'> and Expected the item type is <class 'sagemaker.model_card.model_card.Metric'>"  # noqa E501  # pylint: disable=c0301
        ),
    ):
        metric_group.add_metric({"name": "metric1", "type": "number", "value": 123})


def test_add_evaluation_metrics_from_json():
    evaluation_job = EvaluationJob(name=EVALUATION_JOB_NAME)
    json_path = os.path.join(DATA_DIR, "evaluation_metrics/clarify_bias.json")
    evaluation_job.add_metric_group_from_json(json_path, EvaluationMetricTypeEnum.CLARIFY_BIAS)

    assert len(evaluation_job.metric_groups) == 3
    assert (
        len(
            evaluation_job.get_metric_group(
                "post_training_bias_metrics - label Target = 1 and facet Gender=0"
            ).metric_data
        )
        == 1
    )
    assert (
        len(
            evaluation_job.get_metric_group(
                "pre_training_bias_metrics - label Target = 1 and facet Gender=0"
            ).metric_data
        )
        == 1
    )


@patch("boto3.session.Session")
def test_add_evauation_metrics_from_s3(session, caplog):
    json_path = os.path.join(DATA_DIR, "evaluation_metrics/clarify_bias.json")
    with open(json_path, "r", encoding="utf-8") as istr:
        data = json.dumps(json.load(istr))
        response = {
            "Body": botocore.response.StreamingBody(
                io.BytesIO(bytes(data, "utf-8")), content_length=len(data)
            ),
            "ContentType": "application/json",
        }
    session.client.return_value.get_object.side_effect = [
        response,
        S3_MISSING_KEY_CLIENT_ERROR,
    ]

    evaluation_job = EvaluationJob(name=EVALUATION_JOB_NAME)
    evaluation_job.add_metric_group_from_s3(
        session=session,
        s3_url="s3://test/clarify_bias.json",
        metric_type=EvaluationMetricTypeEnum.CLARIFY_BIAS,
    )
    assert len(evaluation_job.metric_groups) == 3

    evaluation_job.add_metric_group_from_s3(
        session=session,
        s3_url="s3://test/clarify_bias.json",
        metric_type=EvaluationMetricTypeEnum.CLARIFY_BIAS,
    )
    assert "Metric file clarify_bias.json does not exist in test." in caplog.text


def test_metrics_clarify_bias():
    with open(
        os.path.join(DATA_DIR, "evaluation_metrics/clarify_bias.json"),
        "r",
        encoding="utf-8",
    ) as istr:
        json_data = json.load(istr)

    with open(
        os.path.join(DATA_DIR, "evaluation_metrics/translated_clarify_bias.json"),
        "r",
        encoding="utf-8",
    ) as istr:
        expected_translation = json.load(istr)

    parser = EVALUATION_METRIC_PARSERS[EvaluationMetricTypeEnum.CLARIFY_BIAS]
    result = parser.run(json_data)

    assert json.dumps(result, sort_keys=True) == json.dumps(expected_translation, sort_keys=True)


def test_metrics_clarify_explanation():
    with open(
        os.path.join(DATA_DIR, "evaluation_metrics/clarify_explanability.json"),
        "r",
        encoding="utf-8",
    ) as istr:
        json_data = json.load(istr)

    with open(
        os.path.join(DATA_DIR, "evaluation_metrics/translated_clarify_explanation.json"),
        "r",
        encoding="utf-8",
    ) as istr:
        expected_translation = json.load(istr)

    parser = EVALUATION_METRIC_PARSERS[EvaluationMetricTypeEnum.CLARIFY_EXPLAINABILITY]
    result = parser.run(json_data)

    assert json.dumps(result, sort_keys=True) == json.dumps(expected_translation, sort_keys=True)


def test_metrics_model_monitor_model_quality_binary_classification():
    with open(
        os.path.join(
            DATA_DIR,
            "evaluation_metrics/model_monitor_model_quality_binary_classification.json",
        ),
        "r",
        encoding="utf-8",
    ) as istr:
        json_data = json.load(istr)

    with open(
        os.path.join(
            DATA_DIR,
            "evaluation_metrics/translated_model_monitor_model_quality_binary_classification.json",
        ),
        "r",
        encoding="utf-8",
    ) as istr:
        expected_translation = json.load(istr)

    parser = EVALUATION_METRIC_PARSERS[EvaluationMetricTypeEnum.MODEL_MONITOR_MODEL_QUALITY]
    result = parser.run(json_data)

    assert json.dumps(result, sort_keys=True) == json.dumps(expected_translation, sort_keys=True)


def test_metrics_model_monitor_model_quality_regression():
    with open(
        os.path.join(
            DATA_DIR,
            "evaluation_metrics/model_monitor_model_quality_regression.json",
        ),
        "r",
        encoding="utf-8",
    ) as istr:
        json_data = json.load(istr)

    with open(
        os.path.join(
            DATA_DIR,
            "evaluation_metrics/translated_model_monitor_model_quality_regression.json",
        ),
        "r",
        encoding="utf-8",
    ) as istr:
        expected_translation = json.load(istr)

    parser = EVALUATION_METRIC_PARSERS[EvaluationMetricTypeEnum.MODEL_MONITOR_MODEL_QUALITY]
    result = parser.run(json_data)

    assert json.dumps(result, sort_keys=True) == json.dumps(expected_translation, sort_keys=True)


@patch("sagemaker.Session")
def test_create_export_model_card(session, caplog):
    session.sagemaker_client.create_model_card_export_job.side_effect = [
        CREATE_EXPORT_MODEL_CARD_EXAMPLE,
        CREATE_EXPORT_MODEL_CARD_EXAMPLE,
    ]
    session.sagemaker_client.describe_model_card_export_job.side_effect = [
        DESCRIBE_MODEL_CARD_EXPORT_JOB_IN_PROGRESS_EXAMPLE,
        DESCRIBE_MODEL_CARD_EXPORT_JOB_COMPLETED_EXAMPLE,
        DESCRIBE_MODEL_CARD_EXPORT_JOB_FAILED_EXAMPLE,
    ]

    job = ModelCardExportJob(
        model_card_name=MODEL_CARD_NAME,
        model_card_version=MODEL_CARD_VERSION,
        export_job_name=EXPORT_JOB_NAME,
        s3_output_path=S3_URL,
        sagemaker_session=session,
    )
    pdf_s3_url = job.create()
    assert (
        pdf_s3_url
        == DESCRIBE_MODEL_CARD_EXPORT_JOB_COMPLETED_EXAMPLE["ExportArtifacts"]["S3ExportArtifacts"]
    )

    job = ModelCardExportJob(
        model_card_name=MODEL_CARD_NAME,
        model_card_version=MODEL_CARD_VERSION,
        export_job_name=EXPORT_JOB_NAME,
        s3_output_path=S3_URL,
        sagemaker_session=session,
    )
    job.create()
    assert "Failed to export model card" in caplog.text


@patch("sagemaker.Session")
def test_list_export_model_cards(session):
    session.sagemaker_client.list_model_card_export_jobs.side_effect = [
        LIST_MODEL_CARD_EXPORT_JOB_EXAMPLE
    ]

    response = ModelCardExportJob.list_export_jobs(
        model_card_name=MODEL_CARD_NAME, sagemaker_session=session
    )

    assert len(response["ModelCardExportJobSummaries"]) == len(
        LIST_MODEL_CARD_EXPORT_JOB_EXAMPLE["ModelCardExportJobSummaries"]
    )


@patch("sagemaker.Session")
def test_model_card_export_pdf(session, caplog):
    session.sagemaker_client.create_model_card_export_job.side_effect = [
        CREATE_EXPORT_MODEL_CARD_EXAMPLE,
        CREATE_EXPORT_MODEL_CARD_EXAMPLE,
    ]
    session.sagemaker_client.describe_model_card_export_job.side_effect = [
        DESCRIBE_MODEL_CARD_EXPORT_JOB_IN_PROGRESS_EXAMPLE,
        DESCRIBE_MODEL_CARD_EXPORT_JOB_COMPLETED_EXAMPLE,
        DESCRIBE_MODEL_CARD_EXPORT_JOB_FAILED_EXAMPLE,
    ]

    card = ModelCard(name=MODEL_CARD_NAME, sagemaker_session=session)
    pdf_s3_url = card.export_pdf(export_job_name=EXPORT_JOB_NAME, s3_output_path=S3_URL)
    assert (
        pdf_s3_url
        == DESCRIBE_MODEL_CARD_EXPORT_JOB_COMPLETED_EXAMPLE["ExportArtifacts"]["S3ExportArtifacts"]
    )

    card.export_pdf(export_job_name=EXPORT_JOB_NAME, s3_output_path=S3_URL)
    assert "Failed to export model card" in caplog.text


@patch("sagemaker.Session")
def test_list_model_card_version_history(session):
    session.sagemaker_client.list_model_card_versions.side_effect = [
        LIST_MODEL_CARD_VERSION_HISTORY_EXAMPLE
    ]
    card = ModelCard(name=MODEL_CARD_NAME, sagemaker_session=session)
    versions = card.get_version_history()

    assert len(versions) == len(
        LIST_MODEL_CARD_VERSION_HISTORY_EXAMPLE["ModelCardVersionSummaryList"]
    )
