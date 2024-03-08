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
from sagemaker import (
    AutoMLDataChannel,
    AutoMLJobV2,
    AutoMLTabularConfig,
    AutoMLV2,
    CandidateEstimator,
    LocalAutoMLDataChannel,
    PipelineModel,
)
from sagemaker.predictor import Predictor
from sagemaker.session_settings import SessionSettings

from tests.unit import (
    DEFAULT_S3_BUCKET_NAME,
    DEFAULT_S3_OBJECT_KEY_PREFIX_NAME,
    SAGEMAKER_CONFIG_AUTO_ML,
    SAGEMAKER_CONFIG_TRAINING_JOB,
    _test_default_bucket_and_prefix_combinations,
)

MODEL_DATA = "s3://bucket/model.tar.gz"
MODEL_IMAGE = "mi"
ENTRY_POINT = "blah.py"

TIMESTAMP = "2017-11-06-14:14:15.671"
BUCKET_NAME = "mybucket"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c5.2xlarge"
RESOURCE_POOLS = [{"InstanceType": INSTANCE_TYPE, "PoolSize": INSTANCE_COUNT}]
ROLE = "DummyRole"
TARGET_ATTRIBUTE_NAME = "target"
SAMPLE_WEIGHT_ATTRIBUTE_NAME = "sampleWeight"
REGION = "us-west-2"
DEFAULT_S3_INPUT_DATA = "s3://{}/data".format(BUCKET_NAME)
DEFAULT_S3_VALIDATION_DATA = "s3://{}/validation_data".format(BUCKET_NAME)
DEFAULT_OUTPUT_PATH = "s3://{}/".format(BUCKET_NAME)
LOCAL_DATA_PATH = "file://data"
DEFAULT_MAX_CANDIDATES = None
DEFAULT_JOB_NAME = "automl-{}".format(TIMESTAMP)

JOB_NAME = "default-job-name"
JOB_NAME_2 = "banana-auto-ml-job"
JOB_NAME_3_TABULAR = "tabular-auto-ml-job-3"
JOB_NAME_4_IMAGE_CLASSIFICATION = "image-classification-auto-ml-job-4"
JOB_NAME_5_TEXT_CLASSIFICATION = "text-classification-job-name-5"
JOB_NAME_6_TEXT_GENERATION = "text-generation-job-name-6"
JOB_NAME_7_FORECASTING = "forecasting-job-name-7"
VOLUME_KMS_KEY = "volume-kms-key-id-string"
OUTPUT_KMS_KEY = "output-kms-key-id-string"
OUTPUT_PATH = "s3://my_other_bucket/"
BASE_JOB_NAME = "banana"
PROBLEM_TYPE = "BinaryClassification"
LIST_TAGS_RESULT = {"Tags": [{"Key": "key1", "Value": "value1"}]}
MAX_CANDIDATES = 10
MAX_RUNTIME_PER_TRAINING_JOB = 3600
TOTAL_JOB_RUNTIME = 36000
TARGET_OBJECTIVE = "0.01"
JOB_OBJECTIVE = {"MetricName": "F1"}
TAGS = [{"Name": "some-tag", "Value": "value-for-tag"}]
CONTENT_TYPE = "x-application/vnd.amazon+parquet"
S3_DATA_TYPE = "ManifestFile"
FEATURE_SPECIFICATION_S3_URI = "s3://{}/features.json".format(BUCKET_NAME)
VALIDATION_FRACTION = 0.2
MODE = "ENSEMBLING"
AUTO_GENERATE_ENDPOINT_NAME = False
ENDPOINT_NAME = "EndpointName"
VPC_CONFIG = {"SecurityGroupIds": ["group"], "Subnets": ["subnet"]}
COMPRESSION_TYPE = "Gzip"
ENCRYPT_INTER_CONTAINER_TRAFFIC = False
GENERATE_CANDIDATE_DEFINITIONS_ONLY = False
BEST_CANDIDATE = {"best-candidate": "best-trial"}
BEST_CANDIDATE_2 = {"best-candidate": "best-trial-2"}
AUTO_ML_V2_DESC = {"AutoMLJobName": JOB_NAME, "BestCandidate": BEST_CANDIDATE}
AUTO_ML_V2_DESC_2 = {"AutoMLJobName": JOB_NAME_2, "BestCandidate": BEST_CANDIDATE_2}
AUTO_ML_V2_DESC_3_TABULAR = {
    "AutoMLJobName": "mock_automl_job_name",
    "AutoMLJobArn": "automl_job_arn",
    "AutoMLJobInputDataConfig": [
        {
            "ChannelType": "training",
            "CompressionType": "Gzip",
            "ContentType": "x-application/vnd.amazon+parquet",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "ManifestFile",
                    "S3Uri": "s3://mybucket/data",
                }
            },
        },
        {
            "ChannelType": "validation",
            "CompressionType": "Gzip",
            "ContentType": "x-application/vnd.amazon+parquet",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "ManifestFile",
                    "S3Uri": "s3://mybucket/data",
                }
            },
        },
    ],
    "OutputDataConfig": {"KmsKeyId": "string", "S3OutputPath": "s3://output_prefix"},
    "RoleArn": "mock_role_arn",
    "AutoMLJobObjective": {"MetricName": "Auto"},
    "AutoMLProblemTypeConfig": {
        "TabularJobConfig": {
            "CandidateGenerationConfig": {
                "AlgorithmsConfig": [
                    {
                        "AutoMLAlgorithms": [
                            "xgboost",
                        ]
                    },
                ]
            },
            "CompletionCriteria": {
                "MaxCandidates": 28,
                "MaxRuntimePerTrainingJobInSeconds": 100,
                "MaxAutoMLJobRuntimeInSeconds": 3000,
            },
            "FeatureSpecificationS3Uri": "s3://mybucket/features.json",
            "Mode": "ENSEMBLING",
            "GenerateCandidateDefinitionsOnly": False,
            "ProblemType": "Auto",
            "TargetAttributeName": "y",
            "SampleWeightAttributeName": "sampleWeight",
        },
    },
    "AutoMLProblemTypeConfigName": "Tabular",
    "DataSplitConfig": {"ValidationFraction": 0.2},
    "SecurityConfig": {"EnableInterContainerTrafficEncryption": True},
    "AutoMLJobSecondaryStatus": "Completed",
    "AutoMLJobStatus": "Completed",
    "ModelDeployConfig": {
        "AutoGenerateEndpointName": False,
        "EndpointName": "EndpointName",
    },
}
AUTO_ML_V2_DESC_4_IMAGE_CLASSIFICATION = {}
AUTO_ML_V2_DESC_5_TEXT_CLASSIFICATION = {}
AUTO_ML_V2_DESC_6_TEXT_GENERATION = {}
AUTO_ML_V2_DESC_7_FORECASTING = {}

INFERENCE_CONTAINERS = [
    {
        "Environment": {"SAGEMAKER_PROGRAM": "sagemaker_serve"},
        "Image": "account.dkr.ecr.us-west-2.amazonaws.com/sagemaker-auto-ml-data-processing:1.0-cpu-py3",
        "ModelDataUrl": "s3://sagemaker-us-west-2-account/sagemaker-auto-ml-gamma/data-processing/output",
    },
    {
        "Environment": {"MAX_CONTENT_LENGTH": "20000000"},
        "Image": "account.dkr.ecr.us-west-2.amazonaws.com/sagemaker-auto-ml-training:1.0-cpu-py3",
        "ModelDataUrl": "s3://sagemaker-us-west-2-account/sagemaker-auto-ml-gamma/training/output",
    },
    {
        "Environment": {"INVERSE_LABEL_TRANSFORM": "1"},
        "Image": "account.dkr.ecr.us-west-2.amazonaws.com/sagemaker-auto-ml-transform:1.0-cpu-py3",
        "ModelDataUrl": "s3://sagemaker-us-west-2-account/sagemaker-auto-ml-gamma/transform/output",
    },
]

CLASSIFICATION_INFERENCE_CONTAINERS = [
    {
        "Environment": {"SAGEMAKER_PROGRAM": "sagemaker_serve"},
        "Image": "account.dkr.ecr.us-west-2.amazonaws.com/sagemaker-auto-ml-data-processing:1.0-cpu-py3",
        "ModelDataUrl": "s3://sagemaker-us-west-2-account/sagemaker-auto-ml-gamma/data-processing/output",
    },
    {
        "Environment": {
            "MAX_CONTENT_LENGTH": "20000000",
            "SAGEMAKER_INFERENCE_SUPPORTED": "probability,probabilities,predicted_label",
            "SAGEMAKER_INFERENCE_OUTPUT": "predicted_label",
        },
        "Image": "account.dkr.ecr.us-west-2.amazonaws.com/sagemaker-auto-ml-training:1.0-cpu-py3",
        "ModelDataUrl": "s3://sagemaker-us-west-2-account/sagemaker-auto-ml-gamma/training/output",
    },
    {
        "Environment": {
            "INVERSE_LABEL_TRANSFORM": "1",
            "SAGEMAKER_INFERENCE_SUPPORTED": "probability,probabilities,predicted_label,labels",
            "SAGEMAKER_INFERENCE_OUTPUT": "predicted_label",
            "SAGEMAKER_INFERENCE_INPUT": "predicted_label",
        },
        "Image": "account.dkr.ecr.us-west-2.amazonaws.com/sagemaker-auto-ml-transform:1.0-cpu-py3",
        "ModelDataUrl": "s3://sagemaker-us-west-2-account/sagemaker-auto-ml-gamma/transform/output",
    },
]

CANDIDATE_STEPS = [
    {
        "CandidateStepName": "training-job/sagemaker-auto-ml-gamma/data-processing",
        "CandidateStepType": "AWS::Sagemaker::TrainingJob",
    },
    {
        "CandidateStepName": "transform-job/sagemaker-auto-ml-gamma/transform",
        "CandidateStepType": "AWS::Sagemaker::TransformJob",
    },
    {
        "CandidateStepName": "training-job/sagemaker-auto-ml-gamma/training",
        "CandidateStepType": "AWS::Sagemaker::TrainingJob",
    },
]

CANDIDATE_DICT = {
    "CandidateName": "candidate_mock",
    "InferenceContainers": INFERENCE_CONTAINERS,
    "CandidateSteps": CANDIDATE_STEPS,
}

CLASSIFICATION_CANDIDATE_DICT = {
    "CandidateName": "candidate_mock",
    "InferenceContainers": CLASSIFICATION_INFERENCE_CONTAINERS,
    "CandidateSteps": CANDIDATE_STEPS,
}

TRAINING_JOB = {
    "AlgorithmSpecification": {
        "AlgorithmName": "string",
        "TrainingImage": "string",
        "TrainingInputMode": "string",
    },
    "CheckpointConfig": {"LocalPath": "string", "S3Uri": "string"},
    "EnableInterContainerTrafficEncryption": False,
    "EnableManagedSpotTraining": False,
    "EnableNetworkIsolation": False,
    "InputDataConfig": [
        {"DataSource": {"S3DataSource": {"S3DataType": "string", "S3Uri": "string"}}}
    ],
    "OutputDataConfig": {"KmsKeyId": "string", "S3OutputPath": "string"},
    "ResourceConfig": {},
    "RoleArn": "string",
    "StoppingCondition": {},
    "TrainingJobArn": "string",
    "TrainingJobName": "string",
    "TrainingJobStatus": "string",
    "VpcConfig": {},
}

TRANSFORM_JOB = {
    "BatchStrategy": "string",
    "DataProcessing": {},
    "Environment": {"string": "string"},
    "FailureReason": "string",
    "LabelingJobArn": "string",
    "MaxConcurrentTransforms": 1,
    "MaxPayloadInMB": 2000,
    "ModelName": "string",
    "TransformInput": {"DataSource": {"S3DataSource": {"S3DataType": "string", "S3Uri": "string"}}},
    "TransformJobStatus": "string",
    "TransformJobArn": "string",
    "TransformJobName": "string",
    "TransformOutput": {},
    "TransformResources": {},
}


def describe_auto_ml_job_v2_mock(job_name=None):
    if job_name is None or job_name == JOB_NAME:
        return AUTO_ML_V2_DESC
    elif job_name == JOB_NAME_2:
        return AUTO_ML_V2_DESC_2
    elif job_name == JOB_NAME_3_TABULAR:
        return AUTO_ML_V2_DESC_3_TABULAR
    elif job_name == JOB_NAME_4_IMAGE_CLASSIFICATION:
        return AUTO_ML_V2_DESC_4_IMAGE_CLASSIFICATION
    elif job_name == JOB_NAME_5_TEXT_CLASSIFICATION:
        return AUTO_ML_V2_DESC_5_TEXT_CLASSIFICATION
    elif job_name == JOB_NAME_6_TEXT_GENERATION:
        return AUTO_ML_V2_DESC_6_TEXT_GENERATION
    elif job_name == JOB_NAME_7_FORECASTING:
        return AUTO_ML_V2_DESC_7_FORECASTING


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    sms = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
        settings=SessionSettings(),
        default_bucket_prefix=None,
    )
    sms.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    sms.upload_data = Mock(name="upload_data", return_value=DEFAULT_S3_INPUT_DATA)
    sms.expand_role = Mock(name="expand_role", return_value=ROLE)
    sms.describe_auto_ml_job_v2 = Mock(
        name="describe_auto_ml_job_v2", side_effect=describe_auto_ml_job_v2_mock
    )
    sms.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job", return_value=TRAINING_JOB
    )
    sms.sagemaker_client.describe_transform_job = Mock(
        name="describe_transform_job", return_value=TRANSFORM_JOB
    )
    sms.list_candidates = Mock(name="list_candidates", return_value={"Candidates": []})
    sms.sagemaker_client.list_tags = Mock(name="list_tags", return_value=LIST_TAGS_RESULT)
    # For tests which doesn't verify config file injection, operate with empty config
    sms.sagemaker_config = {}
    return sms


@pytest.fixture()
def candidate_mock(sagemaker_session):
    candidate = Mock(
        name="candidate_mock",
        containers=INFERENCE_CONTAINERS,
        steps=CANDIDATE_STEPS,
        sagemaker_session=sagemaker_session,
    )
    return candidate


def test_auto_ml_without_role_parameter(sagemaker_session):
    with pytest.raises(ValueError):
        AutoMLV2(
            problem_config=AutoMLTabularConfig(
                target_attribute_name=TARGET_ATTRIBUTE_NAME,
            ),
            sagemaker_session=sagemaker_session,
        )


def test_framework_initialization_with_sagemaker_config_injection(sagemaker_session):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_AUTO_ML

    auto_ml = AutoMLV2(
        problem_config=AutoMLTabularConfig(
            target_attribute_name=TARGET_ATTRIBUTE_NAME,
        ),
        sagemaker_session=sagemaker_session,
    )

    expected_volume_kms_key_id = SAGEMAKER_CONFIG_AUTO_ML["SageMaker"]["AutoMLJob"][
        "AutoMLJobConfig"
    ]["SecurityConfig"]["VolumeKmsKeyId"]
    expected_role_arn = SAGEMAKER_CONFIG_AUTO_ML["SageMaker"]["AutoMLJob"]["RoleArn"]
    expected_kms_key_id = SAGEMAKER_CONFIG_AUTO_ML["SageMaker"]["AutoMLJob"]["OutputDataConfig"][
        "KmsKeyId"
    ]
    expected_vpc_config = SAGEMAKER_CONFIG_AUTO_ML["SageMaker"]["AutoMLJob"]["AutoMLJobConfig"][
        "SecurityConfig"
    ]["VpcConfig"]
    expected_enable_inter_container_traffic_encryption = SAGEMAKER_CONFIG_AUTO_ML["SageMaker"][
        "AutoMLJob"
    ]["AutoMLJobConfig"]["SecurityConfig"]["EnableInterContainerTrafficEncryption"]
    assert auto_ml.role == expected_role_arn
    assert auto_ml.output_kms_key == expected_kms_key_id
    assert auto_ml.volume_kms_key == expected_volume_kms_key_id
    assert auto_ml.vpc_config == expected_vpc_config
    assert (
        auto_ml.encrypt_inter_container_traffic
        == expected_enable_inter_container_traffic_encryption
    )


def test_auto_ml_default_channel_name(sagemaker_session):
    auto_ml = AutoMLV2(
        role=ROLE,
        problem_config=AutoMLTabularConfig(
            target_attribute_name=TARGET_ATTRIBUTE_NAME,
        ),
        sagemaker_session=sagemaker_session,
    )
    inputs = AutoMLDataChannel(
        s3_uri=DEFAULT_S3_INPUT_DATA,
        s3_data_type="S3Prefix",
    )
    AutoMLJobV2.start_new(auto_ml, inputs)
    sagemaker_session.create_auto_ml_v2.assert_called_once()
    _, args = sagemaker_session.create_auto_ml_v2.call_args
    assert args["input_config"] == [
        {
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": DEFAULT_S3_INPUT_DATA,
                }
            },
        }
    ]
    assert args["problem_config"] == {
        "TabularJobConfig": {
            "TargetAttributeName": TARGET_ATTRIBUTE_NAME,
        },
    }


def test_auto_ml_validation_channel_name(sagemaker_session):
    auto_ml = AutoMLV2(
        role=ROLE,
        problem_config=AutoMLTabularConfig(
            target_attribute_name=TARGET_ATTRIBUTE_NAME,
            sample_weight_attribute_name="sampleWeight",
        ),
        sagemaker_session=sagemaker_session,
    )
    input_training = AutoMLDataChannel(
        s3_uri=DEFAULT_S3_INPUT_DATA,
        s3_data_type="S3Prefix",
        compression_type="Gzip",
        channel_type="training",
    )
    input_validation = AutoMLDataChannel(
        s3_uri=DEFAULT_S3_VALIDATION_DATA,
        s3_data_type="S3Prefix",
        compression_type="Gzip",
        channel_type="validation",
    )
    inputs = [input_training, input_validation]
    AutoMLJobV2.start_new(auto_ml, inputs)
    sagemaker_session.create_auto_ml_v2.assert_called_once()
    _, args = sagemaker_session.create_auto_ml_v2.call_args
    assert args["input_config"] == [
        {
            "ChannelType": "training",
            "CompressionType": "Gzip",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": DEFAULT_S3_INPUT_DATA,
                }
            },
        },
        {
            "ChannelType": "validation",
            "CompressionType": "Gzip",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": DEFAULT_S3_VALIDATION_DATA,
                }
            },
        },
    ]
    assert args["problem_config"] == {
        "TabularJobConfig": {
            "TargetAttributeName": TARGET_ATTRIBUTE_NAME,
            "SampleWeightAttributeName": SAMPLE_WEIGHT_ATTRIBUTE_NAME,
        }
    }


def test_auto_ml_invalid_input_data_format(sagemaker_session):
    auto_ml = AutoMLV2(
        role=ROLE,
        problem_config=AutoMLTabularConfig(
            target_attribute_name=TARGET_ATTRIBUTE_NAME,
        ),
        sagemaker_session=sagemaker_session,
    )
    inputs = {}

    with pytest.raises(ValueError) as excinfo:
        AutoMLJobV2.start_new(auto_ml, inputs)

    expected_error_msg = (
        "Cannot format input {}. Expecting an AutoMLDataChannel or "
        "a list of AutoMLDataChannel or a LocalAutoMLDataChannel or a list of "
        "LocalAutoMLDataChannel."
    )
    assert expected_error_msg.format(inputs) in str(excinfo.value)

    sagemaker_session.auto_ml.assert_not_called()


def test_auto_ml_only_one_of_problem_type_and_job_objective_provided(sagemaker_session):
    with pytest.raises(ValueError) as excinfo:
        AutoMLV2(
            role=ROLE,
            problem_config=AutoMLTabularConfig(
                target_attribute_name=TARGET_ATTRIBUTE_NAME,
                problem_type=PROBLEM_TYPE,
            ),
            sagemaker_session=sagemaker_session,
        )

    message = (
        "One of problem type and objective metric provided. Either both of them "
        "should be provided or none of them should be provided."
    )
    assert message in str(excinfo.value)


@patch("sagemaker.automl.automl.AutoMLJob.start_new")
def test_auto_ml_fit_set_logs_to_false(start_new, sagemaker_session, caplog):
    auto_ml = AutoMLV2(
        role=ROLE,
        problem_config=AutoMLTabularConfig(
            target_attribute_name=TARGET_ATTRIBUTE_NAME,
        ),
        sagemaker_session=sagemaker_session,
    )
    inputs = AutoMLDataChannel(
        s3_uri=DEFAULT_S3_INPUT_DATA,
        s3_data_type="S3Prefix",
    )
    auto_ml.fit(inputs, job_name=JOB_NAME, wait=False, logs=True)
    start_new.wait.assert_not_called()
    assert "Setting logs to False. logs is only meaningful when wait is True." in caplog.text


def test_auto_ml_additional_optional_params(sagemaker_session):
    auto_ml = AutoMLV2(
        role=ROLE,
        problem_config=AutoMLTabularConfig(
            target_attribute_name=TARGET_ATTRIBUTE_NAME,
            problem_type=PROBLEM_TYPE,
            max_candidates=MAX_CANDIDATES,
            max_runtime_per_training_job_in_seconds=MAX_RUNTIME_PER_TRAINING_JOB,
            max_total_job_runtime_in_seconds=TOTAL_JOB_RUNTIME,
            mode=MODE,
            feature_specification_s3_uri=FEATURE_SPECIFICATION_S3_URI,
            generate_candidate_definitions_only=GENERATE_CANDIDATE_DEFINITIONS_ONLY,
            sample_weight_attribute_name=SAMPLE_WEIGHT_ATTRIBUTE_NAME,
        ),
        sagemaker_session=sagemaker_session,
        volume_kms_key=VOLUME_KMS_KEY,
        vpc_config=VPC_CONFIG,
        encrypt_inter_container_traffic=ENCRYPT_INTER_CONTAINER_TRAFFIC,
        output_kms_key=OUTPUT_KMS_KEY,
        output_path=OUTPUT_PATH,
        job_objective=JOB_OBJECTIVE,
        tags=TAGS,
        validation_fraction=VALIDATION_FRACTION,
        auto_generate_endpoint_name=AUTO_GENERATE_ENDPOINT_NAME,
        endpoint_name=ENDPOINT_NAME,
    )
    inputs = AutoMLDataChannel(
        s3_uri=DEFAULT_S3_INPUT_DATA,
        s3_data_type=S3_DATA_TYPE,
        content_type=CONTENT_TYPE,
        compression_type=COMPRESSION_TYPE,
    )
    auto_ml.fit(inputs, job_name=JOB_NAME)
    sagemaker_session.create_auto_ml_v2.assert_called_once()
    _, args = sagemaker_session.create_auto_ml_v2.call_args

    assert args == {
        "input_config": [
            {
                "ContentType": CONTENT_TYPE,
                "CompressionType": COMPRESSION_TYPE,
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": S3_DATA_TYPE,
                        "S3Uri": DEFAULT_S3_INPUT_DATA,
                    }
                },
            }
        ],
        "problem_config": {
            "TabularJobConfig": {
                "TargetAttributeName": TARGET_ATTRIBUTE_NAME,
                "SampleWeightAttributeName": SAMPLE_WEIGHT_ATTRIBUTE_NAME,
                "GenerateCandidateDefinitionsOnly": GENERATE_CANDIDATE_DEFINITIONS_ONLY,
                "FeatureSpecificationS3Uri": FEATURE_SPECIFICATION_S3_URI,
                "CompletionCriteria": {
                    "MaxAutoMLJobRuntimeInSeconds": TOTAL_JOB_RUNTIME,
                    "MaxCandidates": MAX_CANDIDATES,
                    "MaxRuntimePerTrainingJobInSeconds": MAX_RUNTIME_PER_TRAINING_JOB,
                },
                "Mode": MODE,
                "ProblemType": PROBLEM_TYPE,
            }
        },
        "data_split_config": {"ValidationFraction": VALIDATION_FRACTION},
        "output_config": {"S3OutputPath": OUTPUT_PATH, "KmsKeyId": OUTPUT_KMS_KEY},
        "job_name": JOB_NAME,
        "role": ROLE,
        "job_objective": JOB_OBJECTIVE,
        "tags": TAGS,
        "model_deploy_config": {
            "AutoGenerateEndpointName": AUTO_GENERATE_ENDPOINT_NAME,
            "EndpointName": ENDPOINT_NAME,
        },
        "security_config": {
            "VolumeKmsKeyId": VOLUME_KMS_KEY,
            "VpcConfig": VPC_CONFIG,
            "EnableInterContainerTrafficEncryption": ENCRYPT_INTER_CONTAINER_TRAFFIC,
        },
    }


@patch("time.strftime", return_value=TIMESTAMP)
def test_auto_ml_default_fit(strftime, sagemaker_session):
    auto_ml = AutoMLV2(
        role=ROLE,
        problem_config=AutoMLTabularConfig(
            target_attribute_name=TARGET_ATTRIBUTE_NAME,
        ),
        sagemaker_session=sagemaker_session,
        encrypt_inter_container_traffic=False,
    )
    inputs = AutoMLDataChannel(
        s3_uri=DEFAULT_S3_INPUT_DATA,
        s3_data_type=S3_DATA_TYPE,
    )
    auto_ml.fit(inputs)
    sagemaker_session.create_auto_ml_v2.assert_called_once()
    _, args = sagemaker_session.create_auto_ml_v2.call_args
    assert args == {
        "input_config": [
            {
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": S3_DATA_TYPE,
                        "S3Uri": DEFAULT_S3_INPUT_DATA,
                    }
                },
            }
        ],
        "problem_config": {
            "TabularJobConfig": {
                "TargetAttributeName": TARGET_ATTRIBUTE_NAME,
            }
        },
        "output_config": {"S3OutputPath": DEFAULT_OUTPUT_PATH},
        "security_config": {
            "EnableInterContainerTrafficEncryption": ENCRYPT_INTER_CONTAINER_TRAFFIC
        },
        "role": ROLE,
        "job_name": DEFAULT_JOB_NAME,
        "job_objective": None,
        "tags": None,
    }


def test_auto_ml_local_input(sagemaker_session):
    auto_ml = AutoMLV2(
        role=ROLE,
        problem_config=AutoMLTabularConfig(
            target_attribute_name=TARGET_ATTRIBUTE_NAME,
        ),
        sagemaker_session=sagemaker_session,
    )
    inputs = LocalAutoMLDataChannel(
        path=DEFAULT_S3_INPUT_DATA,
        data_type=S3_DATA_TYPE,
    )
    auto_ml.fit(inputs)
    sagemaker_session.create_auto_ml_v2.assert_called_once()
    _, args = sagemaker_session.create_auto_ml_v2.call_args
    assert args["input_config"][0]["DataSource"]["S3DataSource"]["S3Uri"] == DEFAULT_S3_INPUT_DATA


def test_auto_ml_input(sagemaker_session):
    inputs = AutoMLDataChannel(
        s3_uri=DEFAULT_S3_INPUT_DATA,
        s3_data_type=S3_DATA_TYPE,
        compression_type="Gzip",
    )
    auto_ml = AutoMLV2(
        role=ROLE,
        problem_config=AutoMLTabularConfig(
            target_attribute_name=TARGET_ATTRIBUTE_NAME,
            sample_weight_attribute_name="sampleWeight",
        ),
        sagemaker_session=sagemaker_session,
    )
    auto_ml.fit(inputs)
    _, args = sagemaker_session.create_auto_ml_v2.call_args
    assert args["input_config"] == [
        {
            "CompressionType": "Gzip",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": S3_DATA_TYPE,
                    "S3Uri": DEFAULT_S3_INPUT_DATA,
                }
            },
        }
    ]
    assert args["problem_config"] == {
        "TabularJobConfig": {
            "TargetAttributeName": TARGET_ATTRIBUTE_NAME,
            "SampleWeightAttributeName": SAMPLE_WEIGHT_ATTRIBUTE_NAME,
        }
    }


def test_describe_auto_ml_job(sagemaker_session):
    auto_ml = AutoMLV2(
        role=ROLE,
        problem_config=AutoMLTabularConfig(
            target_attribute_name=TARGET_ATTRIBUTE_NAME,
        ),
        sagemaker_session=sagemaker_session,
    )
    auto_ml.describe_auto_ml_job(job_name=JOB_NAME)
    sagemaker_session.describe_auto_ml_job_v2.assert_called_once()
    sagemaker_session.describe_auto_ml_job_v2.assert_called_with(job_name=JOB_NAME)


def test_list_candidates_default(sagemaker_session):
    auto_ml = AutoMLV2(
        role=ROLE,
        problem_config=AutoMLTabularConfig(
            target_attribute_name=TARGET_ATTRIBUTE_NAME,
        ),
        sagemaker_session=sagemaker_session,
    )
    auto_ml.current_job_name = "current_job_name"
    auto_ml.list_candidates()
    sagemaker_session.list_candidates.assert_called_once()
    sagemaker_session.list_candidates.assert_called_with(job_name=auto_ml.current_job_name)


def test_list_candidates_with_optional_args(sagemaker_session):
    auto_ml = AutoMLV2(
        role=ROLE,
        problem_config=AutoMLTabularConfig(
            target_attribute_name=TARGET_ATTRIBUTE_NAME,
        ),
        sagemaker_session=sagemaker_session,
    )
    auto_ml.list_candidates(
        job_name=JOB_NAME,
        status_equals="Completed",
        candidate_name="candidate-name",
        candidate_arn="candidate-arn",
        sort_order="Ascending",
        sort_by="Status",
        max_results=99,
    )
    sagemaker_session.list_candidates.assert_called_once()
    _, args = sagemaker_session.list_candidates.call_args
    assert args == {
        "job_name": JOB_NAME,
        "status_equals": "Completed",
        "candidate_name": "candidate-name",
        "candidate_arn": "candidate-arn",
        "sort_order": "Ascending",
        "sort_by": "Status",
        "max_results": 99,
    }


def test_best_candidate_with_existing_best_candidate(sagemaker_session):
    auto_ml = AutoMLV2(
        role=ROLE,
        problem_config=AutoMLTabularConfig(
            target_attribute_name=TARGET_ATTRIBUTE_NAME,
        ),
        sagemaker_session=sagemaker_session,
    )
    auto_ml._best_candidate = BEST_CANDIDATE
    best_candidate = auto_ml.best_candidate()
    sagemaker_session.describe_auto_ml_job_v2.assert_not_called()
    assert best_candidate == BEST_CANDIDATE


def test_best_candidate_default_job_name(sagemaker_session):
    auto_ml = AutoMLV2(
        role=ROLE,
        problem_config=AutoMLTabularConfig(
            target_attribute_name=TARGET_ATTRIBUTE_NAME,
        ),
        sagemaker_session=sagemaker_session,
    )
    auto_ml.current_job_name = JOB_NAME
    auto_ml._auto_ml_job_desc = AUTO_ML_V2_DESC
    best_candidate = auto_ml.best_candidate()
    sagemaker_session.describe_auto_ml_job_v2.assert_not_called()
    assert best_candidate == BEST_CANDIDATE


def test_best_candidate_job_no_desc(sagemaker_session):
    auto_ml = AutoMLV2(
        role=ROLE,
        problem_config=AutoMLTabularConfig(
            target_attribute_name=TARGET_ATTRIBUTE_NAME,
        ),
        sagemaker_session=sagemaker_session,
    )
    auto_ml.current_job_name = JOB_NAME
    best_candidate = auto_ml.best_candidate()
    sagemaker_session.describe_auto_ml_job_v2.assert_called_once()
    sagemaker_session.describe_auto_ml_job_v2.assert_called_with(job_name=JOB_NAME)
    assert best_candidate == BEST_CANDIDATE


def test_best_candidate_no_desc_no_job_name(sagemaker_session):
    auto_ml = AutoMLV2(
        role=ROLE,
        problem_config=AutoMLTabularConfig(
            target_attribute_name=TARGET_ATTRIBUTE_NAME,
        ),
        sagemaker_session=sagemaker_session,
    )
    best_candidate = auto_ml.best_candidate(job_name=JOB_NAME)
    sagemaker_session.describe_auto_ml_job_v2.assert_called_once()
    sagemaker_session.describe_auto_ml_job_v2.assert_called_with(job_name=JOB_NAME)
    assert best_candidate == BEST_CANDIDATE


def test_best_candidate_job_name_not_match(sagemaker_session):
    auto_ml = AutoMLV2(
        role=ROLE,
        problem_config=AutoMLTabularConfig(
            target_attribute_name=TARGET_ATTRIBUTE_NAME,
        ),
        sagemaker_session=sagemaker_session,
    )
    auto_ml.current_job_name = JOB_NAME
    auto_ml._auto_ml_job_desc = AUTO_ML_V2_DESC
    best_candidate = auto_ml.best_candidate(job_name=JOB_NAME_2)
    sagemaker_session.describe_auto_ml_job_v2.assert_called_once()
    sagemaker_session.describe_auto_ml_job_v2.assert_called_with(job_name=JOB_NAME_2)
    assert best_candidate == BEST_CANDIDATE_2


def test_deploy(sagemaker_session, candidate_mock):
    auto_ml = AutoMLV2(
        role=ROLE,
        problem_config=AutoMLTabularConfig(
            target_attribute_name=TARGET_ATTRIBUTE_NAME,
        ),
        sagemaker_session=sagemaker_session,
    )
    mock_pipeline = Mock(name="pipeline_model")
    mock_pipeline.deploy = Mock(name="model_deploy")
    auto_ml.best_candidate = Mock(name="best_candidate", return_value=CANDIDATE_DICT)
    auto_ml.create_model = Mock(name="create_model", return_value=mock_pipeline)
    auto_ml.deploy(
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
        model_kms_key=OUTPUT_KMS_KEY,
    )
    auto_ml.create_model.assert_called_once()
    mock_pipeline.deploy.assert_called_once()


@patch("sagemaker.automl.automl.CandidateEstimator")
def test_deploy_optional_args(candidate_estimator, sagemaker_session, candidate_mock):
    candidate_estimator.return_value = candidate_mock

    auto_ml = AutoMLV2(
        role=ROLE,
        problem_config=AutoMLTabularConfig(
            target_attribute_name=TARGET_ATTRIBUTE_NAME,
        ),
        sagemaker_session=sagemaker_session,
    )
    mock_pipeline = Mock(name="pipeline_model")
    mock_pipeline.deploy = Mock(name="model_deploy")
    auto_ml.best_candidate = Mock(name="best_candidate", return_value=CANDIDATE_DICT)
    auto_ml.create_model = Mock(name="create_model", return_value=mock_pipeline)

    auto_ml.deploy(
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        candidate=CANDIDATE_DICT,
        sagemaker_session=sagemaker_session,
        name=JOB_NAME,
        endpoint_name=JOB_NAME,
        tags=TAGS,
        wait=False,
        vpc_config=VPC_CONFIG,
        enable_network_isolation=True,
        model_kms_key=OUTPUT_KMS_KEY,
        predictor_cls=Predictor,
        inference_response_keys=None,
    )

    auto_ml.create_model.assert_called_once()
    auto_ml.create_model.assert_called_with(
        name=JOB_NAME,
        sagemaker_session=sagemaker_session,
        candidate=CANDIDATE_DICT,
        inference_response_keys=None,
        vpc_config=VPC_CONFIG,
        enable_network_isolation=True,
        model_kms_key=OUTPUT_KMS_KEY,
        predictor_cls=Predictor,
    )

    mock_pipeline.deploy.assert_called_once()

    mock_pipeline.deploy.assert_called_with(
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        serializer=None,
        deserializer=None,
        endpoint_name=JOB_NAME,
        kms_key=OUTPUT_KMS_KEY,
        volume_size=None,
        model_data_download_timeout=None,
        container_startup_health_check_timeout=None,
        tags=TAGS,
        wait=False,
    )


def test_candidate_estimator_fit_initialization_with_sagemaker_config_injection(
    sagemaker_session,
):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_TRAINING_JOB
    sagemaker_session.train = Mock()
    sagemaker_session.transform = Mock()

    desc_training_job_response = copy.deepcopy(TRAINING_JOB)
    del desc_training_job_response["VpcConfig"]
    del desc_training_job_response["OutputDataConfig"]["KmsKeyId"]

    sagemaker_session.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job", return_value=desc_training_job_response
    )
    candidate_estimator = CandidateEstimator(CANDIDATE_DICT, sagemaker_session=sagemaker_session)
    candidate_estimator._check_all_job_finished = Mock(
        name="_check_all_job_finished", return_value=True
    )
    inputs = DEFAULT_S3_INPUT_DATA
    candidate_estimator.fit(inputs)
    expected_volume_kms_key_id = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "ResourceConfig"
    ]["VolumeKmsKeyId"]
    expected_vpc_config = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"]["VpcConfig"]
    expected_enable_inter_container_traffic_encryption = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"][
        "TrainingJob"
    ]["EnableInterContainerTrafficEncryption"]

    for train_call in sagemaker_session.train.call_args_list:
        train_args = train_call.kwargs
        assert train_args["vpc_config"] == expected_vpc_config
        assert train_args["resource_config"]["VolumeKmsKeyId"] == expected_volume_kms_key_id
        assert (
            train_args["encrypt_inter_container_traffic"]
            == expected_enable_inter_container_traffic_encryption
        )


def test_candidate_estimator_get_steps(sagemaker_session):
    candidate_estimator = CandidateEstimator(CANDIDATE_DICT, sagemaker_session=sagemaker_session)
    steps = candidate_estimator.get_steps()
    assert len(steps) == 3


def test_candidate_estimator_fit(sagemaker_session):
    candidate_estimator = CandidateEstimator(CANDIDATE_DICT, sagemaker_session=sagemaker_session)
    candidate_estimator._check_all_job_finished = Mock(
        name="_check_all_job_finished", return_value=True
    )
    inputs = DEFAULT_S3_INPUT_DATA
    candidate_estimator.fit(inputs)
    sagemaker_session.train.assert_called()
    sagemaker_session.transform.assert_called()


def test_validate_and_update_inference_response():
    cic = copy.copy(CLASSIFICATION_INFERENCE_CONTAINERS)

    AutoMLV2.validate_and_update_inference_response(
        inference_containers=cic,
        inference_response_keys=[
            "predicted_label",
            "labels",
            "probabilities",
            "probability",
        ],
    )

    assert (
        cic[2]["Environment"]["SAGEMAKER_INFERENCE_OUTPUT"]
        == "predicted_label,labels,probabilities,probability"
    )
    assert (
        cic[2]["Environment"]["SAGEMAKER_INFERENCE_INPUT"]
        == "predicted_label,probabilities,probability"
    )
    assert (
        cic[1]["Environment"]["SAGEMAKER_INFERENCE_OUTPUT"]
        == "predicted_label,probabilities,probability"
    )


def test_validate_and_update_inference_response_wrong_input():
    cic = copy.copy(CLASSIFICATION_INFERENCE_CONTAINERS)

    with pytest.raises(ValueError) as excinfo:
        AutoMLV2.validate_and_update_inference_response(
            inference_containers=cic,
            inference_response_keys=[
                "wrong_key",
                "wrong_label",
                "probabilities",
                "probability",
            ],
        )
    message = (
        "Requested inference output keys [wrong_key, wrong_label] are unsupported. "
        "The supported inference keys are [probability, probabilities, predicted_label, labels]"
    )
    assert message in str(excinfo.value)


def test_create_model(sagemaker_session):
    auto_ml = AutoMLV2(
        role=ROLE,
        problem_config=AutoMLTabularConfig(
            target_attribute_name=TARGET_ATTRIBUTE_NAME,
        ),
        sagemaker_session=sagemaker_session,
    )

    pipeline_model = auto_ml.create_model(
        name=JOB_NAME,
        sagemaker_session=sagemaker_session,
        candidate=CLASSIFICATION_CANDIDATE_DICT,
        vpc_config=VPC_CONFIG,
        enable_network_isolation=True,
        model_kms_key=None,
        predictor_cls=None,
        inference_response_keys=None,
    )

    assert isinstance(pipeline_model, PipelineModel)


def test_attach_tabular(sagemaker_session):
    aml = AutoMLV2.attach(auto_ml_job_name=JOB_NAME_3_TABULAR, sagemaker_session=sagemaker_session)
    assert aml.current_job_name == JOB_NAME_3_TABULAR
    assert aml.role == "mock_role_arn"
    assert aml.problem_config.target_attribute_name == "y"
    assert aml.problem_config.problem_type == "Auto"
    assert aml.output_path == "s3://output_prefix"
    assert aml.tags == LIST_TAGS_RESULT["Tags"]
    assert aml.inputs[0].content_type == "x-application/vnd.amazon+parquet"
    assert aml.inputs[0].s3_data_type == "ManifestFile"
    assert aml.problem_config.feature_specification_s3_uri == "s3://{}/features.json".format(
        BUCKET_NAME
    )
    assert aml.validation_fraction == 0.2
    assert aml.problem_config.mode == "ENSEMBLING"
    assert aml.auto_generate_endpoint_name is False
    assert aml.endpoint_name == "EndpointName"


@patch("sagemaker.automl.automlv2.AutoMLJobV2.start_new")
def test_output_path_default_bucket_and_prefix_combinations(start_new):
    inputs = AutoMLDataChannel(
        s3_data_type=S3_DATA_TYPE,
        s3_uri=DEFAULT_S3_INPUT_DATA,
    )

    def with_user_input(sess):
        auto_ml = AutoMLV2(
            role=ROLE,
            problem_config=AutoMLTabularConfig(
                target_attribute_name=TARGET_ATTRIBUTE_NAME,
            ),
            sagemaker_session=sess,
            output_path="s3://test",
        )
        auto_ml.fit(inputs, job_name=JOB_NAME, wait=False, logs=True)
        start_new.assert_called()  # just to make sure this is patched with a mock
        return auto_ml.output_path

    def without_user_input(sess):
        auto_ml = AutoMLV2(
            role=ROLE,
            problem_config=AutoMLTabularConfig(
                target_attribute_name=TARGET_ATTRIBUTE_NAME,
            ),
            sagemaker_session=sess,
        )
        auto_ml.fit(inputs, job_name=JOB_NAME, wait=False, logs=True)
        start_new.assert_called()  # just to make sure this is patched with a mock
        return auto_ml.output_path

    actual, expected = _test_default_bucket_and_prefix_combinations(
        function_with_user_input=with_user_input,
        function_without_user_input=without_user_input,
        expected__without_user_input__with_default_bucket_and_default_prefix=(
            f"s3://{DEFAULT_S3_BUCKET_NAME}/{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/"
        ),
        expected__without_user_input__with_default_bucket_only=f"s3://{DEFAULT_S3_BUCKET_NAME}/",
        expected__with_user_input__with_default_bucket_and_prefix="s3://test",
        expected__with_user_input__with_default_bucket_only="s3://test",
    )
    assert actual == expected
