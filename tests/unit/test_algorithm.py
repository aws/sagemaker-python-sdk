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
import datetime

import pytest
from mock import Mock, patch

from sagemaker.algorithm import AlgorithmEstimator
from sagemaker.estimator import _TrainingJob
from sagemaker.transformer import Transformer

DESCRIBE_ALGORITHM_RESPONSE = {
    "AlgorithmName": "scikit-decision-trees",
    "AlgorithmArn": "arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
    "AlgorithmDescription": "Decision trees using Scikit",
    "CreationTime": datetime.datetime(2018, 8, 3, 22, 44, 54, 437000),
    "TrainingSpecification": {
        "TrainingImage": "123.dkr.ecr.us-east-2.amazonaws.com/decision-trees-sample@sha256:12345",
        "TrainingImageDigest": "sha256:206854b6ea2f0020d216311da732010515169820b898ec29720bcf1d2b46806a",
        "SupportedHyperParameters": [
            {
                "Name": "max_leaf_nodes",
                "Description": "Grow a tree with max_leaf_nodes in best-first fashion.",
                "Type": "Integer",
                "Range": {
                    "IntegerParameterRangeSpecification": {"MinValue": "1", "MaxValue": "100000"}
                },
                "IsTunable": True,
                "IsRequired": False,
                "DefaultValue": "100",
            },
            {
                "Name": "free_text_hp1",
                "Description": "You can write anything here",
                "Type": "FreeText",
                "IsTunable": False,
                "IsRequired": True,
            },
        ],
        "SupportedTrainingInstanceTypes": ["ml.m4.xlarge", "ml.m4.2xlarge", "ml.m4.4xlarge"],
        "SupportsDistributedTraining": False,
        "MetricDefinitions": [
            {"Name": "validation:accuracy", "Regex": "validation-accuracy: (\\S+)"}
        ],
        "TrainingChannels": [
            {
                "Name": "training",
                "Description": "Input channel that provides training data",
                "IsRequired": True,
                "SupportedContentTypes": ["text/csv"],
                "SupportedCompressionTypes": ["None"],
                "SupportedInputModes": ["File"],
            }
        ],
        "SupportedTuningJobObjectiveMetrics": [
            {"Type": "Maximize", "MetricName": "validation:accuracy"}
        ],
    },
    "InferenceSpecification": {
        "InferenceImage": "123.dkr.ecr.us-east-2.amazonaws.com/decision-trees-sample@sha256:123",
        "SupportedTransformInstanceTypes": ["ml.m4.xlarge", "ml.m4.2xlarge"],
        "SupportedContentTypes": ["text/csv"],
        "SupportedResponseMIMETypes": ["text"],
    },
    "ValidationSpecification": {
        "ValidationRole": "arn:aws:iam::764419575721:role/SageMakerRole",
        "ValidationProfiles": [
            {
                "ProfileName": "ValidationProfile1",
                "TrainingJobDefinition": {
                    "TrainingInputMode": "File",
                    "HyperParameters": {},
                    "InputDataConfig": [
                        {
                            "ChannelName": "training",
                            "DataSource": {
                                "S3DataSource": {
                                    "S3DataType": "S3Prefix",
                                    "S3Uri": "s3://sagemaker-us-east-2-7123/-scikit-byo-iris/training-input-data",
                                    "S3DataDistributionType": "FullyReplicated",
                                }
                            },
                            "ContentType": "text/csv",
                            "CompressionType": "None",
                            "RecordWrapperType": "None",
                        }
                    ],
                    "OutputDataConfig": {
                        "KmsKeyId": "",
                        "S3OutputPath": "s3://sagemaker-us-east-2-764419575721/DEMO-scikit-byo-iris/training-output",
                    },
                    "ResourceConfig": {
                        "InstanceType": "ml.c4.xlarge",
                        "InstanceCount": 1,
                        "VolumeSizeInGB": 10,
                    },
                    "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
                },
                "TransformJobDefinition": {
                    "MaxConcurrentTransforms": 0,
                    "MaxPayloadInMB": 0,
                    "TransformInput": {
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": "s3://sagemaker-us-east-2/scikit-byo-iris/batch-inference/transform_test.csv",
                            }
                        },
                        "ContentType": "text/csv",
                        "CompressionType": "None",
                        "SplitType": "Line",
                    },
                    "TransformOutput": {
                        "S3OutputPath": "s3://sagemaker-us-east-2-764419575721/scikit-byo-iris/batch-transform-output",
                        "Accept": "text/csv",
                        "AssembleWith": "Line",
                        "KmsKeyId": "",
                    },
                    "TransformResources": {"InstanceType": "ml.c4.xlarge", "InstanceCount": 1},
                },
            }
        ],
        "ValidationOutputS3Prefix": "s3://sagemaker-us-east-2-764419575721/DEMO-scikit-byo-iris/validation-output",
        "ValidateForMarketplace": True,
    },
    "AlgorithmStatus": "Completed",
    "AlgorithmStatusDetails": {
        "ValidationStatuses": [{"ProfileName": "ValidationProfile1", "Status": "Completed"}]
    },
    "ResponseMetadata": {
        "RequestId": "e04bc28b-61b6-4486-9106-0edf07f5649c",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "e04bc28b-61b6-4486-9106-0edf07f5649c",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "3949",
            "date": "Fri, 03 Aug 2018 23:08:43 GMT",
        },
        "RetryAttempts": 0,
    },
}


@patch("sagemaker.Session")
def test_algorithm_supported_input_mode_with_valid_input_types(session):
    # verify that the Estimator verifies the
    # input mode that an Algorithm supports.
    session.sagemaker_config = {}

    file_mode_algo = copy.deepcopy(DESCRIBE_ALGORITHM_RESPONSE)
    file_mode_algo["TrainingSpecification"]["TrainingChannels"] = [
        {
            "Name": "training",
            "Description": "Input channel that provides training data",
            "IsRequired": True,
            "SupportedContentTypes": ["text/csv"],
            "SupportedCompressionTypes": ["None"],
            "SupportedInputModes": ["File"],
        },
        {
            "Name": "validation",
            "Description": "Input channel that provides validation data",
            "IsRequired": False,
            "SupportedContentTypes": ["text/csv"],
            "SupportedCompressionTypes": ["None"],
            "SupportedInputModes": ["File", "Pipe"],
        },
    ]

    session.sagemaker_client.describe_algorithm = Mock(return_value=file_mode_algo)

    # Creating a File mode Estimator with a File mode algorithm should work
    AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.xlarge",
        instance_count=1,
        sagemaker_session=session,
    )

    pipe_mode_algo = copy.deepcopy(DESCRIBE_ALGORITHM_RESPONSE)
    pipe_mode_algo["TrainingSpecification"]["TrainingChannels"] = [
        {
            "Name": "training",
            "Description": "Input channel that provides training data",
            "IsRequired": True,
            "SupportedContentTypes": ["text/csv"],
            "SupportedCompressionTypes": ["None"],
            "SupportedInputModes": ["Pipe"],
        },
        {
            "Name": "validation",
            "Description": "Input channel that provides validation data",
            "IsRequired": False,
            "SupportedContentTypes": ["text/csv"],
            "SupportedCompressionTypes": ["None"],
            "SupportedInputModes": ["File", "Pipe"],
        },
    ]

    session.sagemaker_client.describe_algorithm = Mock(return_value=pipe_mode_algo)

    # Creating a Pipe mode Estimator with a Pipe mode algorithm should work.
    AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.xlarge",
        instance_count=1,
        input_mode="Pipe",
        sagemaker_session=session,
    )

    any_input_algo = copy.deepcopy(DESCRIBE_ALGORITHM_RESPONSE)
    any_input_algo["TrainingSpecification"]["TrainingChannels"] = [
        {
            "Name": "training",
            "Description": "Input channel that provides training data",
            "IsRequired": True,
            "SupportedContentTypes": ["text/csv"],
            "SupportedCompressionTypes": ["None"],
            "SupportedInputModes": ["File", "Pipe"],
        },
        {
            "Name": "validation",
            "Description": "Input channel that provides validation data",
            "IsRequired": False,
            "SupportedContentTypes": ["text/csv"],
            "SupportedCompressionTypes": ["None"],
            "SupportedInputModes": ["File", "Pipe"],
        },
    ]

    session.sagemaker_client.describe_algorithm = Mock(return_value=any_input_algo)

    # Creating a File mode Estimator with an algorithm that supports both input modes
    # should work.
    AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.xlarge",
        instance_count=1,
        sagemaker_session=session,
    )


@patch("sagemaker.Session")
def test_algorithm_supported_input_mode_with_bad_input_types(session):
    # verify that the Estimator verifies raises exceptions when
    # attempting to train with an incorrect input type
    session.sagemaker_config = {}

    file_mode_algo = copy.deepcopy(DESCRIBE_ALGORITHM_RESPONSE)
    file_mode_algo["TrainingSpecification"]["TrainingChannels"] = [
        {
            "Name": "training",
            "Description": "Input channel that provides training data",
            "IsRequired": True,
            "SupportedContentTypes": ["text/csv"],
            "SupportedCompressionTypes": ["None"],
            "SupportedInputModes": ["File"],
        },
        {
            "Name": "validation",
            "Description": "Input channel that provides validation data",
            "IsRequired": False,
            "SupportedContentTypes": ["text/csv"],
            "SupportedCompressionTypes": ["None"],
            "SupportedInputModes": ["File", "Pipe"],
        },
    ]

    session.sagemaker_client.describe_algorithm = Mock(return_value=file_mode_algo)

    # Creating a Pipe mode Estimator with a File mode algorithm should fail.
    with pytest.raises(ValueError):
        AlgorithmEstimator(
            algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
            role="SageMakerRole",
            instance_type="ml.m4.xlarge",
            instance_count=1,
            input_mode="Pipe",
            sagemaker_session=session,
        )

    pipe_mode_algo = copy.deepcopy(DESCRIBE_ALGORITHM_RESPONSE)
    pipe_mode_algo["TrainingSpecification"]["TrainingChannels"] = [
        {
            "Name": "training",
            "Description": "Input channel that provides training data",
            "IsRequired": True,
            "SupportedContentTypes": ["text/csv"],
            "SupportedCompressionTypes": ["None"],
            "SupportedInputModes": ["Pipe"],
        },
        {
            "Name": "validation",
            "Description": "Input channel that provides validation data",
            "IsRequired": False,
            "SupportedContentTypes": ["text/csv"],
            "SupportedCompressionTypes": ["None"],
            "SupportedInputModes": ["File", "Pipe"],
        },
    ]

    session.sagemaker_client.describe_algorithm = Mock(return_value=pipe_mode_algo)

    # Creating a File mode Estimator with a Pipe mode algorithm should fail.
    with pytest.raises(ValueError):
        AlgorithmEstimator(
            algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
            role="SageMakerRole",
            instance_type="ml.m4.xlarge",
            instance_count=1,
            sagemaker_session=session,
        )


@patch("sagemaker.estimator.EstimatorBase.fit", Mock())
@patch("sagemaker.Session")
def test_algorithm_trainining_channels_with_expected_channels(session):
    session.sagemaker_config = {}
    training_channels = copy.deepcopy(DESCRIBE_ALGORITHM_RESPONSE)

    training_channels["TrainingSpecification"]["TrainingChannels"] = [
        {
            "Name": "training",
            "Description": "Input channel that provides training data",
            "IsRequired": True,
            "SupportedContentTypes": ["text/csv"],
            "SupportedCompressionTypes": ["None"],
            "SupportedInputModes": ["File"],
        },
        {
            "Name": "validation",
            "Description": "Input channel that provides validation data",
            "IsRequired": False,
            "SupportedContentTypes": ["text/csv"],
            "SupportedCompressionTypes": ["None"],
            "SupportedInputModes": ["File"],
        },
    ]

    session.sagemaker_client.describe_algorithm = Mock(return_value=training_channels)

    estimator = AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.xlarge",
        instance_count=1,
        sagemaker_session=session,
    )

    # Pass training and validation channels. This should work
    estimator.fit({"training": "s3://some/place", "validation": "s3://some/other"})

    # Passing only the training channel. Validation is optional so this should also work.
    estimator.fit({"training": "s3://some/place"})


@patch("sagemaker.estimator.EstimatorBase.fit", Mock())
@patch("sagemaker.Session")
def test_algorithm_trainining_channels_with_invalid_channels(session):
    session.sagemaker_config = {}
    training_channels = copy.deepcopy(DESCRIBE_ALGORITHM_RESPONSE)

    training_channels["TrainingSpecification"]["TrainingChannels"] = [
        {
            "Name": "training",
            "Description": "Input channel that provides training data",
            "IsRequired": True,
            "SupportedContentTypes": ["text/csv"],
            "SupportedCompressionTypes": ["None"],
            "SupportedInputModes": ["File"],
        },
        {
            "Name": "validation",
            "Description": "Input channel that provides validation data",
            "IsRequired": False,
            "SupportedContentTypes": ["text/csv"],
            "SupportedCompressionTypes": ["None"],
            "SupportedInputModes": ["File"],
        },
    ]

    session.sagemaker_client.describe_algorithm = Mock(return_value=training_channels)

    estimator = AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.xlarge",
        instance_count=1,
        sagemaker_session=session,
    )

    # Passing only validation should fail as training is required.
    with pytest.raises(ValueError):
        estimator.fit({"validation": "s3://some/thing"})

    # Passing an unknown channel should fail???
    with pytest.raises(ValueError):
        estimator.fit({"training": "s3://some/data", "training2": "s3://some/other/data"})


@patch("sagemaker.Session")
def test_algorithm_train_instance_types_valid_instance_types(session):
    session.sagemaker_config = {}
    describe_algo_response = copy.deepcopy(DESCRIBE_ALGORITHM_RESPONSE)
    instance_types = ["ml.m4.xlarge", "ml.m5.2xlarge"]

    describe_algo_response["TrainingSpecification"][
        "SupportedTrainingInstanceTypes"
    ] = instance_types

    session.sagemaker_client.describe_algorithm = Mock(return_value=describe_algo_response)

    AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.xlarge",
        instance_count=1,
        sagemaker_session=session,
    )

    AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m5.2xlarge",
        instance_count=1,
        sagemaker_session=session,
    )


@patch("sagemaker.Session")
def test_algorithm_train_instance_types_invalid_instance_types(session):
    session.sagemaker_config = {}
    describe_algo_response = copy.deepcopy(DESCRIBE_ALGORITHM_RESPONSE)
    instance_types = ["ml.m4.xlarge", "ml.m5.2xlarge"]

    describe_algo_response["TrainingSpecification"][
        "SupportedTrainingInstanceTypes"
    ] = instance_types

    session.sagemaker_client.describe_algorithm = Mock(return_value=describe_algo_response)

    # invalid instance type, should fail
    with pytest.raises(ValueError):
        AlgorithmEstimator(
            algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
            role="SageMakerRole",
            instance_type="ml.m4.8xlarge",
            instance_count=1,
            sagemaker_session=session,
        )


@patch("sagemaker.Session")
def test_algorithm_distributed_training_validation(session):
    session.sagemaker_config = {}
    distributed_algo = copy.deepcopy(DESCRIBE_ALGORITHM_RESPONSE)
    distributed_algo["TrainingSpecification"]["SupportsDistributedTraining"] = True

    single_instance_algo = copy.deepcopy(DESCRIBE_ALGORITHM_RESPONSE)
    single_instance_algo["TrainingSpecification"]["SupportsDistributedTraining"] = False

    session.sagemaker_client.describe_algorithm = Mock(return_value=distributed_algo)

    # Distributed training should work for Distributed and Single instance.
    AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.xlarge",
        instance_count=1,
        sagemaker_session=session,
    )

    AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.xlarge",
        instance_count=2,
        sagemaker_session=session,
    )

    session.sagemaker_client.describe_algorithm = Mock(return_value=single_instance_algo)

    # distributed training on a single instance algorithm should fail.
    with pytest.raises(ValueError):
        AlgorithmEstimator(
            algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
            role="SageMakerRole",
            instance_type="ml.m5.2xlarge",
            instance_count=2,
            sagemaker_session=session,
        )


@patch("sagemaker.Session")
def test_algorithm_hyperparameter_integer_range_valid_range(session):
    session.sagemaker_config = {}
    hyperparameters = [
        {
            "Description": "Grow a tree with max_leaf_nodes in best-first fashion.",
            "Type": "Integer",
            "Name": "max_leaf_nodes",
            "Range": {
                "IntegerParameterRangeSpecification": {"MinValue": "1", "MaxValue": "100000"}
            },
            "IsTunable": True,
            "IsRequired": False,
            "DefaultValue": "100",
        }
    ]

    some_algo = copy.deepcopy(DESCRIBE_ALGORITHM_RESPONSE)
    some_algo["TrainingSpecification"]["SupportedHyperParameters"] = hyperparameters

    session.sagemaker_client.describe_algorithm = Mock(return_value=some_algo)

    estimator = AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.2xlarge",
        instance_count=1,
        sagemaker_session=session,
    )

    estimator.set_hyperparameters(max_leaf_nodes=1)
    estimator.set_hyperparameters(max_leaf_nodes=100000)


@patch("sagemaker.Session")
def test_algorithm_hyperparameter_integer_range_invalid_range(session):
    session.sagemaker_config = {}
    hyperparameters = [
        {
            "Description": "Grow a tree with max_leaf_nodes in best-first fashion.",
            "Type": "Integer",
            "Name": "max_leaf_nodes",
            "Range": {
                "IntegerParameterRangeSpecification": {"MinValue": "1", "MaxValue": "100000"}
            },
            "IsTunable": True,
            "IsRequired": False,
            "DefaultValue": "100",
        }
    ]

    some_algo = copy.deepcopy(DESCRIBE_ALGORITHM_RESPONSE)
    some_algo["TrainingSpecification"]["SupportedHyperParameters"] = hyperparameters

    session.sagemaker_client.describe_algorithm = Mock(return_value=some_algo)

    estimator = AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.2xlarge",
        instance_count=1,
        sagemaker_session=session,
    )

    with pytest.raises(ValueError):
        estimator.set_hyperparameters(max_leaf_nodes=0)

    with pytest.raises(ValueError):
        estimator.set_hyperparameters(max_leaf_nodes=100001)


@patch("sagemaker.Session")
def test_algorithm_hyperparameter_continuous_range_valid_range(session):
    session.sagemaker_config = {}
    hyperparameters = [
        {
            "Description": "A continuous hyperparameter",
            "Type": "Continuous",
            "Name": "max_leaf_nodes",
            "Range": {
                "ContinuousParameterRangeSpecification": {"MinValue": "0.0", "MaxValue": "1.0"}
            },
            "IsTunable": True,
            "IsRequired": False,
            "DefaultValue": "100",
        }
    ]

    some_algo = copy.deepcopy(DESCRIBE_ALGORITHM_RESPONSE)
    some_algo["TrainingSpecification"]["SupportedHyperParameters"] = hyperparameters

    session.sagemaker_client.describe_algorithm = Mock(return_value=some_algo)

    estimator = AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.2xlarge",
        instance_count=1,
        sagemaker_session=session,
    )

    estimator.set_hyperparameters(max_leaf_nodes=0)
    estimator.set_hyperparameters(max_leaf_nodes=1.0)
    estimator.set_hyperparameters(max_leaf_nodes=0.5)
    estimator.set_hyperparameters(max_leaf_nodes=1)


@patch("sagemaker.Session")
def test_algorithm_hyperparameter_continuous_range_invalid_range(session):
    session.sagemaker_config = {}
    hyperparameters = [
        {
            "Description": "A continuous hyperparameter",
            "Type": "Continuous",
            "Name": "max_leaf_nodes",
            "Range": {
                "ContinuousParameterRangeSpecification": {"MinValue": "0.0", "MaxValue": "1.0"}
            },
            "IsTunable": True,
            "IsRequired": False,
            "DefaultValue": "100",
        }
    ]

    some_algo = copy.deepcopy(DESCRIBE_ALGORITHM_RESPONSE)
    some_algo["TrainingSpecification"]["SupportedHyperParameters"] = hyperparameters

    session.sagemaker_client.describe_algorithm = Mock(return_value=some_algo)

    estimator = AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.2xlarge",
        instance_count=1,
        sagemaker_session=session,
    )

    with pytest.raises(ValueError):
        estimator.set_hyperparameters(max_leaf_nodes=1.1)

    with pytest.raises(ValueError):
        estimator.set_hyperparameters(max_leaf_nodes=-0.1)


@patch("sagemaker.Session")
def test_algorithm_hyperparameter_categorical_range(session):
    session.sagemaker_config = {}
    hyperparameters = [
        {
            "Description": "A continuous hyperparameter",
            "Type": "Categorical",
            "Name": "hp1",
            "Range": {"CategoricalParameterRangeSpecification": {"Values": ["TF", "MXNet"]}},
            "IsTunable": True,
            "IsRequired": False,
            "DefaultValue": "100",
        }
    ]

    some_algo = copy.deepcopy(DESCRIBE_ALGORITHM_RESPONSE)
    some_algo["TrainingSpecification"]["SupportedHyperParameters"] = hyperparameters

    session.sagemaker_client.describe_algorithm = Mock(return_value=some_algo)

    estimator = AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.2xlarge",
        instance_count=1,
        sagemaker_session=session,
    )

    estimator.set_hyperparameters(hp1="MXNet")
    estimator.set_hyperparameters(hp1="TF")

    with pytest.raises(ValueError):
        estimator.set_hyperparameters(hp1="Chainer")

    with pytest.raises(ValueError):
        estimator.set_hyperparameters(hp1="MxNET")


@patch("sagemaker.Session")
def test_algorithm_required_hyperparameters_not_provided(session):
    session.sagemaker_config = {}
    hyperparameters = [
        {
            "Description": "A continuous hyperparameter",
            "Type": "Categorical",
            "Name": "hp1",
            "Range": {"CategoricalParameterRangeSpecification": {"Values": ["TF", "MXNet"]}},
            "IsTunable": True,
            "IsRequired": True,
        },
        {
            "Name": "hp2",
            "Description": "A continuous hyperparameter",
            "Type": "Categorical",
            "IsTunable": False,
            "IsRequired": True,
        },
    ]

    some_algo = copy.deepcopy(DESCRIBE_ALGORITHM_RESPONSE)
    some_algo["TrainingSpecification"]["SupportedHyperParameters"] = hyperparameters

    session.sagemaker_client.describe_algorithm = Mock(return_value=some_algo)

    estimator = AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.2xlarge",
        instance_count=1,
        sagemaker_session=session,
    )

    # hp1 is required and was not provided
    with pytest.raises(ValueError):
        estimator.set_hyperparameters(hp2="TF2")

    # Calling fit with unset required hyperparameters should fail
    # this covers the use case of not calling set_hyperparameters() explicitly
    with pytest.raises(ValueError):
        estimator.fit({"training": "s3://some/place"})


@patch("sagemaker.Session")
@patch("sagemaker.estimator.EstimatorBase.fit", Mock())
def test_algorithm_required_hyperparameters_are_provided(session):
    session.sagemaker_config = {}
    hyperparameters = [
        {
            "Description": "A categorical hyperparameter",
            "Type": "Categorical",
            "Name": "hp1",
            "Range": {"CategoricalParameterRangeSpecification": {"Values": ["TF", "MXNet"]}},
            "IsTunable": True,
            "IsRequired": True,
        },
        {
            "Name": "hp2",
            "Description": "A categorical hyperparameter",
            "Type": "Categorical",
            "IsTunable": False,
            "IsRequired": True,
        },
        {
            "Name": "free_text_hp1",
            "Description": "You can write anything here",
            "Type": "FreeText",
            "IsTunable": False,
            "IsRequired": True,
        },
    ]

    some_algo = copy.deepcopy(DESCRIBE_ALGORITHM_RESPONSE)
    some_algo["TrainingSpecification"]["SupportedHyperParameters"] = hyperparameters

    session.sagemaker_client.describe_algorithm = Mock(return_value=some_algo)

    estimator = AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.2xlarge",
        instance_count=1,
        sagemaker_session=session,
    )

    # All 3 Hyperparameters are provided
    estimator.set_hyperparameters(hp1="TF", hp2="TF2", free_text_hp1="Hello!")


@patch("sagemaker.Session")
def test_algorithm_required_free_text_hyperparameter_not_provided(session):
    session.sagemaker_config = {}
    hyperparameters = [
        {
            "Name": "free_text_hp1",
            "Description": "You can write anything here",
            "Type": "FreeText",
            "IsTunable": False,
            "IsRequired": True,
        },
        {
            "Name": "free_text_hp2",
            "Description": "You can write anything here",
            "Type": "FreeText",
            "IsTunable": False,
            "IsRequired": False,
        },
    ]

    some_algo = copy.deepcopy(DESCRIBE_ALGORITHM_RESPONSE)
    some_algo["TrainingSpecification"]["SupportedHyperParameters"] = hyperparameters

    session.sagemaker_client.describe_algorithm = Mock(return_value=some_algo)

    estimator = AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.2xlarge",
        instance_count=1,
        sagemaker_session=session,
    )

    # Calling fit with unset required hyperparameters should fail
    # this covers the use case of not calling set_hyperparameters() explicitly
    with pytest.raises(ValueError):
        estimator.fit({"training": "s3://some/place"})

    # hp1 is required and was not provided
    with pytest.raises(ValueError):
        estimator.set_hyperparameters(free_text_hp2="some text")


@patch("sagemaker.Session")
@patch("sagemaker.algorithm.AlgorithmEstimator.create_model")
def test_algorithm_create_transformer(create_model, session):
    session.sagemaker_config = {}
    session.sagemaker_client.describe_algorithm = Mock(return_value=DESCRIBE_ALGORITHM_RESPONSE)

    estimator = AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.xlarge",
        instance_count=1,
        sagemaker_session=session,
    )

    estimator.latest_training_job = _TrainingJob(session, "some-job-name")
    model = Mock()
    model.name = "my-model"
    create_model.return_value = model

    transformer = estimator.transformer(instance_count=1, instance_type="ml.m4.xlarge")

    assert isinstance(transformer, Transformer)
    create_model.assert_called()
    assert transformer.model_name == "my-model"


@patch("sagemaker.Session")
def test_algorithm_create_transformer_without_completed_training_job(session):
    session.sagemaker_config = {}
    session.sagemaker_client.describe_algorithm = Mock(return_value=DESCRIBE_ALGORITHM_RESPONSE)

    estimator = AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.xlarge",
        instance_count=1,
        sagemaker_session=session,
    )

    with pytest.raises(RuntimeError) as error:
        estimator.transformer(instance_count=1, instance_type="ml.m4.xlarge")
        assert "No finished training job found associated with this estimator" in str(error)


@patch("sagemaker.algorithm.AlgorithmEstimator.create_model")
@patch("sagemaker.Session")
def test_algorithm_create_transformer_with_product_id(create_model, session):
    session.sagemaker_config = {}
    response = copy.deepcopy(DESCRIBE_ALGORITHM_RESPONSE)
    response["ProductId"] = "some-product-id"
    session.sagemaker_client.describe_algorithm = Mock(return_value=response)

    estimator = AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.xlarge",
        instance_count=1,
        sagemaker_session=session,
    )

    estimator.latest_training_job = _TrainingJob(session, "some-job-name")
    model = Mock()
    model.name = "my-model"
    create_model.return_value = model

    transformer = estimator.transformer(instance_count=1, instance_type="ml.m4.xlarge")
    assert transformer.env is None


@patch("sagemaker.Session")
def test_algorithm_enable_network_isolation_no_product_id(session):
    session.sagemaker_config = {}
    session.sagemaker_client.describe_algorithm = Mock(return_value=DESCRIBE_ALGORITHM_RESPONSE)

    estimator = AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.xlarge",
        instance_count=1,
        sagemaker_session=session,
    )

    network_isolation = estimator.enable_network_isolation()
    assert network_isolation is False


@patch("sagemaker.Session")
def test_algorithm_enable_network_isolation_with_product_id(session):
    session.sagemaker_config = {}
    response = copy.deepcopy(DESCRIBE_ALGORITHM_RESPONSE)
    response["ProductId"] = "some-product-id"
    session.sagemaker_client.describe_algorithm = Mock(return_value=response)

    estimator = AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.xlarge",
        instance_count=1,
        sagemaker_session=session,
    )

    network_isolation = estimator.enable_network_isolation()
    assert network_isolation is True


@patch("sagemaker.Session")
def test_algorithm_encrypt_inter_container_traffic(session):
    session.sagemaker_config = {}
    response = copy.deepcopy(DESCRIBE_ALGORITHM_RESPONSE)
    response["encrypt_inter_container_traffic"] = True
    session.sagemaker_client.describe_algorithm = Mock(return_value=response)

    estimator = AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.xlarge",
        instance_count=1,
        sagemaker_session=session,
        encrypt_inter_container_traffic=True,
    )

    encrypt_inter_container_traffic = estimator.encrypt_inter_container_traffic
    assert encrypt_inter_container_traffic is True


@patch("sagemaker.Session")
def test_algorithm_no_required_hyperparameters(session):
    session.sagemaker_config = {}
    some_algo = copy.deepcopy(DESCRIBE_ALGORITHM_RESPONSE)
    del some_algo["TrainingSpecification"]["SupportedHyperParameters"]

    session.sagemaker_client.describe_algorithm = Mock(return_value=some_algo)

    # Calling AlgorithmEstimator() with unset required hyperparameters
    # should fail if they are required.
    # Pass training and hyperparameters channels. This should work
    assert AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.2xlarge",
        instance_count=1,
        sagemaker_session=session,
    )


def test_algorithm_attach_from_hyperparameter_tuning():
    session = Mock()
    session.sagemaker_config = {}
    job_name = "training-job-that-is-part-of-a-tuning-job"
    algo_arn = "arn:aws:sagemaker:us-east-2:000000000000:algorithm/scikit-decision-trees"
    role_arn = "arn:aws:iam::123412341234:role/SageMakerRole"
    instance_count = 1
    instance_type = "ml.m4.xlarge"
    volume_size = 30
    input_mode = "File"

    session.sagemaker_client.list_tags.return_value = {"Tags": []}
    session.sagemaker_client.describe_algorithm.return_value = DESCRIBE_ALGORITHM_RESPONSE
    session.sagemaker_client.describe_training_job.return_value = {
        "TrainingJobName": job_name,
        "TrainingJobArn": "arn:aws:sagemaker:us-east-2:123412341234:training-job/%s" % job_name,
        "TuningJobArn": "arn:aws:sagemaker:us-east-2:123412341234:hyper-parameter-tuning-job/%s"
        % job_name,
        "ModelArtifacts": {
            "S3ModelArtifacts": "s3://sagemaker-us-east-2-123412341234/output/model.tar.gz"
        },
        "TrainingJobOutput": {
            "S3TrainingJobOutput": "s3://sagemaker-us-east-2-123412341234/output/output.tar.gz"
        },
        "TrainingJobStatus": "Succeeded",
        "HyperParameters": {
            "_tuning_objective_metric": "validation:accuracy",
            "max_leaf_nodes": 1,
            "free_text_hp1": "foo",
        },
        "AlgorithmSpecification": {"AlgorithmName": algo_arn, "TrainingInputMode": input_mode},
        "MetricDefinitions": [
            {"Name": "validation:accuracy", "Regex": "validation-accuracy: (\\S+)"}
        ],
        "RoleArn": role_arn,
        "InputDataConfig": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://sagemaker-us-east-2-123412341234/input/training.csv",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "CompressionType": "None",
                "RecordWrapperType": "None",
            }
        ],
        "OutputDataConfig": {
            "KmsKeyId": "",
            "S3OutputPath": "s3://sagemaker-us-east-2-123412341234/output",
            "RemoveJobNameFromS3OutputPath": False,
        },
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": instance_count,
            "VolumeSizeInGB": volume_size,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
    }

    estimator = AlgorithmEstimator.attach(job_name, sagemaker_session=session)
    assert estimator.hyperparameters() == {"max_leaf_nodes": 1, "free_text_hp1": "foo"}
    assert estimator.algorithm_arn == algo_arn
    assert estimator.role == role_arn
    assert estimator.instance_count == instance_count
    assert estimator.instance_type == instance_type
    assert estimator.volume_size == volume_size
    assert estimator.input_mode == input_mode
    assert estimator.sagemaker_session == session


@patch("sagemaker.Session")
def test_algorithm_supported_with_spot_instances(session):
    session.sagemaker_config = {}
    session.sagemaker_client.describe_algorithm = Mock(return_value=DESCRIBE_ALGORITHM_RESPONSE)

    assert AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        role="SageMakerRole",
        instance_type="ml.m4.xlarge",
        instance_count=1,
        use_spot_instances=True,
        max_wait=500,
        sagemaker_session=session,
    )
