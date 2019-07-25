# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import os

import pytest
from mock import Mock

from sagemaker import RealTimePredictor
from sagemaker.amazon.amazon_estimator import RecordSet
from sagemaker.amazon.pca import PCA
from sagemaker.estimator import Estimator
from sagemaker.mxnet import MXNet
from sagemaker.parameter import (
    CategoricalParameter,
    ContinuousParameter,
    IntegerParameter,
    ParameterRange,
)
from sagemaker.tuner import (
    _TuningJob,
    create_identical_dataset_and_algorithm_tuner,
    create_transfer_learning_tuner,
    HyperparameterTuner,
    WarmStartConfig,
    WarmStartTypes,
)
from sagemaker.session import s3_input

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DATA = "s3://bucket/model.tar.gz"

JOB_NAME = "tuning_job"
REGION = "us-west-2"
BUCKET_NAME = "Some-Bucket"
ROLE = "myrole"
IMAGE_NAME = "image"

TRAIN_INSTANCE_COUNT = 1
TRAIN_INSTANCE_TYPE = "ml.c4.xlarge"
NUM_COMPONENTS = 5

SCRIPT_NAME = "my_script.py"
FRAMEWORK_VERSION = "1.0.0"

INPUTS = "s3://mybucket/train"
OBJECTIVE_METRIC_NAME = "mock_metric"
HYPERPARAMETER_RANGES = {
    "validated": ContinuousParameter(0, 5),
    "elizabeth": IntegerParameter(0, 5),
    "blank": CategoricalParameter([0, 5]),
}
METRIC_DEFINITIONS = "mock_metric_definitions"

TUNING_JOB_DETAILS = {
    "HyperParameterTuningJobConfig": {
        "ResourceLimits": {"MaxParallelTrainingJobs": 1, "MaxNumberOfTrainingJobs": 1},
        "HyperParameterTuningJobObjective": {
            "MetricName": OBJECTIVE_METRIC_NAME,
            "Type": "Minimize",
        },
        "Strategy": "Bayesian",
        "ParameterRanges": {
            "CategoricalParameterRanges": [],
            "ContinuousParameterRanges": [],
            "IntegerParameterRanges": [
                {
                    "MaxValue": "100",
                    "Name": "num_components",
                    "MinValue": "10",
                    "ScalingType": "Auto",
                }
            ],
        },
        "TrainingJobEarlyStoppingType": "Off",
    },
    "HyperParameterTuningJobName": JOB_NAME,
    "TrainingJobDefinition": {
        "RoleArn": ROLE,
        "StaticHyperParameters": {
            "num_components": "1",
            "_tuning_objective_metric": "train:throughput",
            "feature_dim": "784",
            "sagemaker_estimator_module": '"sagemaker.amazon.pca"',
            "sagemaker_estimator_class_name": '"PCA"',
        },
        "ResourceConfig": {
            "VolumeSizeInGB": 30,
            "InstanceType": "ml.c4.xlarge",
            "InstanceCount": 1,
        },
        "AlgorithmSpecification": {
            "TrainingImage": IMAGE_NAME,
            "TrainingInputMode": "File",
            "MetricDefinitions": METRIC_DEFINITIONS,
        },
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataDistributionType": "ShardedByS3Key",
                        "S3Uri": INPUTS,
                        "S3DataType": "ManifestFile",
                    }
                },
            }
        ],
        "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
        "OutputDataConfig": {"S3OutputPath": BUCKET_NAME},
    },
    "TrainingJobCounters": {
        "ClientError": 0,
        "Completed": 1,
        "InProgress": 0,
        "Fault": 0,
        "Stopped": 0,
    },
    "HyperParameterTuningEndTime": 1526605831.0,
    "CreationTime": 1526605605.0,
    "HyperParameterTuningJobArn": "arn:tuning_job",
}

ENDPOINT_DESC = {"EndpointConfigName": "test-endpoint"}

ENDPOINT_CONFIG_DESC = {"ProductionVariants": [{"ModelName": "model-1"}, {"ModelName": "model-2"}]}


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    sms = Mock(name="sagemaker_session", boto_session=boto_mock)
    sms.boto_region_name = REGION
    sms.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    sms.config = None

    sms.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    sms.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)

    return sms


@pytest.fixture()
def estimator(sagemaker_session):
    return Estimator(
        IMAGE_NAME,
        ROLE,
        TRAIN_INSTANCE_COUNT,
        TRAIN_INSTANCE_TYPE,
        output_path="s3://bucket/prefix",
        sagemaker_session=sagemaker_session,
    )


@pytest.fixture()
def tuner(estimator):
    return HyperparameterTuner(
        estimator, OBJECTIVE_METRIC_NAME, HYPERPARAMETER_RANGES, METRIC_DEFINITIONS
    )


def test_prepare_for_training(tuner):
    static_hyperparameters = {"validated": 1, "another_one": 0}
    tuner.estimator.set_hyperparameters(**static_hyperparameters)
    tuner._prepare_for_training()

    assert tuner._current_job_name.startswith(IMAGE_NAME)

    assert len(tuner.static_hyperparameters) == 1
    assert tuner.static_hyperparameters["another_one"] == "0"


def test_prepare_for_training_with_amazon_estimator(tuner, sagemaker_session):
    tuner.estimator = PCA(
        ROLE,
        TRAIN_INSTANCE_COUNT,
        TRAIN_INSTANCE_TYPE,
        NUM_COMPONENTS,
        sagemaker_session=sagemaker_session,
    )

    tuner._prepare_for_training()
    assert "sagemaker_estimator_class_name" not in tuner.static_hyperparameters
    assert "sagemaker_estimator_module" not in tuner.static_hyperparameters


def test_prepare_for_training_include_estimator_cls(tuner):
    tuner._prepare_for_training(include_cls_metadata=True)
    assert "sagemaker_estimator_class_name" in tuner.static_hyperparameters
    assert "sagemaker_estimator_module" in tuner.static_hyperparameters


def test_prepare_for_training_with_job_name(tuner):
    static_hyperparameters = {"validated": 1, "another_one": 0}
    tuner.estimator.set_hyperparameters(**static_hyperparameters)

    tuner._prepare_for_training(job_name="some-other-job-name")
    assert tuner._current_job_name == "some-other-job-name"


def test_validate_parameter_ranges_number_validation_error(sagemaker_session):
    pca = PCA(
        ROLE,
        TRAIN_INSTANCE_COUNT,
        TRAIN_INSTANCE_TYPE,
        NUM_COMPONENTS,
        base_job_name="pca",
        sagemaker_session=sagemaker_session,
    )

    invalid_hyperparameter_ranges = {"num_components": IntegerParameter(-1, 2)}

    with pytest.raises(ValueError) as e:
        HyperparameterTuner(
            estimator=pca,
            objective_metric_name=OBJECTIVE_METRIC_NAME,
            hyperparameter_ranges=invalid_hyperparameter_ranges,
            metric_definitions=METRIC_DEFINITIONS,
        )

    assert "Value must be an integer greater than zero" in str(e)


def test_validate_parameter_ranges_string_value_validation_error(sagemaker_session):
    pca = PCA(
        ROLE,
        TRAIN_INSTANCE_COUNT,
        TRAIN_INSTANCE_TYPE,
        NUM_COMPONENTS,
        base_job_name="pca",
        sagemaker_session=sagemaker_session,
    )

    invalid_hyperparameter_ranges = {"algorithm_mode": CategoricalParameter([0, 5])}

    with pytest.raises(ValueError) as e:
        HyperparameterTuner(
            estimator=pca,
            objective_metric_name=OBJECTIVE_METRIC_NAME,
            hyperparameter_ranges=invalid_hyperparameter_ranges,
            metric_definitions=METRIC_DEFINITIONS,
        )

    assert 'Value must be one of "regular" and "randomized"' in str(e)


def test_fit_pca(sagemaker_session, tuner):
    pca = PCA(
        ROLE,
        TRAIN_INSTANCE_COUNT,
        TRAIN_INSTANCE_TYPE,
        NUM_COMPONENTS,
        base_job_name="pca",
        sagemaker_session=sagemaker_session,
    )

    pca.algorithm_mode = "randomized"
    pca.subtract_mean = True
    pca.extra_components = 5

    tuner.estimator = pca

    tags = [{"Name": "some-tag-without-a-value"}]
    tuner.tags = tags

    hyperparameter_ranges = {
        "num_components": IntegerParameter(2, 4),
        "algorithm_mode": CategoricalParameter(["regular", "randomized"]),
    }
    tuner._hyperparameter_ranges = hyperparameter_ranges

    records = RecordSet(s3_data=INPUTS, num_records=1, feature_dim=1)
    tuner.fit(records, mini_batch_size=9999)

    _, _, tune_kwargs = sagemaker_session.tune.mock_calls[0]

    assert len(tune_kwargs["static_hyperparameters"]) == 4
    assert tune_kwargs["static_hyperparameters"]["extra_components"] == "5"
    assert len(tune_kwargs["parameter_ranges"]["IntegerParameterRanges"]) == 1
    assert tune_kwargs["job_name"].startswith("pca")
    assert tune_kwargs["tags"] == tags
    assert tune_kwargs["early_stopping_type"] == "Off"
    assert tuner.estimator.mini_batch_size == 9999


def test_fit_pca_with_early_stopping(sagemaker_session, tuner):
    pca = PCA(
        ROLE,
        TRAIN_INSTANCE_COUNT,
        TRAIN_INSTANCE_TYPE,
        NUM_COMPONENTS,
        base_job_name="pca",
        sagemaker_session=sagemaker_session,
    )

    tuner.estimator = pca
    tuner.early_stopping_type = "Auto"

    records = RecordSet(s3_data=INPUTS, num_records=1, feature_dim=1)
    tuner.fit(records, mini_batch_size=9999)

    _, _, tune_kwargs = sagemaker_session.tune.mock_calls[0]

    assert tune_kwargs["job_name"].startswith("pca")
    assert tune_kwargs["early_stopping_type"] == "Auto"


def test_fit_mxnet_with_vpc_config(sagemaker_session, tuner):
    subnets = ["foo"]
    security_group_ids = ["bar"]

    pca = PCA(
        ROLE,
        TRAIN_INSTANCE_COUNT,
        TRAIN_INSTANCE_TYPE,
        NUM_COMPONENTS,
        base_job_name="pca",
        sagemaker_session=sagemaker_session,
        subnets=subnets,
        security_group_ids=security_group_ids,
    )
    tuner.estimator = pca

    records = RecordSet(s3_data=INPUTS, num_records=1, feature_dim=1)
    tuner.fit(records, mini_batch_size=9999)

    _, _, tune_kwargs = sagemaker_session.tune.mock_calls[0]
    assert tune_kwargs["vpc_config"] == {"Subnets": subnets, "SecurityGroupIds": security_group_ids}


def test_s3_input_mode(sagemaker_session, tuner):
    expected_input_mode = "Pipe"

    script_path = os.path.join(DATA_DIR, "mxnet_mnist", "failure_script.py")
    mxnet = MXNet(
        entry_point=script_path,
        role=ROLE,
        framework_version=FRAMEWORK_VERSION,
        train_instance_count=TRAIN_INSTANCE_COUNT,
        train_instance_type=TRAIN_INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
    )
    tuner.estimator = mxnet

    tags = [{"Name": "some-tag-without-a-value"}]
    tuner.tags = tags

    hyperparameter_ranges = {
        "num_components": IntegerParameter(2, 4),
        "algorithm_mode": CategoricalParameter(["regular", "randomized"]),
    }
    tuner._hyperparameter_ranges = hyperparameter_ranges

    tuner.fit(inputs=s3_input("s3://mybucket/train_manifest", input_mode=expected_input_mode))

    actual_input_mode = sagemaker_session.method_calls[1][2]["input_mode"]
    assert actual_input_mode == expected_input_mode


def test_fit_pca_with_inter_container_traffic_encryption_flag(sagemaker_session, tuner):
    pca = PCA(
        ROLE,
        TRAIN_INSTANCE_COUNT,
        TRAIN_INSTANCE_TYPE,
        NUM_COMPONENTS,
        base_job_name="pca",
        sagemaker_session=sagemaker_session,
        encrypt_inter_container_traffic=True,
    )

    tuner.estimator = pca

    records = RecordSet(s3_data=INPUTS, num_records=1, feature_dim=1)
    tuner.fit(records, mini_batch_size=9999)

    _, _, tune_kwargs = sagemaker_session.tune.mock_calls[0]

    assert tune_kwargs["job_name"].startswith("pca")
    assert tune_kwargs["encrypt_inter_container_traffic"] is True


def test_attach_tuning_job_with_estimator_from_hyperparameters(sagemaker_session):
    job_details = copy.deepcopy(TUNING_JOB_DETAILS)
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_tuning_job", return_value=job_details
    )
    tuner = HyperparameterTuner.attach(JOB_NAME, sagemaker_session=sagemaker_session)

    assert tuner.latest_tuning_job.name == JOB_NAME
    assert tuner.objective_metric_name == OBJECTIVE_METRIC_NAME
    assert tuner.max_jobs == 1
    assert tuner.max_parallel_jobs == 1
    assert tuner.metric_definitions == METRIC_DEFINITIONS
    assert tuner.strategy == "Bayesian"
    assert tuner.objective_type == "Minimize"
    assert tuner.early_stopping_type == "Off"

    assert isinstance(tuner.estimator, PCA)
    assert tuner.estimator.role == ROLE
    assert tuner.estimator.train_instance_count == 1
    assert tuner.estimator.train_max_run == 24 * 60 * 60
    assert tuner.estimator.input_mode == "File"
    assert tuner.estimator.output_path == BUCKET_NAME
    assert tuner.estimator.output_kms_key == ""

    assert "_tuning_objective_metric" not in tuner.estimator.hyperparameters()
    assert tuner.estimator.hyperparameters()["num_components"] == "10"


def test_attach_tuning_job_with_estimator_from_hyperparameters_with_early_stopping(
    sagemaker_session
):
    job_details = copy.deepcopy(TUNING_JOB_DETAILS)
    job_details["HyperParameterTuningJobConfig"]["TrainingJobEarlyStoppingType"] = "Auto"
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_tuning_job", return_value=job_details
    )
    tuner = HyperparameterTuner.attach(JOB_NAME, sagemaker_session=sagemaker_session)

    assert tuner.latest_tuning_job.name == JOB_NAME
    assert tuner.early_stopping_type == "Auto"

    assert isinstance(tuner.estimator, PCA)


def test_attach_tuning_job_with_job_details(sagemaker_session):
    job_details = copy.deepcopy(TUNING_JOB_DETAILS)
    HyperparameterTuner.attach(
        JOB_NAME, sagemaker_session=sagemaker_session, job_details=job_details
    )
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job.assert_not_called


def test_attach_tuning_job_with_estimator_from_image(sagemaker_session):
    job_details = copy.deepcopy(TUNING_JOB_DETAILS)
    job_details["TrainingJobDefinition"]["AlgorithmSpecification"][
        "TrainingImage"
    ] = "1111.amazonaws.com/pca:1"
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_tuning_job", return_value=job_details
    )

    tuner = HyperparameterTuner.attach(JOB_NAME, sagemaker_session=sagemaker_session)
    assert isinstance(tuner.estimator, PCA)


def test_attach_tuning_job_with_estimator_from_kwarg(sagemaker_session):
    job_details = copy.deepcopy(TUNING_JOB_DETAILS)
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_tuning_job", return_value=job_details
    )
    tuner = HyperparameterTuner.attach(
        JOB_NAME, sagemaker_session=sagemaker_session, estimator_cls="sagemaker.estimator.Estimator"
    )
    assert isinstance(tuner.estimator, Estimator)


def test_attach_with_no_specified_estimator(sagemaker_session):
    job_details = copy.deepcopy(TUNING_JOB_DETAILS)
    del job_details["TrainingJobDefinition"]["StaticHyperParameters"]["sagemaker_estimator_module"]
    del job_details["TrainingJobDefinition"]["StaticHyperParameters"][
        "sagemaker_estimator_class_name"
    ]
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_tuning_job", return_value=job_details
    )

    tuner = HyperparameterTuner.attach(JOB_NAME, sagemaker_session=sagemaker_session)
    assert isinstance(tuner.estimator, Estimator)


def test_attach_with_warm_start_config(sagemaker_session):
    warm_start_config = WarmStartConfig(
        warm_start_type=WarmStartTypes.TRANSFER_LEARNING, parents={"p1", "p2"}
    )
    job_details = copy.deepcopy(TUNING_JOB_DETAILS)
    job_details["WarmStartConfig"] = warm_start_config.to_input_req()

    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_tuning_job", return_value=job_details
    )

    tuner = HyperparameterTuner.attach(JOB_NAME, sagemaker_session=sagemaker_session)
    assert tuner.warm_start_config.type == warm_start_config.type
    assert tuner.warm_start_config.parents == warm_start_config.parents


def test_serialize_parameter_ranges(tuner):
    hyperparameter_ranges = tuner.hyperparameter_ranges()

    for key, value in HYPERPARAMETER_RANGES.items():
        assert hyperparameter_ranges[value.__name__ + "ParameterRanges"][0]["Name"] == key


def test_analytics(tuner):
    tuner.latest_tuning_job = _TuningJob(tuner.sagemaker_session, "testjob")
    tuner_analytics = tuner.analytics()
    assert tuner_analytics is not None
    assert tuner_analytics.name.find("testjob") > -1


def test_serialize_categorical_ranges_for_frameworks(sagemaker_session, tuner):
    tuner.estimator = MXNet(
        entry_point=SCRIPT_NAME,
        role=ROLE,
        framework_version=FRAMEWORK_VERSION,
        train_instance_count=TRAIN_INSTANCE_COUNT,
        train_instance_type=TRAIN_INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
    )

    hyperparameter_ranges = tuner.hyperparameter_ranges()

    assert hyperparameter_ranges["CategoricalParameterRanges"][0]["Name"] == "blank"
    assert hyperparameter_ranges["CategoricalParameterRanges"][0]["Values"] == ['"0"', '"5"']


def test_serialize_nonexistent_parameter_ranges(tuner):
    temp_hyperparameter_ranges = HYPERPARAMETER_RANGES.copy()
    parameter_type = temp_hyperparameter_ranges["validated"].__name__

    temp_hyperparameter_ranges["validated"] = None
    tuner._hyperparameter_ranges = temp_hyperparameter_ranges

    ranges = tuner.hyperparameter_ranges()
    assert len(ranges.keys()) == 3
    assert not ranges[parameter_type + "ParameterRanges"]


def test_stop_tuning_job(sagemaker_session, tuner):
    sagemaker_session.stop_tuning_job = Mock(name="stop_hyper_parameter_tuning_job")
    tuner.latest_tuning_job = _TuningJob(sagemaker_session, JOB_NAME)

    tuner.stop_tuning_job()

    sagemaker_session.stop_tuning_job.assert_called_once_with(name=JOB_NAME)


def test_stop_tuning_job_no_tuning_job(tuner):
    with pytest.raises(ValueError) as e:
        tuner.stop_tuning_job()
    assert "No tuning job available" in str(e)


def test_best_tuning_job(tuner):
    tuning_job_description = {"BestTrainingJob": {"TrainingJobName": JOB_NAME}}

    tuner.estimator.sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_hyper_parameter_tuning_job", return_value=tuning_job_description
    )

    tuner.latest_tuning_job = _TuningJob(tuner.estimator.sagemaker_session, JOB_NAME)
    best_training_job = tuner.best_training_job()

    assert best_training_job == JOB_NAME
    tuner.estimator.sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job.assert_called_once_with(
        HyperParameterTuningJobName=JOB_NAME
    )


def test_best_tuning_job_no_latest_job(tuner):
    with pytest.raises(Exception) as e:
        tuner.best_training_job()

    assert "No tuning job available" in str(e)


def test_best_tuning_job_no_best_job(tuner):
    tuning_job_description = {"BestTrainingJob": {"Mock": None}}

    tuner.estimator.sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_hyper_parameter_tuning_job", return_value=tuning_job_description
    )

    tuner.latest_tuning_job = _TuningJob(tuner.estimator.sagemaker_session, JOB_NAME)

    with pytest.raises(Exception) as e:
        tuner.best_training_job()

    tuner.estimator.sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job.assert_called_once_with(
        HyperParameterTuningJobName=JOB_NAME
    )
    assert "Best training job not available for tuning job:" in str(e)


def test_deploy_default(tuner):
    returned_training_job_description = {
        "AlgorithmSpecification": {
            "TrainingInputMode": "File",
            "TrainingImage": IMAGE_NAME,
            "MetricDefinitions": METRIC_DEFINITIONS,
        },
        "HyperParameters": {
            "sagemaker_submit_directory": '"s3://some/sourcedir.tar.gz"',
            "checkpoint_path": '"s3://other/1508872349"',
            "sagemaker_program": '"iris-dnn-classifier.py"',
            "sagemaker_enable_cloudwatch_metrics": "false",
            "sagemaker_container_log_level": '"logging.INFO"',
            "sagemaker_job_name": '"neo"',
            "training_steps": "100",
            "_tuning_objective_metric": "Validation-accuracy",
        },
        "RoleArn": ROLE,
        "ResourceConfig": {
            "VolumeSizeInGB": 30,
            "InstanceCount": 1,
            "InstanceType": "ml.c4.xlarge",
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 24 * 60 * 60},
        "TrainingJobName": "neo",
        "TrainingJobStatus": "Completed",
        "TrainingJobArn": "arn:aws:sagemaker:us-west-2:336:training-job/neo",
        "OutputDataConfig": {"KmsKeyId": "", "S3OutputPath": "s3://place/output/neo"},
        "TrainingJobOutput": {"S3TrainingJobOutput": "s3://here/output.tar.gz"},
        "ModelArtifacts": {"S3ModelArtifacts": MODEL_DATA},
    }
    tuning_job_description = {"BestTrainingJob": {"TrainingJobName": JOB_NAME}}
    returned_list_tags = {"Tags": [{"Key": "TagtestKey", "Value": "TagtestValue"}]}

    tuner.estimator.sagemaker_session.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job", return_value=returned_training_job_description
    )
    tuner.estimator.sagemaker_session.sagemaker_client.list_tags = Mock(
        name="list_tags", return_value=returned_list_tags
    )
    tuner.estimator.sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_hyper_parameter_tuning_job", return_value=tuning_job_description
    )
    tuner.estimator.sagemaker_session.log_for_jobs = Mock(name="log_for_jobs")

    tuner.latest_tuning_job = _TuningJob(tuner.estimator.sagemaker_session, JOB_NAME)
    predictor = tuner.deploy(TRAIN_INSTANCE_COUNT, TRAIN_INSTANCE_TYPE)

    tuner.estimator.sagemaker_session.create_model.assert_called_once()
    args = tuner.estimator.sagemaker_session.create_model.call_args[0]

    assert args[0] == "neo"
    assert args[1] == ROLE
    assert args[2]["Image"] == IMAGE_NAME
    assert args[2]["ModelDataUrl"] == MODEL_DATA

    assert isinstance(predictor, RealTimePredictor)
    assert predictor.endpoint.startswith(JOB_NAME)
    assert predictor.sagemaker_session == tuner.estimator.sagemaker_session


def test_wait(tuner):
    tuner.latest_tuning_job = _TuningJob(tuner.estimator.sagemaker_session, JOB_NAME)
    tuner.estimator.sagemaker_session.wait_for_tuning_job = Mock(name="wait_for_tuning_job")

    tuner.wait()

    tuner.estimator.sagemaker_session.wait_for_tuning_job.assert_called_once_with(JOB_NAME)


def test_delete_endpoint(tuner):
    tuner.latest_tuning_job = _TuningJob(tuner.estimator.sagemaker_session, JOB_NAME)

    tuning_job_description = {"BestTrainingJob": {"TrainingJobName": JOB_NAME}}
    tuner.estimator.sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_hyper_parameter_tuning_job", return_value=tuning_job_description
    )

    tuner.delete_endpoint()
    tuner.sagemaker_session.delete_endpoint.assert_called_with(JOB_NAME)


def test_fit_no_inputs(tuner, sagemaker_session):
    script_path = os.path.join(DATA_DIR, "mxnet_mnist", "failure_script.py")
    tuner.estimator = MXNet(
        entry_point=script_path,
        role=ROLE,
        framework_version=FRAMEWORK_VERSION,
        train_instance_count=TRAIN_INSTANCE_COUNT,
        train_instance_type=TRAIN_INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
    )

    tuner.fit()

    _, _, tune_kwargs = sagemaker_session.tune.mock_calls[0]

    assert tune_kwargs["input_config"] is None


def test_identical_dataset_and_algorithm_tuner(sagemaker_session):
    job_details = copy.deepcopy(TUNING_JOB_DETAILS)
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_tuning_job", return_value=job_details
    )

    tuner = HyperparameterTuner.attach(JOB_NAME, sagemaker_session=sagemaker_session)
    parent_tuner = tuner.identical_dataset_and_algorithm_tuner(additional_parents={"p1", "p2"})
    assert parent_tuner.warm_start_config.type == WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM
    assert parent_tuner.warm_start_config.parents == {tuner.latest_tuning_job.name, "p1", "p2"}


def test_transfer_learning_tuner_with_estimator(sagemaker_session, estimator):
    job_details = copy.deepcopy(TUNING_JOB_DETAILS)
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_tuning_job", return_value=job_details
    )

    tuner = HyperparameterTuner.attach(JOB_NAME, sagemaker_session=sagemaker_session)
    parent_tuner = tuner.transfer_learning_tuner(
        additional_parents={"p1", "p2"}, estimator=estimator
    )

    assert parent_tuner.warm_start_config.type == WarmStartTypes.TRANSFER_LEARNING
    assert parent_tuner.warm_start_config.parents == {tuner.latest_tuning_job.name, "p1", "p2"}
    assert parent_tuner.estimator == estimator and parent_tuner.estimator != tuner.estimator


def test_transfer_learning_tuner(sagemaker_session):
    job_details = copy.deepcopy(TUNING_JOB_DETAILS)
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_tuning_job", return_value=job_details
    )

    tuner = HyperparameterTuner.attach(JOB_NAME, sagemaker_session=sagemaker_session)
    parent_tuner = tuner.transfer_learning_tuner(additional_parents={"p1", "p2"})

    assert parent_tuner.warm_start_config.type == WarmStartTypes.TRANSFER_LEARNING
    assert parent_tuner.warm_start_config.parents == {tuner.latest_tuning_job.name, "p1", "p2"}
    assert parent_tuner.estimator == tuner.estimator


#################################################################################
# _ParameterRange Tests


def test_continuous_parameter():
    cont_param = ContinuousParameter(0.1, 1e-2)
    assert isinstance(cont_param, ParameterRange)
    assert cont_param.__name__ == "Continuous"


def test_continuous_parameter_ranges():
    cont_param = ContinuousParameter(0.1, 1e-2)
    ranges = cont_param.as_tuning_range("some")
    assert len(ranges.keys()) == 4
    assert ranges["Name"] == "some"
    assert ranges["MinValue"] == "0.1"
    assert ranges["MaxValue"] == "0.01"
    assert ranges["ScalingType"] == "Auto"


def test_continuous_parameter_scaling_type():
    cont_param = ContinuousParameter(0.1, 2, scaling_type="ReverseLogarithmic")
    cont_range = cont_param.as_tuning_range("range")
    assert cont_range["ScalingType"] == "ReverseLogarithmic"


def test_integer_parameter():
    int_param = IntegerParameter(1, 2)
    assert isinstance(int_param, ParameterRange)
    assert int_param.__name__ == "Integer"


def test_integer_parameter_ranges():
    int_param = IntegerParameter(1, 2)
    ranges = int_param.as_tuning_range("some")
    assert len(ranges.keys()) == 4
    assert ranges["Name"] == "some"
    assert ranges["MinValue"] == "1"
    assert ranges["MaxValue"] == "2"
    assert ranges["ScalingType"] == "Auto"


def test_integer_parameter_scaling_type():
    int_param = IntegerParameter(2, 3, scaling_type="Linear")
    int_range = int_param.as_tuning_range("range")
    assert int_range["ScalingType"] == "Linear"


def test_categorical_parameter_list():
    cat_param = CategoricalParameter(["a", "z"])
    assert isinstance(cat_param, ParameterRange)
    assert cat_param.__name__ == "Categorical"


def test_categorical_parameter_list_ranges():
    cat_param = CategoricalParameter([1, 10])
    ranges = cat_param.as_tuning_range("some")
    assert len(ranges.keys()) == 2
    assert ranges["Name"] == "some"
    assert ranges["Values"] == ["1", "10"]


def test_categorical_parameter_value():
    cat_param = CategoricalParameter("a")
    assert isinstance(cat_param, ParameterRange)


def test_categorical_parameter_value_ranges():
    cat_param = CategoricalParameter("a")
    ranges = cat_param.as_tuning_range("some")
    assert len(ranges.keys()) == 2
    assert ranges["Name"] == "some"
    assert ranges["Values"] == ["a"]


#################################################################################
# _TuningJob Tests


def test_start_new(tuner, sagemaker_session):
    tuning_job = _TuningJob(sagemaker_session, JOB_NAME)

    tuner.static_hyperparameters = {}
    started_tuning_job = tuning_job.start_new(tuner, INPUTS)

    assert started_tuning_job.sagemaker_session == sagemaker_session
    sagemaker_session.tune.assert_called_once()


def test_stop(sagemaker_session):
    tuning_job = _TuningJob(sagemaker_session, JOB_NAME)
    tuning_job.stop()

    sagemaker_session.stop_tuning_job.assert_called_once_with(name=JOB_NAME)


def test_tuning_job_wait(sagemaker_session):
    sagemaker_session.wait_for_tuning_job = Mock(name="wait_for_tuning_job")

    tuning_job = _TuningJob(sagemaker_session, JOB_NAME)
    tuning_job.wait()

    sagemaker_session.wait_for_tuning_job.assert_called_once_with(JOB_NAME)


#################################################################################
# WarmStartConfig Tests


@pytest.mark.parametrize(
    "type, parents",
    [
        (WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM, {"p1", "p2", "p3"}),
        (WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM, {"p1", "p3", "p3"}),
        (WarmStartTypes.TRANSFER_LEARNING, {"p3"}),
    ],
)
def test_warm_start_config_init(type, parents):
    warm_start_config = WarmStartConfig(warm_start_type=type, parents=parents)

    assert warm_start_config.type == type, "Warm start type initialization failed."
    assert warm_start_config.parents == set(
        parents
    ), "Warm start parents config initialization failed."

    warm_start_config_req = warm_start_config.to_input_req()
    assert warm_start_config.type == WarmStartTypes(warm_start_config_req["WarmStartType"])
    for parent in warm_start_config_req["ParentHyperParameterTuningJobs"]:
        assert parent["HyperParameterTuningJobName"] in parents


@pytest.mark.parametrize(
    "type, parents",
    [
        ("InvalidType", {"p1", "p2", "p3"}),
        (None, {"p1", "p2", "p3"}),
        ("", {"p1", "p2", "p3"}),
        (WarmStartTypes.TRANSFER_LEARNING, None),
        (WarmStartTypes.TRANSFER_LEARNING, {}),
    ],
)
def test_warm_start_config_init_negative(type, parents):
    with pytest.raises(ValueError):
        WarmStartConfig(warm_start_type=type, parents=parents)


@pytest.mark.parametrize(
    "warm_start_config_req",
    [
        ({}),
        (None),
        ({"WarmStartType": "TransferLearning"}),
        ({"ParentHyperParameterTuningJobs": []}),
    ],
)
def test_prepare_warm_start_config_cls_negative(warm_start_config_req):
    warm_start_config = WarmStartConfig.from_job_desc(warm_start_config_req)
    assert warm_start_config is None, "Warm start config should be None for invalid type/parents"


@pytest.mark.parametrize(
    "warm_start_config_req",
    [
        (
            {
                "WarmStartType": "TransferLearning",
                "ParentHyperParameterTuningJobs": [
                    {"HyperParameterTuningJobName": "p1"},
                    {"HyperParameterTuningJobName": "p2"},
                ],
            }
        ),
        (
            {
                "WarmStartType": "IdenticalDataAndAlgorithm",
                "ParentHyperParameterTuningJobs": [
                    {"HyperParameterTuningJobName": "p1"},
                    {"HyperParameterTuningJobName": "p1"},
                ],
            }
        ),
    ],
)
def test_prepare_warm_start_config_cls(warm_start_config_req):
    warm_start_config = WarmStartConfig.from_job_desc(warm_start_config_req)

    assert warm_start_config.type == WarmStartTypes(
        warm_start_config_req["WarmStartType"]
    ), "Warm start type initialization failed."

    for p in warm_start_config_req["ParentHyperParameterTuningJobs"]:
        assert (
            p["HyperParameterTuningJobName"] in warm_start_config.parents
        ), "Warm start parents config initialization failed."


@pytest.mark.parametrize("additional_parents", [{"p1", "p2"}, {}, None])
def test_create_identical_dataset_and_algorithm_tuner(sagemaker_session, additional_parents):
    job_details = copy.deepcopy(TUNING_JOB_DETAILS)
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_tuning_job", return_value=job_details
    )

    tuner = create_identical_dataset_and_algorithm_tuner(
        parent=JOB_NAME, additional_parents=additional_parents, sagemaker_session=sagemaker_session
    )

    assert tuner.warm_start_config.type == WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM
    if additional_parents:
        additional_parents.add(JOB_NAME)
        assert tuner.warm_start_config.parents == additional_parents
    else:
        assert tuner.warm_start_config.parents == {JOB_NAME}


@pytest.mark.parametrize("additional_parents", [{"p1", "p2"}, {}, None])
def test_create_transfer_learning_tuner(sagemaker_session, estimator, additional_parents):
    job_details = copy.deepcopy(TUNING_JOB_DETAILS)
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_tuning_job", return_value=job_details
    )

    tuner = create_transfer_learning_tuner(
        parent=JOB_NAME,
        additional_parents=additional_parents,
        sagemaker_session=sagemaker_session,
        estimator=estimator,
    )

    assert tuner.warm_start_config.type == WarmStartTypes.TRANSFER_LEARNING
    assert tuner.estimator == estimator
    if additional_parents:
        additional_parents.add(JOB_NAME)
        assert tuner.warm_start_config.parents == additional_parents
    else:
        assert tuner.warm_start_config.parents == {JOB_NAME}
