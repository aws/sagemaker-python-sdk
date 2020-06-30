# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import re

import pytest
from mock import Mock, patch

from sagemaker import Predictor, utils
from sagemaker.amazon.amazon_estimator import RecordSet
from sagemaker.estimator import Framework
from sagemaker.mxnet import MXNet
from sagemaker.parameter import ParameterRange
from sagemaker.session import s3_input
from sagemaker.tuner import (
    _TuningJob,
    create_identical_dataset_and_algorithm_tuner,
    create_transfer_learning_tuner,
    HyperparameterTuner,
)

from .tuner_test_utils import *  # noqa: F403


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    sms = Mock(name="sagemaker_session", boto_session=boto_mock, s3_client=None, s3_resource=None)
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
    tuner._prepare_for_tuning()

    assert tuner._current_job_name.startswith(IMAGE_NAME)

    assert len(tuner.static_hyperparameters) == 1
    assert tuner.static_hyperparameters["another_one"] == "0"


def test_prepare_for_tuning_with_amazon_estimator(tuner, sagemaker_session):
    tuner.estimator = PCA(
        ROLE,
        TRAIN_INSTANCE_COUNT,
        TRAIN_INSTANCE_TYPE,
        NUM_COMPONENTS,
        sagemaker_session=sagemaker_session,
    )

    tuner._prepare_for_tuning()
    assert "sagemaker_estimator_class_name" not in tuner.static_hyperparameters
    assert "sagemaker_estimator_module" not in tuner.static_hyperparameters


def test_prepare_for_tuning_include_estimator_cls(tuner):
    tuner._prepare_for_tuning(include_cls_metadata=True)
    assert "sagemaker_estimator_class_name" in tuner.static_hyperparameters
    assert "sagemaker_estimator_module" in tuner.static_hyperparameters


def test_prepare_for_tuning_with_job_name(tuner):
    static_hyperparameters = {"validated": 1, "another_one": 0}
    tuner.estimator.set_hyperparameters(**static_hyperparameters)

    tuner._prepare_for_tuning(job_name="some-other-job-name")
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

    tuner._hyperparameter_ranges = HYPERPARAMETER_RANGES_TWO

    records = RecordSet(s3_data=INPUTS, num_records=1, feature_dim=1)
    tuner.fit(records, mini_batch_size=9999)

    _, _, tune_kwargs = sagemaker_session.create_tuning_job.mock_calls[0]

    assert tuner.estimator.mini_batch_size == 9999

    assert tune_kwargs["job_name"].startswith("pca")
    assert tune_kwargs["tags"] == tags

    assert len(tune_kwargs["tuning_config"]["parameter_ranges"]["IntegerParameterRanges"]) == 1
    assert tune_kwargs["tuning_config"]["early_stopping_type"] == "Off"
    assert tuner.estimator.mini_batch_size == 9999

    assert "training_config" in tune_kwargs
    assert "training_config_list" not in tune_kwargs

    assert len(tune_kwargs["training_config"]["static_hyperparameters"]) == 4
    assert tune_kwargs["training_config"]["static_hyperparameters"]["extra_components"] == "5"

    assert "estimator_name" not in tune_kwargs["training_config"]
    assert "objective_type" not in tune_kwargs["training_config"]
    assert "objective_metric_name" not in tune_kwargs["training_config"]
    assert "parameter_ranges" not in tune_kwargs["training_config"]


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

    _, _, tune_kwargs = sagemaker_session.create_tuning_job.mock_calls[0]

    assert tune_kwargs["job_name"].startswith("pca")
    assert tune_kwargs["tuning_config"]["early_stopping_type"] == "Auto"


def test_fit_pca_with_vpc_config(sagemaker_session, tuner):
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

    _, _, tune_kwargs = sagemaker_session.create_tuning_job.mock_calls[0]

    assert tune_kwargs["training_config"]["vpc_config"] == {
        "Subnets": subnets,
        "SecurityGroupIds": security_group_ids,
    }


def test_s3_input_mode(sagemaker_session, tuner):
    expected_input_mode = "Pipe"

    script_path = os.path.join(DATA_DIR, "mxnet_mnist", "failure_script.py")
    mxnet = MXNet(
        entry_point=script_path,
        framework_version=FRAMEWORK_VERSION,
        py_version=PY_VERSION,
        role=ROLE,
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

    actual_input_mode = sagemaker_session.method_calls[1][2]["training_config"]["input_mode"]
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

    _, _, tune_kwargs = sagemaker_session.create_tuning_job.mock_calls[0]

    assert tune_kwargs["job_name"].startswith("pca")
    assert tune_kwargs["training_config"]["encrypt_inter_container_traffic"] is True


@pytest.mark.parametrize(
    "inputs,include_cls_metadata,estimator_kwargs,error_message",
    [
        (
            RecordSet(s3_data=INPUTS, num_records=1, feature_dim=1),
            {ESTIMATOR_NAME_TWO: True},
            {},
            re.compile(
                "Argument 'inputs' must be a dictionary using \\['estimator_name', 'estimator_name_two'\\] as keys"
            ),
        ),
        (
            {ESTIMATOR_NAME: RecordSet(s3_data=INPUTS, num_records=1, feature_dim=1)},
            False,
            {},
            re.compile(
                "Argument 'include_cls_metadata' must be a dictionary using \\['estimator_name', "
                "'estimator_name_two'\\] as keys"
            ),
        ),
        (
            {ESTIMATOR_NAME: RecordSet(s3_data=INPUTS, num_records=1, feature_dim=1)},
            {ESTIMATOR_NAME_TWO: True},
            False,
            re.compile(
                "Argument 'estimator_kwargs' must be a dictionary using \\['estimator_name', "
                "'estimator_name_two'\\] as keys"
            ),
        ),
        (
            {
                ESTIMATOR_NAME: RecordSet(s3_data=INPUTS, num_records=1, feature_dim=1),
                "Invalid estimator": RecordSet(s3_data=INPUTS, num_records=10, feature_dim=5),
            },
            {ESTIMATOR_NAME_TWO: True},
            None,
            re.compile(
                "The keys of argument 'inputs' must be a subset of \\['estimator_name', 'estimator_name_two'\\]"
            ),
        ),
    ],
)
def test_fit_multi_estimators_invalid_inputs(
    sagemaker_session, inputs, include_cls_metadata, estimator_kwargs, error_message
):
    (tuner, estimator_one, estimator_two) = _create_multi_estimator_tuner(sagemaker_session)

    with pytest.raises(ValueError, match=error_message):
        tuner.fit(
            inputs=inputs,
            include_cls_metadata=include_cls_metadata,
            estimator_kwargs=estimator_kwargs,
        )


def test_fit_multi_estimators(sagemaker_session):

    (tuner, estimator_one, estimator_two) = _create_multi_estimator_tuner(sagemaker_session)

    records = {ESTIMATOR_NAME_TWO: RecordSet(s3_data=INPUTS, num_records=1, feature_dim=1)}

    estimator_kwargs = {ESTIMATOR_NAME_TWO: {"mini_batch_size": 4000}}

    tuner.fit(inputs=records, include_cls_metadata={}, estimator_kwargs=estimator_kwargs)

    _, _, tune_kwargs = sagemaker_session.create_tuning_job.mock_calls[0]

    assert tune_kwargs["job_name"].startswith(BASE_JOB_NAME)
    assert tune_kwargs["tags"] == TAGS

    assert tune_kwargs["tuning_config"]["strategy"] == STRATEGY
    assert tune_kwargs["tuning_config"]["max_jobs"] == MAX_JOBS
    assert tune_kwargs["tuning_config"]["max_parallel_jobs"] == MAX_PARALLEL_JOBS
    assert tune_kwargs["tuning_config"]["early_stopping_type"] == EARLY_STOPPING_TYPE

    assert "tuning_objective" not in tune_kwargs["tuning_config"]
    assert "parameter_ranges" not in tune_kwargs["tuning_config"]

    assert "training_config" not in tune_kwargs
    assert "training_config_list" in tune_kwargs

    assert len(tune_kwargs["training_config_list"]) == 2

    training_config_one = tune_kwargs["training_config_list"][0]
    training_config_two = tune_kwargs["training_config_list"][1]

    assert training_config_one["estimator_name"] == ESTIMATOR_NAME
    assert training_config_one["objective_type"] == "Minimize"
    assert training_config_one["objective_metric_name"] == OBJECTIVE_METRIC_NAME
    assert training_config_one["input_config"] is None
    assert training_config_one["image"] == estimator_one.train_image()
    assert training_config_one["metric_definitions"] == METRIC_DEFINITIONS
    assert (
        training_config_one["static_hyperparameters"]["sagemaker_estimator_module"]
        == '"sagemaker.mxnet.estimator"'
    )
    _assert_parameter_ranges(
        HYPERPARAMETER_RANGES,
        training_config_one["parameter_ranges"],
        isinstance(estimator_one, Framework),
    )

    assert training_config_two["estimator_name"] == ESTIMATOR_NAME_TWO
    assert training_config_two["objective_type"] == "Minimize"
    assert training_config_two["objective_metric_name"] == OBJECTIVE_METRIC_NAME_TWO
    assert len(training_config_two["input_config"]) == 1
    assert training_config_two["input_config"][0]["DataSource"]["S3DataSource"]["S3Uri"] == INPUTS
    assert training_config_two["image"] == estimator_two.train_image()
    assert training_config_two["metric_definitions"] is None
    assert training_config_two["static_hyperparameters"]["mini_batch_size"] == "4000"
    _assert_parameter_ranges(
        HYPERPARAMETER_RANGES_TWO,
        training_config_two["parameter_ranges"],
        isinstance(estimator_two, Framework),
    )


def _create_multi_estimator_tuner(sagemaker_session):
    mxnet_script_path = os.path.join(DATA_DIR, "mxnet_mnist", "failure_script.py")
    mxnet = MXNet(
        entry_point=mxnet_script_path,
        framework_version=FRAMEWORK_VERSION,
        py_version=PY_VERSION,
        role=ROLE,
        train_instance_count=TRAIN_INSTANCE_COUNT,
        train_instance_type=TRAIN_INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
    )

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

    tuner = HyperparameterTuner.create(
        base_tuning_job_name=BASE_JOB_NAME,
        estimator_dict={ESTIMATOR_NAME: mxnet, ESTIMATOR_NAME_TWO: pca},
        objective_metric_name_dict={
            ESTIMATOR_NAME: OBJECTIVE_METRIC_NAME,
            ESTIMATOR_NAME_TWO: OBJECTIVE_METRIC_NAME_TWO,
        },
        hyperparameter_ranges_dict={
            ESTIMATOR_NAME: HYPERPARAMETER_RANGES,
            ESTIMATOR_NAME_TWO: HYPERPARAMETER_RANGES_TWO,
        },
        metric_definitions_dict={ESTIMATOR_NAME: METRIC_DEFINITIONS},
        strategy=STRATEGY,
        objective_type=OBJECTIVE_TYPE,
        max_jobs=MAX_JOBS,
        max_parallel_jobs=MAX_PARALLEL_JOBS,
        tags=TAGS,
        warm_start_config=WARM_START_CONFIG,
        early_stopping_type=EARLY_STOPPING_TYPE,
    )

    return tuner, mxnet, pca


def _assert_parameter_ranges(expected, actual, is_framework_estimator):
    continuous_ranges = []
    integer_ranges = []
    categorical_ranges = []
    for (name, param_range) in expected.items():
        if isinstance(param_range, ContinuousParameter):
            continuous_ranges.append(param_range.as_tuning_range(name))
        elif isinstance(param_range, IntegerParameter):
            integer_ranges.append(param_range.as_tuning_range(name))
        else:
            categorical_range = (
                param_range.as_json_range(name)
                if is_framework_estimator
                else param_range.as_tuning_range(name)
            )
            categorical_ranges.append(categorical_range)

    assert continuous_ranges == actual["ContinuousParameterRanges"]
    assert integer_ranges == actual["IntegerParameterRanges"]
    assert categorical_ranges == actual["CategoricalParameterRanges"]


def test_attach_tuning_job_with_estimator_from_hyperparameters(sagemaker_session):
    job_details = copy.deepcopy(TUNING_JOB_DETAILS)
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_tuning_job", return_value=job_details
    )
    tuner = HyperparameterTuner.attach(JOB_NAME, sagemaker_session=sagemaker_session)

    assert tuner.latest_tuning_job.name == JOB_NAME
    assert tuner.base_tuning_job_name == JOB_NAME
    assert tuner._current_job_name == JOB_NAME

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
    sagemaker_session,
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


def test_attach_with_generated_job_name(sagemaker_session):
    job_name = utils.name_from_base(BASE_JOB_NAME, max_length=32, short=True)

    job_details = copy.deepcopy(TUNING_JOB_DETAILS)
    job_details["HyperParameterTuningJobName"] = job_name

    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_tuning_job", return_value=job_details
    )

    tuner = HyperparameterTuner.attach(job_name, sagemaker_session=sagemaker_session)
    assert BASE_JOB_NAME == tuner.base_tuning_job_name


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


def test_attach_tuning_job_with_multi_estimators(sagemaker_session):
    job_details = copy.deepcopy(MULTI_ALGO_TUNING_JOB_DETAILS)
    tuner = HyperparameterTuner.attach(
        JOB_NAME,
        sagemaker_session=sagemaker_session,
        estimator_cls={ESTIMATOR_NAME_TWO: "sagemaker.estimator.Estimator"},
        job_details=job_details,
    )

    assert tuner.latest_tuning_job.name == JOB_NAME
    assert tuner.strategy == "Bayesian"
    assert tuner.objective_type == "Minimize"
    assert tuner.max_jobs == 4
    assert tuner.max_parallel_jobs == 2
    assert tuner.early_stopping_type == "Off"
    assert tuner.warm_start_config is None

    assert tuner.estimator is None
    assert tuner.objective_metric_name is None
    assert tuner._hyperparameter_ranges is None
    assert tuner.metric_definitions is None

    assert tuner.estimator_dict is not None
    assert tuner.objective_metric_name_dict is not None
    assert tuner._hyperparameter_ranges_dict is not None
    assert tuner.metric_definitions_dict is not None

    assert len(tuner.estimator_dict) == 2

    estimator_names = tuner.estimator_dict.keys()
    assert tuner.objective_metric_name_dict.keys() == estimator_names
    assert tuner._hyperparameter_ranges_dict.keys() == estimator_names
    assert set(tuner.metric_definitions_dict.keys()).issubset(set(estimator_names))

    assert isinstance(tuner.estimator_dict[ESTIMATOR_NAME], PCA)
    assert isinstance(tuner.estimator_dict[ESTIMATOR_NAME_TWO], Estimator)

    assert tuner.objective_metric_name_dict[ESTIMATOR_NAME] == OBJECTIVE_METRIC_NAME
    assert tuner.objective_metric_name_dict[ESTIMATOR_NAME_TWO] == OBJECTIVE_METRIC_NAME_TWO

    parameter_ranges_one = tuner._hyperparameter_ranges_dict[ESTIMATOR_NAME]
    assert len(parameter_ranges_one) == 1
    assert isinstance(parameter_ranges_one.get("mini_batch_size", None), IntegerParameter)

    parameter_ranges_two = tuner._hyperparameter_ranges_dict[ESTIMATOR_NAME_TWO]
    assert len(parameter_ranges_two) == 2
    assert isinstance(parameter_ranges_two.get("kernel", None), CategoricalParameter)
    assert isinstance(parameter_ranges_two.get("tree_count", None), IntegerParameter)

    assert len(tuner.metric_definitions_dict) == 1
    assert tuner.metric_definitions_dict[ESTIMATOR_NAME_TWO] == METRIC_DEFINITIONS


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
        framework_version=FRAMEWORK_VERSION,
        py_version=PY_VERSION,
        role=ROLE,
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
    tuning_job_description = {"TuningJobName": "a_job"}

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


def test_best_estimator(tuner):
    tuner.sagemaker_session.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job", return_value=TRAINING_JOB_DESCRIPTION
    )

    tuner.sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_hyper_parameter_tuning_job",
        return_value={"BestTrainingJob": {"TrainingJobName": TRAINING_JOB_NAME}},
    )

    tuner.sagemaker_session.sagemaker_client.list_tags = Mock(
        name="list_tags", return_value=LIST_TAGS_RESULT
    )

    tuner.sagemaker_session.log_for_jobs = Mock(name="log_for_jobs")
    tuner.latest_tuning_job = _TuningJob(tuner.sagemaker_session, JOB_NAME)

    best_estimator = tuner.best_estimator()

    assert best_estimator is not None
    assert best_estimator.latest_training_job is not None
    assert best_estimator.latest_training_job.job_name == TRAINING_JOB_NAME
    assert best_estimator.sagemaker_session == tuner.sagemaker_session

    tuner.estimator.sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job.assert_called_once_with(
        HyperParameterTuningJobName=JOB_NAME
    )
    tuner.sagemaker_session.sagemaker_client.describe_training_job.assert_called_once_with(
        TrainingJobName=TRAINING_JOB_NAME
    )


def test_deploy_default(tuner):
    tuner.sagemaker_session.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job", return_value=TRAINING_JOB_DESCRIPTION
    )

    tuner.sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_hyper_parameter_tuning_job",
        return_value={"BestTrainingJob": {"TrainingJobName": TRAINING_JOB_NAME}},
    )

    tuner.sagemaker_session.sagemaker_client.list_tags = Mock(
        name="list_tags", return_value=LIST_TAGS_RESULT
    )

    tuner.sagemaker_session.log_for_jobs = Mock(name="log_for_jobs")

    tuner.latest_tuning_job = _TuningJob(tuner.sagemaker_session, JOB_NAME)
    predictor = tuner.deploy(TRAIN_INSTANCE_COUNT, TRAIN_INSTANCE_TYPE)

    tuner.sagemaker_session.create_model.assert_called_once()
    args = tuner.sagemaker_session.create_model.call_args[0]
    assert args[0].startswith(TRAINING_JOB_NAME)
    assert args[1] == ROLE
    assert args[2]["Image"] == IMAGE_NAME
    assert args[2]["ModelDataUrl"] == MODEL_DATA

    assert isinstance(predictor, Predictor)
    assert predictor.endpoint_name.startswith(TRAINING_JOB_NAME)
    assert predictor.sagemaker_session == tuner.sagemaker_session


def test_deploy_estimator_dict(tuner):
    tuner.estimator_dict = {ESTIMATOR_NAME: tuner.estimator}
    tuner.estimator = None

    tuner.sagemaker_session.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job", return_value=TRAINING_JOB_DESCRIPTION
    )

    tuner.sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_hyper_parameter_tuning_job",
        return_value={
            "BestTrainingJob": {
                "TrainingJobName": TRAINING_JOB_NAME,
                "TrainingJobDefinitionName": ESTIMATOR_NAME,
            }
        },
    )

    tuner.sagemaker_session.sagemaker_client.list_tags = Mock(
        name="list_tags", return_value=LIST_TAGS_RESULT
    )

    tuner.sagemaker_session.log_for_jobs = Mock(name="log_for_jobs")

    tuner.latest_tuning_job = _TuningJob(tuner.sagemaker_session, JOB_NAME)
    predictor = tuner.deploy(TRAIN_INSTANCE_COUNT, TRAIN_INSTANCE_TYPE)

    tuner.sagemaker_session.create_model.assert_called_once()
    args = tuner.sagemaker_session.create_model.call_args[0]
    assert args[0].startswith(TRAINING_JOB_NAME)
    assert args[1] == ROLE
    assert args[2]["Image"] == IMAGE_NAME
    assert args[2]["ModelDataUrl"] == MODEL_DATA

    assert isinstance(predictor, Predictor)
    assert predictor.endpoint_name.startswith(TRAINING_JOB_NAME)
    assert predictor.sagemaker_session == tuner.sagemaker_session


@patch("sagemaker.tuner.HyperparameterTuner.best_estimator")
@patch("sagemaker.tuner.HyperparameterTuner._get_best_training_job")
def test_deploy_optional_params(_get_best_training_job, best_estimator, tuner):
    tuner.fit()

    estimator = Mock()
    best_estimator.return_value = estimator

    training_job = "best-job-ever"
    _get_best_training_job.return_value = training_job

    accelerator = "ml.eia1.medium"
    endpoint_name = "foo"
    model_name = "bar"
    kms_key = "key"
    kwargs = {"some_arg": "some_value"}

    tuner.deploy(
        TRAIN_INSTANCE_COUNT,
        TRAIN_INSTANCE_TYPE,
        accelerator_type=accelerator,
        endpoint_name=endpoint_name,
        wait=False,
        model_name=model_name,
        kms_key=kms_key,
        **kwargs
    )

    best_estimator.assert_called_with(training_job)

    estimator.deploy.assert_called_with(
        initial_instance_count=TRAIN_INSTANCE_COUNT,
        instance_type=TRAIN_INSTANCE_TYPE,
        accelerator_type=accelerator,
        endpoint_name=endpoint_name,
        wait=False,
        model_name=model_name,
        kms_key=kms_key,
        data_capture_config=None,
        **kwargs
    )


def test_wait(tuner):
    tuner.latest_tuning_job = _TuningJob(tuner.estimator.sagemaker_session, JOB_NAME)
    tuner.estimator.sagemaker_session.wait_for_tuning_job = Mock(name="wait_for_tuning_job")

    tuner.wait()

    tuner.estimator.sagemaker_session.wait_for_tuning_job.assert_called_once_with(JOB_NAME)


def test_fit_no_inputs(tuner, sagemaker_session):
    script_path = os.path.join(DATA_DIR, "mxnet_mnist", "failure_script.py")
    tuner.estimator = MXNet(
        entry_point=script_path,
        framework_version=FRAMEWORK_VERSION,
        py_version=PY_VERSION,
        role=ROLE,
        train_instance_count=TRAIN_INSTANCE_COUNT,
        train_instance_type=TRAIN_INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
    )

    tuner.fit()

    _, _, tune_kwargs = sagemaker_session.create_tuning_job.mock_calls[0]

    assert tune_kwargs["training_config"]["input_config"] is None


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


@pytest.mark.parametrize(
    "estimator_dict,obj_metric_name_dict,param_ranges_dict,metric_def_dict",
    [
        (
            {ESTIMATOR_NAME: ESTIMATOR},
            {ESTIMATOR_NAME: OBJECTIVE_METRIC_NAME},
            {ESTIMATOR_NAME: HYPERPARAMETER_RANGES},
            {ESTIMATOR_NAME: METRIC_DEFINITIONS},
        ),
        (
            {ESTIMATOR_NAME: ESTIMATOR, ESTIMATOR_NAME_TWO: ESTIMATOR_TWO},
            {ESTIMATOR_NAME: OBJECTIVE_METRIC_NAME, ESTIMATOR_NAME_TWO: OBJECTIVE_METRIC_NAME_TWO},
            {
                ESTIMATOR_NAME: HYPERPARAMETER_RANGES,
                ESTIMATOR_NAME_TWO: {"gamma": ContinuousParameter(0, 1.5)},
            },
            {ESTIMATOR_NAME: METRIC_DEFINITIONS},
        ),
    ],
)
def test_create_tuner(estimator_dict, obj_metric_name_dict, param_ranges_dict, metric_def_dict):
    tuner = HyperparameterTuner.create(
        base_tuning_job_name=BASE_JOB_NAME,
        estimator_dict=estimator_dict,
        objective_metric_name_dict=obj_metric_name_dict,
        hyperparameter_ranges_dict=param_ranges_dict,
        metric_definitions_dict=metric_def_dict,
        strategy="Bayesian",
        objective_type="Minimize",
        max_jobs=MAX_JOBS,
        max_parallel_jobs=MAX_PARALLEL_JOBS,
        tags=TAGS,
        warm_start_config=WARM_START_CONFIG,
        early_stopping_type="Auto",
    )

    assert tuner is not None

    assert tuner.estimator_dict == estimator_dict
    assert tuner.objective_metric_name_dict == obj_metric_name_dict
    assert tuner._hyperparameter_ranges_dict == param_ranges_dict
    assert tuner.metric_definitions_dict == metric_def_dict

    assert tuner.base_tuning_job_name == BASE_JOB_NAME
    assert tuner.strategy == "Bayesian"
    assert tuner.objective_type == "Minimize"
    assert tuner.max_jobs == MAX_JOBS
    assert tuner.max_parallel_jobs == MAX_PARALLEL_JOBS
    assert tuner.tags == TAGS
    assert tuner.warm_start_config == WARM_START_CONFIG
    assert tuner.early_stopping_type == "Auto"

    assert tuner.sagemaker_session == SAGEMAKER_SESSION


@pytest.mark.parametrize(
    "estimator_dict,obj_metric_name_dict,param_ranges_dict,metric_def_dict,error_message",
    [
        (
            {},
            {ESTIMATOR_NAME: OBJECTIVE_METRIC_NAME},
            {ESTIMATOR_NAME: HYPERPARAMETER_RANGES},
            {ESTIMATOR_NAME: METRIC_DEFINITIONS},
            re.compile("At least one estimator should be provided"),
        ),
        (
            None,
            {ESTIMATOR_NAME: OBJECTIVE_METRIC_NAME},
            {ESTIMATOR_NAME: HYPERPARAMETER_RANGES},
            {ESTIMATOR_NAME: METRIC_DEFINITIONS},
            re.compile("At least one estimator should be provided"),
        ),
        (
            {None: ESTIMATOR},
            {ESTIMATOR_NAME: OBJECTIVE_METRIC_NAME},
            {ESTIMATOR_NAME: HYPERPARAMETER_RANGES},
            {ESTIMATOR_NAME: METRIC_DEFINITIONS},
            "Estimator names cannot be None",
        ),
        (
            {ESTIMATOR_NAME: ESTIMATOR},
            OBJECTIVE_METRIC_NAME,
            {ESTIMATOR_NAME: HYPERPARAMETER_RANGES},
            {ESTIMATOR_NAME: METRIC_DEFINITIONS},
            re.compile(
                "Argument 'objective_metric_name_dict' must be a dictionary using \\['estimator_name'\\] as keys"
            ),
        ),
        (
            {ESTIMATOR_NAME: ESTIMATOR},
            {ESTIMATOR_NAME + "1": OBJECTIVE_METRIC_NAME},
            {ESTIMATOR_NAME: HYPERPARAMETER_RANGES},
            {ESTIMATOR_NAME: METRIC_DEFINITIONS},
            re.compile(
                "The keys of argument 'objective_metric_name_dict' must be the same as \\['estimator_name'\\]"
            ),
        ),
        (
            {ESTIMATOR_NAME: ESTIMATOR},
            {ESTIMATOR_NAME: OBJECTIVE_METRIC_NAME},
            {ESTIMATOR_NAME + "1": HYPERPARAMETER_RANGES},
            {ESTIMATOR_NAME: METRIC_DEFINITIONS},
            re.compile(
                "The keys of argument 'hyperparameter_ranges_dict' must be the same as \\['estimator_name'\\]"
            ),
        ),
        (
            {ESTIMATOR_NAME: ESTIMATOR},
            {ESTIMATOR_NAME: OBJECTIVE_METRIC_NAME},
            {ESTIMATOR_NAME: HYPERPARAMETER_RANGES},
            {ESTIMATOR_NAME + "1": METRIC_DEFINITIONS},
            re.compile(
                "The keys of argument 'metric_definitions_dict' must be a subset of \\['estimator_name'\\]"
            ),
        ),
    ],
)
def test_create_tuner_negative(
    estimator_dict, obj_metric_name_dict, param_ranges_dict, metric_def_dict, error_message
):
    with pytest.raises(ValueError, match=error_message):
        HyperparameterTuner.create(
            base_tuning_job_name=BASE_JOB_NAME,
            estimator_dict=estimator_dict,
            objective_metric_name_dict=obj_metric_name_dict,
            hyperparameter_ranges_dict=param_ranges_dict,
            metric_definitions_dict=metric_def_dict,
            strategy="Bayesian",
            objective_type="Minimize",
            max_jobs=MAX_JOBS,
            max_parallel_jobs=MAX_PARALLEL_JOBS,
            tags=TAGS,
        )


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
    sagemaker_session.create_tuning_job.assert_called_once()


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


@pytest.mark.parametrize(
    "warm_start_type",
    [WarmStartTypes.TRANSFER_LEARNING, WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM],
)
def test_create_warm_start_tuner_with_multi_estimator_dict(
    sagemaker_session, estimator, warm_start_type
):
    job_details = copy.deepcopy(MULTI_ALGO_TUNING_JOB_DETAILS)

    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_tuning_job", return_value=job_details
    )

    additional_parents = {"p1", "p2"}

    with pytest.raises(
        ValueError,
        match="Warm start is not supported currently for tuners with multiple estimators",
    ):
        if warm_start_type == WarmStartTypes.TRANSFER_LEARNING:
            create_transfer_learning_tuner(
                parent=JOB_NAME,
                additional_parents=additional_parents,
                sagemaker_session=sagemaker_session,
                estimator=estimator,
            )
        else:
            create_identical_dataset_and_algorithm_tuner(
                parent=JOB_NAME,
                additional_parents=additional_parents,
                sagemaker_session=sagemaker_session,
            )


@pytest.mark.parametrize(
    "warm_start_type",
    [WarmStartTypes.TRANSFER_LEARNING, WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM],
)
def test_create_warm_start_tuner_with_single_estimator_dict(
    sagemaker_session, estimator, warm_start_type
):
    job_details = _convert_tuning_job_details(TUNING_JOB_DETAILS, ESTIMATOR_NAME)

    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_tuning_job", return_value=job_details
    )

    additional_parents = {"p1", "p2"}

    if warm_start_type == WarmStartTypes.TRANSFER_LEARNING:
        tuner = create_transfer_learning_tuner(
            parent=JOB_NAME,
            additional_parents=additional_parents,
            sagemaker_session=sagemaker_session,
            estimator=estimator,
        )
    else:
        tuner = create_identical_dataset_and_algorithm_tuner(
            parent=JOB_NAME,
            additional_parents=additional_parents,
            sagemaker_session=sagemaker_session,
        )

    assert tuner.warm_start_config.type == warm_start_type

    assert tuner.estimator is None
    assert tuner.estimator_dict is not None

    assert len(tuner.estimator_dict) == 1

    if warm_start_type == WarmStartTypes.TRANSFER_LEARNING:
        assert tuner.estimator_dict[ESTIMATOR_NAME] == estimator
    else:
        assert isinstance(tuner.estimator_dict[ESTIMATOR_NAME], PCA)

    additional_parents.add(JOB_NAME)
    assert tuner.warm_start_config.parents == additional_parents


def test_describe(tuner):
    tuner.describe()
    tuner.sagemaker_session.describe_tuning_job.assert_called_once()


def _convert_tuning_job_details(job_details, estimator_name):
    """Convert a tuning job description using the 'TrainingJobDefinition' field into a new one using a single-item
       'TrainingJobDefinitions' field (list).
    """
    assert "TrainingJobDefinition" in job_details

    job_details_copy = copy.deepcopy(job_details)

    training_details = job_details_copy.pop("TrainingJobDefinition")

    # When the 'TrainingJobDefinitions' field is used, the 'DefinitionName' field is required for each item in it.
    training_details["DefinitionName"] = estimator_name

    # When the 'TrainingJobDefinitions' field is used, tuning objective and parameter ranges must be set in each item
    # in it instead of the tuning job config.
    training_details["TuningObjective"] = job_details_copy["HyperParameterTuningJobConfig"].pop(
        "HyperParameterTuningJobObjective"
    )
    training_details["HyperParameterRanges"] = job_details_copy[
        "HyperParameterTuningJobConfig"
    ].pop("ParameterRanges")

    job_details_copy["TrainingJobDefinitions"] = [training_details]

    return job_details_copy
