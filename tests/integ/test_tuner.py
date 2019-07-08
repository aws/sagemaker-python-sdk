# Copyright 2017-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import gzip
import json
import os
import pickle
import sys
import time

import numpy as np
import pytest
from botocore.exceptions import ClientError
from tests.integ import DATA_DIR, PYTHON_VERSION, TUNING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.record_set import prepare_record_set_from_local_files
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name
from tests.integ import vpc_test_utils

from sagemaker import KMeans, LDA, RandomCutForest
from sagemaker.amazon.amazon_estimator import registry
from sagemaker.amazon.common import read_records
from sagemaker.chainer import Chainer
from sagemaker.estimator import Estimator
from sagemaker.mxnet.estimator import MXNet
from sagemaker.predictor import json_deserializer
from sagemaker.pytorch import PyTorch
from sagemaker.tensorflow import TensorFlow
from sagemaker.tuner import (
    IntegerParameter,
    ContinuousParameter,
    CategoricalParameter,
    HyperparameterTuner,
    WarmStartConfig,
    WarmStartTypes,
    create_transfer_learning_tuner,
    create_identical_dataset_and_algorithm_tuner,
)
from sagemaker.utils import unique_name_from_base

DATA_PATH = os.path.join(DATA_DIR, "iris", "data")


@pytest.fixture(scope="module")
def kmeans_train_set(sagemaker_session):
    data_path = os.path.join(DATA_DIR, "one_p_mnist", "mnist.pkl.gz")
    pickle_args = {} if sys.version_info.major == 2 else {"encoding": "latin1"}
    # Load the data into memory as numpy arrays
    with gzip.open(data_path, "rb") as f:
        train_set, _, _ = pickle.load(f, **pickle_args)

    return train_set


@pytest.fixture(scope="module")
def kmeans_estimator(sagemaker_session):
    kmeans = KMeans(
        role="SageMakerRole",
        train_instance_count=1,
        train_instance_type="ml.c4.xlarge",
        k=10,
        sagemaker_session=sagemaker_session,
        output_path="s3://{}/".format(sagemaker_session.default_bucket()),
    )
    # set kmeans specific hp
    kmeans.init_method = "random"
    kmeans.max_iterators = 1
    kmeans.tol = 1
    kmeans.num_trials = 1
    kmeans.local_init_method = "kmeans++"
    kmeans.half_life_time_size = 1
    kmeans.epochs = 1

    return kmeans


@pytest.fixture(scope="module")
def hyperparameter_ranges():
    return {
        "extra_center_factor": IntegerParameter(1, 10),
        "mini_batch_size": IntegerParameter(10, 100),
        "epochs": IntegerParameter(1, 2),
        "init_method": CategoricalParameter(["kmeans++", "random"]),
    }


def _tune_and_deploy(
    kmeans_estimator,
    kmeans_train_set,
    sagemaker_session,
    hyperparameter_ranges=None,
    job_name=None,
    warm_start_config=None,
    early_stopping_type="Off",
):
    tuner = _tune(
        kmeans_estimator,
        kmeans_train_set,
        hyperparameter_ranges=hyperparameter_ranges,
        warm_start_config=warm_start_config,
        job_name=job_name,
        early_stopping_type=early_stopping_type,
    )
    _deploy(kmeans_train_set, sagemaker_session, tuner, early_stopping_type)


def _deploy(kmeans_train_set, sagemaker_session, tuner, early_stopping_type):
    best_training_job = tuner.best_training_job()
    assert tuner.early_stopping_type == early_stopping_type
    with timeout_and_delete_endpoint_by_name(best_training_job, sagemaker_session):
        predictor = tuner.deploy(1, "ml.c4.xlarge")

        result = predictor.predict(kmeans_train_set[0][:10])

        assert len(result) == 10
        for record in result:
            assert record.label["closest_cluster"] is not None
            assert record.label["distance_to_cluster"] is not None


def _tune(
    kmeans_estimator,
    kmeans_train_set,
    tuner=None,
    hyperparameter_ranges=None,
    job_name=None,
    warm_start_config=None,
    wait_till_terminal=True,
    max_jobs=2,
    max_parallel_jobs=2,
    early_stopping_type="Off",
):
    with timeout(minutes=TUNING_DEFAULT_TIMEOUT_MINUTES):

        if not tuner:
            tuner = HyperparameterTuner(
                estimator=kmeans_estimator,
                objective_metric_name="test:msd",
                hyperparameter_ranges=hyperparameter_ranges,
                objective_type="Minimize",
                max_jobs=max_jobs,
                max_parallel_jobs=max_parallel_jobs,
                warm_start_config=warm_start_config,
                early_stopping_type=early_stopping_type,
            )

        records = kmeans_estimator.record_set(kmeans_train_set[0][:100])
        test_record_set = kmeans_estimator.record_set(kmeans_train_set[0][:100], channel="test")

        tuner.fit([records, test_record_set], job_name=job_name)
        print("Started hyperparameter tuning job with name:" + tuner.latest_tuning_job.name)

        if wait_till_terminal:
            tuner.wait()

    return tuner


@pytest.mark.canary_quick
def test_tuning_kmeans(
    sagemaker_session, kmeans_train_set, kmeans_estimator, hyperparameter_ranges
):
    job_name = unique_name_from_base("test-tune-kmeans")
    _tune_and_deploy(
        kmeans_estimator,
        kmeans_train_set,
        sagemaker_session,
        hyperparameter_ranges=hyperparameter_ranges,
        job_name=job_name,
    )


def test_tuning_kmeans_identical_dataset_algorithm_tuner_raw(
    sagemaker_session, kmeans_train_set, kmeans_estimator, hyperparameter_ranges
):
    parent_tuning_job_name = unique_name_from_base("kmeans-identical", max_length=32)
    child_tuning_job_name = unique_name_from_base("c-kmeans-identical", max_length=32)
    _tune(
        kmeans_estimator,
        kmeans_train_set,
        job_name=parent_tuning_job_name,
        hyperparameter_ranges=hyperparameter_ranges,
        max_parallel_jobs=1,
        max_jobs=1,
    )
    child_tuner = _tune(
        kmeans_estimator,
        kmeans_train_set,
        job_name=child_tuning_job_name,
        hyperparameter_ranges=hyperparameter_ranges,
        warm_start_config=WarmStartConfig(
            warm_start_type=WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM,
            parents=[parent_tuning_job_name],
        ),
        max_parallel_jobs=1,
        max_jobs=1,
    )

    child_warm_start_config_response = WarmStartConfig.from_job_desc(
        sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=child_tuning_job_name
        )["WarmStartConfig"]
    )

    assert child_warm_start_config_response.type == child_tuner.warm_start_config.type
    assert child_warm_start_config_response.parents == child_tuner.warm_start_config.parents


def test_tuning_kmeans_identical_dataset_algorithm_tuner(
    sagemaker_session, kmeans_train_set, kmeans_estimator, hyperparameter_ranges
):
    """Tests Identical dataset and algorithm use case with one parent and child job launched with
        .identical_dataset_and_algorithm_tuner() """

    parent_tuning_job_name = unique_name_from_base("km-iden1-parent", max_length=32)
    child_tuning_job_name = unique_name_from_base("km-iden1-child", max_length=32)

    parent_tuner = _tune(
        kmeans_estimator,
        kmeans_train_set,
        job_name=parent_tuning_job_name,
        hyperparameter_ranges=hyperparameter_ranges,
    )

    child_tuner = parent_tuner.identical_dataset_and_algorithm_tuner()
    _tune(
        kmeans_estimator,
        kmeans_train_set,
        job_name=child_tuning_job_name,
        tuner=child_tuner,
        max_parallel_jobs=1,
        max_jobs=1,
    )

    child_warm_start_config_response = WarmStartConfig.from_job_desc(
        sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=child_tuning_job_name
        )["WarmStartConfig"]
    )

    assert child_warm_start_config_response.type == child_tuner.warm_start_config.type
    assert child_warm_start_config_response.parents == child_tuner.warm_start_config.parents


def test_create_tuning_kmeans_identical_dataset_algorithm_tuner(
    sagemaker_session, kmeans_train_set, kmeans_estimator, hyperparameter_ranges
):
    """Tests Identical dataset and algorithm use case with one parent and child job launched with
        .create_identical_dataset_and_algorithm_tuner() """

    parent_tuning_job_name = unique_name_from_base("km-iden2-parent", max_length=32)
    child_tuning_job_name = unique_name_from_base("km-iden2-child", max_length=32)

    parent_tuner = _tune(
        kmeans_estimator,
        kmeans_train_set,
        job_name=parent_tuning_job_name,
        hyperparameter_ranges=hyperparameter_ranges,
        max_parallel_jobs=1,
        max_jobs=1,
    )

    child_tuner = create_identical_dataset_and_algorithm_tuner(
        parent=parent_tuner.latest_tuning_job.name, sagemaker_session=sagemaker_session
    )

    _tune(
        kmeans_estimator,
        kmeans_train_set,
        job_name=child_tuning_job_name,
        tuner=child_tuner,
        max_parallel_jobs=1,
        max_jobs=1,
    )

    child_warm_start_config_response = WarmStartConfig.from_job_desc(
        sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=child_tuning_job_name
        )["WarmStartConfig"]
    )

    assert child_warm_start_config_response.type == child_tuner.warm_start_config.type
    assert child_warm_start_config_response.parents == child_tuner.warm_start_config.parents


def test_transfer_learning_tuner(
    sagemaker_session, kmeans_train_set, kmeans_estimator, hyperparameter_ranges
):
    """Tests Transfer learning use case with one parent and child job launched with
        .transfer_learning_tuner() """

    parent_tuning_job_name = unique_name_from_base("km-tran1-parent", max_length=32)
    child_tuning_job_name = unique_name_from_base("km-tran1-child", max_length=32)

    parent_tuner = _tune(
        kmeans_estimator,
        kmeans_train_set,
        job_name=parent_tuning_job_name,
        hyperparameter_ranges=hyperparameter_ranges,
        max_jobs=1,
        max_parallel_jobs=1,
    )

    child_tuner = parent_tuner.transfer_learning_tuner()
    _tune(
        kmeans_estimator,
        kmeans_train_set,
        job_name=child_tuning_job_name,
        tuner=child_tuner,
        max_parallel_jobs=1,
        max_jobs=1,
    )

    child_warm_start_config_response = WarmStartConfig.from_job_desc(
        sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=child_tuning_job_name
        )["WarmStartConfig"]
    )

    assert child_warm_start_config_response.type == child_tuner.warm_start_config.type
    assert child_warm_start_config_response.parents == child_tuner.warm_start_config.parents


def test_create_transfer_learning_tuner(
    sagemaker_session, kmeans_train_set, kmeans_estimator, hyperparameter_ranges
):
    """Tests Transfer learning use case with two parents and child job launched with
        create_transfer_learning_tuner() """
    parent_tuning_job_name_1 = unique_name_from_base("km-tran2-parent1", max_length=32)
    parent_tuning_job_name_2 = unique_name_from_base("km-tran2-parent2", max_length=32)
    child_tuning_job_name = unique_name_from_base("km-tran2-child", max_length=32)

    parent_tuner_1 = _tune(
        kmeans_estimator,
        kmeans_train_set,
        job_name=parent_tuning_job_name_1,
        hyperparameter_ranges=hyperparameter_ranges,
        max_parallel_jobs=1,
        max_jobs=1,
    )

    parent_tuner_2 = _tune(
        kmeans_estimator,
        kmeans_train_set,
        job_name=parent_tuning_job_name_2,
        hyperparameter_ranges=hyperparameter_ranges,
        max_parallel_jobs=1,
        max_jobs=1,
    )

    child_tuner = create_transfer_learning_tuner(
        parent=parent_tuner_1.latest_tuning_job.name,
        sagemaker_session=sagemaker_session,
        estimator=kmeans_estimator,
        additional_parents={parent_tuner_2.latest_tuning_job.name},
    )

    _tune(kmeans_estimator, kmeans_train_set, job_name=child_tuning_job_name, tuner=child_tuner)

    child_warm_start_config_response = WarmStartConfig.from_job_desc(
        sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=child_tuning_job_name
        )["WarmStartConfig"]
    )

    assert child_warm_start_config_response.type == child_tuner.warm_start_config.type
    assert child_warm_start_config_response.parents == child_tuner.warm_start_config.parents


def test_tuning_kmeans_identical_dataset_algorithm_tuner_from_non_terminal_parent(
    sagemaker_session, kmeans_train_set, kmeans_estimator, hyperparameter_ranges
):
    """Tests Identical dataset and algorithm use case with one non terminal parent and child job launched with
    .identical_dataset_and_algorithm_tuner() """
    parent_tuning_job_name = unique_name_from_base("km-non-term", max_length=32)
    child_tuning_job_name = unique_name_from_base("km-non-term-child", max_length=32)

    parent_tuner = _tune(
        kmeans_estimator,
        kmeans_train_set,
        job_name=parent_tuning_job_name,
        hyperparameter_ranges=hyperparameter_ranges,
        wait_till_terminal=False,
        max_parallel_jobs=1,
        max_jobs=1,
    )

    child_tuner = parent_tuner.identical_dataset_and_algorithm_tuner()
    with pytest.raises(ClientError):
        _tune(
            kmeans_estimator,
            kmeans_train_set,
            job_name=child_tuning_job_name,
            tuner=child_tuner,
            max_parallel_jobs=1,
            max_jobs=1,
        )


def test_tuning_lda(sagemaker_session):
    with timeout(minutes=TUNING_DEFAULT_TIMEOUT_MINUTES):
        data_path = os.path.join(DATA_DIR, "lda")
        data_filename = "nips-train_1.pbr"

        with open(os.path.join(data_path, data_filename), "rb") as f:
            all_records = read_records(f)

        # all records must be same
        feature_num = int(all_records[0].features["values"].float32_tensor.shape[0])

        lda = LDA(
            role="SageMakerRole",
            train_instance_type="ml.c4.xlarge",
            num_topics=10,
            sagemaker_session=sagemaker_session,
        )

        record_set = prepare_record_set_from_local_files(
            data_path, lda.data_location, len(all_records), feature_num, sagemaker_session
        )
        test_record_set = prepare_record_set_from_local_files(
            data_path, lda.data_location, len(all_records), feature_num, sagemaker_session
        )
        test_record_set.channel = "test"

        # specify which hp you want to optimize over
        hyperparameter_ranges = {
            "alpha0": ContinuousParameter(1, 10),
            "num_topics": IntegerParameter(1, 2),
        }
        objective_metric_name = "test:pwll"

        tuner = HyperparameterTuner(
            estimator=lda,
            objective_metric_name=objective_metric_name,
            hyperparameter_ranges=hyperparameter_ranges,
            objective_type="Maximize",
            max_jobs=2,
            max_parallel_jobs=2,
            early_stopping_type="Auto",
        )

        tuning_job_name = unique_name_from_base("test-lda", max_length=32)
        tuner.fit([record_set, test_record_set], mini_batch_size=1, job_name=tuning_job_name)

        latest_tuning_job_name = tuner.latest_tuning_job.name

        print("Started hyperparameter tuning job with name:" + latest_tuning_job_name)

        time.sleep(15)
        tuner.wait()

    desc = tuner.latest_tuning_job.sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=latest_tuning_job_name
    )
    assert desc["HyperParameterTuningJobConfig"]["TrainingJobEarlyStoppingType"] == "Auto"

    best_training_job = tuner.best_training_job()
    with timeout_and_delete_endpoint_by_name(best_training_job, sagemaker_session):
        predictor = tuner.deploy(1, "ml.c4.xlarge")
        predict_input = np.random.rand(1, feature_num)
        result = predictor.predict(predict_input)

        assert len(result) == 1
        for record in result:
            assert record.label["topic_mixture"] is not None


def test_stop_tuning_job(sagemaker_session):
    feature_num = 14
    train_input = np.random.rand(1000, feature_num)

    rcf = RandomCutForest(
        role="SageMakerRole",
        train_instance_count=1,
        train_instance_type="ml.c4.xlarge",
        num_trees=50,
        num_samples_per_tree=20,
        sagemaker_session=sagemaker_session,
    )

    records = rcf.record_set(train_input)
    records.distribution = "FullyReplicated"

    test_records = rcf.record_set(train_input, channel="test")
    test_records.distribution = "FullyReplicated"

    hyperparameter_ranges = {
        "num_trees": IntegerParameter(50, 100),
        "num_samples_per_tree": IntegerParameter(1, 2),
    }

    objective_metric_name = "test:f1"
    tuner = HyperparameterTuner(
        estimator=rcf,
        objective_metric_name=objective_metric_name,
        hyperparameter_ranges=hyperparameter_ranges,
        objective_type="Maximize",
        max_jobs=2,
        max_parallel_jobs=2,
    )

    tuning_job_name = unique_name_from_base("test-randomcutforest", max_length=32)
    tuner.fit([records, test_records], tuning_job_name)

    time.sleep(15)

    latest_tuning_job_name = tuner.latest_tuning_job.name

    print("Attempting to stop {}".format(latest_tuning_job_name))

    tuner.stop_tuning_job()

    desc = tuner.latest_tuning_job.sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=latest_tuning_job_name
    )
    assert desc["HyperParameterTuningJobStatus"] == "Stopping"


@pytest.mark.canary_quick
def test_tuning_mxnet(sagemaker_session, mxnet_full_version):
    with timeout(minutes=TUNING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        estimator = MXNet(
            entry_point=script_path,
            role="SageMakerRole",
            py_version=PYTHON_VERSION,
            train_instance_count=1,
            train_instance_type="ml.m4.xlarge",
            framework_version=mxnet_full_version,
            sagemaker_session=sagemaker_session,
        )

        hyperparameter_ranges = {"learning-rate": ContinuousParameter(0.01, 0.2)}
        objective_metric_name = "Validation-accuracy"
        metric_definitions = [
            {"Name": "Validation-accuracy", "Regex": "Validation-accuracy=([0-9\\.]+)"}
        ]
        tuner = HyperparameterTuner(
            estimator,
            objective_metric_name,
            hyperparameter_ranges,
            metric_definitions,
            max_jobs=4,
            max_parallel_jobs=2,
        )

        train_input = estimator.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
        )
        test_input = estimator.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
        )

        tuning_job_name = unique_name_from_base("tune-mxnet", max_length=32)
        tuner.fit({"train": train_input, "test": test_input}, job_name=tuning_job_name)

        print("Started hyperparameter tuning job with name:" + tuning_job_name)

        time.sleep(15)
        tuner.wait()

    best_training_job = tuner.best_training_job()
    with timeout_and_delete_endpoint_by_name(best_training_job, sagemaker_session):
        predictor = tuner.deploy(1, "ml.c4.xlarge")
        data = np.zeros(shape=(1, 1, 28, 28))
        predictor.predict(data)


@pytest.mark.canary_quick
def test_tuning_tf_script_mode(sagemaker_session):
    resource_path = os.path.join(DATA_DIR, "tensorflow_mnist")
    script_path = os.path.join(resource_path, "mnist.py")

    estimator = TensorFlow(
        entry_point=script_path,
        role="SageMakerRole",
        train_instance_count=1,
        train_instance_type="ml.m4.xlarge",
        script_mode=True,
        sagemaker_session=sagemaker_session,
        py_version=PYTHON_VERSION,
        framework_version=TensorFlow.LATEST_VERSION,
    )

    hyperparameter_ranges = {"epochs": IntegerParameter(1, 2)}
    objective_metric_name = "accuracy"
    metric_definitions = [{"Name": objective_metric_name, "Regex": "accuracy = ([0-9\\.]+)"}]

    tuner = HyperparameterTuner(
        estimator,
        objective_metric_name,
        hyperparameter_ranges,
        metric_definitions,
        max_jobs=2,
        max_parallel_jobs=2,
    )

    with timeout(minutes=TUNING_DEFAULT_TIMEOUT_MINUTES):
        inputs = estimator.sagemaker_session.upload_data(
            path=os.path.join(resource_path, "data"), key_prefix="scriptmode/mnist"
        )

        tuning_job_name = unique_name_from_base("tune-tf-script-mode", max_length=32)
        tuner.fit(inputs, job_name=tuning_job_name)

        print("Started hyperparameter tuning job with name: " + tuning_job_name)

        time.sleep(15)
        tuner.wait()


@pytest.mark.canary_quick
@pytest.mark.skipif(PYTHON_VERSION != "py2", reason="TensorFlow image supports only python 2.")
def test_tuning_tf(sagemaker_session):
    with timeout(minutes=TUNING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, "iris", "iris-dnn-classifier.py")

        estimator = TensorFlow(
            entry_point=script_path,
            role="SageMakerRole",
            training_steps=1,
            evaluation_steps=1,
            hyperparameters={"input_tensor_name": "inputs"},
            train_instance_count=1,
            train_instance_type="ml.c4.xlarge",
            sagemaker_session=sagemaker_session,
        )

        inputs = sagemaker_session.upload_data(path=DATA_PATH, key_prefix="integ-test-data/tf_iris")
        hyperparameter_ranges = {"learning_rate": ContinuousParameter(0.05, 0.2)}

        objective_metric_name = "loss"
        metric_definitions = [{"Name": "loss", "Regex": "loss = ([0-9\\.]+)"}]

        tuner = HyperparameterTuner(
            estimator,
            objective_metric_name,
            hyperparameter_ranges,
            metric_definitions,
            objective_type="Minimize",
            max_jobs=2,
            max_parallel_jobs=2,
        )

        tuning_job_name = unique_name_from_base("tune-tf", max_length=32)
        tuner.fit(inputs, job_name=tuning_job_name)

        print("Started hyperparameter tuning job with name:" + tuning_job_name)

        time.sleep(15)
        tuner.wait()

    best_training_job = tuner.best_training_job()
    with timeout_and_delete_endpoint_by_name(best_training_job, sagemaker_session):
        predictor = tuner.deploy(1, "ml.c4.xlarge")

        features = [6.4, 3.2, 4.5, 1.5]
        dict_result = predictor.predict({"inputs": features})
        print("predict result: {}".format(dict_result))
        list_result = predictor.predict(features)
        print("predict result: {}".format(list_result))

        assert dict_result == list_result


@pytest.mark.skipif(PYTHON_VERSION != "py2", reason="TensorFlow image supports only python 2.")
def test_tuning_tf_vpc_multi(sagemaker_session):
    """Test Tensorflow multi-instance using the same VpcConfig for training and inference"""
    instance_type = "ml.c4.xlarge"
    instance_count = 2

    script_path = os.path.join(DATA_DIR, "iris", "iris-dnn-classifier.py")

    ec2_client = sagemaker_session.boto_session.client("ec2")
    subnet_ids, security_group_id = vpc_test_utils.get_or_create_vpc_resources(
        ec2_client, sagemaker_session.boto_region_name
    )
    vpc_test_utils.setup_security_group_for_encryption(ec2_client, security_group_id)

    estimator = TensorFlow(
        entry_point=script_path,
        role="SageMakerRole",
        training_steps=1,
        evaluation_steps=1,
        hyperparameters={"input_tensor_name": "inputs"},
        train_instance_count=instance_count,
        train_instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        base_job_name="test-vpc-tf",
        subnets=subnet_ids,
        security_group_ids=[security_group_id],
        encrypt_inter_container_traffic=True,
    )

    inputs = sagemaker_session.upload_data(path=DATA_PATH, key_prefix="integ-test-data/tf_iris")
    hyperparameter_ranges = {"learning_rate": ContinuousParameter(0.05, 0.2)}

    objective_metric_name = "loss"
    metric_definitions = [{"Name": "loss", "Regex": "loss = ([0-9\\.]+)"}]

    tuner = HyperparameterTuner(
        estimator,
        objective_metric_name,
        hyperparameter_ranges,
        metric_definitions,
        objective_type="Minimize",
        max_jobs=2,
        max_parallel_jobs=2,
    )

    tuning_job_name = unique_name_from_base("tune-tf", max_length=32)
    with timeout(minutes=TUNING_DEFAULT_TIMEOUT_MINUTES):
        tuner.fit(inputs, job_name=tuning_job_name)

        print("Started hyperparameter tuning job with name:" + tuning_job_name)

        time.sleep(15)
        tuner.wait()


@pytest.mark.canary_quick
def test_tuning_chainer(sagemaker_session):
    with timeout(minutes=TUNING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, "chainer_mnist", "mnist.py")
        data_path = os.path.join(DATA_DIR, "chainer_mnist")

        estimator = Chainer(
            entry_point=script_path,
            role="SageMakerRole",
            py_version=PYTHON_VERSION,
            train_instance_count=1,
            train_instance_type="ml.c4.xlarge",
            sagemaker_session=sagemaker_session,
            hyperparameters={"epochs": 1},
        )

        train_input = estimator.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/chainer_mnist/train"
        )
        test_input = estimator.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/chainer_mnist/test"
        )

        hyperparameter_ranges = {"alpha": ContinuousParameter(0.001, 0.005)}

        objective_metric_name = "Validation-accuracy"
        metric_definitions = [
            {
                "Name": "Validation-accuracy",
                "Regex": r"\[J1\s+\d\.\d+\s+\d\.\d+\s+\d\.\d+\s+(\d\.\d+)",
            }
        ]

        tuner = HyperparameterTuner(
            estimator,
            objective_metric_name,
            hyperparameter_ranges,
            metric_definitions,
            max_jobs=2,
            max_parallel_jobs=2,
        )

        tuning_job_name = unique_name_from_base("chainer", max_length=32)
        tuner.fit({"train": train_input, "test": test_input}, job_name=tuning_job_name)

        print("Started hyperparameter tuning job with name:" + tuning_job_name)

        time.sleep(15)
        tuner.wait()

    best_training_job = tuner.best_training_job()
    with timeout_and_delete_endpoint_by_name(best_training_job, sagemaker_session):
        predictor = tuner.deploy(1, "ml.c4.xlarge")

        batch_size = 100
        data = np.zeros((batch_size, 784), dtype="float32")
        output = predictor.predict(data)
        assert len(output) == batch_size

        data = np.zeros((batch_size, 1, 28, 28), dtype="float32")
        output = predictor.predict(data)
        assert len(output) == batch_size

        data = np.zeros((batch_size, 28, 28), dtype="float32")
        output = predictor.predict(data)
        assert len(output) == batch_size


@pytest.mark.canary_quick
def test_attach_tuning_pytorch(sagemaker_session):
    mnist_dir = os.path.join(DATA_DIR, "pytorch_mnist")
    mnist_script = os.path.join(mnist_dir, "mnist.py")

    estimator = PyTorch(
        entry_point=mnist_script,
        role="SageMakerRole",
        train_instance_count=1,
        py_version=PYTHON_VERSION,
        train_instance_type="ml.c4.xlarge",
        sagemaker_session=sagemaker_session,
    )

    with timeout(minutes=TUNING_DEFAULT_TIMEOUT_MINUTES):
        objective_metric_name = "evaluation-accuracy"
        metric_definitions = [
            {"Name": "evaluation-accuracy", "Regex": r"Overall test accuracy: (\d+)"}
        ]
        hyperparameter_ranges = {"batch-size": IntegerParameter(50, 100)}

        tuner = HyperparameterTuner(
            estimator,
            objective_metric_name,
            hyperparameter_ranges,
            metric_definitions,
            max_jobs=2,
            max_parallel_jobs=2,
            early_stopping_type="Auto",
        )

        training_data = estimator.sagemaker_session.upload_data(
            path=os.path.join(mnist_dir, "training"),
            key_prefix="integ-test-data/pytorch_mnist/training",
        )

        tuning_job_name = unique_name_from_base("pytorch", max_length=32)
        tuner.fit({"training": training_data}, job_name=tuning_job_name)

        print("Started hyperparameter tuning job with name:" + tuning_job_name)

        time.sleep(15)
        tuner.wait()

    endpoint_name = tuning_job_name
    model_name = "model-name-1"
    attached_tuner = HyperparameterTuner.attach(
        tuning_job_name, sagemaker_session=sagemaker_session
    )
    assert attached_tuner.early_stopping_type == "Auto"

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        predictor = attached_tuner.deploy(
            1, "ml.c4.xlarge", endpoint_name=endpoint_name, model_name=model_name
        )
        data = np.zeros(shape=(1, 1, 28, 28), dtype=np.float32)
        predictor.predict(data)

        batch_size = 100
        data = np.random.rand(batch_size, 1, 28, 28).astype(np.float32)
        output = predictor.predict(data)

        assert output.shape == (batch_size, 10)
        _assert_model_name_match(sagemaker_session.sagemaker_client, endpoint_name, model_name)


@pytest.mark.canary_quick
def test_tuning_byo_estimator(sagemaker_session):
    """Use Factorization Machines algorithm as an example here.

    First we need to prepare data for training. We take standard data set, convert it to the
    format that the algorithm can process and upload it to S3.
    Then we create the Estimator and set hyperparamets as required by the algorithm.
    Next, we can call fit() with path to the S3.
    Later the trained model is deployed and prediction is called against the endpoint.
    Default predictor is updated with json serializer and deserializer.
    """
    image_name = registry(sagemaker_session.boto_session.region_name) + "/factorization-machines:1"
    training_data_path = os.path.join(DATA_DIR, "dummy_tensor")

    with timeout(minutes=TUNING_DEFAULT_TIMEOUT_MINUTES):
        data_path = os.path.join(DATA_DIR, "one_p_mnist", "mnist.pkl.gz")
        pickle_args = {} if sys.version_info.major == 2 else {"encoding": "latin1"}

        with gzip.open(data_path, "rb") as f:
            train_set, _, _ = pickle.load(f, **pickle_args)

        prefix = "test_byo_estimator"
        key = "recordio-pb-data"
        s3_train_data = sagemaker_session.upload_data(
            path=training_data_path, key_prefix=os.path.join(prefix, "train", key)
        )

        estimator = Estimator(
            image_name=image_name,
            role="SageMakerRole",
            train_instance_count=1,
            train_instance_type="ml.c4.xlarge",
            sagemaker_session=sagemaker_session,
        )

        estimator.set_hyperparameters(
            num_factors=10, feature_dim=784, mini_batch_size=100, predictor_type="binary_classifier"
        )

        hyperparameter_ranges = {"mini_batch_size": IntegerParameter(100, 200)}

        tuner = HyperparameterTuner(
            estimator=estimator,
            objective_metric_name="test:binary_classification_accuracy",
            hyperparameter_ranges=hyperparameter_ranges,
            max_jobs=2,
            max_parallel_jobs=2,
        )

        tuner.fit(
            {"train": s3_train_data, "test": s3_train_data},
            include_cls_metadata=False,
            job_name=unique_name_from_base("byo", 32),
        )

        print("Started hyperparameter tuning job with name:" + tuner.latest_tuning_job.name)

        time.sleep(15)
        tuner.wait()

    best_training_job = tuner.best_training_job()
    with timeout_and_delete_endpoint_by_name(best_training_job, sagemaker_session):
        predictor = tuner.deploy(1, "ml.m4.xlarge", endpoint_name=best_training_job)
        predictor.serializer = _fm_serializer
        predictor.content_type = "application/json"
        predictor.deserializer = json_deserializer

        result = predictor.predict(train_set[0][:10])

        assert len(result["predictions"]) == 10
        for prediction in result["predictions"]:
            assert prediction["score"] is not None


# Serializer for the Factorization Machines predictor (for BYO example)
def _fm_serializer(data):
    js = {"instances": []}
    for row in data:
        js["instances"].append({"features": row.tolist()})
    return json.dumps(js)


def _assert_model_name_match(sagemaker_client, endpoint_config_name, model_name):
    endpoint_config_description = sagemaker_client.describe_endpoint_config(
        EndpointConfigName=endpoint_config_name
    )
    assert model_name == endpoint_config_description["ProductionVariants"][0]["ModelName"]
