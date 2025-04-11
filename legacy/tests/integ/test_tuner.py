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
import time

import numpy as np
import pytest
from botocore.exceptions import ClientError
from packaging.version import Version

import tests.integ
from sagemaker import KMeans, LDA, RandomCutForest, image_uris
from sagemaker.amazon.common import read_records
from sagemaker.chainer import Chainer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.estimator import Estimator
from sagemaker.mxnet.estimator import MXNet
from sagemaker.pytorch import PyTorch
from sagemaker.serializers import SimpleBaseSerializer
from sagemaker.tensorflow import TensorFlow
from sagemaker.tuner import (
    IntegerParameter,
    ContinuousParameter,
    CategoricalParameter,
    HyperparameterTuner,
    InstanceConfig,
    WarmStartConfig,
    WarmStartTypes,
    create_transfer_learning_tuner,
    create_identical_dataset_and_algorithm_tuner,
)
from sagemaker.utils import unique_name_from_base
from tests.integ import (
    datasets,
    vpc_test_utils,
    DATA_DIR,
    TUNING_DEFAULT_TIMEOUT_MINUTES,
)
from tests.integ.record_set import prepare_record_set_from_local_files
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


@pytest.fixture(scope="module")
def kmeans_train_set(sagemaker_session):
    return datasets.one_p_mnist()


@pytest.fixture(scope="module")
def kmeans_estimator(sagemaker_session, cpu_instance_type):
    kmeans = KMeans(
        role="SageMakerRole",
        instance_count=1,
        instance_type=cpu_instance_type,
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
    cpu_instance_type,
    hyperparameter_ranges=None,
    job_name=None,
    warm_start_config=None,
    early_stopping_type="Off",
    instance_configs=None,
    autotune=False,
    hyperparameters_to_keep_static=None,
):
    tuner = _tune(
        kmeans_estimator,
        kmeans_train_set,
        hyperparameter_ranges=hyperparameter_ranges,
        warm_start_config=warm_start_config,
        job_name=job_name,
        early_stopping_type=early_stopping_type,
        instance_configs=instance_configs,
        autotune=autotune,
        hyperparameters_to_keep_static=hyperparameters_to_keep_static,
    )
    _deploy(kmeans_train_set, sagemaker_session, tuner, early_stopping_type, cpu_instance_type)


def _deploy(kmeans_train_set, sagemaker_session, tuner, early_stopping_type, cpu_instance_type):
    best_training_job = tuner.best_training_job()
    assert tuner.early_stopping_type == early_stopping_type
    with timeout_and_delete_endpoint_by_name(best_training_job, sagemaker_session):
        predictor = tuner.deploy(1, cpu_instance_type)

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
    wait=True,
    max_jobs=2,
    max_parallel_jobs=2,
    early_stopping_type="Off",
    instance_configs=None,
    autotune=False,
    hyperparameters_to_keep_static=None,
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
                autotune=autotune,
                hyperparameters_to_keep_static=hyperparameters_to_keep_static,
            )
        tuner.override_resource_config(instance_configs=instance_configs)
        records = kmeans_estimator.record_set(kmeans_train_set[0][:100])
        test_record_set = kmeans_estimator.record_set(kmeans_train_set[0][:100], channel="test")

        print("Started hyperparameter tuning job with name: {}".format(job_name))
        tuner.fit([records, test_record_set], job_name=job_name, wait=wait)

    return tuner


@pytest.mark.release
def test_tuning_kmeans(
    sagemaker_session, kmeans_train_set, kmeans_estimator, hyperparameter_ranges, cpu_instance_type
):
    job_name = unique_name_from_base("test-tune-kmeans")
    _tune_and_deploy(
        kmeans_estimator,
        kmeans_train_set,
        sagemaker_session,
        cpu_instance_type,
        hyperparameter_ranges=hyperparameter_ranges,
        job_name=job_name,
    )


def test_tuning_kmeans_autotune(
    sagemaker_session, kmeans_train_set, kmeans_estimator, hyperparameter_ranges, cpu_instance_type
):
    job_name = unique_name_from_base("test-tune-kmeans-autotune", 32)
    _tune_and_deploy(
        kmeans_estimator,
        kmeans_train_set,
        sagemaker_session,
        cpu_instance_type,
        hyperparameter_ranges=hyperparameter_ranges,
        job_name=job_name,
        autotune=True,
        hyperparameters_to_keep_static=[
            "local_lloyd_init_method",
            "local_lloyd_num_trials",
            "local_lloyd_tol",
        ],
    )


def test_tuning_kmeans_with_instance_configs(
    sagemaker_session, kmeans_train_set, kmeans_estimator, hyperparameter_ranges, cpu_instance_type
):
    job_name = unique_name_from_base("tst-fit")
    _tune_and_deploy(
        kmeans_estimator,
        kmeans_train_set,
        sagemaker_session,
        cpu_instance_type,
        hyperparameter_ranges=hyperparameter_ranges,
        job_name=job_name,
        instance_configs=[
            InstanceConfig(instance_count=1, instance_type="ml.m4.2xlarge", volume_size=30),
            InstanceConfig(instance_count=1, instance_type="ml.m4.xlarge", volume_size=30),
        ],
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
    .identical_dataset_and_algorithm_tuner()"""

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
    .create_identical_dataset_and_algorithm_tuner()"""

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
    .transfer_learning_tuner()"""

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
    create_transfer_learning_tuner()"""
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
    """Tests Identical dataset and algorithm use case with
    one non terminal parent and child job launched with
    .identical_dataset_and_algorithm_tuner()
    """
    parent_tuning_job_name = unique_name_from_base("km-non-term", max_length=32)
    child_tuning_job_name = unique_name_from_base("km-non-term-child", max_length=32)

    parent_tuner = _tune(
        kmeans_estimator,
        kmeans_train_set,
        job_name=parent_tuning_job_name,
        hyperparameter_ranges=hyperparameter_ranges,
        wait=False,
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


@pytest.mark.slow_test
@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_LDA_REGIONS,
    reason="LDA image is not supported in certain regions",
)
def test_tuning_lda(sagemaker_session, cpu_instance_type):
    with timeout(minutes=TUNING_DEFAULT_TIMEOUT_MINUTES):
        data_path = os.path.join(DATA_DIR, "lda")
        data_filename = "nips-train_1.pbr"

        with open(os.path.join(data_path, data_filename), "rb") as f:
            all_records = read_records(f)

        # all records must be same
        feature_num = int(all_records[0].features["values"].float32_tensor.shape[0])

        lda = LDA(
            role="SageMakerRole",
            instance_type=cpu_instance_type,
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
        print("Started hyperparameter tuning job with name:" + tuning_job_name)
        tuner.fit([record_set, test_record_set], mini_batch_size=1, job_name=tuning_job_name)

    attached_tuner = HyperparameterTuner.attach(
        tuning_job_name, sagemaker_session=sagemaker_session
    )
    assert attached_tuner.early_stopping_type == "Auto"
    assert attached_tuner.estimator.alpha0 == 1.0
    assert attached_tuner.estimator.num_topics == 1

    best_training_job = attached_tuner.best_training_job()

    with timeout_and_delete_endpoint_by_name(best_training_job, sagemaker_session):
        predictor = tuner.deploy(1, cpu_instance_type)
        predict_input = np.random.rand(1, feature_num)
        result = predictor.predict(predict_input)

        assert len(result) == 1
        for record in result:
            assert record.label["topic_mixture"] is not None


def test_stop_tuning_job(sagemaker_session, cpu_instance_type):
    feature_num = 14
    train_input = np.random.rand(1000, feature_num)

    rcf = RandomCutForest(
        role="SageMakerRole",
        instance_count=1,
        instance_type=cpu_instance_type,
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
    tuner.fit([records, test_records], tuning_job_name, wait=False)

    time.sleep(15)

    latest_tuning_job_name = tuner.latest_tuning_job.name

    print("Attempting to stop {}".format(latest_tuning_job_name))

    tuner.stop_tuning_job()

    desc = tuner.latest_tuning_job.sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=latest_tuning_job_name
    )
    assert desc["HyperParameterTuningJobStatus"] == "Stopping"


@pytest.mark.slow_test
@pytest.mark.release
def test_tuning_mxnet(
    sagemaker_session,
    mxnet_training_latest_version,
    mxnet_training_latest_py_version,
    cpu_instance_type,
):
    with timeout(minutes=TUNING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        estimator = MXNet(
            entry_point=script_path,
            role="SageMakerRole",
            py_version=mxnet_training_latest_py_version,
            instance_count=1,
            instance_type=cpu_instance_type,
            framework_version=mxnet_training_latest_version,
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
        print("Started hyperparameter tuning job with name:" + tuning_job_name)
        tuner.fit({"train": train_input, "test": test_input}, job_name=tuning_job_name)

    best_training_job = tuner.best_training_job()
    with timeout_and_delete_endpoint_by_name(best_training_job, sagemaker_session):
        predictor = tuner.deploy(1, cpu_instance_type)
        data = np.zeros(shape=(1, 1, 28, 28))
        predictor.predict(data)


@pytest.mark.slow_test
def test_tuning_mxnet_autotune(
    sagemaker_session,
    mxnet_training_latest_version,
    mxnet_training_latest_py_version,
    cpu_instance_type,
):
    with timeout(minutes=TUNING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        estimator = MXNet(
            entry_point=script_path,
            role="SageMakerRole",
            py_version=mxnet_training_latest_py_version,
            instance_count=1,
            instance_type=cpu_instance_type,
            framework_version=mxnet_training_latest_version,
            sagemaker_session=sagemaker_session,
        )
        estimator.set_hyperparameters(learning_rate=0.1)

        hyperparameter_ranges = {}
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
            autotune=True,
            hyperparameters_to_keep_static=None,
        )

        train_input = estimator.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
        )
        test_input = estimator.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
        )

        tuning_job_name = unique_name_from_base("tune-mxnet-autotune", max_length=32)
        print("Started hyperparameter tuning job with name:" + tuning_job_name)
        tuner.fit({"train": train_input, "test": test_input}, job_name=tuning_job_name)

    best_training_job = tuner.best_training_job()
    with timeout_and_delete_endpoint_by_name(best_training_job, sagemaker_session):
        predictor = tuner.deploy(1, cpu_instance_type)
        data = np.zeros(shape=(1, 1, 28, 28))
        predictor.predict(data)


@pytest.mark.release
def test_tuning_tf(
    sagemaker_session,
    cpu_instance_type,
    tensorflow_training_latest_version,
    tensorflow_training_latest_py_version,
):
    if Version(tensorflow_training_latest_version) >= Version("2.16"):
        pytest.skip(
            "This test is failing in TensorFlow 2.16 beacuse of an upstream bug: "
            "https://github.com/tensorflow/io/issues/2039"
        )
    resource_path = os.path.join(DATA_DIR, "tensorflow_mnist")
    script_path = "mnist.py"

    estimator = TensorFlow(
        entry_point=script_path,
        source_dir=resource_path,
        role="SageMakerRole",
        instance_count=1,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        framework_version=tensorflow_training_latest_version,
        py_version=tensorflow_training_latest_py_version,
    )

    hyperparameter_ranges = {"epochs": IntegerParameter(1, 2)}
    objective_metric_name = "accuracy"
    metric_definitions = [{"Name": objective_metric_name, "Regex": "Accuracy: ([0-9\\.]+)"}]

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

        tuning_job_name = unique_name_from_base("tune-tf", max_length=32)
        print("Started hyperparameter tuning job with name: " + tuning_job_name)
        tuner.fit(inputs, job_name=tuning_job_name)


def test_tuning_tf_vpc_multi(
    sagemaker_session,
    cpu_instance_type,
    tensorflow_training_latest_version,
    tensorflow_training_latest_py_version,
):
    """Test Tensorflow multi-instance using the same VpcConfig for training and inference"""
    if Version(tensorflow_training_latest_version) >= Version("2.16"):
        pytest.skip(
            "This test is failing in TensorFlow 2.16 beacuse of an upstream bug: "
            "https://github.com/tensorflow/io/issues/2039"
        )
    instance_type = cpu_instance_type
    instance_count = 2

    resource_path = os.path.join(DATA_DIR, "tensorflow_mnist")
    script_path = "mnist.py"

    ec2_client = sagemaker_session.boto_session.client("ec2")
    subnet_ids, security_group_id = vpc_test_utils.get_or_create_vpc_resources(ec2_client)
    vpc_test_utils.setup_security_group_for_encryption(ec2_client, security_group_id)

    estimator = TensorFlow(
        entry_point=script_path,
        source_dir=resource_path,
        role="SageMakerRole",
        framework_version=tensorflow_training_latest_version,
        py_version=tensorflow_training_latest_py_version,
        instance_count=instance_count,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        base_job_name="test-vpc-tf",
        subnets=subnet_ids,
        security_group_ids=[security_group_id],
        encrypt_inter_container_traffic=True,
    )

    hyperparameter_ranges = {"epochs": IntegerParameter(1, 2)}
    objective_metric_name = "accuracy"
    metric_definitions = [{"Name": objective_metric_name, "Regex": "Accuracy: ([0-9\\.]+)"}]

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

        tuning_job_name = unique_name_from_base("tune-tf", max_length=32)
        print(f"Started hyperparameter tuning job with name: {tuning_job_name}")
        tuner.fit(inputs, job_name=tuning_job_name)


@pytest.mark.release
def test_tuning_chainer(
    sagemaker_session, chainer_latest_version, chainer_latest_py_version, cpu_instance_type
):
    with timeout(minutes=TUNING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, "chainer_mnist", "mnist.py")
        data_path = os.path.join(DATA_DIR, "chainer_mnist")

        estimator = Chainer(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version=chainer_latest_version,
            py_version=chainer_latest_py_version,
            instance_count=1,
            instance_type=cpu_instance_type,
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
        print("Started hyperparameter tuning job with name: {}".format(tuning_job_name))
        tuner.fit({"train": train_input, "test": test_input}, job_name=tuning_job_name)

    best_training_job = tuner.best_training_job()
    with timeout_and_delete_endpoint_by_name(best_training_job, sagemaker_session):
        predictor = tuner.deploy(1, cpu_instance_type)

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


@pytest.mark.release
@pytest.mark.skip(
    reason="This test has always failed, but the failure was masked by a bug. "
    "This test should be fixed. Details in https://github.com/aws/sagemaker-python-sdk/pull/968"
)
def test_attach_tuning_pytorch(
    sagemaker_session,
    cpu_instance_type,
    pytorch_inference_latest_version,
    pytorch_inference_latest_py_version,
):
    mnist_dir = os.path.join(DATA_DIR, "pytorch_mnist")
    mnist_script = os.path.join(mnist_dir, "mnist.py")

    estimator = PyTorch(
        entry_point=mnist_script,
        role="SageMakerRole",
        instance_count=1,
        framework_version=pytorch_inference_latest_version,
        py_version=pytorch_inference_latest_py_version,
        instance_type=cpu_instance_type,
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
        print("Started hyperparameter tuning job with name: {}".format(tuning_job_name))
        tuner.fit({"training": training_data}, job_name=tuning_job_name)

    endpoint_name = tuning_job_name
    model_name = "model-name-1"
    attached_tuner = HyperparameterTuner.attach(
        tuning_job_name, sagemaker_session=sagemaker_session
    )
    assert attached_tuner.early_stopping_type == "Auto"

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        predictor = attached_tuner.deploy(
            1, cpu_instance_type, endpoint_name=endpoint_name, model_name=model_name
        )
    best_training_job = tuner.best_training_job()
    with timeout_and_delete_endpoint_by_name(best_training_job, sagemaker_session):
        predictor = attached_tuner.deploy(1, cpu_instance_type)
        data = np.zeros(shape=(1, 1, 28, 28), dtype=np.float32)
        predictor.predict(data)

        batch_size = 100
        data = np.random.rand(batch_size, 1, 28, 28).astype(np.float32)
        output = predictor.predict(data)

        assert output.shape == (batch_size, 10)
        _assert_model_name_match(sagemaker_session.sagemaker_client, endpoint_name, model_name)


@pytest.mark.release
def test_tuning_byo_estimator(sagemaker_session, cpu_instance_type):
    """Use Factorization Machines algorithm as an example here.

    First we need to prepare data for training. We take standard data set, convert it to the
    format that the algorithm can process and upload it to S3.
    Then we create the Estimator and set hyperparamets as required by the algorithm.
    Next, we can call fit() with path to the S3.
    Later the trained model is deployed and prediction is called against the endpoint.
    Default predictor is updated with json serializer and deserializer.
    """
    image_uri = image_uris.retrieve("factorization-machines", sagemaker_session.boto_region_name)
    training_data_path = os.path.join(DATA_DIR, "dummy_tensor")

    with timeout(minutes=TUNING_DEFAULT_TIMEOUT_MINUTES):
        prefix = "test_byo_estimator"
        key = "recordio-pb-data"
        s3_train_data = sagemaker_session.upload_data(
            path=training_data_path, key_prefix=os.path.join(prefix, "train", key)
        )

        estimator = Estimator(
            image_uri=image_uri,
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
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

        tuning_job_name = unique_name_from_base("byo", 32)
        print("Started hyperparameter tuning job with name {}:".format(tuning_job_name))
        tuner.fit(
            {"train": s3_train_data, "test": s3_train_data},
            include_cls_metadata=False,
            job_name=tuning_job_name,
        )

    best_training_job = tuner.best_training_job()
    with timeout_and_delete_endpoint_by_name(best_training_job, sagemaker_session):
        predictor = tuner.deploy(
            1,
            cpu_instance_type,
            endpoint_name=best_training_job,
            serializer=_FactorizationMachineSerializer(),
            deserializer=JSONDeserializer(),
        )

        result = predictor.predict(datasets.one_p_mnist()[0][:10])

        assert len(result["predictions"]) == 10
        for prediction in result["predictions"]:
            assert prediction["score"] is not None


# Serializer for the Factorization Machines predictor (for BYO example)
class _FactorizationMachineSerializer(SimpleBaseSerializer):
    # SimpleBaseSerializer already uses "application/json" CONTENT_TYPE by default

    def serialize(self, data):
        js = {"instances": []}
        for row in data:
            js["instances"].append({"features": row.tolist()})
        return json.dumps(js)


def _assert_model_name_match(sagemaker_client, endpoint_config_name, model_name):
    endpoint_config_description = sagemaker_client.describe_endpoint_config(
        EndpointConfigName=endpoint_config_name
    )
    assert model_name == endpoint_config_description["ProductionVariants"][0]["ModelName"]
