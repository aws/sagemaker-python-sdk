# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import os
import pickle
import sys
import pytest
import tests.integ

import numpy as np

from sagemaker import (
    KMeans,
    FactorizationMachines,
    IPInsights,
    KNN,
    LDA,
    LinearLearner,
    NTM,
    PCA,
    RandomCutForest,
)
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.amazon.common import read_records
from sagemaker.chainer import Chainer
from sagemaker.estimator import Estimator
from sagemaker.mxnet import MXNet
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.sklearn import SKLearn
from sagemaker.tensorflow import TensorFlow
from sagemaker.xgboost.defaults import XGBOOST_LATEST_VERSION
from sagemaker.workflow import airflow as sm_airflow
from sagemaker.utils import sagemaker_timestamp

import airflow
from airflow import DAG
from airflow.contrib.operators.sagemaker_training_operator import SageMakerTrainingOperator
from airflow.contrib.operators.sagemaker_transform_operator import SageMakerTransformOperator

from sagemaker.xgboost import XGBoost
from tests.integ import DATA_DIR, PYTHON_VERSION
from tests.integ.record_set import prepare_record_set_from_local_files
from tests.integ.timeout import timeout

from six.moves.urllib.parse import urlparse

PYTORCH_MNIST_DIR = os.path.join(DATA_DIR, "pytorch_mnist")
PYTORCH_MNIST_SCRIPT = os.path.join(PYTORCH_MNIST_DIR, "mnist.py")
AIRFLOW_CONFIG_TIMEOUT_IN_SECONDS = 10

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
TF_MNIST_RESOURCE_PATH = os.path.join(RESOURCE_PATH, "tensorflow_mnist")
SCRIPT = os.path.join(TF_MNIST_RESOURCE_PATH, "mnist.py")
ROLE = "SageMakerRole"
SINGLE_INSTANCE_COUNT = 1


@pytest.mark.canary_quick
def test_byo_airflow_config_uploads_data_source_to_s3_when_inputs_provided(
    sagemaker_session, cpu_instance_type
):
    with timeout(seconds=AIRFLOW_CONFIG_TIMEOUT_IN_SECONDS):
        training_data_path = os.path.join(DATA_DIR, "dummy_tensor")

        data_source_location = "test-airflow-config-{}".format(sagemaker_timestamp())
        inputs = sagemaker_session.upload_data(
            path=training_data_path, key_prefix=os.path.join(data_source_location, "train")
        )

        estimator = Estimator(
            image_name=get_image_uri(
                sagemaker_session.boto_session.region_name, "factorization-machines"
            ),
            role=ROLE,
            train_instance_count=SINGLE_INSTANCE_COUNT,
            train_instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
        )

        training_config = _build_airflow_workflow(
            estimator=estimator, instance_type=cpu_instance_type, inputs=inputs
        )

        _assert_that_s3_url_contains_data(
            sagemaker_session,
            training_config["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"],
        )


@pytest.mark.canary_quick
def test_kmeans_airflow_config_uploads_data_source_to_s3(sagemaker_session, cpu_instance_type):
    with timeout(seconds=AIRFLOW_CONFIG_TIMEOUT_IN_SECONDS):
        data_path = os.path.join(DATA_DIR, "one_p_mnist", "mnist.pkl.gz")
        pickle_args = {} if sys.version_info.major == 2 else {"encoding": "latin1"}

        # Load the data into memory as numpy arrays
        with gzip.open(data_path, "rb") as f:
            train_set, _, _ = pickle.load(f, **pickle_args)

        kmeans = KMeans(
            role=ROLE,
            train_instance_count=SINGLE_INSTANCE_COUNT,
            train_instance_type=cpu_instance_type,
            k=10,
            sagemaker_session=sagemaker_session,
        )

        kmeans.init_method = "random"
        kmeans.max_iterations = 1
        kmeans.tol = 1
        kmeans.num_trials = 1
        kmeans.local_init_method = "kmeans++"
        kmeans.half_life_time_size = 1
        kmeans.epochs = 1
        kmeans.center_factor = 1
        kmeans.eval_metrics = ["ssd", "msd"]

        records = kmeans.record_set(train_set[0][:100])

        training_config = _build_airflow_workflow(
            estimator=kmeans, instance_type=cpu_instance_type, inputs=records
        )

        _assert_that_s3_url_contains_data(
            sagemaker_session,
            training_config["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"],
        )


def test_fm_airflow_config_uploads_data_source_to_s3(sagemaker_session, cpu_instance_type):
    with timeout(seconds=AIRFLOW_CONFIG_TIMEOUT_IN_SECONDS):
        data_path = os.path.join(DATA_DIR, "one_p_mnist", "mnist.pkl.gz")
        pickle_args = {} if sys.version_info.major == 2 else {"encoding": "latin1"}

        # Load the data into memory as numpy arrays
        with gzip.open(data_path, "rb") as f:
            train_set, _, _ = pickle.load(f, **pickle_args)

        fm = FactorizationMachines(
            role=ROLE,
            train_instance_count=SINGLE_INSTANCE_COUNT,
            train_instance_type=cpu_instance_type,
            num_factors=10,
            predictor_type="regressor",
            epochs=2,
            clip_gradient=1e2,
            eps=0.001,
            rescale_grad=1.0 / 100,
            sagemaker_session=sagemaker_session,
        )

        records = fm.record_set(train_set[0][:200], train_set[1][:200].astype("float32"))

        training_config = _build_airflow_workflow(
            estimator=fm, instance_type=cpu_instance_type, inputs=records
        )

        _assert_that_s3_url_contains_data(
            sagemaker_session,
            training_config["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"],
        )


@pytest.mark.canary_quick
def test_ipinsights_airflow_config_uploads_data_source_to_s3(sagemaker_session, cpu_instance_type):
    with timeout(seconds=AIRFLOW_CONFIG_TIMEOUT_IN_SECONDS):
        data_path = os.path.join(DATA_DIR, "ipinsights")
        data_filename = "train.csv"

        with open(os.path.join(data_path, data_filename), "rb") as f:
            num_records = len(f.readlines())

        ipinsights = IPInsights(
            role=ROLE,
            train_instance_count=SINGLE_INSTANCE_COUNT,
            train_instance_type=cpu_instance_type,
            num_entity_vectors=10,
            vector_dim=100,
            sagemaker_session=sagemaker_session,
        )

        records = prepare_record_set_from_local_files(
            data_path, ipinsights.data_location, num_records, None, sagemaker_session
        )

        training_config = _build_airflow_workflow(
            estimator=ipinsights, instance_type=cpu_instance_type, inputs=records
        )

        _assert_that_s3_url_contains_data(
            sagemaker_session,
            training_config["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"],
        )


def test_knn_airflow_config_uploads_data_source_to_s3(sagemaker_session, cpu_instance_type):
    with timeout(seconds=AIRFLOW_CONFIG_TIMEOUT_IN_SECONDS):
        data_path = os.path.join(DATA_DIR, "one_p_mnist", "mnist.pkl.gz")
        pickle_args = {} if sys.version_info.major == 2 else {"encoding": "latin1"}

        # Load the data into memory as numpy arrays
        with gzip.open(data_path, "rb") as f:
            train_set, _, _ = pickle.load(f, **pickle_args)

        knn = KNN(
            role=ROLE,
            train_instance_count=SINGLE_INSTANCE_COUNT,
            train_instance_type=cpu_instance_type,
            k=10,
            predictor_type="regressor",
            sample_size=500,
            sagemaker_session=sagemaker_session,
        )

        records = knn.record_set(train_set[0][:200], train_set[1][:200].astype("float32"))

        training_config = _build_airflow_workflow(
            estimator=knn, instance_type=cpu_instance_type, inputs=records
        )

        _assert_that_s3_url_contains_data(
            sagemaker_session,
            training_config["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"],
        )


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_LDA_REGIONS,
    reason="LDA image is not supported in certain regions",
)
@pytest.mark.canary_quick
def test_lda_airflow_config_uploads_data_source_to_s3(sagemaker_session, cpu_instance_type):
    with timeout(seconds=AIRFLOW_CONFIG_TIMEOUT_IN_SECONDS):
        data_path = os.path.join(DATA_DIR, "lda")
        data_filename = "nips-train_1.pbr"

        with open(os.path.join(data_path, data_filename), "rb") as f:
            all_records = read_records(f)

        # all records must be same
        feature_num = int(all_records[0].features["values"].float32_tensor.shape[0])

        lda = LDA(
            role=ROLE,
            train_instance_type=cpu_instance_type,
            num_topics=10,
            sagemaker_session=sagemaker_session,
        )

        records = prepare_record_set_from_local_files(
            data_path, lda.data_location, len(all_records), feature_num, sagemaker_session
        )

        training_config = _build_airflow_workflow(
            estimator=lda, instance_type=cpu_instance_type, inputs=records, mini_batch_size=100
        )

        _assert_that_s3_url_contains_data(
            sagemaker_session,
            training_config["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"],
        )


@pytest.mark.canary_quick
def test_linearlearner_airflow_config_uploads_data_source_to_s3(
    sagemaker_session, cpu_instance_type
):
    with timeout(seconds=AIRFLOW_CONFIG_TIMEOUT_IN_SECONDS):
        data_path = os.path.join(DATA_DIR, "one_p_mnist", "mnist.pkl.gz")
        pickle_args = {} if sys.version_info.major == 2 else {"encoding": "latin1"}

        # Load the data into memory as numpy arrays
        with gzip.open(data_path, "rb") as f:
            train_set, _, _ = pickle.load(f, **pickle_args)

        train_set[1][:100] = 1
        train_set[1][100:200] = 0
        train_set = train_set[0], train_set[1].astype(np.dtype("float32"))

        ll = LinearLearner(
            ROLE,
            1,
            cpu_instance_type,
            predictor_type="binary_classifier",
            sagemaker_session=sagemaker_session,
        )
        ll.binary_classifier_model_selection_criteria = "accuracy"
        ll.target_recall = 0.5
        ll.target_precision = 0.5
        ll.positive_example_weight_mult = 0.1
        ll.epochs = 1
        ll.use_bias = True
        ll.num_models = 1
        ll.num_calibration_samples = 1
        ll.init_method = "uniform"
        ll.init_scale = 0.5
        ll.init_sigma = 0.2
        ll.init_bias = 5
        ll.optimizer = "adam"
        ll.loss = "logistic"
        ll.wd = 0.5
        ll.l1 = 0.5
        ll.momentum = 0.5
        ll.learning_rate = 0.1
        ll.beta_1 = 0.1
        ll.beta_2 = 0.1
        ll.use_lr_scheduler = True
        ll.lr_scheduler_step = 2
        ll.lr_scheduler_factor = 0.5
        ll.lr_scheduler_minimum_lr = 0.1
        ll.normalize_data = False
        ll.normalize_label = False
        ll.unbias_data = True
        ll.unbias_label = False
        ll.num_point_for_scaler = 10000
        ll.margin = 1.0
        ll.quantile = 0.5
        ll.loss_insensitivity = 0.1
        ll.huber_delta = 0.1
        ll.early_stopping_tolerance = 0.0001
        ll.early_stopping_patience = 3

        records = ll.record_set(train_set[0][:200], train_set[1][:200])

        training_config = _build_airflow_workflow(
            estimator=ll, instance_type=cpu_instance_type, inputs=records
        )

        _assert_that_s3_url_contains_data(
            sagemaker_session,
            training_config["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"],
        )


@pytest.mark.canary_quick
def test_ntm_airflow_config_uploads_data_source_to_s3(sagemaker_session, cpu_instance_type):
    with timeout(seconds=AIRFLOW_CONFIG_TIMEOUT_IN_SECONDS):
        data_path = os.path.join(DATA_DIR, "ntm")
        data_filename = "nips-train_1.pbr"

        with open(os.path.join(data_path, data_filename), "rb") as f:
            all_records = read_records(f)

        # all records must be same
        feature_num = int(all_records[0].features["values"].float32_tensor.shape[0])

        ntm = NTM(
            role=ROLE,
            train_instance_count=SINGLE_INSTANCE_COUNT,
            train_instance_type=cpu_instance_type,
            num_topics=10,
            sagemaker_session=sagemaker_session,
        )

        records = prepare_record_set_from_local_files(
            data_path, ntm.data_location, len(all_records), feature_num, sagemaker_session
        )

        training_config = _build_airflow_workflow(
            estimator=ntm, instance_type=cpu_instance_type, inputs=records
        )

        _assert_that_s3_url_contains_data(
            sagemaker_session,
            training_config["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"],
        )


@pytest.mark.canary_quick
def test_pca_airflow_config_uploads_data_source_to_s3(sagemaker_session, cpu_instance_type):
    with timeout(seconds=AIRFLOW_CONFIG_TIMEOUT_IN_SECONDS):
        data_path = os.path.join(DATA_DIR, "one_p_mnist", "mnist.pkl.gz")
        pickle_args = {} if sys.version_info.major == 2 else {"encoding": "latin1"}

        # Load the data into memory as numpy arrays
        with gzip.open(data_path, "rb") as f:
            train_set, _, _ = pickle.load(f, **pickle_args)

        pca = PCA(
            role=ROLE,
            train_instance_count=SINGLE_INSTANCE_COUNT,
            train_instance_type=cpu_instance_type,
            num_components=48,
            sagemaker_session=sagemaker_session,
        )

        pca.algorithm_mode = "randomized"
        pca.subtract_mean = True
        pca.extra_components = 5

        records = pca.record_set(train_set[0][:100])

        training_config = _build_airflow_workflow(
            estimator=pca, instance_type=cpu_instance_type, inputs=records
        )

        _assert_that_s3_url_contains_data(
            sagemaker_session,
            training_config["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"],
        )


@pytest.mark.canary_quick
def test_rcf_airflow_config_uploads_data_source_to_s3(sagemaker_session, cpu_instance_type):
    with timeout(seconds=AIRFLOW_CONFIG_TIMEOUT_IN_SECONDS):
        # Generate a thousand 14-dimensional datapoints.
        feature_num = 14
        train_input = np.random.rand(1000, feature_num)

        rcf = RandomCutForest(
            role=ROLE,
            train_instance_count=SINGLE_INSTANCE_COUNT,
            train_instance_type=cpu_instance_type,
            num_trees=50,
            num_samples_per_tree=20,
            eval_metrics=["accuracy", "precision_recall_fscore"],
            sagemaker_session=sagemaker_session,
        )

        records = rcf.record_set(train_input)

        training_config = _build_airflow_workflow(
            estimator=rcf, instance_type=cpu_instance_type, inputs=records
        )

        _assert_that_s3_url_contains_data(
            sagemaker_session,
            training_config["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"],
        )


@pytest.mark.canary_quick
def test_chainer_airflow_config_uploads_data_source_to_s3(
    sagemaker_session, cpu_instance_type, chainer_full_version
):
    with timeout(seconds=AIRFLOW_CONFIG_TIMEOUT_IN_SECONDS):
        script_path = os.path.join(DATA_DIR, "chainer_mnist", "mnist.py")
        data_path = os.path.join(DATA_DIR, "chainer_mnist")

        chainer = Chainer(
            entry_point=script_path,
            role=ROLE,
            train_instance_count=SINGLE_INSTANCE_COUNT,
            train_instance_type="local",
            framework_version=chainer_full_version,
            py_version=PYTHON_VERSION,
            sagemaker_session=sagemaker_session,
            hyperparameters={"epochs": 1},
            use_mpi=True,
            num_processes=2,
            process_slots_per_host=2,
            additional_mpi_options="-x NCCL_DEBUG=INFO",
        )

        train_input = "file://" + os.path.join(data_path, "train")
        test_input = "file://" + os.path.join(data_path, "test")

        training_config = _build_airflow_workflow(
            estimator=chainer,
            instance_type=cpu_instance_type,
            inputs={"train": train_input, "test": test_input},
        )

        _assert_that_s3_url_contains_data(
            sagemaker_session,
            training_config["HyperParameters"]["sagemaker_submit_directory"].strip('"'),
        )


@pytest.mark.canary_quick
def test_mxnet_airflow_config_uploads_data_source_to_s3(
    sagemaker_session, cpu_instance_type, mxnet_full_version
):
    with timeout(seconds=AIRFLOW_CONFIG_TIMEOUT_IN_SECONDS):
        script_path = os.path.join(DATA_DIR, "chainer_mnist", "mnist.py")
        data_path = os.path.join(DATA_DIR, "chainer_mnist")

        mx = MXNet(
            entry_point=script_path,
            role=ROLE,
            framework_version=mxnet_full_version,
            py_version=PYTHON_VERSION,
            train_instance_count=SINGLE_INSTANCE_COUNT,
            train_instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
        )

        train_input = "file://" + os.path.join(data_path, "train")
        test_input = "file://" + os.path.join(data_path, "test")

        training_config = _build_airflow_workflow(
            estimator=mx,
            instance_type=cpu_instance_type,
            inputs={"train": train_input, "test": test_input},
        )

        _assert_that_s3_url_contains_data(
            sagemaker_session,
            training_config["HyperParameters"]["sagemaker_submit_directory"].strip('"'),
        )


@pytest.mark.canary_quick
def test_sklearn_airflow_config_uploads_data_source_to_s3(
    sagemaker_session, cpu_instance_type, sklearn_full_version
):
    with timeout(seconds=AIRFLOW_CONFIG_TIMEOUT_IN_SECONDS):
        script_path = os.path.join(DATA_DIR, "sklearn_mnist", "mnist.py")
        data_path = os.path.join(DATA_DIR, "sklearn_mnist")

        sklearn = SKLearn(
            entry_point=script_path,
            role=ROLE,
            train_instance_type=cpu_instance_type,
            framework_version=sklearn_full_version,
            py_version=PYTHON_VERSION,
            sagemaker_session=sagemaker_session,
            hyperparameters={"epochs": 1},
        )

        train_input = sklearn.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/sklearn_mnist/train"
        )
        test_input = sklearn.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/sklearn_mnist/test"
        )

        training_config = _build_airflow_workflow(
            estimator=sklearn,
            instance_type=cpu_instance_type,
            inputs={"train": train_input, "test": test_input},
        )

        _assert_that_s3_url_contains_data(
            sagemaker_session,
            training_config["HyperParameters"]["sagemaker_submit_directory"].strip('"'),
        )


@pytest.mark.canary_quick
def test_tf_airflow_config_uploads_data_source_to_s3(sagemaker_session, cpu_instance_type):
    with timeout(seconds=AIRFLOW_CONFIG_TIMEOUT_IN_SECONDS):
        tf = TensorFlow(
            image_name=get_image_uri(
                sagemaker_session.boto_session.region_name, "factorization-machines"
            ),
            entry_point=SCRIPT,
            role=ROLE,
            train_instance_count=SINGLE_INSTANCE_COUNT,
            train_instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            script_mode=True,
            framework_version=TensorFlow.LATEST_VERSION,
            py_version=PYTHON_VERSION,
            metric_definitions=[
                {"Name": "train:global_steps", "Regex": r"global_step\/sec:\s(.*)"}
            ],
        )
        inputs = tf.sagemaker_session.upload_data(
            path=os.path.join(TF_MNIST_RESOURCE_PATH, "data"), key_prefix="scriptmode/mnist"
        )

        training_config = _build_airflow_workflow(
            estimator=tf, instance_type=cpu_instance_type, inputs=inputs
        )

        _assert_that_s3_url_contains_data(
            sagemaker_session,
            training_config["HyperParameters"]["sagemaker_submit_directory"].strip('"'),
        )


@pytest.mark.canary_quick
def test_xgboost_airflow_config_uploads_data_source_to_s3(sagemaker_session, cpu_instance_type):
    with timeout(seconds=AIRFLOW_CONFIG_TIMEOUT_IN_SECONDS):
        xgboost = XGBoost(
            entry_point=os.path.join(DATA_DIR, "dummy_script.py"),
            framework_version=XGBOOST_LATEST_VERSION,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            train_instance_type=cpu_instance_type,
            train_instance_count=SINGLE_INSTANCE_COUNT,
            base_job_name="XGBoost job",
            py_version=PYTHON_VERSION,
        )

        training_config = _build_airflow_workflow(
            estimator=xgboost, instance_type=cpu_instance_type
        )

        _assert_that_s3_url_contains_data(
            sagemaker_session,
            training_config["HyperParameters"]["sagemaker_submit_directory"].strip('"'),
        )


@pytest.mark.canary_quick
def test_pytorch_airflow_config_uploads_data_source_to_s3_when_inputs_not_provided(
    sagemaker_session, cpu_instance_type
):
    with timeout(seconds=AIRFLOW_CONFIG_TIMEOUT_IN_SECONDS):
        estimator = PyTorch(
            entry_point=PYTORCH_MNIST_SCRIPT,
            role=ROLE,
            framework_version="1.1.0",
            train_instance_count=2,
            train_instance_type=cpu_instance_type,
            hyperparameters={"epochs": 6, "backend": "gloo"},
            sagemaker_session=sagemaker_session,
        )

        training_config = _build_airflow_workflow(
            estimator=estimator, instance_type=cpu_instance_type
        )

        _assert_that_s3_url_contains_data(
            sagemaker_session,
            training_config["HyperParameters"]["sagemaker_submit_directory"].strip('"'),
        )


def test_pytorch_12_airflow_config_uploads_data_source_to_s3_when_inputs_not_provided(
    sagemaker_session, cpu_instance_type
):
    with timeout(seconds=AIRFLOW_CONFIG_TIMEOUT_IN_SECONDS):
        estimator = PyTorch(
            entry_point=PYTORCH_MNIST_SCRIPT,
            role=ROLE,
            framework_version="1.2.0",
            train_instance_count=2,
            train_instance_type=cpu_instance_type,
            hyperparameters={"epochs": 6, "backend": "gloo"},
            sagemaker_session=sagemaker_session,
        )

        training_config = _build_airflow_workflow(
            estimator=estimator, instance_type=cpu_instance_type
        )

        _assert_that_s3_url_contains_data(
            sagemaker_session,
            training_config["HyperParameters"]["sagemaker_submit_directory"].strip('"'),
        )


def _assert_that_s3_url_contains_data(sagemaker_session, s3_url):
    parsed_s3_url = urlparse(s3_url)
    s3_request = sagemaker_session.boto_session.client("s3").list_objects_v2(
        Bucket=parsed_s3_url.netloc, Prefix=parsed_s3_url.path.lstrip("/")
    )
    assert s3_request["KeyCount"] > 0


def _build_airflow_workflow(estimator, instance_type, inputs=None, mini_batch_size=None):
    training_config = sm_airflow.training_config(
        estimator=estimator, inputs=inputs, mini_batch_size=mini_batch_size
    )

    model = estimator.create_model()
    assert model is not None

    model_config = sm_airflow.model_config(instance_type, model)
    assert model_config is not None

    transform_config = sm_airflow.transform_config_from_estimator(
        estimator=estimator,
        task_id="transform_config",
        task_type="training",
        instance_count=SINGLE_INSTANCE_COUNT,
        instance_type=estimator.train_instance_type,
        data=inputs,
        content_type="text/csv",
    )

    default_args = {
        "owner": "airflow",
        "start_date": airflow.utils.dates.days_ago(2),
        "provide_context": True,
    }

    dag = DAG("tensorflow_example", default_args=default_args, schedule_interval="@once")

    train_op = SageMakerTrainingOperator(
        task_id="tf_training", config=training_config, wait_for_completion=True, dag=dag
    )

    transform_op = SageMakerTransformOperator(
        task_id="transform_operator", config=transform_config, wait_for_completion=True, dag=dag
    )

    transform_op.set_upstream(train_op)

    return training_config
