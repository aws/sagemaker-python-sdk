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

import time

import numpy as np
import pytest

from sagemaker.amazon.linear_learner import LinearLearner, LinearLearnerModel
from sagemaker.utils import unique_name_from_base
from tests.integ import datasets, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


@pytest.fixture
def training_set():
    return datasets.one_p_mnist()


@pytest.mark.release
def test_linear_learner(sagemaker_session, cpu_instance_type, training_set):
    job_name = unique_name_from_base("linear-learner")

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        training_set[1][:100] = 1
        training_set[1][100:200] = 0
        training_set = training_set[0], training_set[1].astype(np.dtype("float32"))

        ll = LinearLearner(
            "SageMakerRole",
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
        ll.fit(ll.record_set(training_set[0][:200], training_set[1][:200]), job_name=job_name)

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        predictor = ll.deploy(1, cpu_instance_type, endpoint_name=job_name)

        result = predictor.predict(training_set[0][0:100])
        assert len(result) == 100
        for record in result:
            assert record.label["predicted_label"] is not None
            assert record.label["score"] is not None


def test_linear_learner_multiclass(sagemaker_session, cpu_instance_type, training_set):
    job_name = unique_name_from_base("linear-learner")

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        training_set = training_set[0], training_set[1].astype(np.dtype("float32"))

        ll = LinearLearner(
            "SageMakerRole",
            1,
            cpu_instance_type,
            predictor_type="multiclass_classifier",
            num_classes=10,
            sagemaker_session=sagemaker_session,
        )

        ll.epochs = 1
        ll.fit(ll.record_set(training_set[0][:200], training_set[1][:200]), job_name=job_name)

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        predictor = ll.deploy(1, cpu_instance_type, endpoint_name=job_name)

        result = predictor.predict(training_set[0][0:100])
        assert len(result) == 100
        for record in result:
            assert record.label["predicted_label"] is not None
            assert record.label["score"] is not None


def test_async_linear_learner(sagemaker_session, cpu_instance_type, training_set):
    job_name = unique_name_from_base("linear-learner")

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        training_set[1][:100] = 1
        training_set[1][100:200] = 0
        training_set = training_set[0], training_set[1].astype(np.dtype("float32"))

        ll = LinearLearner(
            "SageMakerRole",
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
        ll.fit(
            ll.record_set(training_set[0][:200], training_set[1][:200]),
            wait=False,
            job_name=job_name,
        )

        print("Waiting to re-attach to the training job: %s" % job_name)
        time.sleep(20)

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        estimator = LinearLearner.attach(
            training_job_name=job_name, sagemaker_session=sagemaker_session
        )
        model = LinearLearnerModel(
            estimator.model_data, role="SageMakerRole", sagemaker_session=sagemaker_session
        )
        predictor = model.deploy(1, cpu_instance_type, endpoint_name=job_name)

        result = predictor.predict(training_set[0][0:100])
        assert len(result) == 100
        for record in result:
            assert record.label["predicted_label"] is not None
            assert record.label["score"] is not None


def test_linear_learner_serverless_inference(sagemaker_session, cpu_instance_type, training_set):
    job_name = unique_name_from_base("linear-learner-serverless")

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        training_set[1][:100] = 1
        training_set[1][100:200] = 0
        training_set = training_set[0], training_set[1].astype(np.dtype("float32"))

        ll = LinearLearner(
            "SageMakerRole",
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
        ll.fit(ll.record_set(training_set[0][:200], training_set[1][:200]), job_name=job_name)

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        predictor = ll.deploy(1, cpu_instance_type, endpoint_name=job_name)

        result = predictor.predict(training_set[0][0:100])
        assert len(result) == 100
        for record in result:
            assert record.label["predicted_label"] is not None
            assert record.label["score"] is not None
