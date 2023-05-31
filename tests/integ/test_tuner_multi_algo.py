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

import pytest

from sagemaker import image_uris, utils
from sagemaker.analytics import HyperparameterTuningJobAnalytics
from sagemaker.deserializers import JSONDeserializer
from sagemaker.estimator import Estimator
from sagemaker.serializers import SimpleBaseSerializer
from sagemaker.tuner import ContinuousParameter, IntegerParameter, HyperparameterTuner
from tests.integ import datasets, DATA_DIR, TUNING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name

BASE_TUNING_JOB_NAME = "multi-algo-pysdk"

EXECUTION_ROLE = "SageMakerRole"

STRATEGY = "Bayesian"
OBJECTIVE_TYPE = "Minimize"

TAGS = [{"Key": "pysdk-test", "Value": "multi-algo-tuner"}]

ESTIMATOR_FM = "fm-one"
ESTIMATOR_KNN = "knn-two"

# TODO: change to use one of the new standard metrics for 1P algorithm
OBJECTIVE_METRIC_NAME_FM = "test:rmse"
OBJECTIVE_METRIC_NAME_KNN = "test:mse"

HYPER_PARAMETER_RANGES_FM = {
    "factors_wd": ContinuousParameter(1, 30),
    "factors_lr": ContinuousParameter(40, 50),
}
HYPER_PARAMETER_RANGES_KNN = {
    "k": IntegerParameter(3, 400),
    "sample_size": IntegerParameter(40, 550),
}

MAX_JOBS = 2
MAX_PARALLEL_JOBS = 2


@pytest.fixture(scope="module")
def data_set():
    return datasets.one_p_mnist()


@pytest.fixture(scope="function")
def estimator_fm(sagemaker_session, cpu_instance_type):
    fm_image = image_uris.retrieve("factorization-machines", sagemaker_session.boto_region_name)

    estimator = Estimator(
        image_uri=fm_image,
        role=EXECUTION_ROLE,
        instance_count=1,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
    )

    estimator.set_hyperparameters(
        num_factors=10, feature_dim=784, mini_batch_size=100, predictor_type="regressor"
    )

    return estimator


@pytest.fixture(scope="function")
def estimator_knn(sagemaker_session, cpu_instance_type):
    knn_image = image_uris.retrieve("knn", sagemaker_session.boto_region_name)

    estimator = Estimator(
        image_uri=knn_image,
        role=EXECUTION_ROLE,
        instance_count=1,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
    )

    estimator.set_hyperparameters(
        k=10, sample_size=500, feature_dim=784, mini_batch_size=100, predictor_type="regressor"
    )
    return estimator


def test_multi_estimator_tuning(
    sagemaker_session, estimator_fm, estimator_knn, data_set, cpu_instance_type
):
    tuner = HyperparameterTuner.create(
        base_tuning_job_name=BASE_TUNING_JOB_NAME,
        estimator_dict={ESTIMATOR_FM: estimator_fm, ESTIMATOR_KNN: estimator_knn},
        objective_metric_name_dict={
            ESTIMATOR_FM: OBJECTIVE_METRIC_NAME_FM,
            ESTIMATOR_KNN: OBJECTIVE_METRIC_NAME_KNN,
        },
        hyperparameter_ranges_dict={
            ESTIMATOR_FM: HYPER_PARAMETER_RANGES_FM,
            ESTIMATOR_KNN: HYPER_PARAMETER_RANGES_KNN,
        },
        strategy=STRATEGY,
        objective_type=OBJECTIVE_TYPE,
        max_jobs=MAX_JOBS,
        max_parallel_jobs=MAX_PARALLEL_JOBS,
        tags=TAGS,
    )

    _fit_tuner(sagemaker_session, tuner)

    _retrieve_analytics(sagemaker_session, tuner.latest_tuning_job.name)

    tuner_attached = _attach_tuner(sagemaker_session, tuner.latest_tuning_job.name)

    _deploy_and_predict(sagemaker_session, tuner_attached, data_set, cpu_instance_type)


def test_multi_estimator_tuning_autotune(
    sagemaker_session, estimator_fm, estimator_knn, data_set, cpu_instance_type
):
    tuner = HyperparameterTuner.create(
        base_tuning_job_name=BASE_TUNING_JOB_NAME,
        estimator_dict={ESTIMATOR_FM: estimator_fm, ESTIMATOR_KNN: estimator_knn},
        objective_metric_name_dict={
            ESTIMATOR_FM: OBJECTIVE_METRIC_NAME_FM,
            ESTIMATOR_KNN: OBJECTIVE_METRIC_NAME_KNN,
        },
        hyperparameter_ranges_dict={
            ESTIMATOR_FM: HYPER_PARAMETER_RANGES_FM,
            ESTIMATOR_KNN: HYPER_PARAMETER_RANGES_KNN,
        },
        strategy=STRATEGY,
        objective_type=OBJECTIVE_TYPE,
        max_jobs=MAX_JOBS,
        max_parallel_jobs=MAX_PARALLEL_JOBS,
        tags=TAGS,
        autotune=True,
        hyperparameters_to_keep_static_dict={
            ESTIMATOR_FM: ["num_factors", "predictor_type"],
            ESTIMATOR_KNN: ["predictor_type", "mini_batch_size"],
        },
    )
    tuning_job_base_name = "test-multi-autotune"
    _fit_tuner(sagemaker_session, tuner, tuning_job_base_name=tuning_job_base_name)

    _retrieve_analytics(sagemaker_session, tuner.latest_tuning_job.name)

    tuner_attached = _attach_tuner(sagemaker_session, tuner.latest_tuning_job.name)

    _deploy_and_predict(sagemaker_session, tuner_attached, data_set, cpu_instance_type)


def _fit_tuner(sagemaker_session, tuner, tuning_job_base_name=None):
    training_inputs = _create_training_inputs(sagemaker_session)
    if tuning_job_base_name is None:
        tuning_job_base_name = "test-multi-algo-tuning"
    job_name = utils.unique_name_from_base(tuning_job_base_name, max_length=32)

    with timeout(minutes=TUNING_DEFAULT_TIMEOUT_MINUTES):
        tuner.fit(
            inputs={ESTIMATOR_FM: training_inputs, ESTIMATOR_KNN: training_inputs},
            include_cls_metadata={},
            job_name=job_name,
        )
        tuner.wait()


def _retrieve_analytics(sagemaker_session, tuning_job_name):
    tuner_analytics = HyperparameterTuningJobAnalytics(
        hyperparameter_tuning_job_name=tuning_job_name, sagemaker_session=sagemaker_session
    )
    _verify_analytics_dataframe(tuner_analytics)
    _verify_analytics_tuning_ranges(tuner_analytics)


def _verify_analytics_dataframe(tuner_analytics):
    df = tuner_analytics.dataframe()
    assert len(df) == MAX_JOBS


def _verify_analytics_tuning_ranges(tuner_analytics):
    analytics_tuning_ranges = tuner_analytics.tuning_ranges
    assert len(analytics_tuning_ranges) == 2

    expected_tuning_ranges_fm = {
        key: value.as_tuning_range(key) for key, value in HYPER_PARAMETER_RANGES_FM.items()
    }
    assert expected_tuning_ranges_fm == analytics_tuning_ranges[ESTIMATOR_FM]

    expected_tuning_ranges_knn = {
        key: value.as_tuning_range(key) for key, value in HYPER_PARAMETER_RANGES_KNN.items()
    }
    assert expected_tuning_ranges_knn == analytics_tuning_ranges[ESTIMATOR_KNN]


def _attach_tuner(sagemaker_session, tuning_job_name):
    print("Attaching hyperparameter tuning job {} to a new tuner instance".format(tuning_job_name))
    return HyperparameterTuner.attach(
        tuning_job_name,
        sagemaker_session=sagemaker_session,
        estimator_cls={
            ESTIMATOR_FM: "sagemaker.estimator.Estimator",
            ESTIMATOR_KNN: "sagemaker.estimator.Estimator",
        },
    )


def _deploy_and_predict(sagemaker_session, tuner, data_set, cpu_instance_type):
    best_training_job = tuner.best_training_job()
    with timeout_and_delete_endpoint_by_name(best_training_job, sagemaker_session):
        print(
            "Deploying best model of hyperparameter tuning job {}: {}".format(
                tuner.latest_tuning_job.name, best_training_job
            )
        )
        predictor = tuner.deploy(
            1,
            cpu_instance_type,
            endpoint_name=best_training_job,
            serializer=PredictionDataSerializer(),
            deserializer=JSONDeserializer(),
        )

        print("Making prediction using the deployed model")
        data = data_set[0][:10]
        result = predictor.predict(data)

        assert len(result["predictions"]) == len(data)
        for prediction in result["predictions"]:
            assert prediction is not None


def _create_training_inputs(sagemaker_session):
    training_data_path = os.path.join(DATA_DIR, "dummy_tensor")

    prefix = "multi-algo"
    key = "recordio-pb-data"

    s3_train_data = sagemaker_session.upload_data(
        path=training_data_path, key_prefix=os.path.join(prefix, "train", key)
    )

    return {"train": s3_train_data, "test": s3_train_data}


class PredictionDataSerializer(SimpleBaseSerializer):
    # SimpleBaseSerializer already uses "application/json" CONTENT_TYPE by default

    def serialize(self, data):
        js = {"instances": []}
        for row in data:
            js["instances"].append({"features": row.tolist()})
        return json.dumps(js)
