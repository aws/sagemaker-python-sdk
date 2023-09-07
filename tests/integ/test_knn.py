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

import pytest

from sagemaker import KNN, KNNModel
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.utils import unique_name_from_base
from tests.integ import datasets, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


@pytest.fixture
def training_set():
    return datasets.one_p_mnist()


def test_knn_regressor(sagemaker_session, cpu_instance_type, training_set):
    job_name = unique_name_from_base("knn")

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        knn = KNN(
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
            k=10,
            predictor_type="regressor",
            sample_size=500,
            sagemaker_session=sagemaker_session,
        )

        # training labels must be 'float32'
        knn.fit(
            knn.record_set(training_set[0][:200], training_set[1][:200].astype("float32")),
            job_name=job_name,
        )

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        model = KNNModel(knn.model_data, role="SageMakerRole", sagemaker_session=sagemaker_session)
        predictor = model.deploy(1, cpu_instance_type, endpoint_name=job_name)
        result = predictor.predict(training_set[0][:10])

        assert len(result) == 10
        for record in result:
            assert record.label["score"] is not None


def test_async_knn_classifier(sagemaker_session, cpu_instance_type, training_set):
    job_name = unique_name_from_base("knn")

    with timeout(minutes=5):
        knn = KNN(
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
            k=10,
            predictor_type="classifier",
            sample_size=500,
            index_type="faiss.IVFFlat",
            index_metric="L2",
            sagemaker_session=sagemaker_session,
        )

        # training labels must be 'float32'
        knn.fit(
            knn.record_set(training_set[0][:200], training_set[1][:200].astype("float32")),
            wait=False,
            job_name=job_name,
        )

        print("Detached from training job. Will re-attach in 20 seconds")
        time.sleep(20)
        print("attaching now...")

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        estimator = KNN.attach(training_job_name=job_name, sagemaker_session=sagemaker_session)
        model = KNNModel(
            estimator.model_data, role="SageMakerRole", sagemaker_session=sagemaker_session
        )
        predictor = model.deploy(1, cpu_instance_type, endpoint_name=job_name)
        result = predictor.predict(training_set[0][:10])

        assert len(result) == 10
        for record in result:
            assert record.label["score"] is not None


def test_knn_regressor_serverless_inference(sagemaker_session, cpu_instance_type, training_set):
    job_name = unique_name_from_base("knn-serverless")

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        knn = KNN(
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
            k=10,
            predictor_type="regressor",
            sample_size=500,
            sagemaker_session=sagemaker_session,
        )

        # training labels must be 'float32'
        knn.fit(
            knn.record_set(training_set[0][:200], training_set[1][:200].astype("float32")),
            job_name=job_name,
        )

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        model = KNNModel(knn.model_data, role="SageMakerRole", sagemaker_session=sagemaker_session)
        predictor = model.deploy(
            serverless_inference_config=ServerlessInferenceConfig(), endpoint_name=job_name
        )
        result = predictor.predict(training_set[0][:10])

        assert len(result) == 10
        for record in result:
            assert record.label["score"] is not None
