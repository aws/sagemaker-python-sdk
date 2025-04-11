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
import time

import pytest

from sagemaker import KMeans, KMeansModel
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.utils import unique_name_from_base
from tests.integ import datasets, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


@pytest.fixture
def training_set():
    return datasets.one_p_mnist()


def test_kmeans(sagemaker_session, cpu_instance_type, training_set):
    job_name = unique_name_from_base("kmeans")
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        kmeans = KMeans(
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
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

        assert kmeans.hyperparameters() == dict(
            init_method=kmeans.init_method,
            local_lloyd_max_iter=str(kmeans.max_iterations),
            local_lloyd_tol=str(kmeans.tol),
            local_lloyd_num_trials=str(kmeans.num_trials),
            local_lloyd_init_method=kmeans.local_init_method,
            half_life_time_size=str(kmeans.half_life_time_size),
            epochs=str(kmeans.epochs),
            extra_center_factor=str(kmeans.center_factor),
            k=str(kmeans.k),
            eval_metrics=json.dumps(kmeans.eval_metrics),
            force_dense="True",
        )

        kmeans.fit(kmeans.record_set(training_set[0][:100]), job_name=job_name)

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        model = KMeansModel(
            kmeans.model_data, role="SageMakerRole", sagemaker_session=sagemaker_session
        )
        predictor = model.deploy(1, cpu_instance_type, endpoint_name=job_name)
        result = predictor.predict(training_set[0][:10])

        assert len(result) == 10
        for record in result:
            assert record.label["closest_cluster"] is not None
            assert record.label["distance_to_cluster"] is not None
        predictor.delete_model()
        with pytest.raises(Exception) as exception:
            sagemaker_session.sagemaker_client.describe_model(ModelName=model.name)
            assert "Could not find model" in str(exception.value)


def test_async_kmeans(sagemaker_session, cpu_instance_type, training_set):
    job_name = unique_name_from_base("kmeans")

    with timeout(minutes=5):
        kmeans = KMeans(
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
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

        assert kmeans.hyperparameters() == dict(
            init_method=kmeans.init_method,
            local_lloyd_max_iter=str(kmeans.max_iterations),
            local_lloyd_tol=str(kmeans.tol),
            local_lloyd_num_trials=str(kmeans.num_trials),
            local_lloyd_init_method=kmeans.local_init_method,
            half_life_time_size=str(kmeans.half_life_time_size),
            epochs=str(kmeans.epochs),
            extra_center_factor=str(kmeans.center_factor),
            k=str(kmeans.k),
            force_dense="True",
        )

        kmeans.fit(kmeans.record_set(training_set[0][:100]), wait=False, job_name=job_name)

        print("Detached from training job. Will re-attach in 20 seconds")
        time.sleep(20)
        print("attaching now...")

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        estimator = KMeans.attach(training_job_name=job_name, sagemaker_session=sagemaker_session)
        model = KMeansModel(
            estimator.model_data, role="SageMakerRole", sagemaker_session=sagemaker_session
        )
        predictor = model.deploy(1, cpu_instance_type, endpoint_name=job_name)
        result = predictor.predict(training_set[0][:10])

        assert len(result) == 10
        for record in result:
            assert record.label["closest_cluster"] is not None
            assert record.label["distance_to_cluster"] is not None


def test_kmeans_serverless_inference(sagemaker_session, cpu_instance_type, training_set):
    job_name = unique_name_from_base("kmeans-serverless")
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        kmeans = KMeans(
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
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

        assert kmeans.hyperparameters() == dict(
            init_method=kmeans.init_method,
            local_lloyd_max_iter=str(kmeans.max_iterations),
            local_lloyd_tol=str(kmeans.tol),
            local_lloyd_num_trials=str(kmeans.num_trials),
            local_lloyd_init_method=kmeans.local_init_method,
            half_life_time_size=str(kmeans.half_life_time_size),
            epochs=str(kmeans.epochs),
            extra_center_factor=str(kmeans.center_factor),
            k=str(kmeans.k),
            eval_metrics=json.dumps(kmeans.eval_metrics),
            force_dense="True",
        )

        kmeans.fit(kmeans.record_set(training_set[0][:100]), job_name=job_name)

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        model = KMeansModel(
            kmeans.model_data, role="SageMakerRole", sagemaker_session=sagemaker_session
        )
        predictor = model.deploy(
            serverless_inference_config=ServerlessInferenceConfig(), endpoint_name=job_name
        )
        result = predictor.predict(training_set[0][:10])

        assert len(result) == 10
        for record in result:
            assert record.label["closest_cluster"] is not None
            assert record.label["distance_to_cluster"] is not None
        predictor.delete_model()
        with pytest.raises(Exception) as exception:
            sagemaker_session.sagemaker_client.describe_model(ModelName=model.name)
            assert "Could not find model" in str(exception.value)
