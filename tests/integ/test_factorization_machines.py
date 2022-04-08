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

from sagemaker import FactorizationMachines, FactorizationMachinesModel
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.utils import unique_name_from_base
from tests.integ import datasets, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


@pytest.fixture
def training_set():
    return datasets.one_p_mnist()


def test_factorization_machines(sagemaker_session, cpu_instance_type, training_set):
    job_name = unique_name_from_base("fm")

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        fm = FactorizationMachines(
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
            num_factors=10,
            predictor_type="regressor",
            epochs=2,
            clip_gradient=1e2,
            eps=0.001,
            rescale_grad=1.0 / 100,
            sagemaker_session=sagemaker_session,
        )

        # training labels must be 'float32'
        fm.fit(
            fm.record_set(training_set[0][:200], training_set[1][:200].astype("float32")),
            job_name=job_name,
        )

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        model = FactorizationMachinesModel(
            fm.model_data, role="SageMakerRole", sagemaker_session=sagemaker_session
        )
        predictor = model.deploy(1, cpu_instance_type, endpoint_name=job_name)
        result = predictor.predict(training_set[0][:10])

        assert len(result) == 10
        for record in result:
            assert record.label["score"] is not None


def test_async_factorization_machines(sagemaker_session, cpu_instance_type, training_set):
    job_name = unique_name_from_base("fm")

    with timeout(minutes=5):
        fm = FactorizationMachines(
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
            num_factors=10,
            predictor_type="regressor",
            epochs=2,
            clip_gradient=1e2,
            eps=0.001,
            rescale_grad=1.0 / 100,
            sagemaker_session=sagemaker_session,
        )

        # training labels must be 'float32'
        fm.fit(
            fm.record_set(training_set[0][:200], training_set[1][:200].astype("float32")),
            job_name=job_name,
            wait=False,
        )

        print("Detached from training job. Will re-attach in 20 seconds")
        time.sleep(20)
        print("attaching now...")

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        estimator = FactorizationMachines.attach(
            training_job_name=job_name, sagemaker_session=sagemaker_session
        )
        model = FactorizationMachinesModel(
            estimator.model_data, role="SageMakerRole", sagemaker_session=sagemaker_session
        )
        predictor = model.deploy(1, cpu_instance_type, endpoint_name=job_name)
        result = predictor.predict(training_set[0][:10])

        assert len(result) == 10
        for record in result:
            assert record.label["score"] is not None


def test_factorization_machines_serverless_inference(
    sagemaker_session, cpu_instance_type, training_set
):
    job_name = unique_name_from_base("fm-serverless")

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        fm = FactorizationMachines(
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
            num_factors=10,
            predictor_type="regressor",
            epochs=2,
            clip_gradient=1e2,
            eps=0.001,
            rescale_grad=1.0 / 100,
            sagemaker_session=sagemaker_session,
        )

        # training labels must be 'float32'
        fm.fit(
            fm.record_set(training_set[0][:200], training_set[1][:200].astype("float32")),
            job_name=job_name,
        )

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        model = FactorizationMachinesModel(
            fm.model_data, role="SageMakerRole", sagemaker_session=sagemaker_session
        )
        predictor = model.deploy(
            serverless_inference_config=ServerlessInferenceConfig(), endpoint_name=job_name
        )
        result = predictor.predict(training_set[0][:10])

        assert len(result) == 10
        for record in result:
            assert record.label["score"] is not None
