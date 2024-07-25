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

import os
import time

import pytest
import numpy

from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.sklearn import SKLearn, SKLearnModel, SKLearnProcessor
from sagemaker.utils import unique_name_from_base
from tests.integ import DATA_DIR, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name

ROLE = "SageMakerRole"


@pytest.fixture(scope="module")
@pytest.mark.skip(
    reason="This test has always failed, but the failure was masked by a bug. "
    "This test should be fixed. Details in https://github.com/aws/sagemaker-python-sdk/pull/968"
)
def sklearn_training_job(
    sagemaker_session,
    sklearn_latest_version,
    sklearn_latest_py_version,
    cpu_instance_type,
):
    return _run_mnist_training_job(
        sagemaker_session,
        cpu_instance_type,
        sklearn_latest_version,
        sklearn_latest_py_version,
    )
    sagemaker_session.boto_region_name


def test_training_with_additional_hyperparameters(
    sagemaker_session,
    sklearn_latest_version,
    sklearn_latest_py_version,
    cpu_instance_type,
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, "sklearn_mnist", "mnist.py")
        data_path = os.path.join(DATA_DIR, "sklearn_mnist")

        sklearn = SKLearn(
            entry_point=script_path,
            role=ROLE,
            instance_type=cpu_instance_type,
            framework_version=sklearn_latest_version,
            py_version=sklearn_latest_py_version,
            sagemaker_session=sagemaker_session,
            hyperparameters={"epochs": 1},
        )

        train_input = sklearn.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/sklearn_mnist/train"
        )
        test_input = sklearn.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/sklearn_mnist/test"
        )
        job_name = unique_name_from_base("test-sklearn-hp")

        sklearn.fit({"train": train_input, "test": test_input}, job_name=job_name)


def test_training_with_network_isolation(
    sagemaker_session,
    sklearn_latest_version,
    sklearn_latest_py_version,
    cpu_instance_type,
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, "sklearn_mnist", "mnist.py")
        data_path = os.path.join(DATA_DIR, "sklearn_mnist")

        sklearn = SKLearn(
            entry_point=script_path,
            role=ROLE,
            instance_type=cpu_instance_type,
            framework_version=sklearn_latest_version,
            py_version=sklearn_latest_py_version,
            sagemaker_session=sagemaker_session,
            hyperparameters={"epochs": 1},
            enable_network_isolation=True,
        )

        train_input = sklearn.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/sklearn_mnist/train"
        )
        test_input = sklearn.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/sklearn_mnist/test"
        )
        job_name = unique_name_from_base("test-sklearn-hp")

        sklearn.fit({"train": train_input, "test": test_input}, job_name=job_name)
        assert sagemaker_session.sagemaker_client.describe_training_job(TrainingJobName=job_name)[
            "EnableNetworkIsolation"
        ]


@pytest.mark.release
@pytest.mark.skip(
    reason="This test has always failed, but the failure was masked by a bug. "
    "This test should be fixed. Details in https://github.com/aws/sagemaker-python-sdk/pull/968"
)
def test_attach_deploy(sklearn_training_job, sagemaker_session, cpu_instance_type):
    endpoint_name = unique_name_from_base("test-sklearn-attach-deploy")

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        estimator = SKLearn.attach(sklearn_training_job, sagemaker_session=sagemaker_session)
        predictor = estimator.deploy(1, cpu_instance_type, endpoint_name=endpoint_name)
        _predict_and_assert(predictor)


@pytest.mark.skip(
    reason="This test has always failed, but the failure was masked by a bug. "
    "This test should be fixed. Details in https://github.com/aws/sagemaker-python-sdk/pull/968"
)
def test_deploy_model(
    sklearn_training_job,
    sagemaker_session,
    cpu_instance_type,
    sklearn_latest_version,
    sklearn_latest_py_version,
):
    endpoint_name = unique_name_from_base("test-sklearn-deploy-model")
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        desc = sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=sklearn_training_job
        )
        model_data = desc["ModelArtifacts"]["S3ModelArtifacts"]
        script_path = os.path.join(DATA_DIR, "sklearn_mnist", "mnist.py")
        model = SKLearnModel(
            model_data,
            ROLE,
            entry_point=script_path,
            framework_version=sklearn_latest_version,
            sagemaker_session=sagemaker_session,
        )
        predictor = model.deploy(1, cpu_instance_type, endpoint_name=endpoint_name)
        _predict_and_assert(predictor)


def test_deploy_model_with_serverless_inference_config(
    sklearn_training_job,
    sagemaker_session,
):
    endpoint_name = unique_name_from_base("test-sklearn-deploy-model-serverless")
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        desc = sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=sklearn_training_job
        )
        model_data = desc["ModelArtifacts"]["S3ModelArtifacts"]
        script_path = os.path.join(DATA_DIR, "sklearn_mnist", "mnist.py")
        model = SKLearnModel(
            model_data,
            ROLE,
            entry_point=script_path,
            framework_version="1.0-1",
            sagemaker_session=sagemaker_session,
        )
        predictor = model.deploy(
            serverless_inference_config=ServerlessInferenceConfig(), endpoint_name=endpoint_name
        )
        _predict_and_assert(predictor)


@pytest.mark.skip(
    reason="This test has always failed, but the failure was masked by a bug. "
    "This test should be fixed. Details in https://github.com/aws/sagemaker-python-sdk/pull/968"
)
def test_async_fit(
    sagemaker_session,
    cpu_instance_type,
    sklearn_latest_version,
    sklearn_latest_py_version,
):
    endpoint_name = unique_name_from_base("test-sklearn-attach-deploy")

    with timeout(minutes=5):
        training_job_name = _run_mnist_training_job(
            sagemaker_session,
            cpu_instance_type,
            sklearn_version=sklearn_latest_version,
            wait=False,
        )

        print("Waiting to re-attach to the training job: %s" % training_job_name)
        time.sleep(20)

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        print("Re-attaching now to: %s" % training_job_name)
        estimator = SKLearn.attach(
            training_job_name=training_job_name, sagemaker_session=sagemaker_session
        )
        predictor = estimator.deploy(1, cpu_instance_type, endpoint_name=endpoint_name)
        _predict_and_assert(predictor)


def test_failed_training_job(
    sagemaker_session,
    sklearn_latest_version,
    sklearn_latest_py_version,
    cpu_instance_type,
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, "sklearn_mnist", "failure_script.py")
        data_path = os.path.join(DATA_DIR, "sklearn_mnist")

        sklearn = SKLearn(
            entry_point=script_path,
            role=ROLE,
            framework_version=sklearn_latest_version,
            py_version=sklearn_latest_py_version,
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
        )

        train_input = sklearn.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/sklearn_mnist/train"
        )
        job_name = unique_name_from_base("test-sklearn-failed")

        with pytest.raises(ValueError):
            sklearn.fit(train_input, job_name=job_name)


def _run_processing_job(sagemaker_session, instance_type, sklearn_version, py_version, wait=True):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):

        code_path = os.path.join(DATA_DIR, "dummy_code_bundle_with_reqs")
        entry_point = "main_script.py"

        processor = SKLearnProcessor(
            framework_version=sklearn_version,
            py_version=py_version,
            role=ROLE,
            instance_count=1,
            instance_type=instance_type,
            sagemaker_session=sagemaker_session,
            base_job_name="test-sklearn",
        )

        processor.run(
            code=entry_point,
            source_dir=code_path,
            inputs=[],
            wait=wait,
        )
        return processor.latest_job.name


def _run_mnist_training_job(
    sagemaker_session, instance_type, sklearn_version, py_version, wait=True
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):

        script_path = os.path.join(DATA_DIR, "sklearn_mnist", "mnist.py")

        data_path = os.path.join(DATA_DIR, "sklearn_mnist")

        sklearn = SKLearn(
            entry_point=script_path,
            role=ROLE,
            framework_version=sklearn_version,
            py_version=py_version,
            instance_type=instance_type,
            sagemaker_session=sagemaker_session,
            hyperparameters={"epochs": 1},
        )

        train_input = sklearn.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/sklearn_mnist/train"
        )
        test_input = sklearn.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/sklearn_mnist/test"
        )
        job_name = unique_name_from_base("test-sklearn-mnist")

        sklearn.fit({"train": train_input, "test": test_input}, wait=wait, job_name=job_name)
        return sklearn.latest_training_job.name


def _predict_and_assert(predictor):
    batch_size = 100
    data = numpy.zeros((batch_size, 784), dtype="float32")
    output = predictor.predict(data)
    assert len(output) == batch_size
