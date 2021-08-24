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

import numpy
import pytest

from sagemaker.chainer.estimator import Chainer
from sagemaker.chainer.model import ChainerModel
from sagemaker.utils import unique_name_from_base
from tests.integ import DATA_DIR, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


@pytest.fixture(scope="module")
def chainer_local_training_job(
    sagemaker_local_session, chainer_latest_version, chainer_latest_py_version
):
    return _run_mnist_training_job(
        sagemaker_local_session, "local", 1, chainer_latest_version, chainer_latest_py_version
    )


@pytest.mark.local_mode
def test_distributed_cpu_training(
    sagemaker_local_session, chainer_latest_version, chainer_latest_py_version
):
    _run_mnist_training_job(
        sagemaker_local_session, "local", 2, chainer_latest_version, chainer_latest_py_version
    )


@pytest.mark.local_mode
def test_training_with_additional_hyperparameters(
    sagemaker_local_session, chainer_latest_version, chainer_latest_py_version
):
    script_path = os.path.join(DATA_DIR, "chainer_mnist", "mnist.py")
    data_path = os.path.join(DATA_DIR, "chainer_mnist")

    chainer = Chainer(
        entry_point=script_path,
        role="SageMakerRole",
        instance_count=1,
        instance_type="local",
        framework_version=chainer_latest_version,
        py_version=chainer_latest_py_version,
        sagemaker_session=sagemaker_local_session,
        hyperparameters={"epochs": 1},
        use_mpi=True,
        num_processes=2,
        process_slots_per_host=2,
        additional_mpi_options="-x NCCL_DEBUG=INFO",
    )

    train_input = "file://" + os.path.join(data_path, "train")
    test_input = "file://" + os.path.join(data_path, "test")

    chainer.fit({"train": train_input, "test": test_input})


def test_attach_deploy(
    sagemaker_session, chainer_latest_version, chainer_latest_py_version, cpu_instance_type
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, "chainer_mnist", "mnist.py")
        data_path = os.path.join(DATA_DIR, "chainer_mnist")

        chainer = Chainer(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version=chainer_latest_version,
            py_version=chainer_latest_py_version,
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            hyperparameters={"epochs": 1},
        )

        train_input = sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/chainer_mnist/train"
        )

        test_input = sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/chainer_mnist/test"
        )

        job_name = unique_name_from_base("test-chainer-training")
        chainer.fit({"train": train_input, "test": test_input}, wait=False, job_name=job_name)

    endpoint_name = unique_name_from_base("test-chainer-attach-deploy")

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        estimator = Chainer.attach(
            chainer.latest_training_job.name, sagemaker_session=sagemaker_session
        )
        predictor = estimator.deploy(1, cpu_instance_type, endpoint_name=endpoint_name)
        _predict_and_assert(predictor)


@pytest.mark.local_mode
def test_deploy_model(
    chainer_local_training_job,
    sagemaker_local_session,
    chainer_latest_version,
    chainer_latest_py_version,
):
    script_path = os.path.join(DATA_DIR, "chainer_mnist", "mnist.py")

    model = ChainerModel(
        chainer_local_training_job.model_data,
        "SageMakerRole",
        entry_point=script_path,
        sagemaker_session=sagemaker_local_session,
        framework_version=chainer_latest_version,
        py_version=chainer_latest_py_version,
    )

    predictor = model.deploy(1, "local")
    try:
        _predict_and_assert(predictor)
    finally:
        predictor.delete_endpoint()


def _run_mnist_training_job(
    sagemaker_session, instance_type, instance_count, chainer_version, py_version
):
    script_path = (
        os.path.join(DATA_DIR, "chainer_mnist", "mnist.py")
        if instance_type == 1
        else os.path.join(DATA_DIR, "chainer_mnist", "distributed_mnist.py")
    )

    data_path = os.path.join(DATA_DIR, "chainer_mnist")

    chainer = Chainer(
        entry_point=script_path,
        role="SageMakerRole",
        framework_version=chainer_version,
        py_version=py_version,
        instance_count=instance_count,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        hyperparameters={"epochs": 1},
        # test output_path without trailing slash
        output_path="s3://{}".format(sagemaker_session.default_bucket()),
    )

    train_input = "file://" + os.path.join(data_path, "train")
    test_input = "file://" + os.path.join(data_path, "test")

    job_name = unique_name_from_base("test-chainer-training")
    chainer.fit({"train": train_input, "test": test_input}, job_name=job_name)
    return chainer


def _predict_and_assert(predictor):
    batch_size = 100
    data = numpy.zeros((batch_size, 784), dtype="float32")
    output = predictor.predict(data)
    assert len(output) == batch_size

    data = numpy.zeros((batch_size, 1, 28, 28), dtype="float32")
    output = predictor.predict(data)
    assert len(output) == batch_size

    data = numpy.zeros((batch_size, 28, 28), dtype="float32")
    output = predictor.predict(data)
    assert len(output) == batch_size
