# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import numpy
import pytest
import tests.integ
from tests.integ import DATA_DIR, PYTHON_VERSION, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name

from sagemaker.pytorch.estimator import PyTorch
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.utils import sagemaker_timestamp

MNIST_DIR = os.path.join(DATA_DIR, "pytorch_mnist")
MNIST_SCRIPT = os.path.join(MNIST_DIR, "mnist.py")


@pytest.fixture(scope="module", name="pytorch_training_job")
def fixture_training_job(sagemaker_session, pytorch_full_version):
    instance_type = "ml.c4.xlarge"
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        pytorch = _get_pytorch_estimator(sagemaker_session, pytorch_full_version, instance_type)

        pytorch.fit({"training": _upload_training_data(pytorch)})
        return pytorch.latest_training_job.name


@pytest.mark.canary_quick
@pytest.mark.regional_testing
def test_sync_fit_deploy(pytorch_training_job, sagemaker_session):
    # TODO: add tests against local mode when it's ready to be used
    endpoint_name = "test-pytorch-sync-fit-attach-deploy{}".format(sagemaker_timestamp())
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        estimator = PyTorch.attach(pytorch_training_job, sagemaker_session=sagemaker_session)
        predictor = estimator.deploy(1, "ml.c4.xlarge", endpoint_name=endpoint_name)
        data = numpy.zeros(shape=(1, 1, 28, 28), dtype=numpy.float32)
        predictor.predict(data)

        batch_size = 100
        data = numpy.random.rand(batch_size, 1, 28, 28).astype(numpy.float32)
        output = predictor.predict(data)

        assert output.shape == (batch_size, 10)


def test_deploy_model(pytorch_training_job, sagemaker_session):
    endpoint_name = "test-pytorch-deploy-model-{}".format(sagemaker_timestamp())

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        desc = sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=pytorch_training_job
        )
        model_data = desc["ModelArtifacts"]["S3ModelArtifacts"]
        model = PyTorchModel(
            model_data,
            "SageMakerRole",
            entry_point=MNIST_SCRIPT,
            sagemaker_session=sagemaker_session,
        )
        predictor = model.deploy(1, "ml.m4.xlarge", endpoint_name=endpoint_name)

        batch_size = 100
        data = numpy.random.rand(batch_size, 1, 28, 28).astype(numpy.float32)
        output = predictor.predict(data)

        assert output.shape == (batch_size, 10)


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.HOSTING_NO_P2_REGIONS
    or tests.integ.test_region() in tests.integ.TRAINING_NO_P2_REGIONS,
    reason="no ml.p2 instances in these regions",
)
def test_async_fit_deploy(sagemaker_session, pytorch_full_version):
    training_job_name = ""
    # TODO: add tests against local mode when it's ready to be used
    instance_type = "ml.p2.xlarge"

    with timeout(minutes=10):
        pytorch = _get_pytorch_estimator(sagemaker_session, pytorch_full_version, instance_type)

        pytorch.fit({"training": _upload_training_data(pytorch)}, wait=False)
        training_job_name = pytorch.latest_training_job.name

        print("Waiting to re-attach to the training job: %s" % training_job_name)
        time.sleep(20)

    if not _is_local_mode(instance_type):
        endpoint_name = "test-pytorch-async-fit-attach-deploy-{}".format(sagemaker_timestamp())

        with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
            print("Re-attaching now to: %s" % training_job_name)
            estimator = PyTorch.attach(
                training_job_name=training_job_name, sagemaker_session=sagemaker_session
            )
            predictor = estimator.deploy(1, instance_type, endpoint_name=endpoint_name)

            batch_size = 100
            data = numpy.random.rand(batch_size, 1, 28, 28).astype(numpy.float32)
            output = predictor.predict(data)

            assert output.shape == (batch_size, 10)


def _upload_training_data(pytorch):
    return pytorch.sagemaker_session.upload_data(
        path=os.path.join(MNIST_DIR, "training"),
        key_prefix="integ-test-data/pytorch_mnist/training",
    )


def _get_pytorch_estimator(
    sagemaker_session, pytorch_full_version, instance_type="ml.c4.xlarge", entry_point=MNIST_SCRIPT
):
    return PyTorch(
        entry_point=entry_point,
        role="SageMakerRole",
        framework_version=pytorch_full_version,
        py_version=PYTHON_VERSION,
        train_instance_count=1,
        train_instance_type=instance_type,
        sagemaker_session=sagemaker_session,
    )


def _is_local_mode(instance_type):
    return instance_type == "local"
