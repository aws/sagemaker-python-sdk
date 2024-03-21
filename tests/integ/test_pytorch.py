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

import numpy
import os
import pytest

from sagemaker.pytorch.estimator import PyTorch
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.utils import unique_name_from_base
from tests.integ import (
    DATA_DIR,
    TRAINING_DEFAULT_TIMEOUT_MINUTES,
)
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name

MNIST_DIR = os.path.join(DATA_DIR, "pytorch_mnist")
MNIST_SCRIPT = os.path.join(MNIST_DIR, "mnist.py")
PACKED_MODEL = os.path.join(MNIST_DIR, "packed_model.tar.gz")

EIA_DIR = os.path.join(DATA_DIR, "pytorch_eia")
EIA_MODEL = os.path.join(EIA_DIR, "model_mnist.tar.gz")
EIA_SCRIPT = os.path.join(EIA_DIR, "empty_inference_script.py")


@pytest.fixture(scope="module", name="pytorch_mpi_training_job")
def fixture_mpi_training_job(
    sagemaker_session,
    pytorch_training_latest_version,
    pytorch_training_latest_py_version,
    cpu_instance_type,
):

    distribution_dict = {"mpi": {"enabled": True}}
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        pytorch = _get_pytorch_estimator(
            sagemaker_session,
            pytorch_training_latest_version,
            pytorch_training_latest_py_version,
            cpu_instance_type,
            distributions_dict=distribution_dict,
        )

        pytorch.fit({"training": _upload_training_data(pytorch)})
        return pytorch.latest_training_job.name


@pytest.fixture(scope="module", name="pytorch_training_job")
def fixture_training_job(
    sagemaker_session,
    pytorch_training_latest_version,
    pytorch_training_latest_py_version,
    cpu_instance_type,
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        pytorch = _get_pytorch_estimator(
            sagemaker_session,
            pytorch_training_latest_version,
            pytorch_training_latest_py_version,
            cpu_instance_type,
        )

        pytorch.fit({"training": _upload_training_data(pytorch)})
        return pytorch.latest_training_job.name


@pytest.fixture(scope="module", name="pytorch_training_job_with_latest_infernce_version")
def fixture_training_job_with_latest_inference_version(
    sagemaker_session,
    pytorch_inference_latest_version,
    pytorch_inference_latest_py_version,
    cpu_instance_type,
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        pytorch = _get_pytorch_estimator(
            sagemaker_session,
            pytorch_inference_latest_version,
            pytorch_inference_latest_py_version,
            cpu_instance_type,
        )
        pytorch.fit({"training": _upload_training_data(pytorch)})
        return pytorch.latest_training_job.name


@pytest.mark.release
def test_framework_processing_job_with_deps(
    sagemaker_session,
    pytorch_training_latest_version,
    pytorch_training_latest_py_version,
    cpu_instance_type,
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        code_path = os.path.join(DATA_DIR, "dummy_code_bundle_with_reqs")
        entry_point = "main_script.py"

        processor = PyTorchProcessor(
            framework_version=pytorch_training_latest_version,
            py_version=pytorch_training_latest_py_version,
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            base_job_name="test-pytorch",
        )

        processor.run(
            code=entry_point,
            source_dir=code_path,
            inputs=[],
            wait=True,
        )


@pytest.mark.release
def test_fit_deploy(
    pytorch_training_job_with_latest_infernce_version, sagemaker_session, cpu_instance_type
):
    endpoint_name = unique_name_from_base("test-pytorch-sync-fit-attach-deploy")
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        estimator = PyTorch.attach(
            pytorch_training_job_with_latest_infernce_version, sagemaker_session=sagemaker_session
        )
        predictor = estimator.deploy(1, cpu_instance_type, endpoint_name=endpoint_name)
        data = numpy.zeros(shape=(1, 1, 28, 28), dtype=numpy.float32)
        predictor.predict(data)

        batch_size = 100
        data = numpy.random.rand(batch_size, 1, 28, 28).astype(numpy.float32)
        output = predictor.predict(data)

        assert output.shape == (batch_size, 10)


@pytest.mark.local_mode
def test_local_fit_deploy(
    sagemaker_local_session, pytorch_inference_latest_version, pytorch_inference_latest_py_version
):
    pytorch = PyTorch(
        entry_point=MNIST_SCRIPT,
        role="SageMakerRole",
        framework_version=pytorch_inference_latest_version,
        py_version=pytorch_inference_latest_py_version,
        instance_count=1,
        instance_type="local",
        sagemaker_session=sagemaker_local_session,
    )

    pytorch.fit({"training": "file://" + os.path.join(MNIST_DIR, "training")})

    predictor = pytorch.deploy(1, "local")
    try:
        batch_size = 100
        data = numpy.random.rand(batch_size, 1, 28, 28).astype(numpy.float32)
        output = predictor.predict(data)

        assert output.shape == (batch_size, 10)
    finally:
        predictor.delete_endpoint()


def test_deploy_model(
    pytorch_training_job,
    sagemaker_session,
    cpu_instance_type,
    pytorch_inference_latest_version,
    pytorch_inference_latest_py_version,
):
    endpoint_name = unique_name_from_base("test-pytorch-deploy-model")

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        desc = sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=pytorch_training_job
        )
        model_data = desc["ModelArtifacts"]["S3ModelArtifacts"]
        model = PyTorchModel(
            model_data,
            "SageMakerRole",
            entry_point=MNIST_SCRIPT,
            framework_version=pytorch_inference_latest_version,
            py_version=pytorch_inference_latest_py_version,
            sagemaker_session=sagemaker_session,
        )
        predictor = model.deploy(1, cpu_instance_type, endpoint_name=endpoint_name)

        batch_size = 100
        data = numpy.random.rand(batch_size, 1, 28, 28).astype(numpy.float32)
        output = predictor.predict(data)

        assert output.shape == (batch_size, 10)


def test_deploy_packed_model_with_entry_point_name(
    sagemaker_session,
    cpu_instance_type,
    pytorch_inference_latest_version,
    pytorch_inference_latest_py_version,
):
    endpoint_name = unique_name_from_base("test-pytorch-deploy-model")

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        model_data = sagemaker_session.upload_data(path=PACKED_MODEL)
        model = PyTorchModel(
            model_data,
            "SageMakerRole",
            entry_point="mnist.py",
            framework_version=pytorch_inference_latest_version,
            py_version=pytorch_inference_latest_py_version,
            sagemaker_session=sagemaker_session,
        )
        predictor = model.deploy(1, cpu_instance_type, endpoint_name=endpoint_name)

        batch_size = 100
        data = numpy.random.rand(batch_size, 1, 28, 28).astype(numpy.float32)
        output = predictor.predict(data)

        assert output.shape == (batch_size, 10)


def test_deploy_model_with_serverless_inference_config(
    pytorch_training_job,
    sagemaker_session,
    cpu_instance_type,
    pytorch_inference_latest_version,
    pytorch_inference_latest_py_version,
):
    endpoint_name = unique_name_from_base("test-pytorch-deploy-model-serverless")

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        desc = sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=pytorch_training_job
        )
        model_data = desc["ModelArtifacts"]["S3ModelArtifacts"]
        model = PyTorchModel(
            model_data,
            "SageMakerRole",
            entry_point=MNIST_SCRIPT,
            framework_version=pytorch_inference_latest_version,
            py_version=pytorch_inference_latest_py_version,
            sagemaker_session=sagemaker_session,
        )
        predictor = model.deploy(
            serverless_inference_config=ServerlessInferenceConfig(), endpoint_name=endpoint_name
        )

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
    sagemaker_session,
    pytorch_version,
    py_version,
    instance_type,
    entry_point=MNIST_SCRIPT,
    distributions_dict={},
):
    return PyTorch(
        entry_point=entry_point,
        role="SageMakerRole",
        framework_version=pytorch_version,
        py_version=py_version,
        instance_count=1,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        distributions=distributions_dict,
    )


def _is_local_mode(instance_type):
    return instance_type == "local"
