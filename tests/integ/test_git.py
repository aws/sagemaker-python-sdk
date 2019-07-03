# Copyright 2017-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import tempfile

from tests.integ import lock as lock
from sagemaker.mxnet.estimator import MXNet
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.mxnet.model import MXNetModel
from sagemaker.sklearn.model import SKLearnModel
from tests.integ import DATA_DIR, PYTHON_VERSION

GIT_REPO = "https://github.com/aws/sagemaker-python-sdk.git"
BRANCH = "test-branch-git-config"
COMMIT = "ae15c9d7d5b97ea95ea451e4662ee43da3401d73"

# endpoint tests all use the same port, so we use this lock to prevent concurrent execution
LOCK_PATH = os.path.join(tempfile.gettempdir(), "sagemaker_test_git_lock")


@pytest.mark.local_mode
def test_git_support_with_pytorch(sagemaker_local_session):
    script_path = "mnist.py"
    data_path = os.path.join(DATA_DIR, "pytorch_mnist")
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    pytorch = PyTorch(
        entry_point=script_path,
        role="SageMakerRole",
        source_dir="pytorch",
        framework_version=PyTorch.LATEST_VERSION,
        py_version=PYTHON_VERSION,
        train_instance_count=1,
        train_instance_type="local",
        sagemaker_session=sagemaker_local_session,
        git_config=git_config,
    )

    pytorch.fit({"training": "file://" + os.path.join(data_path, "training")})

    with lock.lock(LOCK_PATH):
        try:
            predictor = pytorch.deploy(initial_instance_count=1, instance_type="local")

            data = numpy.zeros(shape=(1, 1, 28, 28)).astype(numpy.float32)
            result = predictor.predict(data)
            assert result is not None
        finally:
            predictor.delete_endpoint()


@pytest.mark.local_mode
def test_git_support_with_mxnet(sagemaker_local_session):
    script_path = "mnist.py"
    data_path = os.path.join(DATA_DIR, "mxnet_mnist")
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    source_dir = "mxnet"
    dependencies = ["foo/bar.py"]
    mx = MXNet(
        entry_point=script_path,
        role="SageMakerRole",
        source_dir=source_dir,
        dependencies=dependencies,
        framework_version=MXNet.LATEST_VERSION,
        py_version=PYTHON_VERSION,
        train_instance_count=1,
        train_instance_type="local",
        sagemaker_session=sagemaker_local_session,
        git_config=git_config,
    )

    mx.fit(
        {
            "train": "file://" + os.path.join(data_path, "train"),
            "test": "file://" + os.path.join(data_path, "test"),
        }
    )

    files = [file for file in os.listdir(mx.source_dir)]
    assert "some_file" in files
    assert "mnist.py" in files
    assert os.path.exists(mx.dependencies[0])

    with lock.lock(LOCK_PATH):
        try:
            serving_script_path = "mnist_hosting_with_custom_handlers.py"
            client = sagemaker_local_session.sagemaker_client
            desc = client.describe_training_job(TrainingJobName=mx.latest_training_job.name)
            model_data = desc["ModelArtifacts"]["S3ModelArtifacts"]
            model = MXNetModel(
                model_data,
                "SageMakerRole",
                entry_point=serving_script_path,
                source_dir=source_dir,
                dependencies=dependencies,
                py_version=PYTHON_VERSION,
                sagemaker_session=sagemaker_local_session,
                framework_version=MXNet.LATEST_VERSION,
                git_config=git_config,
            )
            predictor = model.deploy(initial_instance_count=1, instance_type="local")

            data = numpy.zeros(shape=(1, 1, 28, 28))
            result = predictor.predict(data)
            assert result is not None
        finally:
            predictor.delete_endpoint()


@pytest.mark.skipif(PYTHON_VERSION != "py3", reason="Scikit-learn image supports only python 3.")
@pytest.mark.local_mode
def test_git_support_with_sklearn(sagemaker_local_session, sklearn_full_version):
    script_path = "mnist.py"
    data_path = os.path.join(DATA_DIR, "sklearn_mnist")
    git_config = {
        "repo": "https://github.com/GaryTu1020/python-sdk-testing.git",
        "branch": "branch1",
        "commit": "aafa4e96237dd78a015d5df22bfcfef46845c3c5",
    }
    source_dir = "sklearn"
    sklearn = SKLearn(
        entry_point=script_path,
        role="SageMakerRole",
        source_dir=source_dir,
        py_version=PYTHON_VERSION,
        train_instance_count=1,
        train_instance_type="local",
        sagemaker_session=sagemaker_local_session,
        framework_version=sklearn_full_version,
        hyperparameters={"epochs": 1},
        git_config=git_config,
    )
    train_input = "file://" + os.path.join(data_path, "train")
    test_input = "file://" + os.path.join(data_path, "test")
    sklearn.fit({"train": train_input, "test": test_input})

    assert os.path.isdir(sklearn.source_dir)

    with lock.lock(LOCK_PATH):
        try:
            client = sagemaker_local_session.sagemaker_client
            desc = client.describe_training_job(TrainingJobName=sklearn.latest_training_job.name)
            model_data = desc["ModelArtifacts"]["S3ModelArtifacts"]
            model = SKLearnModel(
                model_data,
                "SageMakerRole",
                entry_point=script_path,
                source_dir=source_dir,
                sagemaker_session=sagemaker_local_session,
                git_config=git_config,
            )
            predictor = model.deploy(1, "local")

            data = numpy.zeros((100, 784), dtype="float32")
            result = predictor.predict(data)
            assert result is not None
        finally:
            predictor.delete_endpoint()
