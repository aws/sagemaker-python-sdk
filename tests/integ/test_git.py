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
import subprocess
import tempfile

from tests.integ import lock as lock
from sagemaker.mxnet.estimator import MXNet
from sagemaker.pytorch.defaults import PYTORCH_VERSION
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.mxnet.model import MXNetModel
from sagemaker.sklearn.model import SKLearnModel
from tests.integ import DATA_DIR, PYTHON_VERSION

GIT_REPO = "https://github.com/aws/sagemaker-python-sdk.git"
BRANCH = "test-branch-git-config"
COMMIT = "ae15c9d7d5b97ea95ea451e4662ee43da3401d73"
PRIVATE_GIT_REPO = "https://github.com/git-support-test/test-git.git"
PRIVATE_BRANCH = "master"
PRIVATE_COMMIT = "a46d6f9add3532ca3e4e231e4108b6bad15b7373"
PRIVATE_GIT_REPO_2FA = "https://github.com/git-support-test-2fa/test-git.git"
PRIVATE_GIT_REPO_2FA_SSH = "git@github.com:git-support-test-2fa/test-git.git"
PRIVATE_BRANCH_2FA = "master"
PRIVATE_COMMIT_2FA = "52381dee030eb332a7e42d9992878d7261eb21d4"
CODECOMMIT_REPO = (
    "https://git-codecommit.us-west-2.amazonaws.com/v1/repos/sagemaker-python-sdk-git-testing-repo/"
)
CODECOMMIT_BRANCH = "master"

# Since personal access tokens will delete themselves if they are committed to GitHub repos,
# we cannot hard code them here, but have to encrypt instead
ENCRYPTED_PRIVATE_REPO_TOKEN = "e-4_1-1dc_71-f0e_f7b54a0f3b7db2757163da7b5e8c3"
PRIVATE_REPO_TOKEN = ENCRYPTED_PRIVATE_REPO_TOKEN.replace("-", "").replace("_", "")

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
        framework_version=PYTORCH_VERSION,
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
    git_config = {
        "repo": PRIVATE_GIT_REPO,
        "branch": PRIVATE_BRANCH,
        "commit": PRIVATE_COMMIT,
        "2FA_enabled": False,
        "username": "git-support-test",
        "password": "passw0rd@ %",
    }
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
        "repo": PRIVATE_GIT_REPO_2FA,
        "branch": PRIVATE_BRANCH_2FA,
        "commit": PRIVATE_COMMIT_2FA,
        "2FA_enabled": True,
        "token": PRIVATE_REPO_TOKEN,
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


@pytest.mark.local_mode
def test_git_support_with_sklearn_ssh_passphrase_not_configured(
    sagemaker_local_session, sklearn_full_version
):
    script_path = "mnist.py"
    data_path = os.path.join(DATA_DIR, "sklearn_mnist")
    git_config = {
        "repo": PRIVATE_GIT_REPO_2FA_SSH,
        "branch": PRIVATE_BRANCH_2FA,
        "commit": PRIVATE_COMMIT_2FA,
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
    with pytest.raises(subprocess.CalledProcessError) as error:
        sklearn.fit({"train": train_input, "test": test_input})
    assert "returned non-zero exit status" in str(error)


@pytest.mark.local_mode
def test_git_support_codecommit_with_mxnet(sagemaker_local_session):
    script_path = "mnist.py"
    data_path = os.path.join(DATA_DIR, "mxnet_mnist")
    git_config = {
        "repo": CODECOMMIT_REPO,
        "branch": CODECOMMIT_BRANCH,
        "username": "GitTest-at-142577830533",
        "password": "22LcZpWMtjpDG3fbOuHPooIoKoRxF36rQj7zdUvXooA=",
    }
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
            client = sagemaker_local_session.sagemaker_client
            desc = client.describe_training_job(TrainingJobName=mx.latest_training_job.name)
            model_data = desc["ModelArtifacts"]["S3ModelArtifacts"]
            model = MXNetModel(
                model_data,
                "SageMakerRole",
                entry_point=script_path,
                source_dir=source_dir,
                dependencies=dependencies,
                py_version=PYTHON_VERSION,
                sagemaker_session=sagemaker_local_session,
                framework_version=MXNet.LATEST_VERSION,
                git_config=git_config,
            )
            predictor = model.deploy(1, "local")

            data = numpy.zeros(shape=(1, 1, 28, 28))
            result = predictor.predict(data)
            assert result is not None
        finally:
            predictor.delete_endpoint()
