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
import subprocess
import tempfile

from tests.integ import lock as lock
from sagemaker.mxnet.estimator import MXNet
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from tests.integ import DATA_DIR


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

# endpoint tests all use the same port, so we use this lock to prevent concurrent execution
LOCK_PATH = os.path.join(tempfile.gettempdir(), "sagemaker_test_git_lock")


@pytest.mark.local_mode
def test_github(
    sagemaker_local_session, pytorch_inference_latest_version, pytorch_inference_latest_py_version
):
    script_path = "mnist.py"
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}

    pytorch = PyTorch(
        entry_point=script_path,
        role="SageMakerRole",
        source_dir="pytorch",
        framework_version=pytorch_inference_latest_version,
        py_version=pytorch_inference_latest_py_version,
        instance_count=1,
        instance_type="local",
        sagemaker_session=sagemaker_local_session,
        git_config=git_config,
    )

    data_path = os.path.join(DATA_DIR, "pytorch_mnist")
    pytorch.fit({"training": "file://" + os.path.join(data_path, "training")})

    with lock.lock(LOCK_PATH):
        try:
            predictor = pytorch.deploy(initial_instance_count=1, instance_type="local")
            data = numpy.zeros(shape=(1, 1, 28, 28)).astype(numpy.float32)
            result = predictor.predict(data)
            assert 10 == len(result[0])  # check that there is a probability for each label
        finally:
            predictor.delete_endpoint()


@pytest.mark.local_mode
@pytest.mark.skip("needs a secure authentication approach")
def test_private_github(
    sagemaker_local_session, mxnet_training_latest_version, mxnet_training_latest_py_version
):
    script_path = "mnist.py"
    data_path = os.path.join(DATA_DIR, "mxnet_mnist")
    git_config = {
        "repo": PRIVATE_GIT_REPO,
        "branch": PRIVATE_BRANCH,
        "commit": PRIVATE_COMMIT,
        "2FA_enabled": False,
        "username": "git-support-test",
        "password": "",  # TODO: find a secure approach
    }
    source_dir = "mxnet"
    dependencies = ["foo/bar.py"]
    mx = MXNet(
        entry_point=script_path,
        role="SageMakerRole",
        source_dir=source_dir,
        dependencies=dependencies,
        framework_version=mxnet_training_latest_version,
        py_version=mxnet_training_latest_py_version,
        instance_count=1,
        instance_type="local",
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
            predictor = mx.deploy(1, "local", entry_point=serving_script_path)

            data = numpy.zeros(shape=(1, 1, 28, 28))
            result = predictor.predict(data)
            assert result is not None
        finally:
            predictor.delete_endpoint()


@pytest.mark.local_mode
@pytest.mark.skip("needs a secure authentication approach")
def test_private_github_with_2fa(
    sagemaker_local_session, sklearn_latest_version, sklearn_latest_py_version
):
    script_path = "mnist.py"
    data_path = os.path.join(DATA_DIR, "sklearn_mnist")
    git_config = {
        "repo": PRIVATE_GIT_REPO_2FA,
        "branch": PRIVATE_BRANCH_2FA,
        "commit": PRIVATE_COMMIT_2FA,
        "2FA_enabled": True,
        "token": "",  # TODO: find a secure approach
    }
    source_dir = "sklearn"

    sklearn = SKLearn(
        entry_point=script_path,
        role="SageMakerRole",
        source_dir=source_dir,
        py_version=sklearn_latest_py_version,
        instance_count=1,
        instance_type="local",
        sagemaker_session=sagemaker_local_session,
        framework_version=sklearn_latest_version,
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
                framework_version=sklearn_latest_version,
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
def test_github_with_ssh_passphrase_not_configured(
    sagemaker_local_session, sklearn_latest_version, sklearn_latest_py_version
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
        instance_count=1,
        instance_type="local",
        sagemaker_session=sagemaker_local_session,
        framework_version=sklearn_latest_version,
        py_version=sklearn_latest_py_version,
        hyperparameters={"epochs": 1},
        git_config=git_config,
    )
    train_input = "file://" + os.path.join(data_path, "train")
    test_input = "file://" + os.path.join(data_path, "test")

    with pytest.raises(subprocess.CalledProcessError) as error:
        sklearn.fit({"train": train_input, "test": test_input})
    assert "returned non-zero exit status" in str(error.value)


@pytest.mark.local_mode
@pytest.mark.skip("needs a secure authentication approach")
def test_codecommit(
    sagemaker_local_session, mxnet_training_latest_version, mxnet_training_latest_py_version
):
    script_path = "mnist.py"
    data_path = os.path.join(DATA_DIR, "mxnet_mnist")
    git_config = {
        "repo": CODECOMMIT_REPO,
        "branch": CODECOMMIT_BRANCH,
        "username": "GitTest-at-142577830533",
        "password": "",  # TODO: assume a role to get temporary credentials
    }
    source_dir = "mxnet"
    dependencies = ["foo/bar.py"]
    mx = MXNet(
        entry_point=script_path,
        role="SageMakerRole",
        source_dir=source_dir,
        dependencies=dependencies,
        framework_version=mxnet_training_latest_version,
        py_version=mxnet_training_latest_py_version,
        instance_count=1,
        instance_type="local",
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
            predictor = mx.deploy(1, "local")

            data = numpy.zeros(shape=(1, 1, 28, 28))
            result = predictor.predict(data)
            assert result is not None
        finally:
            predictor.delete_endpoint()
