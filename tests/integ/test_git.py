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
from tests.integ import DATA_DIR, PYTHON_VERSION

GIT_REPO = "https://github.com/aws/sagemaker-python-sdk.git"
BRANCH = "test-branch-git-config"
COMMIT = "329bfcf884482002c05ff7f44f62599ebc9f445a"

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
def test_git_support_with_mxnet(sagemaker_local_session, mxnet_full_version):
    script_path = "mnist.py"
    data_path = os.path.join(DATA_DIR, "mxnet_mnist")
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    dependencies = ["foo/bar.py"]
    mx = MXNet(
        entry_point=script_path,
        role="SageMakerRole",
        source_dir="mxnet",
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
            predictor = mx.deploy(initial_instance_count=1, instance_type="local")

            data = numpy.zeros(shape=(1, 1, 28, 28))
            result = predictor.predict(data)
            assert result is not None
        finally:
            predictor.delete_endpoint()
