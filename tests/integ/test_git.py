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

from sagemaker.mxnet.estimator import MXNet
from sagemaker.mxnet.model import MXNetModel
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.utils import sagemaker_timestamp
from tests.integ import DATA_DIR, PYTHON_VERSION

GIT_REPO = 'https://github.com/GaryTu1020/sagemaker-python-sdk.git'
BRANCH = 'git_support_testing'
COMMIT = 'b8724a04ee00cb74c12c1b9a0c79d4f065c3801d'


def test_git_support_with_pytorch(sagemaker_local_session):
    script_path = 'mnist.py'
    data_path = os.path.join(DATA_DIR, 'pytorch_mnist')
    git_config = {'repo': GIT_REPO, 'branch': BRANCH, 'commit': COMMIT}
    pytorch = PyTorch(entry_point=script_path, role='SageMakerRole', source_dir='pytorch',
                      framework_version=PyTorch.LATEST_VERSION, py_version=PYTHON_VERSION,
                      train_instance_count=1, train_instance_type='local',
                      sagemaker_session=sagemaker_local_session, git_config=git_config)

    train_input = pytorch.sagemaker_session.upload_data(path=os.path.join(data_path, 'training'),
                                                        key_prefix='integ-test-data/pytorch_mnist/training')
    pytorch.fit({'training': train_input})

    predictor = pytorch.deploy(initial_instance_count=1, instance_type='local')

    data = numpy.zeros(shape=(1, 1, 28, 28)).astype(numpy.float32)
    result = predictor.predict(data)
    assert result is not None


def test_git_support_with_mxnet(sagemaker_local_session, mxnet_full_version):
    script_path = 'mnist.py'
    data_path = os.path.join(DATA_DIR, 'mxnet_mnist')
    git_config = {'repo': GIT_REPO, 'branch': BRANCH, 'commit': COMMIT}
    dependencies = ['foo/bar.py']
    mx = MXNet(entry_point=script_path, role='SageMakerRole',
               source_dir='mxnet', dependencies=dependencies,
               framework_version=MXNet.LATEST_VERSION, py_version=PYTHON_VERSION,
               train_instance_count=1, train_instance_type='local',
               sagemaker_session=sagemaker_local_session, git_config=git_config)

    train_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'train'),
                                                   key_prefix='integ-test-data/mxnet_mnist/train')
    test_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'test'),
                                                  key_prefix='integ-test-data/mxnet_mnist/test')

    mx.fit({'train': train_input, 'test': test_input})

    files = [file for file in os.listdir(mx.source_dir)]
    assert 'some_file' in files and 'mnist.py' in files
