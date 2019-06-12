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
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.utils import sagemaker_timestamp
from tests.integ import DATA_DIR, PYTHON_VERSION, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name
GIT_REPO = 'https://github.com/GaryTu1020/python-sdk-testing.git'
BRANCH = 'branch1'
COMMIT = 'b61c450200d6a309c8d24ac14b8adddc405acc56'


def test_git_support_with_pytorch(sagemaker_local_session):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = 'mnist.py'
        data_path = os.path.join(DATA_DIR, 'pytorch_mnist')
        git_config = {'repo': GIT_REPO, 'branch': BRANCH, 'commit': COMMIT}
        pytorch = PyTorch(entry_point=script_path, role='SageMakerRole', source_dir='pytorch',
                          framework_version=PyTorch.LATEST_VERSION, py_version=PYTHON_VERSION,
                          train_instance_count=1, train_instance_type='ml.c4.xlarge',
                          sagemaker_session=sagemaker_local_session, git_config=git_config)

        train_input = pytorch.sagemaker_session.upload_data(path=os.path.join(data_path, 'training'),
                                                            key_prefix='integ-test-data/pytorch_mnist/training')
        pytorch.fit({'training': train_input})

    files = [file for file in os.listdir(pytorch.source_dir)]
    assert files == ['some-file', 'mnist.py']

    endpoint_name = 'test-git_support-with-pytorch-{}'.format(sagemaker_timestamp())

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_local_session):
        desc = sagemaker_local_session.sagemaker_client.describe_training_job(pytorch.latest_training_job.name)
        model_data = desc['ModelArtifacts']['S3ModelArtifacts']
        model = PyTorchModel(model_data, 'SageMakerRole', entry_point=script_path,
                             sagemaker_session=sagemaker_local_session)
        predictor = model.deploy(1, 'ml.m4.xlarge', endpoint_name=endpoint_name)

        data = numpy.zeros(shape=(1, 1, 28, 28))
        result = predictor.predict(data)
        assert result is not None


def test_git_support_with_mxnet(sagemaker_local_session, mxnet_full_version):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = 'mnist.py'
        data_path = os.path.join(DATA_DIR, 'mxnet_mnist')
        git_config = {'repo': GIT_REPO, 'branch': BRANCH, 'commit': COMMIT}
        dependencies = ['foo/bar.py']
        mx = MXNet(entry_point=script_path,  role='SageMakerRole',
                   source_dir='mxnet', dependencies=dependencies,
                   framework_version=MXNet.LATEST_VERSION, py_version=PYTHON_VERSION,
                   train_instance_count=1, train_instance_type='ml.c4.xlarge',
                   sagemaker_session=sagemaker_local_session, git_config=git_config)

        train_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'train'),
                                                       key_prefix='integ-test-data/mxnet_mnist/train')
        test_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'test'),
                                                      key_prefix='integ-test-data/mxnet_mnist/test')

        mx.fit({'train': train_input, 'test': test_input})

    files = [file for file in os.listdir(mx.source_dir)]
    assert files == ['some_file', 'mnist.py']

    endpoint_name = 'test-git_support-with-mxnet-{}'.format(sagemaker_timestamp())

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_local_session):
        desc = sagemaker_local_session.sagemaker_client.describe_training_job(mx.latest_training_job.name)
        model_data = desc['ModelArtifacts']['S3ModelArtifacts']
        model = MXNetModel(model_data, 'SageMakerRole', entry_point=script_path,
                           py_version=PYTHON_VERSION, sagemaker_session=sagemaker_local_session,
                           framework_version=mxnet_full_version)
        predictor = model.deploy(1, 'ml.m4.xlarge', endpoint_name=endpoint_name)

        data = numpy.zeros(shape=(1, 1, 28, 28))
        result = predictor.predict(data)
        assert result is not None


# def test_git_support_for_source_dirs_and_dependencies(sagemaker_local_session):
#     source_dir = 'pytorch_source_dirs'
#     lib = 'alexa.py'
#     git_config = {'repo': GIT_REPO, 'branch': BRANCH, 'commit': COMMIT}
#
#     with open(lib, 'w') as f:
#         f.write('def question(to_anything): return 42')
#
#     estimator = PyTorch(entry_point='train.py', role='SageMakerRole', source_dir=source_dir,
#                         dependencies=[lib], git_config=git_config,
#                         py_version=PYTHON_VERSION, train_instance_count=1,
#                         train_instance_type='local',
#                         sagemaker_session=sagemaker_local_session)
#     estimator.fit()
#
#     with local_mode_utils.lock():
#         try:
#             predictor = estimator.deploy(initial_instance_count=1, instance_type='local')
#             predict_response = predictor.predict([7])
#             assert predict_response == [49]
#         finally:
#             estimator.delete_endpoint()
