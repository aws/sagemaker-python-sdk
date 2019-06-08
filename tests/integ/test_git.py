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

from sagemaker.pytorch.estimator import PyTorch
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.utils import sagemaker_timestamp
from tests.integ import DATA_DIR, PYTHON_VERSION, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


def test_git_support_with_pytorch(sagemaker_local_session):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = 'mnist.py'
        data_path = os.path.join(DATA_DIR, 'pytorch_mnist')
        git_config = {'repo': 'https://github.com/GaryTu1020/python-sdk-testing.git',
                      'branch': 'branch1',
                      'commit': '1867259a76ee740b99ce7ab00d6a32b582c85e06'}
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
