# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import pytest

import boto3
from sagemaker.tensorflow import TensorFlow
from six.moves.urllib.parse import urlparse
import tests.integ as integ
import tests.integ.timeout as timeout

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tensorflow_mnist')
SCRIPT = os.path.join(RESOURCE_PATH, 'mnist.py')
DISTRIBUTION_ENABLED = {'parameter_server': {'enabled': True}}


@pytest.fixture(scope='session', params=['ml.c5.xlarge', 'ml.p2.xlarge'])
def instance_type(request):
    return request.param


def test_mnist(sagemaker_session, instance_type):
    estimator = TensorFlow(entry_point=SCRIPT,
                           role='SageMakerRole',
                           train_instance_count=1,
                           train_instance_type=instance_type,
                           sagemaker_session=sagemaker_session,
                           py_version='py3',
                           framework_version='1.11',
                           base_job_name='test-tf-sm-mnist')
    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(RESOURCE_PATH, 'data'),
        key_prefix='scriptmode/mnist')

    with timeout.timeout(minutes=integ.TRAINING_DEFAULT_TIMEOUT_MINUTES):
        estimator.fit(inputs)
    _assert_s3_files_exist(estimator.model_dir,
                           ['graph.pbtxt', 'model.ckpt-0.index', 'model.ckpt-0.meta', 'saved_model.pb'])


def test_mnist_distributed(sagemaker_session, instance_type):
    estimator = TensorFlow(entry_point=SCRIPT,
                           role='SageMakerRole',
                           train_instance_count=2,
                           train_instance_type=instance_type,
                           sagemaker_session=sagemaker_session,
                           py_version=integ.PYTHON_VERSION,
                           script_mode=True,
                           framework_version='1.11',
                           distributions=DISTRIBUTION_ENABLED,
                           base_job_name='test-tf-sm-mnist')
    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(RESOURCE_PATH, 'data'),
        key_prefix='scriptmode/distributed_mnist')

    with timeout.timeout(minutes=integ.TRAINING_DEFAULT_TIMEOUT_MINUTES):
        estimator.fit(inputs)
    _assert_s3_files_exist(estimator.model_dir,
                           ['graph.pbtxt', 'model.ckpt-0.index', 'model.ckpt-0.meta', 'saved_model.pb'])


def _assert_s3_files_exist(s3_url, files):
    parsed_url = urlparse(s3_url)
    s3 = boto3.client('s3')
    contents = s3.list_objects_v2(Bucket=parsed_url.netloc, Prefix=parsed_url.path.lstrip('/'))["Contents"]
    for f in files:
        found = [x['Key'] for x in contents if x['Key'].endswith(f)]
        if not found:
            raise ValueError('File {} is not found under {}'.format(f, s3_url))
