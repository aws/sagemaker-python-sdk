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

import numpy as np
import os
import time

import pytest

import boto3
from sagemaker.tensorflow import TensorFlow
from six.moves.urllib.parse import urlparse
from sagemaker.utils import unique_name_from_base

import tests.integ
from tests.integ import timeout

ROLE = 'SageMakerRole'

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
MNIST_RESOURCE_PATH = os.path.join(RESOURCE_PATH, 'tensorflow_mnist')
TFS_RESOURCE_PATH = os.path.join(RESOURCE_PATH, 'tfs', 'tfs-test-entrypoint-with-handler')

SCRIPT = os.path.join(MNIST_RESOURCE_PATH, 'mnist.py')
PARAMETER_SERVER_DISTRIBUTION = {'parameter_server': {'enabled': True}}
MPI_DISTRIBUTION = {'mpi': {'enabled': True}}
TAGS = [{'Key': 'some-key', 'Value': 'some-value'}]


@pytest.fixture(scope='session', params=[
    'ml.c5.xlarge',
    pytest.param('ml.p2.xlarge',
                 marks=pytest.mark.skipif(
                     tests.integ.test_region() in tests.integ.HOSTING_NO_P2_REGIONS,
                     reason='no ml.p2 instances in this region'))])
def instance_type(request):
    return request.param


@pytest.mark.skipif(tests.integ.PYTHON_VERSION != 'py3',
                    reason="Script Mode tests are only configured to run with Python 3")
def test_mnist(sagemaker_session, instance_type):
    estimator = TensorFlow(entry_point=SCRIPT,
                           role='SageMakerRole',
                           train_instance_count=1,
                           train_instance_type=instance_type,
                           sagemaker_session=sagemaker_session,
                           py_version='py3',
                           framework_version=TensorFlow.LATEST_VERSION,
                           metric_definitions=[
                               {'Name': 'train:global_steps', 'Regex': r'global_step\/sec:\s(.*)'}])
    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(MNIST_RESOURCE_PATH, 'data'),
        key_prefix='scriptmode/mnist')

    with tests.integ.timeout.timeout(minutes=tests.integ.TRAINING_DEFAULT_TIMEOUT_MINUTES):
        estimator.fit(inputs=inputs, job_name=unique_name_from_base('test-tf-sm-mnist'))
    _assert_s3_files_exist(estimator.model_dir,
                           ['graph.pbtxt', 'model.ckpt-0.index', 'model.ckpt-0.meta'])
    df = estimator.training_job_analytics.dataframe()
    assert df.size > 0


def test_server_side_encryption(sagemaker_session):
    boto_session = sagemaker_session.boto_session
    with tests.integ.kms_utils.bucket_with_encryption(boto_session, ROLE) as (
            bucket_with_kms, kms_key):
        output_path = os.path.join(bucket_with_kms, 'test-server-side-encryption',
                                   time.strftime('%y%m%d-%H%M'))

        estimator = TensorFlow(entry_point=SCRIPT,
                               role=ROLE,
                               train_instance_count=1,
                               train_instance_type='ml.c5.xlarge',
                               sagemaker_session=sagemaker_session,
                               py_version='py3',
                               framework_version=TensorFlow.LATEST_VERSION,
                               code_location=output_path,
                               output_path=output_path,
                               model_dir='/opt/ml/model',
                               output_kms_key=kms_key)

        inputs = estimator.sagemaker_session.upload_data(
            path=os.path.join(MNIST_RESOURCE_PATH, 'data'),
            key_prefix='scriptmode/mnist')

        with tests.integ.timeout.timeout(minutes=tests.integ.TRAINING_DEFAULT_TIMEOUT_MINUTES):
            estimator.fit(inputs=inputs,
                          job_name=unique_name_from_base('test-server-side-encryption'))


@pytest.mark.canary_quick
@pytest.mark.skipif(tests.integ.PYTHON_VERSION != 'py3',
                    reason="Script Mode tests are only configured to run with Python 3")
def test_mnist_distributed(sagemaker_session, instance_type):
    estimator = TensorFlow(entry_point=SCRIPT,
                           role=ROLE,
                           train_instance_count=2,
                           train_instance_type=instance_type,
                           sagemaker_session=sagemaker_session,
                           py_version=tests.integ.PYTHON_VERSION,
                           script_mode=True,
                           framework_version=TensorFlow.LATEST_VERSION,
                           distributions=PARAMETER_SERVER_DISTRIBUTION)
    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(MNIST_RESOURCE_PATH, 'data'),
        key_prefix='scriptmode/distributed_mnist')

    with tests.integ.timeout.timeout(minutes=tests.integ.TRAINING_DEFAULT_TIMEOUT_MINUTES):
        estimator.fit(inputs=inputs, job_name=unique_name_from_base('test-tf-sm-distributed'))
    _assert_s3_files_exist(estimator.model_dir,
                           ['graph.pbtxt', 'model.ckpt-0.index', 'model.ckpt-0.meta'])


def test_mnist_async(sagemaker_session):
    estimator = TensorFlow(entry_point=SCRIPT,
                           role=ROLE,
                           train_instance_count=1,
                           train_instance_type='ml.c5.4xlarge',
                           sagemaker_session=sagemaker_session,
                           py_version='py3',
                           framework_version=TensorFlow.LATEST_VERSION,
                           tags=TAGS)
    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(MNIST_RESOURCE_PATH, 'data'),
        key_prefix='scriptmode/mnist')
    estimator.fit(inputs=inputs, wait=False, job_name=unique_name_from_base('test-tf-sm-async'))
    training_job_name = estimator.latest_training_job.name
    time.sleep(20)
    endpoint_name = training_job_name
    _assert_training_job_tags_match(sagemaker_session.sagemaker_client,
                                    estimator.latest_training_job.name, TAGS)
    with tests.integ.timeout.timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        estimator = TensorFlow.attach(training_job_name=training_job_name,
                                      sagemaker_session=sagemaker_session)
        predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge',
                                     endpoint_name=endpoint_name)

        result = predictor.predict(np.zeros(784))
        print('predict result: {}'.format(result))
        _assert_endpoint_tags_match(sagemaker_session.sagemaker_client, predictor.endpoint, TAGS)
        _assert_model_tags_match(sagemaker_session.sagemaker_client,
                                 estimator.latest_training_job.name, TAGS)


@pytest.mark.skipif(tests.integ.PYTHON_VERSION != 'py3',
                    reason="Script Mode tests are only configured to run with Python 3")
def test_deploy_with_input_handlers(sagemaker_session, instance_type):
    estimator = TensorFlow(entry_point='inference.py',
                           source_dir=TFS_RESOURCE_PATH,
                           role=ROLE,
                           train_instance_count=1,
                           train_instance_type=instance_type,
                           sagemaker_session=sagemaker_session,
                           py_version='py3',
                           framework_version=TensorFlow.LATEST_VERSION,
                           tags=TAGS)

    estimator.fit(job_name=unique_name_from_base('test-tf-tfs-deploy'))

    endpoint_name = estimator.latest_training_job.name

    with timeout.timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):

        predictor = estimator.deploy(initial_instance_count=1, instance_type=instance_type,
                                     endpoint_name=endpoint_name)

        input_data = {'instances': [1.0, 2.0, 5.0]}
        expected_result = {'predictions': [4.0, 4.5, 6.0]}

        result = predictor.predict(input_data)
        assert expected_result == result


def _assert_s3_files_exist(s3_url, files):
    parsed_url = urlparse(s3_url)
    s3 = boto3.client('s3')
    contents = s3.list_objects_v2(Bucket=parsed_url.netloc, Prefix=parsed_url.path.lstrip('/'))[
        "Contents"]
    for f in files:
        found = [x['Key'] for x in contents if x['Key'].endswith(f)]
        if not found:
            raise ValueError('File {} is not found under {}'.format(f, s3_url))


def _assert_tags_match(sagemaker_client, resource_arn, tags):
    actual_tags = sagemaker_client.list_tags(ResourceArn=resource_arn)['Tags']
    assert actual_tags == tags


def _assert_model_tags_match(sagemaker_client, model_name, tags):
    model_description = sagemaker_client.describe_model(ModelName=model_name)
    _assert_tags_match(sagemaker_client, model_description['ModelArn'], tags)


def _assert_endpoint_tags_match(sagemaker_client, endpoint_name, tags):
    endpoint_description = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    _assert_tags_match(sagemaker_client, endpoint_description['EndpointArn'], tags)


def _assert_training_job_tags_match(sagemaker_client, training_job_name, tags):
    training_job_description = sagemaker_client.describe_training_job(
        TrainingJobName=training_job_name)
    _assert_tags_match(sagemaker_client, training_job_description['TrainingJobArn'], tags)
