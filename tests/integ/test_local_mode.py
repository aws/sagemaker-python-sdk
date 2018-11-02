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

import fcntl
import os
import time

import boto3
import numpy
import pytest

from sagemaker.local import LocalSession, LocalSagemakerRuntimeClient, LocalSagemakerClient
from sagemaker.mxnet import MXNet, MXNetModel
from sagemaker.tensorflow import TensorFlow
from sagemaker.fw_utils import tar_and_upload_dir
from tests.integ import DATA_DIR, PYTHON_VERSION
from tests.integ.timeout import timeout

DATA_PATH = os.path.join(DATA_DIR, 'iris', 'data')
LOCK_PATH = os.path.join(DATA_DIR, 'local_mode_lock')
DEFAULT_REGION = 'us-west-2'


class LocalNoS3Session(LocalSession):
    """
    This Session sets  local_code: True regardless of any config file settings
    """
    def __init__(self):
        super(LocalSession, self).__init__()

    def _initialize(self, boto_session, sagemaker_client, sagemaker_runtime_client):
        self.boto_session = boto3.Session(region_name=DEFAULT_REGION)
        if self.config is None:
            self.config = {
                'local':
                    {
                        'local_code': True,
                        'region_name': DEFAULT_REGION
                    }
            }

        self._region_name = DEFAULT_REGION
        self.sagemaker_client = LocalSagemakerClient(self)
        self.sagemaker_runtime_client = LocalSagemakerRuntimeClient(self.config)
        self.local_mode = True


@pytest.fixture(scope='module')
def mxnet_model(sagemaker_local_session, mxnet_full_version):
    script_path = os.path.join(DATA_DIR, 'mxnet_mnist', 'mnist.py')
    data_path = os.path.join(DATA_DIR, 'mxnet_mnist')

    mx = MXNet(entry_point=script_path, role='SageMakerRole',
               train_instance_count=1, train_instance_type='local', launch_parameter_server=True,
               sagemaker_session=sagemaker_local_session, framework_version=mxnet_full_version)

    train_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'train'),
                                                   key_prefix='integ-test-data/mxnet_mnist/train')
    test_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'test'),
                                                  key_prefix='integ-test-data/mxnet_mnist/test')

    mx.fit({'train': train_input, 'test': test_input})
    model = mx.create_model(1)
    return model


@pytest.mark.skipif(PYTHON_VERSION != 'py2', reason="TensorFlow image supports only python 2.")
def test_tf_local_mode(tf_full_version, sagemaker_local_session):
    local_mode_lock_fd = open(LOCK_PATH, 'w')
    local_mode_lock = local_mode_lock_fd.fileno()
    with timeout(minutes=5):
        script_path = os.path.join(DATA_DIR, 'iris', 'iris-dnn-classifier.py')

        estimator = TensorFlow(entry_point=script_path,
                               role='SageMakerRole',
                               framework_version=tf_full_version,
                               training_steps=1,
                               evaluation_steps=1,
                               hyperparameters={'input_tensor_name': 'inputs'},
                               train_instance_count=1,
                               train_instance_type='local',
                               base_job_name='test-tf',
                               sagemaker_session=sagemaker_local_session)

        inputs = estimator.sagemaker_session.upload_data(path=DATA_PATH,
                                                         key_prefix='integ-test-data/tf_iris')
        estimator.fit(inputs)
        print('job succeeded: {}'.format(estimator.latest_training_job.name))

    endpoint_name = estimator.latest_training_job.name
    try:
        # Since Local Mode uses the same port for serving, we need a lock in order
        # to allow concurrent test execution. The serving test is really fast so it still
        # makes sense to allow this behavior.
        fcntl.lockf(local_mode_lock, fcntl.LOCK_EX)
        json_predictor = estimator.deploy(initial_instance_count=1,
                                          instance_type='local',
                                          endpoint_name=endpoint_name)

        features = [6.4, 3.2, 4.5, 1.5]
        dict_result = json_predictor.predict({'inputs': features})
        print('predict result: {}'.format(dict_result))
        list_result = json_predictor.predict(features)
        print('predict result: {}'.format(list_result))

        assert dict_result == list_result
    finally:
        estimator.delete_endpoint()
        time.sleep(5)
        fcntl.lockf(local_mode_lock, fcntl.LOCK_UN)


@pytest.mark.skipif(PYTHON_VERSION != 'py2', reason="TensorFlow image supports only python 2.")
def test_tf_distributed_local_mode(sagemaker_local_session):
    local_mode_lock_fd = open(LOCK_PATH, 'w')
    local_mode_lock = local_mode_lock_fd.fileno()
    with timeout(minutes=5):
        script_path = os.path.join(DATA_DIR, 'iris', 'iris-dnn-classifier.py')

        estimator = TensorFlow(entry_point=script_path,
                               role='SageMakerRole',
                               training_steps=1,
                               evaluation_steps=1,
                               hyperparameters={'input_tensor_name': 'inputs'},
                               train_instance_count=3,
                               train_instance_type='local',
                               base_job_name='test-tf',
                               sagemaker_session=sagemaker_local_session)

        inputs = 'file://' + DATA_PATH
        estimator.fit(inputs)
        print('job succeeded: {}'.format(estimator.latest_training_job.name))

    endpoint_name = estimator.latest_training_job.name

    try:
        # Since Local Mode uses the same port for serving, we need a lock in order
        # to allow concurrent test execution. The serving test is really fast so it still
        # makes sense to allow this behavior.
        fcntl.lockf(local_mode_lock, fcntl.LOCK_EX)
        json_predictor = estimator.deploy(initial_instance_count=1,
                                          instance_type='local',
                                          endpoint_name=endpoint_name)

        features = [6.4, 3.2, 4.5, 1.5]
        dict_result = json_predictor.predict({'inputs': features})
        print('predict result: {}'.format(dict_result))
        list_result = json_predictor.predict(features)
        print('predict result: {}'.format(list_result))

        assert dict_result == list_result
    finally:
        estimator.delete_endpoint()
        time.sleep(5)
        fcntl.lockf(local_mode_lock, fcntl.LOCK_UN)


@pytest.mark.skipif(PYTHON_VERSION != 'py2', reason="TensorFlow image supports only python 2.")
def test_tf_local_data(sagemaker_local_session):
    local_mode_lock_fd = open(LOCK_PATH, 'w')
    local_mode_lock = local_mode_lock_fd.fileno()
    with timeout(minutes=5):
        script_path = os.path.join(DATA_DIR, 'iris', 'iris-dnn-classifier.py')

        estimator = TensorFlow(entry_point=script_path,
                               role='SageMakerRole',
                               training_steps=1,
                               evaluation_steps=1,
                               hyperparameters={'input_tensor_name': 'inputs'},
                               train_instance_count=1,
                               train_instance_type='local',
                               base_job_name='test-tf',
                               sagemaker_session=sagemaker_local_session)

        inputs = 'file://' + DATA_PATH
        estimator.fit(inputs)
        print('job succeeded: {}'.format(estimator.latest_training_job.name))

    endpoint_name = estimator.latest_training_job.name
    try:
        # Since Local Mode uses the same port for serving, we need a lock in order
        # to allow concurrent test execution. The serving test is really fast so it still
        # makes sense to allow this behavior.
        fcntl.lockf(local_mode_lock, fcntl.LOCK_EX)
        json_predictor = estimator.deploy(initial_instance_count=1,
                                          instance_type='local',
                                          endpoint_name=endpoint_name)

        features = [6.4, 3.2, 4.5, 1.5]
        dict_result = json_predictor.predict({'inputs': features})
        print('predict result: {}'.format(dict_result))
        list_result = json_predictor.predict(features)
        print('predict result: {}'.format(list_result))

        assert dict_result == list_result
    finally:
        estimator.delete_endpoint()
        fcntl.lockf(local_mode_lock, fcntl.LOCK_UN)


@pytest.mark.skipif(PYTHON_VERSION != 'py2', reason="TensorFlow image supports only python 2.")
def test_tf_local_data_local_script():
    local_mode_lock_fd = open(LOCK_PATH, 'w')
    local_mode_lock = local_mode_lock_fd.fileno()
    with timeout(minutes=5):

        script_path = os.path.join(DATA_DIR, 'iris', 'iris-dnn-classifier.py')

        estimator = TensorFlow(entry_point=script_path,
                               role='SageMakerRole',
                               training_steps=1,
                               evaluation_steps=1,
                               hyperparameters={'input_tensor_name': 'inputs'},
                               train_instance_count=1,
                               train_instance_type='local',
                               base_job_name='test-tf',
                               sagemaker_session=LocalNoS3Session())

        inputs = 'file://' + DATA_PATH

        estimator.fit(inputs)
        print('job succeeded: {}'.format(estimator.latest_training_job.name))

    endpoint_name = estimator.latest_training_job.name
    try:
        # Since Local Mode uses the same port for serving, we need a lock in order
        # to allow concurrent test execution. The serving test is really fast so it still
        # makes sense to allow this behavior.
        fcntl.lockf(local_mode_lock, fcntl.LOCK_EX)
        json_predictor = estimator.deploy(initial_instance_count=1,
                                          instance_type='local',
                                          endpoint_name=endpoint_name)

        features = [6.4, 3.2, 4.5, 1.5]
        dict_result = json_predictor.predict({'inputs': features})
        print('predict result: {}'.format(dict_result))
        list_result = json_predictor.predict(features)
        print('predict result: {}'.format(list_result))

        assert dict_result == list_result
    finally:
        estimator.delete_endpoint()
        time.sleep(5)
        fcntl.lockf(local_mode_lock, fcntl.LOCK_UN)


def test_local_mode_serving_from_s3_model(sagemaker_local_session, mxnet_model):
    local_mode_lock_fd = open(LOCK_PATH, 'w')
    local_mode_lock = local_mode_lock_fd.fileno()

    model_data = mxnet_model.model_data
    boto_session = sagemaker_local_session.boto_session
    default_bucket = sagemaker_local_session.default_bucket()
    uploaded_data = tar_and_upload_dir(boto_session, default_bucket,
                                       'test_mxnet_local_mode', '', model_data)

    s3_model = MXNetModel(model_data=uploaded_data.s3_prefix, role='SageMakerRole',
                          entry_point=mxnet_model.entry_point, image=mxnet_model.image,
                          sagemaker_session=sagemaker_local_session)

    predictor = None
    try:
        # Since Local Mode uses the same port for serving, we need a lock in order
        # to allow concurrent test execution. The serving test is really fast so it still
        # makes sense to allow this behavior.
        fcntl.lockf(local_mode_lock, fcntl.LOCK_EX)
        predictor = s3_model.deploy(initial_instance_count=1, instance_type='local')
        data = numpy.zeros(shape=(1, 1, 28, 28))
        predictor.predict(data)
    finally:
        if predictor:
            predictor.delete_endpoint()
            time.sleep(5)
        fcntl.lockf(local_mode_lock, fcntl.LOCK_UN)


def test_local_mode_serving_from_local_model(sagemaker_local_session, mxnet_model):
    local_mode_lock_fd = open(LOCK_PATH, 'w')
    local_mode_lock = local_mode_lock_fd.fileno()
    predictor = None

    try:
        # Since Local Mode uses the same port for serving, we need a lock in order
        # to allow concurrent test execution. The serving test is really fast so it still
        # makes sense to allow this behavior.
        fcntl.lockf(local_mode_lock, fcntl.LOCK_EX)
        mxnet_model.sagemaker_session = sagemaker_local_session
        predictor = mxnet_model.deploy(initial_instance_count=1, instance_type='local')
        data = numpy.zeros(shape=(1, 1, 28, 28))
        predictor.predict(data)
    finally:
        if predictor:
            predictor.delete_endpoint()
            time.sleep(5)
        fcntl.lockf(local_mode_lock, fcntl.LOCK_UN)


def test_mxnet_local_mode(sagemaker_local_session, mxnet_full_version):
    local_mode_lock_fd = open(LOCK_PATH, 'w')
    local_mode_lock = local_mode_lock_fd.fileno()

    script_path = os.path.join(DATA_DIR, 'mxnet_mnist', 'mnist.py')
    data_path = os.path.join(DATA_DIR, 'mxnet_mnist')

    mx = MXNet(entry_point=script_path, role='SageMakerRole', py_version=PYTHON_VERSION,
               train_instance_count=1, train_instance_type='local',
               sagemaker_session=sagemaker_local_session, framework_version=mxnet_full_version)

    train_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'train'),
                                                   key_prefix='integ-test-data/mxnet_mnist/train')
    test_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'test'),
                                                  key_prefix='integ-test-data/mxnet_mnist/test')

    mx.fit({'train': train_input, 'test': test_input})
    endpoint_name = mx.latest_training_job.name
    try:
        # Since Local Mode uses the same port for serving, we need a lock in order
        # to allow concurrent test execution. The serving test is really fast so it still
        # makes sense to allow this behavior.
        fcntl.lockf(local_mode_lock, fcntl.LOCK_EX)
        predictor = mx.deploy(1, 'local', endpoint_name=endpoint_name)
        data = numpy.zeros(shape=(1, 1, 28, 28))
        predictor.predict(data)
    finally:
        mx.delete_endpoint()
        time.sleep(5)
        fcntl.lockf(local_mode_lock, fcntl.LOCK_UN)


def test_mxnet_local_data_local_script():
    local_mode_lock_fd = open(LOCK_PATH, 'w')
    local_mode_lock = local_mode_lock_fd.fileno()

    data_path = os.path.join(DATA_DIR, 'mxnet_mnist')
    script_path = os.path.join(data_path, 'mnist_framework_mode.py')

    mx = MXNet(entry_point=script_path, role='SageMakerRole',
               train_instance_count=1, train_instance_type='local',
               sagemaker_session=LocalNoS3Session())

    train_input = 'file://' + os.path.join(data_path, 'train')
    test_input = 'file://' + os.path.join(data_path, 'test')

    mx.fit({'train': train_input, 'test': test_input})
    endpoint_name = mx.latest_training_job.name
    try:
        # Since Local Mode uses the same port for serving, we need a lock in order
        # to allow concurrent test execution. The serving test is really fast so it still
        # makes sense to allow this behavior.
        fcntl.lockf(local_mode_lock, fcntl.LOCK_EX)
        predictor = mx.deploy(1, 'local', endpoint_name=endpoint_name)
        data = numpy.zeros(shape=(1, 1, 28, 28))
        predictor.predict(data)
    finally:
        mx.delete_endpoint()
        time.sleep(5)
        fcntl.lockf(local_mode_lock, fcntl.LOCK_UN)


def test_local_transform_mxnet(sagemaker_local_session, tmpdir, mxnet_full_version):
    local_mode_lock_fd = open(LOCK_PATH, 'w')
    local_mode_lock = local_mode_lock_fd.fileno()
    data_path = os.path.join(DATA_DIR, 'mxnet_mnist')
    script_path = os.path.join(data_path, 'mnist.py')

    mx = MXNet(entry_point=script_path, role='SageMakerRole', train_instance_count=1,
               train_instance_type='ml.c4.xlarge', framework_version=mxnet_full_version,
               sagemaker_session=sagemaker_local_session, launch_parameter_server=True)

    train_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'train'),
                                                   key_prefix='integ-test-data/mxnet_mnist/train')
    test_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'test'),
                                                  key_prefix='integ-test-data/mxnet_mnist/test')

    with timeout(minutes=15):
        mx.fit({'train': train_input, 'test': test_input})

    transform_input_path = os.path.join(data_path, 'transform')
    transform_input_key_prefix = 'integ-test-data/mxnet_mnist/transform'
    transform_input = mx.sagemaker_session.upload_data(path=transform_input_path,
                                                       key_prefix=transform_input_key_prefix)

    output_path = 'file://%s' % (str(tmpdir))
    transformer = mx.transformer(1, 'local', assemble_with='Line', max_payload=1,
                                 strategy='SingleRecord', output_path=output_path)

    # Since Local Mode uses the same port for serving, we need a lock in order
    # to allow concurrent test execution.
    fcntl.lockf(local_mode_lock, fcntl.LOCK_EX)
    transformer.transform(transform_input, content_type='text/csv', split_type='Line')
    transformer.wait()
    time.sleep(5)
    fcntl.lockf(local_mode_lock, fcntl.LOCK_UN)

    assert os.path.exists(os.path.join(str(tmpdir), 'data.csv.out'))
