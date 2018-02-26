# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import os
import time

import boto3
import numpy
import pytest
from sagemaker import Session
from sagemaker.mxnet.estimator import MXNet
from sagemaker.mxnet.model import MXNetModel
from sagemaker.utils import sagemaker_timestamp

from tests.integ import DATA_DIR, REGION
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


@pytest.fixture(scope='module')
def sagemaker_session():
    return Session(boto_session=boto3.Session(region_name=REGION))


@pytest.fixture(scope='module')
def mxnet_training_job(sagemaker_session, mxnet_full_version):
    with timeout(minutes=15):
        script_path = os.path.join(DATA_DIR, 'mxnet_mnist', 'mnist.py')
        data_path = os.path.join(DATA_DIR, 'mxnet_mnist')

        mx = MXNet(entry_point=script_path, role='SageMakerRole', framework_version=mxnet_full_version,
                   train_instance_count=1, train_instance_type='ml.c4.xlarge',
                   sagemaker_session=sagemaker_session)

        train_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'train'),
                                                       key_prefix='integ-test-data/mxnet_mnist/train')
        test_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'test'),
                                                      key_prefix='integ-test-data/mxnet_mnist/test')

        mx.fit({'train': train_input, 'test': test_input})
        return mx.latest_training_job.name


def test_attach_deploy(mxnet_training_job, sagemaker_session):
    endpoint_name = 'test-mxnet-attach-deploy-{}'.format(sagemaker_timestamp())

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session, minutes=20):
        estimator = MXNet.attach(mxnet_training_job, sagemaker_session=sagemaker_session)
        predictor = estimator.deploy(1, 'ml.m4.xlarge', endpoint_name=endpoint_name)
        data = numpy.zeros(shape=(1, 1, 28, 28))
        predictor.predict(data)


def test_async_fit(sagemaker_session, mxnet_full_version):

    training_job_name = ""
    endpoint_name = 'test-mxnet-attach-deploy-{}'.format(sagemaker_timestamp())

    with timeout(minutes=5):
        script_path = os.path.join(DATA_DIR, 'mxnet_mnist', 'mnist.py')
        data_path = os.path.join(DATA_DIR, 'mxnet_mnist')

        mx = MXNet(entry_point=script_path, role='SageMakerRole', framework_version=mxnet_full_version,
                   train_instance_count=1, train_instance_type='ml.c4.xlarge',
                   sagemaker_session=sagemaker_session)

        train_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'train'),
                                                       key_prefix='integ-test-data/mxnet_mnist/train')
        test_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'test'),
                                                      key_prefix='integ-test-data/mxnet_mnist/test')

        mx.fit({'train': train_input, 'test': test_input}, wait=False)
        training_job_name = mx.latest_training_job.name

        print("Waiting to re-attach to the training job: %s" % training_job_name)
        time.sleep(20)

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session, minutes=35):
        print("Re-attaching now to: %s" % training_job_name)
        estimator = MXNet.attach(training_job_name=training_job_name, sagemaker_session=sagemaker_session)
        predictor = estimator.deploy(1, 'ml.m4.xlarge', endpoint_name=endpoint_name)
        data = numpy.zeros(shape=(1, 1, 28, 28))
        predictor.predict(data)


def test_deploy_model(mxnet_training_job, sagemaker_session):
    endpoint_name = 'test-mxnet-deploy-model-{}'.format(sagemaker_timestamp())

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session, minutes=20):
        desc = sagemaker_session.sagemaker_client.describe_training_job(TrainingJobName=mxnet_training_job)
        model_data = desc['ModelArtifacts']['S3ModelArtifacts']
        script_path = os.path.join(DATA_DIR, 'mxnet_mnist', 'mnist.py')
        model = MXNetModel(model_data, 'SageMakerRole', entry_point=script_path, sagemaker_session=sagemaker_session)
        predictor = model.deploy(1, 'ml.m4.xlarge', endpoint_name=endpoint_name)

        data = numpy.zeros(shape=(1, 1, 28, 28))
        predictor.predict(data)


def test_failed_training_job(sagemaker_session, mxnet_full_version):
    with timeout(minutes=15):
        script_path = os.path.join(DATA_DIR, 'mxnet_mnist', 'failure_script.py')
        data_path = os.path.join(DATA_DIR, 'mxnet_mnist')

        mx = MXNet(entry_point=script_path, role='SageMakerRole', framework_version=mxnet_full_version,
                   train_instance_count=1, train_instance_type='ml.c4.xlarge',
                   sagemaker_session=sagemaker_session)

        train_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'train'),
                                                       key_prefix='integ-test-data/mxnet_mnist/train-failure')

        with pytest.raises(ValueError) as e:
            mx.fit(train_input)
        assert 'This failure is expected' in str(e.value)
