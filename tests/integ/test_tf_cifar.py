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
import pickle

import boto3
import numpy as np
import os
import pytest

from sagemaker import Session
from sagemaker.tensorflow import TensorFlow
from tests.integ import DATA_DIR, REGION
from tests.integ.timeout import timeout_and_delete_endpoint, timeout

PICKLE_CONTENT_TYPE = 'application/python-pickle'


@pytest.fixture(scope='module')
def sagemaker_session():
    return Session(boto_session=boto3.Session(region_name=REGION))


class PickleSerializer(object):
    def __init__(self):
        self.content_type = PICKLE_CONTENT_TYPE

    def __call__(self, data):
        return pickle.dumps(data, protocol=2)


def test_cifar(sagemaker_session, tf_full_version):
    with timeout(minutes=15):
        script_path = os.path.join(DATA_DIR, 'cifar_10', 'source')

        dataset_path = os.path.join(DATA_DIR, 'cifar_10', 'data')

        estimator = TensorFlow(entry_point='resnet_cifar_10.py', source_dir=script_path, role='SageMakerRole',
                               framework_version=tf_full_version, training_steps=20, evaluation_steps=5,
                               train_instance_count=2, train_instance_type='ml.p2.xlarge',
                               sagemaker_session=sagemaker_session,
                               base_job_name='test-cifar')

        inputs = estimator.sagemaker_session.upload_data(path=dataset_path, key_prefix='data/cifar10')
        estimator.fit(inputs, logs=False)
        print('job succeeded: {}'.format(estimator.latest_training_job.name))

    with timeout_and_delete_endpoint(estimator=estimator, minutes=20):
        predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.p2.xlarge')
        predictor.serializer = PickleSerializer()
        predictor.content_type = PICKLE_CONTENT_TYPE

        data = np.random.randn(32, 32, 3)
        predict_response = predictor.predict(data)
        assert len(predict_response['outputs']['probabilities']['floatVal']) == 10
