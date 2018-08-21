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
import time

import pytest

from sagemaker.tensorflow import TensorFlow
from tests.integ import DATA_DIR, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout_and_delete_endpoint_by_name, timeout
from tests.integ.vpc_utils import get_or_create_subnet_and_security_group

DATA_PATH = os.path.join(DATA_DIR, 'iris', 'data')
VPC_NAME = 'training-job-test'


@pytest.mark.continuous_testing
def test_tf(sagemaker_session, tf_full_version):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, 'iris', 'iris-dnn-classifier.py')

        estimator = TensorFlow(entry_point=script_path,
                               role='SageMakerRole',
                               framework_version=tf_full_version,
                               training_steps=1,
                               evaluation_steps=1,
                               hyperparameters={'input_tensor_name': 'inputs'},
                               train_instance_count=1,
                               train_instance_type='ml.c4.xlarge',
                               sagemaker_session=sagemaker_session,
                               base_job_name='test-tf')

        inputs = sagemaker_session.upload_data(path=DATA_PATH, key_prefix='integ-test-data/tf_iris')
        estimator.fit(inputs)
        print('job succeeded: {}'.format(estimator.latest_training_job.name))

    endpoint_name = estimator.latest_training_job.name
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        json_predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge',
                                          endpoint_name=endpoint_name)

        features = [6.4, 3.2, 4.5, 1.5]
        dict_result = json_predictor.predict({'inputs': features})
        print('predict result: {}'.format(dict_result))
        list_result = json_predictor.predict(features)
        print('predict result: {}'.format(list_result))

        assert dict_result == list_result


def test_tf_async(sagemaker_session):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, 'iris', 'iris-dnn-classifier.py')

        estimator = TensorFlow(entry_point=script_path,
                               role='SageMakerRole',
                               training_steps=1,
                               evaluation_steps=1,
                               hyperparameters={'input_tensor_name': 'inputs'},
                               train_instance_count=1,
                               train_instance_type='ml.c4.xlarge',
                               sagemaker_session=sagemaker_session,
                               base_job_name='test-tf')

        inputs = estimator.sagemaker_session.upload_data(path=DATA_PATH, key_prefix='integ-test-data/tf_iris')
        estimator.fit(inputs, wait=False)
        training_job_name = estimator.latest_training_job.name
        time.sleep(20)

    endpoint_name = training_job_name
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        estimator = TensorFlow.attach(training_job_name=training_job_name, sagemaker_session=sagemaker_session)
        json_predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge',
                                          endpoint_name=endpoint_name)

        result = json_predictor.predict([6.4, 3.2, 4.5, 1.5])
        print('predict result: {}'.format(result))


def test_failed_tf_training(sagemaker_session, tf_full_version):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, 'iris', 'failure_script.py')
        ec2_client = sagemaker_session.boto_session.client('ec2')
        subnet, security_group_id = get_or_create_subnet_and_security_group(ec2_client, VPC_NAME)
        estimator = TensorFlow(entry_point=script_path,
                               role='SageMakerRole',
                               framework_version=tf_full_version,
                               training_steps=1,
                               evaluation_steps=1,
                               hyperparameters={'input_tensor_name': 'inputs'},
                               train_instance_count=1,
                               train_instance_type='ml.c4.xlarge',
                               sagemaker_session=sagemaker_session,
                               subnets=[subnet],
                               security_group_ids=[security_group_id])

        inputs = estimator.sagemaker_session.upload_data(path=DATA_PATH, key_prefix='integ-test-data/tf-failure')

        with pytest.raises(ValueError) as e:
            estimator.fit(inputs)
        assert 'This failure is expected' in str(e.value)

        job_desc = estimator.sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=estimator.latest_training_job.name)
        assert [subnet] == job_desc['VpcConfig']['Subnets']
        assert [security_group_id] == job_desc['VpcConfig']['SecurityGroupIds']
