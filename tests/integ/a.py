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
import boto3
import os

from sagemaker import Session
from sagemaker.tensorflow import TensorFlow
from tests.integ import DATA_DIR, REGION

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


sagemaker_session =  Session(boto_session=boto3.Session(region_name=REGION))


script_path = os.path.join(DATA_DIR, 'iris', 'iris-dnn-classifier.py')
data_path = os.path.join(DATA_DIR, 'iris', 'data')

estimator = TensorFlow(entry_point=script_path,
                       role='SageMakerRole',
                       training_steps=1000,
                       evaluation_steps=100,
                       hyperparameters={'input_tensor_name': 'inputs'},
                       train_instance_count=2,
                       train_instance_type='ml.c4.xlarge',
                       sagemaker_session=sagemaker_session,
                       base_job_name='test-tf')

inputs = estimator.sagemaker_session.upload_data(path=data_path, key_prefix='integ-test-data/tf_iris')
estimator.fit(inputs)
print('job succeeded: {}'.format(estimator.latest_training_job.name))
