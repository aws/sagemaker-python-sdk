# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import pytest
from mock import Mock
from sagemaker.tensorflow import TensorFlow


SCRIPT = 'resnet_cifar_10.py'
TIMESTAMP = '2017-11-06-14:14:15.673'
TIME = 1510006209.073025
BUCKET_NAME = 'mybucket'
INSTANCE_COUNT = 1
INSTANCE_TYPE_GPU = 'ml.p2.xlarge'
INSTANCE_TYPE_CPU = 'ml.m4.xlarge'
CPU_IMAGE_NAME = 'sagemaker-tensorflow-py2-cpu'
GPU_IMAGE_NAME = 'sagemaker-tensorflow-py2-gpu'
REGION = 'us-west-2'
IMAGE_URI_FORMAT_STRING = "520713654638.dkr.ecr.{}.amazonaws.com/{}:{}-{}-{}"
REGION = 'us-west-2'
ROLE = 'SagemakerRole'
SOURCE_DIR = 's3://fefergerger'


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name='boto_session', region_name=REGION)
    ims = Mock(name='sagemaker_session', boto_session=boto_mock)
    ims.default_bucket = Mock(name='default_bucket', return_value=BUCKET_NAME)
    ims.expand_role = Mock(name="expand_role", return_value=ROLE)
    ims.sagemaker_client.describe_training_job = Mock(return_value={'ModelArtifacts':
                                                                    {'S3ModelArtifacts': 's3://m/m.tar.gz'}})
    return ims


# Test that we pass all necessary fields from estimator to the session when we call deploy
def test_deploy(sagemaker_session, tf_version):
    estimator = TensorFlow(entry_point=SCRIPT, source_dir=SOURCE_DIR, role=ROLE,
                           framework_version=tf_version,
                           train_instance_count=2, train_instance_type=INSTANCE_TYPE_CPU,
                           sagemaker_session=sagemaker_session,
                           base_job_name='test-cifar')

    estimator.fit('s3://mybucket/train')
    print('job succeeded: {}'.format(estimator.latest_training_job.name))

    estimator.deploy(initial_instance_count=1, instance_type=INSTANCE_TYPE_CPU)
    image = IMAGE_URI_FORMAT_STRING.format(REGION, CPU_IMAGE_NAME, tf_version, 'cpu', 'py2')
    sagemaker_session.create_model.assert_called_with(
        estimator._current_job_name,
        ROLE,
        {'Environment':
         {'SAGEMAKER_ENABLE_CLOUDWATCH_METRICS': 'false',
          'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
          'SAGEMAKER_SUBMIT_DIRECTORY': SOURCE_DIR,
          'SAGEMAKER_REGION': REGION,
          'SAGEMAKER_PROGRAM': SCRIPT},
         'Image': image,
         'ModelDataUrl': 's3://m/m.tar.gz'})
