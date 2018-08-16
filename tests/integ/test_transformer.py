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
from __future__ import absolute_import

import os

from sagemaker.mxnet import MXNet
from tests.integ import DATA_DIR, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout


def test_transform_mxnet(sagemaker_session):
    data_path = os.path.join(DATA_DIR, 'mxnet_mnist')
    script_path = os.path.join(data_path, 'mnist.py')

    mx = MXNet(entry_point=script_path, role='SageMakerRole', train_instance_count=1,
               train_instance_type='ml.c4.xlarge', sagemaker_session=sagemaker_session)

    train_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'train'),
                                                   key_prefix='integ-test-data/mxnet_mnist/train')
    test_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'test'),
                                                  key_prefix='integ-test-data/mxnet_mnist/test')

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        mx.fit({'train': train_input, 'test': test_input})

    transform_input_path = os.path.join(data_path, 'transform', 'data.csv')
    transform_input_key_prefix = 'integ-test-data/mxnet_mnist/transform'
    transform_input = mx.sagemaker_session.upload_data(path=transform_input_path,
                                                       key_prefix=transform_input_key_prefix)

    transformer = mx.transformer(instance_count=1, instance_type='ml.m4.xlarge')
    transformer.transform(transform_input, content_type='text/csv')
    transformer.wait()
