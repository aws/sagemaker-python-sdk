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
import tarfile
import tempfile

import boto3
import pytest

from sagemaker.tensorflow import TensorFlow
from six.moves.urllib.parse import urlparse
import tests.integ as integ
from tests.integ import timeout, vpc_test_utils

horovod_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'horovod')


@pytest.mark.parametrize('instance_type', ['ml.p3.2xlarge', 'ml.p2.xlarge'])
def test_horovod_tf_benchmarks(sagemaker_session, instance_type):
    ec2_client = sagemaker_session.boto_session.client('ec2')
    subnet_ids, security_group_id = vpc_test_utils.get_or_create_vpc_resources(
        ec2_client, sagemaker_session.boto_session.region_name)

    estimator = TensorFlow(entry_point=os.path.join(horovod_dir, 'launcher.sh'),
                           role='SageMakerRole',
                           dependencies=[os.path.join(horovod_dir, 'benchmarks')],
                           train_instance_count=2,
                           train_instance_type='ml.p3.2xlarge',
                           sagemaker_session=sagemaker_session,
                           subnets=subnet_ids,
                           security_group_ids=[security_group_id],
                           py_version=integ.PYTHON_VERSION,
                           script_mode=True,
                           framework_version='1.12',
                           distributions={'mpi': {'enabled': True}},
                           base_job_name='test-tf-horovod')

    with timeout.timeout(minutes=integ.TRAINING_DEFAULT_TIMEOUT_MINUTES):
        estimator.fit()

    _assert_s3_files_exist_in_tar(estimator.model_data, ['graph.pbtxt', 'checkpoint'])


def _assert_s3_files_exist_in_tar(s3_url, files):
    parsed_url = urlparse(s3_url)
    tmp_file = tempfile.NamedTemporaryFile()
    s3 = boto3.resource('s3')
    object = s3.Bucket(parsed_url.netloc).Object(parsed_url.path.lstrip('/'))

    with open(tmp_file.name, 'wb') as temp_file:
        object.download_fileobj(temp_file)
        with tarfile.open(tmp_file.name, 'r') as tar_file:
            for f in files:
                found = [x for x in tar_file.getnames() if x.endswith(f)]
                if not found:
                    raise ValueError('File {} is not found in {}'.format(f, s3_url))
