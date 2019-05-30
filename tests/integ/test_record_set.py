# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import gzip
import os
import pickle
import sys

from six.moves.urllib.parse import urlparse

from sagemaker import KMeans
from tests.integ import DATA_DIR


def test_record_set(sagemaker_session):
    """Test the method ``AmazonAlgorithmEstimatorBase.record_set``.

    In particular, test that the objects uploaded to the S3 bucket are encrypted.
    """
    data_path = os.path.join(DATA_DIR, 'one_p_mnist', 'mnist.pkl.gz')
    pickle_args = {} if sys.version_info.major == 2 else {'encoding': 'latin1'}
    with gzip.open(data_path, 'rb') as file_object:
        train_set, _, _ = pickle.load(file_object, **pickle_args)
    kmeans = KMeans(role='SageMakerRole', train_instance_count=1,
                    train_instance_type='ml.c4.xlarge',
                    k=10, sagemaker_session=sagemaker_session)
    record_set = kmeans.record_set(train_set[0][:100], encrypt=True)
    parsed_url = urlparse(record_set.s3_data)
    s3_client = sagemaker_session.boto_session.client('s3')
    head = s3_client.head_object(Bucket=parsed_url.netloc, Key=parsed_url.path.lstrip('/'))
    assert head['ServerSideEncryption'] == 'AES256'
