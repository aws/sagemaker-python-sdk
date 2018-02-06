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
import numpy as np
import os
from six.moves.urllib.parse import urlparse

import sagemaker
from sagemaker import LDA, LDAModel
from sagemaker.amazon.amazon_estimator import RecordSet
from sagemaker.amazon.common import read_records
from sagemaker.utils import name_from_base, sagemaker_timestamp

from tests.integ import DATA_DIR, REGION
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


def test_lda():

    with timeout(minutes=15):
        sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))
        data_path = os.path.join(DATA_DIR, 'lda')
        data_filename = 'nips-train_1.pbr'

        with open(os.path.join(data_path, data_filename), 'rb') as f:
            all_records = read_records(f)

        # all records must be same
        feature_num = int(all_records[0].features['values'].float32_tensor.shape[0])

        lda = LDA(role='SageMakerRole', train_instance_type='ml.c4.xlarge', num_topics=10,
                  sagemaker_session=sagemaker_session, base_job_name='test-lda')

        record_set = _prepare_record_set_from_local_files(data_path, lda.data_location,
                                                          len(all_records), feature_num, sagemaker_session)
        lda.fit(record_set, 100)

    endpoint_name = name_from_base('lda')
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session, minutes=20):
        model = LDAModel(lda.model_data, role='SageMakerRole', sagemaker_session=sagemaker_session)
        predictor = model.deploy(1, 'ml.c4.xlarge', endpoint_name=endpoint_name)

        predict_input = np.random.rand(1, feature_num)
        result = predictor.predict(predict_input)

        assert len(result) == 1
        for record in result:
            assert record.label["topic_mixture"] is not None


def _prepare_record_set_from_local_files(dir_path, destination, num_records, feature_dim, sagemaker_session):
    """Build a :class:`~RecordSet` by pointing to local files.

    Args:
        dir_path (string): Path to local directory from where the files shall be uploaded.
        destination (string): S3 path to upload the file to.
        num_records (int): Number of records in all the files
        feature_dim (int): Number of features in the data set
        sagemaker_session (sagemaker.session.Session): Session object to manage interactions with Amazon SageMaker APIs.
    Returns:
        RecordSet: A RecordSet specified by S3Prefix to to be used in training.
    """
    key_prefix = urlparse(destination).path
    key_prefix = key_prefix + '{}-{}'.format("testfiles", sagemaker_timestamp())
    key_prefix = key_prefix.lstrip('/')
    uploaded_location = sagemaker_session.upload_data(path=dir_path, key_prefix=key_prefix)
    return RecordSet(uploaded_location, num_records, feature_dim, s3_data_type='S3Prefix')
