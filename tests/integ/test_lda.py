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
        data_filename = 'nips-train_1.pbr'
        data_path = os.path.join(DATA_DIR, 'lda', data_filename)

        with open(data_path, 'rb') as f:
            all_records = read_records(f)

        # all records must be same
        feature_num = int(all_records[0].features['values'].float32_tensor.shape[0])

        lda = LDA(role='SageMakerRole', train_instance_type='ml.c4.xlarge', num_topics=10,
                  sagemaker_session=sagemaker_session, base_job_name='test-lda')

        # upload data and prepare the set
        data_location_key = "integ-test-data/lda-" + sagemaker_timestamp()
        sagemaker_session.upload_data(path=data_path, key_prefix=data_location_key)
        record_set = RecordSet.from_s3("s3://{}/{}".format(sagemaker_session.default_bucket(), data_location_key),
                                       num_records=len(all_records),
                                       feature_dim=feature_num,
                                       channel='train')
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
