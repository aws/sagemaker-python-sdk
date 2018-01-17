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
import gzip
import pickle
import sys

import boto3
import os

import sagemaker
from sagemaker import ImageClassification, ImageClassificationModel
from sagemaker.utils import name_from_base
from tests.integ import DATA_DIR, REGION
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name
import urllib

def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)

        
def upload_to_s3(channel, file, bucket):
    s3 = boto3.resource('s3')
    data = open(file, "rb")
    key = channel + '/' + file
    s3.Bucket(bucket).put_object(Key=key, Body=data)

def test_image_classification():

    with timeout(minutes=15):
        sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))

        # caltech-256
        download('http://data.mxnet.io/data/caltech-256/caltech-256-60-train.rec')
        upload_to_s3('train', 'caltech-256-60-train.rec', sagemaker_session.default_bucket())
        download('http://data.mxnet.io/data/caltech-256/caltech-256-60-val.rec')
        upload_to_s3('validation', 'caltech-256-60-val.rec', sagemaker_session.default_bucket())
        ic = ImageClassification(role='SageMakerRole', train_instance_count=1,
                        train_instance_type='ml.c4.xlarge', data_location = 's3://' + sagemaker_session.default_bucket(),
                        num_classes=257, num_training_samples=15420, epochs = 1, image_shape= '3,32,32',
                        sagemaker_session=sagemaker_session, base_job_name='test-ic')

        ic.epochs = 1
        records = []
        records.append(ic.s3_record_set( 'train', channel = 'train'))
        records.append(ic.s3_record_set( 'validation', channel = 'validation'))    
        import pdb
        pdb.set_trace()           
        ic.fit(records)
    """
    endpoint_name = name_from_base('ic')
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session, minutes=20):
        model = ImageClassificationModel(ic.model_data, role='SageMakerRole', sagemaker_session=sagemaker_session)
        predictor = model.deploy(1, 'ml.c4.xlarge', endpoint_name=endpoint_name)
        result = predictor.predict(train_set[0][:10])

        assert len(result) == 10
        for record in result:
            assert record.label["closest_cluster"] is not None
            assert record.label["distance_to_cluster"] is not None
    """
