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
import io
import json
import numpy as np
import os
import pickle
import sys

import boto3

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.amazon.amazon_estimator import registry
from sagemaker.amazon.common import write_numpy_to_dense_tensor
from sagemaker.utils import name_from_base
from tests.integ import DATA_DIR, REGION
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


def fm_serializer(data):
    js = {'instances': []}
    for row in data:
        js['instances'].append({'features': row.tolist()})
    return json.dumps(js)


def test_byo_estimator():
    """Use Factorization Machines algorithm as an example here.

    First we need to prepare data for training. We take standard data set, convert it to the
    format that the algorithm can process and upload it to S3.
    Then we create the Estimator and set hyperparamets as required by the algorithm.
    Next, we can call fit() with path to the S3.
    Later the trained model is deployed and prediction is called against the endpoint.
    Default predictor is updated with json serializer and deserializer.

    """
    image_name = registry(REGION) + "/factorization-machines:1"

    with timeout(minutes=15):
        sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))
        data_path = os.path.join(DATA_DIR, 'one_p_mnist', 'mnist.pkl.gz')
        pickle_args = {} if sys.version_info.major == 2 else {'encoding': 'latin1'}

        with gzip.open(data_path, 'rb') as f:
            train_set, _, _ = pickle.load(f, **pickle_args)

        # take 100 examples for faster execution
        vectors = np.array([t.tolist() for t in train_set[0][:100]]).astype('float32')
        labels = np.where(np.array([t.tolist() for t in train_set[1][:100]]) == 0, 1.0, 0.0).astype('float32')

        buf = io.BytesIO()
        write_numpy_to_dense_tensor(buf, vectors, labels)
        buf.seek(0)

        bucket = sagemaker_session.default_bucket()
        prefix = 'test_byo_estimator'
        key = 'recordio-pb-data'
        boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)
        s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)

        estimator = Estimator(image_name=image_name,
                              role='SageMakerRole', train_instance_count=1,
                              train_instance_type='ml.c4.xlarge',
                              sagemaker_session=sagemaker_session, base_job_name='test-byo')

        estimator.set_hyperparameters(num_factors=10,
                                      feature_dim=784,
                                      mini_batch_size=100,
                                      predictor_type='binary_classifier')

        # training labels must be 'float32'
        estimator.fit({'train': s3_train_data})

    endpoint_name = name_from_base('byo')

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session, minutes=20):
        model = estimator.create_model()
        predictor = model.deploy(1, 'ml.m4.xlarge', endpoint_name=endpoint_name)
        predictor.serializer = fm_serializer
        predictor.content_type = 'application/json'
        predictor.deserializer = sagemaker.predictor.json_deserializer

        result = predictor.predict(train_set[0][:10])

        assert len(result['predictions']) == 10
        for prediction in result['predictions']:
            assert prediction['score'] is not None


def test_async_byo_estimator():
    image_name = registry(REGION) + "/factorization-machines:1"
    endpoint_name = name_from_base('byo')
    training_job_name = ""

    with timeout(minutes=5):
        sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))
        data_path = os.path.join(DATA_DIR, 'one_p_mnist', 'mnist.pkl.gz')
        pickle_args = {} if sys.version_info.major == 2 else {'encoding': 'latin1'}

        with gzip.open(data_path, 'rb') as f:
            train_set, _, _ = pickle.load(f, **pickle_args)

        # take 100 examples for faster execution
        vectors = np.array([t.tolist() for t in train_set[0][:100]]).astype('float32')
        labels = np.where(np.array([t.tolist() for t in train_set[1][:100]]) == 0, 1.0, 0.0).astype('float32')

        buf = io.BytesIO()
        write_numpy_to_dense_tensor(buf, vectors, labels)
        buf.seek(0)

        bucket = sagemaker_session.default_bucket()
        prefix = 'test_byo_estimator'
        key = 'recordio-pb-data'
        boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)
        s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)

        estimator = Estimator(image_name=image_name,
                              role='SageMakerRole', train_instance_count=1,
                              train_instance_type='ml.c4.xlarge',
                              sagemaker_session=sagemaker_session, base_job_name='test-byo')

        estimator.set_hyperparameters(num_factors=10,
                                      feature_dim=784,
                                      mini_batch_size=100,
                                      predictor_type='binary_classifier')

        # training labels must be 'float32'
        estimator.fit({'train': s3_train_data}, wait=False)
        training_job_name = estimator.latest_training_job.name

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session, minutes=30):
        estimator = Estimator.attach(training_job_name=training_job_name, sagemaker_session=sagemaker_session)
        model = estimator.create_model()
        predictor = model.deploy(1, 'ml.m4.xlarge', endpoint_name=endpoint_name)
        predictor.serializer = fm_serializer
        predictor.content_type = 'application/json'
        predictor.deserializer = sagemaker.predictor.json_deserializer

        result = predictor.predict(train_set[0][:10])

        assert len(result['predictions']) == 10
        for prediction in result['predictions']:
            assert prediction['score'] is not None

        assert estimator.train_image() == image_name
