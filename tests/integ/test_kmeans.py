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
import time

import sagemaker
from sagemaker import KMeans, KMeansModel
from sagemaker.utils import name_from_base
from tests.integ import DATA_DIR, REGION
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


def test_kmeans():

    with timeout(minutes=15):
        sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))
        data_path = os.path.join(DATA_DIR, 'one_p_mnist', 'mnist.pkl.gz')
        pickle_args = {} if sys.version_info.major == 2 else {'encoding': 'latin1'}

        # Load the data into memory as numpy arrays
        with gzip.open(data_path, 'rb') as f:
            train_set, _, _ = pickle.load(f, **pickle_args)

        kmeans = KMeans(role='SageMakerRole', train_instance_count=1,
                        train_instance_type='ml.c4.xlarge',
                        k=10, sagemaker_session=sagemaker_session, base_job_name='test-kmeans')

        kmeans.init_method = 'random'
        kmeans.max_iterators = 1
        kmeans.tol = 1
        kmeans.num_trials = 1
        kmeans.local_init_method = 'kmeans++'
        kmeans.half_life_time_size = 1
        kmeans.epochs = 1
        kmeans.center_factor = 1

        kmeans.fit(kmeans.record_set(train_set[0][:100]))

    endpoint_name = name_from_base('kmeans')
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session, minutes=20):
        model = KMeansModel(kmeans.model_data, role='SageMakerRole', sagemaker_session=sagemaker_session)
        predictor = model.deploy(1, 'ml.c4.xlarge', endpoint_name=endpoint_name)
        result = predictor.predict(train_set[0][:10])

        assert len(result) == 10
        for record in result:
            assert record.label["closest_cluster"] is not None
            assert record.label["distance_to_cluster"] is not None


def test_async_kmeans():

    training_job_name = ""
    endpoint_name = name_from_base('kmeans')

    with timeout(minutes=5):
        sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))
        data_path = os.path.join(DATA_DIR, 'one_p_mnist', 'mnist.pkl.gz')
        pickle_args = {} if sys.version_info.major == 2 else {'encoding': 'latin1'}

        # Load the data into memory as numpy arrays
        with gzip.open(data_path, 'rb') as f:
            train_set, _, _ = pickle.load(f, **pickle_args)

        kmeans = KMeans(role='SageMakerRole', train_instance_count=1,
                        train_instance_type='ml.c4.xlarge',
                        k=10, sagemaker_session=sagemaker_session, base_job_name='test-kmeans')

        kmeans.init_method = 'random'
        kmeans.max_iterators = 1
        kmeans.tol = 1
        kmeans.num_trials = 1
        kmeans.local_init_method = 'kmeans++'
        kmeans.half_life_time_size = 1
        kmeans.epochs = 1
        kmeans.center_factor = 1

        kmeans.fit(kmeans.record_set(train_set[0][:100]), wait=False)
        training_job_name = kmeans.latest_training_job.name

        print("Detached from training job. Will re-attach in 20 seconds")
        time.sleep(20)
        print("attaching now...")

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session, minutes=35):
        estimator = KMeans.attach(training_job_name=training_job_name, sagemaker_session=sagemaker_session)
        model = KMeansModel(estimator.model_data, role='SageMakerRole', sagemaker_session=sagemaker_session)
        predictor = model.deploy(1, 'ml.c4.xlarge', endpoint_name=endpoint_name)
        result = predictor.predict(train_set[0][:10])

        assert len(result) == 10
        for record in result:
            assert record.label["closest_cluster"] is not None
            assert record.label["distance_to_cluster"] is not None
