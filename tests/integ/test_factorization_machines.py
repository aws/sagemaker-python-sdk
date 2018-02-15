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
import time

import boto3
import os

import sagemaker
from sagemaker import FactorizationMachines, FactorizationMachinesModel
from sagemaker.utils import name_from_base
from tests.integ import DATA_DIR, REGION
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


def test_factorization_machines():

    with timeout(minutes=15):
        sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))
        data_path = os.path.join(DATA_DIR, 'one_p_mnist', 'mnist.pkl.gz')
        pickle_args = {} if sys.version_info.major == 2 else {'encoding': 'latin1'}

        # Load the data into memory as numpy arrays
        with gzip.open(data_path, 'rb') as f:
            train_set, _, _ = pickle.load(f, **pickle_args)

        fm = FactorizationMachines(role='SageMakerRole', train_instance_count=1,
                                   train_instance_type='ml.c4.xlarge',
                                   num_factors=10, predictor_type='regressor',
                                   epochs=2, clip_gradient=1e2, eps=0.001, rescale_grad=1.0/100,
                                   sagemaker_session=sagemaker_session, base_job_name='test-fm')

        # training labels must be 'float32'
        fm.fit(fm.record_set(train_set[0][:200], train_set[1][:200].astype('float32')))

    endpoint_name = name_from_base('fm')
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session, minutes=20):
        model = FactorizationMachinesModel(fm.model_data, role='SageMakerRole', sagemaker_session=sagemaker_session)
        predictor = model.deploy(1, 'ml.c4.xlarge', endpoint_name=endpoint_name)
        result = predictor.predict(train_set[0][:10])

        assert len(result) == 10
        for record in result:
            assert record.label["score"] is not None


def test_async_factorization_machines():

    training_job_name = ""
    endpoint_name = name_from_base('factorizationMachines')
    sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))

    with timeout(minutes=5):

        data_path = os.path.join(DATA_DIR, 'one_p_mnist', 'mnist.pkl.gz')
        pickle_args = {} if sys.version_info.major == 2 else {'encoding': 'latin1'}

        # Load the data into memory as numpy arrays
        with gzip.open(data_path, 'rb') as f:
            train_set, _, _ = pickle.load(f, **pickle_args)

        fm = FactorizationMachines(role='SageMakerRole', train_instance_count=1,
                                   train_instance_type='ml.c4.xlarge',
                                   num_factors=10, predictor_type='regressor',
                                   epochs=2, clip_gradient=1e2, eps=0.001, rescale_grad=1.0 / 100,
                                   sagemaker_session=sagemaker_session, base_job_name='test-fm')

        # training labels must be 'float32'
        fm.fit(fm.record_set(train_set[0][:200], train_set[1][:200].astype('float32')), wait=False)
        training_job_name = fm.latest_training_job.name

        print("Detached from training job. Will re-attach in 20 seconds")
        time.sleep(20)
        print("attaching now...")

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session, minutes=35):
        estimator = FactorizationMachines.attach(training_job_name=training_job_name,
                                                 sagemaker_session=sagemaker_session)
        model = FactorizationMachinesModel(estimator.model_data, role='SageMakerRole',
                                           sagemaker_session=sagemaker_session)
        predictor = model.deploy(1, 'ml.c4.xlarge', endpoint_name=endpoint_name)
        result = predictor.predict(train_set[0][:10])

        assert len(result) == 10
        for record in result:
            assert record.label["score"] is not None
