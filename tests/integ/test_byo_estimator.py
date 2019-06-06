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

import json
import os

import pytest

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.utils import unique_name_from_base
from tests.integ import DATA_DIR, TRAINING_DEFAULT_TIMEOUT_MINUTES, dummy_container
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


@pytest.fixture(scope='module')
def region(sagemaker_session):
    return sagemaker_session.boto_session.region_name


def fm_serializer(data):
    js = {'instances': []}
    for row in data:
        js['instances'].append({'features': row})
    return json.dumps(js)


@pytest.mark.canary_quick
def test_byo_estimator(sagemaker_integ_session, region):
    """Use Factorization Machines algorithm as an example here.

    First we need to prepare data for training. We take standard data set, convert it to the
    format that the algorithm can process and upload it to S3.
    Then we create the Estimator and set hyperparamets as required by the algorithm.
    Next, we can call fit() with path to the S3.
    Later the trained model is deployed and prediction is called against the endpoint.
    Default predictor is updated with json serializer and deserializer.

    """

    expected_hyperparamters = {
        "feature_dim": '784', "mini_batch_size": '100',
        "num_factors": '10', "predictor_type": "binary_classifier"
    }

    requests = [dummy_container.Request(input='{"instances": [{"features": [2]}]}',
                                        content_type='application/json',
                                        accept=None,
                                        response='3')]

    expected_inputdataconfig = {
        'train': {
            'RecordWrapperType': 'None',
            'S3DistributionType': 'FullyReplicated',
            'TrainingInputMode': 'File'
        }
    }

    image_name = dummy_container.build_and_push(expected_hyperparameters=expected_hyperparamters,
                                                expected_inputdataconfig=expected_inputdataconfig,
                                                expected_hosts=1,
                                                expected_requests=requests,
                                                sagemaker_session=sagemaker_integ_session)

    job_name = unique_name_from_base('byo')

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        prefix = 'test_byo_estimator'
        key = 'recordio-pb-data'

        s3_train_data = sagemaker_integ_session.upload_data(
            path=os.path.join(DATA_DIR, 'dummy_tensor'),
            key_prefix=os.path.join(prefix, 'train', key))

        estimator = Estimator(image_name=image_name,
                              role='SageMakerRole', train_instance_count=1,
                              train_instance_type='ml.c4.xlarge',
                              sagemaker_session=sagemaker_integ_session)

        estimator.set_hyperparameters(num_factors=10,
                                      feature_dim=784,
                                      mini_batch_size=100,
                                      predictor_type='binary_classifier')

        # training labels must be 'float32'
        estimator.fit({'train': s3_train_data}, job_name=job_name)

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_integ_session):
        model = estimator.create_model()
        predictor = model.deploy(1, 'ml.m4.xlarge', endpoint_name=job_name)
        predictor.serializer = fm_serializer
        predictor.content_type = 'application/json'
        predictor.deserializer = sagemaker.predictor.json_deserializer

        assert predictor.predict([[2]]) == 3


@pytest.mark.canary_quick
def test_async_byo_estimator(sagemaker_session, region):
    """Use Factorization Machines algorithm as an example here.

    First we need to prepare data for training. We take standard data set, convert it to the
    format that the algorithm can process and upload it to S3.
    Then we create the Estimator and set hyperparamets as required by the algorithm.
    Next, we can call fit() with path to the S3.
    Later the trained model is deployed and prediction is called against the endpoint.
    Default predictor is updated with json serializer and deserializer.

    """

    expected_hyperparamters = {
        "feature_dim": '784', "mini_batch_size": '100',
        "num_factors": '10', "predictor_type": "binary_classifier"
    }

    requests = [dummy_container.Request(input='{"instances": [{"features": [2]}]}',
                                        content_type='application/json',
                                        accept=None,
                                        response='3')]

    expected_inputdataconfig = {
        'train': {
            'RecordWrapperType': 'None',
            'S3DistributionType': 'FullyReplicated',
            'TrainingInputMode': 'File'
        }
    }

    image_name = dummy_container.build_and_push(expected_hyperparameters=expected_hyperparamters,
                                                expected_inputdataconfig=expected_inputdataconfig,
                                                expected_hosts=1,
                                                expected_requests=requests,
                                                sagemaker_session=sagemaker_session)

    job_name = unique_name_from_base('byo')

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        prefix = 'test_byo_estimator'
        key = 'recordio-pb-data'

        s3_train_data = sagemaker_session.upload_data(path=os.path.join(DATA_DIR, 'dummy_tensor'),
                                                      key_prefix=os.path.join(prefix, 'train', key))

        estimator = Estimator(image_name=image_name,
                              role='SageMakerRole', train_instance_count=1,
                              train_instance_type='ml.c4.xlarge',
                              sagemaker_session=sagemaker_session)

        estimator.set_hyperparameters(num_factors=10,
                                      feature_dim=784,
                                      mini_batch_size=100,
                                      predictor_type='binary_classifier')

        estimator.fit({'train': s3_train_data}, wait=False, job_name=job_name)

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        estimator = Estimator.attach(training_job_name=job_name,
                                     sagemaker_session=sagemaker_session)
        model = estimator.create_model()
        predictor = model.deploy(1, 'ml.m4.xlarge', endpoint_name=job_name)
        predictor.serializer = fm_serializer
        predictor.content_type = 'application/json'
        predictor.deserializer = sagemaker.predictor.json_deserializer

        assert predictor.predict([[2]]) == 3
