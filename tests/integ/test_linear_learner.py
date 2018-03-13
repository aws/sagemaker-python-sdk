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
import os
import pickle
import sys
import time
import pytest  # noqa
import boto3
import numpy as np

import sagemaker
from sagemaker.amazon.linear_learner import LinearLearner, LinearLearnerModel
from sagemaker.utils import name_from_base, sagemaker_timestamp

from tests.integ import DATA_DIR, REGION
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


def test_linear_learner():
    with timeout(minutes=15):
        sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))
        data_path = os.path.join(DATA_DIR, 'one_p_mnist', 'mnist.pkl.gz')
        pickle_args = {} if sys.version_info.major == 2 else {'encoding': 'latin1'}

        # Load the data into memory as numpy arrays
        with gzip.open(data_path, 'rb') as f:
            train_set, _, _ = pickle.load(f, **pickle_args)

        train_set[1][:100] = 1
        train_set[1][100:200] = 0
        train_set = train_set[0], train_set[1].astype(np.dtype('float32'))

        ll = LinearLearner('SageMakerRole', 1, 'ml.c4.2xlarge', base_job_name='test-linear-learner',
                           sagemaker_session=sagemaker_session)
        ll.binary_classifier_model_selection_criteria = 'accuracy'
        ll.target_recall = 0.5
        ll.target_precision = 0.5
        ll.positive_example_weight_mult = 0.1
        ll.epochs = 1
        ll.predictor_type = 'binary_classifier'
        ll.use_bias = True
        ll.num_models = 1
        ll.num_calibration_samples = 1
        ll.init_method = 'uniform'
        ll.init_scale = 0.5
        ll.init_sigma = 0.2
        ll.init_bias = 5
        ll.optimizer = 'adam'
        ll.loss = 'logistic'
        ll.wd = 0.5
        ll.l1 = 0.5
        ll.momentum = 0.5
        ll.learning_rate = 0.1
        ll.beta_1 = 0.1
        ll.beta_2 = 0.1
        ll.use_lr_scheduler = True
        ll.lr_scheduler_step = 2
        ll.lr_scheduler_factor = 0.5
        ll.lr_scheduler_minimum_lr = 0.1
        ll.normalize_data = False
        ll.normalize_label = False
        ll.unbias_data = True
        ll.unbias_label = False
        ll.num_point_for_scaler = 10000
        ll.fit(ll.record_set(train_set[0][:200], train_set[1][:200]))

    endpoint_name = name_from_base('linear-learner')
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session, minutes=20):

        predictor = ll.deploy(1, 'ml.c4.xlarge', endpoint_name=endpoint_name)

        result = predictor.predict(train_set[0][0:100])
        assert len(result) == 100
        for record in result:
            assert record.label["predicted_label"] is not None
            assert record.label["score"] is not None


def test_async_linear_learner():

    training_job_name = ""
    endpoint_name = 'test-linear-learner-async-{}'.format(sagemaker_timestamp())
    sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))

    with timeout(minutes=5):

        data_path = os.path.join(DATA_DIR, 'one_p_mnist', 'mnist.pkl.gz')
        pickle_args = {} if sys.version_info.major == 2 else {'encoding': 'latin1'}

        # Load the data into memory as numpy arrays
        with gzip.open(data_path, 'rb') as f:
            train_set, _, _ = pickle.load(f, **pickle_args)

        train_set[1][:100] = 1
        train_set[1][100:200] = 0
        train_set = train_set[0], train_set[1].astype(np.dtype('float32'))

        ll = LinearLearner('SageMakerRole', 1, 'ml.c4.2xlarge', base_job_name='test-linear-learner',
                           sagemaker_session=sagemaker_session)
        ll.binary_classifier_model_selection_criteria = 'accuracy'
        ll.target_recall = 0.5
        ll.target_precision = 0.5
        ll.positive_example_weight_mult = 0.1
        ll.epochs = 1
        ll.predictor_type = 'binary_classifier'
        ll.use_bias = True
        ll.num_models = 1
        ll.num_calibration_samples = 1
        ll.init_method = 'uniform'
        ll.init_scale = 0.5
        ll.init_sigma = 0.2
        ll.init_bias = 5
        ll.optimizer = 'adam'
        ll.loss = 'logistic'
        ll.wd = 0.5
        ll.l1 = 0.5
        ll.momentum = 0.5
        ll.learning_rate = 0.1
        ll.beta_1 = 0.1
        ll.beta_2 = 0.1
        ll.use_lr_scheduler = True
        ll.lr_scheduler_step = 2
        ll.lr_scheduler_factor = 0.5
        ll.lr_scheduler_minimum_lr = 0.1
        ll.normalize_data = False
        ll.normalize_label = False
        ll.unbias_data = True
        ll.unbias_label = False
        ll.num_point_for_scaler = 10000
        ll.fit(ll.record_set(train_set[0][:200], train_set[1][:200]), wait=False)
        training_job_name = ll.latest_training_job.name

        print("Waiting to re-attach to the training job: %s" % training_job_name)
        time.sleep(20)

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session, minutes=35):
        estimator = LinearLearner.attach(training_job_name=training_job_name, sagemaker_session=sagemaker_session)
        model = LinearLearnerModel(estimator.model_data, role='SageMakerRole', sagemaker_session=sagemaker_session)
        predictor = model.deploy(1, 'ml.c4.xlarge', endpoint_name=endpoint_name)

        result = predictor.predict(train_set[0][0:100])
        assert len(result) == 100
        for record in result:
            assert record.label["predicted_label"] is not None
            assert record.label["score"] is not None
