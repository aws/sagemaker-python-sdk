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
from __future__ import absolute_import

import gzip
import os
import pickle
import sys

import pytest

from sagemaker.amazon.kmeans import KMeans
from sagemaker.mxnet.estimator import MXNet
from sagemaker.tuner import IntegerParameter, ContinuousParameter, CategoricalParameter, HyperparameterTuner
from tests.integ import DATA_DIR
from tests.integ.timeout import timeout


@pytest.mark.skip(reason='functionality is not ready yet')
def test_fit_1p(sagemaker_session):
    data_path = os.path.join(DATA_DIR, 'one_p_mnist', 'mnist.pkl.gz')
    pickle_args = {} if sys.version_info.major == 2 else {'encoding': 'latin1'}

    # Load the data into memory as numpy arrays
    with gzip.open(data_path, 'rb') as f:
        train_set, _, _ = pickle.load(f, **pickle_args)

    kmeans = KMeans(role='SageMakerRole', train_instance_count=1,
                    train_instance_type='ml.c4.xlarge',
                    k=10, sagemaker_session=sagemaker_session, base_job_name='tk',
                    output_path='s3://{}/'.format(sagemaker_session.default_bucket()))

    # set kmeans specific hp
    kmeans.init_method = 'random'
    kmeans.max_iterators = 1
    kmeans.tol = 1
    kmeans.num_trials = 1
    kmeans.local_init_method = 'kmeans++'
    kmeans.half_life_time_size = 1
    kmeans.epochs = 1

    records = kmeans.record_set(train_set[0][:100])
    test_records = kmeans.record_set(train_set[0][:100], channel='test')

    # specify which hp you want to optimize over
    hyperparameter_ranges = {'extra_center_factor': IntegerParameter(1, 10),
                             'mini_batch_size': IntegerParameter(10, 100),
                             'epochs': IntegerParameter(1, 2),
                             'init_method': CategoricalParameter(['kmeans++', 'random'])}
    objective_metric_name = 'test:msd'

    tuner = HyperparameterTuner(estimator=kmeans, objective_metric_name=objective_metric_name,
                                hyperparameter_ranges=hyperparameter_ranges, objective_type='Minimize', max_jobs=2,
                                max_parallel_jobs=2)

    tuner.fit([records, test_records])

    print('Started HPO job with name:' + tuner.latest_tuning_job.name)


@pytest.mark.skip(reason='functionality is not ready yet')
def test_mxnet_tuning(sagemaker_session, mxnet_full_version):
    with timeout(minutes=15):
        script_path = os.path.join(DATA_DIR, 'mxnet_mnist', 'tuning.py')
        data_path = os.path.join(DATA_DIR, 'mxnet_mnist')

        estimator = MXNet(entry_point=script_path,
                          role='SageMakerRole',
                          framework_version=mxnet_full_version,
                          train_instance_count=1,
                          train_instance_type='ml.m4.xlarge',
                          sagemaker_session=sagemaker_session,
                          base_job_name='hpo')

        hyperparameter_ranges = {'learning_rate': ContinuousParameter(0.01, 0.2)}
        objective_metric_name = 'Validation-accuracy'
        metric_definitions = [{'Name': 'Validation-accuracy', 'Regex': 'Validation-accuracy=([0-9\\.]+)'}]
        tuner = HyperparameterTuner(estimator, objective_metric_name, hyperparameter_ranges, metric_definitions,
                                    max_jobs=4, max_parallel_jobs=2)

        train_input = estimator.sagemaker_session.upload_data(path=os.path.join(data_path, 'train'),
                                                              key_prefix='integ-test-data/mxnet_mnist/train')
        test_input = estimator.sagemaker_session.upload_data(path=os.path.join(data_path, 'test'),
                                                             key_prefix='integ-test-data/mxnet_mnist/test')
        tuner.fit({'train': train_input, 'test': test_input})

        print('tuning job successfully created: {}'.format(tuner.latest_tuning_job.name))
