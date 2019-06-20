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

import os
import time

import pytest
import numpy

from sagemaker.sklearn.defaults import SKLEARN_VERSION
from sagemaker.sklearn import SKLearn
from sagemaker.sklearn import SKLearnModel
from sagemaker.utils import sagemaker_timestamp, unique_name_from_base
from tests.integ import DATA_DIR, PYTHON_VERSION, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


@pytest.fixture(scope='module')
def sklearn_training_job(sagemaker_session, sklearn_full_version):
    return _run_mnist_training_job(sagemaker_session, "ml.c4.xlarge", sklearn_full_version)


@pytest.mark.skipif(PYTHON_VERSION != 'py3', reason="Scikit-learn image supports only python 3.")
def test_training_with_additional_hyperparameters(sagemaker_session, sklearn_full_version):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, 'sklearn_mnist', 'mnist.py')
        data_path = os.path.join(DATA_DIR, 'sklearn_mnist')

        sklearn = SKLearn(entry_point=script_path,
                          role='SageMakerRole',
                          train_instance_type="ml.c4.xlarge",
                          framework_version=sklearn_full_version,
                          py_version=PYTHON_VERSION,
                          sagemaker_session=sagemaker_session,
                          hyperparameters={'epochs': 1})

        train_input = sklearn.sagemaker_session.upload_data(path=os.path.join(data_path, 'train'),
                                                            key_prefix='integ-test-data/sklearn_mnist/train')
        test_input = sklearn.sagemaker_session.upload_data(path=os.path.join(data_path, 'test'),
                                                           key_prefix='integ-test-data/sklearn_mnist/test')
        job_name = unique_name_from_base('test-sklearn-hp')

        sklearn.fit({'train': train_input, 'test': test_input}, job_name=job_name)
        return sklearn.latest_training_job.name


@pytest.mark.skipif(PYTHON_VERSION != 'py3', reason="Scikit-learn image supports only python 3.")
def test_training_with_network_isolation(sagemaker_session, sklearn_full_version):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, 'sklearn_mnist', 'mnist.py')
        data_path = os.path.join(DATA_DIR, 'sklearn_mnist')

        sklearn = SKLearn(entry_point=script_path,
                          role='SageMakerRole',
                          train_instance_type="ml.c4.xlarge",
                          framework_version=sklearn_full_version,
                          py_version=PYTHON_VERSION,
                          sagemaker_session=sagemaker_session,
                          hyperparameters={'epochs': 1},
                          enable_network_isolation=True)

        train_input = sklearn.sagemaker_session.upload_data(path=os.path.join(data_path, 'train'),
                                                            key_prefix='integ-test-data/sklearn_mnist/train')
        test_input = sklearn.sagemaker_session.upload_data(path=os.path.join(data_path, 'test'),
                                                           key_prefix='integ-test-data/sklearn_mnist/test')
        job_name = unique_name_from_base('test-sklearn-hp')

        sklearn.fit({'train': train_input, 'test': test_input}, job_name=job_name)
        assert sagemaker_session.sagemaker_client \
            .describe_training_job(TrainingJobName=job_name)['EnableNetworkIsolation']
        return sklearn.latest_training_job.name


@pytest.mark.canary_quick
@pytest.mark.regional_testing
@pytest.mark.skipif(PYTHON_VERSION != 'py3', reason="Scikit-learn image supports only python 3.")
def test_attach_deploy(sklearn_training_job, sagemaker_session):
    endpoint_name = 'test-sklearn-attach-deploy-{}'.format(sagemaker_timestamp())

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        estimator = SKLearn.attach(sklearn_training_job, sagemaker_session=sagemaker_session)
        predictor = estimator.deploy(1, 'ml.m4.xlarge', endpoint_name=endpoint_name)
        _predict_and_assert(predictor)


@pytest.mark.skipif(PYTHON_VERSION != 'py3', reason="Scikit-learn image supports only python 3.")
def test_deploy_model(sklearn_training_job, sagemaker_session):
    endpoint_name = 'test-sklearn-deploy-model-{}'.format(sagemaker_timestamp())
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        desc = sagemaker_session.sagemaker_client.describe_training_job(TrainingJobName=sklearn_training_job)
        model_data = desc['ModelArtifacts']['S3ModelArtifacts']
        script_path = os.path.join(DATA_DIR, 'sklearn_mnist', 'mnist.py')
        model = SKLearnModel(model_data, 'SageMakerRole', entry_point=script_path, sagemaker_session=sagemaker_session)
        predictor = model.deploy(1, "ml.m4.xlarge", endpoint_name=endpoint_name)
        _predict_and_assert(predictor)


@pytest.mark.skipif(PYTHON_VERSION != 'py3', reason="Scikit-learn image supports only python 3.")
def test_async_fit(sagemaker_session):
    endpoint_name = 'test-sklearn-attach-deploy-{}'.format(sagemaker_timestamp())

    with timeout(minutes=5):
        training_job_name = _run_mnist_training_job(sagemaker_session, "ml.c4.xlarge",
                                                    sklearn_full_version=SKLEARN_VERSION, wait=False)

        print("Waiting to re-attach to the training job: %s" % training_job_name)
        time.sleep(20)

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        print("Re-attaching now to: %s" % training_job_name)
        estimator = SKLearn.attach(training_job_name=training_job_name, sagemaker_session=sagemaker_session)
        predictor = estimator.deploy(1, "ml.c4.xlarge", endpoint_name=endpoint_name)
        _predict_and_assert(predictor)


@pytest.mark.skipif(PYTHON_VERSION != 'py3', reason="Scikit-learn image supports only python 3.")
def test_failed_training_job(sagemaker_session, sklearn_full_version):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, 'sklearn_mnist', 'failure_script.py')
        data_path = os.path.join(DATA_DIR, 'sklearn_mnist')

        sklearn = SKLearn(entry_point=script_path, role='SageMakerRole',
                          framework_version=sklearn_full_version, py_version=PYTHON_VERSION,
                          train_instance_count=1, train_instance_type='ml.c4.xlarge',
                          sagemaker_session=sagemaker_session)

        train_input = sklearn.sagemaker_session.upload_data(path=os.path.join(data_path, 'train'),
                                                            key_prefix='integ-test-data/sklearn_mnist/train')
        job_name = unique_name_from_base('test-sklearn-failed')

        with pytest.raises(ValueError):
            sklearn.fit(train_input, job_name=job_name)


def _run_mnist_training_job(sagemaker_session, instance_type, sklearn_full_version, wait=True):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):

        script_path = os.path.join(DATA_DIR, 'sklearn_mnist', 'mnist.py')

        data_path = os.path.join(DATA_DIR, 'sklearn_mnist')

        sklearn = SKLearn(entry_point=script_path, role='SageMakerRole',
                          framework_version=sklearn_full_version, py_version=PYTHON_VERSION,
                          train_instance_type=instance_type,
                          sagemaker_session=sagemaker_session, hyperparameters={'epochs': 1})

        train_input = sklearn.sagemaker_session.upload_data(path=os.path.join(data_path, 'train'),
                                                            key_prefix='integ-test-data/sklearn_mnist/train')
        test_input = sklearn.sagemaker_session.upload_data(path=os.path.join(data_path, 'test'),
                                                           key_prefix='integ-test-data/sklearn_mnist/test')
        job_name = unique_name_from_base('test-sklearn-mnist')

        sklearn.fit({'train': train_input, 'test': test_input}, wait=wait, job_name=job_name)
        return sklearn.latest_training_job.name


def _predict_and_assert(predictor):
    batch_size = 100
    data = numpy.zeros((batch_size, 784), dtype='float32')
    output = predictor.predict(data)
    assert len(output) == batch_size

    data = numpy.zeros((batch_size, 1, 28, 28), dtype='float32')
    output = predictor.predict(data)
    assert len(output) == batch_size

    data = numpy.zeros((batch_size, 28, 28), dtype='float32')
    output = predictor.predict(data)
    assert len(output) == batch_size
