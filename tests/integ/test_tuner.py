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
import io
import json
import os
import pickle
import sys
import time

import boto3
import numpy as np
import pytest

from sagemaker import KMeans, LDA, RandomCutForest
from sagemaker.amazon.amazon_estimator import registry
from sagemaker.amazon.common import read_records, write_numpy_to_dense_tensor
from sagemaker.chainer import Chainer
from sagemaker.estimator import Estimator
from sagemaker.mxnet.estimator import MXNet
from sagemaker.predictor import json_deserializer
from sagemaker.pytorch import PyTorch
from sagemaker.tensorflow import TensorFlow
from sagemaker.tuner import IntegerParameter, ContinuousParameter, CategoricalParameter, HyperparameterTuner
from tests.integ import DATA_DIR
from tests.integ.record_set import prepare_record_set_from_local_files
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name

DATA_PATH = os.path.join(DATA_DIR, 'iris', 'data')


@pytest.mark.continuous_testing
def test_tuning_kmeans(sagemaker_session):
    with timeout(minutes=20):
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

        print('Started hyperparameter tuning job with name:' + tuner.latest_tuning_job.name)

        time.sleep(15)
        tuner.wait()

    best_training_job = tuner.best_training_job()
    with timeout_and_delete_endpoint_by_name(best_training_job, sagemaker_session):
        predictor = tuner.deploy(1, 'ml.c4.xlarge')
        result = predictor.predict(train_set[0][:10])

        assert len(result) == 10
        for record in result:
            assert record.label['closest_cluster'] is not None
            assert record.label['distance_to_cluster'] is not None


def test_tuning_lda(sagemaker_session):
    with timeout(minutes=20):
        data_path = os.path.join(DATA_DIR, 'lda')
        data_filename = 'nips-train_1.pbr'

        with open(os.path.join(data_path, data_filename), 'rb') as f:
            all_records = read_records(f)

        # all records must be same
        feature_num = int(all_records[0].features['values'].float32_tensor.shape[0])

        lda = LDA(role='SageMakerRole', train_instance_type='ml.c4.xlarge', num_topics=10,
                  sagemaker_session=sagemaker_session, base_job_name='test-lda')

        record_set = prepare_record_set_from_local_files(data_path, lda.data_location,
                                                         len(all_records), feature_num, sagemaker_session)
        test_record_set = prepare_record_set_from_local_files(data_path, lda.data_location,
                                                              len(all_records), feature_num, sagemaker_session)
        test_record_set.channel = 'test'

        # specify which hp you want to optimize over
        hyperparameter_ranges = {'alpha0': ContinuousParameter(1, 10),
                                 'num_topics': IntegerParameter(1, 2)}
        objective_metric_name = 'test:pwll'

        tuner = HyperparameterTuner(estimator=lda, objective_metric_name=objective_metric_name,
                                    hyperparameter_ranges=hyperparameter_ranges, objective_type='Maximize', max_jobs=2,
                                    max_parallel_jobs=2)

        tuner.fit([record_set, test_record_set], mini_batch_size=1)

        print('Started hyperparameter tuning job with name:' + tuner.latest_tuning_job.name)

        time.sleep(15)
        tuner.wait()

    best_training_job = tuner.best_training_job()
    with timeout_and_delete_endpoint_by_name(best_training_job, sagemaker_session):
        predictor = tuner.deploy(1, 'ml.c4.xlarge')
        predict_input = np.random.rand(1, feature_num)
        result = predictor.predict(predict_input)

        assert len(result) == 1
        for record in result:
            assert record.label['topic_mixture'] is not None


@pytest.mark.continuous_testing
def test_stop_tuning_job(sagemaker_session):
    feature_num = 14
    train_input = np.random.rand(1000, feature_num)

    rcf = RandomCutForest(role='SageMakerRole', train_instance_count=1, train_instance_type='ml.c4.xlarge',
                          num_trees=50, num_samples_per_tree=20, sagemaker_session=sagemaker_session,
                          base_job_name='test-randomcutforest')

    records = rcf.record_set(train_input)
    records.distribution = 'FullyReplicated'

    test_records = rcf.record_set(train_input, channel='test')
    test_records.distribution = 'FullyReplicated'

    hyperparameter_ranges = {'num_trees': IntegerParameter(50, 100),
                             'num_samples_per_tree': IntegerParameter(1, 2)}

    objective_metric_name = 'test:f1'
    tuner = HyperparameterTuner(estimator=rcf, objective_metric_name=objective_metric_name,
                                hyperparameter_ranges=hyperparameter_ranges, objective_type='Maximize', max_jobs=2,
                                max_parallel_jobs=2)

    tuner.fit([records, test_records])

    time.sleep(15)

    latest_tuning_job_name = tuner.latest_tuning_job.name

    print('Attempting to stop {}'.format(latest_tuning_job_name))

    tuner.stop_tuning_job()

    desc = tuner.latest_tuning_job.sagemaker_session.sagemaker_client\
        .describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=latest_tuning_job_name)
    assert desc['HyperParameterTuningJobStatus'] == 'Stopping'


@pytest.mark.continuous_testing
def test_tuning_mxnet(sagemaker_session):
    with timeout(minutes=15):
        script_path = os.path.join(DATA_DIR, 'mxnet_mnist', 'tuning.py')
        data_path = os.path.join(DATA_DIR, 'mxnet_mnist')

        estimator = MXNet(entry_point=script_path,
                          role='SageMakerRole',
                          train_instance_count=1,
                          train_instance_type='ml.m4.xlarge',
                          sagemaker_session=sagemaker_session,
                          base_job_name='tune-mxnet')

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

        print('Started hyperparameter tuning job with name:' + tuner.latest_tuning_job.name)

        time.sleep(15)
        tuner.wait()

    best_training_job = tuner.best_training_job()
    with timeout_and_delete_endpoint_by_name(best_training_job, sagemaker_session):
        predictor = tuner.deploy(1, 'ml.c4.xlarge')
        data = np.zeros(shape=(1, 1, 28, 28))
        predictor.predict(data)


@pytest.mark.continuous_testing
def test_tuning_tf(sagemaker_session):
    with timeout(minutes=15):
        script_path = os.path.join(DATA_DIR, 'iris', 'iris-dnn-classifier.py')

        estimator = TensorFlow(entry_point=script_path,
                               role='SageMakerRole',
                               training_steps=1,
                               evaluation_steps=1,
                               hyperparameters={'input_tensor_name': 'inputs'},
                               train_instance_count=1,
                               train_instance_type='ml.c4.xlarge',
                               sagemaker_session=sagemaker_session,
                               base_job_name='tune-tf')

        inputs = sagemaker_session.upload_data(path=DATA_PATH, key_prefix='integ-test-data/tf_iris')
        hyperparameter_ranges = {'learning_rate': ContinuousParameter(0.05, 0.2)}

        objective_metric_name = 'loss'
        metric_definitions = [{'Name': 'loss', 'Regex': 'loss = ([0-9\\.]+)'}]

        tuner = HyperparameterTuner(estimator, objective_metric_name, hyperparameter_ranges, metric_definitions,
                                    objective_type='Minimize', max_jobs=2, max_parallel_jobs=2)

        tuner.fit(inputs)

        print('Started hyperparameter tuning job with name:' + tuner.latest_tuning_job.name)

        time.sleep(15)
        tuner.wait()

    best_training_job = tuner.best_training_job()
    with timeout_and_delete_endpoint_by_name(best_training_job, sagemaker_session):
        predictor = tuner.deploy(1, 'ml.c4.xlarge')

        features = [6.4, 3.2, 4.5, 1.5]
        dict_result = predictor.predict({'inputs': features})
        print('predict result: {}'.format(dict_result))
        list_result = predictor.predict(features)
        print('predict result: {}'.format(list_result))

        assert dict_result == list_result


@pytest.mark.continuous_testing
def test_tuning_chainer(sagemaker_session):
    with timeout(minutes=15):
        script_path = os.path.join(DATA_DIR, 'chainer_mnist', 'mnist.py')
        data_path = os.path.join(DATA_DIR, 'chainer_mnist')

        estimator = Chainer(entry_point=script_path,
                            role='SageMakerRole',
                            train_instance_count=1,
                            train_instance_type='ml.c4.xlarge',
                            sagemaker_session=sagemaker_session,
                            hyperparameters={'epochs': 1})

        train_input = estimator.sagemaker_session.upload_data(path=os.path.join(data_path, 'train'),
                                                              key_prefix='integ-test-data/chainer_mnist/train')
        test_input = estimator.sagemaker_session.upload_data(path=os.path.join(data_path, 'test'),
                                                             key_prefix='integ-test-data/chainer_mnist/test')

        hyperparameter_ranges = {'alpha': ContinuousParameter(0.001, 0.005)}

        objective_metric_name = 'Validation-accuracy'
        metric_definitions = [
            {'Name': 'Validation-accuracy', 'Regex': '\[J1\s+\d\.\d+\s+\d\.\d+\s+\d\.\d+\s+(\d\.\d+)'}]

        tuner = HyperparameterTuner(estimator, objective_metric_name, hyperparameter_ranges, metric_definitions,
                                    max_jobs=2, max_parallel_jobs=2)

        tuner.fit({'train': train_input, 'test': test_input})

        print('Started hyperparameter tuning job with name:' + tuner.latest_tuning_job.name)

        time.sleep(15)
        tuner.wait()

    best_training_job = tuner.best_training_job()
    with timeout_and_delete_endpoint_by_name(best_training_job, sagemaker_session):
        predictor = tuner.deploy(1, 'ml.c4.xlarge')

        batch_size = 100
        data = np.zeros((batch_size, 784), dtype='float32')
        output = predictor.predict(data)
        assert len(output) == batch_size

        data = np.zeros((batch_size, 1, 28, 28), dtype='float32')
        output = predictor.predict(data)
        assert len(output) == batch_size

        data = np.zeros((batch_size, 28, 28), dtype='float32')
        output = predictor.predict(data)
        assert len(output) == batch_size


@pytest.mark.continuous_testing
def test_attach_tuning_pytorch(sagemaker_session):
    mnist_dir = os.path.join(DATA_DIR, 'pytorch_mnist')
    mnist_script = os.path.join(mnist_dir, 'mnist.py')

    estimator = PyTorch(entry_point=mnist_script, role='SageMakerRole', train_instance_count=1,
                        train_instance_type='ml.c4.xlarge', sagemaker_session=sagemaker_session)

    with timeout(minutes=15):
        objective_metric_name = 'evaluation-accuracy'
        metric_definitions = [{'Name': 'evaluation-accuracy', 'Regex': 'Overall test accuracy: (\d+)'}]
        hyperparameter_ranges = {'batch-size': IntegerParameter(50, 100)}

        tuner = HyperparameterTuner(estimator, objective_metric_name, hyperparameter_ranges, metric_definitions,
                                    max_jobs=2, max_parallel_jobs=2)

        training_data = estimator.sagemaker_session.upload_data(path=os.path.join(mnist_dir, 'training'),
                                                                key_prefix='integ-test-data/pytorch_mnist/training')
        tuner.fit({'training': training_data})

        tuning_job_name = tuner.latest_tuning_job.name

        print('Started hyperparameter tuning job with name:' + tuning_job_name)

        time.sleep(15)
        tuner.wait()

    attached_tuner = HyperparameterTuner.attach(tuning_job_name)
    best_training_job = tuner.best_training_job()
    with timeout_and_delete_endpoint_by_name(best_training_job, sagemaker_session, minutes=20):
        predictor = attached_tuner.deploy(1, 'ml.c4.xlarge')
        data = np.zeros(shape=(1, 1, 28, 28), dtype=np.float32)
        predictor.predict(data)

        batch_size = 100
        data = np.random.rand(batch_size, 1, 28, 28).astype(np.float32)
        output = predictor.predict(data)

        assert output.shape == (batch_size, 10)


@pytest.mark.continuous_testing
def test_tuning_byo_estimator(sagemaker_session):
    """Use Factorization Machines algorithm as an example here.

    First we need to prepare data for training. We take standard data set, convert it to the
    format that the algorithm can process and upload it to S3.
    Then we create the Estimator and set hyperparamets as required by the algorithm.
    Next, we can call fit() with path to the S3.
    Later the trained model is deployed and prediction is called against the endpoint.
    Default predictor is updated with json serializer and deserializer.
    """
    image_name = registry(sagemaker_session.boto_session.region_name) + '/factorization-machines:1'

    with timeout(minutes=15):
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

        hyperparameter_ranges = {'mini_batch_size': IntegerParameter(100, 200)}

        tuner = HyperparameterTuner(estimator=estimator, base_tuning_job_name='byo',
                                    objective_metric_name='test:binary_classification_accuracy',
                                    hyperparameter_ranges=hyperparameter_ranges,
                                    max_jobs=2, max_parallel_jobs=2)

        tuner.fit({'train': s3_train_data, 'test': s3_train_data}, include_cls_metadata=False)

        print('Started hyperparameter tuning job with name:' + tuner.latest_tuning_job.name)

        time.sleep(15)
        tuner.wait()

    best_training_job = tuner.best_training_job()
    with timeout_and_delete_endpoint_by_name(best_training_job, sagemaker_session):
        predictor = tuner.deploy(1, 'ml.m4.xlarge', endpoint_name=best_training_job)
        predictor.serializer = _fm_serializer
        predictor.content_type = 'application/json'
        predictor.deserializer = json_deserializer

        result = predictor.predict(train_set[0][:10])

        assert len(result['predictions']) == 10
        for prediction in result['predictions']:
            assert prediction['score'] is not None


# Serializer for the Factorization Machines predictor (for BYO example)
def _fm_serializer(data):
    js = {'instances': []}
    for row in data:
        js['instances'].append({'features': row.tolist()})
    return json.dumps(js)
