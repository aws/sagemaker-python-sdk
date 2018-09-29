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

import gzip
import json
import os
import pickle
import sys

import pytest

from sagemaker import KNN
from sagemaker.amazon.amazon_estimator import registry
from sagemaker.chainer import Chainer
from sagemaker.estimator import Estimator
from sagemaker.mxnet import MXNet
from sagemaker.predictor import json_deserializer
from sagemaker.pytorch import PyTorch
from sagemaker.tensorflow import TensorFlow
from tests.integ import DATA_DIR, TRAINING_DEFAULT_TIMEOUT_MINUTES, PYTHON_VERSION
from tests.integ.timeout import timeout_and_delete_endpoint_by_name, timeout
from tests.integ.vpc_utils import get_or_create_subnets_and_security_group

DATA_PATH = os.path.join(DATA_DIR, 'iris', 'data')


@pytest.mark.skipif(PYTHON_VERSION != 'py2', reason="TensorFlow image supports only python 2.")
def test_vpc_tf(sagemaker_session, tf_full_version):
    instance_type = 'ml.c4.xlarge'

    train_input = sagemaker_session.upload_data(path=DATA_PATH, key_prefix='integ-test-data/tf_iris')
    script_path = os.path.join(DATA_DIR, 'iris', 'iris-dnn-classifier.py')

    ec2_client = sagemaker_session.boto_session.client('ec2')
    subnet_ids, security_group_id = get_or_create_subnets_and_security_group(ec2_client,
                                                                             sagemaker_session.boto_session.region_name)

    estimator = TensorFlow(entry_point=script_path,
                           role='SageMakerRole',
                           framework_version=tf_full_version,
                           training_steps=1,
                           evaluation_steps=1,
                           hyperparameters={'input_tensor_name': 'inputs'},
                           train_instance_count=1,
                           train_instance_type=instance_type,
                           sagemaker_session=sagemaker_session,
                           base_job_name='test-vpc-tf',
                           subnets=subnet_ids,
                           security_group_ids=[security_group_id])

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        estimator.fit(train_input)
        print('training job succeeded: {}'.format(estimator.latest_training_job.name))

    job_desc = sagemaker_session.sagemaker_client.describe_training_job(
        TrainingJobName=estimator.latest_training_job.name)
    assert set(subnet_ids) == set(job_desc['VpcConfig']['Subnets'])
    assert [security_group_id] == job_desc['VpcConfig']['SecurityGroupIds']

    endpoint_name = estimator.latest_training_job.name
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        model = estimator.create_model()
        json_predictor = model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge',
                                      endpoint_name=endpoint_name)

        features = [6.4, 3.2, 4.5, 1.5]
        dict_result = json_predictor.predict({'inputs': features})
        print('predict result: {}'.format(dict_result))
        list_result = json_predictor.predict(features)
        print('predict result: {}'.format(list_result))

        assert dict_result == list_result

        model_desc = sagemaker_session.sagemaker_client.describe_model(ModelName=model.name)
        assert set(subnet_ids) == set(model_desc['VpcConfig']['Subnets'])
        assert [security_group_id] == model_desc['VpcConfig']['SecurityGroupIds']


def test_vpc_mxnet(sagemaker_session, mxnet_full_version):
    script_path = os.path.join(DATA_DIR, 'mxnet_mnist', 'mnist.py')
    data_path = os.path.join(DATA_DIR, 'mxnet_mnist')

    train_input = sagemaker_session.upload_data(path=os.path.join(data_path, 'train'),
                                                key_prefix='integ-test-data/mxnet_mnist/train')
    test_input = sagemaker_session.upload_data(path=os.path.join(data_path, 'test'),
                                               key_prefix='integ-test-data/mxnet_mnist/test')

    ec2_client = sagemaker_session.boto_session.client('ec2')
    subnet_ids, security_group_id = get_or_create_subnets_and_security_group(ec2_client,
                                                                             sagemaker_session.boto_session.region_name)

    estimator = MXNet(entry_point=script_path,
                      role='SageMakerRole',
                      framework_version=mxnet_full_version,
                      py_version=PYTHON_VERSION,
                      train_instance_count=1,
                      train_instance_type='ml.c4.xlarge',
                      sagemaker_session=sagemaker_session,
                      base_job_name='test-vpc-mxnet',
                      subnets=subnet_ids,
                      security_group_ids=[security_group_id])

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        estimator.fit({'train': train_input, 'test': test_input})
        print('training job succeeded: {}'.format(estimator.latest_training_job.name))

    job_desc = sagemaker_session.sagemaker_client.describe_training_job(
        TrainingJobName=estimator.latest_training_job.name)
    assert set(subnet_ids) == set(job_desc['VpcConfig']['Subnets'])
    assert [security_group_id] == job_desc['VpcConfig']['SecurityGroupIds']

    endpoint_name = estimator.latest_training_job.name
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        model = estimator.create_model()
        json_predictor = model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge',
                                      endpoint_name=endpoint_name)

        model_desc = sagemaker_session.sagemaker_client.describe_model(ModelName=model.name)
        assert set(subnet_ids) == set(model_desc['VpcConfig']['Subnets'])
        assert [security_group_id] == model_desc['VpcConfig']['SecurityGroupIds']

        features = [6.4, 3.2, 4.5, 1.5]
        dict_result = json_predictor.predict({'inputs': features})
        print('predict result: {}'.format(dict_result))
        list_result = json_predictor.predict(features)
        print('predict result: {}'.format(list_result))

        assert dict_result == list_result


def test_vpc_pytorch(sagemaker_session, pytorch_full_version):
    instance_type = 'ml.c4.xlarge'
    mnist_dir = os.path.join(DATA_DIR, 'pytorch_mnist')
    mnist_script = os.path.join(mnist_dir, 'mnist.py')

    train_input = sagemaker_session.upload_data(path=os.path.join(mnist_dir, 'training'),
                                                key_prefix='integ-test-data/pytorch_mnist/training')

    ec2_client = sagemaker_session.boto_session.client('ec2')
    subnet_ids, security_group_id = get_or_create_subnets_and_security_group(ec2_client,
                                                                             sagemaker_session.boto_session.region_name)

    estimator = PyTorch(entry_point=mnist_script,
                        role='SageMakerRole',
                        framework_version=pytorch_full_version,
                        py_version=PYTHON_VERSION,
                        train_instance_count=1,
                        train_instance_type=instance_type,
                        sagemaker_session=sagemaker_session,
                        base_job_name='test-vpc-pytorch',
                        subnets=subnet_ids,
                        security_group_ids=[security_group_id])

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        estimator.fit({'training': train_input})
        print('training job succeeded: {}'.format(estimator.latest_training_job.name))

    job_desc = sagemaker_session.sagemaker_client.describe_training_job(
        TrainingJobName=estimator.latest_training_job.name)
    assert set(subnet_ids) == set(job_desc['VpcConfig']['Subnets'])
    assert [security_group_id] == job_desc['VpcConfig']['SecurityGroupIds']

    endpoint_name = estimator.latest_training_job.name
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        model = estimator.create_model()
        json_predictor = model.deploy(initial_instance_count=1, instance_type=instance_type,
                                      endpoint_name=endpoint_name)

        model_desc = sagemaker_session.sagemaker_client.describe_model(ModelName=model.name)
        assert set(subnet_ids) == set(model_desc['VpcConfig']['Subnets'])
        assert [security_group_id] == model_desc['VpcConfig']['SecurityGroupIds']

        features = [6.4, 3.2, 4.5, 1.5]
        dict_result = json_predictor.predict({'inputs': features})
        print('predict result: {}'.format(dict_result))
        list_result = json_predictor.predict(features)
        print('predict result: {}'.format(list_result))

        assert dict_result == list_result


def test_vpc_chainer(sagemaker_session, chainer_full_version):
    instance_type = 'ml.c4.xlarge'

    script_path = os.path.join(DATA_DIR, 'chainer_mnist', 'mnist.py')
    data_path = os.path.join(DATA_DIR, 'chainer_mnist')

    train_input = sagemaker_session.upload_data(path=os.path.join(data_path, 'train'),
                                                key_prefix='integ-test-data/chainer_mnist/train')
    test_input = sagemaker_session.upload_data(path=os.path.join(data_path, 'test'),
                                               key_prefix='integ-test-data/chainer_mnist/test')

    ec2_client = sagemaker_session.boto_session.client('ec2')
    subnet_ids, security_group_id = get_or_create_subnets_and_security_group(ec2_client,
                                                                             sagemaker_session.boto_session.region_name)

    estimator = Chainer(entry_point=script_path,
                        role='SageMakerRole',
                        framework_version=chainer_full_version,
                        py_version=PYTHON_VERSION,
                        train_instance_count=1,
                        train_instance_type=instance_type,
                        sagemaker_session=sagemaker_session,
                        hyperparameters={'epochs': 1},
                        base_job_name='test-vpc-chainer',
                        subnets=subnet_ids,
                        security_group_ids=[security_group_id])

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        estimator.fit({'train': train_input, 'test': test_input})
        print('training job succeeded: {}'.format(estimator.latest_training_job.name))

    job_desc = sagemaker_session.sagemaker_client.describe_training_job(
        TrainingJobName=estimator.latest_training_job.name)
    assert set(subnet_ids) == set(job_desc['VpcConfig']['Subnets'])
    assert [security_group_id] == job_desc['VpcConfig']['SecurityGroupIds']

    endpoint_name = estimator.latest_training_job.name
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        model = estimator.create_model()
        json_predictor = model.deploy(initial_instance_count=1, instance_type=instance_type,
                                      endpoint_name=endpoint_name)

        model_desc = sagemaker_session.sagemaker_client.describe_model(ModelName=model.name)
        assert set(subnet_ids) == set(model_desc['VpcConfig']['Subnets'])
        assert [security_group_id] == model_desc['VpcConfig']['SecurityGroupIds']

        features = [6.4, 3.2, 4.5, 1.5]
        dict_result = json_predictor.predict({'inputs': features})
        print('predict result: {}'.format(dict_result))
        list_result = json_predictor.predict(features)
        print('predict result: {}'.format(list_result))

        assert dict_result == list_result


def test_vpc_knn_multi(sagemaker_session):
    instance_type = 'ml.c4.xlarge'
    train_instance_count = 2

    data_path = os.path.join(DATA_DIR, 'one_p_mnist', 'mnist.pkl.gz')
    pickle_args = {} if sys.version_info.major == 2 else {'encoding': 'latin1'}
    # Load the data into memory as numpy arrays
    with gzip.open(data_path, 'rb') as f:
        train_set, _, _ = pickle.load(f, **pickle_args)

    ec2_client = sagemaker_session.boto_session.client('ec2')
    subnet_ids, security_group_id = get_or_create_subnets_and_security_group(ec2_client,
                                                                             sagemaker_session.boto_session.region_name)

    estimator = KNN(role='SageMakerRole',
                    train_instance_count=train_instance_count,
                    train_instance_type=instance_type,
                    k=10,
                    predictor_type='regressor',
                    sample_size=500,
                    sagemaker_session=sagemaker_session,
                    base_job_name='test-vpc-knn',
                    subnets=subnet_ids,
                    security_group_ids=[security_group_id])

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        # training labels must be 'float32'
        estimator.fit(estimator.record_set(train_set[0][:200], train_set[1][:200].astype('float32')))
        print('training job succeeded: {}'.format(estimator.latest_training_job.name))

    job_desc = sagemaker_session.sagemaker_client.describe_training_job(
        TrainingJobName=estimator.latest_training_job.name)
    assert set(subnet_ids) == set(job_desc['VpcConfig']['Subnets'])
    assert [security_group_id] == job_desc['VpcConfig']['SecurityGroupIds']

    endpoint_name = estimator.latest_training_job.name
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        model = estimator.create_model()
        predictor = model.deploy(initial_instance_count=1, instance_type=instance_type, endpoint_name=endpoint_name)

        model_desc = sagemaker_session.sagemaker_client.describe_model(ModelName=model.name)
        assert set(subnet_ids) == set(model_desc['VpcConfig']['Subnets'])
        assert [security_group_id] == model_desc['VpcConfig']['SecurityGroupIds']

        result = predictor.predict(train_set[0][:10])
        print('predict result: {}'.format(result))

        assert len(result) == 10
        for record in result:
            assert record.label["score"] is not None


def test_vpc_byo(sagemaker_session):
    instance_type = 'ml.c4.xlarge'
    image_name = registry(sagemaker_session.boto_session.region_name) + "/factorization-machines:1"

    data_path = os.path.join(DATA_DIR, 'one_p_mnist', 'mnist.pkl.gz')
    pickle_args = {} if sys.version_info.major == 2 else {'encoding': 'latin1'}
    with gzip.open(data_path, 'rb') as f:
        test_set, _, _ = pickle.load(f, **pickle_args)

    training_data_path = os.path.join(DATA_DIR, 'dummy_tensor')
    prefix = 'test_byo_estimator'
    key = 'recordio-pb-data'
    s3_train_data = sagemaker_session.upload_data(path=training_data_path,
                                                  key_prefix=os.path.join(prefix, 'train', key))

    ec2_client = sagemaker_session.boto_session.client('ec2')
    subnet_ids, security_group_id = get_or_create_subnets_and_security_group(ec2_client,
                                                                             sagemaker_session.boto_session.region_name)

    estimator = Estimator(image_name=image_name,
                          role='SageMakerRole',
                          train_instance_count=1,
                          train_instance_type=instance_type,
                          sagemaker_session=sagemaker_session,
                          base_job_name='test-vpc-byo-fm',
                          subnets=subnet_ids,
                          security_group_ids=[security_group_id])
    estimator.set_hyperparameters(num_factors=10,
                                  feature_dim=784,
                                  mini_batch_size=100,
                                  predictor_type='binary_classifier')

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        estimator.fit({'train': s3_train_data})
        print('training job succeeded: {}'.format(estimator.latest_training_job.name))

    job_desc = sagemaker_session.sagemaker_client.describe_training_job(
        TrainingJobName=estimator.latest_training_job.name)
    assert set(subnet_ids) == set(job_desc['VpcConfig']['Subnets'])
    assert [security_group_id] == job_desc['VpcConfig']['SecurityGroupIds']

    endpoint_name = estimator.latest_training_job.name
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        model = estimator.create_model()
        predictor = model.deploy(1, 'ml.m4.xlarge', endpoint_name=endpoint_name)
        predictor.serializer = fm_serializer
        predictor.content_type = 'application/json'
        predictor.deserializer = json_deserializer

        model_desc = sagemaker_session.sagemaker_client.describe_model(ModelName=model.name)
        assert set(subnet_ids) == set(model_desc['VpcConfig']['Subnets'])
        assert [security_group_id] == model_desc['VpcConfig']['SecurityGroupIds']

        result = predictor.predict(test_set[0][:10])

        assert len(result['predictions']) == 10
        for prediction in result['predictions']:
            assert prediction['score'] is not None


def fm_serializer(data):
    js = {'instances': []}
    for row in data:
        js['instances'].append({'features': row.tolist()})
    return json.dumps(js)
