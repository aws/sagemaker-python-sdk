# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from sagemaker import KMeans
from sagemaker.mxnet import MXNet
from sagemaker.transformer import Transformer
from sagemaker.utils import unique_name_from_base
from tests.integ import DATA_DIR, TRAINING_DEFAULT_TIMEOUT_MINUTES, TRANSFORM_DEFAULT_TIMEOUT_MINUTES
from tests.integ.kms_utils import get_or_create_kms_key
from tests.integ.timeout import timeout, timeout_and_delete_model_with_transformer
from tests.integ.vpc_test_utils import get_or_create_vpc_resources


@pytest.mark.canary_quick
def test_transform_mxnet(sagemaker_session, mxnet_full_version):
    data_path = os.path.join(DATA_DIR, 'mxnet_mnist')
    script_path = os.path.join(data_path, 'mnist.py')

    mx = MXNet(entry_point=script_path, role='SageMakerRole', train_instance_count=1,
               train_instance_type='ml.c4.xlarge', sagemaker_session=sagemaker_session,
               framework_version=mxnet_full_version)

    train_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'train'),
                                                   key_prefix='integ-test-data/mxnet_mnist/train')
    test_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'test'),
                                                  key_prefix='integ-test-data/mxnet_mnist/test')
    job_name = unique_name_from_base('test-mxnet-transform')

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        mx.fit({'train': train_input, 'test': test_input}, job_name=job_name)

    transform_input_path = os.path.join(data_path, 'transform', 'data.csv')
    transform_input_key_prefix = 'integ-test-data/mxnet_mnist/transform'
    transform_input = mx.sagemaker_session.upload_data(path=transform_input_path,
                                                       key_prefix=transform_input_key_prefix)

    kms_key_arn = get_or_create_kms_key(sagemaker_session)

    transformer = _create_transformer_and_transform_job(mx, transform_input, kms_key_arn,
            input_filter=None, output_filter="$", join_source=None)
    with timeout_and_delete_model_with_transformer(transformer, sagemaker_session,
                                                   minutes=TRANSFORM_DEFAULT_TIMEOUT_MINUTES):
        transformer.wait()

    job_desc = transformer.sagemaker_session.sagemaker_client.describe_transform_job(
        TransformJobName=transformer.latest_transform_job.name)
    assert kms_key_arn == job_desc['TransformResources']['VolumeKmsKeyId']


@pytest.mark.canary_quick
def test_attach_transform_kmeans(sagemaker_session):
    data_path = os.path.join(DATA_DIR, 'one_p_mnist')
    pickle_args = {} if sys.version_info.major == 2 else {'encoding': 'latin1'}

    # Load the data into memory as numpy arrays
    train_set_path = os.path.join(data_path, 'mnist.pkl.gz')
    with gzip.open(train_set_path, 'rb') as f:
        train_set, _, _ = pickle.load(f, **pickle_args)

    kmeans = KMeans(role='SageMakerRole', train_instance_count=1,
                    train_instance_type='ml.c4.xlarge', k=10, sagemaker_session=sagemaker_session,
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

    job_name = unique_name_from_base('test-kmeans-attach')

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        kmeans.fit(records, job_name=job_name)

    transform_input_path = os.path.join(data_path, 'transform_input.csv')
    transform_input_key_prefix = 'integ-test-data/one_p_mnist/transform'
    transform_input = kmeans.sagemaker_session.upload_data(path=transform_input_path,
                                                           key_prefix=transform_input_key_prefix)

    transformer = _create_transformer_and_transform_job(kmeans, transform_input)

    attached_transformer = Transformer.attach(transformer.latest_transform_job.name,
                                              sagemaker_session=sagemaker_session)
    with timeout_and_delete_model_with_transformer(transformer, sagemaker_session,
                                                   minutes=TRANSFORM_DEFAULT_TIMEOUT_MINUTES):
        attached_transformer.wait()


def test_transform_mxnet_vpc(sagemaker_session, mxnet_full_version):
    data_path = os.path.join(DATA_DIR, 'mxnet_mnist')
    script_path = os.path.join(data_path, 'mnist.py')

    ec2_client = sagemaker_session.boto_session.client('ec2')
    subnet_ids, security_group_id = get_or_create_vpc_resources(ec2_client,
                                                                sagemaker_session.boto_session.region_name)

    mx = MXNet(entry_point=script_path, role='SageMakerRole', train_instance_count=1,
               train_instance_type='ml.c4.xlarge', sagemaker_session=sagemaker_session,
               framework_version=mxnet_full_version, subnets=subnet_ids,
               security_group_ids=[security_group_id])

    train_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'train'),
                                                   key_prefix='integ-test-data/mxnet_mnist/train')
    test_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'test'),
                                                  key_prefix='integ-test-data/mxnet_mnist/test')
    job_name = unique_name_from_base('test-mxnet-vpc')

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        mx.fit({'train': train_input, 'test': test_input}, job_name=job_name)

    job_desc = sagemaker_session.sagemaker_client.describe_training_job(TrainingJobName=mx.latest_training_job.name)
    assert set(subnet_ids) == set(job_desc['VpcConfig']['Subnets'])
    assert [security_group_id] == job_desc['VpcConfig']['SecurityGroupIds']

    transform_input_path = os.path.join(data_path, 'transform', 'data.csv')
    transform_input_key_prefix = 'integ-test-data/mxnet_mnist/transform'
    transform_input = mx.sagemaker_session.upload_data(path=transform_input_path,
                                                       key_prefix=transform_input_key_prefix)

    transformer = _create_transformer_and_transform_job(mx, transform_input)
    with timeout_and_delete_model_with_transformer(transformer, sagemaker_session,
                                                   minutes=TRANSFORM_DEFAULT_TIMEOUT_MINUTES):
        transformer.wait()
        model_desc = sagemaker_session.sagemaker_client.describe_model(ModelName=transformer.model_name)
        assert set(subnet_ids) == set(model_desc['VpcConfig']['Subnets'])
        assert [security_group_id] == model_desc['VpcConfig']['SecurityGroupIds']


def _create_transformer_and_transform_job(estimator, transform_input, volume_kms_key=None,
        input_filter=None, output_filter=None, join_source=None):
    transformer = estimator.transformer(1, 'ml.m4.xlarge', volume_kms_key=volume_kms_key)
    transformer.transform(transform_input, content_type='text/csv',
        input_filter=input_filter, output_filter=output_filter, join_source=join_source)
    return transformer
