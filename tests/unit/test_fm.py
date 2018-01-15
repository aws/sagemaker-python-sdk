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
import pytest
from mock import Mock

from sagemaker.amazon.factorization_machines import FactorizationMachines
from sagemaker.amazon.amazon_estimator import registry


COMMON_TRAIN_ARGS = {'role': 'myrole', 'train_instance_count': 1, 'train_instance_type': 'ml.c4.xlarge'}
ALL_REQ_ARGS = dict({'num_factors': 3, 'predictor_type': 'regressor'}, **COMMON_TRAIN_ARGS)

REGION = "us-west-2"
BUCKET_NAME = "Some-Bucket"


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name='boto_session', region_name=REGION)
    sms = Mock(name='sagemaker_session', boto_session=boto_mock)
    sms.boto_region_name = REGION
    sms.default_bucket = Mock(name='default_bucket', return_value=BUCKET_NAME)
    return sms


def test_init_required_positional(sagemaker_session):
    fm = FactorizationMachines('myrole', 1, 'ml.c4.xlarge', 3, 'regressor',
                               sagemaker_session=sagemaker_session)
    assert fm.role == 'myrole'
    assert fm.train_instance_count == 1
    assert fm.train_instance_type == 'ml.c4.xlarge'
    assert fm.num_factors == 3
    assert fm.predictor_type == 'regressor'


def test_init_required_named(sagemaker_session):
    fm = FactorizationMachines(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    assert fm.role == COMMON_TRAIN_ARGS['role']
    assert fm.train_instance_count == COMMON_TRAIN_ARGS['train_instance_count']
    assert fm.train_instance_type == COMMON_TRAIN_ARGS['train_instance_type']
    assert fm.num_factors == ALL_REQ_ARGS['num_factors']
    assert fm.predictor_type == ALL_REQ_ARGS['predictor_type']


def test_all_hyperparameters(sagemaker_session):
    fm = FactorizationMachines(sagemaker_session=sagemaker_session,
                               epochs=2, clip_gradient=1e2, eps=0.001, rescale_grad=2.2,
                               bias_lr=0.01, linear_lr=0.002, factors_lr=0.0003,
                               bias_wd=0.0004, linear_wd=1.01, factors_wd=1.002,
                               bias_init_method='uniform', bias_init_scale=0.1, bias_init_sigma=0.05,
                               bias_init_value=2.002, linear_init_method='constant', linear_init_scale=0.02,
                               linear_init_sigma=0.003, linear_init_value=1.0, factors_init_method='normal',
                               factors_init_scale=1.101, factors_init_sigma=1.202, factors_init_value=1.303,
                               **ALL_REQ_ARGS)
    assert fm.hyperparameters() == dict(
        num_factors=str(ALL_REQ_ARGS['num_factors']),
        predictor_type=ALL_REQ_ARGS['predictor_type'],
        epochs='2',
        clip_gradient='100.0',
        eps='0.001',
        rescale_grad='2.2',
        bias_lr='0.01',
        linear_lr='0.002',
        factors_lr='0.0003',
        bias_wd='0.0004',
        linear_wd='1.01',
        factors_wd='1.002',
        bias_init_method='uniform',
        bias_init_scale='0.1',
        bias_init_sigma='0.05',
        bias_init_value='2.002',
        linear_init_method='constant',
        linear_init_scale='0.02',
        linear_init_sigma='0.003',
        linear_init_value='1.0',
        factors_init_method='normal',
        factors_init_scale='1.101',
        factors_init_sigma='1.202',
        factors_init_value='1.303',
    )


def test_image(sagemaker_session):
    fm = FactorizationMachines(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    assert fm.train_image() == registry(REGION) + '/factorization-machines:1'
