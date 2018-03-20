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
from mock import Mock, patch

from sagemaker.amazon.kmeans import KMeans, KMeansPredictor
from sagemaker.amazon.amazon_estimator import registry, RecordSet

ROLE = 'myrole'
TRAIN_INSTANCE_COUNT = 1
TRAIN_INSTANCE_TYPE = 'ml.c4.xlarge'
K = 2

COMMON_TRAIN_ARGS = {'role': ROLE, 'train_instance_count': TRAIN_INSTANCE_COUNT,
                     'train_instance_type': TRAIN_INSTANCE_TYPE}
ALL_REQ_ARGS = dict({'k': K}, **COMMON_TRAIN_ARGS)

REGION = "us-west-2"
BUCKET_NAME = "Some-Bucket"

DESCRIBE_TRAINING_JOB_RESULT = {
    'ModelArtifacts': {
        'S3ModelArtifacts': "s3://bucket/model.tar.gz"
    }
}


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name='boto_session', region_name=REGION)
    sms = Mock(name='sagemaker_session', boto_session=boto_mock)
    sms.boto_region_name = REGION
    sms.default_bucket = Mock(name='default_bucket', return_value=BUCKET_NAME)
    sms.sagemaker_client.describe_training_job = Mock(name='describe_training_job',
                                                      return_value=DESCRIBE_TRAINING_JOB_RESULT)

    return sms


def test_init_required_positional(sagemaker_session):
    kmeans = KMeans(ROLE, TRAIN_INSTANCE_COUNT, TRAIN_INSTANCE_TYPE, K, sagemaker_session=sagemaker_session)
    assert kmeans.role == ROLE
    assert kmeans.train_instance_count == TRAIN_INSTANCE_COUNT
    assert kmeans.train_instance_type == TRAIN_INSTANCE_TYPE
    assert kmeans.k == K


def test_init_required_named(sagemaker_session):
    kmeans = KMeans(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    assert kmeans.role == COMMON_TRAIN_ARGS['role']
    assert kmeans.train_instance_count == TRAIN_INSTANCE_COUNT
    assert kmeans.train_instance_type == COMMON_TRAIN_ARGS['train_instance_type']
    assert kmeans.k == ALL_REQ_ARGS['k']


def test_all_hyperparameters(sagemaker_session):
    kmeans = KMeans(sagemaker_session=sagemaker_session, init_method='random', max_iterations=3, tol=0.5,
                    num_trials=5, local_init_method='kmeans++', half_life_time_size=0, epochs=10, center_factor=2,
                    eval_metrics=['msd', 'ssd'], **ALL_REQ_ARGS)
    assert kmeans.hyperparameters() == dict(
        k=str(ALL_REQ_ARGS['k']),
        init_method='random',
        local_lloyd_max_iterations='3',
        local_lloyd_tol='0.5',
        local_lloyd_num_trials='5',
        local_lloyd_init_method='kmeans++',
        half_life_time_size='0',
        epochs='10',
        extra_center_factor='2',
        eval_metrics='[\'msd\', \'ssd\']',
        force_dense='True'
    )


def test_image(sagemaker_session):
    kmeans = KMeans(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    assert kmeans.train_image() == registry(REGION, "kmeans") + '/kmeans:1'


def test_k_validation_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        KMeans(k='invalid', sagemaker_session=sagemaker_session, **COMMON_TRAIN_ARGS)


def test_k_validation_fail_value(sagemaker_session):
    with pytest.raises(ValueError):
        KMeans(k=0, sagemaker_session=sagemaker_session, **COMMON_TRAIN_ARGS)


def test_init_method_validation_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        KMeans(init_method=0, sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_init_method_validation_fail_value(sagemaker_session):
    with pytest.raises(ValueError):
        KMeans(init_method='invalid', sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_max_iterations_validation_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        KMeans(max_iterations='invalid', sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_max_iterations_validation_fail_value(sagemaker_session):
    with pytest.raises(ValueError):
        KMeans(max_iterations=0, sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_tol_validation_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        KMeans(tol='invalid', sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_tol_validation_fail_value_lower(sagemaker_session):
    with pytest.raises(ValueError):
        KMeans(tol=-0.1, sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_tol_validation_fail_value_upper(sagemaker_session):
    with pytest.raises(ValueError):
        KMeans(tol=1.1, sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_num_trials_validation_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        KMeans(num_trials='invalid', sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_num_trials_validation_fail_value(sagemaker_session):
    with pytest.raises(ValueError):
        KMeans(num_trials=0, sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_local_init_method_validation_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        KMeans(local_init_method=0, sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_local_init_method_validation_fail_value(sagemaker_session):
    with pytest.raises(ValueError):
        KMeans(local_init_method='invalid', sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_half_life_time_size_validation_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        KMeans(half_life_time_size='invalid', sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_half_life_time_size_validation_fail_value(sagemaker_session):
    with pytest.raises(ValueError):
        KMeans(half_life_time_size=-1, sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_epochs_validation_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        KMeans(epochs='invalid', sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_epochs_validation_fail_value(sagemaker_session):
    with pytest.raises(ValueError):
        KMeans(epochs=0, sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_center_factor_validation_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        KMeans(center_factor='invalid', sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_center_factor_validation_fail_value(sagemaker_session):
    with pytest.raises(ValueError):
        KMeans(center_factor=0, sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_eval_metrics_validation_fail_type(sagemaker_session):
    with pytest.raises(TypeError):
        KMeans(eval_metrics=0, sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


PREFIX = "prefix"
FEATURE_DIM = 10
MINI_BATCH_SIZE = 200


@patch("sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit")
def test_call_fit(base_fit, sagemaker_session):
    kmeans = KMeans(base_job_name="kmeans", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet("s3://{}/{}".format(BUCKET_NAME, PREFIX), num_records=1, feature_dim=FEATURE_DIM, channel='train')

    kmeans.fit(data, MINI_BATCH_SIZE)

    base_fit.assert_called_once()
    assert len(base_fit.call_args[0]) == 2
    assert base_fit.call_args[0][0] == data
    assert base_fit.call_args[0][1] == MINI_BATCH_SIZE


def test_call_fit_none_mini_batch_size(sagemaker_session):
    kmeans = KMeans(base_job_name="kmeans", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet("s3://{}/{}".format(BUCKET_NAME, PREFIX), num_records=1, feature_dim=FEATURE_DIM,
                     channel='train')
    kmeans.fit(data)


def test_call_fit_wrong_type_mini_batch_size(sagemaker_session):
    kmeans = KMeans(base_job_name="kmeans", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet("s3://{}/{}".format(BUCKET_NAME, PREFIX), num_records=1, feature_dim=FEATURE_DIM,
                     channel='train')

    with pytest.raises((TypeError, ValueError)):
        kmeans.fit(data, "some")


def test_call_fit_wrong_value_mini_batch_size(sagemaker_session):
    kmeans = KMeans(base_job_name="kmeans", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet("s3://{}/{}".format(BUCKET_NAME, PREFIX), num_records=1, feature_dim=FEATURE_DIM,
                     channel='train')
    with pytest.raises(ValueError):
        kmeans.fit(data, 0)


def test_model_image(sagemaker_session):
    kmeans = KMeans(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet("s3://{}/{}".format(BUCKET_NAME, PREFIX), num_records=1, feature_dim=FEATURE_DIM, channel='train')
    kmeans.fit(data, MINI_BATCH_SIZE)

    model = kmeans.create_model()
    assert model.image == registry(REGION, "kmeans") + '/kmeans:1'


def test_predictor_type(sagemaker_session):
    kmeans = KMeans(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet("s3://{}/{}".format(BUCKET_NAME, PREFIX), num_records=1, feature_dim=FEATURE_DIM, channel='train')
    kmeans.fit(data, MINI_BATCH_SIZE)
    model = kmeans.create_model()
    predictor = model.deploy(1, TRAIN_INSTANCE_TYPE)

    assert isinstance(predictor, KMeansPredictor)
