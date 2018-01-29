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

from sagemaker.amazon.lda import LDA, LDAPredictor
from sagemaker.amazon.amazon_estimator import registry, RecordSet

ROLE = 'myrole'
TRAIN_INSTANCE_COUNT = 1
TRAIN_INSTANCE_TYPE = 'ml.c4.xlarge'
NUM_TOPICS = 3

COMMON_TRAIN_ARGS = {'role': ROLE, 'train_instance_type': TRAIN_INSTANCE_TYPE}
ALL_REQ_ARGS = dict({'num_topics': NUM_TOPICS}, **COMMON_TRAIN_ARGS)

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
    lda = LDA(ROLE, TRAIN_INSTANCE_TYPE, NUM_TOPICS, sagemaker_session=sagemaker_session)
    assert lda.role == ROLE
    assert lda.train_instance_count == TRAIN_INSTANCE_COUNT
    assert lda.train_instance_type == TRAIN_INSTANCE_TYPE
    assert lda.num_topics == NUM_TOPICS


def test_init_required_named(sagemaker_session):
    lda = LDA(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    assert lda.role == COMMON_TRAIN_ARGS['role']
    assert lda.train_instance_count == TRAIN_INSTANCE_COUNT
    assert lda.train_instance_type == COMMON_TRAIN_ARGS['train_instance_type']
    assert lda.num_topics == ALL_REQ_ARGS['num_topics']


def test_all_hyperparameters(sagemaker_session):
    lda = LDA(sagemaker_session=sagemaker_session,
              alpha0=2.2, max_restarts=3, max_iterations=10, tol=3.3,
              **ALL_REQ_ARGS)
    assert lda.hyperparameters() == dict(
        num_topics=str(ALL_REQ_ARGS['num_topics']),
        alpha0='2.2',
        max_restarts='3',
        max_iterations='10',
        tol='3.3',
    )


def test_image(sagemaker_session):
    lda = LDA(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    assert lda.train_image() == registry(REGION, "lda") + '/lda:1'


def test_num_topics_validation_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LDA(num_topics='other', sagemaker_session=sagemaker_session, **COMMON_TRAIN_ARGS)


def test_num_topics_validation_fail_value(sagemaker_session):
    with pytest.raises(ValueError):
        LDA(num_topics=0, sagemaker_session=sagemaker_session, **COMMON_TRAIN_ARGS)


def test_alpha0_validation_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LDA(alpha0='other', sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_max_restarts_validation_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LDA(max_restarts='other', sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_max_restarts_validation_fail_type2(sagemaker_session):
    with pytest.raises(ValueError):
        LDA(max_restarts=0.1, sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_max_restarts_validation_fail_value(sagemaker_session):
    with pytest.raises(ValueError):
        LDA(max_restarts=0, sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_max_iterations_validation_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LDA(max_iterations='other', sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_max_iterations_validation_fail_value(sagemaker_session):
    with pytest.raises(ValueError):
        LDA(max_iterations=0, sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_tol_validation_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LDA(tol='other', sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


def test_tol_validation_fail_value(sagemaker_session):
    with pytest.raises(ValueError):
        LDA(tol=0, sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)


PREFIX = "prefix"
BASE_TRAIN_CALL = {
    'hyperparameters': {},
    'image': registry(REGION, "lda") + '/lda:1',
    'input_config': [{
        'DataSource': {
            'S3DataSource': {
                'S3DataDistributionType': 'ShardedByS3Key',
                'S3DataType': 'ManifestFile',
                'S3Uri': 's3://{}/{}'.format(BUCKET_NAME, PREFIX)
            }
        },
        'ChannelName': 'train'
    }],
    'input_mode': 'File',
    'output_config': {'S3OutputPath': 's3://{}/'.format(BUCKET_NAME)},
    'resource_config': {
        'InstanceCount': TRAIN_INSTANCE_COUNT,
        'InstanceType': TRAIN_INSTANCE_TYPE,
        'VolumeSizeInGB': 30
    },
    'stop_condition': {'MaxRuntimeInSeconds': 86400}
}

FEATURE_DIM = 10
MINI_BATCH_SZIE = 200
HYPERPARAMS = {'num_topics': NUM_TOPICS, 'feature_dim': FEATURE_DIM, 'mini_batch_size': MINI_BATCH_SZIE}
STRINGIFIED_HYPERPARAMS = dict([(x, str(y)) for x, y in HYPERPARAMS.items()])
HP_TRAIN_CALL = dict(BASE_TRAIN_CALL)
HP_TRAIN_CALL.update({'hyperparameters': STRINGIFIED_HYPERPARAMS})


@patch("sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit")
def test_call_fit(base_fit, sagemaker_session):
    lda = LDA(base_job_name="lda", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet("s3://{}/{}".format(BUCKET_NAME, PREFIX), num_records=1, feature_dim=FEATURE_DIM, channel='train')

    lda.fit(data, MINI_BATCH_SZIE)

    base_fit.assert_called_once()
    assert len(base_fit.call_args[0]) == 2
    assert base_fit.call_args[0][0] == data
    assert base_fit.call_args[0][1] == MINI_BATCH_SZIE


def test_call_fit_none_mini_batch_size(sagemaker_session):
    lda = LDA(base_job_name="lda", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet("s3://{}/{}".format(BUCKET_NAME, PREFIX), num_records=1, feature_dim=FEATURE_DIM,
                     channel='train')
    with pytest.raises(ValueError):
        lda.fit(data, None)


def test_call_fit_wrong_type_mini_batch_size(sagemaker_session):
    lda = LDA(base_job_name="lda", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet("s3://{}/{}".format(BUCKET_NAME, PREFIX), num_records=1, feature_dim=FEATURE_DIM,
                     channel='train')

    with pytest.raises(ValueError):
        lda.fit(data, "some")


def test_call_fit_wrong_value_mini_batch_size(sagemaker_session):
    lda = LDA(base_job_name="lda", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet("s3://{}/{}".format(BUCKET_NAME, PREFIX), num_records=1, feature_dim=FEATURE_DIM,
                     channel='train')
    with pytest.raises(ValueError):
        lda.fit(data, 0)


def test_model_image(sagemaker_session):
    lda = LDA(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet("s3://{}/{}".format(BUCKET_NAME, PREFIX), num_records=1, feature_dim=FEATURE_DIM, channel='train')
    lda.fit(data, MINI_BATCH_SZIE)

    model = lda.create_model()
    assert model.image == registry(REGION, "lda") + '/lda:1'


def test_predictor_type(sagemaker_session):
    lda = LDA(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet("s3://{}/{}".format(BUCKET_NAME, PREFIX), num_records=1, feature_dim=FEATURE_DIM, channel='train')
    lda.fit(data, MINI_BATCH_SZIE)
    model = lda.create_model()
    predictor = model.deploy(1, TRAIN_INSTANCE_TYPE)

    assert isinstance(predictor, LDAPredictor)
