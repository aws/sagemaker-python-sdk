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

from sagemaker.amazon.linear_learner import LinearLearner, LinearLearnerPredictor
from sagemaker.amazon.amazon_estimator import registry, RecordSet

ROLE = 'myrole'
TRAIN_INSTANCE_COUNT = 1
TRAIN_INSTANCE_TYPE = 'ml.c4.xlarge'

DEFAULT_PREDICTOR_TYPE = 'binary_classifier'

REQ_ARGS = {'role': ROLE, 'train_instance_count': TRAIN_INSTANCE_COUNT, 'train_instance_type': TRAIN_INSTANCE_TYPE}

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
    lr = LinearLearner(ROLE, TRAIN_INSTANCE_COUNT, TRAIN_INSTANCE_TYPE, sagemaker_session=sagemaker_session)
    assert lr.role == ROLE
    assert lr.train_instance_count == TRAIN_INSTANCE_COUNT
    assert lr.train_instance_type == TRAIN_INSTANCE_TYPE
    assert lr.predictor_type == DEFAULT_PREDICTOR_TYPE


def test_init_required_named(sagemaker_session):
    lr = LinearLearner(sagemaker_session=sagemaker_session, **REQ_ARGS)

    assert lr.role == REQ_ARGS['role']
    assert lr.train_instance_count == REQ_ARGS['train_instance_count']
    assert lr.train_instance_type == REQ_ARGS['train_instance_type']
    assert lr.predictor_type == DEFAULT_PREDICTOR_TYPE


def test_all_hyperparameters(sagemaker_session):
    lr = LinearLearner(sagemaker_session=sagemaker_session,
                       predictor_type='regressor', binary_classifier_model_selection_criteria='accuracy',
                       target_recall=0.5, target_precision=0.6,
                       positive_example_weight_mult=0.1, epochs=1, use_bias=True, num_models=5,
                       num_calibration_samples=6, init_method='uniform', init_scale=-0.1, init_sigma=0.001,
                       init_bias=0, optimizer='sgd', loss='logistic', wd=0.4, l1=0.04, momentum=0.1,
                       learning_rate=0.001, beta_1=0.2, beta_2=0.03, bias_lr_mult=5.5, bias_wd_mult=6.6,
                       use_lr_scheduler=False, lr_scheduler_step=2, lr_scheduler_factor=0.03,
                       lr_scheduler_minimum_lr=0.001, normalize_data=False, normalize_label=True,
                       unbias_data=True, unbias_label=False, num_point_for_scaler=3,
                       **REQ_ARGS)

    assert lr.hyperparameters() == dict(
        predictor_type='regressor', binary_classifier_model_selection_criteria='accuracy',
        target_recall='0.5', target_precision='0.6', positive_example_weight_mult='0.1', epochs='1',
        use_bias='True', num_models='5', num_calibration_samples='6', init_method='uniform',
        init_scale='-0.1', init_sigma='0.001', init_bias='0.0', optimizer='sgd', loss='logistic',
        wd='0.4', l1='0.04', momentum='0.1', learning_rate='0.001', beta_1='0.2', beta_2='0.03',
        bias_lr_mult='5.5', bias_wd_mult='6.6', use_lr_scheduler='False', lr_scheduler_step='2',
        lr_scheduler_factor='0.03', lr_scheduler_minimum_lr='0.001', normalize_data='False',
        normalize_label='True', unbias_data='True', unbias_label='False', num_point_for_scaler='3',
    )


def test_image(sagemaker_session):
    lr = LinearLearner(sagemaker_session=sagemaker_session, **REQ_ARGS)
    assert lr.train_image() == registry(REGION, "linear-learner") + '/linear-learner:1'


def test_predictor_type_fail(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(predictor_type='other', sagemaker_session=sagemaker_session, **REQ_ARGS)


def test_binary_classifier_model_selection_criteria_fail(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(binary_classifier_model_selection_criteria='other',
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_target_recall_fail_value_low(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(target_recall=0,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_target_recall_fail_value_high(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(target_recall=1,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_target_recall_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(target_recall='blah',
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_target_precision_fail_value_low(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(target_precision=0,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_target_precision_fail_value_high(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(target_precision=1,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_target_precision_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(target_precision='blah',
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_positive_example_weight_mult_fail_value(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(positive_example_weight_mult=0,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_positive_example_weight_mult_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(positive_example_weight_mult='blah',
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_epochs_fail_value(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(epochs=0,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_epochs_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(epochs='blah',
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_num_models_fail_value(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(num_models=0,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_num_models_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(num_models='blah',
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_num_calibration_samples_fail_value(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(num_calibration_samples=0,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_num_calibration_samples_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(num_calibration_samples='blah',
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_init_method_fail(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(init_method='other', sagemaker_session=sagemaker_session, **REQ_ARGS)


def test_init_scale_fail_value_low(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(init_scale=1.01,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_init_scale_fail_value_high(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(init_scale=1,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_init_scale_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(init_scale='blah',
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_init_sigma_fail_value_low(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(init_sigma=0,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_init_sigma_fail_value_high(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(init_sigma=1,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_init_sigma_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(init_sigma='blah',
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_init_bias_fail(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(init_bias='other', sagemaker_session=sagemaker_session, **REQ_ARGS)


def test_optimizer_fail(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(optimizer='other', sagemaker_session=sagemaker_session, **REQ_ARGS)


def test_loss_fail(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(loss='other', sagemaker_session=sagemaker_session, **REQ_ARGS)


def test_wd_fail_value_low(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(wd=0,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_wd_fail_value_high(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(wd=1,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_wd_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(wd='blah',
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_l1_fail_value_low(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(l1=0,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_l1_fail_value_high(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(l1=1,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_l1_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(l1='blah',
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_momentum_fail_value_low(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(momentum=0,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_momentum_fail_value_high(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(momentum=1,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_momentum_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(momentum='blah',
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_learning_rate_fail_value_low(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(learning_rate=0,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_learning_rate_fail_value_high(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(learning_rate=1,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_learning_rate_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(learning_rate='blah',
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_beta_1_fail_value_low(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(beta_1=0,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_beta_1_fail_value_high(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(beta_1=1,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_beta_1_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(beta_1='blah',
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_beta_2_fail_value_low(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(beta_2=0,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_beta_2_fail_value_high(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(beta_2=1,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_beta_2_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(beta_2='blah',
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_bias_lr_mult_fail_value_low(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(bias_lr_mult=0,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_bias_lr_mult_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(bias_lr_mult='blah',
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_bias_wd_mult_fail_value_low(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(bias_wd_mult=0,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_bias_wd_mult_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(bias_wd_mult='blah',
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_lr_scheduler_step_fail_value(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(lr_scheduler_step=0,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_lr_scheduler_step_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(lr_scheduler_step='other',
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_lr_scheduler_factor_fail_value_low(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(lr_scheduler_factor=0,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_lr_scheduler_factor_fail_value_high(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(lr_scheduler_factor=1,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_lr_scheduler_factor_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(lr_scheduler_factor='blah',
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_lr_scheduler_minimum_lr_fail_value(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(lr_scheduler_minimum_lr=0,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_lr_scheduler_minimum_lr_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(lr_scheduler_minimum_lr='blah',
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_num_point_for_scaler_fail_value(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(num_point_for_scaler=0,
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


def test_num_point_for_scaler_fail_type(sagemaker_session):
    with pytest.raises(ValueError):
        LinearLearner(num_point_for_scaler='other',
                      sagemaker_session=sagemaker_session,
                      **REQ_ARGS)


PREFIX = "prefix"
FEATURE_DIM = 10
DEFAULT_MINI_BATCH_SIZE = 1000


@patch("sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit")
def test_call_fit_calculate_batch_size_1(base_fit, sagemaker_session):
    lr = LinearLearner(base_job_name="lr", sagemaker_session=sagemaker_session, **REQ_ARGS)

    data = RecordSet("s3://{}/{}".format(BUCKET_NAME, PREFIX), num_records=1, feature_dim=FEATURE_DIM, channel='train')

    lr.fit(data)

    base_fit.assert_called_once()
    assert len(base_fit.call_args[0]) == 2
    assert base_fit.call_args[0][0] == data
    assert base_fit.call_args[0][1] == 1


@patch("sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit")
def test_call_fit_calculate_batch_size_2(base_fit, sagemaker_session):
    lr = LinearLearner(base_job_name="lr", sagemaker_session=sagemaker_session, **REQ_ARGS)

    data = RecordSet("s3://{}/{}".format(BUCKET_NAME, PREFIX),
                     num_records=10000,
                     feature_dim=FEATURE_DIM,
                     channel='train')

    lr.fit(data)

    base_fit.assert_called_once()
    assert len(base_fit.call_args[0]) == 2
    assert base_fit.call_args[0][0] == data
    assert base_fit.call_args[0][1] == DEFAULT_MINI_BATCH_SIZE


@patch("sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit")
def test_call_fit_pass_batch_size(base_fit, sagemaker_session):
    lr = LinearLearner(base_job_name="lr", sagemaker_session=sagemaker_session, **REQ_ARGS)

    data = RecordSet("s3://{}/{}".format(BUCKET_NAME, PREFIX),
                     num_records=10000,
                     feature_dim=FEATURE_DIM,
                     channel='train')

    lr.fit(data, 10)

    base_fit.assert_called_once()
    assert len(base_fit.call_args[0]) == 2
    assert base_fit.call_args[0][0] == data
    assert base_fit.call_args[0][1] == 10


def test_model_image(sagemaker_session):
    lr = LinearLearner(sagemaker_session=sagemaker_session, **REQ_ARGS)
    data = RecordSet("s3://{}/{}".format(BUCKET_NAME, PREFIX), num_records=1, feature_dim=FEATURE_DIM, channel='train')
    lr.fit(data)

    model = lr.create_model()
    assert model.image == registry(REGION, 'linear-learner') + '/linear-learner:1'


def test_predictor_type(sagemaker_session):
    lr = LinearLearner(sagemaker_session=sagemaker_session, **REQ_ARGS)
    data = RecordSet("s3://{}/{}".format(BUCKET_NAME, PREFIX), num_records=1, feature_dim=FEATURE_DIM, channel='train')
    lr.fit(data)
    model = lr.create_model()
    predictor = model.deploy(1, TRAIN_INSTANCE_TYPE)

    assert isinstance(predictor, LinearLearnerPredictor)
