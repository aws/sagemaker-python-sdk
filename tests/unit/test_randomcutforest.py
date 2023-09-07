# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import pytest
from mock import Mock, patch

from sagemaker import image_uris
from sagemaker.amazon.randomcutforest import RandomCutForest, RandomCutForestPredictor
from sagemaker.amazon.amazon_estimator import RecordSet
from sagemaker.session_settings import SessionSettings

ROLE = "myrole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.xlarge"
NUM_SAMPLES_PER_TREE = 20
NUM_TREES = 50
EVAL_METRICS = ["accuracy", "precision_recall_fscore"]

COMMON_TRAIN_ARGS = {
    "role": ROLE,
    "instance_count": INSTANCE_COUNT,
    "instance_type": INSTANCE_TYPE,
}
ALL_REQ_ARGS = dict(**COMMON_TRAIN_ARGS)

REGION = "us-west-2"
BUCKET_NAME = "Some-Bucket"

DESCRIBE_TRAINING_JOB_RESULT = {"ModelArtifacts": {"S3ModelArtifacts": "s3://bucket/model.tar.gz"}}

ENDPOINT_DESC = {"EndpointConfigName": "test-endpoint"}

ENDPOINT_CONFIG_DESC = {"ProductionVariants": [{"ModelName": "model-1"}, {"ModelName": "model-2"}]}


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    sms = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        region_name=REGION,
        config=None,
        local_mode=False,
        settings=SessionSettings(),
        default_bucket_prefix=None,
    )
    sms.boto_region_name = REGION
    sms.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    sms.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job", return_value=DESCRIBE_TRAINING_JOB_RESULT
    )
    sms.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    sms.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)
    # For tests which doesn't verify config file injection, operate with empty config
    sms.sagemaker_config = {}
    return sms


def test_init_required_positional(sagemaker_session):
    randomcutforest = RandomCutForest(
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        NUM_SAMPLES_PER_TREE,
        NUM_TREES,
        EVAL_METRICS,
        sagemaker_session=sagemaker_session,
    )
    assert randomcutforest.role == ROLE
    assert randomcutforest.instance_count == INSTANCE_COUNT
    assert randomcutforest.instance_type == INSTANCE_TYPE
    assert randomcutforest.num_trees == NUM_TREES
    assert randomcutforest.num_samples_per_tree == NUM_SAMPLES_PER_TREE
    assert randomcutforest.eval_metrics == EVAL_METRICS


def test_init_required_named(sagemaker_session):
    randomcutforest = RandomCutForest(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    assert randomcutforest.role == COMMON_TRAIN_ARGS["role"]
    assert randomcutforest.instance_count == INSTANCE_COUNT
    assert randomcutforest.instance_type == COMMON_TRAIN_ARGS["instance_type"]


def test_all_hyperparameters(sagemaker_session):
    randomcutforest = RandomCutForest(
        sagemaker_session=sagemaker_session,
        num_trees=NUM_TREES,
        num_samples_per_tree=NUM_SAMPLES_PER_TREE,
        eval_metrics=EVAL_METRICS,
        **ALL_REQ_ARGS,
    )
    assert randomcutforest.hyperparameters() == dict(
        num_samples_per_tree=str(NUM_SAMPLES_PER_TREE),
        num_trees=str(NUM_TREES),
        eval_metrics='["accuracy", "precision_recall_fscore"]',
    )


def test_image(sagemaker_session):
    randomcutforest = RandomCutForest(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    assert image_uris.retrieve("randomcutforest", REGION) == randomcutforest.training_image_uri()


@pytest.mark.parametrize("iterable_hyper_parameters, value", [("eval_metrics", 0)])
def test_iterable_hyper_parameters_type(sagemaker_session, iterable_hyper_parameters, value):
    with pytest.raises(TypeError):
        test_params = ALL_REQ_ARGS.copy()
        test_params.update({iterable_hyper_parameters: value})
        RandomCutForest(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize(
    "optional_hyper_parameters, value",
    [("num_trees", "string"), ("num_samples_per_tree", "string")],
)
def test_optional_hyper_parameters_type(sagemaker_session, optional_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params.update({optional_hyper_parameters: value})
        RandomCutForest(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize(
    "optional_hyper_parameters, value",
    [
        ("num_trees", 49),
        ("num_trees", 1001),
        ("num_samples_per_tree", 0),
        ("num_samples_per_tree", 2049),
    ],
)
def test_optional_hyper_parameters_value(sagemaker_session, optional_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params.update({optional_hyper_parameters: value})
        RandomCutForest(sagemaker_session=sagemaker_session, **test_params)


PREFIX = "prefix"
FEATURE_DIM = 10
MAX_FEATURE_DIM = 10000
MINI_BATCH_SIZE = 1000


@patch("sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit")
def test_call_fit(base_fit, sagemaker_session):
    randomcutforest = RandomCutForest(
        base_job_name="randomcutforest", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS
    )

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )

    randomcutforest.fit(data, MINI_BATCH_SIZE)

    base_fit.assert_called_once()
    assert len(base_fit.call_args[0]) == 2
    assert base_fit.call_args[0][0] == data
    assert base_fit.call_args[0][1] == MINI_BATCH_SIZE


def test_prepare_for_training_no_mini_batch_size(sagemaker_session):
    randomcutforest = RandomCutForest(
        base_job_name="randomcutforest", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS
    )

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    randomcutforest._prepare_for_training(data)

    assert randomcutforest.mini_batch_size == MINI_BATCH_SIZE


def test_prepare_for_training_wrong_type_mini_batch_size(sagemaker_session):
    randomcutforest = RandomCutForest(
        base_job_name="randomcutforest", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS
    )

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )

    with pytest.raises((TypeError, ValueError)):
        randomcutforest._prepare_for_training(data, 1234)


def test_prepare_for_training_feature_dim_greater_than_max_allowed(sagemaker_session):
    randomcutforest = RandomCutForest(
        base_job_name="randomcutforest", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS
    )

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=MAX_FEATURE_DIM + 1,
        channel="train",
    )

    with pytest.raises((TypeError, ValueError)):
        randomcutforest._prepare_for_training(data)


def test_model_image(sagemaker_session):
    randomcutforest = RandomCutForest(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    randomcutforest.fit(data, MINI_BATCH_SIZE)

    model = randomcutforest.create_model()
    assert image_uris.retrieve("randomcutforest", REGION) == model.image_uri


def test_predictor_type(sagemaker_session):
    randomcutforest = RandomCutForest(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    randomcutforest.fit(data, MINI_BATCH_SIZE)
    model = randomcutforest.create_model()
    predictor = model.deploy(1, INSTANCE_TYPE)

    assert isinstance(predictor, RandomCutForestPredictor)
