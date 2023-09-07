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
from sagemaker.amazon.lda import LDA, LDAPredictor
from sagemaker.amazon.amazon_estimator import RecordSet
from sagemaker.session_settings import SessionSettings

ROLE = "myrole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.xlarge"
NUM_TOPICS = 3

COMMON_TRAIN_ARGS = {"role": ROLE, "instance_type": INSTANCE_TYPE}
ALL_REQ_ARGS = dict({"num_topics": NUM_TOPICS}, **COMMON_TRAIN_ARGS)

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
        config=None,
        local_mode=False,
        s3_client=None,
        s3_resource=None,
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
    lda = LDA(ROLE, INSTANCE_TYPE, NUM_TOPICS, sagemaker_session=sagemaker_session)
    assert lda.role == ROLE
    assert lda.instance_count == INSTANCE_COUNT
    assert lda.instance_type == INSTANCE_TYPE
    assert lda.num_topics == NUM_TOPICS


def test_init_required_named(sagemaker_session):
    lda = LDA(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    assert lda.role == COMMON_TRAIN_ARGS["role"]
    assert lda.instance_count == INSTANCE_COUNT
    assert lda.instance_type == COMMON_TRAIN_ARGS["instance_type"]
    assert lda.num_topics == ALL_REQ_ARGS["num_topics"]


def test_all_hyperparameters(sagemaker_session):
    lda = LDA(
        sagemaker_session=sagemaker_session,
        alpha0=2.2,
        max_restarts=3,
        max_iterations=10,
        tol=3.3,
        **ALL_REQ_ARGS,
    )
    assert lda.hyperparameters() == dict(
        num_topics=str(ALL_REQ_ARGS["num_topics"]),
        alpha0="2.2",
        max_restarts="3",
        max_iterations="10",
        tol="3.3",
    )


def test_image(sagemaker_session):
    lda = LDA(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    assert image_uris.retrieve("lda", REGION) == lda.training_image_uri()


@pytest.mark.parametrize("required_hyper_parameters, value", [("num_topics", "string")])
def test_required_hyper_parameters_type(sagemaker_session, required_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params[required_hyper_parameters] = value
        LDA(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize("required_hyper_parameters, value", [("num_topics", 0)])
def test_required_hyper_parameters_value(sagemaker_session, required_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params[required_hyper_parameters] = value
        LDA(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize(
    "optional_hyper_parameters, value",
    [
        ("alpha0", "string"),
        ("max_restarts", "string"),
        ("max_iterations", "string"),
        ("tol", "string"),
    ],
)
def test_optional_hyper_parameters_type(sagemaker_session, optional_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params.update({optional_hyper_parameters: value})
        LDA(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize(
    "optional_hyper_parameters, value", [("max_restarts", 0), ("max_iterations", 0), ("tol", 0)]
)
def test_optional_hyper_parameters_value(sagemaker_session, optional_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params.update({optional_hyper_parameters: value})
        LDA(sagemaker_session=sagemaker_session, **test_params)


PREFIX = "prefix"
FEATURE_DIM = 10
MINI_BATCH_SZIE = 200


@patch("sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit")
def test_call_fit(base_fit, sagemaker_session):
    lda = LDA(base_job_name="lda", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )

    lda.fit(data, MINI_BATCH_SZIE)

    base_fit.assert_called_once()
    assert len(base_fit.call_args[0]) == 2
    assert base_fit.call_args[0][0] == data
    assert base_fit.call_args[0][1] == MINI_BATCH_SZIE


def test_prepare_for_training_no_mini_batch_size(sagemaker_session):
    lda = LDA(base_job_name="lda", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    with pytest.raises(ValueError):
        lda._prepare_for_training(data, None)


def test_prepare_for_training_wrong_type_mini_batch_size(sagemaker_session):
    lda = LDA(base_job_name="lda", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )

    with pytest.raises(ValueError):
        lda._prepare_for_training(data, "some")


def test_prepare_for_training_wrong_value_mini_batch_size(sagemaker_session):
    lda = LDA(base_job_name="lda", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    with pytest.raises(ValueError):
        lda._prepare_for_training(data, 0)


def test_model_image(sagemaker_session):
    lda = LDA(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    lda.fit(data, MINI_BATCH_SZIE)

    model = lda.create_model()
    assert image_uris.retrieve("lda", REGION) == model.image_uri


def test_predictor_type(sagemaker_session):
    lda = LDA(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    lda.fit(data, MINI_BATCH_SZIE)
    model = lda.create_model()
    predictor = model.deploy(1, INSTANCE_TYPE)

    assert isinstance(predictor, LDAPredictor)


def test_predictor_custom_serialization(sagemaker_session):
    lda = LDA(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    lda.fit(data, MINI_BATCH_SZIE)
    model = lda.create_model()
    custom_serializer = Mock()
    custom_deserializer = Mock()
    predictor = model.deploy(
        1,
        INSTANCE_TYPE,
        serializer=custom_serializer,
        deserializer=custom_deserializer,
    )

    assert isinstance(predictor, LDAPredictor)
    assert predictor.serializer is custom_serializer
    assert predictor.deserializer is custom_deserializer
