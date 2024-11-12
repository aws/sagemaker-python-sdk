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
from sagemaker.amazon.ipinsights import IPInsights, IPInsightsPredictor
from sagemaker.amazon.amazon_estimator import RecordSet
from sagemaker.session_settings import SessionSettings

# Mocked training config
ROLE = "myrole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.xlarge"

# Required algorithm hyperparameters
NUM_ENTITY_VECTORS = 10000
VECTOR_DIM = 128

COMMON_TRAIN_ARGS = {
    "role": ROLE,
    "instance_count": INSTANCE_COUNT,
    "instance_type": INSTANCE_TYPE,
}
ALL_REQ_ARGS = dict(
    {"num_entity_vectors": NUM_ENTITY_VECTORS, "vector_dim": VECTOR_DIM}, **COMMON_TRAIN_ARGS
)
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
    ipinsights = IPInsights(
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        NUM_ENTITY_VECTORS,
        VECTOR_DIM,
        sagemaker_session=sagemaker_session,
    )
    assert ipinsights.role == ROLE
    assert ipinsights.instance_count == INSTANCE_COUNT
    assert ipinsights.instance_type == INSTANCE_TYPE
    assert ipinsights.num_entity_vectors == NUM_ENTITY_VECTORS
    assert ipinsights.vector_dim == VECTOR_DIM


def test_init_required_named(sagemaker_session):
    ipinsights = IPInsights(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    assert ipinsights.role == COMMON_TRAIN_ARGS["role"]
    assert ipinsights.instance_count == INSTANCE_COUNT
    assert ipinsights.instance_type == COMMON_TRAIN_ARGS["instance_type"]
    assert ipinsights.num_entity_vectors == NUM_ENTITY_VECTORS
    assert ipinsights.vector_dim == VECTOR_DIM


def test_all_hyperparameters(sagemaker_session):
    ipinsights = IPInsights(
        sagemaker_session=sagemaker_session,
        batch_metrics_publish_interval=100,
        epochs=10,
        learning_rate=0.001,
        num_ip_encoder_layers=3,
        random_negative_sampling_rate=5,
        shuffled_negative_sampling_rate=5,
        weight_decay=5.0,
        **ALL_REQ_ARGS,
    )
    assert ipinsights.hyperparameters() == dict(
        num_entity_vectors=str(ALL_REQ_ARGS["num_entity_vectors"]),
        vector_dim=str(ALL_REQ_ARGS["vector_dim"]),
        batch_metrics_publish_interval="100",
        epochs="10",
        learning_rate="0.001",
        num_ip_encoder_layers="3",
        random_negative_sampling_rate="5",
        shuffled_negative_sampling_rate="5",
        weight_decay="5.0",
    )


def test_image(sagemaker_session):
    ipinsights = IPInsights(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    assert image_uris.retrieve("ipinsights", REGION) == ipinsights.training_image_uri()


@pytest.mark.parametrize(
    "required_hyper_parameters, value", [("num_entity_vectors", "string"), ("vector_dim", "string")]
)
def test_required_hyper_parameters_type(sagemaker_session, required_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params[required_hyper_parameters] = value
        IPInsights(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize(
    "required_hyper_parameters, value",
    [
        ("num_entity_vectors", 0),
        ("num_entity_vectors", 500000001),
        ("vector_dim", 3),
        ("vector_dim", 4097),
    ],
)
def test_required_hyper_parameters_value(sagemaker_session, required_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params[required_hyper_parameters] = value
        IPInsights(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize(
    "optional_hyper_parameters, value",
    [
        ("batch_metrics_publish_interval", "string"),
        ("epochs", "string"),
        ("learning_rate", "string"),
        ("num_ip_encoder_layers", "string"),
        ("random_negative_sampling_rate", "string"),
        ("shuffled_negative_sampling_rate", "string"),
        ("weight_decay", "string"),
    ],
)
def test_optional_hyper_parameters_type(sagemaker_session, optional_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params.update({optional_hyper_parameters: value})
        IPInsights(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize(
    "optional_hyper_parameters, value",
    [
        ("batch_metrics_publish_interval", 0),
        ("epochs", 0),
        ("learning_rate", 0),
        ("learning_rate", 11),
        ("num_ip_encoder_layers", -1),
        ("num_ip_encoder_layers", 101),
        ("random_negative_sampling_rate", -1),
        ("random_negative_sampling_rate", 501),
        ("shuffled_negative_sampling_rate", -1),
        ("shuffled_negative_sampling_rate", 501),
        ("weight_decay", -1),
        ("weight_decay", 11),
    ],
)
def test_optional_hyper_parameters_value(sagemaker_session, optional_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params.update({optional_hyper_parameters: value})
        IPInsights(sagemaker_session=sagemaker_session, **test_params)


PREFIX = "prefix"
FEATURE_DIM = None
MINI_BATCH_SIZE = 200


@patch("sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit")
def test_call_fit(base_fit, sagemaker_session):
    ipinsights = IPInsights(
        base_job_name="ipinsights", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS
    )

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )

    ipinsights.fit(data, MINI_BATCH_SIZE)

    base_fit.assert_called_once()
    assert len(base_fit.call_args[0]) == 2
    assert base_fit.call_args[0][0] == data
    assert base_fit.call_args[0][1] == MINI_BATCH_SIZE


def test_call_fit_none_mini_batch_size(sagemaker_session):
    ipinsights = IPInsights(
        base_job_name="ipinsights", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS
    )

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    ipinsights.fit(data)


def test_prepare_for_training_wrong_type_mini_batch_size(sagemaker_session):
    ipinsights = IPInsights(
        base_job_name="ipinsights", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS
    )

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )

    with pytest.raises((TypeError, ValueError)):
        ipinsights._prepare_for_training(data, "some")


def test_prepare_for_training_wrong_value_lower_mini_batch_size(sagemaker_session):
    ipinsights = IPInsights(
        base_job_name="ipinsights", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS
    )

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    with pytest.raises(ValueError):
        ipinsights._prepare_for_training(data, 0)


def test_prepare_for_training_wrong_value_upper_mini_batch_size(sagemaker_session):
    ipinsights = IPInsights(
        base_job_name="ipinsights", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS
    )

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    with pytest.raises(ValueError):
        ipinsights._prepare_for_training(data, 500001)


def test_model_image(sagemaker_session):
    ipinsights = IPInsights(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    ipinsights.fit(data, MINI_BATCH_SIZE)

    model = ipinsights.create_model()
    assert image_uris.retrieve("ipinsights", REGION) == model.image_uri


def test_predictor_type(sagemaker_session):
    ipinsights = IPInsights(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    ipinsights.fit(data, MINI_BATCH_SIZE)
    model = ipinsights.create_model()
    predictor = model.deploy(1, INSTANCE_TYPE)

    assert isinstance(predictor, IPInsightsPredictor)


def test_predictor_custom_serialization(sagemaker_session):
    ipinsights = IPInsights(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    ipinsights.fit(data, MINI_BATCH_SIZE)
    model = ipinsights.create_model()
    custom_serializer = Mock()
    custom_deserializer = Mock()
    predictor = model.deploy(
        1,
        INSTANCE_TYPE,
        serializer=custom_serializer,
        deserializer=custom_deserializer,
    )

    assert isinstance(predictor, IPInsightsPredictor)
    assert predictor.serializer is custom_serializer
    assert predictor.deserializer is custom_deserializer
