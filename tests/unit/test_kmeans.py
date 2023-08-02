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
from sagemaker.amazon.kmeans import KMeans, KMeansPredictor
from sagemaker.amazon.amazon_estimator import RecordSet
from sagemaker.session_settings import SessionSettings

ROLE = "myrole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.xlarge"
K = 2

COMMON_TRAIN_ARGS = {
    "role": ROLE,
    "instance_count": INSTANCE_COUNT,
    "instance_type": INSTANCE_TYPE,
}
ALL_REQ_ARGS = dict({"k": K}, **COMMON_TRAIN_ARGS)

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
    kmeans = KMeans(ROLE, INSTANCE_COUNT, INSTANCE_TYPE, K, sagemaker_session=sagemaker_session)
    assert kmeans.role == ROLE
    assert kmeans.instance_count == INSTANCE_COUNT
    assert kmeans.instance_type == INSTANCE_TYPE
    assert kmeans.k == K


def test_init_required_named(sagemaker_session):
    kmeans = KMeans(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    assert kmeans.role == COMMON_TRAIN_ARGS["role"]
    assert kmeans.instance_count == INSTANCE_COUNT
    assert kmeans.instance_type == COMMON_TRAIN_ARGS["instance_type"]
    assert kmeans.k == ALL_REQ_ARGS["k"]


def test_all_hyperparameters(sagemaker_session):
    kmeans = KMeans(
        sagemaker_session=sagemaker_session,
        init_method="random",
        max_iterations=3,
        tol=0.5,
        num_trials=5,
        local_init_method="kmeans++",
        half_life_time_size=0,
        epochs=10,
        center_factor=2,
        eval_metrics=["msd", "ssd"],
        **ALL_REQ_ARGS,
    )
    assert kmeans.hyperparameters() == dict(
        k=str(ALL_REQ_ARGS["k"]),
        init_method="random",
        local_lloyd_max_iter="3",
        local_lloyd_tol="0.5",
        local_lloyd_num_trials="5",
        local_lloyd_init_method="kmeans++",
        half_life_time_size="0",
        epochs="10",
        extra_center_factor="2",
        eval_metrics='["msd", "ssd"]',
        force_dense="True",
    )


def test_image(sagemaker_session):
    kmeans = KMeans(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    assert image_uris.retrieve("kmeans", REGION) == kmeans.training_image_uri()


@pytest.mark.parametrize("required_hyper_parameters, value", [("k", "string")])
def test_required_hyper_parameters_type(sagemaker_session, required_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params[required_hyper_parameters] = value
        KMeans(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize("required_hyper_parameters, value", [("k", 0)])
def test_required_hyper_parameters_value(sagemaker_session, required_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params[required_hyper_parameters] = value
        KMeans(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize("iterable_hyper_parameters, value", [("eval_metrics", 0)])
def test_iterable_hyper_parameters_type(sagemaker_session, iterable_hyper_parameters, value):
    with pytest.raises(TypeError):
        test_params = ALL_REQ_ARGS.copy()
        test_params.update({iterable_hyper_parameters: value})
        KMeans(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize(
    "optional_hyper_parameters, value",
    [
        ("init_method", 0),
        ("max_iterations", "string"),
        ("tol", "string"),
        ("num_trials", "string"),
        ("local_init_method", 0),
        ("half_life_time_size", "string"),
        ("epochs", "string"),
        ("center_factor", "string"),
    ],
)
def test_optional_hyper_parameters_type(sagemaker_session, optional_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params.update({optional_hyper_parameters: value})
        KMeans(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize(
    "optional_hyper_parameters, value",
    [
        ("init_method", "string"),
        ("max_iterations", 0),
        ("tol", -0.1),
        ("tol", 1.1),
        ("num_trials", 0),
        ("local_init_method", "string"),
        ("half_life_time_size", -1),
        ("epochs", 0),
        ("center_factor", 0),
    ],
)
def test_optional_hyper_parameters_value(sagemaker_session, optional_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params.update({optional_hyper_parameters: value})
        KMeans(sagemaker_session=sagemaker_session, **test_params)


PREFIX = "prefix"
FEATURE_DIM = 10
MINI_BATCH_SIZE = 200


@patch("sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit")
def test_call_fit(base_fit, sagemaker_session):
    kmeans = KMeans(base_job_name="kmeans", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )

    kmeans.fit(data, MINI_BATCH_SIZE)

    base_fit.assert_called_once()
    assert len(base_fit.call_args[0]) == 2
    assert base_fit.call_args[0][0] == data
    assert base_fit.call_args[0][1] == MINI_BATCH_SIZE


def test_prepare_for_training_no_mini_batch_size(sagemaker_session):
    kmeans = KMeans(base_job_name="kmeans", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    kmeans._prepare_for_training(data)

    assert kmeans.mini_batch_size == 5000


def test_prepare_for_training_wrong_type_mini_batch_size(sagemaker_session):
    kmeans = KMeans(base_job_name="kmeans", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )

    with pytest.raises((TypeError, ValueError)):
        kmeans._prepare_for_training(data, "some")


def test_prepare_for_training_wrong_value_mini_batch_size(sagemaker_session):
    kmeans = KMeans(base_job_name="kmeans", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    with pytest.raises(ValueError):
        kmeans._prepare_for_training(data, 0)


def test_model_image(sagemaker_session):
    kmeans = KMeans(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    kmeans.fit(data, MINI_BATCH_SIZE)

    model = kmeans.create_model()
    assert image_uris.retrieve("kmeans", REGION) == model.image_uri


def test_predictor_type(sagemaker_session):
    kmeans = KMeans(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    kmeans.fit(data, MINI_BATCH_SIZE)
    model = kmeans.create_model()
    predictor = model.deploy(1, INSTANCE_TYPE)

    assert isinstance(predictor, KMeansPredictor)


def test_predictor_custom_serialization(sagemaker_session):
    kmeans = KMeans(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    kmeans.fit(data, MINI_BATCH_SIZE)
    model = kmeans.create_model()
    custom_serializer = Mock()
    custom_deserializer = Mock()
    predictor = model.deploy(
        1,
        INSTANCE_TYPE,
        serializer=custom_serializer,
        deserializer=custom_deserializer,
    )

    assert isinstance(predictor, KMeansPredictor)
    assert predictor.serializer is custom_serializer
    assert predictor.deserializer is custom_deserializer
