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
from sagemaker.amazon.pca import PCA, PCAPredictor
from sagemaker.amazon.amazon_estimator import RecordSet
from sagemaker.session_settings import SessionSettings

ROLE = "myrole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.xlarge"
NUM_COMPONENTS = 5

COMMON_TRAIN_ARGS = {
    "role": ROLE,
    "instance_count": INSTANCE_COUNT,
    "instance_type": INSTANCE_TYPE,
}
ALL_REQ_ARGS = dict({"num_components": NUM_COMPONENTS}, **COMMON_TRAIN_ARGS)

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
    pca = PCA(
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        NUM_COMPONENTS,
        sagemaker_session=sagemaker_session,
    )
    assert pca.role == ROLE
    assert pca.instance_count == INSTANCE_COUNT
    assert pca.instance_type == INSTANCE_TYPE
    assert pca.num_components == NUM_COMPONENTS


def test_init_required_named(sagemaker_session):
    pca = PCA(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    assert pca.role == COMMON_TRAIN_ARGS["role"]
    assert pca.instance_count == INSTANCE_COUNT
    assert pca.instance_type == COMMON_TRAIN_ARGS["instance_type"]
    assert pca.num_components == ALL_REQ_ARGS["num_components"]


def test_all_hyperparameters(sagemaker_session):
    pca = PCA(
        sagemaker_session=sagemaker_session,
        algorithm_mode="regular",
        subtract_mean="True",
        extra_components=1,
        **ALL_REQ_ARGS,
    )
    assert pca.hyperparameters() == dict(
        num_components=str(ALL_REQ_ARGS["num_components"]),
        algorithm_mode="regular",
        subtract_mean="True",
        extra_components="1",
    )


def test_image(sagemaker_session):
    pca = PCA(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    assert image_uris.retrieve("pca", REGION) == pca.training_image_uri()


@pytest.mark.parametrize("required_hyper_parameters, value", [("num_components", "string")])
def test_required_hyper_parameters_type(sagemaker_session, required_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params[required_hyper_parameters] = value
        PCA(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize("required_hyper_parameters, value", [("num_components", 0)])
def test_required_hyper_parameters_value(sagemaker_session, required_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params[required_hyper_parameters] = value
        PCA(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize(
    "optional_hyper_parameters, value", [("algorithm_mode", 0), ("extra_components", "string")]
)
def test_optional_hyper_parameters_type(sagemaker_session, optional_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params.update({optional_hyper_parameters: value})
        PCA(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize("optional_hyper_parameters, value", [("algorithm_mode", "string")])
def test_optional_hyper_parameters_value(sagemaker_session, optional_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params.update({optional_hyper_parameters: value})
        PCA(sagemaker_session=sagemaker_session, **test_params)


PREFIX = "prefix"
FEATURE_DIM = 10
MINI_BATCH_SIZE = 200


@patch("sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit")
def test_call_fit(base_fit, sagemaker_session):
    pca = PCA(base_job_name="pca", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )

    pca.fit(data, MINI_BATCH_SIZE)

    base_fit.assert_called_once()
    assert len(base_fit.call_args[0]) == 2
    assert base_fit.call_args[0][0] == data
    assert base_fit.call_args[0][1] == MINI_BATCH_SIZE


def test_prepare_for_training_no_mini_batch_size(sagemaker_session):
    pca = PCA(base_job_name="pca", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    pca._prepare_for_training(data)

    assert pca.mini_batch_size == 1


def test_prepare_for_training_wrong_type_mini_batch_size(sagemaker_session):
    pca = PCA(base_job_name="pca", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )

    with pytest.raises((TypeError, ValueError)):
        pca.fit(data, "some")


def test_prepare_for_training_multiple_channel(sagemaker_session):
    lr = PCA(base_job_name="lr", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )

    lr._prepare_for_training([data, data])

    assert lr.mini_batch_size == 1


def test_prepare_for_training_multiple_channel_no_train(sagemaker_session):
    lr = PCA(base_job_name="lr", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="mock",
    )

    with pytest.raises(ValueError) as ex:
        lr._prepare_for_training([data, data])

    assert "Must provide train channel." in str(ex)


def test_model_image(sagemaker_session):
    pca = PCA(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    pca.fit(data, MINI_BATCH_SIZE)

    model = pca.create_model()
    assert image_uris.retrieve("pca", REGION) == model.image_uri


def test_predictor_type(sagemaker_session):
    pca = PCA(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    pca.fit(data, MINI_BATCH_SIZE)
    model = pca.create_model()
    predictor = model.deploy(1, INSTANCE_TYPE)

    assert isinstance(predictor, PCAPredictor)


def test_predictor_custom_serialization(sagemaker_session):
    pca = PCA(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    pca.fit(data, MINI_BATCH_SIZE)
    model = pca.create_model()
    custom_serializer = Mock()
    custom_deserializer = Mock()
    predictor = model.deploy(
        1,
        INSTANCE_TYPE,
        serializer=custom_serializer,
        deserializer=custom_deserializer,
    )

    assert isinstance(predictor, PCAPredictor)
    assert predictor.serializer is custom_serializer
    assert predictor.deserializer is custom_deserializer
