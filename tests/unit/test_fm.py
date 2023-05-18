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
from sagemaker.amazon.factorization_machines import (
    FactorizationMachines,
    FactorizationMachinesPredictor,
)
from sagemaker.amazon.amazon_estimator import RecordSet
from sagemaker.session_settings import SessionSettings

ROLE = "myrole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.xlarge"
NUM_FACTORS = 3
PREDICTOR_TYPE = "regressor"

COMMON_TRAIN_ARGS = {
    "role": ROLE,
    "instance_count": INSTANCE_COUNT,
    "instance_type": INSTANCE_TYPE,
}
ALL_REQ_ARGS = dict(
    {"num_factors": NUM_FACTORS, "predictor_type": PREDICTOR_TYPE}, **COMMON_TRAIN_ARGS
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
        s3_client=False,
        s3_resource=False,
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
    fm = FactorizationMachines(
        "myrole", 1, "ml.c4.xlarge", 3, "regressor", sagemaker_session=sagemaker_session
    )
    assert fm.role == "myrole"
    assert fm.instance_count == 1
    assert fm.instance_type == "ml.c4.xlarge"
    assert fm.num_factors == 3
    assert fm.predictor_type == "regressor"


def test_init_required_named(sagemaker_session):
    fm = FactorizationMachines(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    assert fm.role == COMMON_TRAIN_ARGS["role"]
    assert fm.instance_count == COMMON_TRAIN_ARGS["instance_count"]
    assert fm.instance_type == COMMON_TRAIN_ARGS["instance_type"]
    assert fm.num_factors == ALL_REQ_ARGS["num_factors"]
    assert fm.predictor_type == ALL_REQ_ARGS["predictor_type"]


def test_all_hyperparameters(sagemaker_session):
    fm = FactorizationMachines(
        sagemaker_session=sagemaker_session,
        epochs=2,
        clip_gradient=1e2,
        eps=0.001,
        rescale_grad=2.2,
        bias_lr=0.01,
        linear_lr=0.002,
        factors_lr=0.0003,
        bias_wd=0.0004,
        linear_wd=1.01,
        factors_wd=1.002,
        bias_init_method="uniform",
        bias_init_scale=0.1,
        bias_init_sigma=0.05,
        bias_init_value=2.002,
        linear_init_method="constant",
        linear_init_scale=0.02,
        linear_init_sigma=0.003,
        linear_init_value=1.0,
        factors_init_method="normal",
        factors_init_scale=1.101,
        factors_init_sigma=1.202,
        factors_init_value=1.303,
        **ALL_REQ_ARGS,
    )
    assert fm.hyperparameters() == dict(
        num_factors=str(ALL_REQ_ARGS["num_factors"]),
        predictor_type=ALL_REQ_ARGS["predictor_type"],
        epochs="2",
        clip_gradient="100.0",
        eps="0.001",
        rescale_grad="2.2",
        bias_lr="0.01",
        linear_lr="0.002",
        factors_lr="0.0003",
        bias_wd="0.0004",
        linear_wd="1.01",
        factors_wd="1.002",
        bias_init_method="uniform",
        bias_init_scale="0.1",
        bias_init_sigma="0.05",
        bias_init_value="2.002",
        linear_init_method="constant",
        linear_init_scale="0.02",
        linear_init_sigma="0.003",
        linear_init_value="1.0",
        factors_init_method="normal",
        factors_init_scale="1.101",
        factors_init_sigma="1.202",
        factors_init_value="1.303",
    )


def test_image(sagemaker_session):
    fm = FactorizationMachines(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    assert image_uris.retrieve("factorization-machines", REGION) == fm.training_image_uri()


@pytest.mark.parametrize(
    "required_hyper_parameters, value", [("num_factors", "string"), ("predictor_type", 0)]
)
def test_required_hyper_parameters_type(sagemaker_session, required_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params[required_hyper_parameters] = value
        FactorizationMachines(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize(
    "required_hyper_parameters, value", [("num_factors", 0), ("predictor_type", "string")]
)
def test_required_hyper_parameters_value(sagemaker_session, required_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params[required_hyper_parameters] = value
        FactorizationMachines(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize(
    "optional_hyper_parameters, value",
    [
        ("epochs", "string"),
        ("clip_gradient", "string"),
        ("eps", "string"),
        ("rescale_grad", "string"),
        ("bias_lr", "string"),
        ("linear_lr", "string"),
        ("factors_lr", "string"),
        ("bias_wd", "string"),
        ("linear_wd", "string"),
        ("factors_wd", "string"),
        ("bias_init_method", 0),
        ("bias_init_scale", "string"),
        ("bias_init_sigma", "string"),
        ("bias_init_value", "string"),
        ("linear_init_method", 0),
        ("linear_init_scale", "string"),
        ("linear_init_sigma", "string"),
        ("linear_init_value", "string"),
        ("factors_init_method", 0),
        ("factors_init_scale", "string"),
        ("factors_init_sigma", "string"),
        ("factors_init_value", "string"),
    ],
)
def test_optional_hyper_parameters_type(sagemaker_session, optional_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params.update({optional_hyper_parameters: value})
        FactorizationMachines(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize(
    "optional_hyper_parameters, value",
    [
        ("epochs", 0),
        ("bias_lr", -1),
        ("linear_lr", -1),
        ("factors_lr", -1),
        ("bias_wd", -1),
        ("linear_wd", -1),
        ("factors_wd", -1),
        ("bias_init_method", "string"),
        ("bias_init_scale", -1),
        ("bias_init_sigma", -1),
        ("linear_init_method", "string"),
        ("linear_init_scale", -1),
        ("linear_init_sigma", -1),
        ("factors_init_method", "string"),
        ("factors_init_scale", -1),
        ("factors_init_sigma", -1),
    ],
)
def test_optional_hyper_parameters_value(sagemaker_session, optional_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params.update({optional_hyper_parameters: value})
        FactorizationMachines(sagemaker_session=sagemaker_session, **test_params)


PREFIX = "prefix"
FEATURE_DIM = 10
MINI_BATCH_SIZE = 200


@patch("sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit")
def test_call_fit(base_fit, sagemaker_session):
    fm = FactorizationMachines(
        base_job_name="fm", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS
    )

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )

    fm.fit(data, MINI_BATCH_SIZE)

    base_fit.assert_called_once()
    assert len(base_fit.call_args[0]) == 2
    assert base_fit.call_args[0][0] == data
    assert base_fit.call_args[0][1] == MINI_BATCH_SIZE


def test_prepare_for_training_no_mini_batch_size(sagemaker_session):
    fm = FactorizationMachines(
        base_job_name="fm", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS
    )

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    fm._prepare_for_training(data)


def test_prepare_for_training_wrong_type_mini_batch_size(sagemaker_session):
    fm = FactorizationMachines(
        base_job_name="fm", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS
    )

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )

    with pytest.raises((TypeError, ValueError)):
        fm._prepare_for_training(data, "some")


def test_prepare_for_training_wrong_value_mini_batch_size(sagemaker_session):
    fm = FactorizationMachines(
        base_job_name="fm", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS
    )

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    with pytest.raises(ValueError):
        fm._prepare_for_training(data, 0)


def test_model_image(sagemaker_session):
    fm = FactorizationMachines(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    fm.fit(data, MINI_BATCH_SIZE)

    model = fm.create_model()
    assert image_uris.retrieve("factorization-machines", REGION) == model.image_uri


def test_predictor_type(sagemaker_session):
    fm = FactorizationMachines(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    fm.fit(data, MINI_BATCH_SIZE)
    model = fm.create_model()
    predictor = model.deploy(1, INSTANCE_TYPE)

    assert isinstance(predictor, FactorizationMachinesPredictor)


def test_predictor_custom_serialization(sagemaker_session):
    fm = FactorizationMachines(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    fm.fit(data, MINI_BATCH_SIZE)
    model = fm.create_model()
    custom_serializer = Mock()
    custom_deserializer = Mock()
    predictor = model.deploy(
        1,
        INSTANCE_TYPE,
        serializer=custom_serializer,
        deserializer=custom_deserializer,
    )

    assert isinstance(predictor, FactorizationMachinesPredictor)
    assert predictor.serializer is custom_serializer
    assert predictor.deserializer is custom_deserializer
