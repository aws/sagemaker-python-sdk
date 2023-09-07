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
from sagemaker.amazon.object2vec import Object2Vec
from sagemaker.predictor import Predictor
from sagemaker.amazon.amazon_estimator import RecordSet
from sagemaker.session_settings import SessionSettings

ROLE = "myrole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.xlarge"
EPOCHS = 5
ENC0_MAX_SEQ_LEN = 100
ENC0_VOCAB_SIZE = 500

MINI_BATCH_SIZE = 32

COMMON_TRAIN_ARGS = {
    "role": ROLE,
    "instance_count": INSTANCE_COUNT,
    "instance_type": INSTANCE_TYPE,
}
ALL_REQ_ARGS = dict(
    {"epochs": EPOCHS, "enc0_max_seq_len": ENC0_MAX_SEQ_LEN, "enc0_vocab_size": ENC0_VOCAB_SIZE},
    **COMMON_TRAIN_ARGS,
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
    object2vec = Object2Vec(
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        EPOCHS,
        ENC0_MAX_SEQ_LEN,
        ENC0_VOCAB_SIZE,
        sagemaker_session=sagemaker_session,
    )
    assert object2vec.role == ROLE
    assert object2vec.instance_count == INSTANCE_COUNT
    assert object2vec.instance_type == INSTANCE_TYPE
    assert object2vec.epochs == EPOCHS
    assert object2vec.enc0_max_seq_len == ENC0_MAX_SEQ_LEN
    assert object2vec.enc0_vocab_size == ENC0_VOCAB_SIZE


def test_init_required_named(sagemaker_session):
    object2vec = Object2Vec(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    assert object2vec.role == COMMON_TRAIN_ARGS["role"]
    assert object2vec.instance_count == INSTANCE_COUNT
    assert object2vec.instance_type == COMMON_TRAIN_ARGS["instance_type"]
    assert object2vec.epochs == ALL_REQ_ARGS["epochs"]
    assert object2vec.enc0_max_seq_len == ALL_REQ_ARGS["enc0_max_seq_len"]
    assert object2vec.enc0_vocab_size == ALL_REQ_ARGS["enc0_vocab_size"]


def test_all_hyperparameters(sagemaker_session):
    object2vec = Object2Vec(
        sagemaker_session=sagemaker_session,
        enc_dim=1024,
        mini_batch_size=100,
        early_stopping_patience=3,
        early_stopping_tolerance=0.001,
        dropout=0.1,
        weight_decay=0.001,
        bucket_width=0,
        num_classes=5,
        mlp_layers=3,
        mlp_dim=1024,
        mlp_activation="tanh",
        output_layer="softmax",
        optimizer="adam",
        learning_rate=0.0001,
        negative_sampling_rate=1,
        comparator_list="hadamard, abs_diff",
        tied_token_embedding_weight=True,
        token_embedding_storage_type="row_sparse",
        enc0_network="bilstm",
        enc1_network="hcnn",
        enc0_cnn_filter_width=3,
        enc1_cnn_filter_width=3,
        enc1_max_seq_len=300,
        enc0_token_embedding_dim=300,
        enc1_token_embedding_dim=300,
        enc1_vocab_size=300,
        enc0_layers=3,
        enc1_layers=3,
        enc0_freeze_pretrained_embedding=True,
        enc1_freeze_pretrained_embedding=False,
        **ALL_REQ_ARGS,
    )

    hp = object2vec.hyperparameters()
    assert hp["epochs"] == str(EPOCHS)
    assert hp["mlp_activation"] == "tanh"


def test_image(sagemaker_session):
    object2vec = Object2Vec(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    assert image_uris.retrieve("object2vec", REGION) == object2vec.training_image_uri()


@pytest.mark.parametrize("required_hyper_parameters, value", [("epochs", "string")])
def test_required_hyper_parameters_type(sagemaker_session, required_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params[required_hyper_parameters] = value
        Object2Vec(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize(
    "required_hyper_parameters, value", [("enc0_vocab_size", 0), ("enc0_vocab_size", 1000000000)]
)
def test_required_hyper_parameters_value(sagemaker_session, required_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params[required_hyper_parameters] = value
        Object2Vec(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize(
    "optional_hyper_parameters, value",
    [
        ("epochs", "string"),
        ("optimizer", 0),
        ("enc0_cnn_filter_width", "string"),
        ("weight_decay", "string"),
        ("learning_rate", "string"),
        ("negative_sampling_rate", "some_string"),
        ("comparator_list", 0),
        ("comparator_list", ["foobar"]),
        ("token_embedding_storage_type", 123),
    ],
)
def test_optional_hyper_parameters_type(sagemaker_session, optional_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params.update({optional_hyper_parameters: value})
        Object2Vec(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize(
    "optional_hyper_parameters, value",
    [
        ("epochs", 0),
        ("epochs", 1000),
        ("optimizer", "string"),
        ("early_stopping_tolerance", 0),
        ("early_stopping_tolerance", 0.5),
        ("early_stopping_patience", 0),
        ("early_stopping_patience", 100),
        ("weight_decay", -1),
        ("weight_decay", 200000),
        ("enc0_cnn_filter_width", 2000),
        ("learning_rate", 0),
        ("learning_rate", 2),
        ("negative_sampling_rate", -1),
        ("comparator_list", "hadamard,foobar"),
        ("token_embedding_storage_type", "foobar"),
    ],
)
def test_optional_hyper_parameters_value(sagemaker_session, optional_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params.update({optional_hyper_parameters: value})
        Object2Vec(sagemaker_session=sagemaker_session, **test_params)


PREFIX = "prefix"
FEATURE_DIM = 10


@patch("sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit")
def test_call_fit(base_fit, sagemaker_session):
    object2vec = Object2Vec(
        base_job_name="object2vec", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS
    )

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )

    object2vec.fit(data, MINI_BATCH_SIZE)

    base_fit.assert_called_once()
    assert len(base_fit.call_args[0]) == 2
    assert base_fit.call_args[0][0] == data
    assert base_fit.call_args[0][1] == MINI_BATCH_SIZE


def test_call_fit_none_mini_batch_size(sagemaker_session):
    object2vec = Object2Vec(
        base_job_name="object2vec", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS
    )

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    object2vec.fit(data)


def test_prepare_for_training_wrong_type_mini_batch_size(sagemaker_session):
    object2vec = Object2Vec(
        base_job_name="object2vec", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS
    )

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )

    with pytest.raises((TypeError, ValueError)):
        object2vec._prepare_for_training(data, "some")


def test_prepare_for_training_wrong_value_lower_mini_batch_size(sagemaker_session):
    object2vec = Object2Vec(
        base_job_name="object2vec", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS
    )

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    with pytest.raises(ValueError):
        object2vec._prepare_for_training(data, 0)


def test_prepare_for_training_wrong_value_upper_mini_batch_size(sagemaker_session):
    object2vec = Object2Vec(
        base_job_name="object2vec", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS
    )

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    with pytest.raises(ValueError):
        object2vec._prepare_for_training(data, 10001)


def test_model_image(sagemaker_session):
    object2vec = Object2Vec(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    object2vec.fit(data, MINI_BATCH_SIZE)

    model = object2vec.create_model()
    assert image_uris.retrieve("object2vec", REGION) == model.image_uri


def test_predictor_type(sagemaker_session):
    object2vec = Object2Vec(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    object2vec.fit(data, MINI_BATCH_SIZE)
    model = object2vec.create_model()
    predictor = model.deploy(1, INSTANCE_TYPE)

    assert isinstance(predictor, Predictor)
