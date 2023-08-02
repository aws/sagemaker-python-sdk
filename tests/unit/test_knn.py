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
from sagemaker.amazon.knn import KNN, KNNPredictor
from sagemaker.amazon.amazon_estimator import RecordSet
from sagemaker.session_settings import SessionSettings

ROLE = "myrole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.xlarge"
K = 5
SAMPLE_SIZE = 1000
PREDICTOR_TYPE_REGRESSOR = "regressor"
PREDICTOR_TYPE_CLASSIFIER = "classifier"

COMMON_TRAIN_ARGS = {
    "role": ROLE,
    "instance_count": INSTANCE_COUNT,
    "instance_type": INSTANCE_TYPE,
}
ALL_REQ_ARGS = dict(
    {"k": K, "sample_size": SAMPLE_SIZE, "predictor_type": PREDICTOR_TYPE_REGRESSOR},
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
    knn = KNN(
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        K,
        SAMPLE_SIZE,
        PREDICTOR_TYPE_REGRESSOR,
        sagemaker_session=sagemaker_session,
    )
    assert knn.role == ROLE
    assert knn.instance_count == INSTANCE_COUNT
    assert knn.instance_type == INSTANCE_TYPE
    assert knn.k == K


def test_init_required_named(sagemaker_session):
    knn = KNN(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    assert knn.role == COMMON_TRAIN_ARGS["role"]
    assert knn.instance_count == INSTANCE_COUNT
    assert knn.instance_type == COMMON_TRAIN_ARGS["instance_type"]
    assert knn.k == ALL_REQ_ARGS["k"]


def test_all_hyperparameters_regressor(sagemaker_session):
    knn = KNN(
        sagemaker_session=sagemaker_session,
        dimension_reduction_type="sign",
        dimension_reduction_target="2",
        index_type="faiss.Flat",
        index_metric="COSINE",
        faiss_index_ivf_nlists="auto",
        faiss_index_pq_m=1,
        **ALL_REQ_ARGS,
    )
    assert knn.hyperparameters() == dict(
        k=str(ALL_REQ_ARGS["k"]),
        sample_size=str(ALL_REQ_ARGS["sample_size"]),
        predictor_type=str(ALL_REQ_ARGS["predictor_type"]),
        dimension_reduction_type="sign",
        dimension_reduction_target="2",
        index_type="faiss.Flat",
        index_metric="COSINE",
        faiss_index_ivf_nlists="auto",
        faiss_index_pq_m="1",
    )


def test_all_hyperparameters_classifier(sagemaker_session):
    test_params = ALL_REQ_ARGS.copy()
    test_params["predictor_type"] = PREDICTOR_TYPE_CLASSIFIER

    knn = KNN(
        sagemaker_session=sagemaker_session,
        dimension_reduction_type="fjlt",
        dimension_reduction_target="2",
        index_type="faiss.IVFFlat",
        index_metric="L2",
        faiss_index_ivf_nlists="20",
        **test_params,
    )
    assert knn.hyperparameters() == dict(
        k=str(ALL_REQ_ARGS["k"]),
        sample_size=str(ALL_REQ_ARGS["sample_size"]),
        predictor_type=str(PREDICTOR_TYPE_CLASSIFIER),
        dimension_reduction_type="fjlt",
        dimension_reduction_target="2",
        index_type="faiss.IVFFlat",
        index_metric="L2",
        faiss_index_ivf_nlists="20",
    )


def test_image(sagemaker_session):
    knn = KNN(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    assert image_uris.retrieve("knn", REGION) == knn.training_image_uri()


@pytest.mark.parametrize(
    "required_hyper_parameters, value",
    [("k", "string"), ("sample_size", "string"), ("predictor_type", 1)],
)
def test_required_hyper_parameters_type(sagemaker_session, required_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params[required_hyper_parameters] = value
        KNN(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize("required_hyper_parameters, value", [("predictor_type", "random_string")])
def test_required_hyper_parameters_value(sagemaker_session, required_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params[required_hyper_parameters] = value
        KNN(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize(
    "iterable_hyper_parameters, value", [("index_type", 1), ("index_metric", "string")]
)
def test_error_optional_hyper_parameters_type(sagemaker_session, iterable_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params.update({iterable_hyper_parameters: value})
        KNN(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize(
    "optional_hyper_parameters, value",
    [("index_type", "faiss.random"), ("index_metric", "randomstring"), ("faiss_index_pq_m", -1)],
)
def test_error_optional_hyper_parameters_value(sagemaker_session, optional_hyper_parameters, value):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params.update({optional_hyper_parameters: value})
        KNN(sagemaker_session=sagemaker_session, **test_params)


@pytest.mark.parametrize(
    "conditional_hyper_parameters",
    [
        {"dimension_reduction_type": "sign"},  # errors due to missing dimension_reduction_target
        {"dimension_reduction_type": "sign", "dimension_reduction_target": -2},
        {"dimension_reduction_type": "sign", "dimension_reduction_target": "string"},
        {"dimension_reduction_type": 2, "dimension_reduction_target": 20},
        {"dimension_reduction_type": "randomstring", "dimension_reduction_target": 20},
    ],
)
def test_error_conditional_hyper_parameters_value(sagemaker_session, conditional_hyper_parameters):
    with pytest.raises(ValueError):
        test_params = ALL_REQ_ARGS.copy()
        test_params.update(conditional_hyper_parameters)
        KNN(sagemaker_session=sagemaker_session, **test_params)


PREFIX = "prefix"
FEATURE_DIM = 10
MINI_BATCH_SIZE = 200


@patch("sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit")
def test_call_fit(base_fit, sagemaker_session):
    knn = KNN(base_job_name="knn", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )

    knn.fit(data, MINI_BATCH_SIZE)

    base_fit.assert_called_once()
    assert len(base_fit.call_args[0]) == 2
    assert base_fit.call_args[0][0] == data
    assert base_fit.call_args[0][1] == MINI_BATCH_SIZE


def test_call_fit_none_mini_batch_size(sagemaker_session):
    knn = KNN(base_job_name="knn", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    knn.fit(data)


def test_prepare_for_training_wrong_type_mini_batch_size(sagemaker_session):
    knn = KNN(base_job_name="knn", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )

    with pytest.raises((TypeError, ValueError)):
        knn._prepare_for_training(data, "some")


def test_prepare_for_training_wrong_value_lower_mini_batch_size(sagemaker_session):
    knn = KNN(base_job_name="knn", sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)

    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    with pytest.raises(ValueError):
        knn._prepare_for_training(data, 0)


def test_model_image(sagemaker_session):
    knn = KNN(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    knn.fit(data, MINI_BATCH_SIZE)

    model = knn.create_model()
    assert image_uris.retrieve("knn", REGION) == model.image_uri


def test_predictor_type(sagemaker_session):
    knn = KNN(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    knn.fit(data, MINI_BATCH_SIZE)
    model = knn.create_model()
    predictor = model.deploy(1, INSTANCE_TYPE)

    assert isinstance(predictor, KNNPredictor)


def test_predictor_custom_serialization(sagemaker_session):
    knn = KNN(sagemaker_session=sagemaker_session, **ALL_REQ_ARGS)
    data = RecordSet(
        "s3://{}/{}".format(BUCKET_NAME, PREFIX),
        num_records=1,
        feature_dim=FEATURE_DIM,
        channel="train",
    )
    knn.fit(data, MINI_BATCH_SIZE)
    model = knn.create_model()
    custom_serializer = Mock()
    custom_deserializer = Mock()
    predictor = model.deploy(
        1,
        INSTANCE_TYPE,
        serializer=custom_serializer,
        deserializer=custom_deserializer,
    )

    assert isinstance(predictor, KNNPredictor)
    assert predictor.serializer is custom_serializer
    assert predictor.deserializer is custom_deserializer
