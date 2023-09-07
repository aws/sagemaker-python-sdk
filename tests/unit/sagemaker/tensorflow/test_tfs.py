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

import io
import json
import logging

import mock
import pytest
from mock import Mock, patch, ANY

from sagemaker.serializers import CSVSerializer, IdentitySerializer
from sagemaker.tensorflow import TensorFlow, TensorFlowModel, TensorFlowPredictor

JSON_CONTENT_TYPE = "application/json"
CSV_CONTENT_TYPE = "text/csv"
IMAGE = "tensorflow-inference:2.0.0-cpu"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.4xlarge"
ACCELERATOR_TYPE = "ml.eia1.medium"
ROLE = "Dummy"
REGION = "us-west-2"
PREDICT_INPUT = {"instances": [1.0, 2.0, 5.0]}
PREDICT_RESPONSE = {"predictions": [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}
CLASSIFY_INPUT = {
    "signature_name": "tensorflow/serving/classify",
    "examples": [{"x": 1.0}, {"x": 2.0}],
}
CLASSIFY_RESPONSE = {"result": [[0.4, 0.6], [0.2, 0.8]]}
REGRESS_INPUT = {
    "signature_name": "tensorflow/serving/regress",
    "examples": [{"x": 1.0}, {"x": 2.0}],
}
REGRESS_RESPONSE = {"results": [3.5, 4.0]}

ENDPOINT_DESC = {"EndpointConfigName": "test-endpoint"}

ENDPOINT_CONFIG_DESC = {"ProductionVariants": [{"ModelName": "model-1"}, {"ModelName": "model-2"}]}


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    session = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
        s3_resource=None,
        s3_client=None,
        default_bucket_prefix=None,
    )
    session.default_bucket = Mock(name="default_bucket", return_value="my_bucket")
    session.expand_role = Mock(name="expand_role", return_value=ROLE)
    describe = {"ModelArtifacts": {"S3ModelArtifacts": "s3://m/m.tar.gz"}}
    session.sagemaker_client.describe_training_job = Mock(return_value=describe)
    session.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    session.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)
    # For tests which doesn't verify config file injection, operate with empty config
    session.sagemaker_config = {}
    return session


@patch("sagemaker.image_uris.retrieve", return_value=IMAGE)
def test_tfs_model(retrieve_image_uri, sagemaker_session, tensorflow_inference_version):
    model = TensorFlowModel(
        "s3://some/data.tar.gz",
        role=ROLE,
        framework_version=tensorflow_inference_version,
        sagemaker_session=sagemaker_session,
    )
    cdef = model.prepare_container_def(INSTANCE_TYPE)
    retrieve_image_uri.assert_called_with(
        "tensorflow",
        REGION,
        version=tensorflow_inference_version,
        instance_type=INSTANCE_TYPE,
        accelerator_type=None,
        image_scope="inference",
        serverless_inference_config=None,
    )
    assert IMAGE == cdef["Image"]
    assert {} == cdef["Environment"]

    predictor = model.deploy(INSTANCE_COUNT, INSTANCE_TYPE)
    assert isinstance(predictor, TensorFlowPredictor)


@patch("sagemaker.image_uris.retrieve", return_value=IMAGE)
def test_tfs_model_accelerator(retrieve_image_uri, sagemaker_session, tensorflow_eia_version):
    model = TensorFlowModel(
        "s3://some/data.tar.gz",
        role=ROLE,
        framework_version=tensorflow_eia_version,
        sagemaker_session=sagemaker_session,
    )
    cdef = model.prepare_container_def(INSTANCE_TYPE, accelerator_type=ACCELERATOR_TYPE)
    retrieve_image_uri.assert_called_with(
        "tensorflow",
        REGION,
        version=tensorflow_eia_version,
        instance_type=INSTANCE_TYPE,
        accelerator_type=ACCELERATOR_TYPE,
        image_scope="inference",
        serverless_inference_config=None,
    )
    assert IMAGE == cdef["Image"]

    predictor = model.deploy(INSTANCE_COUNT, INSTANCE_TYPE)
    assert isinstance(predictor, TensorFlowPredictor)


def test_tfs_model_image_accelerator_not_supported(sagemaker_session):
    model = TensorFlowModel(
        "s3://some/data.tar.gz",
        role=ROLE,
        framework_version="1.13.0",
        sagemaker_session=sagemaker_session,
    )

    # assert error is not raised

    model.deploy(
        instance_type="ml.c4.xlarge", initial_instance_count=1, accelerator_type="ml.eia1.medium"
    )

    model = TensorFlowModel(
        "s3://some/data.tar.gz",
        role=ROLE,
        framework_version="2.1",
        sagemaker_session=sagemaker_session,
    )

    # assert error is not raised

    model.deploy(instance_type="ml.c4.xlarge", initial_instance_count=1)

    with pytest.raises(AttributeError) as e:
        model.deploy(
            instance_type="ml.c4.xlarge",
            accelerator_type="ml.eia1.medium",
            initial_instance_count=1,
        )

    assert str(e.value) == "The TensorFlow version 2.1 doesn't support EIA."


def test_tfs_model_with_log_level(sagemaker_session, tensorflow_inference_version):
    model = TensorFlowModel(
        "s3://some/data.tar.gz",
        role=ROLE,
        framework_version=tensorflow_inference_version,
        container_log_level=logging.INFO,
        sagemaker_session=sagemaker_session,
    )
    cdef = model.prepare_container_def(INSTANCE_TYPE)
    assert cdef["Environment"] == {TensorFlowModel.LOG_LEVEL_PARAM_NAME: "info"}


def test_tfs_model_with_custom_image(sagemaker_session, tensorflow_inference_version):
    model = TensorFlowModel(
        "s3://some/data.tar.gz",
        role=ROLE,
        framework_version=tensorflow_inference_version,
        image_uri="my-image",
        sagemaker_session=sagemaker_session,
    )
    cdef = model.prepare_container_def(INSTANCE_TYPE)
    assert cdef["Image"] == "my-image"


@mock.patch("sagemaker.fw_utils.model_code_key_prefix", return_value="key-prefix")
@mock.patch("sagemaker.utils.repack_model")
def test_tfs_model_with_entry_point(
    repack_model, model_code_key_prefix, sagemaker_session, tensorflow_inference_version
):
    model = TensorFlowModel(
        "s3://some/data.tar.gz",
        entry_point="train.py",
        role=ROLE,
        framework_version=tensorflow_inference_version,
        image_uri="my-image",
        sagemaker_session=sagemaker_session,
        model_kms_key="kms-key",
    )

    model.prepare_container_def(INSTANCE_TYPE)

    model_code_key_prefix.assert_called_with(model.key_prefix, model.name, model.image_uri)

    repack_model.assert_called_with(
        "train.py",
        None,
        [],
        "s3://some/data.tar.gz",
        "s3://my_bucket/key-prefix/model.tar.gz",
        sagemaker_session,
        kms_key="kms-key",
    )


@mock.patch("sagemaker.fw_utils.model_code_key_prefix", return_value="key-prefix")
@mock.patch("sagemaker.utils.repack_model")
def test_tfs_model_with_source(
    repack_model, model_code_key_prefix, sagemaker_session, tensorflow_inference_version
):
    model = TensorFlowModel(
        "s3://some/data.tar.gz",
        entry_point="train.py",
        source_dir="src",
        role=ROLE,
        framework_version=tensorflow_inference_version,
        image_uri="my-image",
        sagemaker_session=sagemaker_session,
    )

    model.prepare_container_def(INSTANCE_TYPE)

    model_code_key_prefix.assert_called_with(model.key_prefix, model.name, model.image_uri)

    repack_model.assert_called_with(
        "train.py",
        "src",
        [],
        "s3://some/data.tar.gz",
        "s3://my_bucket/key-prefix/model.tar.gz",
        sagemaker_session,
        kms_key=None,
    )


@mock.patch("sagemaker.fw_utils.model_code_key_prefix", return_value="key-prefix")
@mock.patch("sagemaker.utils.repack_model")
def test_tfs_model_with_dependencies(
    repack_model, model_code_key_prefix, sagemaker_session, tensorflow_inference_version
):
    model = TensorFlowModel(
        "s3://some/data.tar.gz",
        entry_point="train.py",
        dependencies=["src", "lib"],
        role=ROLE,
        framework_version=tensorflow_inference_version,
        image_uri="my-image",
        sagemaker_session=sagemaker_session,
    )

    model.prepare_container_def(INSTANCE_TYPE)

    model_code_key_prefix.assert_called_with(model.key_prefix, model.name, model.image_uri)

    repack_model.assert_called_with(
        "train.py",
        None,
        ["src", "lib"],
        "s3://some/data.tar.gz",
        "s3://my_bucket/key-prefix/model.tar.gz",
        sagemaker_session,
        kms_key=None,
    )


def test_model_prepare_container_def_no_instance_type_or_image(tensorflow_inference_version):
    model = TensorFlowModel(
        "s3://some/data.tar.gz", role=ROLE, framework_version=tensorflow_inference_version
    )

    with pytest.raises(ValueError) as e:
        model.prepare_container_def()

    expected_msg = "Must supply either an instance type (for choosing CPU vs GPU) or an image URI."
    assert expected_msg in str(e)


def test_estimator_deploy(sagemaker_session):
    container_log_level = '"logging.INFO"'
    source_dir = "s3://mybucket/source"
    custom_image = "custom:1.0"
    tf = TensorFlow(
        entry_point="script.py",
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        image_uri=custom_image,
        container_log_level=container_log_level,
        base_job_name="job",
        source_dir=source_dir,
    )

    job_name = "doing something"
    tf.fit(inputs="s3://mybucket/train", job_name=job_name)
    predictor = tf.deploy(INSTANCE_COUNT, INSTANCE_TYPE, endpoint_name="endpoint")
    assert isinstance(predictor, TensorFlowPredictor)


def test_predictor(sagemaker_session):
    predictor = TensorFlowPredictor("endpoint", sagemaker_session)

    mock_response(json.dumps(PREDICT_RESPONSE).encode("utf-8"), sagemaker_session)
    result = predictor.predict(PREDICT_INPUT)

    assert_invoked(
        sagemaker_session,
        EndpointName="endpoint",
        ContentType=JSON_CONTENT_TYPE,
        Accept=JSON_CONTENT_TYPE,
        Body=json.dumps(PREDICT_INPUT),
    )

    assert PREDICT_RESPONSE == result


def test_predictor_jsons(sagemaker_session):
    predictor = TensorFlowPredictor(
        "endpoint",
        sagemaker_session,
        serializer=IdentitySerializer(content_type="application/jsons"),
    )

    mock_response(json.dumps(PREDICT_RESPONSE).encode("utf-8"), sagemaker_session)
    result = predictor.predict("[1.0, 2.0, 3.0]\n[4.0, 5.0, 6.0]")

    assert_invoked(
        sagemaker_session,
        EndpointName="endpoint",
        ContentType="application/jsons",
        Accept=JSON_CONTENT_TYPE,
        Body="[1.0, 2.0, 3.0]\n[4.0, 5.0, 6.0]",
    )

    assert PREDICT_RESPONSE == result


def test_predictor_csv(sagemaker_session):
    predictor = TensorFlowPredictor("endpoint", sagemaker_session, serializer=CSVSerializer())

    mock_response(json.dumps(PREDICT_RESPONSE).encode("utf-8"), sagemaker_session)
    result = predictor.predict([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert_invoked(
        sagemaker_session,
        EndpointName="endpoint",
        ContentType=CSV_CONTENT_TYPE,
        Accept=JSON_CONTENT_TYPE,
        Body="1.0,2.0,3.0\n4.0,5.0,6.0",
    )

    assert PREDICT_RESPONSE == result


def test_predictor_model_attributes(sagemaker_session):
    predictor = TensorFlowPredictor(
        "endpoint", sagemaker_session, model_name="model", model_version="123"
    )

    mock_response(json.dumps(PREDICT_RESPONSE).encode("utf-8"), sagemaker_session)
    result = predictor.predict(PREDICT_INPUT)

    assert_invoked(
        sagemaker_session,
        EndpointName="endpoint",
        ContentType=JSON_CONTENT_TYPE,
        Accept=JSON_CONTENT_TYPE,
        CustomAttributes="tfs-model-name=model,tfs-model-version=123",
        Body=json.dumps(PREDICT_INPUT),
    )

    assert PREDICT_RESPONSE == result


def test_predictor_classify(sagemaker_session):
    predictor = TensorFlowPredictor("endpoint", sagemaker_session)

    mock_response(json.dumps(CLASSIFY_RESPONSE).encode("utf-8"), sagemaker_session)
    result = predictor.classify(CLASSIFY_INPUT)

    assert_invoked_with_body_dict(
        sagemaker_session,
        EndpointName="endpoint",
        ContentType=JSON_CONTENT_TYPE,
        Accept=JSON_CONTENT_TYPE,
        CustomAttributes="tfs-method=classify",
        Body=json.dumps(CLASSIFY_INPUT),
    )

    assert CLASSIFY_RESPONSE == result


def test_predictor_regress(sagemaker_session):
    predictor = TensorFlowPredictor(
        "endpoint", sagemaker_session, model_name="model", model_version="123"
    )

    mock_response(json.dumps(REGRESS_RESPONSE).encode("utf-8"), sagemaker_session)
    result = predictor.regress(REGRESS_INPUT)

    assert_invoked_with_body_dict(
        sagemaker_session,
        EndpointName="endpoint",
        ContentType=JSON_CONTENT_TYPE,
        Accept=JSON_CONTENT_TYPE,
        CustomAttributes="tfs-method=regress,tfs-model-name=model,tfs-model-version=123",
        Body=json.dumps(REGRESS_INPUT),
    )

    assert REGRESS_RESPONSE == result


def test_predictor_regress_bad_content_type(sagemaker_session):
    predictor = TensorFlowPredictor("endpoint", sagemaker_session, CSVSerializer())

    with pytest.raises(ValueError):
        predictor.regress(REGRESS_INPUT)


def test_predictor_classify_bad_content_type(sagemaker_session):
    predictor = TensorFlowPredictor("endpoint", sagemaker_session, CSVSerializer())

    with pytest.raises(ValueError):
        predictor.classify(CLASSIFY_INPUT)


def assert_invoked(sagemaker_session, **kwargs):
    sagemaker_session.sagemaker_runtime_client.invoke_endpoint.assert_called_once_with(**kwargs)


def assert_invoked_with_body_dict(sagemaker_session, **kwargs):
    call = sagemaker_session.sagemaker_runtime_client.invoke_endpoint.call_args
    cargs, ckwargs = call
    assert not cargs
    assert len(kwargs) == len(ckwargs)
    for k in ckwargs:
        if k != "Body":
            assert kwargs[k] == ckwargs[k]
        else:
            actual_body = json.loads(ckwargs[k])
            expected_body = json.loads(kwargs[k])
            assert len(actual_body) == len(expected_body)
            for k2 in actual_body:
                assert actual_body[k2] == expected_body[k2]


def mock_response(expected_response, sagemaker_session, content_type=JSON_CONTENT_TYPE):
    sagemaker_session.sagemaker_runtime_client.invoke_endpoint.return_value = {
        "ContentType": content_type,
        "Body": io.BytesIO(expected_response),
    }


def test_register_tfs_model_auto_infer_framework(sagemaker_session, tensorflow_inference_version):
    model_package_group_name = "test-tfs-register-model"
    content_types = ["application/json"]
    response_types = ["application/json"]
    inference_instances = ["ml.m4.xlarge"]
    transform_instances = ["ml.m4.xlarge"]
    image_uri = "fakeimage"

    tfs_model = TensorFlowModel(
        "s3://some/data.tar.gz",
        role=ROLE,
        framework_version=tensorflow_inference_version,
        sagemaker_session=sagemaker_session,
    )

    tfs_model.register(
        content_types,
        response_types,
        inference_instances,
        transform_instances,
        model_package_group_name=model_package_group_name,
        marketplace_cert=True,
        image_uri=image_uri,
    )

    expected_create_model_package_request = {
        "containers": [
            {
                "Image": image_uri,
                "Environment": ANY,
                "ModelDataUrl": ANY,
                "Framework": "TENSORFLOW",
                "FrameworkVersion": tensorflow_inference_version,
            },
        ],
        "content_types": content_types,
        "response_types": response_types,
        "inference_instances": inference_instances,
        "transform_instances": transform_instances,
        "model_package_group_name": model_package_group_name,
        "marketplace_cert": True,
    }

    sagemaker_session.create_model_package_from_containers.assert_called_with(
        **expected_create_model_package_request
    )
