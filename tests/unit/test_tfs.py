# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import pytest
from mock import Mock

from sagemaker.tensorflow import TensorFlow
from sagemaker.tensorflow.predictor import csv_serializer
from sagemaker.tensorflow.serving import Model, Predictor

JSON_CONTENT_TYPE = 'application/json'
CSV_CONTENT_TYPE = 'text/csv'
INSTANCE_COUNT = 1
INSTANCE_TYPE = 'ml.c4.4xlarge'
ROLE = 'Dummy'
REGION = 'us-west-2'
PREDICT_INPUT = {'instances': [1.0, 2.0, 5.0]}
PREDICT_RESPONSE = {'predictions': [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}
CLASSIFY_INPUT = {
    'signature_name': 'tensorflow/serving/classify',
    'examples': [{'x': 1.0}, {'x': 2.0}]
}
CLASSIFY_RESPONSE = {'result': [[0.4, 0.6], [0.2, 0.8]]}
REGRESS_INPUT = {
    'signature_name': 'tensorflow/serving/regress',
    'examples': [{'x': 1.0}, {'x': 2.0}]
}
REGRESS_RESPONSE = {'results': [3.5, 4.0]}


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name='boto_session', region_name=REGION)
    session = Mock(name='sagemaker_session', boto_session=boto_mock,
                   boto_region_name=REGION, config=None, local_mode=False)
    session.default_bucket = Mock(name='default_bucket', return_value='my_bucket')
    session.expand_role = Mock(name="expand_role", return_value=ROLE)
    describe = {'ModelArtifacts': {'S3ModelArtifacts': 's3://m/m.tar.gz'}}
    session.sagemaker_client.describe_training_job = Mock(return_value=describe)
    return session


def test_tfs_model(sagemaker_session, tf_version):
    model = Model("s3://some/data.tar.gz", role=ROLE, framework_version=tf_version,
                  sagemaker_session=sagemaker_session)
    cdef = model.prepare_container_def(INSTANCE_TYPE)
    assert cdef['Image'].endswith('sagemaker-tensorflow-serving:{}-cpu'.format(tf_version))
    assert cdef['Environment'] == {}

    predictor = model.deploy(INSTANCE_COUNT, INSTANCE_TYPE)
    assert isinstance(predictor, Predictor)


def test_tfs_model_with_log_level(sagemaker_session, tf_version):
    model = Model("s3://some/data.tar.gz", role=ROLE, framework_version=tf_version,
                  container_log_level=logging.INFO,
                  sagemaker_session=sagemaker_session)
    cdef = model.prepare_container_def(INSTANCE_TYPE)
    assert cdef['Environment'] == {Model.LOG_LEVEL_PARAM_NAME: 'info'}


def test_tfs_model_with_custom_image(sagemaker_session, tf_version):
    model = Model("s3://some/data.tar.gz", role=ROLE, framework_version=tf_version,
                  image='my-image',
                  sagemaker_session=sagemaker_session)
    cdef = model.prepare_container_def(INSTANCE_TYPE)
    assert cdef['Image'] == 'my-image'


def test_estimator_deploy(sagemaker_session):
    container_log_level = '"logging.INFO"'
    source_dir = 's3://mybucket/source'
    custom_image = 'custom:1.0'
    tf = TensorFlow(entry_point='script.py', role=ROLE, sagemaker_session=sagemaker_session,
                    training_steps=1000, evaluation_steps=10, train_instance_count=INSTANCE_COUNT,
                    train_instance_type=INSTANCE_TYPE, image_name=custom_image,
                    container_log_level=container_log_level, base_job_name='job',
                    source_dir=source_dir)

    job_name = 'doing something'
    tf.fit(inputs='s3://mybucket/train', job_name=job_name)
    predictor = tf.deploy(INSTANCE_COUNT, INSTANCE_TYPE, 'endpoint',
                          endpoint_type='tensorflow-serving')
    assert isinstance(predictor, Predictor)


def test_predictor(sagemaker_session):
    predictor = Predictor('endpoint', sagemaker_session)

    mock_response(json.dumps(PREDICT_RESPONSE).encode('utf-8'), sagemaker_session)
    result = predictor.predict(PREDICT_INPUT)

    assert_invoked(sagemaker_session,
                   EndpointName='endpoint',
                   ContentType=JSON_CONTENT_TYPE,
                   Accept=JSON_CONTENT_TYPE,
                   Body=json.dumps(PREDICT_INPUT))

    assert PREDICT_RESPONSE == result


def test_predictor_csv(sagemaker_session):
    predictor = Predictor('endpoint', sagemaker_session, serializer=csv_serializer)

    mock_response(json.dumps(PREDICT_RESPONSE).encode('utf-8'), sagemaker_session)
    result = predictor.predict([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert_invoked(sagemaker_session,
                   EndpointName='endpoint',
                   ContentType=CSV_CONTENT_TYPE,
                   Accept=JSON_CONTENT_TYPE,
                   Body='1.0,2.0,3.0\n4.0,5.0,6.0')

    assert PREDICT_RESPONSE == result


def test_predictor_model_attributes(sagemaker_session):
    predictor = Predictor('endpoint', sagemaker_session, model_name='model', model_version='123')

    mock_response(json.dumps(PREDICT_RESPONSE).encode('utf-8'), sagemaker_session)
    result = predictor.predict(PREDICT_INPUT)

    assert_invoked(sagemaker_session,
                   EndpointName='endpoint',
                   ContentType=JSON_CONTENT_TYPE,
                   Accept=JSON_CONTENT_TYPE,
                   CustomAttributes='tfs-model-name=model,tfs-model-version=123',
                   Body=json.dumps(PREDICT_INPUT))

    assert PREDICT_RESPONSE == result


def test_predictor_classify(sagemaker_session):
    predictor = Predictor('endpoint', sagemaker_session)

    mock_response(json.dumps(CLASSIFY_RESPONSE).encode('utf-8'), sagemaker_session)
    result = predictor.classify(CLASSIFY_INPUT)

    assert_invoked(sagemaker_session,
                   EndpointName='endpoint',
                   ContentType=JSON_CONTENT_TYPE,
                   Accept=JSON_CONTENT_TYPE,
                   CustomAttributes='tfs-method=classify',
                   Body=json.dumps(CLASSIFY_INPUT))

    assert CLASSIFY_RESPONSE == result


def test_predictor_regress(sagemaker_session):
    predictor = Predictor('endpoint', sagemaker_session, model_name='model', model_version='123')

    mock_response(json.dumps(REGRESS_RESPONSE).encode('utf-8'), sagemaker_session)
    result = predictor.regress(REGRESS_INPUT)

    assert_invoked(sagemaker_session,
                   EndpointName='endpoint',
                   ContentType=JSON_CONTENT_TYPE,
                   Accept=JSON_CONTENT_TYPE,
                   CustomAttributes='tfs-method=regress,tfs-model-name=model,tfs-model-version=123',
                   Body=json.dumps(REGRESS_INPUT))

    assert REGRESS_RESPONSE == result


def test_predictor_regress_bad_content_type():
    predictor = Predictor('endpoint', sagemaker_session, csv_serializer)

    with pytest.raises(ValueError):
        predictor.regress(REGRESS_INPUT)


def test_predictor_classify_bad_content_type():
    predictor = Predictor('endpoint', sagemaker_session, csv_serializer)

    with pytest.raises(ValueError):
        predictor.classify(CLASSIFY_INPUT)


def assert_invoked(sagemaker_session, **kwargs):
    call = sagemaker_session.sagemaker_runtime_client.invoke_endpoint.call_args
    cargs, ckwargs = call
    assert not cargs
    assert len(kwargs) == len(ckwargs)
    for k in ckwargs:
        assert kwargs[k] == ckwargs[k]


def mock_response(expected_response, sagemaker_session, content_type=JSON_CONTENT_TYPE):
    sagemaker_session.sagemaker_runtime_client.invoke_endpoint.return_value = {
        'ContentType': content_type,
        'Body': io.BytesIO(expected_response)
    }
