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

import botocore.exceptions
import pytest

import sagemaker
import sagemaker.predictor
import sagemaker.utils
from sagemaker.tensorflow.serving import Model, Predictor
from tests.integ.timeout import timeout_and_delete_endpoint_by_name


@pytest.fixture(scope='session', params=['ml.c5.xlarge', 'ml.p3.2xlarge'])
def instance_type(request):
    return request.param


@pytest.fixture(scope='module')
def tfs_predictor(instance_type, sagemaker_session, tf_full_version):
    endpoint_name = sagemaker.utils.name_from_base('sagemaker-tensorflow-serving')
    model_data = sagemaker_session.upload_data(
        path='tests/data/tensorflow-serving-test-model.tar.gz',
        key_prefix='tensorflow-serving/models')
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        model = Model(model_data=model_data, role='SageMakerRole',
                      framework_version=tf_full_version,
                      sagemaker_session=sagemaker_session)
        predictor = model.deploy(1, instance_type, endpoint_name=endpoint_name)
        yield predictor


# @pytest.mark.continuous_testing
# @pytest.mark.regional_testing
def test_predict(tfs_predictor):
    input_data = {'instances': [1.0, 2.0, 5.0]}
    expected_result = {'predictions': [3.5, 4.0, 5.5]}

    result = tfs_predictor.predict(input_data)
    assert expected_result == result


def test_predict_generic_json(tfs_predictor):
    input_data = [[1.0, 2.0, 5.0], [1.0, 2.0, 5.0]]
    expected_result = {'predictions': [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}

    result = tfs_predictor.predict(input_data)
    assert expected_result == result


def test_predict_jsons(tfs_predictor):
    input_data = '[1.0, 2.0, 5.0]\n[1.0, 2.0, 5.0]'
    expected_result = {'predictions': [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}

    predictor = sagemaker.RealTimePredictor(tfs_predictor.endpoint,
                                            tfs_predictor.sagemaker_session, serializer=None,
                                            deserializer=sagemaker.predictor.json_deserializer,
                                            content_type='application/json',
                                            accept='application/json')

    result = predictor.predict(input_data)
    assert expected_result == result


def test_predict_csv(tfs_predictor):
    input_data = '1.0,2.0,5.0\n1.0,2.0,5.0'
    expected_result = {'predictions': [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}

    predictor = Predictor(tfs_predictor.endpoint, tfs_predictor.sagemaker_session,
                          serializer=sagemaker.predictor.csv_serializer)

    result = predictor.predict(input_data)
    assert expected_result == result


def test_predict_bad_input(tfs_predictor):
    input_data = {'junk': 'data'}
    with pytest.raises(botocore.exceptions.ClientError):
        tfs_predictor.predict(input_data)
