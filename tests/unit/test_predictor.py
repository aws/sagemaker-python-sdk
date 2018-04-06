# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import io
import json
import os
import pytest
from mock import Mock

import numpy as np

from sagemaker.predictor import RealTimePredictor
from sagemaker.predictor import json_serializer, json_deserializer, csv_serializer, BytesDeserializer, \
    StringDeserializer, StreamDeserializer, NumpyDeserializer
from tests.unit import DATA_DIR

# testing serialization functions


def test_json_serializer_numpy_valid():
    result = json_serializer(np.array([1, 2, 3]))

    assert result == '[1, 2, 3]'


def test_json_serializer_numpy_valid_2dimensional():
    result = json_serializer(np.array([[1, 2, 3], [3, 4, 5]]))

    assert result == '[[1, 2, 3], [3, 4, 5]]'


def test_json_serializer_numpy_invalid_empty():
    with pytest.raises(ValueError) as invalid_input:
        json_serializer(np.array([]))

    assert "empty array" in str(invalid_input)


def test_json_serializer_python_array():
    result = json_serializer([1, 2, 3])

    assert result == '[1, 2, 3]'


def test_json_serializer_python_dictionary():
    d = {"gender": "m", "age": 22, "city": "Paris"}

    result = json_serializer(d)

    assert json.loads(result) == d


def test_json_serializer_python_invalid_empty():
    with pytest.raises(ValueError) as error:
        json_serializer([])
    assert "empty array" in str(error)


def test_json_serializer_python_dictionary_invalid_empty():
    with pytest.raises(ValueError) as error:
        json_serializer({})
    assert "empty dictionary" in str(error)


def test_json_serializer_csv_buffer():
    csv_file_path = os.path.join(DATA_DIR, "with_integers.csv")
    with open(csv_file_path) as csv_file:
        validation_value = csv_file.read()
        csv_file.seek(0)
        result = json_serializer(csv_file)
        assert result == validation_value


def test_csv_serializer_str():
    original = '1,2,3'
    result = csv_serializer('1,2,3')

    assert result == original


def test_csv_serializer_python_array():
    result = csv_serializer([1, 2, 3])

    assert result == '1,2,3'


def test_csv_serializer_numpy_valid():
    result = csv_serializer(np.array([1, 2, 3]))

    assert result == '1,2,3'


def test_csv_serializer_numpy_valid_2dimensional():
    result = csv_serializer(np.array([[1, 2, 3], [3, 4, 5]]))

    assert result == '1,2,3\n3,4,5'


def test_csv_serializer_list_of_str():
    result = csv_serializer(['1,2,3', '4,5,6'])

    assert result == '1,2,3\n4,5,6'


def test_csv_serializer_list_of_list():
    result = csv_serializer([[1, 2, 3], [3, 4, 5]])

    assert result == '1,2,3\n3,4,5'


def test_csv_serializer_list_of_empty():
    with pytest.raises(ValueError) as invalid_input:
        csv_serializer(np.array([[], []]))

    assert "empty array" in str(invalid_input)


def test_csv_serializer_numpy_invalid_empty():
    with pytest.raises(ValueError) as invalid_input:
        csv_serializer(np.array([]))

    assert "empty array" in str(invalid_input)


def test_csv_serializer_python_invalid_empty():
    with pytest.raises(ValueError) as error:
        csv_serializer([])
    assert "empty array" in str(error)


def test_csv_serializer_csv_reader():
    csv_file_path = os.path.join(DATA_DIR, "with_integers.csv")
    with open(csv_file_path) as csv_file:
        validation_data = csv_file.read()
        csv_file.seek(0)
        result = csv_serializer(csv_file)
        assert result == validation_data


def test_json_deserializer_array():
    result = json_deserializer(io.BytesIO(b'[1, 2, 3]'), 'application/json')

    assert result == [1, 2, 3]


def test_json_deserializer_2dimensional():
    result = json_deserializer(io.BytesIO(b'[[1, 2, 3], [3, 4, 5]]'), 'application/json')

    assert result == [[1, 2, 3], [3, 4, 5]]


def test_json_deserializer_invalid_data():
    with pytest.raises(ValueError) as error:
        json_deserializer(io.BytesIO(b'[[1]'), 'application/json')
    assert "column" in str(error)


def test_bytes_deserializer():
    result = BytesDeserializer()(io.BytesIO(b'[1, 2, 3]'), 'application/json')

    assert result == b'[1, 2, 3]'


def test_string_deserializer():
    result = StringDeserializer()(io.BytesIO(b'[1, 2, 3]'), 'application/json')

    assert result == '[1, 2, 3]'


def test_stream_deserializer():
    stream, content_type = StreamDeserializer()(io.BytesIO(b'[1, 2, 3]'), 'application/json')
    result = stream.read()
    assert result == b'[1, 2, 3]'
    assert content_type == 'application/json'


def test_numpy_deser_from_csv():
    arr = NumpyDeserializer()(io.BytesIO(b'1,2,3\n4,5,6'), 'text/csv')
    assert np.array_equal(arr, np.array([[1, 2, 3], [4, 5, 6]]))


def test_numpy_deser_from_csv_ragged():
    with pytest.raises(ValueError) as error:
        NumpyDeserializer()(io.BytesIO(b'1,2,3\n4,5,6,7'), 'text/csv')
    assert "errors were detected" in str(error)


def test_numpy_deser_from_csv_alpha():
    arr = NumpyDeserializer(dtype='U5')(io.BytesIO(b'hello,2,3\n4,5,6'), 'text/csv')
    assert np.array_equal(arr, np.array([['hello', 2, 3], [4, 5, 6]]))


def test_numpy_deser_from_json():
    arr = NumpyDeserializer()(io.BytesIO(b'[[1,2,3],\n[4,5,6]]'), 'application/json')
    assert np.array_equal(arr, np.array([[1, 2, 3], [4, 5, 6]]))


# Sadly, ragged arrays work fine in JSON (giving us a 1D array of Python lists
def test_numpy_deser_from_json_ragged():
    arr = NumpyDeserializer()(io.BytesIO(b'[[1,2,3],\n[4,5,6,7]]'), 'application/json')
    assert np.array_equal(arr, np.array([[1, 2, 3], [4, 5, 6, 7]]))


def test_numpy_deser_from_json_alpha():
    arr = NumpyDeserializer(dtype='U5')(io.BytesIO(b'[["hello",2,3],\n[4,5,6]]'), 'application/json')
    assert np.array_equal(arr, np.array([['hello', 2, 3], [4, 5, 6]]))


# testing 'predict' invocations

ENDPOINT = 'mxnet_endpoint'
BUCKET_NAME = 'mxnet_endpoint'
DEFAULT_CONTENT_TYPE = 'application/json'
CSV_CONTENT_TYPE = 'text/csv'
RETURN_VALUE = 0
CSV_RETURN_VALUE = "1,2,3\r\n"


def empty_sagemaker_session():
    ims = Mock(name='sagemaker_session')
    ims.default_bucket = Mock(name='default_bucket', return_value=BUCKET_NAME)
    ims.sagemaker_runtime_client = Mock(name='sagemaker_runtime')

    response_body = Mock('body')
    response_body.read = Mock('read', return_value=RETURN_VALUE)
    response_body.close = Mock('close', return_value=None)
    ims.sagemaker_runtime_client.invoke_endpoint = Mock(name='invoke_endpoint', return_value={'Body': response_body})
    return ims


def test_predict_call_pass_through():
    sagemaker_session = empty_sagemaker_session()
    predictor = RealTimePredictor(ENDPOINT, sagemaker_session)

    data = "untouched"
    result = predictor.predict(data)

    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint.called

    expected_request_args = {
        'Body': data,
        'EndpointName': ENDPOINT
    }
    call_args, kwargs = sagemaker_session.sagemaker_runtime_client.invoke_endpoint.call_args
    assert kwargs == expected_request_args

    assert result == RETURN_VALUE


def test_predict_call_with_headers():
    sagemaker_session = empty_sagemaker_session()
    predictor = RealTimePredictor(ENDPOINT, sagemaker_session,
                                  content_type=DEFAULT_CONTENT_TYPE,
                                  accept=DEFAULT_CONTENT_TYPE)

    data = "untouched"
    result = predictor.predict(data)

    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint.called

    expected_request_args = {
        'Accept': DEFAULT_CONTENT_TYPE,
        'Body': data,
        'ContentType': DEFAULT_CONTENT_TYPE,
        'EndpointName': ENDPOINT
    }
    call_args, kwargs = sagemaker_session.sagemaker_runtime_client.invoke_endpoint.call_args
    assert kwargs == expected_request_args

    assert result == RETURN_VALUE


def json_sagemaker_session():
    ims = Mock(name='sagemaker_session')
    ims.default_bucket = Mock(name='default_bucket', return_value=BUCKET_NAME)
    ims.sagemaker_runtime_client = Mock(name='sagemaker_runtime')

    response_body = Mock('body')
    response_body.read = Mock('read', return_value=json.dumps([RETURN_VALUE]))
    response_body.close = Mock('close', return_value=None)
    ims.sagemaker_runtime_client.invoke_endpoint = Mock(name='invoke_endpoint',
                                                        return_value={'Body': response_body,
                                                                      'ContentType': DEFAULT_CONTENT_TYPE})
    return ims


def test_predict_call_with_headers_and_json():
    sagemaker_session = json_sagemaker_session()
    predictor = RealTimePredictor(ENDPOINT, sagemaker_session,
                                  content_type='not/json',
                                  accept='also/not-json',
                                  serializer=json_serializer)

    data = [1, 2]
    result = predictor.predict(data)

    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint.called

    expected_request_args = {
        'Accept': 'also/not-json',
        'Body': json.dumps(data),
        'ContentType': 'not/json',
        'EndpointName': ENDPOINT
    }
    call_args, kwargs = sagemaker_session.sagemaker_runtime_client.invoke_endpoint.call_args
    assert kwargs == expected_request_args

    assert result == json.dumps([RETURN_VALUE])


def ret_csv_sagemaker_session():
    ims = Mock(name='sagemaker_session')
    ims.default_bucket = Mock(name='default_bucket', return_value=BUCKET_NAME)
    ims.sagemaker_runtime_client = Mock(name='sagemaker_runtime')

    response_body = Mock('body')
    response_body.read = Mock('read', return_value=CSV_RETURN_VALUE)
    response_body.close = Mock('close', return_value=None)
    ims.sagemaker_runtime_client.invoke_endpoint = Mock(name='invoke_endpoint',
                                                        return_value={'Body': response_body,
                                                                      'ContentType': CSV_CONTENT_TYPE})
    return ims


def test_predict_call_with_headers_and_csv():
    sagemaker_session = ret_csv_sagemaker_session()
    predictor = RealTimePredictor(ENDPOINT, sagemaker_session,
                                  accept=CSV_CONTENT_TYPE,
                                  serializer=csv_serializer)

    data = [1, 2]
    result = predictor.predict(data)

    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint.called

    expected_request_args = {
        'Accept': CSV_CONTENT_TYPE,
        'Body': '1,2',
        'ContentType': CSV_CONTENT_TYPE,
        'EndpointName': ENDPOINT
    }
    call_args, kwargs = sagemaker_session.sagemaker_runtime_client.invoke_endpoint.call_args
    assert kwargs == expected_request_args

    assert result == CSV_RETURN_VALUE
