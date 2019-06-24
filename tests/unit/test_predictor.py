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
import os
import pytest
from mock import Mock, call

import numpy as np

from sagemaker.predictor import RealTimePredictor
from sagemaker.predictor import (
    json_serializer,
    json_deserializer,
    csv_serializer,
    csv_deserializer,
    BytesDeserializer,
    StringDeserializer,
    StreamDeserializer,
    numpy_deserializer,
    npy_serializer,
    _NumpyDeserializer,
)
from tests.unit import DATA_DIR

# testing serialization functions


def test_json_serializer_numpy_valid():
    result = json_serializer(np.array([1, 2, 3]))

    assert result == "[1, 2, 3]"


def test_json_serializer_numpy_valid_2dimensional():
    result = json_serializer(np.array([[1, 2, 3], [3, 4, 5]]))

    assert result == "[[1, 2, 3], [3, 4, 5]]"


def test_json_serializer_empty():
    assert json_serializer(np.array([])) == "[]"


def test_json_serializer_python_array():
    result = json_serializer([1, 2, 3])

    assert result == "[1, 2, 3]"


def test_json_serializer_python_dictionary():
    d = {"gender": "m", "age": 22, "city": "Paris"}

    result = json_serializer(d)

    assert json.loads(result) == d


def test_json_serializer_python_invalid_empty():
    assert json_serializer([]) == "[]"


def test_json_serializer_python_dictionary_invalid_empty():
    assert json_serializer({}) == "{}"


def test_json_serializer_csv_buffer():
    csv_file_path = os.path.join(DATA_DIR, "with_integers.csv")
    with open(csv_file_path) as csv_file:
        validation_value = csv_file.read()
        csv_file.seek(0)
        result = json_serializer(csv_file)
        assert result == validation_value


def test_csv_serializer_str():
    original = "1,2,3"
    result = csv_serializer("1,2,3")

    assert result == original


def test_csv_serializer_python_array():
    result = csv_serializer([1, 2, 3])

    assert result == "1,2,3"


def test_csv_serializer_numpy_valid():
    result = csv_serializer(np.array([1, 2, 3]))

    assert result == "1,2,3"


def test_csv_serializer_numpy_valid_2dimensional():
    result = csv_serializer(np.array([[1, 2, 3], [3, 4, 5]]))

    assert result == "1,2,3\n3,4,5"


def test_csv_serializer_list_of_str():
    result = csv_serializer(["1,2,3", "4,5,6"])

    assert result == "1,2,3\n4,5,6"


def test_csv_serializer_list_of_list():
    result = csv_serializer([[1, 2, 3], [3, 4, 5]])

    assert result == "1,2,3\n3,4,5"


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


def test_csv_deserializer_single_element():
    result = csv_deserializer(io.BytesIO(b"1"), "text/csv")
    assert result == [["1"]]


def test_csv_deserializer_array():
    result = csv_deserializer(io.BytesIO(b"1,2,3"), "text/csv")
    assert result == [["1", "2", "3"]]


def test_csv_deserializer_2dimensional():
    result = csv_deserializer(io.BytesIO(b"1,2,3\n3,4,5"), "text/csv")
    assert result == [["1", "2", "3"], ["3", "4", "5"]]


def test_json_deserializer_array():
    result = json_deserializer(io.BytesIO(b"[1, 2, 3]"), "application/json")

    assert result == [1, 2, 3]


def test_json_deserializer_2dimensional():
    result = json_deserializer(io.BytesIO(b"[[1, 2, 3], [3, 4, 5]]"), "application/json")

    assert result == [[1, 2, 3], [3, 4, 5]]


def test_json_deserializer_invalid_data():
    with pytest.raises(ValueError) as error:
        json_deserializer(io.BytesIO(b"[[1]"), "application/json")
    assert "column" in str(error)


def test_bytes_deserializer():
    result = BytesDeserializer()(io.BytesIO(b"[1, 2, 3]"), "application/json")

    assert result == b"[1, 2, 3]"


def test_string_deserializer():
    result = StringDeserializer()(io.BytesIO(b"[1, 2, 3]"), "application/json")

    assert result == "[1, 2, 3]"


def test_stream_deserializer():
    stream, content_type = StreamDeserializer()(io.BytesIO(b"[1, 2, 3]"), "application/json")
    result = stream.read()
    assert result == b"[1, 2, 3]"
    assert content_type == "application/json"


def test_npy_serializer_python_array():
    array = [1, 2, 3]
    result = npy_serializer(array)

    assert np.array_equal(array, np.load(io.BytesIO(result)))


def test_npy_serializer_python_array_with_dtype():
    array = [1, 2, 3]
    dtype = "float16"

    result = npy_serializer(array, dtype)

    deserialized = np.load(io.BytesIO(result))
    assert np.array_equal(array, deserialized)
    assert deserialized.dtype == dtype


def test_npy_serializer_numpy_valid_2_dimensional():
    array = np.array([[1, 2, 3], [3, 4, 5]])
    result = npy_serializer(array)

    assert np.array_equal(array, np.load(io.BytesIO(result)))


def test_npy_serializer_numpy_valid_multidimensional():
    array = np.ones((10, 10, 10, 10))
    result = npy_serializer(array)

    assert np.array_equal(array, np.load(io.BytesIO(result)))


def test_npy_serializer_numpy_valid_list_of_strings():
    array = np.array(["one", "two", "three"])
    result = npy_serializer(array)

    assert np.array_equal(array, np.load(io.BytesIO(result)))


def test_npy_serializer_from_buffer_or_file():
    array = np.ones((2, 3))
    stream = io.BytesIO()
    np.save(stream, array)
    stream.seek(0)

    result = npy_serializer(stream)

    assert np.array_equal(array, np.load(io.BytesIO(result)))


def test_npy_serializer_object():
    object = {1, 2, 3}

    result = npy_serializer(object)

    assert np.array_equal(np.array(object), np.load(io.BytesIO(result), allow_pickle=True))


def test_npy_serializer_list_of_empty():
    with pytest.raises(ValueError) as invalid_input:
        npy_serializer(np.array([[], []]))

    assert "empty array" in str(invalid_input)


def test_npy_serializer_numpy_invalid_empty():
    with pytest.raises(ValueError) as invalid_input:
        npy_serializer(np.array([]))

    assert "empty array" in str(invalid_input)


def test_npy_serializer_python_invalid_empty():
    with pytest.raises(ValueError) as error:
        npy_serializer([])
    assert "empty array" in str(error)


def test_numpy_deser_from_csv():
    arr = numpy_deserializer(io.BytesIO(b"1,2,3\n4,5,6"), "text/csv")
    assert np.array_equal(arr, np.array([[1, 2, 3], [4, 5, 6]]))


def test_numpy_deser_from_csv_ragged():
    with pytest.raises(ValueError) as error:
        numpy_deserializer(io.BytesIO(b"1,2,3\n4,5,6,7"), "text/csv")
    assert "errors were detected" in str(error)


def test_numpy_deser_from_csv_alpha():
    arr = _NumpyDeserializer(dtype="U5")(io.BytesIO(b"hello,2,3\n4,5,6"), "text/csv")
    assert np.array_equal(arr, np.array([["hello", 2, 3], [4, 5, 6]]))


def test_numpy_deser_from_json():
    arr = numpy_deserializer(io.BytesIO(b"[[1,2,3],\n[4,5,6]]"), "application/json")
    assert np.array_equal(arr, np.array([[1, 2, 3], [4, 5, 6]]))


# Sadly, ragged arrays work fine in JSON (giving us a 1D array of Python lists
def test_numpy_deser_from_json_ragged():
    arr = numpy_deserializer(io.BytesIO(b"[[1,2,3],\n[4,5,6,7]]"), "application/json")
    assert np.array_equal(arr, np.array([[1, 2, 3], [4, 5, 6, 7]]))


def test_numpy_deser_from_json_alpha():
    arr = _NumpyDeserializer(dtype="U5")(
        io.BytesIO(b'[["hello",2,3],\n[4,5,6]]'), "application/json"
    )
    assert np.array_equal(arr, np.array([["hello", 2, 3], [4, 5, 6]]))


def test_numpy_deser_from_npy():
    array = np.ones((2, 3))
    stream = io.BytesIO()
    np.save(stream, array)
    stream.seek(0)

    result = numpy_deserializer(stream)

    assert np.array_equal(array, result)


def test_numpy_deser_from_npy_object_array():
    array = np.array(["one", "two"])
    stream = io.BytesIO()
    np.save(stream, array)
    stream.seek(0)

    result = numpy_deserializer(stream)

    assert np.array_equal(array, result)


# testing 'predict' invocations


ENDPOINT = "mxnet_endpoint"
BUCKET_NAME = "mxnet_endpoint"
DEFAULT_CONTENT_TYPE = "application/json"
CSV_CONTENT_TYPE = "text/csv"
RETURN_VALUE = 0
CSV_RETURN_VALUE = "1,2,3\r\n"

ENDPOINT_DESC = {"EndpointConfigName": ENDPOINT}

ENDPOINT_CONFIG_DESC = {"ProductionVariants": [{"ModelName": "model-1"}, {"ModelName": "model-2"}]}


def empty_sagemaker_session():
    ims = Mock(name="sagemaker_session")
    ims.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    ims.sagemaker_runtime_client = Mock(name="sagemaker_runtime")
    ims.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    ims.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)

    response_body = Mock("body")
    response_body.read = Mock("read", return_value=RETURN_VALUE)
    response_body.close = Mock("close", return_value=None)
    ims.sagemaker_runtime_client.invoke_endpoint = Mock(
        name="invoke_endpoint", return_value={"Body": response_body}
    )
    return ims


def test_predict_call_pass_through():
    sagemaker_session = empty_sagemaker_session()
    predictor = RealTimePredictor(ENDPOINT, sagemaker_session)

    data = "untouched"
    result = predictor.predict(data)

    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint.called

    expected_request_args = {"Body": data, "EndpointName": ENDPOINT}
    call_args, kwargs = sagemaker_session.sagemaker_runtime_client.invoke_endpoint.call_args
    assert kwargs == expected_request_args

    assert result == RETURN_VALUE


def test_predict_call_with_headers():
    sagemaker_session = empty_sagemaker_session()
    predictor = RealTimePredictor(
        ENDPOINT, sagemaker_session, content_type=DEFAULT_CONTENT_TYPE, accept=DEFAULT_CONTENT_TYPE
    )

    data = "untouched"
    result = predictor.predict(data)

    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint.called

    expected_request_args = {
        "Accept": DEFAULT_CONTENT_TYPE,
        "Body": data,
        "ContentType": DEFAULT_CONTENT_TYPE,
        "EndpointName": ENDPOINT,
    }
    call_args, kwargs = sagemaker_session.sagemaker_runtime_client.invoke_endpoint.call_args
    assert kwargs == expected_request_args

    assert result == RETURN_VALUE


def json_sagemaker_session():
    ims = Mock(name="sagemaker_session")
    ims.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    ims.sagemaker_runtime_client = Mock(name="sagemaker_runtime")
    ims.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    ims.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)

    ims.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    ims.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)

    response_body = Mock("body")
    response_body.read = Mock("read", return_value=json.dumps([RETURN_VALUE]))
    response_body.close = Mock("close", return_value=None)
    ims.sagemaker_runtime_client.invoke_endpoint = Mock(
        name="invoke_endpoint",
        return_value={"Body": response_body, "ContentType": DEFAULT_CONTENT_TYPE},
    )
    return ims


def test_predict_call_with_headers_and_json():
    sagemaker_session = json_sagemaker_session()
    predictor = RealTimePredictor(
        ENDPOINT,
        sagemaker_session,
        content_type="not/json",
        accept="also/not-json",
        serializer=json_serializer,
    )

    data = [1, 2]
    result = predictor.predict(data)

    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint.called

    expected_request_args = {
        "Accept": "also/not-json",
        "Body": json.dumps(data),
        "ContentType": "not/json",
        "EndpointName": ENDPOINT,
    }
    call_args, kwargs = sagemaker_session.sagemaker_runtime_client.invoke_endpoint.call_args
    assert kwargs == expected_request_args

    assert result == json.dumps([RETURN_VALUE])


def ret_csv_sagemaker_session():
    ims = Mock(name="sagemaker_session")
    ims.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    ims.sagemaker_runtime_client = Mock(name="sagemaker_runtime")
    ims.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    ims.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)

    ims.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    ims.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)

    response_body = Mock("body")
    response_body.read = Mock("read", return_value=CSV_RETURN_VALUE)
    response_body.close = Mock("close", return_value=None)
    ims.sagemaker_runtime_client.invoke_endpoint = Mock(
        name="invoke_endpoint",
        return_value={"Body": response_body, "ContentType": CSV_CONTENT_TYPE},
    )
    return ims


def test_predict_call_with_headers_and_csv():
    sagemaker_session = ret_csv_sagemaker_session()
    predictor = RealTimePredictor(
        ENDPOINT, sagemaker_session, accept=CSV_CONTENT_TYPE, serializer=csv_serializer
    )

    data = [1, 2]
    result = predictor.predict(data)

    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint.called

    expected_request_args = {
        "Accept": CSV_CONTENT_TYPE,
        "Body": "1,2",
        "ContentType": CSV_CONTENT_TYPE,
        "EndpointName": ENDPOINT,
    }
    call_args, kwargs = sagemaker_session.sagemaker_runtime_client.invoke_endpoint.call_args
    assert kwargs == expected_request_args

    assert result == CSV_RETURN_VALUE


def test_delete_endpoint_with_config():
    sagemaker_session = empty_sagemaker_session()
    sagemaker_session.sagemaker_client.describe_endpoint = Mock(
        return_value={"EndpointConfigName": "endpoint-config"}
    )
    predictor = RealTimePredictor(ENDPOINT, sagemaker_session=sagemaker_session)
    predictor.delete_endpoint()

    sagemaker_session.delete_endpoint.assert_called_with(ENDPOINT)
    sagemaker_session.delete_endpoint_config.assert_called_with("endpoint-config")


def test_delete_endpoint_only():
    sagemaker_session = empty_sagemaker_session()
    predictor = RealTimePredictor(ENDPOINT, sagemaker_session=sagemaker_session)
    predictor.delete_endpoint(delete_endpoint_config=False)

    sagemaker_session.delete_endpoint.assert_called_with(ENDPOINT)
    sagemaker_session.delete_endpoint_config.assert_not_called()


def test_delete_model():
    sagemaker_session = empty_sagemaker_session()
    predictor = RealTimePredictor(ENDPOINT, sagemaker_session=sagemaker_session)

    predictor.delete_model()

    expected_call_count = 2
    expected_call_args_list = [call("model-1"), call("model-2")]
    assert sagemaker_session.delete_model.call_count == expected_call_count
    assert sagemaker_session.delete_model.call_args_list == expected_call_args_list


def test_delete_model_fail():
    sagemaker_session = empty_sagemaker_session()
    sagemaker_session.sagemaker_client.delete_model = Mock(
        side_effect=Exception("Could not find model.")
    )
    expected_error_message = "One or more models cannot be deleted, please retry."

    predictor = RealTimePredictor(ENDPOINT, sagemaker_session=sagemaker_session)

    with pytest.raises(Exception) as exception:
        predictor.delete_model()
        assert expected_error_message in str(exception.val)
