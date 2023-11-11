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

from io import BytesIO
from typing import IO
import numpy as np
from pandas import DataFrame

from sagemaker.serve import SchemaBuilder, CustomPayloadTranslator
from sagemaker.serve.builder.schema_builder import JSONSerializerWrapper
from sagemaker.deserializers import (
    BytesDeserializer,
    NumpyDeserializer,
    JSONDeserializer,
    PandasDeserializer,
)
from sagemaker.serializers import (
    DataSerializer,
    NumpySerializer,
    CSVSerializer,
)

NUMPY_CONTENT_TYPE = "application/x-npy"
DATAFRAME_CONTENT_TYPE = "text/csv"
JSON_CONTENT_TYPE = "application/json"
TORCH_TENSOR_CONTENT_TYPE = "tensor/pt"
CLOUDPICKLE_CONTENT_TYPE = "application/x-pkl"


class SomeRandomClass:
    def __init__(self, number: int, text: str) -> None:
        self.some_random_number = number
        self.some_random_text = text


class MyPayloadTranslator(CustomPayloadTranslator):
    """A simple converter class that serializes/deserializes a string."""

    def serialize_payload_to_bytes(self, payload: str) -> bytes:
        return str.encode(payload)

    def deserialize_payload_from_stream(self, stream: IO) -> object:
        return stream.read().decode()


@pytest.fixture
def numpy_array():
    shape = (3, 4)
    return np.random.rand(*shape)


@pytest.fixture
def pandas_df(numpy_array):
    return DataFrame(numpy_array, columns=["col1", "col2", "col3", "col4"])


@pytest.fixture
def jsonable_obj():
    return {"key1": "value1"}


@pytest.fixture
def jsonable_obj_2():
    return ["value 1", "value 2"]


@pytest.fixture
def some_bytes():
    return b"some_bytes"


@pytest.fixture
def unsupported_object():
    return SomeRandomClass(
        number=42, text="This class will be serialized/deserialized with cloudpickle."
    )


@pytest.fixture
def custom_translator():
    return MyPayloadTranslator()


# @pytest.fixture
# def torch_tensor():
#     return torch.rand(3, 4)


def test_schema_builder_with_numpy(numpy_array):
    schema_builder = SchemaBuilder(numpy_array, numpy_array)
    assert isinstance(schema_builder.input_serializer, NumpySerializer)
    assert isinstance(schema_builder.output_serializer, NumpySerializer)
    assert isinstance(schema_builder.input_deserializer._deserializer, NumpyDeserializer)
    assert schema_builder.input_deserializer.ACCEPT == NUMPY_CONTENT_TYPE
    assert isinstance(schema_builder.output_deserializer._deserializer, NumpyDeserializer)
    assert schema_builder.output_deserializer.ACCEPT == NUMPY_CONTENT_TYPE


def test_schema_builder_with_pandas_dataframe(pandas_df):
    schema_builder = SchemaBuilder(pandas_df, pandas_df)
    assert isinstance(schema_builder.input_serializer, CSVSerializer)
    assert isinstance(schema_builder.output_serializer, CSVSerializer)
    assert isinstance(schema_builder.input_deserializer._deserializer, PandasDeserializer)
    assert schema_builder.input_deserializer.ACCEPT == DATAFRAME_CONTENT_TYPE
    assert isinstance(schema_builder.output_deserializer._deserializer, PandasDeserializer)
    assert schema_builder.output_deserializer.ACCEPT == DATAFRAME_CONTENT_TYPE


def test_schema_builder_with_jsonable(jsonable_obj):
    schema_builder = SchemaBuilder(jsonable_obj, jsonable_obj)
    assert isinstance(schema_builder.input_serializer, JSONSerializerWrapper)
    assert isinstance(schema_builder.output_serializer, JSONSerializerWrapper)
    assert isinstance(schema_builder.input_deserializer._deserializer, JSONDeserializer)
    assert schema_builder.input_deserializer.ACCEPT == JSON_CONTENT_TYPE
    assert isinstance(schema_builder.output_deserializer._deserializer, JSONDeserializer)
    assert schema_builder.output_deserializer.ACCEPT == JSON_CONTENT_TYPE


def test_schema_builder_with_bytes(some_bytes):
    schema_builder = SchemaBuilder(some_bytes, some_bytes)
    assert isinstance(schema_builder.input_serializer, DataSerializer)
    assert isinstance(schema_builder.output_serializer, DataSerializer)
    assert isinstance(schema_builder.input_deserializer._deserializer, BytesDeserializer)
    assert isinstance(schema_builder.output_deserializer._deserializer, BytesDeserializer)


def test_schema_builder_with_cloudpickle(unsupported_object):
    with pytest.raises(ValueError, match="SchemaBuilder cannot determine"):
        SchemaBuilder(unsupported_object, unsupported_object)


@pytest.mark.parametrize("jsonable", ["jsonable_obj", "jsonable_obj_2"])
def test_json_serializer_wrapper(jsonable):
    b = JSONSerializerWrapper().serialize(jsonable)
    stream = BytesIO(b)
    JSONDeserializer().deserialize(stream, content_type="application/json")


def test_schema_builder_with_payload_translator(custom_translator):
    payload = "payload"
    schema_builder = SchemaBuilder(
        payload, payload, input_translator=custom_translator, output_translator=custom_translator
    )
    assert isinstance(schema_builder.custom_input_translator, MyPayloadTranslator)
    assert isinstance(schema_builder.custom_output_translator, MyPayloadTranslator)


def test_schema_builder_with_payload_converter_invalid_payload(custom_translator):
    invalid_payload = 42
    with pytest.raises(Exception):
        SchemaBuilder(
            invalid_payload,
            invalid_payload,
            input_translator=custom_translator,
            output_translator=custom_translator,
        )
