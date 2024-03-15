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
import unittest
from unittest.mock import patch, ANY

import numpy as np
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.marshalling.custom_payload_translator import CustomPayloadTranslator
from sagemaker.serve.utils.predictors import retrieve_predictor

ENDPOINT_NAME = "ENDPOINT_NAME"


class MyNumpyTranslator(CustomPayloadTranslator):
    def serialize_payload_to_bytes(self, np_array: object) -> bytes:
        buffer = io.BytesIO()
        np.save(buffer, np_array)
        return buffer.getvalue()

    def deserialize_payload_from_stream(self, stream) -> object:
        return np.load(io.BytesIO(stream.read()))


class TestPredictors(unittest.TestCase):
    @patch("sagemaker.serve.utils.predictors.Session")
    @patch("sagemaker.serve.utils.predictors.Predictor")
    def test_retrieve_predictor_happy(self, mock_predictor, mock_session):

        shape = (3, 4)
        numpy_array = np.random.rand(*shape)
        schema_builder = SchemaBuilder(sample_input=numpy_array, sample_output=numpy_array)
        retrieve_predictor(
            endpoint_name=ENDPOINT_NAME,
            sagemaker_session=mock_session,
            schema_builder=schema_builder,
        )
        mock_predictor.assert_called_once_with(
            endpoint_name=ENDPOINT_NAME,
            sagemaker_session=ANY,
            serializer=schema_builder.input_serializer,
            deserializer=schema_builder.output_deserializer,
        )

    @patch("sagemaker.serve.utils.predictors.Session")
    @patch("sagemaker.serve.utils.predictors.Predictor")
    def test_retrieve_predictor_custom_translator_happy(self, mock_predictor, mock_session):

        shape = (3, 4)
        numpy_array = np.random.rand(*shape)
        numpy_translator = MyNumpyTranslator()
        schema_builder = SchemaBuilder(
            sample_input=numpy_array,
            sample_output=numpy_array,
            input_translator=numpy_translator,
            output_translator=numpy_translator,
        )
        retrieve_predictor(
            endpoint_name=ENDPOINT_NAME,
            sagemaker_session=mock_session,
            schema_builder=schema_builder,
        )
        mock_predictor.assert_called_once_with(
            endpoint_name=ENDPOINT_NAME,
            sagemaker_session=ANY,
            serializer=schema_builder.custom_input_translator,
            deserializer=schema_builder.custom_output_translator,
        )
