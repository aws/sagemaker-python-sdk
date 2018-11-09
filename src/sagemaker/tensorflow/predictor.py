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

import json

import google.protobuf.json_format as json_format
from google.protobuf.message import DecodeError
from protobuf_to_dict import protobuf_to_dict
from tensorflow.core.framework import tensor_pb2  # pylint: disable=no-name-in-module
from tensorflow.python.framework import tensor_util  # pylint: disable=no-name-in-module

from sagemaker.content_types import CONTENT_TYPE_JSON, CONTENT_TYPE_OCTET_STREAM, CONTENT_TYPE_CSV
from sagemaker.predictor import json_serializer, csv_serializer
from tensorflow_serving.apis import predict_pb2, classification_pb2, inference_pb2, regression_pb2

_POSSIBLE_RESPONSES = [predict_pb2.PredictResponse, classification_pb2.ClassificationResponse,
                       inference_pb2.MultiInferenceResponse, regression_pb2.RegressionResponse,
                       tensor_pb2.TensorProto]

REGRESSION_REQUEST = 'RegressionRequest'
MULTI_INFERENCE_REQUEST = 'MultiInferenceRequest'
CLASSIFICATION_REQUEST = 'ClassificationRequest'
PREDICT_REQUEST = 'PredictRequest'


class _TFProtobufSerializer(object):
    def __init__(self):
        self.content_type = CONTENT_TYPE_OCTET_STREAM

    def __call__(self, data):
        # isinstance does not work here because a same protobuf message can be imported from a different module.
        # for example sagemaker.tensorflow.tensorflow_serving.regression_pb2 and tensorflow_serving.apis.regression_pb2
        predict_type = data.__class__.__name__

        available_requests = [PREDICT_REQUEST, CLASSIFICATION_REQUEST, MULTI_INFERENCE_REQUEST, REGRESSION_REQUEST]

        if predict_type not in available_requests:
            raise ValueError('request type {} is not supported'.format(predict_type))
        return data.SerializeToString()


tf_serializer = _TFProtobufSerializer()


class _TFProtobufDeserializer(object):
    def __init__(self):
        self.accept = CONTENT_TYPE_OCTET_STREAM

    def __call__(self, stream, content_type):
        try:
            data = stream.read()
        finally:
            stream.close()

        for possible_response in _POSSIBLE_RESPONSES:
            try:
                response = possible_response()
                response.ParseFromString(data)
                return response
            except (UnicodeDecodeError, DecodeError):
                # given that the payload does not have the response type, there no way to infer
                # the response without keeping state, so I'm iterating all the options.
                pass
        raise ValueError('data is not in the expected format')


tf_deserializer = _TFProtobufDeserializer()


class _TFJsonSerializer(object):
    def __init__(self):
        self.content_type = CONTENT_TYPE_JSON

    def __call__(self, data):
        if isinstance(data, tensor_pb2.TensorProto):
            return json_format.MessageToJson(data)
        else:
            return json_serializer(data)


tf_json_serializer = _TFJsonSerializer()


class _TFJsonDeserializer(object):
    def __init__(self):
        self.accept = CONTENT_TYPE_JSON

    def __call__(self, stream, content_type):
        try:
            data = stream.read()
        finally:
            stream.close()

        for possible_response in _POSSIBLE_RESPONSES:
            try:
                return protobuf_to_dict(json_format.Parse(data, possible_response()))
            except (UnicodeDecodeError, DecodeError, json_format.ParseError):
                # given that the payload does not have the response type, there no way to infer
                # the response without keeping state, so I'm iterating all the options.
                pass
        return json.loads(data.decode())


tf_json_deserializer = _TFJsonDeserializer()


class _TFCsvSerializer(object):
    def __init__(self):
        self.content_type = CONTENT_TYPE_CSV

    def __call__(self, data):
        to_serialize = data
        if isinstance(data, tensor_pb2.TensorProto):
            to_serialize = tensor_util.MakeNdarray(data)
        return csv_serializer(to_serialize)


tf_csv_serializer = _TFCsvSerializer()
