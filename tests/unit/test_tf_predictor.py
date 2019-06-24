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
import sys

from google.protobuf import json_format
import numpy as np
import pytest
from mock import Mock
import tensorflow as tf
import six
from six import BytesIO
from tensorflow.python.saved_model.signature_constants import (
    DEFAULT_SERVING_SIGNATURE_DEF_KEY,
    PREDICT_INPUTS,
)

from sagemaker.predictor import RealTimePredictor
from sagemaker.tensorflow.predictor import (
    tf_csv_serializer,
    tf_deserializer,
    tf_json_deserializer,
    tf_json_serializer,
    tf_serializer,
)
from sagemaker.tensorflow.tensorflow_serving.apis import classification_pb2

BUCKET_NAME = "mybucket"
ENDPOINT = "myendpoint"
REGION = "us-west-2"

CLASSIFICATION_RESPONSE = {
    "result": {
        "classifications": [
            {
                "classes": [
                    {"label": "0", "score": 0.0012890376383438706},
                    {"label": "1", "score": 0.9814321994781494},
                    {"label": "2", "score": 0.017278732731938362},
                ]
            }
        ]
    }
}

CSV_CONTENT_TYPE = "text/csv"
JSON_CONTENT_TYPE = "application/json"
PROTO_CONTENT_TYPE = "application/octet-stream"

ENDPOINT_DESC = {"EndpointConfigName": ENDPOINT}

ENDPOINT_CONFIG_DESC = {"ProductionVariants": [{"ModelName": "model-1"}, {"ModelName": "model-2"}]}


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    ims = Mock(name="sagemaker_session", boto_session=boto_mock)
    ims.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    ims.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    ims.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)
    return ims


def test_endpoint_initialization(sagemaker_session):
    endpoint_name = "endpoint"
    predictor = RealTimePredictor(endpoint=endpoint_name, sagemaker_session=sagemaker_session)
    assert predictor.endpoint == endpoint_name


def test_classification_request_json(sagemaker_session):
    data = [1, 2, 3]
    predictor = RealTimePredictor(
        endpoint=ENDPOINT,
        sagemaker_session=sagemaker_session,
        deserializer=tf_json_deserializer,
        serializer=tf_json_serializer,
    )

    mock_response(
        json.dumps(CLASSIFICATION_RESPONSE).encode("utf-8"), sagemaker_session, JSON_CONTENT_TYPE
    )

    result = predictor.predict(data)

    sagemaker_session.sagemaker_runtime_client.invoke_endpoint.assert_called_once_with(
        Accept=JSON_CONTENT_TYPE,
        Body="[1, 2, 3]",
        ContentType=JSON_CONTENT_TYPE,
        EndpointName="myendpoint",
    )

    assert result == CLASSIFICATION_RESPONSE


def test_classification_request_csv(sagemaker_session):
    data = [1, 2, 3]
    predictor = RealTimePredictor(
        serializer=tf_csv_serializer,
        deserializer=tf_deserializer,
        sagemaker_session=sagemaker_session,
        endpoint=ENDPOINT,
    )

    expected_response = json_format.Parse(
        json.dumps(CLASSIFICATION_RESPONSE), classification_pb2.ClassificationResponse()
    ).SerializeToString()

    mock_response(expected_response, sagemaker_session, PROTO_CONTENT_TYPE)

    result = predictor.predict(data)

    sagemaker_session.sagemaker_runtime_client.invoke_endpoint.assert_called_once_with(
        Accept=PROTO_CONTENT_TYPE,
        Body="1,2,3",
        ContentType=CSV_CONTENT_TYPE,
        EndpointName="myendpoint",
    )

    # python 2 and 3 protobuf serialization has different precision so I'm checking
    # the version here
    if sys.version_info < (3, 0):
        assert (
            str(result)
            == """result {
  classifications {
    classes {
      label: "0"
      score: 0.00128903763834
    }
    classes {
      label: "1"
      score: 0.981432199478
    }
    classes {
      label: "2"
      score: 0.0172787327319
    }
  }
}
"""
        )
    else:
        assert (
            str(result)
            == """result {
  classifications {
    classes {
      label: "0"
      score: 0.0012890376383438706
    }
    classes {
      label: "1"
      score: 0.9814321994781494
    }
    classes {
      label: "2"
      score: 0.017278732731938362
    }
  }
}
"""
        )


def test_json_deserializer_should_work_with_predict_response():
    data = b"""{
"outputs": {
    "example_strings": {
      "dtype": "DT_STRING",
      "tensorShape": {
        "dim": [
          {
            "size": "3"
          }
        ]
      },
      "stringVal": [
        "YXBwbGU=",
        "YmFuYW5h",
        "b3Jhbmdl"
      ]
    },
    "ages": {
      "dtype": "DT_FLOAT",
      "floatVal": [
        4.954165935516357
      ],
      "tensorShape": {
        "dim": [
          {
            "size": "1"
          }
        ]
      }
    }
  },
  "modelSpec": {
    "version": "1531758457",
    "name": "generic_model",
    "signatureName": "serving_default"
  }
}"""

    stream = BytesIO(data)

    response = tf_json_deserializer(stream, "application/json")

    if six.PY2:
        string_vals = ["apple", "banana", "orange"]
    else:
        string_vals = [b"apple", b"banana", b"orange"]

    assert response == {
        "model_spec": {
            "name": u"generic_model",
            "signature_name": u"serving_default",
            "version": {"value": 1531758457.0 if six.PY2 else 1531758457},
        },
        "outputs": {
            u"ages": {
                "dtype": 1,
                "float_val": [4.954165935516357],
                "tensor_shape": {"dim": [{"size": 1.0 if six.PY2 else 1}]},
            },
            u"example_strings": {
                "dtype": 7,
                "string_val": string_vals,
                "tensor_shape": {"dim": [{"size": 3.0 if six.PY2 else 3}]},
            },
        },
    }


def test_classification_request_pb(sagemaker_session):
    request = classification_pb2.ClassificationRequest()
    request.model_spec.name = "generic_model"
    request.model_spec.signature_name = DEFAULT_SERVING_SIGNATURE_DEF_KEY
    example = request.input.example_list.examples.add()
    example.features.feature[PREDICT_INPUTS].float_list.value.extend([6.4, 3.2, 4.5, 1.5])

    predictor = RealTimePredictor(
        sagemaker_session=sagemaker_session,
        endpoint=ENDPOINT,
        deserializer=tf_deserializer,
        serializer=tf_serializer,
    )

    expected_response = classification_pb2.ClassificationResponse()
    classes = expected_response.result.classifications.add().classes

    class_0 = classes.add()
    class_0.label = "0"
    class_0.score = 0.00128903763834

    class_1 = classes.add()
    class_1.label = "1"
    class_1.score = 0.981432199478

    class_2 = classes.add()
    class_2.label = "2"
    class_2.score = 0.0172787327319

    mock_response(expected_response.SerializeToString(), sagemaker_session, PROTO_CONTENT_TYPE)

    result = predictor.predict(request)

    sagemaker_session.sagemaker_runtime_client.invoke_endpoint.assert_called_once_with(
        Accept=PROTO_CONTENT_TYPE,
        Body=request.SerializeToString(),
        ContentType=PROTO_CONTENT_TYPE,
        EndpointName="myendpoint",
    )

    # python 2 and 3 protobuf serialization has different precision so I'm checking
    # the version here
    if sys.version_info < (3, 0):
        assert (
            str(result)
            == """result {
  classifications {
    classes {
      label: "0"
      score: 0.00128903763834
    }
    classes {
      label: "1"
      score: 0.981432199478
    }
    classes {
      label: "2"
      score: 0.0172787327319
    }
  }
}
"""
        )
    else:
        assert (
            str(result)
            == """result {
  classifications {
    classes {
      label: "0"
      score: 0.0012890376383438706
    }
    classes {
      label: "1"
      score: 0.9814321994781494
    }
    classes {
      label: "2"
      score: 0.017278732731938362
    }
  }
}
"""
        )


def test_predict_request_json(sagemaker_session):
    data = [6.4, 3.2, 0.5, 1.5]
    tensor_proto = tf.make_tensor_proto(
        values=np.asarray(data), shape=[1, len(data)], dtype=tf.float32
    )
    predictor = RealTimePredictor(
        sagemaker_session=sagemaker_session,
        endpoint=ENDPOINT,
        deserializer=tf_json_deserializer,
        serializer=tf_json_serializer,
    )

    mock_response(
        json.dumps(CLASSIFICATION_RESPONSE).encode("utf-8"), sagemaker_session, JSON_CONTENT_TYPE
    )

    result = predictor.predict(tensor_proto)

    sagemaker_session.sagemaker_runtime_client.invoke_endpoint.assert_called_once_with(
        Accept=JSON_CONTENT_TYPE,
        Body=json_format.MessageToJson(tensor_proto),
        ContentType=JSON_CONTENT_TYPE,
        EndpointName="myendpoint",
    )

    assert result == CLASSIFICATION_RESPONSE


def test_predict_tensor_request_csv(sagemaker_session):
    data = [6.4, 3.2, 0.5, 1.5]
    tensor_proto = tf.make_tensor_proto(
        values=np.asarray(data), shape=[1, len(data)], dtype=tf.float32
    )
    predictor = RealTimePredictor(
        serializer=tf_csv_serializer,
        deserializer=tf_json_deserializer,
        sagemaker_session=sagemaker_session,
        endpoint=ENDPOINT,
    )

    mock_response(
        json.dumps(CLASSIFICATION_RESPONSE).encode("utf-8"), sagemaker_session, JSON_CONTENT_TYPE
    )

    result = predictor.predict(tensor_proto)

    sagemaker_session.sagemaker_runtime_client.invoke_endpoint.assert_called_once_with(
        Accept=JSON_CONTENT_TYPE,
        Body="6.4,3.2,0.5,1.5",
        ContentType=CSV_CONTENT_TYPE,
        EndpointName="myendpoint",
    )

    assert result == CLASSIFICATION_RESPONSE


def mock_response(expected_response, sagemaker_session, content_type):
    sagemaker_session.sagemaker_runtime_client.invoke_endpoint.return_value = {
        "ContentType": content_type,
        "Body": io.BytesIO(expected_response),
    }


def test_json_serialize_dict():
    data = {"tensor1": [1, 2, 3], "tensor2": [4, 5, 6]}
    serialized = tf_json_serializer(data)
    # deserialize again for assertion, since dict order is not guaranteed
    deserialized = json.loads(serialized)
    assert deserialized == data


def test_json_serialize_dict_with_numpy():
    data = {"tensor1": np.asarray([1, 2, 3]), "tensor2": np.asarray([4, 5, 6])}
    serialized = tf_json_serializer(data)
    # deserialize again for assertion, since dict order is not guaranteed
    deserialized = json.loads(serialized)
    assert deserialized == {"tensor1": [1, 2, 3], "tensor2": [4, 5, 6]}


def test_json_serialize_numpy():
    data = np.asarray([[1, 2, 3], [4, 5, 6]])
    assert tf_json_serializer(data) == "[[1, 2, 3], [4, 5, 6]]"
