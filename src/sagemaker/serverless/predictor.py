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
"""Predictors that are hosted on serverless compute."""
from __future__ import absolute_import

from typing import Optional, Tuple

import boto3
import botocore

from sagemaker import deserializers, serializers
from sagemaker.predictor import PredictorBase


class LambdaPredictor(PredictorBase):
    """A deployed model hosted on Lambda."""

    def __init__(
        self, function_name: str, client: Optional[botocore.client.BaseClient] = None
    ) -> None:
        """Initialize instance attributes.

        Arguments:
            function_name: The name of the function.
            client: The Lambda client used to interact with Lambda.
        """
        self._client = client or boto3.client("lambda")
        self._function_name = function_name
        self._serializer = serializers.JSONSerializer()
        self._deserializer = deserializers.JSONDeserializer()

    def predict(self, data: dict) -> dict:
        """Invoke the Lambda function specified in the constructor.

        This function is synchronous. It will only return after the function
        has produced a prediction.

        Arguments:
            data: The data sent to the Lambda function as input.

        Returns:
            The data returned by the Lambda function.
        """
        response = self._client.invoke(
            FunctionName=self._function_name,
            InvocationType="RequestResponse",
            Payload=self._serializer.serialize(data),
        )
        return self._deserializer.deserialize(
            response["Payload"],
            response["ResponseMetadata"]["HTTPHeaders"]["content-type"],
        )

    def delete_predictor(self) -> None:
        """Destroy the Lambda function specified in the constructor."""
        self._client.delete_function(FunctionName=self._function_name)

    @property
    def content_type(self) -> str:
        """The MIME type of the data sent to the Lambda function."""
        return self._serializer.CONTENT_TYPE

    @property
    def accept(self) -> Tuple[str]:
        """The content type(s) that are expected from the Lambda function."""
        return self._deserializer.ACCEPT

    @property
    def function_name(self) -> str:
        """The name of the Lambda function this predictor invokes."""
        return self._function_name
