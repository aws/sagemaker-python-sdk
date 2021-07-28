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
"""Models that can be deployed to serverless compute."""
from __future__ import absolute_import

import time
from typing import Optional

import boto3
import botocore

from sagemaker.model import ModelBase

from .predictor import LambdaPredictor


class LambdaModel(ModelBase):
    """A model that can be deployed to Lambda."""

    def __init__(
        self, image_uri: str, role: str, client: Optional[botocore.client.BaseClient] = None
    ) -> None:
        """Initialize instance attributes.

        Arguments:
            image_uri: URI of a container image in the Amazon ECR registry. The image
                should contain a handler that performs inference.
            role: The Amazon Resource Name (ARN) of the IAM role that Lambda will assume
                when it performs inference
            client: The Lambda client used to interact with Lambda.
        """
        self._client = client or boto3.client("lambda")
        self._image_uri = image_uri
        self._role = role

    def deploy(
        self, function_name: str, timeout: int, memory_size: int, wait: bool = True
    ) -> LambdaPredictor:
        """Create a Lambda function using the image specified in the constructor.

        Arguments:
            function_name: The name of the function.
            timeout: The number of seconds that the function can run for before being terminated.
            memory_size: The amount of memory in MB that the function has access to.
            wait: If true, wait until the deployment completes (default: True).

        Returns:
            A LambdaPredictor instance that performs inference using the specified image.
        """
        response = self._client.create_function(
            FunctionName=function_name,
            PackageType="Image",
            Role=self._role,
            Code={
                "ImageUri": self._image_uri,
            },
            Timeout=timeout,
            MemorySize=memory_size,
        )

        if not wait:
            return LambdaPredictor(function_name, client=self._client)

        # Poll function state.
        polling_interval = 5
        while response["State"] == "Pending":
            time.sleep(polling_interval)
            response = self._client.get_function_configuration(FunctionName=function_name)

        if response["State"] != "Active":
            raise RuntimeError("Failed to deploy model to Lambda: %s" % response["StateReason"])

        return LambdaPredictor(function_name, client=self._client)

    def delete_model(self) -> None:
        """Destroy resources associated with this model.

        This method does not delete the image specified in the constructor. As
        a result, this method is a no-op.
        """
