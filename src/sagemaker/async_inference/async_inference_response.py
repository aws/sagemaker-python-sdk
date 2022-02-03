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
"""A class for AsyncInferenceResponse"""

from __future__ import print_function, absolute_import

from botocore.exceptions import ClientError
from sagemaker.s3 import parse_s3_url
from sagemaker.async_inference import WaiterConfig
from sagemaker.exceptions import ObjectNotExistedError, UnexpectedClientError


class AsyncInferenceResponse(object):
    """Response from Async Inference endpoint

    This response object provides a method to check for an async inference result in the
    Amazon S3 output path specified. If result object exists in that path, get and return
    the result
    """

    def __init__(
        self,
        predictor_async,
        output_path,
    ):
        """Initialize an AsyncInferenceResponse object.

        AsyncInferenceResponse can help users to get async inference result
        from the Amazon S3 output path

        Args:
            predictor_async (sagemaker.predictor.AsyncPredictor): The ``AsyncPredictor``
                that return this response.
            output_path (str): The Amazon S3 location that endpoints upload inference responses
                to.
        """
        self.predictor_async = predictor_async
        self.output_path = output_path
        self._result = None

    def get_result(
        self,
        waiter_config=None,
    ):
        """Get async inference result in the Amazon S3 output path specified

        Args:
            waiter_config (sagemaker.async_inference.waiter_config.WaiterConfig): Configuration
                for the waiter. The pre-defined value for the delay between poll is 15 seconds
                and the default max attempts is 60
        Raises:
            ValueError: If a wrong type of object is provided as ``waiter_config``
        Returns:
            object: Inference result in the given Amazon S3 output path. If a deserializer was
                specified when creating the AsyncPredictor, the result of the deserializer is
                returned. Otherwise the response returns the sequence of bytes
                as is.
        """
        if waiter_config is not None and not isinstance(waiter_config, WaiterConfig):
            raise ValueError("waiter_config should be a WaiterConfig object")

        if self._result is None:
            if waiter_config is None:
                self._result = self._get_result_from_s3(self.output_path)
            else:
                self._result = self.predictor_async._wait_for_output(
                    self.output_path, waiter_config
                )
        return self._result

    def _get_result_from_s3(
        self,
        output_path,
    ):
        """Get inference result from the output Amazon S3 path"""
        bucket, key = parse_s3_url(output_path)
        try:
            response = self.predictor_async.s3_client.get_object(Bucket=bucket, Key=key)
            return self.predictor_async.predictor._handle_response(response)
        except ClientError as ex:
            if ex.response["Error"]["Code"] == "NoSuchKey":
                raise ObjectNotExistedError(
                    message="Inference could still be running",
                    output_path=output_path,
                )
            raise UnexpectedClientError(
                message=ex.response["Error"]["Message"],
            )
