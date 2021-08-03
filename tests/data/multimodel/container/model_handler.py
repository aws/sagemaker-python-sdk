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

"""
ModelHandler defines a dummy base model handler.
Returns invoked model name.
"""
import logging


class ModelHandler(object):
    """
    A base Model handler implementation.
    """

    def __init__(self):
        self.error = None
        self._context = None
        self.initialized = False

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time

        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self.initialized = True

    def handle(self, data, context):
        """
        Custom service entry point function.

        :param data: list of objects, raw input from request
        :param context: model server context
        :return: list of target model to send back to client
        """
        self.error = None  # reset earlier errors

        try:
            target_model = context.get_request_header(0, "X-Amzn-SageMaker-Target-Model")
            return ["Invoked model: {}".format(target_model)]
        except Exception as e:
            logging.error(e, exc_info=True)
            request_processor = context.request_processor
            request_processor.report_status(500, "Unknown inference error")
            return str(e)
