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
"""Holds templated classes to enable users to provide custom inference scripting capabilities"""
from __future__ import absolute_import
from abc import ABC, abstractmethod


class CustomOrchestrator(ABC):
    """Templated class to standardize sync entrypoint-based inference scripts"""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Boto3 SageMaker runtime client to use with custom orchestrator"""
        if not hasattr(self, "_client") or not self._client:
            from boto3 import Session

            self._client = Session().client("sagemaker-runtime")
        return self._client

    @abstractmethod
    def handle(self, data, context=None):
        """Abstract class for defining an entrypoint for the model server"""
        return NotImplemented


class AsyncCustomOrchestrator(ABC):
    """Templated class to standardize async entrypoint-based inference scripts"""

    @abstractmethod
    async def handle(self, data, context=None):
        """Abstract class for defining an aynchronous entrypoint for the model server"""
        return NotImplemented
