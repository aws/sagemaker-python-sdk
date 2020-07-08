# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Implements methods for deserializing data returned from an inference endpoint."""
from __future__ import absolute_import

import csv

import abc


class BaseDeserializer(abc.ABC):
    """Abstract base class for creation of new deserializers.

    Provides a skeleton for customization requiring the overriding of the method
    deserialize and the class attribute ACCEPT.
    """

    @abc.abstractmethod
    def deserialize(self, data, content_type):
        """Deserialize data received from an inference endpoint.

        Args:
            data (object): Data to be deserialized.
            content_type (str): The MIME type of the data.

        Returns:
            object: The data deserialized into an object.
        """

    @property
    @abc.abstractmethod
    def ACCEPT(self):
        """The content type that is expected from the inference endpoint."""


class BytesDeserializer(BaseDeserializer):
    """Deserialize a stream of bytes into a bytes object."""

    ACCEPT = "*/*"

    def deserialize(self, data, content_type):
        """Read a stream of bytes returned from an inference endpoint.

        Args:
            data (object): A stream of bytes.
            content_type (str): The MIME type of the data.

        Returns:
            bytes: The bytes object read from the stream.
        """
        try:
            return data.read()
        finally:
            data.close()


class CSVDeserializer(BaseDeserializer):
    """Deserialize a stream of bytes into a list of lists."""

    ACCEPT = "test/csv"

    def __init__(self, encoding="utf-8"):
        """Initialize the string encoding.

        Args:
            encoding (str): The string encoding to use (default: "utf-8").
        """
        self.encoding = encoding

    def deserialize(self, data, content_type):
        """Deserialize data from an inference endpoint into a list of lists.

        Args:
            data (botocore.response.StreamingBody): Data to be deserialized.
            content_type (str): The MIME type of the data.

        Returns:
            list: The data deserialized into a list of lists representing the
                contents of a CSV file.
        """
        try:
            decoded_string = data.read().decode(self.encoding)
            return list(csv.reader(decoded_string.splitlines()))
        finally:
            data.close()
