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


class StringDeserializer(BaseDeserializer):
    """Deserialize data from an inference endpoint into a decoded string."""

    ACCEPT = "application/json"

    def __init__(self, encoding="UTF-8"):
        """Initialize the string encoding.

        Args:
            encoding (str): The string encoding to use (default: UTF-8).
        """
        self.encoding = encoding

    def deserialize(self, data, content_type):
        """Deserialize data from an inference endpoint into a decoded string.

        Args:
            data (object): Data to be deserialized.
            content_type (str): The MIME type of the data.

        Returns:
            str: The data deserialized into a decoded string.
        """
        try:
            return data.read().decode(self.encoding)
        finally:
            data.close()


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


class StreamDeserializer(BaseDeserializer):
    """Returns the data and content-type received from an inference endpoint.

    It is the user's responsibility to close the data stream once they're done
    reading it.
    """

    ACCEPT = "*/*"

    def deserialize(self, data, content_type):
        """Returns a stream of the response body and the MIME type of the data.

        Args:
            data (object): A stream of bytes.
            content_type (str): The MIME type of the data.

        Returns:
            tuple: A two-tuple containing the stream and content-type.
        """
        return data, content_type
