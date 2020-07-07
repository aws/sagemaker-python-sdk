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

from sagemaker.utils import parse_mime_type


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


class StringDeserializer(object):
    """Deserialize data from an inference endpoint into a decoded string."""

    def __init__(self, encoding="UTF-8"):
        """Initialize the default encoding.

        Args:
            encoding (str): The string encoding to use, if a charset is not
                provided by the server (default: UTF-8).
        """
        self.encoding = encoding

    def deserialize(self, data, content_type):
        """Deserialize data from an inference endpoint into a decoded string.

        Args:
            data (object): A string or a byte stream.
            content_type (str): The MIME type of the data.

        Returns:
            str: The data deserialized into a decoded string.
        """
        category, _, parameters = parse_mime_type(content_type)

        if category == "text":
            return data

        try:
            encoding = parameters.get("charset", self.encoding)
            return data.read().decode(encoding)
        finally:
            data.close()
