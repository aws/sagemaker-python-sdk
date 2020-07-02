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
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def ACCEPT(self):
        """The content type that is expected from the inference endpoint."""
        raise NotImplementedError
