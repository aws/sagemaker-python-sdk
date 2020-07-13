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
"""Implements methods for serializing data for an inference endpoint."""
from __future__ import absolute_import

import abc
import io
import json

import numpy as np


class BaseSerializer(abc.ABC):
    """Abstract base class for creation of new serializers.

    Provides a skeleton for customization requiring the overriding of the method
    serialize and the class attribute CONTENT_TYPE.
    """

    @abc.abstractmethod
    def serialize(self, data):
        """Serialize data into the media type specified by CONTENT_TYPE.

        Args:
            data (object): Data to be serialized.

        Returns:
            object: Serialized data used for a request.
        """

    @property
    @abc.abstractmethod
    def CONTENT_TYPE(self):
        """The MIME type of the data sent to the inference endpoint."""


class NumpySerializer(BaseSerializer):
    """Serialize data to a buffer using the .npy format."""

    CONTENT_TYPE = "application/x-npy"

    def __init__(self, dtype=None):
        """Initialize the dtype.

        Args:
            dtype (str): The dtype of the data.
        """
        self.dtype = dtype

    def serialize(self, data):
        """Serialize data to a buffer using the .npy format.

        Args:
            data (object): Data to be serialized. Can be a NumPy array, list,
                file, or buffer.

        Returns:
            io.BytesIO: A buffer containing data serialzied in the .npy format.
        """
        if isinstance(data, np.ndarray):
            if not data.size > 0:
                raise ValueError("Cannot serialize empty array.")
            return self._serialize_array(data)

        if isinstance(data, list):
            if not len(data) > 0:
                raise ValueError("Cannot serialize empty array.")
            return self._serialize_array(np.array(data, self.dtype))

        # files and buffers. Assumed to hold npy-formatted data.
        if hasattr(data, "read"):
            return data.read()

        return self._serialize_array(np.array(data))

    def _serialize_array(self, array):
        """Saves a NumPy array in a buffer.

        Args:
            array (numpy.ndarray): The array to serialize.

        Returns:
            io.BytesIO: A buffer containing the serialized array.
        """
        buffer = io.BytesIO()
        np.save(buffer, array)
        return buffer.getvalue()


class JSONSerializer(BaseSerializer):
    """Serialize data to a JSON formatted string."""

    CONTENT_TYPE = "application/json"

    def serialize(self, data):
        """Serialize data of various formats to a JSON formatted string.

        Args:
            data (object): Data to be serialized.

        Returns:
            str: The data serialized as a JSON string.
        """
        if isinstance(data, dict):
            return json.dumps(
                {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in data.items()
                }
            )

        if hasattr(data, "read"):
            return data.read()

        if isinstance(data, np.ndarray):
            return json.dumps(data.tolist())

        return json.dumps(data)
