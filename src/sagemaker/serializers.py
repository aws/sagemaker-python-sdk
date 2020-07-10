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
import csv
import io

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


class CSVSerializer(BaseSerializer):
    """Placeholder docstring"""

    CONTENT_TYPE = "text/csv"

    def serialize(self, data):
        """Take data of various data formats and serialize them into CSV.

        Args:
            data (object): Data to be serialized.

        Returns:
            object: Sequence of bytes to be used for the request body.
        """
        # For inputs which represent multiple "rows", the result should be newline-separated CSV
        # rows
        if _is_mutable_sequence_like(data) and len(data) > 0 and _is_sequence_like(data[0]):
            return "\n".join([CSVSerializer._serialize_row(row) for row in data])
        return CSVSerializer._serialize_row(data)

    @staticmethod
    def _serialize_row(data):
        # Don't attempt to re-serialize a string
        """
        Args:
            data:
        """
        if isinstance(data, str):
            return data
        if isinstance(data, np.ndarray):
            data = np.ndarray.flatten(data)
        if hasattr(data, "__len__"):
            if len(data) == 0:
                raise ValueError("Cannot serialize empty array")
            return _csv_serialize_python_array(data)

        # files and buffers
        if hasattr(data, "read"):
            return _csv_serialize_from_buffer(data)

        raise ValueError("Unable to handle input format: ", type(data))


def _csv_serialize_python_array(data):
    """
    Args:
        data:
    """
    return _csv_serialize_object(data)


def _csv_serialize_from_buffer(buff):
    """
    Args:
        buff:
    """
    return buff.read()


def _csv_serialize_object(data):
    """
    Args:
        data:
    """
    csv_buffer = io.StringIO()

    csv_writer = csv.writer(csv_buffer, delimiter=",")
    csv_writer.writerow(data)
    return csv_buffer.getvalue().rstrip("\r\n")


def _is_mutable_sequence_like(obj):
    """
    Args:
        obj:
    """
    return _is_sequence_like(obj) and hasattr(obj, "__setitem__")


def _is_sequence_like(obj):
    """
    Args:
        obj:
    """
    return hasattr(obj, "__iter__") and hasattr(obj, "__getitem__")


def _row_to_csv(obj):
    """
    Args:
        obj:
    """
    if isinstance(obj, str):
        return obj
    return ",".join(obj)
