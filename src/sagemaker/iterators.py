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
"""Implements iterators for deserializing data returned from an inference streaming endpoint."""
from __future__ import absolute_import

from abc import ABC, abstractmethod
import io

from sagemaker.exceptions import ModelStreamError, InternalStreamFailure


def handle_stream_errors(chunk):
    """Handle API Response errors within `invoke_endpoint_with_response_stream` API if any.

    Args:
        chunk (dict): A chunk of response received as part of `botocore.eventstream.EventStream`
            response object.

    Raises:
        ModelStreamError: If `ModelStreamError` error is detected in a chunk of
            `botocore.eventstream.EventStream` response object.
        InternalStreamFailure: If `InternalStreamFailure` error is detected in a chunk of
            `botocore.eventstream.EventStream` response object.
    """
    if "ModelStreamError" in chunk:
        raise ModelStreamError(
            chunk["ModelStreamError"]["Message"], code=chunk["ModelStreamError"]["ErrorCode"]
        )
    if "InternalStreamFailure" in chunk:
        raise InternalStreamFailure(chunk["InternalStreamFailure"]["Message"])


class BaseIterator(ABC):
    """Abstract base class for Inference Streaming iterators.

    Provides a skeleton for customization requiring the overriding of iterator methods
    __iter__ and __next__.

    Tenets of iterator class for Streaming Inference API Response
    (https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/
    sagemaker-runtime/client/invoke_endpoint_with_response_stream.html):
    1. Needs to accept an botocore.eventstream.EventStream response.
    2. Needs to implement logic in __next__ to:
        2.1. Concatenate and provide next chunk of response from botocore.eventstream.EventStream.
            While doing so parse the response_chunk["PayloadPart"]["Bytes"].
        2.2. If PayloadPart not in EventStream response, handle Errors
            [Recommended to use `iterators.handle_stream_errors` method].
    """

    def __init__(self, event_stream):
        """Initialises a Iterator object to help parse the byte event stream input.

        Args:
            event_stream: (botocore.eventstream.EventStream): Event Stream object to be iterated.
        """
        self.event_stream = event_stream

    @abstractmethod
    def __iter__(self):
        """Abstract method, returns an iterator object itself"""
        return self

    @abstractmethod
    def __next__(self):
        """Abstract method, is responsible for returning the next element in the iteration"""


class ByteIterator(BaseIterator):
    """A helper class for parsing the byte Event Stream input to provide Byte iteration."""

    def __init__(self, event_stream):
        """Initialises a BytesIterator Iterator object

        Args:
            event_stream: (botocore.eventstream.EventStream): Event Stream object to be iterated.
        """
        super().__init__(event_stream)
        self.byte_iterator = iter(event_stream)

    def __iter__(self):
        """Returns an iterator object itself, which allows the object to be iterated.

        Returns:
            iter : object
                    An iterator object representing the iterable.
        """
        return self

    def __next__(self):
        """Returns the next chunk of Byte directly."""
        # Even with "while True" loop the function still behaves like a generator
        # and sends the next new byte chunk.
        while True:
            chunk = next(self.byte_iterator)
            if "PayloadPart" not in chunk:
                # handle API response errors and force terminate.
                handle_stream_errors(chunk)
                # print and move on to next response byte
                print("Unknown event type:" + chunk)
                continue
            return chunk["PayloadPart"]["Bytes"]


class LineIterator(BaseIterator):
    """A helper class for parsing the byte Event Stream input to provide Line iteration."""

    def __init__(self, event_stream):
        """Initialises a LineIterator Iterator object

        Args:
            event_stream: (botocore.eventstream.EventStream): Event Stream object to be iterated.
        """
        super().__init__(event_stream)
        self.byte_iterator = iter(self.event_stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0

    def __iter__(self):
        """Returns an iterator object itself, which allows the object to be iterated.

        Returns:
            iter : object
                    An iterator object representing the iterable.
        """
        return self

    def __next__(self):
        r"""Returns the next Line for an Line iterable.

        The output of the event stream will be in the following format:

        ```
        b'{"outputs": [" a"]}\n'
        b'{"outputs": [" challenging"]}\n'
        b'{"outputs": [" problem"]}\n'
        ...
        ```

        While usually each PayloadPart event from the event stream will contain a byte array
        with a full json, this is not guaranteed and some of the json objects may be split across
        PayloadPart events. For example:
        ```
        {'PayloadPart': {'Bytes': b'{"outputs": '}}
        {'PayloadPart': {'Bytes': b'[" problem"]}\n'}}
        ```

        This class accounts for this by concatenating bytes written via the 'write' function
        and then exposing a method which will return lines (ending with a '\n' character) within
        the buffer via the 'scan_lines' function. It maintains the position of the last read
        position to ensure that previous bytes are not exposed again.

        Returns:
            str: Read and return one line from the event stream.
        """
        # Even with "while True" loop the function still behaves like a generator
        # and sends the next new concatenated line
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == ord("\n"):
                self.read_pos += len(line)
                return line[:-1]
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if "PayloadPart" not in chunk:
                # handle API response errors and force terminate.
                handle_stream_errors(chunk)
                # print and move on to next response byte
                print("Unknown event type:" + chunk)
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk["PayloadPart"]["Bytes"])
