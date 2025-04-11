"""Defines CustomPayloadTranslator class that holds custom serialization/deserialization code"""

from __future__ import absolute_import
import abc
from typing import IO

CONTENT_TYPE = "application/custom"
ACCEPT_TYPE = "application/custom"


class CustomPayloadTranslator(abc.ABC):
    """Abstract base class for handling custom payload serialization and deserialization.

    Provides a skeleton for customization requiring the overriding of the
    `serialize_payload` and `deserialize_payload` methods.

    Args:
        content_type (str): The content type of the endpoint input data.
        accept_type (str): The content type of the data accepted from the endpoint.
    """

    # pylint: disable=E0601
    def __init__(self, content_type: str = CONTENT_TYPE, accept_type: str = ACCEPT_TYPE) -> None:
        # pylint: disable=unused-argument
        self._content_type = content_type
        self._accept_type = accept_type

    @abc.abstractmethod
    def serialize_payload_to_bytes(self, payload: object) -> bytes:
        """Serialize payload into bytes

        Args:
            payload (object): Data to be serialized into bytes.

        Returns:
            bytes: bytes of serialized data
        """

    @abc.abstractmethod
    def deserialize_payload_from_stream(self, stream: IO) -> object:
        """Deserialize stream into object.

        Args:
            stream (IO): Stream of bytes

        Returns:
            object: Deserialized data
        """

    def serialize(self, payload: object, content_type: str = CONTENT_TYPE) -> bytes:
        """Placeholder docstring"""
        # pylint: disable=unused-argument
        return self.serialize_payload_to_bytes(payload)

    def deserialize(self, stream: IO, content_type: str = CONTENT_TYPE) -> object:
        """Placeholder docstring"""
        # pylint: disable=unused-argument
        return self.deserialize_payload_from_stream(stream)

    @property
    def CONTENT_TYPE(self):
        """Placeholder docstring"""
        return self._content_type

    @property
    def ACCEPT(self):
        """Placeholder docstring"""
        return self._accept_type
