"""This module stores types related to SageMaker JumpStart."""
from __future__ import absolute_import
import re
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

class JumpStartDataHolderType:
    """Base class for many JumpStart types.

    Allows objects to be added to dicts and sets,
    and improves string representation. This class overrides the ``__eq__``
    and ``__hash__`` methods so that different objects with the same attributes/types
    can be compared.
    """

    __slots__: List[str] = []

    _non_serializable_slots: List[str] = []

    def __eq__(self, other: Any) -> bool:
        """Returns True if ``other`` is of the same type and has all attributes equal.

        Args:
            other (Any): Other object to which to compare this object.
        """

        if not isinstance(other, type(self)):
            return False
        if getattr(other, "__slots__", None) is None:
            return False
        if self.__slots__ != other.__slots__:
            return False
        for attribute in self.__slots__:
            if (hasattr(self, attribute) and not hasattr(other, attribute)) or (
                hasattr(other, attribute) and not hasattr(self, attribute)
            ):
                return False
            if hasattr(self, attribute) and hasattr(other, attribute):
                if getattr(self, attribute) != getattr(other, attribute):
                    return False
        return True

class JumpStartLaunchedRegionInfo(JumpStartDataHolderType):
    """Data class for launched region info."""

    __slots__ = ["content_bucket", "region_name", "gated_content_bucket", "neo_content_bucket"]

    def __init__(
        self,
        content_bucket: str,
        region_name: str,
        gated_content_bucket: Optional[str] = None,
        neo_content_bucket: Optional[str] = None,
    ):
        """Instantiates JumpStartLaunchedRegionInfo object.

        Args:
            content_bucket (str): Name of JumpStart s3 content bucket associated with region.
            region_name (str): Name of JumpStart launched region.
            gated_content_bucket (Optional[str[]): Name of JumpStart gated s3 content bucket
                optionally associated with region.
            neo_content_bucket (Optional[str]): Name of Neo service s3 content bucket
                optionally associated with region.
        """
        self.content_bucket = content_bucket
        self.gated_content_bucket = gated_content_bucket
        self.region_name = region_name
        self.neo_content_bucket = neo_content_bucket
