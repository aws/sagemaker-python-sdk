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
"""Feature Definitions for FeatureStore."""
from __future__ import absolute_import

from enum import Enum
from typing import Optional, Union

from sagemaker.core.shapes import (
    FeatureDefinition,
    CollectionConfig,
    VectorConfig,
)

class FeatureTypeEnum(Enum):
    """Feature data types: Fractional, Integral, or String."""

    FRACTIONAL = "Fractional"
    INTEGRAL = "Integral"
    STRING = "String"

class CollectionTypeEnum(Enum):
    """Collection types: List, Set, or Vector."""

    LIST = "List"
    SET = "Set"
    VECTOR = "Vector"

class ListCollectionType:
    """List collection type."""

    collection_type = CollectionTypeEnum.LIST.value
    collection_config = None

class SetCollectionType:
    """Set collection type."""

    collection_type = CollectionTypeEnum.SET.value
    collection_config = None

class VectorCollectionType:
    """Vector collection type with dimension."""

    collection_type = CollectionTypeEnum.VECTOR.value

    def __init__(self, dimension: int):
        self.collection_config = CollectionConfig(
            vector_config=VectorConfig(dimension=dimension)
        )

CollectionType = Union[ListCollectionType, SetCollectionType, VectorCollectionType]

def _create_feature_definition(
        feature_name: str,
        feature_type: FeatureTypeEnum,
        collection_type: Optional[CollectionType] = None,
) -> FeatureDefinition:
    """Internal helper to create FeatureDefinition from collection type."""
    return FeatureDefinition(
        feature_name=feature_name,
        feature_type=feature_type.value,
        collection_type=collection_type.collection_type if collection_type else None,
        collection_config=collection_type.collection_config if collection_type else None,
    )

def FractionalFeatureDefinition(
        feature_name: str,
        collection_type: Optional[CollectionType] = None,
) -> FeatureDefinition:
    """Create a feature definition with Fractional type."""
    return _create_feature_definition(feature_name, FeatureTypeEnum.FRACTIONAL, collection_type)

def IntegralFeatureDefinition(
        feature_name: str,
        collection_type: Optional[CollectionType] = None,
) -> FeatureDefinition:
    """Create a feature definition with Integral type."""
    return _create_feature_definition(feature_name, FeatureTypeEnum.INTEGRAL, collection_type)

def StringFeatureDefinition(
        feature_name: str,
        collection_type: Optional[CollectionType] = None,
) -> FeatureDefinition:
    """Create a feature definition with String type."""
    return _create_feature_definition(feature_name, FeatureTypeEnum.STRING, collection_type)

__all__ = [
    "FeatureDefinition",
    "FeatureTypeEnum",
    "CollectionTypeEnum",
    "ListCollectionType",
    "SetCollectionType",
    "VectorCollectionType",
    "FractionalFeatureDefinition",
    "IntegralFeatureDefinition",
    "StringFeatureDefinition",
]
