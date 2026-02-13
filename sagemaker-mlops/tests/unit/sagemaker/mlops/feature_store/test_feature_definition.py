# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Unit tests for feature_definition.py"""
import pytest

from sagemaker.mlops.feature_store.feature_definition import (
    FeatureDefinition,
    FeatureTypeEnum,
    CollectionTypeEnum,
    IntegralFeatureDefinition,
    FractionalFeatureDefinition,
    StringFeatureDefinition,
    VectorCollectionType,
    ListCollectionType,
    SetCollectionType,
)


class TestFeatureTypeEnum:
    def test_fractional_value(self):
        assert FeatureTypeEnum.FRACTIONAL.value == "Fractional"

    def test_integral_value(self):
        assert FeatureTypeEnum.INTEGRAL.value == "Integral"

    def test_string_value(self):
        assert FeatureTypeEnum.STRING.value == "String"


class TestCollectionTypeEnum:
    def test_list_value(self):
        assert CollectionTypeEnum.LIST.value == "List"

    def test_set_value(self):
        assert CollectionTypeEnum.SET.value == "Set"

    def test_vector_value(self):
        assert CollectionTypeEnum.VECTOR.value == "Vector"


class TestCollectionTypes:
    def test_list_collection_type(self):
        collection = ListCollectionType()
        assert collection.collection_type == "List"
        assert collection.collection_config is None

    def test_set_collection_type(self):
        collection = SetCollectionType()
        assert collection.collection_type == "Set"
        assert collection.collection_config is None

    def test_vector_collection_type(self):
        collection = VectorCollectionType(dimension=128)
        assert collection.collection_type == "Vector"
        assert collection.collection_config is not None
        assert collection.collection_config.vector_config.dimension == 128


class TestFeatureDefinitionFactories:
    def test_integral_feature_definition(self):
        definition = IntegralFeatureDefinition(feature_name="my_int_feature")
        assert definition.feature_name == "my_int_feature"
        assert definition.feature_type == "Integral"
        assert definition.collection_type is None

    def test_fractional_feature_definition(self):
        definition = FractionalFeatureDefinition(feature_name="my_float_feature")
        assert definition.feature_name == "my_float_feature"
        assert definition.feature_type == "Fractional"
        assert definition.collection_type is None

    def test_string_feature_definition(self):
        definition = StringFeatureDefinition(feature_name="my_string_feature")
        assert definition.feature_name == "my_string_feature"
        assert definition.feature_type == "String"
        assert definition.collection_type is None

    def test_integral_with_list_collection(self):
        definition = IntegralFeatureDefinition(
            feature_name="my_int_list",
            collection_type=ListCollectionType(),
        )
        assert definition.feature_name == "my_int_list"
        assert definition.feature_type == "Integral"
        assert definition.collection_type == "List"

    def test_string_with_set_collection(self):
        definition = StringFeatureDefinition(
            feature_name="my_string_set",
            collection_type=SetCollectionType(),
        )
        assert definition.feature_name == "my_string_set"
        assert definition.feature_type == "String"
        assert definition.collection_type == "Set"

    def test_fractional_with_vector_collection(self):
        definition = FractionalFeatureDefinition(
            feature_name="my_embedding",
            collection_type=VectorCollectionType(dimension=256),
        )
        assert definition.feature_name == "my_embedding"
        assert definition.feature_type == "Fractional"
        assert definition.collection_type == "Vector"
        assert definition.collection_config.vector_config.dimension == 256


class TestFeatureDefinitionSerialization:
    """Test that FeatureDefinition can be serialized (Pydantic model_dump)."""

    def test_simple_definition_serialization(self):
        definition = IntegralFeatureDefinition(feature_name="id")
        # Pydantic model - use model_dump
        data = definition.model_dump(exclude_none=True)
        assert data["feature_name"] == "id"
        assert data["feature_type"] == "Integral"

    def test_collection_definition_serialization(self):
        definition = FractionalFeatureDefinition(
            feature_name="vector",
            collection_type=VectorCollectionType(dimension=10),
        )
        data = definition.model_dump(exclude_none=True)
        assert data["feature_name"] == "vector"
        assert data["feature_type"] == "Fractional"
        assert data["collection_type"] == "Vector"
        assert data["collection_config"]["vector_config"]["dimension"] == 10
