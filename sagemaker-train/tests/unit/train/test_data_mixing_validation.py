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
"""Validation tests for DataMixingConfig."""
from __future__ import absolute_import

import pytest
from pydantic import ValidationError

from sagemaker.train.data_mixing_config import DataMixingConfig
from sagemaker.train.common_utils.data_mixing_utils import validate_data_mixing_categories


class TestCustomerDataPercentRange:
    """Customer data percent range enforcement.

    For any float value outside the range [0, 100], constructing a DataMixingConfig
    with that value as customer_data_percent SHALL raise a ValidationError.
    """

    @pytest.mark.parametrize(
        "invalid_percent",
        [
            -1,
            -0.01,
            100.01,
            200,
            -100,
        ],
        ids=[
            "negative_one",
            "just_below_zero",
            "just_above_hundred",
            "two_hundred",
            "negative_hundred",
        ],
    )
    def test_invalid_customer_data_percent_raises(self, invalid_percent):
        """Values outside [0, 100] must raise ValidationError."""
        with pytest.raises(ValidationError, match="customer_data_percent must be between 0 and 100"):
            DataMixingConfig(customer_data_percent=invalid_percent)


class TestNovaSumInvariant:
    """Nova data percentage sum invariant.

    For any DataMixingConfig where customer_data_percent < 100 and
    nova_data_percentages is provided (non-None), the values must sum to 100.
    When customer_data_percent == 100 or nova_data_percentages is None,
    the config is accepted without error.
    """

    @pytest.mark.parametrize(
        "customer_data_percent,nova_data_percentages",
        [
            # sum < 100 should fail
            (50.0, {"code": 30.0, "math": 20.0}),
            (0.0, {"en-entertainment": 10.0, "code": 10.0, "math": 10.0}),
            (99.0, {"code": 50.0, "math": 49.0}),
        ],
        ids=[
            "sum_50_less_than_100",
            "sum_30_less_than_100",
            "sum_99_less_than_100",
        ],
    )
    def test_nova_sum_less_than_100_raises(self, customer_data_percent, nova_data_percentages):
        """When customer_data_percent < 100 and nova percentages sum < 100, raise ValidationError."""
        with pytest.raises(ValidationError, match="nova_data_percentages must sum to 100"):
            DataMixingConfig(
                customer_data_percent=customer_data_percent,
                nova_data_percentages=nova_data_percentages,
            )

    @pytest.mark.parametrize(
        "customer_data_percent,nova_data_percentages",
        [
            # sum > 100 should fail
            (50.0, {"code": 60.0, "math": 50.0}),
            (0.0, {"en-entertainment": 50.0, "code": 50.0, "math": 1.0}),
            (10.0, {"code": 100.0, "math": 0.01}),
        ],
        ids=[
            "sum_110_greater_than_100",
            "sum_101_greater_than_100",
            "sum_100.01_greater_than_100",
        ],
    )
    def test_nova_sum_greater_than_100_raises(self, customer_data_percent, nova_data_percentages):
        """When customer_data_percent < 100 and nova percentages sum > 100, raise ValidationError."""
        with pytest.raises(ValidationError, match="nova_data_percentages must sum to 100"):
            DataMixingConfig(
                customer_data_percent=customer_data_percent,
                nova_data_percentages=nova_data_percentages,
            )

    @pytest.mark.parametrize(
        "customer_data_percent,nova_data_percentages",
        [
            # valid sum = 100 should pass
            (50.0, {"code": 50.0, "math": 50.0}),
            (0.0, {"en-entertainment": 25.0, "code": 25.0, "math": 25.0, "en-scientific": 25.0}),
            (99.0, {"code": 100.0}),
            (30.0, {"code": 33.33, "math": 33.33, "en-scientific": 33.34}),
        ],
        ids=[
            "two_categories_sum_100",
            "four_categories_sum_100",
            "single_category_100",
            "three_categories_sum_100",
        ],
    )
    def test_nova_sum_equals_100_valid(self, customer_data_percent, nova_data_percentages):
        """When customer_data_percent < 100 and nova percentages sum to 100, config is valid."""
        config = DataMixingConfig(
            customer_data_percent=customer_data_percent,
            nova_data_percentages=nova_data_percentages,
        )
        assert config.customer_data_percent == customer_data_percent
        assert config.nova_data_percentages == nova_data_percentages

    @pytest.mark.parametrize(
        "nova_data_percentages",
        [
            # customer_data_percent = 100 bypasses sum check
            {"code": 30.0, "math": 20.0},  # sum = 50, but should pass
            {"code": 60.0, "math": 60.0},  # sum = 120, but should pass
            {"code": 0.0},  # sum = 0, but should pass
            {},  # empty dict, sum = 0, but should pass
            {"en-entertainment": 25.0, "code": 25.0, "math": 25.0, "en-scientific": 25.0},  # sum = 100
        ],
        ids=[
            "sum_50_bypassed",
            "sum_120_bypassed",
            "sum_0_bypassed",
            "empty_dict_bypassed",
            "sum_100_bypassed",
        ],
    )
    def test_customer_100_bypasses_sum_check(self, nova_data_percentages):
        """When customer_data_percent == 100, any nova_data_percentages sum is accepted."""
        config = DataMixingConfig(
            customer_data_percent=100.0,
            nova_data_percentages=nova_data_percentages,
        )
        assert config.customer_data_percent == 100.0
        assert config.nova_data_percentages == nova_data_percentages

    @pytest.mark.parametrize(
        "customer_data_percent",
        [
            0.0,
            50.0,
            99.0,
            100.0,
        ],
        ids=[
            "customer_0_none_bypassed",
            "customer_50_none_bypassed",
            "customer_99_none_bypassed",
            "customer_100_none_bypassed",
        ],
    )
    def test_none_nova_percentages_bypasses_sum_check(self, customer_data_percent):
        """When nova_data_percentages is None, sum check is bypassed regardless of customer_data_percent."""
        config = DataMixingConfig(
            customer_data_percent=customer_data_percent,
            nova_data_percentages=None,
        )
        assert config.customer_data_percent == customer_data_percent
        assert config.nova_data_percentages is None


class TestNovaCategoryValueRange:
    """Nova data category value range enforcement.

    For any dictionary provided as nova_data_percentages containing at least one
    value outside the range [0, 100], constructing a DataMixingConfig SHALL raise
    a ValidationError.
    """

    @pytest.mark.parametrize(
        "nova_data_percentages",
        [
            {"code": -1.0, "math": 101.0},
            {"en-entertainment": -50.0, "code": 75.0, "math": 75.0},
            {"code": 200.0},
            {"math": -1.0, "code": 101.0},
            {"en-scientific": 101.0, "code": -0.01},
        ],
        ids=[
            "negative_and_over_100",
            "negative_50",
            "value_200",
            "negative_1_and_101",
            "over_100_and_slightly_negative",
        ],
    )
    def test_category_value_outside_range_raises(self, nova_data_percentages):
        """When any nova category value is outside [0, 100], raise ValidationError."""
        with pytest.raises(
            ValidationError,
            match="Each nova data category percent must be between 0 and 100 inclusive",
        ):
            DataMixingConfig(
                customer_data_percent=100.0,
                nova_data_percentages=nova_data_percentages,
            )


class TestSerializationRoundTrip:
    """Serialization round-trip.

    For any valid DataMixingConfig instance, serializing via to_recipe_config()
    and then reconstructing via from_recipe_config() SHALL produce a DataMixingConfig
    that is equal to the original.
    """

    @pytest.mark.parametrize(
        "config",
        [
            DataMixingConfig(
                customer_data_percent=50.0,
                nova_data_percentages={"code": 50.0, "math": 50.0},
            ),
            DataMixingConfig(
                customer_data_percent=0.0,
                nova_data_percentages={
                    "en-entertainment": 25.0,
                    "code": 25.0,
                    "math": 25.0,
                    "en-scientific": 25.0,
                },
            ),
            DataMixingConfig(
                customer_data_percent=100.0,
                nova_data_percentages={"code": 60.0, "math": 40.0},
            ),
            DataMixingConfig(
                customer_data_percent=75.5,
                nova_data_percentages={"code": 100.0},
            ),
            DataMixingConfig(
                customer_data_percent=33.33,
                nova_data_percentages={"a": 33.33, "b": 33.33, "c": 33.34},
            ),
        ],
        ids=[
            "two_categories_50_50",
            "four_categories_equal_split",
            "customer_100_with_nova_categories",
            "single_category_100",
            "three_categories_precise_sum",
        ],
    )
    def test_round_trip_with_nova_percentages(self, config):
        """Serializing and deserializing a config with nova_data_percentages produces an equal config."""
        serialized = config.to_recipe_config()
        reconstructed = DataMixingConfig.from_recipe_config(serialized)
        assert reconstructed == config

    def test_round_trip_with_none_nova_percentages(self):
        """Config with nova_data_percentages=None serializes to empty nova_data and round-trips correctly.

        Note: When nova_data_percentages is None, to_recipe_config() produces an empty
        nova_data dict, and from_recipe_config() interprets that as None. This preserves
        the semantic equivalence.
        """
        config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages=None,
        )
        serialized = config.to_recipe_config()
        reconstructed = DataMixingConfig.from_recipe_config(serialized)
        assert reconstructed == config

    @pytest.mark.parametrize(
        "customer_data_percent",
        [0.0, 25.0, 50.0, 75.0, 100.0],
        ids=["0_percent", "25_percent", "50_percent", "75_percent", "100_percent"],
    )
    def test_round_trip_various_customer_percents(self, customer_data_percent):
        """Round-trip preserves customer_data_percent across the valid range."""
        nova_percentages = {"code": 50.0, "math": 50.0}
        if customer_data_percent < 100:
            config = DataMixingConfig(
                customer_data_percent=customer_data_percent,
                nova_data_percentages=nova_percentages,
            )
        else:
            config = DataMixingConfig(
                customer_data_percent=customer_data_percent,
                nova_data_percentages=nova_percentages,
            )
        serialized = config.to_recipe_config()
        reconstructed = DataMixingConfig.from_recipe_config(serialized)
        assert reconstructed == config


class TestCategoryKeysValidatedAgainstRecipe:
    """Category keys validated against recipe.

    For any DataMixingConfig and recipe category set, if the user-provided category
    keys contain any key not present in the recipe category set, category validation
    SHALL raise a ValueError identifying the invalid keys. If all user-provided keys
    are a subset of recipe keys, validation SHALL succeed.
    """

    RECIPE_CATEGORIES = {
        "en-entertainment": 25.0,
        "code": 25.0,
        "math": 25.0,
        "en-scientific": 25.0,
    }

    @pytest.mark.parametrize(
        "nova_data_percentages",
        [
            {"invalid-category": 100.0},
            {"en-entertainment": 50.0, "nonexistent": 50.0},
            {"code": 25.0, "math": 25.0, "fiction": 25.0, "poetry": 25.0},
            {"totally-wrong": 50.0, "also-wrong": 50.0},
        ],
        ids=[
            "single_invalid_key",
            "one_valid_one_invalid_key",
            "mix_of_valid_and_invalid_keys",
            "all_invalid_keys",
        ],
    )
    def test_invalid_category_keys_raise_value_error(self, nova_data_percentages):
        """User-provided keys not in recipe raise ValueError listing valid categories."""
        config = DataMixingConfig(
            customer_data_percent=100.0,
            nova_data_percentages=nova_data_percentages,
        )
        with pytest.raises(ValueError, match="Unrecognized data mixing categories"):
            validate_data_mixing_categories(config, self.RECIPE_CATEGORIES)

    @pytest.mark.parametrize(
        "nova_data_percentages",
        [
            {"en-entertainment": 100.0},
            {"code": 50.0, "math": 50.0},
            {"en-entertainment": 25.0, "code": 25.0, "math": 25.0, "en-scientific": 25.0},
        ],
        ids=[
            "single_valid_key",
            "two_valid_keys",
            "all_recipe_keys",
        ],
    )
    def test_valid_subset_of_recipe_keys_passes(self, nova_data_percentages):
        """User-provided keys that are a subset of recipe keys pass without error."""
        config = DataMixingConfig(
            customer_data_percent=100.0,
            nova_data_percentages=nova_data_percentages,
        )
        result = validate_data_mixing_categories(config, self.RECIPE_CATEGORIES)
        # Returned config should have all recipe categories populated
        assert set(result.nova_data_percentages.keys()) == set(self.RECIPE_CATEGORIES.keys())
        # User-provided values are preserved
        for key, value in nova_data_percentages.items():
            assert result.nova_data_percentages[key] == value

    def test_none_nova_percentages_uses_recipe_defaults(self):
        """When nova_data_percentages is None, all recipe defaults are used."""
        config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages=None,
        )
        result = validate_data_mixing_categories(config, self.RECIPE_CATEGORIES)
        assert result.nova_data_percentages == self.RECIPE_CATEGORIES


class TestUnspecifiedCategoriesZeroed:
    """Unspecified categories zeroed when overrides provided.

    For any DataMixingConfig whose nova_data_percentages is provided and whose keys
    are a subset of the recipe categories, after category validation and merging,
    the resulting config SHALL contain all recipe category keys — with user-provided
    values preserved and all unspecified categories set to 0.
    """

    @pytest.mark.parametrize(
        "user_percentages,recipe_categories,expected_nova",
        [
            # User specifies a subset → unspecified ones zeroed
            (
                {"code": 100.0},
                {"code": 25.0, "math": 25.0, "en-entertainment": 25.0, "en-scientific": 25.0},
                {"code": 100.0, "math": 0.0, "en-entertainment": 0.0, "en-scientific": 0.0},
            ),
            # User specifies two of four categories → other two zeroed
            (
                {"code": 60.0, "math": 40.0},
                {"code": 25.0, "math": 25.0, "en-entertainment": 25.0, "en-scientific": 25.0},
                {"code": 60.0, "math": 40.0, "en-entertainment": 0.0, "en-scientific": 0.0},
            ),
            # User specifies all recipe categories → result matches user values exactly
            (
                {"code": 30.0, "math": 20.0, "en-entertainment": 25.0, "en-scientific": 25.0},
                {"code": 25.0, "math": 25.0, "en-entertainment": 25.0, "en-scientific": 25.0},
                {"code": 30.0, "math": 20.0, "en-entertainment": 25.0, "en-scientific": 25.0},
            ),
            # Multiple recipe categories with partial user overrides (3 of 5)
            (
                {"code": 50.0, "math": 30.0, "science": 20.0},
                {"code": 20.0, "math": 20.0, "science": 20.0, "history": 20.0, "art": 20.0},
                {"code": 50.0, "math": 30.0, "science": 20.0, "history": 0.0, "art": 0.0},
            ),
            # Single recipe category, user specifies it → no zeroing needed
            (
                {"code": 100.0},
                {"code": 50.0},
                {"code": 100.0},
            ),
            # User specifies one of three categories
            (
                {"en-entertainment": 100.0},
                {"en-entertainment": 40.0, "code": 30.0, "math": 30.0},
                {"en-entertainment": 100.0, "code": 0.0, "math": 0.0},
            ),
        ],
        ids=[
            "subset_one_of_four_zeroes_rest",
            "subset_two_of_four_zeroes_rest",
            "all_categories_specified_no_zeroing",
            "partial_three_of_five_zeroes_rest",
            "single_category_recipe_no_zeroing",
            "one_of_three_zeroes_rest",
        ],
    )
    def test_unspecified_categories_zeroed(
        self, user_percentages, recipe_categories, expected_nova
    ):
        """After validation, all recipe categories are present with user values preserved and missing ones set to 0."""
        # Use customer_data_percent=100 to bypass the nova sum validation at construction
        config = DataMixingConfig(
            customer_data_percent=100.0,
            nova_data_percentages=user_percentages,
        )

        result = validate_data_mixing_categories(config, recipe_categories)

        # Result should contain ALL recipe categories
        assert set(result.nova_data_percentages.keys()) == set(recipe_categories.keys())

        # User-specified values should be preserved exactly
        for key, value in user_percentages.items():
            assert result.nova_data_percentages[key] == value

        # Unspecified categories should be 0.0
        unspecified = set(recipe_categories.keys()) - set(user_percentages.keys())
        for key in unspecified:
            assert result.nova_data_percentages[key] == 0.0

        # Full expected output should match
        assert result.nova_data_percentages == expected_nova

        # customer_data_percent should be preserved
        assert result.customer_data_percent == config.customer_data_percent
