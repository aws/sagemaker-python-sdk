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
"""Unit tests for DataMixingConfig class."""
from __future__ import absolute_import

import pytest
from pydantic import ValidationError

from sagemaker.train.data_mixing_config import DataMixingConfig


class TestDataMixingConfigConstruction:
    """Test valid construction of DataMixingConfig with typical values."""

    def test_valid_construction_typical_values(self):
        """DataMixingConfig can be constructed with typical customer_data_percent and nova categories."""
        config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages={"code": 60.0, "math": 40.0},
        )
        assert config.customer_data_percent == 50.0
        assert config.nova_data_percentages == {"code": 60.0, "math": 40.0}

    def test_valid_construction_multiple_categories(self):
        """DataMixingConfig supports multiple nova data categories."""
        config = DataMixingConfig(
            customer_data_percent=30.0,
            nova_data_percentages={
                "en-entertainment": 25.0,
                "code": 25.0,
                "math": 25.0,
                "en-scientific": 25.0,
            },
        )
        assert config.customer_data_percent == 30.0
        assert len(config.nova_data_percentages) == 4

    def test_valid_construction_single_category(self):
        """DataMixingConfig works with a single nova data category at 100%."""
        config = DataMixingConfig(
            customer_data_percent=20.0,
            nova_data_percentages={"code": 100.0},
        )
        assert config.customer_data_percent == 20.0
        assert config.nova_data_percentages == {"code": 100.0}


class TestDataMixingConfigBoundaryValues:
    """Test boundary values for DataMixingConfig fields."""

    def test_customer_data_percent_zero(self):
        """customer_data_percent of 0 is valid (all Nova data)."""
        config = DataMixingConfig(
            customer_data_percent=0.0,
            nova_data_percentages={"code": 100.0},
        )
        assert config.customer_data_percent == 0.0

    def test_customer_data_percent_hundred(self):
        """customer_data_percent of 100 is valid (all customer data)."""
        config = DataMixingConfig(
            customer_data_percent=100.0,
        )
        assert config.customer_data_percent == 100.0

    def test_nova_category_value_zero(self):
        """Nova category value of 0 is valid."""
        config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages={"code": 0.0, "math": 100.0},
        )
        assert config.nova_data_percentages["code"] == 0.0

    def test_nova_category_value_hundred(self):
        """Nova category value of 100 is valid."""
        config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages={"code": 100.0},
        )
        assert config.nova_data_percentages["code"] == 100.0

    def test_nova_categories_exactly_sum_to_100(self):
        """Nova categories that sum exactly to 100 are valid."""
        config = DataMixingConfig(
            customer_data_percent=25.0,
            nova_data_percentages={"a": 33.33, "b": 33.33, "c": 33.34},
        )
        total = sum(config.nova_data_percentages.values())
        assert abs(total - 100.0) < 1e-9


class TestDataMixingConfigTypeCoercion:
    """Test type coercion from int to float for DataMixingConfig fields."""

    def test_customer_data_percent_int_coerced_to_float(self):
        """Integer customer_data_percent is coerced to float by Pydantic."""
        config = DataMixingConfig(
            customer_data_percent=50,
            nova_data_percentages={"code": 100.0},
        )
        assert isinstance(config.customer_data_percent, float)
        assert config.customer_data_percent == 50.0

    def test_nova_category_int_values_coerced_to_float(self):
        """Integer nova data category values are coerced to float by Pydantic."""
        config = DataMixingConfig(
            customer_data_percent=50,
            nova_data_percentages={"code": 60, "math": 40},
        )
        assert isinstance(config.nova_data_percentages["code"], float)
        assert isinstance(config.nova_data_percentages["math"], float)
        assert config.nova_data_percentages["code"] == 60.0
        assert config.nova_data_percentages["math"] == 40.0


class TestDataMixingConfigInvalidCustomerDataPercent:
    """Test ValidationError for invalid customer_data_percent values."""

    def test_negative_customer_data_percent_raises(self):
        """Negative customer_data_percent raises ValidationError."""
        with pytest.raises(ValidationError, match="customer_data_percent must be between 0 and 100"):
            DataMixingConfig(customer_data_percent=-1.0)

    def test_over_hundred_customer_data_percent_raises(self):
        """customer_data_percent over 100 raises ValidationError."""
        with pytest.raises(ValidationError, match="customer_data_percent must be between 0 and 100"):
            DataMixingConfig(customer_data_percent=100.01)

    def test_large_negative_customer_data_percent_raises(self):
        """Large negative customer_data_percent raises ValidationError."""
        with pytest.raises(ValidationError, match="customer_data_percent must be between 0 and 100"):
            DataMixingConfig(customer_data_percent=-500.0)

    def test_non_numeric_customer_data_percent_raises(self):
        """Non-numeric customer_data_percent raises ValidationError."""
        with pytest.raises(ValidationError):
            DataMixingConfig(customer_data_percent="not_a_number")


class TestDataMixingConfigInvalidCategoryValues:
    """Test ValidationError for invalid nova_data_percentages category values."""

    def test_negative_category_value_raises(self):
        """Negative nova category value raises ValidationError."""
        with pytest.raises(
            ValidationError,
            match="Each nova data category percent must be between 0 and 100 inclusive",
        ):
            DataMixingConfig(
                customer_data_percent=100.0,
                nova_data_percentages={"code": -1.0},
            )

    def test_over_hundred_category_value_raises(self):
        """Nova category value over 100 raises ValidationError."""
        with pytest.raises(
            ValidationError,
            match="Each nova data category percent must be between 0 and 100 inclusive",
        ):
            DataMixingConfig(
                customer_data_percent=100.0,
                nova_data_percentages={"code": 101.0},
            )

    def test_multiple_invalid_category_values_raises(self):
        """Multiple invalid nova category values raises ValidationError."""
        with pytest.raises(
            ValidationError,
            match="Each nova data category percent must be between 0 and 100 inclusive",
        ):
            DataMixingConfig(
                customer_data_percent=100.0,
                nova_data_percentages={"code": -5.0, "math": 200.0},
            )


class TestDataMixingConfigNonSummingCategories:
    """Test ValidationError for nova_data_percentages that don't sum to 100."""

    def test_nova_sum_below_100_raises(self):
        """Nova data percentages summing to less than 100 raises ValidationError."""
        with pytest.raises(ValidationError, match="nova_data_percentages must sum to 100"):
            DataMixingConfig(
                customer_data_percent=50.0,
                nova_data_percentages={"code": 30.0, "math": 20.0},
            )

    def test_nova_sum_above_100_raises(self):
        """Nova data percentages summing to more than 100 raises ValidationError."""
        with pytest.raises(ValidationError, match="nova_data_percentages must sum to 100"):
            DataMixingConfig(
                customer_data_percent=50.0,
                nova_data_percentages={"code": 60.0, "math": 50.0},
            )

    def test_customer_100_bypasses_sum_validation(self):
        """When customer_data_percent is 100, nova sum validation is skipped."""
        config = DataMixingConfig(
            customer_data_percent=100.0,
            nova_data_percentages={"code": 30.0, "math": 20.0},
        )
        assert config.customer_data_percent == 100.0
        assert config.nova_data_percentages == {"code": 30.0, "math": 20.0}


class TestDataMixingConfigNoneNovaPercentages:
    """Test that None nova_data_percentages is accepted."""

    def test_none_nova_percentages_default(self):
        """When nova_data_percentages is not provided, it defaults to None."""
        config = DataMixingConfig(customer_data_percent=50.0)
        assert config.nova_data_percentages is None

    def test_explicit_none_nova_percentages(self):
        """Explicitly setting nova_data_percentages to None is valid."""
        config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages=None,
        )
        assert config.nova_data_percentages is None

    def test_none_nova_percentages_any_customer_percent(self):
        """None nova_data_percentages is valid for any customer_data_percent value."""
        for percent in [0.0, 25.0, 50.0, 75.0, 100.0]:
            config = DataMixingConfig(
                customer_data_percent=percent,
                nova_data_percentages=None,
            )
            assert config.nova_data_percentages is None


class TestDataMixingConfigToRecipeConfig:
    """Test to_recipe_config() output format."""

    def test_to_recipe_config_structure(self):
        """to_recipe_config() returns correct dictionary structure."""
        config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages={"code": 60.0, "math": 40.0},
        )
        result = config.to_recipe_config()

        assert "customer_data" in result
        assert "nova_data" in result
        assert result["customer_data"] == {"percent": 50.0}
        assert result["nova_data"] == {
            "code": {"percent": 60.0},
            "math": {"percent": 40.0},
        }

    def test_to_recipe_config_preserves_category_names(self):
        """to_recipe_config() preserves category names in nova_data."""
        config = DataMixingConfig(
            customer_data_percent=30.0,
            nova_data_percentages={
                "en-entertainment": 25.0,
                "code": 25.0,
                "math": 25.0,
                "en-scientific": 25.0,
            },
        )
        result = config.to_recipe_config()

        for category in ["en-entertainment", "code", "math", "en-scientific"]:
            assert category in result["nova_data"]
            assert result["nova_data"][category] == {"percent": 25.0}

    def test_to_recipe_config_none_nova_returns_empty_dict(self):
        """to_recipe_config() with None nova_data_percentages returns empty nova_data dict."""
        config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages=None,
        )
        result = config.to_recipe_config()

        assert result["customer_data"] == {"percent": 50.0}
        assert result["nova_data"] == {}

    def test_to_recipe_config_single_category(self):
        """to_recipe_config() works with a single nova data category."""
        config = DataMixingConfig(
            customer_data_percent=75.0,
            nova_data_percentages={"code": 100.0},
        )
        result = config.to_recipe_config()

        assert result["nova_data"] == {"code": {"percent": 100.0}}


class TestDataMixingConfigFromRecipeConfig:
    """Test from_recipe_config() reconstruction."""

    def test_from_recipe_config_basic(self):
        """from_recipe_config() reconstructs a DataMixingConfig from recipe format."""
        recipe_config = {
            "customer_data": {"percent": 50.0},
            "nova_data": {
                "code": {"percent": 60.0},
                "math": {"percent": 40.0},
            },
        }
        config = DataMixingConfig.from_recipe_config(recipe_config)

        assert config.customer_data_percent == 50.0
        assert config.nova_data_percentages == {"code": 60.0, "math": 40.0}

    def test_from_recipe_config_empty_nova_data(self):
        """from_recipe_config() with empty nova_data produces None nova_data_percentages."""
        recipe_config = {
            "customer_data": {"percent": 75.0},
            "nova_data": {},
        }
        config = DataMixingConfig.from_recipe_config(recipe_config)

        assert config.customer_data_percent == 75.0
        assert config.nova_data_percentages is None

    def test_from_recipe_config_round_trip(self):
        """from_recipe_config(config.to_recipe_config()) produces an equivalent config."""
        original = DataMixingConfig(
            customer_data_percent=40.0,
            nova_data_percentages={"en-entertainment": 50.0, "code": 50.0},
        )
        serialized = original.to_recipe_config()
        reconstructed = DataMixingConfig.from_recipe_config(serialized)

        assert reconstructed == original

    def test_from_recipe_config_multiple_categories(self):
        """from_recipe_config() correctly handles multiple nova data categories."""
        recipe_config = {
            "customer_data": {"percent": 20.0},
            "nova_data": {
                "en-entertainment": {"percent": 25.0},
                "code": {"percent": 25.0},
                "math": {"percent": 25.0},
                "en-scientific": {"percent": 25.0},
            },
        }
        config = DataMixingConfig.from_recipe_config(recipe_config)

        assert config.customer_data_percent == 20.0
        assert config.nova_data_percentages == {
            "en-entertainment": 25.0,
            "code": 25.0,
            "math": 25.0,
            "en-scientific": 25.0,
        }

    def test_from_recipe_config_missing_nova_data_key(self):
        """from_recipe_config() handles missing nova_data key gracefully."""
        recipe_config = {
            "customer_data": {"percent": 100.0},
        }
        config = DataMixingConfig.from_recipe_config(recipe_config)

        assert config.customer_data_percent == 100.0
        assert config.nova_data_percentages is None
