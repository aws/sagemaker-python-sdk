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
"""Configuration for blending customer training data with Nova curated datasets."""
from __future__ import absolute_import

from typing import Any, Dict, Optional

from pydantic import BaseModel, field_validator, model_validator


class DataMixingConfig(BaseModel):
    """Configuration for blending customer training data with Nova curated datasets.

    Attributes:
        customer_data_percent: Percentage of total training mix that is customer data (0-100).
        nova_data_percentages: Optional per-category percentage distribution within the Nova
            data portion. Keys are category names (e.g., "en-entertainment", "code", "math").
            Values must each be 0-100 and must sum to 100 when provided.
            If None, all default percentages from the recipe template are used at submission time.
            If provided, unspecified recipe categories are set to 0.
    """

    customer_data_percent: float
    nova_data_percentages: Optional[Dict[str, float]] = None

    @field_validator("customer_data_percent")
    @classmethod
    def _validate_customer_percent(cls, v: float) -> float:
        """Validate that customer_data_percent is between 0 and 100 inclusive."""
        if not (0 <= v <= 100):
            raise ValueError(
                f"customer_data_percent must be between 0 and 100 inclusive, got {v}"
            )
        return v

    @field_validator("nova_data_percentages")
    @classmethod
    def _validate_category_ranges(
        cls, v: Optional[Dict[str, float]]
    ) -> Optional[Dict[str, float]]:
        """Validate that each nova data category percentage is between 0 and 100 inclusive."""
        if v is None:
            return v
        for category, percent in v.items():
            if not (0 <= percent <= 100):
                raise ValueError(
                    f"Each nova data category percent must be between 0 and 100 inclusive, "
                    f"but '{category}' has value {percent}"
                )
        return v

    @model_validator(mode="after")
    def _validate_nova_sum(self) -> "DataMixingConfig":
        """Validate that nova_data_percentages sum to 100 when provided and customer_data_percent < 100."""
        if self.customer_data_percent < 100 and self.nova_data_percentages is not None:
            total = sum(self.nova_data_percentages.values())
            if abs(total - 100.0) > 1e-9:
                raise ValueError(
                    f"nova_data_percentages must sum to 100 when customer_data_percent < 100, "
                    f"but they sum to {total}"
                )
        return self

    def to_recipe_config(self) -> Dict[str, Any]:
        """Serialize to the recipe configuration format.

        Returns a dictionary structured for the data_mixing.sources section
        of the training recipe format.

        Returns:
            Dictionary with structure:
            {
                "customer_data": {"percent": <float>},
                "nova_data": {"<category>": {"percent": <float>}, ...}
            }

            When nova_data_percentages is None, nova_data will be an empty dictionary.
        """
        nova_data: Dict[str, Any] = {}
        if self.nova_data_percentages is not None:
            nova_data = {
                category: {"percent": percent}
                for category, percent in self.nova_data_percentages.items()
            }
        return {
            "customer_data": {"percent": self.customer_data_percent},
            "nova_data": nova_data,
        }

    def to_hyperparameters(self) -> Dict[str, str]:
        """Serialize to flat hyperparameter format for SMTJ serverless.

        Returns a dictionary of hyperparameter names to string values, using the
        naming convention expected by the serverless training platform:
        - ``customer_data_percent`` for the customer data portion
        - ``nova_<category>_percent`` for each Nova data category

        When nova_data_percentages is None, only customer_data_percent is returned.

        Returns:
            Dictionary with flat hyperparameter keys and string values, e.g.:
            {
                "customer_data_percent": "70",
                "nova_code_percent": "30",
                "nova_math_percent": "20",
                ...
            }
        """
        params: Dict[str, str] = {
            "customer_data_percent": str(int(self.customer_data_percent))
            if self.customer_data_percent == int(self.customer_data_percent)
            else str(self.customer_data_percent),
        }
        if self.nova_data_percentages is not None:
            for category, percent in self.nova_data_percentages.items():
                value = str(int(percent)) if percent == int(percent) else str(percent)
                params[f"nova_{category}_percent"] = value
        return params

    @classmethod
    def from_recipe_config(cls, config: Dict[str, Any]) -> "DataMixingConfig":
        """Reconstruct a DataMixingConfig from the serialized recipe config format.

        Args:
            config: Dictionary in the to_recipe_config() output format, with
                "customer_data" containing a "percent" field and "nova_data"
                containing per-category entries each with a "percent" field.

        Returns:
            A DataMixingConfig instance equivalent to the one that produced the config.
        """
        customer_percent = config["customer_data"]["percent"]
        nova_data = config.get("nova_data", {})
        nova_percentages: Optional[Dict[str, float]] = None
        if nova_data:
            nova_percentages = {
                category: entry["percent"]
                for category, entry in nova_data.items()
            }
        return cls(
            customer_data_percent=customer_percent,
            nova_data_percentages=nova_percentages,
        )
