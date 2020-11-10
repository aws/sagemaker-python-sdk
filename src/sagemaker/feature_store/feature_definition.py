# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""The Feature Definitions for FeatureStore."""
from __future__ import absolute_import

from enum import Enum
from typing import Dict, Any

import attr

from sagemaker.feature_store.inputs import Config


class FeatureTypeEnum(Enum):
    """Enum of feature types."""

    FRACTIONAL = "Fractional"
    INTEGRAL = "Integral"
    STRING = "String"


@attr.s
class FeatureDefinition(Config):
    """Feature definition.

    Attributes:
        feature_name (str): The name of the feature
        feature_type (FeatureTypeEnum): The type of the feature
    """

    feature_name: str = attr.ib()
    feature_type: FeatureTypeEnum = attr.ib()

    def to_dict(self) -> Dict[str, Any]:
        """Constructs a dictionary based on the attributes"""
        return Config.construct_dict(
            FeatureName=self.feature_name, FeatureType=self.feature_type.value
        )


class FractionalFeatureDefinition(FeatureDefinition):
    """Fractional feature definition.

    Attributes:
        feature_name (str): The name of the feature
        feature_type (FeatureTypeEnum): A `FeatureTypeEnum.FRACTIONAL` type
    """

    def __init__(self, feature_name: str):
        """Constructs an instance of FractionalFeatureDefinition.

        Args:
            feature_name (str): the name of the feature.
        """
        super(FractionalFeatureDefinition, self).__init__(feature_name, FeatureTypeEnum.FRACTIONAL)


class IntegralFeatureDefinition(FeatureDefinition):
    """Fractional feature definition.

    Attributes:
        feature_name (str): the name of the feature.
        feature_type (FeatureTypeEnum): a `FeatureTypeEnum.INTEGRAL` type.
    """

    def __init__(self, feature_name: str):
        """Constructs an instance of IntegralFeatureDefinition.

        Args:
            feature_name (str): the name of the feature.
        """
        super(IntegralFeatureDefinition, self).__init__(feature_name, FeatureTypeEnum.INTEGRAL)


class StringFeatureDefinition(FeatureDefinition):
    """Fractional feature definition.

    Attributes:
        feature_name (str): the name of the feature.
        feature_type (FeatureTypeEnum): a `FeatureTypeEnum.STRING` type.
    """

    def __init__(self, feature_name: str):
        """Constructs an instance of StringFeatureDefinition.

        Args:
            feature_name (str): the name of the feature.
        """
        super(StringFeatureDefinition, self).__init__(feature_name, FeatureTypeEnum.STRING)
