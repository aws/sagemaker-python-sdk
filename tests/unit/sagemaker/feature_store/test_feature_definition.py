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
# language governing permissions and limitations under the License.
from __future__ import absolute_import

from sagemaker.feature_store.feature_definition import (
    FeatureDefinition,
    FeatureTypeEnum,
    IntegralFeatureDefinition,
)


def ordered(obj):
    """Helper function for dict comparison"""
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(ordered(x) for x in obj)
    else:
        return obj


def test_feature_definition():
    definition = FeatureDefinition(feature_name="MyFeature", feature_type=FeatureTypeEnum.INTEGRAL)
    assert ordered(definition.to_dict()) == ordered(
        {
            "FeatureName": "MyFeature",
            "FeatureType": "Integral",
        }
    )


def test_integral_feature_definition():
    definition = IntegralFeatureDefinition(feature_name="MyFeature")
    assert ordered(definition.to_dict()) == ordered(
        {
            "FeatureName": "MyFeature",
            "FeatureType": "Integral",
        }
    )
