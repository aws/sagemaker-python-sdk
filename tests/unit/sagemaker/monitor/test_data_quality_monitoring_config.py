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
from __future__ import absolute_import

from sagemaker.model_monitor import (
    DataQualityMonitoringConfig,
    DataQualityDistributionConstraints,
)

INVALID_CATEGORICAL_DRIFT_METHOD = "Invalid"
CHISQUARED_CATEGORICAL_DRIFT_METHOD = "ChiSquared"
LINFINITY_CATEGORICAL_DRIFT_METHOD = "LInfinity"


def test_valid_distribution_constraints_with_valid_input():
    valid_distribution_constraints = DataQualityDistributionConstraints()
    assert DataQualityDistributionConstraints.valid_distribution_constraints(
        valid_distribution_constraints
    )


def test_valid_distribution_constraints_with_invalid_input():
    invalid_distribution_constraints = DataQualityDistributionConstraints(
        categorical_drift_method=INVALID_CATEGORICAL_DRIFT_METHOD
    )
    assert not DataQualityDistributionConstraints.valid_distribution_constraints(
        invalid_distribution_constraints
    )


def test_valid_categorical_drift_method_with_valid_input():
    assert DataQualityDistributionConstraints.valid_categorical_drift_method(
        CHISQUARED_CATEGORICAL_DRIFT_METHOD
    )
    assert DataQualityDistributionConstraints.valid_categorical_drift_method(
        LINFINITY_CATEGORICAL_DRIFT_METHOD
    )


def test_valid_categorical_drift_method_with_invalid_input():
    assert not DataQualityDistributionConstraints.valid_categorical_drift_method(
        INVALID_CATEGORICAL_DRIFT_METHOD
    )


def test_valid_monitoring_config_with_valid_input():
    valid_monitoring_config = DataQualityMonitoringConfig()
    assert DataQualityMonitoringConfig.valid_monitoring_config(valid_monitoring_config)


def test_valid_monitoring_config_with_invalid_input():
    invalid_monitoring_config = DataQualityMonitoringConfig(
        distribution_constraints=DataQualityDistributionConstraints(
            categorical_drift_method=INVALID_CATEGORICAL_DRIFT_METHOD
        ),
    )
    assert not DataQualityMonitoringConfig.valid_monitoring_config(invalid_monitoring_config)
