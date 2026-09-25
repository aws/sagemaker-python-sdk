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
"""Unit tests for the Constraints.set_monitoring feature-level override."""
from __future__ import absolute_import

import pytest

from sagemaker.model_monitor.monitoring_files import Constraints

CONSTRAINTS_S3_URI = "s3://bucket/constraints.json"


def _body_dict():
    """A constraints body with one non-string (binary/integral) and one string feature."""
    return {
        "version": 0.0,
        "features": [
            {
                "name": "Churn",
                "inferred_type": "Integral",
                "num_constraints": {"is_non_negative": True},
            },
            {
                "name": "State",
                "inferred_type": "String",
                "string_constraints": {"domains": ["OH", "NJ"]},
            },
        ],
        "monitoring_config": {"evaluate_constraints": "Enabled"},
    }


def _make_constraints():
    return Constraints(body_dict=_body_dict(), constraints_file_s3_uri=CONSTRAINTS_S3_URI)


def _feature(constraints, name):
    return next(f for f in constraints.body_dict["features"] if f["name"] == name)


def test_set_monitoring_disables_non_string_feature():
    # Regression test for GitHub issue #2745: set_monitoring must work for a
    # non-string (e.g. binary/integral) feature that has no string_constraints section.
    constraints = _make_constraints()

    constraints.set_monitoring(False, feature_name="Churn")

    churn = _feature(constraints, "Churn")
    assert churn["monitoring_config_overrides"] == {"evaluate_constraints": "Disabled"}
    # The override must live at the feature level, not nested inside num_constraints.
    assert "monitoring_config_overrides" not in churn["num_constraints"]


def test_set_monitoring_override_is_feature_level_for_string_feature():
    # Even for a string feature, the override belongs at the feature level (per the
    # constraints.json schema), not inside the string_constraints section.
    constraints = _make_constraints()

    constraints.set_monitoring(False, feature_name="State")

    state = _feature(constraints, "State")
    assert state["monitoring_config_overrides"] == {"evaluate_constraints": "Disabled"}
    assert "monitoring_config_overrides" not in state["string_constraints"]


def test_set_monitoring_enable_maps_to_enabled():
    constraints = _make_constraints()

    constraints.set_monitoring(True, feature_name="Churn")

    assert _feature(constraints, "Churn")["monitoring_config_overrides"] == {
        "evaluate_constraints": "Enabled"
    }


def test_set_monitoring_preserves_existing_overrides():
    body = _body_dict()
    body["features"][0]["monitoring_config_overrides"] = {"some_other_flag": "keep"}
    constraints = Constraints(body_dict=body, constraints_file_s3_uri=CONSTRAINTS_S3_URI)

    constraints.set_monitoring(False, feature_name="Churn")

    overrides = _feature(constraints, "Churn")["monitoring_config_overrides"]
    assert overrides == {"some_other_flag": "keep", "evaluate_constraints": "Disabled"}


def test_set_monitoring_without_feature_name_sets_top_level_flag():
    constraints = _make_constraints()

    constraints.set_monitoring(False)

    assert constraints.body_dict["monitoring_config"]["evaluate_constraints"] == "Disabled"


@pytest.mark.parametrize("enable,expected", [(True, "Enabled"), (False, "Disabled")])
def test_monitoring_api_map(enable, expected):
    constraints = _make_constraints()

    constraints.set_monitoring(enable, feature_name="Churn")

    assert (
        _feature(constraints, "Churn")["monitoring_config_overrides"]["evaluate_constraints"]
        == expected
    )
