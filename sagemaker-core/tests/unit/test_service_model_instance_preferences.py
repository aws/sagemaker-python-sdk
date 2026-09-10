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
"""Pin the Instance Preferences contract in the bundled SageMaker service model.

The bundled ``sample/sagemaker/2017-07-24/service-2.json`` is both the codegen
input for ``shapes.py`` and the model the runtime botocore loader injects, so
these tests guard the Instance Preferences API surface against accidental
regeneration/edit drift:

- training and processing preferences are DISTINCT shapes
  (``InstancePreference`` vs ``ProcessingInstancePreference``), matching the
  service model where the two planes use different member types;
- per-preference training plans are training-only and capped at 1
  (``TrainingPlanArnList``);
- the output-only ``SelectedInstanceType``/``SelectedInstanceCount`` exist on
  both ``ResourceConfig`` and ``ProcessingClusterConfig``;
- ``ProcessingClusterConfig`` no longer requires ``InstanceType``/
  ``InstanceCount`` (mutually exclusive with ``InstancePreferences``,
  enforced server-side).
"""
from __future__ import absolute_import

import json
import os

import pytest

SERVICE_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "sample",
    "sagemaker",
    "2017-07-24",
    "service-2.json",
)


@pytest.fixture(scope="module")
def shapes():
    with open(SERVICE_JSON) as f:
        return json.load(f)["shapes"]


class TestTrainingInstancePreferenceModel:
    def test_instance_preference_shape(self, shapes):
        shape = shapes["InstancePreference"]
        assert shape["type"] == "structure"
        assert shape["required"] == ["InstanceType"]
        members = shape["members"]
        assert members["InstanceType"]["shape"] == "TrainingInstanceType"
        assert members["InstanceCount"]["shape"] == "TrainingInstanceCount"
        assert members["TrainingPlanArns"]["shape"] == "TrainingPlanArnList"

    def test_instance_preference_list_capped_at_5(self, shapes):
        lst = shapes["InstancePreferenceList"]
        assert lst["type"] == "list"
        assert lst["member"]["shape"] == "InstancePreference"
        assert lst["min"] == 1
        assert lst["max"] == 5

    def test_training_plan_arn_list_capped_at_1(self, shapes):
        lst = shapes["TrainingPlanArnList"]
        assert lst["type"] == "list"
        assert lst["member"]["shape"] == "TrainingPlanArn"
        assert lst["min"] == 1
        assert lst["max"] == 1

    def test_resource_config_members(self, shapes):
        members = shapes["ResourceConfig"]["members"]
        assert members["InstancePreferences"]["shape"] == "InstancePreferenceList"
        assert members["SelectedInstanceType"]["shape"] == "TrainingInstanceType"
        assert members["SelectedInstanceCount"]["shape"] == "TrainingInstanceCount"


class TestProcessingInstancePreferenceModel:
    def test_processing_instance_preference_shape(self, shapes):
        shape = shapes["ProcessingInstancePreference"]
        assert shape["type"] == "structure"
        assert shape["required"] == ["InstanceType"]
        members = shape["members"]
        assert members["InstanceType"]["shape"] == "ProcessingInstanceType"
        assert members["InstanceCount"]["shape"] == "ProcessingInstanceCount"
        # per-type training plans are training-only
        assert "TrainingPlanArns" not in members

    def test_processing_instance_preference_list_capped_at_5(self, shapes):
        lst = shapes["ProcessingInstancePreferenceList"]
        assert lst["type"] == "list"
        assert lst["member"]["shape"] == "ProcessingInstancePreference"
        assert lst["min"] == 1
        assert lst["max"] == 5

    def test_processing_cluster_config_members(self, shapes):
        pcc = shapes["ProcessingClusterConfig"]
        members = pcc["members"]
        assert members["InstancePreferences"]["shape"] == "ProcessingInstancePreferenceList"
        assert members["SelectedInstanceType"]["shape"] == "ProcessingInstanceType"
        assert members["SelectedInstanceCount"]["shape"] == "ProcessingInstanceCount"

    def test_processing_cluster_config_type_count_not_required(self, shapes):
        # InstanceType/InstanceCount are mutually exclusive with
        # InstancePreferences; the exclusivity is enforced server-side, so the
        # client model must not hard-require them.
        required = shapes["ProcessingClusterConfig"]["required"]
        assert "InstanceType" not in required
        assert "InstanceCount" not in required
        assert "VolumeSizeInGB" in required


class TestPreferenceShapesAreDistinct:
    def test_training_and_processing_preferences_do_not_share_shapes(self, shapes):
        assert (
            shapes["ResourceConfig"]["members"]["InstancePreferences"]["shape"]
            != shapes["ProcessingClusterConfig"]["members"]["InstancePreferences"]["shape"]
        )
