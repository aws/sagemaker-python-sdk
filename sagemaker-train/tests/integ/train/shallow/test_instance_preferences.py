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
"""Shallow submission tests for ``Compute.instance_preferences``.

Each accepted case submits a real ``CreateTrainingJob`` carrying an ordered
``InstancePreferences`` list (no top-level ``InstanceType``), asserts the
service returned an ARN and echoed the list on Describe, then stops the job.
Rejected cases assert the request is refused with the documented reason.

Which instance type wins, and how the job runs on it, is training behaviour and
belongs in the deep suites.
"""

from __future__ import absolute_import

import os

import pytest
from sagemaker.core import shapes
from sagemaker.core.shapes import InstancePreference
from sagemaker.core.training.configs import Compute, SourceCode
from sagemaker.train.model_trainer import ModelTrainer

from .harness import (
    MAX_RUNTIME_IN_SECONDS,
    assert_rejected,
    assert_submitted,
    cpu_image,
    submitted,
    unique_name,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
PARAM_SCRIPT_SOURCE_DIR = os.path.join(DATA_DIR, "params_script")

# Two small CPU types. INSTANCE_PREFERENCES_TEST_INSTANCE_TYPES (comma-separated)
# overrides them for accounts where these lack quota.
PREFERENCE_TYPES = [
    t.strip()
    for t in os.environ.get(
        "INSTANCE_PREFERENCES_TEST_INSTANCE_TYPES", "ml.m5.large,ml.m5.xlarge"
    ).split(",")
    if t.strip()
]


def _trainer(sagemaker_session, name, compute):
    return ModelTrainer(
        sagemaker_session=sagemaker_session,
        training_image=cpu_image(sagemaker_session),
        source_code=SourceCode(
            source_dir=PARAM_SCRIPT_SOURCE_DIR,
            requirements="requirements.txt",
            entry_script="train.py",
        ),
        compute=compute,
        stopping_condition=shapes.StoppingCondition(max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS),
        base_job_name=name,
    )


def _preferences(counts=None):
    if counts is None:
        return [InstancePreference(instance_type=t) for t in PREFERENCE_TYPES]
    return [
        InstancePreference(instance_type=t, instance_count=c)
        for t, c in zip(PREFERENCE_TYPES, counts)
    ]


def _echoed_types(job):
    job.refresh()
    prefs = job.resource_config.instance_preferences
    return [p.instance_type for p in prefs] if prefs else []


class TestInstancePreferencesAccepted:
    def test_uniform_count_is_accepted_and_echoed(self, sagemaker_session):
        name = unique_name("shallow-ip-uniform")
        trainer = _trainer(
            sagemaker_session,
            name,
            Compute(instance_preferences=_preferences(), instance_count=1),
        )

        with submitted(trainer) as job:
            assert_submitted(job)
            assert _echoed_types(job) == PREFERENCE_TYPES
            # The customer never set a top-level type; the service must not invent one.
            assert not job.resource_config.instance_type

    def test_per_preference_counts_are_accepted_and_echoed(self, sagemaker_session):
        name = unique_name("shallow-ip-per-pref")
        trainer = _trainer(
            sagemaker_session,
            name,
            Compute(instance_preferences=_preferences(counts=[1, 1])),
        )

        with submitted(trainer) as job:
            assert_submitted(job)
            assert _echoed_types(job) == PREFERENCE_TYPES


class TestInstancePreferencesRejected:
    """Client-side rules must fail fast: a payload that reached the service with
    both a fixed type and a list would be a regression in the SDK, not a
    behaviour to leave for the backend to catch."""

    def test_instance_type_with_preferences_is_rejected(self, sagemaker_session):
        with pytest.raises(ValueError, match="mutually exclusive with instance_type"):
            Compute(
                instance_type=PREFERENCE_TYPES[0],
                instance_count=1,
                instance_preferences=_preferences(),
            )

    def test_managed_spot_with_preferences_is_rejected(self, sagemaker_session):
        with pytest.raises(ValueError, match="mutually exclusive with managed spot training"):
            Compute(
                instance_count=1,
                enable_managed_spot_training=True,
                instance_preferences=_preferences(),
            )

    def test_duplicate_types_are_rejected(self, sagemaker_session):
        with pytest.raises(ValueError, match="duplicate instance types"):
            Compute(
                instance_count=1,
                instance_preferences=[
                    InstancePreference(instance_type=PREFERENCE_TYPES[0]),
                    InstancePreference(instance_type=PREFERENCE_TYPES[0]),
                ],
            )

    def test_server_rejects_more_than_five_preferences(self, sagemaker_session):
        """The list cap is deliberately not enforced client-side (it is a
        server-side tunable), so this is the one rule that must be asserted
        against the service."""
        six = [
            "ml.m5.large",
            "ml.m5.xlarge",
            "ml.m5.2xlarge",
            "ml.m4.xlarge",
            "ml.c5.xlarge",
            "ml.c5.2xlarge",
        ]
        trainer = _trainer(
            sagemaker_session,
            unique_name("shallow-ip-six"),
            Compute(
                instance_preferences=[InstancePreference(instance_type=t) for t in six],
                instance_count=1,
            ),
        )
        assert_rejected(trainer, ("InstancePreferences", "Member must have length"))
