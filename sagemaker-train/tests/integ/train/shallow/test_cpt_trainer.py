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
"""Shallow submission tests for CPTTrainer (continued pre-training).

Shallow counterpart of test_cpt_hyperpod.py.

CPT differs from the other recipe trainers in two verified ways: it accepts no
training_type (there is no LoRA/full distinction for continued pre-training),
and its compute is HyperPodCompute-only.

The whole class is marked gpu_intensive and skips unless a cluster is
configured, because CPT refuses to submit without HyperPod compute --

    ValueError: CPT requires HyperPod compute.
    Pass compute=HyperPodCompute(...) when creating the CPTTrainer.

-- and HyperPod submits to a pre-provisioned cluster rather than through
CreateTrainingJob, so there is nothing this suite can create on demand. Written in
the shallow style anyway so it becomes gate-eligible by dropping one marker once a
cluster exists in the PR account.
"""

from __future__ import absolute_import

import os

import pytest
from sagemaker.core.training.configs import HyperPodCompute
from sagemaker.train.cpt_trainer import CPTTrainer

from .harness import assert_submitted, submitted
from .recipe_cases import RecipeTrainerCases


@pytest.mark.gpu_intensive
class TestCPTTrainerSubmission(RecipeTrainerCases):
    """CPT submits only via HyperPod, so the shared cases are not inherited as-is."""

    TRAINER = CPTTrainer
    SUPPORTS_TRAINING_TYPE = False
    SUPPORTS_SERVERFUL = False

    @pytest.fixture(autouse=True)
    def _require_hyperpod(self):
        """Skip the whole class unless a HyperPod cluster is configured."""
        cluster = os.environ.get("SHALLOW_HYPERPOD_CLUSTER")
        if not cluster:
            pytest.skip("CPT requires HyperPod; set SHALLOW_HYPERPOD_CLUSTER to run")
        self._cluster = cluster

    def build(self, sagemaker_session, dataset, name, **overrides):
        """Add the required HyperPod compute to every CPT submission."""
        overrides.setdefault("compute", HyperPodCompute(cluster_name=self._cluster))
        return super().build(sagemaker_session, dataset, name, **overrides)
