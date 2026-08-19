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
"""Shallow submission tests for ``MultiTurnRLTrainer`` (Agentic RFT).

Shallow counterpart of ``test_multi_turn_rl_trainer_integration.py``.

MTRL is the one trainer here that does not create a TrainingJob at all: it calls
the generic Job API and returns an ``AgentRFTJob``, so its ARN segment is ``job``
rather than ``training-job`` and the harness resolves it via ``_latest_job``.
"""

from __future__ import absolute_import

import logging
import os

import pytest
from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer

from .harness import assert_submitted, submitted, unique_name

logger = logging.getLogger(__name__)


@pytest.mark.gpu_intensive
class TestMultiTurnRLSubmission:
    """AgentRFT Job acceptance for ``MultiTurnRLTrainer``.

    Marked ``gpu_intensive`` (and therefore excluded from the PR gate, per the
    marker's definition in ``tox.ini``) because unlike every other test in this
    suite it cannot be made self-contained: MTRL requires a pre-provisioned agent
    runtime and an MLflow app, neither of which this suite creates. The existing
    ``test_multi_turn_rl_trainer_integration.py`` hardcodes both.

    They are still written using the shallow pattern rather than omitted, so that
    when the prerequisites are provisioned in the PR account these become
    PR-gate-eligible by deleting one marker. Prerequisites are resolved from the
    environment and the tests skip when absent, so they never fail for
    infrastructure reasons.
    """

    @pytest.fixture(scope="class")
    def mtrl_prerequisites(self, sagemaker_session, account_id, region):
        """Resolve MTRL prerequisites, skipping if they are not configured.

        Read from the environment rather than hardcoded so this does not bake in
        another account-specific constant.
        """
        agent_env = os.environ.get("SHALLOW_MTRL_AGENT_ENV")
        mlflow_app_arn = os.environ.get("SHALLOW_MTRL_MLFLOW_APP_ARN")
        dataset = os.environ.get("SHALLOW_MTRL_DATASET")

        missing = [
            name
            for name, value in (
                ("SHALLOW_MTRL_AGENT_ENV", agent_env),
                ("SHALLOW_MTRL_MLFLOW_APP_ARN", mlflow_app_arn),
                ("SHALLOW_MTRL_DATASET", dataset),
            )
            if not value
        ]
        if missing:
            pytest.skip("MTRL prerequisites not configured; set " + ", ".join(missing))

        return {
            "agent_env": agent_env,
            "mlflow_app_arn": mlflow_app_arn,
            "dataset": dataset,
            "model": os.environ.get("SHALLOW_MTRL_MODEL", "mock-oss-test"),
        }

    def test_agent_rft_job_is_accepted(self, sagemaker_session, mtrl_prerequisites):
        """The AgentRFT job config document must be accepted by the Job API.

        Note the different ARN resource segment: this is a ``job``, not a
        ``training-job``.
        """
        trainer = MultiTurnRLTrainer(
            model=mtrl_prerequisites["model"],
            agent_env=mtrl_prerequisites["agent_env"],
            training_dataset=mtrl_prerequisites["dataset"],
            mlflow_app_arn=mtrl_prerequisites["mlflow_app_arn"],
            accept_eula=True,
            sagemaker_session=sagemaker_session,
            base_job_name=unique_name("shallow-mtrl"),
        )

        with submitted(trainer) as job:
            assert_submitted(job, resource="job")

    def test_hyperparameter_mutation_is_accepted(self, sagemaker_session, mtrl_prerequisites):
        """``trainer.hyperparameters`` mutation must reach the job config
        document, which the service validates on submission."""
        trainer = MultiTurnRLTrainer(
            model=mtrl_prerequisites["model"],
            agent_env=mtrl_prerequisites["agent_env"],
            training_dataset=mtrl_prerequisites["dataset"],
            mlflow_app_arn=mtrl_prerequisites["mlflow_app_arn"],
            accept_eula=True,
            sagemaker_session=sagemaker_session,
            base_job_name=unique_name("shallow-mtrl-hp"),
        )
        trainer.hyperparameters.global_batch_size = 32

        with submitted(trainer) as job:
            assert_submitted(job, resource="job")
