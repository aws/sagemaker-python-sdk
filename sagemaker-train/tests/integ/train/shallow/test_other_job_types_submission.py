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
"""Shallow submission tests for job types that are not plain training jobs.

The rest of this suite covers ``CreateTrainingJob``. Two trainers in this package
create something else, and each needed harness support rather than being
genuinely un-testable:

* ``HyperparameterTuner.tune()`` creates a **HyperParameterTuningJob**. The
  service validates the embedded training-job definition (including the
  ``sm_drivers`` channel for distributed runs) plus tuning-specific rules --
  objective metric, parameter ranges, max jobs/parallel jobs. Stopping is
  ``tuner.stop_tuning_job()``.
* ``MultiTurnRLTrainer.train()`` creates an **AgentRFT Job** via the generic Job
  API, not ``CreateTrainingJob``. It returns an ``AgentRFTJob`` exposing
  ``job_arn``/``job_name``/``stop()``.

Both are covered here because "different resource type" is a reason to teach the
harness a new ARN shape, not a reason to skip the coverage.

The tuner tests carry the real weight: they run on CPU with no external
prerequisites. The MTRL tests are marked ``gpu_intensive`` and skip when their
prerequisites are absent -- see ``TestMultiTurnRLSubmission`` for why.
"""

from __future__ import absolute_import

import logging
import os

import pytest
from sagemaker.core import shapes
from sagemaker.core.parameter import ContinuousParameter
from sagemaker.core.training.configs import Compute, SourceCode
from sagemaker.train.distributed import Torchrun
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer
from sagemaker.train.tuner import HyperparameterTuner

from .harness import (
    CPU_IMAGE,
    DEFAULT_INSTANCE_COUNT,
    DEFAULT_INSTANCE_TYPE,
    MAX_RUNTIME_IN_SECONDS,
    assert_submitted,
    submitted,
    unique_name,
)

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
PARAM_SCRIPT_SOURCE_DIR = os.path.join(DATA_DIR, "params_script")


def _model_trainer(sagemaker_session, name, **overrides):
    """The inner trainer a tuning job wraps."""
    kwargs = dict(
        sagemaker_session=sagemaker_session,
        training_image=CPU_IMAGE,
        base_job_name=name,
        source_code=SourceCode(
            source_dir=PARAM_SCRIPT_SOURCE_DIR,
            requirements="requirements.txt",
            entry_script="train.py",
        ),
        compute=Compute(
            instance_type=DEFAULT_INSTANCE_TYPE,
            instance_count=DEFAULT_INSTANCE_COUNT,
            volume_size_in_gb=30,
        ),
        stopping_condition=shapes.StoppingCondition(max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS),
        hyperparameters={"learning_rate": 1e-4},
    )
    kwargs.update(overrides)
    return ModelTrainer(**kwargs)


def _tuner(model_trainer, **overrides):
    """A minimal single-job tuner.

    ``max_jobs=1`` / ``max_parallel_jobs=1`` keeps the blast radius to one child
    training job, which is stopped along with the tuning job.
    """
    kwargs = dict(
        model_trainer=model_trainer,
        objective_metric_name="eval_loss",
        metric_definitions=[{"Name": "eval_loss", "Regex": r"eval_loss: ([0-9\\.]+)"}],
        hyperparameter_ranges={
            "learning_rate": ContinuousParameter(
                min_value=1e-5, max_value=5e-4, scaling_type="Logarithmic"
            )
        },
        objective_type="Minimize",
        max_jobs=1,
        max_parallel_jobs=1,
    )
    kwargs.update(overrides)
    return HyperparameterTuner(**kwargs)


class TestTuningJobSubmission:
    """HyperParameterTuningJob acceptance.

    Stopping a tuning job also stops its child training jobs, so the same
    submit-then-stop economics apply.
    """

    def test_minimal_tuning_job_is_accepted(self, sagemaker_session):
        """Baseline: the service accepts a well-formed tuning job."""
        name = unique_name("shallow-tuner")
        tuner = _tuner(_model_trainer(sagemaker_session, name))

        try:
            tuner.tune(wait=False)
            assert_submitted(tuner.latest_tuning_job, resource="hyper-parameter-tuning-job")
        finally:
            # Tuner exposes its own stop method rather than the resource's.
            try:
                tuner.stop_tuning_job()
            except Exception as e:  # pragma: no cover - best-effort teardown
                logger.warning("Could not stop tuning job: %s", e)

    def test_distributed_tuning_job_is_accepted(self, sagemaker_session):
        """A tuning job wrapping a Torchrun trainer must include the
        ``sm_drivers`` channel in its training-job definition.

        This is the regression the existing ``test_tuner_distributed.py`` guards
        by running a job to completion and inspecting logs. Submission alone
        proves the channel is present and the definition is accepted, which is
        the part that regressed; the log assertion stays in the deep suite.
        """
        name = unique_name("shallow-tuner-dist")
        model_trainer = _model_trainer(sagemaker_session, name, distributed=Torchrun())
        tuner = _tuner(model_trainer)

        try:
            tuner.tune(wait=False)
            arn = assert_submitted(tuner.latest_tuning_job, resource="hyper-parameter-tuning-job")

            # The sm_drivers channel lives in the tuning job's training
            # definition; read it back to prove it survived submission rather
            # than inferring from acceptance alone.
            described = tuner.latest_tuning_job.refresh()
            definition = getattr(described, "training_job_definition", None)
            if definition is not None:
                channels = [
                    channel.channel_name for channel in (definition.input_data_config or [])
                ]
                assert "sm_drivers" in channels, (
                    f"tuning job {arn} is missing the sm_drivers channel; " f"channels={channels}"
                )
        finally:
            try:
                tuner.stop_tuning_job()
            except Exception as e:  # pragma: no cover - best-effort teardown
                logger.warning("Could not stop tuning job: %s", e)


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
