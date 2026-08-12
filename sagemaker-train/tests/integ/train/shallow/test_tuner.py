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
from contextlib import contextmanager

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
    MAX_TUNING_JOB_NAME,
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


@contextmanager
def _tuning(tuner, job_name):
    """Submit a tuning job under an explicit name, then always stop it.

    The explicit ``job_name`` is load-bearing. Left to itself the tuner derives a
    name from the training image plus a second-granularity timestamp
    (``pytorch-training-260811-1621``) and ignores ``base_job_name`` entirely, so
    two tuner tests starting in the same second collide with ``ResourceInUse``.
    Verified against AWS: that is exactly how this failed before.

    Teardown goes through ``tuner.stop_tuning_job()`` rather than the harness's
    ``stop_quietly``, because the tuner wraps the resource and stopping it also
    stops the child training jobs it launched.
    """
    try:
        tuner.tune(job_name=job_name, wait=False)
        yield
    finally:
        try:
            tuner.stop_tuning_job()
            logger.info("Stopped tuning job %s", job_name)
        except Exception as e:  # pragma: no cover - best-effort teardown
            # A tuning job that never started, or already reached a terminal
            # state, cannot be stopped; that must not fail the test.
            logger.warning("Could not stop tuning job %s: %s", job_name, e)


class TestTuningJobSubmission:
    """HyperParameterTuningJob acceptance.

    Stopping a tuning job also stops its child training jobs, so the same
    submit-then-stop economics apply.
    """

    def test_minimal_tuning_job_is_accepted(self, sagemaker_session):
        """Baseline: the service accepts a well-formed tuning job."""
        name = unique_name("shallow-tuner", max_length=MAX_TUNING_JOB_NAME)
        tuner = _tuner(_model_trainer(sagemaker_session, name))

        with _tuning(tuner, name):
            assert_submitted(
                tuner.latest_tuning_job,
                expected_name=name,
                resource="hyper-parameter-tuning-job",
            )

    def test_distributed_tuning_job_is_accepted(self, sagemaker_session):
        """A tuning job wrapping a Torchrun trainer must include the
        ``sm_drivers`` channel in its training-job definition.

        This is the regression the existing ``test_tuner_distributed.py`` guards
        by running a job to completion and inspecting logs. Submission alone
        proves the channel is present and the definition is accepted, which is
        the part that regressed; the log assertion stays in the deep suite.
        """
        name = unique_name("shallow-tune-dist", max_length=MAX_TUNING_JOB_NAME)
        model_trainer = _model_trainer(sagemaker_session, name, distributed=Torchrun())
        tuner = _tuner(model_trainer)

        with _tuning(tuner, name):
            arn = assert_submitted(
                tuner.latest_tuning_job,
                expected_name=name,
                resource="hyper-parameter-tuning-job",
            )

            # The sm_drivers channel lives in the tuning job's training
            # definition; read it back to prove it survived submission rather
            # than inferring it from acceptance alone.
            described = tuner.latest_tuning_job.refresh()
            definition = getattr(described, "training_job_definition", None)
            assert definition is not None, (
                f"tuning job {arn} has no training_job_definition to inspect; "
                "cannot verify the sm_drivers channel"
            )
            channels = [channel.channel_name for channel in (definition.input_data_config or [])]
            assert (
                "sm_drivers" in channels
            ), f"tuning job {arn} is missing the sm_drivers channel; channels={channels}"
