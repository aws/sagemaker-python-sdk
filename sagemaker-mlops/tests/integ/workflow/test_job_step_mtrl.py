"""Integration test for a `JobStep` captured from `MultiTurnRLTrainer`.

Agent RFT needs real assets, so inputs come from environment variables and the
test skips when they are absent:

    SAGEMAKER_INTEG_MTRL_MODEL           model id for multi-turn RL
    SAGEMAKER_INTEG_MTRL_AGENT_ENV       Bedrock AgentCore runtime ARN (or Lambda ARN)
    SAGEMAKER_INTEG_MTRL_DATASET         training dataset S3 URI
    SAGEMAKER_INTEG_MODEL_PACKAGE_GROUP  output model package group
    SAGEMAKER_INTEG_MTRL_MLFLOW_APP      MLflow app ARN

Proves the second `CreateJob` producer end to end: the pipeline service accepts
a `JobStep` whose arguments were captured from `MultiTurnRLTrainer.train()`,
resolves the execution-scoped output path inside `JobConfigDocument`, and
creates the AgentRFT job from it. The execution is stopped once the job exists.
"""
from __future__ import absolute_import

import json
import os
import time
import uuid

import pytest

from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import JobStep
from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer

_REQUIRED_ENV = (
    "SAGEMAKER_INTEG_MTRL_MODEL",
    "SAGEMAKER_INTEG_MTRL_AGENT_ENV",
    "SAGEMAKER_INTEG_MTRL_DATASET",
    "SAGEMAKER_INTEG_MODEL_PACKAGE_GROUP",
    "SAGEMAKER_INTEG_MTRL_MLFLOW_APP",
)

pytestmark = pytest.mark.skipif(
    any(not os.environ.get(name) for name in _REQUIRED_ENV),
    reason="requires AgentRFT assets: %s" % ", ".join(_REQUIRED_ENV),
)


@pytest.fixture
def sagemaker_session():
    return Session()


@pytest.fixture
def pipeline_session():
    return PipelineSession()


@pytest.fixture
def role():
    return get_execution_role()


def _wait_for_created_job(execution, timeout_seconds=900):
    """Wait until the step reports its created job, or terminal failure."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        steps = execution.list_steps()
        if steps:
            metadata = steps[0].get("Metadata", {})
            if "Job" in metadata:
                return steps[0]
            if steps[0].get("StepStatus") in ("Failed", "Stopped"):
                raise AssertionError("step ended %s: %s" % (steps[0]["StepStatus"], steps[0]))
        time.sleep(30)
    raise TimeoutError("step did not create a job in time")


def test_job_step_from_mtrl_capture(sagemaker_session, pipeline_session, role):
    bucket = sagemaker_session.default_bucket()
    pipeline_name = "integ-mtrl-job-step-%s" % uuid.uuid4().hex[:8]

    trainer = MultiTurnRLTrainer(
        model=os.environ["SAGEMAKER_INTEG_MTRL_MODEL"],
        agent_env=os.environ["SAGEMAKER_INTEG_MTRL_AGENT_ENV"],
        training_dataset=os.environ["SAGEMAKER_INTEG_MTRL_DATASET"],
        output_model_package_group=os.environ["SAGEMAKER_INTEG_MODEL_PACKAGE_GROUP"],
        mlflow_app_arn=os.environ["SAGEMAKER_INTEG_MTRL_MLFLOW_APP"],
        s3_output_path="s3://%s/%s/output" % (bucket, pipeline_name),
        accept_eula=True,
        sagemaker_session=pipeline_session,
    )

    step = JobStep(name="mtrl", step_args=trainer.train(wait=False))
    pipeline = Pipeline(name=pipeline_name, steps=[step], sagemaker_session=pipeline_session)

    definition = json.loads(pipeline.definition())
    assert definition["Steps"][0]["Type"] == "Job"
    document = definition["Steps"][0]["Arguments"]["JobConfigDocument"]
    # JobStep scopes the output path per execution inside the document.
    assert "Std:Join" in document
    assert {"Get": "Execution.PipelineExecutionId"} in document["Std:Join"]["Values"]

    try:
        pipeline.upsert(role_arn=role)
        execution = pipeline.start()
        step_state = _wait_for_created_job(execution)
        assert step_state["Metadata"]["Job"]["Arn"]
    finally:
        try:
            execution.stop()
        except Exception:  # noqa: BLE001 -- may already be terminal
            pass
        try:
            pipeline.delete()
        except Exception:  # noqa: BLE001 -- best-effort cleanup
            pass
