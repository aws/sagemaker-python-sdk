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
"""Integration test for the LineageStep.

Creates a pipeline containing a single ``LineageStep`` that records a
SageMaker lineage Action, executes it end-to-end against the real
service, and asserts the execution reaches ``Succeeded``. Cleans up the
Action, the pipeline, and the S3 pipeline definition artifact.

Requires the execution role to have ``sagemaker:CreateAction`` (and
related lineage permissions). ``SageMakerRole`` — the standard fixture
role used across the SDK's integ tests — has broad SageMaker access and
satisfies this requirement.

This test represents the SDK-side end-to-end validation of the
LineageStep. The inference deployment steps are covered separately by
``test_deployment_steps.py``.
"""

from __future__ import absolute_import

import time
import uuid

import pytest

from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.lineage.action import Action
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.mlops.workflow.lineage_step import LineageStep
from sagemaker.mlops.workflow.pipeline import Pipeline


@pytest.fixture
def sagemaker_session():
    return Session()


@pytest.fixture
def pipeline_session():
    return PipelineSession()


@pytest.fixture
def role():
    return get_execution_role()


def test_lineage_step_execute_end_to_end(sagemaker_session, pipeline_session, role):
    """Full end-to-end run of a LineageStep pipeline against the real service.

    Builds a pipeline with a single ``LineageStep`` that creates one
    lineage ``Action``. Verifies the pipeline execution succeeds and the
    server-reported step metadata contains the created action ARN.
    """
    stamp = uuid.uuid4().hex[:8]
    action_name = f"lineage-integ-{stamp}"
    pipeline_name = f"integ-lineage-{stamp}"

    step_args = Action.create(
        action_name=action_name,
        source_uri=f"s3://lineage-integ-test/{stamp}/model.tar.gz",
        source_type="MODEL",
        action_type="ModelTraining",
        status="Completed",
        description="Lineage integ test action",
        sagemaker_session=pipeline_session,
    )
    step = LineageStep(name="RecordLineage", step_args=step_args)
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step],
        sagemaker_session=pipeline_session,
    )

    try:
        pipeline.upsert(role_arn=role)
        execution = pipeline.start()

        # LineageStep is metadata-only; execution completes quickly. Poll
        # up to 5 minutes to give the service plenty of headroom under load.
        timeout = 300
        start_time = time.time()
        final_status = None
        while time.time() - start_time < timeout:
            execution_desc = execution.describe()
            status = execution_desc["PipelineExecutionStatus"]
            if status in ("Succeeded", "Failed", "Stopped"):
                final_status = status
                break
            time.sleep(10)

        if final_status != "Succeeded":
            steps = sagemaker_session.sagemaker_client.list_pipeline_execution_steps(
                PipelineExecutionArn=execution.arn,
            )["PipelineExecutionSteps"]
            failure_details = "\n".join(
                f"{s['StepName']}: {s.get('FailureReason', 'no reason')}"
                for s in steps
                if s.get("StepStatus") == "Failed"
            )
            pytest.fail(f"Pipeline execution status={final_status}. Details:\n{failure_details}")

        # Verify the step metadata reports the created action ARN.
        steps = sagemaker_session.sagemaker_client.list_pipeline_execution_steps(
            PipelineExecutionArn=execution.arn,
        )["PipelineExecutionSteps"]
        lineage_step = next(s for s in steps if s["StepName"] == "RecordLineage")
        assert lineage_step["StepStatus"] == "Succeeded"
        metadata = lineage_step.get("Metadata", {})
        action_arns = metadata.get("Lineage", {}).get("ActionArns", {})
        assert (
            action_name in action_arns
        ), f"expected {action_name} in ActionArns, got: {action_arns}"
        assert action_arns[action_name].endswith(f":action/{action_name}")

    finally:
        # Delete the lineage Action.
        try:
            sagemaker_session.sagemaker_client.delete_action(ActionName=action_name)
        except Exception:
            pass
        # Delete the pipeline.
        try:
            sagemaker_session.sagemaker_client.delete_pipeline(PipelineName=pipeline_name)
        except Exception:
            pass
