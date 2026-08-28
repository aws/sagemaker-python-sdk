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
"""Integration test for the LineageStep (v2)."""

from __future__ import absolute_import

import time
import uuid

import pytest

from sagemaker import get_execution_role, utils
from sagemaker.workflow.lineage_step import LineageStep
from sagemaker.workflow.pipeline import Pipeline


@pytest.fixture
def role(sagemaker_session):
    return get_execution_role(sagemaker_session)


@pytest.fixture
def pipeline_name():
    return utils.unique_name_from_base("my-pipeline-lineage")


def test_lineage_step_execute_end_to_end(sagemaker_session, role, pipeline_name):
    """End-to-end run of a LineageStep pipeline against the real service.

    Builds a pipeline with a single ``LineageStep`` that creates one
    lineage ``Action``. Verifies the pipeline execution succeeds and
    that the server-reported step metadata contains the created action
    ARN. Cleans up the Action, the pipeline, and any lingering S3
    definition artifact.

    Requires the execution role to have ``sagemaker:CreateAction``. The
    ``SageMakerRole`` fixture role has broad SageMaker access and
    satisfies this requirement.
    """
    stamp = uuid.uuid4().hex[:8]
    action_name = f"lineage-integ-{stamp}"

    step = LineageStep(
        name="RecordLineage",
        arguments={
            "Actions": [
                {
                    "ActionName": action_name,
                    "ActionType": "ModelTraining",
                    "Status": "Completed",
                    "Source": {
                        "SourceUri": f"s3://lineage-integ-test/{stamp}/model.tar.gz",
                        "SourceType": "MODEL",
                    },
                    "Description": "Lineage v2 integ test action",
                }
            ]
        },
    )
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step],
        sagemaker_session=sagemaker_session,
    )

    sm_client = sagemaker_session.sagemaker_client
    try:
        pipeline.create(role_arn=role)
        execution = pipeline.start()

        # LineageStep is metadata-only and completes quickly. Poll up to
        # 5 minutes for headroom under load.
        timeout = 300
        start_time = time.time()
        final_status = None
        while time.time() - start_time < timeout:
            description = execution.describe()
            status = description["PipelineExecutionStatus"]
            if status in ("Succeeded", "Failed", "Stopped"):
                final_status = status
                break
            time.sleep(10)

        if final_status != "Succeeded":
            steps = sm_client.list_pipeline_execution_steps(
                PipelineExecutionArn=execution.arn,
            )["PipelineExecutionSteps"]
            failure_details = "\n".join(
                f"{s['StepName']}: {s.get('FailureReason', 'no reason')}"
                for s in steps
                if s.get("StepStatus") == "Failed"
            )
            pytest.fail(f"Pipeline execution status={final_status}. Details:\n{failure_details}")

        # Verify the step metadata reports the created action ARN.
        steps = sm_client.list_pipeline_execution_steps(
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
            sm_client.delete_action(ActionName=action_name)
        except Exception:
            pass
        # Delete the pipeline.
        try:
            sm_client.delete_pipeline(PipelineName=pipeline_name)
        except Exception:
            pass
