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
SageMaker Action, Artifact and Context together, plus associations between
them referenced by name and type, executes it end-to-end against the real
service, and asserts the execution reaches ``Succeeded``. Cleans up every
created entity, the associations, and the pipeline.

Requires the execution role to have ``sagemaker:CreateAction``,
``CreateArtifact``, ``CreateContext`` and ``AddAssociation`` (and the
matching delete permissions for cleanup). ``SageMakerRole`` — the standard
fixture role used across the SDK's integ tests — has broad SageMaker access
and satisfies this requirement.

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
from sagemaker.core.lineage.artifact import Artifact
from sagemaker.core.lineage.context import Context
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.mlops.workflow.lineage_step import (
    LineageAssociation,
    LineageEntityReference,
    LineageStep,
)
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

    Builds a pipeline with a single ``LineageStep`` that creates an Action,
    an Artifact and a Context, plus associations between them referenced by
    name and type. Verifies the execution succeeds and the server-reported
    step metadata contains an ARN for every created entity.
    """
    stamp = uuid.uuid4().hex[:8]
    action_name = f"lineage-integ-action-{stamp}"
    artifact_name = f"lineage-integ-artifact-{stamp}"
    context_name = f"lineage-integ-context-{stamp}"
    pipeline_name = f"integ-lineage-{stamp}"

    action_args = Action.create(
        action_name=action_name,
        source_uri=f"s3://lineage-integ-test/{stamp}/run",
        source_type="MODEL",
        action_type="ModelTraining",
        status="Completed",
        description="Lineage integ test action",
        sagemaker_session=pipeline_session,
    )
    artifact_args = Artifact.create(
        artifact_name=artifact_name,
        source_uri=f"s3://lineage-integ-test/{stamp}/model.tar.gz",
        artifact_type="Model",
        sagemaker_session=pipeline_session,
    )
    context_args = Context.create(
        context_name=context_name,
        source_uri=f"s3://lineage-integ-test/{stamp}/experiment",
        context_type="Experiment",
        description="Lineage integ test context",
        sagemaker_session=pipeline_session,
    )

    # Associations reference entities created by this same step by name and
    # type. The service resolves them against the entities it just created,
    # so a single step covers create-then-associate end to end.
    step = LineageStep(
        name="RecordLineage",
        step_args=[action_args, artifact_args, context_args],
        associations=[
            LineageAssociation(
                source=LineageEntityReference(name=action_name, type="Action"),
                destination=LineageEntityReference(name=artifact_name, type="Artifact"),
                association_type="Produced",
            ),
            LineageAssociation(
                source=LineageEntityReference(name=context_name, type="Context"),
                destination=LineageEntityReference(name=action_name, type="Action"),
                association_type="AssociatedWith",
            ),
        ],
    )
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step],
        sagemaker_session=pipeline_session,
    )

    # Bound before the try so cleanup never raises NameError if the execution
    # fails before the metadata is read.
    action_arns: dict = {}
    artifact_arns: dict = {}
    context_arns: dict = {}

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
        lineage_metadata = lineage_step.get("Metadata", {}).get("Lineage", {})

        action_arns = lineage_metadata.get("ActionArns", {})
        artifact_arns = lineage_metadata.get("ArtifactArns", {})
        context_arns = lineage_metadata.get("ContextArns", {})
        assert (
            action_name in action_arns
        ), f"expected {action_name} in ActionArns, got: {action_arns}"
        assert action_arns[action_name].endswith(f":action/{action_name}")
        assert (
            artifact_name in artifact_arns
        ), f"expected {artifact_name} in ArtifactArns, got: {artifact_arns}"
        assert ":artifact/" in artifact_arns[artifact_name]
        assert (
            context_name in context_arns
        ), f"expected {context_name} in ContextArns, got: {context_arns}"
        assert context_arns[context_name].endswith(f":context/{context_name}")

        # Both associations were added, proving the name-and-type references
        # resolved against the entities this same step created.
        associations = lineage_metadata.get("Associations", [])
        assert len(associations) == 2, f"expected 2 associations, got: {associations}"

    finally:
        client = sagemaker_session.sagemaker_client
        action_arn = action_arns.get(action_name)
        artifact_arn = artifact_arns.get(artifact_name)
        context_arn = context_arns.get(context_name)

        deletes = []
        if action_arn and artifact_arn:
            deletes.append(
                lambda: client.delete_association(SourceArn=action_arn, DestinationArn=artifact_arn)
            )
        if context_arn and action_arn:
            deletes.append(
                lambda: client.delete_association(SourceArn=context_arn, DestinationArn=action_arn)
            )
        deletes.append(lambda: client.delete_action(ActionName=action_name))
        if artifact_arn:
            deletes.append(lambda: client.delete_artifact(ArtifactArn=artifact_arn))
        deletes.append(lambda: client.delete_context(ContextName=context_name))
        deletes.append(lambda: client.delete_pipeline(PipelineName=pipeline_name))

        for delete in deletes:
            try:
                delete()
            except Exception:  # noqa: BLE001 -- best-effort cleanup
                pass
