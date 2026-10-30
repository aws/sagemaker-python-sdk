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
"""Integration tests for HubContentStep.

Imports Model hub content through SageMaker Pipelines and checks that:

* the step succeeds, a pipeline parameter inside the document resolves, and a
  ``ConditionStep`` can read the step's ``HubContentStatus`` and ``HubContentType``,
* re-running a pipeline that omits ``hub_content_version`` imports a new version,
* importing into a hub that does not exist drives the step to ``Failed``.

Requirements on top of the shared workflow integ fixtures:

* The test identity needs ``sagemaker:CreateHub``, ``DescribeHub``, ``DeleteHub``,
  ``ListHubContentVersions``, ``DescribeHubContent`` and ``DeleteHubContent``.
* The pipeline execution role (``role`` fixture) needs ``sagemaker:ImportHubContent``
  and ``sagemaker:DescribeHubContent``.
* The HubContent step type must be available to the account and Region;
  otherwise ``CreatePipeline`` rejects the pipeline definition.
"""

from __future__ import absolute_import

import time

import pytest

from sagemaker.core.common_utils import unique_name_from_base
from sagemaker.core.workflow.conditions import ConditionEquals
from sagemaker.core.workflow.parameters import ParameterString
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.mlops.workflow import ConditionStep, FailStep, HubContentStep
from sagemaker.mlops.workflow.pipeline import Pipeline

from tests.integ.workflow.helpers import wait_pipeline_execution

_MODEL_SCHEMA_VERSION = "2.4.0"
_URL_PARAMETER = "ModelUrl"
_MODEL_URL = "https://example.com/hub-content-step-integ-model"


def _model_document(url):
    return {
        "Url": url,
        "ModelTypes": ["OPEN_WEIGHTS"],
        "Capabilities": [],
        "TrainingSupported": False,
        "IncrementalTrainingSupported": False,
    }


@pytest.fixture(scope="module")
def pipeline_session_for_hub(sagemaker_session_for_pipeline):
    return PipelineSession(
        boto_session=sagemaker_session_for_pipeline.boto_session,
        sagemaker_client=sagemaker_session_for_pipeline.sagemaker_client,
    )


@pytest.fixture(scope="module")
def hub_name(sagemaker_session_for_pipeline):
    client = sagemaker_session_for_pipeline.sagemaker_client
    name = unique_name_from_base("hub-content-step-integ")
    client.create_hub(HubName=name, HubDescription="HubContentStep integ test")
    for _ in range(30):
        if client.describe_hub(HubName=name)["HubStatus"] == "InService":
            break
        time.sleep(2)
    yield name
    for content in client.list_hub_contents(HubName=name, HubContentType="Model")[
        "HubContentSummaries"
    ]:
        versions = client.list_hub_content_versions(
            HubName=name, HubContentType="Model", HubContentName=content["HubContentName"]
        )["HubContentSummaries"]
        for version in versions:
            try:
                client.delete_hub_content(
                    HubName=name,
                    HubContentType="Model",
                    HubContentName=content["HubContentName"],
                    HubContentVersion=version["HubContentVersion"],
                )
            except Exception:
                pass
    try:
        client.delete_hub(HubName=name)
    except Exception:
        pass


def _import_step(pipeline_session, hub, content_name, url):
    return HubContentStep(
        name="ImportModel",
        step_args=pipeline_session.import_hub_content(
            hub_name=hub,
            hub_content_name=content_name,
            hub_content_type="Model",
            document_schema_version=_MODEL_SCHEMA_VERSION,
            hub_content_document=_model_document(url),
        ),
    )


def _run(pipeline, parameters=None):
    """Start one execution, wait for it and return ``(status, steps by name)``."""
    execution = pipeline.start(parameters=parameters or {})
    wait_pipeline_execution(execution=execution, delay=10, max_attempts=60)
    status = execution.describe()["PipelineExecutionStatus"]
    steps = {step["StepName"]: step for step in execution.list_steps()}
    return status, steps


def test_hub_content_step_imports_and_exposes_outputs(
    sagemaker_session_for_pipeline, pipeline_session_for_hub, hub_name, role
):
    """The import succeeds with a parameterized document, and its outputs feed a condition."""
    content_name = unique_name_from_base("hc-model")
    url = ParameterString(name=_URL_PARAMETER, default_value=_MODEL_URL)
    import_step = _import_step(pipeline_session_for_hub, hub_name, content_name, url)
    verify = ConditionStep(
        name="VerifyOutputs",
        conditions=[
            ConditionEquals(left=import_step.properties.HubContentStatus, right="Available"),
            ConditionEquals(left=import_step.properties.HubContentType, right="Model"),
        ],
        if_steps=[],
        else_steps=[FailStep(name="OutputsWrong", error_message="unexpected outputs")],
    )
    pipeline = Pipeline(
        name=unique_name_from_base("hub-content-step-integ"),
        parameters=[url],
        steps=[import_step, verify],
        sagemaker_session=pipeline_session_for_hub,
    )

    try:
        pipeline.create(role_arn=role)
        status, steps = _run(pipeline)
        assert status == "Succeeded", {n: s.get("FailureReason") for n, s in steps.items()}
        assert steps["ImportModel"]["StepStatus"] == "Succeeded"
        assert steps["VerifyOutputs"]["Metadata"]["Condition"]["Outcome"] == "True"

        content = sagemaker_session_for_pipeline.sagemaker_client.describe_hub_content(
            HubName=hub_name, HubContentType="Model", HubContentName=content_name
        )
        assert content["HubContentStatus"] == "Available"
        assert _MODEL_URL in content["HubContentDocument"]
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_hub_content_step_unpinned_version_increments(
    sagemaker_session_for_pipeline, pipeline_session_for_hub, hub_name, role
):
    """Without ``hub_content_version`` each execution imports a new version."""
    content_name = unique_name_from_base("hc-model")
    pipeline = Pipeline(
        name=unique_name_from_base("hub-content-step-integ"),
        steps=[_import_step(pipeline_session_for_hub, hub_name, content_name, _MODEL_URL)],
        sagemaker_session=pipeline_session_for_hub,
    )

    try:
        pipeline.create(role_arn=role)
        for _ in range(2):
            status, steps = _run(pipeline)
            assert status == "Succeeded", steps["ImportModel"].get("FailureReason")

        versions = sagemaker_session_for_pipeline.sagemaker_client.list_hub_content_versions(
            HubName=hub_name, HubContentType="Model", HubContentName=content_name
        )["HubContentSummaries"]
        assert len(versions) == 2, versions
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_hub_content_step_missing_hub_fails(pipeline_session_for_hub, role):
    """Importing into a hub that does not exist fails the step with a reason."""
    pipeline = Pipeline(
        name=unique_name_from_base("hub-content-step-integ"),
        steps=[
            _import_step(
                pipeline_session_for_hub,
                unique_name_from_base("missing-hub"),
                unique_name_from_base("hc-model"),
                _MODEL_URL,
            )
        ],
        sagemaker_session=pipeline_session_for_hub,
    )

    try:
        pipeline.create(role_arn=role)
        status, steps = _run(pipeline)
        assert status == "Failed"
        assert steps["ImportModel"]["StepStatus"] == "Failed"
        assert steps["ImportModel"].get("FailureReason")
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass
