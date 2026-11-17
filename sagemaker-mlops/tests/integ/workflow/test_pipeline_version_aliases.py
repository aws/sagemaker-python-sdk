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
"""Integration test for pipeline version aliases.

Uses a FailStep-only pipeline so each execution ends immediately without compute.
The pipeline role defaults to the ``SageMakerRole`` fixture and can be overridden
with ``PIPELINE_VERSION_ALIASES_ROLE_ARN``.
"""

from __future__ import absolute_import

import json
import logging
import os
import time
import uuid

import pytest
from botocore.exceptions import ClientError

from sagemaker.mlops.workflow.fail_step import FailStep
from sagemaker.mlops.workflow.pipeline import Pipeline


def _error_message(describe_response):
    # DescribePipeline returns no version ID, so each version is identified by its definition.
    return json.loads(describe_response["PipelineDefinition"])["Steps"][0]["Arguments"][
        "ErrorMessage"
    ]


@pytest.fixture(scope="module")
def pipeline_role(request):
    override = os.environ.get("PIPELINE_VERSION_ALIASES_ROLE_ARN")
    return override or request.getfixturevalue("role")


def test_pipeline_version_aliases(sagemaker_session_for_pipeline, pipeline_role):
    role = pipeline_role
    pipeline = Pipeline(
        name=f"integ-version-aliases-{uuid.uuid4().hex[:8]}",
        steps=[FailStep(name="Fail", error_message="v1")],
        sagemaker_session=sagemaker_session_for_pipeline,
    )

    try:
        pipeline.create(role_arn=role, pipeline_version_aliases=["stable"])

        stable = pipeline.describe(pipeline_version_alias="stable")
        assert _error_message(stable) == "v1"
        assert stable["PipelineVersionAliases"] == ["stable"]

        # Without a definition change there is no new version to alias.
        with pytest.raises(ClientError, match="Cannot assign aliases without a new version"):
            pipeline.update(role_arn=role, pipeline_version_aliases=["unused"])

        # A definition change creates version 2, which takes the new alias.
        pipeline.steps = [FailStep(name="Fail", error_message="v2")]
        update_response = pipeline.update(role_arn=role, pipeline_version_aliases=["candidate"])
        assert update_response["PipelineVersionId"] == 2
        assert _error_message(pipeline.describe(pipeline_version_alias="candidate")) == "v2"

        pipeline.update_pipeline_version(
            pipeline_version_id=1,
            aliases_to_add=["prod"],
            aliases_to_remove=["stable"],
        )
        assert pipeline.describe(pipeline_version_id=1)["PipelineVersionAliases"] == ["prod"]

        execution = pipeline.start(pipeline_version_alias="prod")
        assert execution.describe()["PipelineVersionId"] == 1

        # A pipeline with a running execution cannot be deleted, so wait for the FailStep.
        response = execution.describe()
        deadline = time.time() + 300
        while response["PipelineExecutionStatus"] == "Executing" and time.time() < deadline:
            time.sleep(10)
            response = execution.describe()
        assert response["PipelineExecutionStatus"] == "Failed"
        step = execution.list_steps()[0]
        assert step["Metadata"]["Fail"]["ErrorMessage"] == "v1"
    finally:
        try:
            pipeline.delete()
        except Exception as e:
            logging.warning("Failed to delete pipeline %s: %s", pipeline.name, e)
