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
"""Integration test for managed pipelines.

Lists managed pipelines with ``managed_by="AWS"``, starts one with the caller's
role, waits for it to succeed, and checks the role is reported on the execution.
Skips when no managed pipeline is visible to the calling account.

The execution role defaults to the ``SageMakerRole`` fixture and can be
overridden with ``MANAGED_PIPELINE_EXECUTION_ROLE_ARN``.
"""

from __future__ import absolute_import

import os

import pytest

from sagemaker.core.resources import Pipeline as CorePipeline
from sagemaker.core.resources import PipelineExecution as CorePipelineExecution
from sagemaker.mlops.workflow.pipeline import Pipeline

MANAGED_PIPELINE_NAME = "aws-system-pipeline-definition-validation"


@pytest.fixture(scope="module")
def execution_role(request):
    override = os.environ.get("MANAGED_PIPELINE_EXECUTION_ROLE_ARN")
    return override or request.getfixturevalue("role")


@pytest.fixture(scope="module")
def managed_pipeline_arn(sagemaker_session):
    pipelines = list(
        CorePipeline.get_all(
            managed_by="AWS",
            session=sagemaker_session.boto_session,
            region=sagemaker_session.boto_region_name,
        )
    )
    for pipeline in pipelines:
        assert ":aws:pipeline/" in pipeline.pipeline_arn
    arns = [p.pipeline_arn for p in pipelines if p.pipeline_name == MANAGED_PIPELINE_NAME]
    if not arns:
        pytest.skip(f"{MANAGED_PIPELINE_NAME} is not visible to this account")
    return arns[0]


def test_start_managed_pipeline_with_role(sagemaker_session, managed_pipeline_arn, execution_role):
    pipeline = Pipeline(name=managed_pipeline_arn, sagemaker_session=sagemaker_session)

    execution = pipeline.start(role_arn=execution_role)
    execution.wait(delay=15, max_attempts=40)

    assert execution.describe()["PipelineExecutionStatus"] == "Succeeded"
    described = CorePipelineExecution.get(
        pipeline_execution_arn=execution.arn,
        session=sagemaker_session.boto_session,
        region=sagemaker_session.boto_region_name,
    )
    assert described.role_arn == execution_role
