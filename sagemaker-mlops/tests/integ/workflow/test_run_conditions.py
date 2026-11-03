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
"""Integration test for run conditions (StepDependency triggers).

A Processing step fails on purpose. A step that depends on it with the Failed trigger
runs, and a step with a plain (Completed) dependency does not.
"""

from __future__ import absolute_import

import logging
import time
import uuid

from sagemaker.core import image_uris
from sagemaker.core.processing import Processor
from sagemaker.mlops.workflow import StepDependency, TriggerCondition
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import ProcessingStep


def _shell_step(name, command, image_uri, role, pipeline_session, depends_on=None):
    processor = Processor(
        role=role,
        image_uri=image_uri,
        instance_count=1,
        instance_type="ml.m5.large",
        entrypoint=["/bin/sh", "-c", command],
        sagemaker_session=pipeline_session,
    )
    return ProcessingStep(name=name, step_args=processor.run(), depends_on=depends_on)


def test_failed_trigger_runs_after_upstream_failure(pipeline_session, role, sklearn_latest_version):
    stamp = uuid.uuid4().hex[:8]
    image_uri = image_uris.retrieve(
        framework="sklearn",
        region=pipeline_session.boto_region_name,
        version=sklearn_latest_version,
        py_version="py3",
        instance_type="ml.m5.large",
    )

    work = _shell_step("Work", "exit 1", image_uri, role, pipeline_session)
    on_failure = _shell_step(
        "OnFailure",
        "echo recovered",
        image_uri,
        role,
        pipeline_session,
        depends_on=[StepDependency(work, TriggerCondition.FAILED)],
    )
    on_success = _shell_step(
        "OnSuccess", "echo done", image_uri, role, pipeline_session, depends_on=[work]
    )
    pipeline = Pipeline(
        name=f"integ-run-conditions-{stamp}",
        steps=[work, on_failure, on_success],
        sagemaker_session=pipeline_session,
    )

    try:
        pipeline.upsert(role_arn=role)
        execution = pipeline.start()

        status = None
        deadline = time.time() + 1800
        while time.time() < deadline:
            status = execution.describe()["PipelineExecutionStatus"]
            if status in ("Succeeded", "Failed", "Stopped", "CompletedWithFailures"):
                break
            time.sleep(30)

        step_status = {s["StepName"]: s["StepStatus"] for s in execution.list_steps()}
        assert status == "CompletedWithFailures", f"status={status}, steps={step_status}"
        assert step_status["Work"] == "Failed"
        assert step_status["OnFailure"] == "Succeeded"
        assert "OnSuccess" not in step_status
    finally:
        try:
            pipeline.delete()
        except Exception as e:
            logging.warning("Failed to delete pipeline %s: %s", pipeline.name, e)
