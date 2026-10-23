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
"""Integration test for CleanupStep.

Runs a Processing step that writes two objects, then a CleanupStep that
deletes one of them by URI. Requires the execution role to have
``s3:DeleteObject`` on the session's default bucket.
"""

from __future__ import absolute_import

import time
import uuid

import pytest
from botocore.exceptions import ClientError

from sagemaker.core import image_uris
from sagemaker.core.processing import Processor
from sagemaker.core.shapes import ProcessingOutput, ProcessingS3Output
from sagemaker.mlops.workflow import CleanupStep, S3CleanupSource
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import ProcessingStep


def _object_exists(s3_client, bucket, key):
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def test_cleanup_step_deletes_processing_output(
    sagemaker_session, pipeline_session, role, sklearn_latest_version
):
    stamp = uuid.uuid4().hex[:8]
    bucket = sagemaker_session.default_bucket()
    prefix = f"integ-test-cleanup-step/{stamp}"
    output_uri = f"s3://{bucket}/{prefix}/output"
    region = pipeline_session.boto_region_name

    processor = Processor(
        role=role,
        image_uri=image_uris.retrieve(
            framework="sklearn",
            region=region,
            version=sklearn_latest_version,
            py_version="py3",
            instance_type="ml.m5.large",
        ),
        instance_count=1,
        instance_type="ml.m5.large",
        entrypoint=[
            "/bin/sh",
            "-c",
            "echo keep > /opt/ml/processing/output/keep.txt && "
            "echo delete > /opt/ml/processing/output/delete.txt",
        ],
        sagemaker_session=pipeline_session,
    )
    step_process = ProcessingStep(
        name="WriteOutput",
        step_args=processor.run(
            outputs=[
                ProcessingOutput(
                    output_name="output",
                    s3_output=ProcessingS3Output(
                        s3_uri=output_uri,
                        local_path="/opt/ml/processing/output",
                        s3_upload_mode="EndOfJob",
                    ),
                )
            ],
        ),
    )
    step_cleanup = CleanupStep(
        name="CleanupOutput",
        s3_sources=[S3CleanupSource(step=step_process, s3_uris=[f"{output_uri}/delete.txt"])],
    )
    pipeline = Pipeline(
        name=f"integ-cleanup-step-{stamp}",
        steps=[step_process, step_cleanup],
        sagemaker_session=pipeline_session,
    )

    client = sagemaker_session.sagemaker_client
    s3_client = sagemaker_session.boto_session.client("s3")
    try:
        pipeline.upsert(role_arn=role)
        execution = pipeline.start()

        final_status = None
        deadline = time.time() + 1200
        while time.time() < deadline:
            final_status = execution.describe()["PipelineExecutionStatus"]
            if final_status in ("Succeeded", "Failed", "Stopped", "CompletedWithFailures"):
                break
            time.sleep(30)

        steps = client.list_pipeline_execution_steps(PipelineExecutionArn=execution.arn)[
            "PipelineExecutionSteps"
        ]
        if final_status != "Succeeded":
            failures = "\n".join(
                f"{s['StepName']}: {s.get('FailureReason', 'no reason')}"
                for s in steps
                if s.get("StepStatus") == "Failed"
            )
            pytest.fail(f"Pipeline execution status={final_status}. Details:\n{failures}")

        cleanup = next(s for s in steps if s["StepName"] == "CleanupOutput")
        assert cleanup["Metadata"]["Cleanup"]["CleanedSteps"] == ["WriteOutput"]
        assert not _object_exists(s3_client, bucket, f"{prefix}/output/delete.txt")
        assert _object_exists(s3_client, bucket, f"{prefix}/output/keep.txt")
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass
        for obj in s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", []):
            s3_client.delete_object(Bucket=bucket, Key=obj["Key"])
