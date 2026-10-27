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
"""End-to-end integration test locking in issue #5805.

#5805: ``FrameworkProcessor.run(requirements=...)`` generated a ``runproc.sh``
that hard-coded ``requirements.txt`` and never threaded the caller's
requirements file name through, so a differently named requirements file (or
none) was silently ignored and dependencies were not installed in the container.

Two assertions, one cheap and one full:

1. Cheap, job-outcome-independent (the exact regression): download the generated
   ``runproc.sh`` from the job's uploaded ``entrypoint`` code location and assert
   it references the requirements file name that was passed to ``run``.
2. Full: run the job to completion with an entry script that ``import art`` (a
   small pure-python package NOT baked into the sklearn image) and writes a
   marker file; a Completed status proves ``pip install -r <requirements>`` ran
   inside the container.
"""

from __future__ import absolute_import

import os
import tempfile
import time
import uuid

import boto3
import pytest

from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.processing import FrameworkProcessor

ROLE = "SageMakerRole"
REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
INSTANCE_TYPE = "ml.m5.xlarge"
REQUIREMENTS_FILE_NAME = "requirements.txt"
# Small, pure-python, and NOT present in the sklearn processing image, so a
# Completed job proves the requirements install actually ran.
EXTRA_PACKAGE = "art==6.4"
WAIT_TIMEOUT_SECONDS = 30 * 60
POLL_SECONDS = 30
TERMINAL = ("Completed", "Failed", "Stopped")

ENTRY_SCRIPT = """\
import os

import art  # noqa: F401  -- from requirements.txt, absent in the base image

out_dir = "/opt/ml/processing/output"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "marker.txt"), "w") as fh:
    fh.write("art import succeeded\\n")
"""


def _stop_quietly(client, job_name):
    try:
        if (
            client.describe_processing_job(ProcessingJobName=job_name)["ProcessingJobStatus"]
            in TERMINAL
        ):
            return
        client.stop_processing_job(ProcessingJobName=job_name)
    except Exception:  # pylint: disable=broad-except
        pass


def _processing_image():
    return image_uris.retrieve(
        "sklearn", REGION, version="1.2-1", py_version="py3", instance_type=INSTANCE_TYPE
    )


def _entrypoint_s3_uri(described):
    """FrameworkProcessor uploads the generated runproc.sh as a ProcessingInput
    named ``entrypoint``."""
    for inp in described.get("ProcessingInputs", []):
        if inp.get("InputName") == "entrypoint":
            return inp["S3Input"]["S3Uri"]
    raise AssertionError(f"no 'entrypoint' ProcessingInput in Describe: {described}")


@pytest.mark.serial
def test_framework_processor_requirements_threaded_into_runproc_5805():
    """requirements= must reach runproc.sh and get installed in the container."""
    client = boto3.client("sagemaker", region_name=REGION)
    session = Session(sagemaker_client=client)
    s3_client = boto3.client("s3", region_name=REGION)

    processor = FrameworkProcessor(
        image_uri=_processing_image(),
        role=ROLE,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        command=["python3"],
        volume_size_in_gb=30,
        max_runtime_in_seconds=1800,
        sagemaker_session=session,
    )

    job_name = f"fw-proc-reqs-{uuid.uuid4().hex[:8]}"
    source_dir = tempfile.mkdtemp()
    entry = "process.py"
    with open(os.path.join(source_dir, entry), "w") as fh:
        fh.write(ENTRY_SCRIPT)
    with open(os.path.join(source_dir, REQUIREMENTS_FILE_NAME), "w") as fh:
        fh.write(EXTRA_PACKAGE + "\n")

    try:
        processor.run(
            code=entry,
            source_dir=source_dir,
            requirements=REQUIREMENTS_FILE_NAME,
            wait=False,
            logs=False,
            job_name=job_name,
        )

        described = client.describe_processing_job(ProcessingJobName=job_name)

        # --- (1) The exact regression: runproc.sh must name the requirements file.
        runproc_uri = _entrypoint_s3_uri(described)
        bucket, key = runproc_uri[len("s3://") :].split("/", 1)
        runproc_body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        assert REQUIREMENTS_FILE_NAME in runproc_body, (
            f"runproc.sh does not reference {REQUIREMENTS_FILE_NAME!r} -- requirements were "
            f"not threaded through (#5805). runproc.sh:\n{runproc_body}"
        )

        # --- (2) Full proof: the install actually happened -> job Completed.
        deadline = time.time() + WAIT_TIMEOUT_SECONDS
        status = described["ProcessingJobStatus"]
        while time.time() < deadline and status not in TERMINAL:
            time.sleep(POLL_SECONDS)
            status = client.describe_processing_job(ProcessingJobName=job_name)[
                "ProcessingJobStatus"
            ]

        failure = client.describe_processing_job(ProcessingJobName=job_name).get(
            "FailureReason", ""
        )
        assert status == "Completed", (
            f"job status={status!r} (expected Completed); if the container could not "
            f"`import art`, requirements were not installed (job={job_name}, "
            f"failure={failure!r})"
        )
    finally:
        _stop_quietly(client, job_name)
        for name in (entry, REQUIREMENTS_FILE_NAME):
            try:
                os.remove(os.path.join(source_dir, name))
            except OSError:
                pass
        try:
            os.rmdir(source_dir)
        except OSError:
            pass
