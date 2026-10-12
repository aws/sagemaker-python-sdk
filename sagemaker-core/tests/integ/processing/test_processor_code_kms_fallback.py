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
"""End-to-end integration test locking in issue #4874.

#4874: ``ScriptProcessor.run(code=...)`` uploaded the user code file with no
server-side encryption even when the processor was configured with an
``output_kms_key``. The fix makes ``kms_key`` fall back to the processor's
``output_kms_key`` when the caller passes no explicit ``kms_key``.

A pass proves the *upload path*, not the job outcome: we submit a real
processing job with ``output_kms_key`` set and NO ``kms_key`` argument, then
read the uploaded code object's S3 metadata straight from Describe and assert it
was encrypted with that same KMS key. The job is stopped immediately -- the fix
lives entirely in the request/upload path, so we never wait for completion.
"""

from __future__ import absolute_import

import os
import tempfile
import uuid

import boto3

from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.processing import ScriptProcessor
from tests.integ.integ_test_kms_helpers import get_or_create_kms_key

ROLE = "SageMakerRole"
REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
INSTANCE_TYPE = "ml.m5.xlarge"
TERMINAL = ("Completed", "Failed", "Stopped")


def _stop_quietly(client, job_name):
    """The upload has already happened by the time Describe returns the code
    input, so there is no reason to let the job run on shared quota."""
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


def _code_s3_uri(described):
    """Pull the uploaded code object's S3 URI out of the Describe response.

    ScriptProcessor uploads the user code as a ProcessingInput named ``code``.
    """
    for inp in described.get("ProcessingInputs", []):
        if inp.get("InputName") == "code":
            return inp["S3Input"]["S3Uri"]
    raise AssertionError(f"no 'code' ProcessingInput in Describe: {described}")


def test_script_processor_code_kms_falls_back_to_output_kms_key_4874():
    """output_kms_key must encrypt the uploaded code when no kms_key is given."""
    client = boto3.client("sagemaker", region_name=REGION)
    session = Session(sagemaker_client=client)

    role_arn = boto3.client("iam").get_role(RoleName=ROLE)["Role"]["Arn"]
    kms_key_arn = get_or_create_kms_key(session, role_arn=role_arn)
    key_id = kms_key_arn.split("/")[-1]

    processor = ScriptProcessor(
        role=ROLE,
        image_uri=_processing_image(),
        command=["python3"],
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=30,
        max_runtime_in_seconds=1800,
        output_kms_key=kms_key_arn,
        sagemaker_session=session,
    )

    job_name = f"code-kms-fallback-{uuid.uuid4().hex[:8]}"
    tmpdir = tempfile.mkdtemp()
    code_path = os.path.join(tmpdir, "noop.py")
    with open(code_path, "w") as fh:
        fh.write("print('hello from kms fallback test')\n")

    try:
        # NOTE: no kms_key argument -- the fix must fall back to output_kms_key.
        processor.run(code=code_path, wait=False, logs=False, job_name=job_name)

        described = client.describe_processing_job(ProcessingJobName=job_name)
        code_uri = _code_s3_uri(described)

        bucket, key = code_uri[len("s3://") :].split("/", 1)
        head = boto3.client("s3", region_name=REGION).head_object(Bucket=bucket, Key=key)

        assert head.get("ServerSideEncryption") == "aws:kms", (
            f"uploaded code was not KMS-encrypted (SSE={head.get('ServerSideEncryption')!r}); "
            f"output_kms_key did not flow to the code upload (job={job_name})"
        )
        sse_key = head.get("SSEKMSKeyId", "")
        assert sse_key.endswith(key_id) or sse_key == kms_key_arn, (
            f"code encrypted with {sse_key!r}, expected the processor's output_kms_key "
            f"{kms_key_arn!r} (job={job_name})"
        )
    finally:
        _stop_quietly(client, job_name)
        try:
            os.remove(code_path)
            os.rmdir(tmpdir)
        except OSError:
            pass
