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
"""End-to-end Local Mode tests for the sagemaker-core session/runtime fixes.

Each test locks in one bug fixed in this PR and states what a pass proves:

- #5562 -- ``Processor`` no longer requires an IAM role when ``instance_type`` is
  ``local``/``local_gpu``. A pass proves a real local processing job runs through
  Docker with ``role=None`` and reaches ``Completed`` with no role ``ValueError``
  and no ``expand_role(None)`` failure while building the request.
- #3348 + #4996 -- ``LocalSagemakerRuntimeClient.invoke_endpoint`` returns the
  full SageMaker runtime response shape (``Body``, ``ContentType`` taken from the
  container's response header, ``InvokedProductionVariant`` and
  ``ResponseMetadata``) and binary payloads round-trip through the local invoke
  path unchanged. A pass drives a real container endpoint end-to-end through
  Docker and asserts the response contract and the bytes round-trip.

#4417 (``describe_user_profile`` passthrough) is covered by unit tests only: an
integ test would just prove that a boto client can call the service.

These tests need Docker (local mode); #5562 pulls a public SageMaker ECR image.
Where a local blocker prevents a step (ECR pull denied) the test skips with a
precise reason rather than weakening the assertion. Marked ``local_mode`` and
``serial`` so local-mode tests do not run concurrently.
"""

from __future__ import absolute_import

import fcntl
import os
import shutil
import tempfile
import textwrap
import time
import uuid
from contextlib import contextmanager

import boto3
import botocore.exceptions
import pytest

from sagemaker.core import image_uris
from sagemaker.core.local.local_session import (
    LocalSagemakerClient,
    LocalSagemakerRuntimeClient,
    LocalSession,
)
from sagemaker.core.processing import Processor

REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
LOCK_PATH = os.path.join(tempfile.gettempdir(), "sagemaker_test_local_mode_lock")

pytestmark = [pytest.mark.local_mode, pytest.mark.serial]


@contextmanager
def _local_mode_lock(path=LOCK_PATH):
    """Serialize local-mode tests that share Docker and the fixed 8080 port."""
    f = open(path, "w")
    try:
        fcntl.lockf(f.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        time.sleep(5)
        fcntl.lockf(f.fileno(), fcntl.LOCK_UN)
        f.close()


def _boto_session():
    return boto3.Session(region_name=REGION)


# --------------------------------------------------------------------------- #
# #5562 -- local processing job without a role
# --------------------------------------------------------------------------- #
def test_local_processor_runs_without_role_5562():
    """A local processing job must run with ``role=None``.

    Before the fix, ``Processor.__init__`` raised ``ValueError('An AWS IAM role
    is required...')`` even for ``instance_type='local'`` where the role is never
    used. A pass proves the constructor accepts ``role=None`` in local mode and
    the job runs through Docker to ``Completed``.
    """
    with _local_mode_lock():
        image_uri = image_uris.retrieve(
            "sklearn", REGION, version="1.2-1", py_version="py3", instance_type="ml.m5.large"
        )

        # Constructing with role=None must NOT raise -- this is the fix itself.
        processor = Processor(
            image_uri=image_uri,
            instance_type="local",
            instance_count=1,
            role=None,
            sagemaker_session=LocalSession(boto_session=_boto_session()),
            entrypoint=["python3", "-c", 'print("ok")'],
        )
        assert processor.role is None

        job_name = "local-proc-no-role-%s" % uuid.uuid4().hex[:8]
        try:
            # LocalSagemakerClient.create_processing_job runs the container synchronously,
            # so the job is finished when run() returns. wait=False because the V3
            # ProcessingJob resource waiter polls the real service client, not local mode.
            processor.run(wait=False, logs=False, job_name=job_name)
        except botocore.exceptions.ClientError as err:
            code = err.response.get("Error", {}).get("Code", "")
            if code in ("AccessDenied", "AccessDeniedException", "UnrecognizedClientException"):
                pytest.skip(f"ECR pull for the sklearn image denied here: {code}")
            raise
        except Exception as err:  # pylint: disable=broad-except
            msg = str(err).lower()
            if "pull" in msg and ("denied" in msg or "unauthorized" in msg or "403" in msg):
                pytest.skip(f"ECR image pull denied locally: {err}")
            if "toomanyrequests" in msg or "rate limit" in msg:
                pytest.skip(f"Registry pull rate-limited locally: {err}")
            raise

        described = processor.sagemaker_session.sagemaker_client.describe_processing_job(
            ProcessingJobName=job_name
        )
        assert described["ProcessingJobStatus"] == "Completed", described


# --------------------------------------------------------------------------- #
# #3348 + #4996 -- local endpoint invoke response shape and binary payload
# --------------------------------------------------------------------------- #
_DOCKERFILE = """\
FROM python:3.10-slim
COPY serve.py /serve.py
ENTRYPOINT ["python3", "/serve.py"]
"""

_SERVE_PY = textwrap.dedent('''\
    """Tiny SageMaker-style serving container for local endpoint tests.

    GET /ping -> 200. POST /invocations -> echoes the request body back with
    Content-Type: application/octet-stream and an x-amzn-RequestId header, so the
    test can prove ContentType comes from the container response (not the echoed
    Accept) and that a binary payload round-trips unchanged.
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer


    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/ping":
                self.send_response(200)
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/invocations":
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("x-amzn-RequestId", "local-echo-req-id")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args):
            pass


    if __name__ == "__main__":
        HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
    ''')


def _docker_available():
    import subprocess

    try:
        return subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    except (OSError, ValueError):
        return False


def _build_serving_image(tag):
    import subprocess

    build_dir = tempfile.mkdtemp(prefix="local-serve-")
    with open(os.path.join(build_dir, "Dockerfile"), "w") as f:
        f.write(_DOCKERFILE)
    with open(os.path.join(build_dir, "serve.py"), "w") as f:
        f.write(_SERVE_PY)
    result = subprocess.run(
        ["docker", "build", "-t", tag, build_dir], capture_output=True, text=True
    )
    return result


def test_local_endpoint_invoke_response_shape_and_binary_3348_4996():
    """Local endpoint invoke returns the full runtime shape and round-trips bytes.

    Before the fix, ``invoke_endpoint`` returned only ``{"Body", "ContentType"}``
    (ContentType echoing the request Accept), so callers written against the real
    runtime client's ``ResponseMetadata``/``InvokedProductionVariant`` shape broke
    in local mode (#3348), and the response ContentType did not reflect what the
    container actually returned. A pass drives a real container endpoint through
    Docker and asserts:

    - ``Body`` is present and the bytes payload round-trips unchanged (locks the
      binary path shared with the #4996 MultiRecordStrategy fix -- see note below);
    - ``ContentType`` comes from the container's response header
      (``application/octet-stream``), not the echoed ``Accept``;
    - ``InvokedProductionVariant`` is present;
    - ``ResponseMetadata`` carries ``RequestId`` (from the container's
      ``x-amzn-RequestId``), ``HTTPStatusCode``, ``HTTPHeaders`` and
      ``RetryAttempts``.
    """
    if not _docker_available():
        pytest.skip("Docker is not available for local endpoint test.")

    with _local_mode_lock():
        tag = "sagemaker-local-echo:%s" % uuid.uuid4().hex[:8]
        build = _build_serving_image(tag)
        if build.returncode != 0:
            stderr = (build.stderr or "").lower()
            if "toomanyrequests" in stderr or "rate limit" in stderr:
                pytest.skip("Docker Hub pull rate-limited building the serving image.")
            if "pull access denied" in stderr or "unauthorized" in stderr:
                pytest.skip("python:3.10-slim base image pull denied locally.")
            raise AssertionError("docker build failed:\n%s" % build.stderr)

        session = LocalSession(boto_session=_boto_session())
        client = LocalSagemakerClient(session)
        runtime = LocalSagemakerRuntimeClient(session.config)

        suffix = uuid.uuid4().hex[:8]
        model_name = "local-echo-model-%s" % suffix
        config_name = "local-echo-config-%s" % suffix
        endpoint_name = "local-echo-endpoint-%s" % suffix

        # _LocalEndpoint mounts ModelDataUrl at /opt/ml/model; an empty local dir is enough.
        model_dir = tempfile.mkdtemp(prefix="sagemaker-local-echo-model-")
        try:
            client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    "Image": tag,
                    "ModelDataUrl": "file://" + model_dir,
                    "Environment": {},
                },
            )
            client.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        "VariantName": "AllTraffic",
                        "ModelName": model_name,
                        "InitialInstanceCount": 1,
                        "InstanceType": "local",
                    }
                ],
            )
            client.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)

            payload = bytes(range(256))  # non-UTF-8 bytes -> proves binary round-trip
            response = runtime.invoke_endpoint(
                Body=payload,
                EndpointName=endpoint_name,
                ContentType="application/octet-stream",
                Accept="application/json",
            )

            # Response shape (#3348)
            assert "Body" in response
            assert response["ContentType"] == "application/octet-stream", response["ContentType"]
            assert "InvokedProductionVariant" in response
            metadata = response["ResponseMetadata"]
            assert metadata["RequestId"] == "local-echo-req-id", metadata
            assert metadata["HTTPStatusCode"] == 200, metadata
            assert isinstance(metadata["HTTPHeaders"], dict)
            assert metadata["RetryAttempts"] == 0

            # Binary round-trip (path shared with the #4996 buffer-type fix)
            echoed = response["Body"].read()
            assert echoed == payload, (len(echoed), len(payload))
        finally:
            client.delete_endpoint(endpoint_name)
            client.delete_endpoint_config(config_name)
            client.delete_model(model_name)
            shutil.rmtree(model_dir, ignore_errors=True)
