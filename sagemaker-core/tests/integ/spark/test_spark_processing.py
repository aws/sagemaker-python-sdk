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
"""End-to-end integration tests for the V3 ``PySparkProcessor``.

V3 shipped with NO Spark processing integ tests; this one is ported from V2's
``tests/integ/test_spark_processing.py`` to the V3 APIs and locks in two fixes
that are only observable against the real service:

- #6253: the Spark event-log output stopped landing in S3. The job below
  asserts the event-log S3 prefix is non-empty after the run.
- #6252: conf / py-files input channels were emitted in the wrong shape. We
  assert their Describe ProcessingInputs carry an ``S3Input`` with a
  ``LocalPath`` under ``/opt/ml/processing/input/``.

#3809 (``submit_py_files`` must be a list) is an argument guard that fires before
any API call and is covered by unit tests.
"""

from __future__ import absolute_import

import os
import time
import uuid

import boto3
import pytest

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.spark.processing import PySparkProcessor

ROLE = "SageMakerRole"
REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
INSTANCE_TYPE = "ml.m5.xlarge"
FRAMEWORK_VERSION = "3.3"
WAIT_TIMEOUT_SECONDS = 30 * 60
POLL_SECONDS = 20
TERMINAL = ("Completed", "Failed", "Stopped")

_DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data", "spark")
_SPARK_APP = os.path.join(_DATA_DIR, "code", "python", "hello_py_spark", "hello_py_spark_app.py")
_SPARK_UDFS = os.path.join(_DATA_DIR, "code", "python", "hello_py_spark", "hello_py_spark_udfs.py")
_DATA_JSONL = os.path.join(_DATA_DIR, "files", "data.jsonl")

CONFIGURATION = [
    {
        "Classification": "spark-defaults",
        "Properties": {"spark.executor.memory": "2g", "spark.executor.cores": "1"},
    },
]


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


def _input_by_name(described, name):
    for inp in described.get("ProcessingInputs", []):
        if inp.get("InputName") == name:
            return inp
    return None


@pytest.mark.serial
def test_pyspark_multinode_event_logs_and_input_shape_6253_6252():
    """#6253 (event logs land in S3) and #6252 (conf/py-files input shape)."""
    client = boto3.client("sagemaker", region_name=REGION)
    session = Session(sagemaker_client=client)
    s3_client = boto3.client("s3", region_name=REGION)

    processor = PySparkProcessor(
        role=ROLE,
        instance_count=2,
        instance_type=INSTANCE_TYPE,
        framework_version=FRAMEWORK_VERSION,
        max_runtime_in_seconds=1800,
        sagemaker_session=session,
    )

    bucket = session.default_bucket()
    run_id = uuid.uuid4().hex[:8]
    output_data_uri = f"s3://{bucket}/spark/output/sales/{run_id}"
    event_logs_prefix = f"spark/spark-events/{run_id}"
    event_logs_s3_uri = f"s3://{bucket}/{event_logs_prefix}"

    # Upload the input data set.
    input_data_uri = f"s3://{bucket}/spark/input/{run_id}/data.jsonl"
    with open(_DATA_JSONL) as data:
        s3_client.put_object(
            Bucket=bucket, Key=input_data_uri[len(f"s3://{bucket}/") :], Body=data.read()
        )

    job_name = f"pyspark-multinode-{run_id}"
    try:
        processor.run(
            submit_app=_SPARK_APP,
            submit_py_files=[_SPARK_UDFS],
            arguments=["--input", input_data_uri, "--output", output_data_uri],
            configuration=CONFIGURATION,
            spark_event_logs_s3_uri=event_logs_s3_uri,
            job_name=job_name,
            wait=False,
        )

        described = client.describe_processing_job(ProcessingJobName=job_name)

        # --- #6252: conf and py-files channels carry the corrected S3Input shape.
        for channel in ("conf", "py-files"):
            inp = _input_by_name(described, channel)
            assert inp is not None, f"no {channel!r} ProcessingInput: {described}"
            assert "S3Input" in inp, f"{channel!r} input missing S3Input: {inp}"
            local_path = inp["S3Input"].get("LocalPath", "")
            assert local_path.startswith("/opt/ml/processing/input/"), (
                f"{channel!r} LocalPath={local_path!r} not under /opt/ml/processing/input/ "
                f"(#6252 shape regression)"
            )

        # --- Wait for terminal, then #6253: event logs must be present in S3.
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
        assert (
            status == "Completed"
        ), f"job status={status!r} (expected Completed) job={job_name} failure={failure!r}"

        listed = s3_client.list_objects_v2(Bucket=bucket, Prefix=event_logs_prefix)
        assert listed.get("KeyCount", 0) > 0, (
            f"no Spark event logs under s3://{bucket}/{event_logs_prefix} -- "
            f"event-log output did not land in S3 (#6253)"
        )
    finally:
        _stop_quietly(client, job_name)
