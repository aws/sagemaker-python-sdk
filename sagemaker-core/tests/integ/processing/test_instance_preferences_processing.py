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
"""End-to-end integration test for Instance Preferences on processing jobs.

Launches a REAL processing job whose ``ClusterConfig`` carries an ordered
``InstancePreferences`` list (no ``InstanceType``) through the ``Processor``
path, then asserts the Describe contract:

- the create request is accepted with ``InstancePreferences`` + the uniform
  ``InstanceCount`` only;
- Describe echoes ``InstancePreferences`` and does not return the top-level
  ``InstanceType`` the customer never set;
- once the job leaves the pre-instance states, ``SelectedInstanceType`` /
  ``SelectedInstanceCount`` report the resolved winner, which must be one of
  the submitted preferences.

Runs in the standard integration-test account: the execution role is the
suite's ``SageMakerRole`` and the image is resolved through ``image_uris``, the
same way the other ``sagemaker-core`` integration tests obtain theirs.
``PROCESSING_INSTANCE_PREFERENCES_TEST_INSTANCE_TYPES`` (comma-separated)
overrides the candidate types for accounts where the defaults lack quota.
"""

from __future__ import absolute_import

import os
import time
import uuid

import boto3

from sagemaker.core import image_uris

ROLE = "SageMakerRole"
REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))

PREFERENCE_TYPES = [
    t.strip()
    for t in os.environ.get(
        "PROCESSING_INSTANCE_PREFERENCES_TEST_INSTANCE_TYPES", "ml.m5.xlarge,ml.m5.large"
    ).split(",")
    if t.strip()
]
WAIT_TIMEOUT_SECONDS = 30 * 60
POLL_SECONDS = 30
TERMINAL = ("Completed", "Failed", "Stopped")


def _stop_quietly(client, job_name):
    """The job has done its part once a winner is selected; do not leave it
    running to max_runtime on shared quota."""
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
        "sklearn", REGION, version="1.2-1", py_version="py3", instance_type=PREFERENCE_TYPES[0]
    )


def test_processor_instance_preferences_e2e():
    """Create a real processing job with InstancePreferences and verify the winner."""
    from sagemaker.core.helper.session_helper import Session
    from sagemaker.core.processing import Processor

    client = boto3.client("sagemaker", region_name=REGION)
    session = Session(sagemaker_client=client)

    processor = Processor(
        role=ROLE,
        image_uri=_processing_image(),
        instance_count=1,
        instance_preferences=[{"InstanceType": t} for t in PREFERENCE_TYPES],
        volume_size_in_gb=30,
        max_runtime_in_seconds=1800,
        sagemaker_session=session,
    )

    job_name = f"instance-prefs-proc-integ-{uuid.uuid4().hex[:8]}"
    processor.run(wait=False, logs=False, job_name=job_name)

    try:

        # --- Create accepted; Describe echoes the request contract -------------
        described = client.describe_processing_job(ProcessingJobName=job_name)
        cluster_config = described["ProcessingResources"]["ClusterConfig"]
        assert [p["InstanceType"] for p in cluster_config.get("InstancePreferences", [])] == (
            PREFERENCE_TYPES
        ), f"Describe did not echo InstancePreferences: {cluster_config}"
        # The customer never set the top-level InstanceType; it must not come
        # back populated on Describe.
        assert "InstanceType" not in cluster_config, cluster_config

        # --- Wait for a terminal-or-resolved state ------------------------------
        deadline = time.time() + WAIT_TIMEOUT_SECONDS
        status = described["ProcessingJobStatus"]
        while time.time() < deadline:
            described = client.describe_processing_job(ProcessingJobName=job_name)
            status = described["ProcessingJobStatus"]
            cluster_config = described["ProcessingResources"]["ClusterConfig"]
            if status in TERMINAL:
                break
            if status == "InProgress" and cluster_config.get("SelectedInstanceType"):
                break
            time.sleep(POLL_SECONDS)

        failure_reason = described.get("FailureReason", "")

        # --- Full contract: resolved winner is surfaced and is a submitted pref -
        cluster_config = described["ProcessingResources"]["ClusterConfig"]
        selected_type = cluster_config.get("SelectedInstanceType")
        selected_count = cluster_config.get("SelectedInstanceCount")
        assert selected_type in PREFERENCE_TYPES, (
            f"SelectedInstanceType={selected_type!r} not among submitted preferences "
            f"{PREFERENCE_TYPES} (job={job_name}, status={status}, "
            f"failure={failure_reason!r})"
        )
        assert selected_count == 1
        assert "InstanceType" not in cluster_config, cluster_config
    finally:
        _stop_quietly(client, job_name)
