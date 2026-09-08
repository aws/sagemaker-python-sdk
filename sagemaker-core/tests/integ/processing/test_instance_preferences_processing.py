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

Configuration (environment variables; SKIPPED when unset):

- ``PROCESSING_INSTANCE_PREFERENCES_TEST_ROLE_ARN``  - execution role ARN
- ``PROCESSING_INSTANCE_PREFERENCES_TEST_IMAGE_URI`` - processing image
- ``SAGEMAKER_ENDPOINT``                             - optional endpoint
  override (e.g. a pre-GA stage endpoint).

Rollout note: like the training analog
(``sagemaker-train/tests/integ/train/test_instance_preferences.py``), while
the server-side scheduling/write-back stages are not yet deployed on the
target endpoint an accepted job fails with ``InternalServerError`` before an
instance type is resolved; the test reports that as XFAIL (rollout
incomplete) and enforces the full contract automatically once deployed.
"""
from __future__ import absolute_import

import os
import time
import uuid

import pytest

ROLE_ARN = os.environ.get("PROCESSING_INSTANCE_PREFERENCES_TEST_ROLE_ARN")
IMAGE_URI = os.environ.get("PROCESSING_INSTANCE_PREFERENCES_TEST_IMAGE_URI")
ENDPOINT = os.environ.get("SAGEMAKER_ENDPOINT")

PREFERENCE_TYPES = ["ml.m5.xlarge", "ml.m5.large"]
WAIT_TIMEOUT_SECONDS = 30 * 60
POLL_SECONDS = 30

pytestmark = pytest.mark.skipif(
    not (ROLE_ARN and IMAGE_URI),
    reason=(
        "Processing Instance Preferences integ test requires "
        "PROCESSING_INSTANCE_PREFERENCES_TEST_ROLE_ARN and "
        "PROCESSING_INSTANCE_PREFERENCES_TEST_IMAGE_URI"
    ),
)


def _sagemaker_client():
    """Build a botocore client on the bundled service model (which carries the
    pre-GA InstancePreferences shapes), honoring SAGEMAKER_ENDPOINT."""
    import pathlib

    import botocore.loaders
    import botocore.session as bc_session_mod
    from boto3.session import Session as Boto3Session

    # .../sagemaker-core/tests/integ/processing/<this file> -> .../sagemaker-core/sample
    sample_dir = pathlib.Path(__file__).resolve().parents[3] / "sample"
    assert (sample_dir / "sagemaker").exists(), f"bundled model dir not found: {sample_dir}"
    sample_dir = str(sample_dir)
    bc_session = bc_session_mod.get_session()
    loader = botocore.loaders.Loader(
        extra_search_paths=[sample_dir], include_default_search_paths=True
    )
    bc_session.register_component("data_loader", loader)
    region = os.environ.get("AWS_REGION", "us-west-2")
    return Boto3Session(botocore_session=bc_session, region_name=region).client(
        "sagemaker", endpoint_url=ENDPOINT
    )


def test_processor_instance_preferences_e2e():
    """Create a real processing job with InstancePreferences and verify the winner."""
    from sagemaker.core.helper.session_helper import Session
    from sagemaker.core.processing import Processor

    client = _sagemaker_client()
    session = Session(sagemaker_client=client)

    processor = Processor(
        role=ROLE_ARN,
        image_uri=IMAGE_URI,
        instance_count=1,
        instance_preferences=[{"InstanceType": t} for t in PREFERENCE_TYPES],
        volume_size_in_gb=30,
        max_runtime_in_seconds=1800,
        sagemaker_session=session,
    )

    job_name = f"instance-prefs-proc-integ-{uuid.uuid4().hex[:8]}"
    try:
        processor.run(wait=False, logs=False, job_name=job_name)
    except Exception as e:  # botocore ClientError
        message = str(e)
        if "ValidationException" in message and (
            "instanceType' failed to satisfy constraint: Member must not be null" in message
            or "InstanceType and InstanceCount greater than 0 must be specified" in message
        ):
            # The public processing model still carries @required on
            # InstanceType/InstanceCount (the relaxation is bundled with the
            # GA/Trebuchet ungating), so a preferences-only create is rejected
            # by the frontend model validation on this endpoint. The SDK-side
            # request construction and serialization succeeded.
            pytest.xfail(
                "Server rollout incomplete on this endpoint: CreateProcessingJob "
                "rejected a preferences-only ClusterConfig (public model @required "
                "relaxation on InstanceType/InstanceCount not yet deployed): "
                f"{message[:200]}"
            )
        raise

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
        if status in ("Completed", "Failed", "Stopped"):
            break
        if status == "InProgress" and cluster_config.get("SelectedInstanceType"):
            break
        time.sleep(POLL_SECONDS)

    failure_reason = described.get("FailureReason", "")
    if status == "Failed" and "InternalServerError" in failure_reason:
        pytest.xfail(
            "Server rollout incomplete on this endpoint: processing job accepted "
            "with InstancePreferences but failed with InternalServerError before "
            f"type resolution (job={job_name})"
        )

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
