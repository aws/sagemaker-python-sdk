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
"""End-to-end integration test for Instance Preferences (multi-instance-type).

Launches a REAL training job whose ``ResourceConfig`` carries an ordered
``InstancePreferences`` list (no top-level ``InstanceType``) through the
``ModelTrainer`` + ``Compute`` path, then asserts the Describe contract:

- the create request is accepted with ``instance_preferences`` only;
- once the job leaves PENDING, ``selected_instance_type`` /
  ``selected_instance_count`` report the resolved winner, which must be one
  of the submitted preferences;
- the top-level ``instance_type`` the customer never set is not echoed back
  populated.

Configuration (environment variables; the test is SKIPPED when unset so the
suite stays green on hosts without the test-account setup):

- ``INSTANCE_PREFERENCES_TEST_ROLE_ARN``   - SageMaker execution role ARN
- ``INSTANCE_PREFERENCES_TEST_IMAGE_URI``  - training image the account can pull
- ``INSTANCE_PREFERENCES_TEST_S3_OUTPUT``  - s3:// output path
- ``INSTANCE_PREFERENCES_TEST_S3_INPUT``   - optional s3:// input channel
- ``INSTANCE_PREFERENCES_TEST_INSTANCE_TYPES`` - optional comma-separated
  candidate types (default: ml.m5.xlarge, ml.m4.xlarge)
- ``SAGEMAKER_ENDPOINT``                   - optional endpoint override;
  honored by the sagemaker-core client loader.
"""

from __future__ import absolute_import

import os
import time
import uuid

import pytest

from sagemaker.core.utils.utils import Unassigned

ROLE_ARN = os.environ.get("INSTANCE_PREFERENCES_TEST_ROLE_ARN")
IMAGE_URI = os.environ.get("INSTANCE_PREFERENCES_TEST_IMAGE_URI")
S3_OUTPUT = os.environ.get("INSTANCE_PREFERENCES_TEST_S3_OUTPUT")
S3_INPUT = os.environ.get("INSTANCE_PREFERENCES_TEST_S3_INPUT")

PREFERENCE_TYPES = [
    t.strip()
    for t in os.environ.get(
        "INSTANCE_PREFERENCES_TEST_INSTANCE_TYPES", "ml.m5.xlarge,ml.m4.xlarge"
    ).split(",")
    if t.strip()
]
WAIT_TIMEOUT_SECONDS = 30 * 60
POLL_SECONDS = 30

pytestmark = pytest.mark.skipif(
    not (ROLE_ARN and IMAGE_URI and S3_OUTPUT),
    reason=(
        "Instance Preferences integ test requires INSTANCE_PREFERENCES_TEST_ROLE_ARN, "
        "INSTANCE_PREFERENCES_TEST_IMAGE_URI and INSTANCE_PREFERENCES_TEST_S3_OUTPUT"
    ),
)


def _get_value(field):
    """None for Unassigned/None, else the raw value."""
    if field is None or isinstance(field, Unassigned):
        return None
    return field


def test_model_trainer_instance_preferences_e2e():
    """Create a real training job with instance_preferences and verify the winner."""
    from sagemaker.core.resources import TrainingJob
    from sagemaker.core.shapes import shapes
    from sagemaker.train.model_trainer import ModelTrainer
    from sagemaker.train.configs import Compute, InputData, OutputDataConfig

    job_prefix = f"instance-prefs-integ-{uuid.uuid4().hex[:8]}"

    compute = Compute(
        instance_preferences=[shapes.InstancePreference(instance_type=t) for t in PREFERENCE_TYPES],
        instance_count=1,
        volume_size_in_gb=50,
    )
    # The Compute config must forward the preference list and leave the
    # top-level instance type unset.
    resource_config = compute._to_resource_config()
    assert [p.instance_type for p in resource_config.instance_preferences] == PREFERENCE_TYPES
    assert _get_value(resource_config.instance_type) is None

    trainer = ModelTrainer(
        base_job_name=job_prefix,
        training_image=IMAGE_URI,
        role=ROLE_ARN,
        compute=compute,
        output_data_config=OutputDataConfig(s3_output_path=S3_OUTPUT),
    )

    input_data_config = None
    if S3_INPUT:
        input_data_config = [InputData(channel_name="train", data_source=S3_INPUT)]

    trainer.train(input_data_config=input_data_config, wait=False)
    job_name = trainer._latest_training_job.training_job_name

    # --- Create accepted; Describe echoes the request contract -------------
    described = TrainingJob.get(training_job_name=job_name)
    assert described.training_job_name == job_name
    echoed = described.resource_config
    # Top-level instance type was never set; must not come back populated.
    assert _get_value(echoed.instance_type) is None
    assert [p.instance_type for p in _get_value(echoed.instance_preferences) or []] == (
        PREFERENCE_TYPES
    ), f"Describe did not echo instance_preferences (job={job_name})"

    # Wait until terminal or winner visible (Selected* propagation lags the
    # secondary-status transitions; gate on the winner, not on secondary).
    deadline = time.time() + WAIT_TIMEOUT_SECONDS
    status = described.training_job_status
    while time.time() < deadline:
        described.refresh()
        status = described.training_job_status
        if status in ("Completed", "Failed", "Stopped"):
            break
        if _get_value(described.resource_config.selected_instance_type) is not None:
            break
        time.sleep(POLL_SECONDS)

    failure_reason = _get_value(described.failure_reason) or ""

    # --- Full contract: resolved winner is surfaced and is a submitted pref -
    final_rc = described.resource_config
    selected_type = _get_value(final_rc.selected_instance_type)
    selected_count = _get_value(final_rc.selected_instance_count)
    assert selected_type in PREFERENCE_TYPES, (
        f"selected_instance_type={selected_type!r} not among submitted "
        f"preferences {PREFERENCE_TYPES} (job={job_name}, status={status}, "
        f"failure={failure_reason!r})"
    )
    assert selected_count == 1
    assert _get_value(final_rc.instance_type) is None
