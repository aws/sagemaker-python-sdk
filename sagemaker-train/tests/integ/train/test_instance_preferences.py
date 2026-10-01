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
"""End-to-end training test for ``Compute.instance_preferences``.

Submits a real training job with an ordered list of CPU instance types and
no top-level ``instance_type``, then follows it until the service has picked
a winner and the job has run to completion on it. The shallow suite covers
acceptance and rejection; this test covers the part only the service can
prove: which type was selected and that training actually ran on it.
"""

from __future__ import absolute_import

import os
import time

from sagemaker.core.shapes import InstancePreference, StoppingCondition
from sagemaker.train.configs import Compute, SourceCode
from sagemaker.train.model_trainer import ModelTrainer

DATA_DIR = os.path.join(os.path.dirname(__file__), "../..", "data")
PARAM_SCRIPT_SOURCE_CODE = SourceCode(
    source_dir=f"{DATA_DIR}/params_script",
    requirements="requirements.txt",
    entry_script="train.py",
)
HYPERPARAMETERS = {
    "integer": 1,
    "boolean": True,
    "float": 3.14,
    "string": "Hello World",
    "list": [1, 2, 3],
    "dict": {
        "string": "value",
        "integer": 3,
        "float": 3.14,
        "list": [1, 2, 3],
        "dict": {"key": "value"},
        "boolean": True,
    },
}
DEFAULT_CPU_IMAGE = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.0-cpu-py310"

# CPU-only candidates. INSTANCE_PREFERENCES_TEST_INSTANCE_TYPES (comma-separated)
# overrides them for accounts where these lack quota.
PREFERENCE_TYPES = [
    t.strip()
    for t in os.environ.get(
        "INSTANCE_PREFERENCES_TEST_INSTANCE_TYPES", "ml.m5.large,ml.m5.xlarge"
    ).split(",")
    if t.strip()
]
TERMINAL = ("Completed", "Failed", "Stopped")
MAX_RUNTIME_SECONDS = 1800
WAIT_TIMEOUT_SECONDS = 40 * 60
POLL_SECONDS = 30


def _wait(client, job_name, until):
    deadline = time.time() + WAIT_TIMEOUT_SECONDS
    while True:
        described = client.describe_training_job(TrainingJobName=job_name)
        if until(described) or time.time() >= deadline:
            return described
        time.sleep(POLL_SECONDS)


def _stop_quietly(client, job_name):
    try:
        if client.describe_training_job(TrainingJobName=job_name)["TrainingJobStatus"] in TERMINAL:
            return
        client.stop_training_job(TrainingJobName=job_name)
    except Exception:  # pylint: disable=broad-except
        pass


def test_instance_preferences_select_a_winner_and_complete(sagemaker_session):
    trainer = ModelTrainer(
        sagemaker_session=sagemaker_session,
        training_image=DEFAULT_CPU_IMAGE,
        hyperparameters=HYPERPARAMETERS,
        source_code=PARAM_SCRIPT_SOURCE_CODE,
        compute=Compute(
            instance_preferences=[InstancePreference(instance_type=t) for t in PREFERENCE_TYPES],
            instance_count=1,
        ),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=MAX_RUNTIME_SECONDS),
        base_job_name="instance-prefs-e2e",
    )
    client = sagemaker_session.sagemaker_client

    trainer.train(wait=False, logs=False)
    job_name = trainer._latest_training_job.training_job_name

    try:
        described = client.describe_training_job(TrainingJobName=job_name)
        resource_config = described["ResourceConfig"]
        assert [p["InstanceType"] for p in resource_config["InstancePreferences"]] == (
            PREFERENCE_TYPES
        ), resource_config
        assert "InstanceType" not in resource_config, resource_config

        described = _wait(
            client,
            job_name,
            lambda d: d["TrainingJobStatus"] in TERMINAL
            or d["ResourceConfig"].get("SelectedInstanceType"),
        )
        resource_config = described["ResourceConfig"]
        selected_type = resource_config.get("SelectedInstanceType")
        assert selected_type in PREFERENCE_TYPES, (
            f"SelectedInstanceType={selected_type!r} not among {PREFERENCE_TYPES} "
            f"(job={job_name}, status={described['TrainingJobStatus']}, "
            f"failure={described.get('FailureReason', '')!r})"
        )
        assert resource_config.get("SelectedInstanceCount") == 1
        assert "InstanceType" not in resource_config, resource_config

        described = _wait(client, job_name, lambda d: d["TrainingJobStatus"] in TERMINAL)
        assert described["TrainingJobStatus"] == "Completed", (
            f"job={job_name} ended {described['TrainingJobStatus']} on {selected_type}: "
            f"{described.get('FailureReason', '')!r}"
        )
        # The winner must not change once training has run on it.
        assert described["ResourceConfig"]["SelectedInstanceType"] == selected_type
    finally:
        _stop_quietly(client, job_name)
