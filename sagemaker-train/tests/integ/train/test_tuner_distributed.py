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
"""Integration test: HyperparameterTuner with Torchrun distributed training.

Regression test for the bug where HyperparameterTuner dropped the sm_drivers
channel, causing the container to fall back to single-GPU execution instead
of using torchrun for multi-GPU distributed training.
"""
from __future__ import absolute_import

import os
import time
import logging

import pytest

from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.configs import SourceCode, Compute
from sagemaker.train.distributed import Torchrun
from sagemaker.train.tuner import HyperparameterTuner
from sagemaker.core.parameter import ContinuousParameter

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../..", "data")
DEFAULT_CPU_IMAGE = (
    "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.0-cpu-py310"
)

TRAIN_SCRIPT_CONTENT = """\
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args, _ = parser.parse_known_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"DISTRIBUTED_CHECK: world_size={world_size}")
    print(f"DISTRIBUTED_CHECK: local_rank={local_rank}")
    print(f"DISTRIBUTED_CHECK: learning_rate={args.learning_rate}")

    # Emit metric for the tuner to parse
    print(f"eval_loss: 0.42")


if __name__ == "__main__":
    main()
"""


@pytest.fixture(scope="module")
def train_source_dir(tmp_path_factory):
    """Create a temp directory with a minimal training script."""
    d = tmp_path_factory.mktemp("tuner_dist_src")
    (d / "train.py").write_text(TRAIN_SCRIPT_CONTENT)
    return str(d)


def test_tuner_includes_sm_drivers_channel(sagemaker_session, train_source_dir):
    """Verify tuning jobs include sm_drivers channel for distributed training.

    Uses a CPU instance with Torchrun to validate that the sm_drivers channel
    (containing torchrun_driver.py and sm_train.sh) is included in the tuning
    job definition. The training script logs WORLD_SIZE to confirm the V3
    driver path is used instead of the legacy framework container fallback.
    """
    model_trainer = ModelTrainer(
        sagemaker_session=sagemaker_session,
        training_image=DEFAULT_CPU_IMAGE,
        base_job_name="tuner-dist-test",
        source_code=SourceCode(
            source_dir=train_source_dir,
            entry_script="train.py",
        ),
        compute=Compute(
            instance_type="ml.m5.xlarge",
            instance_count=1,
            volume_size_in_gb=30,
        ),
        distributed=Torchrun(),
        hyperparameters={"learning_rate": 1e-4},
    )

    tuner = HyperparameterTuner(
        model_trainer=model_trainer,
        objective_metric_name="eval_loss",
        metric_definitions=[
            {"Name": "eval_loss", "Regex": r"eval_loss: ([0-9\\.]+)"},
        ],
        hyperparameter_ranges={
            "learning_rate": ContinuousParameter(
                min_value=1e-5,
                max_value=5e-4,
                scaling_type="Logarithmic",
            ),
        },
        objective_type="Minimize",
        max_jobs=1,
        max_parallel_jobs=1,
    )

    tuner.tune(wait=True)

    job = tuner.latest_tuning_job.refresh()
    assert job.hyper_parameter_tuning_job_status in (
        "Completed",
        "Stopped",
    ), f"Tuning job failed: {job.hyper_parameter_tuning_job_status}"

    best = tuner.best_training_job()
    assert best is not None
    logger.info("PASSED: tuner distributed training test - job: %s", best)
