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
"""Integration test: HyperparameterTuner with Torchrun distributed training."""

import os
import time
import logging
import pytest

from sagemaker.train import ModelTrainer
from sagemaker.train.model_trainer import Mode
from sagemaker.train.configs import SourceCode, Compute
from sagemaker.train.distributed import Torchrun
from sagemaker.train.tuner import HyperparameterTuner
from sagemaker.core.parameter import ContinuousParameter
from sagemaker.core.shapes import StoppingCondition

logger = logging.getLogger(__name__)

TRAINING_IMAGE = (
    "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
    "huggingface-pytorch-training:2.8.0-transformers4.56.2"
    "-gpu-py312-cu129-ubuntu22.04-v1.0"
)
ROLE = os.environ.get(
    "SAGEMAKER_ROLE_ARN",
    "arn:aws:iam::123456789012:role/SageMakerRole",
)
INSTANCE_TYPE = "ml.g5.12xlarge"


TRAIN_SCRIPT = '''\
import os
import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args, _ = parser.parse_known_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"DISTRIBUTED_CHECK: world_size={world_size}, local_rank={local_rank}")
    print(f"DISTRIBUTED_CHECK: gpu_count={torch.cuda.device_count()}")

    if world_size > 1:
        print("DISTRIBUTED_CHECK: MULTI_GPU_MODE=true")
        torch.distributed.init_process_group(backend="nccl")
        torch.distributed.destroy_process_group()
    else:
        print("DISTRIBUTED_CHECK: MULTI_GPU_MODE=false")

    # Emit metric for the tuner to parse
    print("eval_loss: 0.5")


if __name__ == "__main__":
    main()
'''


@pytest.fixture(scope="module")
def training_script_dir(tmp_path_factory):
    """Create a temp directory with a minimal training script."""
    d = tmp_path_factory.mktemp("src")
    (d / "train.py").write_text(TRAIN_SCRIPT)
    return str(d)


@pytest.mark.slow
class TestTunerDistributedTraining:
    """Regression test: tuner must preserve Torchrun distributed config."""

    def test_tuner_launches_torchrun(self, training_script_dir):
        """Tuning job should use torchrun (multi-GPU), not single-GPU fallback."""
        model_trainer = ModelTrainer(
            training_mode=Mode.SAGEMAKER_TRAINING_JOB,
            role=ROLE,
            training_image=TRAINING_IMAGE,
            base_job_name=f"tuner-dist-{int(time.time())}",
            source_code=SourceCode(
                source_dir=training_script_dir,
                entry_script="train.py",
            ),
            compute=Compute(
                instance_type=INSTANCE_TYPE,
                instance_count=1,
                volume_size_in_gb=30,
            ),
            distributed=Torchrun(),
            stopping_condition=StoppingCondition(
                max_runtime_in_seconds=600,
            ),
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
