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
"""This module contains code to test image builder"""
from __future__ import absolute_import

from tests.integ import DATA_DIR

from sagemaker.modules.train import ModelTrainer
from sagemaker.modules.configs import SourceCode, Compute
from sagemaker.modules.distributed import MPI, Torchrun

EXPECTED_HYPERPARAMETERS = {
    "integer": 1,
    "boolean": True,
    "float": 3.14,
    "string": "Hello World",
    "list": [1, 2, 3],
    "dict": {
        "string": "value",
        "integer": 3,
        "list": [1, 2, 3],
        "dict": {"key": "value"},
        "boolean": True,
    },
}

DEFAULT_CPU_IMAGE = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.0-cpu-py310"


def test_hp_contract_basic_py_script(modules_sagemaker_session):
    source_code = SourceCode(
        source_dir=f"{DATA_DIR}/modules/params-script",
        entry_script="train.py",
    )

    model_trainer = ModelTrainer(
        session=modules_sagemaker_session,
        training_image=DEFAULT_CPU_IMAGE,
        hyperparameters=EXPECTED_HYPERPARAMETERS,
        source_code=source_code,
        base_job_name="hp-contract-basic-py-script",
    )

    model_trainer.train()


def test_hp_contract_basic_sh_script(modules_sagemaker_session):
    source_code = SourceCode(
        source_dir=f"{DATA_DIR}/modules/params-script",
        entry_script="train.sh",
    )
    model_trainer = ModelTrainer(
        session=modules_sagemaker_session,
        training_image=DEFAULT_CPU_IMAGE,
        hyperparameters=EXPECTED_HYPERPARAMETERS,
        source_code=source_code,
        base_job_name="hp-contract-basic-sh-script",
    )

    model_trainer.train()


def test_hp_contract_mpi_script(modules_sagemaker_session):
    source_code = SourceCode(
        source_dir=f"{DATA_DIR}/modules/params-script",
        entry_script="train.py",
    )
    compute = Compute(instance_type="ml.m5.xlarge", instance_count=2)
    model_trainer = ModelTrainer(
        session=modules_sagemaker_session,
        training_image=DEFAULT_CPU_IMAGE,
        compute=compute,
        hyperparameters=EXPECTED_HYPERPARAMETERS,
        source_code=source_code,
        distributed_runner=MPI(),
        base_job_name="hp-contract-mpi-script",
    )

    model_trainer.train()


def test_hp_contract_torchrun_script(modules_sagemaker_session):
    source_code = SourceCode(
        source_dir=f"{DATA_DIR}/modules/params-script",
        entry_script="train.py",
    )
    compute = Compute(instance_type="ml.m5.xlarge", instance_count=2)
    model_trainer = ModelTrainer(
        session=modules_sagemaker_session,
        training_image=DEFAULT_CPU_IMAGE,
        compute=compute,
        hyperparameters=EXPECTED_HYPERPARAMETERS,
        source_code=source_code,
        distributed_runner=Torchrun(),
        base_job_name="hp-contract-torchrun-script",
    )

    model_trainer.train()
