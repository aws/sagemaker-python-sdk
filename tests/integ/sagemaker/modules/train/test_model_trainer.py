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
from sagemaker.modules.distributed import MPI, Torchrun, DistributedConfig

EXPECTED_HYPERPARAMETERS = {
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

PARAM_SCRIPT_SOURCE_DIR = f"{DATA_DIR}/modules/params_script"
PARAM_SCRIPT_SOURCE_CODE = SourceCode(
    source_dir=PARAM_SCRIPT_SOURCE_DIR,
    requirements="requirements.txt",
    entry_script="train.py",
)

DEFAULT_CPU_IMAGE = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.0-cpu-py310"

TAR_FILE_SOURCE_DIR = f"{DATA_DIR}/modules/script_mode/code.tar.gz"
TAR_FILE_SOURCE_CODE = SourceCode(
    source_dir=TAR_FILE_SOURCE_DIR,
    requirements="requirements.txt",
    entry_script="custom_script.py",
)


def test_source_dir_local_tar_file(modules_sagemaker_session):
    model_trainer = ModelTrainer(
        sagemaker_session=modules_sagemaker_session,
        training_image=DEFAULT_CPU_IMAGE,
        source_code=TAR_FILE_SOURCE_CODE,
        base_job_name="source_dir_local_tar_file",
    )

    model_trainer.train()


def test_hp_contract_basic_py_script(modules_sagemaker_session):
    model_trainer = ModelTrainer(
        sagemaker_session=modules_sagemaker_session,
        training_image=DEFAULT_CPU_IMAGE,
        hyperparameters=EXPECTED_HYPERPARAMETERS,
        source_code=PARAM_SCRIPT_SOURCE_CODE,
        base_job_name="hp-contract-basic-py-script",
    )

    model_trainer.train()


def test_hp_contract_basic_sh_script(modules_sagemaker_session):
    source_code = SourceCode(
        source_dir=f"{DATA_DIR}/modules/params_script",
        requirements="requirements.txt",
        entry_script="train.sh",
    )
    model_trainer = ModelTrainer(
        sagemaker_session=modules_sagemaker_session,
        training_image=DEFAULT_CPU_IMAGE,
        hyperparameters=EXPECTED_HYPERPARAMETERS,
        source_code=source_code,
        base_job_name="hp-contract-basic-sh-script",
    )

    model_trainer.train()


def test_hp_contract_mpi_script(modules_sagemaker_session):
    compute = Compute(instance_type="ml.m5.xlarge", instance_count=2)
    model_trainer = ModelTrainer(
        sagemaker_session=modules_sagemaker_session,
        training_image=DEFAULT_CPU_IMAGE,
        compute=compute,
        hyperparameters=EXPECTED_HYPERPARAMETERS,
        source_code=PARAM_SCRIPT_SOURCE_CODE,
        distributed=MPI(),
        base_job_name="hp-contract-mpi-script",
    )

    model_trainer.train()


def test_hp_contract_torchrun_script(modules_sagemaker_session):
    compute = Compute(instance_type="ml.m5.xlarge", instance_count=2)
    model_trainer = ModelTrainer(
        sagemaker_session=modules_sagemaker_session,
        training_image=DEFAULT_CPU_IMAGE,
        compute=compute,
        hyperparameters=EXPECTED_HYPERPARAMETERS,
        source_code=PARAM_SCRIPT_SOURCE_CODE,
        distributed=Torchrun(),
        base_job_name="hp-contract-torchrun-script",
    )

    model_trainer.train()


def test_hp_contract_hyperparameter_json(modules_sagemaker_session):
    model_trainer = ModelTrainer(
        sagemaker_session=modules_sagemaker_session,
        training_image=DEFAULT_CPU_IMAGE,
        hyperparameters=f"{PARAM_SCRIPT_SOURCE_DIR}/hyperparameters.json",
        source_code=PARAM_SCRIPT_SOURCE_CODE,
        base_job_name="hp-contract-hyperparameter-json",
    )
    assert model_trainer.hyperparameters == EXPECTED_HYPERPARAMETERS
    model_trainer.train()


def test_hp_contract_hyperparameter_yaml(modules_sagemaker_session):
    model_trainer = ModelTrainer(
        sagemaker_session=modules_sagemaker_session,
        training_image=DEFAULT_CPU_IMAGE,
        hyperparameters=f"{PARAM_SCRIPT_SOURCE_DIR}/hyperparameters.yaml",
        source_code=PARAM_SCRIPT_SOURCE_CODE,
        base_job_name="hp-contract-hyperparameter-yaml",
    )
    assert model_trainer.hyperparameters == EXPECTED_HYPERPARAMETERS
    model_trainer.train()


def test_custom_distributed_driver(modules_sagemaker_session):
    class CustomDriver(DistributedConfig):
        process_count_per_node: int = None

        @property
        def driver_dir(self) -> str:
            return f"{DATA_DIR}/modules/custom_drivers"

        @property
        def driver_script(self) -> str:
            return "driver.py"

    source_code = SourceCode(
        source_dir=f"{DATA_DIR}/modules/scripts",
        entry_script="entry_script.py",
    )

    hyperparameters = {"epochs": 10}

    custom_driver = CustomDriver(process_count_per_node=2)

    model_trainer = ModelTrainer(
        sagemaker_session=modules_sagemaker_session,
        training_image=DEFAULT_CPU_IMAGE,
        hyperparameters=hyperparameters,
        source_code=source_code,
        distributed=custom_driver,
        base_job_name="custom-distributed-driver",
    )
    model_trainer.train()
