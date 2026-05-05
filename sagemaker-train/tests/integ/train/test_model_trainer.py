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

import os
import logging
import traceback
import pytest

from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.configs import SourceCode, Compute
from sagemaker.train.distributed import MPI, Torchrun, DistributedConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../..", "data")
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

PARAM_SCRIPT_SOURCE_DIR = f"{DATA_DIR}/params_script"
PARAM_SCRIPT_SOURCE_CODE = SourceCode(
    source_dir=PARAM_SCRIPT_SOURCE_DIR,
    requirements="requirements.txt",
    entry_script="train.py",
)

DEFAULT_CPU_IMAGE = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.0-cpu-py310"


TAR_FILE_SOURCE_DIR = f"{DATA_DIR}/script_mode/code.tar.gz"
TAR_FILE_SOURCE_CODE = SourceCode(
    source_dir=TAR_FILE_SOURCE_DIR,
    requirements="requirements.txt",
    entry_script="custom_script.py",
)


def test_source_dir_local_tar_file(sagemaker_session):
    model_trainer = ModelTrainer(
        sagemaker_session=sagemaker_session,
        training_image=DEFAULT_CPU_IMAGE,
        source_code=TAR_FILE_SOURCE_CODE,
        base_job_name="source_dir_local_tar_file",
    )

    model_trainer.train()


def test_hp_contract_basic_py_script(sagemaker_session):
    model_trainer = ModelTrainer(
        sagemaker_session=sagemaker_session,
        training_image=DEFAULT_CPU_IMAGE,
        hyperparameters=EXPECTED_HYPERPARAMETERS,
        source_code=PARAM_SCRIPT_SOURCE_CODE,
        base_job_name="hp-contract-basic-py-script",
    )

    model_trainer.train()


def test_hp_contract_basic_sh_script(sagemaker_session):
    source_code = SourceCode(
        source_dir=f"{DATA_DIR}/params_script",
        requirements="requirements.txt",
        entry_script="train.sh",
    )
    model_trainer = ModelTrainer(
        sagemaker_session=sagemaker_session,
        training_image=DEFAULT_CPU_IMAGE,
        hyperparameters=EXPECTED_HYPERPARAMETERS,
        source_code=source_code,
        base_job_name="hp-contract-basic-sh-script",
    )

    model_trainer.train()


# skip this test for now as requirments.txt is not resolved
# @pytest.mark.skip
def test_hp_contract_mpi_script(sagemaker_session):
    logger.info("=== START test_hp_contract_mpi_script ===")
    logger.info(f"sagemaker_session region: {sagemaker_session.boto_region_name}")
    logger.info(f"PARAM_SCRIPT_SOURCE_DIR: {PARAM_SCRIPT_SOURCE_DIR}")
    logger.info(f"Source dir exists: {os.path.exists(PARAM_SCRIPT_SOURCE_DIR)}")

    requirements_path = os.path.join(PARAM_SCRIPT_SOURCE_DIR, "requirements.txt")
    logger.info(f"requirements.txt path: {requirements_path}")
    logger.info(f"requirements.txt exists: {os.path.exists(requirements_path)}")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r") as f:
            logger.info(f"requirements.txt contents:\n{f.read()}")

    # Log all files in source dir
    if os.path.exists(PARAM_SCRIPT_SOURCE_DIR):
        logger.info(f"Files in source dir: {os.listdir(PARAM_SCRIPT_SOURCE_DIR)}")

    try:
        compute = Compute(instance_type="ml.m5.xlarge", instance_count=2)
        logger.info(f"Compute config: instance_type=ml.m5.xlarge, instance_count=2")
        logger.info(f"Source code: source_dir={PARAM_SCRIPT_SOURCE_CODE.source_dir}, requirements={PARAM_SCRIPT_SOURCE_CODE.requirements}, entry_script={PARAM_SCRIPT_SOURCE_CODE.entry_script}")

        model_trainer = ModelTrainer(
            sagemaker_session=sagemaker_session,
            training_image=DEFAULT_CPU_IMAGE,
            compute=compute,
            hyperparameters=EXPECTED_HYPERPARAMETERS,
            source_code=PARAM_SCRIPT_SOURCE_CODE,
            distributed=MPI(),
            base_job_name="hp-contract-mpi-script",
        )
        logger.info(f"ModelTrainer created successfully")
        logger.info(f"ModelTrainer training_image: {model_trainer.training_image}")

        logger.info("Calling model_trainer.train()...")
        model_trainer.train()
        logger.info("model_trainer.train() completed successfully")
    except Exception as e:
        logger.error(f"test_hp_contract_mpi_script FAILED: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        raise
    logger.info("=== END test_hp_contract_mpi_script - PASSED ===")


def test_hp_contract_torchrun_script(sagemaker_session):
    compute = Compute(instance_type="ml.m5.xlarge", instance_count=2)
    model_trainer = ModelTrainer(
        sagemaker_session=sagemaker_session,
        training_image=DEFAULT_CPU_IMAGE,
        compute=compute,
        hyperparameters=EXPECTED_HYPERPARAMETERS,
        source_code=PARAM_SCRIPT_SOURCE_CODE,
        distributed=Torchrun(),
        base_job_name="hp-contract-torchrun-script",
    )

    model_trainer.train()


def test_hp_contract_hyperparameter_json(sagemaker_session):
    model_trainer = ModelTrainer(
        sagemaker_session=sagemaker_session,
        training_image=DEFAULT_CPU_IMAGE,
        hyperparameters=f"{PARAM_SCRIPT_SOURCE_DIR}/hyperparameters.json",
        source_code=PARAM_SCRIPT_SOURCE_CODE,
        base_job_name="hp-contract-hyperparameter-json",
    )
    assert model_trainer.hyperparameters == EXPECTED_HYPERPARAMETERS
    model_trainer.train()


def test_hp_contract_hyperparameter_yaml(sagemaker_session):
    model_trainer = ModelTrainer(
        sagemaker_session=sagemaker_session,
        training_image=DEFAULT_CPU_IMAGE,
        hyperparameters=f"{PARAM_SCRIPT_SOURCE_DIR}/hyperparameters.yaml",
        source_code=PARAM_SCRIPT_SOURCE_CODE,
        base_job_name="hp-contract-hyperparameter-yaml",
    )
    assert model_trainer.hyperparameters == EXPECTED_HYPERPARAMETERS
    model_trainer.train()


def test_custom_distributed_driver(sagemaker_session):
    class CustomDriver(DistributedConfig):
        process_count_per_node: int = None

        @property
        def driver_dir(self) -> str:
            return f"{DATA_DIR}/custom_drivers"

        @property
        def driver_script(self) -> str:
            return "driver.py"

    source_code = SourceCode(
        source_dir=f"{DATA_DIR}/scripts",
        entry_script="entry_script.py",
    )

    hyperparameters = {"epochs": 10}

    custom_driver = CustomDriver(process_count_per_node=2)

    model_trainer = ModelTrainer(
        sagemaker_session=sagemaker_session,
        training_image=DEFAULT_CPU_IMAGE,
        hyperparameters=hyperparameters,
        source_code=source_code,
        distributed=custom_driver,
        base_job_name="custom-distributed-driver",
    )
    model_trainer.train()
