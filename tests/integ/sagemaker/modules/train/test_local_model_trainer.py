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
# language governing peCWDissions and limitations under the License.
"""This module contains code to test image builder with local mode"""
from __future__ import absolute_import
import os
import errno

import shutil
import tempfile

from tests.integ import DATA_DIR
import tests.integ.lock as lock

from sagemaker.modules.configs import Compute, InputData, SourceCode
from sagemaker.modules.distributed import Torchrun
from sagemaker.modules.train.model_trainer import Mode, ModelTrainer
import subprocess

DEFAULT_CPU_IMAGE = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.0-cpu-py310"
CWD = os.getcwd()
SOURCE_DIR = os.path.join(DATA_DIR, "modules/local_script")
LOCK_PATH = os.path.join(tempfile.gettempdir(), "sagemaker_test_local_mode_lock")


def delete_local_path(path):
    try:
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Removed directory: {path}")
        else:
            print(f"Directory does not exist: {path}")
    except OSError as exc:
        # on Linux, when docker writes to any mounted volume, it uses the container's user. In most
        # cases this is root. When the container exits and we try to delete them we can't because
        # root owns those files. We expect this to happen, so we handle EACCESS. Any other error
        # we will raise the exception up.
        if exc.errno == errno.EACCES:
            print(f"Failed to delete: {path} Please remove it manually.")
        else:
            print(f"Failed to delete: {path}")
            raise


def test_single_container_local_mode_local_data(modules_sagemaker_session):
    with lock.lock(LOCK_PATH):
        try:
            source_code = SourceCode(
                source_dir=SOURCE_DIR,
                entry_script="local_training_script.py",
            )

            compute = Compute(
                instance_type="local_cpu",
                instance_count=1,
            )

            train_data = InputData(
                channel_name="train",
                data_source=os.path.join(SOURCE_DIR, "data/train/"),
            )

            test_data = InputData(
                channel_name="test",
                data_source=os.path.join(SOURCE_DIR, "data/test/"),
            )

            model_trainer = ModelTrainer(
                training_image=DEFAULT_CPU_IMAGE,
                sagemaker_session=modules_sagemaker_session,
                source_code=source_code,
                compute=compute,
                input_data_config=[train_data, test_data],
                base_job_name="local_mode_single_container_local_data",
                training_mode=Mode.LOCAL_CONTAINER,
            )

            model_trainer.train()
            assert os.path.exists(os.path.join(CWD, "compressed_artifacts/model.tar.gz"))
        finally:
            subprocess.run(["docker", "compose", "down", "-v"])
            directories = [
                "compressed_artifacts",
                "artifacts",
                "model",
                "shared",
                "input",
                "output",
                "algo-1",
            ]

            for directory in directories:
                path = os.path.join(CWD, directory)
                delete_local_path(path)


def test_single_container_local_mode_s3_data(modules_sagemaker_session):
    with lock.lock(LOCK_PATH):
        try:
            # upload local data to s3
            session = modules_sagemaker_session
            bucket = session.default_bucket()
            session.upload_data(
                path=os.path.join(SOURCE_DIR, "data/train/"),
                bucket=bucket,
                key_prefix="data/train",
            )
            session.upload_data(
                path=os.path.join(SOURCE_DIR, "data/test/"),
                bucket=bucket,
                key_prefix="data/test",
            )

            source_code = SourceCode(
                source_dir=SOURCE_DIR,
                entry_script="local_training_script.py",
            )

            compute = Compute(
                instance_type="local_cpu",
                instance_count=1,
            )

            # read input data from s3
            train_data = InputData(channel_name="train", data_source=f"s3://{bucket}/data/train/")

            test_data = InputData(channel_name="test", data_source=f"s3://{bucket}/data/test/")

            model_trainer = ModelTrainer(
                training_image=DEFAULT_CPU_IMAGE,
                sagemaker_session=modules_sagemaker_session,
                source_code=source_code,
                compute=compute,
                input_data_config=[train_data, test_data],
                base_job_name="local_mode_single_container_s3_data",
                training_mode=Mode.LOCAL_CONTAINER,
            )

            model_trainer.train()
            assert os.path.exists(os.path.join(CWD, "compressed_artifacts/model.tar.gz"))
        finally:
            subprocess.run(["docker", "compose", "down", "-v"])
            directories = [
                "compressed_artifacts",
                "artifacts",
                "model",
                "shared",
                "input",
                "output",
                "algo-1",
            ]

            for directory in directories:
                path = os.path.join(CWD, directory)
                delete_local_path(path)


def test_multi_container_local_mode(modules_sagemaker_session):
    with lock.lock(LOCK_PATH):
        try:
            source_code = SourceCode(
                source_dir=SOURCE_DIR,
                entry_script="local_training_script.py",
            )

            distributed = Torchrun(
                process_count_per_node=1,
            )

            compute = Compute(
                instance_type="local_cpu",
                instance_count=2,
            )

            train_data = InputData(
                channel_name="train",
                data_source=os.path.join(SOURCE_DIR, "data/train/"),
            )

            test_data = InputData(
                channel_name="test",
                data_source=os.path.join(SOURCE_DIR, "data/test/"),
            )

            model_trainer = ModelTrainer(
                training_image=DEFAULT_CPU_IMAGE,
                sagemaker_session=modules_sagemaker_session,
                source_code=source_code,
                distributed=distributed,
                compute=compute,
                input_data_config=[train_data, test_data],
                base_job_name="local_mode_multi_container",
                training_mode=Mode.LOCAL_CONTAINER,
            )

            model_trainer.train()
            assert os.path.exists(os.path.join(CWD, "compressed_artifacts/model.tar.gz"))
            assert os.path.exists(os.path.join(CWD, "algo-1"))
            assert os.path.exists(os.path.join(CWD, "algo-2"))

        finally:
            subprocess.run(["docker", "compose", "down", "-v"])
            directories = [
                "compressed_artifacts",
                "artifacts",
                "model",
                "shared",
                "input",
                "output",
                "algo-1",
                "algo-2",
            ]

            for directory in directories:
                path = os.path.join(CWD, directory)
                delete_local_path(path)
