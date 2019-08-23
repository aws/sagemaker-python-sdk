# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from __future__ import absolute_import

import os
import time

import pytest

from sagemaker.inputs import FileSystemInput
from sagemaker.parameter import IntegerParameter
from sagemaker.tensorflow import TensorFlow
from sagemaker.tuner import HyperparameterTuner
from sagemaker.utils import unique_name_from_base
from tests.integ import TRAINING_DEFAULT_TIMEOUT_MINUTES, TUNING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.s3_utils import assert_s3_files_exist
from tests.integ.file_system_input_utils import tear_down, set_up_efs_fsx
from tests.integ.timeout import timeout

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
MNIST_RESOURCE_PATH = os.path.join(RESOURCE_PATH, "tensorflow_mnist")
SCRIPT = os.path.join(MNIST_RESOURCE_PATH, "mnist.py")
TFS_RESOURCE_PATH = os.path.join(RESOURCE_PATH, "tfs", "tfs-test-entrypoint-with-handler")
INSTANCE_TYPE = "ml.c4.xlarge"
EFS_DIR_PATH = "/tensorflow"
FSX_DIR_PATH = "/fsx/tensorflow"
MAX_JOBS = 2
MAX_PARALLEL_JOBS = 2
PY_VERSION = "py3"


@pytest.fixture(scope="module")
def efs_fsx_setup(sagemaker_session):
    fs_resources = set_up_efs_fsx(sagemaker_session)
    try:
        yield fs_resources
    finally:
        tear_down(sagemaker_session, fs_resources)


@pytest.mark.canary_quick
def test_mnist_efs(efs_fsx_setup, sagemaker_session):
    role = efs_fsx_setup.role_name
    subnets = [efs_fsx_setup.subnet_id]
    security_group_ids = efs_fsx_setup.security_group_ids

    estimator = TensorFlow(
        entry_point=SCRIPT,
        role=role,
        train_instance_count=1,
        train_instance_type=INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
        script_mode=True,
        framework_version=TensorFlow.LATEST_VERSION,
        py_version=PY_VERSION,
        subnets=subnets,
        security_group_ids=security_group_ids,
    )

    file_system_efs_id = efs_fsx_setup.file_system_efs_id
    file_system_input = FileSystemInput(
        file_system_id=file_system_efs_id, file_system_type="EFS", directory_path=EFS_DIR_PATH
    )
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        estimator.fit(inputs=file_system_input, job_name=unique_name_from_base("test-mnist-efs"))

    assert_s3_files_exist(
        sagemaker_session,
        estimator.model_dir,
        ["graph.pbtxt", "model.ckpt-0.index", "model.ckpt-0.meta"],
    )


@pytest.mark.canary_quick
def test_mnist_lustre(efs_fsx_setup, sagemaker_session):
    role = efs_fsx_setup.role_name
    subnets = [efs_fsx_setup.subnet_id]
    security_group_ids = efs_fsx_setup.security_group_ids

    estimator = TensorFlow(
        entry_point=SCRIPT,
        role=role,
        train_instance_count=1,
        train_instance_type=INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
        script_mode=True,
        framework_version=TensorFlow.LATEST_VERSION,
        py_version=PY_VERSION,
        subnets=subnets,
        security_group_ids=security_group_ids,
    )

    file_system_fsx_id = efs_fsx_setup.file_system_fsx_id
    file_system_input = FileSystemInput(
        file_system_id=file_system_fsx_id, file_system_type="FSxLustre", directory_path=FSX_DIR_PATH
    )

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        estimator.fit(inputs=file_system_input, job_name=unique_name_from_base("test-mnist-lustre"))
    assert_s3_files_exist(
        sagemaker_session,
        estimator.model_dir,
        ["graph.pbtxt", "model.ckpt-0.index", "model.ckpt-0.meta"],
    )


def test_tuning_tf_script_mode_efs(efs_fsx_setup, sagemaker_session):
    role = efs_fsx_setup.role_name
    subnets = [efs_fsx_setup.subnet_id]
    security_group_ids = efs_fsx_setup.security_group_ids

    estimator = TensorFlow(
        entry_point=SCRIPT,
        role=role,
        train_instance_count=1,
        train_instance_type=INSTANCE_TYPE,
        script_mode=True,
        sagemaker_session=sagemaker_session,
        py_version=PY_VERSION,
        framework_version=TensorFlow.LATEST_VERSION,
        subnets=subnets,
        security_group_ids=security_group_ids,
    )

    hyperparameter_ranges = {"epochs": IntegerParameter(1, 2)}
    objective_metric_name = "accuracy"
    metric_definitions = [{"Name": objective_metric_name, "Regex": "accuracy = ([0-9\\.]+)"}]
    tuner = HyperparameterTuner(
        estimator,
        objective_metric_name,
        hyperparameter_ranges,
        metric_definitions,
        max_jobs=MAX_JOBS,
        max_parallel_jobs=MAX_PARALLEL_JOBS,
    )

    file_system_efs_id = efs_fsx_setup.file_system_efs_id
    file_system_input = FileSystemInput(
        file_system_id=file_system_efs_id, file_system_type="EFS", directory_path=EFS_DIR_PATH
    )

    with timeout(minutes=TUNING_DEFAULT_TIMEOUT_MINUTES):
        tuning_job_name = unique_name_from_base("test-tuning-tf-script-mode-efs", max_length=32)
        tuner.fit(file_system_input, job_name=tuning_job_name)
        time.sleep(15)
        tuner.wait()
    best_training_job = tuner.best_training_job()
    assert best_training_job


def test_tuning_tf_script_mode_lustre(efs_fsx_setup, sagemaker_session):
    role = efs_fsx_setup.role_name
    subnets = [efs_fsx_setup.subnet_id]
    security_group_ids = efs_fsx_setup.security_group_ids

    estimator = TensorFlow(
        entry_point=SCRIPT,
        role=role,
        train_instance_count=1,
        train_instance_type=INSTANCE_TYPE,
        script_mode=True,
        sagemaker_session=sagemaker_session,
        py_version=PY_VERSION,
        framework_version=TensorFlow.LATEST_VERSION,
        subnets=subnets,
        security_group_ids=security_group_ids,
    )

    hyperparameter_ranges = {"epochs": IntegerParameter(1, 2)}
    objective_metric_name = "accuracy"
    metric_definitions = [{"Name": objective_metric_name, "Regex": "accuracy = ([0-9\\.]+)"}]
    tuner = HyperparameterTuner(
        estimator,
        objective_metric_name,
        hyperparameter_ranges,
        metric_definitions,
        max_jobs=MAX_JOBS,
        max_parallel_jobs=MAX_PARALLEL_JOBS,
    )

    file_system_fsx_id = efs_fsx_setup.file_system_fsx_id
    file_system_input = FileSystemInput(
        file_system_id=file_system_fsx_id, file_system_type="FSxLustre", directory_path=FSX_DIR_PATH
    )

    with timeout(minutes=TUNING_DEFAULT_TIMEOUT_MINUTES):
        tuning_job_name = unique_name_from_base("test-tuning-tf-script-mode-lustre", max_length=32)
        tuner.fit(file_system_input, job_name=tuning_job_name)
        time.sleep(15)
        tuner.wait()
    best_training_job = tuner.best_training_job()
    assert best_training_job
