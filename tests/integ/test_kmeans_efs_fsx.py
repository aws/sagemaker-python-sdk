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
from __future__ import absolute_import

import pytest

from sagemaker import KMeans
from sagemaker.amazon.amazon_estimator import FileSystemRecordSet
from sagemaker.parameter import IntegerParameter, CategoricalParameter
from sagemaker.tuner import HyperparameterTuner
from sagemaker.utils import unique_name_from_base
import tests
from tests.integ import TRAINING_DEFAULT_TIMEOUT_MINUTES, TUNING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.file_system_input_utils import set_up_efs_fsx, tear_down
from tests.integ.s3_utils import assert_s3_files_exist
from tests.integ.timeout import timeout

INSTANCE_COUNT = 1
OBJECTIVE_METRIC_NAME = "test:msd"
EFS_DIR_PATH = "/one_p_mnist"
FSX_DIR_PATH = "/fsx/one_p_mnist"
MAX_JOBS = 2
MAX_PARALLEL_JOBS = 2
K = 10
NUM_RECORDS = 784
FEATURE_DIM = 784


@pytest.fixture(scope="module")
def efs_fsx_setup(sagemaker_session, ec2_instance_type):
    fs_resources = None
    try:
        fs_resources = set_up_efs_fsx(sagemaker_session, ec2_instance_type)
        yield fs_resources
    finally:
        if fs_resources:
            tear_down(sagemaker_session, fs_resources)


@pytest.mark.skipif(
    tests.integ.test_region() not in tests.integ.EFS_TEST_ENABLED_REGION,
    reason="EFS integration tests need to be fixed before running in all regions.",
)
def test_kmeans_efs(efs_fsx_setup, sagemaker_session, cpu_instance_type):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        role = efs_fsx_setup["role_name"]
        subnets = [efs_fsx_setup["subnet_id"]]
        security_group_ids = efs_fsx_setup["security_group_ids"]

        kmeans = KMeans(
            role=role,
            instance_count=INSTANCE_COUNT,
            instance_type=cpu_instance_type,
            k=K,
            sagemaker_session=sagemaker_session,
            subnets=subnets,
            security_group_ids=security_group_ids,
        )

        file_system_efs_id = efs_fsx_setup["file_system_efs_id"]
        records = FileSystemRecordSet(
            file_system_id=file_system_efs_id,
            file_system_type="EFS",
            directory_path=EFS_DIR_PATH,
            num_records=NUM_RECORDS,
            feature_dim=FEATURE_DIM,
        )

        job_name = unique_name_from_base("kmeans-efs")
        kmeans.fit(records, job_name=job_name)
        model_path, _ = kmeans.model_data.rsplit("/", 1)
        assert_s3_files_exist(sagemaker_session, model_path, ["model.tar.gz"])


@pytest.mark.skipif(
    tests.integ.test_region() not in tests.integ.EFS_TEST_ENABLED_REGION,
    reason="EFS integration tests need to be fixed before running in all regions.",
)
def test_kmeans_fsx(efs_fsx_setup, sagemaker_session, cpu_instance_type):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        role = efs_fsx_setup["role_name"]
        subnets = [efs_fsx_setup["subnet_id"]]
        security_group_ids = efs_fsx_setup["security_group_ids"]
        kmeans = KMeans(
            role=role,
            instance_count=INSTANCE_COUNT,
            instance_type=cpu_instance_type,
            k=K,
            sagemaker_session=sagemaker_session,
            subnets=subnets,
            security_group_ids=security_group_ids,
        )

        file_system_fsx_id = efs_fsx_setup["file_system_fsx_id"]
        records = FileSystemRecordSet(
            file_system_id=file_system_fsx_id,
            file_system_type="FSxLustre",
            directory_path=FSX_DIR_PATH,
            num_records=NUM_RECORDS,
            feature_dim=FEATURE_DIM,
        )

        job_name = unique_name_from_base("kmeans-fsx")
        kmeans.fit(records, job_name=job_name)
        model_path, _ = kmeans.model_data.rsplit("/", 1)
        assert_s3_files_exist(sagemaker_session, model_path, ["model.tar.gz"])


@pytest.mark.skipif(
    tests.integ.test_region() not in tests.integ.EFS_TEST_ENABLED_REGION,
    reason="EFS integration tests need to be fixed before running in all regions.",
)
def test_tuning_kmeans_efs(efs_fsx_setup, sagemaker_session, cpu_instance_type):
    role = efs_fsx_setup["role_name"]
    subnets = [efs_fsx_setup["subnet_id"]]
    security_group_ids = efs_fsx_setup["security_group_ids"]
    kmeans = KMeans(
        role=role,
        instance_count=INSTANCE_COUNT,
        instance_type=cpu_instance_type,
        k=K,
        sagemaker_session=sagemaker_session,
        subnets=subnets,
        security_group_ids=security_group_ids,
    )

    hyperparameter_ranges = {
        "extra_center_factor": IntegerParameter(4, 10),
        "mini_batch_size": IntegerParameter(10, 100),
        "epochs": IntegerParameter(1, 2),
        "init_method": CategoricalParameter(["kmeans++", "random"]),
    }

    with timeout(minutes=TUNING_DEFAULT_TIMEOUT_MINUTES):
        tuner = HyperparameterTuner(
            estimator=kmeans,
            objective_metric_name=OBJECTIVE_METRIC_NAME,
            hyperparameter_ranges=hyperparameter_ranges,
            objective_type="Minimize",
            max_jobs=MAX_JOBS,
            max_parallel_jobs=MAX_PARALLEL_JOBS,
        )

        file_system_efs_id = efs_fsx_setup["file_system_efs_id"]
        train_records = FileSystemRecordSet(
            file_system_id=file_system_efs_id,
            file_system_type="EFS",
            directory_path=EFS_DIR_PATH,
            num_records=NUM_RECORDS,
            feature_dim=FEATURE_DIM,
        )

        test_records = FileSystemRecordSet(
            file_system_id=file_system_efs_id,
            file_system_type="EFS",
            directory_path=EFS_DIR_PATH,
            num_records=NUM_RECORDS,
            feature_dim=FEATURE_DIM,
            channel="test",
        )

        job_name = unique_name_from_base("tune-kmeans-efs")
        tuner.fit([train_records, test_records], job_name=job_name)
        tuner.wait()
        best_training_job = tuner.best_training_job()
        assert best_training_job


@pytest.mark.skipif(
    tests.integ.test_region() not in tests.integ.EFS_TEST_ENABLED_REGION,
    reason="EFS integration tests need to be fixed before running in all regions.",
)
def test_tuning_kmeans_fsx(efs_fsx_setup, sagemaker_session, cpu_instance_type):
    role = efs_fsx_setup["role_name"]
    subnets = [efs_fsx_setup["subnet_id"]]
    security_group_ids = efs_fsx_setup["security_group_ids"]
    kmeans = KMeans(
        role=role,
        instance_count=INSTANCE_COUNT,
        instance_type=cpu_instance_type,
        k=K,
        sagemaker_session=sagemaker_session,
        subnets=subnets,
        security_group_ids=security_group_ids,
    )

    hyperparameter_ranges = {
        "extra_center_factor": IntegerParameter(4, 10),
        "mini_batch_size": IntegerParameter(10, 100),
        "epochs": IntegerParameter(1, 2),
        "init_method": CategoricalParameter(["kmeans++", "random"]),
    }

    with timeout(minutes=TUNING_DEFAULT_TIMEOUT_MINUTES):
        tuner = HyperparameterTuner(
            estimator=kmeans,
            objective_metric_name=OBJECTIVE_METRIC_NAME,
            hyperparameter_ranges=hyperparameter_ranges,
            objective_type="Minimize",
            max_jobs=MAX_JOBS,
            max_parallel_jobs=MAX_PARALLEL_JOBS,
        )

        file_system_fsx_id = efs_fsx_setup["file_system_fsx_id"]
        train_records = FileSystemRecordSet(
            file_system_id=file_system_fsx_id,
            file_system_type="FSxLustre",
            directory_path=FSX_DIR_PATH,
            num_records=NUM_RECORDS,
            feature_dim=FEATURE_DIM,
        )

        test_records = FileSystemRecordSet(
            file_system_id=file_system_fsx_id,
            file_system_type="FSxLustre",
            directory_path=FSX_DIR_PATH,
            num_records=NUM_RECORDS,
            feature_dim=FEATURE_DIM,
            channel="test",
        )

        job_name = unique_name_from_base("tune-kmeans-fsx")
        tuner.fit([train_records, test_records], job_name=job_name)
        tuner.wait()
        best_training_job = tuner.best_training_job()
        assert best_training_job
