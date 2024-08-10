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

import os
import pytest

from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.utils import unique_name_from_base
from sagemaker.xgboost import XGBoost, XGBoostModel
from sagemaker.xgboost.processing import XGBoostProcessor
from tests.integ import DATA_DIR, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name

ROLE = "SageMakerRole"


@pytest.fixture(scope="module")
def xgboost_training_job(
    sagemaker_session,
    xgboost_latest_version,
    xgboost_latest_py_version,
    cpu_instance_type,
):
    return _run_mnist_training_job(
        sagemaker_session,
        cpu_instance_type,
        xgboost_latest_version,
        xgboost_latest_py_version,
    )


def test_sourcedir_naming(
    sagemaker_session,
    xgboost_latest_version,
    xgboost_latest_py_version,
    cpu_instance_type,
):
    with pytest.raises(RuntimeError):
        processor = XGBoostProcessor(
            framework_version=xgboost_latest_version,
            role=ROLE,
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
        )
        processor.run(
            source_dir="s3://bucket/deps.tar.gz",
            code="main_script.py",
        )


@pytest.mark.release
def test_framework_processing_job_with_deps(
    sagemaker_session,
    xgboost_latest_version,
    xgboost_latest_py_version,
    cpu_instance_type,
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        code_path = os.path.join(DATA_DIR, "dummy_code_bundle_with_reqs")
        entry_point = "main_script.py"

        processor = XGBoostProcessor(
            framework_version=xgboost_latest_version,
            py_version=xgboost_latest_py_version,
            role=ROLE,
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            base_job_name="test-xgboost",
        )

        processor.run(
            code=entry_point,
            source_dir=code_path,
            inputs=[],
            wait=True,
        )


def test_training_with_network_isolation(
    sagemaker_session,
    xgboost_latest_version,
    xgboost_latest_py_version,
    cpu_instance_type,
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        base_job_name = "test-network-isolation-xgboost"

        xgboost = XGBoost(
            entry_point=os.path.join(DATA_DIR, "xgboost_abalone", "abalone.py"),
            role=ROLE,
            instance_type=cpu_instance_type,
            instance_count=1,
            framework_version=xgboost_latest_version,
            py_version=xgboost_latest_py_version,
            base_job_name=base_job_name,
            sagemaker_session=sagemaker_session,
            enable_network_isolation=True,
        )

        train_input = xgboost.sagemaker_session.upload_data(
            path=os.path.join(DATA_DIR, "xgboost_abalone", "abalone"),
            key_prefix="integ-test-data/xgboost_abalone/abalone",
        )
        job_name = unique_name_from_base(base_job_name)
        xgboost.fit(inputs={"train": train_input}, job_name=job_name)
        assert sagemaker_session.sagemaker_client.describe_training_job(TrainingJobName=job_name)[
            "EnableNetworkIsolation"
        ]


def test_xgboost_serverless_inference(
    xgboost_training_job,
    sagemaker_session,
):
    endpoint_name = unique_name_from_base("test-xgboost-deploy-model-serverless")
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        desc = sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=xgboost_training_job
        )
        model_data = desc["ModelArtifacts"]["S3ModelArtifacts"]

        xgboost = XGBoostModel(
            sagemaker_session=sagemaker_session,
            model_data=model_data,
            role=ROLE,
            entry_point=os.path.join(DATA_DIR, "xgboost_abalone", "abalone.py"),
            framework_version="1.5-1",
        )

        xgboost.deploy(
            serverless_inference_config=ServerlessInferenceConfig(), endpoint_name=endpoint_name
        )


def _run_mnist_training_job(
    sagemaker_session, cpu_instance_type, xgboost_latest_version, xgboost_latest_py_version
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        base_job_name = "test-xgboost-mnist"

        xgboost = XGBoost(
            entry_point=os.path.join(DATA_DIR, "xgboost_abalone", "abalone.py"),
            role=ROLE,
            instance_type=cpu_instance_type,
            instance_count=1,
            framework_version=xgboost_latest_version,
            py_version=xgboost_latest_py_version,
            base_job_name=base_job_name,
            sagemaker_session=sagemaker_session,
            enable_network_isolation=True,
        )

        train_input = xgboost.sagemaker_session.upload_data(
            path=os.path.join(DATA_DIR, "xgboost_abalone", "abalone"),
            key_prefix="integ-test-data/xgboost_abalone/abalone",
        )
        job_name = unique_name_from_base(base_job_name)
        xgboost.fit(inputs={"train": train_input}, job_name=job_name)

        return xgboost.latest_training_job.name
