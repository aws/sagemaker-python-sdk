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

import boto3
import botocore
import pytest

from sagemaker.modules.train import ModelTrainer
from sagemaker.modules.configs import SourceCode, InputData, Compute

from sagemaker.aws_batch.training_queue import TrainingQueue

from tests.integ import DATA_DIR
from tests.integ.sagemaker.modules.conftest import modules_sagemaker_session  # noqa: F401
from tests.integ.sagemaker.modules.train.test_model_trainer import (
    DEFAULT_CPU_IMAGE,
)
from tests.integ.sagemaker.aws_batch.manager import BatchTestResourceManager


@pytest.fixture(scope="module")
def batch_client():
    return boto3.client("batch", region_name="us-west-2")


@pytest.fixture(scope="function")
def batch_test_resource_manager(batch_client):
    resource_manager = BatchTestResourceManager(batch_client=batch_client)
    resource_manager.get_or_create_resources()
    return resource_manager


def test_model_trainer_submit(batch_test_resource_manager, modules_sagemaker_session):  # noqa: F811
    queue_name = batch_test_resource_manager.queue_name

    source_code = SourceCode(
        source_dir=f"{DATA_DIR}/modules/script_mode/",
        requirements="requirements.txt",
        entry_script="custom_script.py",
    )
    hyperparameters = {
        "batch-size": 32,
        "epochs": 1,
        "learning-rate": 0.01,
    }
    compute = Compute(instance_type="ml.m5.2xlarge")
    model_trainer = ModelTrainer(
        sagemaker_session=modules_sagemaker_session,
        training_image=DEFAULT_CPU_IMAGE,
        source_code=source_code,
        compute=compute,
        hyperparameters=hyperparameters,
        base_job_name="test-batch-model-trainer",
    )
    train_data = InputData(
        channel_name="train",
        data_source=f"{DATA_DIR}/modules/script_mode/data/train/",
    )
    test_data = InputData(
        channel_name="test",
        data_source=f"{DATA_DIR}/modules/script_mode/data/test/",
    )

    training_queue = TrainingQueue(queue_name=queue_name)

    try:
        queued_job = training_queue.submit(
            training_job=model_trainer,
            inputs=[train_data, test_data],
        )
    except botocore.exceptions.ClientError as e:
        print(e.response["ResponseMetadata"])
        print(e.response["Error"]["Message"])
        raise e
    res = queued_job.describe()
    assert res is not None
    assert res["status"] == "SUBMITTED"

    queued_job.wait(timeout=1800)
    res = queued_job.describe()
    assert res is not None
    assert res["status"] == "SUCCEEDED"
