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

from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.configs import SourceCode, InputData, Compute

from sagemaker.train.aws_batch.training_queue import TrainingQueue

from tests.integ import DATA_DIR
from tests.integ.train.conftest import sagemaker_session  # noqa: F401
from tests.integ.train.test_model_trainer import (
    DEFAULT_CPU_IMAGE,
)
from .manager import BatchTestResourceManager


@pytest.fixture(scope="module")
def batch_client():
    return boto3.client("batch", region_name="us-west-2")


@pytest.fixture(scope="function")
def batch_test_resource_manager(batch_client):
    resource_manager = BatchTestResourceManager(batch_client=batch_client)
    resource_manager.get_or_create_resources()
    yield resource_manager
    resource_manager.delete_resources()


def test_model_trainer_submit(batch_test_resource_manager, sagemaker_session):  # noqa: F811
    queue_name = batch_test_resource_manager.queue_name

    source_code = SourceCode(command="echo 'Hello World'")
    compute = Compute(instance_type="ml.m5.2xlarge")
    model_trainer = ModelTrainer(
        sagemaker_session=sagemaker_session,
        training_image=DEFAULT_CPU_IMAGE,
        source_code=source_code,
        compute=compute,
        base_job_name="test-batch-model-trainer",
    )

    training_queue = TrainingQueue(queue_name=queue_name)

    try:
        queued_job = training_queue.submit(
            training_job=model_trainer,
            inputs=None,
            job_name="pysdk_integ_test_job",
            retry_config={
                "attempts": 1,
                "evaluateOnExit": [
                    {
                        "action": "Retry",
                        "onStatusReason": "Received status from SageMaker: AlgorithmError: *"
                    },
                    {
                        "action": "EXIT",
                        "onStatusReason": "*"
                    }
                ]
            },
            priority=1,
            tags={"pysdk-integ-test-tag-key": "pysdk-integ-test-tag-value"},
            quota_share_name=batch_test_resource_manager.quota_share_name,
            preemption_config={"preemptionRetriesBeforeTermination": 0}
        )
    except botocore.exceptions.ClientError as e:
        print(e.response["ResponseMetadata"])
        print(e.response["Error"]["Message"])
        raise e

    res = queued_job.describe()
    assert res is not None
    assert res["status"] in {"SUBMITTED", "RUNNABLE", "SCHEDULED"}

    res = queued_job.update(2)
    assert res is not None
    assert res["jobArn"] == queued_job.job_arn

    # Job termination results in FAILED
    queued_job.terminate()

    res = queued_job.wait(timeout=900)
    assert res is not None
    assert res["status"] == "FAILED"

    list_by_job_name = training_queue.list_jobs(queued_job.job_name)
    list_by_job_status = training_queue.list_jobs(status="FAILED")
    assert queued_job.job_arn in [job.job_arn for job in list_by_job_name]
    assert queued_job.job_name in [job.job_name for job in list_by_job_status]
