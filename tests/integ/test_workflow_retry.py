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
import re
import time

import pytest

from botocore.exceptions import WaiterError
from sagemaker.processing import ProcessingInput
from sagemaker.session import get_execution_role
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.dataset_definition.inputs import (
    DatasetDefinition,
    AthenaDatasetDefinition,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.retry import (
    StepRetryPolicy,
    StepExceptionTypeEnum,
    SageMakerJobStepRetryPolicy,
    SageMakerJobExceptionTypeEnum,
)
from sagemaker.inputs import TrainingInput, CreateModelInput
from sagemaker.workflow.steps import (
    CreateModelStep,
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.model import Model
from sagemaker.workflow.step_collections import RegisterModel
from tests.integ import DATA_DIR


@pytest.fixture(scope="module")
def region_name(sagemaker_session):
    return sagemaker_session.boto_session.region_name


@pytest.fixture(scope="module")
def role(sagemaker_session):
    return get_execution_role(sagemaker_session)


@pytest.fixture(scope="module")
def script_dir():
    return os.path.join(DATA_DIR, "sklearn_processing")


@pytest.fixture
def pipeline_name():
    return f"my-pipeline-{int(time.time() * 10**7)}"


@pytest.fixture
def smclient(sagemaker_session):
    return sagemaker_session.boto_session.client("sagemaker")


@pytest.fixture
def athena_dataset_definition(sagemaker_session):
    return DatasetDefinition(
        local_path="/opt/ml/processing/input/add",
        data_distribution_type="FullyReplicated",
        input_mode="File",
        athena_dataset_definition=AthenaDatasetDefinition(
            catalog="AwsDataCatalog",
            database="default",
            work_group="workgroup",
            query_string='SELECT * FROM "default"."s3_test_table_$STAGE_$REGIONUNDERSCORED";',
            output_s3_uri=f"s3://{sagemaker_session.default_bucket()}/add",
            output_format="JSON",
            output_compression="GZIP",
        ),
    )


def test_pipeline_execution_processing_step_with_retry(
    sagemaker_session,
    smclient,
    role,
    sklearn_latest_version,
    cpu_instance_type,
    pipeline_name,
    athena_dataset_definition,
):
    instance_count = ParameterInteger(name="InstanceCount", default_value=2)
    script_path = os.path.join(DATA_DIR, "dummy_script.py")
    input_file_path = os.path.join(DATA_DIR, "dummy_input.txt")
    inputs = [
        ProcessingInput(source=input_file_path, destination="/opt/ml/processing/inputs/"),
        ProcessingInput(dataset_definition=athena_dataset_definition),
    ]

    sklearn_processor = SKLearnProcessor(
        framework_version=sklearn_latest_version,
        role=role,
        instance_type=cpu_instance_type,
        instance_count=instance_count,
        command=["python3"],
        sagemaker_session=sagemaker_session,
        base_job_name="test-sklearn",
    )

    step_sklearn = ProcessingStep(
        name="sklearn-process",
        processor=sklearn_processor,
        inputs=inputs,
        code=script_path,
        retry_policies=[
            StepRetryPolicy(
                exception_types=[
                    StepExceptionTypeEnum.SERVICE_FAULT,
                    StepExceptionTypeEnum.THROTTLING,
                ],
                backoff_rate=2.0,
                interval_seconds=30,
                expire_after_mins=5,
            ),
            SageMakerJobStepRetryPolicy(
                exception_types=[SageMakerJobExceptionTypeEnum.CAPACITY_ERROR],
                max_attempts=10,
            ),
        ],
    )
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count],
        steps=[step_sklearn],
        sagemaker_session=sagemaker_session,
    )

    try:
        pipeline.create(role)
        execution = pipeline.start(parameters={})

        try:
            execution.wait(delay=30, max_attempts=3)
        except WaiterError:
            pass
        execution_steps = execution.list_steps()
        assert len(execution_steps) == 1
        assert execution_steps[0]["StepName"] == "sklearn-process"
        # assert execution_steps[0]["AttemptCount"] >= 1
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_model_registration_with_model_repack(
    sagemaker_session,
    role,
    pipeline_name,
    region_name,
):
    base_dir = os.path.join(DATA_DIR, "pytorch_mnist")
    entry_point = os.path.join(base_dir, "mnist.py")
    input_path = sagemaker_session.upload_data(
        path=os.path.join(base_dir, "training"),
        key_prefix="integ-test-data/pytorch_mnist/training",
    )
    inputs = TrainingInput(s3_data=input_path)

    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")
    good_enough_input = ParameterInteger(name="GoodEnoughInput", default_value=1)

    pytorch_estimator = PyTorch(
        entry_point=entry_point,
        role=role,
        framework_version="1.5.0",
        py_version="py3",
        instance_count=instance_count,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
    )
    step_train = TrainingStep(
        name="pytorch-train",
        estimator=pytorch_estimator,
        inputs=inputs,
        retry_policies=[
            StepRetryPolicy(exception_types=[StepExceptionTypeEnum.THROTTLING], max_attempts=3)
        ],
    )

    step_register = RegisterModel(
        name="pytorch-register-model",
        estimator=pytorch_estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        description="test-description",
        entry_point=entry_point,
        register_model_step_retry_policies=[
            StepRetryPolicy(exception_types=[StepExceptionTypeEnum.THROTTLING], max_attempts=3)
        ],
        repack_model_step_retry_policies=[
            StepRetryPolicy(exception_types=[StepExceptionTypeEnum.THROTTLING], max_attempts=3)
        ],
    )

    model = Model(
        image_uri=pytorch_estimator.training_image_uri(),
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=sagemaker_session,
        role=role,
    )
    model_inputs = CreateModelInput(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium",
    )
    step_model = CreateModelStep(
        name="pytorch-model",
        model=model,
        inputs=model_inputs,
    )

    step_cond = ConditionStep(
        name="cond-good-enough",
        conditions=[ConditionGreaterThanOrEqualTo(left=good_enough_input, right=1)],
        if_steps=[step_train, step_register],
        else_steps=[step_model],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[good_enough_input, instance_count, instance_type],
        steps=[step_cond],
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

        execution = pipeline.start(parameters={})
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )

        execution = pipeline.start(parameters={"GoodEnoughInput": 0})
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass
