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

from tests.integ.sagemaker.workflow.helpers import wait_pipeline_execution
from sagemaker.processing import ProcessingInput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.dataset_definition.inputs import (
    DatasetDefinition,
    AthenaDatasetDefinition,
)
from sagemaker.workflow.model_step import (
    ModelStep,
    _REGISTER_MODEL_NAME_BASE,
    _CREATE_MODEL_NAME_BASE,
)
from sagemaker.workflow.parameters import ParameterInteger
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.retry import (
    StepRetryPolicy,
    StepExceptionTypeEnum,
    SageMakerJobStepRetryPolicy,
    SageMakerJobExceptionTypeEnum,
)
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.model import Model
from tests.integ import DATA_DIR


@pytest.fixture
def pipeline_name():
    return f"my-pipeline-{int(time.time() * 10**7)}"


@pytest.fixture
def athena_dataset_definition(sagemaker_session_for_pipeline):
    return DatasetDefinition(
        local_path="/opt/ml/processing/input/add",
        data_distribution_type="FullyReplicated",
        input_mode="File",
        athena_dataset_definition=AthenaDatasetDefinition(
            catalog="AwsDataCatalog",
            database="default",
            work_group="workgroup",
            query_string='SELECT * FROM "default"."s3_test_table_$STAGE_$REGIONUNDERSCORED";',
            output_s3_uri=f"s3://{sagemaker_session_for_pipeline.default_bucket()}/add",
            output_format="JSON",
            output_compression="GZIP",
        ),
    )


def test_pipeline_execution_processing_step_with_retry(
    sagemaker_session_for_pipeline,
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
        sagemaker_session=sagemaker_session_for_pipeline,
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
        sagemaker_session=sagemaker_session_for_pipeline,
    )

    try:
        pipeline.create(role)
        execution = pipeline.start(parameters={})

        wait_pipeline_execution(execution=execution, max_attempts=3)
        execution_steps = execution.list_steps()
        assert len(execution_steps) == 1
        assert execution_steps[0]["StepName"] == "sklearn-process"
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_model_registration_with_model_repack(
    pipeline_session,
    role,
    pipeline_name,
    region_name,
):
    base_dir = os.path.join(DATA_DIR, "pytorch_mnist")
    entry_point = os.path.join(base_dir, "mnist.py")
    input_path = pipeline_session.upload_data(
        path=os.path.join(base_dir, "training"),
        key_prefix="integ-test-data/pytorch_mnist/training",
    )
    inputs = TrainingInput(s3_data=input_path)

    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = "ml.m5.xlarge"
    good_enough_input = ParameterInteger(name="GoodEnoughInput", default_value=1)

    # If image_uri is not provided, the instance_type should not be a pipeline variable
    # since instance_type is used to retrieve image_uri in compile time (PySDK)
    pytorch_estimator = PyTorch(
        entry_point=entry_point,
        role=role,
        framework_version="1.5.0",
        py_version="py3",
        instance_count=instance_count,
        instance_type=instance_type,
        sagemaker_session=pipeline_session,
    )
    step_train = TrainingStep(
        name="pytorch-train",
        estimator=pytorch_estimator,
        inputs=inputs,
        retry_policies=[
            StepRetryPolicy(exception_types=[StepExceptionTypeEnum.THROTTLING], max_attempts=3)
        ],
    )
    model = Model(
        image_uri=pytorch_estimator.training_image_uri(),
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
        entry_point=entry_point,
        source_dir=base_dir,
    )
    # register model with repack
    regis_model_step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        description="test-description",
        model_package_group_name=f"{pipeline_name}TestModelPackageGroup",
    )
    step_model_regis = ModelStep(
        name="pytorch-register-model",
        step_args=regis_model_step_args,
        retry_policies=dict(
            register_model_retry_policies=[
                StepRetryPolicy(exception_types=[StepExceptionTypeEnum.THROTTLING], max_attempts=3)
            ],
            repack_model_retry_policies=[
                StepRetryPolicy(exception_types=[StepExceptionTypeEnum.THROTTLING], max_attempts=3)
            ],
        ),
    )
    # create model with repack
    create_model_step_args = model.create(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium",
    )
    step_model_create = ModelStep(
        name="pytorch-model",
        step_args=create_model_step_args,
    )
    step_cond = ConditionStep(
        name="cond-good-enough",
        conditions=[ConditionGreaterThanOrEqualTo(left=good_enough_input, right=1)],
        if_steps=[step_model_regis],
        else_steps=[step_model_create],
        depends_on=[step_train],
    )
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[good_enough_input, instance_count],
        steps=[step_train, step_cond],
        sagemaker_session=pipeline_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

        execution1 = pipeline.start(parameters={})
        execution2 = pipeline.start(parameters={"GoodEnoughInput": 0})

        wait_pipeline_execution(execution=execution1)
        execution1_steps = execution1.list_steps()
        for step in execution1_steps:
            assert not step.get("FailureReason", None)
            assert step["StepStatus"] == "Succeeded"
            if _REGISTER_MODEL_NAME_BASE in step["StepName"]:
                assert step["Metadata"]["RegisterModel"]
        assert len(execution1_steps) == 4

        wait_pipeline_execution(execution=execution2)
        execution2_steps = execution2.list_steps()
        for step in execution2_steps:
            assert not step.get("FailureReason", None)
            assert step["StepStatus"] == "Succeeded"
            if _CREATE_MODEL_NAME_BASE in step["StepName"]:
                assert step["Metadata"]["Model"]
        assert len(execution2_steps) == 4

    finally:
        try:
            pipeline.delete()
        except Exception:
            pass
