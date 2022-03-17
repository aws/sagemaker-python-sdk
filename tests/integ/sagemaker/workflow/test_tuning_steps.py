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

import pytest

from sagemaker import TrainingInput, Model, get_execution_role, utils
from sagemaker.dataset_definition import DatasetDefinition, AthenaDatasetDefinition
from sagemaker.inputs import CreateModelInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch import PyTorch
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.tuner import HyperparameterTuner, IntegerParameter
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline

from sagemaker.workflow.steps import (
    CreateModelStep,
    ProcessingStep,
    TuningStep,
    PropertyFile,
)
from tests.integ import DATA_DIR


@pytest.fixture
def role(sagemaker_session):
    return get_execution_role(sagemaker_session)


@pytest.fixture
def pipeline_name():
    return utils.unique_name_from_base("my-pipeline-training")


@pytest.fixture
def region_name(sagemaker_session):
    return sagemaker_session.boto_session.region_name


@pytest.fixture
def script_dir():
    return os.path.join(DATA_DIR, "sklearn_processing")


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
            query_string=('SELECT * FROM "default"."s3_test_table_$STAGE_$REGIONUNDERSCORED";'),
            output_s3_uri=f"s3://{sagemaker_session.default_bucket()}/add",
            output_format="JSON",
            output_compression="GZIP",
        ),
    )


def test_tuning_single_algo(
    sagemaker_session,
    role,
    cpu_instance_type,
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

    pytorch_estimator = PyTorch(
        entry_point=entry_point,
        role=role,
        framework_version="1.5.0",
        py_version="py3",
        instance_count=instance_count,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        enable_sagemaker_metrics=True,
        max_retry_attempts=3,
    )

    min_batch_size = ParameterString(name="MinBatchSize", default_value="64")
    max_batch_size = ParameterString(name="MaxBatchSize", default_value="128")
    hyperparameter_ranges = {
        "batch-size": IntegerParameter(min_batch_size, max_batch_size),
    }

    tuner = HyperparameterTuner(
        estimator=pytorch_estimator,
        objective_metric_name="test:acc",
        objective_type="Maximize",
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=[{"Name": "test:acc", "Regex": "Overall test accuracy: (.*?);"}],
        max_jobs=2,
        max_parallel_jobs=2,
    )

    step_tune = TuningStep(
        name="my-tuning-step",
        tuner=tuner,
        inputs=inputs,
    )

    best_model = Model(
        image_uri=pytorch_estimator.training_image_uri(),
        model_data=step_tune.get_top_model_s3_uri(
            top_k=0,
            s3_bucket=sagemaker_session.default_bucket(),
        ),
        sagemaker_session=sagemaker_session,
        role=role,
    )
    model_inputs = CreateModelInput(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium",
    )
    step_best_model = CreateModelStep(
        name="1st-model",
        model=best_model,
        inputs=model_inputs,
    )

    second_best_model = Model(
        image_uri=pytorch_estimator.training_image_uri(),
        model_data=step_tune.get_top_model_s3_uri(
            top_k=1,
            s3_bucket=sagemaker_session.default_bucket(),
        ),
        sagemaker_session=sagemaker_session,
        role=role,
    )

    step_second_best_model = CreateModelStep(
        name="2nd-best-model",
        model=second_best_model,
        inputs=model_inputs,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count, instance_type, min_batch_size, max_batch_size],
        steps=[step_tune, step_best_model, step_second_best_model],
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
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_tuning_multi_algos(
    sagemaker_session,
    role,
    cpu_instance_type,
    pipeline_name,
    region_name,
    script_dir,
    athena_dataset_definition,
):
    base_dir = os.path.join(DATA_DIR, "pytorch_mnist")
    entry_point = os.path.join(base_dir, "mnist.py")
    input_path = sagemaker_session.upload_data(
        path=os.path.join(base_dir, "training"),
        key_prefix="integ-test-data/pytorch_mnist/training",
    )

    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")

    input_data = f"s3://sagemaker-sample-data-{region_name}/processing/census/census-income.csv"

    sklearn_processor = SKLearnProcessor(
        framework_version="0.20.0",
        instance_type=instance_type,
        instance_count=instance_count,
        base_job_name="test-sklearn",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    property_file = PropertyFile(
        name="DataAttributes", output_name="attributes", path="attributes.json"
    )

    step_process = ProcessingStep(
        name="my-process",
        display_name="ProcessingStep",
        description="description for Processing step",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),
            ProcessingInput(dataset_definition=athena_dataset_definition),
        ],
        outputs=[
            ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="attributes", source="/opt/ml/processing/attributes.json"),
        ],
        property_files=[property_file],
        code=os.path.join(script_dir, "preprocessing.py"),
    )

    static_hp_1 = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")
    json_get_hp = JsonGet(
        step_name=step_process.name, property_file=property_file, json_path="train_size"
    )
    pytorch_estimator = PyTorch(
        entry_point=entry_point,
        role=role,
        framework_version="1.5.0",
        py_version="py3",
        instance_count=instance_count,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        enable_sagemaker_metrics=True,
        max_retry_attempts=3,
        hyperparameters={"static-hp": static_hp_1, "train_size": json_get_hp},
    )

    min_batch_size = ParameterString(name="MinBatchSize", default_value="64")
    max_batch_size = json_get_hp

    tuner = HyperparameterTuner.create(
        estimator_dict={
            "estimator-1": pytorch_estimator,
            "estimator-2": pytorch_estimator,
        },
        objective_metric_name_dict={
            "estimator-1": "test:acc",
            "estimator-2": "test:acc",
        },
        hyperparameter_ranges_dict={
            "estimator-1": {"batch-size": IntegerParameter(min_batch_size, max_batch_size)},
            "estimator-2": {"batch-size": IntegerParameter(min_batch_size, max_batch_size)},
        },
        metric_definitions_dict={
            "estimator-1": [{"Name": "test:acc", "Regex": "Overall test accuracy: (.*?);"}],
            "estimator-2": [{"Name": "test:acc", "Regex": "Overall test accuracy: (.*?);"}],
        },
    )

    inputs = {
        "estimator-1": TrainingInput(s3_data=input_path),
        "estimator-2": TrainingInput(s3_data=input_path),
    }

    step_tune = TuningStep(
        name="my-tuning-step",
        tuner=tuner,
        inputs=inputs,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count, instance_type, min_batch_size, max_batch_size],
        steps=[step_process, step_tune],
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
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass
