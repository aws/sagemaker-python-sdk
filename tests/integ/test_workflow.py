# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import json
import os
import re
import time
import uuid

import boto3
import pytest

from botocore.config import Config
from botocore.exceptions import WaiterError
from sagemaker.debugger import (
    DebuggerHookConfig,
    Rule,
    rule_configs,
)
from sagemaker.inputs import CreateModelInput, TrainingInput
from sagemaker.model import Model
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.session import get_execution_role, Session
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.dataset_definition.inputs import DatasetDefinition, AthenaDatasetDefinition
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.steps import (
    CreateModelStep,
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline import Pipeline
from tests.integ import DATA_DIR


def ordered(obj):
    """Helper function for dict comparison"""
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(ordered(x) for x in obj)
    else:
        return obj


@pytest.fixture(scope="module")
def region_name(sagemaker_session):
    return sagemaker_session.boto_session.region_name


@pytest.fixture(scope="module")
def role(sagemaker_session):
    return get_execution_role(sagemaker_session)


# TODO-reinvent-2020: remove use of specific region and this session
@pytest.fixture(scope="module")
def region():
    return "us-east-2"


# TODO-reinvent-2020: remove use of specific region and this session
@pytest.fixture(scope="module")
def workflow_session(region):
    boto_session = boto3.Session(region_name=region)

    sagemaker_client_config = dict()
    sagemaker_client_config.setdefault("config", Config(retries=dict(max_attempts=2)))
    sagemaker_client = boto_session.client("sagemaker", **sagemaker_client_config)

    return Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=None,
    )


@pytest.fixture(scope="module")
def script_dir():
    return os.path.join(DATA_DIR, "sklearn_processing")


@pytest.fixture
def pipeline_name():
    return f"my-pipeline-{int(time.time() * 10**7)}"


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


def test_three_step_definition(
    sagemaker_session,
    workflow_session,
    region_name,
    role,
    script_dir,
    pipeline_name,
    athena_dataset_definition,
):
    framework_version = "0.20.0"
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)

    input_data = f"s3://sagemaker-sample-data-{region_name}/processing/census/census-income.csv"

    sklearn_processor = SKLearnProcessor(
        framework_version=framework_version,
        instance_type=instance_type,
        instance_count=instance_count,
        base_job_name="test-sklearn",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    step_process = ProcessingStep(
        name="my-process",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),
            ProcessingInput(dataset_definition=athena_dataset_definition),
        ],
        outputs=[
            ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="test_data", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(script_dir, "preprocessing.py"),
    )

    sklearn_train = SKLearn(
        framework_version=framework_version,
        entry_point=os.path.join(script_dir, "train.py"),
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        role=role,
    )
    step_train = TrainingStep(
        name="my-train",
        estimator=sklearn_train,
        inputs=TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                "train_data"
            ].S3Output.S3Uri
        ),
    )

    model = Model(
        image_uri=sklearn_train.image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=sagemaker_session,
        role=role,
    )
    model_inputs = CreateModelInput(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium",
    )
    step_model = CreateModelStep(
        name="my-model",
        model=model,
        inputs=model_inputs,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_type, instance_count],
        steps=[step_process, step_train, step_model],
        sagemaker_session=workflow_session,
    )

    definition = json.loads(pipeline.definition())
    assert definition["Version"] == "2020-12-01"

    assert set(tuple(param.items()) for param in definition["Parameters"]) == set(
        [
            tuple(
                {"Name": "InstanceType", "Type": "String", "DefaultValue": "ml.m5.xlarge"}.items()
            ),
            tuple({"Name": "InstanceCount", "Type": "Integer", "DefaultValue": 1}.items()),
        ]
    )

    steps = definition["Steps"]
    assert len(steps) == 3

    names_and_types = []
    processing_args = {}
    training_args = {}
    for step in steps:
        names_and_types.append((step["Name"], step["Type"]))
        if step["Type"] == "Processing":
            processing_args = step["Arguments"]
        if step["Type"] == "Training":
            training_args = step["Arguments"]
        if step["Type"] == "Model":
            model_args = step["Arguments"]

    assert set(names_and_types) == set(
        [
            ("my-process", "Processing"),
            ("my-train", "Training"),
            ("my-model", "Model"),
        ]
    )

    assert processing_args["ProcessingResources"]["ClusterConfig"] == {
        "InstanceType": {"Get": "Parameters.InstanceType"},
        "InstanceCount": {"Get": "Parameters.InstanceCount"},
        "VolumeSizeInGB": 30,
    }

    assert training_args["ResourceConfig"] == {
        "InstanceCount": 1,
        "InstanceType": {"Get": "Parameters.InstanceType"},
        "VolumeSizeInGB": 30,
    }
    assert training_args["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"] == {
        "Get": "Steps.my-process.ProcessingOutputConfig.Outputs['train_data'].S3Output.S3Uri"
    }
    assert model_args["PrimaryContainer"]["ModelDataUrl"] == {
        "Get": "Steps.my-train.ModelArtifacts.S3ModelArtifacts"
    }


# TODO-reinvent-2020: Modify use of the workflow client
def test_one_step_sklearn_processing_pipeline(
    sagemaker_session,
    workflow_session,
    role,
    sklearn_latest_version,
    cpu_instance_type,
    pipeline_name,
    region,
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
    )
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count],
        steps=[step_sklearn],
        sagemaker_session=workflow_session,
    )

    try:
        # NOTE: We should exercise the case when role used in the pipeline execution is
        # different than that required of the steps in the pipeline itself. The role in
        # the pipeline definition needs to create training and processing jobs and other
        # sagemaker entities. However, the jobs created in the steps themselves execute
        # under a potentially different role, often requiring access to S3 and other
        # artifacts not required to during creation of the jobs in the pipeline steps.
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            fr"arn:aws:sagemaker:{region}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

        pipeline.parameters = [ParameterInteger(name="InstanceCount", default_value=1)]
        response = pipeline.update(role)
        update_arn = response["PipelineArn"]
        assert re.match(
            fr"arn:aws:sagemaker:{region}:\d{{12}}:pipeline/{pipeline_name}",
            update_arn,
        )

        execution = pipeline.start(parameters={})
        assert re.match(
            fr"arn:aws:sagemaker:{region}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )

        response = execution.describe()
        assert response["PipelineArn"] == create_arn

        try:
            execution.wait(delay=30, max_attempts=3)
        except WaiterError:
            pass
        execution_steps = execution.list_steps()
        assert len(execution_steps) == 1
        assert execution_steps[0]["StepName"] == "sklearn-process"
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


# TODO-reinvent-2020: Modify use of the workflow client
def test_conditional_pytorch_training_model_registration(
    sagemaker_session,
    workflow_session,
    role,
    cpu_instance_type,
    pipeline_name,
    region,
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
    )

    step_register = RegisterModel(
        name="pytorch-register-model",
        estimator=pytorch_estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["*"],
        response_types=["*"],
        inference_instances=["*"],
        transform_instances=["*"],
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
        sagemaker_session=workflow_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            fr"arn:aws:sagemaker:{region}:\d{{12}}:pipeline/{pipeline_name}", create_arn
        )

        execution = pipeline.start(parameters={})
        assert re.match(
            fr"arn:aws:sagemaker:{region}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )

        execution = pipeline.start(parameters={"GoodEnoughInput": 0})
        assert re.match(
            fr"arn:aws:sagemaker:{region}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_training_job_with_debugger(
    sagemaker_session,
    pipeline_name,
    role,
    pytorch_training_latest_version,
    pytorch_training_latest_py_version,
):
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")

    rules = [
        Rule.sagemaker(rule_configs.vanishing_gradient()),
        Rule.sagemaker(base_config=rule_configs.all_zero(), rule_parameters={"tensor_regex": ".*"}),
        Rule.sagemaker(rule_configs.loss_not_decreasing()),
    ]
    debugger_hook_config = DebuggerHookConfig(
        s3_output_path=f"s3://{sagemaker_session.default_bucket()}/{uuid.uuid4()}/tensors"
    )

    base_dir = os.path.join(DATA_DIR, "pytorch_mnist")
    script_path = os.path.join(base_dir, "mnist.py")
    input_path = sagemaker_session.upload_data(
        path=os.path.join(base_dir, "training"),
        key_prefix="integ-test-data/pytorch_mnist/training",
    )
    inputs = TrainingInput(s3_data=input_path)

    pytorch_estimator = PyTorch(
        entry_point=script_path,
        role="SageMakerRole",
        framework_version=pytorch_training_latest_version,
        py_version=pytorch_training_latest_py_version,
        instance_count=instance_count,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        rules=rules,
        debugger_hook_config=debugger_hook_config,
    )

    step_train = TrainingStep(
        name="pytorch-train",
        estimator=pytorch_estimator,
        inputs=inputs,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count, instance_type],
        steps=[step_train],
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]

        execution = pipeline.start()
        response = execution.describe()
        assert response["PipelineArn"] == create_arn

        try:
            execution.wait(delay=10, max_attempts=60)
        except WaiterError:
            pass
        execution_steps = execution.list_steps()
        training_job_arn = execution_steps[0]["Metadata"]["TrainingJob"]["Arn"]
        job_description = sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=training_job_arn.split("/")[1]
        )

        assert len(execution_steps) == 1
        assert execution_steps[0]["StepName"] == "pytorch-train"
        assert execution_steps[0]["StepStatus"] == "Succeeded"

        for index, rule in enumerate(rules):
            config = job_description["DebugRuleConfigurations"][index]
            assert config["RuleConfigurationName"] == rule.name
            assert config["RuleEvaluatorImage"] == rule.image_uri
            assert config["VolumeSizeInGB"] == 0
            assert (
                config["RuleParameters"]["rule_to_invoke"] == rule.rule_parameters["rule_to_invoke"]
            )
        assert job_description["DebugHookConfig"] == debugger_hook_config._to_request_dict()
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass
