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

import json
import os
import re
import tempfile
import time
import shutil

from contextlib import contextmanager
import pytest

import pandas as pd

from sagemaker.utils import retry_with_backoff
from tests.integ.sagemaker.workflow.helpers import wait_pipeline_execution
from tests.integ.s3_utils import extract_files_from_s3
from sagemaker.workflow.model_step import (
    ModelStep,
    _REGISTER_MODEL_NAME_BASE,
    _REPACK_MODEL_NAME_BASE,
)
from sagemaker.parameter import IntegerParameter
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker.tuner import HyperparameterTuner
from tests.integ.timeout import timeout

from sagemaker.session import Session
from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.inputs import CreateModelInput, TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    FeatureStoreOutput,
    ScriptProcessor,
)
from sagemaker.s3 import S3Uploader
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.transformer import Transformer
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import (
    ConditionGreaterThanOrEqualTo,
    ConditionLessThanOrEqualTo,
)
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.callback_step import (
    CallbackStep,
    CallbackOutput,
    CallbackOutputTypeEnum,
)
from sagemaker.wrangler.processing import DataWranglerProcessor
from sagemaker.dataset_definition.inputs import (
    DatasetDefinition,
    AthenaDatasetDefinition,
)
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.wrangler.ingestion import generate_data_ingestion_flow_from_s3_input
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.steps import (
    CreateModelStep,
    ProcessingStep,
    TrainingStep,
    TransformStep,
    TransformInput,
    PropertyFile,
    TuningStep,
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.utilities import hash_files_or_dirs
from sagemaker.feature_store.feature_group import (
    FeatureGroup,
    FeatureDefinition,
    FeatureTypeEnum,
)
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
def feature_store_session(sagemaker_session_for_pipeline):
    boto_session = sagemaker_session_for_pipeline.boto_session
    sagemaker_client = boto_session.client("sagemaker")
    featurestore_runtime_client = boto_session.client("sagemaker-featurestore-runtime")

    return Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_featurestore_runtime_client=featurestore_runtime_client,
    )


@pytest.fixture
def pipeline_name():
    return f"my-pipeline-{int(time.time() * 10 ** 7)}"


@pytest.fixture(scope="module")
def athena_dataset_definition(sagemaker_session_for_pipeline):
    return DatasetDefinition(
        local_path="/opt/ml/processing/input/add",
        data_distribution_type="FullyReplicated",
        input_mode="File",
        athena_dataset_definition=AthenaDatasetDefinition(
            catalog="AwsDataCatalog",
            database="default",
            work_group="workgroup",
            query_string=('SELECT * FROM "default"."s3_test_table_$STAGE_$REGIONUNDERSCORED";'),
            output_s3_uri=f"s3://{sagemaker_session_for_pipeline.default_bucket()}/add",
            output_format="JSON",
            output_compression="GZIP",
        ),
    )


def test_three_step_definition(
    pipeline_session,
    region_name,
    role,
    script_dir,
    pipeline_name,
    athena_dataset_definition,
):
    framework_version = "0.20.0"
    instance_type = "ml.m5.xlarge"
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    output_prefix = ParameterString(name="OutputPrefix", default_value="output")

    input_data = f"s3://sagemaker-sample-data-{region_name}/processing/census/census-income.csv"

    # The instance_type should not be a pipeline variable
    # since it is used to retrieve image_uri in compile time (PySDK)
    sklearn_processor = SKLearnProcessor(
        framework_version=framework_version,
        instance_type=instance_type,
        instance_count=instance_count,
        base_job_name="test-sklearn",
        sagemaker_session=pipeline_session,
        role=role,
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
            ProcessingOutput(
                output_name="test_data",
                source="/opt/ml/processing/test",
                destination=Join(
                    on="/",
                    values=[
                        "s3:/",
                        pipeline_session.default_bucket(),
                        "test-sklearn",
                        output_prefix,
                        ExecutionVariables.PIPELINE_EXECUTION_ID,
                    ],
                ),
            ),
        ],
        code=os.path.join(script_dir, "preprocessing.py"),
    )

    # If image_uri is not provided, the instance_type should not be a pipeline variable
    # since instance_type is used to retrieve image_uri in compile time (PySDK)
    sklearn_train = SKLearn(
        framework_version=framework_version,
        entry_point=os.path.join(script_dir, "train.py"),
        instance_type=instance_type,
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_train = TrainingStep(
        name="my-train",
        display_name="TrainingStep",
        description="description for Training step",
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
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_model_args = model.create(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium",
    )
    step_model = ModelStep(
        name="my-model",
        display_name="ModelStep",
        description="description for Model step",
        step_args=step_model_args,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count, output_prefix],
        steps=[step_process, step_train, step_model],
        sagemaker_session=pipeline_session,
    )

    definition = json.loads(pipeline.definition())
    assert definition["Version"] == "2020-12-01"

    assert set(tuple(param.items()) for param in definition["Parameters"]) == set(
        [
            tuple({"Name": "InstanceCount", "Type": "Integer", "DefaultValue": 1}.items()),
            tuple(
                {
                    "Name": "OutputPrefix",
                    "Type": "String",
                    "DefaultValue": "output",
                }.items()
            ),
        ]
    )

    steps = definition["Steps"]
    assert len(steps) == 3

    names_and_types = []
    display_names_and_desc = []
    processing_args = {}
    training_args = {}
    for step in steps:
        names_and_types.append((step["Name"], step["Type"]))
        display_names_and_desc.append((step["DisplayName"], step["Description"]))
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
            ("my-model-CreateModel", "Model"),
        ]
    )

    assert set(display_names_and_desc) == set(
        [
            ("ProcessingStep", "description for Processing step"),
            ("TrainingStep", "description for Training step"),
            ("ModelStep", "description for Model step"),
        ]
    )
    assert processing_args["ProcessingResources"]["ClusterConfig"] == {
        "InstanceType": "ml.m5.xlarge",
        "InstanceCount": {"Get": "Parameters.InstanceCount"},
        "VolumeSizeInGB": 30,
    }

    assert training_args["ResourceConfig"] == {
        "InstanceCount": 1,
        "InstanceType": "ml.m5.xlarge",
        "VolumeSizeInGB": 30,
    }
    assert training_args["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"] == {
        "Get": "Steps.my-process.ProcessingOutputConfig.Outputs['train_data'].S3Output.S3Uri"
    }
    assert model_args["PrimaryContainer"]["ModelDataUrl"] == {
        "Get": "Steps.my-train.ModelArtifacts.S3ModelArtifacts"
    }
    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )
        assert pipeline.latest_pipeline_version_id == 1
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_steps_with_map_params_pipeline(
    pipeline_session,
    role,
    script_dir,
    pipeline_name,
    region_name,
    athena_dataset_definition,
):
    instance_count = ParameterInteger(name="InstanceCount", default_value=2)
    framework_version = "0.20.0"
    instance_type = "ml.m5.xlarge"
    output_prefix = ParameterString(name="OutputPrefix", default_value="output")
    input_data = f"s3://sagemaker-sample-data-{region_name}/processing/census/census-income.csv"

    # The instance_type should not be a pipeline variable
    # since it is used to retrieve image_uri in compile time (PySDK)
    sklearn_processor = SKLearnProcessor(
        framework_version=framework_version,
        instance_type=instance_type,
        instance_count=instance_count,
        base_job_name="test-sklearn",
        sagemaker_session=pipeline_session,
        role=role,
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
            ProcessingOutput(
                output_name="test_data",
                source="/opt/ml/processing/test",
                destination=Join(
                    on="/",
                    values=[
                        "s3:/",
                        pipeline_session.default_bucket(),
                        "test-sklearn",
                        output_prefix,
                        ExecutionVariables.PIPELINE_EXECUTION_ID,
                    ],
                ),
            ),
        ],
        code=os.path.join(script_dir, "preprocessing.py"),
    )

    # If image_uri is not provided, the instance_type should not be a pipeline variable
    # since instance_type is used to retrieve image_uri in compile time (PySDK)
    sklearn_train = SKLearn(
        framework_version=framework_version,
        entry_point=os.path.join(script_dir, "train.py"),
        instance_type=instance_type,
        sagemaker_session=pipeline_session,
        role=role,
        hyperparameters={
            "batch-size": 500,
            "epochs": 5,
        },
    )
    step_train = TrainingStep(
        name="my-train",
        display_name="TrainingStep",
        description="description for Training step",
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
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_model_args = model.create(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium",
    )
    step_model = ModelStep(
        name="my-model",
        display_name="ModelStep",
        description="description for Model step",
        step_args=step_model_args,
    )

    # Condition step for evaluating model quality and branching execution
    cond_lte = ConditionGreaterThanOrEqualTo(
        left=step_train.properties.HyperParameters["batch-size"],
        right=6.0,
    )

    step_cond = ConditionStep(
        name="CustomerChurnAccuracyCond",
        conditions=[cond_lte],
        if_steps=[],
        else_steps=[step_model],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count, output_prefix],
        steps=[step_process, step_train, step_cond],
        sagemaker_session=pipeline_session,
    )

    definition = json.loads(pipeline.definition())
    assert definition["Version"] == "2020-12-01"

    steps = definition["Steps"]
    assert len(steps) == 3
    training_args = {}
    condition_args = {}
    for step in steps:
        if step["Type"] == "Training":
            training_args = step["Arguments"]
        if step["Type"] == "Condition":
            condition_args = step["Arguments"]

    assert training_args["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"] == {
        "Get": "Steps.my-process.ProcessingOutputConfig.Outputs['train_data'].S3Output.S3Uri"
    }
    assert condition_args["Conditions"][0]["LeftValue"] == {
        "Get": "Steps.my-train.HyperParameters['batch-size']"
    }

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_one_step_ingestion_pipeline(
    sagemaker_session_for_pipeline, feature_store_session, feature_definitions, role, pipeline_name
):
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.4xlarge")

    input_name = "features.csv"
    input_file_path = os.path.join(DATA_DIR, "workflow", "features.csv")
    input_data_uri = os.path.join(
        "s3://",
        sagemaker_session_for_pipeline.default_bucket(),
        "py-sdk-ingestion-test-input/features.csv",
    )

    with open(input_file_path, "r") as data:
        body = data.read()
        S3Uploader.upload_string_as_file_body(
            body=body,
            desired_s3_uri=input_data_uri,
            sagemaker_session=sagemaker_session_for_pipeline,
        )

    inputs = [
        ProcessingInput(
            input_name=input_name,
            source=input_data_uri,
            destination="/opt/ml/processing/features.csv",
        )
    ]

    feature_group_name = f"py-sdk-integ-fg-{int(time.time() * 10**7)}"
    feature_group = FeatureGroup(
        name=feature_group_name,
        feature_definitions=feature_definitions,
        sagemaker_session=feature_store_session,
    )

    ingestion_only_flow, output_name = generate_data_ingestion_flow_from_s3_input(
        input_name,
        input_data_uri,
        s3_content_type="csv",
        s3_has_header=True,
    )

    outputs = [
        ProcessingOutput(
            output_name=output_name,
            app_managed=True,
            feature_store_output=FeatureStoreOutput(feature_group_name=feature_group_name),
        )
    ]

    output_content_type = "CSV"
    output_config = {output_name: {"content_type": output_content_type}}
    job_argument = [f"--output-config '{json.dumps(output_config)}'"]

    temp_flow_path = "./ingestion.flow"
    with cleanup_feature_group(feature_group):
        with open(temp_flow_path, "w") as f:
            json.dump(ingestion_only_flow, f)

        data_wrangler_processor = DataWranglerProcessor(
            role=role,
            data_wrangler_flow_source=temp_flow_path,
            instance_count=instance_count,
            instance_type=instance_type,
            sagemaker_session=sagemaker_session_for_pipeline,
            max_runtime_in_seconds=86400,
        )

        data_wrangler_step = ProcessingStep(
            name="ingestion-step",
            processor=data_wrangler_processor,
            inputs=inputs,
            outputs=outputs,
            job_arguments=job_argument,
        )

        pipeline = Pipeline(
            name=pipeline_name,
            parameters=[instance_count, instance_type],
            steps=[data_wrangler_step],
            sagemaker_session=sagemaker_session_for_pipeline,
        )

        try:
            response = pipeline.create(role)
            create_arn = response["PipelineArn"]

            offline_store_s3_uri = os.path.join(
                "s3://", sagemaker_session_for_pipeline.default_bucket(), feature_group_name
            )
            feature_group.create(
                s3_uri=offline_store_s3_uri,
                record_identifier_name="f11",
                event_time_feature_name="f10",
                role_arn=role,
                enable_online_store=False,
            )
            _wait_for_feature_group_create(feature_group)

            execution = pipeline.start()
            response = execution.describe()
            assert response["PipelineArn"] == create_arn

            wait_pipeline_execution(execution=execution, delay=60, max_attempts=10)

            execution_steps = execution.list_steps()

            assert len(execution_steps) == 1
            assert execution_steps[0]["StepName"] == "ingestion-step"
            assert execution_steps[0]["StepStatus"] == "Succeeded"

            athena_query = feature_group.athena_query()
            with timeout(minutes=10):
                athena_query.run(
                    query_string=f'SELECT * FROM "{athena_query.table_name}"',
                    output_location=f"{offline_store_s3_uri}/query_results",
                )
                athena_query.wait()
                assert "SUCCEEDED" == athena_query.get_query_execution().get("QueryExecution").get(
                    "Status"
                ).get("State")

                df = athena_query.as_dataframe()
                assert pd.read_csv(input_file_path).shape[0] == df.shape[0]
        finally:
            try:
                pipeline.delete()
            except Exception as e:
                print(f"Delete pipeline failed with error: {e}")
            os.remove(temp_flow_path)


@pytest.mark.skip(
    reason="""This test creates a long-running pipeline that
                            runs actual training jobs, processing jobs, etc.
                            All of the functionality in this test is covered in
                            shallow tests in this suite; as such, this is disabled
                            and only run as part of the 'lineage' test suite."""
)
def test_end_to_end_pipeline_successful_execution(
    sagemaker_session_for_pipeline, region_name, role, pipeline_name, wait=False
):
    model_package_group_name = f"{pipeline_name}ModelPackageGroup"
    data_path = os.path.join(DATA_DIR, "workflow")
    default_bucket = sagemaker_session_for_pipeline.default_bucket()

    # download the input data
    local_input_path = os.path.join(data_path, "abalone-dataset.csv")
    s3 = sagemaker_session_for_pipeline.boto_session.resource("s3")
    s3.Bucket(f"sagemaker-servicecatalog-seedcode-{region_name}").download_file(
        "dataset/abalone-dataset.csv", local_input_path
    )

    # # upload the input data to our bucket
    base_uri = f"s3://{default_bucket}/{pipeline_name}"
    with open(local_input_path) as data:
        body = data.read()
        input_data_uri = S3Uploader.upload_string_as_file_body(
            body=body,
            desired_s3_uri=f"{base_uri}/abalone-dataset.csv",
            sagemaker_session=sagemaker_session_for_pipeline,
        )

    # download batch transform data
    local_batch_path = os.path.join(data_path, "abalone-dataset-batch")
    s3.Bucket(f"sagemaker-servicecatalog-seedcode-{region_name}").download_file(
        "dataset/abalone-dataset-batch", local_batch_path
    )

    # upload the batch transform data
    with open(local_batch_path) as data:
        body = data.read()
        batch_data_uri = S3Uploader.upload_string_as_file_body(
            body=body,
            desired_s3_uri=f"{base_uri}/abalone-dataset-batch",
            sagemaker_session=sagemaker_session_for_pipeline,
        )

    # define parameters
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.m5.xlarge"
    )
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="Approved")
    input_data = ParameterString(
        name="InputData",
        default_value=input_data_uri,
    )
    batch_data = ParameterString(
        name="BatchData",
        default_value=batch_data_uri,
    )

    # define processing step
    framework_version = "0.23-1"
    sklearn_processor = SKLearnProcessor(
        framework_version=framework_version,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{pipeline_name}-process",
        role=role,
        sagemaker_session=sagemaker_session_for_pipeline,
    )
    step_process = ProcessingStep(
        name="AbaloneProcess",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(data_path, "abalone/preprocessing.py"),
    )

    # define training step
    model_path = f"s3://{default_bucket}/{pipeline_name}Train"
    image_uri = image_uris.retrieve(
        framework="xgboost",
        region=region_name,
        version="1.0-1",
        py_version="py3",
        instance_type=training_instance_type,
    )
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        role=role,
        sagemaker_session=sagemaker_session_for_pipeline,
    )
    xgb_train.set_hyperparameters(
        objective="reg:linear",
        num_round=50,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
        silent=0,
    )
    step_train = TrainingStep(
        name="AbaloneTrain",
        estimator=xgb_train,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    # define evaluation step
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{pipeline_name}-eval",
        role=role,
        sagemaker_session=sagemaker_session_for_pipeline,
    )
    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )
    step_eval = ProcessingStep(
        name="AbaloneEval",
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(data_path, "abalone/evaluation.py"),
        property_files=[evaluation_report],
    )

    # define create model step
    model = Model(
        image_uri=image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=sagemaker_session_for_pipeline,
        role=role,
    )
    inputs = CreateModelInput(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium",
    )
    step_create_model = CreateModelStep(
        name="AbaloneCreateModel",
        model=model,
        inputs=inputs,
    )

    # define transform step
    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        output_path=f"s3://{default_bucket}/{pipeline_name}Transform",
        sagemaker_session=sagemaker_session_for_pipeline,
    )
    step_transform = TransformStep(
        name="AbaloneTransform",
        transformer=transformer,
        inputs=TransformInput(data=batch_data),
    )

    # define register model step
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json",
        )
    )
    step_register = RegisterModel(
        name="AbaloneRegisterModel",
        estimator=xgb_train,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    # define condition step
    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.mse.value",
        ),
        right=20.0,
    )

    step_cond = ConditionStep(
        name="AbaloneMSECond",
        conditions=[cond_lte],
        if_steps=[step_register, step_create_model, step_transform],
        else_steps=[],
    )

    # define pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
            batch_data,
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=sagemaker_session_for_pipeline,
    )

    pipeline.create(role)
    execution = pipeline.start()
    execution_arn = execution.arn

    if wait:
        execution.wait()

    return execution_arn


def _wait_for_feature_group_create(feature_group: FeatureGroup):
    status = feature_group.describe().get("FeatureGroupStatus")
    while status == "Creating":
        print("Waiting for Feature Group Creation")
        time.sleep(5)
        status = feature_group.describe().get("FeatureGroupStatus")
    if status != "Created":
        print(feature_group.describe())
        raise RuntimeError(f"Failed to create feature group {feature_group.name}")
    print(f"FeatureGroup {feature_group.name} successfully created.")


@pytest.fixture
def feature_definitions():
    return [
        FeatureDefinition(feature_name="f1", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="f2", feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name="f3", feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name="f4", feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name="f5", feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name="f6", feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name="f7", feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name="f8", feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name="f9", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="f10", feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name="f11", feature_type=FeatureTypeEnum.STRING),
    ]


@contextmanager
def cleanup_feature_group(feature_group: FeatureGroup):
    try:
        yield
    finally:
        try:
            feature_group.delete()
            print("FeatureGroup cleaned up")
        except Exception as e:
            print(f"Delete FeatureGroup failed with error: {e}.")
            pass


def test_large_pipeline(sagemaker_session_for_pipeline, role, pipeline_name, region_name):
    instance_count = ParameterInteger(name="InstanceCount", default_value=2)

    outputParam = CallbackOutput(output_name="output", output_type=CallbackOutputTypeEnum.String)

    callback_steps = [
        CallbackStep(
            name=f"callback-step{count}",
            sqs_queue_url="https://sqs.us-east-2.amazonaws.com/123456789012/MyQueue",
            inputs={"arg1": "foo"},
            outputs=[outputParam],
        )
        for count in range(2000)
    ]
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count],
        steps=callback_steps,
        sagemaker_session=sagemaker_session_for_pipeline,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )
        assert len(json.loads(pipeline.describe()["PipelineDefinition"])["Steps"]) == 2000

        pipeline.parameters = [ParameterInteger(name="InstanceCount", default_value=1)]
        response = pipeline.update(role)
        update_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            update_arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_create_and_update_with_parallelism_config(
    sagemaker_session_for_pipeline, role, pipeline_name, region_name
):
    instance_count = ParameterInteger(name="InstanceCount", default_value=2)

    outputParam = CallbackOutput(output_name="output", output_type=CallbackOutputTypeEnum.String)

    callback_steps = [
        CallbackStep(
            name=f"callback-step{count}",
            sqs_queue_url="https://sqs.us-east-2.amazonaws.com/123456789012/MyQueue",
            inputs={"arg1": "foo"},
            outputs=[outputParam],
        )
        for count in range(500)
    ]
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count],
        steps=callback_steps,
        sagemaker_session=sagemaker_session_for_pipeline,
    )

    try:
        response = pipeline.create(role, parallelism_config={"MaxParallelExecutionSteps": 50})
        create_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )
        response = pipeline.describe()
        assert response["ParallelismConfiguration"]["MaxParallelExecutionSteps"] == 50

        pipeline.parameters = [ParameterInteger(name="InstanceCount", default_value=1)]
        response = pipeline.upsert(role, parallelism_config={"MaxParallelExecutionSteps": 55})
        update_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            update_arn,
        )

        response = pipeline.describe()
        assert response["ParallelismConfiguration"]["MaxParallelExecutionSteps"] == 55

    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_create_and_start_without_parallelism_config_override(
    pipeline_session, role, pipeline_name, script_dir
):
    sklearn_train = SKLearn(
        framework_version="0.20.0",
        entry_point=os.path.join(script_dir, "train.py"),
        instance_type="ml.m5.xlarge",
        sagemaker_session=pipeline_session,
        role=role,
    )

    train_steps = [
        TrainingStep(
            name=f"my-train-{count}",
            display_name="TrainingStep",
            description="description for Training step",
            step_args=sklearn_train.fit(),
        )
        for count in range(2)
    ]
    pipeline = Pipeline(
        name=pipeline_name,
        steps=train_steps,
        sagemaker_session=pipeline_session,
    )

    try:
        pipeline.create(role, parallelism_config=dict(MaxParallelExecutionSteps=1))
        # No ParallelismConfiguration given in pipeline.start, so it won't override that in pipeline.create
        execution = pipeline.start()

        def validate():
            # Only one step would be scheduled initially
            assert len(execution.list_steps()) == 1

        retry_with_backoff(validate, num_attempts=4)

        wait_pipeline_execution(execution=execution)

    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_create_and_start_with_parallelism_config_override(
    pipeline_session, role, pipeline_name, script_dir
):
    sklearn_train = SKLearn(
        framework_version="0.20.0",
        entry_point=os.path.join(script_dir, "train.py"),
        instance_type="ml.m5.xlarge",
        sagemaker_session=pipeline_session,
        role=role,
    )

    train_steps = [
        TrainingStep(
            name=f"my-train-{count}",
            display_name="TrainingStep",
            description="description for Training step",
            step_args=sklearn_train.fit(),
        )
        for count in range(2)
    ]
    pipeline = Pipeline(
        name=pipeline_name,
        steps=train_steps,
        sagemaker_session=pipeline_session,
    )

    try:
        pipeline.create(role, parallelism_config=dict(MaxParallelExecutionSteps=1))
        # Override ParallelismConfiguration in pipeline.start
        execution = pipeline.start(parallelism_config=dict(MaxParallelExecutionSteps=2))

        def validate():
            assert len(execution.list_steps()) == 2
            for step in execution.list_steps():
                assert step["StepStatus"] == "Executing"

        retry_with_backoff(validate, num_attempts=4)

        wait_pipeline_execution(execution=execution)

    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_model_registration_with_tuning_model(
    pipeline_session,
    role,
    cpu_instance_type,
    pipeline_name,
    region_name,
):
    base_dir = os.path.join(DATA_DIR, "pipeline/model_step/pytorch_mnist")
    entry_point = "mnist.py"
    input_path = pipeline_session.upload_data(
        path=os.path.join(base_dir, "training"),
        key_prefix="integ-test-data/pytorch_mnist/training",
    )
    inputs = TrainingInput(s3_data=input_path)

    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = "ml.m5.xlarge"

    # If image_uri is not provided, the instance_type should not be a pipeline variable
    # since instance_type is used to retrieve image_uri in compile time (PySDK)
    pytorch_estimator = PyTorch(
        entry_point=entry_point,
        source_dir=base_dir,
        role=role,
        framework_version="1.13.1",
        py_version="py39",
        instance_count=instance_count,
        instance_type=instance_type,
        sagemaker_session=pipeline_session,
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
    model = PyTorchModel(
        image_uri=pytorch_estimator.training_image_uri(),
        role=role,
        model_data=step_tune.get_top_model_s3_uri(
            top_k=0,
            s3_bucket=pipeline_session.default_bucket(),
        ),
        entry_point=entry_point,
        source_dir=base_dir,
        framework_version="1.13.1",
        py_version="py39",
        sagemaker_session=pipeline_session,
    )
    step_model_regis_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=f"{pipeline_name}TestModelPackageGroup",
    )
    step_register_best = ModelStep(
        name="my-model-regis",
        step_args=step_model_regis_args,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count, min_batch_size, max_batch_size],
        steps=[step_tune, step_register_best],
        sagemaker_session=pipeline_session,
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
        wait_pipeline_execution(execution=execution)
        execution_steps = execution.list_steps()

        for step in execution_steps:
            assert not step.get("FailureReason", None)
            assert step["StepStatus"] == "Succeeded"
            if _REGISTER_MODEL_NAME_BASE in step["StepName"]:
                assert step["Metadata"]["RegisterModel"]
            if _REPACK_MODEL_NAME_BASE in step["StepName"]:
                _verify_repack_output(step, pipeline_session)
        assert len(execution_steps) == 3
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def _verify_repack_output(repack_step_dict, sagemaker_session_for_pipeline):
    # This is to verify if the `requirements.txt` provided in ModelStep
    # is not auto installed in the Repack step but is successfully repacked
    # in the new model.tar.gz
    # The repack step is using an old version of SKLearn framework "0.23-1"
    # so if the `requirements.txt` is auto installed, it should raise an exception
    # caused by the unsupported library version listed in the `requirements.txt`
    training_job_arn = repack_step_dict["Metadata"]["TrainingJob"]["Arn"]
    job_description = sagemaker_session_for_pipeline.sagemaker_client.describe_training_job(
        TrainingJobName=training_job_arn.split("/")[1]
    )
    model_uri = job_description["ModelArtifacts"]["S3ModelArtifacts"]
    with tempfile.TemporaryDirectory() as tmp:
        extract_files_from_s3(
            s3_url=model_uri, tmpdir=tmp, sagemaker_session=sagemaker_session_for_pipeline
        )

        def walk():
            results = set()
            for root, dirs, files in os.walk(tmp):
                relative_path = root.replace(tmp, "")
                for f in files:
                    results.add(f"{relative_path}/{f}")
            return results

        tar_files = walk()
        assert {"/code/mnist.py", "/code/requirements.txt", "/model.pth"}.issubset(tar_files)


def test_caching_behavior(
    pipeline_session,
    role,
    cpu_instance_type,
    pipeline_name,
    script_dir,
    athena_dataset_definition,
    region_name,
):
    default_bucket = pipeline_session.default_bucket()
    data_path = os.path.join(DATA_DIR, "workflow")

    framework_version = "0.20.0"
    instance_type = "ml.m5.xlarge"
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    output_prefix = ParameterString(name="OutputPrefix", default_value="output")

    input_data = f"s3://sagemaker-sample-data-{region_name}/processing/census/census-income.csv"

    # additionally add abalone input, so we can test input s3 file from local upload
    abalone_input = ProcessingInput(
        input_name="abalone_data",
        source=os.path.join(data_path, "abalone-dataset.csv"),
        destination="/opt/ml/processing/input",
    )

    # define processing step
    sklearn_processor = SKLearnProcessor(
        framework_version=framework_version,
        instance_type=instance_type,
        instance_count=instance_count,
        base_job_name="test-sklearn",
        sagemaker_session=pipeline_session,
        role=role,
    )
    processor_args = sklearn_processor.run(
        inputs=[
            ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),
            ProcessingInput(dataset_definition=athena_dataset_definition),
            abalone_input,
        ],
        outputs=[
            ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train"),
            ProcessingOutput(
                output_name="test_data",
                source="/opt/ml/processing/test",
                destination=Join(
                    on="/",
                    values=[
                        "s3:/",
                        pipeline_session.default_bucket(),
                        "test-sklearn",
                        output_prefix,
                        ExecutionVariables.PIPELINE_EXECUTION_ID,
                    ],
                ),
            ),
        ],
        code=os.path.join(script_dir, "preprocessing.py"),
    )
    step_process = ProcessingStep(
        name="my-process",
        display_name="ProcessingStep",
        description="description for Processing step",
        step_args=processor_args,
    )

    # define training step
    sklearn_train = SKLearn(
        framework_version=framework_version,
        source_dir=script_dir,
        entry_point=os.path.join(script_dir, "train.py"),
        instance_type=instance_type,
        sagemaker_session=pipeline_session,
        role=role,
    )
    train_args = sklearn_train.fit(
        inputs=TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                "train_data"
            ].S3Output.S3Uri
        ),
    )
    step_train = TrainingStep(
        name="my-train",
        display_name="TrainingStep",
        description="description for Training step",
        step_args=train_args,
    )

    # define pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count, output_prefix],
        steps=[step_process, step_train],
        sagemaker_session=pipeline_session,
    )

    try:
        # create pipeline
        pipeline.create(role)
        definition = json.loads(pipeline.definition())

        # verify input path
        expected_abalone_input_path = f"{pipeline_name}/{step_process.name}" f"/input/abalone_data"
        expected_abalone_input_file = f"{expected_abalone_input_path}/abalone-dataset.csv"

        s3_input_objects = pipeline_session.list_s3_files(
            bucket=default_bucket, key_prefix=expected_abalone_input_path
        )
        assert expected_abalone_input_file in s3_input_objects

        # verify code path
        expected_code_path = f"{pipeline_name}/code/" f"{hash_files_or_dirs([script_dir])}"
        expected_training_file = f"{expected_code_path}/sourcedir.tar.gz"

        s3_code_objects = pipeline_session.list_s3_files(
            bucket=default_bucket, key_prefix=expected_code_path
        )
        assert expected_training_file in s3_code_objects

        # update pipeline
        pipeline.update(role)

        # verify no changes
        definition2 = json.loads(pipeline.definition())
        assert definition == definition2

        # add dummy file to source_dir
        shutil.copyfile(DATA_DIR + "/dummy_script.py", script_dir + "/dummy_script.py")

        # update pipeline again
        pipeline.update(role)

        # verify changes
        definition3 = json.loads(pipeline.definition())
        assert definition != definition3

    finally:
        try:
            os.remove(script_dir + "/dummy_script.py")
            pipeline.delete()
        except Exception:
            os.remove(script_dir + "/dummy_script.py")
            pass


def test_pipeline_versioning(pipeline_session, role, pipeline_name, script_dir):
    sklearn_train = SKLearn(
        framework_version="0.20.0",
        entry_point=os.path.join(script_dir, "train.py"),
        instance_type="ml.m5.xlarge",
        sagemaker_session=pipeline_session,
        role=role,
    )

    step1 = TrainingStep(
        name="my-train-1",
        display_name="TrainingStep",
        description="description for Training step",
        step_args=sklearn_train.fit(),
    )

    step2 = TrainingStep(
        name="my-train-2",
        display_name="TrainingStep",
        description="description for Training step",
        step_args=sklearn_train.fit(),
    )
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step1],
        sagemaker_session=pipeline_session,
    )

    try:
        pipeline.create(role)

        assert pipeline.latest_pipeline_version_id == 1

        describe_response = pipeline.describe(pipeline_version_id=1)
        assert len(json.loads(describe_response["PipelineDefinition"])["Steps"]) == 1

        pipeline.steps.append(step2)
        pipeline.upsert(role)

        assert pipeline.latest_pipeline_version_id == 2

        describe_response = pipeline.describe(pipeline_version_id=2)
        assert len(json.loads(describe_response["PipelineDefinition"])["Steps"]) == 2

        assert len(pipeline.list_pipeline_versions()["PipelineVersionSummaries"]) == 2

    finally:
        try:
            pipeline.delete()
        except Exception:
            pass
