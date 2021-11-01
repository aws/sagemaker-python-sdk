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
import subprocess
import time
import uuid
import logging

from contextlib import contextmanager
import pytest

from botocore.exceptions import WaiterError
import pandas as pd
from tests.integ.timeout import timeout

from sagemaker.debugger import (
    DebuggerHookConfig,
    Rule,
    rule_configs,
)
from datetime import datetime
from sagemaker.session import Session
from sagemaker import image_uris
from sagemaker.inputs import CreateModelInput, TrainingInput
from sagemaker.model import Model
from sagemaker import PipelineModel
from sagemaker.processing import ProcessingInput, ProcessingOutput, FeatureStoreOutput
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.tuner import HyperparameterTuner, IntegerParameter
from sagemaker.s3 import S3Uploader
from sagemaker.session import get_execution_role
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn import SKLearnModel
from sagemaker.mxnet.model import MXNetModel
from sagemaker.xgboost import XGBoostModel
from sagemaker.xgboost import XGBoost
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.spark.processing import PySparkProcessor, SparkJarProcessor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo, ConditionIn
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.callback_step import CallbackStep, CallbackOutput, CallbackOutputTypeEnum
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum
from sagemaker.wrangler.processing import DataWranglerProcessor
from sagemaker.dataset_definition.inputs import DatasetDefinition, AthenaDatasetDefinition
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.wrangler.ingestion import generate_data_ingestion_flow_from_s3_input
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.steps import (
    CreateModelStep,
    ProcessingStep,
    TrainingStep,
    CacheConfig,
    TuningStep,
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.lambda_helper import Lambda
from sagemaker.feature_store.feature_group import FeatureGroup, FeatureDefinition, FeatureTypeEnum
from tests.integ import DATA_DIR
from tests.integ.kms_utils import get_or_create_kms_key
from tests.integ.retry import retries


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


@pytest.fixture(scope="module")
def script_dir():
    return os.path.join(DATA_DIR, "sklearn_processing")


@pytest.fixture(scope="module")
def feature_store_session(sagemaker_session):
    boto_session = sagemaker_session.boto_session
    sagemaker_client = boto_session.client("sagemaker")
    featurestore_runtime_client = boto_session.client("sagemaker-featurestore-runtime")

    return Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_featurestore_runtime_client=featurestore_runtime_client,
    )


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


@pytest.fixture
def configuration() -> list:
    configuration = [
        {
            "Classification": "spark-defaults",
            "Properties": {"spark.executor.memory": "2g", "spark.executor.cores": "1"},
        },
        {
            "Classification": "hadoop-env",
            "Properties": {},
            "Configurations": [
                {
                    "Classification": "export",
                    "Properties": {
                        "HADOOP_DATANODE_HEAPSIZE": "2048",
                        "HADOOP_NAMENODE_OPTS": "-XX:GCTimeRatio=19",
                    },
                    "Configurations": [],
                }
            ],
        },
        {
            "Classification": "core-site",
            "Properties": {"spark.executor.memory": "2g", "spark.executor.cores": "1"},
        },
        {"Classification": "hadoop-log4j", "Properties": {"key": "value"}},
        {
            "Classification": "hive-env",
            "Properties": {},
            "Configurations": [
                {
                    "Classification": "export",
                    "Properties": {
                        "HADOOP_DATANODE_HEAPSIZE": "2048",
                        "HADOOP_NAMENODE_OPTS": "-XX:GCTimeRatio=19",
                    },
                    "Configurations": [],
                }
            ],
        },
        {"Classification": "hive-log4j", "Properties": {"key": "value"}},
        {"Classification": "hive-exec-log4j", "Properties": {"key": "value"}},
        {"Classification": "hive-site", "Properties": {"key": "value"}},
        {"Classification": "spark-defaults", "Properties": {"key": "value"}},
        {
            "Classification": "spark-env",
            "Properties": {},
            "Configurations": [
                {
                    "Classification": "export",
                    "Properties": {
                        "HADOOP_DATANODE_HEAPSIZE": "2048",
                        "HADOOP_NAMENODE_OPTS": "-XX:GCTimeRatio=19",
                    },
                    "Configurations": [],
                }
            ],
        },
        {"Classification": "spark-log4j", "Properties": {"key": "value"}},
        {"Classification": "spark-hive-site", "Properties": {"key": "value"}},
        {"Classification": "spark-metrics", "Properties": {"key": "value"}},
        {"Classification": "yarn-site", "Properties": {"key": "value"}},
        {
            "Classification": "yarn-env",
            "Properties": {},
            "Configurations": [
                {
                    "Classification": "export",
                    "Properties": {
                        "HADOOP_DATANODE_HEAPSIZE": "2048",
                        "HADOOP_NAMENODE_OPTS": "-XX:GCTimeRatio=19",
                    },
                    "Configurations": [],
                }
            ],
        },
    ]
    return configuration


@pytest.fixture(scope="module")
def build_jar():
    spark_path = os.path.join(DATA_DIR, "spark")
    java_file_path = os.path.join("com", "amazonaws", "sagemaker", "spark", "test")
    java_version_pattern = r"(\d+\.\d+).*"
    jar_file_path = os.path.join(spark_path, "code", "java", "hello-java-spark")
    # compile java file
    java_version = subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT).decode(
        "utf-8"
    )
    java_version = re.search(java_version_pattern, java_version).groups()[0]

    if float(java_version) > 1.8:
        subprocess.run(
            [
                "javac",
                "--release",
                "8",
                os.path.join(jar_file_path, java_file_path, "HelloJavaSparkApp.java"),
            ]
        )
    else:
        subprocess.run(
            ["javac", os.path.join(jar_file_path, java_file_path, "HelloJavaSparkApp.java")]
        )

    subprocess.run(
        [
            "jar",
            "cfm",
            os.path.join(jar_file_path, "hello-spark-java.jar"),
            os.path.join(jar_file_path, "manifest.txt"),
            "-C",
            jar_file_path,
            ".",
        ]
    )
    yield
    subprocess.run(["rm", os.path.join(jar_file_path, "hello-spark-java.jar")])
    subprocess.run(["rm", os.path.join(jar_file_path, java_file_path, "HelloJavaSparkApp.class")])


def test_three_step_definition(
    sagemaker_session,
    region_name,
    role,
    script_dir,
    pipeline_name,
    athena_dataset_definition,
):
    framework_version = "0.20.0"
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    output_prefix = ParameterString(name="OutputPrefix", default_value="output")

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
                        sagemaker_session.default_bucket(),
                        "test-sklearn",
                        output_prefix,
                        ExecutionVariables.PIPELINE_EXECUTION_ID,
                    ],
                ),
            ),
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
        sagemaker_session=sagemaker_session,
        role=role,
    )
    model_inputs = CreateModelInput(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium",
    )
    step_model = CreateModelStep(
        name="my-model",
        display_name="ModelStep",
        description="description for Model step",
        model=model,
        inputs=model_inputs,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_type, instance_count, output_prefix],
        steps=[step_process, step_train, step_model],
        sagemaker_session=sagemaker_session,
    )

    definition = json.loads(pipeline.definition())
    assert definition["Version"] == "2020-12-01"

    assert set(tuple(param.items()) for param in definition["Parameters"]) == set(
        [
            tuple(
                {"Name": "InstanceType", "Type": "String", "DefaultValue": "ml.m5.xlarge"}.items()
            ),
            tuple({"Name": "InstanceCount", "Type": "Integer", "DefaultValue": 1}.items()),
            tuple({"Name": "OutputPrefix", "Type": "String", "DefaultValue": "output"}.items()),
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
            ("my-model", "Model"),
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
    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_one_step_sklearn_processing_pipeline(
    sagemaker_session,
    role,
    sklearn_latest_version,
    cpu_instance_type,
    pipeline_name,
    region_name,
    athena_dataset_definition,
):
    instance_count = ParameterInteger(name="InstanceCount", default_value=2)
    script_path = os.path.join(DATA_DIR, "dummy_script.py")
    input_file_path = os.path.join(DATA_DIR, "dummy_input.txt")
    inputs = [
        ProcessingInput(source=input_file_path, destination="/opt/ml/processing/inputs/"),
        ProcessingInput(dataset_definition=athena_dataset_definition),
    ]

    cache_config = CacheConfig(enable_caching=True, expire_after="T30m")

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
        cache_config=cache_config,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count],
        steps=[step_sklearn],
        sagemaker_session=sagemaker_session,
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
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

        pipeline.parameters = [ParameterInteger(name="InstanceCount", default_value=1)]
        execution = pipeline.start(parameters={})
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )

        response = execution.describe()
        assert response["PipelineArn"] == create_arn

        # Check CacheConfig
        response = json.loads(pipeline.describe()["PipelineDefinition"])["Steps"][0]["CacheConfig"]
        assert response["Enabled"] == cache_config.enable_caching
        assert response["ExpireAfter"] == cache_config.expire_after

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


def test_one_step_framework_processing_pipeline(
    sagemaker_session,
    role,
    sklearn_latest_version,
    cpu_instance_type,
    pipeline_name,
    region_name,
    athena_dataset_definition,
):
    """Use `SKLearnProcessor` to test `FrameworkProcessor`."""
    instance_count = ParameterInteger(name="InstanceCount", default_value=2)
    script_path = os.path.join(DATA_DIR, "dummy_script.py")
    input_file_path = os.path.join(DATA_DIR, "dummy_input.txt")

    inputs = [
        ProcessingInput(source=input_file_path, destination="/opt/ml/processing/inputs/"),
        ProcessingInput(dataset_definition=athena_dataset_definition),
    ]

    cache_config = CacheConfig(enable_caching=True, expire_after="T30m")

    sklearn_processor = SKLearnProcessor(
        framework_version=sklearn_latest_version,
        role=role,
        instance_type=cpu_instance_type,
        instance_count=instance_count,
        sagemaker_session=sagemaker_session,
        base_job_name="test-sklearn",
    )

    run_args = sklearn_processor.get_run_args(code=script_path, inputs=inputs)

    step_sklearn = ProcessingStep(
        name="sklearn-process",
        processor=sklearn_processor,
        inputs=run_args.inputs,
        outputs=run_args.outputs,
        job_arguments=run_args.arguments,
        code=run_args.code,
        cache_config=cache_config,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count],
        steps=[step_sklearn],
        sagemaker_session=sagemaker_session,
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
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

        pipeline.parameters = [ParameterInteger(name="InstanceCount", default_value=1)]
        response = pipeline.update(role)
        update_arn = response["PipelineArn"]
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            update_arn,
        )

        execution = pipeline.start(parameters={})
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )

        response = execution.describe()
        assert response["PipelineArn"] == create_arn

        # Check CacheConfig
        response = json.loads(pipeline.describe()["PipelineDefinition"])["Steps"][0]["CacheConfig"]
        assert response["Enabled"] == cache_config.enable_caching
        assert response["ExpireAfter"] == cache_config.expire_after

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


def test_one_step_pyspark_processing_pipeline(
    sagemaker_session,
    role,
    cpu_instance_type,
    pipeline_name,
    region_name,
):
    instance_count = ParameterInteger(name="InstanceCount", default_value=2)
    script_path = os.path.join(DATA_DIR, "dummy_script.py")

    cache_config = CacheConfig(enable_caching=True, expire_after="T30m")

    pyspark_processor = PySparkProcessor(
        base_job_name="sm-spark",
        framework_version="2.4",
        role=role,
        instance_count=instance_count,
        instance_type=cpu_instance_type,
        max_runtime_in_seconds=1200,
        sagemaker_session=sagemaker_session,
    )

    spark_run_args = pyspark_processor.get_run_args(
        submit_app=script_path,
        arguments=[
            "--s3_input_bucket",
            sagemaker_session.default_bucket(),
            "--s3_input_key_prefix",
            "spark-input",
            "--s3_output_bucket",
            sagemaker_session.default_bucket(),
            "--s3_output_key_prefix",
            "spark-output",
        ],
    )

    step_pyspark = ProcessingStep(
        name="pyspark-process",
        processor=pyspark_processor,
        inputs=spark_run_args.inputs,
        outputs=spark_run_args.outputs,
        job_arguments=spark_run_args.arguments,
        code=spark_run_args.code,
        cache_config=cache_config,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count],
        steps=[step_pyspark],
        sagemaker_session=sagemaker_session,
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
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

        pipeline.parameters = [ParameterInteger(name="InstanceCount", default_value=1)]
        response = pipeline.update(role)
        update_arn = response["PipelineArn"]
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            update_arn,
        )

        execution = pipeline.start(parameters={})
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )

        response = execution.describe()
        assert response["PipelineArn"] == create_arn

        # Check CacheConfig
        response = json.loads(pipeline.describe()["PipelineDefinition"])["Steps"][0]["CacheConfig"]
        assert response["Enabled"] == cache_config.enable_caching
        assert response["ExpireAfter"] == cache_config.expire_after

        try:
            execution.wait(delay=30, max_attempts=3)
        except WaiterError:
            pass
        execution_steps = execution.list_steps()
        assert len(execution_steps) == 1
        assert execution_steps[0]["StepName"] == "pyspark-process"
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_one_step_sparkjar_processing_pipeline(
    sagemaker_session, role, cpu_instance_type, pipeline_name, region_name, configuration, build_jar
):
    instance_count = ParameterInteger(name="InstanceCount", default_value=2)
    cache_config = CacheConfig(enable_caching=True, expire_after="T30m")
    spark_path = os.path.join(DATA_DIR, "spark")

    spark_jar_processor = SparkJarProcessor(
        role=role,
        instance_count=2,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        framework_version="2.4",
    )
    bucket = spark_jar_processor.sagemaker_session.default_bucket()
    with open(os.path.join(spark_path, "files", "data.jsonl")) as data:
        body = data.read()
        input_data_uri = f"s3://{bucket}/spark/input/data.jsonl"
        S3Uploader.upload_string_as_file_body(
            body=body, desired_s3_uri=input_data_uri, sagemaker_session=sagemaker_session
        )
    output_data_uri = f"s3://{bucket}/spark/output/sales/{datetime.now().isoformat()}"

    java_project_dir = os.path.join(spark_path, "code", "java", "hello-java-spark")
    spark_run_args = spark_jar_processor.get_run_args(
        submit_app=f"{java_project_dir}/hello-spark-java.jar",
        submit_class="com.amazonaws.sagemaker.spark.test.HelloJavaSparkApp",
        arguments=["--input", input_data_uri, "--output", output_data_uri],
        configuration=configuration,
    )

    step_pyspark = ProcessingStep(
        name="sparkjar-process",
        processor=spark_jar_processor,
        inputs=spark_run_args.inputs,
        outputs=spark_run_args.outputs,
        job_arguments=spark_run_args.arguments,
        code=spark_run_args.code,
        cache_config=cache_config,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count],
        steps=[step_pyspark],
        sagemaker_session=sagemaker_session,
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
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

        pipeline.parameters = [ParameterInteger(name="InstanceCount", default_value=1)]
        response = pipeline.update(role)
        update_arn = response["PipelineArn"]
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            update_arn,
        )

        execution = pipeline.start(parameters={})
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )

        response = execution.describe()
        assert response["PipelineArn"] == create_arn

        # Check CacheConfig
        response = json.loads(pipeline.describe()["PipelineDefinition"])["Steps"][0]["CacheConfig"]
        assert response["Enabled"] == cache_config.enable_caching
        assert response["ExpireAfter"] == cache_config.expire_after

        try:
            execution.wait(delay=30, max_attempts=3)
        except WaiterError:
            pass
        execution_steps = execution.list_steps()
        assert len(execution_steps) == 1
        assert execution_steps[0]["StepName"] == "sparkjar-process"
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_one_step_callback_pipeline(sagemaker_session, role, pipeline_name, region_name):
    instance_count = ParameterInteger(name="InstanceCount", default_value=2)

    outputParam1 = CallbackOutput(output_name="output1", output_type=CallbackOutputTypeEnum.String)
    step_callback = CallbackStep(
        name="callback-step",
        sqs_queue_url="https://sqs.us-east-2.amazonaws.com/123456789012/MyQueue",
        inputs={"arg1": "foo"},
        outputs=[outputParam1],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count],
        steps=[step_callback],
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

        pipeline.parameters = [ParameterInteger(name="InstanceCount", default_value=1)]
        response = pipeline.update(role)
        update_arn = response["PipelineArn"]
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            update_arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_steps_with_map_params_pipeline(
    sagemaker_session, role, script_dir, pipeline_name, region_name, athena_dataset_definition
):
    instance_count = ParameterInteger(name="InstanceCount", default_value=2)
    framework_version = "0.20.0"
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")
    output_prefix = ParameterString(name="OutputPrefix", default_value="output")
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
                        sagemaker_session.default_bucket(),
                        "test-sklearn",
                        output_prefix,
                        ExecutionVariables.PIPELINE_EXECUTION_ID,
                    ],
                ),
            ),
        ],
        code=os.path.join(script_dir, "preprocessing.py"),
    )

    sklearn_train = SKLearn(
        framework_version=framework_version,
        entry_point=os.path.join(script_dir, "train.py"),
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
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
        sagemaker_session=sagemaker_session,
        role=role,
    )
    model_inputs = CreateModelInput(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium",
    )
    step_model = CreateModelStep(
        name="my-model",
        display_name="ModelStep",
        description="description for Model step",
        model=model,
        inputs=model_inputs,
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
        parameters=[instance_type, instance_count, output_prefix],
        steps=[step_process, step_train, step_cond],
        sagemaker_session=sagemaker_session,
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
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_two_step_callback_pipeline_with_output_reference(
    sagemaker_session, role, pipeline_name, region_name
):
    instance_count = ParameterInteger(name="InstanceCount", default_value=2)

    outputParam1 = CallbackOutput(output_name="output1", output_type=CallbackOutputTypeEnum.String)
    step_callback1 = CallbackStep(
        name="callback-step1",
        sqs_queue_url="https://sqs.us-east-2.amazonaws.com/123456789012/MyQueue",
        inputs={"arg1": "foo"},
        outputs=[outputParam1],
    )

    step_callback2 = CallbackStep(
        name="callback-step2",
        sqs_queue_url="https://sqs.us-east-2.amazonaws.com/123456789012/MyQueue",
        inputs={"arg1": outputParam1},
        outputs=[],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count],
        steps=[step_callback1, step_callback2],
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_one_step_lambda_pipeline(sagemaker_session, role, pipeline_name, region_name):
    instance_count = ParameterInteger(name="InstanceCount", default_value=2)

    outputParam1 = LambdaOutput(output_name="output1", output_type=LambdaOutputTypeEnum.String)
    step_lambda = LambdaStep(
        name="lambda-step",
        lambda_func=Lambda(
            function_arn="arn:aws:lambda:us-west-2:123456789012:function:sagemaker_test_lambda",
            session=sagemaker_session,
        ),
        inputs={"arg1": "foo"},
        outputs=[outputParam1],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count],
        steps=[step_lambda],
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

        pipeline.parameters = [ParameterInteger(name="InstanceCount", default_value=1)]
        response = pipeline.update(role)
        update_arn = response["PipelineArn"]
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            update_arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_two_step_lambda_pipeline_with_output_reference(
    sagemaker_session, role, pipeline_name, region_name
):
    instance_count = ParameterInteger(name="InstanceCount", default_value=2)

    outputParam1 = LambdaOutput(output_name="output1", output_type=LambdaOutputTypeEnum.String)
    step_lambda1 = LambdaStep(
        name="lambda-step1",
        lambda_func=Lambda(
            function_arn="arn:aws:lambda:us-west-2:123456789012:function:sagemaker_test_lambda",
            session=sagemaker_session,
        ),
        inputs={"arg1": "foo"},
        outputs=[outputParam1],
    )

    step_lambda2 = LambdaStep(
        name="lambda-step2",
        lambda_func=Lambda(
            function_arn="arn:aws:lambda:us-west-2:123456789012:function:sagemaker_test_lambda",
            session=sagemaker_session,
        ),
        inputs={"arg1": outputParam1},
        outputs=[],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count],
        steps=[step_lambda1, step_lambda2],
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_conditional_pytorch_training_model_registration(
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
    good_enough_input = ParameterInteger(name="GoodEnoughInput", default_value=1)
    in_condition_input = ParameterString(name="Foo", default_value="Foo")

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
        description="test-description",
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
        conditions=[
            ConditionGreaterThanOrEqualTo(left=good_enough_input, right=1),
            ConditionIn(value=in_condition_input, in_values=["foo", "bar"]),
        ],
        if_steps=[step_train, step_register],
        else_steps=[step_model],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[in_condition_input, good_enough_input, instance_count, instance_type],
        steps=[step_cond],
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}", create_arn
        )

        execution = pipeline.start(parameters={})
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )

        execution = pipeline.start(parameters={"GoodEnoughInput": 0})
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


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
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}", create_arn
        )

        execution = pipeline.start(parameters={})
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
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
):
    base_dir = os.path.join(DATA_DIR, "pytorch_mnist")
    entry_point = os.path.join(base_dir, "mnist.py")
    input_path = sagemaker_session.upload_data(
        path=os.path.join(base_dir, "training"),
        key_prefix="integ-test-data/pytorch_mnist/training",
    )

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
        steps=[step_tune],
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}", create_arn
        )

        execution = pipeline.start(parameters={})
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_mxnet_model_registration(
    sagemaker_session,
    role,
    cpu_instance_type,
    pipeline_name,
    region_name,
):
    base_dir = os.path.join(DATA_DIR, "mxnet_mnist")
    source_dir = os.path.join(base_dir, "code")
    entry_point = os.path.join(source_dir, "inference.py")
    mx_mnist_model_data = os.path.join(base_dir, "model.tar.gz")

    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")

    model = MXNetModel(
        entry_point=entry_point,
        source_dir=source_dir,
        role=role,
        model_data=mx_mnist_model_data,
        framework_version="1.7.0",
        py_version="py3",
        sagemaker_session=sagemaker_session,
    )

    step_register = RegisterModel(
        name="mxnet-register-model",
        model=model,
        content_types=["*"],
        response_types=["*"],
        inference_instances=["ml.m5.xlarge"],
        transform_instances=["*"],
        description="test-description",
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count, instance_type],
        steps=[step_register],
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}", create_arn
        )

        execution = pipeline.start(parameters={})
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )

        execution = pipeline.start()
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_sklearn_xgboost_sip_model_registration(
    sagemaker_session,
    role,
    cpu_instance_type,
    pipeline_name,
    region_name,
):
    prefix = "sip"
    bucket_name = sagemaker_session.default_bucket()
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")

    sklearn_processor = SKLearnProcessor(
        role=role,
        instance_type=instance_type,
        instance_count=instance_count,
        framework_version="0.20.0",
        sagemaker_session=sagemaker_session,
    )

    # The path to the raw data.
    raw_data_path = "s3://{0}/{1}/data/raw/".format(bucket_name, prefix)
    raw_data_path_param = ParameterString(name="raw_data_path", default_value=raw_data_path)

    # The output path to the training data.
    train_data_path = "s3://{0}/{1}/data/preprocessed/train/".format(bucket_name, prefix)
    train_data_path_param = ParameterString(name="train_data_path", default_value=train_data_path)

    # The output path to the validation data.
    val_data_path = "s3://{0}/{1}/data/preprocessed/val/".format(bucket_name, prefix)
    val_data_path_param = ParameterString(name="val_data_path", default_value=val_data_path)

    # The training output path for the model.
    output_path = "s3://{0}/{1}/output/".format(bucket_name, prefix)
    output_path_param = ParameterString(name="output_path", default_value=output_path)

    # The output path to the featurizer model.
    model_path = "s3://{0}/{1}/output/sklearn/".format(bucket_name, prefix)
    model_path_param = ParameterString(name="model_path", default_value=model_path)

    inputs = [
        ProcessingInput(
            input_name="raw_data",
            source=raw_data_path_param,
            destination="/opt/ml/processing/input",
        )
    ]

    outputs = [
        ProcessingOutput(
            output_name="train_data",
            source="/opt/ml/processing/train",
            destination=train_data_path_param,
        ),
        ProcessingOutput(
            output_name="val_data", source="/opt/ml/processing/val", destination=val_data_path_param
        ),
        ProcessingOutput(
            output_name="model", source="/opt/ml/processing/model", destination=model_path_param
        ),
    ]

    base_dir = os.path.join(DATA_DIR, "sip")
    code_path = os.path.join(base_dir, "preprocessor.py")

    processing_step = ProcessingStep(
        name="Processing",
        code=code_path,
        processor=sklearn_processor,
        inputs=inputs,
        outputs=outputs,
        job_arguments=["--train-test-split-ratio", "0.2"],
    )

    entry_point = "training.py"
    source_dir = base_dir
    code_location = "s3://{0}/{1}/code".format(bucket_name, prefix)

    estimator = XGBoost(
        entry_point=entry_point,
        source_dir=source_dir,
        output_path=output_path_param,
        code_location=code_location,
        instance_type=instance_type,
        instance_count=instance_count,
        framework_version="0.90-2",
        sagemaker_session=sagemaker_session,
        py_version="py3",
        role=role,
    )

    training_step = TrainingStep(
        name="Training",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                    "train_data"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                    "val_data"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    code_location = "s3://{0}/{1}/code".format(bucket_name, prefix)
    source_dir = os.path.join(base_dir, "sklearn_source_dir")

    sklearn_model = SKLearnModel(
        name="sklearn-model",
        model_data=processing_step.properties.ProcessingOutputConfig.Outputs[
            "model"
        ].S3Output.S3Uri,
        entry_point="inference.py",
        source_dir=source_dir,
        code_location=code_location,
        role=role,
        sagemaker_session=sagemaker_session,
        framework_version="0.20.0",
        py_version="py3",
    )

    code_location = "s3://{0}/{1}/code".format(bucket_name, prefix)
    source_dir = os.path.join(base_dir, "xgboost_source_dir")

    xgboost_model = XGBoostModel(
        name="xgboost-model",
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        entry_point="inference.py",
        source_dir=source_dir,
        code_location=code_location,
        framework_version="0.90-2",
        py_version="py3",
        role=role,
        sagemaker_session=sagemaker_session,
    )

    pipeline_model = PipelineModel(
        [xgboost_model, sklearn_model], role, sagemaker_session=sagemaker_session
    )

    step_register = RegisterModel(
        name="AbaloneRegisterModel",
        model=pipeline_model,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name="windturbine",
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            raw_data_path_param,
            train_data_path_param,
            val_data_path_param,
            model_path_param,
            instance_type,
            instance_count,
            output_path_param,
        ],
        steps=[processing_step, training_step, step_register],
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}", create_arn
        )

        execution = pipeline.start(parameters={})
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )

        execution = pipeline.start()
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )
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
    kms_key = get_or_create_kms_key(sagemaker_session, role)
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
        output_kms_key=kms_key,
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
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        description="test-description",
        entry_point=entry_point,
        model_kms_key=kms_key,
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
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}", create_arn
        )

        execution = pipeline.start(parameters={})
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )

        execution = pipeline.start(parameters={"GoodEnoughInput": 0})
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_training_job_with_debugger_and_profiler(
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

    for _ in retries(
        max_retry_count=5,
        exception_message_prefix="Waiting for a successful execution of pipeline",
        seconds_to_sleep=10,
    ):
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

            assert len(execution_steps) == 1
            failure_reason = execution_steps[0].get("FailureReason", "")
            if failure_reason != "":
                logging.error(f"Pipeline execution failed with error: {failure_reason}.Retrying..")
                continue
            assert execution_steps[0]["StepName"] == "pytorch-train"
            assert execution_steps[0]["StepStatus"] == "Succeeded"

            training_job_arn = execution_steps[0]["Metadata"]["TrainingJob"]["Arn"]
            job_description = sagemaker_session.sagemaker_client.describe_training_job(
                TrainingJobName=training_job_arn.split("/")[1]
            )

            for index, rule in enumerate(rules):
                config = job_description["DebugRuleConfigurations"][index]
                assert config["RuleConfigurationName"] == rule.name
                assert config["RuleEvaluatorImage"] == rule.image_uri
                assert config["VolumeSizeInGB"] == 0
                assert (
                    config["RuleParameters"]["rule_to_invoke"]
                    == rule.rule_parameters["rule_to_invoke"]
                )
            assert job_description["DebugHookConfig"] == debugger_hook_config._to_request_dict()

            assert job_description["ProfilingStatus"] == "Enabled"
            assert job_description["ProfilerConfig"]["ProfilingIntervalInMilliseconds"] == 500
            break
        finally:
            try:
                pipeline.delete()
            except Exception:
                pass


def test_two_processing_job_depends_on(
    sagemaker_session,
    role,
    pipeline_name,
    region_name,
    cpu_instance_type,
):
    instance_count = ParameterInteger(name="InstanceCount", default_value=2)
    script_path = os.path.join(DATA_DIR, "dummy_script.py")

    pyspark_processor = PySparkProcessor(
        base_job_name="sm-spark",
        framework_version="2.4",
        role=role,
        instance_count=instance_count,
        instance_type=cpu_instance_type,
        max_runtime_in_seconds=1200,
        sagemaker_session=sagemaker_session,
    )

    spark_run_args = pyspark_processor.get_run_args(
        submit_app=script_path,
        arguments=[
            "--s3_input_bucket",
            sagemaker_session.default_bucket(),
            "--s3_input_key_prefix",
            "spark-input",
            "--s3_output_bucket",
            sagemaker_session.default_bucket(),
            "--s3_output_key_prefix",
            "spark-output",
        ],
    )

    step_pyspark_1 = ProcessingStep(
        name="pyspark-process-1",
        processor=pyspark_processor,
        inputs=spark_run_args.inputs,
        outputs=spark_run_args.outputs,
        job_arguments=spark_run_args.arguments,
        code=spark_run_args.code,
    )

    step_pyspark_2 = ProcessingStep(
        name="pyspark-process-2",
        depends_on=[step_pyspark_1],
        processor=pyspark_processor,
        inputs=spark_run_args.inputs,
        outputs=spark_run_args.outputs,
        job_arguments=spark_run_args.arguments,
        code=spark_run_args.code,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count],
        steps=[step_pyspark_1, step_pyspark_2],
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

        pipeline.parameters = [ParameterInteger(name="InstanceCount", default_value=1)]
        response = pipeline.update(role)
        update_arn = response["PipelineArn"]
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            update_arn,
        )

        execution = pipeline.start(parameters={})
        assert re.match(
            fr"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )

        response = execution.describe()
        assert response["PipelineArn"] == create_arn

        try:
            execution.wait(delay=60)
        except WaiterError:
            pass

        execution_steps = execution.list_steps()
        assert len(execution_steps) == 2
        time_stamp = {}
        for execution_step in execution_steps:
            name = execution_step["StepName"]
            if name == "pyspark-process-1":
                time_stamp[name] = execution_step["EndTime"]
            else:
                time_stamp[name] = execution_step["StartTime"]
        assert time_stamp["pyspark-process-1"] < time_stamp["pyspark-process-2"]
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_one_step_data_wrangler_processing_pipeline(sagemaker_session, role, pipeline_name):
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.4xlarge")

    recipe_file_path = os.path.join(DATA_DIR, "workflow", "dummy_recipe.flow")
    input_file_path = os.path.join(DATA_DIR, "workflow", "dummy_data.csv")

    output_name = "3f74973c-fd1e-4845-89f8-0dd400031be9.default"
    output_content_type = "CSV"
    output_config = {output_name: {"content_type": output_content_type}}
    job_argument = [f"--output-config '{json.dumps(output_config)}'"]

    inputs = [
        ProcessingInput(
            input_name="dummy_data.csv",
            source=input_file_path,
            destination="/opt/ml/processing/dummy_data.csv",
        )
    ]

    output_s3_uri = f"s3://{sagemaker_session.default_bucket()}/output"
    outputs = [
        ProcessingOutput(
            output_name=output_name,
            source="/opt/ml/processing/output",
            destination=output_s3_uri,
            s3_upload_mode="EndOfJob",
        )
    ]

    data_wrangler_processor = DataWranglerProcessor(
        role=role,
        data_wrangler_flow_source=recipe_file_path,
        instance_count=instance_count,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        max_runtime_in_seconds=86400,
    )

    data_wrangler_step = ProcessingStep(
        name="data-wrangler-step",
        processor=data_wrangler_processor,
        inputs=inputs,
        outputs=outputs,
        job_arguments=job_argument,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count, instance_type],
        steps=[data_wrangler_step],
        sagemaker_session=sagemaker_session,
    )

    definition = json.loads(pipeline.definition())
    expected_image_uri = image_uris.retrieve(
        "data-wrangler", region=sagemaker_session.boto_region_name
    )
    assert len(definition["Steps"]) == 1
    assert definition["Steps"][0]["Arguments"]["AppSpecification"]["ImageUri"] is not None
    assert definition["Steps"][0]["Arguments"]["AppSpecification"]["ImageUri"] == expected_image_uri

    assert definition["Steps"][0]["Arguments"]["ProcessingInputs"] is not None
    processing_inputs = definition["Steps"][0]["Arguments"]["ProcessingInputs"]
    assert len(processing_inputs) == 2
    for processing_input in processing_inputs:
        if processing_input["InputName"] == "flow":
            assert processing_input["S3Input"]["S3Uri"].endswith(".flow")
            assert processing_input["S3Input"]["LocalPath"] == "/opt/ml/processing/flow"
        elif processing_input["InputName"] == "dummy_data.csv":
            assert processing_input["S3Input"]["S3Uri"].endswith(".csv")
            assert processing_input["S3Input"]["LocalPath"] == "/opt/ml/processing/dummy_data.csv"
        else:
            raise AssertionError("Unknown input name")
    assert definition["Steps"][0]["Arguments"]["ProcessingOutputConfig"] is not None
    processing_outputs = definition["Steps"][0]["Arguments"]["ProcessingOutputConfig"]["Outputs"]
    assert len(processing_outputs) == 1
    assert processing_outputs[0]["OutputName"] == output_name
    assert processing_outputs[0]["S3Output"] is not None
    assert processing_outputs[0]["S3Output"]["LocalPath"] == "/opt/ml/processing/output"
    assert processing_outputs[0]["S3Output"]["S3Uri"] == output_s3_uri

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]

        execution = pipeline.start()
        response = execution.describe()
        assert response["PipelineArn"] == create_arn

        try:
            execution.wait(delay=60, max_attempts=10)
        except WaiterError:
            pass

        execution_steps = execution.list_steps()
        assert len(execution_steps) == 1
        assert execution_steps[0]["StepName"] == "data-wrangler-step"
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_one_step_ingestion_pipeline(
    sagemaker_session, feature_store_session, feature_definitions, role, pipeline_name
):
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.4xlarge")

    input_name = "features.csv"
    input_file_path = os.path.join(DATA_DIR, "workflow", "features.csv")
    input_data_uri = os.path.join(
        "s3://", sagemaker_session.default_bucket(), "py-sdk-ingestion-test-input/features.csv"
    )

    with open(input_file_path, "r") as data:
        body = data.read()
        S3Uploader.upload_string_as_file_body(
            body=body, desired_s3_uri=input_data_uri, sagemaker_session=sagemaker_session
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
        json.dump(ingestion_only_flow, open(temp_flow_path, "w"))

        data_wrangler_processor = DataWranglerProcessor(
            role=role,
            data_wrangler_flow_source=temp_flow_path,
            instance_count=instance_count,
            instance_type=instance_type,
            sagemaker_session=sagemaker_session,
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
            sagemaker_session=sagemaker_session,
        )

        try:
            response = pipeline.create(role)
            create_arn = response["PipelineArn"]

            offline_store_s3_uri = os.path.join(
                "s3://", sagemaker_session.default_bucket(), feature_group_name
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

            try:
                execution.wait(delay=60, max_attempts=10)
            except WaiterError:
                pass

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
