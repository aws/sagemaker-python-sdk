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
from datetime import datetime
from pathlib import Path

import pytest
from botocore.exceptions import WaiterError
from sagemaker.workflow.utilities import hash_files_or_dirs, hash_object

from sagemaker import image_uris, get_execution_role, utils
from sagemaker.dataset_definition import DatasetDefinition, AthenaDatasetDefinition
from sagemaker.processing import ProcessingInput, ProcessingOutput, FrameworkProcessor
from sagemaker.s3 import S3Uploader, S3Downloader
from sagemaker.sklearn import SKLearnProcessor, SKLearn
from sagemaker.tensorflow import TensorFlow
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    ProcessingStep,
    CacheConfig,
)
from sagemaker.spark.processing import PySparkProcessor, SparkJarProcessor
from sagemaker.wrangler.processing import DataWranglerProcessor
from tests.integ import DATA_DIR


@pytest.fixture
def role(sagemaker_session):
    return get_execution_role(sagemaker_session)


@pytest.fixture
def pipeline_name():
    return utils.unique_name_from_base("my-pipeline-processing")


@pytest.fixture
def region_name(sagemaker_session):
    return sagemaker_session.boto_session.region_name


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
    java_file_path = os.path.join("com", "amazonaws", "..", "spark", "test")
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
            [
                "javac",
                os.path.join(jar_file_path, java_file_path, "HelloJavaSparkApp.java"),
            ]
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
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

        pipeline.parameters = [ParameterInteger(name="InstanceCount", default_value=1)]
        execution = pipeline.start(parameters={})
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
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
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

        pipeline.parameters = [ParameterInteger(name="InstanceCount", default_value=1)]
        response = pipeline.update(role)
        update_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            update_arn,
        )

        execution = pipeline.start(parameters={})
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
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


def test_multi_step_framework_processing_pipeline_same_source_dir(
    pipeline_session, role, pipeline_name
):
    default_bucket = pipeline_session.default_bucket()
    cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")

    SOURCE_DIR = "/pipeline/test_source_dir"

    framework_processor_tf = FrameworkProcessor(
        role=role,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        estimator_cls=TensorFlow,
        framework_version="2.9",
        py_version="py39",
        sagemaker_session=pipeline_session,
    )

    framework_processor_sk = FrameworkProcessor(
        framework_version="1.0-1",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name="my-job",
        role=role,
        estimator_cls=SKLearn,
        sagemaker_session=pipeline_session,
    )

    step_1 = ProcessingStep(
        name="Step-1",
        step_args=framework_processor_tf.run(
            code="script_1.py",
            source_dir=DATA_DIR + SOURCE_DIR,
            outputs=[ProcessingOutput(output_name="test", source="/opt/ml/processing/test")],
        ),
        cache_config=cache_config,
    )

    step_2 = ProcessingStep(
        name="Step-2",
        step_args=framework_processor_sk.run(
            code="script_2.py",
            source_dir=DATA_DIR + SOURCE_DIR,
            inputs=[
                ProcessingInput(
                    source=step_1.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                    destination="/opt/ml/processing/test",
                ),
            ],
        ),
        cache_config=cache_config,
    )

    pipeline = Pipeline(
        name=pipeline_name, steps=[step_1, step_2], sagemaker_session=pipeline_session
    )
    try:
        pipeline.create(role)
        definition = json.loads(pipeline.definition())

        source_dir_1_s3_uri, entry_point_1 = _verify_code_artifacts_of_framework_processing_step(
            pipeline_session,
            framework_processor_tf,
            default_bucket,
            pipeline_name,
            definition["Steps"][0],
            SOURCE_DIR,
            "script_1.py",
        )
        source_dir_2_s3_uri, entry_point_2 = _verify_code_artifacts_of_framework_processing_step(
            pipeline_session,
            framework_processor_sk,
            default_bucket,
            pipeline_name,
            definition["Steps"][1],
            SOURCE_DIR,
            "script_2.py",
        )

        # the same local source_dirs should have the same s3 paths
        assert source_dir_1_s3_uri == source_dir_2_s3_uri

        # verify different entry_point paths
        assert entry_point_1 != entry_point_2

        execution = pipeline.start(parameters={})
        try:
            execution.wait(delay=540, max_attempts=3)
        except WaiterError:
            pass

        execution_steps = execution.list_steps()
        assert len(execution_steps) == 2
        for step in execution_steps:
            assert step["StepStatus"] == "Succeeded"

    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_multi_step_framework_processing_pipeline_different_source_dir(
    pipeline_session, role, pipeline_name
):
    default_bucket = pipeline_session.default_bucket()
    cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")

    SOURCE_DIR_1 = "/pipeline/test_source_dir"
    SOURCE_DIR_2 = "/pipeline/test_source_dir_2"

    framework_processor_tf = FrameworkProcessor(
        role=role,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        estimator_cls=TensorFlow,
        framework_version="2.9",
        py_version="py39",
        sagemaker_session=pipeline_session,
    )

    step_1 = ProcessingStep(
        name="Step-1",
        step_args=framework_processor_tf.run(
            code="script_1.py",
            source_dir=DATA_DIR + SOURCE_DIR_1,
            outputs=[ProcessingOutput(output_name="test", source="/opt/ml/processing/test")],
        ),
        cache_config=cache_config,
    )

    step_2 = ProcessingStep(
        name="Step-2",
        step_args=framework_processor_tf.run(
            code="script_2.py",
            source_dir=DATA_DIR + SOURCE_DIR_2,
            inputs=[
                ProcessingInput(
                    source=step_1.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                    destination="/opt/ml/processing/test",
                ),
            ],
        ),
        cache_config=cache_config,
    )

    pipeline = Pipeline(
        name=pipeline_name, steps=[step_1, step_2], sagemaker_session=pipeline_session
    )
    try:
        pipeline.create(role)
        definition = json.loads(pipeline.definition())

        source_dir_1_s3_uri, entry_point_1 = _verify_code_artifacts_of_framework_processing_step(
            pipeline_session,
            framework_processor_tf,
            default_bucket,
            pipeline_name,
            definition["Steps"][0],
            SOURCE_DIR_1,
            "script_1.py",
        )
        source_dir_2_s3_uri, entry_point_2 = _verify_code_artifacts_of_framework_processing_step(
            pipeline_session,
            framework_processor_tf,
            default_bucket,
            pipeline_name,
            definition["Steps"][1],
            SOURCE_DIR_2,
            "script_2.py",
        )

        # different local source_dirs should have different s3 paths
        assert source_dir_1_s3_uri != source_dir_2_s3_uri

        # verify different entry_point paths
        assert entry_point_1 != entry_point_2

        execution = pipeline.start(parameters={})
        try:
            execution.wait(delay=540, max_attempts=3)
        except WaiterError:
            pass

        execution_steps = execution.list_steps()
        assert len(execution_steps) == 2
        for step in execution_steps:
            assert step["StepStatus"] == "Succeeded"

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
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

        pipeline.parameters = [ParameterInteger(name="InstanceCount", default_value=1)]
        response = pipeline.update(role)
        update_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            update_arn,
        )

        execution = pipeline.start(parameters={})
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
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
    sagemaker_session,
    role,
    cpu_instance_type,
    pipeline_name,
    region_name,
    configuration,
    build_jar,
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
            body=body,
            desired_s3_uri=input_data_uri,
            sagemaker_session=sagemaker_session,
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
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

        pipeline.parameters = [ParameterInteger(name="InstanceCount", default_value=1)]
        response = pipeline.update(role)
        update_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            update_arn,
        )

        execution = pipeline.start(parameters={})
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
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
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

        pipeline.parameters = [ParameterInteger(name="InstanceCount", default_value=1)]
        response = pipeline.update(role)
        update_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            update_arn,
        )

        execution = pipeline.start(parameters={})
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
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


def _verify_code_artifacts_of_framework_processing_step(
    pipeline_session, processor, bucket, pipeline_name, step_definition, source_dir, entry_point
):

    source_dir_s3_uri = (
        f"s3://{bucket}/{pipeline_name}" f"/code/{hash_files_or_dirs([f'{DATA_DIR}/{source_dir}'])}"
    )

    # verify runproc.sh prefix is different from code artifact prefix
    runprocs = []
    for input_obj in step_definition["Arguments"]["ProcessingInputs"]:
        if input_obj["InputName"] == "entrypoint":
            s3_uri = input_obj["S3Input"]["S3Uri"]
            runprocs.append(s3_uri)

            assert Path(s3_uri).parent != source_dir_s3_uri

    # verify only one entrypoint generated per step
    assert len(runprocs) == 1

    expected_source_dir_tar = (
        f"{pipeline_name}"
        f"/code/{hash_files_or_dirs([DATA_DIR + '/pipeline/test_source_dir'])}/sourcedir.tar.gz"
    )

    step_script = processor._generate_framework_script(entry_point)
    expected_step_artifact = f"{pipeline_name}/code/{hash_object(step_script)}/runproc.sh"

    expected_prefix = f"{pipeline_name}/code"
    s3_code_objects = pipeline_session.list_s3_files(bucket=bucket, key_prefix=expected_prefix)

    # verify all distinct artifacts were uploaded
    assert expected_source_dir_tar in s3_code_objects
    assert expected_step_artifact in s3_code_objects

    # verify runprocs contain the correct commands
    step_runproc = S3Downloader.read_file(
        f"s3://{bucket}/{expected_step_artifact}", pipeline_session
    )
    assert f"python {entry_point}" in step_runproc
    return source_dir, expected_step_artifact
