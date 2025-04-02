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

import re
import os
import subprocess
import time
import urllib3

from datetime import datetime
from urllib3.util.retry import Retry

import boto3
import pytest
from sagemaker.s3 import S3Downloader, S3Uploader
from sagemaker.spark.processing import PySparkProcessor, SparkJarProcessor
from tests.integ import DATA_DIR
from unittest.case import TestCase

HISTORY_SERVER_ENDPOINT = "http://0.0.0.0/proxy/15050"
JAVA_FILE_PATH = os.path.join("com", "amazonaws", "sagemaker", "spark", "test")
JAVA_VERSION_PATTERN = r"(\d+\.\d+).*"
SPARK_APPLICATION_URL_SUFFIX = "/history/application_1594922484246_0001/1/jobs/"
SPARK_PATH = os.path.join(DATA_DIR, "spark")


@pytest.fixture(scope="module", autouse=True)
def build_jar():
    jar_file_path = os.path.join(SPARK_PATH, "code", "java", "hello-java-spark")
    # compile java file
    java_version = subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT).decode(
        "utf-8"
    )
    java_version = re.search(JAVA_VERSION_PATTERN, java_version).groups()[0]

    if float(java_version) > 1.8:
        subprocess.run(
            [
                "javac",
                "--release",
                "8",
                os.path.join(jar_file_path, JAVA_FILE_PATH, "HelloJavaSparkApp.java"),
            ]
        )
    else:
        subprocess.run(
            ["javac", os.path.join(jar_file_path, JAVA_FILE_PATH, "HelloJavaSparkApp.java")]
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


@pytest.fixture(scope="module")
def spark_py_processor(sagemaker_session, cpu_instance_type):
    spark_py_processor = PySparkProcessor(
        role="SageMakerRole",
        instance_count=2,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        framework_version="2.4",
    )

    return spark_py_processor


@pytest.fixture(scope="module")
def spark_v3_py_processor(sagemaker_session, cpu_instance_type):
    spark_py_processor = PySparkProcessor(
        role="SageMakerRole",
        instance_count=2,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        framework_version="3.1",
    )

    return spark_py_processor


@pytest.fixture(scope="module")
def spark_jar_processor(sagemaker_session, cpu_instance_type):
    spark_jar_processor = SparkJarProcessor(
        role="SageMakerRole",
        instance_count=2,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        framework_version="2.4",
    )

    return spark_jar_processor


@pytest.fixture(scope="module")
def spark_v3_jar_processor(sagemaker_session, cpu_instance_type):
    spark_jar_processor = SparkJarProcessor(
        role="SageMakerRole",
        instance_count=2,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        framework_version="3.1",
    )

    return spark_jar_processor


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


def test_sagemaker_pyspark_v3(
    spark_v3_py_processor, spark_v3_jar_processor, sagemaker_session, configuration
):
    test_sagemaker_pyspark_multinode(spark_v3_py_processor, sagemaker_session, configuration)
    test_sagemaker_java_jar_multinode(spark_v3_jar_processor, sagemaker_session, configuration)


def test_sagemaker_pyspark_multinode(spark_py_processor, sagemaker_session, configuration):
    """Test that basic multinode case works on 32KB of data"""
    bucket = spark_py_processor.sagemaker_session.default_bucket()
    timestamp = datetime.now().isoformat()
    output_data_uri = f"s3://{bucket}/spark/output/sales/{timestamp}"
    spark_event_logs_key_prefix = f"spark/spark-events/{timestamp}"
    spark_event_logs_s3_uri = f"s3://{bucket}/{spark_event_logs_key_prefix}"

    with open(os.path.join(SPARK_PATH, "files", "data.jsonl")) as data:
        body = data.read()
        input_data_uri = f"s3://{bucket}/spark/input/data.jsonl"
        S3Uploader.upload_string_as_file_body(
            body=body, desired_s3_uri=input_data_uri, sagemaker_session=sagemaker_session
        )

    spark_py_processor.run(
        submit_app=os.path.join(
            SPARK_PATH, "code", "python", "hello_py_spark", "hello_py_spark_app.py"
        ),
        submit_py_files=[
            os.path.join(SPARK_PATH, "code", "python", "hello_py_spark", "hello_py_spark_udfs.py")
        ],
        arguments=["--input", input_data_uri, "--output", output_data_uri],
        configuration=configuration,
        spark_event_logs_s3_uri=spark_event_logs_s3_uri,
        wait=False,
    )
    processing_job = spark_py_processor.latest_job

    s3_client = boto3.client(
        "s3", region_name=spark_py_processor.sagemaker_session.boto_region_name
    )

    file_size = 0
    latest_file_size = None
    updated_times_count = 0
    time_out = time.time() + 900
    while not processing_job_not_fail_or_complete(
        sagemaker_session.sagemaker_client, processing_job.job_name
    ):
        response = s3_client.list_objects(Bucket=bucket, Prefix=spark_event_logs_key_prefix)
        if "Contents" in response:
            # somehow when call list_objects the first file size is always 0, this for loop
            # is to skip that.
            for event_log_file in response["Contents"]:
                if event_log_file["Size"] != 0:
                    latest_file_size = event_log_file["Size"]

        # update the file size if it increased
        if latest_file_size and latest_file_size > file_size:
            updated_times_count += 1
            file_size = latest_file_size

        if time.time() > time_out:
            raise RuntimeError("Timeout")

        time.sleep(20)

    # verify that spark event logs are periodically written to s3
    assert file_size != 0

    output_contents = S3Downloader.list(output_data_uri, sagemaker_session=sagemaker_session)
    assert len(output_contents) != 0


def test_sagemaker_java_jar_multinode(spark_jar_processor, sagemaker_session, configuration):
    """Test SparkJarProcessor using Java application jar"""
    bucket = spark_jar_processor.sagemaker_session.default_bucket()
    with open(os.path.join(SPARK_PATH, "files", "data.jsonl")) as data:
        body = data.read()
        input_data_uri = f"s3://{bucket}/spark/input/data.jsonl"
        S3Uploader.upload_string_as_file_body(
            body=body, desired_s3_uri=input_data_uri, sagemaker_session=sagemaker_session
        )
    output_data_uri = f"s3://{bucket}/spark/output/sales/{datetime.now().isoformat()}"

    java_project_dir = os.path.join(SPARK_PATH, "code", "java", "hello-java-spark")
    spark_jar_processor.run(
        submit_app=f"{java_project_dir}/hello-spark-java.jar",
        submit_class="com.amazonaws.sagemaker.spark.test.HelloJavaSparkApp",
        arguments=["--input", input_data_uri, "--output", output_data_uri],
        configuration=configuration,
    )
    processing_job = spark_jar_processor.latest_job

    waiter = sagemaker_session.sagemaker_client.get_waiter("processing_job_completed_or_stopped")
    waiter.wait(
        ProcessingJobName=processing_job.job_name,
        # poll every 15 seconds. timeout after 15 minutes.
        WaiterConfig={"Delay": 15, "MaxAttempts": 60},
    )

    describe_response = sagemaker_session.sagemaker_client.describe_processing_job(
        ProcessingJobName=processing_job.job_name
    )
    assert describe_response["ProcessingJobStatus"] == "Completed"


def processing_job_not_fail_or_complete(sagemaker_client, job_name):
    response = sagemaker_client.describe_processing_job(ProcessingJobName=job_name)

    if not response or "ProcessingJobStatus" not in response:
        raise ValueError("Response is none or does not have ProcessingJobStatus")
    status = response["ProcessingJobStatus"]
    return status == "Failed" or status == "Completed" or status == "Stopped"


def test_integ_history_server(spark_py_processor, sagemaker_session):
    bucket = spark_py_processor.sagemaker_session.default_bucket()
    spark_event_logs_key_prefix = "spark/spark-history-fs"
    spark_event_logs_s3_uri = f"s3://{bucket}/{spark_event_logs_key_prefix}"

    with open(os.path.join(SPARK_PATH, "files", "sample_spark_event_logs")) as data:
        body = data.read()
        S3Uploader.upload_string_as_file_body(
            body=body,
            desired_s3_uri=spark_event_logs_s3_uri + "/sample_spark_event_logs",
            sagemaker_session=sagemaker_session,
        )

    # sleep 3 seconds to avoid s3 eventual consistency issue
    time.sleep(3)
    spark_py_processor.start_history_server(spark_event_logs_s3_uri=spark_event_logs_s3_uri)

    try:
        response = _request_with_retry(HISTORY_SERVER_ENDPOINT)
        assert response.status == 200
    finally:
        spark_py_processor.terminate_history_server()


def test_integ_history_server_with_expected_failure(spark_py_processor):
    with TestCase.assertLogs("sagemaker", level="ERROR") as cm:
        spark_py_processor.start_history_server(spark_event_logs_s3_uri="invalids3uri")
    response = _request_with_retry(HISTORY_SERVER_ENDPOINT, max_retries=5)
    assert response is None
    assert (
        "History server failed to start. Please run 'docker logs history_server' to see logs"
        in cm.output[0]
    )


def _request_with_retry(url, max_retries=10):
    http = urllib3.PoolManager(
        retries=Retry(
            max_retries,
            redirect=max_retries,
            status=max_retries,
            status_forcelist=[502, 404],
            backoff_factor=0.2,
        )
    )
    try:
        return http.request("GET", url)
    except Exception:  # pylint: disable=W0703
        return None
