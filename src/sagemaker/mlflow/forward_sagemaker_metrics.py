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

"""This module contains code related to forwarding SageMaker TrainingJob Metrics to MLflow."""

from __future__ import absolute_import

import os
import platform
import re
from typing import Set, Tuple, List, Dict, Generator
import boto3
import mlflow
from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunTag

from packaging import version


def encode(name: str, existing_names: Set[str]) -> str:
    """Encode a string to comply with MLflow naming restrictions and ensure uniqueness.

    Args:
        name (str): The original string to be encoded.
        existing_names (Set[str]): Set of existing encoded names to avoid collisions.

    Returns:
        str: The encoded string if changes were necessary, otherwise the original string.
    """

    def encode_char(match):
        return f"_{ord(match.group(0)):02x}_"

    # Check if we're on Mac/Unix and using MLflow 2.16.0 or greater
    is_unix = platform.system() != "Windows"
    mlflow_version = version.parse(mlflow.__version__)
    allow_colon = is_unix and mlflow_version >= version.parse("2.16.0")

    if allow_colon:
        pattern = r"[^\w\-./:\s]"
    else:
        pattern = r"[^\w\-./\s]"

    encoded = re.sub(pattern, encode_char, name)
    base_name = encoded[:240]  # Leave room for potential suffix to accommodate duplicates

    if base_name in existing_names:
        suffix = 1
        # Edge case where even with suffix space there is a collision
        # we will override one of the keys.
        while f"{base_name}_{suffix}" in existing_names:
            suffix += 1
        encoded = f"{base_name}_{suffix}"

    # Max length is 250 for mlflow metric/params
    encoded = encoded[:250]

    existing_names.add(encoded)
    return encoded


def decode(encoded_metric_name: str) -> str:
    """Decodes an encoded metric name by replacing hexadecimal representations with ASCII

    This function reverses the encoding process by converting hexadecimal codes
    back to their original characters. It looks for patterns of the form "_XX_"
    where XX is a two-digit hexadecimal code, and replaces them with the
    corresponding ASCII character.

    Args:
        encoded_metric_name (str): The encoded metric name to be decoded.

    Returns:
        str: The decoded metric name with hexadecimal codes replaced by their
             corresponding characters.

    Example:
        >>> decode("loss_3a_val")
        "loss:val"
    """

    def replace_code(match):
        code = match.group(1)
        return chr(int(code, 16))

    # Replace encoded characters
    decoded = re.sub(r"_([0-9a-f]{2})_", replace_code, encoded_metric_name)

    return decoded


def get_training_job_details(job_arn: str) -> dict:
    """Retrieve details of a SageMaker training job.

    Args:
        job_arn (str): The ARN of the SageMaker training job.

    Returns:
        dict: A dictionary containing the details of the training job.

    Raises:
        boto3.exceptions.Boto3Error: If there's an issue with the AWS API call.
    """
    sagemaker_client = boto3.client("sagemaker")
    job_name = job_arn.split("/")[-1]
    return sagemaker_client.describe_training_job(TrainingJobName=job_name)


def create_metric_queries(job_arn: str, metric_definitions: list) -> list:
    """Create metric queries for SageMaker metrics.

    Args:
        job_arn (str): The ARN of the SageMaker training job.
        metric_definitions (list): List of metric definitions from the training job.

    Returns:
        list: A list of dictionaries, each representing a metric query.
    """
    metric_queries = []
    for metric in metric_definitions:
        query = {
            "MetricName": metric["Name"],
            "XAxisType": "Timestamp",
            "MetricStat": "Avg",
            "Period": "OneMinute",
            "ResourceArn": job_arn,
        }
        metric_queries.append(query)
    return metric_queries


def get_metric_data(metric_queries: list) -> dict:
    """Retrieve metric data from SageMaker.

    Args:
        metric_queries (list): A list of metric queries.

    Returns:
        dict: A dictionary containing the metric data results.

    Raises:
        boto3.exceptions.Boto3Error: If there's an issue with the AWS API call.
    """
    sagemaker_metrics_client = boto3.client("sagemaker-metrics")
    metric_data = sagemaker_metrics_client.batch_get_metrics(MetricQueries=metric_queries)
    return metric_data


def prepare_mlflow_metrics(
    metric_queries: list, metric_results: list
) -> Tuple[List[Metric], Dict[str, str]]:
    """Prepare metrics for MLflow logging, encoding metric names if necessary.

    Args:
        metric_queries (list): The original metric queries sent to SageMaker.
        metric_results (list): The metric results from SageMaker batch_get_metrics.

    Returns:
        Tuple[List[Metric], Dict[str, str]]:
            - A list of Metric objects with encoded names (if necessary)
            - A mapping of encoded to original names for metrics (only for encoded metrics)
    """
    mlflow_metrics = []
    metric_name_mapping = {}
    existing_names = set()

    for query, result in zip(metric_queries, metric_results):
        if result["Status"] == "Complete":
            metric_name = query["MetricName"]
            encoded_name = encode(metric_name, existing_names)
            metric_name_mapping[encoded_name] = metric_name

            for step, (timestamp, value) in enumerate(
                zip(result["XAxisValues"], result["MetricValues"])
            ):
                metric = Metric(key=encoded_name, value=value, timestamp=timestamp, step=step)
                mlflow_metrics.append(metric)

    return mlflow_metrics, metric_name_mapping


def prepare_mlflow_params(hyperparameters: Dict[str, str]) -> Tuple[List[Param], Dict[str, str]]:
    """Prepare hyperparameters for MLflow logging, encoding parameter names if necessary.

    Args:
        hyperparameters (Dict[str, str]): The hyperparameters from the SageMaker job.

    Returns:
        Tuple[List[Param], Dict[str, str]]:
            - A list of Param objects with encoded names (if necessary)
            - A mapping of encoded to original names for
                hyperparameters (only for encoded parameters)
    """
    mlflow_params = []
    param_name_mapping = {}
    existing_names = set()

    for key, value in hyperparameters.items():
        encoded_key = encode(key, existing_names)
        param_name_mapping[encoded_key] = key
        mlflow_params.append(Param(encoded_key, str(value)))

    return mlflow_params, param_name_mapping


def batch_items(items: list, batch_size: int) -> Generator:
    """Yield successive batch_size chunks from items.

    Args:
        items (list): The list of items to be batched.
        batch_size (int): The size of each batch.

    Yields:
        list: A batch of items.
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def log_to_mlflow(metrics: list, params: list, tags: dict) -> None:
    """Log metrics, parameters, and tags to MLflow.

    Args:
        metrics (list): List of metrics to log.
        params (list): List of parameters to log.
        tags (dict): Dictionary of tags to set.

    Raises:
        mlflow.exceptions.MlflowException: If there's an issue with MLflow logging.
    """
    client = MlflowClient()

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    if experiment_name is None or experiment_name.strip() == "":
        experiment_name = "Default"
        print("MLFLOW_EXPERIMENT_NAME not set. Using Default")

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    run = client.create_run(experiment_id)

    for metric_batch in batch_items(metrics, 1000):
        client.log_batch(
            run.info.run_id,
            metrics=metric_batch,
        )
    for param_batch in batch_items(params, 1000):
        client.log_batch(run.info.run_id, params=param_batch)

    tag_items = list(tags.items())
    for tag_batch in batch_items(tag_items, 1000):
        tag_objects = [RunTag(key, str(value)) for key, value in tag_batch]
        client.log_batch(run.info.run_id, tags=tag_objects)
    client.set_terminated(run.info.run_id)


def log_sagemaker_job_to_mlflow(training_job_arn: str) -> None:
    """Retrieve SageMaker metrics and hyperparameters and log them to MLflow.

    Args:
        training_job_arn (str): The ARN of the SageMaker training job.

    Raises:
        Exception: If there's any error during the process.
    """
    # Get training job details
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    job_details = get_training_job_details(training_job_arn)

    # Extract hyperparameters and metric definitions
    hyperparameters = job_details["HyperParameters"]
    metric_definitions = job_details["AlgorithmSpecification"]["MetricDefinitions"]

    # Create and get metric queries
    metric_queries = create_metric_queries(job_details["TrainingJobArn"], metric_definitions)
    metric_data = get_metric_data(metric_queries)

    # Create a mapping of encoded to original metric names
    # Prepare data for MLflow
    mlflow_metrics, metric_name_mapping = prepare_mlflow_metrics(
        metric_queries, metric_data["MetricQueryResults"]
    )

    # Create a mapping of encoded to original hyperparameter names
    # Prepare data for MLflow
    mlflow_params, param_name_mapping = prepare_mlflow_params(hyperparameters)

    mlflow_tags = {
        "training_job_arn": training_job_arn,
        "metric_name_mapping": str(metric_name_mapping),
        "param_name_mapping": str(param_name_mapping),
    }

    # Log to MLflow
    log_to_mlflow(mlflow_metrics, mlflow_params, mlflow_tags)
    print(f"Logged {len(mlflow_metrics)} metric datapoints to MLflow")
    print(f"Logged {len(mlflow_params)} hyperparameters to MLflow")
