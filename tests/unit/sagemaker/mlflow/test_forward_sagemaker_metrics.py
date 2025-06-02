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
from unittest.mock import patch, MagicMock, Mock
import json
import pytest
from mlflow.entities import Metric, Param
import requests


from sagemaker.mlflow.forward_sagemaker_metrics import (
    encode,
    log_sagemaker_job_to_mlflow,
    decode,
    prepare_mlflow_metrics,
    prepare_mlflow_params,
    batch_items,
    create_metric_queries,
    get_metric_data,
    log_to_mlflow,
    get_training_job_details,
)


@pytest.fixture
def mock_boto3_client():
    with patch("boto3.client") as mock_client:
        yield mock_client


@pytest.fixture
def mock_mlflow_client():
    with patch("mlflow.MlflowClient") as mock_client:
        yield mock_client


def test_encode():
    existing_names = set()
    assert encode("test-name", existing_names) == "test-name"
    assert encode("test:name", existing_names) == "test:name"
    assert encode("test-name", existing_names) == "test-name_1"


def test_encode_colon_allowed():
    # Test case where colon is allowed (Unix-like system and MLflow >= 2.16.0)
    with patch("platform.system") as mock_system, patch("mlflow.__version__", new="2.16.0"):

        mock_system.return_value = "Darwin"  # MacOS
        existing_names = set()

        assert encode("test:name", existing_names) == "test:name"
        assert encode("test/name", existing_names) == "test/name"
        assert encode("test name", existing_names) == "test name"
        assert encode("test@name", existing_names) == "test_40_name"

        # Test name longer than 250 characters
        long_name = "a" * 250
        encoded_long_name = encode(long_name, existing_names)
        assert len(encoded_long_name) == 250
        assert encoded_long_name == "a" * 250

        # Test suffix addition for duplicate names
        assert encode("duplicate", existing_names) == "duplicate"
        assert encode("duplicate", existing_names) == "duplicate_1"
        assert encode("duplicate", existing_names) == "duplicate_2"


def test_decode():
    assert decode("test_3a_name") == "test:name"
    assert decode("normal_name") == "normal_name"


def test_get_training_job_details(mock_boto3_client):
    mock_sagemaker = MagicMock()
    mock_boto3_client.return_value = mock_sagemaker
    mock_sagemaker.describe_training_job.return_value = {"JobName": "test-job"}

    result = get_training_job_details(
        "arn:aws:sagemaker:us-west-2:123456789012:training-job/test-job"
    )
    assert result == {"JobName": "test-job"}
    mock_sagemaker.describe_training_job.assert_called_once_with(TrainingJobName="test-job")


def test_create_metric_queries():
    job_arn = "arn:aws:sagemaker:us-west-2:123456789012:training-job/test-job"
    metric_definitions = [{"Name": "loss"}, {"Name": "accuracy"}]
    result = create_metric_queries(job_arn, metric_definitions)
    assert len(result) == 2
    assert result[0]["MetricName"] == "loss"
    assert result[1]["MetricName"] == "accuracy"


def test_get_metric_data(mock_boto3_client):
    mock_metrics = MagicMock()
    mock_boto3_client.return_value = mock_metrics
    mock_metrics.batch_get_metrics.return_value = {"MetricResults": []}

    metric_queries = [{"MetricName": "loss"}]
    result = get_metric_data(metric_queries)
    assert result == {"MetricResults": []}
    mock_metrics.batch_get_metrics.assert_called_once_with(MetricQueries=metric_queries)


def test_prepare_mlflow_metrics():
    metric_queries = [{"MetricName": "loss"}, {"MetricName": "accuracy!"}]
    metric_results = [
        {"Status": "Complete", "XAxisValues": [1, 2], "MetricValues": [0.1, 0.2]},
        {"Status": "Complete", "XAxisValues": [1, 2], "MetricValues": [0.8, 0.9]},
    ]
    expected_encoded = {"loss": "loss", "accuracy_21_": "accuracy!"}

    metrics, mapping = prepare_mlflow_metrics(metric_queries, metric_results)

    assert len(metrics) == sum(len(result["MetricValues"]) for result in metric_results)

    expected_metrics = [
        ("loss", 0.1, 1, 0),
        ("loss", 0.2, 2, 1),
        ("accuracy_21_", 0.8, 1, 0),
        ("accuracy_21_", 0.9, 2, 1),
    ]

    for metric, (exp_key, exp_value, exp_timestamp, exp_step) in zip(metrics, expected_metrics):
        assert metric.key == exp_key
        assert metric.value == exp_value
        assert metric.timestamp == exp_timestamp
        assert metric.step == exp_step

    assert mapping == {v: k for v, k in expected_encoded.items()}


def test_prepare_mlflow_params():
    hyperparameters = {"learning_rate": "0.01", "batch_!size": "32"}
    expected_encoded = {"learning_rate": "learning_rate", "batch__21_size": "batch_!size"}

    params, mapping = prepare_mlflow_params(hyperparameters)

    assert len(params) == len(hyperparameters)

    for param in params:
        assert param.key in expected_encoded
        assert param.value == hyperparameters[mapping[param.key]]

    assert mapping == {v: k for v, k in expected_encoded.items()}


def test_batch_items():
    items = [1, 2, 3, 4, 5]
    batches = list(batch_items(items, 2))
    assert batches == [[1, 2], [3, 4], [5]]


@patch("os.getenv")
@patch("requests.Session.request")
def test_log_to_mlflow(mock_request, mock_getenv):
    # Set up return values for os.getenv calls
    def getenv_side_effect(arg, default=None):
        values = {
            "MLFLOW_TRACKING_URI": "https://test.sagemaker.aws",
            "MLFLOW_REGISTRY_URI": "https://registry.uri",
            "MLFLOW_EXPERIMENT_NAME": "test_experiment",
            "MLFLOW_ALLOW_HTTP_REDIRECTS": "true",
        }
        return values.get(arg, default)

    mock_getenv.side_effect = getenv_side_effect

    # Mock the HTTP requests
    mock_responses = {
        "https://test.sagemaker.aws/api/2.0/mlflow/experiments/get-by-name": Mock(
            spec=requests.Response
        ),
        "https://test.sagemaker.aws/api/2.0/mlflow/runs/create": Mock(spec=requests.Response),
        "https://test.sagemaker.aws/api/2.0/mlflow/runs/update": Mock(spec=requests.Response),
        "https://test.sagemaker.aws/api/2.0/mlflow/runs/log-batch": [
            Mock(spec=requests.Response),
            Mock(spec=requests.Response),
            Mock(spec=requests.Response),
        ],
        "https://test.sagemaker.aws/api/2.0/mlflow/runs/terminate": Mock(spec=requests.Response),
    }

    mock_responses[
        "https://test.sagemaker.aws/api/2.0/mlflow/experiments/get-by-name"
    ].status_code = 200
    mock_responses["https://test.sagemaker.aws/api/2.0/mlflow/experiments/get-by-name"].text = (
        json.dumps(
            {
                "experiment_id": "existing_experiment_id",
                "name": "test_experiment",
                "artifact_location": "some/path",
                "lifecycle_stage": "active",
                "tags": {},
            }
        )
    )

    mock_responses["https://test.sagemaker.aws/api/2.0/mlflow/runs/create"].status_code = 200
    mock_responses["https://test.sagemaker.aws/api/2.0/mlflow/runs/create"].text = json.dumps(
        {"run_id": "test_run_id"}
    )

    mock_responses["https://test.sagemaker.aws/api/2.0/mlflow/runs/update"].status_code = 200
    mock_responses["https://test.sagemaker.aws/api/2.0/mlflow/runs/update"].text = json.dumps(
        {"run_id": "test_run_id"}
    )

    for mock_response in mock_responses["https://test.sagemaker.aws/api/2.0/mlflow/runs/log-batch"]:
        mock_response.status_code = 200
        mock_response.text = json.dumps({})

    mock_responses["https://test.sagemaker.aws/api/2.0/mlflow/runs/terminate"].status_code = 200
    mock_responses["https://test.sagemaker.aws/api/2.0/mlflow/runs/terminate"].text = json.dumps({})

    mock_request.side_effect = [
        mock_responses["https://test.sagemaker.aws/api/2.0/mlflow/experiments/get-by-name"],
        mock_responses["https://test.sagemaker.aws/api/2.0/mlflow/runs/create"],
        mock_responses["https://test.sagemaker.aws/api/2.0/mlflow/runs/update"],
        *mock_responses["https://test.sagemaker.aws/api/2.0/mlflow/runs/log-batch"],
        mock_responses["https://test.sagemaker.aws/api/2.0/mlflow/runs/terminate"],
    ]

    metrics = [Metric("loss", 0.1, 1, 0)]
    params = [Param("learning_rate", "0.01")]
    tags = {"tag1": "value1"}

    log_to_mlflow(metrics, params, tags)

    assert mock_request.call_count == 7  # Total number of API calls


@patch("sagemaker.mlflow.forward_sagemaker_metrics.get_training_job_details")
@patch("sagemaker.mlflow.forward_sagemaker_metrics.create_metric_queries")
@patch("sagemaker.mlflow.forward_sagemaker_metrics.get_metric_data")
@patch("sagemaker.mlflow.forward_sagemaker_metrics.prepare_mlflow_metrics")
@patch("sagemaker.mlflow.forward_sagemaker_metrics.prepare_mlflow_params")
@patch("sagemaker.mlflow.forward_sagemaker_metrics.log_to_mlflow")
def test_log_sagemaker_job_to_mlflow(
    mock_log_to_mlflow,
    mock_prepare_params,
    mock_prepare_metrics,
    mock_get_metric_data,
    mock_create_queries,
    mock_get_job_details,
):
    mock_get_job_details.return_value = {
        "HyperParameters": {"learning_rate": "0.01"},
        "AlgorithmSpecification": {"MetricDefinitions": [{"Name": "loss"}]},
        "TrainingJobArn": "arn:aws:sagemaker:us-west-2:123456789012:training-job/test-job",
    }
    mock_create_queries.return_value = [{"MetricName": "loss"}]
    mock_get_metric_data.return_value = {"MetricQueryResults": []}
    mock_prepare_metrics.return_value = ([], {})
    mock_prepare_params.return_value = ([], {})

    log_sagemaker_job_to_mlflow("test-job")

    mock_get_job_details.assert_called_once()
    mock_create_queries.assert_called_once()
    mock_get_metric_data.assert_called_once()
    mock_prepare_metrics.assert_called_once()
    mock_prepare_params.assert_called_once()
    mock_log_to_mlflow.assert_called_once()


if __name__ == "__main__":
    pytest.main()
