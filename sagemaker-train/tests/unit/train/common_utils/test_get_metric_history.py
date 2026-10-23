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
"""Tests for ``get_metric_history``.

Before this change the SDK had no public function that returned MLflow metric history as
data. ``plot_training_metrics`` rendered a PNG and ``get_available_metrics`` returned only
names, so callers set the tracking URI and called ``MlflowClient().get_metric_history``
themselves, with recipe-specific metric-name aliases copied into each caller.
"""

from __future__ import absolute_import

from unittest.mock import Mock, patch

import pytest
from mlflow.entities import Metric

from sagemaker.train.common_utils.metrics_visualizer import (
    DEFAULT_METRIC_ALIASES,
    MetricPoint,
    get_metric_history,
)

TRACKING_ARN = "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/abc"
RUN_ID = "run-123"


def _training_job():
    job = Mock()
    job.training_job_name = "my-job"
    job.mlflow_config.mlflow_resource_arn = TRACKING_ARN
    job.mlflow_details.mlflow_run_id = RUN_ID
    return job


def _metric(name, step, value):
    return Metric(key=name, value=value, timestamp=1_700_000_000_000 + step, step=step)


def _history_side_effect(histories):
    def get_metric_history(run_id, metric_name):
        assert run_id == RUN_ID
        return histories[metric_name]

    return get_metric_history


@pytest.fixture
def mlflow_run():
    with (
        patch("mlflow.set_tracking_uri") as set_uri,
        patch("mlflow.get_run") as get_run,
        patch("mlflow.tracking.MlflowClient") as client_cls,
    ):
        yield set_uri, get_run, client_cls.return_value


def _logged(get_run, names):
    get_run.return_value.data.metrics = {name: 0.0 for name in names}


def test_alias_resolves_to_train_loss_when_total_loss_is_absent(mlflow_run):
    set_uri, get_run, client = mlflow_run
    _logged(get_run, ["train/loss", "val/loss", "lr"])
    client.get_metric_history.side_effect = _history_side_effect(
        {"train/loss": [_metric("train/loss", 1, 2.0), _metric("train/loss", 2, 1.5)]}
    )

    history = get_metric_history(_training_job(), metrics=["loss"])

    set_uri.assert_called_once_with(TRACKING_ARN)
    get_run.assert_called_once_with(RUN_ID)
    client.get_metric_history.assert_called_once_with(RUN_ID, "train/loss")
    assert list(history) == ["loss"]
    assert history["loss"] == [
        MetricPoint(step=1, value=2.0, timestamp=1_700_000_000_001),
        MetricPoint(step=2, value=1.5, timestamp=1_700_000_000_002),
    ]


def test_alias_prefers_total_loss_when_present(mlflow_run):
    _, get_run, client = mlflow_run
    _logged(get_run, ["total_loss", "train/loss"])
    client.get_metric_history.side_effect = _history_side_effect(
        {"total_loss": [_metric("total_loss", 1, 3.0)]}
    )

    history = get_metric_history(_training_job(), metrics=["loss"])

    client.get_metric_history.assert_called_once_with(RUN_ID, "total_loss")
    assert history["loss"][0].value == 3.0


def test_explicit_metric_list_returns_only_those_metrics(mlflow_run):
    _, get_run, client = mlflow_run
    _logged(get_run, ["lr", "grad_norm", "total_loss"])
    client.get_metric_history.side_effect = _history_side_effect(
        {"lr": [_metric("lr", 1, 0.1)], "grad_norm": [_metric("grad_norm", 1, 0.5)]}
    )

    history = get_metric_history(_training_job(), metrics=["lr", "grad_norm"])

    assert set(history) == {"lr", "grad_norm"}
    assert history["lr"] == [MetricPoint(step=1, value=0.1, timestamp=1_700_000_000_001)]
    assert client.get_metric_history.call_count == 2


def test_no_metric_list_returns_every_logged_metric(mlflow_run):
    _, get_run, client = mlflow_run
    _logged(get_run, ["lr", "total_loss"])
    client.get_metric_history.side_effect = _history_side_effect(
        {"lr": [_metric("lr", 1, 0.1)], "total_loss": [_metric("total_loss", 1, 3.0)]}
    )

    history = get_metric_history(_training_job())

    assert list(history) == ["lr", "total_loss"]


@patch("sagemaker.train.common_utils.metrics_visualizer.TrainingJob")
def test_job_name_string_goes_through_training_job_get(training_job_cls, mlflow_run):
    _, get_run, client = mlflow_run
    training_job_cls.get.return_value = _training_job()
    _logged(get_run, ["lr"])
    client.get_metric_history.side_effect = _history_side_effect({"lr": [_metric("lr", 1, 0.1)]})

    history = get_metric_history("my-job", metrics=["lr"])

    training_job_cls.get.assert_called_once_with(training_job_name="my-job")
    assert history["lr"][0].step == 1


def test_unknown_metric_raises_with_available_names(mlflow_run):
    _, get_run, client = mlflow_run
    _logged(get_run, ["lr", "total_loss"])

    with pytest.raises(ValueError) as excinfo:
        get_metric_history(_training_job(), metrics=["val_loss"])

    message = str(excinfo.value)
    assert "val_loss" in message
    assert "val_eval_total_loss / val/loss / eval_loss / val_loss" in message
    assert "Available metrics: lr, total_loss" in message
    client.get_metric_history.assert_not_called()


def test_points_are_sorted_by_step(mlflow_run):
    _, get_run, client = mlflow_run
    _logged(get_run, ["lr"])
    client.get_metric_history.side_effect = _history_side_effect(
        {"lr": [_metric("lr", 3, 0.3), _metric("lr", 1, 0.1), _metric("lr", 2, 0.2)]}
    )

    history = get_metric_history(_training_job(), metrics=["lr"])

    assert [point.step for point in history["lr"]] == [1, 2, 3]
    assert [point.value for point in history["lr"]] == [0.1, 0.2, 0.3]


def test_custom_aliases_replace_the_defaults(mlflow_run):
    _, get_run, client = mlflow_run
    _logged(get_run, ["my/reward"])
    client.get_metric_history.side_effect = _history_side_effect(
        {"my/reward": [_metric("my/reward", 1, 0.9)]}
    )

    history = get_metric_history(
        _training_job(), metrics=["reward"], aliases={"reward": ("my/reward",)}
    )

    assert history["reward"][0].value == 0.9
    assert "loss" in DEFAULT_METRIC_ALIASES


def test_job_without_mlflow_run_raises(mlflow_run):
    job = _training_job()
    job.mlflow_details.mlflow_run_id = None

    with pytest.raises(ValueError, match="has no MLflow run"):
        get_metric_history(job, metrics=["lr"])


def test_export_resolves_from_sagemaker_train():
    import sagemaker.train as train_module

    assert train_module.get_metric_history is get_metric_history
    assert train_module.MetricPoint is MetricPoint
