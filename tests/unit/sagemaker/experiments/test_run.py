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

import datetime
import unittest
from math import inf, nan
from unittest.mock import patch, Mock, MagicMock

import pandas as pd
import pytest

from sagemaker import Session
from sagemaker.experiments import _environment
from sagemaker.experiments._api_types import (
    TrialComponentSource,
    TrialComponentArtifact,
    TrialComponentSummary,
    TrialComponentStatus,
    TrialComponentSearchResult,
)
from sagemaker.experiments.experiment import _Experiment
from sagemaker.experiments.run import (
    Run,
    RUN_NAME_BASE,
    TRIAL_NAME_TEMPLATE,
    MAX_RUN_TC_ARTIFACTS_LEN,
    MAX_TRIAL_NAME_LEN,
    UNKNOWN_EXP_NAME,
)
from sagemaker.experiments.trial import _Trial
from sagemaker.experiments.trial_component import _TrialComponent
from sagemaker.utilities.search_expression import Filter, Operator, SearchExpression


@pytest.fixture
def client():
    """Mock client.

    Considerations when appropriate:

        * utilize botocore.stub.Stubber
        * separate runtime client from client
    """
    client_mock = unittest.mock.Mock()
    client_mock._client_config.user_agent = (
        "Boto3/1.14.24 Python/3.8.5 Linux/5.4.0-42-generic Botocore/1.17.24 Resource"
    )
    return client_mock


@pytest.fixture
def sagemaker_session(client):
    return Session(
        sagemaker_client=client,
    )


@pytest.fixture
def source_arn():
    return "source_arn"


def mock_tc_load_or_create_func(
    trial_component_name, display_name=None, tags=None, sagemaker_session=None
):
    return _TrialComponent(
        trial_component_name=trial_component_name,
        display_name=display_name,
        tags=tags,
        sagemaker_session=sagemaker_session,
    )


def mock_trial_load_or_create_func(
    experiment_name, trial_name, display_name=None, tags=None, sagemaker_session=None
):
    return _Trial(
        trial_name=trial_name,
        experiment_name=experiment_name,
        display_name=display_name,
        tags=tags,
        sagemaker_session=sagemaker_session,
    )


def mock_trial_component_load_func(trial_component_name, sagemaker_session=None):
    return _TrialComponent(
        trial_component_name=trial_component_name, sagemaker_session=sagemaker_session
    )


_exp_name = "my-experiment"
_run_name = "my-run"


@patch(
    "sagemaker.experiments.run._Experiment._load_or_create",
    MagicMock(return_value=_Experiment(experiment_name=_exp_name)),
)
@patch(
    "sagemaker.experiments.run._Trial._load_or_create",
    MagicMock(side_effect=mock_trial_load_or_create_func),
)
@patch.object(_Trial, "add_trial_component", MagicMock(return_value=None))
@patch(
    "sagemaker.experiments.run._TrialComponent._load_or_create",
    MagicMock(side_effect=mock_tc_load_or_create_func),
)
@patch.object(_TrialComponent, "save")
def test_run_init_outside_sm_job(mock_tc_save, sagemaker_session):
    with Run.init(
        experiment_name=_exp_name, run_name=_run_name, sagemaker_session=sagemaker_session
    ) as run_obj:
        assert not run_obj._in_sagemaker_job
        assert not run_obj._job_has_run
        assert _run_name == run_obj.run_name
        assert run_obj.run_name == run_obj._trial_component.trial_component_name
        assert not run_obj._trial_component.parameters
        assert TRIAL_NAME_TEMPLATE.format(_exp_name) == run_obj._trial.trial_name
        assert run_obj._experiment.experiment_name == _exp_name

    # trail_component.save is called when exiting the with block
    mock_tc_save.assert_called_once()


@patch(
    "sagemaker.experiments.run._Experiment._load_or_create",
    MagicMock(return_value=_Experiment(experiment_name=_exp_name)),
)
@patch(
    "sagemaker.experiments.run._Trial._load_or_create",
    MagicMock(side_effect=mock_trial_load_or_create_func),
)
@patch(
    "sagemaker.experiments.run._TrialComponent._load_or_create",
    MagicMock(
        return_value=_TrialComponent(
            trial_component_name=_run_name, source=TrialComponentSource("arn:")
        )
    ),
)
def test_run_init_outside_sm_job_but_with_job_tc_returned(sagemaker_session):
    with pytest.raises(RuntimeError) as err:
        Run.init(experiment_name=_exp_name, run_name=_run_name, sagemaker_session=sagemaker_session)

    assert f"Invalid run_name input {_run_name}" in str(err)


_load_run_tc_name = "load-run-tc"


# TODO: we may need to update/remove this once simplify the Run
@patch(
    "sagemaker.experiments.run._Experiment._load_or_create",
    MagicMock(return_value=_Experiment(experiment_name=_exp_name)),
)
@patch(
    "sagemaker.experiments.run._Trial._load_or_create",
    MagicMock(side_effect=mock_trial_load_or_create_func),
)
@patch.object(_Trial, "add_trial_component", MagicMock(return_value=None))
@patch(
    "sagemaker.experiments.run._TrialComponent.load",
    MagicMock(side_effect=mock_trial_component_load_func),
)
@patch("sagemaker.experiments.run._RunEnvironment")
def test_run_init_in_sm_training_job_load_run_tc(mock_run_env, sagemaker_session):
    rv = Mock()
    rv.source_arn = "arn:1234/my-train-job"
    rv.environment_type = _environment.EnvironmentType.SageMakerTrainingJob
    mock_run_env.load.return_value = rv

    sagemaker_session.sagemaker_client.describe_training_job.return_value = {
        "TrainingJobName": "train-job-experiments",
        # The Run object has been created else where
        "ExperimentConfig": {
            "ExperimentName": _exp_name,
            "TrialName": f"Default-Run-Group-{_exp_name}",
            "RunName": _load_run_tc_name,
        },
    }
    # run_name is not given, which will be auto generated
    run_obj = Run.init(
        experiment_name=_exp_name,
        sagemaker_session=sagemaker_session,
    )
    assert run_obj._in_sagemaker_job
    assert not run_obj._job_has_run
    assert RUN_NAME_BASE in run_obj.run_name  # This run_name will be ignored
    assert run_obj._trial_component.trial_component_name == _load_run_tc_name
    assert TRIAL_NAME_TEMPLATE.format(_exp_name) in run_obj._trial.trial_name
    assert run_obj._trial.display_name is None
    assert run_obj._experiment.experiment_name == _exp_name


# TODO: we may need to update/remove this once simplify the Run
@patch(
    "sagemaker.experiments.run._Experiment._load_or_create",
    MagicMock(return_value=_Experiment(experiment_name=_exp_name)),
)
@patch(
    "sagemaker.experiments.run._Trial._load_or_create",
    MagicMock(side_effect=mock_trial_load_or_create_func),
)
@patch.object(_Trial, "add_trial_component", MagicMock(return_value=None))
@patch.object(
    _TrialComponent, "_load_or_create", MagicMock(side_effect=mock_tc_load_or_create_func)
)
@patch("sagemaker.experiments.run._RunEnvironment")
def test_run_init_in_sm_training_job_only(mock_run_env, sagemaker_session):
    rv = Mock()
    rv.source_arn = "arn:1234/my-train-job"
    rv.environment_type = _environment.EnvironmentType.SageMakerTrainingJob
    mock_run_env.load.return_value = rv

    # No Run object is created else where
    sagemaker_session.sagemaker_client.describe_training_job.return_value = {
        "TrainingJobName": "train-job-experiments",
    }
    # run_name is given
    run_obj = Run.init(
        experiment_name=_exp_name,
        run_name=_run_name,
        sagemaker_session=sagemaker_session,
    )
    assert run_obj._in_sagemaker_job
    assert run_obj._job_has_run
    assert run_obj.run_name == _run_name
    assert run_obj._trial_component.trial_component_name == _run_name
    assert TRIAL_NAME_TEMPLATE.format(_exp_name) in run_obj._trial.trial_name
    assert run_obj._trial.display_name is None
    assert run_obj._experiment.experiment_name == _exp_name


@patch(
    "sagemaker.experiments.run._Experiment._load_or_create",
    MagicMock(return_value=_Experiment(experiment_name=_exp_name)),
)
@patch(
    "sagemaker.experiments.run._Trial._load_or_create",
    MagicMock(side_effect=mock_trial_load_or_create_func),
)
@patch("sagemaker.experiments.run._RunEnvironment")
def test_run_init_in_sm_processing_job(mock_run_env, sagemaker_session):
    rv = unittest.mock.Mock()
    rv.source_arn = "arn:1234"
    rv.environment_type = _environment.EnvironmentType.SageMakerProcessingJob
    mock_run_env.load.return_value = rv

    with pytest.raises(RuntimeError) as err:
        Run.init(
            experiment_name=_exp_name,
            run_name=_run_name,
            sagemaker_session=sagemaker_session,
        )

    assert (
        "Experiment Run init is not currently supported "
        "in Sagemaker jobs other than the Training job"
    ) in str(err)


@pytest.fixture
def local_run_obj(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    client.update_trial_component.return_value = {}
    client.associate_trial_component.return_value = {}
    with patch(
        "sagemaker.experiments.run._Experiment._load_or_create",
        MagicMock(
            return_value=_Experiment(experiment_name=_exp_name, sagemaker_session=sagemaker_session)
        ),
    ):
        with patch(
            "sagemaker.experiments.run._TrialComponent._load_or_create",
            MagicMock(side_effect=mock_tc_load_or_create_func),
        ):
            with patch(
                "sagemaker.experiments.run._Trial._load_or_create",
                MagicMock(side_effect=mock_trial_load_or_create_func),
            ):
                run = Run.init(
                    experiment_name=_exp_name,
                    run_name=_run_name,
                    sagemaker_session=sagemaker_session,
                )
                run._artifact_uploader = Mock()
                run._lineage_artifact_tracker = Mock()
                run._metrics_manager = Mock()
                return run


# TODO: we may need to update/remove this once simplify the Run
@pytest.fixture
def train_run_obj(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    client.update_trial_component.return_value = {}
    client.associate_trial_component.return_value = {}
    with patch(
        "sagemaker.experiments.run._Experiment._load_or_create",
        MagicMock(
            return_value=_Experiment(experiment_name=_exp_name, sagemaker_session=sagemaker_session)
        ),
    ):
        with patch(
            "sagemaker.experiments.run._Trial._load_or_create",
            MagicMock(side_effect=mock_trial_load_or_create_func),
        ):
            with patch(
                "sagemaker.experiments.run._TrialComponent._load_or_create",
                MagicMock(side_effect=mock_tc_load_or_create_func),
            ):
                with patch("sagemaker.experiments.run._RunEnvironment") as mock_run_env:
                    rv = Mock()
                    rv.source_arn = "arn:1234"
                    rv.environment_type = _environment.EnvironmentType.SageMakerTrainingJob
                    mock_run_env.load.return_value = rv

                    sagemaker_session.sagemaker_client.describe_training_job.return_value = {
                        "TrainingJobName": "train-job-experiments",
                    }
                    # The job TC will be used
                    return Run.init(
                        experiment_name=_exp_name,
                        run_name=_run_name,
                        sagemaker_session=sagemaker_session,
                    )


def test_log_parameter(local_run_obj):
    local_run_obj.log_parameter("foo", "bar")
    assert local_run_obj._trial_component.parameters["foo"] == "bar"
    local_run_obj.log_parameter("whizz", 1)
    assert local_run_obj._trial_component.parameters["whizz"] == 1


def test_log_parameter_skip_invalid_value(local_run_obj):
    local_run_obj.log_parameter("key", nan)
    assert "key" not in local_run_obj._trial_component.parameters


def test_log_parameters(local_run_obj):
    local_run_obj.log_parameters({"a": "b", "c": "d", "e": 5})
    assert local_run_obj._trial_component.parameters == {"a": "b", "c": "d", "e": 5}


def test_log_parameters_skip_invalid_values(local_run_obj):
    local_run_obj.log_parameters({"a": "b", "c": "d", "e": 5, "f": nan})
    assert local_run_obj._trial_component.parameters == {"a": "b", "c": "d", "e": 5}


def test_log_input(local_run_obj):
    local_run_obj.log_input("foo", "baz", "text/text")
    assert local_run_obj._trial_component.input_artifacts == {
        "foo": TrialComponentArtifact(value="baz", media_type="text/text")
    }


def test_log_output(local_run_obj):
    local_run_obj.log_output("foo", "baz", "text/text")
    assert local_run_obj._trial_component.output_artifacts == {
        "foo": TrialComponentArtifact(value="baz", media_type="text/text")
    }


def test_log_metric(local_run_obj):
    now = datetime.datetime.now()
    local_run_obj.log_metric(name="foo", value=1.0, step=1, timestamp=now)
    local_run_obj._metrics_manager.log_metric.assert_called_with(
        metric_name="foo", value=1.0, step=1, timestamp=now
    )


def test_log_metric_skip_invalid_value(local_run_obj):
    local_run_obj.log_metric(None, nan, None, None)
    assert not local_run_obj._metrics_manager.log_metric.called


def test_log_metric_attribute_error(local_run_obj):
    now = datetime.datetime.now()

    local_run_obj._metrics_manager.log_metric.side_effect = AttributeError

    with pytest.raises(AttributeError):
        local_run_obj.log_metric("foo", 1.0, 1, now)


def test_log_metric_attribute_error_warned(local_run_obj):
    now = datetime.datetime.now()

    local_run_obj._metrics_manager = None
    local_run_obj._warned_on_metrics = None

    local_run_obj.log_metric("foo", 1.0, 1, now)

    assert local_run_obj._warned_on_metrics


def test_log_output_artifact(local_run_obj):
    local_run_obj._artifact_uploader.upload_artifact.return_value = ("s3uri_value", "etag_value")

    local_run_obj.log_artifact_file("foo.txt", "name", "whizz/bang")
    local_run_obj._artifact_uploader.upload_artifact.assert_called_with("foo.txt")
    assert "whizz/bang" == local_run_obj._trial_component.output_artifacts["name"].media_type

    local_run_obj.log_artifact_file("foo.txt")
    local_run_obj._artifact_uploader.upload_artifact.assert_called_with("foo.txt")
    assert "foo.txt" in local_run_obj._trial_component.output_artifacts
    assert "text/plain" == local_run_obj._trial_component.output_artifacts["foo.txt"].media_type


def test_log_input_artifact(local_run_obj):
    local_run_obj._artifact_uploader.upload_artifact.return_value = ("s3uri_value", "etag_value")

    local_run_obj.log_artifact_file("foo.txt", "name", "whizz/bang", is_output=False)
    local_run_obj._artifact_uploader.upload_artifact.assert_called_with("foo.txt")
    assert "whizz/bang" == local_run_obj._trial_component.input_artifacts["name"].media_type

    local_run_obj.log_artifact_file("foo.txt", is_output=False)
    local_run_obj._artifact_uploader.upload_artifact.assert_called_with("foo.txt")
    assert "foo.txt" in local_run_obj._trial_component.input_artifacts
    assert "text/plain" == local_run_obj._trial_component.input_artifacts["foo.txt"].media_type


def test_log_lineage_output_artifact(local_run_obj):
    local_run_obj._artifact_uploader.upload_artifact.return_value = ("s3uri_value", "etag_value")

    local_run_obj.log_lineage_artifact("foo.txt", "name", "whizz/bang")
    local_run_obj._artifact_uploader.upload_artifact.assert_called_with("foo.txt")
    local_run_obj._lineage_artifact_tracker.add_output_artifact.assert_called_with(
        "name", "s3uri_value", "etag_value", "whizz/bang"
    )

    local_run_obj.log_lineage_artifact("foo.txt")
    local_run_obj._artifact_uploader.upload_artifact.assert_called_with("foo.txt")
    local_run_obj._lineage_artifact_tracker.add_output_artifact.assert_called_with(
        "foo.txt", "s3uri_value", "etag_value", "text/plain"
    )


def test_log_lineage_input_artifact(local_run_obj):
    local_run_obj._artifact_uploader.upload_artifact.return_value = ("s3uri_value", "etag_value")

    local_run_obj.log_lineage_artifact("foo.txt", "name", "whizz/bang", is_output=False)
    local_run_obj._artifact_uploader.upload_artifact.assert_called_with("foo.txt")
    local_run_obj._lineage_artifact_tracker.add_input_artifact.assert_called_with(
        "name", "s3uri_value", "etag_value", "whizz/bang"
    )

    local_run_obj.log_lineage_artifact("foo.txt", is_output=False)
    local_run_obj._artifact_uploader.upload_artifact.assert_called_with("foo.txt")
    local_run_obj._lineage_artifact_tracker.add_input_artifact.assert_called_with(
        "foo.txt", "s3uri_value", "etag_value", "text/plain"
    )


def test_log_multiple_inputs(local_run_obj):
    for index in range(0, MAX_RUN_TC_ARTIFACTS_LEN):
        file_path = "foo" + str(index) + ".txt"
        local_run_obj._trial_component.input_artifacts[file_path] = {
            "foo": TrialComponentArtifact(value="baz" + str(index), media_type="text/text")
        }
    with pytest.raises(ValueError) as error:
        local_run_obj.log_input("foo.txt", "name", "whizz/bang")
    assert f"Cannot add more than {MAX_RUN_TC_ARTIFACTS_LEN} input_artifacts" in str(error)


def test_log_multiple_outputs(local_run_obj):
    for index in range(0, MAX_RUN_TC_ARTIFACTS_LEN):
        file_path = "foo" + str(index) + ".txt"
        local_run_obj._trial_component.output_artifacts[file_path] = {
            "foo": TrialComponentArtifact(value="baz" + str(index), media_type="text/text")
        }
    with pytest.raises(ValueError) as error:
        local_run_obj.log_output("foo.txt", "name", "whizz/bang")
    assert f"Cannot add more than {MAX_RUN_TC_ARTIFACTS_LEN} output_artifacts" in str(error)


def test_log_multiple_input_artifacts(local_run_obj):
    for index in range(0, MAX_RUN_TC_ARTIFACTS_LEN):
        file_path = "foo" + str(index) + ".txt"
        local_run_obj._artifact_uploader.upload_artifact.return_value = (
            "s3uri_value" + str(index),
            "etag_value" + str(index),
        )
        local_run_obj.log_artifact_file(
            file_path, "name" + str(index), "whizz/bang" + str(index), is_output=False
        )
        local_run_obj._artifact_uploader.upload_artifact.assert_called_with(file_path)

    local_run_obj._artifact_uploader.upload_artifact.return_value = ("s3uri_value", "etag_value")

    # log an output artifact, should be fine
    local_run_obj.log_artifact_file("foo.txt", "name", "whizz/bang", is_output=True)

    # log an extra input artifact, should raise exception
    with pytest.raises(ValueError) as error:
        local_run_obj.log_artifact_file("foo.txt", "name", "whizz/bang", is_output=False)
    assert f"Cannot add more than {MAX_RUN_TC_ARTIFACTS_LEN} input_artifacts" in str(error)


def test_log_multiple_output_artifacts(local_run_obj):
    for index in range(0, MAX_RUN_TC_ARTIFACTS_LEN):
        file_path = "foo" + str(index) + ".txt"
        local_run_obj._artifact_uploader.upload_artifact.return_value = (
            "s3uri_value" + str(index),
            "etag_value" + str(index),
        )
        local_run_obj.log_artifact_file(file_path, "name" + str(index), "whizz/bang" + str(index))
        local_run_obj._artifact_uploader.upload_artifact.assert_called_with(file_path)

    local_run_obj._artifact_uploader.upload_artifact.return_value = ("s3uri_value", "etag_value")

    # log an input artifact, should be fine
    local_run_obj.log_artifact_file("foo.txt", "name", "whizz/bang", is_output=False)

    # log an extra output artifact, should raise exception
    with pytest.raises(ValueError) as error:
        local_run_obj.log_artifact_file("foo.txt", "name", "whizz/bang")
    assert f"Cannot add more than {MAX_RUN_TC_ARTIFACTS_LEN} output_artifacts" in str(error)


def test_log_precision_recall(local_run_obj):
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8]
    no_skill = 0.1
    title = "TestPrecisionRecall"

    local_run_obj._artifact_uploader.upload_object_artifact.return_value = (
        "s3uri_value",
        "etag_value",
    )

    local_run_obj.log_precision_recall(
        y_true, y_scores, 0, title=title, no_skill=no_skill, is_output=False
    )

    expected_data = {
        "type": "PrecisionRecallCurve",
        "version": 0,
        "title": title,
        "precision": [0.5, 0.3333333333333333, 0.5, 0.0, 1.0],
        "recall": [1.0, 0.5, 0.5, 0.0, 0.0],
        "averagePrecisionScore": 0.5,
        "noSkill": 0.1,
    }
    local_run_obj._artifact_uploader.upload_object_artifact.assert_called_with(
        title, expected_data, file_extension="json"
    )

    local_run_obj._lineage_artifact_tracker.add_input_artifact.assert_called_with(
        title, "s3uri_value", "etag_value", "PrecisionRecallCurve"
    )


def test_log_precision_recall_invalid_input(local_run_obj):
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35]
    no_skill = 0.1

    with pytest.raises(ValueError) as error:
        local_run_obj.log_precision_recall(
            y_true, y_scores, 0, title="TestPrecisionRecall", no_skill=no_skill, is_output=False
        )
    assert "Lengths mismatch between true labels and predicted probabilities" in str(error)


def test_log_confusion_matrix(local_run_obj):
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]

    local_run_obj._artifact_uploader.upload_object_artifact.return_value = (
        "s3uri_value",
        "etag_value",
    )

    local_run_obj.log_confusion_matrix(y_true, y_pred, title="TestConfusionMatrix")

    expected_data = {
        "type": "ConfusionMatrix",
        "version": 0,
        "title": "TestConfusionMatrix",
        "confusionMatrix": [[2, 0, 0], [0, 0, 1], [1, 0, 2]],
    }

    local_run_obj._artifact_uploader.upload_object_artifact.assert_called_with(
        "TestConfusionMatrix", expected_data, file_extension="json"
    )

    local_run_obj._lineage_artifact_tracker.add_output_artifact.assert_called_with(
        "TestConfusionMatrix", "s3uri_value", "etag_value", "ConfusionMatrix"
    )


def test_log_confusion_matrix_invalid_input(local_run_obj):
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0]

    with pytest.raises(ValueError) as error:
        local_run_obj.log_confusion_matrix(y_true, y_pred, title="TestConfusionMatrix")
    assert "Lengths mismatch between true labels and predicted labels" in str(error)


def test_log_table_both_specified(local_run_obj):
    with pytest.raises(ValueError) as error:
        local_run_obj.log_table(title="test", values={"foo": "bar"}, data_frame={"foo": "bar"})
    assert "either values or data_frame should be provided" in str(error)


def test_log_table_neither_specified(local_run_obj):
    with pytest.raises(ValueError) as error:
        local_run_obj.log_table(title="test")
    assert "either values or data_frame should be provided" in str(error)


def test_log_table_invalid_values(local_run_obj):
    values = {"x": "foo", "y": [4, 5, 6]}

    with pytest.raises(ValueError) as error:
        local_run_obj.log_table(title="test", values=values)
    assert "Table values should be list" in str(error)


def test_log_table(local_run_obj):
    values = {"x": [1, 2, 3], "y": [4, 5, 6]}

    local_run_obj._artifact_uploader.upload_object_artifact.return_value = (
        "s3uri_value",
        "etag_value",
    )

    local_run_obj.log_table(title="TestTable", values=values, is_output=False)
    expected_data = {
        "type": "Table",
        "version": 0,
        "title": "TestTable",
        "fields": [
            {"name": "x", "type": "string"},
            {"name": "y", "type": "string"},
        ],
        "data": {"x": [1, 2, 3], "y": [4, 5, 6]},
    }
    local_run_obj._artifact_uploader.upload_object_artifact.assert_called_with(
        "TestTable", expected_data, file_extension="json"
    )

    local_run_obj._lineage_artifact_tracker.add_input_artifact.assert_called_with(
        "TestTable", "s3uri_value", "etag_value", "Table"
    )


def test_log_table_dataframe(local_run_obj):
    dataframe = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    local_run_obj._artifact_uploader.upload_object_artifact.return_value = (
        "s3uri_value",
        "etag_value",
    )

    local_run_obj.log_table(title="TestTable", data_frame=dataframe)
    expected_data = {
        "type": "Table",
        "version": 0,
        "title": "TestTable",
        "fields": [{"name": "x", "type": "number"}, {"name": "y", "type": "number"}],
        "data": {"x": [1, 2, 3], "y": [4, 5, 6]},
    }
    local_run_obj._artifact_uploader.upload_object_artifact.assert_called_with(
        "TestTable", expected_data, file_extension="json"
    )

    local_run_obj._lineage_artifact_tracker.add_output_artifact.assert_called_with(
        "TestTable", "s3uri_value", "etag_value", "Table"
    )


def test_log_roc_curve(local_run_obj):
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8]

    local_run_obj._artifact_uploader.upload_object_artifact.return_value = (
        "s3uri_value",
        "etag_value",
    )

    local_run_obj.log_roc_curve(y_true, y_scores, title="TestROCCurve", is_output=False)

    expected_data = {
        "type": "ROCCurve",
        "version": 0,
        "title": "TestROCCurve",
        "falsePositiveRate": [0.0, 0.0, 0.5, 0.5, 1.0],
        "truePositiveRate": [0.0, 0.5, 0.5, 1.0, 1.0],
        "areaUnderCurve": 0.75,
    }
    local_run_obj._artifact_uploader.upload_object_artifact.assert_called_with(
        "TestROCCurve", expected_data, file_extension="json"
    )

    local_run_obj._lineage_artifact_tracker.add_input_artifact.assert_called_with(
        "TestROCCurve", "s3uri_value", "etag_value", "ROCCurve"
    )


def test_log_roc_curve_invalid_input(local_run_obj):
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35]

    with pytest.raises(ValueError) as error:
        local_run_obj.log_roc_curve(y_true, y_scores, title="TestROCCurve", is_output=False)
    assert "Lengths mismatch between true labels and predicted scores" in str(error)


@patch("sagemaker.experiments.run._TrialComponent.load")
@patch("sagemaker.experiments.run._TrialComponent.list")
def test_list(mock_tc_list, mock_tc_load, local_run_obj, sagemaker_session):
    start_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
    end_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=2)
    creation_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=3)
    last_modified_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=4)
    run_list_len = 20
    mock_tc_list.return_value = [
        TrialComponentSummary(
            trial_component_name="A" + str(i),
            trial_component_arn="B" + str(i),
            display_name="C" + str(i),
            source_arn="D" + str(i),
            status=TrialComponentStatus(primary_status="InProgress", message="E" + str(i)),
            start_time=start_time + datetime.timedelta(hours=i),
            end_time=end_time + datetime.timedelta(hours=i),
            creation_time=creation_time + datetime.timedelta(hours=i),
            last_modified_time=last_modified_time + datetime.timedelta(hours=i),
            last_modified_by={},
        )
        for i in range(run_list_len)
    ]
    mock_tc_load.side_effect = [
        _TrialComponent(
            trial_component_name="A" + str(i),
            trial_component_arn="B" + str(i),
            display_name="C" + str(i),
            source_arn="D" + str(i),
            status=TrialComponentStatus(primary_status="InProgress", message="E" + str(i)),
            start_time=start_time + datetime.timedelta(hours=i),
            end_time=end_time + datetime.timedelta(hours=i),
            creation_time=creation_time + datetime.timedelta(hours=i),
            last_modified_time=last_modified_time + datetime.timedelta(hours=i),
            last_modified_by={},
        )
        for i in range(run_list_len)
    ]

    run_list = Run.list(
        experiment_name=_exp_name,
        sort_by="CreationTime",
        sort_order="Ascending",
        sagemaker_session=sagemaker_session,
    )

    mock_tc_list.assert_called_once_with(
        experiment_name=_exp_name,
        created_before=None,
        created_after=None,
        sort_by="CreationTime",
        sort_order="Ascending",
        sagemaker_session=sagemaker_session,
        max_results=None,
        next_token=None,
    )
    assert len(run_list) == run_list_len
    for i in range(run_list_len):
        run = run_list[i]
        assert run.experiment_name == _exp_name
        assert run.run_name == "A" + str(i)
        assert run._experiment is None
        assert run._trial is None
        assert isinstance(run._trial_component, _TrialComponent)
        assert run._trial_component.trial_component_name == "A" + str(i)
        assert run._in_sagemaker_job is False
        assert run._job_has_run is False
        assert run._artifact_uploader
        assert run._lineage_artifact_tracker
        assert not run._metrics_manager


@patch("sagemaker.experiments.run._TrialComponent.list")
def test_list_empty(mock_tc_list, sagemaker_session):
    mock_tc_list.return_value = []
    assert [] == Run.list(experiment_name="my-exp", sagemaker_session=sagemaker_session)


@patch("sagemaker.experiments.run._TrialComponent.load")
@patch("sagemaker.experiments.run._TrialComponent.search")
def test_search(mock_tc_search, mock_tc_load, local_run_obj, sagemaker_session):
    run_list_len = 20
    mock_tc_search.return_value = [
        TrialComponentSearchResult(
            trial_component_name=f"A{i}",
            trial_component_arn=f"B{i}",
            display_name=f"C{i}",
            parents=[{"ExperimentName": f"Exp-{i}-0"}, {"ExperimentName": f"Exp-{i}-1"}],
        )
        for i in range(run_list_len)
    ]
    mock_tc_load.side_effect = [
        _TrialComponent(
            trial_component_name=f"A{i}",
            trial_component_arn=f"B{i}",
            display_name=f"C{i}",
        )
        for i in range(run_list_len)
    ]

    run_list = Run.search(
        sort_by="CreationTime",
        sort_order="Ascending",
        sagemaker_session=sagemaker_session,
    )

    mock_tc_search.assert_called_once_with(
        sort_by="CreationTime",
        sort_order="Ascending",
        sagemaker_session=sagemaker_session,
        search_expression=None,
        max_results=50,
    )
    assert len(run_list) == run_list_len * 2
    for i in range(run_list_len):
        run = run_list[i]
        assert run.experiment_name == f"Exp-{int(i / 2)}-{(i % 2)}"
        assert run.run_name == f"A{int(i / 2)}"
        assert run._experiment is None
        assert run._trial is None
        assert isinstance(run._trial_component, _TrialComponent)
        assert run._trial_component.trial_component_name == f"A{int(i / 2)}"
        assert run._in_sagemaker_job is False
        assert run._artifact_uploader
        assert run._lineage_artifact_tracker
        assert not run._metrics_manager


@patch("sagemaker.experiments.run._TrialComponent.load")
@patch("sagemaker.experiments.run._TrialComponent.search")
def test_search_empty_parents(mock_tc_search, mock_tc_load, local_run_obj, sagemaker_session):
    run_list_len = 20
    mock_tc_search.return_value = [
        TrialComponentSearchResult(
            trial_component_name=f"A{i}",
            trial_component_arn=f"B{i}",
            display_name=f"C{i}",
            parents=[],
        )
        for i in range(run_list_len)
    ]
    mock_tc_load.side_effect = [
        _TrialComponent(
            trial_component_name=f"A{i}",
            trial_component_arn=f"B{i}",
            display_name=f"C{i}",
        )
        for i in range(run_list_len)
    ]

    run_list = Run.search(
        sort_by="CreationTime",
        sort_order="Ascending",
        sagemaker_session=sagemaker_session,
    )

    mock_tc_search.assert_called_once_with(
        sort_by="CreationTime",
        sort_order="Ascending",
        sagemaker_session=sagemaker_session,
        search_expression=None,
        max_results=50,
    )
    assert len(run_list) == run_list_len
    for i in range(run_list_len):
        run = run_list[i]
        assert run.experiment_name == UNKNOWN_EXP_NAME
        assert run.run_name == f"A{i}"
        assert run._experiment is None
        assert run._trial is None
        assert isinstance(run._trial_component, _TrialComponent)
        assert run._trial_component.trial_component_name == f"A{i}"
        assert run._in_sagemaker_job is False
        assert run._job_has_run is False
        assert run._artifact_uploader
        assert run._lineage_artifact_tracker
        assert not run._metrics_manager


@patch("sagemaker.experiments.run._TrialComponent.search")
def test_search_empty(mock_tc_search, sagemaker_session):
    mock_tc_search.return_value = []
    search_filter = Filter(name="ExperimentName", operator=Operator.EQUALS, value="unknown")
    search_expression = SearchExpression(filters=[search_filter])
    assert [] == Run.search(
        search_expression=search_expression, sagemaker_session=sagemaker_session
    )


def test_enter(local_run_obj):
    local_run_obj.__enter__()
    assert isinstance(local_run_obj._trial_component.start_time, datetime.datetime)
    assert local_run_obj._trial_component.status.primary_status == "InProgress"


def test_exit(sagemaker_session, local_run_obj):
    sagemaker_session.sagemaker_client.update_trial_component.return_value = {}
    with local_run_obj:
        pass
    assert local_run_obj._trial_component.status.primary_status == "Completed"
    assert isinstance(local_run_obj._trial_component.end_time, datetime.datetime)


def test_exit_fail(sagemaker_session, local_run_obj):
    sagemaker_session.sagemaker_client.update_trial_component.return_value = {}
    try:
        with local_run_obj:
            raise ValueError("Foo")
    except ValueError:
        pass

    assert local_run_obj._trial_component.status.primary_status == "Failed"
    assert local_run_obj._trial_component.status.message
    assert isinstance(local_run_obj._trial_component.end_time, datetime.datetime)


# TODO: we may need to update/remove this once simplify the Run
def test_enter_sagemaker_job_only(train_run_obj, sagemaker_session):
    # The Run object is initialized in job env only
    # meaning that no RunName found in the job's experiment config.
    # Its trail component's timestamp and status can still be correctly set
    sagemaker_session.sagemaker_client.update_trial_component.return_value = {}
    with train_run_obj:
        pass
    assert isinstance(train_run_obj._trial_component.start_time, datetime.datetime)
    assert isinstance(train_run_obj._trial_component.end_time, datetime.datetime)
    assert train_run_obj._trial_component.status.primary_status == "Completed"


@pytest.mark.parametrize(
    "metric_value",
    [1.3, "nan", "inf", "-inf", None],
)
def test_is_input_valid(local_run_obj, metric_value):
    assert local_run_obj._is_input_valid("metric", "Name", metric_value)


@pytest.mark.parametrize(
    "metric_value",
    [nan, inf, -inf],
)
def test_is_input_valid_false(local_run_obj, metric_value):
    assert not local_run_obj._is_input_valid("parameter", "Name", metric_value)


def test_generate_trial_name():
    base_name = "x" * MAX_TRIAL_NAME_LEN
    trial_name = Run._generate_trial_name(base_name=base_name)
    assert len(trial_name) <= MAX_TRIAL_NAME_LEN
