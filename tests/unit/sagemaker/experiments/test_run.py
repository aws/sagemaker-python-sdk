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

import dateutil
import pandas as pd
import pytest

from sagemaker.experiments import _environment
from sagemaker.experiments._api_types import (
    TrialComponentSource,
    TrialComponentArtifact,
    TrialComponentSummary,
    TrialComponentStatus,
    TrialComponentSearchResult,
    Parent,
)
from sagemaker.experiments.experiment import _Experiment
from sagemaker.experiments.run import (
    Run,
    TRIAL_NAME_TEMPLATE,
    MAX_RUN_TC_ARTIFACTS_LEN,
    MAX_TRIAL_NAME_LEN,
    UNKNOWN_NAME,
    EXPERIMENT_NAME,
    RUN_NAME,
    TRIAL_NAME,
)
from sagemaker.experiments.trial import _Trial
from sagemaker.experiments.trial_component import _TrialComponent
from sagemaker.utilities.search_expression import Filter, Operator, SearchExpression
from tests.unit.sagemaker.experiments.helpers import (
    mock_trial_load_or_create_func,
    mock_tc_load_or_create_func,
    TEST_EXP_NAME,
    TEST_RUN_NAME,
    TEST_RUN_GRP_NAME,
    mock_trial_component_load_func,
)


@patch(
    "sagemaker.experiments.run._Experiment._load_or_create",
    MagicMock(return_value=_Experiment(experiment_name=TEST_EXP_NAME)),
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
def test_run_init(mock_tc_save, sagemaker_session):
    with Run.init(
        experiment_name=TEST_EXP_NAME, run_name=TEST_RUN_NAME, sagemaker_session=sagemaker_session
    ) as run_obj:
        assert not run_obj._in_load
        assert not run_obj._inside_load_context
        assert run_obj._inside_init_context
        assert not run_obj._trial_component.parameters

        assert run_obj.experiment_name == TEST_EXP_NAME
        assert run_obj.run_name == TEST_RUN_NAME
        assert run_obj.run_group_name == TRIAL_NAME_TEMPLATE.format(TEST_EXP_NAME)
        assert run_obj._trial_component.trial_component_name == TEST_RUN_NAME
        assert run_obj._trial.trial_name == TRIAL_NAME_TEMPLATE.format(TEST_EXP_NAME)
        assert run_obj._experiment.experiment_name == TEST_EXP_NAME
        assert run_obj.experiment_config == {
            EXPERIMENT_NAME: TEST_EXP_NAME,
            TRIAL_NAME: run_obj.run_group_name,
            RUN_NAME: TEST_RUN_NAME,
        }

    # trail_component.save is called when exiting the with block
    mock_tc_save.assert_called_once()


@patch(
    "sagemaker.experiments.run._Experiment._load_or_create",
    MagicMock(return_value=_Experiment(experiment_name=TEST_EXP_NAME)),
)
@patch(
    "sagemaker.experiments.run._Trial._load_or_create",
    MagicMock(side_effect=mock_trial_load_or_create_func),
)
@patch(
    "sagemaker.experiments.run._TrialComponent._load_or_create",
    MagicMock(
        return_value=_TrialComponent(
            trial_component_name=TEST_RUN_NAME, source=TrialComponentSource("arn:")
        )
    ),
)
def test_run_init_with_job_tc_as_run_name(sagemaker_session):
    with pytest.raises(ValueError) as err:
        Run.init(
            experiment_name=TEST_EXP_NAME,
            run_name=TEST_RUN_NAME,
            sagemaker_session=sagemaker_session,
        )

    assert f"Invalid run_name input {TEST_RUN_NAME}" in str(err)


@patch("sagemaker.experiments.run._RunEnvironment")
def test_run_init_in_sm_processing_job(mock_run_env, sagemaker_session):
    rv = unittest.mock.Mock()
    rv.source_arn = "arn:1234"
    rv.environment_type = _environment.EnvironmentType.SageMakerProcessingJob
    mock_run_env.load.return_value = rv

    with pytest.raises(RuntimeError) as err:
        Run.init(
            experiment_name=TEST_EXP_NAME,
            run_name=TEST_RUN_NAME,
            sagemaker_session=sagemaker_session,
        )

    assert (
        "Experiment Run init is not currently supported "
        "in Sagemaker jobs other than the Training job"
    ) in str(err)


@patch.object(_TrialComponent, "save", MagicMock(return_value=None))
@patch(
    "sagemaker.experiments.run._TrialComponent.load",
    MagicMock(side_effect=mock_trial_component_load_func),
)
@patch("sagemaker.experiments.run._RunEnvironment")
def test_run_load_no_run_name_and_in_train_job(mock_run_env, sagemaker_session):
    rv = Mock()
    rv.source_arn = "arn:1234/my-train-job"
    rv.environment_type = _environment.EnvironmentType.SageMakerTrainingJob
    mock_run_env.load.return_value = rv

    exp_config = {
        EXPERIMENT_NAME: TEST_EXP_NAME,
        TRIAL_NAME: TEST_RUN_GRP_NAME,
        RUN_NAME: TEST_RUN_NAME,
    }
    sagemaker_session.sagemaker_client.describe_training_job.return_value = {
        "TrainingJobName": "train-job-experiments",
        # The Run object has been created else where
        "ExperimentConfig": exp_config,
    }
    with Run.load(sagemaker_session=sagemaker_session) as run_obj:
        assert run_obj._in_load
        assert not run_obj._inside_init_context
        assert run_obj._inside_load_context
        assert run_obj.run_name == TEST_RUN_NAME
        assert run_obj._trial_component.trial_component_name == run_obj.run_name
        assert run_obj.run_group_name == TEST_RUN_GRP_NAME
        assert not run_obj._trial
        assert run_obj.experiment_name == TEST_EXP_NAME
        assert not run_obj._experiment
        assert run_obj.experiment_config == exp_config


@patch("sagemaker.experiments.run._RunEnvironment")
def test_run_load_no_run_name_and_in_train_job_but_fail_to_get_exp_cfg(
    mock_run_env, sagemaker_session
):
    rv = Mock()
    rv.source_arn = "arn:1234/my-train-job"
    rv.environment_type = _environment.EnvironmentType.SageMakerTrainingJob
    mock_run_env.load.return_value = rv

    # No Run object is created else where
    sagemaker_session.sagemaker_client.describe_training_job.return_value = {
        "TrainingJobName": "train-job-experiments",
    }

    with pytest.raises(RuntimeError) as err:
        with Run.load(sagemaker_session=sagemaker_session):
            pass

    assert "Not able to fetch RunName in ExperimentConfig of the sagemaker job" in str(err)


def test_run_load_no_run_name_and_not_in_train_job(run_obj, sagemaker_session):
    with run_obj:
        with Run.load(sagemaker_session=sagemaker_session) as run:
            assert run_obj == run


def test_run_load_no_run_name_and_not_in_train_job_but_no_obj_in_context(sagemaker_session):
    with pytest.raises(RuntimeError) as err:
        with Run.load(sagemaker_session=sagemaker_session):
            pass

    assert "Failed to load a Run object" in str(err)

    # experiment_name is given but is not supplied along with the run_name so it's ignored.
    with pytest.raises(RuntimeError) as err:
        with Run.load(experiment_name=TEST_EXP_NAME, sagemaker_session=sagemaker_session):
            pass

    assert "Failed to load a Run object" in str(err)


@patch.object(_TrialComponent, "save", MagicMock(return_value=None))
@patch(
    "sagemaker.experiments.run._TrialComponent.load",
    MagicMock(side_effect=mock_trial_component_load_func),
)
@patch("sagemaker.experiments.run._TrialComponent.search")
def test_run_load_with_run_name(mock_tc_search, sagemaker_session):
    mock_tc_search.return_value = [
        TrialComponentSearchResult(
            trial_component_name=TEST_RUN_NAME,
            parents=[
                Parent(
                    experiment_name=TEST_EXP_NAME,
                    trial_name=TEST_RUN_GRP_NAME,
                    run_name=TEST_RUN_NAME,
                ),
                Parent(
                    experiment_name=f"{TEST_EXP_NAME}-2",
                    trial_name=f"{TEST_RUN_GRP_NAME}-2",
                    run_name=TEST_RUN_NAME,
                ),
            ],
        )
    ]
    # No experiment_name or run_group_name is given.
    # Will fetch the first parent from the TC search results
    with Run.load(
        run_name=TEST_RUN_NAME,
        sagemaker_session=sagemaker_session,
    ) as run_obj:
        assert run_obj._in_load
        assert run_obj._inside_load_context
        assert not run_obj._inside_init_context
        assert run_obj.run_name == TEST_RUN_NAME
        assert run_obj._trial_component.trial_component_name == run_obj.run_name
        assert run_obj.run_group_name == TEST_RUN_GRP_NAME
        assert not run_obj._trial
        assert run_obj.experiment_name == TEST_EXP_NAME
        assert not run_obj._experiment
        assert run_obj.experiment_config == {
            EXPERIMENT_NAME: TEST_EXP_NAME,
            TRIAL_NAME: TEST_RUN_GRP_NAME,
            RUN_NAME: TEST_RUN_NAME,
        }


@patch.object(_TrialComponent, "save", MagicMock(return_value=None))
@patch(
    "sagemaker.experiments.run._TrialComponent.load",
    MagicMock(side_effect=mock_trial_component_load_func),
)
@patch("sagemaker.experiments.run._TrialComponent.search")
def test_run_load_with_run_name_and_exp_name(mock_tc_search, sagemaker_session):
    mock_tc_search.return_value = [
        TrialComponentSearchResult(
            trial_component_name=TEST_RUN_NAME,
            parents=[
                Parent(
                    experiment_name=TEST_EXP_NAME,
                    trial_name=TEST_RUN_GRP_NAME,
                    run_name=TEST_RUN_NAME,
                ),
                Parent(
                    experiment_name=f"{TEST_EXP_NAME}-2",
                    trial_name=f"{TEST_RUN_GRP_NAME}-2",
                    run_name=TEST_RUN_NAME,
                ),
                Parent(
                    experiment_name=f"{TEST_EXP_NAME}-2",
                    trial_name=f"{TEST_RUN_GRP_NAME}-2-2",
                    run_name=TEST_RUN_NAME,
                ),
            ],
        )
    ]
    # Specify the run_name and experiment_name.
    # Will fetch the first parent matching the given experiment_name
    # from the TC search results.
    with Run.load(
        run_name=TEST_RUN_NAME,
        experiment_name=f"{TEST_EXP_NAME}-2",
        sagemaker_session=sagemaker_session,
    ) as run_obj:
        assert run_obj.run_name == TEST_RUN_NAME
        assert run_obj.run_group_name == f"{TEST_RUN_GRP_NAME}-2"
        assert run_obj.experiment_name == f"{TEST_EXP_NAME}-2"

    # Specify the run_name, experiment_name and run_group_name.
    # Will fetch the parent matches all of them from TC search results.
    with Run.load(
        run_name=TEST_RUN_NAME,
        experiment_name=f"{TEST_EXP_NAME}-2",
        run_group_name=f"{TEST_RUN_GRP_NAME}-2-2",
        sagemaker_session=sagemaker_session,
    ) as run_obj:
        assert run_obj.run_name == TEST_RUN_NAME
        assert run_obj.run_group_name == f"{TEST_RUN_GRP_NAME}-2-2"
        assert run_obj.experiment_name == f"{TEST_EXP_NAME}-2"

    # Specify the run_name, experiment_name and run_group_name.
    # But the run_group_name does not exist
    name_not_exist = "NAME-NOT-EXIST"
    with pytest.raises(ValueError) as err:
        with Run.load(
            run_name=TEST_RUN_NAME,
            experiment_name=f"{TEST_EXP_NAME}-2",
            run_group_name=name_not_exist,
            sagemaker_session=sagemaker_session,
        ):
            pass

    expected_err_msg = (
        f"Not able to load the Run object given the supplied experiment_name ({TEST_EXP_NAME}-2), "
        f"run_group_name ({name_not_exist}),  run_name ({TEST_RUN_NAME})."
    )
    assert expected_err_msg in str(err)


@patch(
    "sagemaker.experiments.run._TrialComponent.load",
    MagicMock(side_effect=mock_trial_component_load_func),
)
@patch("sagemaker.experiments.run._TrialComponent.search")
def test_run_load_with_run_name_but_fail_to_retrieve_exp_name(mock_tc_search, sagemaker_session):
    mock_tc_search.return_value = [
        TrialComponentSearchResult(
            trial_component_name=TEST_RUN_NAME,
            parents=[],
        )
    ]
    # No user supplied experiment_name
    with pytest.raises(ValueError) as err:
        with Run.load(
            run_name=TEST_RUN_NAME,
            sagemaker_session=sagemaker_session,
        ):
            pass

    assert f"Failed to load a Run object with name '{TEST_RUN_NAME}'" in str(err)

    # With user supplied experiment_name
    with pytest.raises(ValueError) as err:
        with Run.load(
            run_name=TEST_RUN_NAME,
            experiment_name=TEST_EXP_NAME,
            run_group_name=TEST_RUN_GRP_NAME,
            sagemaker_session=sagemaker_session,
        ):
            pass

    assert "Not able to load the Run object given the supplied" in str(err)


@patch("sagemaker.experiments.run._RunEnvironment")
def test_run_load_in_sm_processing_job(mock_run_env, sagemaker_session):
    rv = unittest.mock.Mock()
    rv.source_arn = "arn:1234"
    rv.environment_type = _environment.EnvironmentType.SageMakerProcessingJob
    mock_run_env.load.return_value = rv

    with pytest.raises(RuntimeError) as err:
        with Run.load(sagemaker_session=sagemaker_session):
            pass

    assert (
        "Experiment Run load is not currently supported "
        "in Sagemaker jobs other than the Training job"
    ) in str(err)


def test_log_parameter_outside_run_context(run_obj):
    with pytest.raises(RuntimeError) as err:
        run_obj.log_parameter("foo", "bar")
    assert "This method should be called inside context of 'with' statement" in str(err)


def test_log_parameter(run_obj):
    with run_obj:
        run_obj.log_parameter("foo", "bar")
        assert run_obj._trial_component.parameters["foo"] == "bar"
        run_obj.log_parameter("whizz", 1)
        assert run_obj._trial_component.parameters["whizz"] == 1


def test_log_parameter_skip_invalid_value(run_obj):
    with run_obj:
        run_obj.log_parameter("key", nan)
        assert "key" not in run_obj._trial_component.parameters


def test_log_parameters_outside_run_context(run_obj):
    with pytest.raises(RuntimeError) as err:
        run_obj.log_parameters({"a": "b", "c": "d", "e": 5})
    assert "This method should be called inside context of 'with' statement" in str(err)


def test_log_parameters(run_obj):
    with run_obj:
        run_obj.log_parameters({"a": "b", "c": "d", "e": 5})
        assert run_obj._trial_component.parameters == {"a": "b", "c": "d", "e": 5}


def test_log_parameters_skip_invalid_values(run_obj):
    with run_obj:
        run_obj.log_parameters({"a": "b", "c": "d", "e": 5, "f": nan})
        assert run_obj._trial_component.parameters == {"a": "b", "c": "d", "e": 5}


def test_log_input_outside_run_context(run_obj):
    with pytest.raises(RuntimeError) as err:
        run_obj.log_input("foo", "baz", "text/text")
    assert "This method should be called inside context of 'with' statement" in str(err)


def test_log_input(run_obj):
    with run_obj:
        run_obj.log_input("foo", "baz", "text/text")
        assert run_obj._trial_component.input_artifacts == {
            "foo": TrialComponentArtifact(value="baz", media_type="text/text")
        }


def test_log_output_outside_run_context(run_obj):
    with pytest.raises(RuntimeError) as err:
        run_obj.log_output("foo", "baz", "text/text")
    assert "This method should be called inside context of 'with' statement" in str(err)


def test_log_output(run_obj):
    with run_obj:
        run_obj.log_output("foo", "baz", "text/text")
        assert run_obj._trial_component.output_artifacts == {
            "foo": TrialComponentArtifact(value="baz", media_type="text/text")
        }


def test_log_metric_outside_run_context(run_obj):
    with pytest.raises(RuntimeError) as err:
        run_obj.log_metric(name="foo", value=1.0, step=1)
    assert "This method should be called inside context of 'with' statement" in str(err)


def test_log_metric(run_obj):
    now = datetime.datetime.now()
    with run_obj:
        run_obj.log_metric(name="foo", value=1.0, step=1, timestamp=now)
        run_obj._metrics_manager.log_metric.assert_called_with(
            metric_name="foo", value=1.0, step=1, timestamp=now
        )


def test_log_metric_skip_invalid_value(run_obj):
    with run_obj:
        run_obj.log_metric(None, nan, None, None)
        assert not run_obj._metrics_manager.log_metric.called


def test_log_metric_attribute_error(run_obj):
    now = datetime.datetime.now()
    with run_obj:
        run_obj._metrics_manager.log_metric.side_effect = AttributeError

        with pytest.raises(AttributeError):
            run_obj.log_metric("foo", 1.0, 1, now)


def test_log_output_artifact_outside_run_context(run_obj):
    with pytest.raises(RuntimeError) as err:
        run_obj.log_artifact_file("foo.txt", "name", "whizz/bang")
    assert "This method should be called inside context of 'with' statement" in str(err)


def test_log_output_artifact(run_obj):
    run_obj._artifact_uploader.upload_artifact.return_value = ("s3uri_value", "etag_value")
    with run_obj:
        run_obj.log_artifact_file("foo.txt", "name", "whizz/bang")
        run_obj._artifact_uploader.upload_artifact.assert_called_with("foo.txt")
        assert "whizz/bang" == run_obj._trial_component.output_artifacts["name"].media_type

        run_obj.log_artifact_file("foo.txt")
        run_obj._artifact_uploader.upload_artifact.assert_called_with("foo.txt")
        assert "foo.txt" in run_obj._trial_component.output_artifacts
        assert "text/plain" == run_obj._trial_component.output_artifacts["foo.txt"].media_type


def test_log_input_artifact_outside_run_context(run_obj):
    with pytest.raises(RuntimeError) as err:
        run_obj.log_artifact_file("foo.txt", "name", "whizz/bang", is_output=False)
    assert "This method should be called inside context of 'with' statement" in str(err)


def test_log_input_artifact(run_obj):
    run_obj._artifact_uploader.upload_artifact.return_value = ("s3uri_value", "etag_value")
    with run_obj:
        run_obj.log_artifact_file("foo.txt", "name", "whizz/bang", is_output=False)
        run_obj._artifact_uploader.upload_artifact.assert_called_with("foo.txt")
        assert "whizz/bang" == run_obj._trial_component.input_artifacts["name"].media_type

        run_obj.log_artifact_file("foo.txt", is_output=False)
        run_obj._artifact_uploader.upload_artifact.assert_called_with("foo.txt")
        assert "foo.txt" in run_obj._trial_component.input_artifacts
        assert "text/plain" == run_obj._trial_component.input_artifacts["foo.txt"].media_type


def test_log_lineage_output_artifact_outside_run_context(run_obj):
    with pytest.raises(RuntimeError) as err:
        run_obj.log_lineage_artifact("foo.txt", "name", "whizz/bang")
    assert "This method should be called inside context of 'with' statement" in str(err)


def test_log_lineage_output_artifact(run_obj):
    run_obj._artifact_uploader.upload_artifact.return_value = ("s3uri_value", "etag_value")
    with run_obj:
        run_obj.log_lineage_artifact("foo.txt", "name", "whizz/bang")
        run_obj._artifact_uploader.upload_artifact.assert_called_with("foo.txt")
        run_obj._lineage_artifact_tracker.add_output_artifact.assert_called_with(
            "name", "s3uri_value", "etag_value", "whizz/bang"
        )

        run_obj.log_lineage_artifact("foo.txt")
        run_obj._artifact_uploader.upload_artifact.assert_called_with("foo.txt")
        run_obj._lineage_artifact_tracker.add_output_artifact.assert_called_with(
            "foo.txt", "s3uri_value", "etag_value", "text/plain"
        )


def test_log_lineage_input_artifact_outside_run_context(run_obj):
    with pytest.raises(RuntimeError) as err:
        run_obj.log_lineage_artifact("foo.txt", "name", "whizz/bang", is_output=False)
    assert "This method should be called inside context of 'with' statement" in str(err)


def test_log_lineage_input_artifact(run_obj):
    run_obj._artifact_uploader.upload_artifact.return_value = ("s3uri_value", "etag_value")
    with run_obj:
        run_obj.log_lineage_artifact("foo.txt", "name", "whizz/bang", is_output=False)
        run_obj._artifact_uploader.upload_artifact.assert_called_with("foo.txt")
        run_obj._lineage_artifact_tracker.add_input_artifact.assert_called_with(
            "name", "s3uri_value", "etag_value", "whizz/bang"
        )

        run_obj.log_lineage_artifact("foo.txt", is_output=False)
        run_obj._artifact_uploader.upload_artifact.assert_called_with("foo.txt")
        run_obj._lineage_artifact_tracker.add_input_artifact.assert_called_with(
            "foo.txt", "s3uri_value", "etag_value", "text/plain"
        )


def test_log_multiple_inputs(run_obj):
    with run_obj:
        for index in range(0, MAX_RUN_TC_ARTIFACTS_LEN):
            file_path = "foo" + str(index) + ".txt"
            run_obj._trial_component.input_artifacts[file_path] = {
                "foo": TrialComponentArtifact(value="baz" + str(index), media_type="text/text")
            }
        with pytest.raises(ValueError) as error:
            run_obj.log_input("foo.txt", "name", "whizz/bang")
        assert f"Cannot add more than {MAX_RUN_TC_ARTIFACTS_LEN} input_artifacts" in str(error)


def test_log_multiple_outputs(run_obj):
    with run_obj:
        for index in range(0, MAX_RUN_TC_ARTIFACTS_LEN):
            file_path = "foo" + str(index) + ".txt"
            run_obj._trial_component.output_artifacts[file_path] = {
                "foo": TrialComponentArtifact(value="baz" + str(index), media_type="text/text")
            }
        with pytest.raises(ValueError) as error:
            run_obj.log_output("foo.txt", "name", "whizz/bang")
        assert f"Cannot add more than {MAX_RUN_TC_ARTIFACTS_LEN} output_artifacts" in str(error)


def test_log_multiple_input_artifacts(run_obj):
    with run_obj:
        for index in range(0, MAX_RUN_TC_ARTIFACTS_LEN):
            file_path = "foo" + str(index) + ".txt"
            run_obj._artifact_uploader.upload_artifact.return_value = (
                "s3uri_value" + str(index),
                "etag_value" + str(index),
            )
            run_obj.log_artifact_file(
                file_path, "name" + str(index), "whizz/bang" + str(index), is_output=False
            )
            run_obj._artifact_uploader.upload_artifact.assert_called_with(file_path)

        run_obj._artifact_uploader.upload_artifact.return_value = (
            "s3uri_value",
            "etag_value",
        )

        # log an output artifact, should be fine
        run_obj.log_artifact_file("foo.txt", "name", "whizz/bang", is_output=True)

        # log an extra input artifact, should raise exception
        with pytest.raises(ValueError) as error:
            run_obj.log_artifact_file("foo.txt", "name", "whizz/bang", is_output=False)
        assert f"Cannot add more than {MAX_RUN_TC_ARTIFACTS_LEN} input_artifacts" in str(error)


def test_log_multiple_output_artifacts(run_obj):
    with run_obj:
        for index in range(0, MAX_RUN_TC_ARTIFACTS_LEN):
            file_path = "foo" + str(index) + ".txt"
            run_obj._artifact_uploader.upload_artifact.return_value = (
                "s3uri_value" + str(index),
                "etag_value" + str(index),
            )
            run_obj.log_artifact_file(file_path, "name" + str(index), "whizz/bang" + str(index))
            run_obj._artifact_uploader.upload_artifact.assert_called_with(file_path)

        run_obj._artifact_uploader.upload_artifact.return_value = (
            "s3uri_value",
            "etag_value",
        )

        # log an input artifact, should be fine
        run_obj.log_artifact_file("foo.txt", "name", "whizz/bang", is_output=False)

        # log an extra output artifact, should raise exception
        with pytest.raises(ValueError) as error:
            run_obj.log_artifact_file("foo.txt", "name", "whizz/bang")
        assert f"Cannot add more than {MAX_RUN_TC_ARTIFACTS_LEN} output_artifacts" in str(error)


def test_log_precision_recall_outside_run_context(run_obj):
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8]
    no_skill = 0.1
    title = "TestPrecisionRecall"

    with pytest.raises(RuntimeError) as err:
        run_obj.log_precision_recall(
            y_true, y_scores, 0, title=title, no_skill=no_skill, is_output=False
        )
    assert "This method should be called inside context of 'with' statement" in str(err)


def test_log_precision_recall(run_obj):
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8]
    no_skill = 0.1
    title = "TestPrecisionRecall"

    run_obj._artifact_uploader.upload_object_artifact.return_value = (
        "s3uri_value",
        "etag_value",
    )
    with run_obj:
        run_obj.log_precision_recall(
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
        run_obj._artifact_uploader.upload_object_artifact.assert_called_with(
            title, expected_data, file_extension="json"
        )

        run_obj._lineage_artifact_tracker.add_input_artifact.assert_called_with(
            title, "s3uri_value", "etag_value", "PrecisionRecallCurve"
        )


def test_log_precision_recall_invalid_input(run_obj):
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35]
    no_skill = 0.1

    with run_obj:
        with pytest.raises(ValueError) as error:
            run_obj.log_precision_recall(
                y_true, y_scores, 0, title="TestPrecisionRecall", no_skill=no_skill, is_output=False
            )
        assert "Lengths mismatch between true labels and predicted probabilities" in str(error)


def test_log_confusion_matrix_outside_run_context(run_obj):
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]

    with pytest.raises(RuntimeError) as err:
        run_obj.log_confusion_matrix(y_true, y_pred, title="TestConfusionMatrix")
    assert "This method should be called inside context of 'with' statement" in str(err)


def test_log_confusion_matrix(run_obj):
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]

    run_obj._artifact_uploader.upload_object_artifact.return_value = (
        "s3uri_value",
        "etag_value",
    )
    with run_obj:
        run_obj.log_confusion_matrix(y_true, y_pred, title="TestConfusionMatrix")

        expected_data = {
            "type": "ConfusionMatrix",
            "version": 0,
            "title": "TestConfusionMatrix",
            "confusionMatrix": [[2, 0, 0], [0, 0, 1], [1, 0, 2]],
        }

        run_obj._artifact_uploader.upload_object_artifact.assert_called_with(
            "TestConfusionMatrix", expected_data, file_extension="json"
        )

        run_obj._lineage_artifact_tracker.add_output_artifact.assert_called_with(
            "TestConfusionMatrix", "s3uri_value", "etag_value", "ConfusionMatrix"
        )


def test_log_confusion_matrix_invalid_input(run_obj):
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0]

    with run_obj:
        with pytest.raises(ValueError) as error:
            run_obj.log_confusion_matrix(y_true, y_pred, title="TestConfusionMatrix")
        assert "Lengths mismatch between true labels and predicted labels" in str(error)


def test_log_table_outside_run_context(run_obj):
    values = {"x": [1, 2, 3], "y": [4, 5, 6]}

    with pytest.raises(RuntimeError) as err:
        run_obj.log_table(title="TestTable", values=values, is_output=False)
    assert "This method should be called inside context of 'with' statement" in str(err)


def test_log_table_both_specified(run_obj):
    with run_obj:
        with pytest.raises(ValueError) as error:
            run_obj.log_table(title="test", values={"foo": "bar"}, data_frame={"foo": "bar"})
        assert "either values or data_frame should be provided" in str(error)


def test_log_table_neither_specified(run_obj):
    with run_obj:
        with pytest.raises(ValueError) as error:
            run_obj.log_table(title="test")
        assert "either values or data_frame should be provided" in str(error)


def test_log_table_invalid_values(run_obj):
    values = {"x": "foo", "y": [4, 5, 6]}

    with run_obj:
        with pytest.raises(ValueError) as error:
            run_obj.log_table(title="test", values=values)
        assert "Table values should be list" in str(error)


def test_log_table(run_obj):
    values = {"x": [1, 2, 3], "y": [4, 5, 6]}

    run_obj._artifact_uploader.upload_object_artifact.return_value = (
        "s3uri_value",
        "etag_value",
    )
    with run_obj:
        run_obj.log_table(title="TestTable", values=values, is_output=False)
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
        run_obj._artifact_uploader.upload_object_artifact.assert_called_with(
            "TestTable", expected_data, file_extension="json"
        )

        run_obj._lineage_artifact_tracker.add_input_artifact.assert_called_with(
            "TestTable", "s3uri_value", "etag_value", "Table"
        )


def test_log_table_dataframe(run_obj):
    dataframe = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    run_obj._artifact_uploader.upload_object_artifact.return_value = (
        "s3uri_value",
        "etag_value",
    )
    with run_obj:
        run_obj.log_table(title="TestTable", data_frame=dataframe)
        expected_data = {
            "type": "Table",
            "version": 0,
            "title": "TestTable",
            "fields": [{"name": "x", "type": "number"}, {"name": "y", "type": "number"}],
            "data": {"x": [1, 2, 3], "y": [4, 5, 6]},
        }
        run_obj._artifact_uploader.upload_object_artifact.assert_called_with(
            "TestTable", expected_data, file_extension="json"
        )

        run_obj._lineage_artifact_tracker.add_output_artifact.assert_called_with(
            "TestTable", "s3uri_value", "etag_value", "Table"
        )


def test_log_roc_curve_outside_run_context(run_obj):
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8]

    with pytest.raises(RuntimeError) as err:
        run_obj.log_roc_curve(y_true, y_scores, title="TestROCCurve", is_output=False)
    assert "This method should be called inside context of 'with' statement" in str(err)


def test_log_roc_curve(run_obj):
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8]
    with run_obj:
        run_obj._artifact_uploader.upload_object_artifact.return_value = (
            "s3uri_value",
            "etag_value",
        )

        run_obj.log_roc_curve(y_true, y_scores, title="TestROCCurve", is_output=False)

        expected_data = {
            "type": "ROCCurve",
            "version": 0,
            "title": "TestROCCurve",
            "falsePositiveRate": [0.0, 0.0, 0.5, 0.5, 1.0],
            "truePositiveRate": [0.0, 0.5, 0.5, 1.0, 1.0],
            "areaUnderCurve": 0.75,
        }
        run_obj._artifact_uploader.upload_object_artifact.assert_called_with(
            "TestROCCurve", expected_data, file_extension="json"
        )

        run_obj._lineage_artifact_tracker.add_input_artifact.assert_called_with(
            "TestROCCurve", "s3uri_value", "etag_value", "ROCCurve"
        )


def test_log_roc_curve_invalid_input(run_obj):
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35]

    with run_obj:
        with pytest.raises(ValueError) as error:
            run_obj.log_roc_curve(y_true, y_scores, title="TestROCCurve", is_output=False)
        assert "Lengths mismatch between true labels and predicted scores" in str(error)


@patch("sagemaker.experiments.run._TrialComponent.load")
@patch("sagemaker.experiments.run._TrialComponent.list")
def test_list(mock_tc_list, mock_tc_load, run_obj, sagemaker_session):
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
        experiment_name=TEST_EXP_NAME,
        sort_by="CreationTime",
        sort_order="Ascending",
        sagemaker_session=sagemaker_session,
    )

    mock_tc_list.assert_called_once_with(
        experiment_name=TEST_EXP_NAME,
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
        assert run.experiment_name == TEST_EXP_NAME
        assert run.run_name == "A" + str(i)
        assert run._experiment is None
        assert run._trial is None
        assert isinstance(run._trial_component, _TrialComponent)
        assert run._trial_component.trial_component_name == "A" + str(i)
        assert run._in_load is False
        assert run._inside_load_context is False
        assert run._inside_init_context is False
        assert run._artifact_uploader
        assert run._lineage_artifact_tracker
        assert run._metrics_manager


@patch("sagemaker.experiments.run._TrialComponent.list")
def test_list_empty(mock_tc_list, sagemaker_session):
    mock_tc_list.return_value = []
    assert [] == Run.list(experiment_name=TEST_EXP_NAME, sagemaker_session=sagemaker_session)


@patch("sagemaker.experiments.run._TrialComponent.load")
@patch("sagemaker.experiments.run._TrialComponent.search")
def test_search(mock_tc_search, mock_tc_load, run_obj, sagemaker_session):
    run_list_len = 20
    mock_tc_search.return_value = [
        TrialComponentSearchResult(
            trial_component_name=f"A{i}",
            trial_component_arn=f"B{i}",
            display_name=f"C{i}",
            parents=[
                Parent(experiment_name=f"Exp-{i}-0"),
                Parent(experiment_name=f"Exp-{i}-1"),
            ],
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
        assert run.run_group_name is None
        assert run._experiment is None
        assert run._trial is None
        assert isinstance(run._trial_component, _TrialComponent)
        assert run._trial_component.trial_component_name == f"A{int(i / 2)}"
        assert run._in_load is False
        assert run._inside_load_context is False
        assert run._inside_init_context is False
        assert run._artifact_uploader
        assert run._lineage_artifact_tracker
        assert run._metrics_manager


@patch("sagemaker.experiments.run._TrialComponent.load")
@patch("sagemaker.experiments.run._TrialComponent.search")
def test_search_empty_parents(mock_tc_search, mock_tc_load, run_obj, sagemaker_session):
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
        assert run.experiment_name == UNKNOWN_NAME
        assert run.run_group_name == UNKNOWN_NAME
        assert run.run_name == f"A{i}"
        assert run._experiment is None
        assert run._trial is None
        assert isinstance(run._trial_component, _TrialComponent)
        assert run._trial_component.trial_component_name == f"A{i}"
        assert run._in_load is False
        assert run._inside_load_context is False
        assert run._inside_init_context is False
        assert run._artifact_uploader
        assert run._lineage_artifact_tracker
        assert run._metrics_manager


@patch("sagemaker.experiments.run._TrialComponent.search")
def test_search_empty(mock_tc_search, sagemaker_session):
    mock_tc_search.return_value = []
    search_filter = Filter(name=EXPERIMENT_NAME, operator=Operator.EQUALS, value="unknown")
    search_expression = SearchExpression(filters=[search_filter])
    assert [] == Run.search(
        search_expression=search_expression, sagemaker_session=sagemaker_session
    )


def test_enter_exit_locally(sagemaker_session, run_obj):
    sagemaker_session.sagemaker_client.update_trial_component.return_value = {}
    _verify_tc_status_before_enter_init(run_obj._trial_component)

    with run_obj:
        _verify_tc_init_status(run_obj._trial_component)
        init_start_time = run_obj._trial_component.start_time

        with Run.load(sagemaker_session=sagemaker_session):
            _verify_load_does_not_change_tc_status(
                trial_component=run_obj._trial_component,
                init_start_time=init_start_time,
            )

        _verify_load_does_not_change_tc_status(
            trial_component=run_obj._trial_component,
            init_start_time=init_start_time,
        )

    _verify_tc_status_when_successfully_exit_init(run_obj._trial_component)


def test_exit_fail(sagemaker_session, run_obj):
    sagemaker_session.sagemaker_client.update_trial_component.return_value = {}
    try:
        with run_obj:
            raise ValueError("Foo")
    except ValueError:
        pass

    assert run_obj._trial_component.status.primary_status == "Failed"
    assert run_obj._trial_component.status.message
    assert isinstance(run_obj._trial_component.end_time, datetime.datetime)


@patch(
    "sagemaker.experiments.run._TrialComponent.load",
    MagicMock(side_effect=mock_trial_component_load_func),
)
@patch("sagemaker.experiments.run._RunEnvironment")
def test_enter_exit_sagemaker_job_only(mock_run_env, run_obj, sagemaker_session):
    # The Run object is initialized and loaded in job env only
    # Note this test also applies to Run object initialized locally
    # and loaded in job env as the Run.init does not depend on environment
    rv = Mock()
    rv.source_arn = "arn:1234/my-train-job"
    rv.environment_type = _environment.EnvironmentType.SageMakerTrainingJob
    mock_run_env.load.return_value = rv

    exp_config = {
        EXPERIMENT_NAME: TEST_EXP_NAME,
        TRIAL_NAME: TEST_RUN_GRP_NAME,
        RUN_NAME: TEST_RUN_NAME,
    }
    sagemaker_session.sagemaker_client.describe_training_job.return_value = {
        "TrainingJobName": "train-job-experiments",
        "ExperimentConfig": exp_config,
    }

    sagemaker_session.sagemaker_client.update_trial_component.return_value = {}
    _verify_tc_status_before_enter_init(run_obj._trial_component)

    with run_obj:
        _verify_tc_init_status(run_obj._trial_component)
        init_start_time = run_obj._trial_component.start_time

        with Run.load(sagemaker_session=sagemaker_session):
            _verify_load_does_not_change_tc_status(
                trial_component=run_obj._trial_component,
                init_start_time=init_start_time,
            )

        _verify_load_does_not_change_tc_status(
            trial_component=run_obj._trial_component,
            init_start_time=init_start_time,
        )

    _verify_tc_status_when_successfully_exit_init(run_obj._trial_component)


@pytest.mark.parametrize(
    "metric_value",
    [1.3, "nan", "inf", "-inf", None],
)
def test_is_input_valid(run_obj, metric_value):
    assert run_obj._is_input_valid("metric", "Name", metric_value)


@pytest.mark.parametrize(
    "metric_value",
    [nan, inf, -inf],
)
def test_is_input_valid_false(run_obj, metric_value):
    assert not run_obj._is_input_valid("parameter", "Name", metric_value)


def test_generate_trial_name():
    base_name = "x" * MAX_TRIAL_NAME_LEN
    trial_name = Run._generate_trial_name(base_name=base_name)
    assert len(trial_name) <= MAX_TRIAL_NAME_LEN


def _verify_tc_status_before_enter_init(trial_component):
    assert not trial_component.start_time
    assert not trial_component.end_time
    assert not trial_component.status


def _verify_tc_init_status(trial_component):
    assert isinstance(trial_component.start_time, datetime.datetime)
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    assert (now.timestamp() - trial_component.start_time.timestamp()) < 1
    assert not trial_component.end_time
    assert trial_component.status.primary_status == "InProgress"


def _verify_load_does_not_change_tc_status(trial_component, init_start_time):
    assert trial_component.start_time == init_start_time
    assert not trial_component.end_time
    assert trial_component.status.primary_status == "InProgress"


def _verify_tc_status_when_successfully_exit_init(trial_component):
    assert trial_component.status.primary_status == "Completed"
    assert isinstance(trial_component.start_time, datetime.datetime)
    assert isinstance(trial_component.end_time, datetime.datetime)
