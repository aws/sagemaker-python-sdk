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

from unittest.mock import patch, MagicMock

import pytest

from sagemaker import Processor
from sagemaker.estimator import Estimator, _TrainingJob
from sagemaker.experiments.experiment import Experiment
from sagemaker.experiments.run import _RunContext
from sagemaker.experiments import load_run, Run
from sagemaker.experiments.trial import _Trial
from sagemaker.experiments.trial_component import _TrialComponent
from sagemaker.processing import ProcessingJob
from sagemaker.transformer import _TransformJob, Transformer
from tests.unit.sagemaker.experiments.helpers import (
    TEST_EXP_NAME,
    mock_trial_load_or_create_func,
    mock_tc_load_or_create_func,
)

_bucket = "my-bucket"
_train_input_path = f"s3://{_bucket}/data.csv"
_train_output_path = f"s3://{_bucket}"


@patch.object(_TrainingJob, "start_new")
def test_auto_pass_in_exp_config_to_train_job(mock_start_job, run_obj, sagemaker_session):
    mock_start_job.return_value = _TrainingJob(sagemaker_session, "my-job")
    with run_obj:
        estimator = Estimator(
            role="arn:my-role",
            image_uri="my-image",
            sagemaker_session=sagemaker_session,
            output_path=_train_output_path,
        )
        estimator.fit(
            inputs=_train_input_path,
            wait=False,
        )

        assert _RunContext.get_current_run() == run_obj

    expected_exp_config = run_obj.experiment_config
    mock_start_job.assert_called_once_with(estimator, _train_input_path, expected_exp_config, None)

    # _RunContext is cleaned up after exiting the with statement
    assert not _RunContext.get_current_run()


@patch.object(_Trial, "add_trial_component", MagicMock(return_value=None))
@patch.object(_TrialComponent, "save", MagicMock(return_value=None))
@patch("sagemaker.experiments.run.Experiment._load_or_create")
@patch("sagemaker.experiments.run._Trial._load_or_create")
@patch("sagemaker.experiments.run._TrialComponent._load_or_create")
@patch.object(_TrainingJob, "start_new")
def test_auto_pass_in_exp_config_under_load_run(
    mock_start_job, mock_load_tc, mock_load_trial, mock_load_exp, run_obj, sagemaker_session
):
    mock_start_job.return_value = _TrainingJob(sagemaker_session, "my-job")
    mock_load_tc.return_value = run_obj._trial_component, True
    mock_load_trial.return_value = run_obj._trial
    mock_load_exp.return_value = run_obj._experiment
    with load_run(
        run_name=run_obj.run_name,
        experiment_name=run_obj.experiment_name,
        sagemaker_session=sagemaker_session,
    ):
        estimator = Estimator(
            role="arn:my-role",
            image_uri="my-image",
            sagemaker_session=sagemaker_session,
            output_path=_train_output_path,
        )
        estimator.fit(
            inputs=_train_input_path,
            wait=False,
        )

        loaded_run = _RunContext.get_current_run()
        assert loaded_run.run_name == run_obj.run_name
        assert loaded_run.experiment_config == run_obj.experiment_config

    expected_exp_config = run_obj.experiment_config
    mock_start_job.assert_called_once_with(estimator, _train_input_path, expected_exp_config, None)

    # _RunContext is cleaned up after exiting the with statement
    assert not _RunContext.get_current_run()


@patch.object(ProcessingJob, "start_new")
def test_auto_pass_in_exp_config_to_process_job(mock_start_job, run_obj, sagemaker_session):
    mock_start_job.return_value = ProcessingJob(sagemaker_session, "my-job", [], [], "")
    with run_obj:
        processor = Processor(
            role="arn:my-role",
            image_uri="my-image",
            instance_count=1,
            instance_type="ml.m5.large",
            sagemaker_session=sagemaker_session,
        )
        processor.run(wait=False, logs=False)

        assert _RunContext.get_current_run() == run_obj

    expected_exp_config = run_obj.experiment_config
    assert mock_start_job.call_args[1]["experiment_config"] == expected_exp_config

    # _RunContext is cleaned up after exiting the with statement
    assert not _RunContext.get_current_run()


@patch.object(_TransformJob, "start_new")
def test_auto_pass_in_exp_config_to_transform_job(mock_start_job, run_obj, sagemaker_session):
    bucket_name = "my-bucket"
    job_name = "my-job"
    mock_start_job.return_value = _TransformJob(sagemaker_session, job_name)
    with run_obj:
        transformer = Transformer(
            model_name="my-model",
            instance_count=1,
            instance_type="ml.m5.large",
            output_path=f"s3://{bucket_name}/output",
            sagemaker_session=sagemaker_session,
        )
        transformer.transform(
            data=f"s3://{bucket_name}/data", wait=False, logs=False, job_name=job_name
        )

        assert _RunContext.get_current_run() == run_obj

    expected_exp_config = run_obj.experiment_config
    assert mock_start_job.call_args[0][9] == expected_exp_config

    # _RunContext is cleaned up after exiting the with statement
    assert not _RunContext.get_current_run()


# TODO: add unit test for test_auto_pass_in_exp_config_to_tuning_job once ready


@patch.object(_TrainingJob, "start_new")
def test_user_supply_exp_config_to_train_job(mock_start_job, run_obj, sagemaker_session):
    mock_start_job.return_value = _TrainingJob(sagemaker_session, "my-job")
    supplied_exp_cfg = {
        "ExperimentName": "my-supplied-exp-name",
        "TrialName": "my-supplied-run-group-name",
        "RunName": "my-supplied-run-name",
    }
    with run_obj:
        estimator = Estimator(
            role="arn:my-role",
            image_uri="my-image",
            sagemaker_session=sagemaker_session,
            output_path=_train_output_path,
        )
        estimator.fit(
            experiment_config=supplied_exp_cfg,
            inputs=_train_input_path,
            wait=False,
        )

        assert _RunContext.get_current_run() == run_obj

    mock_start_job.assert_called_once_with(estimator, _train_input_path, supplied_exp_cfg, None)

    # _RunContext is cleaned up after exiting the with statement
    assert not _RunContext.get_current_run()


def test_auto_fetch_created_run_obj_from_context(run_obj, sagemaker_session):
    assert not run_obj._inside_init_context
    assert not run_obj._inside_load_context
    assert not run_obj._in_load
    assert not _RunContext.get_current_run()

    def train():
        with load_run(sagemaker_session=sagemaker_session) as run_load:
            assert run_load == run_obj
            assert _RunContext.get_current_run() == run_obj
            assert run_obj._inside_init_context
            assert run_obj._inside_load_context
            assert run_obj._in_load

            run_load.log_parameter("foo", "bar")
            run_load.log_parameter("whizz", 1)

    with run_obj:
        assert run_obj._inside_init_context
        assert not run_obj._inside_load_context
        assert not run_obj._in_load
        assert _RunContext.get_current_run() == run_obj

        train()

        assert run_obj._inside_init_context
        assert not run_obj._inside_load_context
        assert not run_obj._in_load
        assert _RunContext.get_current_run() == run_obj

        run_obj.log_parameters({"a": "b", "c": 2})

        assert run_obj._trial_component.parameters["foo"] == "bar"
        assert run_obj._trial_component.parameters["whizz"] == 1
        assert run_obj._trial_component.parameters["a"] == "b"
        assert run_obj._trial_component.parameters["c"] == 2

        # Verify separating load_run and with statement in different lines still work
        run_load2 = load_run(sagemaker_session=sagemaker_session)
        with run_load2:
            assert run_load2 == run_obj
            assert run_obj._inside_init_context
            assert run_obj._inside_load_context
            assert run_obj._in_load

        assert run_obj._inside_init_context
        assert not run_obj._inside_load_context
        assert not run_obj._in_load
        assert _RunContext.get_current_run() == run_obj

    assert not run_obj._inside_init_context
    assert not run_obj._inside_load_context
    assert not run_obj._in_load
    assert not _RunContext.get_current_run()


def test_nested_run_init_context_on_same_run_object(run_obj, sagemaker_session):
    assert not _RunContext.get_current_run()

    with pytest.raises(RuntimeError) as err:
        with run_obj:
            assert _RunContext.get_current_run()

            with run_obj:
                pass
    assert "It is not allowed to use nested 'with' statements on the Run" in str(err)


@patch(
    "sagemaker.experiments.run.Experiment._load_or_create",
    MagicMock(return_value=Experiment(experiment_name=TEST_EXP_NAME)),
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
def test_nested_run_init_context_on_different_run_object(run_obj, sagemaker_session):
    assert not _RunContext.get_current_run()

    with pytest.raises(RuntimeError) as err:
        with Run(experiment_name=TEST_EXP_NAME, sagemaker_session=sagemaker_session):
            assert _RunContext.get_current_run()

            with run_obj:
                pass
    assert "It is not allowed to use nested 'with' statements on the Run" in str(err)

    with pytest.raises(RuntimeError) as err:
        with Run(experiment_name=TEST_EXP_NAME, sagemaker_session=sagemaker_session):
            assert _RunContext.get_current_run()

            with Run(experiment_name=TEST_EXP_NAME, sagemaker_session=sagemaker_session):
                pass
    assert "It is not allowed to use nested 'with' statements on the Run" in str(err)


def test_nested_run_load_context(run_obj, sagemaker_session):
    assert not _RunContext.get_current_run()

    with pytest.raises(RuntimeError) as err:
        with run_obj:
            assert _RunContext.get_current_run()

            with load_run():
                run_load = load_run()
                with run_load:
                    pass
    assert "It is not allowed to use nested 'with' statements on the load_run" in str(err)

    with pytest.raises(RuntimeError) as err:
        with run_obj:
            assert _RunContext.get_current_run()

            with load_run():
                with load_run():
                    pass
    assert "It is not allowed to use nested 'with' statements on the load_run" in str(err)


@patch.object(_Trial, "add_trial_component", MagicMock(return_value=None))
@patch("sagemaker.experiments.run.Experiment._load_or_create")
@patch("sagemaker.experiments.run._Trial._load_or_create")
@patch("sagemaker.experiments.run._TrialComponent._load_or_create")
def test_run_init_under_run_load_context(
    mock_load_tc, mock_load_trial, mock_load_exp, run_obj, sagemaker_session
):
    mock_load_tc.return_value = run_obj._trial_component, True
    mock_load_trial.return_value = run_obj._trial
    mock_load_exp.return_value = run_obj._experiment
    assert not _RunContext.get_current_run()

    with pytest.raises(RuntimeError) as err:
        with load_run(
            run_name=run_obj.run_name,
            experiment_name=run_obj.experiment_name,
            sagemaker_session=sagemaker_session,
        ):
            assert _RunContext.get_current_run()

            with Run(experiment_name=TEST_EXP_NAME, sagemaker_session=sagemaker_session):
                pass
    assert "It is not allowed to use nested 'with' statements on the Run" in str(err)
