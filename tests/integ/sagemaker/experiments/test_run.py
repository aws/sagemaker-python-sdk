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

import os

import pytest

from tests.integ import lock
from tests.integ import DATA_DIR
from sagemaker.experiments.metrics import _MetricsManager
from sagemaker.experiments.trial_component import _TrialComponent
from sagemaker.sklearn import SKLearn
from sagemaker.utils import retry_with_backoff
from tests.integ.sagemaker.experiments.helpers import name, cleanup_exp_resources
from sagemaker.experiments.run import (
    Run,
    RUN_NAME_BASE,
    DELIMITER,
    RUN_TC_TAG_KEY,
    RUN_TC_TAG_VALUE,
)
from sagemaker.experiments._helper import _DEFAULT_ARTIFACT_PREFIX


EXECUTION_ROLE = "SageMakerRole"


@pytest.fixture
def artifact_file_path(tempdir):
    file_contents = "test artifact file"
    file_path = os.path.join(tempdir, "artifact_file.txt")
    with open(file_path, "w") as foo_file:
        foo_file.write(file_contents)
    return file_path


@pytest.fixture
def artifact_directory(tempdir):
    file_contents = "test artifact file"
    file_path1 = os.path.join(tempdir, "artifact_file1.txt")
    file_path2 = os.path.join(tempdir, "artifact_file2.txt")
    with open(file_path1, "w") as f1:
        f1.write(file_contents)
    with open(file_path2, "w") as f2:
        f2.write(file_contents)
    return tempdir


@pytest.fixture
def lineage_artifact_path(tempdir):
    file_contents = "test lineage artifact"
    file_path = os.path.join(tempdir, "lineage_file.txt")
    with open(file_path, "w") as foo_file:
        foo_file.write(file_contents)
    return file_path


file_artifact_name = f"file-artifact-{name()}"
lineage_artifact_name = f"lineage-file-artifact-{name()}"
metric_name = "test-x-step"


def test_local_run_with_load_specifying_names(
    sagemaker_session, artifact_file_path, artifact_directory, lineage_artifact_path
):
    exp_name = f"my-local-exp-{name()}"
    with cleanup_exp_resources(exp_names=[exp_name], sagemaker_session=sagemaker_session):
        # Run name is not provided, will create a new TC
        with Run.init(experiment_name=exp_name, sagemaker_session=sagemaker_session) as run1:
            run1_name = run1.run_name
            assert RUN_NAME_BASE in run1_name

            run1.log_parameter("p1", 1.0)
            run1.log_parameter("p2", "p2-value")
            run1.log_parameters({"p3": 2.0, "p4": "p4-value"})

            run1.log_artifact_file(file_path=artifact_file_path, name=file_artifact_name)
            run1.log_artifact_directory(directory=artifact_directory, is_output=False)
            run1.log_lineage_artifact(file_path=lineage_artifact_path, name=lineage_artifact_name)

            for i in range(_MetricsManager._BATCH_SIZE):
                run1.log_metric(name=metric_name, value=i, step=i)

        with Run.load(
            experiment_name=exp_name,
            run_name=run1_name,
            sagemaker_session=sagemaker_session,
        ) as run2:
            assert run2.run_name == run1_name
            assert run2._trial_component.trial_component_name == f"{exp_name}{DELIMITER}{run1_name}"
            _check_run_from_local_end_result(
                sagemaker_session=sagemaker_session, tc=run2._trial_component
            )


def _check_run_from_local_end_result(sagemaker_session, tc):
    def validate_tc_artifact_association(is_output, expected_artifact_name):
        if is_output:
            # It's an output association from the tc
            response = sagemaker_session.sagemaker_client.list_associations(
                SourceArn=tc.trial_component_arn
            )
        else:
            # It's an input association to the tc
            response = sagemaker_session.sagemaker_client.list_associations(
                DestinationArn=tc.trial_component_arn
            )
        associations = response["AssociationSummaries"]

        assert len(associations) == 1
        summary = associations[0]
        if is_output:
            assert summary["SourceArn"] == tc.trial_component_arn
            assert summary["DestinationName"] == expected_artifact_name
        else:
            assert summary["DestinationArn"] == tc.trial_component_arn
            assert summary["SourceName"] == expected_artifact_name

    assert tc.parameters == {"p1": 1.0, "p2": "p2-value", "p3": 2.0, "p4": "p4-value"}

    s3_prefix = f"s3://{sagemaker_session.default_bucket()}/{_DEFAULT_ARTIFACT_PREFIX}"
    assert s3_prefix in tc.output_artifacts[file_artifact_name].value
    assert "text/plain" == tc.output_artifacts[file_artifact_name].media_type
    assert s3_prefix in tc.input_artifacts["artifact_file1"].value
    assert "text/plain" == tc.input_artifacts["artifact_file1"].media_type
    assert s3_prefix in tc.input_artifacts["artifact_file2"].value
    assert "text/plain" == tc.input_artifacts["artifact_file2"].media_type

    assert len(tc.metrics) == 1
    metric_summary = tc.metrics[0]
    assert metric_summary.metric_name == metric_name
    assert metric_summary.max == 9.0
    assert metric_summary.min == 0.0

    validate_tc_artifact_association(
        is_output=True,
        expected_artifact_name=lineage_artifact_name,
    )


def test_two_local_run_init_with_same_run_name_and_different_exp_names(sagemaker_session):
    exp_name1 = f"my-two-local-exp1-{name()}"
    exp_name2 = f"my-two-local-exp2-{name()}"
    run_name = "test-run"
    with cleanup_exp_resources(
        exp_names=[exp_name1, exp_name2], sagemaker_session=sagemaker_session
    ):
        # Run name is not provided, will create a new TC
        with Run.init(
            experiment_name=exp_name1, run_name=run_name, sagemaker_session=sagemaker_session
        ) as run1:
            pass
        with Run.init(
            experiment_name=exp_name2, run_name=run_name, sagemaker_session=sagemaker_session
        ) as run2:
            pass

        assert run1.experiment_name != run2.experiment_name
        assert run1.run_name == run2.run_name
        assert (
            run1._trial_component.trial_component_name != run2._trial_component.trial_component_name
        )
        assert run1._trial_component.trial_component_name == f"{exp_name1}{DELIMITER}{run_name}"
        assert run2._trial_component.trial_component_name == f"{exp_name2}{DELIMITER}{run_name}"


@pytest.mark.parametrize(
    "input_names",
    [
        (f"my-local-exp-{name()}", "test-run", None),  # both have delimiter -
        ("my-test-1", "my-test-1", None),  # exp_name equals run_name
        ("my-test-3", "my-test-3-run", None),  # <exp_name><delimiter> is subset of run_name
        ("x" * 59, "test-run", None),  # long exp_name
        ("test-exp", "y" * 59, None),  # long run_name
        ("x" * 59, "y" * 59, None),  # long exp_name and run_name
        ("my-test4", "test-run", "run-display-name-test"),  # with supplied display name
    ],
)
def test_run_name_vs_trial_component_name_edge_cases(
    sagemaker_session, artifact_file_path, artifact_directory, lineage_artifact_path, input_names
):
    exp_name, run_name, run_display_name = input_names
    with cleanup_exp_resources(exp_names=[exp_name], sagemaker_session=sagemaker_session):
        with Run.init(
            experiment_name=exp_name,
            sagemaker_session=sagemaker_session,
            run_name=run_name,
            run_display_name=run_display_name,
        ) as run1:
            assert not run1._experiment.tags
            assert not run1._trial.tags
            tags = run1._trial_component.tags
            assert len(tags) == 1
            assert tags[0]["Key"] == RUN_TC_TAG_KEY
            assert tags[0]["Value"] == RUN_TC_TAG_VALUE

        with Run.load(
            experiment_name=exp_name,
            run_name=run_name,
            sagemaker_session=sagemaker_session,
        ) as run2:
            assert run2.experiment_name == exp_name
            assert run2.run_name == run_name
            assert run2._trial_component.trial_component_name == f"{exp_name}{DELIMITER}{run_name}"
            assert run2._trial_component.display_name in (
                run_display_name,
                run2._trial_component.trial_component_name,
            )


# TODO: Need to update the rest of tests
# once we update the Run behavior in job env in M1
_EXP_NAME_IN_SCRIPT = "my-train-job-exp-in-script"
_RUN_NAME_IN_SCRIPT = "my-train-job-run-in-script"
_EXP_NAME_IN_LOCAL = "my-train-job-exp-in-local"
_RUN_NAME_IN_LOCAL = "my-train-job-run-in-local"
_ENTRY_POINT_PATH = os.path.join(DATA_DIR, "experiment/scripts/train_job_script_for_run_clz.py")
_PYTHON_SCRIPT_PATH = os.path.join(DATA_DIR, "experiment/scripts/launcher.sh")


@pytest.mark.skip(
    reason=(
        "Waiting for the CR https://code.amazon.com/reviews/CR-75915367/revisions/1#/details "
        "to deploy to us-west-2"
    )
)
def test_run_from_local_and_train_job_and_all_exp_cfg_match(sagemaker_session, job_resource_dir):
    # Notes:
    # 1. The 1st Run TC created locally
    # 2. In training job, the same exp and run names are given in Run.init
    # which will load the 1st Run TC in training job and log parameters
    # and metrics there
    with lock.lock():
        with cleanup_exp_resources(
            exp_names=[_EXP_NAME_IN_SCRIPT], sagemaker_session=sagemaker_session
        ):
            with Run.init(
                experiment_name=_EXP_NAME_IN_SCRIPT,
                run_name=_RUN_NAME_IN_SCRIPT,
                sagemaker_session=sagemaker_session,
            ) as run:
                estimator = SKLearn(
                    framework_version="0.23-1",
                    entry_point=_PYTHON_SCRIPT_PATH,
                    dependencies=[job_resource_dir, _ENTRY_POINT_PATH],
                    role=EXECUTION_ROLE,
                    instance_type="ml.m5.large",
                    instance_count=1,
                    volume_size=10,
                    max_run=900,
                    enable_sagemaker_metrics=True,
                    sagemaker_session=sagemaker_session,
                )
                estimator.fit(
                    job_name=f"train-job-{name()}",
                    experiment_config=run._experiment_config,
                    wait=True,  # wait the training job to finish
                    logs="None",  # set to "All" to display logs fetched from the training job
                )

            assert run.run_name == _RUN_NAME_IN_SCRIPT
            _check_run_from_job_result(
                tc_name=run.run_name,
                sagemaker_session=sagemaker_session,
            )


@pytest.mark.skip(
    reason=(
        "Waiting for the CR https://code.amazon.com/reviews/CR-75915367/revisions/1#/details "
        "to deploy to us-west-2"
    )
)
def test_run_from_local_and_train_job_and_exp_cfg_not_match(sagemaker_session, job_resource_dir):
    # Notes:
    # 1. The 1st Run TC created locally
    # 2. In training job, different exp and run names are given in Run.init
    # which will still load the 1st Run TC in training job
    # and log parameters and metrics there
    with lock.lock():
        with cleanup_exp_resources(
            exp_names=[_EXP_NAME_IN_LOCAL, _EXP_NAME_IN_SCRIPT], sagemaker_session=sagemaker_session
        ):
            with Run.init(
                experiment_name=_EXP_NAME_IN_LOCAL,
                run_name=_RUN_NAME_IN_LOCAL,
                sagemaker_session=sagemaker_session,
            ) as run:
                estimator = SKLearn(
                    framework_version="0.23-1",
                    entry_point=_PYTHON_SCRIPT_PATH,
                    dependencies=[job_resource_dir, _ENTRY_POINT_PATH],
                    role=EXECUTION_ROLE,
                    instance_type="ml.m5.large",
                    instance_count=1,
                    volume_size=10,
                    max_run=900,
                    enable_sagemaker_metrics=True,
                    sagemaker_session=sagemaker_session,
                )
                estimator.fit(
                    job_name=f"train-job-{name()}",
                    experiment_config=run._experiment_config,
                    wait=True,  # wait the training job to finish
                    logs="None",  # set to "All" to display logs fetched from the training job
                )

            assert run.run_name != _RUN_NAME_IN_SCRIPT
            _check_run_from_job_result(
                tc_name=run.run_name,
                sagemaker_session=sagemaker_session,
            )


@pytest.mark.skip(
    reason=(
        "Waiting for the CR https://code.amazon.com/reviews/CR-75915367/revisions/1#/details "
        "to deploy to us-west-2"
    )
)
def test_run_from_train_job(sagemaker_session, job_resource_dir):
    # Notes:
    # 1. No Run TC created locally or specified in experiment config
    # 2. In training job, Run.init is invoked
    # which will load/create a Run TC according to the run_name passed in there
    # 3. Both metrics and parameters are logged in the Run TC loaded/created in job
    with lock.lock():
        with cleanup_exp_resources(
            exp_names=[_EXP_NAME_IN_SCRIPT], sagemaker_session=sagemaker_session
        ):
            estimator = SKLearn(
                framework_version="0.23-1",
                entry_point=_PYTHON_SCRIPT_PATH,
                dependencies=[job_resource_dir, _ENTRY_POINT_PATH],
                role=EXECUTION_ROLE,
                instance_type="ml.m5.large",
                instance_count=1,
                volume_size=10,
                max_run=900,
                enable_sagemaker_metrics=True,
                sagemaker_session=sagemaker_session,
            )
            estimator.fit(
                job_name=f"train-job-{name()}",
                wait=True,  # wait the training job to finish
                logs="None",  # set to "All" to display logs fetched from the training job
            )
            _check_run_from_job_result(
                tc_name=_RUN_NAME_IN_SCRIPT,
                sagemaker_session=sagemaker_session,
            )


def _check_run_from_job_result(sagemaker_session, tc_name=None):
    def validate_tc():
        # parameters were logged in Run TC
        tc = _TrialComponent.load(trial_component_name=tc_name, sagemaker_session=sagemaker_session)

        assert tc.start_time
        assert tc.end_time
        assert tc.status.primary_status == "Completed"
        assert tc.parameters["p1"] == 1.0
        assert tc.parameters["p2"] == 2.0
        assert len(tc.metrics) == 5
        for metric_summary in tc.metrics:
            # metrics deletion is not supported at this point
            # so its count would accumulate
            assert metric_summary.count > 0
            assert metric_summary.min == 0.0
            assert metric_summary.max == 1.0

    # Add retry since the load behavior is inconsistent sometimes
    retry_with_backoff(validate_tc, 4)
