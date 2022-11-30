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
import os

import pytest

from sagemaker.experiments._api_types import _TrialComponentStatusType
from sagemaker.processing import FrameworkProcessor
from sagemaker.pytorch import PyTorch
from sagemaker.utilities.search_expression import Filter, Operator, SearchExpression
from tests.integ import DATA_DIR
from sagemaker.experiments.metrics import _MetricsManager
from sagemaker.experiments.trial_component import _TrialComponent
from sagemaker.sklearn import SKLearn
from sagemaker.utils import retry_with_backoff, unique_name_from_base
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
metric_name = "test-local-init-log-metric"


def test_local_run_with_load(
    sagemaker_session, artifact_file_path, artifact_directory, lineage_artifact_path
):
    exp_name = f"my-local-exp-{name()}"
    with cleanup_exp_resources(exp_names=[exp_name], sagemaker_session=sagemaker_session):
        # Run name is not provided, will create a new TC
        with Run.init(experiment_name=exp_name, sagemaker_session=sagemaker_session) as run1:
            run1_name = run1.run_name
            assert RUN_NAME_BASE in run1_name
            _local_run_log_behaviors(
                artifact_file_path=artifact_file_path,
                artifact_directory=artifact_directory,
                lineage_artifact_path=lineage_artifact_path,
                sagemaker_session=sagemaker_session,
            )

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
        ("e" * 59, "y" * 59, None),  # long exp_name and run_name
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
            _check_run_trial_component_tags(
                trial_component=run1._trial_component, sagemaker_session=sagemaker_session
            )

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


_EXP_NAME_BASE_IN_SCRIPT = "train-job-exp-in-script"
_RUN_NAME_IN_SCRIPT = "train-job-run-in-script"

_EXP_DIR = os.path.join(DATA_DIR, "experiment")
_ENTRY_POINT_PATH = os.path.join(_EXP_DIR, "scripts/launcher.sh")
_PYTHON_TRAIN_SCRIPT_PATH = os.path.join(_EXP_DIR, "scripts/train_job_script_for_run_clz.py")
_PYTHON_PROCESS_SCRIPT = "process_job_script_for_run_clz.py"

_RUN_INIT = "init"
_RUN_LOAD = "load"


def test_run_from_local_and_train_job_and_all_exp_cfg_match(sagemaker_session, job_resource_dir):
    # Notes:
    # 1. The 1st Run TC created locally and its exp config was auto passed to the job
    # 2. In training job, the same exp and run names are given in Run.init
    # which will load the 1st Run TC in training job and log parameters
    # and metrics there
    # 3. In a different training job, load the same Run TC and log more parameters there.
    exp_name = unique_name_from_base(_EXP_NAME_BASE_IN_SCRIPT)
    estimator = _generate_estimator(
        job_resource_dir=job_resource_dir, sagemaker_session=sagemaker_session, exp_name=exp_name
    )
    tc_name = Run._generate_trial_component_name(
        experiment_name=exp_name, run_name=_RUN_NAME_IN_SCRIPT
    )

    with cleanup_exp_resources(exp_names=[exp_name], sagemaker_session=sagemaker_session):
        with Run.init(
            experiment_name=exp_name,
            run_name=_RUN_NAME_IN_SCRIPT,
            sagemaker_session=sagemaker_session,
        ) as run:
            init_start_time = _check_tc_status_when_entering(run._trial_component)
            _local_run_log_behaviors(is_complete_log=False, sagemaker_session=sagemaker_session)
            # experiment_config is auto passed in by _RunContext
            estimator.fit(
                job_name=f"train-job-{name()}",
                wait=True,  # wait the training job to finish
                logs="None",  # set to "All" to display logs fetched from the training job
            )
            old_end_time = _check_tc_status_when_exiting(
                trial_component_name=run._trial_component.trial_component_name,
                init_start_time=init_start_time,
                sagemaker_session=sagemaker_session,
            )

        _check_tc_status_when_exiting(
            trial_component_name=run._trial_component.trial_component_name,
            init_start_time=init_start_time,
            old_end_time=old_end_time,
            sagemaker_session=sagemaker_session,
        )
        assert run.experiment_name == exp_name
        assert run.run_name == _RUN_NAME_IN_SCRIPT
        _check_run_from_local_end_result(
            tc=run._trial_component,
            sagemaker_session=sagemaker_session,
            is_complete_log=False,
        )
        _check_run_from_job_result(
            tc_name=tc_name,
            sagemaker_session=sagemaker_session,
        )

        with run:
            estimator.environment["RUN_OPERATION"] = _RUN_LOAD
            estimator.environment["CALL_RUN_LOAD_WITH_NO_NAME_ARGS"] = "True"
            estimator.fit(
                job_name=f"train-job-{name()}",
                wait=True,  # wait the training job to finish
                logs="None",  # set to "All" to display logs fetched from the training job
            )

            old_end_time = _check_tc_status_when_exiting(
                trial_component_name=run._trial_component.trial_component_name,
                init_start_time=init_start_time,
                old_end_time=old_end_time,
                sagemaker_session=sagemaker_session,
            )

        _check_tc_status_when_exiting(
            trial_component_name=run._trial_component.trial_component_name,
            init_start_time=init_start_time,
            old_end_time=old_end_time,
            sagemaker_session=sagemaker_session,
        )
        _check_run_from_job_result(
            tc_name=tc_name,
            sagemaker_session=sagemaker_session,
            is_init=False,
            has_extra_load=True,
        )


def test_run_from_local_and_train_job_and_exp_cfg_not_match(sagemaker_session, job_resource_dir):
    # Notes:
    # 1. The 1st Run TC created locally and its exp config was auto passed to the job
    # 2. In training job, different exp and run names (i.e. 2nd Run TC) are given
    # in Run.init which will create a Run TC according to the run_name
    # passed in there and ignore the exp config in the job
    # 3. Both metrics and parameters are logged in the Run TC created in job
    # 4. In a different training job, load the 2nd Run TC and log more parameters there.
    exp_name = unique_name_from_base(_EXP_NAME_BASE_IN_SCRIPT)
    exp_name2 = unique_name_from_base(_EXP_NAME_BASE_IN_SCRIPT)
    estimator = _generate_estimator(
        job_resource_dir=job_resource_dir, sagemaker_session=sagemaker_session, exp_name=exp_name
    )
    tc_name = Run._generate_trial_component_name(
        experiment_name=exp_name, run_name=_RUN_NAME_IN_SCRIPT
    )

    with cleanup_exp_resources(
        exp_names=[exp_name, exp_name2], sagemaker_session=sagemaker_session
    ):
        with Run.init(
            experiment_name=exp_name2,
            run_name=f"{_RUN_NAME_IN_SCRIPT}2",
            sagemaker_session=sagemaker_session,
        ) as run:
            init_start_time = _check_tc_status_when_entering(run._trial_component)
            # experiment_config is auto passed in by _RunContext
            estimator.fit(
                job_name=f"train-job-{name()}",
                wait=True,  # wait the training job to finish
                logs="None",  # set to "All" to display logs fetched from the training job
            )
            _check_tc_status_intermediate(
                trial_component=run._trial_component,
                sagemaker_session=sagemaker_session,
                init_start_time=init_start_time,
            )

        old_end_time = _check_tc_status_when_exiting(
            trial_component_name=run._trial_component.trial_component_name,
            init_start_time=init_start_time,
            sagemaker_session=sagemaker_session,
        )
        assert run.experiment_name != exp_name
        assert run.run_name != _RUN_NAME_IN_SCRIPT
        _check_run_from_job_result(
            tc_name=tc_name,
            sagemaker_session=sagemaker_session,
        )

        with run:
            estimator.environment["RUN_OPERATION"] = _RUN_LOAD
            estimator.fit(
                job_name=f"train-job-{name()}",
                wait=True,  # wait the training job to finish
                logs="None",  # set to "All" to display logs fetched from the training job
            )
            _check_tc_status_intermediate(
                trial_component=run._trial_component,
                sagemaker_session=sagemaker_session,
                init_start_time=init_start_time,
                old_end_time=old_end_time,
            )

        _check_tc_status_when_exiting(
            trial_component_name=run._trial_component.trial_component_name,
            init_start_time=init_start_time,
            old_end_time=old_end_time,
            sagemaker_session=sagemaker_session,
        )
        _check_run_from_job_result(
            tc_name=tc_name, sagemaker_session=sagemaker_session, is_init=False
        )


def test_run_from_train_job_only(sagemaker_session, job_resource_dir):
    # Notes:
    # 1. No Run TC created locally or specified in experiment config
    # 2. In training job, Run.init is invoked
    # which will create a Run TC according to the run_name passed in there
    # 3. Both metrics and parameters are logged in the Run TC created in job
    # 4. In a different training job, load the same Run TC and log more parameters there.
    exp_name = unique_name_from_base(_EXP_NAME_BASE_IN_SCRIPT)
    estimator = _generate_estimator(
        job_resource_dir=job_resource_dir,
        sagemaker_session=sagemaker_session,
        exp_name=exp_name,
    )
    tc_name = Run._generate_trial_component_name(
        experiment_name=exp_name, run_name=_RUN_NAME_IN_SCRIPT
    )

    with cleanup_exp_resources(exp_names=[exp_name], sagemaker_session=sagemaker_session):
        estimator.fit(
            job_name=f"train-job-{name()}",
            wait=True,  # wait the training job to finish
            logs="None",  # set to "All" to display logs fetched from the training job
        )
        _check_run_from_job_result(
            tc_name=tc_name,
            sagemaker_session=sagemaker_session,
        )

        estimator.environment["RUN_OPERATION"] = _RUN_LOAD
        estimator.fit(
            job_name=f"train-job-{name()}",
            wait=True,  # wait the training job to finish
            logs="None",  # set to "All" to display logs fetched from the training job
        )
        _check_run_from_job_result(
            tc_name=tc_name, sagemaker_session=sagemaker_session, is_init=False
        )


def test_run_from_processing_job_and_override_default_exp_confg(
    sagemaker_session, job_resource_dir, run_obj
):
    # Notes:
    # 1. The 1st Run TC (run) created locally
    # 2. Within the 2nd Run TC (run_obj)'s context, invoke processor.run
    # but override the default experiment config in context of 2nd Run TC
    # with the experiment config of the 1st Run TC
    # 3. In the processing job script, load the 1st Run TC via the experiment config
    # fetched from the job env
    # 4. All data are logged in the Run TC either locally or in the processing job
    exp_name = unique_name_from_base(_EXP_NAME_BASE_IN_SCRIPT)
    processor = FrameworkProcessor(
        estimator_cls=PyTorch,
        framework_version="1.8",
        py_version="py3",
        instance_count=1,
        instance_type="ml.m5.xlarge",
        role=EXECUTION_ROLE,
        sagemaker_session=sagemaker_session,
    )

    with cleanup_exp_resources(exp_names=[exp_name], sagemaker_session=sagemaker_session):
        with Run.init(
            experiment_name=exp_name,
            run_name=_RUN_NAME_IN_SCRIPT,
            sagemaker_session=sagemaker_session,
        ) as run:
            _local_run_log_behaviors(is_complete_log=False, sagemaker_session=sagemaker_session)

        with run_obj:
            # Override the default experiment_config in _RunContext of run_obj
            # with the experiment_config of run
            processor.run(
                code=_PYTHON_PROCESS_SCRIPT,
                source_dir=_EXP_DIR,
                job_name=f"process-job-{name()}",
                wait=True,  # wait the job to finish
                logs=False,
                experiment_config=run._experiment_config,
            )

        assert run_obj.experiment_name != run.experiment_name
        assert run_obj.run_name != run.run_name
        _check_run_from_local_end_result(
            tc=run._trial_component,
            sagemaker_session=sagemaker_session,
            is_complete_log=False,
        )
        tc_name = Run._generate_trial_component_name(
            experiment_name=run.experiment_name, run_name=run.run_name
        )
        _check_run_from_job_result(
            tc_name=tc_name, sagemaker_session=sagemaker_session, is_init=False
        )


def _generate_estimator(exp_name, job_resource_dir, sagemaker_session):
    return SKLearn(
        framework_version="0.23-1",
        entry_point=_ENTRY_POINT_PATH,
        dependencies=[job_resource_dir, _PYTHON_TRAIN_SCRIPT_PATH],
        role=EXECUTION_ROLE,
        instance_type="ml.m5.large",
        instance_count=1,
        volume_size=10,
        max_run=900,
        enable_sagemaker_metrics=True,
        environment={
            "EXPERIMENT_NAME": exp_name,
            "RUN_NAME": _RUN_NAME_IN_SCRIPT,
            "RUN_OPERATION": _RUN_INIT,
        },
        sagemaker_session=sagemaker_session,
    )


def _local_run_log_behaviors(
    sagemaker_session,
    artifact_file_path=None,
    artifact_directory=None,
    lineage_artifact_path=None,
    is_complete_log=True,
):
    with Run.load(sagemaker_session=sagemaker_session) as run:
        run.log_parameter("pa", 1.0)
        run.log_parameter("pb", "p2-value")
        run.log_parameters({"pc": 2.0, "pd": "p4-value"})

        if is_complete_log:
            run.log_artifact_file(file_path=artifact_file_path, name=file_artifact_name)
            run.log_artifact_directory(directory=artifact_directory, is_output=False)
            run.log_lineage_artifact(file_path=lineage_artifact_path, name=lineage_artifact_name)

            for i in range(_MetricsManager._BATCH_SIZE):
                run.log_metric(name=metric_name, value=i, step=i)


def _check_run_from_local_end_result(sagemaker_session, tc, is_complete_log=True):
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

    assert tc.parameters == {"pa": 1.0, "pb": "p2-value", "pc": 2.0, "pd": "p4-value"}

    if not is_complete_log:
        return

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


def _check_run_from_job_result(sagemaker_session, tc_name=None, is_init=True, has_extra_load=False):
    def validate_tc_updated_in_init():
        assert tc.start_time
        assert tc.end_time
        assert tc.status.primary_status == _TrialComponentStatusType.Completed.value
        assert tc.parameters["p1"] == 1.0
        assert tc.parameters["p2"] == 2.0
        assert len(tc.metrics) == 5
        for metric_summary in tc.metrics:
            # metrics deletion is not supported at this point
            # so its count would accumulate
            assert metric_summary.count > 0
            assert metric_summary.min == 0.0
            assert metric_summary.max == 1.0

    def validate_tc_updated_in_load():
        assert tc.parameters["p3"] == 3.0
        assert tc.parameters["p4"] == 4.0
        assert len(tc.metrics) > 0
        for metric_summary in tc.metrics:
            if metric_summary.metric_name != "test-job-load-log-metric":
                continue
            assert metric_summary.last == 0.1
            assert metric_summary.max == 0.1
            assert metric_summary.min == 0.1
        if has_extra_load:
            assert tc.parameters["p5"] == 5.0
            assert tc.parameters["p6"] == 6.0

    tc = _TrialComponent.load(trial_component_name=tc_name, sagemaker_session=sagemaker_session)
    if is_init:
        # Add retry since the load behavior is inconsistent sometimes
        retry_with_backoff(validate_tc_updated_in_init, 4)
    else:
        retry_with_backoff(validate_tc_updated_in_load, 4)


def _check_tc_status_when_entering(trial_component):
    assert isinstance(trial_component.start_time, datetime.datetime)
    assert not trial_component.end_time
    assert trial_component.status.primary_status == _TrialComponentStatusType.InProgress.value
    return trial_component.start_time


def _check_tc_status_when_exiting(
    trial_component_name, sagemaker_session, init_start_time, old_end_time=None
):
    tc = _TrialComponent.load(
        trial_component_name=trial_component_name, sagemaker_session=sagemaker_session
    )
    # There will be deviation (< 1s) caused by different TS precisions used in Backend and SDK
    assert abs(tc.start_time.timestamp() - init_start_time.timestamp()) < 1
    assert tc.status.primary_status == _TrialComponentStatusType.Completed.value
    assert isinstance(tc.end_time, datetime.datetime)
    if old_end_time:
        assert tc.end_time > old_end_time
    return tc.end_time


def _check_tc_status_intermediate(
    trial_component, sagemaker_session, init_start_time, old_end_time=None
):
    tc_load = _TrialComponent.load(
        trial_component_name=trial_component.trial_component_name,
        sagemaker_session=sagemaker_session,
    )
    assert abs(tc_load.start_time.timestamp() - init_start_time.timestamp()) < 1
    assert tc_load.status.primary_status == _TrialComponentStatusType.InProgress.value
    if not old_end_time:
        assert not trial_component.end_time
        return
    assert isinstance(tc_load.end_time, datetime.datetime)
    assert tc_load.end_time == old_end_time


def _check_run_trial_component_tags(trial_component, sagemaker_session):
    search_filter = Filter(
        name="TrialComponentName",
        operator=Operator.CONTAINS,
        value=trial_component.trial_component_name,
    )
    search_expression = SearchExpression(filters=[search_filter])

    def validate():
        tc_search_res = list(
            _TrialComponent.search(
                search_expression=search_expression,
                max_results=1,
                sagemaker_session=sagemaker_session,
            )
        )
        assert len(tc_search_res) == 1
        assert len(tc_search_res[0].tags) > 0
        expected_tc_tag = {"Key": RUN_TC_TAG_KEY, "Value": RUN_TC_TAG_VALUE}
        assert expected_tc_tag in tc_search_res[0].tags

    retry_with_backoff(validate, 4)
