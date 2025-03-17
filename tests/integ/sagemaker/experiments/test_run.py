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
import json
import os
import time

import pytest

from tests.conftest import CUSTOM_S3_OBJECT_KEY_PREFIX
from tests.integ.sagemaker.experiments.conftest import TAGS
from sagemaker.experiments._api_types import _TrialComponentStatusType
from sagemaker.experiments._utils import is_run_trial_component
from sagemaker.processing import FrameworkProcessor
from sagemaker.pytorch import PyTorch
from sagemaker.s3 import S3Uploader
from sagemaker.xgboost import XGBoostModel
from tests.integ import DATA_DIR
from sagemaker.experiments._metrics import BATCH_SIZE
from sagemaker.experiments.trial_component import _TrialComponent
from sagemaker.sklearn import SKLearn
from sagemaker.utils import retry_with_backoff, unique_name_from_base
from tests.integ.sagemaker.experiments.helpers import name, cleanup_exp_resources, clear_run_context
from sagemaker.experiments.run import (
    RUN_NAME_BASE,
    DELIMITER,
)
from sagemaker.experiments import Run, load_run, list_runs
from sagemaker.experiments._helper import _DEFAULT_ARTIFACT_PREFIX


@pytest.fixture
def artifact_file_path(tempdir):
    file_contents = "test artifact file"
    file_path = os.path.join(tempdir, "artifact_file.txt")
    with open(file_path, "w") as foo_file:
        foo_file.write(file_contents)
    return file_path


artifact_name = unique_name_from_base("Test-Artifact")
file_artifact_name = f"File-Artifact-{name()}"
metric_name = "Test-Local-Init-Log-Metric"


def test_local_run_with_load(sagemaker_session, artifact_file_path, clear_run_context):
    exp_name = f"My-Local-Exp-{name()}"
    with cleanup_exp_resources(exp_names=[exp_name], sagemaker_session=sagemaker_session):
        # Run name is not provided, will create a new TC
        with Run(experiment_name=exp_name, sagemaker_session=sagemaker_session) as run1:
            run1_name = run1.run_name
            assert RUN_NAME_BASE in run1_name
            _local_run_log_behaviors(
                artifact_file_path=artifact_file_path,
                sagemaker_session=sagemaker_session,
            )

        def verify_load_run():
            with load_run(
                experiment_name=exp_name,
                run_name=run1_name,
                sagemaker_session=sagemaker_session,
            ) as run2:
                assert run2.run_name == run1_name
                assert (
                    run2._trial_component.trial_component_name
                    == f"{run2.experiment_name}{DELIMITER}{run1_name}"
                )
                _check_run_from_local_end_result(
                    sagemaker_session=sagemaker_session, tc=run2._trial_component
                )

        # Add retry to make sure metrics -> eureka propagation is consistent
        retry_with_backoff(verify_load_run, 4)


def test_two_local_run_init_with_same_run_name_and_different_exp_names(
    sagemaker_session, clear_run_context
):
    exp_name1 = f"my-two-local-exp1-{name()}"
    exp_name2 = f"my-two-local-exp2-{name()}"
    run_name = "test-run"
    with cleanup_exp_resources(
        exp_names=[exp_name1, exp_name2], sagemaker_session=sagemaker_session
    ):
        # Run name is not provided, will create a new TC
        with Run(
            experiment_name=exp_name1, run_name=run_name, sagemaker_session=sagemaker_session
        ) as run1:
            pass
        with Run(
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
    sagemaker_session, input_names, clear_run_context
):
    exp_name, run_name, run_display_name = input_names
    with cleanup_exp_resources(exp_names=[exp_name], sagemaker_session=sagemaker_session):
        with Run(
            experiment_name=exp_name,
            sagemaker_session=sagemaker_session,
            run_name=run_name,
            run_display_name=run_display_name,
        ) as run1:
            assert not run1._experiment.tags
            assert not run1._trial.tags

            def verify_is_run():
                is_run_tc = is_run_trial_component(
                    trial_component_name=run1._trial_component.trial_component_name,
                    sagemaker_session=sagemaker_session,
                )
                assert is_run_tc

            retry_with_backoff(verify_is_run, 4)

        with load_run(
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


_EXP_NAME_BASE_IN_SCRIPT = "job-exp-in-script"
_RUN_NAME_IN_SCRIPT = "job-run-in-script"

_EXP_DIR = os.path.join(DATA_DIR, "experiment")
_ENTRY_POINT_PATH = os.path.join(_EXP_DIR, "train_job_script_for_run_clz.py")
_PYTHON_PROCESS_SCRIPT = "process_job_script_for_run_clz.py"
_TRANSFORM_MATERIALS = os.path.join(_EXP_DIR, "transform_job_materials")

_RUN_INIT = "init"
_RUN_LOAD = "load"


def test_run_from_local_and_train_job_and_all_exp_cfg_match(
    sagemaker_session,
    dev_sdk_tar,
    execution_role,
    sagemaker_client_config,
    sagemaker_metrics_config,
    clear_run_context,
):
    # Notes:
    # 1. The 1st Run created locally and its exp config was auto passed to the job
    # 2. In training job, the same exp and run names are given in the Run constructor
    # which will load the 1st Run in training job and log parameters
    # and metrics there
    # 3. In a different training job, load the same Run and log more parameters there.
    exp_name = unique_name_from_base(_EXP_NAME_BASE_IN_SCRIPT)
    estimator = _generate_estimator(
        sdk_tar=dev_sdk_tar,
        sagemaker_session=sagemaker_session,
        exp_name=exp_name,
        execution_role=execution_role,
        sagemaker_client_config=sagemaker_client_config,
        sagemaker_metrics_config=sagemaker_metrics_config,
    )
    tc_name = Run._generate_trial_component_name(
        experiment_name=exp_name, run_name=_RUN_NAME_IN_SCRIPT
    )

    with cleanup_exp_resources(exp_names=[exp_name], sagemaker_session=sagemaker_session):
        with Run(
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

            # the above estimator has wait=True but the job TC could still be receiving updates
            # after wait is complete resulting in run TC being updated, then when the above with
            # statement is exited another update trial component call is made _sometimes_
            # resulting in a ConflictException
            time.sleep(3)

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


def test_run_from_local_and_train_job_and_exp_cfg_not_match(
    sagemaker_session,
    dev_sdk_tar,
    execution_role,
    sagemaker_client_config,
    sagemaker_metrics_config,
    clear_run_context,
):
    # Notes:
    # 1. The 1st Run created locally and its exp config was auto passed to the job
    # 2. In training job, different exp and run names (i.e. 2nd Run) are given
    # in the Run constructor which will create a Run according to the run_name
    # passed in there and ignore the exp config in the job
    # 3. Both metrics and parameters are logged in the Run created in job
    # 4. In a different training job, load the 2nd Run and log more parameters there.
    exp_name = unique_name_from_base(_EXP_NAME_BASE_IN_SCRIPT)
    exp_name2 = unique_name_from_base(_EXP_NAME_BASE_IN_SCRIPT)
    estimator = _generate_estimator(
        sdk_tar=dev_sdk_tar,
        sagemaker_session=sagemaker_session,
        exp_name=exp_name,
        execution_role=execution_role,
        sagemaker_client_config=sagemaker_client_config,
        sagemaker_metrics_config=sagemaker_metrics_config,
    )
    tc_name = Run._generate_trial_component_name(
        experiment_name=exp_name, run_name=_RUN_NAME_IN_SCRIPT
    )

    with cleanup_exp_resources(
        exp_names=[exp_name, exp_name2], sagemaker_session=sagemaker_session
    ):
        with Run(
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


def test_run_from_train_job_only(
    sagemaker_session,
    dev_sdk_tar,
    execution_role,
    sagemaker_client_config,
    sagemaker_metrics_config,
    clear_run_context,
):
    # Notes:
    # 1. No Run created locally or specified in experiment config
    # 2. In training job, Run is initialized
    # which will create a Run according to the run_name passed in there
    # 3. Both metrics and parameters are logged in the Run created in job
    # 4. In a different training job, load the same Run and log more parameters there.
    exp_name = unique_name_from_base(_EXP_NAME_BASE_IN_SCRIPT)
    estimator = _generate_estimator(
        sdk_tar=dev_sdk_tar,
        sagemaker_session=sagemaker_session,
        exp_name=exp_name,
        execution_role=execution_role,
        sagemaker_client_config=sagemaker_client_config,
        sagemaker_metrics_config=sagemaker_metrics_config,
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


# dev_sdk_tar is required to trigger generating the dev SDK tar
def test_run_from_processing_job_and_override_default_exp_config(
    sagemaker_session,
    dev_sdk_tar,
    run_obj,
    execution_role,
    sagemaker_client_config,
    sagemaker_metrics_config,
    clear_run_context,
):
    # Notes:
    # 1. The 1st Run (run) created locally
    # 2. Within the 2nd Run (run_obj)'s context, invoke processor.run
    # but override the default experiment config in context of 2nd Run
    # with the experiment config of the 1st Run
    # 3. In the processing job script, load the 1st Run via the experiment config
    # fetched from the job env
    # 4. All data are logged in the Run either locally or in the processing job
    exp_name = unique_name_from_base(_EXP_NAME_BASE_IN_SCRIPT)
    processor = _generate_processor(
        exp_name=exp_name,
        sagemaker_session=sagemaker_session,
        execution_role=execution_role,
        sagemaker_client_config=sagemaker_client_config,
        sagemaker_metrics_config=sagemaker_metrics_config,
    )

    with cleanup_exp_resources(exp_names=[exp_name], sagemaker_session=sagemaker_session):
        with Run(
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
                experiment_config=run.experiment_config,
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

        with run_obj:
            # Not to override the exp config and use the default one in the context
            processor.run(
                code=_PYTHON_PROCESS_SCRIPT,
                source_dir=_EXP_DIR,
                job_name=f"process-job-{name()}",
                wait=True,  # wait the job to finish
                logs=False,
            )

        tc_name = Run._generate_trial_component_name(
            experiment_name=run_obj.experiment_name, run_name=run_obj.run_name
        )
        _check_run_from_job_result(
            tc_name=tc_name, sagemaker_session=sagemaker_session, is_init=False
        )


# dev_sdk_tar is required to trigger generating the dev SDK tar
@pytest.mark.skip(reason="This test is failing regularly and blocking code pipeline.")
def test_run_from_transform_job(
    sagemaker_session,
    dev_sdk_tar,
    xgboost_latest_version,
    execution_role,
    sagemaker_client_config,
    sagemaker_metrics_config,
    clear_run_context,
):
    # Notes:
    # 1. The 1st Run (run) created locally
    # 2. In the inference script running in a transform job, load the 1st Run twice and log data
    # 1) via explicitly passing the experiment_name and run_name of the 1st Run
    # 2) use load_run() without explicitly supplying the names
    # 3. All data are logged in the Run either locally or in the transform job
    exp_name = unique_name_from_base(_EXP_NAME_BASE_IN_SCRIPT)
    xgb_model_data_s3 = sagemaker_session.upload_data(
        path=os.path.join(_TRANSFORM_MATERIALS, "xgb_model.tar.gz"),
        key_prefix="integ-test-data/xgboost/model",
    )
    env = _update_env_with_client_config(
        env={
            "EXPERIMENT_NAME": exp_name,
            "RUN_NAME": _RUN_NAME_IN_SCRIPT,
        },
        sagemaker_metrics_config=sagemaker_metrics_config,
        sagemaker_client_config=sagemaker_client_config,
    )
    xgboost_model = XGBoostModel(
        sagemaker_session=sagemaker_session,
        model_data=xgb_model_data_s3,
        role=execution_role,
        entry_point="inference.py",
        source_dir=_EXP_DIR,
        framework_version=xgboost_latest_version,
        env=env,
    )
    transformer = xgboost_model.transformer(
        instance_count=1,
        instance_type="ml.m5.4xlarge",
        max_concurrent_transforms=5,
        max_payload=1,
        strategy="MultiRecord",
    )
    uri = "s3://{}/{}/input/data/{}".format(
        sagemaker_session.default_bucket(),
        "transform-test",
        unique_name_from_base("json-data"),
    )
    input_data = S3Uploader.upload(
        os.path.join(_TRANSFORM_MATERIALS, "data.csv"), uri, sagemaker_session=sagemaker_session
    )

    with cleanup_exp_resources(exp_names=[exp_name], sagemaker_session=sagemaker_session):
        with Run(
            experiment_name=exp_name,
            run_name=_RUN_NAME_IN_SCRIPT,
            sagemaker_session=sagemaker_session,
        ) as run:
            _local_run_log_behaviors(is_complete_log=False, sagemaker_session=sagemaker_session)
            transformer.transform(
                data=input_data,
                content_type="text/libsvm",
                split_type="Line",
                wait=True,
                logs=False,
                job_name=f"transform-job-{name()}",
            )

        _check_run_from_local_end_result(
            tc=run._trial_component,
            sagemaker_session=sagemaker_session,
            is_complete_log=False,
        )
        tc_name = Run._generate_trial_component_name(
            experiment_name=run.experiment_name, run_name=run.run_name
        )
        _check_run_from_job_result(
            tc_name=tc_name, sagemaker_session=sagemaker_session, is_init=False, has_extra_load=True
        )


# dev_sdk_tar is required to trigger generating the dev SDK tar
def test_load_run_auto_pass_in_exp_config_to_job(
    sagemaker_session,
    dev_sdk_tar,
    execution_role,
    sagemaker_client_config,
    sagemaker_metrics_config,
    clear_run_context,
):
    # Notes:
    # 1. In local side, load the Run created previously and invoke a job under the load context
    # 2. In the job script, load the 1st Run via exp config auto-passed to the job env
    # 3. All data are logged in the Run either locally or in the transform job
    exp_name = unique_name_from_base(_EXP_NAME_BASE_IN_SCRIPT)
    processor = _generate_processor(
        exp_name=exp_name,
        sagemaker_session=sagemaker_session,
        execution_role=execution_role,
        sagemaker_client_config=sagemaker_client_config,
        sagemaker_metrics_config=sagemaker_metrics_config,
    )

    with cleanup_exp_resources(exp_names=[exp_name], sagemaker_session=sagemaker_session):
        with Run(
            experiment_name=exp_name,
            run_name=_RUN_NAME_IN_SCRIPT,
            sagemaker_session=sagemaker_session,
        ) as run:
            _local_run_log_behaviors(is_complete_log=False, sagemaker_session=sagemaker_session)

        with load_run(
            experiment_name=run.experiment_name,
            run_name=run.run_name,
            sagemaker_session=sagemaker_session,
        ):
            processor.run(
                code=_PYTHON_PROCESS_SCRIPT,
                source_dir=_EXP_DIR,
                job_name=f"process-job-{name()}",
                wait=True,  # wait the job to finish
                logs=False,
            )

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


def test_list(run_obj, sagemaker_session, clear_run_context):
    tc1 = _TrialComponent.create(
        trial_component_name=f"non-run-tc1-{name()}",
        sagemaker_session=sagemaker_session,
    )
    tc2 = _TrialComponent.create(
        trial_component_name=f"non-run-tc2-{name()}",
        sagemaker_session=sagemaker_session,
        tags=TAGS,
    )
    run_obj._trial.add_trial_component(tc1)
    run_obj._trial.add_trial_component(tc2)

    run_tcs = list_runs(
        experiment_name=run_obj.experiment_name, sagemaker_session=sagemaker_session
    )
    assert len(run_tcs) == 1
    assert run_tcs[0].run_name == run_obj.run_name
    assert run_tcs[0].experiment_name == run_obj.experiment_name
    assert run_tcs[0].experiment_config == run_obj.experiment_config


def test_list_twice(run_obj, sagemaker_session, clear_run_context):
    tc1 = _TrialComponent.create(
        trial_component_name=f"non-run-tc1-{name()}",
        sagemaker_session=sagemaker_session,
    )
    tc2 = _TrialComponent.create(
        trial_component_name=f"non-run-tc2-{name()}",
        sagemaker_session=sagemaker_session,
        tags=TAGS,
    )
    run_obj._trial.add_trial_component(tc1)
    run_obj._trial.add_trial_component(tc2)

    run_tcs = list_runs(
        experiment_name=run_obj.experiment_name, sagemaker_session=sagemaker_session
    )
    assert len(run_tcs) == 1
    assert run_tcs[0].run_name == run_obj.run_name
    assert run_tcs[0].experiment_name == run_obj.experiment_name
    assert run_tcs[0].experiment_config == run_obj.experiment_config

    # note the experiment name used by run_obj is already mixed case and so
    # covers the mixed case experiment name double create issue
    run_tcs_second_result = list_runs(
        experiment_name=run_obj.experiment_name, sagemaker_session=sagemaker_session
    )
    assert len(run_tcs) == 1
    assert run_tcs_second_result[0].run_name == run_obj.run_name
    assert run_tcs_second_result[0].experiment_name == run_obj.experiment_name
    assert run_tcs_second_result[0].experiment_config == run_obj.experiment_config


def _generate_estimator(
    exp_name,
    sdk_tar,
    sagemaker_session,
    execution_role,
    sagemaker_client_config,
    sagemaker_metrics_config,
):
    env = _update_env_with_client_config(
        env={
            "EXPERIMENT_NAME": exp_name,
            "RUN_NAME": _RUN_NAME_IN_SCRIPT,
            "RUN_OPERATION": _RUN_INIT,
        },
        sagemaker_metrics_config=sagemaker_metrics_config,
        sagemaker_client_config=sagemaker_client_config,
    )
    return SKLearn(
        framework_version="1.2-1",
        entry_point=_ENTRY_POINT_PATH,
        dependencies=[sdk_tar],
        role=execution_role,
        instance_type="ml.m5.large",
        instance_count=1,
        volume_size=10,
        max_run=900,
        enable_sagemaker_metrics=True,
        environment=env,
        sagemaker_session=sagemaker_session,
    )


def _generate_processor(
    exp_name, sagemaker_session, execution_role, sagemaker_metrics_config, sagemaker_client_config
):
    env = _update_env_with_client_config(
        env={
            "EXPERIMENT_NAME": exp_name,
            "RUN_NAME": _RUN_NAME_IN_SCRIPT,
        },
        sagemaker_metrics_config=sagemaker_metrics_config,
        sagemaker_client_config=sagemaker_client_config,
    )
    return FrameworkProcessor(
        estimator_cls=PyTorch,
        framework_version="1.10",
        py_version="py39",
        instance_count=1,
        instance_type="ml.m5.xlarge",
        role=execution_role,
        sagemaker_session=sagemaker_session,
        env=env,
    )


def _local_run_log_behaviors(
    sagemaker_session,
    artifact_file_path=None,
    is_complete_log=True,
):
    with load_run(sagemaker_session=sagemaker_session) as run:
        run.log_parameter("pa", 1.0)
        run.log_parameter("pb", "p2-value")
        run.log_parameters({"pc": 2.0, "pd": "p4-value"})

        if is_complete_log:
            run.log_file(file_path=artifact_file_path, name=file_artifact_name)
            run.log_artifact(name=artifact_name, value="s3://Output")
            run.log_artifact(name=artifact_name, value="s3://Input", is_output=False)

            for i in range(BATCH_SIZE):
                run.log_metric(name=metric_name, value=i, step=i)


def _check_run_from_local_end_result(sagemaker_session, tc, is_complete_log=True):
    assert tc.parameters == {"pa": 1.0, "pb": "p2-value", "pc": 2.0, "pd": "p4-value"}

    if not is_complete_log:
        return

    s3_prefix = f"s3://{sagemaker_session.default_bucket()}/{CUSTOM_S3_OBJECT_KEY_PREFIX}/{_DEFAULT_ARTIFACT_PREFIX}"
    assert s3_prefix in tc.output_artifacts[file_artifact_name].value
    assert "text/plain" == tc.output_artifacts[file_artifact_name].media_type
    assert "s3://Output" == tc.output_artifacts[artifact_name].value
    assert not tc.output_artifacts[artifact_name].media_type
    assert "s3://Input" == tc.input_artifacts[artifact_name].value
    assert not tc.input_artifacts[artifact_name].media_type

    assert len(tc.metrics) == 1
    metric_summary = tc.metrics[0]
    assert metric_summary.metric_name == metric_name
    assert metric_summary.max == 9.0
    assert metric_summary.min == 0.0


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


def _update_env_with_client_config(env, sagemaker_client_config, sagemaker_metrics_config):
    if sagemaker_client_config and sagemaker_client_config.get("endpoint_url", None):
        env["SM_CLIENT_CONFIG"] = json.dumps(
            {"endpoint_url": sagemaker_client_config["endpoint_url"]}
        )
    if sagemaker_metrics_config and sagemaker_metrics_config.get("endpoint_url", None):
        env["SM_METRICS_CONFIG"] = json.dumps(
            {"endpoint_url": sagemaker_metrics_config["endpoint_url"]}
        )
    return env
