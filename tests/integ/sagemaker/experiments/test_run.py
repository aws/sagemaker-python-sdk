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

from sagemaker import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.experiments.metrics import _MetricsManager
from sagemaker.experiments.trial_component import _TrialComponent
from sagemaker.utils import retry_with_backoff
from tests.integ.sagemaker.experiments.helpers import name
from sagemaker.experiments.run import Run, RUN_NAME_BASE
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


def test_local_run(
    sagemaker_session, artifact_file_path, artifact_directory, lineage_artifact_path
):
    exp_name = f"my-local-exp-{name()}"
    file_artifact_name = "file-artifact"
    lineage_artifact_name = "lineage-file-artifact"
    table_artifact_name = "TestTableTitle"
    metric_name = "test-x-step"

    # Run name is not provided, will create a new TC
    with Run.init(experiment_name=exp_name, sagemaker_session=sagemaker_session) as run1:
        run1_name = run1.run_name
        run1_exp_name = run1.experiment_name
        run1_trial_name = run1._trial.trial_name

        run1.log_parameter("p1", 1.0)
        run1.log_parameter("p2", "p2-value")
        run1.log_parameters({"p3": 2.0, "p4": "p4-value"})

        run1.log_artifact_file(file_path=artifact_file_path, name=file_artifact_name)
        run1.log_artifact_directory(directory=artifact_directory, is_output=False)
        run1.log_lineage_artifact(file_path=lineage_artifact_path, name=lineage_artifact_name)
        run1.log_table(
            title=table_artifact_name, values={"x": [1, 2, 3], "y": [4, 5, 6]}, is_output=False
        )

        for i in range(_MetricsManager._BATCH_SIZE):
            run1.log_metric(name=metric_name, value=i, step=i)

    assert RUN_NAME_BASE in run1_name

    # Run name is passed from the name of an existing TC.
    # Meanwhile, the experiment_name is changed.
    # Should load it from backend.
    run2 = Run.init(
        experiment_name=f"{exp_name}-2",
        run_name=run1_name,
        sagemaker_session=sagemaker_session,
    )

    assert run1_exp_name != run2.experiment_name
    assert run1_trial_name != run2._trial.trial_name
    assert run1_name == run2.run_name

    tc = run2._trial_component
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

    validate_tc_artifact_association(
        is_output=True,
        expected_artifact_name=lineage_artifact_name,
    )
    validate_tc_artifact_association(
        is_output=False,
        expected_artifact_name=table_artifact_name,
    )


# TODO: Need to update the rest of tests
# once we update the Run behavior in job env in M1
_EXP_NAME_IN_SCRIPT = "my-train-job-exp-in-script"
_RUN_NAME_IN_SCRIPT = "my-train-job-run-in-script"
_EXP_NAME_IN_LOCAL = "my-train-job-exp-in-local"
_RUN_NAME_IN_LOCAL = "my-train-job-run-in-local"


@pytest.mark.slow_test
def test_run_from_local_and_train_job_and_all_exp_cfg_match(
    sagemaker_session,
    docker_image,
    training_input_s3_uri,
    training_output_s3_uri,
):
    # Notes:
    # 1. The 1st Run TC created locally
    # 2. In training job, the same exp and run names are given in Run.init
    # which will load the 1st Run TC in training job and log parameters there
    # 3. metrics are logged in job TC only and are not copied to Run TC
    with Run.init(
        experiment_name=_EXP_NAME_IN_SCRIPT,
        run_name=_RUN_NAME_IN_SCRIPT,
        sagemaker_session=sagemaker_session,
    ) as run:
        estimator = Estimator(
            image_uri=docker_image,
            role=EXECUTION_ROLE,
            instance_type="ml.m5.large",
            instance_count=1,
            volume_size=10,
            max_run=900,
            output_path=training_output_s3_uri,
            enable_sagemaker_metrics=True,
            sagemaker_session=sagemaker_session,
        )
        estimator.fit(
            job_name=f"train-job-{name()}",
            inputs={"train": TrainingInput(s3_data=training_input_s3_uri)},
            experiment_config=run.experiment_config,
            wait=True,  # wait the training job to finish
            logs="All",  # display all logs fetched from the training job
        )

    assert run.run_name == _RUN_NAME_IN_SCRIPT
    _check_run_from_job_result(
        tc_name=run.run_name,
        sagemaker_session=sagemaker_session,
    )


@pytest.mark.slow_test
def test_run_from_local_and_train_job_and_exp_cfg_not_match(
    sagemaker_session,
    docker_image,
    training_input_s3_uri,
    training_output_s3_uri,
):
    # Notes:
    # 1. The 1st Run TC created locally
    # 2. In training job, different exp and run names are given in Run.init
    # which will still load the 1st Run TC in training job and log parameters there
    # 3. metrics are logged in job TC only and are not copied to Run TC
    with Run.init(
        experiment_name=_EXP_NAME_IN_LOCAL,
        run_name=_RUN_NAME_IN_LOCAL,
        sagemaker_session=sagemaker_session,
    ) as run:
        estimator = Estimator(
            image_uri=docker_image,
            role=EXECUTION_ROLE,
            instance_type="ml.m5.large",
            instance_count=1,
            volume_size=10,
            max_run=900,
            output_path=training_output_s3_uri,
            enable_sagemaker_metrics=True,
            sagemaker_session=sagemaker_session,
        )
        estimator.fit(
            job_name=f"train-job-{name()}",
            inputs={"train": TrainingInput(s3_data=training_input_s3_uri)},
            experiment_config=run.experiment_config,
            wait=True,  # wait the training job to finish
            logs="All",  # display all logs fetched from the training job
        )

    assert run.run_name != _RUN_NAME_IN_SCRIPT
    _check_run_from_job_result(
        tc_name=run.run_name,
        sagemaker_session=sagemaker_session,
    )


@pytest.mark.slow_test
def test_run_from_train_job(
    sagemaker_session,
    docker_image,
    training_input_s3_uri,
    training_output_s3_uri,
):
    # Notes:
    # 1. No Run TC created locally or specified in experiment config
    # 2. In training job, Run.init is invoked
    # which will load/create a Run TC according to the run_name passed in there
    # 3. Both metrics and parameters are logged in the Run TC loaded/created in job
    estimator = Estimator(
        image_uri=docker_image,
        role=EXECUTION_ROLE,
        instance_type="ml.m5.large",
        instance_count=1,
        volume_size=10,
        max_run=900,
        output_path=training_output_s3_uri,
        enable_sagemaker_metrics=True,
        sagemaker_session=sagemaker_session,
    )
    estimator.fit(
        job_name=f"train-job-{name()}",
        inputs={"train": TrainingInput(s3_data=training_input_s3_uri)},
        wait=True,  # wait the training job to finish
        logs="All",  # display all logs fetched from the training job
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
            # if delete the Run tc
            # the metrics agent (in backend?) still persists.
            # then if recreate the Run tc with the same name
            # the metric count accumulate each time
            # assert metric_summary.count == 2
            assert metric_summary.min == 0.0
            assert metric_summary.max == 1.0

    # Add retry since the load behavior is inconsistent sometimes
    retry_with_backoff(validate_tc, 4)
