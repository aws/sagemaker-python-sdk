from __future__ import absolute_import


import os
import pytest
import time
import re
from datetime import datetime, timedelta

from sagemaker.workflow.functions import Join

from sagemaker.workflow.parameters import ParameterInteger, ParameterString

from sagemaker import TrainingInput, image_uris, get_execution_role, utils
from sagemaker.estimator import Estimator

from sagemaker.workflow.steps import TrainingStep

from sagemaker.workflow.triggers import PipelineSchedule

from sagemaker.workflow.pipeline import Pipeline
from tests.integ.sagemaker.workflow.helpers import validate_scheduled_pipeline_execution
from tests.integ import DATA_DIR

INSTANCE_TYPE = "ml.m5.large"


@pytest.fixture
def role(sagemaker_session):
    return get_execution_role(sagemaker_session)


@pytest.fixture
def region_name(sagemaker_session):
    return sagemaker_session.boto_session.region_name


def test_pipeline_scheduler_using_training_step(
    sagemaker_session, role, tf_full_version, tf_full_py_version, region_name
):
    pipeline_name = utils.unique_name_from_base("Scheduled-Training-Step-Pipeline")
    input_path = sagemaker_session.upload_data(
        path=os.path.join(DATA_DIR, "xgboost_abalone", "abalone"),
        key_prefix="integ-test-data/xgboost_abalone/abalone",
    )
    inputs = {"train": TrainingInput(s3_data=input_path)}
    entry_point = "dummy1"
    src_base_dir = os.path.join(DATA_DIR, "xgboost_abalone/estimator_source_code")
    source_dir = sagemaker_session.upload_data(
        path=os.path.join(src_base_dir, "estimator_source_code_dummy1.tar.gz"),
        key_prefix="integ-test-data/estimator/training",
    )
    # Build params -- make sure to provide defaults
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")
    entry_point_param = ParameterString(name="EntryPoint", default_value=entry_point)
    source_dir_param = ParameterString(name="SourceDir", default_value=source_dir)
    output_path = Join(
        on="/", values=["s3:/", f"{sagemaker_session.default_bucket()}", f"{pipeline_name}"]
    )
    image_uri = image_uris.retrieve(
        framework="xgboost",
        region=sagemaker_session.boto_session.region_name,
        version="1.0-1",
        py_version="py3",
        instance_type="ml.m5.xlarge",
    )
    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        output_path=output_path,
        source_dir=source_dir_param,
        entry_point=entry_point_param,
        base_job_name="TestJob",
    )
    estimator.set_hyperparameters(
        objective="reg:linear",
        num_round=50,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
    )
    step_train = TrainingStep(
        name="MyScheduledTrainingStep",
        estimator=estimator,
        inputs=inputs,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count, instance_type, source_dir_param, entry_point_param],
        steps=[step_train],
        sagemaker_session=sagemaker_session,
    )

    try:

        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

        # (at=) PipelineSchedules using datetime
        datetime_at = datetime.now() + timedelta(0, 120)  # 2 minutes from now
        schedule1 = PipelineSchedule(name="datetime-schedule", at=datetime_at)
        # (rate=) PipelineSchedules using rate-based
        schedule2 = PipelineSchedule(name="test-rate", rate=(2, "minutes"))
        # (cron=) PipelineSchedules using cron-based
        schedule3 = PipelineSchedule(cron="0/2 * ? * * *")  # run job every 2 minutes

        schedules = [schedule1, schedule2, schedule3]
        triggers = pipeline.put_triggers(triggers=schedules, role_arn=role)
        assert len(triggers) == 3

        describe1 = pipeline.describe_trigger(trigger_name=schedule1.name)
        describe2 = pipeline.describe_trigger(trigger_name=schedule2.name)
        describe3 = pipeline.describe_trigger(trigger_name=schedule3.name)

        assert "at" in describe1["Schedule_Expression"]
        assert "rate" in describe2["Schedule_Expression"]
        assert "cron" in describe3["Schedule_Expression"]

        schedule1_arn = describe1["Schedule_Arn"]
        schedule2_arn = describe2["Schedule_Arn"]
        schedule3_arn = describe3["Schedule_Arn"]
        schedule_arns = [schedule1_arn, schedule2_arn, schedule3_arn]

        for schedule, expected_arn in zip(schedules, schedule_arns):
            assert re.match(
                rf"arn:aws:scheduler:{region_name}:\d{{12}}:schedule/default/{schedule.name}",
                expected_arn,
            )

        time.sleep(180)  # wait 3 minutes

        list_response = pipeline.list_executions()
        executions = list_response["PipelineExecutionSummaries"]
        assert executions is not None
        # At least 3 executions should have been started, one from each type of schedule.
        assert len(executions) > 3

        # Attempt async cleanup. Must delete schedules so no new executions are created
        # between now and attempting to delete the pipeline.
        pipeline.delete_triggers(trigger_names=[s.name for s in schedules])

        # Validate all running, scheduled executions.
        for execution in executions:
            execution_arn = execution["PipelineExecutionArn"]
            validate_scheduled_pipeline_execution(
                execution_arn=execution_arn,
                pipeline_arn=create_arn,
                no_of_steps=1,
                last_step_name="MyScheduledTrainingStep",
                status="Succeeded",
                session=sagemaker_session,
            )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass
