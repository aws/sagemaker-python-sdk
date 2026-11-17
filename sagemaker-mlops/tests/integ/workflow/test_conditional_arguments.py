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
"""Integration test for If / ElseIf / Omit step arguments built through step_args.

One pipeline runs with two parameter sets. Each job is described afterwards to check
that the selected branch was sent and that an omitted member is absent from the request.
"""

from __future__ import absolute_import

import logging
import time
import uuid

from sagemaker.core import image_uris
from sagemaker.core.processing import Processor
from sagemaker.core.shapes import ProcessingOutput, ProcessingS3Output
from sagemaker.core.workflow import ElseIf, If, Omit
from sagemaker.core.workflow.conditions import ConditionEquals
from sagemaker.core.workflow.parameters import ParameterBoolean, ParameterString
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, InputData, StoppingCondition

# Writes 200 numeric rows (label first, no header) for XGBoost.
MAKE_DATA = (
    "d=/opt/ml/processing/train; mkdir -p $d; for i in $(seq 200); do echo $i,$i; done > $d/t.csv"
)


def _wait(execution):
    deadline = time.time() + 2400
    while time.time() < deadline:
        status = execution.describe()["PipelineExecutionStatus"]
        if status not in ("Executing", "Stopping"):
            return status
        time.sleep(30)
    return "TimedOut"


def _jobs(sagemaker_client, execution):
    jobs = {}
    for step in execution.list_steps():
        metadata = step.get("Metadata", {})
        if "ProcessingJob" in metadata:
            name = metadata["ProcessingJob"]["Arn"].split("/")[-1]
            jobs["processing"] = sagemaker_client.describe_processing_job(ProcessingJobName=name)
        if "TrainingJob" in metadata:
            name = metadata["TrainingJob"]["Arn"].split("/")[-1]
            jobs["training"] = sagemaker_client.describe_training_job(TrainingJobName=name)
    return jobs


def test_if_else_if_and_omit_through_step_args(pipeline_session, role):
    stamp = uuid.uuid4().hex[:8]
    region = pipeline_session.boto_region_name
    image = image_uris.retrieve(framework="xgboost", region=region, version="1.7-1")

    enable = ParameterBoolean("Enable", default_value=False)
    env = ParameterString("Env", default_value="dev")
    enabled = ConditionEquals(left=enable, right=True)
    stage = If(
        ConditionEquals(left=env, right="prod"),
        then="release",
        else_if=[ElseIf(ConditionEquals(left=env, right="beta"), then="preview")],
        else_="dev",
    )

    processor = Processor(
        role=role,
        image_uri=image,
        instance_count=1,
        instance_type="ml.m5.large",
        entrypoint=["/bin/sh", "-c", MAKE_DATA],
        env={"STAGE": stage, "EXTRA": If(enabled, then="on", else_=Omit())},
        sagemaker_session=pipeline_session,
    )
    output_uri = f"s3://{pipeline_session.default_bucket()}/integ-if-omit-{stamp}/train"
    process = ProcessingStep(
        name="MakeData",
        step_args=processor.run(
            outputs=[
                ProcessingOutput(
                    output_name="train",
                    s3_output=ProcessingS3Output(
                        s3_uri=output_uri,
                        local_path="/opt/ml/processing/train",
                        s3_upload_mode="EndOfJob",
                    ),
                )
            ]
        ),
    )

    trainer = ModelTrainer(
        training_image=image,
        role=role,
        base_job_name=f"integ-if-omit-{stamp}",
        compute=Compute(instance_type="ml.m5.large", instance_count=1),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=1800),
        hyperparameters={
            "objective": "reg:squarederror",
            "num_round": "5",
            "max_depth": If(enabled, then="3", else_=Omit()),
        },
        environment={"EXTRA": If(enabled, then="on", else_=Omit())},
        input_data_config=[
            InputData(
                channel_name="train",
                data_source=process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            )
        ],
        sagemaker_session=pipeline_session,
    )
    train = TrainingStep(name="Train", step_args=trainer.train())

    pipeline = Pipeline(
        name=f"integ-if-omit-{stamp}",
        parameters=[enable, env],
        steps=[process, train],
        sagemaker_session=pipeline_session,
    )

    try:
        pipeline.upsert(role_arn=role)
        enabled_run = pipeline.start(parameters={"Enable": True, "Env": "prod"})
        omitted_run = pipeline.start(parameters={"Enable": False, "Env": "beta"})
        client = pipeline_session.sagemaker_client

        for execution in (enabled_run, omitted_run):
            status = _wait(execution)
            steps = {s["StepName"]: s["StepStatus"] for s in execution.list_steps()}
            assert status == "Succeeded", f"status={status}, steps={steps}"

        enabled_jobs = _jobs(client, enabled_run)
        assert enabled_jobs["processing"]["Environment"] == {"STAGE": "release", "EXTRA": "on"}
        assert enabled_jobs["training"]["HyperParameters"]["max_depth"] == "3"
        assert enabled_jobs["training"]["Environment"] == {"EXTRA": "on"}

        omitted_jobs = _jobs(client, omitted_run)
        assert omitted_jobs["processing"]["Environment"] == {"STAGE": "preview"}
        assert "max_depth" not in omitted_jobs["training"]["HyperParameters"]
        assert "EXTRA" not in omitted_jobs["training"].get("Environment", {})
    finally:
        try:
            pipeline.delete()
        except Exception as e:
            logging.warning("Failed to delete pipeline %s: %s", pipeline.name, e)
