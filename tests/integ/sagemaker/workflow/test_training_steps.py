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
import uuid
import logging

import pytest
from botocore.exceptions import WaiterError

from sagemaker import TrainingInput, get_execution_role, utils
from sagemaker.debugger import (
    DebuggerHookConfig,
    Rule,
    rule_configs,
)
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from tests.integ.retry import retries
from tests.integ import DATA_DIR


@pytest.fixture
def role(sagemaker_session):
    return get_execution_role(sagemaker_session)


@pytest.fixture
def pipeline_name():
    return utils.unique_name_from_base("my-pipeline-training")


@pytest.fixture
def region_name(sagemaker_session):
    return sagemaker_session.boto_session.region_name


def test_training_job_with_debugger_and_profiler(
    sagemaker_session,
    pipeline_name,
    role,
    pytorch_training_latest_version,
    pytorch_training_latest_py_version,
):
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")

    rules = [
        Rule.sagemaker(rule_configs.vanishing_gradient()),
        Rule.sagemaker(base_config=rule_configs.all_zero(), rule_parameters={"tensor_regex": ".*"}),
        Rule.sagemaker(rule_configs.loss_not_decreasing()),
    ]
    debugger_hook_config = DebuggerHookConfig(
        s3_output_path=(f"s3://{sagemaker_session.default_bucket()}/{uuid.uuid4()}/tensors")
    )

    base_dir = os.path.join(DATA_DIR, "pytorch_mnist")
    script_path = os.path.join(base_dir, "mnist.py")
    input_path = sagemaker_session.upload_data(
        path=os.path.join(base_dir, "training"),
        key_prefix="integ-test-data/pytorch_mnist/training",
    )
    inputs = TrainingInput(s3_data=input_path)

    pytorch_estimator = PyTorch(
        entry_point=script_path,
        role="SageMakerRole",
        framework_version=pytorch_training_latest_version,
        py_version=pytorch_training_latest_py_version,
        instance_count=instance_count,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        rules=rules,
        debugger_hook_config=debugger_hook_config,
    )

    step_train = TrainingStep(
        name="pytorch-train",
        estimator=pytorch_estimator,
        inputs=inputs,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count, instance_type],
        steps=[step_train],
        sagemaker_session=sagemaker_session,
    )

    for _ in retries(
        max_retry_count=5,
        exception_message_prefix="Waiting for a successful execution of pipeline",
        seconds_to_sleep=10,
    ):
        try:
            response = pipeline.create(role)
            create_arn = response["PipelineArn"]

            execution = pipeline.start()
            response = execution.describe()
            assert response["PipelineArn"] == create_arn

            try:
                execution.wait(delay=10, max_attempts=60)
            except WaiterError:
                pass
            execution_steps = execution.list_steps()

            assert len(execution_steps) == 1
            failure_reason = execution_steps[0].get("FailureReason", "")
            if failure_reason != "":
                logging.error(f"Pipeline execution failed with error: {failure_reason}.Retrying..")
                continue
            assert execution_steps[0]["StepName"] == "pytorch-train"
            assert execution_steps[0]["StepStatus"] == "Succeeded"

            training_job_arn = execution_steps[0]["Metadata"]["TrainingJob"]["Arn"]
            job_description = sagemaker_session.sagemaker_client.describe_training_job(
                TrainingJobName=training_job_arn.split("/")[1]
            )

            for index, rule in enumerate(rules):
                config = job_description["DebugRuleConfigurations"][index]
                assert config["RuleConfigurationName"] == rule.name
                assert config["RuleEvaluatorImage"] == rule.image_uri
                assert config["VolumeSizeInGB"] == 0
                assert (
                    config["RuleParameters"]["rule_to_invoke"]
                    == rule.rule_parameters["rule_to_invoke"]
                )
            assert job_description["DebugHookConfig"] == debugger_hook_config._to_request_dict()

            assert job_description["ProfilingStatus"] == "Enabled"
            assert job_description["ProfilerConfig"]["ProfilingIntervalInMilliseconds"] == 500
            break
        finally:
            try:
                pipeline.delete()
            except Exception:
                pass
