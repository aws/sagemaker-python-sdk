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
from botocore.exceptions import WaiterError

from sagemaker.workflow.automl_step import AutoMLStep
from sagemaker.automl.automl import AutoML, AutoMLInput

from sagemaker import utils, get_execution_role
from sagemaker.utils import unique_name_from_base
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline import Pipeline

from tests.integ import DATA_DIR

PREFIX = "sagemaker/automl-agt"
TARGET_ATTRIBUTE_NAME = "virginica"
DATA_DIR = os.path.join(DATA_DIR, "automl", "data")
TRAINING_DATA = os.path.join(DATA_DIR, "iris_training.csv")
VALIDATION_DATA = os.path.join(DATA_DIR, "iris_validation.csv")
MODE = "ENSEMBLING"


@pytest.fixture
def role(sagemaker_session):
    return get_execution_role(sagemaker_session)


@pytest.fixture
def pipeline_name():
    return utils.unique_name_from_base("my-pipeline-automl")


def test_automl_step(pipeline_session, role, pipeline_name):
    auto_ml = AutoML(
        role=role,
        target_attribute_name=TARGET_ATTRIBUTE_NAME,
        sagemaker_session=pipeline_session,
        max_candidates=1,
        mode=MODE,
    )
    job_name = unique_name_from_base("auto-ml", max_length=32)
    s3_input_training = pipeline_session.upload_data(
        path=TRAINING_DATA, key_prefix=PREFIX + "/input"
    )
    s3_input_validation = pipeline_session.upload_data(
        path=VALIDATION_DATA, key_prefix=PREFIX + "/input"
    )
    input_training = AutoMLInput(
        inputs=s3_input_training,
        target_attribute_name=TARGET_ATTRIBUTE_NAME,
        channel_type="training",
    )
    input_validation = AutoMLInput(
        inputs=s3_input_validation,
        target_attribute_name=TARGET_ATTRIBUTE_NAME,
        channel_type="validation",
    )
    inputs = [input_training, input_validation]

    step_args = auto_ml.fit(inputs=inputs, job_name=job_name)

    automl_step = AutoMLStep(
        name="MyAutoMLStep",
        step_args=step_args,
    )

    automl_model = automl_step.get_best_auto_ml_model(sagemaker_session=pipeline_session, role=role)

    step_args_create_model = automl_model.create(
        instance_type="c4.4xlarge",
    )

    automl_model_step = ModelStep(
        name="MyAutoMLModelStep",
        step_args=step_args_create_model,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[automl_step, automl_model_step],
        sagemaker_session=pipeline_session,
    )

    try:
        _ = pipeline.create(role)
        execution = pipeline.start(parameters={})
        try:
            execution.wait(delay=30, max_attempts=60)
        except WaiterError:
            pass

        execution_steps = execution.list_steps()
        has_automl_job = False
        for step in execution_steps:
            assert not step.get("FailureReason", None)
            assert step["StepStatus"] == "Succeeded"
            if "AutoMLJob" in step["Metadata"]:
                has_automl_job = True
                assert step["Metadata"]["AutoMLJob"]["Arn"] is not None

        assert has_automl_job
        assert len(execution_steps) == 2
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass
