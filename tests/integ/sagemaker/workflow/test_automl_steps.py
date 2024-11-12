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
import re

import pytest

from tests.integ.sagemaker.workflow.helpers import wait_pipeline_execution
from sagemaker.workflow import ParameterString
from sagemaker.workflow.automl_step import AutoMLStep
from sagemaker.automl.automl import AutoML, AutoMLInput

from sagemaker import utils, get_execution_role, ModelMetrics, MetricsSource
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
        mode=MODE,
    )
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

    step_args = auto_ml.fit(inputs=inputs)

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

    model_package_group_name = ParameterString(
        name="ModelPackageName", default_value="AutoMlModelPackageGroup"
    )
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="Approved")
    model_insights_json_report_path = (
        automl_step.properties.BestCandidateProperties.ModelInsightsJsonReportPath
    )
    explainability_json_report_path = (
        automl_step.properties.BestCandidateProperties.ExplainabilityJsonReportPath
    )
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=model_insights_json_report_path,
            content_type="application/json",
        ),
        explainability=MetricsSource(
            s3_uri=explainability_json_report_path,
            content_type="application/json",
        ),
    )
    step_args_register_model = automl_model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    register_model_step = ModelStep(
        name="ModelRegistrationStep", step_args=step_args_register_model
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            model_approval_status,
            model_package_group_name,
        ],
        steps=[automl_step, automl_model_step, register_model_step],
        sagemaker_session=pipeline_session,
    )

    try:
        _ = pipeline.create(role)
        execution = pipeline.start(parameters={})
        wait_pipeline_execution(execution=execution)

        execution_steps = execution.list_steps()
        has_automl_job = False
        for step in execution_steps:
            assert not step.get("FailureReason", None)
            assert step["StepStatus"] == "Succeeded"
            if "AutoMLJob" in step["Metadata"]:
                has_automl_job = True
                automl_job_arn = step["Metadata"]["AutoMLJob"]["Arn"]
                assert automl_job_arn is not None
                automl_job_name = re.findall(r"(?<=automl-job/).*", automl_job_arn)[0]
                auto_ml_desc = auto_ml.describe_auto_ml_job(job_name=automl_job_name)
                model_insights_json_from_automl = (
                    auto_ml_desc["BestCandidate"]["CandidateProperties"][
                        "CandidateArtifactLocations"
                    ]["ModelInsights"]
                    + "/statistics.json"
                )
                explainability_json_from_automl = (
                    auto_ml_desc["BestCandidate"]["CandidateProperties"][
                        "CandidateArtifactLocations"
                    ]["Explainability"]
                    + "/analysis.json"
                )

        assert has_automl_job
        assert len(execution_steps) == 3
        sagemaker_client = pipeline_session.boto_session.client("sagemaker")
        model_package = sagemaker_client.list_model_packages(
            ModelPackageGroupName="AutoMlModelPackageGroup"
        )["ModelPackageSummaryList"][0]
        response = sagemaker_client.describe_model_package(
            ModelPackageName=model_package["ModelPackageArn"]
        )
        model_insights_json_report_path = response["ModelMetrics"]["ModelQuality"]["Statistics"][
            "S3Uri"
        ]
        explainability_json_report_path = response["ModelMetrics"]["Explainability"]["Report"][
            "S3Uri"
        ]

        assert model_insights_json_report_path == model_insights_json_from_automl
        assert explainability_json_report_path == explainability_json_from_automl

    finally:
        try:
            sagemaker_client.delete_model_package(ModelPackageName=model_package["ModelPackageArn"])
            sagemaker_client.delete_model_package_group(
                ModelPackageGroupName="AutoMlModelPackageGroup"
            )

            pipeline.delete()
        except Exception:
            pass
