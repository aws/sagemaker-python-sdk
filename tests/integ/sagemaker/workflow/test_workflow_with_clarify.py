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

import json
import os
import statistics
import time
import tempfile

import pytest
import numpy as np
import pandas as pd

from tests.integ.sagemaker.workflow.helpers import wait_pipeline_execution
from sagemaker.amazon.linear_learner import LinearLearner, LinearLearnerPredictor
from sagemaker.clarify import (
    BiasConfig,
    DataConfig,
    ModelConfig,
    ModelPredictedLabelConfig,
    SageMakerClarifyProcessor,
)
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep, JsonGet

from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.pipeline import Pipeline

from sagemaker import utils
from tests import integ
from tests.integ import timeout


@pytest.fixture
def pipeline_name():
    return f"my-pipeline-clarify-{int(time.time() * 10**7)}"


@pytest.fixture(scope="module")
def training_set():
    label = (np.random.rand(100, 1) > 0.5).astype(np.int32)
    features = np.random.rand(100, 4)
    return features, label


@pytest.yield_fixture(scope="module")
def data_path(training_set):
    features, label = training_set
    data = pd.concat([pd.DataFrame(label), pd.DataFrame(features)], axis=1, sort=False)
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = os.path.join(tmpdirname, "train.csv")
        data.to_csv(filename, index=False, header=False)
        yield filename


@pytest.fixture(scope="module")
def headers():
    return [
        "Label",
        "F1",
        "F2",
        "F3",
        "F4",
    ]


@pytest.fixture(scope="module")
def data_config(sagemaker_session_for_pipeline, data_path, headers):
    output_path = (
        f"s3://{sagemaker_session_for_pipeline.default_bucket()}/linear_learner_analysis_result"
    )
    return DataConfig(
        s3_data_input_path=data_path,
        s3_output_path=output_path,
        label="Label",
        headers=headers,
        dataset_type="text/csv",
    )


@pytest.fixture(scope="module")
def data_bias_config():
    return BiasConfig(
        label_values_or_threshold=[1],
        facet_name="F1",
        facet_values_or_threshold=[0.5],
        group_name="F2",
    )


@pytest.yield_fixture(scope="module")
def model_name(sagemaker_session_for_pipeline, cpu_instance_type, training_set):
    job_name = utils.unique_name_from_base("clarify-xgb")

    with timeout.timeout(minutes=integ.TRAINING_DEFAULT_TIMEOUT_MINUTES):
        ll = LinearLearner(
            "SageMakerRole",
            1,
            cpu_instance_type,
            predictor_type="binary_classifier",
            sagemaker_session=sagemaker_session_for_pipeline,
            disable_profiler=True,
        )
        ll.binary_classifier_model_selection_criteria = "accuracy"
        ll.early_stopping_tolerance = 0.0001
        ll.early_stopping_patience = 3
        ll.num_models = 1
        ll.epochs = 1
        ll.num_calibration_samples = 1

        features, label = training_set
        ll.fit(
            ll.record_set(features.astype(np.float32), label.reshape(-1).astype(np.float32)),
            job_name=job_name,
        )

    with timeout.timeout_and_delete_endpoint_by_name(job_name, sagemaker_session_for_pipeline):
        ll.deploy(1, cpu_instance_type, endpoint_name=job_name, model_name=job_name, wait=True)
        yield job_name


@pytest.fixture(scope="module")
def model_config(model_name):
    return ModelConfig(
        model_name=model_name,
        instance_type="ml.c5.xlarge",
        instance_count=1,
        accept_type="application/jsonlines",
    )


@pytest.fixture(scope="module")
def model_predicted_label_config(sagemaker_session_for_pipeline, model_name, training_set):
    predictor = LinearLearnerPredictor(
        model_name,
        sagemaker_session=sagemaker_session_for_pipeline,
    )
    result = predictor.predict(training_set[0].astype(np.float32))
    predictions = [float(record.label["score"].float32_tensor.values[0]) for record in result]
    probability_threshold = statistics.median(predictions)
    return ModelPredictedLabelConfig(label="score", probability_threshold=probability_threshold)


def test_workflow_with_clarify(
    data_config,
    data_bias_config,
    model_config,
    model_predicted_label_config,
    pipeline_name,
    role,
    sagemaker_session_for_pipeline,
):

    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)

    analysis_config = data_config.get_config()
    analysis_config.update(data_bias_config.get_config())
    (
        probability_threshold,
        predictor_config,
    ) = model_predicted_label_config.get_predictor_config()
    predictor_config.update(model_config.get_predictor_config())
    analysis_config["methods"] = {"post_training_bias": {"methods": "all"}}
    analysis_config["predictor"] = predictor_config
    analysis_config["probability_threshold"] = probability_threshold
    analysis_config["methods"]["report"] = {"name": "report", "title": "Analysis Report"}

    with tempfile.TemporaryDirectory() as tmpdirname:
        analysis_config_file = os.path.join(tmpdirname, "analysis_config.json")
        with open(analysis_config_file, "w") as f:
            json.dump(analysis_config, f)
        config_input = ProcessingInput(
            input_name="analysis_config",
            source=analysis_config_file,
            destination="/opt/ml/processing/input/config",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
            s3_compression_type="None",
        )

        data_input = ProcessingInput(
            input_name="dataset",
            source=data_config.s3_data_input_path,
            destination="/opt/ml/processing/input/data",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
            s3_data_distribution_type=data_config.s3_data_distribution_type,
            s3_compression_type=data_config.s3_compression_type,
        )

        result_output = ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=data_config.s3_output_path,
            output_name="analysis_result",
            s3_upload_mode="EndOfJob",
        )

        processor = SageMakerClarifyProcessor(
            role="SageMakerRole",
            instance_count=instance_count,
            instance_type=instance_type,
            sagemaker_session=sagemaker_session_for_pipeline,
        )

        property_file = PropertyFile(
            name="BiasOutput",
            output_name="analysis_result",
            path="analysis.json",
        )

        step_process = ProcessingStep(
            name="my-process",
            processor=processor,
            inputs=[data_input, config_input],
            outputs=[result_output],
            property_files=[property_file],
        )

        # Keep the deprecated JsonGet in test to verify it's compatible with new changes
        cond_left = JsonGet(
            step=step_process,
            property_file="BiasOutput",
            json_path="post_training_bias_metrics.facets.F1[0].metrics[0].value",
        )

        step_condition = ConditionStep(
            name="bias-condition",
            conditions=[ConditionLessThanOrEqualTo(left=cond_left, right=1)],
            if_steps=[],
            else_steps=[],
        )

        pipeline = Pipeline(
            name=pipeline_name,
            parameters=[instance_type, instance_count],
            steps=[step_process, step_condition],
            sagemaker_session=sagemaker_session_for_pipeline,
        )

        try:
            response = pipeline.create(role)
            create_arn = response["PipelineArn"]

            execution = pipeline.start(parameters={})

            response = execution.describe()
            assert response["PipelineArn"] == create_arn

            wait_pipeline_execution(execution=execution)
            execution_steps = execution.list_steps()

            assert len(execution_steps) == 2
            assert execution_steps[1]["StepName"] == "my-process"
            assert execution_steps[1]["StepStatus"] == "Succeeded"
            assert execution_steps[0]["StepName"] == "bias-condition"
            assert execution_steps[0]["StepStatus"] == "Succeeded"
            assert execution_steps[0]["Metadata"]["Condition"]["Outcome"] == "True"

        finally:
            try:
                pipeline.delete()
            except Exception:
                pass
