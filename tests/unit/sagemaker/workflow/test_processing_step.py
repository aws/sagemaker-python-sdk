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
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import json

import pytest
import sagemaker
import warnings

from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker.processing import Processor, ScriptProcessor, FrameworkProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.tensorflow.processing import TensorFlowProcessor
from sagemaker.huggingface.processing import HuggingFaceProcessor
from sagemaker.xgboost.processing import XGBoostProcessor
from sagemaker.mxnet.processing import MXNetProcessor
from sagemaker.wrangler.processing import DataWranglerProcessor
from sagemaker.spark.processing import SparkJarProcessor, PySparkProcessor

from sagemaker.processing import ProcessingInput

from sagemaker.workflow.steps import CacheConfig, ProcessingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile

from sagemaker.network import NetworkConfig
from sagemaker.pytorch.estimator import PyTorch
from sagemaker import utils

from sagemaker.clarify import (
    SageMakerClarifyProcessor,
    BiasConfig,
    DataConfig,
    ModelConfig,
    ModelPredictedLabelConfig,
    SHAPConfig,
)
from tests.unit.sagemaker.workflow.helpers import CustomStep

REGION = "us-west-2"
IMAGE_URI = "fakeimage"
MODEL_NAME = "gisele"
DUMMY_S3_SCRIPT_PATH = "s3://dummy-s3/dummy_script.py"
INSTANCE_TYPE = "ml.m4.xlarge"


@pytest.fixture
def pipeline_session():
    return PipelineSession()


@pytest.fixture
def bucket(pipeline_session):
    return pipeline_session.default_bucket()


@pytest.fixture
def processing_input(bucket):
    return [
        ProcessingInput(
            source=f"s3://{bucket}/processing_manifest",
            destination="processing_manifest",
        )
    ]


@pytest.fixture
def network_config():
    return NetworkConfig(
        subnets=["my_subnet_id"],
        security_group_ids=["my_security_group_id"],
        enable_network_isolation=True,
        encrypt_inter_container_traffic=True,
    )


def test_processing_step_with_processor(pipeline_session, processing_input):
    custom_step1 = CustomStep("TestStep")
    custom_step2 = CustomStep("SecondTestStep")
    processor = Processor(
        image_uri=IMAGE_URI,
        role=sagemaker.get_execution_role(),
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        sagemaker_session=pipeline_session,
    )

    with warnings.catch_warnings(record=True) as w:
        step_args = processor.run(inputs=processing_input)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Running within a PipelineSession" in str(w[-1].message)

    cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")
    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )

    with warnings.catch_warnings(record=True) as w:
        step = ProcessingStep(
            name="MyProcessingStep",
            step_args=step_args,
            description="ProcessingStep description",
            display_name="MyProcessingStep",
            depends_on=["TestStep", "SecondTestStep"],
            cache_config=cache_config,
            property_files=[evaluation_report],
        )
        assert len(w) == 0

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step, custom_step1, custom_step2],
        sagemaker_session=pipeline_session,
    )
    assert json.loads(pipeline.definition())["Steps"][0] == {
        "Name": "MyProcessingStep",
        "Description": "ProcessingStep description",
        "DisplayName": "MyProcessingStep",
        "Type": "Processing",
        "DependsOn": ["TestStep", "SecondTestStep"],
        "Arguments": step_args,
        "CacheConfig": {"Enabled": True, "ExpireAfter": "PT1H"},
        "PropertyFiles": [
            {
                "FilePath": "evaluation.json",
                "OutputName": "evaluation",
                "PropertyFileName": "EvaluationReport",
            }
        ],
    }
    assert step.properties.ProcessingJobName.expr == {
        "Get": "Steps.MyProcessingStep.ProcessingJobName"
    }


def test_processing_step_with_processor_and_step_args(pipeline_session, processing_input):
    processor = Processor(
        image_uri=IMAGE_URI,
        role=sagemaker.get_execution_role(),
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        sagemaker_session=pipeline_session,
    )

    step_args = processor.run(inputs=processing_input)

    try:
        ProcessingStep(
            name="MyProcessingStep",
            step_args=step_args,
            processor=processor,
        )
        assert False
    except Exception as e:
        assert isinstance(e, ValueError)

    try:
        ProcessingStep(
            name="MyProcessingStep",
        )
        assert False
    except Exception as e:
        assert isinstance(e, ValueError)


def test_processing_step_with_script_processor(pipeline_session, processing_input, network_config):
    processor = ScriptProcessor(
        role=sagemaker.get_execution_role(),
        image_uri=IMAGE_URI,
        command=["python3"],
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        volume_size_in_gb=100,
        volume_kms_key="volume-kms-key",
        output_kms_key="output-kms-key",
        max_runtime_in_seconds=3600,
        base_job_name="my_sklearn_processor",
        env={"my_env_variable": "my_env_variable_value"},
        tags=[{"Key": "my-tag", "Value": "my-tag-value"}],
        network_config=network_config,
        sagemaker_session=pipeline_session,
    )

    step_args = processor.run(
        inputs=processing_input, code=DUMMY_S3_SCRIPT_PATH, job_name="my-processing-job"
    )

    step = ProcessingStep(
        name="MyProcessingStep",
        step_args=step_args,
    )

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step],
        sagemaker_session=pipeline_session,
    )

    assert json.loads(pipeline.definition())["Steps"][0] == {
        "Name": "MyProcessingStep",
        "Type": "Processing",
        "Arguments": step_args,
    }


@pytest.mark.parametrize(
    "framework_processor",
    [
        (
            FrameworkProcessor(
                framework_version="1.8",
                instance_type=INSTANCE_TYPE,
                instance_count=1,
                role=sagemaker.get_execution_role(),
                estimator_cls=PyTorch,
            ),
            {"code": DUMMY_S3_SCRIPT_PATH},
        ),
        (
            SKLearnProcessor(
                framework_version="1.0-1",
                instance_type=INSTANCE_TYPE,
                instance_count=1,
                role=sagemaker.get_execution_role(),
            ),
            {"code": DUMMY_S3_SCRIPT_PATH},
        ),
        (
            PyTorchProcessor(
                role=sagemaker.get_execution_role(),
                instance_type=INSTANCE_TYPE,
                instance_count=1,
                framework_version="1.8.0",
                py_version="py3",
            ),
            {"code": DUMMY_S3_SCRIPT_PATH},
        ),
        (
            TensorFlowProcessor(
                role=sagemaker.get_execution_role(),
                instance_type=INSTANCE_TYPE,
                instance_count=1,
                framework_version="2.0",
            ),
            {"code": DUMMY_S3_SCRIPT_PATH},
        ),
        (
            HuggingFaceProcessor(
                transformers_version="4.6",
                pytorch_version="1.7",
                role=sagemaker.get_execution_role(),
                instance_count=1,
                instance_type="ml.p3.2xlarge",
            ),
            {"code": DUMMY_S3_SCRIPT_PATH},
        ),
        (
            XGBoostProcessor(
                framework_version="1.3-1",
                py_version="py3",
                role=sagemaker.get_execution_role(),
                instance_count=1,
                instance_type=INSTANCE_TYPE,
                base_job_name="test-xgboost",
            ),
            {"code": DUMMY_S3_SCRIPT_PATH},
        ),
        (
            MXNetProcessor(
                framework_version="1.4.1",
                py_version="py3",
                role=sagemaker.get_execution_role(),
                instance_count=1,
                instance_type=INSTANCE_TYPE,
                base_job_name="test-mxnet",
            ),
            {"code": DUMMY_S3_SCRIPT_PATH},
        ),
        (
            DataWranglerProcessor(
                role=sagemaker.get_execution_role(),
                data_wrangler_flow_source="s3://my-bucket/dw.flow",
                instance_count=1,
                instance_type=INSTANCE_TYPE,
            ),
            {},
        ),
        (
            SparkJarProcessor(
                role=sagemaker.get_execution_role(),
                framework_version="2.4",
                instance_count=1,
                instance_type=INSTANCE_TYPE,
            ),
            {
                "submit_app": "s3://my-jar",
                "submit_class": "com.amazonaws.sagemaker.spark.test.HelloJavaSparkApp",
                "arguments": ["--input", "input-data-uri", "--output", "output-data-uri"],
            },
        ),
        (
            PySparkProcessor(
                role=sagemaker.get_execution_role(),
                framework_version="2.4",
                instance_count=1,
                instance_type=INSTANCE_TYPE,
            ),
            {
                "submit_app": "s3://my-jar",
                "arguments": ["--input", "input-data-uri", "--output", "output-data-uri"],
            },
        ),
    ],
)
def test_processing_step_with_framework_processor(
    framework_processor, pipeline_session, processing_input, network_config
):

    processor, run_inputs = framework_processor
    processor.sagemaker_session = pipeline_session
    processor.role = sagemaker.get_execution_role()

    processor.volume_kms_key = "volume-kms-key"
    processor.network_config = network_config

    run_inputs["inputs"] = processing_input

    step_args = processor.run(**run_inputs)

    step = ProcessingStep(
        name="MyProcessingStep",
        step_args=step_args,
    )
    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step],
        sagemaker_session=pipeline_session,
    )

    assert json.loads(pipeline.definition())["Steps"][0] == {
        "Name": "MyProcessingStep",
        "Type": "Processing",
        "Arguments": step_args,
    }


def test_processing_step_with_clarify_processor(pipeline_session):
    def headers():
        return [
            "Label",
            "F1",
            "F2",
            "F3",
            "F4",
        ]

    def data_bias_config():
        return BiasConfig(
            label_values_or_threshold=[1],
            facet_name="F1",
            facet_values_or_threshold=[0.5],
            group_name="F2",
        )

    def model_config(model_name):
        return ModelConfig(
            model_name=model_name,
            instance_type="ml.c5.xlarge",
            instance_count=1,
            accept_type="application/jsonlines",
            endpoint_name_prefix="myprefix",
        )

    def shap_config():
        return SHAPConfig(
            baseline=[
                [
                    0.94672389,
                    0.47108862,
                    0.63350081,
                    0.00604642,
                ]
            ],
            num_samples=2,
            agg_method="mean_sq",
            seed=123,
        )

    def verfiy(step_args):
        step = ProcessingStep(
            name="MyProcessingStep",
            step_args=step_args,
        )
        pipeline = Pipeline(
            name="MyPipeline",
            steps=[step],
            sagemaker_session=pipeline_session,
        )
        assert json.loads(pipeline.definition())["Steps"][0] == {
            "Name": "MyProcessingStep",
            "Type": "Processing",
            "Arguments": step_args,
        }

    test_run = utils.unique_name_from_base("test_run")
    output_path = "s3://{}/{}/{}".format(
        pipeline_session.default_bucket(), "linear_learner_analysis_result", test_run
    )
    data_config = DataConfig(
        s3_data_input_path=f"s3://{pipeline_session.default_bucket()}/{input}/train.csv",
        s3_output_path=output_path,
        label="Label",
        headers=headers(),
        dataset_type="text/csv",
    )

    clarify_processor = SageMakerClarifyProcessor(
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        sagemaker_session=pipeline_session,
        role=sagemaker.get_execution_role(),
    )

    run_bias_args = clarify_processor.run_bias(
        data_config=data_config,
        bias_config=data_bias_config(),
        model_config=model_config("1st-model-rpyndy0uyo"),
    )
    verfiy(run_bias_args)

    run_pre_training_bias_args = clarify_processor.run_pre_training_bias(
        data_config=data_config,
        data_bias_config=data_bias_config(),
    )
    verfiy(run_pre_training_bias_args)

    run_post_training_bias_args = clarify_processor.run_post_training_bias(
        data_config=data_config,
        data_bias_config=data_bias_config(),
        model_config=model_config("1st-model-rpyndy0uyo"),
        model_predicted_label_config=ModelPredictedLabelConfig(probability_threshold=0.9),
    )
    verfiy(run_post_training_bias_args)

    run_explainability_args = clarify_processor.run_explainability(
        data_config=data_config,
        model_config=model_config("1st-model-rpyndy0uyo"),
        explainability_config=shap_config(),
    )
    verfiy(run_explainability_args)
