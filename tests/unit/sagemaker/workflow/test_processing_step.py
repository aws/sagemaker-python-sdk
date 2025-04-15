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
from mock import patch

import pytest
import warnings

from copy import deepcopy

from sagemaker.estimator import Estimator
from sagemaker.parameter import IntegerParameter
from sagemaker.transformer import Transformer
from sagemaker.tuner import HyperparameterTuner

from sagemaker.processing import (
    Processor,
    ScriptProcessor,
    FrameworkProcessor,
    ProcessingOutput,
    ProcessingInput,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.tensorflow.processing import TensorFlowProcessor
from sagemaker.huggingface.processing import HuggingFaceProcessor
from sagemaker.xgboost.processing import XGBoostProcessor
from sagemaker.mxnet.processing import MXNetProcessor
from sagemaker.wrangler.processing import DataWranglerProcessor
from sagemaker.spark.processing import SparkJarProcessor, PySparkProcessor


from sagemaker.workflow.steps import CacheConfig, ProcessingStep
from sagemaker.workflow.pipeline import Pipeline, PipelineGraph
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.pipeline_context import _PipelineConfig
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.functions import Join
from sagemaker.workflow.utilities import hash_files_or_dirs
from sagemaker.workflow import is_pipeline_variable

from sagemaker.network import NetworkConfig
from sagemaker.pytorch.estimator import PyTorch
from sagemaker import utils, Model

from sagemaker.clarify import (
    SageMakerClarifyProcessor,
    BiasConfig,
    DataConfig,
    ModelConfig,
    ModelPredictedLabelConfig,
    SHAPConfig,
)
from tests.unit.sagemaker.workflow.helpers import CustomStep, ordered, get_step_args_helper
from tests.unit import DATA_DIR
from tests.unit.sagemaker.workflow.conftest import ROLE, BUCKET, IMAGE_URI, INSTANCE_TYPE

HF_INSTANCE_TYPE = "ml.p3.2xlarge"
DUMMY_S3_SCRIPT_PATH = "s3://dummy-s3/dummy_script.py"
LOCAL_SCRIPT_PATH = os.path.join(DATA_DIR, "workflow/abalone/preprocessing.py")
SPARK_APP_JAR_PATH = os.path.join(
    DATA_DIR, "spark/code/java/hello-java-spark/HelloJavaSparkApp.jar"
)
SPARK_DEP_JAR = os.path.join(DATA_DIR, "spark/code/java/TestJarFile.jar")
SPARK_APP_PY_PATH = os.path.join(DATA_DIR, "spark/code/python/hello_py_spark/hello_py_spark_app.py")
SPARK_PY_FILE1 = os.path.join(DATA_DIR, "spark/code/python/hello_py_spark/__init__.py")
SPARK_PY_FILE2 = os.path.join(DATA_DIR, "spark/code/python/hello_py_spark/hello_py_spark_udfs.py")
SPARK_SUBMIT_FILE1 = os.path.join(DATA_DIR, "spark/files/data.jsonl")
SPARK_SUBMIT_FILE2 = os.path.join(DATA_DIR, "spark/files/sample_spark_event_logs")
MOCKED_PIPELINE_CONFIG = _PipelineConfig(
    "MyPipeline",
    "MyProcessingStep",
    None,
    hash_files_or_dirs([LOCAL_SCRIPT_PATH]),
    "config-hash-abcdefg",
    None,
)

_DEFINITION_CONFIG = PipelineDefinitionConfig(use_custom_job_prefix=True)
MOCKED_PIPELINE_CONFIG_WITH_CUSTOM_PREFIX = _PipelineConfig(
    "MyPipelineWithCustomPrefix",
    "MyProcessingStep",
    None,
    None,
    None,
    _DEFINITION_CONFIG,
)

FRAMEWORK_PROCESSOR = [
    (
        FrameworkProcessor(
            framework_version="1.8",
            instance_type=INSTANCE_TYPE,
            instance_count=1,
            role=ROLE,
            estimator_cls=PyTorch,
        ),
        {"code": DUMMY_S3_SCRIPT_PATH},
    ),
    (
        SKLearnProcessor(
            framework_version="0.23-1",
            instance_type=INSTANCE_TYPE,
            instance_count=1,
            role=ROLE,
        ),
        {"code": DUMMY_S3_SCRIPT_PATH},
    ),
    (
        PyTorchProcessor(
            role=ROLE,
            instance_type=INSTANCE_TYPE,
            instance_count=1,
            framework_version="1.8.0",
            py_version="py3",
        ),
        {"code": DUMMY_S3_SCRIPT_PATH},
    ),
    (
        TensorFlowProcessor(
            role=ROLE,
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
            role=ROLE,
            instance_count=1,
            instance_type=HF_INSTANCE_TYPE,
        ),
        {"code": DUMMY_S3_SCRIPT_PATH},
    ),
    (
        XGBoostProcessor(
            framework_version="1.3-1",
            py_version="py3",
            role=ROLE,
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
            role=ROLE,
            instance_count=1,
            instance_type=INSTANCE_TYPE,
            base_job_name="test-mxnet",
        ),
        {"code": DUMMY_S3_SCRIPT_PATH},
    ),
    (
        DataWranglerProcessor(
            role=ROLE,
            data_wrangler_flow_source="s3://my-bucket/dw.flow",
            instance_count=1,
            instance_type=INSTANCE_TYPE,
        ),
        {},
    ),
]

FRAMEWORK_PROCESSOR_LOCAL_CODE = [
    (
        FrameworkProcessor(
            framework_version="1.8",
            instance_type=INSTANCE_TYPE,
            instance_count=1,
            role=ROLE,
            estimator_cls=PyTorch,
        ),
        {"code": LOCAL_SCRIPT_PATH},
    ),
]

PROCESSING_INPUT = [
    ProcessingInput(source="s3://my-bucket/processing_manifest", destination="processing_manifest"),
    ProcessingInput(
        source=ParameterString(name="my-processing-input"),
        destination="processing-input",
    ),
    ProcessingInput(
        source=ParameterString(
            name="my-processing-input", default_value="s3://my-bucket/my-processing"
        ),
        destination="processing-input",
    ),
    ProcessingInput(
        source=Join(on="/", values=["s3://my-bucket", "my-input"]),
        destination="processing-input",
    ),
]

PROCESSING_OUTPUT = [
    ProcessingOutput(source="/opt/ml/output", destination="s3://my-bucket/my-output"),
    ProcessingOutput(source="/opt/ml/output", destination=ParameterString(name="my-output")),
    ProcessingOutput(
        source="/opt/ml/output",
        destination=ParameterString(name="my-output", default_value="s3://my-bucket/my-output"),
    ),
    ProcessingOutput(
        source="/opt/ml/output",
        destination=Join(on="/", values=["s3://my-bucket", "my-output"]),
    ),
]


@pytest.fixture
def processing_input():
    return [
        ProcessingInput(
            source=f"s3://{BUCKET}/processing_manifest",
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


@pytest.mark.parametrize(
    "experiment_config, expected_experiment_config",
    [
        (
            {
                "ExperimentName": "experiment-name",
                "TrialName": "trial-name",
                "TrialComponentDisplayName": "display-name",
            },
            {"TrialComponentDisplayName": "display-name"},
        ),
        (
            {"TrialComponentDisplayName": "display-name"},
            {"TrialComponentDisplayName": "display-name"},
        ),
        (
            {
                "ExperimentName": "experiment-name",
                "TrialName": "trial-name",
            },
            None,
        ),
        (None, None),
    ],
)
def test_processing_step_with_processor(
    pipeline_session, processing_input, experiment_config, expected_experiment_config
):
    custom_step1 = CustomStep("TestStep")
    custom_step2 = CustomStep("SecondTestStep")
    processor = Processor(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        sagemaker_session=pipeline_session,
    )

    with warnings.catch_warnings(record=True) as w:
        step_args = processor.run(inputs=processing_input, experiment_config=experiment_config)
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
    step_args = get_step_args_helper(step_args, "Processing")

    expected_step_arguments = deepcopy(step_args)
    if expected_experiment_config is None:
        expected_step_arguments.pop("ExperimentConfig", None)
    else:
        expected_step_arguments["ExperimentConfig"] = expected_experiment_config

    step_def = json.loads(pipeline.definition())["Steps"][0]
    assert step_def == {
        "Name": "MyProcessingStep",
        "Description": "ProcessingStep description",
        "DisplayName": "MyProcessingStep",
        "Type": "Processing",
        "DependsOn": ["TestStep", "SecondTestStep"],
        "Arguments": expected_step_arguments,
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
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered(
        {
            "TestStep": ["MyProcessingStep"],
            "SecondTestStep": ["MyProcessingStep"],
            "MyProcessingStep": [],
        }
    )

    # test idempotency
    step_def2 = json.loads(pipeline.definition())["Steps"][0]
    assert step_def == step_def2


@pytest.mark.parametrize(
    "image_uri",
    [
        IMAGE_URI,
        ParameterString(name="MyImage"),
        ParameterString(name="MyImage", default_value="my-image-uri"),
        Join(on="/", values=["docker", "my-fake-image"]),
    ],
)
def test_processing_step_with_processor_and_step_args(
    pipeline_session, processing_input, image_uri
):
    processor = Processor(
        image_uri=image_uri,
        role=ROLE,
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


@patch("sagemaker.workflow.utilities._pipeline_config", MOCKED_PIPELINE_CONFIG)
@pytest.mark.parametrize("code_artifact", [DUMMY_S3_SCRIPT_PATH, LOCAL_SCRIPT_PATH])
def test_processing_step_with_script_processor(
    pipeline_session, processing_input, network_config, code_artifact
):
    processor = ScriptProcessor(
        role=ROLE,
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
        inputs=processing_input, code=code_artifact, job_name="my-processing-job"
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

    step_args = get_step_args_helper(step_args, "Processing")
    step_def = json.loads(pipeline.definition())["Steps"][0]
    assert step_def == {"Name": "MyProcessingStep", "Type": "Processing", "Arguments": step_args}

    # test idempotency
    step_def2 = json.loads(pipeline.definition())["Steps"][0]
    assert step_def == step_def2


@pytest.mark.parametrize("framework_processor", FRAMEWORK_PROCESSOR)
@pytest.mark.parametrize("processing_input", PROCESSING_INPUT)
@pytest.mark.parametrize("processing_output", PROCESSING_OUTPUT)
def test_processing_step_with_framework_processor(
    framework_processor, pipeline_session, processing_input, processing_output, network_config
):

    processor, run_inputs = framework_processor
    default_instance_type = (
        HF_INSTANCE_TYPE if type(processor) is HuggingFaceProcessor else INSTANCE_TYPE
    )
    instance_type_param = ParameterString(
        name="ProcessingInstanceType", default_value=default_instance_type
    )
    processor.sagemaker_session = pipeline_session
    processor.role = ROLE
    processor.instance_type = instance_type_param

    processor.volume_kms_key = "volume-kms-key"
    processor.network_config = network_config

    run_inputs["inputs"] = [processing_input]
    run_inputs["outputs"] = [processing_output]

    step_args = processor.run(**run_inputs)

    step = ProcessingStep(
        name="MyProcessingStep",
        step_args=step_args,
    )
    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step],
        sagemaker_session=pipeline_session,
        parameters=[instance_type_param],
    )

    step_args = get_step_args_helper(step_args, "Processing")
    step_def = json.loads(pipeline.definition())["Steps"][0]

    assert step_args["ProcessingInputs"][0]["S3Input"]["S3Uri"] == processing_input.source
    assert (
        step_args["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
        == processing_output.destination
    )
    assert (
        type(step_args["ProcessingResources"]["ClusterConfig"]["InstanceType"]) is ParameterString
    )
    step_args["ProcessingResources"]["ClusterConfig"]["InstanceType"] = step_args[
        "ProcessingResources"
    ]["ClusterConfig"]["InstanceType"].expr

    del step_args["ProcessingInputs"][0]["S3Input"]["S3Uri"]
    del step_def["Arguments"]["ProcessingInputs"][0]["S3Input"]["S3Uri"]

    del step_args["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
    del step_def["Arguments"]["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]

    assert step_def == {
        "Name": "MyProcessingStep",
        "Type": "Processing",
        "Arguments": step_args,
    }

    # test idempotency
    step_def2 = json.loads(pipeline.definition())["Steps"][0]
    del step_def2["Arguments"]["ProcessingInputs"][0]["S3Input"]["S3Uri"]
    del step_def2["Arguments"]["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
    assert step_def == step_def2


@patch("sagemaker.workflow.utilities._pipeline_config", MOCKED_PIPELINE_CONFIG)
@pytest.mark.parametrize("framework_processor", FRAMEWORK_PROCESSOR_LOCAL_CODE)
def test_processing_step_with_framework_processor_local_code(
    framework_processor, pipeline_session, network_config
):
    processor, run_inputs = framework_processor
    processor.sagemaker_session = pipeline_session
    processor.role = ROLE

    processor.volume_kms_key = "volume-kms-key"
    processor.network_config = network_config

    processing_input = ProcessingInput(
        source="s3://my-bucket/processing_manifest",
        destination="processing_manifest",
        input_name="manifest",
    )
    processing_output = ProcessingOutput(
        output_name="framework_output", source="/opt/ml/processing/framework_output"
    )

    run_inputs["inputs"] = [processing_input]
    run_inputs["outputs"] = [processing_output]

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

    step_args = get_step_args_helper(step_args, "Processing")
    step_def = json.loads(pipeline.definition())["Steps"][0]

    del step_args["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
    del step_def["Arguments"]["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]

    assert step_def == {
        "Name": "MyProcessingStep",
        "Type": "Processing",
        "Arguments": step_args,
    }

    # test idempotency
    step_def2 = json.loads(pipeline.definition())["Steps"][0]
    del step_def2["Arguments"]["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
    assert step_def == step_def2


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

    def verify(step_args):
        step = ProcessingStep(
            name="MyProcessingStep",
            step_args=step_args,
        )
        pipeline = Pipeline(
            name="MyPipeline",
            steps=[step],
            sagemaker_session=pipeline_session,
        )
        step_def = json.loads(pipeline.definition())["Steps"][0]
        assert step_def == {
            "Name": "MyProcessingStep",
            "Type": "Processing",
            "Arguments": get_step_args_helper(step_args, "Processing"),
        }

        # test idempotency
        step_def2 = json.loads(pipeline.definition())["Steps"][0]
        assert step_def == step_def2

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
        role=ROLE,
    )

    run_bias_args = clarify_processor.run_bias(
        data_config=data_config,
        bias_config=data_bias_config(),
        model_config=model_config("1st-model-rpyndy0uyo"),
    )
    verify(run_bias_args)

    run_pre_training_bias_args = clarify_processor.run_pre_training_bias(
        data_config=data_config,
        data_bias_config=data_bias_config(),
    )
    verify(run_pre_training_bias_args)

    run_post_training_bias_args = clarify_processor.run_post_training_bias(
        data_config=data_config,
        data_bias_config=data_bias_config(),
        model_config=model_config("1st-model-rpyndy0uyo"),
        model_predicted_label_config=ModelPredictedLabelConfig(probability_threshold=0.9),
    )
    verify(run_post_training_bias_args)

    run_explainability_args = clarify_processor.run_explainability(
        data_config=data_config,
        model_config=model_config("1st-model-rpyndy0uyo"),
        explainability_config=shap_config(),
    )
    verify(run_explainability_args)


@pytest.mark.parametrize(
    "inputs",
    [
        (
            Transformer(
                model_name="model_name",
                instance_type="ml.m5.xlarge",
                instance_count=1,
                output_path="s3://Transform",
            ),
            dict(
                target_fun="transform",
                func_args=dict(data="s3://data", job_name="test"),
            ),
        ),
        (
            Estimator(
                role=ROLE,
                instance_count=1,
                instance_type=INSTANCE_TYPE,
                image_uri=IMAGE_URI,
            ),
            dict(
                target_fun="fit",
                func_args={},
            ),
        ),
        (
            HyperparameterTuner(
                estimator=Estimator(
                    role=ROLE,
                    instance_count=1,
                    instance_type=INSTANCE_TYPE,
                    image_uri=IMAGE_URI,
                ),
                objective_metric_name="test:acc",
                hyperparameter_ranges={"batch-size": IntegerParameter(64, 128)},
            ),
            dict(target_fun="fit", func_args={}),
        ),
        (
            Model(
                image_uri=IMAGE_URI,
                role=ROLE,
            ),
            dict(target_fun="create", func_args={}),
        ),
    ],
)
def test_insert_wrong_step_args_into_processing_step(inputs, pipeline_session):
    downstream_obj, target_func_cfg = inputs
    if isinstance(downstream_obj, HyperparameterTuner):
        downstream_obj.estimator.sagemaker_session = pipeline_session
    else:
        downstream_obj.sagemaker_session = pipeline_session
    func_name = target_func_cfg["target_fun"]
    func_args = target_func_cfg["func_args"]
    step_args = getattr(downstream_obj, func_name)(**func_args)

    with pytest.raises(ValueError) as error:
        ProcessingStep(
            name="MyProcessingStep",
            step_args=step_args,
        )

    assert "The step_args of ProcessingStep must be obtained from processor.run()" in str(
        error.value
    )


@pytest.mark.parametrize(
    "spark_processor",
    [
        (
            SparkJarProcessor(
                role=ROLE,
                framework_version="2.4",
                instance_count=1,
                instance_type=INSTANCE_TYPE,
            ),
            {
                "submit_app": "s3://my-jar",
                "submit_class": "com.amazonaws.sagemaker.spark.test.HelloJavaSparkApp",
                "arguments": [
                    "--input",
                    "input-data-uri",
                    "--output",
                    ParameterString("MyArgOutput"),
                ],
                "submit_jars": [
                    "s3://my-jar",
                    ParameterString("MyJars"),
                    "s3://her-jar",
                    ParameterString("OurJar"),
                ],
                "submit_files": [
                    "s3://my-files",
                    ParameterString("MyFiles"),
                    "s3://her-files",
                    ParameterString("OurFiles"),
                ],
                "spark_event_logs_s3_uri": ParameterString("MySparkEventLogS3Uri"),
            },
        ),
        (
            PySparkProcessor(
                role=ROLE,
                framework_version="2.4",
                instance_count=1,
                instance_type=INSTANCE_TYPE,
            ),
            {
                "submit_app": "s3://my-jar",
                "arguments": [
                    "--input",
                    "input-data-uri",
                    "--output",
                    ParameterString("MyArgOutput"),
                ],
                "submit_py_files": [
                    "s3://my-py-files",
                    ParameterString("MyPyFiles"),
                    "s3://her-pyfiles",
                    ParameterString("OurPyFiles"),
                ],
                "submit_jars": [
                    "s3://my-jar",
                    ParameterString("MyJars"),
                    "s3://her-jar",
                    ParameterString("OurJar"),
                ],
                "submit_files": [
                    "s3://my-files",
                    ParameterString("MyFiles"),
                    "s3://her-files",
                    ParameterString("OurFiles"),
                ],
                "spark_event_logs_s3_uri": ParameterString("MySparkEventLogS3Uri"),
            },
        ),
    ],
)
def test_spark_processor(spark_processor, processing_input, pipeline_session):

    processor, run_inputs = spark_processor
    processor.sagemaker_session = pipeline_session
    processor.role = ROLE
    arguments_output = [
        "--input",
        "input-data-uri",
        "--output",
        '{"Get": "Parameters.MyArgOutput"}',
    ]
    run_inputs["inputs"] = processing_input

    step_args = processor.run(**run_inputs)
    step = ProcessingStep(
        name="MyProcessingStep",
        step_args=step_args,
    )

    step_args = get_step_args_helper(step_args, "Processing")

    assert step_args["AppSpecification"]["ContainerArguments"] == arguments_output

    entry_points = step_args["AppSpecification"]["ContainerEntrypoint"]
    entry_points_expr = []
    for entry_point in entry_points:
        if is_pipeline_variable(entry_point):
            entry_points_expr.append(entry_point.expr)
        else:
            entry_points_expr.append(entry_point)

    if "submit_py_files" in run_inputs:
        expected = [
            "smspark-submit",
            "--py-files",
            {
                "Std:Join": {
                    "On": ",",
                    "Values": [
                        "s3://my-py-files",
                        {"Get": "Parameters.MyPyFiles"},
                        "s3://her-pyfiles",
                        {"Get": "Parameters.OurPyFiles"},
                    ],
                }
            },
            "--jars",
            {
                "Std:Join": {
                    "On": ",",
                    "Values": [
                        "s3://my-jar",
                        {"Get": "Parameters.MyJars"},
                        "s3://her-jar",
                        {"Get": "Parameters.OurJar"},
                    ],
                }
            },
            "--files",
            {
                "Std:Join": {
                    "On": ",",
                    "Values": [
                        "s3://my-files",
                        {"Get": "Parameters.MyFiles"},
                        "s3://her-files",
                        {"Get": "Parameters.OurFiles"},
                    ],
                }
            },
            "--local-spark-event-logs-dir",
            "/opt/ml/processing/spark-events/",
            "/opt/ml/processing/input/code",
        ]
        # py spark
    else:
        expected = [
            "smspark-submit",
            "--class",
            "com.amazonaws.sagemaker.spark.test.HelloJavaSparkApp",
            "--jars",
            {
                "Std:Join": {
                    "On": ",",
                    "Values": [
                        "s3://my-jar",
                        {"Get": "Parameters.MyJars"},
                        "s3://her-jar",
                        {"Get": "Parameters.OurJar"},
                    ],
                }
            },
            "--files",
            {
                "Std:Join": {
                    "On": ",",
                    "Values": [
                        "s3://my-files",
                        {"Get": "Parameters.MyFiles"},
                        "s3://her-files",
                        {"Get": "Parameters.OurFiles"},
                    ],
                }
            },
            "--local-spark-event-logs-dir",
            "/opt/ml/processing/spark-events/",
            "/opt/ml/processing/input/code",
        ]

    assert entry_points_expr == expected
    for output in step_args["ProcessingOutputConfig"]["Outputs"]:
        if is_pipeline_variable(output["S3Output"]["S3Uri"]):
            output["S3Output"]["S3Uri"] = output["S3Output"]["S3Uri"].expr

    assert step_args["ProcessingOutputConfig"]["Outputs"] == [
        {
            "OutputName": "output-1",
            "AppManaged": False,
            "S3Output": {
                "S3Uri": {"Get": "Parameters.MySparkEventLogS3Uri"},
                "LocalPath": "/opt/ml/processing/spark-events/",
                "S3UploadMode": "Continuous",
            },
        }
    ]

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step],
        sagemaker_session=pipeline_session,
    )

    # test for idempotency
    step_def = json.loads(pipeline.definition())["Steps"][0]
    step_def_2 = json.loads(pipeline.definition())["Steps"][0]
    assert step_def == step_def_2


@pytest.mark.parametrize(
    "spark_processor",
    [
        (
            SparkJarProcessor(
                role=ROLE,
                framework_version="2.4",
                instance_count=1,
                instance_type=INSTANCE_TYPE,
            ),
            {
                "submit_app": SPARK_APP_JAR_PATH,
                "submit_class": "com.amazonaws.sagemaker.spark.test.HelloJavaSparkApp",
                "arguments": [
                    "--input",
                    "input-data-uri",
                    "--output",
                    ParameterString("MyArgOutput"),
                ],
                "submit_jars": [
                    SPARK_DEP_JAR,
                ],
                "submit_files": [
                    SPARK_SUBMIT_FILE1,
                    SPARK_SUBMIT_FILE2,
                ],
                "spark_event_logs_s3_uri": ParameterString("MySparkEventLogS3Uri"),
                "configuration": {
                    "Classification": "core-site",
                    "Properties": {"hadoop.security.groups.cache.secs": "250"},
                },
            },
        ),
        (
            PySparkProcessor(
                role=ROLE,
                framework_version="2.4",
                instance_count=1,
                instance_type=INSTANCE_TYPE,
            ),
            {
                "submit_app": SPARK_APP_PY_PATH,
                "arguments": [
                    "--input",
                    "input-data-uri",
                    "--output",
                    ParameterString("MyArgOutput"),
                ],
                "submit_py_files": [
                    SPARK_PY_FILE1,
                    SPARK_PY_FILE2,
                ],
                "submit_jars": [SPARK_DEP_JAR],
                "submit_files": [SPARK_SUBMIT_FILE1, SPARK_SUBMIT_FILE2],
                "spark_event_logs_s3_uri": ParameterString("MySparkEventLogS3Uri"),
                "configuration": {
                    "Classification": "core-site",
                    "Properties": {"hadoop.security.groups.cache.secs": "250"},
                },
            },
        ),
    ],
)
def test_spark_processor_local_code(spark_processor, processing_input, pipeline_session):
    processor, run_inputs = spark_processor
    processor.sagemaker_session = pipeline_session
    processor.role = ROLE
    arguments_output = [
        "--input",
        "input-data-uri",
        "--output",
        '{"Get": "Parameters.MyArgOutput"}',
    ]

    run_inputs["inputs"] = processing_input

    step_args = processor.run(**run_inputs)
    step = ProcessingStep(
        name="MyProcessingStep",
        step_args=step_args,
    )

    step_args = get_step_args_helper(step_args, "Processing")

    assert step_args["AppSpecification"]["ContainerArguments"] == arguments_output

    entry_points = step_args["AppSpecification"]["ContainerEntrypoint"]
    entry_points_expr = []
    for entry_point in entry_points:
        if is_pipeline_variable(entry_point):
            entry_points_expr.append(entry_point.expr)
        else:
            entry_points_expr.append(entry_point)

    if "submit_py_files" in run_inputs:
        expected = [
            "smspark-submit",
            "--py-files",
            "/opt/ml/processing/input/py-files",
            "--jars",
            "/opt/ml/processing/input/jars",
            "--files",
            "/opt/ml/processing/input/files",
            "--local-spark-event-logs-dir",
            "/opt/ml/processing/spark-events/",
            "/opt/ml/processing/input/code/hello_py_spark_app.py",
        ]
        # py spark
    else:
        expected = [
            "smspark-submit",
            "--class",
            "com.amazonaws.sagemaker.spark.test.HelloJavaSparkApp",
            "--jars",
            "/opt/ml/processing/input/jars",
            "--files",
            "/opt/ml/processing/input/files",
            "--local-spark-event-logs-dir",
            "/opt/ml/processing/spark-events/",
            "/opt/ml/processing/input/code/HelloJavaSparkApp.jar",
        ]

    assert entry_points_expr == expected
    for output in step_args["ProcessingOutputConfig"]["Outputs"]:
        if is_pipeline_variable(output["S3Output"]["S3Uri"]):
            output["S3Output"]["S3Uri"] = output["S3Output"]["S3Uri"].expr

    assert step_args["ProcessingOutputConfig"]["Outputs"] == [
        {
            "OutputName": "output-1",
            "AppManaged": False,
            "S3Output": {
                "S3Uri": {"Get": "Parameters.MySparkEventLogS3Uri"},
                "LocalPath": "/opt/ml/processing/spark-events/",
                "S3UploadMode": "Continuous",
            },
        }
    ]

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step],
        sagemaker_session=pipeline_session,
    )

    # test for idempotency
    step_def = json.loads(pipeline.definition())["Steps"][0]
    step_def2 = json.loads(pipeline.definition())["Steps"][0]
    assert step_def == step_def2


_PARAM_ROLE_NAME = "Role"


@pytest.mark.parametrize(
    "processor_args",
    [
        (
            ScriptProcessor(
                role=ParameterString(name=_PARAM_ROLE_NAME, default_value=ROLE),
                image_uri=IMAGE_URI,
                instance_count=1,
                instance_type="ml.m4.xlarge",
                command=["python3"],
            ),
            {"code": DUMMY_S3_SCRIPT_PATH},
        ),
        (
            Processor(
                role=ParameterString(name=_PARAM_ROLE_NAME, default_value=ROLE),
                image_uri=IMAGE_URI,
                instance_count=1,
                instance_type="ml.m4.xlarge",
            ),
            {},
        ),
    ],
)
@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_processor_with_role_as_pipeline_parameter(
    exists_mock, isfile_mock, processor_args, pipeline_session
):
    processor, run_inputs = processor_args
    processor.sagemaker_session = pipeline_session
    processor.run(**run_inputs)

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

    step_def = json.loads(pipeline.definition())["Steps"][0]
    assert step_def["Arguments"]["RoleArn"] == {"Get": f"Parameters.{_PARAM_ROLE_NAME}"}


@patch("sagemaker.workflow.utilities._pipeline_config", MOCKED_PIPELINE_CONFIG_WITH_CUSTOM_PREFIX)
def test_processing_step_with_processor_using_custom_job_prefixes(
    pipeline_session, processing_input, network_config
):
    custom_job_prefix = "ProcessingJobPrefix-2023-06-22"

    processor = Processor(
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        role=ROLE,
        base_job_name=custom_job_prefix,
    )
    processor.sagemaker_session = pipeline_session
    processor.role = ROLE
    processor.volume_kms_key = "volume-kms-key"
    processor.network_config = network_config

    processor_args = processor.run(inputs=processing_input)

    step = ProcessingStep(
        name="MyProcessingStep",
        step_args=processor_args,
    )

    pipeline = Pipeline(
        name="MyPipelineWithCustomPrefix",
        steps=[step],
        sagemaker_session=pipeline_session,
    )

    # Default the custom-prefixing feature is OFF for this pipeline
    # JobName not present in step_args
    step_args = get_step_args_helper(processor_args, "Processing", False)
    step_def = json.loads(pipeline.definition())["Steps"][0]

    assert "ProcessingJobName" not in step_def["Arguments"]
    assert step_def == {
        "Name": "MyProcessingStep",
        "Type": "Processing",
        "Arguments": step_args,
    }

    # Toggle on the custom-prefixing feature, and update the pipeline
    definition_config = PipelineDefinitionConfig(use_custom_job_prefix=True)
    pipeline.pipeline_definition_config = definition_config
    pipeline.upsert(role_arn=ROLE)

    # ProcessingJobPrefix-2023-06-20-20-42-13-030 trimmed to ProcessingJobPrefix-2023-06-22
    step_args2 = get_step_args_helper(processor_args, "Processing", True)
    step_def2 = json.loads(pipeline.definition())["Steps"][0]

    assert step_def2["Arguments"]["ProcessingJobName"] == custom_job_prefix
    assert step_def2 == {
        "Name": "MyProcessingStep",
        "Type": "Processing",
        "Arguments": step_args2,
    }
