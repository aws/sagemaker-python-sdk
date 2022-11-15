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
from mock import Mock, PropertyMock

import pytest
import warnings

from copy import deepcopy

from sagemaker.estimator import Estimator
from sagemaker.parameter import IntegerParameter
from sagemaker.transformer import Transformer
from sagemaker.tuner import HyperparameterTuner
from sagemaker.workflow.pipeline_context import PipelineSession

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
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.functions import Join
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

REGION = "us-west-2"
BUCKET = "my-bucket"
ROLE = "DummyRole"
IMAGE_URI = "fakeimage"
MODEL_NAME = "gisele"
DUMMY_S3_SCRIPT_PATH = "s3://dummy-s3/dummy_script.py"
INSTANCE_TYPE = "ml.m4.xlarge"

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
            instance_type="ml.p3.2xlarge",
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
def client():
    """Mock client.
    Considerations when appropriate:
        * utilize botocore.stub.Stubber
        * separate runtime client from client
    """
    client_mock = Mock()
    client_mock._client_config.user_agent = (
        "Boto3/1.14.24 Python/3.8.5 Linux/5.4.0-42-generic Botocore/1.17.24 Resource"
    )
    return client_mock


@pytest.fixture
def boto_session(client):
    role_mock = Mock()
    type(role_mock).arn = PropertyMock(return_value=ROLE)

    resource_mock = Mock()
    resource_mock.Role.return_value = role_mock

    session_mock = Mock(region_name=REGION)
    session_mock.resource.return_value = resource_mock
    session_mock.client.return_value = client

    return session_mock


@pytest.fixture
def pipeline_session(boto_session, client):
    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=client,
        default_bucket=BUCKET,
    )


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

    assert json.loads(pipeline.definition())["Steps"][0] == {
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


def test_processing_step_with_script_processor(pipeline_session, processing_input, network_config):
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
        "Arguments": get_step_args_helper(step_args, "Processing"),
    }


@pytest.mark.parametrize("framework_processor", FRAMEWORK_PROCESSOR)
@pytest.mark.parametrize("processing_input", PROCESSING_INPUT)
@pytest.mark.parametrize("processing_output", PROCESSING_OUTPUT)
def test_processing_step_with_framework_processor(
    framework_processor, pipeline_session, processing_input, processing_output, network_config
):

    processor, run_inputs = framework_processor
    processor.sagemaker_session = pipeline_session
    processor.role = ROLE

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
    )

    step_args = get_step_args_helper(step_args, "Processing")
    step_def = json.loads(pipeline.definition())["Steps"][0]

    assert step_args["ProcessingInputs"][0]["S3Input"]["S3Uri"] == processing_input.source
    assert (
        step_args["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
        == processing_output.destination
    )

    del step_args["ProcessingInputs"][0]["S3Input"]["S3Uri"]
    del step_def["Arguments"]["ProcessingInputs"][0]["S3Input"]["S3Uri"]

    del step_args["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
    del step_def["Arguments"]["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]

    assert step_def == {
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
        assert json.loads(pipeline.definition())["Steps"][0] == {
            "Name": "MyProcessingStep",
            "Type": "Processing",
            "Arguments": get_step_args_helper(step_args, "Processing"),
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

    run_inputs["inputs"] = processing_input

    step_args = processor.run(**run_inputs)
    step = ProcessingStep(
        name="MyProcessingStep",
        step_args=step_args,
    )

    step_args = get_step_args_helper(step_args, "Processing")

    assert step_args["AppSpecification"]["ContainerArguments"] == run_inputs["arguments"]

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
    pipeline.definition()
