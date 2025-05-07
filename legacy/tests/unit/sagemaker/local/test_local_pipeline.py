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
from mock import Mock, patch, PropertyMock

from botocore.exceptions import ClientError
from mock.mock import MagicMock

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.debugger import ProfilerConfig
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor
from sagemaker.remote_function.core.stored_function import RESULTS_FOLDER
from sagemaker.remote_function.job import (
    SCRIPT_AND_DEPENDENCIES_CHANNEL_NAME,
    RUNTIME_SCRIPTS_CHANNEL_NAME,
)
from sagemaker.s3_utils import s3_path_join
from sagemaker.transformer import Transformer
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import (
    ConditionEquals,
    ConditionGreaterThan,
    ConditionGreaterThanOrEqualTo,
    ConditionIn,
    ConditionLessThan,
    ConditionLessThanOrEqualTo,
    ConditionNot,
    ConditionOr,
)
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.function_step import step
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.step_outputs import get_step
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    TransformStep,
    TransformInput,
    StepTypeEnum,
)
from sagemaker.utils.local.pipeline import (
    _ConditionStepExecutor,
    _FailStepExecutor,
    _ProcessingStepExecutor,
    _StepExecutorFactory,
    _TrainingStepExecutor,
    _TransformStepExecutor,
    _CreateModelStepExecutor,
    LocalPipelineExecutor,
    StepExecutionException,
)
from sagemaker.utils.local.entities import _LocalExecutionStatus, _LocalPipelineExecution
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join, JsonGet, PropertyFile
from sagemaker.utils.local import LocalSession
from tests.unit.sagemaker.workflow.helpers import CustomStep
from tests.unit import DATA_DIR

STRING_PARAMETER = ParameterString("MyStr", "DefaultParameter")
INSTANCE_COUNT_PIPELINE_PARAMETER = ParameterInteger(name="InstanceCount", default_value=6)
INPUT_STEP = CustomStep(name="InputStep")
IMAGE_URI = "fakeimage"
ROLE = "DummyRole"
BUCKET = "my-bucket"
REGION = "us-west-2"
INSTANCE_TYPE = "ml.m4.xlarge"
PROPERTY_FILE_CONTENT = (
    "{"
    '  "my-processing-output": {'
    '    "nested_object1": {'
    '      "metric1": 45.22,'
    '      "metric2": 76'
    "    },"
    '    "nested_object2": {'
    '      "nested_list": ['
    "        {"
    '          "list_object1": {'
    '            "metric1": 55,'
    '            "metric2": 66.34'
    "          }"
    "        },"
    "        {"
    '          "list_object2": {'
    '            "metric1": 33'
    "           }"
    "         }"
    "      ]"
    "    }"
    "  }"
    "}"
)
S3_FILE_CONTENT = """
{
    "Result": [1, 2, 3], "Exception": null
}
"""
TEST_JOB_NAME = "test-job-name"


@pytest.fixture
def role_arn():
    return "arn:role"


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
    client_mock.describe_model.return_value = {"PrimaryContainer": {}, "Containers": {}}
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
    pipeline_session_mock = PipelineSession(
        boto_session=boto_session,
        sagemaker_client=client,
        default_bucket=BUCKET,
    )
    # For tests which doesn't verify config file injection, operate with empty config

    pipeline_session.sagemaker_config = {}
    return pipeline_session_mock


@pytest.fixture()
def local_sagemaker_session(boto_session):
    local_session_mock = LocalSession(boto_session=boto_session, default_bucket="my-bucket")
    # For tests which doesn't verify config file injection, operate with empty config

    local_session_mock.sagemaker_config = {}
    return local_session_mock


@pytest.fixture
def training_step(pipeline_session):
    estimator = Estimator(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_count=INSTANCE_COUNT_PIPELINE_PARAMETER,
        instance_type="c4.4xlarge",
        profiler_config=ProfilerConfig(system_monitor_interval_millis=500),
        hyperparameters={
            "batch-size": 500,
            "epochs": 5,
        },
        rules=[],
        sagemaker_session=pipeline_session,
        output_path="s3://a/b",
        use_spot_instances=False,
        # base_job_name would be popped out if no pipeline_definition_config configured
        base_job_name=TEST_JOB_NAME,
    )
    training_input = TrainingInput(s3_data=f"s3://{BUCKET}/train_manifest")
    step_args = estimator.fit(inputs=training_input)
    return TrainingStep(
        name="MyTrainingStep",
        description="TrainingStep description",
        display_name="MyTrainingStep",
        step_args=step_args,
    )


@pytest.fixture
def processing_step(pipeline_session):
    processor = Processor(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        sagemaker_session=pipeline_session,
        # base_job_name would be popped out if no pipeline_definition_config configured
        base_job_name=TEST_JOB_NAME,
    )
    processing_input = [
        ProcessingInput(
            source=f"s3://{BUCKET}/processing_manifest",
            destination="processing_manifest",
        )
    ]
    processing_output = [
        ProcessingOutput(
            output_name="output1",
            source="/opt/ml/processing/output/output1",
            destination="s3://some-bucket/some-path/output1",
            s3_upload_mode="EndOfJob",
        )
    ]
    step_args = processor.run(inputs=processing_input, outputs=processing_output)
    return ProcessingStep(
        name="MyProcessingStep",
        step_args=step_args,
        description="ProcessingStep description",
        display_name="MyProcessingStep",
    )


@pytest.fixture
def transform_step(pipeline_session):
    transformer = Transformer(
        model_name="my-model",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        output_path="s3://my-bucket/my-output-path",
        sagemaker_session=pipeline_session,
        # base_transform_job_name would be popped out if no pipeline_definition_config configured
        base_transform_job_name=TEST_JOB_NAME,
    )
    transform_inputs = TransformInput(data="s3://my-bucket/my-data")
    step_args = transformer.transform(
        data=transform_inputs.data,
        data_type=transform_inputs.data_type,
        content_type=transform_inputs.content_type,
        compression_type=transform_inputs.compression_type,
        split_type=transform_inputs.split_type,
        input_filter=transform_inputs.input_filter,
        output_filter=transform_inputs.output_filter,
        join_source=transform_inputs.join_source,
        model_client_config=transform_inputs.model_client_config,
    )
    return TransformStep(
        name="MyTransformStep",
        step_args=step_args,
    )


def test_evaluate_parameter(local_sagemaker_session):
    step = CustomStep(name="MyStep", input_data=STRING_PARAMETER)
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[STRING_PARAMETER],
        steps=[step],
        sagemaker_session=local_sagemaker_session,
    )

    execution = _LocalPipelineExecution("my-execution", pipeline, {"MyStr": "test_string"})
    evaluated_args = LocalPipelineExecutor(
        execution, local_sagemaker_session
    ).evaluate_step_arguments(step)
    assert evaluated_args["input_data"] == "test_string"


@pytest.mark.parametrize(
    "property_reference, expected",
    [
        (INPUT_STEP.properties.TrainingJobArn, "my-training-arn"),
        (INPUT_STEP.properties.ExperimentConfig.TrialName, "trial-bar"),
        (INPUT_STEP.properties.FinalMetricDataList[0].Value, 24),
        (INPUT_STEP.properties.FailureReason, "Error: bad input!"),
        (INPUT_STEP.properties.AlgorithmSpecification.AlgorithmName, "fooAlgorithm"),
        (INPUT_STEP.properties.AlgorithmSpecification.MetricDefinitions[0].Name, "mse"),
        (INPUT_STEP.properties.Environment["max-depth"], "10"),
    ],
)
def test_evaluate_property_reference(local_sagemaker_session, property_reference, expected):
    step = CustomStep(name="MyStep", input_data=property_reference)
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[STRING_PARAMETER],
        steps=[INPUT_STEP, step],
        sagemaker_session=local_sagemaker_session,
    )

    execution = _LocalPipelineExecution("my-execution", pipeline)
    execution.step_execution[INPUT_STEP.name].properties = {
        "AlgorithmSpecification": {
            "AlgorithmName": "fooAlgorithm",
            "MetricDefinitions": [{"Name": "mse", "Regex": ".*MeanSquaredError.*"}],
        },
        "TrainingJobArn": "my-training-arn",
        "FinalMetricDataList": [{"MetricName": "mse", "Timestamp": 1656281030, "Value": 24}],
        "ExperimentConfig": {
            "ExperimentName": "my-exp",
            "TrialComponentDisplayName": "trial-component-foo",
            "TrialName": "trial-bar",
        },
        "Environment": {"max-depth": "10"},
        "FailureReason": "Error: bad input!",
    }
    evaluated_args = LocalPipelineExecutor(
        execution, local_sagemaker_session
    ).evaluate_step_arguments(step)
    assert evaluated_args["input_data"] == expected


def test_evaluate_property_reference_undefined(local_sagemaker_session):
    step = CustomStep(name="MyStep", input_data=INPUT_STEP.properties.FailureReason)
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[STRING_PARAMETER],
        steps=[INPUT_STEP, step],
        sagemaker_session=local_sagemaker_session,
    )

    execution = _LocalPipelineExecution("my-execution", pipeline)
    execution.step_execution[INPUT_STEP.name].properties = {"TrainingJobArn": "my-training-arn"}
    with pytest.raises(StepExecutionException) as e:
        LocalPipelineExecutor(execution, local_sagemaker_session).evaluate_step_arguments(step)
    assert f"{INPUT_STEP.properties.FailureReason.expr} is undefined." in str(e.value)


@pytest.mark.parametrize(
    "join_value, expected",
    [
        (ExecutionVariables.PIPELINE_NAME, "blah-MyPipeline-blah"),
        (STRING_PARAMETER, "blah-DefaultParameter-blah"),
        (INPUT_STEP.properties.TrainingJobArn, "blah-my-training-arn-blah"),
        (
            Join(on=".", values=["test1", "test2", "test3"]),
            "blah-test1.test2.test3-blah",
        ),
        (
            Join(on=".", values=["test", ExecutionVariables.PIPELINE_NAME, "test"]),
            "blah-test.MyPipeline.test-blah",
        ),
    ],
)
def test_evaluate_join_function(local_sagemaker_session, join_value, expected):
    step = CustomStep(name="TestStep", input_data=Join(on="-", values=["blah", join_value, "blah"]))
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[STRING_PARAMETER],
        steps=[INPUT_STEP, step],
        sagemaker_session=local_sagemaker_session,
    )

    execution = _LocalPipelineExecution("my-execution", pipeline)
    execution.step_execution["InputStep"].properties = {"TrainingJobArn": "my-training-arn"}
    evaluated_args = LocalPipelineExecutor(
        execution, local_sagemaker_session
    ).evaluate_step_arguments(step)
    assert evaluated_args["input_data"] == expected


@pytest.mark.parametrize(
    "json_path_value, expected",
    [
        ("my-processing-output.nested_object1.metric1", 45.22),
        ("my-processing-output.nested_object1['metric2']", 76),
        ("my-processing-output.nested_object2.nested_list[0].list_object1.metric1", 55),
        ("my-processing-output.nested_object2.nested_list[0].list_object1['metric2']", 66.34),
        ("my-processing-output.nested_object2.nested_list[1].list_object2.metric1", 33),
    ],
)
@patch("sagemaker.session.Session.read_s3_file", return_value=PROPERTY_FILE_CONTENT)
def test_evaluate_json_get_function(
    read_s3_file, local_sagemaker_session, json_path_value, expected
):
    property_file = PropertyFile(
        name="my-property-file", output_name="TestOutputName", path="processing_output.json"
    )
    processor = Processor(
        image_uri="some_image_uri",
        role="DummyRole",
        instance_count=1,
        instance_type="c4.4xlarge",
        sagemaker_session=local_sagemaker_session,
    )
    processing_step = ProcessingStep(
        name="inputProcessingStep",
        processor=processor,
        outputs=[ProcessingOutput(output_name="TestOutputName")],
        property_files=[property_file],
    )

    step = CustomStep(
        name="TestStep",
        input_data=JsonGet(
            step_name=processing_step.name, property_file=property_file, json_path=json_path_value
        ),
    )
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[STRING_PARAMETER],
        steps=[processing_step, step],
        sagemaker_session=local_sagemaker_session,
    )

    execution = _LocalPipelineExecution("my-execution", pipeline)
    execution.step_execution["inputProcessingStep"].properties = {
        "ProcessingOutputConfig": {
            "Outputs": {
                "TestOutputName": {
                    "OutputName": "TestOutputName",
                    "S3Output": {"S3Uri": "s3://my-bucket/processing/output"},
                }
            }
        }
    }
    evaluated_args = LocalPipelineExecutor(
        execution, local_sagemaker_session
    ).evaluate_step_arguments(step)
    assert evaluated_args["input_data"] == expected


def test_evaluate_json_get_function_processing_output_not_available(local_sagemaker_session):
    property_file = PropertyFile(
        name="my-property-file", output_name="TestOutputName", path="processing_output.json"
    )
    processor = Processor(
        image_uri="some_image_uri",
        role="DummyRole",
        instance_count=1,
        instance_type="c4.4xlarge",
        sagemaker_session=local_sagemaker_session,
    )
    processing_step = ProcessingStep(
        name="inputProcessingStep",
        processor=processor,
        outputs=[ProcessingOutput(output_name="TestOutputName")],
        property_files=[property_file],
    )
    step = CustomStep(
        name="TestStep",
        input_data=JsonGet(
            step_name=processing_step.name, property_file=property_file, json_path="mse"
        ),
    )
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[STRING_PARAMETER],
        steps=[processing_step, step],
        sagemaker_session=local_sagemaker_session,
    )
    execution = _LocalPipelineExecution("my-execution", pipeline)
    with pytest.raises(StepExecutionException) as e:
        LocalPipelineExecutor(execution, local_sagemaker_session).evaluate_step_arguments(step)
    assert f"Step '{processing_step.name}' does not yet contain processing outputs." in str(e.value)


@patch(
    "sagemaker.session.Session.read_s3_file",
    side_effect=ClientError({"Code": "NoSuchKey", "Message": "bad key"}, "GetObject"),
)
def test_evaluate_json_get_function_s3_client_error(read_s3_file, local_sagemaker_session):
    property_file = PropertyFile(
        name="my-property-file", output_name="TestOutputName", path="processing_output.json"
    )
    processor = Processor(
        image_uri="some_image_uri",
        role="DummyRole",
        instance_count=1,
        instance_type="c4.4xlarge",
        sagemaker_session=local_sagemaker_session,
    )
    processing_step = ProcessingStep(
        name="inputProcessingStep",
        processor=processor,
        outputs=[ProcessingOutput(output_name="TestOutputName")],
        property_files=[property_file],
    )
    step = CustomStep(
        name="TestStep",
        input_data=JsonGet(
            step_name=processing_step.name, property_file=property_file, json_path="mse"
        ),
    )
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[STRING_PARAMETER],
        steps=[processing_step, step],
        sagemaker_session=local_sagemaker_session,
    )
    execution = _LocalPipelineExecution("my-execution", pipeline)
    processing_output_path = "s3://my-bucket/processing/output"
    execution.step_execution["inputProcessingStep"].properties = {
        "ProcessingOutputConfig": {
            "Outputs": {
                "TestOutputName": {
                    "OutputName": "TestOutputName",
                    "S3Output": {"S3Uri": processing_output_path},
                }
            }
        }
    }
    with pytest.raises(StepExecutionException) as e:
        LocalPipelineExecutor(execution, local_sagemaker_session).evaluate_step_arguments(step)
    assert (
        f"Received an error while reading file {processing_output_path}/{property_file.path} from S3"
        in str(e.value)
    )


@patch("sagemaker.session.Session.read_s3_file", return_value="['invalid_json']")
def test_evaluate_json_get_function_bad_json_in_property_file(
    read_s3_file, local_sagemaker_session
):
    property_file = PropertyFile(
        name="my-property-file", output_name="TestOutputName", path="processing_output.json"
    )
    processor = Processor(
        image_uri="some_image_uri",
        role="DummyRole",
        instance_count=1,
        instance_type="c4.4xlarge",
        sagemaker_session=local_sagemaker_session,
    )
    processing_step = ProcessingStep(
        name="inputProcessingStep",
        processor=processor,
        outputs=[ProcessingOutput(output_name="TestOutputName")],
        property_files=[property_file],
    )
    step = CustomStep(
        name="TestStep",
        input_data=JsonGet(
            step_name=processing_step.name, property_file=property_file, json_path="mse"
        ),
    )
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[STRING_PARAMETER],
        steps=[processing_step, step],
        sagemaker_session=local_sagemaker_session,
    )

    execution = _LocalPipelineExecution("my-execution", pipeline)
    processing_output_path = "s3://my-bucket/processing/output"
    execution.step_execution["inputProcessingStep"].properties = {
        "ProcessingOutputConfig": {
            "Outputs": {
                "TestOutputName": {
                    "OutputName": "TestOutputName",
                    "S3Output": {"S3Uri": processing_output_path},
                }
            }
        }
    }
    with pytest.raises(StepExecutionException) as e:
        LocalPipelineExecutor(execution, local_sagemaker_session).evaluate_step_arguments(step)
    assert (
        f"Contents of file {processing_output_path}/{property_file.path} are not in valid JSON format."
    ) in str(e.value)


@patch("sagemaker.session.Session.read_s3_file", return_value=PROPERTY_FILE_CONTENT)
def test_evaluate_json_get_function_invalid_json_path(read_s3_file, local_sagemaker_session):
    property_file = PropertyFile(
        name="my-property-file", output_name="TestOutputName", path="processing_output.json"
    )
    processor = Processor(
        image_uri="some_image_uri",
        role="DummyRole",
        instance_count=1,
        instance_type="c4.4xlarge",
        sagemaker_session=local_sagemaker_session,
    )
    processing_step = ProcessingStep(
        name="inputProcessingStep",
        processor=processor,
        outputs=[ProcessingOutput(output_name="TestOutputName")],
        property_files=[property_file],
    )
    step = CustomStep(
        name="TestStep",
        input_data=JsonGet(
            step_name=processing_step.name,
            property_file=property_file,
            json_path="some.json.path[1].does.not['exist']",
        ),
    )
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[STRING_PARAMETER],
        steps=[processing_step, step],
        sagemaker_session=local_sagemaker_session,
    )
    execution = _LocalPipelineExecution("my-execution", pipeline)
    execution.step_execution["inputProcessingStep"].properties = {
        "ProcessingOutputConfig": {
            "Outputs": {
                "TestOutputName": {
                    "OutputName": "TestOutputName",
                    "S3Output": {"S3Uri": "s3://my-bucket/processing/output"},
                }
            }
        }
    }
    with pytest.raises(StepExecutionException) as e:
        LocalPipelineExecutor(execution, local_sagemaker_session).evaluate_step_arguments(step)
    assert "Invalid json path 'some.json.path[1].does.not['exist']'" in str(e.value)


@pytest.mark.parametrize(
    "step, step_executor_class",
    [
        (Mock(step_type=StepTypeEnum.TRAINING), _TrainingStepExecutor),
        (Mock(step_type=StepTypeEnum.PROCESSING), _ProcessingStepExecutor),
        (Mock(step_type=StepTypeEnum.TRANSFORM), _TransformStepExecutor),
        (Mock(step_type=StepTypeEnum.CONDITION), _ConditionStepExecutor),
        (Mock(step_type=StepTypeEnum.FAIL), _FailStepExecutor),
        (Mock(step_type=StepTypeEnum.CREATE_MODEL), _CreateModelStepExecutor),
    ],
)
def test_step_executor_factory(step, step_executor_class):
    local_pipeline_executor = Mock()
    step_executor_factory = _StepExecutorFactory(local_pipeline_executor)
    step_executor = step_executor_factory.get(step)
    assert isinstance(step_executor, step_executor_class)


@patch(
    "sagemaker.local.image._SageMakerContainer.train",
    return_value="/some/path/to/model",
)
def test_execute_pipeline_training_step(train, local_sagemaker_session, training_step):
    pipeline = Pipeline(
        name="MyPipeline1",
        parameters=[INSTANCE_COUNT_PIPELINE_PARAMETER],
        steps=[training_step],
        sagemaker_session=local_sagemaker_session,
    )
    execution = LocalPipelineExecutor(
        _LocalPipelineExecution("my-execution-1", pipeline), local_sagemaker_session
    ).execute()
    assert execution.status == _LocalExecutionStatus.SUCCEEDED.value
    assert execution.pipeline_execution_name == "my-execution-1"

    step_execution = execution.step_execution
    expected_must_have = {
        "ResourceConfig": {"InstanceCount": 6},
        "TrainingJobStatus": "Completed",
        "ModelArtifacts": {"S3ModelArtifacts": "/some/path/to/model"},
    }
    assert step_execution["MyTrainingStep"].status == "Succeeded"
    assert expected_must_have.items() <= step_execution["MyTrainingStep"].properties.items()

    with pytest.raises(ValueError) as e:
        execution.result(step_name=training_step.name)

    assert "This method can only be used on pipeline steps created using @step decorator." in str(e)


@patch(
    "sagemaker.local.image._SageMakerContainer.train",
    return_value="/some/path/to/model",
)
@patch("sagemaker.workflow.pipeline.deserialize_obj_from_s3")
def test_execute_pipeline_step_decorator(mock_deserialize, mock_train, local_sagemaker_session):
    dependencies_path = os.path.join(DATA_DIR, "workflow", "requirements.txt")
    step_settings = dict(
        role="SageMakerRole",
        instance_type="local",
        image_uri=IMAGE_URI,
        keep_alive_period_in_seconds=60,
        dependencies=dependencies_path,
    )

    @step(**step_settings)
    def generator():
        return 3, 4

    @step(**step_settings)
    def sum(a, b):
        """adds two numbers"""
        return a + b

    step_output_a = generator()
    step_output_b = sum(step_output_a[0], step_output_a[1])

    pipeline_name = "MyPipeline1"
    execution_id = "my-execution-1"

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_output_b],
        sagemaker_session=local_sagemaker_session,
    )
    execution = LocalPipelineExecutor(
        _LocalPipelineExecution(execution_id, pipeline, local_session=local_sagemaker_session),
        local_sagemaker_session,
    ).execute()

    input_data_configs = mock_train.call_args[0][0]
    assert len(input_data_configs) == 2
    for input_data_config in input_data_configs:
        assert input_data_config["ChannelName"] in {
            RUNTIME_SCRIPTS_CHANNEL_NAME,
            SCRIPT_AND_DEPENDENCIES_CHANNEL_NAME,
        }

    assert execution.status == _LocalExecutionStatus.SUCCEEDED.value
    assert execution.pipeline_execution_name == "my-execution-1"

    step_execution = execution.step_execution
    expected_must_have = {
        "ResourceConfig": {"InstanceCount": 1},
        "TrainingJobStatus": "Completed",
        "ModelArtifacts": {"S3ModelArtifacts": "/some/path/to/model"},
    }
    assert step_execution[get_step(step_output_a).name].status == "Succeeded"
    assert step_execution[get_step(step_output_b).name].status == "Succeeded"
    assert (
        expected_must_have.items()
        <= step_execution[get_step(step_output_a).name].properties.items()
    )
    assert (
        expected_must_have.items()
        <= step_execution[get_step(step_output_b).name].properties.items()
    )

    execution.result(step_name=get_step(step_output_a).name)

    assert mock_deserialize.call_args[1]["sagemaker_session"] == local_sagemaker_session
    assert mock_deserialize.call_args[1]["s3_uri"] == s3_path_join(
        "s3://",
        local_sagemaker_session.default_bucket(),
        pipeline_name,
        execution_id,
        get_step(step_output_a).name,
        RESULTS_FOLDER,
    )


@patch(
    "sagemaker.local.image._SageMakerContainer.train",
    MagicMock(side_effect=Exception()),
)
def test_execute_pipeline_step_decorator_failure_case(local_sagemaker_session):
    step_settings = dict(
        role="SageMakerRole",
        instance_type="local",
        image_uri=IMAGE_URI,
        keep_alive_period_in_seconds=60,
    )

    @step(**step_settings)
    def generator():
        return 3, 4

    step_output_a = generator()

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step_output_a],
        sagemaker_session=local_sagemaker_session,
    )
    execution = LocalPipelineExecutor(
        _LocalPipelineExecution("my-execution", pipeline, local_session=local_sagemaker_session),
        local_sagemaker_session,
    ).execute()

    assert execution.status == _LocalExecutionStatus.FAILED.value

    step_execution = execution.step_execution
    assert step_execution[get_step(step_output_a).name].status == "Failed"

    step_name = get_step(step_output_a).name
    with pytest.raises(RuntimeError) as e:
        execution.result(step_name=step_name)

    assert f"step {step_name} is not in Completed status." in str(e)


@patch(
    "sagemaker.local.image._SageMakerContainer.train",
    return_value="/some/path/to/model",
)
@patch("sagemaker.session.Session.read_s3_file", return_value=S3_FILE_CONTENT)
def test_execute_pipeline_step_decorator_with_condition_step(
    mock_s3_rd, mock_train, local_sagemaker_session
):
    step_settings = dict(
        role="SageMakerRole",
        instance_type="local",
        keep_alive_period_in_seconds=60,
        image_uri=IMAGE_URI,
    )

    @step(**step_settings)
    def left_condition_step():
        # The return values should match the contents in S3_FILE_CONTENT
        return 1, 2, 3

    @step(**step_settings)
    def if_step():
        return "In if branch"

    @step(**step_settings)
    def else_step():
        return "In else branch"

    @step(**step_settings)
    def depends_step():
        return "Condition depending step"

    step_output = left_condition_step()
    depends_step_output = depends_step()
    if_step_output = if_step()
    else_step_output = else_step()

    cond_gt = ConditionGreaterThan(left=step_output[1], right=1)
    cond_step = ConditionStep(
        name="MyConditionStep",
        conditions=[cond_gt],
        if_steps=[if_step_output],
        else_steps=[else_step_output],
        depends_on=[depends_step_output],
    )

    pipeline = Pipeline(
        name="local_pipeline_step_decorator",
        steps=[cond_step],
        sagemaker_session=local_sagemaker_session,
    )

    execution = LocalPipelineExecutor(
        _LocalPipelineExecution("my-execution-1", pipeline), local_sagemaker_session
    ).execute()

    assert execution.status == _LocalExecutionStatus.SUCCEEDED.value
    assert execution.pipeline_execution_name == "my-execution-1"

    step_execution = execution.step_execution

    expected_must_have = {
        "ResourceConfig": {"InstanceCount": 1},
        "TrainingJobStatus": "Completed",
        "ModelArtifacts": {"S3ModelArtifacts": "/some/path/to/model"},
    }
    assert not step_execution[get_step(else_step_output).name].status
    assert step_execution[get_step(step_output).name].status == "Succeeded"
    assert step_execution[get_step(depends_step_output).name].status == "Succeeded"
    assert step_execution[get_step(if_step_output).name].status == "Succeeded"
    assert step_execution[cond_step.name].status == "Succeeded"
    assert step_execution[cond_step.name].properties["Outcome"]
    assert (
        expected_must_have.items() <= step_execution[get_step(step_output).name].properties.items()
    )
    assert (
        expected_must_have.items()
        <= step_execution[get_step(depends_step_output).name].properties.items()
    )
    assert (
        expected_must_have.items()
        <= step_execution[get_step(if_step_output).name].properties.items()
    )


@patch("sagemaker.local.image._SageMakerContainer.process", MagicMock())
def test_execute_pipeline_processing_step(local_sagemaker_session, processing_step):
    pipeline = Pipeline(
        name="MyPipeline2",
        steps=[processing_step],
        sagemaker_session=local_sagemaker_session,
    )
    execution = LocalPipelineExecutor(
        _LocalPipelineExecution("my-execution-2", pipeline), local_sagemaker_session
    ).execute()
    assert execution.status == _LocalExecutionStatus.SUCCEEDED.value
    assert execution.pipeline_execution_name == "my-execution-2"

    step_execution = execution.step_execution
    step_properties = step_execution["MyProcessingStep"].properties
    assert step_execution["MyProcessingStep"].status == "Succeeded"
    assert "MyProcessingStep-" in step_properties["ProcessingJobArn"]
    assert "MyProcessingStep-" in step_properties["ProcessingJobName"]
    assert step_properties["AppSpecification"]["ImageUri"] == IMAGE_URI
    s3_input = step_properties["ProcessingInputs"]["input-1"][
        "S3Input"
    ]  # input name "input-1" is auto-generated
    assert s3_input["S3Uri"] == f"s3://{BUCKET}/processing_manifest"
    assert s3_input["LocalPath"] == "processing_manifest"
    cluster_config = step_properties["ProcessingResources"]["ClusterConfig"]
    assert cluster_config["InstanceCount"] == 1
    assert cluster_config["InstanceType"] == INSTANCE_TYPE
    assert step_properties["ProcessingJobStatus"] == "Completed"
    expected_processing_output = {
        "OutputName": "output1",
        "AppManaged": False,
        "S3Output": {
            "S3Uri": "s3://some-bucket/some-path/output1",
            "LocalPath": "/opt/ml/processing/output/output1",
            "S3UploadMode": "EndOfJob",
        },
    }
    processing_output = step_properties["ProcessingOutputConfig"]["Outputs"]["output1"]
    assert processing_output == expected_processing_output

    with pytest.raises(ValueError) as e:
        execution.result(step_name=processing_step.name)

    assert "This method can only be used on pipeline steps created using @step decorator." in str(e)


@patch("sagemaker.local.local_session._LocalTransformJob")
def test_execute_pipeline_transform_step(
    _LocalTransformJob, local_sagemaker_session, transform_step
):
    pipeline = Pipeline(
        name="MyPipeline3",
        steps=[transform_step],
        sagemaker_session=local_sagemaker_session,
    )
    execution = LocalPipelineExecutor(
        _LocalPipelineExecution("my-execution-3", pipeline), local_sagemaker_session
    ).execute()

    _LocalTransformJob().start.assert_called_with(
        {
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://my-bucket/my-data",
                }
            }
        },
        {"S3OutputPath": "s3://my-bucket/my-output-path"},
        {"InstanceCount": 1, "InstanceType": "ml.m5.xlarge"},
    )

    _LocalTransformJob().describe.assert_called()

    assert execution.status == _LocalExecutionStatus.SUCCEEDED.value
    assert execution.pipeline_execution_name == "my-execution-3"

    step_execution = execution.step_execution
    assert step_execution["MyTransformStep"].status == _LocalExecutionStatus.SUCCEEDED.value


def test_execute_pipeline_fail_step(local_sagemaker_session):
    param = ParameterString(name="foo", default_value="bar")
    step_fail = FailStep(
        name="FailStep",
        error_message=Join(on=": ", values=["Failed due to foo has value", param]),
    )
    pipeline = Pipeline(
        name="MyPipeline4",
        steps=[step_fail],
        parameters=[param],
        sagemaker_session=local_sagemaker_session,
    )

    execution = LocalPipelineExecutor(
        _LocalPipelineExecution("my-execution-4", pipeline), local_sagemaker_session
    ).execute()

    assert execution.status == _LocalExecutionStatus.FAILED.value
    assert execution.pipeline_execution_name == "my-execution-4"

    fail_step_execution = execution.step_execution.get(step_fail.name)
    assert fail_step_execution.status == _LocalExecutionStatus.FAILED.value
    assert fail_step_execution.properties == {"ErrorMessage": "Failed due to foo has value: bar"}
    assert fail_step_execution.failure_reason == "Failed due to foo has value: bar"


@pytest.mark.parametrize(
    "condition, condition_outcome, succeeded_steps, executing_steps",
    [
        (
            ConditionEquals(left=INSTANCE_COUNT_PIPELINE_PARAMETER, right=1),
            False,
            ["MyProcessingStep"],
            ["MyTrainingStep"],
        ),
        (
            ConditionGreaterThan(left=INSTANCE_COUNT_PIPELINE_PARAMETER, right=1),
            True,
            ["MyTrainingStep"],
            ["MyProcessingStep"],
        ),
        (
            ConditionGreaterThanOrEqualTo(left=INSTANCE_COUNT_PIPELINE_PARAMETER, right=6),
            True,
            ["MyTrainingStep"],
            ["MyProcessingStep"],
        ),
        (
            ConditionLessThan(left=INSTANCE_COUNT_PIPELINE_PARAMETER, right=1),
            False,
            ["MyProcessingStep"],
            ["MyTrainingStep"],
        ),
        (
            ConditionLessThanOrEqualTo(left=INSTANCE_COUNT_PIPELINE_PARAMETER, right=6),
            True,
            ["MyTrainingStep"],
            ["MyProcessingStep"],
        ),
        (
            ConditionIn(value=INSTANCE_COUNT_PIPELINE_PARAMETER, in_values=[3, 6, 9]),
            True,
            ["MyTrainingStep"],
            ["MyProcessingStep"],
        ),
        (
            ConditionNot(ConditionEquals(left=INSTANCE_COUNT_PIPELINE_PARAMETER, right=1)),
            True,
            ["MyTrainingStep"],
            ["MyProcessingStep"],
        ),
        (
            ConditionOr(
                conditions=[
                    ConditionEquals(left=INSTANCE_COUNT_PIPELINE_PARAMETER, right=3),
                    ConditionEquals(left=INSTANCE_COUNT_PIPELINE_PARAMETER, right=6),
                    ConditionEquals(left=INSTANCE_COUNT_PIPELINE_PARAMETER, right=9),
                ]
            ),
            True,
            ["MyTrainingStep"],
            ["MyProcessingStep"],
        ),
        (
            ConditionOr(
                conditions=[
                    ConditionEquals(left=INSTANCE_COUNT_PIPELINE_PARAMETER, right=3),
                    ConditionEquals(left=INSTANCE_COUNT_PIPELINE_PARAMETER, right=7),
                    ConditionEquals(left=INSTANCE_COUNT_PIPELINE_PARAMETER, right=9),
                ]
            ),
            False,
            ["MyProcessingStep"],
            ["MyTrainingStep"],
        ),
    ],
)
@patch(
    "sagemaker.local.image._SageMakerContainer.train",
    return_value="/some/path/to/model",
)
@patch("sagemaker.local.image._SageMakerContainer.process")
def test_execute_pipeline_condition_step_test_conditions(
    process,
    train,
    local_sagemaker_session,
    training_step,
    processing_step,
    condition,
    condition_outcome,
    succeeded_steps,
    executing_steps,
):
    condition_step = ConditionStep(
        name="MyCondStep",
        conditions=[condition],
        if_steps=[training_step],
        else_steps=[processing_step],
    )
    pipeline = Pipeline(
        name="MyPipeline5",
        steps=[condition_step],
        parameters=[INSTANCE_COUNT_PIPELINE_PARAMETER],
        sagemaker_session=local_sagemaker_session,
    )

    execution = LocalPipelineExecutor(
        _LocalPipelineExecution("my-execution-5", pipeline), local_sagemaker_session
    ).execute()

    assert execution.status == _LocalExecutionStatus.SUCCEEDED.value
    assert (
        execution.step_execution.get("MyCondStep").status == _LocalExecutionStatus.SUCCEEDED.value
    )
    assert execution.step_execution.get("MyCondStep").properties == {"Outcome": condition_outcome}

    for succeeded_step in succeeded_steps:
        assert (
            execution.step_execution.get(succeeded_step).status
            == _LocalExecutionStatus.SUCCEEDED.value
        )
        assert execution.step_execution.get(succeeded_step).name == succeeded_step
        assert execution.step_execution.get(succeeded_step).properties != {}
        assert execution.step_execution.get(succeeded_step).failure_reason is None

    for executing_step in executing_steps:
        assert execution.step_execution.get(executing_step).name == executing_step
        assert execution.step_execution.get(executing_step).properties == {}
        assert execution.step_execution.get(executing_step).failure_reason is None


#         ┌──►F
#         │
# A──►B──►C──►E──►G──►H
#     │           ▲
#     └──►D──►I───┘
@pytest.mark.parametrize(
    "left_value_1, left_value_2, expected_path",
    [
        (2, 2, ["stepA", "stepB", "stepC", "stepE"]),
        (2, 1, ["stepA", "stepB", "stepC", "stepF"]),
        (1, 2, ["stepA", "stepB", "stepD", "stepI"]),
        (1, 1, ["stepA", "stepB", "stepD", "stepI"]),
    ],
)
@patch(
    "sagemaker.local.local_session.LocalSagemakerClient.describe_training_job",
    return_value={},
)
@patch("sagemaker.local.local_session.LocalSagemakerClient.create_training_job")
def test_pipeline_execution_condition_step_execution_path(
    create_training_job,
    describe_training_job,
    local_sagemaker_session,
    left_value_1,
    left_value_2,
    expected_path,
):
    condition_1 = ConditionEquals(left=left_value_1, right=2)
    condition_2 = ConditionEquals(left=left_value_2, right=2)
    step_a = CustomStep(name="stepA")
    step_e = CustomStep(name="stepE")
    step_f = CustomStep(name="stepF")
    step_d = CustomStep(name="stepD")
    step_i = CustomStep(name="stepI", depends_on=[step_d.name])
    step_c = ConditionStep(
        name="stepC",
        conditions=[condition_2],
        if_steps=[step_e],
        else_steps=[step_f],
    )
    step_b = ConditionStep(
        name="stepB",
        depends_on=[step_a.name],
        conditions=[condition_1],
        if_steps=[step_c],
        else_steps=[step_d],
    )
    step_g = CustomStep(name="stepG", depends_on=[step_e.name, step_i.name])
    step_h = CustomStep(name="stepH", depends_on=[step_g.name])

    pipeline = Pipeline(
        name="MyPipeline5-1",
        parameters=[INSTANCE_COUNT_PIPELINE_PARAMETER],
        steps=[step_a, step_b, step_g, step_h, step_i],
        sagemaker_session=local_sagemaker_session,
    )

    execution = LocalPipelineExecutor(
        _LocalPipelineExecution("my-execution-5-1", pipeline), local_sagemaker_session
    ).execute()

    actual_path = []
    for step_name, step_execution in execution.step_execution.items():
        if step_execution.status is not None:
            actual_path.append(step_name)
    assert actual_path == expected_path


def test_condition_step_incompatible_types(local_sagemaker_session):

    step_a = CustomStep(name="stepA")
    step_b = CustomStep(name="stepB")
    step_cond = ConditionStep(
        name="stepCondition",
        conditions=[ConditionEquals(left=INSTANCE_COUNT_PIPELINE_PARAMETER, right="some_string")],
        if_steps=[step_a],
        else_steps=[step_b],
    )

    pipeline = Pipeline(
        name="MyPipeline5-2",
        parameters=[INSTANCE_COUNT_PIPELINE_PARAMETER],
        steps=[step_cond],
        sagemaker_session=local_sagemaker_session,
    )

    execution = LocalPipelineExecutor(
        _LocalPipelineExecution("my-execution-5-1", pipeline), local_sagemaker_session
    ).execute()

    assert execution.status == _LocalExecutionStatus.FAILED.value
    assert (
        "LeftValue [6] of type [<class 'int'>] and RightValue [some_string] of "
        + "type [<class 'str'>] are not of the same type."
        in execution.failure_reason
    )
    assert execution.step_execution["stepCondition"].status == _LocalExecutionStatus.FAILED.value


@patch("sagemaker.local.local_session._LocalTrainingJob")
@patch("sagemaker.local.image._SageMakerContainer.process")
def test_processing_and_training_steps_with_data_dependency(
    process,
    _LocalTrainingJob,
    pipeline_session,
    local_sagemaker_session,
    processing_step,
):

    estimator = Estimator(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_count=INSTANCE_COUNT_PIPELINE_PARAMETER,
        instance_type="c4.4xlarge",
        profiler_config=ProfilerConfig(system_monitor_interval_millis=500),
        hyperparameters={
            "batch-size": 500,
            "epochs": 5,
        },
        rules=[],
        sagemaker_session=pipeline_session,
        output_path="s3://a/b",
        use_spot_instances=False,
    )
    training_input = TrainingInput(
        s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["output1"].S3Output.S3Uri
    )
    step_args = estimator.fit(inputs=training_input)
    training_step = TrainingStep(
        name="MyTrainingStep",
        description="TrainingStep description",
        display_name="MyTrainingStep",
        step_args=step_args,
    )

    pipeline = Pipeline(
        name="MyPipeline6",
        parameters=[INSTANCE_COUNT_PIPELINE_PARAMETER],
        steps=[processing_step, training_step],
        sagemaker_session=local_sagemaker_session,
    )

    execution = LocalPipelineExecutor(
        _LocalPipelineExecution("my-execution-6", pipeline), local_sagemaker_session
    ).execute()

    args_called_with = _LocalTrainingJob().start.call_args.args

    # input_data_config
    assert args_called_with[0] == [
        {
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://some-bucket/some-path/output1",  # from depended processing step
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
            "ChannelName": "training",
        }
    ]

    # output_data_config
    assert args_called_with[1] == {"S3OutputPath": "s3://a/b"}

    # hyperparameters
    assert args_called_with[2] == {"batch-size": "500", "epochs": "5"}

    # environment
    assert args_called_with[3] == {}

    # job_name
    assert args_called_with[4].startswith("MyTrainingStep-")

    assert (
        execution.step_execution.get("MyProcessingStep").status
        == _LocalExecutionStatus.SUCCEEDED.value
    )
    assert (
        execution.step_execution.get("MyTrainingStep").status
        == _LocalExecutionStatus.SUCCEEDED.value
    )
    assert execution.status == _LocalExecutionStatus.SUCCEEDED.value


@patch(
    "sagemaker.local.local_session.LocalSagemakerClient.create_training_job",
    side_effect=RuntimeError("Dummy RuntimeError"),
)
def test_execute_pipeline_step_create_training_job_fail(
    create_training_job, local_sagemaker_session, pipeline_session, training_step
):
    pipeline = Pipeline(
        name="MyPipelineX-" + training_step.name,
        steps=[training_step],
        parameters=[INSTANCE_COUNT_PIPELINE_PARAMETER],
        sagemaker_session=local_sagemaker_session,
    )
    execution = LocalPipelineExecutor(
        _LocalPipelineExecution("my-execution-x-" + training_step.name, pipeline),
        local_sagemaker_session,
    ).execute()

    assert execution.status == _LocalExecutionStatus.FAILED.value
    assert execution.pipeline_execution_name == "my-execution-x-" + training_step.name

    step_execution = execution.step_execution
    assert step_execution[training_step.name].status == _LocalExecutionStatus.FAILED.value
    assert "Dummy RuntimeError" in step_execution[training_step.name].failure_reason


@patch(
    "sagemaker.local.local_session.LocalSagemakerClient.create_processing_job",
    side_effect=RuntimeError("Dummy RuntimeError"),
)
def test_execute_pipeline_step_create_processing_job_fail(
    create_processing_job, local_sagemaker_session, pipeline_session, processing_step
):
    pipeline = Pipeline(
        name="MyPipelineX-" + processing_step.name,
        steps=[processing_step],
        sagemaker_session=local_sagemaker_session,
    )
    execution = LocalPipelineExecutor(
        _LocalPipelineExecution("my-execution-x-" + processing_step.name, pipeline),
        local_sagemaker_session,
    ).execute()

    assert execution.status == _LocalExecutionStatus.FAILED.value
    assert execution.pipeline_execution_name == "my-execution-x-" + processing_step.name

    step_execution = execution.step_execution
    assert step_execution[processing_step.name].status == _LocalExecutionStatus.FAILED.value
    assert "Dummy RuntimeError" in step_execution[processing_step.name].failure_reason


@patch(
    "sagemaker.local.local_session.LocalSagemakerClient.create_transform_job",
    side_effect=RuntimeError("Dummy RuntimeError"),
)
def test_execute_pipeline_step_create_transform_job_fail(
    create_transform_job, local_sagemaker_session, pipeline_session, transform_step
):
    pipeline = Pipeline(
        name="MyPipelineX-" + transform_step.name,
        steps=[transform_step],
        sagemaker_session=local_sagemaker_session,
    )
    execution = LocalPipelineExecutor(
        _LocalPipelineExecution("my-execution-x-" + transform_step.name, pipeline),
        local_sagemaker_session,
    ).execute()

    assert execution.status == _LocalExecutionStatus.FAILED.value
    assert execution.pipeline_execution_name == "my-execution-x-" + transform_step.name

    step_execution = execution.step_execution
    assert step_execution[transform_step.name].status == _LocalExecutionStatus.FAILED.value
    assert "Dummy RuntimeError" in step_execution[transform_step.name].failure_reason


@patch(
    "sagemaker.local.image._SageMakerContainer.train",
    MagicMock(return_value="/some/path/to/model"),
)
@patch("sagemaker.local.image._SageMakerContainer.process", MagicMock())
def test_pipeline_definition_config_in_local_mode_for_train_process_steps(
    processing_step,
    training_step,
    local_sagemaker_session,
):
    exe_steps = [processing_step, training_step]

    def _verify_execution(exe_step_name, execution, with_custom_job_prefix):
        assert not execution.failure_reason
        assert execution.status == _LocalExecutionStatus.SUCCEEDED.value

        step_execution = execution.step_execution[exe_step_name]
        assert step_execution.status == _LocalExecutionStatus.SUCCEEDED.value

        if step_execution.type == StepTypeEnum.PROCESSING:
            job_name_field = "ProcessingJobName"
        elif step_execution.type == StepTypeEnum.TRAINING:
            job_name_field = "TrainingJobName"

        if with_custom_job_prefix:
            assert step_execution.properties[job_name_field] == TEST_JOB_NAME
        else:
            assert step_execution.properties[job_name_field].startswith(step_execution.name)

    for exe_step in exe_steps:
        pipeline = Pipeline(
            name="MyPipelineX-" + exe_step.name,
            steps=[exe_step],
            sagemaker_session=local_sagemaker_session,
            parameters=[INSTANCE_COUNT_PIPELINE_PARAMETER],
        )

        execution = LocalPipelineExecutor(
            _LocalPipelineExecution("my-execution-x-" + exe_step.name, pipeline),
            local_sagemaker_session,
        ).execute()

        _verify_execution(
            exe_step_name=exe_step.name, execution=execution, with_custom_job_prefix=False
        )

        pipeline.pipeline_definition_config = PipelineDefinitionConfig(use_custom_job_prefix=True)
        execution = LocalPipelineExecutor(
            _LocalPipelineExecution("my-execution-x-" + exe_step.name, pipeline),
            local_sagemaker_session,
        ).execute()

        _verify_execution(
            exe_step_name=exe_step.name, execution=execution, with_custom_job_prefix=True
        )


@patch("sagemaker.local.local_session.LocalSagemakerClient.create_transform_job")
def test_pipeline_definition_config_in_local_mode_for_transform_step(
    create_transform_job, local_sagemaker_session, transform_step
):
    pipeline = Pipeline(
        name="MyPipelineX-" + transform_step.name,
        steps=[transform_step],
        sagemaker_session=local_sagemaker_session,
    )
    LocalPipelineExecutor(
        _LocalPipelineExecution("my-execution-x-" + transform_step.name, pipeline),
        local_sagemaker_session,
    ).execute()

    assert create_transform_job.call_args.args[0].startswith(transform_step.name)

    pipeline.pipeline_definition_config = PipelineDefinitionConfig(use_custom_job_prefix=True)

    LocalPipelineExecutor(
        _LocalPipelineExecution("my-execution-x-" + transform_step.name, pipeline),
        local_sagemaker_session,
    ).execute()

    assert create_transform_job.call_args.args[0] == TEST_JOB_NAME
