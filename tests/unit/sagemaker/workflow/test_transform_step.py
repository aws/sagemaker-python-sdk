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

from sagemaker import Model, Processor
from sagemaker.estimator import Estimator
from sagemaker.parameter import IntegerParameter
from sagemaker.tuner import HyperparameterTuner
from sagemaker.workflow.pipeline_context import PipelineSession
from tests.unit.sagemaker.workflow.helpers import CustomStep, get_step_args_helper

from sagemaker.workflow.steps import TransformStep, TransformInput
from sagemaker.workflow.pipeline import Pipeline, PipelineGraph
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.functions import Join
from sagemaker.workflow import is_pipeline_variable

from sagemaker.transformer import Transformer

REGION = "us-west-2"
ROLE = "DummyRole"
IMAGE_URI = "fakeimage"
MODEL_NAME = "gisele"
DUMMY_S3_SCRIPT_PATH = "s3://dummy-s3/dummy_script.py"
DUMMY_S3_SOURCE_DIR = "s3://dummy-s3-source-dir/"
INSTANCE_TYPE = "ml.m4.xlarge"
BUCKET = "my-bucket"
custom_step = CustomStep(name="my-custom-step")


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
    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=client,
        default_bucket=BUCKET,
    )


@pytest.mark.parametrize(
    "model_name",
    [
        "my-model",
        ParameterString("ModelName"),
        ParameterString("ModelName", default_value="my-model"),
        Join(on="-", values=["my", "model"]),
        custom_step.properties.RoleArn,
    ],
)
@pytest.mark.parametrize(
    "data",
    [
        "s3://my-bucket/my-data",
        ParameterString("MyTransformInput"),
        ParameterString("MyTransformInput", default_value="s3://my-model"),
        Join(on="/", values=["s3://my-bucket", "my-transform-data", "input"]),
        custom_step.properties.OutputDataConfig.S3OutputPath,
    ],
)
@pytest.mark.parametrize(
    "output_path",
    [
        "s3://my-bucket/my-output-path",
        ParameterString("MyOutputPath"),
        ParameterString("MyOutputPath", default_value="s3://my-output"),
        Join(on="/", values=["s3://my-bucket", "my-transform-data", "output"]),
        custom_step.properties.OutputDataConfig.S3OutputPath,
    ],
)
def test_transform_step_with_transformer(model_name, data, output_path, pipeline_session):
    transformer = Transformer(
        model_name=model_name,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        output_path=output_path,
        sagemaker_session=pipeline_session,
    )
    transform_inputs = TransformInput(data=data)

    with warnings.catch_warnings(record=True) as w:
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
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Running within a PipelineSession" in str(w[-1].message)

    with warnings.catch_warnings(record=True) as w:
        step = TransformStep(
            name="MyTransformStep",
            step_args=step_args,
        )
        assert len(w) == 0

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step, custom_step],
        parameters=[model_name, data],
        sagemaker_session=pipeline_session,
    )

    step_args = get_step_args_helper(step_args, "Transform")
    expected_step_arguments = deepcopy(step_args)
    expected_step_arguments["ModelName"] = (
        model_name.expr if is_pipeline_variable(model_name) else model_name
    )
    expected_step_arguments["TransformInput"]["DataSource"]["S3DataSource"]["S3Uri"] = (
        data.expr if is_pipeline_variable(data) else data
    )
    expected_step_arguments["TransformOutput"]["S3OutputPath"] = (
        output_path.expr if is_pipeline_variable(output_path) else output_path
    )

    step_def = json.loads(pipeline.definition())["Steps"][0]
    assert step_def == {
        "Name": "MyTransformStep",
        "Type": "Transform",
        "Arguments": expected_step_arguments,
    }


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
def test_transform_step_with_transformer_experiment_config(
    experiment_config, expected_experiment_config, pipeline_session
):
    transformer = Transformer(
        model_name="my_model",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        output_path="s3://my-bucket/my-output-path",
        sagemaker_session=pipeline_session,
    )
    transform_inputs = TransformInput(data="s3://my-bucket/my-data")

    with warnings.catch_warnings(record=True) as w:
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
            experiment_config=experiment_config,
        )
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Running within a PipelineSession" in str(w[-1].message)

    with warnings.catch_warnings(record=True) as w:
        step = TransformStep(
            name="MyTransformStep",
            step_args=step_args,
        )
        assert len(w) == 0

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step],
        sagemaker_session=pipeline_session,
    )

    step_args = get_step_args_helper(step_args, "Transform")
    expected_step_arguments = deepcopy(step_args)
    if expected_experiment_config is None:
        expected_step_arguments.pop("ExperimentConfig", None)
    else:
        expected_step_arguments["ExperimentConfig"] = expected_experiment_config

    step_def = json.loads(pipeline.definition())["Steps"][0]
    assert step_def == {
        "Name": "MyTransformStep",
        "Type": "Transform",
        "Arguments": expected_step_arguments,
    }

    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert adjacency_list == {"MyTransformStep": []}


@pytest.mark.parametrize(
    "inputs",
    [
        (
            Processor(
                image_uri=IMAGE_URI,
                role=ROLE,
                instance_count=1,
                instance_type=INSTANCE_TYPE,
            ),
            dict(target_fun="run", func_args={}),
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
def test_insert_wrong_step_args_into_transform_step(inputs, pipeline_session):
    downstream_obj, target_func_cfg = inputs
    if isinstance(downstream_obj, HyperparameterTuner):
        downstream_obj.estimator.sagemaker_session = pipeline_session
    else:
        downstream_obj.sagemaker_session = pipeline_session
    func_name = target_func_cfg["target_fun"]
    func_args = target_func_cfg["func_args"]
    step_args = getattr(downstream_obj, func_name)(**func_args)

    with pytest.raises(ValueError) as error:
        TransformStep(
            name="MyTransformStep",
            step_args=step_args,
        )

    assert "The step_args of TransformStep must be obtained from transformer.transform()" in str(
        error.value
    )
