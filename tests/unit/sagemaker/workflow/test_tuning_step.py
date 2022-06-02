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
import json
from mock import Mock, PropertyMock
import pytest
import warnings

from sagemaker import Model, Processor
from sagemaker.estimator import Estimator
from sagemaker.transformer import Transformer
from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker.workflow.steps import TuningStep
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline import Pipeline

from sagemaker.tuner import HyperparameterTuner, IntegerParameter
from sagemaker.pytorch.estimator import PyTorch

from tests.unit import DATA_DIR


REGION = "us-west-2"
BUCKET = "my-bucket"
ROLE = "DummyRole"
IMAGE_URI = "fakeimage"
INSTANCE_TYPE = "ml.m4.xlarge"


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
def entry_point():
    base_dir = os.path.join(DATA_DIR, "pytorch_mnist")
    return os.path.join(base_dir, "mnist.py")


def test_tuning_step_with_single_algo_tuner(pipeline_session, entry_point):
    inputs = TrainingInput(s3_data=f"s3://{pipeline_session.default_bucket()}/training-data")

    pytorch_estimator = PyTorch(
        entry_point=entry_point,
        role=ROLE,
        framework_version="1.5.0",
        py_version="py3",
        instance_count=1,
        instance_type="ml.m5.xlarge",
        sagemaker_session=pipeline_session,
        enable_sagemaker_metrics=True,
        max_retry_attempts=3,
    )

    hyperparameter_ranges = {
        "batch-size": IntegerParameter(64, 128),
    }

    tuner = HyperparameterTuner(
        estimator=pytorch_estimator,
        objective_metric_name="test:acc",
        objective_type="Maximize",
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=[{"Name": "test:acc", "Regex": "Overall test accuracy: (.*?);"}],
        max_jobs=2,
        max_parallel_jobs=2,
    )

    with warnings.catch_warnings(record=True) as w:
        step_args = tuner.fit(inputs=inputs)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Running within a PipelineSession" in str(w[-1].message)

    with warnings.catch_warnings(record=True) as w:
        step = TuningStep(
            name="MyTuningStep",
            step_args=step_args,
        )
        assert len(w) == 0

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step],
        sagemaker_session=pipeline_session,
    )

    assert json.loads(pipeline.definition())["Steps"][0] == {
        "Name": "MyTuningStep",
        "Type": "Tuning",
        "Arguments": step_args.args,
    }


def test_tuning_step_with_multi_algo_tuner(pipeline_session, entry_point):
    pytorch_estimator = PyTorch(
        entry_point=entry_point,
        role=ROLE,
        framework_version="1.5.0",
        py_version="py3",
        instance_count=1,
        instance_type="ml.m5.xlarge",
        sagemaker_session=pipeline_session,
        enable_sagemaker_metrics=True,
        max_retry_attempts=3,
        hyperparameters={"static-hp": "hp1", "train_size": "1280"},
    )

    tuner = HyperparameterTuner.create(
        estimator_dict={
            "estimator-1": pytorch_estimator,
            "estimator-2": pytorch_estimator,
        },
        objective_metric_name_dict={
            "estimator-1": "test:acc",
            "estimator-2": "test:acc",
        },
        hyperparameter_ranges_dict={
            "estimator-1": {"batch-size": IntegerParameter(64, 128)},
            "estimator-2": {"batch-size": IntegerParameter(256, 512)},
        },
        metric_definitions_dict={
            "estimator-1": [{"Name": "test:acc", "Regex": "Overall test accuracy: (.*?);"}],
            "estimator-2": [{"Name": "test:acc", "Regex": "Overall test accuracy: (.*?);"}],
        },
    )
    input_path = f"s3://{pipeline_session.default_bucket()}/training-data"
    inputs = {
        "estimator-1": TrainingInput(s3_data=input_path),
        "estimator-2": TrainingInput(s3_data=input_path),
    }
    step_args = tuner.fit(
        inputs=inputs,
        include_cls_metadata={
            "estimator-1": False,
            "estimator-2": False,
        },
    )

    step = TuningStep(
        name="MyTuningStep",
        step_args=step_args,
    )

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step],
        sagemaker_session=pipeline_session,
    )

    assert json.loads(pipeline.definition())["Steps"][0] == {
        "Name": "MyTuningStep",
        "Type": "Tuning",
        "Arguments": step_args.args,
    }


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
            Processor(
                image_uri=IMAGE_URI,
                role=ROLE,
                instance_count=1,
                instance_type=INSTANCE_TYPE,
            ),
            dict(target_fun="run", func_args={}),
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
def test_insert_wrong_step_args_into_tuning_step(inputs, pipeline_session):
    downstream_obj, target_func_cfg = inputs
    downstream_obj.sagemaker_session = pipeline_session
    func_name = target_func_cfg["target_fun"]
    func_args = target_func_cfg["func_args"]
    step_args = getattr(downstream_obj, func_name)(**func_args)

    with pytest.raises(ValueError) as error:
        TuningStep(
            name="MyTuningStep",
            step_args=step_args,
        )

    assert "The step_args of TuningStep must be obtained from tuner.fit()" in str(error.value)
