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

import os
import json
import pytest
import warnings

import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker.workflow.steps import TuningStep
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline import Pipeline

from sagemaker.tuner import HyperparameterTuner, IntegerParameter
from sagemaker.pytorch.estimator import PyTorch

from tests.unit import DATA_DIR


@pytest.fixture
def pipeline_session():
    return PipelineSession()


@pytest.fixture
def entry_point():
    base_dir = os.path.join(DATA_DIR, "pytorch_mnist")
    return os.path.join(base_dir, "mnist.py")


def test_tuning_step_with_single_algo_tuner(pipeline_session, entry_point):
    inputs = TrainingInput(s3_data=f"s3://{pipeline_session.default_bucket()}/training-data")

    pytorch_estimator = PyTorch(
        entry_point=entry_point,
        role=sagemaker.get_execution_role(),
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
        run_args = tuner.fit(inputs=inputs)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Running within a PipelineSession" in str(w[-1].message)

    with warnings.catch_warnings(record=True) as w:
        step = TuningStep(
            name="MyTuningStep",
            run_args=run_args,
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
        "Arguments": run_args,
    }


def test_tuning_step_with_multi_algo_tuner(pipeline_session, entry_point):
    pytorch_estimator = PyTorch(
        entry_point=entry_point,
        role=sagemaker.get_execution_role(),
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
    run_args = tuner.fit(inputs=inputs)

    step = TuningStep(
        name="MyTuningStep",
        run_args=run_args,
    )

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step],
        sagemaker_session=pipeline_session,
    )

    assert json.loads(pipeline.definition())["Steps"][0] == {
        "Name": "MyTuningStep",
        "Type": "Tuning",
        "Arguments": run_args,
    }
