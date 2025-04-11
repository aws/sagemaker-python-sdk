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
import pytest
import warnings

from sagemaker import Model, Processor
from sagemaker.estimator import Estimator
from sagemaker.transformer import Transformer
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig

from sagemaker.workflow.steps import TuningStep
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline import Pipeline, PipelineGraph
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.functions import Join

from sagemaker.tuner import HyperparameterTuner, IntegerParameter
from sagemaker.pytorch.estimator import PyTorch

from tests.unit import DATA_DIR
from tests.unit.sagemaker.workflow.helpers import get_step_args_helper
from tests.unit.sagemaker.workflow.conftest import ROLE, INSTANCE_TYPE, IMAGE_URI


@pytest.fixture
def entry_point():
    base_dir = os.path.join(DATA_DIR, "pytorch_mnist")
    return os.path.join(base_dir, "mnist.py")


@pytest.mark.parametrize(
    "training_input",
    [
        "s3://my-bucket/my-training-input",
        ParameterString(name="training_input", default_value="s3://my-bucket/my-input"),
        ParameterString(name="training_input"),
        Join(on="/", values=["s3://my-bucket", "my-input"]),
    ],
)
def test_tuning_step_with_single_algo_tuner(pipeline_session, training_input, entry_point):
    inputs = TrainingInput(s3_data=training_input)

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

    step_args = get_step_args_helper(step_args, "HyperParameterTuning")
    step_def = json.loads(pipeline.definition())["Steps"][0]

    assert (
        step_args["TrainingJobDefinition"]["InputDataConfig"][0]["DataSource"]["S3DataSource"][
            "S3Uri"
        ]
        == training_input
    )
    del step_args["TrainingJobDefinition"]["InputDataConfig"][0]["DataSource"]["S3DataSource"][
        "S3Uri"
    ]
    del step_def["Arguments"]["TrainingJobDefinition"]["InputDataConfig"][0]["DataSource"][
        "S3DataSource"
    ]["S3Uri"]

    # delete sagemaker_job_name b/c of timestamp collision
    del step_args["TrainingJobDefinition"]["StaticHyperParameters"]["sagemaker_job_name"]
    del step_def["Arguments"]["TrainingJobDefinition"]["StaticHyperParameters"][
        "sagemaker_job_name"
    ]

    # delete S3 path assertions for now because job name is included with timestamp. These will be re-enabled after
    # caching improvements phase 2.
    del step_args["TrainingJobDefinition"]["StaticHyperParameters"]["sagemaker_submit_directory"]
    del step_def["Arguments"]["TrainingJobDefinition"]["StaticHyperParameters"][
        "sagemaker_submit_directory"
    ]

    assert step_def == {
        "Name": "MyTuningStep",
        "Type": "Tuning",
        "Arguments": step_args,
    }
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert adjacency_list == {"MyTuningStep": []}

    # test idempotency
    step_def2 = json.loads(pipeline.definition())["Steps"][0]
    del step_def2["Arguments"]["TrainingJobDefinition"]["StaticHyperParameters"][
        "sagemaker_job_name"
    ]
    del step_def2["Arguments"]["TrainingJobDefinition"]["StaticHyperParameters"][
        "sagemaker_submit_directory"
    ]
    assert step_def == step_def2


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

    step_args = get_step_args_helper(step_args, "HyperParameterTuning")
    step_def = json.loads(pipeline.definition())["Steps"][0]

    for i, step in enumerate(step_args["TrainingJobDefinitions"]):
        # delete sagemaker_job_name b/c of timestamp collision
        del step_args["TrainingJobDefinitions"][i]["StaticHyperParameters"]["sagemaker_job_name"]
        del step_def["Arguments"]["TrainingJobDefinitions"][i]["StaticHyperParameters"][
            "sagemaker_job_name"
        ]

        # delete S3 path assertions for now because job name is included with timestamp. These will be re-enabled after
        # caching improvements phase 2.
        del step_args["TrainingJobDefinitions"][i]["StaticHyperParameters"][
            "sagemaker_submit_directory"
        ]
        del step_def["Arguments"]["TrainingJobDefinitions"][i]["StaticHyperParameters"][
            "sagemaker_submit_directory"
        ]

    assert step_def == {
        "Name": "MyTuningStep",
        "Type": "Tuning",
        "Arguments": step_args,
    }
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert adjacency_list == {"MyTuningStep": []}

    # test idempotency
    step_def2 = json.loads(pipeline.definition())["Steps"][0]
    for i, step in enumerate(step_def2["Arguments"]["TrainingJobDefinitions"]):
        del step_def2["Arguments"]["TrainingJobDefinitions"][i]["StaticHyperParameters"][
            "sagemaker_job_name"
        ]

        del step_def2["Arguments"]["TrainingJobDefinitions"][i]["StaticHyperParameters"][
            "sagemaker_submit_directory"
        ]
    assert step_def == step_def2


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


def test_single_tuning_step_using_custom_job_prefixes(pipeline_session, entry_point):
    pytorch_estimator = PyTorch(
        entry_point=entry_point,
        role=ROLE,
        framework_version="1.5.0",
        py_version="py3",
        instance_count=1,
        instance_type="ml.m5.xlarge",
        sagemaker_session=pipeline_session,
    )

    hyperparameter_ranges = {
        "batch-size": IntegerParameter(64, 128),
    }

    tuner = HyperparameterTuner(
        estimator=pytorch_estimator,
        base_tuning_job_name="MyTuningJobPrefix",  # custom-job-prefix
        objective_metric_name="test:acc",
        objective_type="Maximize",
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=[{"Name": "test:acc", "Regex": "Overall test accuracy: (.*?);"}],
        max_jobs=2,
        max_parallel_jobs=2,
    )

    inputs = TrainingInput(s3_data="s3://my-bucket/my-training-input")
    step_args = tuner.fit(inputs=inputs)

    step = TuningStep(
        name="MyTuningStep",
        step_args=step_args,
    )

    definition_config = PipelineDefinitionConfig(use_custom_job_prefix=True)

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step],
        sagemaker_session=pipeline_session,
        pipeline_definition_config=definition_config,
    )

    step_def = json.loads(pipeline.definition())["Steps"][0]
    assert step_def["Arguments"]["HyperParameterTuningJobName"] == "MyTuningJobPrefix"
