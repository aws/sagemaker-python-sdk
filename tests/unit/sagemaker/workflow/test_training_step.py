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
from mock import patch

import pytest
import warnings

from copy import deepcopy

from sagemaker import Processor, Model
from sagemaker.parameter import IntegerParameter
from sagemaker.processing import ProcessingOutput
from sagemaker.transformer import Transformer
from sagemaker.tuner import HyperparameterTuner
from sagemaker.workflow.pipeline_context import _PipelineConfig
from sagemaker.workflow.parameters import ParameterString, ParameterBoolean
from sagemaker.workflow.properties import PropertyFile

from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline import Pipeline, PipelineGraph
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.utilities import hash_files_or_dirs
from sagemaker.workflow.functions import Join, JsonGet

from sagemaker.estimator import Estimator
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.tensorflow.estimator import TensorFlow
from sagemaker.huggingface.estimator import HuggingFace
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.mxnet.estimator import MXNet
from sagemaker.rl.estimator import RLEstimator, RLToolkit, RLFramework
from sagemaker.chainer.estimator import Chainer

from sagemaker.amazon.amazon_estimator import RecordSet
from sagemaker.amazon.knn import KNN
from sagemaker.amazon.kmeans import KMeans
from sagemaker.amazon.linear_learner import LinearLearner
from sagemaker.amazon.lda import LDA
from sagemaker.amazon.pca import PCA
from sagemaker.amazon.factorization_machines import FactorizationMachines
from sagemaker.amazon.ipinsights import IPInsights
from sagemaker.amazon.randomcutforest import RandomCutForest
from sagemaker.amazon.ntm import NTM
from sagemaker.amazon.object2vec import Object2Vec

from tests.unit import DATA_DIR

from sagemaker.inputs import TrainingInput
from tests.unit.sagemaker.workflow.helpers import CustomStep, ordered, get_step_args_helper
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from tests.unit.sagemaker.workflow.conftest import ROLE, BUCKET, IMAGE_URI, INSTANCE_TYPE

LOCAL_ENTRY_POINT = os.path.join(DATA_DIR, "tfs/tfs-test-entrypoint-with-handler/training.py")
LOCAL_SOURCE_DIR = os.path.join(DATA_DIR, "tfs/tfs-test-entrypoint-with-handler")
LOCAL_DEPS = [
    os.path.join(DATA_DIR, "tfs/tfs-test-entrypoint-and-dependencies"),
]
DUMMY_S3_SCRIPT_PATH = "s3://dummy-s3/dummy_script.py"
MOCKED_PIPELINE_CONFIG = _PipelineConfig(
    "MyPipeline",
    "MyTrainingStep",
    None,
    hash_files_or_dirs([LOCAL_SOURCE_DIR] + LOCAL_DEPS),
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

ESTIMATOR_LISTS = [
    SKLearn(
        framework_version="0.23-1",
        py_version="py3",
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        role=ROLE,
        entry_point=ParameterString(name="EntryPoint"),
        source_dir=ParameterString(name="SourceDir"),
    ),
    PyTorch(
        role=ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        framework_version="1.8.0",
        py_version="py36",
        entry_point=ParameterString(name="EntryPoint"),
        source_dir=ParameterString(name="SourceDir"),
    ),
    TensorFlow(
        role=ROLE,
        entry_point=ParameterString(name="EntryPoint"),
        source_dir=ParameterString(name="SourceDir"),
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        framework_version="2.0",
        py_version="py3",
    ),
    HuggingFace(
        transformers_version="4.6",
        pytorch_version="1.7",
        role=ROLE,
        instance_type="ml.p3.2xlarge",
        instance_count=1,
        py_version="py36",
        entry_point=ParameterString(name="EntryPoint"),
        source_dir=ParameterString(name="SourceDir"),
    ),
    XGBoost(
        framework_version="1.3-1",
        py_version="py3",
        role=ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        entry_point=ParameterString(name="EntryPoint"),
        source_dir=ParameterString(name="SourceDir"),
    ),
    MXNet(
        framework_version="1.4.1",
        py_version="py3",
        role=ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        entry_point=ParameterString(name="EntryPoint"),
        source_dir=ParameterString(name="SourceDir"),
        toolkit=RLToolkit.RAY,
        framework=RLFramework.TENSORFLOW,
        toolkit_version="0.8.5",
    ),
    RLEstimator(
        entry_point=ParameterString(name="EntryPoint"),
        source_dir=ParameterString(name="SourceDir"),
        toolkit=RLToolkit.RAY,
        framework=RLFramework.TENSORFLOW,
        toolkit_version="0.8.5",
        role=ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
    ),
    Chainer(
        role=ROLE,
        entry_point=ParameterString(name="EntryPoint"),
        source_dir=ParameterString(name="SourceDir"),
        use_mpi=True,
        num_processes=4,
        framework_version="5.0.0",
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        py_version="py3",
    ),
]

ESTIMATOR_LISTS_LOCAL_CODE = [
    SKLearn(
        framework_version="0.23-1",
        py_version="py3",
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        role=ROLE,
        entry_point=LOCAL_ENTRY_POINT,
        source_dir=LOCAL_SOURCE_DIR,
    ),
    PyTorch(
        role=ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        framework_version="1.8.0",
        py_version="py36",
        entry_point=LOCAL_ENTRY_POINT,
        source_dir=LOCAL_SOURCE_DIR,
    ),
    TensorFlow(
        role=ROLE,
        entry_point=LOCAL_ENTRY_POINT,
        source_dir=LOCAL_SOURCE_DIR,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        framework_version="2.0",
        py_version="py3",
    ),
    HuggingFace(
        transformers_version="4.6",
        pytorch_version="1.7",
        role=ROLE,
        instance_type="ml.p3.2xlarge",
        instance_count=1,
        py_version="py36",
        entry_point=LOCAL_ENTRY_POINT,
        source_dir=LOCAL_SOURCE_DIR,
    ),
    XGBoost(
        framework_version="1.3-1",
        py_version="py3",
        role=ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        entry_point=LOCAL_ENTRY_POINT,
        source_dir=LOCAL_SOURCE_DIR,
    ),
    MXNet(
        framework_version="1.4.1",
        py_version="py3",
        role=ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        entry_point=LOCAL_ENTRY_POINT,
        source_dir=LOCAL_SOURCE_DIR,
        toolkit=RLToolkit.RAY,
        framework=RLFramework.TENSORFLOW,
        toolkit_version="0.8.5",
    ),
    RLEstimator(
        entry_point=LOCAL_ENTRY_POINT,
        source_dir=LOCAL_SOURCE_DIR,
        toolkit=RLToolkit.RAY,
        framework=RLFramework.TENSORFLOW,
        toolkit_version="0.8.5",
        role=ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
    ),
    Chainer(
        role=ROLE,
        entry_point=LOCAL_ENTRY_POINT,
        source_dir=LOCAL_SOURCE_DIR,
        use_mpi=True,
        num_processes=4,
        framework_version="5.0.0",
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        py_version="py3",
    ),
]


INPUT_PARAM_LISTS = [
    "s3://my-bucket/my-training-input",
    ParameterString(name="training_input", default_value="s3://my-bucket/my-input"),
    ParameterString(name="training_input"),
    Join(on="/", values=["s3://my-bucket", "my-input"]),
]

OUTPUT_PARAM_LIST = ["s3://my-bucket/my-output-path", ParameterString(name="OutputPath")]


@pytest.fixture
def training_input():
    return TrainingInput(s3_data=f"s3://{BUCKET}/my-training-input")


@pytest.fixture
def hyperparameters():
    return {"test-key": "test-val"}


@patch("sagemaker.workflow.utilities._pipeline_config", MOCKED_PIPELINE_CONFIG)
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
def test_training_step_with_estimator(
    pipeline_session, training_input, hyperparameters, experiment_config, expected_experiment_config
):
    custom_step1 = CustomStep("TestStep")
    custom_step2 = CustomStep("SecondTestStep")
    enable_network_isolation = ParameterBoolean(name="enable_network_isolation")
    encrypt_container_traffic = ParameterBoolean(name="encrypt_container_traffic")
    estimator = Estimator(
        role=ROLE,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        sagemaker_session=pipeline_session,
        image_uri=IMAGE_URI,
        hyperparameters=hyperparameters,
        enable_network_isolation=enable_network_isolation,
        encrypt_inter_container_traffic=encrypt_container_traffic,
        entry_point=LOCAL_ENTRY_POINT,
        source_dir=LOCAL_SOURCE_DIR,
        dependencies=LOCAL_DEPS,
    )

    with warnings.catch_warnings(record=True) as w:
        # TODO: remove job_name once we merge
        # https://github.com/aws/sagemaker-python-sdk/pull/3158/files
        step_args = estimator.fit(
            inputs=training_input, job_name="TestJob", experiment_config=experiment_config
        )
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Running within a PipelineSession" in str(w[-1].message)

    with warnings.catch_warnings(record=True) as w:
        step_train = TrainingStep(
            name="MyTrainingStep",
            step_args=step_args,
            description="TrainingStep description",
            display_name="MyTrainingStep",
            depends_on=["TestStep"],
        )
        assert len(w) == 0

    step_condition = ConditionStep(
        name="MyConditionStep",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=step_train.properties.FinalMetricDataList["val:acc"].Value, right=0.95
            )
        ],
        if_steps=[custom_step2],
    )

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step_train, step_condition, custom_step1],
        parameters=[enable_network_isolation, encrypt_container_traffic],
        sagemaker_session=pipeline_session,
    )
    step_args = get_step_args_helper(step_args, "Training")

    step_args["EnableInterContainerTrafficEncryption"] = {
        "Get": "Parameters.encrypt_container_traffic"
    }
    step_args["EnableNetworkIsolation"] = {"Get": "Parameters.enable_network_isolation"}
    if expected_experiment_config is None:
        step_args.pop("ExperimentConfig", None)
    else:
        step_args["ExperimentConfig"] = expected_experiment_config

    assert step_condition.conditions[0].left.expr == {
        "Get": "Steps.MyTrainingStep.FinalMetricDataList['val:acc'].Value"
    }
    step_definition = json.loads(pipeline.definition())["Steps"][0]

    assert step_definition == {
        "Name": "MyTrainingStep",
        "Description": "TrainingStep description",
        "DisplayName": "MyTrainingStep",
        "Type": "Training",
        "DependsOn": ["TestStep"],
        "Arguments": step_args,
    }
    assert step_train.properties.TrainingJobName.expr == {
        "Get": "Steps.MyTrainingStep.TrainingJobName"
    }
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered(
        {
            "MyConditionStep": ["SecondTestStep"],
            "MyTrainingStep": ["MyConditionStep"],
            "SecondTestStep": [],
            "TestStep": ["MyTrainingStep"],
        }
    )

    # test idempotency
    step_def2 = json.loads(pipeline.definition())["Steps"][0]
    assert step_definition == step_def2


def test_training_step_estimator_with_param_code_input(
    pipeline_session, training_input, hyperparameters
):
    entry_point = ParameterString(name="EntryPoint")
    source_dir = ParameterString(name="SourceDir")
    estimator = Estimator(
        role=ROLE,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        sagemaker_session=pipeline_session,
        image_uri=IMAGE_URI,
        hyperparameters=hyperparameters,
        entry_point=entry_point,
        source_dir=source_dir,
    )

    with warnings.catch_warnings(record=True) as w:
        # TODO: remove job_name once we merge
        # https://github.com/aws/sagemaker-python-sdk/pull/3158/files
        step_args = estimator.fit(inputs=training_input, job_name="TestJob")
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Running within a PipelineSession" in str(w[-1].message)

    with warnings.catch_warnings(record=True) as w:
        step = TrainingStep(
            name="MyTrainingStep",
            step_args=step_args,
            description="TrainingStep description",
            display_name="MyTrainingStep",
        )
        assert len(w) == 0

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step],
        sagemaker_session=pipeline_session,
    )
    step_args = get_step_args_helper(step_args, "Training")
    step_args["HyperParameters"]["sagemaker_program"] = {"Get": "Parameters.EntryPoint"}
    step_args["HyperParameters"]["sagemaker_submit_directory"] = {"Get": "Parameters.SourceDir"}
    step_def = json.loads(pipeline.definition())["Steps"][0]
    assert step_def == {
        "Name": "MyTrainingStep",
        "Description": "TrainingStep description",
        "DisplayName": "MyTrainingStep",
        "Type": "Training",
        "Arguments": step_args,
    }

    # test idempotency
    step_def2 = json.loads(pipeline.definition())["Steps"][0]
    assert step_def == step_def2


@pytest.mark.skip(reason="incompatible with pytest-xdist")
@pytest.mark.parametrize("estimator", ESTIMATOR_LISTS)
@pytest.mark.parametrize("training_input", INPUT_PARAM_LISTS)
@pytest.mark.parametrize(
    "output_path", ["s3://my-bucket/my-output-path", ParameterString(name="OutputPath")]
)
def test_training_step_with_framework_estimator(
    estimator, pipeline_session, training_input, output_path, hyperparameters
):
    estimator.set_hyperparameters(**hyperparameters)
    estimator.volume_kms_key = "volume-kms-key"
    estimator.output_kms_key = "output-kms-key"
    estimator.dependencies = ["dep-1", "dep-2"]
    estimator.output_path = output_path
    # TODO: remove job_name once we merge
    # https://github.com/aws/sagemaker-python-sdk/pull/3158/files
    estimator.base_job_name = "TestJob"

    estimator.sagemaker_session = pipeline_session
    step_args = estimator.fit(inputs=TrainingInput(s3_data=training_input))

    step = TrainingStep(
        name="MyTrainingStep",
        step_args=step_args,
    )
    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step],
        sagemaker_session=pipeline_session,
    )

    step_args = get_step_args_helper(step_args, "Training")
    expected_step_args = deepcopy(step_args)
    step_def = json.loads(pipeline.definition())["Steps"][0]

    assert (
        expected_step_args["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"]
        == training_input
    )
    assert expected_step_args["OutputDataConfig"]["S3OutputPath"] == output_path
    expected_step_args["HyperParameters"]["sagemaker_program"] = {"Get": "Parameters.EntryPoint"}
    expected_step_args["HyperParameters"]["sagemaker_submit_directory"] = {
        "Get": "Parameters.SourceDir"
    }

    del expected_step_args["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"]
    del step_def["Arguments"]["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"]

    del expected_step_args["OutputDataConfig"]["S3OutputPath"]
    del step_def["Arguments"]["OutputDataConfig"]["S3OutputPath"]

    if "sagemaker_s3_output" in step_args["HyperParameters"]:
        del expected_step_args["HyperParameters"]["sagemaker_s3_output"]
        del step_def["Arguments"]["HyperParameters"]["sagemaker_s3_output"]

    assert step_def == {
        "Name": "MyTrainingStep",
        "Type": "Training",
        "Arguments": expected_step_args,
    }

    # test idempotency
    step_def2 = json.loads(pipeline.definition())["Steps"][0]
    del step_def2["Arguments"]["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"]
    del step_def2["Arguments"]["OutputDataConfig"]["S3OutputPath"]
    if "sagemaker_s3_output" in step_def2["Arguments"]["HyperParameters"]:
        del step_def2["Arguments"]["HyperParameters"]["sagemaker_s3_output"]
    assert step_def == step_def2


@pytest.mark.skip(reason="incompatible with pytest-xdist")
@patch("sagemaker.workflow.utilities._pipeline_config", MOCKED_PIPELINE_CONFIG)
@pytest.mark.parametrize("estimator", ESTIMATOR_LISTS_LOCAL_CODE)
@pytest.mark.parametrize("training_input", INPUT_PARAM_LISTS)
@pytest.mark.parametrize(
    "output_path", ["s3://my-bucket/my-output-path", ParameterString(name="OutputPath")]
)
def test_training_step_with_framework_estimator_local_code(
    estimator, pipeline_session, training_input, output_path, hyperparameters
):
    estimator.set_hyperparameters(**hyperparameters)
    estimator.volume_kms_key = "volume-kms-key"
    estimator.output_kms_key = "output-kms-key"
    estimator.dependencies = LOCAL_DEPS
    estimator.output_path = output_path
    # TODO: remove job_name once we merge
    # https://github.com/aws/sagemaker-python-sdk/pull/3158/files
    estimator.base_job_name = "TestJob"

    estimator.sagemaker_session = pipeline_session
    step_args = estimator.fit(inputs=TrainingInput(s3_data=training_input))

    step = TrainingStep(
        name="MyTrainingStep",
        step_args=step_args,
    )
    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step],
        sagemaker_session=pipeline_session,
    )

    step_args = get_step_args_helper(step_args, "Training")
    expected_step_args = deepcopy(step_args)
    step_def = json.loads(pipeline.definition())["Steps"][0]

    assert (
        expected_step_args["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"]
        == training_input
    )
    assert expected_step_args["OutputDataConfig"]["S3OutputPath"] == output_path

    del expected_step_args["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"]
    del step_def["Arguments"]["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"]

    del expected_step_args["OutputDataConfig"]["S3OutputPath"]
    del step_def["Arguments"]["OutputDataConfig"]["S3OutputPath"]

    if "sagemaker_s3_output" in step_args["HyperParameters"]:
        del expected_step_args["HyperParameters"]["sagemaker_s3_output"]
        del step_def["Arguments"]["HyperParameters"]["sagemaker_s3_output"]

    assert step_def == {
        "Name": "MyTrainingStep",
        "Type": "Training",
        "Arguments": expected_step_args,
    }

    # test idempotency
    step_def2 = json.loads(pipeline.definition())["Steps"][0]
    del step_def2["Arguments"]["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"]
    del step_def2["Arguments"]["OutputDataConfig"]["S3OutputPath"]
    if "sagemaker_s3_output" in step_def2["Arguments"]["HyperParameters"]:
        del step_def2["Arguments"]["HyperParameters"]["sagemaker_s3_output"]
    assert step_def == step_def2


@pytest.mark.parametrize(
    "algo_estimator",
    [
        KNN,
        KMeans,
        LinearLearner,
        RandomCutForest,
        LDA,
        Object2Vec,
        NTM,
        PCA,
        FactorizationMachines,
        IPInsights,
    ],
)
@pytest.mark.parametrize(
    "training_input",
    INPUT_PARAM_LISTS,
)
def test_training_step_with_algorithm_base(algo_estimator, training_input, pipeline_session):
    estimator = algo_estimator(
        role=ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        sagemaker_session=pipeline_session,
        entry_point=ParameterString(name="EntryPoint"),
        source_dir=ParameterString(name="SourceDir"),
        # TODO: remove job_name once we merge
        # https://github.com/aws/sagemaker-python-sdk/pull/3158/files
        base_job_name="TestJob",
    )
    data = RecordSet(
        s3_data=training_input,
        num_records=1000,
        feature_dim=128,
        channel="train",
    )

    with warnings.catch_warnings(record=True) as w:
        step_args = estimator.fit(
            records=data,
            mini_batch_size=1000,
        )
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Running within a PipelineSession" in str(w[-1].message)

    with warnings.catch_warnings(record=True) as w:
        step = TrainingStep(
            name="MyTrainingStep",
            step_args=step_args,
        )
        assert len(w) == 0

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step],
        sagemaker_session=pipeline_session,
    )

    step_args = get_step_args_helper(step_args, "Training")

    step_def = json.loads(pipeline.definition())["Steps"][0]
    assert step_args["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"] == training_input
    step_args["HyperParameters"]["sagemaker_program"] = {"Get": "Parameters.EntryPoint"}
    step_args["HyperParameters"]["sagemaker_submit_directory"] = {"Get": "Parameters.SourceDir"}
    del step_args["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"]
    del step_def["Arguments"]["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"]

    assert step_def == {
        "Name": "MyTrainingStep",
        "Type": "Training",
        "Arguments": step_args,
    }

    # test idempotency
    step_def2 = json.loads(pipeline.definition())["Steps"][0]
    del step_def2["Arguments"]["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"]
    assert step_def == step_def2


@patch("sagemaker.workflow.utilities._pipeline_config", MOCKED_PIPELINE_CONFIG)
@pytest.mark.parametrize(
    "algo_estimator",
    [
        KNN,
        KMeans,
        LinearLearner,
        RandomCutForest,
        LDA,
        Object2Vec,
        NTM,
        PCA,
        FactorizationMachines,
        IPInsights,
    ],
)
@pytest.mark.parametrize(
    "training_input",
    INPUT_PARAM_LISTS,
)
def test_training_step_with_algorithm_base_local_code(
    algo_estimator, training_input, pipeline_session
):
    estimator = algo_estimator(
        role=ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        sagemaker_session=pipeline_session,
        entry_point=LOCAL_ENTRY_POINT,
        source_dir=LOCAL_SOURCE_DIR,
        dependencies=LOCAL_DEPS,
        # TODO: remove job_name once we merge
        # https://github.com/aws/sagemaker-python-sdk/pull/3158/files
        base_job_name="TestJob",
    )
    data = RecordSet(
        s3_data=training_input,
        num_records=1000,
        feature_dim=128,
        channel="train",
    )

    with warnings.catch_warnings(record=True) as w:
        step_args = estimator.fit(
            records=data,
            mini_batch_size=1000,
        )
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Running within a PipelineSession" in str(w[-1].message)

    with warnings.catch_warnings(record=True) as w:
        step = TrainingStep(
            name="MyTrainingStep",
            step_args=step_args,
        )
        assert len(w) == 0

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step],
        sagemaker_session=pipeline_session,
    )

    step_args = get_step_args_helper(step_args, "Training")

    step_def = json.loads(pipeline.definition())["Steps"][0]
    assert step_args["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"] == training_input
    del step_args["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"]
    del step_def["Arguments"]["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"]

    assert step_def == {
        "Name": "MyTrainingStep",
        "Type": "Training",
        "Arguments": step_args,
    }

    # test idempotency
    step_def2 = json.loads(pipeline.definition())["Steps"][0]
    del step_def2["Arguments"]["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"]
    assert step_def == step_def2


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
def test_insert_wrong_step_args_into_training_step(inputs, pipeline_session):
    downstream_obj, target_func_cfg = inputs
    if isinstance(downstream_obj, HyperparameterTuner):
        downstream_obj.estimator.sagemaker_session = pipeline_session
    else:
        downstream_obj.sagemaker_session = pipeline_session
    func_name = target_func_cfg["target_fun"]
    func_args = target_func_cfg["func_args"]
    step_args = getattr(downstream_obj, func_name)(**func_args)

    with pytest.raises(ValueError) as error:
        TrainingStep(
            name="MyTrainingStep",
            step_args=step_args,
        )

    assert "The step_args of TrainingStep must be obtained from estimator.fit()" in str(error.value)


@patch("sagemaker.workflow.utilities._pipeline_config", MOCKED_PIPELINE_CONFIG_WITH_CUSTOM_PREFIX)
def test_training_step_with_estimator_using_custom_prefixes(
    pipeline_session, training_input, hyperparameters
):
    entry_point = ParameterString(name="EntryPoint")
    source_dir = ParameterString(name="SourceDir")

    custom_job_prefix = "TrainingJobPrefix"
    estimator = Estimator(
        role=ROLE,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        sagemaker_session=pipeline_session,
        image_uri=IMAGE_URI,
        hyperparameters=hyperparameters,
        entry_point=entry_point,
        source_dir=source_dir,
        base_job_name=custom_job_prefix,  # Include a custom prefix for the job
    )

    with warnings.catch_warnings(record=True) as w:
        step_args = estimator.fit(inputs=training_input)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Running within a PipelineSession" in str(w[-1].message)

    with warnings.catch_warnings(record=True) as w:
        step = TrainingStep(
            name="MyTrainingStep",
            step_args=step_args,
            description="TrainingStep description",
            display_name="MyTrainingStep",
        )
        assert len(w) == 0

    # Toggle the custom prefixing feature ON for this pipeline
    definition_config = PipelineDefinitionConfig(use_custom_job_prefix=True)

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step],
        sagemaker_session=pipeline_session,
        pipeline_definition_config=definition_config,
    )
    step_args = get_step_args_helper(step_args, "Training", True)
    step_args["HyperParameters"]["sagemaker_program"] = {"Get": "Parameters.EntryPoint"}
    step_args["HyperParameters"]["sagemaker_submit_directory"] = {"Get": "Parameters.SourceDir"}
    step_def = json.loads(pipeline.definition())["Steps"][0]

    assert step_def["Arguments"]["TrainingJobName"] == "TrainingJobPrefix"
    assert step_def == {
        "Name": "MyTrainingStep",
        "Description": "TrainingStep description",
        "DisplayName": "MyTrainingStep",
        "Type": "Training",
        "Arguments": step_args,
    }


def test_training_step_with_jsonget_instance_type(pipeline_session):
    property_file = PropertyFile(
        name="my-property-file", output_name="TestOutputName", path="processing_output.json"
    )
    processor = Processor(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_count=1,
        instance_type="c4.4xlarge",
        sagemaker_session=pipeline_session,
    )
    process_arg = processor.run(outputs=[ProcessingOutput(output_name="TestOutputName")])
    processing_step = ProcessingStep(
        name="inputProcessingStep",
        step_args=process_arg,
        property_files=[property_file],
    )

    json_get_function = JsonGet(
        step_name=processing_step.name, property_file=property_file.name, json_path="mse"
    )

    estimator = Estimator(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_count=1,
        instance_type=json_get_function,
        sagemaker_session=pipeline_session,
    )

    training_step = TrainingStep(
        name="MyTrainingStep",
        step_args=estimator.fit(inputs=TrainingInput(s3_data=f"s3://{BUCKET}/train_manifest")),
    )
    pipeline = Pipeline(
        name="MyPipeline",
        steps=[processing_step, training_step],
        sagemaker_session=pipeline_session,
    )

    steps = json.loads(pipeline.definition())["Steps"]
    for step in steps:
        if step["Type"] == "Processing":
            continue
        assert step["Arguments"]["ResourceConfig"]["InstanceType"] == {
            "Std:JsonGet": {
                "Path": "mse",
                "PropertyFile": {"Get": "Steps.inputProcessingStep.PropertyFiles.my-property-file"},
            }
        }
