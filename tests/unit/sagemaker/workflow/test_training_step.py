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
import re

import pytest
import warnings

from sagemaker import Processor, Model
from sagemaker.parameter import IntegerParameter
from sagemaker.transformer import Transformer
from sagemaker.tuner import HyperparameterTuner
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString

from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline import Pipeline

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
from tests.unit.sagemaker.workflow.helpers import CustomStep

REGION = "us-west-2"
BUCKET = "my-bucket"
ROLE = "DummyRole"
IMAGE_URI = "fakeimage"
MODEL_NAME = "gisele"
DUMMY_LOCAL_SCRIPT_PATH = os.path.join(DATA_DIR, "dummy_script.py")
DUMMY_S3_SCRIPT_PATH = "s3://dummy-s3/dummy_script.py"
DUMMY_S3_SOURCE_DIR = "s3://dummy-s3-source-dir/"
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
def training_input():
    return TrainingInput(s3_data=f"s3://{BUCKET}/my-training-input")


@pytest.fixture
def hyperparameters():
    return {"test-key": "test-val"}


def test_training_step_with_estimator(pipeline_session, training_input, hyperparameters):
    custom_step1 = CustomStep("TestStep")
    custom_step2 = CustomStep("SecondTestStep")
    estimator = Estimator(
        role=ROLE,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        sagemaker_session=pipeline_session,
        image_uri=IMAGE_URI,
        hyperparameters=hyperparameters,
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
            depends_on=["TestStep", "SecondTestStep"],
        )
        assert len(w) == 0

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step, custom_step1, custom_step2],
        sagemaker_session=pipeline_session,
    )
    assert json.loads(pipeline.definition())["Steps"][0] == {
        "Name": "MyTrainingStep",
        "Description": "TrainingStep description",
        "DisplayName": "MyTrainingStep",
        "Type": "Training",
        "DependsOn": ["TestStep", "SecondTestStep"],
        "Arguments": step_args.args,
    }
    assert step.properties.TrainingJobName.expr == {"Get": "Steps.MyTrainingStep.TrainingJobName"}


def test_estimator_with_parameterized_output(pipeline_session, training_input):
    output_path = ParameterString(name="OutputPath")
    # XGBoost
    estimator = XGBoost(
        framework_version="1.3-1",
        py_version="py3",
        role=ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        entry_point=DUMMY_LOCAL_SCRIPT_PATH,
        output_path=output_path,
        sagemaker_session=pipeline_session,
    )
    step_args = estimator.fit(inputs=training_input)
    step1 = TrainingStep(
        name="MyTrainingStep1",
        step_args=step_args,
        description="TrainingStep description",
        display_name="MyTrainingStep",
    )

    # TensorFlow
    # If model_dir is None and output_path is a pipeline variable
    # a default model_dir will be generated with default bucket
    estimator = TensorFlow(
        framework_version="2.4.1",
        py_version="py37",
        role=ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        entry_point=DUMMY_LOCAL_SCRIPT_PATH,
        output_path=output_path,
        sagemaker_session=pipeline_session,
    )
    step_args = estimator.fit(inputs=training_input)
    step2 = TrainingStep(
        name="MyTrainingStep2",
        step_args=step_args,
        description="TrainingStep description",
        display_name="MyTrainingStep",
    )
    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step1, step2],
        parameters=[output_path],
        sagemaker_session=pipeline_session,
    )
    step_defs = json.loads(pipeline.definition())["Steps"]
    for step_def in step_defs:
        assert step_def["Arguments"]["OutputDataConfig"]["S3OutputPath"] == {
            "Get": "Parameters.OutputPath"
        }
        if step_def["Name"] != "MyTrainingStep2":
            continue
        model_dir = step_def["Arguments"]["HyperParameters"]["model_dir"]
        assert re.match(rf'"s3://{BUCKET}/.*/model"', model_dir)


@pytest.mark.parametrize(
    "estimator",
    [
        SKLearn(
            framework_version="0.23-1",
            py_version="py3",
            instance_type=INSTANCE_TYPE,
            instance_count=1,
            role=ROLE,
            entry_point=DUMMY_LOCAL_SCRIPT_PATH,
        ),
        PyTorch(
            role=ROLE,
            instance_type=INSTANCE_TYPE,
            instance_count=1,
            framework_version="1.8.0",
            py_version="py36",
            entry_point=DUMMY_LOCAL_SCRIPT_PATH,
        ),
        TensorFlow(
            role=ROLE,
            instance_type=INSTANCE_TYPE,
            instance_count=1,
            framework_version="2.0",
            py_version="py3",
            entry_point=DUMMY_LOCAL_SCRIPT_PATH,
        ),
        HuggingFace(
            transformers_version="4.6",
            pytorch_version="1.7",
            role=ROLE,
            instance_type="ml.p3.2xlarge",
            instance_count=1,
            py_version="py36",
            entry_point=DUMMY_LOCAL_SCRIPT_PATH,
        ),
        XGBoost(
            framework_version="1.3-1",
            py_version="py3",
            role=ROLE,
            instance_type=INSTANCE_TYPE,
            instance_count=1,
            entry_point=DUMMY_LOCAL_SCRIPT_PATH,
        ),
        MXNet(
            framework_version="1.4.1",
            py_version="py3",
            role=ROLE,
            instance_type=INSTANCE_TYPE,
            instance_count=1,
            entry_point=DUMMY_LOCAL_SCRIPT_PATH,
        ),
        RLEstimator(
            entry_point="cartpole.py",
            toolkit=RLToolkit.RAY,
            framework=RLFramework.TENSORFLOW,
            toolkit_version="0.8.5",
            role=ROLE,
            instance_type=INSTANCE_TYPE,
            instance_count=1,
        ),
        Chainer(
            role=ROLE,
            entry_point=DUMMY_LOCAL_SCRIPT_PATH,
            use_mpi=True,
            num_processes=4,
            framework_version="5.0.0",
            instance_type=INSTANCE_TYPE,
            instance_count=1,
            py_version="py3",
        ),
    ],
)
def test_training_step_with_framework_estimator(
    estimator, pipeline_session, training_input, hyperparameters
):
    estimator.source_dir = DUMMY_S3_SOURCE_DIR
    estimator.set_hyperparameters(**hyperparameters)
    estimator.volume_kms_key = "volume-kms-key"
    estimator.output_kms_key = "output-kms-key"
    estimator.dependencies = ["dep-1", "dep-2"]

    estimator.sagemaker_session = pipeline_session
    step_args = estimator.fit(inputs=training_input)

    step = TrainingStep(
        name="MyTrainingStep",
        step_args=step_args,
    )
    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step],
        sagemaker_session=pipeline_session,
    )

    assert json.loads(pipeline.definition())["Steps"][0] == {
        "Name": "MyTrainingStep",
        "Type": "Training",
        "Arguments": step_args.args,
    }


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
def test_training_step_with_algorithm_base(algo_estimator, pipeline_session):
    estimator = algo_estimator(
        role=ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        sagemaker_session=pipeline_session,
    )
    data = RecordSet(
        "s3://{}/{}".format(BUCKET, "dummy"),
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
    assert json.loads(pipeline.definition())["Steps"][0] == {
        "Name": "MyTrainingStep",
        "Type": "Training",
        "Arguments": step_args.args,
    }
    assert step.properties.TrainingJobName.expr == {"Get": "Steps.MyTrainingStep.TrainingJobName"}


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
