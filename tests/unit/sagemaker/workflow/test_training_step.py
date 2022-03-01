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


from sagemaker.inputs import TrainingInput

REGION = "us-west-2"
IMAGE_URI = "fakeimage"
MODEL_NAME = "gisele"
DUMMY_S3_SCRIPT_PATH = "s3://dummy-s3/dummy_script.py"
DUMMY_S3_SOURCE_DIR = "s3://dummy-s3-source-dir/"
INSTANCE_TYPE = "ml.m4.xlarge"


@pytest.fixture
def pipeline_session():
    return PipelineSession()


@pytest.fixture
def bucket(pipeline_session):
    return pipeline_session.default_bucket()


@pytest.fixture
def training_input(bucket):
    return TrainingInput(s3_data=f"s3://{bucket}/my-training-input")


@pytest.fixture
def hyperparameters():
    return {"test-key": "test-val"}


def test_training_step_with_estimator(pipeline_session, training_input, hyperparameters):
    estimator = Estimator(
        role=sagemaker.get_execution_role(),
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        sagemaker_session=pipeline_session,
        image_uri=IMAGE_URI,
        hyperparameters=hyperparameters,
    )

    with warnings.catch_warnings(record=True) as w:
        run_args = estimator.fit(inputs=training_input)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Running within a PipelineSession" in str(w[-1].message)

    with warnings.catch_warnings(record=True) as w:
        step = TrainingStep(
            name="MyTrainingStep",
            run_args=run_args,
            description="TrainingStep description",
            display_name="MyTrainingStep",
            depends_on=["TestStep", "SecondTestStep"],
        )
        assert len(w) == 0

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step],
        sagemaker_session=pipeline_session,
    )
    assert json.loads(pipeline.definition())["Steps"][0] == {
        "Name": "MyTrainingStep",
        "Description": "TrainingStep description",
        "DisplayName": "MyTrainingStep",
        "Type": "Training",
        "DependsOn": ["TestStep", "SecondTestStep"],
        "Arguments": run_args,
    }
    assert step.properties.TrainingJobName.expr == {"Get": "Steps.MyTrainingStep.TrainingJobName"}


@pytest.mark.parametrize(
    "estimator",
    [
        SKLearn(
            framework_version="0.23-1",
            py_version="py3",
            instance_type=INSTANCE_TYPE,
            instance_count=1,
            role=sagemaker.get_execution_role(),
            entry_point="entry_point.py",
        ),
        PyTorch(
            role=sagemaker.get_execution_role(),
            instance_type=INSTANCE_TYPE,
            instance_count=1,
            framework_version="1.8.0",
            py_version="py36",
            entry_point="entry_point.py",
        ),
        TensorFlow(
            role=sagemaker.get_execution_role(),
            instance_type=INSTANCE_TYPE,
            instance_count=1,
            framework_version="2.0",
            py_version="py3",
            entry_point="entry_point.py",
        ),
        HuggingFace(
            transformers_version="4.6",
            pytorch_version="1.7",
            role=sagemaker.get_execution_role(),
            instance_type="ml.p3.2xlarge",
            instance_count=1,
            py_version="py36",
            entry_point="entry_point.py",
        ),
        XGBoost(
            framework_version="1.3-1",
            py_version="py3",
            role=sagemaker.get_execution_role(),
            instance_type=INSTANCE_TYPE,
            instance_count=1,
            entry_point="entry_point.py",
        ),
        MXNet(
            framework_version="1.4.1",
            py_version="py3",
            role=sagemaker.get_execution_role(),
            instance_type=INSTANCE_TYPE,
            instance_count=1,
            entry_point="entry_point.py",
        ),
        RLEstimator(
            entry_point="cartpole.py",
            toolkit=RLToolkit.RAY,
            framework=RLFramework.TENSORFLOW,
            toolkit_version="0.8.5",
            role=sagemaker.get_execution_role(),
            instance_type=INSTANCE_TYPE,
            instance_count=1,
        ),
        Chainer(
            role=sagemaker.get_execution_role(),
            entry_point="entry_point.py",
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
    run_args = estimator.fit(inputs=training_input)

    step = TrainingStep(
        name="MyTrainingStep",
        run_args=run_args,
    )
    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step],
        sagemaker_session=pipeline_session,
    )

    assert json.loads(pipeline.definition())["Steps"][0] == {
        "Name": "MyTrainingStep",
        "Type": "Training",
        "Arguments": run_args,
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
        role=sagemaker.get_execution_role(),
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        sagemaker_session=pipeline_session,
    )
    data = RecordSet(
        "s3://{}/{}".format(pipeline_session.default_bucket(), "dummy"),
        num_records=1000,
        feature_dim=128,
        channel="train",
    )

    with warnings.catch_warnings(record=True) as w:
        run_args = estimator.fit(
            records=data,
            mini_batch_size=1000,
        )
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Running within a PipelineSession" in str(w[-1].message)

    with warnings.catch_warnings(record=True) as w:
        step = TrainingStep(
            name="MyTrainingStep",
            run_args=run_args,
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
        "Arguments": run_args,
    }
    assert step.properties.TrainingJobName.expr == {"Get": "Steps.MyTrainingStep.TrainingJobName"}
