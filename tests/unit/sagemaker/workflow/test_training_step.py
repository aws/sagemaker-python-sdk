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

from sagemaker import Processor, Model
from sagemaker.parameter import IntegerParameter
from sagemaker.transformer import Transformer
from sagemaker.tuner import HyperparameterTuner
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString, ParameterBoolean

from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline import Pipeline, PipelineGraph
from sagemaker.workflow.functions import Join

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
from tests.unit.sagemaker.workflow.helpers import CustomStep, ordered

REGION = "us-west-2"
BUCKET = "my-bucket"
ROLE = "DummyRole"
IMAGE_URI = "fakeimage"
MODEL_NAME = "gisele"
DUMMY_LOCAL_SCRIPT_PATH = os.path.join(DATA_DIR, "dummy_script.py")
DUMMY_S3_SCRIPT_PATH = "s3://dummy-s3/dummy_script.py"
INSTANCE_TYPE = "ml.m4.xlarge"

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


INPUT_PARAM_LISTS = [
    "s3://my-bucket/my-training-input",
    ParameterString(name="training_input", default_value="s3://my-bucket/my-input"),
    ParameterString(name="training_input"),
    Join(on="/", values=["s3://my-bucket", "my-input"]),
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
def training_input():
    return TrainingInput(s3_data=f"s3://{BUCKET}/my-training-input")


@pytest.fixture
def hyperparameters():
    return {"test-key": "test-val"}


def test_training_step_with_estimator(pipeline_session, training_input, hyperparameters):
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
            depends_on=["TestStep", "SecondTestStep"],
        )
        assert len(w) == 0

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step, custom_step1, custom_step2],
        parameters=[enable_network_isolation, encrypt_container_traffic],
        sagemaker_session=pipeline_session,
    )
    step_args.args["EnableInterContainerTrafficEncryption"] = {
        "Get": "Parameters.encrypt_container_traffic"
    }
    step_args.args["EnableNetworkIsolation"] = {"Get": "Parameters.encrypt_container_traffic"}
    assert json.loads(pipeline.definition())["Steps"][0] == {
        "Name": "MyTrainingStep",
        "Description": "TrainingStep description",
        "DisplayName": "MyTrainingStep",
        "Type": "Training",
        "DependsOn": ["TestStep", "SecondTestStep"],
        "Arguments": step_args.args,
    }
    assert step.properties.TrainingJobName.expr == {"Get": "Steps.MyTrainingStep.TrainingJobName"}
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered(
        {"MyTrainingStep": [], "SecondTestStep": ["MyTrainingStep"], "TestStep": ["MyTrainingStep"]}
    )


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
    step_args.args["HyperParameters"]["sagemaker_program"] = {"Get": "Parameters.EntryPoint"}
    step_args.args["HyperParameters"]["sagemaker_submit_directory"] = {
        "Get": "Parameters.SourceDir"
    }
    assert json.loads(pipeline.definition())["Steps"][0] == {
        "Name": "MyTrainingStep",
        "Description": "TrainingStep description",
        "DisplayName": "MyTrainingStep",
        "Type": "Training",
        "Arguments": step_args.args,
    }


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

    step_args = step_args.args
    step_def = json.loads(pipeline.definition())["Steps"][0]

    assert step_args["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"] == training_input
    assert step_args["OutputDataConfig"]["S3OutputPath"] == output_path
    step_args["HyperParameters"]["sagemaker_program"] = {"Get": "Parameters.EntryPoint"}
    step_args["HyperParameters"]["sagemaker_submit_directory"] = {"Get": "Parameters.SourceDir"}

    del step_args["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"]
    del step_def["Arguments"]["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"]

    del step_args["OutputDataConfig"]["S3OutputPath"]
    del step_def["Arguments"]["OutputDataConfig"]["S3OutputPath"]

    if "sagemaker_s3_output" in step_args["HyperParameters"]:
        del step_args["HyperParameters"]["sagemaker_s3_output"]
        del step_def["Arguments"]["HyperParameters"]["sagemaker_s3_output"]

    assert step_def == {
        "Name": "MyTrainingStep",
        "Type": "Training",
        "Arguments": step_args,
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

    step_args = step_args.args

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
