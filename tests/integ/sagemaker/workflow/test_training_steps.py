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
import uuid
import logging

import pytest

from packaging.version import Version
from packaging.specifiers import SpecifierSet

from tests.integ.sagemaker.workflow.helpers import wait_pipeline_execution
from sagemaker import TrainingInput, get_execution_role, utils, image_uris
from sagemaker.debugger import (
    DebuggerHookConfig,
    Rule,
    rule_configs,
)
from sagemaker.estimator import Estimator
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.tensorflow import TensorFlow
from sagemaker.utils import sagemaker_timestamp
from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from tests.integ.retry import retries
from tests.integ import DATA_DIR


@pytest.fixture
def role(sagemaker_session):
    return get_execution_role(sagemaker_session)


@pytest.fixture
def pipeline_name():
    return utils.unique_name_from_base("my-pipeline-training")


@pytest.fixture
def region_name(sagemaker_session):
    return sagemaker_session.boto_session.region_name


def test_training_job_with_debugger_and_profiler(
    sagemaker_session,
    pipeline_name,
    role,
    pytorch_training_latest_version,
    pytorch_training_latest_py_version,
):
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = "ml.m5.xlarge"

    rules = [
        Rule.sagemaker(rule_configs.vanishing_gradient()),
        Rule.sagemaker(base_config=rule_configs.all_zero(), rule_parameters={"tensor_regex": ".*"}),
        Rule.sagemaker(rule_configs.loss_not_decreasing()),
    ]
    debugger_hook_config = DebuggerHookConfig(
        s3_output_path=f"s3://{sagemaker_session.default_bucket()}/{uuid.uuid4()}/tensors"
    )

    base_dir = os.path.join(DATA_DIR, "pytorch_mnist")
    entry_point = "mnist.py"
    source_dir = sagemaker_session.upload_data(
        path=os.path.join(base_dir, "pytorch_mnist_source_code.tar.gz"),
        key_prefix="integ-test-data/pytorch_mnist/training",
    )
    entry_point_param = ParameterString(name="EntryPoint")
    source_dir_param = ParameterString(name="SourceDir")
    input_path = sagemaker_session.upload_data(
        path=os.path.join(base_dir, "training"),
        key_prefix="integ-test-data/pytorch_mnist/training",
    )
    inputs = TrainingInput(s3_data=input_path)

    # If image_uri is not provided, the instance_type should not be a pipeline variable
    # since instance_type is used to retrieve image_uri in compile time (PySDK)
    pytorch_estimator = PyTorch(
        entry_point=entry_point_param,
        source_dir=source_dir_param,
        role="SageMakerRole",
        framework_version=pytorch_training_latest_version,
        py_version=pytorch_training_latest_py_version,
        instance_count=instance_count,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        rules=rules,
        debugger_hook_config=debugger_hook_config,
        # TODO: remove base_job_name once we merge
        # https://github.com/aws/sagemaker-python-sdk/pull/3158/files
        base_job_name="TestJob",
    )

    step_train = TrainingStep(
        name="pytorch-train",
        estimator=pytorch_estimator,
        inputs=inputs,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count, entry_point_param, source_dir_param],
        steps=[step_train],
        sagemaker_session=sagemaker_session,
    )

    try:
        pipeline.create(role)
        execution_steps = _start_and_verify_execution_with_retry(
            pipeline=pipeline,
            parameters={"EntryPoint": entry_point, "SourceDir": source_dir},
        )
        training_job_arn = execution_steps[0]["Metadata"]["TrainingJob"]["Arn"]
        job_description = sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=training_job_arn.split("/")[1]
        )

        for index, rule in enumerate(rules):
            config = job_description["DebugRuleConfigurations"][index]
            assert config["RuleConfigurationName"] == rule.name
            assert config["RuleEvaluatorImage"] == rule.image_uri
            assert config["VolumeSizeInGB"] == 0
            assert (
                config["RuleParameters"]["rule_to_invoke"] == rule.rule_parameters["rule_to_invoke"]
            )
        assert job_description["DebugHookConfig"] == debugger_hook_config._to_request_dict()
        assert job_description["ProfilingStatus"] == "Enabled"
        assert job_description["ProfilerConfig"]["ProfilingIntervalInMilliseconds"] == 500
    finally:
        try:
            pipeline.delete()
        except Exception as error:
            logging.error(error)


def test_training_step_with_output_path_as_join(
    sagemaker_session, role, tf_full_version, tf_full_py_version, pipeline_name, region_name
):
    input_path = sagemaker_session.upload_data(
        path=os.path.join(DATA_DIR, "xgboost_abalone", "abalone"),
        key_prefix="integ-test-data/xgboost_abalone/abalone",
    )
    inputs = {"train": TrainingInput(s3_data=input_path)}

    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")
    entry_point1 = "dummy1"
    entry_point2 = "dummy2"
    src_base_dir = os.path.join(DATA_DIR, "xgboost_abalone/estimator_source_code")
    source_dir1 = sagemaker_session.upload_data(
        path=os.path.join(src_base_dir, "estimator_source_code_dummy1.tar.gz"),
        key_prefix="integ-test-data/estimator/training",
    )
    source_dir2 = sagemaker_session.upload_data(
        path=os.path.join(src_base_dir, "estimator_source_code_dummy2.tar.gz"),
        key_prefix="integ-test-data/estimator/training",
    )
    entry_point_param = ParameterString(name="EntryPoint")
    source_dir_param = ParameterString(name="SourceDir")
    output_path = Join(
        on="/", values=["s3:/", f"{sagemaker_session.default_bucket()}", f"{pipeline_name}Train"]
    )
    image_uri = image_uris.retrieve(
        framework="xgboost",
        region=sagemaker_session.boto_session.region_name,
        version="1.0-1",
        py_version="py3",
        instance_type="ml.m5.xlarge",
    )
    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        output_path=output_path,
        source_dir=source_dir_param,
        entry_point=entry_point_param,
        # TODO: remove base_job_name once we merge
        # https://github.com/aws/sagemaker-python-sdk/pull/3158/files
        base_job_name="TestJob",
    )
    estimator.set_hyperparameters(
        objective="reg:linear",
        num_round=50,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
    )
    step_train = TrainingStep(
        name="MyTrain",
        estimator=estimator,
        inputs=inputs,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count, instance_type, source_dir_param, entry_point_param],
        steps=[step_train],
        sagemaker_session=sagemaker_session,
    )

    try:
        pipeline.create(role)
        # execution1
        _start_and_verify_execution_with_retry(
            pipeline=pipeline,
            parameters={"EntryPoint": entry_point1, "SourceDir": source_dir1},
        )
        # execution2 updates parameters to different values
        _start_and_verify_execution_with_retry(
            pipeline=pipeline,
            parameters={"EntryPoint": entry_point2, "SourceDir": source_dir2},
        )
    finally:
        try:
            pipeline.delete()
        except Exception as error:
            logging.error(error)


def test_tensorflow_training_step_with_parameterized_code_input(
    pipeline_session, role, tf_full_version, tf_full_py_version, pipeline_name
):
    if Version(tf_full_version) in SpecifierSet("==2.16.*"):
        pytest.skip(
            "This test is failing in TensorFlow 2.16 beacuse of an upstream bug: "
            "https://github.com/tensorflow/io/issues/2039"
        )

    base_dir = os.path.join(DATA_DIR, "tensorflow_mnist")
    entry_point1 = "mnist_v2.py"
    entry_point2 = "mnist_dummy.py"
    source_dir1 = pipeline_session.upload_data(
        path=os.path.join(base_dir, "tensorflow_mnist_source_code.tar.gz"),
        key_prefix="integ-test-data/tf-scriptmode/mnist/training",
    )
    source_dir2 = pipeline_session.upload_data(
        path=os.path.join(base_dir, "tensorflow_mnist_source_code_dummy.tar.gz"),
        key_prefix="integ-test-data/tf-scriptmode/mnist/training",
    )
    entry_point_param = ParameterString(name="EntryPoint")
    source_dir_param = ParameterString(name="SourceDir")
    input_path = pipeline_session.upload_data(
        path=os.path.join(base_dir, "data"),
        key_prefix="integ-test-data/tf-scriptmode/mnist/training",
    )
    inputs = TrainingInput(s3_data=input_path)
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    output_path = ParameterString(
        name="OutputPath", default_value=f"s3://{pipeline_session.default_bucket()}"
    )
    checkpoint_s3_uri1 = "s3://{}/checkpoints/tf1-{}".format(
        pipeline_session.default_bucket(), sagemaker_timestamp()
    )
    checkpoint_s3_uri2 = "s3://{}/checkpoints/tf2-{}".format(
        pipeline_session.default_bucket(), sagemaker_timestamp()
    )
    checkpoint_s3_param = ParameterString(name="CheckpointS3Uri")

    # If image_uri is not provided, the instance_type should not be a pipeline variable
    # since instance_type is used to retrieve image_uri in compile time (PySDK)
    tensorflow_estimator = TensorFlow(
        entry_point=entry_point_param,
        source_dir=source_dir_param,
        role=role,
        instance_count=instance_count,
        instance_type="ml.m5.xlarge",
        framework_version=tf_full_version,
        py_version=tf_full_py_version,
        sagemaker_session=pipeline_session,
        output_path=output_path,
        checkpoint_s3_uri=checkpoint_s3_param,
    )
    # TODO: remove job_name once we merge
    # https://github.com/aws/sagemaker-python-sdk/pull/3158/files
    train_step_args = tensorflow_estimator.fit(inputs=inputs, job_name="TestJob")
    step_train = TrainingStep(
        name="MyTrain",
        step_args=train_step_args,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            instance_count,
            output_path,
            entry_point_param,
            source_dir_param,
            checkpoint_s3_param,
        ],
        steps=[step_train],
        sagemaker_session=pipeline_session,
    )

    try:
        pipeline.create(role)
        # execution1
        _start_and_verify_execution_with_retry(
            pipeline=pipeline,
            parameters={
                "EntryPoint": entry_point1,
                "SourceDir": source_dir1,
                "CheckpointS3Uri": checkpoint_s3_uri1,
            },
        )
        # execution2 updates parameters to different values
        _start_and_verify_execution_with_retry(
            pipeline=pipeline,
            parameters={
                "EntryPoint": entry_point2,
                "SourceDir": source_dir2,
                "CheckpointS3Uri": checkpoint_s3_uri2,
            },
        )
    finally:
        try:
            pipeline.delete()
        except Exception as error:
            logging.error(error)


def _start_and_verify_execution_with_retry(pipeline: Pipeline, parameters: dict) -> list:
    for _ in retries(
        max_retry_count=5,
        exception_message_prefix="Waiting for a successful execution of pipeline",
        seconds_to_sleep=10,
    ):
        execution = pipeline.start(parameters=parameters)
        wait_pipeline_execution(execution=execution)
        execution_steps = execution.list_steps()
        assert len(execution_steps) == 1
        failure_reason = execution_steps[0].get("FailureReason", "")
        if failure_reason != "":
            logging.error(f"Pipeline execution failed with error: {failure_reason}." " Retrying..")
            continue
        assert execution_steps[0]["StepStatus"] == "Succeeded"
        return execution_steps
