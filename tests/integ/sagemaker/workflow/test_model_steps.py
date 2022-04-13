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

import logging
import os

import pytest
from botocore.exceptions import WaiterError

from sagemaker.tensorflow import TensorFlow, TensorFlowModel
from sagemaker.workflow.model_step import (
    ModelStep,
    _REGISTER_MODEL_NAME_BASE,
    _CREATE_MODEL_NAME_BASE,
)
from tests.integ.retry import retries
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker import (
    PipelineModel,
    TrainingInput,
    Model,
    ModelMetrics,
    MetricsSource,
    get_execution_role,
)
from sagemaker import FileSource, utils
from sagemaker.pytorch import PyTorch
from sagemaker.s3 import S3Uploader
from sagemaker.mxnet.model import MXNetModel
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.xgboost import XGBoost
from sagemaker.workflow.conditions import (
    ConditionGreaterThanOrEqualTo,
)
from tests.integ.kms_utils import get_or_create_kms_key
from tests.integ import DATA_DIR


_REGISTER_MODEL_TYPE = "RegisterModel"
_CREATE_MODEL_TYPE = "Model"


@pytest.fixture
def role(pipeline_session):
    return get_execution_role(pipeline_session)


@pytest.fixture
def pipeline_name():
    return utils.unique_name_from_base("my-pipeline-model-step")


def test_conditional_pytorch_training_model_registration(
    pipeline_session,
    role,
    cpu_instance_type,
    pipeline_name,
):
    base_dir = os.path.join(DATA_DIR, "pytorch_mnist")
    entry_point = os.path.join(base_dir, "mnist.py")
    input_path = pipeline_session.upload_data(
        path=os.path.join(base_dir, "training"),
        key_prefix="integ-test-data/pytorch_mnist/training",
    )
    inputs = TrainingInput(s3_data=input_path)

    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")
    good_enough_input = ParameterInteger(name="GoodEnoughInput", default_value=1)

    pytorch_estimator = PyTorch(
        entry_point=entry_point,
        role=role,
        framework_version="1.5.0",
        py_version="py3",
        instance_count=instance_count,
        instance_type=instance_type,
        sagemaker_session=pipeline_session,
    )
    train_step_args = pytorch_estimator.fit(inputs=inputs)
    step_train = TrainingStep(
        name="pytorch-train",
        step_args=train_step_args,
    )
    model = Model(
        image_uri=pytorch_estimator.training_image_uri(),
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
    )
    regis_model_step_args = model.register(
        content_types=["*"],
        response_types=["application/json"],
        inference_instances=["ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        description="test-description",
    )
    step_model_regis = ModelStep(
        name="pytorch-register-model",
        step_args=regis_model_step_args,
    )
    create_model_step_args = model.create(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium",
    )

    step_model_create = ModelStep(
        name="pytorch-model",
        step_args=create_model_step_args,
    )

    step_cond = ConditionStep(
        name="cond-good-enough",
        conditions=[
            ConditionGreaterThanOrEqualTo(left=good_enough_input, right=1),
        ],
        if_steps=[step_model_regis],
        else_steps=[step_model_create],
        depends_on=[step_train],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            good_enough_input,
            instance_count,
            instance_type,
        ],
        steps=[step_train, step_cond],
        sagemaker_session=pipeline_session,
    )
    try:
        pipeline.create(role)
        execution1 = pipeline.start(parameters={})
        execution2 = pipeline.start(parameters={"GoodEnoughInput": 0})
        try:
            execution1.wait(delay=30, max_attempts=60)
        except WaiterError:
            pass
        execution1_steps = execution1.list_steps()
        for step in execution1_steps:
            assert not step.get("FailureReason", None)
            assert step["StepStatus"] == "Succeeded"
            if _REGISTER_MODEL_NAME_BASE in step["StepName"]:
                assert step["Metadata"][_REGISTER_MODEL_TYPE]
        assert len(execution1_steps) == 3

        try:
            execution2.wait(delay=30, max_attempts=60)
        except WaiterError:
            pass
        execution2_steps = execution2.list_steps()
        for step in execution2_steps:
            assert not step.get("FailureReason", None)
            assert step["StepStatus"] == "Succeeded"
            if _CREATE_MODEL_NAME_BASE in step["StepName"]:
                assert step["Metadata"][_CREATE_MODEL_TYPE]
        assert len(execution2_steps) == 3
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_mxnet_model_registration(
    pipeline_session,
    role,
    cpu_instance_type,
    pipeline_name,
):
    base_dir = os.path.join(DATA_DIR, "mxnet_mnist")
    source_dir = os.path.join(base_dir, "code")
    entry_point = os.path.join(source_dir, "inference.py")
    mx_mnist_model_data = os.path.join(base_dir, "model.tar.gz")

    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")

    model = MXNetModel(
        entry_point=entry_point,
        source_dir=source_dir,
        role=role,
        model_data=mx_mnist_model_data,
        framework_version="1.7.0",
        py_version="py3",
        sagemaker_session=pipeline_session,
    )

    step_args = model.register(
        content_types=["*"],
        response_types=["application/json"],
        model_package_group_name=f"{pipeline_name}TestModelPackageGroup",
        inference_instances=["ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        description="test-description",
    )
    step_model_regis = ModelStep(
        name="mxnet-register-model",
        step_args=step_args,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count, instance_type],
        steps=[step_model_regis],
        sagemaker_session=pipeline_session,
    )

    try:
        pipeline.create(role)

        for _ in retries(
            max_retry_count=5,
            exception_message_prefix="Waiting for a successful execution of pipeline",
            seconds_to_sleep=10,
        ):
            execution = pipeline.start()
            try:
                execution.wait(delay=30, max_attempts=60)
            except WaiterError:
                pass
            execution_steps = execution.list_steps()

            assert len(execution_steps) == 1
            failure_reason = execution_steps[0].get("FailureReason", "")
            if failure_reason != "":
                logging.error(
                    f"Pipeline execution failed with error: {failure_reason}." " Retrying.."
                )
                continue
            assert execution_steps[0]["StepStatus"] == "Succeeded"
            if _REGISTER_MODEL_NAME_BASE in execution_steps[0]["StepName"]:
                assert execution_steps[0]["Metadata"][_REGISTER_MODEL_TYPE]
            break

    finally:
        try:
            pipeline.delete()
        except Exception as error:
            logging.error(error)


def test_model_registration_with_drift_check_baselines(
    pipeline_session,
    role,
    pipeline_name,
):
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")

    # upload model data to s3
    model_local_path = os.path.join(DATA_DIR, "mxnet_mnist/model.tar.gz")
    model_base_uri = "s3://{}/{}/input/model/{}".format(
        pipeline_session.default_bucket(),
        "register_model_test_with_drift_baseline",
        utils.unique_name_from_base("model"),
    )
    model_uri = S3Uploader.upload(
        model_local_path, model_base_uri, sagemaker_session=pipeline_session
    )
    model_uri_param = ParameterString(name="model_uri", default_value=model_uri)

    # upload metrics to s3
    metrics_data = (
        '{"regression_metrics": {"mse": {"value": 4.925353410353891, '
        '"standard_deviation": 2.219186917819692}}}'
    )
    metrics_base_uri = "s3://{}/{}/input/metrics/{}".format(
        pipeline_session.default_bucket(),
        "register_model_test_with_drift_baseline",
        utils.unique_name_from_base("metrics"),
    )
    metrics_uri = S3Uploader.upload_string_as_file_body(
        body=metrics_data,
        desired_s3_uri=metrics_base_uri,
        sagemaker_session=pipeline_session,
    )
    metrics_uri_param = ParameterString(name="metrics_uri", default_value=metrics_uri)

    model_metrics = ModelMetrics(
        bias=MetricsSource(
            s3_uri=metrics_uri_param,
            content_type="application/json",
        ),
        explainability=MetricsSource(
            s3_uri=metrics_uri_param,
            content_type="application/json",
        ),
        bias_pre_training=MetricsSource(
            s3_uri=metrics_uri_param,
            content_type="application/json",
        ),
        bias_post_training=MetricsSource(
            s3_uri=metrics_uri_param,
            content_type="application/json",
        ),
    )
    drift_check_baselines = DriftCheckBaselines(
        model_statistics=MetricsSource(
            s3_uri=metrics_uri_param,
            content_type="application/json",
        ),
        model_constraints=MetricsSource(
            s3_uri=metrics_uri_param,
            content_type="application/json",
        ),
        model_data_statistics=MetricsSource(
            s3_uri=metrics_uri_param,
            content_type="application/json",
        ),
        model_data_constraints=MetricsSource(
            s3_uri=metrics_uri_param,
            content_type="application/json",
        ),
        bias_config_file=FileSource(
            s3_uri=metrics_uri_param,
            content_type="application/json",
        ),
        bias_pre_training_constraints=MetricsSource(
            s3_uri=metrics_uri_param,
            content_type="application/json",
        ),
        bias_post_training_constraints=MetricsSource(
            s3_uri=metrics_uri_param,
            content_type="application/json",
        ),
        explainability_constraints=MetricsSource(
            s3_uri=metrics_uri_param,
            content_type="application/json",
        ),
        explainability_config_file=FileSource(
            s3_uri=metrics_uri_param,
            content_type="application/json",
        ),
    )
    customer_metadata_properties = {"key1": "value1"}
    estimator = XGBoost(
        entry_point="training.py",
        source_dir=os.path.join(DATA_DIR, "sip"),
        instance_type=instance_type,
        instance_count=instance_count,
        framework_version="0.90-2",
        sagemaker_session=pipeline_session,
        py_version="py3",
        role=role,
    )
    model = Model(
        image_uri=estimator.training_image_uri(),
        model_data=model_uri_param,
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name="testModelPackageGroup",
        model_metrics=model_metrics,
        drift_check_baselines=drift_check_baselines,
        customer_metadata_properties=customer_metadata_properties,
    )
    step_model_register = ModelStep(
        name="MyRegisterModelStep",
        step_args=step_args,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            model_uri_param,
            metrics_uri_param,
            instance_type,
            instance_count,
        ],
        steps=[step_model_register],
        sagemaker_session=pipeline_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]

        for _ in retries(
            max_retry_count=5,
            exception_message_prefix="Waiting for a successful execution of pipeline",
            seconds_to_sleep=10,
        ):
            execution = pipeline.start(
                parameters={"model_uri": model_uri, "metrics_uri": metrics_uri}
            )
            response = execution.describe()

            assert response["PipelineArn"] == create_arn

            try:
                execution.wait(delay=30, max_attempts=60)
            except WaiterError:
                pass
            execution_steps = execution.list_steps()

            assert len(execution_steps) == 1
            failure_reason = execution_steps[0].get("FailureReason", "")
            if failure_reason != "":
                logging.error(
                    f"Pipeline execution failed with error: {failure_reason}." " Retrying.."
                )
                continue
            assert execution_steps[0]["StepStatus"] == "Succeeded"

            response = pipeline_session.sagemaker_client.describe_model_package(
                ModelPackageName=execution_steps[0]["Metadata"]["RegisterModel"]["Arn"]
            )

            assert (
                response["ModelMetrics"]["Explainability"]["Report"]["ContentType"]
                == "application/json"
            )
            assert (
                response["DriftCheckBaselines"]["Bias"]["PreTrainingConstraints"]["ContentType"]
                == "application/json"
            )
            assert (
                response["DriftCheckBaselines"]["Explainability"]["Constraints"]["ContentType"]
                == "application/json"
            )
            assert (
                response["DriftCheckBaselines"]["ModelQuality"]["Statistics"]["ContentType"]
                == "application/json"
            )
            assert (
                response["DriftCheckBaselines"]["ModelDataQuality"]["Statistics"]["ContentType"]
                == "application/json"
            )
            assert response["CustomerMetadataProperties"] == customer_metadata_properties
            break
    finally:
        try:
            pipeline.delete()
        except Exception as error:
            logging.error(error)


def test_model_registration_with_model_repack(
    pipeline_session,
    role,
    pipeline_name,
):
    kms_key = get_or_create_kms_key(pipeline_session, role)
    base_dir = os.path.join(DATA_DIR, "pytorch_mnist")
    entry_point = os.path.join(base_dir, "mnist.py")
    input_path = pipeline_session.upload_data(
        path=os.path.join(base_dir, "training"),
        key_prefix="integ-test-data/pytorch_mnist/training",
    )
    inputs = TrainingInput(s3_data=input_path)

    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")
    good_enough_input = ParameterInteger(name="GoodEnoughInput", default_value=1)

    pytorch_estimator = PyTorch(
        entry_point=entry_point,
        role=role,
        framework_version="1.5.0",
        py_version="py3",
        instance_count=instance_count,
        instance_type=instance_type,
        sagemaker_session=pipeline_session,
        output_kms_key=kms_key,
    )
    train_step_args = pytorch_estimator.fit(inputs=inputs)
    step_train = TrainingStep(
        name="pytorch-train",
        step_args=train_step_args,
    )
    model = Model(
        name="MyModel",
        image_uri=pytorch_estimator.training_image_uri(),
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
        entry_point=entry_point,
        source_dir=base_dir,
        model_kms_key=kms_key,
    )
    # register model with runtime repack
    regis_model_step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        description="test-description",
        model_package_group_name=f"{pipeline_name}TestModelPackageGroup",
    )
    step_model_regis = ModelStep(
        name="pytorch-register-model",
        step_args=regis_model_step_args,
    )

    # create model without runtime repack
    model.entry_point = None
    model.source_dir = None
    create_model_step_args = model.create(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium",
    )

    step_model_create = ModelStep(
        name="pytorch-model",
        step_args=create_model_step_args,
    )

    step_cond = ConditionStep(
        name="cond-good-enough",
        conditions=[ConditionGreaterThanOrEqualTo(left=good_enough_input, right=1)],
        if_steps=[step_model_regis],
        else_steps=[step_model_create],
        depends_on=[step_train],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[good_enough_input, instance_count, instance_type],
        steps=[step_train, step_cond],
        sagemaker_session=pipeline_session,
    )

    try:
        pipeline.create(role)
        execution1 = pipeline.start(parameters={})
        execution2 = pipeline.start(parameters={"GoodEnoughInput": 0})

        try:
            execution1.wait(delay=30, max_attempts=60)
        except WaiterError:
            pass
        execution1_steps = execution1.list_steps()
        for step in execution1_steps:
            assert not step.get("FailureReason", None)
            assert step["StepStatus"] == "Succeeded"
            if _REGISTER_MODEL_NAME_BASE in step["StepName"]:
                assert step["Metadata"][_REGISTER_MODEL_TYPE]
        assert len(execution1_steps) == 4

        try:
            execution2.wait(delay=30, max_attempts=60)
        except WaiterError:
            pass
        execution2_steps = execution2.list_steps()
        for step in execution2_steps:
            assert not step.get("FailureReason", None)
            assert step["StepStatus"] == "Succeeded"
            if _CREATE_MODEL_NAME_BASE in step["StepName"]:
                assert step["Metadata"][_CREATE_MODEL_TYPE]
        assert len(execution2_steps) == 3
    finally:
        try:
            pipeline.delete()
        except Exception as error:
            logging.error(error)


def test_model_registration_with_tensorflow_model_with_pipeline_model(
    pipeline_session, role, tf_full_version, tf_full_py_version, pipeline_name
):
    base_dir = os.path.join(DATA_DIR, "tensorflow_mnist")
    entry_point = os.path.join(base_dir, "mnist_v2.py")
    input_path = pipeline_session.upload_data(
        path=os.path.join(base_dir, "data"),
        key_prefix="integ-test-data/tf-scriptmode/mnist/training",
    )
    inputs = TrainingInput(s3_data=input_path)
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")

    tensorflow_estimator = TensorFlow(
        entry_point=entry_point,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        framework_version=tf_full_version,
        py_version=tf_full_py_version,
        sagemaker_session=pipeline_session,
    )
    train_step_args = tensorflow_estimator.fit(inputs=inputs)
    step_train = TrainingStep(
        name="MyTrain",
        step_args=train_step_args,
    )
    model = TensorFlowModel(
        entry_point=entry_point,
        framework_version="2.4",
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        sagemaker_session=pipeline_session,
    )
    pipeline_model = PipelineModel(
        name="MyModelPipeline", models=[model], role=role, sagemaker_session=pipeline_session
    )
    step_args = pipeline_model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=f"{pipeline_name}TestModelPackageGroup",
    )
    step_register_model = ModelStep(
        name="MyModelStep",
        step_args=step_args,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            instance_count,
            instance_type,
        ],
        steps=[step_train, step_register_model],
        sagemaker_session=pipeline_session,
    )

    try:
        pipeline.create(role)
        execution = pipeline.start(parameters={})
        try:
            execution.wait(delay=30, max_attempts=60)
        except WaiterError:
            pass
        execution_steps = execution.list_steps()

        for step in execution_steps:
            assert not step.get("FailureReason", None)
            assert step["StepStatus"] == "Succeeded"
            if _REGISTER_MODEL_NAME_BASE in step["StepName"]:
                assert step["Metadata"][_REGISTER_MODEL_TYPE]
        assert len(execution_steps) == 3
    finally:
        try:
            pipeline.delete()
        except Exception as error:
            logging.error(error)
