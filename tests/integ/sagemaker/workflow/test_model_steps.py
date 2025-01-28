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

from packaging.version import Version
from packaging.specifiers import SpecifierSet

from tests.integ.sagemaker.workflow.helpers import wait_pipeline_execution
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join
from tests.integ.timeout import timeout_and_delete_endpoint_by_name
from sagemaker.tensorflow import TensorFlow, TensorFlowModel, TensorFlowPredictor
from sagemaker.utils import unique_name_from_base
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
    ModelPackage,
)
from sagemaker import FileSource, utils
from sagemaker.pytorch import PyTorch
from sagemaker.s3 import S3Uploader
from sagemaker.mxnet.model import MXNetModel
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.xgboost import XGBoost, XGBoostModel, XGBoostPredictor
from tests.integ.kms_utils import get_or_create_kms_key
from tests.integ import DATA_DIR


_REGISTER_MODEL_TYPE = "RegisterModel"
_CREATE_MODEL_TYPE = "Model"
_XGBOOST_PATH = os.path.join(DATA_DIR, "xgboost_abalone")
_XGBOOST_TEST_DATA = "6 1:3 2:0.37 3:0.29 4:0.095 5:0.249 6:0.1045 7:0.058 8:0.067"
_TENSORFLOW_PATH = os.path.join(DATA_DIR, "tfs/tfs-test-entrypoint-and-dependencies")
_TENSORFLOW_TEST_DATA = {"instances": [1.0, 2.0, 5.0]}


@pytest.fixture
def role(pipeline_session):
    return get_execution_role(pipeline_session)


@pytest.fixture
def pipeline_name():
    return utils.unique_name_from_base("my-pipeline-model-step")


def test_pytorch_training_model_registration_and_creation_without_custom_inference(
    pipeline_session,
    role,
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

    # If image_uri is not provided, the instance_type should not be a pipeline variable
    # since instance_type is used to retrieve image_uri in compile time (PySDK)
    pytorch_estimator = PyTorch(
        entry_point=entry_point,
        role=role,
        framework_version="1.5.0",
        py_version="py3",
        instance_count=instance_count,
        instance_type="ml.m5.xlarge",
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
        model_package_name="model-pkg-name-will-be-popped-out",
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
    # Use FailStep error_message to reference model step properties
    step_fail = FailStep(
        name="fail-step",
        error_message=Join(
            on=", ",
            values=[
                "Fail the execution on purpose to check model step properties",
                "register model",
                step_model_regis.properties.ModelPackageName,
                "create model",
                step_model_create.properties.ModelName,
            ],
        ),
    )
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count],
        steps=[step_train, step_model_regis, step_model_create, step_fail],
        sagemaker_session=pipeline_session,
    )
    try:
        pipeline.create(role)

        for _ in retries(
            max_retry_count=5,
            exception_message_prefix="Waiting for a successful execution of pipeline",
            seconds_to_sleep=10,
        ):
            execution = pipeline.start(parameters={})
            wait_pipeline_execution(execution=execution)
            execution_steps = execution.list_steps()
            is_execution_fail = False
            for step in execution_steps:
                if step["StepName"] == "fail-step":
                    assert step["StepStatus"] == "Failed"
                    assert "pytorch-register" in step["FailureReason"]
                    assert "pytorch-model" in step["FailureReason"]
                    continue
                failure_reason = step.get("FailureReason", "")
                if failure_reason != "":
                    logging.error(
                        f"Pipeline execution failed with error: {failure_reason}." " Retrying.."
                    )
                    is_execution_fail = True
                    break
                assert step["StepStatus"] == "Succeeded"
                if _REGISTER_MODEL_NAME_BASE in step["StepName"]:
                    assert step["Metadata"][_REGISTER_MODEL_TYPE]
                if _CREATE_MODEL_NAME_BASE in step["StepName"]:
                    assert step["Metadata"][_CREATE_MODEL_TYPE]
            if is_execution_fail:
                continue
            assert len(execution_steps) == 4
            break
    finally:
        try:
            pipeline.delete()
        except Exception as error:
            logging.error(error)


def test_pytorch_training_model_registration_and_creation_with_custom_inference(
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

    # If image_uri is not provided, the instance_type should not be a pipeline variable
    # since instance_type is used to retrieve image_uri in compile time (PySDK)
    pytorch_estimator = PyTorch(
        entry_point=entry_point,
        role=role,
        framework_version="1.5.0",
        py_version="py3",
        instance_count=instance_count,
        instance_type="ml.m5.xlarge",
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
    create_model_step_args = model.create(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium",
    )
    step_model_create = ModelStep(
        name="pytorch-model",
        step_args=create_model_step_args,
    )
    # Use FailStep error_message to reference model step properties
    step_fail = FailStep(
        name="fail-step",
        error_message=Join(
            on=", ",
            values=[
                "Fail the execution on purpose to check model step properties",
                "register model",
                step_model_regis.properties.ModelApprovalStatus,
                "create model",
                step_model_create.properties.ModelName,
            ],
        ),
    )
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count],
        steps=[step_train, step_model_regis, step_model_create, step_fail],
        sagemaker_session=pipeline_session,
    )

    try:
        pipeline.create(role)

        for _ in retries(
            max_retry_count=5,
            exception_message_prefix="Waiting for a successful execution of pipeline",
            seconds_to_sleep=10,
        ):
            execution = pipeline.start(parameters={})
            wait_pipeline_execution(execution=execution)
            execution_steps = execution.list_steps()
            is_execution_fail = False
            for step in execution_steps:
                if step["StepName"] == "fail-step":
                    assert step["StepStatus"] == "Failed"
                    assert "PendingManualApproval" in step["FailureReason"]
                    assert "pytorch-model" in step["FailureReason"]
                    continue
                failure_reason = step.get("FailureReason", "")
                if failure_reason != "":
                    logging.error(
                        f"Pipeline execution failed with error: {failure_reason}." " Retrying.."
                    )
                    is_execution_fail = True
                    break
                assert step["StepStatus"] == "Succeeded"
                if _REGISTER_MODEL_NAME_BASE in step["StepName"]:
                    assert step["Metadata"][_REGISTER_MODEL_TYPE]
                if _CREATE_MODEL_NAME_BASE in step["StepName"]:
                    assert step["Metadata"][_CREATE_MODEL_TYPE]
            if is_execution_fail:
                continue
            assert len(execution_steps) == 6
            break
    finally:
        try:
            pipeline.delete()
        except Exception as error:
            logging.error(error)


def test_mxnet_model_registration_with_custom_inference(
    pipeline_session,
    role,
    pipeline_name,
):
    base_dir = os.path.join(DATA_DIR, "mxnet_mnist")
    source_dir = os.path.join(base_dir, "code")
    entry_point = os.path.join(source_dir, "inference.py")
    mx_mnist_model_data = os.path.join(base_dir, "model.tar.gz")

    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")

    # No runtime repack needed as the model_data is not a PipelineVariable
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
            wait_pipeline_execution(execution=execution)
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


def test_model_registration_with_drift_check_baselines_and_model_metrics(
    pipeline_session,
    role,
    pipeline_name,
):
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)

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
    # If image_uri is not provided, the instance_type should not be a pipeline variable
    # since instance_type is used to retrieve image_uri in compile time (PySDK)
    estimator = XGBoost(
        entry_point="training.py",
        source_dir=os.path.join(DATA_DIR, "sip"),
        instance_type="ml.m5.xlarge",
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

            wait_pipeline_execution(execution=execution)
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


def test_model_registration_with_tensorflow_model_with_pipeline_model(
    pipeline_session, role, tf_full_version, tf_full_py_version, pipeline_name
):
    if Version(tf_full_version) >= Version("2.16"):
        pytest.skip(
            "This test is failing in TensorFlow 2.16 beacuse of an upstream bug: "
            "https://github.com/tensorflow/io/issues/2039"
        )
    base_dir = os.path.join(DATA_DIR, "tensorflow_mnist")
    entry_point = os.path.join(base_dir, "mnist_v2.py")
    input_path = pipeline_session.upload_data(
        path=os.path.join(base_dir, "data"),
        key_prefix="integ-test-data/tf-scriptmode/mnist/training",
    )
    inputs = TrainingInput(s3_data=input_path)
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    output_path = ParameterString(
        name="OutputPath", default_value=f"s3://{pipeline_session.default_bucket()}"
    )

    # If image_uri is not provided, the instance_type should not be a pipeline variable
    # since instance_type is used to retrieve image_uri in compile time (PySDK)
    tensorflow_estimator = TensorFlow(
        entry_point=entry_point,
        role=role,
        instance_count=instance_count,
        instance_type="ml.m5.xlarge",
        framework_version=tf_full_version,
        py_version=tf_full_py_version,
        sagemaker_session=pipeline_session,
        output_path=output_path,
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
        parameters=[instance_count, output_path],
        steps=[step_train, step_register_model],
        sagemaker_session=pipeline_session,
    )

    try:
        pipeline.create(role)

        for _ in retries(
            max_retry_count=5,
            exception_message_prefix="Waiting for a successful execution of pipeline",
            seconds_to_sleep=10,
        ):
            execution = pipeline.start(parameters={})
            wait_pipeline_execution(execution=execution)
            execution_steps = execution.list_steps()
            is_execution_fail = False
            for step in execution_steps:
                failure_reason = step.get("FailureReason", "")
                if failure_reason != "":
                    logging.error(
                        f"Pipeline execution failed with error: {failure_reason}." " Retrying.."
                    )
                    is_execution_fail = True
                    break
                assert step["StepStatus"] == "Succeeded"
                if _REGISTER_MODEL_NAME_BASE in step["StepName"]:
                    assert step["Metadata"][_REGISTER_MODEL_TYPE]
            if is_execution_fail:
                continue
            assert len(execution_steps) == 3
            break
    finally:
        try:
            pipeline.delete()
        except Exception as error:
            logging.error(error)


# E2E tests
@pytest.mark.skip(
    reason="""Skip this test as when running in parallel,
    it can lead to XGBoost endpoint conflicts
    and cause the test_inference_pipeline_model_deploy_and_update_endpoint to fail.
    Have created a backlog task for this and will remove the skip once issue resolved"""
)
def test_xgboost_model_register_and_deploy_with_runtime_repack(
    pipeline_session, sagemaker_session, role, pipeline_name
):
    # Assign entry_point and enable_network_isolation to the model
    # which will trigger a model runtime repacking and update the container env with
    # SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/model/code
    endpoint_name = unique_name_from_base("model-xgboost-integ")
    model_package_group_name = f"{pipeline_name}TestModelPackageGroup"
    xgb_model_data_s3 = pipeline_session.upload_data(
        path=os.path.join(_XGBOOST_PATH, "xgb_model.tar.gz"),
        key_prefix="integ-test-data/xgboost/model",
    )
    xgb_model_data_param = ParameterString(name="ModelData", default_value=xgb_model_data_s3)
    xgb_model = XGBoostModel(
        model_data=xgb_model_data_param,
        framework_version="1.3-1",
        role=role,
        sagemaker_session=pipeline_session,
        entry_point=os.path.join(_XGBOOST_PATH, "inference.py"),
        enable_network_isolation=True,
    )
    step_args = xgb_model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status="Approved",
    )
    step_model = ModelStep(
        name="MyModelStep",
        step_args=step_args,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[xgb_model_data_param],
        steps=[step_model],
        sagemaker_session=pipeline_session,
    )
    try:
        pipeline.create(role)

        for _ in retries(
            max_retry_count=5,
            exception_message_prefix="Waiting for a successful execution of pipeline",
            seconds_to_sleep=10,
        ):
            execution = pipeline.start(parameters={})
            wait_pipeline_execution(execution=execution)

            # Verify the pipeline execution succeeded
            step_register_model = None
            execution_steps = execution.list_steps()
            is_execution_fail = False
            for step in execution_steps:
                failure_reason = step.get("FailureReason", "")
                if failure_reason != "":
                    logging.error(
                        f"Pipeline execution failed with error: {failure_reason}." " Retrying.."
                    )
                    is_execution_fail = True
                    break
                assert step["StepStatus"] == "Succeeded"
                if _REGISTER_MODEL_NAME_BASE in step["StepName"]:
                    step_register_model = step
            if is_execution_fail:
                continue
            assert len(execution_steps) == 2
            assert step_register_model
            break

        # Verify the registered model can work as expected
        with timeout_and_delete_endpoint_by_name(
            endpoint_name=endpoint_name, sagemaker_session=pipeline_session
        ):
            model_pkg_arn = step_register_model["Metadata"][_REGISTER_MODEL_TYPE]["Arn"]
            response = pipeline_session.sagemaker_client.describe_model_package(
                ModelPackageName=model_pkg_arn
            )
            model_package = ModelPackage(
                role=role,
                model_data=response["InferenceSpecification"]["Containers"][0]["ModelDataUrl"],
                model_package_arn=model_pkg_arn,
                sagemaker_session=sagemaker_session,
            )
            model_package.deploy(
                initial_instance_count=1,
                instance_type="ml.m5.large",
                endpoint_name=endpoint_name,
            )
            predictor = XGBoostPredictor(
                endpoint_name=endpoint_name,
                sagemaker_session=pipeline_session,
            )
            _, *features = _XGBOOST_TEST_DATA.strip().split()
            test_data = " ".join(["-99"] + features)
            response = predictor.predict(test_data)
            assert len(response) == 1
            # extra data are appended by the inference.py
            assert len(response[0]) > 1
    finally:
        try:
            pipeline.delete()
            pipeline_session.sagemaker_client.delete_model_package(
                ModelPackageName=model_package.model_package_arn
            )
            pipeline_session.sagemaker_client.delete_model_package_group(
                ModelPackageGroupName=model_package_group_name
            )
        except Exception as error:
            logging.error(error)


def test_tensorflow_model_register_and_deploy_with_runtime_repack(
    pipeline_session, sagemaker_session, role, pipeline_name, tensorflow_inference_latest_version
):
    # Assign entry_point to the model
    # which will trigger a model runtime repacking
    endpoint_name = unique_name_from_base("model-tensorflow-integ")
    model_package_group_name = f"{pipeline_name}TestModelPackageGroup"
    tf_model_data_s3 = pipeline_session.upload_data(
        path=os.path.join(DATA_DIR, "tensorflow-serving-test-model.tar.gz"),
        key_prefix="integ-test-data/tensorflow/models",
    )
    tf_model_data_param = ParameterString(name="ModelData", default_value=tf_model_data_s3)
    tf_model = TensorFlowModel(
        model_data=tf_model_data_param,
        framework_version=tensorflow_inference_latest_version,
        role=role,
        sagemaker_session=pipeline_session,
        entry_point=os.path.join(_TENSORFLOW_PATH, "inference.py"),
        dependencies=[os.path.join(_TENSORFLOW_PATH, "dependency.py")],
        code_location=f"s3://{pipeline_session.default_bucket()}/model-code",
    )
    step_args = tf_model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status="Approved",
    )
    step_model = ModelStep(
        name="MyModelStep",
        step_args=step_args,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[tf_model_data_param],
        steps=[step_model],
        sagemaker_session=pipeline_session,
    )
    try:
        pipeline.create(role)

        for _ in retries(
            max_retry_count=5,
            exception_message_prefix="Waiting for a successful execution of pipeline",
            seconds_to_sleep=10,
        ):
            execution = pipeline.start(parameters={})
            wait_pipeline_execution(execution=execution)

            # Verify the pipeline execution succeeded
            step_register_model = None
            execution_steps = execution.list_steps()
            is_execution_fail = False
            for step in execution_steps:
                failure_reason = step.get("FailureReason", "")
                if failure_reason != "":
                    logging.error(
                        f"Pipeline execution failed with error: {failure_reason}." " Retrying.."
                    )
                    is_execution_fail = True
                    break
                assert step["StepStatus"] == "Succeeded"
                if _REGISTER_MODEL_NAME_BASE in step["StepName"]:
                    step_register_model = step
            if is_execution_fail:
                continue
            assert len(execution_steps) == 2
            assert step_register_model
            break

        # Verify the registered model can work as expected
        with timeout_and_delete_endpoint_by_name(
            endpoint_name=endpoint_name, sagemaker_session=pipeline_session
        ):
            model_pkg_arn = step_register_model["Metadata"][_REGISTER_MODEL_TYPE]["Arn"]
            response = pipeline_session.sagemaker_client.describe_model_package(
                ModelPackageName=model_pkg_arn
            )
            model_package = ModelPackage(
                role=role,
                model_data=response["InferenceSpecification"]["Containers"][0]["ModelDataUrl"],
                model_package_arn=model_pkg_arn,
                sagemaker_session=sagemaker_session,
            )
            model_package.deploy(
                initial_instance_count=1,
                instance_type="ml.m5.large",
                endpoint_name=endpoint_name,
            )
            predictor = TensorFlowPredictor(
                endpoint_name=endpoint_name,
                sagemaker_session=pipeline_session,
            )
            response = predictor.predict(_TENSORFLOW_TEST_DATA)
            assert response == {"predictions": [4.0, 4.5, 6.0]}
    finally:
        try:
            pipeline.delete()
            pipeline_session.sagemaker_client.delete_model_package(
                ModelPackageName=model_package.model_package_arn
            )
            pipeline_session.sagemaker_client.delete_model_package_group(
                ModelPackageGroupName=model_package_group_name
            )
        except Exception as error:
            logging.error(error)
