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

# TODO: This file should be removed once we completely deprecate the RegisterModel
# and deprecate the old usage of CreateModelStep (i.e. without step_args)
# Most of the tests in this file have been reproduced in
# `tests/integ/sagemaker/workflow/test_model_steps.py` etc.
# and the RegisterModel and CreateModelStep have been replaced with the new interface - ModelStep
from __future__ import absolute_import

import logging
import os
import re

import pytest

import tests
from tests.integ.sagemaker.workflow.helpers import wait_pipeline_execution
from sagemaker.tensorflow import TensorFlow, TensorFlowModel
from tests.integ.retry import retries
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker import (
    PipelineModel,
    TrainingInput,
    Model,
    ModelMetrics,
    MetricsSource,
)
from sagemaker import FileSource, utils
from sagemaker.inputs import CreateModelInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch import PyTorch
from sagemaker.s3 import S3Uploader
from sagemaker.sklearn import SKLearnModel, SKLearnProcessor
from sagemaker.mxnet.model import MXNetModel
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import CreateModelStep, ProcessingStep, TrainingStep
from sagemaker.xgboost import XGBoostModel
from sagemaker.xgboost import XGBoost
from sagemaker.workflow.conditions import (
    ConditionGreaterThanOrEqualTo,
    ConditionIn,
)
from tests.integ.kms_utils import get_or_create_kms_key
from tests.integ import DATA_DIR


@pytest.fixture
def pipeline_name():
    return utils.unique_name_from_base("my-pipeline-model-regis")


def test_conditional_pytorch_training_model_registration(
    sagemaker_session_for_pipeline,
    role,
    cpu_instance_type,
    pipeline_name,
    region_name,
):
    base_dir = os.path.join(DATA_DIR, "pytorch_mnist")
    entry_point = os.path.join(base_dir, "mnist.py")
    input_path = sagemaker_session_for_pipeline.upload_data(
        path=os.path.join(base_dir, "training"),
        key_prefix="integ-test-data/pytorch_mnist/training",
    )
    inputs = TrainingInput(s3_data=input_path)

    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = "ml.m5.xlarge"
    good_enough_input = ParameterInteger(name="GoodEnoughInput", default_value=1)
    in_condition_input = ParameterString(name="Foo", default_value="Foo")

    task = "IMAGE_CLASSIFICATION"
    sample_payload_url = "s3://test-bucket/model"
    framework = "TENSORFLOW"
    framework_version = "2.9"
    nearest_model_name = "resnet50"

    # If image_uri is not provided, the instance_type should not be a pipeline variable
    # since instance_type is used to retrieve image_uri in compile time (PySDK)
    pytorch_estimator = PyTorch(
        entry_point=entry_point,
        role=role,
        framework_version="1.5.0",
        py_version="py3",
        instance_count=instance_count,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session_for_pipeline,
    )
    step_train = TrainingStep(
        name="pytorch-train",
        estimator=pytorch_estimator,
        inputs=inputs,
    )

    step_register = RegisterModel(
        name="pytorch-register-model",
        estimator=pytorch_estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["*"],
        response_types=["*"],
        inference_instances=["ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        description="test-description",
        sample_payload_url=sample_payload_url,
        task=task,
        framework=framework,
        framework_version=framework_version,
        nearest_model_name=nearest_model_name,
    )

    model = Model(
        image_uri=pytorch_estimator.training_image_uri(),
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=sagemaker_session_for_pipeline,
        role=role,
    )
    model_inputs = CreateModelInput(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium",
    )
    step_model = CreateModelStep(
        name="pytorch-model",
        model=model,
        inputs=model_inputs,
    )

    step_cond = ConditionStep(
        name="cond-good-enough",
        conditions=[
            ConditionGreaterThanOrEqualTo(left=good_enough_input, right=1),
            ConditionIn(value=in_condition_input, in_values=["foo", "bar"]),
        ],
        if_steps=[step_register],
        else_steps=[step_model],
        depends_on=[step_train],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            in_condition_input,
            good_enough_input,
            instance_count,
        ],
        steps=[step_train, step_cond],
        sagemaker_session=sagemaker_session_for_pipeline,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

        execution = pipeline.start(parameters={})
        wait_pipeline_execution(execution=execution)
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )

        execution = pipeline.start(parameters={"GoodEnoughInput": 0})
        wait_pipeline_execution(execution=execution)
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_mxnet_model_registration(
    sagemaker_session_for_pipeline,
    role,
    cpu_instance_type,
    pipeline_name,
    region_name,
):
    base_dir = os.path.join(DATA_DIR, "mxnet_mnist")
    source_dir = os.path.join(base_dir, "code")
    entry_point = os.path.join(source_dir, "inference.py")
    mx_mnist_model_data = os.path.join(base_dir, "model.tar.gz")

    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")

    task = "IMAGE_CLASSIFICATION"
    sample_payload_url = "s3://test-bucket/model"
    framework = "TENSORFLOW"
    framework_version = "2.9"
    nearest_model_name = "resnet50"

    model = MXNetModel(
        entry_point=entry_point,
        source_dir=source_dir,
        role=role,
        model_data=mx_mnist_model_data,
        framework_version="1.7.0",
        py_version="py3",
        sagemaker_session=sagemaker_session_for_pipeline,
    )

    step_register = RegisterModel(
        name="mxnet-register-model",
        model=model,
        content_types=["*"],
        response_types=["*"],
        inference_instances=["ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        description="test-description",
        sample_payload_url=sample_payload_url,
        task=task,
        framework=framework,
        framework_version=framework_version,
        nearest_model_name=nearest_model_name,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count, instance_type],
        steps=[step_register],
        sagemaker_session=sagemaker_session_for_pipeline,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

        execution = pipeline.start(parameters={})
        wait_pipeline_execution(execution=execution)
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )

        execution = pipeline.start()
        wait_pipeline_execution(execution=execution)
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_sklearn_xgboost_sip_model_registration(
    sagemaker_session_for_pipeline, role, pipeline_name, region_name
):
    prefix = "sip"
    bucket_name = sagemaker_session_for_pipeline.default_bucket()
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = "ml.m5.xlarge"

    task = "IMAGE_CLASSIFICATION"
    sample_payload_url = "s3://test-bucket/model"
    framework = "TENSORFLOW"
    framework_version = "2.9"
    nearest_model_name = "resnet50"

    # The instance_type should not be a pipeline variable
    # since it is used to retrieve image_uri in compile time (PySDK)
    sklearn_processor = SKLearnProcessor(
        role=role,
        instance_type=instance_type,
        instance_count=instance_count,
        framework_version="0.20.0",
        sagemaker_session=sagemaker_session_for_pipeline,
    )

    # The path to the raw data.
    raw_data_path = "s3://{0}/{1}/data/raw/".format(bucket_name, prefix)
    raw_data_path_param = ParameterString(name="raw_data_path", default_value=raw_data_path)

    # The output path to the training data.
    train_data_path = "s3://{0}/{1}/data/preprocessed/train/".format(bucket_name, prefix)
    train_data_path_param = ParameterString(name="train_data_path", default_value=train_data_path)

    # The output path to the validation data.
    val_data_path = "s3://{0}/{1}/data/preprocessed/val/".format(bucket_name, prefix)
    val_data_path_param = ParameterString(name="val_data_path", default_value=val_data_path)

    # The training output path for the model.
    output_path = "s3://{0}/{1}/output/".format(bucket_name, prefix)
    output_path_param = ParameterString(name="output_path", default_value=output_path)

    # The output path to the featurizer model.
    model_path = "s3://{0}/{1}/output/sklearn/".format(bucket_name, prefix)
    model_path_param = ParameterString(name="model_path", default_value=model_path)

    inputs = [
        ProcessingInput(
            input_name="raw_data",
            source=raw_data_path_param,
            destination="/opt/ml/processing/input",
        )
    ]

    outputs = [
        ProcessingOutput(
            output_name="train_data",
            source="/opt/ml/processing/train",
            destination=train_data_path_param,
        ),
        ProcessingOutput(
            output_name="val_data",
            source="/opt/ml/processing/val",
            destination=val_data_path_param,
        ),
        ProcessingOutput(
            output_name="model",
            source="/opt/ml/processing/model",
            destination=model_path_param,
        ),
    ]

    base_dir = os.path.join(DATA_DIR, "sip")
    code_path = os.path.join(base_dir, "preprocessor.py")

    processing_step = ProcessingStep(
        name="Processing",
        code=code_path,
        processor=sklearn_processor,
        inputs=inputs,
        outputs=outputs,
        job_arguments=["--train-test-split-ratio", "0.2"],
    )

    entry_point = "training.py"
    source_dir = base_dir
    code_location = "s3://{0}/{1}/code".format(bucket_name, prefix)

    # If image_uri is not provided, the instance_type should not be a pipeline variable
    # since instance_type is used to retrieve image_uri in compile time (PySDK)
    estimator = XGBoost(
        entry_point=entry_point,
        source_dir=source_dir,
        output_path=output_path_param,
        code_location=code_location,
        instance_type=instance_type,
        instance_count=instance_count,
        framework_version="0.90-2",
        sagemaker_session=sagemaker_session_for_pipeline,
        py_version="py3",
        role=role,
    )

    training_step = TrainingStep(
        name="Training",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                    "train_data"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                    "val_data"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    code_location = "s3://{0}/{1}/code".format(bucket_name, prefix)
    source_dir = os.path.join(base_dir, "sklearn_source_dir")

    sklearn_model = SKLearnModel(
        name="sklearn-model",
        model_data=processing_step.properties.ProcessingOutputConfig.Outputs[
            "model"
        ].S3Output.S3Uri,
        entry_point="inference.py",
        source_dir=source_dir,
        code_location=code_location,
        role=role,
        sagemaker_session=sagemaker_session_for_pipeline,
        framework_version="0.20.0",
        py_version="py3",
    )

    code_location = "s3://{0}/{1}/code".format(bucket_name, prefix)
    source_dir = os.path.join(base_dir, "xgboost_source_dir")

    xgboost_model = XGBoostModel(
        name="xgboost-model",
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        entry_point="inference.py",
        source_dir=source_dir,
        code_location=code_location,
        framework_version="0.90-2",
        py_version="py3",
        role=role,
        sagemaker_session=sagemaker_session_for_pipeline,
    )

    pipeline_model = PipelineModel(
        [xgboost_model, sklearn_model], role, sagemaker_session=sagemaker_session_for_pipeline
    )

    step_register = RegisterModel(
        name="AbaloneRegisterModel",
        model=pipeline_model,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name="windturbine",
        sample_payload_url=sample_payload_url,
        task=task,
        framework=framework,
        framework_version=framework_version,
        nearest_model_name=nearest_model_name,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            raw_data_path_param,
            train_data_path_param,
            val_data_path_param,
            model_path_param,
            instance_count,
            output_path_param,
        ],
        steps=[processing_step, training_step, step_register],
        sagemaker_session=sagemaker_session_for_pipeline,
    )

    try:
        response = pipeline.upsert(role_arn=role)
        create_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

        execution = pipeline.start(parameters={})
        wait_pipeline_execution(execution=execution)
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )

        execution = pipeline.start()
        wait_pipeline_execution(execution=execution)
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


@pytest.mark.skipif(
    tests.integ.test_region() not in tests.integ.DRIFT_CHECK_BASELINES_SUPPORTED_REGIONS,
    reason=(
        "DriftCheckBaselines changes are not fully deployed in" f" {tests.integ.test_region()}."
    ),
)
def test_model_registration_with_drift_check_baselines(
    sagemaker_session_for_pipeline,
    role,
    pipeline_name,
):
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = "ml.m5.xlarge"

    # upload model data to s3
    model_local_path = os.path.join(DATA_DIR, "mxnet_mnist/model.tar.gz")
    model_base_uri = "s3://{}/{}/input/model/{}".format(
        sagemaker_session_for_pipeline.default_bucket(),
        "register_model_test_with_drift_baseline",
        utils.unique_name_from_base("model"),
    )
    model_uri = S3Uploader.upload(
        model_local_path, model_base_uri, sagemaker_session=sagemaker_session_for_pipeline
    )
    model_uri_param = ParameterString(name="model_uri", default_value=model_uri)

    # upload metrics to s3
    metrics_data = (
        '{"regression_metrics": {"mse": {"value": 4.925353410353891, '
        '"standard_deviation": 2.219186917819692}}}'
    )
    metrics_base_uri = "s3://{}/{}/input/metrics/{}".format(
        sagemaker_session_for_pipeline.default_bucket(),
        "register_model_test_with_drift_baseline",
        utils.unique_name_from_base("metrics"),
    )
    metrics_uri = S3Uploader.upload_string_as_file_body(
        body=metrics_data,
        desired_s3_uri=metrics_base_uri,
        sagemaker_session=sagemaker_session_for_pipeline,
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
    domain = "COMPUTER_VISION"
    task = "IMAGE_CLASSIFICATION"
    sample_payload_url = "s3://test-bucket/model"
    framework = "TENSORFLOW"
    framework_version = "2.9"
    nearest_model_name = "resnet50"
    data_input_configuration = '{"input_1":[1,224,224,3]}'
    skip_model_validation = "All"

    # If image_uri is not provided, the instance_type should not be a pipeline variable
    # since instance_type is used to retrieve image_uri in compile time (PySDK)
    estimator = XGBoost(
        entry_point="training.py",
        source_dir=os.path.join(DATA_DIR, "sip"),
        instance_type=instance_type,
        instance_count=instance_count,
        framework_version="0.90-2",
        sagemaker_session=sagemaker_session_for_pipeline,
        py_version="py3",
        role=role,
    )

    step_register = RegisterModel(
        name="MyRegisterModelStep",
        estimator=estimator,
        model_data=model_uri_param,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name="testModelPackageGroup",
        model_metrics=model_metrics,
        drift_check_baselines=drift_check_baselines,
        customer_metadata_properties=customer_metadata_properties,
        domain=domain,
        sample_payload_url=sample_payload_url,
        task=task,
        framework=framework,
        framework_version=framework_version,
        nearest_model_name=nearest_model_name,
        data_input_configuration=data_input_configuration,
        skip_model_validation=skip_model_validation,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            model_uri_param,
            metrics_uri_param,
            instance_count,
        ],
        steps=[step_register],
        sagemaker_session=sagemaker_session_for_pipeline,
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
            assert execution_steps[0]["StepName"] == "MyRegisterModelStep-RegisterModel"

            response = sagemaker_session_for_pipeline.sagemaker_client.describe_model_package(
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
            assert response["Domain"] == domain
            assert response["Task"] == task
            assert response["SamplePayloadUrl"] == sample_payload_url
            assert response["SkipModelValidation"] == skip_model_validation
            break
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_model_registration_with_model_repack(
    sagemaker_session_for_pipeline,
    role,
    pipeline_name,
    region_name,
):
    kms_key = get_or_create_kms_key(sagemaker_session_for_pipeline, role)
    base_dir = os.path.join(DATA_DIR, "pytorch_mnist")
    entry_point = os.path.join(base_dir, "mnist.py")
    input_path = sagemaker_session_for_pipeline.upload_data(
        path=os.path.join(base_dir, "training"),
        key_prefix="integ-test-data/pytorch_mnist/training",
    )
    inputs = TrainingInput(s3_data=input_path)

    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = "ml.m5.xlarge"
    good_enough_input = ParameterInteger(name="GoodEnoughInput", default_value=1)

    # If image_uri is not provided, the instance_type should not be a pipeline variable
    # since instance_type is used to retrieve image_uri in compile time (PySDK)
    pytorch_estimator = PyTorch(
        entry_point=entry_point,
        role=role,
        framework_version="1.5.0",
        py_version="py3",
        instance_count=instance_count,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session_for_pipeline,
        output_kms_key=kms_key,
    )
    step_train = TrainingStep(
        name="pytorch-train",
        estimator=pytorch_estimator,
        inputs=inputs,
    )

    step_register = RegisterModel(
        name="pytorch-register-model",
        estimator=pytorch_estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        description="test-description",
        entry_point=entry_point,
        model_kms_key=kms_key,
    )

    model = Model(
        image_uri=pytorch_estimator.training_image_uri(),
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=sagemaker_session_for_pipeline,
        role=role,
    )
    model_inputs = CreateModelInput(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium",
    )
    step_model = CreateModelStep(
        name="pytorch-model",
        model=model,
        inputs=model_inputs,
    )

    step_cond = ConditionStep(
        name="cond-good-enough",
        conditions=[ConditionGreaterThanOrEqualTo(left=good_enough_input, right=1)],
        if_steps=[step_train, step_register],
        else_steps=[step_model],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[good_enough_input, instance_count],
        steps=[step_cond],
        sagemaker_session=sagemaker_session_for_pipeline,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

        execution = pipeline.start(parameters={})
        wait_pipeline_execution(execution=execution)
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )

        execution = pipeline.start(parameters={"GoodEnoughInput": 0})
        wait_pipeline_execution(execution=execution)
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_model_registration_with_tensorflow_model_with_pipeline_model(
    sagemaker_session_for_pipeline,
    role,
    tf_full_version,
    tf_full_py_version,
    pipeline_name,
    region_name,
):
    base_dir = os.path.join(DATA_DIR, "tensorflow_mnist")
    entry_point = os.path.join(base_dir, "mnist_v2.py")
    input_path = sagemaker_session_for_pipeline.upload_data(
        path=os.path.join(base_dir, "data"),
        key_prefix="integ-test-data/tf-scriptmode/mnist/training",
    )
    inputs = TrainingInput(s3_data=input_path)

    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = "ml.m5.xlarge"

    # If image_uri is not provided, the instance_type should not be a pipeline variable
    # since instance_type is used to retrieve image_uri in compile time (PySDK)
    tensorflow_estimator = TensorFlow(
        entry_point=entry_point,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        framework_version=tf_full_version,
        py_version=tf_full_py_version,
        sagemaker_session=sagemaker_session_for_pipeline,
    )
    step_train = TrainingStep(
        name="MyTrain",
        estimator=tensorflow_estimator,
        inputs=inputs,
    )

    model = TensorFlowModel(
        entry_point=entry_point,
        framework_version="2.4",
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        sagemaker_session=sagemaker_session_for_pipeline,
    )

    pipeline_model = PipelineModel(
        name="MyModelPipeline",
        models=[model],
        role=role,
        sagemaker_session=sagemaker_session_for_pipeline,
    )

    step_register_model = RegisterModel(
        name="MyRegisterModel",
        model=pipeline_model,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=f"{pipeline_name}TestModelPackageGroup",
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count],
        steps=[step_train, step_register_model],
        sagemaker_session=sagemaker_session_for_pipeline,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]

        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )

        execution = pipeline.start(parameters={})
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}/execution/",
            execution.arn,
        )
        wait_pipeline_execution(execution=execution)
        execution_steps = execution.list_steps()

        for step in execution_steps:
            assert not step.get("FailureReason", None)
            assert step["StepStatus"] == "Succeeded"
        assert len(execution_steps) == 3
    finally:
        try:
            pipeline.delete()
        except Exception as error:
            logging.error(error)
