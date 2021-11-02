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

import pytest
import sagemaker
import os

from mock import (
    Mock,
    PropertyMock,
    patch,
)

from sagemaker.debugger import DEBUGGER_FLAG, ProfilerConfig
from sagemaker.estimator import Estimator
from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import TrainingInput, TransformInput, CreateModelInput, CompilationInput
from sagemaker.model import Model
from sagemaker.processing import (
    Processor,
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.tuner import (
    HyperparameterTuner,
    ContinuousParameter,
    WarmStartConfig,
    WarmStartTypes,
)
from sagemaker.network import NetworkConfig
from sagemaker.transformer import Transformer
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.workflow.retry import (
    StepRetryPolicy,
    StepExceptionTypeEnum,
    SageMakerJobStepRetryPolicy,
    SageMakerJobExceptionTypeEnum,
)
from sagemaker.workflow.steps import (
    ProcessingStep,
    CompilationStep,
    ConfigurableRetryStep,
    StepTypeEnum,
    TrainingStep,
    TuningStep,
    TransformStep,
    CreateModelStep,
    CacheConfig,
)
from tests.unit import DATA_DIR

SCRIPT_FILE = "dummy_script.py"
SCRIPT_PATH = os.path.join(DATA_DIR, SCRIPT_FILE)

REGION = "us-west-2"
BUCKET = "my-bucket"
IMAGE_URI = "fakeimage"
ROLE = "DummyRole"
MODEL_NAME = "gisele"


class CustomStep(ConfigurableRetryStep):
    def __init__(self, name, display_name=None, description=None, retry_policies=None):
        super(CustomStep, self).__init__(
            name, StepTypeEnum.TRAINING, display_name, description, None, retry_policies
        )
        self._properties = Properties(path=f"Steps.{name}")

    @property
    def arguments(self):
        return dict()

    @property
    def properties(self):
        return self._properties


@pytest.fixture
def boto_session():
    role_mock = Mock()
    type(role_mock).arn = PropertyMock(return_value=ROLE)

    resource_mock = Mock()
    resource_mock.Role.return_value = role_mock

    session_mock = Mock(region_name=REGION)
    session_mock.resource.return_value = resource_mock

    return session_mock


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
def sagemaker_session(boto_session, client):
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=client,
        sagemaker_runtime_client=client,
        default_bucket=BUCKET,
    )


def test_custom_step():
    step = CustomStep(
        name="MyStep", display_name="CustomStepDisplayName", description="CustomStepDescription"
    )
    assert step.to_request() == {
        "Name": "MyStep",
        "DisplayName": "CustomStepDisplayName",
        "Description": "CustomStepDescription",
        "Type": "Training",
        "Arguments": dict(),
    }


def test_custom_step_without_display_name():
    step = CustomStep(name="MyStep", description="CustomStepDescription")
    assert step.to_request() == {
        "Name": "MyStep",
        "Description": "CustomStepDescription",
        "Type": "Training",
        "Arguments": dict(),
    }


def test_custom_step_without_description():
    step = CustomStep(name="MyStep", display_name="CustomStepDisplayName")
    assert step.to_request() == {
        "Name": "MyStep",
        "DisplayName": "CustomStepDisplayName",
        "Type": "Training",
        "Arguments": dict(),
    }


def test_custom_step_with_retry_policy():
    step = CustomStep(
        name="MyStep",
        retry_policies=[
            StepRetryPolicy(
                exception_types=[
                    StepExceptionTypeEnum.SERVICE_FAULT,
                    StepExceptionTypeEnum.THROTTLING,
                ],
                expire_after_mins=1,
            ),
            SageMakerJobStepRetryPolicy(
                exception_types=[SageMakerJobExceptionTypeEnum.CAPACITY_ERROR],
                max_attempts=3,
            ),
        ],
    )
    assert step.to_request() == {
        "Name": "MyStep",
        "Type": "Training",
        "RetryPolicies": [
            {
                "ExceptionType": ["Step.SERVICE_FAULT", "Step.THROTTLING"],
                "IntervalSeconds": 1,
                "BackoffRate": 2.0,
                "ExpireAfterMin": 1,
            },
            {
                "ExceptionType": ["SageMaker.CAPACITY_ERROR"],
                "IntervalSeconds": 1,
                "BackoffRate": 2.0,
                "MaxAttempts": 3,
            },
        ],
        "Arguments": dict(),
    }

    step.add_retry_policy(
        SageMakerJobStepRetryPolicy(
            exception_types=[SageMakerJobExceptionTypeEnum.INTERNAL_ERROR],
            interval_seconds=5,
            backoff_rate=2.0,
            expire_after_mins=5,
        )
    )
    assert step.to_request() == {
        "Name": "MyStep",
        "Type": "Training",
        "RetryPolicies": [
            {
                "ExceptionType": ["Step.SERVICE_FAULT", "Step.THROTTLING"],
                "IntervalSeconds": 1,
                "BackoffRate": 2.0,
                "ExpireAfterMin": 1,
            },
            {
                "ExceptionType": ["SageMaker.CAPACITY_ERROR"],
                "IntervalSeconds": 1,
                "BackoffRate": 2.0,
                "MaxAttempts": 3,
            },
            {
                "ExceptionType": ["SageMaker.JOB_INTERNAL_ERROR"],
                "IntervalSeconds": 5,
                "BackoffRate": 2.0,
                "ExpireAfterMin": 5,
            },
        ],
        "Arguments": dict(),
    }

    step = CustomStep(name="MyStep")
    assert step.to_request() == {
        "Name": "MyStep",
        "Type": "Training",
        "Arguments": dict(),
    }


def test_training_step_base_estimator(sagemaker_session):
    instance_type_parameter = ParameterString(name="InstanceType", default_value="c4.4xlarge")
    instance_count_parameter = ParameterInteger(name="InstanceCount", default_value=1)
    data_source_uri_parameter = ParameterString(
        name="DataSourceS3Uri", default_value=f"s3://{BUCKET}/train_manifest"
    )
    training_epochs_parameter = ParameterInteger(name="TrainingEpochs", default_value=5)
    training_batch_size_parameter = ParameterInteger(name="TrainingBatchSize", default_value=500)
    estimator = Estimator(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_count=instance_count_parameter,
        instance_type=instance_type_parameter,
        profiler_config=ProfilerConfig(system_monitor_interval_millis=500),
        hyperparameters={
            "batch-size": training_batch_size_parameter,
            "epochs": training_epochs_parameter,
        },
        rules=[],
        sagemaker_session=sagemaker_session,
    )
    inputs = TrainingInput(s3_data=data_source_uri_parameter)
    cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")
    step = TrainingStep(
        name="MyTrainingStep",
        depends_on=["TestStep"],
        description="TrainingStep description",
        display_name="MyTrainingStep",
        estimator=estimator,
        inputs=inputs,
        cache_config=cache_config,
    )
    step.add_depends_on(["AnotherTestStep"])
    assert step.to_request() == {
        "Name": "MyTrainingStep",
        "Type": "Training",
        "Description": "TrainingStep description",
        "DisplayName": "MyTrainingStep",
        "DependsOn": ["TestStep", "AnotherTestStep"],
        "Arguments": {
            "AlgorithmSpecification": {"TrainingImage": IMAGE_URI, "TrainingInputMode": "File"},
            "HyperParameters": {
                "batch-size": training_batch_size_parameter,
                "epochs": training_epochs_parameter,
            },
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataDistributionType": "FullyReplicated",
                            "S3DataType": "S3Prefix",
                            "S3Uri": data_source_uri_parameter,
                        }
                    },
                }
            ],
            "OutputDataConfig": {"S3OutputPath": f"s3://{BUCKET}/"},
            "ResourceConfig": {
                "InstanceCount": instance_count_parameter,
                "InstanceType": instance_type_parameter,
                "VolumeSizeInGB": 30,
            },
            "RoleArn": ROLE,
            "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
            "ProfilerConfig": {
                "ProfilingIntervalInMilliseconds": 500,
                "S3OutputPath": f"s3://{BUCKET}/",
            },
        },
        "CacheConfig": {"Enabled": True, "ExpireAfter": "PT1H"},
    }
    assert step.properties.TrainingJobName.expr == {"Get": "Steps.MyTrainingStep.TrainingJobName"}
    assert step.properties.HyperParameters.expr == {"Get": "Steps.MyTrainingStep.HyperParameters"}


def test_training_step_tensorflow(sagemaker_session):
    instance_type_parameter = ParameterString(name="InstanceType", default_value="ml.p3.16xlarge")
    instance_count_parameter = ParameterInteger(name="InstanceCount", default_value=1)
    data_source_uri_parameter = ParameterString(
        name="DataSourceS3Uri", default_value=f"s3://{BUCKET}/train_manifest"
    )
    training_epochs_parameter = ParameterInteger(name="TrainingEpochs", default_value=5)
    training_batch_size_parameter = ParameterInteger(name="TrainingBatchSize", default_value=500)
    estimator = TensorFlow(
        entry_point=os.path.join(DATA_DIR, SCRIPT_FILE),
        role=ROLE,
        model_dir=False,
        image_uri=IMAGE_URI,
        source_dir="s3://mybucket/source",
        framework_version="2.4.1",
        py_version="py37",
        instance_count=instance_count_parameter,
        instance_type=instance_type_parameter,
        sagemaker_session=sagemaker_session,
        # subnets=subnets,
        hyperparameters={
            "batch-size": training_batch_size_parameter,
            "epochs": training_epochs_parameter,
        },
        # security_group_ids=security_group_ids,
        debugger_hook_config=False,
        # Training using SMDataParallel Distributed Training Framework
        distribution={"smdistributed": {"dataparallel": {"enabled": True}}},
    )

    inputs = TrainingInput(s3_data=data_source_uri_parameter)
    cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")
    step = TrainingStep(
        name="MyTrainingStep", estimator=estimator, inputs=inputs, cache_config=cache_config
    )
    step_request = step.to_request()
    step_request["Arguments"]["HyperParameters"].pop("sagemaker_job_name", None)
    step_request["Arguments"]["HyperParameters"].pop("sagemaker_program", None)
    step_request["Arguments"].pop("ProfilerRuleConfigurations", None)
    assert step_request == {
        "Name": "MyTrainingStep",
        "Type": "Training",
        "Arguments": {
            "AlgorithmSpecification": {
                "TrainingInputMode": "File",
                "TrainingImage": "fakeimage",
                "EnableSageMakerMetricsTimeSeries": True,
            },
            "OutputDataConfig": {"S3OutputPath": "s3://my-bucket/"},
            "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
            "ResourceConfig": {
                "InstanceCount": instance_count_parameter,
                "InstanceType": instance_type_parameter,
                "VolumeSizeInGB": 30,
            },
            "RoleArn": "DummyRole",
            "InputDataConfig": [
                {
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": data_source_uri_parameter,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "ChannelName": "training",
                }
            ],
            "HyperParameters": {
                "batch-size": training_batch_size_parameter,
                "epochs": training_epochs_parameter,
                "sagemaker_submit_directory": '"s3://mybucket/source"',
                "sagemaker_container_log_level": "20",
                "sagemaker_region": '"us-west-2"',
                "sagemaker_distributed_dataparallel_enabled": "true",
                "sagemaker_instance_type": instance_type_parameter,
                "sagemaker_distributed_dataparallel_custom_mpi_options": '""',
            },
            "ProfilerConfig": {"S3OutputPath": "s3://my-bucket/"},
            "Environment": {DEBUGGER_FLAG: "0"},
        },
        "CacheConfig": {"Enabled": True, "ExpireAfter": "PT1H"},
    }
    assert step.properties.TrainingJobName.expr == {"Get": "Steps.MyTrainingStep.TrainingJobName"}


def test_processing_step(sagemaker_session):
    processing_input_data_uri_parameter = ParameterString(
        name="ProcessingInputDataUri", default_value=f"s3://{BUCKET}/processing_manifest"
    )
    instance_type_parameter = ParameterString(name="InstanceType", default_value="ml.m4.4xlarge")
    instance_count_parameter = ParameterInteger(name="InstanceCount", default_value=1)
    processor = Processor(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_count=instance_count_parameter,
        instance_type=instance_type_parameter,
        sagemaker_session=sagemaker_session,
    )
    inputs = [
        ProcessingInput(
            source=processing_input_data_uri_parameter,
            destination="processing_manifest",
        )
    ]
    cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")
    step = ProcessingStep(
        name="MyProcessingStep",
        description="ProcessingStep description",
        display_name="MyProcessingStep",
        depends_on=["TestStep", "SecondTestStep"],
        processor=processor,
        inputs=inputs,
        outputs=[],
        cache_config=cache_config,
    )
    step.add_depends_on(["ThirdTestStep"])
    assert step.to_request() == {
        "Name": "MyProcessingStep",
        "Description": "ProcessingStep description",
        "DisplayName": "MyProcessingStep",
        "Type": "Processing",
        "DependsOn": ["TestStep", "SecondTestStep", "ThirdTestStep"],
        "Arguments": {
            "AppSpecification": {"ImageUri": "fakeimage"},
            "ProcessingInputs": [
                {
                    "InputName": "input-1",
                    "AppManaged": False,
                    "S3Input": {
                        "LocalPath": "processing_manifest",
                        "S3CompressionType": "None",
                        "S3DataDistributionType": "FullyReplicated",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                        "S3Uri": processing_input_data_uri_parameter,
                    },
                }
            ],
            "ProcessingResources": {
                "ClusterConfig": {
                    "InstanceCount": instance_count_parameter,
                    "InstanceType": instance_type_parameter,
                    "VolumeSizeInGB": 30,
                }
            },
            "RoleArn": "DummyRole",
        },
        "CacheConfig": {"Enabled": True, "ExpireAfter": "PT1H"},
    }
    assert step.properties.ProcessingJobName.expr == {
        "Get": "Steps.MyProcessingStep.ProcessingJobName"
    }


@patch("sagemaker.processing.ScriptProcessor._normalize_args")
def test_processing_step_normalizes_args(mock_normalize_args, sagemaker_session):
    processor = ScriptProcessor(
        role=ROLE,
        image_uri="012345678901.dkr.ecr.us-west-2.amazonaws.com/my-custom-image-uri",
        command=["python3"],
        instance_type="ml.m4.xlarge",
        instance_count=1,
        volume_size_in_gb=100,
        volume_kms_key="arn:aws:kms:us-west-2:012345678901:key/volume-kms-key",
        output_kms_key="arn:aws:kms:us-west-2:012345678901:key/output-kms-key",
        max_runtime_in_seconds=3600,
        base_job_name="my_sklearn_processor",
        env={"my_env_variable": "my_env_variable_value"},
        tags=[{"Key": "my-tag", "Value": "my-tag-value"}],
        network_config=NetworkConfig(
            subnets=["my_subnet_id"],
            security_group_ids=["my_security_group_id"],
            enable_network_isolation=True,
            encrypt_inter_container_traffic=True,
        ),
        sagemaker_session=sagemaker_session,
    )
    cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")
    inputs = [
        ProcessingInput(
            source=f"s3://{BUCKET}/processing_manifest",
            destination="processing_manifest",
        )
    ]
    outputs = [
        ProcessingOutput(
            source=f"s3://{BUCKET}/processing_manifest",
            destination="processing_manifest",
        )
    ]
    step = ProcessingStep(
        name="MyProcessingStep",
        processor=processor,
        code="foo.py",
        inputs=inputs,
        outputs=outputs,
        job_arguments=["arg1", "arg2"],
        cache_config=cache_config,
    )
    mock_normalize_args.return_value = [step.inputs, step.outputs]
    step.to_request()
    mock_normalize_args.assert_called_with(
        arguments=step.job_arguments,
        inputs=step.inputs,
        outputs=step.outputs,
        code=step.code,
    )


def test_create_model_step(sagemaker_session):
    model = Model(
        image_uri=IMAGE_URI,
        role=ROLE,
        sagemaker_session=sagemaker_session,
    )
    inputs = CreateModelInput(
        instance_type="c4.4xlarge",
        accelerator_type="ml.eia1.medium",
    )
    step = CreateModelStep(
        name="MyCreateModelStep",
        depends_on=["TestStep"],
        display_name="MyCreateModelStep",
        description="TestDescription",
        model=model,
        inputs=inputs,
    )
    step.add_depends_on(["SecondTestStep"])

    assert step.to_request() == {
        "Name": "MyCreateModelStep",
        "Type": "Model",
        "Description": "TestDescription",
        "DisplayName": "MyCreateModelStep",
        "DependsOn": ["TestStep", "SecondTestStep"],
        "Arguments": {
            "ExecutionRoleArn": "DummyRole",
            "PrimaryContainer": {"Environment": {}, "Image": "fakeimage"},
        },
    }
    assert step.properties.ModelName.expr == {"Get": "Steps.MyCreateModelStep.ModelName"}


def test_transform_step(sagemaker_session):
    transformer = Transformer(
        model_name=MODEL_NAME,
        instance_count=1,
        instance_type="c4.4xlarge",
        sagemaker_session=sagemaker_session,
    )
    inputs = TransformInput(data=f"s3://{BUCKET}/transform_manifest")
    cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")
    step = TransformStep(
        name="MyTransformStep",
        depends_on=["TestStep"],
        transformer=transformer,
        display_name="TransformStep",
        description="TestDescription",
        inputs=inputs,
        cache_config=cache_config,
    )
    step.add_depends_on(["SecondTestStep"])
    assert step.to_request() == {
        "Name": "MyTransformStep",
        "Type": "Transform",
        "Description": "TestDescription",
        "DisplayName": "TransformStep",
        "DependsOn": ["TestStep", "SecondTestStep"],
        "Arguments": {
            "ModelName": "gisele",
            "TransformInput": {
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://my-bucket/transform_manifest",
                    }
                }
            },
            "TransformOutput": {"S3OutputPath": None},
            "TransformResources": {
                "InstanceCount": 1,
                "InstanceType": "c4.4xlarge",
            },
        },
        "CacheConfig": {"Enabled": True, "ExpireAfter": "PT1H"},
    }
    assert step.properties.TransformJobName.expr == {
        "Get": "Steps.MyTransformStep.TransformJobName"
    }


def test_properties_describe_training_job_response():
    prop = Properties("Steps.MyStep", "DescribeTrainingJobResponse")
    some_prop_names = ["TrainingJobName", "TrainingJobArn", "HyperParameters", "OutputDataConfig"]
    for name in some_prop_names:
        assert name in prop.__dict__.keys()
    assert prop.CreationTime.expr == {"Get": "Steps.MyStep.CreationTime"}
    assert prop.OutputDataConfig.S3OutputPath.expr == {
        "Get": "Steps.MyStep.OutputDataConfig.S3OutputPath"
    }


def test_properties_describe_processing_job_response():
    prop = Properties("Steps.MyStep", "DescribeProcessingJobResponse")
    some_prop_names = ["ProcessingInputs", "ProcessingOutputConfig", "ProcessingEndTime"]
    for name in some_prop_names:
        assert name in prop.__dict__.keys()
    assert prop.ProcessingJobName.expr == {"Get": "Steps.MyStep.ProcessingJobName"}
    assert prop.ProcessingOutputConfig.Outputs["MyOutputName"].S3Output.S3Uri.expr == {
        "Get": "Steps.MyStep.ProcessingOutputConfig.Outputs['MyOutputName'].S3Output.S3Uri"
    }


def test_add_depends_on(sagemaker_session):
    processing_input_data_uri_parameter = ParameterString(
        name="ProcessingInputDataUri", default_value=f"s3://{BUCKET}/processing_manifest"
    )
    instance_type_parameter = ParameterString(name="InstanceType", default_value="ml.m4.4xlarge")
    instance_count_parameter = ParameterInteger(name="InstanceCount", default_value=1)
    processor = Processor(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_count=instance_count_parameter,
        instance_type=instance_type_parameter,
        sagemaker_session=sagemaker_session,
    )
    inputs = [
        ProcessingInput(
            source=processing_input_data_uri_parameter,
            destination="processing_manifest",
        )
    ]
    cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")

    step_1 = ProcessingStep(
        name="MyProcessingStep-1",
        processor=processor,
        inputs=inputs,
        outputs=[],
        cache_config=cache_config,
    )

    step_2 = ProcessingStep(
        name="MyProcessingStep-2",
        depends_on=[step_1],
        processor=processor,
        inputs=inputs,
        outputs=[],
        cache_config=cache_config,
    )

    step_3 = ProcessingStep(
        name="MyProcessingStep-3",
        depends_on=[step_1],
        processor=processor,
        inputs=inputs,
        outputs=[],
        cache_config=cache_config,
    )
    step_3.add_depends_on([step_2.name])

    assert "DependsOn" not in step_1.to_request()
    assert step_2.to_request()["DependsOn"] == ["MyProcessingStep-1"]
    assert step_3.to_request()["DependsOn"] == ["MyProcessingStep-1", "MyProcessingStep-2"]


def test_single_algo_tuning_step(sagemaker_session):
    data_source_uri_parameter = ParameterString(
        name="DataSourceS3Uri", default_value=f"s3://{BUCKET}/train_manifest"
    )
    estimator = Estimator(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_count=1,
        instance_type="ml.c5.4xlarge",
        profiler_config=ProfilerConfig(system_monitor_interval_millis=500),
        rules=[],
        sagemaker_session=sagemaker_session,
    )
    estimator.set_hyperparameters(
        num_layers=18,
        image_shape="3,224,224",
        num_classes=257,
        num_training_samples=15420,
        mini_batch_size=128,
        epochs=10,
        optimizer="sgd",
        top_k="2",
        precision_dtype="float32",
        augmentation_type="crop",
    )

    hyperparameter_ranges = {
        "learning_rate": ContinuousParameter(0.0001, 0.05),
        "momentum": ContinuousParameter(0.0, 0.99),
        "weight_decay": ContinuousParameter(0.0, 0.99),
    }

    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name="val:accuracy",
        hyperparameter_ranges=hyperparameter_ranges,
        objective_type="Maximize",
        max_jobs=5,
        max_parallel_jobs=2,
        early_stopping_type="OFF",
        strategy="Bayesian",
        warm_start_config=WarmStartConfig(
            warm_start_type=WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM,
            parents=set(["parent-hpo"]),
        ),
    )

    inputs = TrainingInput(s3_data=data_source_uri_parameter)

    tuning_step = TuningStep(
        name="MyTuningStep",
        tuner=tuner,
        inputs=inputs,
    )

    assert tuning_step.to_request() == {
        "Name": "MyTuningStep",
        "Type": "Tuning",
        "Arguments": {
            "HyperParameterTuningJobConfig": {
                "Strategy": "Bayesian",
                "ResourceLimits": {"MaxNumberOfTrainingJobs": 5, "MaxParallelTrainingJobs": 2},
                "TrainingJobEarlyStoppingType": "OFF",
                "HyperParameterTuningJobObjective": {
                    "Type": "Maximize",
                    "MetricName": "val:accuracy",
                },
                "ParameterRanges": {
                    "ContinuousParameterRanges": [
                        {
                            "Name": "learning_rate",
                            "MinValue": "0.0001",
                            "MaxValue": "0.05",
                            "ScalingType": "Auto",
                        },
                        {
                            "Name": "momentum",
                            "MinValue": "0.0",
                            "MaxValue": "0.99",
                            "ScalingType": "Auto",
                        },
                        {
                            "Name": "weight_decay",
                            "MinValue": "0.0",
                            "MaxValue": "0.99",
                            "ScalingType": "Auto",
                        },
                    ],
                    "CategoricalParameterRanges": [],
                    "IntegerParameterRanges": [],
                },
            },
            "TrainingJobDefinition": {
                "StaticHyperParameters": {
                    "num_layers": "18",
                    "image_shape": "3,224,224",
                    "num_classes": "257",
                    "num_training_samples": "15420",
                    "mini_batch_size": "128",
                    "epochs": "10",
                    "optimizer": "sgd",
                    "top_k": "2",
                    "precision_dtype": "float32",
                    "augmentation_type": "crop",
                },
                "RoleArn": "DummyRole",
                "OutputDataConfig": {"S3OutputPath": "s3://my-bucket/"},
                "ResourceConfig": {
                    "InstanceCount": 1,
                    "InstanceType": "ml.c5.4xlarge",
                    "VolumeSizeInGB": 30,
                },
                "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
                "AlgorithmSpecification": {
                    "TrainingInputMode": "File",
                    "TrainingImage": "fakeimage",
                },
                "InputDataConfig": [
                    {
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": data_source_uri_parameter,
                                "S3DataDistributionType": "FullyReplicated",
                            }
                        },
                        "ChannelName": "training",
                    }
                ],
            },
            "WarmStartConfig": {
                "WarmStartType": "IdenticalDataAndAlgorithm",
                "ParentHyperParameterTuningJobs": [
                    {
                        "HyperParameterTuningJobName": "parent-hpo",
                    }
                ],
            },
        },
    }

    assert tuning_step.properties.HyperParameterTuningJobName.expr == {
        "Get": "Steps.MyTuningStep.HyperParameterTuningJobName"
    }
    assert tuning_step.properties.TrainingJobSummaries[0].TrainingJobName.expr == {
        "Get": "Steps.MyTuningStep.TrainingJobSummaries[0].TrainingJobName"
    }
    assert tuning_step.get_top_model_s3_uri(0, "my-bucket", "my-prefix").expr == {
        "Std:Join": {
            "On": "/",
            "Values": [
                "s3:/",
                "my-bucket",
                "my-prefix",
                {"Get": "Steps.MyTuningStep.TrainingJobSummaries[0].TrainingJobName"},
                "output/model.tar.gz",
            ],
        }
    }


def test_multi_algo_tuning_step(sagemaker_session):
    data_source_uri_parameter = ParameterString(
        name="DataSourceS3Uri", default_value=f"s3://{BUCKET}/train_manifest"
    )
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    estimator = Estimator(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_count=instance_count,
        instance_type="ml.c5.4xlarge",
        profiler_config=ProfilerConfig(system_monitor_interval_millis=500),
        rules=[],
        sagemaker_session=sagemaker_session,
        max_retry_attempts=10,
    )

    estimator.set_hyperparameters(
        num_layers=18,
        image_shape="3,224,224",
        num_classes=257,
        num_training_samples=15420,
        mini_batch_size=128,
        epochs=10,
        optimizer="sgd",
        top_k="2",
        precision_dtype="float32",
        augmentation_type="crop",
    )

    initial_lr_param = ParameterString(name="InitialLR", default_value="0.0001")
    hyperparameter_ranges = {
        "learning_rate": ContinuousParameter(initial_lr_param, 0.05),
        "momentum": ContinuousParameter(0.0, 0.99),
        "weight_decay": ContinuousParameter(0.0, 0.99),
    }

    tuner = HyperparameterTuner.create(
        estimator_dict={
            "estimator-1": estimator,
            "estimator-2": estimator,
        },
        objective_type="Minimize",
        objective_metric_name_dict={
            "estimator-1": "val:loss",
            "estimator-2": "val:loss",
        },
        hyperparameter_ranges_dict={
            "estimator-1": hyperparameter_ranges,
            "estimator-2": hyperparameter_ranges,
        },
    )

    inputs = TrainingInput(s3_data=data_source_uri_parameter)

    tuning_step = TuningStep(
        name="MyTuningStep",
        tuner=tuner,
        inputs={
            "estimator-1": inputs,
            "estimator-2": inputs,
        },
    )

    assert tuning_step.to_request() == {
        "Name": "MyTuningStep",
        "Type": "Tuning",
        "Arguments": {
            "HyperParameterTuningJobConfig": {
                "Strategy": "Bayesian",
                "ResourceLimits": {"MaxNumberOfTrainingJobs": 1, "MaxParallelTrainingJobs": 1},
                "TrainingJobEarlyStoppingType": "Off",
            },
            "TrainingJobDefinitions": [
                {
                    "StaticHyperParameters": {
                        "num_layers": "18",
                        "image_shape": "3,224,224",
                        "num_classes": "257",
                        "num_training_samples": "15420",
                        "mini_batch_size": "128",
                        "epochs": "10",
                        "optimizer": "sgd",
                        "top_k": "2",
                        "precision_dtype": "float32",
                        "augmentation_type": "crop",
                    },
                    "RoleArn": "DummyRole",
                    "OutputDataConfig": {"S3OutputPath": "s3://my-bucket/"},
                    "ResourceConfig": {
                        "InstanceCount": 1,
                        "InstanceType": "ml.c5.4xlarge",
                        "VolumeSizeInGB": 30,
                    },
                    "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
                    "AlgorithmSpecification": {
                        "TrainingInputMode": "File",
                        "TrainingImage": "fakeimage",
                    },
                    "InputDataConfig": [
                        {
                            "DataSource": {
                                "S3DataSource": {
                                    "S3DataType": "S3Prefix",
                                    "S3Uri": data_source_uri_parameter,
                                    "S3DataDistributionType": "FullyReplicated",
                                }
                            },
                            "ChannelName": "training",
                        }
                    ],
                    "DefinitionName": "estimator-1",
                    "TuningObjective": {"Type": "Minimize", "MetricName": "val:loss"},
                    "HyperParameterRanges": {
                        "ContinuousParameterRanges": [
                            {
                                "Name": "learning_rate",
                                "MinValue": initial_lr_param,
                                "MaxValue": "0.05",
                                "ScalingType": "Auto",
                            },
                            {
                                "Name": "momentum",
                                "MinValue": "0.0",
                                "MaxValue": "0.99",
                                "ScalingType": "Auto",
                            },
                            {
                                "Name": "weight_decay",
                                "MinValue": "0.0",
                                "MaxValue": "0.99",
                                "ScalingType": "Auto",
                            },
                        ],
                        "CategoricalParameterRanges": [],
                        "IntegerParameterRanges": [],
                    },
                    "RetryStrategy": {
                        "MaximumRetryAttempts": 10,
                    },
                },
                {
                    "StaticHyperParameters": {
                        "num_layers": "18",
                        "image_shape": "3,224,224",
                        "num_classes": "257",
                        "num_training_samples": "15420",
                        "mini_batch_size": "128",
                        "epochs": "10",
                        "optimizer": "sgd",
                        "top_k": "2",
                        "precision_dtype": "float32",
                        "augmentation_type": "crop",
                    },
                    "RoleArn": "DummyRole",
                    "OutputDataConfig": {"S3OutputPath": "s3://my-bucket/"},
                    "ResourceConfig": {
                        "InstanceCount": 1,
                        "InstanceType": "ml.c5.4xlarge",
                        "VolumeSizeInGB": 30,
                    },
                    "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
                    "AlgorithmSpecification": {
                        "TrainingInputMode": "File",
                        "TrainingImage": "fakeimage",
                    },
                    "InputDataConfig": [
                        {
                            "DataSource": {
                                "S3DataSource": {
                                    "S3DataType": "S3Prefix",
                                    "S3Uri": data_source_uri_parameter,
                                    "S3DataDistributionType": "FullyReplicated",
                                }
                            },
                            "ChannelName": "training",
                        }
                    ],
                    "DefinitionName": "estimator-2",
                    "TuningObjective": {"Type": "Minimize", "MetricName": "val:loss"},
                    "HyperParameterRanges": {
                        "ContinuousParameterRanges": [
                            {
                                "Name": "learning_rate",
                                "MinValue": initial_lr_param,
                                "MaxValue": "0.05",
                                "ScalingType": "Auto",
                            },
                            {
                                "Name": "momentum",
                                "MinValue": "0.0",
                                "MaxValue": "0.99",
                                "ScalingType": "Auto",
                            },
                            {
                                "Name": "weight_decay",
                                "MinValue": "0.0",
                                "MaxValue": "0.99",
                                "ScalingType": "Auto",
                            },
                        ],
                        "CategoricalParameterRanges": [],
                        "IntegerParameterRanges": [],
                    },
                    "RetryStrategy": {
                        "MaximumRetryAttempts": 10,
                    },
                },
            ],
        },
    }


def test_compilation_step(sagemaker_session):
    estimator = Estimator(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_count=1,
        instance_type="ml.c5.4xlarge",
        profiler_config=ProfilerConfig(system_monitor_interval_millis=500),
        rules=[],
        sagemaker_session=sagemaker_session,
    )

    model = Model(
        image_uri=IMAGE_URI,
        model_data="s3://output/tensorflow.tar.gz",
        sagemaker_session=sagemaker_session,
    )

    compilation_input = CompilationInput(
        target_instance_type="ml_inf",
        input_shape={"data": [1, 3, 1024, 1024]},
        output_path="s3://output",
        compile_max_run=100,
        framework="tensorflow",
        job_name="compile-model",
        compiler_options=None,
    )
    compilation_step = CompilationStep(
        name="MyCompilationStep", estimator=estimator, model=model, inputs=compilation_input
    )

    assert compilation_step.to_request() == {
        "Name": "MyCompilationStep",
        "Type": "Compilation",
        "Arguments": {
            "InputConfig": {
                "DataInputConfig": '{"data": [1, 3, 1024, 1024]}',
                "Framework": "TENSORFLOW",
                "S3Uri": "s3://output/tensorflow.tar.gz",
            },
            "OutputConfig": {"S3OutputLocation": "s3://output", "TargetDevice": "ml_inf"},
            "RoleArn": ROLE,
            "StoppingCondition": {"MaxRuntimeInSeconds": 100},
            "Tags": [],
        },
    }
