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

import json

import pytest
from sagemaker.automl.automl import AutoML, AutoMLInput
from sagemaker.exceptions import AutoMLStepInvalidModeError
from sagemaker.workflow import ParameterString

from sagemaker.workflow.automl_step import AutoMLStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from tests.unit.sagemaker.workflow.conftest import ROLE


def test_single_automl_step(pipeline_session):
    auto_ml = AutoML(
        role=ROLE,
        target_attribute_name="y",
        sagemaker_session=pipeline_session,
        volume_kms_key="volume-kms-key-id-string",
        vpc_config={"SecurityGroupIds": ["group"], "Subnets": ["subnet"]},
        encrypt_inter_container_traffic=False,
        compression_type="Gzip",
        output_kms_key="output-kms-key-id-string",
        output_path="s3://my_other_bucket/",
        problem_type="BinaryClassification",
        max_candidates=1,
        max_runtime_per_training_job_in_seconds=3600,
        total_job_runtime_in_seconds=36000,
        job_objective={"MetricName": "F1"},
        generate_candidate_definitions_only=False,
        tags=[{"Name": "some-tag", "Value": "value-for-tag"}],
        content_type="x-application/vnd.amazon+parquet",
        s3_data_type="ManifestFile",
        feature_specification_s3_uri="s3://bucket/features.json",
        validation_fraction=0.3,
        mode="ENSEMBLING",
        auto_generate_endpoint_name=False,
        endpoint_name="EndpointName",
        base_job_name="AutoMLJobPrefix",
    )
    input_training = AutoMLInput(
        inputs="s3://bucket/data",
        target_attribute_name="target",
        compression="Gzip",
        channel_type="training",
        sample_weight_attribute_name="sampleWeight",
    )
    input_validation = AutoMLInput(
        inputs="s3://bucket/validation_data",
        target_attribute_name="target",
        compression="Gzip",
        channel_type="validation",
        sample_weight_attribute_name="sampleWeight",
    )
    inputs = [input_training, input_validation]

    step_args = auto_ml.fit(
        inputs=inputs,
    )

    automl_step = AutoMLStep(
        name="MyAutoMLStep",
        step_args=step_args,
    )

    definition_config = PipelineDefinitionConfig(use_custom_job_prefix=True)

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[automl_step],
        sagemaker_session=pipeline_session,
        pipeline_definition_config=definition_config,
    )

    # AutoMLJobName trimmed to 8 char + timestamp :: "AutoMLJo-2023-06-23-22-57-39-083"
    step_dsl_list = json.loads(pipeline.definition())["Steps"]
    assert step_dsl_list[0] == {
        "Name": "MyAutoMLStep",
        "Type": "AutoML",
        "Arguments": {
            "AutoMLJobName": "AutoMLJo",
            "AutoMLJobConfig": {
                "CandidateGenerationConfig": {
                    "FeatureSpecificationS3Uri": "s3://bucket/features.json"
                },
                "DataSplitConfig": {"ValidationFraction": 0.3},
                "Mode": "ENSEMBLING",
                "CompletionCriteria": {
                    "MaxAutoMLJobRuntimeInSeconds": 36000,
                    "MaxCandidates": 1,
                    "MaxRuntimePerTrainingJobInSeconds": 3600,
                },
                "SecurityConfig": {
                    "EnableInterContainerTrafficEncryption": False,
                    "VolumeKmsKeyId": "volume-kms-key-id-string",
                    "VpcConfig": {"SecurityGroupIds": ["group"], "Subnets": ["subnet"]},
                },
            },
            "AutoMLJobObjective": {"MetricName": "F1"},
            "InputDataConfig": [
                {
                    "ChannelType": "training",
                    "CompressionType": "Gzip",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": "s3://bucket/data",
                        }
                    },
                    "TargetAttributeName": "target",
                    "SampleWeightAttributeName": "sampleWeight",
                },
                {
                    "ChannelType": "validation",
                    "CompressionType": "Gzip",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": "s3://bucket/validation_data",
                        }
                    },
                    "TargetAttributeName": "target",
                    "SampleWeightAttributeName": "sampleWeight",
                },
            ],
            "OutputDataConfig": {
                "KmsKeyId": "output-kms-key-id-string",
                "S3OutputPath": "s3://my_other_bucket/",
            },
            "ProblemType": "BinaryClassification",
            "RoleArn": "DummyRole",
            "Tags": [{"Name": "some-tag", "Value": "value-for-tag"}],
        },
    }


def test_single_automl_step_with_parameter(pipeline_session):
    target_parameter = ParameterString(name="y")
    auto_ml = AutoML(
        role=ROLE,
        target_attribute_name=target_parameter,
        sagemaker_session=pipeline_session,
        volume_kms_key="volume-kms-key-id-string",
        vpc_config={"SecurityGroupIds": ["group"], "Subnets": ["subnet"]},
        encrypt_inter_container_traffic=False,
        compression_type="Gzip",
        output_kms_key="output-kms-key-id-string",
        output_path="s3://my_other_bucket/",
        problem_type="BinaryClassification",
        max_candidates=1,
        max_runtime_per_training_job_in_seconds=3600,
        total_job_runtime_in_seconds=36000,
        job_objective={"MetricName": "F1"},
        generate_candidate_definitions_only=False,
        tags=[{"Name": "some-tag", "Value": "value-for-tag"}],
        content_type="x-application/vnd.amazon+parquet",
        s3_data_type="ManifestFile",
        feature_specification_s3_uri="s3://bucket/features.json",
        validation_fraction=0.3,
        mode="ENSEMBLING",
        auto_generate_endpoint_name=False,
        endpoint_name="EndpointName",
    )
    input_training = AutoMLInput(
        inputs="s3://bucket/data",
        target_attribute_name=target_parameter,
        compression="Gzip",
        channel_type="training",
    )
    input_validation = AutoMLInput(
        inputs="s3://bucket/validation_data",
        target_attribute_name=target_parameter,
        compression="Gzip",
        channel_type="validation",
    )
    inputs = [input_training, input_validation]

    step_args = auto_ml.fit(
        inputs=inputs,
    )

    automl_step = AutoMLStep(
        name="MyAutoMLStep",
        step_args=step_args,
    )

    assert automl_step.properties.BestCandidateProperties.ModelInsightsJsonReportPath.expr == {
        "Get": "Steps.MyAutoMLStep.BestCandidateProperties.ModelInsightsJsonReportPath"
    }
    assert (
        automl_step.properties.BestCandidateProperties.ModelInsightsJsonReportPath._referenced_steps
        == [automl_step]
    )
    assert automl_step.properties.BestCandidateProperties.ExplainabilityJsonReportPath.expr == {
        "Get": "Steps.MyAutoMLStep.BestCandidateProperties.ExplainabilityJsonReportPath"
    }

    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[target_parameter],
        steps=[automl_step],
        sagemaker_session=pipeline_session,
    )

    step_dsl_list = json.loads(pipeline.definition())["Steps"]
    assert step_dsl_list[0] == {
        "Name": "MyAutoMLStep",
        "Type": "AutoML",
        "Arguments": {
            "AutoMLJobConfig": {
                "CandidateGenerationConfig": {
                    "FeatureSpecificationS3Uri": "s3://bucket/features.json"
                },
                "DataSplitConfig": {"ValidationFraction": 0.3},
                "Mode": "ENSEMBLING",
                "CompletionCriteria": {
                    "MaxAutoMLJobRuntimeInSeconds": 36000,
                    "MaxCandidates": 1,
                    "MaxRuntimePerTrainingJobInSeconds": 3600,
                },
                "SecurityConfig": {
                    "EnableInterContainerTrafficEncryption": False,
                    "VolumeKmsKeyId": "volume-kms-key-id-string",
                    "VpcConfig": {"SecurityGroupIds": ["group"], "Subnets": ["subnet"]},
                },
            },
            "AutoMLJobObjective": {"MetricName": "F1"},
            "InputDataConfig": [
                {
                    "ChannelType": "training",
                    "CompressionType": "Gzip",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": "s3://bucket/data",
                        }
                    },
                    "TargetAttributeName": {"Get": "Parameters.y"},
                },
                {
                    "ChannelType": "validation",
                    "CompressionType": "Gzip",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": "s3://bucket/validation_data",
                        }
                    },
                    "TargetAttributeName": {"Get": "Parameters.y"},
                },
            ],
            "OutputDataConfig": {
                "KmsKeyId": "output-kms-key-id-string",
                "S3OutputPath": "s3://my_other_bucket/",
            },
            "ProblemType": "BinaryClassification",
            "RoleArn": "DummyRole",
            "Tags": [{"Name": "some-tag", "Value": "value-for-tag"}],
        },
    }


def test_get_best_auto_ml_model(pipeline_session):
    auto_ml = AutoML(
        role=ROLE,
        target_attribute_name="y",
        sagemaker_session=pipeline_session,
        volume_kms_key="volume-kms-key-id-string",
        vpc_config={"SecurityGroupIds": ["group"], "Subnets": ["subnet"]},
        encrypt_inter_container_traffic=False,
        compression_type="Gzip",
        output_kms_key="output-kms-key-id-string",
        output_path="s3://my_other_bucket/",
        problem_type="BinaryClassification",
        max_candidates=1,
        max_runtime_per_training_job_in_seconds=3600,
        total_job_runtime_in_seconds=36000,
        job_objective={"MetricName": "F1"},
        generate_candidate_definitions_only=False,
        tags=[{"Name": "some-tag", "Value": "value-for-tag"}],
        content_type="x-application/vnd.amazon+parquet",
        s3_data_type="ManifestFile",
        feature_specification_s3_uri="s3://bucket/features.json",
        validation_fraction=0.3,
        mode="ENSEMBLING",
        auto_generate_endpoint_name=False,
        endpoint_name="EndpointName",
    )
    input_training = AutoMLInput(
        inputs="s3://bucket/data",
        target_attribute_name="target",
        compression="Gzip",
        channel_type="training",
    )
    input_validation = AutoMLInput(
        inputs="s3://bucket/validation_data",
        target_attribute_name="target",
        compression="Gzip",
        channel_type="validation",
    )
    inputs = [input_training, input_validation]

    step_args = auto_ml.fit(
        inputs=inputs,
    )

    automl_step = AutoMLStep(
        name="MyAutoMLStep",
        step_args=step_args,
    )

    automl_model = automl_step.get_best_auto_ml_model(sagemaker_session=pipeline_session, role=ROLE)

    step_args_create_model = automl_model.create(
        instance_type="c4.4xlarge",
    )

    automl_model_step = ModelStep(
        name="MyAutoMLModelStep",
        step_args=step_args_create_model,
    )

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[automl_step, automl_model_step],
        sagemaker_session=pipeline_session,
    )

    step_dsl_list = json.loads(pipeline.definition())["Steps"]
    assert len(step_dsl_list) == 2
    assert step_dsl_list[1] == {
        "Name": "MyAutoMLModelStep-CreateModel",
        "Type": "Model",
        "Arguments": {
            "ExecutionRoleArn": "DummyRole",
            "PrimaryContainer": {
                "Environment": {
                    "MODEL_NAME": {
                        "Get": "Steps.MyAutoMLStep.BestCandidate.InferenceContainers[0]."
                        "Environment['MODEL_NAME']"
                    },
                    "SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT": {
                        "Get": "Steps.MyAutoMLStep.BestCandidate.InferenceContainers[0]."
                        "Environment['SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT']"
                    },
                    "SAGEMAKER_INFERENCE_OUTPUT": {
                        "Get": "Steps.MyAutoMLStep.BestCandidate.InferenceContainers[0]."
                        "Environment['SAGEMAKER_INFERENCE_OUTPUT']"
                    },
                    "SAGEMAKER_INFERENCE_SUPPORTED": {
                        "Get": "Steps.MyAutoMLStep.BestCandidate.InferenceContainers[0]."
                        "Environment['SAGEMAKER_INFERENCE_SUPPORTED']"
                    },
                    "SAGEMAKER_PROGRAM": {
                        "Get": "Steps.MyAutoMLStep.BestCandidate.InferenceContainers[0]."
                        "Environment['SAGEMAKER_PROGRAM']"
                    },
                    "SAGEMAKER_SUBMIT_DIRECTORY": {
                        "Get": "Steps.MyAutoMLStep.BestCandidate.InferenceContainers[0]."
                        "Environment['SAGEMAKER_SUBMIT_DIRECTORY']"
                    },
                },
                "Image": {"Get": "Steps.MyAutoMLStep.BestCandidate.InferenceContainers[0].Image"},
                "ModelDataUrl": {
                    "Get": "Steps.MyAutoMLStep.BestCandidate.InferenceContainers[0].ModelDataUrl"
                },
            },
        },
    }


def test_automl_step_with_invalid_mode(pipeline_session):
    auto_ml = AutoML(
        role=ROLE,
        target_attribute_name="y",
        sagemaker_session=pipeline_session,
        volume_kms_key="volume-kms-key-id-string",
        vpc_config={"SecurityGroupIds": ["group"], "Subnets": ["subnet"]},
        encrypt_inter_container_traffic=False,
        compression_type="Gzip",
        output_kms_key="output-kms-key-id-string",
        output_path="s3://my_other_bucket/",
        problem_type="BinaryClassification",
        max_candidates=1,
        max_runtime_per_training_job_in_seconds=3600,
        total_job_runtime_in_seconds=36000,
        job_objective={"MetricName": "F1"},
        generate_candidate_definitions_only=False,
        tags=[{"Name": "some-tag", "Value": "value-for-tag"}],
        content_type="x-application/vnd.amazon+parquet",
        s3_data_type="ManifestFile",
        feature_specification_s3_uri="s3://bucket/features.json",
        validation_fraction=0.3,
        mode="HPO",
        auto_generate_endpoint_name=False,
        endpoint_name="EndpointName",
    )
    input_training = AutoMLInput(
        inputs="s3://bucket/data",
        target_attribute_name="target",
        compression="Gzip",
        channel_type="training",
    )
    input_validation = AutoMLInput(
        inputs="s3://bucket/validation_data",
        target_attribute_name="target",
        compression="Gzip",
        channel_type="validation",
    )
    inputs = [input_training, input_validation]

    step_args = auto_ml.fit(
        inputs=inputs,
    )

    with pytest.raises(AutoMLStepInvalidModeError) as error:
        automl_step = AutoMLStep(
            name="MyAutoMLStep",
            step_args=step_args,
        )
        _ = automl_step.arguments()
    assert (
        "Mode in AutoMLJobConfig must be defined for AutoMLStep. "
        "AutoMLStep currently only supports ENSEMBLING mode" in str(error.value)
    )


def test_automl_step_with_no_mode(pipeline_session):
    auto_ml = AutoML(
        role=ROLE,
        target_attribute_name="y",
        sagemaker_session=pipeline_session,
        volume_kms_key="volume-kms-key-id-string",
        vpc_config={"SecurityGroupIds": ["group"], "Subnets": ["subnet"]},
        encrypt_inter_container_traffic=False,
        compression_type="Gzip",
        output_kms_key="output-kms-key-id-string",
        output_path="s3://my_other_bucket/",
        problem_type="BinaryClassification",
        max_candidates=1,
        max_runtime_per_training_job_in_seconds=3600,
        total_job_runtime_in_seconds=36000,
        job_objective={"MetricName": "F1"},
        generate_candidate_definitions_only=False,
        tags=[{"Name": "some-tag", "Value": "value-for-tag"}],
        content_type="x-application/vnd.amazon+parquet",
        s3_data_type="ManifestFile",
        feature_specification_s3_uri="s3://bucket/features.json",
        validation_fraction=0.3,
        auto_generate_endpoint_name=False,
        endpoint_name="EndpointName",
    )
    input_training = AutoMLInput(
        inputs="s3://bucket/data",
        target_attribute_name="target",
        compression="Gzip",
        channel_type="training",
    )
    input_validation = AutoMLInput(
        inputs="s3://bucket/validation_data",
        target_attribute_name="target",
        compression="Gzip",
        channel_type="validation",
    )
    inputs = [input_training, input_validation]

    step_args = auto_ml.fit(
        inputs=inputs,
    )

    with pytest.raises(AutoMLStepInvalidModeError) as error:
        automl_step = AutoMLStep(
            name="MyAutoMLStep",
            step_args=step_args,
        )
        _ = automl_step.arguments()
    assert (
        "Mode in AutoMLJobConfig must be defined for AutoMLStep. "
        "AutoMLStep currently only supports ENSEMBLING mode" in str(error.value)
    )
