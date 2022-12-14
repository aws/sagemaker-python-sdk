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

from sagemaker.model_monitor import DatasetFormat
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.quality_check_step import (
    QualityCheckStep,
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
    QualityCheckConfig,
)
from sagemaker.workflow.steps import CacheConfig
from sagemaker.workflow.check_job_config import CheckJobConfig

_ROLE = "DummyRole"


_expected_data_quality_dsl = {
    "Name": "DataQualityCheckStep",
    "Type": "QualityCheck",
    "Arguments": {
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceType": "ml.m5.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 60,
            }
        },
        "AppSpecification": {
            "ImageUri": "159807026194.dkr.ecr.us-west-2.amazonaws.com/sagemaker-model-monitor-analyzer",
        },
        "RoleArn": "DummyRole",
        "ProcessingInputs": [
            {
                "InputName": "baseline_dataset_input",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": {"Get": "Parameters.BaselineDataset"},
                    "LocalPath": "/opt/ml/processing/input/baseline_dataset_input",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "post_analytics_processor_script_input",
                "AppManaged": False,
                "S3Input": {
                    "LocalPath": "/opt/ml/processing/input/post_analytics_processor_script_input",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "record_preprocessor_script_input",
                "AppManaged": False,
                "S3Input": {
                    "LocalPath": "/opt/ml/processing/input/record_preprocessor_script_input",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "quality_check_output",
                    "AppManaged": False,
                    "S3Output": {
                        "S3Uri": "s3://...",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        },
        "Environment": {
            "output_path": "/opt/ml/processing/output",
            "publish_cloudwatch_metrics": "Disabled",
            "dataset_format": '{"csv": {"header": true, "output_columns_position": "START"}}',
            "record_preprocessor_script": "/opt/ml/processing/input/record_preprocessor_script_input/preprocessor.py",
            "post_analytics_processor_script": "/opt/ml/processing/input/"
            + "post_analytics_processor_script_input/postprocessor.py",
            "dataset_source": "/opt/ml/processing/input/baseline_dataset_input",
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 1800},
    },
    "CheckType": "DATA_QUALITY",
    "ModelPackageGroupName": {"Get": "Parameters.MyModelPackageGroup"},
    "SkipCheck": False,
    "FailOnViolation": False,
    "RegisterNewBaseline": False,
    "SuppliedBaselineStatistics": {"Get": "Parameters.SuppliedBaselineStatisticsUri"},
    "SuppliedBaselineConstraints": {"Get": "Parameters.SuppliedBaselineConstraintsUri"},
    "CacheConfig": {"Enabled": True, "ExpireAfter": "PT1H"},
}

_expected_model_quality_dsl = {
    "Name": "ModelQualityCheckStep",
    "Type": "QualityCheck",
    "Arguments": {
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceType": "ml.m5.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 60,
            }
        },
        "AppSpecification": {
            "ImageUri": "159807026194.dkr.ecr.us-west-2.amazonaws.com/sagemaker-model-monitor-analyzer"
        },
        "RoleArn": "DummyRole",
        "ProcessingInputs": [
            {
                "InputName": "baseline_dataset_input",
                "AppManaged": False,
                "S3Input": {
                    "LocalPath": "/opt/ml/processing/input/baseline_dataset_input",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "post_analytics_processor_script_input",
                "AppManaged": False,
                "S3Input": {
                    "LocalPath": "/opt/ml/processing/input/post_analytics_processor_script_input",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "quality_check_output",
                    "AppManaged": False,
                    "S3Output": {
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        },
        "Environment": {
            "output_path": "/opt/ml/processing/output",
            "publish_cloudwatch_metrics": "Disabled",
            "dataset_format": '{"csv": {"header": true, "output_columns_position": "START"}}',
            "post_analytics_processor_script": "/opt/ml/processing/input/post_analytics_processor_script_input/"
            + "postprocessor.py",
            "dataset_source": "/opt/ml/processing/input/baseline_dataset_input",
            "analysis_type": "MODEL_QUALITY",
            "problem_type": "BinaryClassification",
            "probability_attribute": "0",
            "probability_threshold_attribute": "0.5",
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 1800},
    },
    "CheckType": "MODEL_QUALITY",
    "ModelPackageGroupName": {"Get": "Parameters.MyModelPackageGroup"},
    "SkipCheck": False,
    "FailOnViolation": True,
    "RegisterNewBaseline": False,
    "SuppliedBaselineStatistics": {"Get": "Parameters.SuppliedBaselineStatisticsUri"},
    "SuppliedBaselineConstraints": {"Get": "Parameters.SuppliedBaselineConstraintsUri"},
}


@pytest.fixture
def model_package_group_name():
    return ParameterString(name="MyModelPackageGroup", default_value="")


@pytest.fixture
def supplied_baseline_statistics_uri():
    return ParameterString(name="SuppliedBaselineStatisticsUri", default_value="")


@pytest.fixture
def supplied_baseline_constraints_uri():
    return ParameterString(name="SuppliedBaselineConstraintsUri", default_value="")


@pytest.fixture
def check_job_config(sagemaker_session):
    return CheckJobConfig(
        role=_ROLE,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=60,
        max_runtime_in_seconds=1800,
        sagemaker_session=sagemaker_session,
    )


def test_data_quality_check_step(
    sagemaker_session,
    check_job_config,
    model_package_group_name,
    supplied_baseline_statistics_uri,
    supplied_baseline_constraints_uri,
):
    data_quality_check_config = DataQualityCheckConfig(
        baseline_dataset=ParameterString(name="BaselineDataset"),
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri="s3://...",
        record_preprocessor_script="s3://my_bucket/data_quality/preprocessor.py",
        post_analytics_processor_script="s3://my_bucket/data_quality/postprocessor.py",
    )
    data_quality_check_step = QualityCheckStep(
        name="DataQualityCheckStep",
        skip_check=False,
        fail_on_violation=False,
        register_new_baseline=False,
        quality_check_config=data_quality_check_config,
        check_job_config=check_job_config,
        model_package_group_name=model_package_group_name,
        supplied_baseline_statistics=supplied_baseline_statistics_uri,
        supplied_baseline_constraints=supplied_baseline_constraints_uri,
        cache_config=CacheConfig(enable_caching=True, expire_after="PT1H"),
    )
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[
            supplied_baseline_statistics_uri,
            supplied_baseline_constraints_uri,
            model_package_group_name,
        ],
        steps=[data_quality_check_step],
        sagemaker_session=sagemaker_session,
    )
    step_definition = _get_step_definition_for_test(
        pipeline, ["baseline_dataset_input", "quality_check_output"]
    )

    assert step_definition == _expected_data_quality_dsl


def test_model_quality_check_step(
    sagemaker_session,
    check_job_config,
    model_package_group_name,
    supplied_baseline_statistics_uri,
    supplied_baseline_constraints_uri,
):
    model_quality_check_config = ModelQualityCheckConfig(
        baseline_dataset="baseline_dataset_s3_url",
        dataset_format=DatasetFormat.csv(header=True),
        problem_type="BinaryClassification",
        probability_attribute=0,  # the integer should be converted to str by SDK
        ground_truth_attribute=None,
        probability_threshold_attribute=0.5,  # the float should be converted to str by SDK
        post_analytics_processor_script="s3://my_bucket/data_quality/postprocessor.py",
        output_s3_uri="",
    )
    model_quality_check_step = QualityCheckStep(
        name="ModelQualityCheckStep",
        register_new_baseline=False,
        skip_check=False,
        fail_on_violation=True,
        quality_check_config=model_quality_check_config,
        check_job_config=check_job_config,
        model_package_group_name=model_package_group_name,
        supplied_baseline_statistics=supplied_baseline_statistics_uri,
        supplied_baseline_constraints=supplied_baseline_constraints_uri,
    )
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[
            supplied_baseline_statistics_uri,
            supplied_baseline_constraints_uri,
            model_package_group_name,
        ],
        steps=[model_quality_check_step],
        sagemaker_session=sagemaker_session,
    )

    step_definition = _get_step_definition_for_test(pipeline)

    assert step_definition == _expected_model_quality_dsl


def test_quality_check_step_properties(
    check_job_config,
    model_package_group_name,
    supplied_baseline_statistics_uri,
    supplied_baseline_constraints_uri,
):
    model_quality_check_config = ModelQualityCheckConfig(
        baseline_dataset="baseline_dataset_s3_url",
        dataset_format=DatasetFormat.csv(header=True),
        problem_type="BinaryClassification",
        probability_attribute="0",
        probability_threshold_attribute="0.5",
        post_analytics_processor_script="s3://my_bucket/data_quality/postprocessor.py",
        output_s3_uri="",
    )
    model_quality_check_step = QualityCheckStep(
        name="ModelQualityCheckStep",
        register_new_baseline=False,
        skip_check=False,
        quality_check_config=model_quality_check_config,
        check_job_config=check_job_config,
        model_package_group_name=model_package_group_name,
        supplied_baseline_statistics=supplied_baseline_statistics_uri,
        supplied_baseline_constraints=supplied_baseline_constraints_uri,
    )

    assert model_quality_check_step.properties.CalculatedBaselineConstraints.expr == {
        "Get": "Steps.ModelQualityCheckStep.CalculatedBaselineConstraints"
    }
    assert model_quality_check_step.properties.CalculatedBaselineStatistics.expr == {
        "Get": "Steps.ModelQualityCheckStep.CalculatedBaselineStatistics"
    }
    assert model_quality_check_step.properties.BaselineUsedForDriftCheckStatistics.expr == {
        "Get": "Steps.ModelQualityCheckStep.BaselineUsedForDriftCheckStatistics"
    }
    assert model_quality_check_step.properties.BaselineUsedForDriftCheckConstraints.expr == {
        "Get": "Steps.ModelQualityCheckStep.BaselineUsedForDriftCheckConstraints"
    }


def test_quality_check_step_invalid_config(
    check_job_config,
    model_package_group_name,
    supplied_baseline_statistics_uri,
    supplied_baseline_constraints_uri,
):
    quality_check_config = QualityCheckConfig(
        baseline_dataset="baseline_dataset_s3_url",
        dataset_format=DatasetFormat.csv(header=True),
        post_analytics_processor_script="s3://my_bucket/data_quality/postprocessor.py",
        output_s3_uri="",
    )
    with pytest.raises(Exception) as error:
        QualityCheckStep(
            name="QualityCheckStep",
            register_new_baseline=False,
            skip_check=False,
            quality_check_config=quality_check_config,
            check_job_config=check_job_config,
            model_package_group_name=model_package_group_name,
            supplied_baseline_statistics=supplied_baseline_statistics_uri,
            supplied_baseline_constraints=supplied_baseline_constraints_uri,
        )

    assert (
        str(error.value)
        == "The quality_check_config can only be object of DataQualityCheckConfig or ModelQualityCheckConfig"
    )


def _get_step_definition_for_test(pipeline: Pipeline, skip_pop_fields: list = []) -> dict:
    step_definition = json.loads(pipeline.definition())["Steps"][0]
    # pop out the S3Uri as it may be dynamically changed due to timestamp
    for processing_input in step_definition["Arguments"]["ProcessingInputs"]:
        if processing_input["InputName"] in skip_pop_fields:
            continue
        processing_input["S3Input"].pop("S3Uri")
    for output in step_definition["Arguments"]["ProcessingOutputConfig"]["Outputs"]:
        if output["OutputName"] in skip_pop_fields:
            continue
        output["S3Output"].pop("S3Uri")
    return step_definition
