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
import re
import pytest

from sagemaker.clarify import (
    DataConfig,
    BiasConfig,
    ModelConfig,
    ModelPredictedLabelConfig,
    SHAPConfig,
)
from sagemaker.workflow.clarify_check_step import (
    DataBiasCheckConfig,
    ClarifyCheckStep,
    ModelBiasCheckConfig,
    ModelExplainabilityCheckConfig,
    ClarifyCheckConfig,
    _EXPLAINABILITY_MONITORING_CFG_BASE_NAME,
    _BIAS_MONITORING_CFG_BASE_NAME,
)
from sagemaker.model_monitor.model_monitoring import _MODEL_MONITOR_S3_PATH
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline import PipelineDefinitionConfig
from sagemaker.workflow.steps import CacheConfig
from sagemaker.workflow.check_job_config import CheckJobConfig

_REGION = "us-west-2"
_ROLE = "DummyRole"
_DEFAULT_BUCKET = "my-bucket"
_S3_INPUT_PATH = "s3://my_bucket/input"
_S3_OUTPUT_PATH = "s3://my_bucket/output"
_S3_ANALYSIS_CONFIG_OUTPUT_PATH = "s3://my_bucket/analysis_cfg_output"
_CHECK_JOB_PREFIX = "CheckJobPrefix"


_expected_data_bias_dsl = {
    "Name": "DataBiasCheckStep",
    "Type": "ClarifyCheck",
    "Arguments": {
        "ProcessingJobName": _CHECK_JOB_PREFIX,
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceType": "ml.m5.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            }
        },
        "AppSpecification": {
            "ImageUri": "306415355426.dkr.ecr.us-west-2.amazonaws.com/sagemaker-clarify-processing:1.0"
        },
        "RoleArn": "DummyRole",
        "ProcessingInputs": [
            {
                "InputName": "analysis_config",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": f"{_S3_ANALYSIS_CONFIG_OUTPUT_PATH}/analysis_config.json",
                    "LocalPath": "/opt/ml/processing/input/config",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "dataset",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": f"{_S3_INPUT_PATH}",
                    "LocalPath": "/opt/ml/processing/input/data",
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
                    "OutputName": "analysis_result",
                    "AppManaged": False,
                    "S3Output": {
                        "S3Uri": f"{_S3_OUTPUT_PATH}",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ],
            "KmsKeyId": "output_kms_key",
        },
    },
    "CheckType": "DATA_BIAS",
    "ModelPackageGroupName": {"Get": "Parameters.MyModelPackageGroup"},
    "SkipCheck": False,
    "FailOnViolation": False,
    "RegisterNewBaseline": False,
    "SuppliedBaselineConstraints": "supplied_baseline_constraints",
    "CacheConfig": {"Enabled": True, "ExpireAfter": "PT1H"},
}

_expected_model_bias_dsl = {
    "Name": "ModelBiasCheckStep",
    "Type": "ClarifyCheck",
    "Arguments": {
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceType": "ml.m5.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            }
        },
        "AppSpecification": {
            "ImageUri": "306415355426.dkr.ecr.us-west-2.amazonaws.com/sagemaker-clarify-processing:1.0"
        },
        "RoleArn": "DummyRole",
        "ProcessingInputs": [
            {
                "InputName": "analysis_config",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": f"{_S3_OUTPUT_PATH}/analysis_config.json",
                    "LocalPath": "/opt/ml/processing/input/config",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "dataset",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": f"{_S3_INPUT_PATH}",
                    "LocalPath": "/opt/ml/processing/input/data",
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
                    "OutputName": "analysis_result",
                    "AppManaged": False,
                    "S3Output": {
                        "S3Uri": f"{_S3_OUTPUT_PATH}",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ],
            "KmsKeyId": "output_kms_key",
        },
    },
    "CheckType": "MODEL_BIAS",
    "ModelPackageGroupName": {"Get": "Parameters.MyModelPackageGroup"},
    "SkipCheck": False,
    "FailOnViolation": True,
    "RegisterNewBaseline": False,
    "SuppliedBaselineConstraints": "supplied_baseline_constraints",
    "ModelName": "model_name",
}

_expected_model_explainability_dsl = {
    "Name": "ModelExplainabilityCheckStep",
    "Type": "ClarifyCheck",
    "Arguments": {
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceType": "ml.m5.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            }
        },
        "AppSpecification": {
            "ImageUri": "306415355426.dkr.ecr.us-west-2.amazonaws.com/sagemaker-clarify-processing:1.0"
        },
        "RoleArn": "DummyRole",
        "ProcessingInputs": [
            {
                "InputName": "analysis_config",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": f"{_S3_OUTPUT_PATH}/analysis_config.json",
                    "LocalPath": "/opt/ml/processing/input/config",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "dataset",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": f"{_S3_INPUT_PATH}",
                    "LocalPath": "/opt/ml/processing/input/data",
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
                    "OutputName": "analysis_result",
                    "AppManaged": False,
                    "S3Output": {
                        "S3Uri": f"{_S3_OUTPUT_PATH}",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ],
            "KmsKeyId": "output_kms_key",
        },
    },
    "CheckType": "MODEL_EXPLAINABILITY",
    "ModelPackageGroupName": {"Get": "Parameters.MyModelPackageGroup"},
    "SkipCheck": False,
    "FailOnViolation": False,
    "RegisterNewBaseline": False,
    "SuppliedBaselineConstraints": "supplied_baseline_constraints",
    "ModelName": "model_name",
}


@pytest.fixture
def model_package_group_name():
    return ParameterString(name="MyModelPackageGroup", default_value="")


@pytest.fixture
def check_job_config(sagemaker_session):
    return CheckJobConfig(
        role=_ROLE,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        sagemaker_session=sagemaker_session,
        output_kms_key="output_kms_key",
        base_job_name=_CHECK_JOB_PREFIX,
    )


@pytest.fixture
def data_config():
    return DataConfig(
        s3_data_input_path=_S3_INPUT_PATH,
        s3_output_path=_S3_OUTPUT_PATH,
        label="fraud",
        dataset_type="text/csv",
    )


@pytest.fixture()
def bias_config():
    return BiasConfig(
        label_values_or_threshold=[0],
        facet_name="customer_gender_female",
        facet_values_or_threshold=[1],
    )


@pytest.fixture()
def model_config():
    return ModelConfig(
        model_name="model_name",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        accept_type="text/csv",
        content_type="text/csv",
    )


@pytest.fixture()
def predictions_config():
    return ModelPredictedLabelConfig(probability_threshold=0.8)


@pytest.fixture()
def shap_config():
    return SHAPConfig(
        baseline=[],
        num_samples=15,
        agg_method="mean_abs",
        save_local_shap_values=True,
    )


def test_data_bias_check_step(
    sagemaker_session, check_job_config, model_package_group_name, bias_config
):
    data_bias_data_config = DataConfig(
        s3_data_input_path=_S3_INPUT_PATH,
        s3_output_path=_S3_OUTPUT_PATH,
        s3_analysis_config_output_path=_S3_ANALYSIS_CONFIG_OUTPUT_PATH,
        label="fraud",
        dataset_type="text/csv",
    )
    data_bias_check_config = DataBiasCheckConfig(
        data_config=data_bias_data_config,
        data_bias_config=bias_config,
        methods="all",
        kms_key="kms_key",
    )
    data_bias_check_step = ClarifyCheckStep(
        name="DataBiasCheckStep",
        clarify_check_config=data_bias_check_config,
        check_job_config=check_job_config,
        skip_check=False,
        fail_on_violation=False,
        register_new_baseline=False,
        model_package_group_name=model_package_group_name,
        supplied_baseline_constraints="supplied_baseline_constraints",
        cache_config=CacheConfig(enable_caching=True, expire_after="PT1H"),
    )

    definition_config = PipelineDefinitionConfig(use_custom_job_prefix=True)
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[model_package_group_name],
        steps=[data_bias_check_step],
        sagemaker_session=sagemaker_session,
        pipeline_definition_config=definition_config,
    )

    step_def = json.loads(pipeline.definition())["Steps"][0]
    assert step_def == _expected_data_bias_dsl
    assert re.match(
        f"{_S3_ANALYSIS_CONFIG_OUTPUT_PATH}/{_BIAS_MONITORING_CFG_BASE_NAME}-configuration"
        + f"/{_BIAS_MONITORING_CFG_BASE_NAME}-config.*/.*/analysis_config.json",
        data_bias_check_config.monitoring_analysis_config_uri,
    )


def test_model_bias_check_step(
    sagemaker_session,
    check_job_config,
    model_package_group_name,
    data_config,
    bias_config,
    model_config,
    predictions_config,
):
    model_bias_check_config = ModelBiasCheckConfig(
        data_config=data_config,
        data_bias_config=bias_config,
        model_config=model_config,
        model_predicted_label_config=predictions_config,
        methods="all",
    )
    model_bias_check_step = ClarifyCheckStep(
        name="ModelBiasCheckStep",
        clarify_check_config=model_bias_check_config,
        check_job_config=check_job_config,
        skip_check=False,
        fail_on_violation=True,
        register_new_baseline=False,
        model_package_group_name=model_package_group_name,
        supplied_baseline_constraints="supplied_baseline_constraints",
    )
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[model_package_group_name],
        steps=[model_bias_check_step],
        sagemaker_session=sagemaker_session,
    )

    assert json.loads(pipeline.definition())["Steps"][0] == _expected_model_bias_dsl
    assert re.match(
        f"s3://{_DEFAULT_BUCKET}/{_MODEL_MONITOR_S3_PATH}"
        + f"/{_BIAS_MONITORING_CFG_BASE_NAME}-configuration"
        + f"/{_BIAS_MONITORING_CFG_BASE_NAME}-config.*/.*/analysis_config.json",
        model_bias_check_config.monitoring_analysis_config_uri,
    )


def test_model_explainability_check_step(
    sagemaker_session,
    check_job_config,
    model_package_group_name,
    data_config,
    model_config,
    shap_config,
):
    model_explainability_check_config = ModelExplainabilityCheckConfig(
        data_config=data_config,
        model_config=model_config,
        explainability_config=shap_config,
    )
    model_explainability_check_step = ClarifyCheckStep(
        name="ModelExplainabilityCheckStep",
        clarify_check_config=model_explainability_check_config,
        check_job_config=check_job_config,
        skip_check=False,
        fail_on_violation=False,
        register_new_baseline=False,
        model_package_group_name=model_package_group_name,
        supplied_baseline_constraints="supplied_baseline_constraints",
    )
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[model_package_group_name],
        steps=[model_explainability_check_step],
        sagemaker_session=sagemaker_session,
    )

    assert json.loads(pipeline.definition())["Steps"][0] == _expected_model_explainability_dsl
    assert re.match(
        f"s3://{_DEFAULT_BUCKET}/{_MODEL_MONITOR_S3_PATH}"
        + f"/{_EXPLAINABILITY_MONITORING_CFG_BASE_NAME}-configuration"
        + f"/{_EXPLAINABILITY_MONITORING_CFG_BASE_NAME}-config-.*/.*/analysis_config.json",
        model_explainability_check_config.monitoring_analysis_config_uri,
    )


def test_clarify_check_step_properties(
    check_job_config,
    model_package_group_name,
    data_config,
    model_config,
    shap_config,
):
    model_explainability_check_config = ModelExplainabilityCheckConfig(
        data_config=data_config,
        model_config=model_config,
        explainability_config=shap_config,
    )
    model_explainability_check_step = ClarifyCheckStep(
        name="ModelExplainabilityCheckStep",
        clarify_check_config=model_explainability_check_config,
        check_job_config=check_job_config,
        skip_check=False,
        register_new_baseline=False,
        model_package_group_name=model_package_group_name,
        supplied_baseline_constraints="supplied_baseline_constraints",
    )

    assert model_explainability_check_step.properties.CalculatedBaselineConstraints.expr == {
        "Get": "Steps.ModelExplainabilityCheckStep.CalculatedBaselineConstraints"
    }
    assert (
        model_explainability_check_step.properties.CalculatedBaselineConstraints._referenced_steps
        == [model_explainability_check_step]
    )
    assert model_explainability_check_step.properties.BaselineUsedForDriftCheckConstraints.expr == {
        "Get": "Steps.ModelExplainabilityCheckStep.BaselineUsedForDriftCheckConstraints"
    }


def test_clarify_check_step_invalid_config(
    check_job_config,
    model_package_group_name,
    data_config,
):
    clarify_check_config = ClarifyCheckConfig(data_config=data_config)
    with pytest.raises(Exception) as error:
        ClarifyCheckStep(
            name="ClarifyCheckStep",
            clarify_check_config=clarify_check_config,
            check_job_config=check_job_config,
            skip_check=False,
            register_new_baseline=False,
            model_package_group_name=model_package_group_name,
            supplied_baseline_constraints="supplied_baseline_constraints",
        )

    assert (
        str(error.value)
        == "The clarify_check_config can only be object of DataBiasCheckConfig, ModelBiasCheckConfig"
        " or ModelExplainabilityCheckConfig"
    )


def test_clarify_check_step_with_none_or_invalid_s3_analysis_config_output_uri(
    bias_config,
    check_job_config,
    model_package_group_name,
):
    # s3_analysis_config_output is None and s3_output_path is valid s3 path str
    data_config = DataConfig(
        s3_data_input_path=_S3_INPUT_PATH,
        s3_output_path=_S3_OUTPUT_PATH,
        label="fraud",
        dataset_type="text/csv",
    )
    clarify_check_config = DataBiasCheckConfig(
        data_config=data_config,
        data_bias_config=bias_config,
    )

    ClarifyCheckStep(
        name="ClarifyCheckStep",
        clarify_check_config=clarify_check_config,
        check_job_config=check_job_config,
        skip_check=False,
        register_new_baseline=False,
        model_package_group_name=model_package_group_name,
        supplied_baseline_constraints="supplied_baseline_constraints",
    )

    # s3_analysis_config_output is empty but s3_output_path is Parameter
    data_config = DataConfig(
        s3_data_input_path=_S3_INPUT_PATH,
        s3_output_path=ParameterString(name="S3OutputPath", default_value=_S3_OUTPUT_PATH),
        s3_analysis_config_output_path="",
        label="fraud",
        dataset_type="text/csv",
    )
    clarify_check_config = DataBiasCheckConfig(
        data_config=data_config,
        data_bias_config=bias_config,
    )

    with pytest.raises(Exception) as error:
        ClarifyCheckStep(
            name="ClarifyCheckStep",
            clarify_check_config=clarify_check_config,
            check_job_config=check_job_config,
            skip_check=False,
            register_new_baseline=False,
            model_package_group_name=model_package_group_name,
            supplied_baseline_constraints="supplied_baseline_constraints",
        )

    assert (
        str(error.value)
        == "`s3_output_path` cannot be of type ExecutionVariable/Expression/Parameter/Properties "
        + "if `s3_analysis_config_output_path` is none or empty "
    )

    # s3_analysis_config_output is invalid
    data_config = DataConfig(
        s3_data_input_path=_S3_INPUT_PATH,
        s3_output_path=ParameterString(name="S3OutputPath", default_value=_S3_OUTPUT_PATH),
        s3_analysis_config_output_path=ParameterString(name="S3OAnalysisCfgOutput"),
        label="fraud",
        dataset_type="text/csv",
    )
    clarify_check_config = DataBiasCheckConfig(
        data_config=data_config,
        data_bias_config=bias_config,
    )

    with pytest.raises(Exception) as error:
        ClarifyCheckStep(
            name="ClarifyCheckStep",
            clarify_check_config=clarify_check_config,
            check_job_config=check_job_config,
            skip_check=False,
            register_new_baseline=False,
            model_package_group_name=model_package_group_name,
            supplied_baseline_constraints="supplied_baseline_constraints",
        )

    assert (
        str(error.value) == "s3_analysis_config_output_path cannot be of type "
        "ExecutionVariable/Expression/Parameter/Properties"
    )


def test_get_s3_base_uri_for_monitoring_analysis_config(
    check_job_config,
    data_config,
    bias_config,
    model_config,
    shap_config,
    predictions_config,
):
    # ModelExplainabilityCheckStep without specifying s3_analysis_config_output_path
    model_explainability_check_config_1 = ModelExplainabilityCheckConfig(
        data_config=data_config,
        model_config=model_config,
        explainability_config=shap_config,
    )
    model_explainability_check_step_1 = ClarifyCheckStep(
        name="ModelExplainabilityCheckStep",
        clarify_check_config=model_explainability_check_config_1,
        check_job_config=check_job_config,
    )

    assert (
        f"s3://{_DEFAULT_BUCKET}/{_MODEL_MONITOR_S3_PATH}"
        + f"/{_EXPLAINABILITY_MONITORING_CFG_BASE_NAME}-configuration"
        == model_explainability_check_step_1._get_s3_base_uri_for_monitoring_analysis_config()
    )

    # ModelExplainabilityCheckStep with specifying s3_analysis_config_output_path
    model_explainability_data_config = DataConfig(
        s3_data_input_path=_S3_INPUT_PATH,
        s3_output_path=ParameterString(name="S3OutputPath", default_value=_S3_OUTPUT_PATH),
        s3_analysis_config_output_path=_S3_ANALYSIS_CONFIG_OUTPUT_PATH,
    )
    model_explainability_check_config_2 = ModelExplainabilityCheckConfig(
        data_config=model_explainability_data_config,
        model_config=model_config,
        explainability_config=shap_config,
    )
    model_explainability_check_step_2 = ClarifyCheckStep(
        name="ModelExplainabilityCheckStep",
        clarify_check_config=model_explainability_check_config_2,
        check_job_config=check_job_config,
    )

    assert (
        f"{_S3_ANALYSIS_CONFIG_OUTPUT_PATH}/{_EXPLAINABILITY_MONITORING_CFG_BASE_NAME}-configuration"
        == model_explainability_check_step_2._get_s3_base_uri_for_monitoring_analysis_config()
    )

    # ModelBiasCheckStep with specifying s3_analysis_config_output_path
    model_bias_data_config = DataConfig(
        s3_data_input_path=_S3_INPUT_PATH,
        s3_output_path=_S3_OUTPUT_PATH,
        s3_analysis_config_output_path=_S3_ANALYSIS_CONFIG_OUTPUT_PATH,
    )
    model_bias_check_config = ModelBiasCheckConfig(
        data_config=model_bias_data_config,
        data_bias_config=bias_config,
        model_config=model_config,
        model_predicted_label_config=predictions_config,
    )
    model_bias_check_step = ClarifyCheckStep(
        name="ModelBiasCheckStep",
        clarify_check_config=model_bias_check_config,
        check_job_config=check_job_config,
    )

    assert (
        f"{_S3_ANALYSIS_CONFIG_OUTPUT_PATH}/{_BIAS_MONITORING_CFG_BASE_NAME}-configuration"
        == model_bias_check_step._get_s3_base_uri_for_monitoring_analysis_config()
    )

    # DataBiasCheckStep without specifying s3_analysis_config_output_path
    data_bias_check_config = DataBiasCheckConfig(
        data_config=data_config,
        data_bias_config=bias_config,
    )
    data_bias_check_step = ClarifyCheckStep(
        name="DataBiasCheckStep",
        clarify_check_config=data_bias_check_config,
        check_job_config=check_job_config,
    )
    assert (
        f"s3://{_DEFAULT_BUCKET}/{_MODEL_MONITOR_S3_PATH}"
        + f"/{_BIAS_MONITORING_CFG_BASE_NAME}-configuration"
        == data_bias_check_step._get_s3_base_uri_for_monitoring_analysis_config()
    )
