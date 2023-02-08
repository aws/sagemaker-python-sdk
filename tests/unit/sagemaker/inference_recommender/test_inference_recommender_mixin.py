from __future__ import absolute_import

from unittest.mock import patch, MagicMock, ANY

from sagemaker.model import Model, ModelPackage
from sagemaker.parameter import CategoricalParameter
from sagemaker.inference_recommender import ModelLatencyThreshold, Phase
from sagemaker.async_inference import AsyncInferenceConfig
from sagemaker.serverless import ServerlessInferenceConfig

import pytest

REGION = "us-west-2"

MODEL_NAME = "model-name-for-ir"
MODEL_DATA = "s3://bucket/model.tar.gz"
MODEL_IMAGE = "model-image-for-ir"
MODEL_PACKAGE_ARN = "model-package-for-ir"

IR_ROLE_ARN = "arn:aws:iam::123456789123:role/service-role/AmazonSageMaker-ExecutionRole-UnitTest"
IR_SAMPLE_PAYLOAD_URL = "s3://sagemaker-us-west-2-123456789123/payload/payload.tar.gz"
IR_SAMPLE_FRAMEWORK = "SAGEMAKER-SCIKIT-LEARN"
IR_SUPPORTED_CONTENT_TYPES = ["text/csv"]
IR_JOB_NAME = "SMPYTHONSDK-1234567891"
IR_SAMPLE_INSTANCE_TYPE = "ml.c5.xlarge"

IR_SAMPLE_LIST_OF_INSTANCES_HYPERPARAMETER_RANGES = [
    {
        "instance_types": CategoricalParameter(["ml.m5.xlarge", "ml.g4dn.xlarge"]),
        "TEST_PARAM": CategoricalParameter(["TEST_PARAM_VALUE_1", "TEST_PARAM_VALUE_2"]),
    }
]

IR_SAMPLE_SINGLE_INSTANCES_HYPERPARAMETER_RANGES = [
    {
        "instance_types": CategoricalParameter(["ml.m5.xlarge"]),
        "TEST_PARAM": CategoricalParameter(["TEST_PARAM_VALUE_1", "TEST_PARAM_VALUE_2"]),
    },
    {
        "instance_types": CategoricalParameter(["ml.g4dn.xlarge"]),
        "TEST_PARAM": CategoricalParameter(["TEST_PARAM_VALUE_1", "TEST_PARAM_VALUE_2"]),
    },
]

IR_SAMPLE_INVALID_HYPERPARAMETERS_RANGES = [
    {
        "TEST_PARAM": CategoricalParameter(["TEST_PARAM_VALUE_1", "TEST_PARAM_VALUE_2"]),
        "TEST_PARAM2": CategoricalParameter(["TEST_PARAM_VALUE_1", "TEST_PARAM_VALUE_2"]),
    }
]

IR_SAMPLE_PHASES = [
    Phase(duration_in_seconds=300, initial_number_of_users=2, spawn_rate=2),
    Phase(duration_in_seconds=300, initial_number_of_users=14, spawn_rate=2),
]

IR_SAMPLE_MODEL_LATENCY_THRESHOLDS = [
    ModelLatencyThreshold(percentile="P95", value_in_milliseconds=100)
]

IR_RIGHT_SIZE_INSTANCE_TYPE = "ml.m5.xlarge"
IR_RIGHT_SIZE_INITIAL_INSTANCE_COUNT = 1

IR_SAMPLE_INFERENCE_RESPONSE = {
    "JobName": "SMPYTHONSDK-1671044837",
    "JobDescription": "#python-sdk-create",
    "PlaceHolder": "...",
    "InferenceRecommendations": [
        {
            "Metrics": {"PlaceHolder": "..."},
            "EndpointConfiguration": {
                "EndpointName": "sm-epc-test",
                "VariantName": "sm-epc-test",
                "InstanceType": IR_RIGHT_SIZE_INSTANCE_TYPE,
                "InitialInstanceCount": IR_RIGHT_SIZE_INITIAL_INSTANCE_COUNT,
            },
            "ModelConfiguration": {"PlaceHolder": "..."},
        }
    ],
    "PlaceHolder": "...",
}

IR_DEPLOY_ENDPOINT_NAME = "ir-endpoint-test"

IR_SAMPLE_ENDPOINT_CONFIG = [
    {
        "EnvironmentParameterRanges": {
            "CategoricalParameterRanges": [
                {
                    "Name": "TEST_PARAM",
                    "Value": ["TEST_PARAM_VALUE_1", "TEST_PARAM_VALUE_2"],
                },
            ],
        },
        "InstanceType": "ml.m5.xlarge",
    },
    {
        "EnvironmentParameterRanges": {
            "CategoricalParameterRanges": [
                {
                    "Name": "TEST_PARAM",
                    "Value": ["TEST_PARAM_VALUE_1", "TEST_PARAM_VALUE_2"],
                },
            ],
        },
        "InstanceType": "ml.g4dn.xlarge",
    },
]

IR_SAMPLE_TRAFFIC_PATTERN = {
    "Phases": [
        {
            "DurationInSeconds": 300,
            "InitialNumberOfUsers": 2,
            "SpawnRate": 2,
        },
        {
            "DurationInSeconds": 300,
            "InitialNumberOfUsers": 14,
            "SpawnRate": 2,
        },
    ],
    "TrafficType": "PHASES",
}

IR_SAMPLE_STOPPING_CONDITIONS = {
    "MaxInvocations": 100,
    "ModelLatencyThresholds": [
        {
            "Percentile": "P95",
            "ValueInMilliseconds": 100,
        }
    ],
}

IR_SAMPLE_RESOURCE_LIMIT = {
    "MaxNumberOfTests": 5,
    "MaxParallelOfTests": 5,
}


@pytest.fixture()
def sagemaker_session():
    session = MagicMock(boto_region_name=REGION)

    session.create_inference_recommendations_job.return_value = IR_JOB_NAME
    session.wait_for_inference_recommendations_job.return_value = IR_SAMPLE_INFERENCE_RESPONSE

    return session


@pytest.fixture()
def model_package(sagemaker_session):
    return ModelPackage(
        role=IR_ROLE_ARN, model_package_arn=MODEL_PACKAGE_ARN, sagemaker_session=sagemaker_session
    )


@pytest.fixture()
def model(sagemaker_session):
    return Model(MODEL_IMAGE, MODEL_DATA, role=IR_ROLE_ARN, sagemaker_session=sagemaker_session)


@pytest.fixture()
def default_right_sized_model(model_package):
    return model_package.right_size(
        sample_payload_url=IR_SAMPLE_PAYLOAD_URL,
        supported_content_types=IR_SUPPORTED_CONTENT_TYPES,
        supported_instance_types=[IR_SAMPLE_INSTANCE_TYPE],
        job_name=IR_JOB_NAME,
        framework=IR_SAMPLE_FRAMEWORK,
    )


def test_right_size_default_with_model_package_successful(sagemaker_session, model_package):
    inference_recommender_model_pkg = model_package.right_size(
        sample_payload_url=IR_SAMPLE_PAYLOAD_URL,
        supported_content_types=IR_SUPPORTED_CONTENT_TYPES,
        supported_instance_types=[IR_SAMPLE_INSTANCE_TYPE],
        job_name=IR_JOB_NAME,
        framework=IR_SAMPLE_FRAMEWORK,
    )

    # assert that the create api has been called with default parameters
    assert sagemaker_session.create_inference_recommendations_job.called_with(
        role=IR_ROLE_ARN,
        job_name=IR_JOB_NAME,
        job_type="Default",
        job_duration_in_seconds=None,
        model_package_version_arn=model_package.model_package_arn,
        framework=IR_SAMPLE_FRAMEWORK,
        framework_version=None,
        sample_payload_url=IR_SAMPLE_PAYLOAD_URL,
        supported_content_types=IR_SUPPORTED_CONTENT_TYPES,
        supported_instance_types=[IR_SAMPLE_INSTANCE_TYPE],
        endpoint_configurations=None,
        traffic_pattern=None,
        stopping_conditions=None,
        resource_limit=None,
    )

    assert sagemaker_session.wait_for_inference_recomendations_job.called_with(IR_JOB_NAME)

    # confirm that the IR instance attributes have been set
    assert (
        inference_recommender_model_pkg.inference_recommender_job_results
        == IR_SAMPLE_INFERENCE_RESPONSE
    )
    assert (
        inference_recommender_model_pkg.inference_recommendations
        == IR_SAMPLE_INFERENCE_RESPONSE["InferenceRecommendations"]
    )

    # confirm that the returned object of right_size is itself
    assert inference_recommender_model_pkg == model_package


def test_right_size_advanced_list_instances_model_package_successful(
    sagemaker_session, model_package
):
    inference_recommender_model_pkg = model_package.right_size(
        sample_payload_url=IR_SAMPLE_PAYLOAD_URL,
        supported_content_types=IR_SUPPORTED_CONTENT_TYPES,
        framework="SAGEMAKER-SCIKIT-LEARN",
        job_duration_in_seconds=7200,
        hyperparameter_ranges=IR_SAMPLE_LIST_OF_INSTANCES_HYPERPARAMETER_RANGES,
        phases=IR_SAMPLE_PHASES,
        traffic_type="PHASES",
        max_invocations=100,
        model_latency_thresholds=IR_SAMPLE_MODEL_LATENCY_THRESHOLDS,
        max_tests=5,
        max_parallel_tests=5,
    )

    # assert that the create api has been called with advanced parameters
    assert sagemaker_session.create_inference_recommendations_job.called_with(
        role=IR_ROLE_ARN,
        job_name=IR_JOB_NAME,
        job_type="Advanced",
        job_duration_in_seconds=7200,
        model_package_version_arn=model_package.model_package_arn,
        framework=IR_SAMPLE_FRAMEWORK,
        framework_version=None,
        sample_payload_url=IR_SAMPLE_PAYLOAD_URL,
        supported_content_types=IR_SUPPORTED_CONTENT_TYPES,
        supported_instance_types=[IR_SAMPLE_INSTANCE_TYPE],
        endpoint_configurations=IR_SAMPLE_ENDPOINT_CONFIG,
        traffic_pattern=IR_SAMPLE_TRAFFIC_PATTERN,
        stopping_conditions=IR_SAMPLE_STOPPING_CONDITIONS,
        resource_limit=IR_SAMPLE_RESOURCE_LIMIT,
    )

    assert sagemaker_session.wait_for_inference_recomendations_job.called_with(IR_JOB_NAME)

    # confirm that the IR instance attributes have been set
    assert (
        inference_recommender_model_pkg.inference_recommender_job_results
        == IR_SAMPLE_INFERENCE_RESPONSE
    )
    assert (
        inference_recommender_model_pkg.inference_recommendations
        == IR_SAMPLE_INFERENCE_RESPONSE["InferenceRecommendations"]
    )

    # confirm that the returned object of right_size is itself
    assert inference_recommender_model_pkg == model_package


def test_right_size_advanced_single_instances_model_package_successful(
    sagemaker_session, model_package
):
    model_package.right_size(
        sample_payload_url=IR_SAMPLE_PAYLOAD_URL,
        supported_content_types=IR_SUPPORTED_CONTENT_TYPES,
        framework="SAGEMAKER-SCIKIT-LEARN",
        job_duration_in_seconds=7200,
        hyperparameter_ranges=IR_SAMPLE_SINGLE_INSTANCES_HYPERPARAMETER_RANGES,
        phases=IR_SAMPLE_PHASES,
        traffic_type="PHASES",
        max_invocations=100,
        model_latency_thresholds=IR_SAMPLE_MODEL_LATENCY_THRESHOLDS,
        max_tests=5,
        max_parallel_tests=5,
    )

    # assert that the create api has been called with advanced parameters
    assert sagemaker_session.create_inference_recommendations_job.called_with(
        role=IR_ROLE_ARN,
        job_name=IR_JOB_NAME,
        job_type="Advanced",
        job_duration_in_seconds=7200,
        model_package_version_arn=model_package.model_package_arn,
        framework=IR_SAMPLE_FRAMEWORK,
        framework_version=None,
        sample_payload_url=IR_SAMPLE_PAYLOAD_URL,
        supported_content_types=IR_SUPPORTED_CONTENT_TYPES,
        supported_instance_types=[IR_SAMPLE_INSTANCE_TYPE],
        endpoint_configurations=IR_SAMPLE_ENDPOINT_CONFIG,
        traffic_pattern=IR_SAMPLE_TRAFFIC_PATTERN,
        stopping_conditions=IR_SAMPLE_STOPPING_CONDITIONS,
        resource_limit=IR_SAMPLE_RESOURCE_LIMIT,
    )


def test_right_size_advanced_model_package_partial_params_successful(
    sagemaker_session, model_package
):
    model_package.right_size(
        sample_payload_url=IR_SAMPLE_PAYLOAD_URL,
        supported_content_types=IR_SUPPORTED_CONTENT_TYPES,
        framework="SAGEMAKER-SCIKIT-LEARN",
        job_duration_in_seconds=7200,
        hyperparameter_ranges=IR_SAMPLE_SINGLE_INSTANCES_HYPERPARAMETER_RANGES,
        phases=IR_SAMPLE_PHASES,
        traffic_type="PHASES",
        max_invocations=100,
        model_latency_thresholds=IR_SAMPLE_MODEL_LATENCY_THRESHOLDS,
    )

    # assert that the create api has been called with advanced parameters
    assert sagemaker_session.create_inference_recommendations_job.called_with(
        role=IR_ROLE_ARN,
        job_name=IR_JOB_NAME,
        job_type="Advanced",
        job_duration_in_seconds=7200,
        model_package_version_arn=model_package.model_package_arn,
        framework=IR_SAMPLE_FRAMEWORK,
        framework_version=None,
        sample_payload_url=IR_SAMPLE_PAYLOAD_URL,
        supported_content_types=IR_SUPPORTED_CONTENT_TYPES,
        supported_instance_types=[IR_SAMPLE_INSTANCE_TYPE],
        endpoint_configurations=IR_SAMPLE_ENDPOINT_CONFIG,
        traffic_pattern=IR_SAMPLE_TRAFFIC_PATTERN,
        stopping_conditions=IR_SAMPLE_STOPPING_CONDITIONS,
        resource_limit=None,
    )


def test_right_size_invalid_hyperparameter_ranges(sagemaker_session, model_package):
    with pytest.raises(
        ValueError,
        match="instance_type must be defined as a hyperparameter_range",
    ):
        model_package.right_size(
            sample_payload_url=IR_SAMPLE_PAYLOAD_URL,
            supported_content_types=IR_SUPPORTED_CONTENT_TYPES,
            framework="SAGEMAKER-SCIKIT-LEARN",
            job_duration_in_seconds=7200,
            hyperparameter_ranges=IR_SAMPLE_INVALID_HYPERPARAMETERS_RANGES,
            phases=IR_SAMPLE_PHASES,
            traffic_type="PHASES",
            max_invocations=100,
            model_latency_thresholds=IR_SAMPLE_MODEL_LATENCY_THRESHOLDS,
            max_tests=5,
            max_parallel_tests=5,
        )


# TODO -> removed once model registry is decoupled
def test_right_size_missing_model_package_arn(sagemaker_session, model):
    with pytest.raises(
        ValueError,
        match="right_size\\(\\) is currently only supported with a registered model",
    ):
        model.right_size(
            sample_payload_url=IR_SAMPLE_PAYLOAD_URL,
            supported_content_types=IR_SUPPORTED_CONTENT_TYPES,
            supported_instance_types=[IR_SAMPLE_INSTANCE_TYPE],
            job_name=IR_JOB_NAME,
            framework=IR_SAMPLE_FRAMEWORK,
        )


# TODO check our framework mapping when we add in inference_recommendation_id support


@patch("sagemaker.production_variant")
@patch("sagemaker.utils.name_from_base", return_value=MODEL_NAME)
def test_deploy_right_size_with_model_package_succeeds(
    production_variant, default_right_sized_model
):
    default_right_sized_model.deploy(endpoint_name=IR_DEPLOY_ENDPOINT_NAME)

    assert production_variant.called_with(
        model_name=MODEL_NAME,
        instance_type=IR_RIGHT_SIZE_INSTANCE_TYPE,
        initial_instance_count=IR_RIGHT_SIZE_INITIAL_INSTANCE_COUNT,
        accelerator_type=None,
        serverless_inference_config=None,
        volume_size=None,
        model_data_download_timeout=None,
        container_startup_health_check_timeout=None,
    )


@patch("sagemaker.production_variant")
@patch("sagemaker.utils.name_from_base", return_value=MODEL_NAME)
def test_deploy_right_size_with_both_overrides_succeeds(
    production_variant, default_right_sized_model
):
    default_right_sized_model.deploy(
        instance_type="ml.c5.2xlarge",
        initial_instance_count=5,
        endpoint_name=IR_DEPLOY_ENDPOINT_NAME,
    )

    assert production_variant.called_with(
        model_name=MODEL_NAME,
        instance_type="ml.c5.2xlarge",
        initial_instance_count=5,
        accelerator_type=None,
        serverless_inference_config=None,
        volume_size=None,
        model_data_download_timeout=None,
        container_startup_health_check_timeout=None,
    )


def test_deploy_right_size_instance_type_override_fails(default_right_sized_model):
    with pytest.raises(
        ValueError,
        match="Must specify instance type and instance count unless using serverless inference",
    ):
        default_right_sized_model.deploy(
            instance_type="ml.c5.2xlarge",
            endpoint_name=IR_DEPLOY_ENDPOINT_NAME,
        )


def test_deploy_right_size_initial_instance_count_override_fails(default_right_sized_model):
    with pytest.raises(
        ValueError,
        match="Must specify instance type and instance count unless using serverless inference",
    ):
        default_right_sized_model.deploy(
            initial_instance_count=2,
            endpoint_name=IR_DEPLOY_ENDPOINT_NAME,
        )


def test_deploy_right_size_accelerator_type_fails(default_right_sized_model):
    with pytest.raises(
        ValueError,
        match="accelerator_type is not compatible with right_size\\(\\).",
    ):
        default_right_sized_model.deploy(accelerator_type="ml.eia.medium")


@patch("sagemaker.production_variant")
@patch("sagemaker.utils.name_from_base", return_value=MODEL_NAME)
def test_deploy_right_size_serverless_override(production_variant, default_right_sized_model):
    serverless_inference_config = ServerlessInferenceConfig()
    default_right_sized_model.deploy(serverless_inference_config=serverless_inference_config)

    assert production_variant.called_with(
        model_name=MODEL_NAME,
        instance_type=None,
        initial_instance_count=None,
        accelerator_type=None,
        serverless_inference_config=serverless_inference_config._to_request_dict,
        volume_size=None,
        model_data_download_timeout=None,
        container_startup_health_check_timeout=None,
    )


@patch("sagemaker.utils.name_from_base", return_value=MODEL_NAME)
def test_deploy_right_size_async_override(sagemaker_session, default_right_sized_model):
    async_inference_config = AsyncInferenceConfig(output_path="s3://some-path")
    default_right_sized_model.deploy(
        instance_type="ml.c5.2xlarge",
        initial_instance_count=1,
        async_inference_config=async_inference_config,
    )

    assert sagemaker_session.endpoint_from_production_variants.called_with(
        name=MODEL_NAME,
        production_variants=[ANY],
        tags=None,
        kms_key=None,
        wait=None,
        data_capture_config_dict=None,
        async_inference_config_dict=async_inference_config._to_request_dict,
    )


# TODO -> cover inference_recommendation_id cases
# ...
