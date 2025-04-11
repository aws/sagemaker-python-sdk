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

import copy

import pytest
from mock import Mock, patch

import sagemaker
from sagemaker.model import Model, ModelPackage
from sagemaker.async_inference import AsyncInferenceConfig
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.explainer import ExplainerConfig
from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements
from sagemaker.enums import EndpointType
from tests.unit.sagemaker.inference_recommender.constants import (
    DESCRIBE_COMPILATION_JOB_RESPONSE,
    DESCRIBE_MODEL_PACKAGE_RESPONSE,
    DESCRIBE_MODEL_RESPONSE,
    MODEL_RECOMMENDATION_ENV,
    MODEL_RECOMMENDATION_ID,
    INVALID_RECOMMENDATION_ID,
    IR_COMPILATION_JOB_NAME,
    IR_ENV,
    IR_IMAGE,
    IR_MODEL_DATA,
    IR_MODEL_NAME,
    IR_MODEL_PACKAGE_VERSION_ARN,
    IR_COMPILATION_IMAGE,
    IR_COMPILATION_MODEL_DATA,
    NOT_EXISTED_MODEL_RECOMMENDATION_ID,
    RECOMMENDATION_ID,
    NOT_EXISTED_RECOMMENDATION_ID,
    IR_CONTAINER_DEF,
    DEPLOYMENT_RECOMMENDATION_CONTAINER_DEF,
    IR_COMPILATION_CONTAINER_DEF,
    IR_MODEL_PACKAGE_CONTAINER_DEF,
    IR_COMPILATION_MODEL_PACKAGE_CONTAINER_DEF,
)
from tests.unit.sagemaker.inference_recommender.constructs import (
    create_inference_recommendations_job_default_with_model_name,
    create_inference_recommendations_job_default_with_model_name_and_compilation,
    create_inference_recommendations_job_default_with_model_package_arn,
    create_inference_recommendations_job_default_with_model_package_arn_and_compilation,
)

MODEL_DATA = "s3://bucket/model.tar.gz"
MODEL_IMAGE = "mi"
TIMESTAMP = "2020-07-02-20-10-30-288"
MODEL_NAME = "{}-{}".format(MODEL_IMAGE, TIMESTAMP)
ENDPOINT_NAME = "endpoint-{}".format(TIMESTAMP)

ACCELERATOR_TYPE = "ml.eia.medium"
INSTANCE_COUNT = 2
INSTANCE_TYPE = "ml.c4.4xlarge"
INFERENCE_RECOMMENDATION_ID = "ir-job/6ab0ff22"
ROLE = "some-role"

BASE_PRODUCTION_VARIANT = {
    "ModelName": MODEL_NAME,
    "InstanceType": INSTANCE_TYPE,
    "InitialInstanceCount": INSTANCE_COUNT,
    "VariantName": "AllTraffic",
    "InitialVariantWeight": 1,
}

SHAP_BASELINE = '1,2,3,"good product"'
CSV_MIME_TYPE = "text/csv"
BUCKET_NAME = "mybucket"
RESOURCES = ResourceRequirements(
    requests={
        "num_cpus": 1,  # NumberOfCpuCoresRequired
        "memory": 1024,  # MinMemoryRequiredInMb (required)
        "copies": 1,
    },
    limits={},
)


@pytest.fixture
def sagemaker_session():
    session = Mock(
        default_bucket_prefix=None,
    )
    session.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    # For tests which doesn't verify config file injection, operate with empty config
    session.sagemaker_config = {}
    return session


@patch("sagemaker.production_variant")
@patch("sagemaker.model.Model.prepare_container_def")
@patch("sagemaker.utils.name_from_base", return_value=MODEL_NAME)
def test_deploy(name_from_base, prepare_container_def, production_variant, sagemaker_session):
    production_variant.return_value = BASE_PRODUCTION_VARIANT

    container_def = {"Image": MODEL_IMAGE, "Environment": {}, "ModelDataUrl": MODEL_DATA}
    prepare_container_def.return_value = container_def

    model = Model(MODEL_IMAGE, MODEL_DATA, role=ROLE, sagemaker_session=sagemaker_session)
    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=INSTANCE_COUNT)

    name_from_base.assert_called_with(MODEL_IMAGE)
    assert 2 == name_from_base.call_count

    prepare_container_def.assert_called_with(
        INSTANCE_TYPE,
        accelerator_type=None,
        serverless_inference_config=None,
        accept_eula=None,
        model_reference_arn=None,
    )
    production_variant.assert_called_with(
        MODEL_NAME,
        INSTANCE_TYPE,
        INSTANCE_COUNT,
        accelerator_type=None,
        serverless_inference_config=None,
        volume_size=None,
        model_data_download_timeout=None,
        container_startup_health_check_timeout=None,
        routing_config=None,
        inference_ami_version=None,
    )

    sagemaker_session.create_model.assert_called_with(
        name=MODEL_NAME,
        role=ROLE,
        container_defs=container_def,
        vpc_config=None,
        enable_network_isolation=False,
        tags=None,
    )

    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=MODEL_NAME,
        production_variants=[BASE_PRODUCTION_VARIANT],
        tags=None,
        kms_key=None,
        wait=True,
        explainer_config_dict=None,
        data_capture_config_dict=None,
        async_inference_config_dict=None,
        live_logging=False,
    )


@patch("sagemaker.utils.name_from_base", return_value=ENDPOINT_NAME)
@patch("sagemaker.model.Model._create_sagemaker_model")
@patch("sagemaker.production_variant")
def test_deploy_accelerator_type(
    production_variant, create_sagemaker_model, name_from_base, sagemaker_session
):
    model = Model(
        MODEL_IMAGE, MODEL_DATA, role=ROLE, name=MODEL_NAME, sagemaker_session=sagemaker_session
    )

    production_variant_result = copy.deepcopy(BASE_PRODUCTION_VARIANT)
    production_variant_result["AcceleratorType"] = ACCELERATOR_TYPE
    production_variant.return_value = production_variant_result

    model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
        accelerator_type=ACCELERATOR_TYPE,
    )

    create_sagemaker_model.assert_called_with(
        instance_type=INSTANCE_TYPE,
        accelerator_type=ACCELERATOR_TYPE,
        tags=None,
        serverless_inference_config=None,
        accept_eula=None,
        model_reference_arn=None,
    )
    production_variant.assert_called_with(
        MODEL_NAME,
        INSTANCE_TYPE,
        INSTANCE_COUNT,
        accelerator_type=ACCELERATOR_TYPE,
        serverless_inference_config=None,
        volume_size=None,
        model_data_download_timeout=None,
        container_startup_health_check_timeout=None,
        routing_config=None,
        inference_ami_version=None,
    )

    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=ENDPOINT_NAME,
        production_variants=[production_variant_result],
        tags=None,
        kms_key=None,
        wait=True,
        explainer_config_dict=None,
        data_capture_config_dict=None,
        async_inference_config_dict=None,
        live_logging=False,
    )


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
def test_deploy_endpoint_name(sagemaker_session):
    sagemaker_session.sagemaker_config = {}
    model = Model(MODEL_IMAGE, MODEL_DATA, role=ROLE, sagemaker_session=sagemaker_session)

    endpoint_name = "blah"
    model.deploy(
        endpoint_name=endpoint_name,
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
    )

    assert endpoint_name == model.endpoint_name
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=endpoint_name,
        production_variants=[BASE_PRODUCTION_VARIANT],
        tags=None,
        kms_key=None,
        wait=True,
        explainer_config_dict=None,
        data_capture_config_dict=None,
        async_inference_config_dict=None,
        live_logging=False,
    )


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.utils.name_from_base")
@patch("sagemaker.utils.base_from_name")
@patch("sagemaker.production_variant")
def test_deploy_generates_endpoint_name_each_time_from_model_name(
    production_variant, base_from_name, name_from_base, sagemaker_session
):
    model = Model(
        MODEL_IMAGE, MODEL_DATA, name=MODEL_NAME, role=ROLE, sagemaker_session=sagemaker_session
    )

    model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
    )
    model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
    )

    base_from_name.assert_called_with(MODEL_NAME)
    name_from_base.assert_called_with(base_from_name.return_value)
    assert 2 == name_from_base.call_count


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.utils.name_from_base")
@patch("sagemaker.utils.base_from_name")
@patch("sagemaker.production_variant")
def test_deploy_generates_endpoint_name_each_time_from_base_name(
    production_variant, base_from_name, name_from_base, sagemaker_session
):
    model = Model(MODEL_IMAGE, MODEL_DATA, role=ROLE, sagemaker_session=sagemaker_session)

    base_name = "foo"
    model._base_name = base_name

    model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
    )
    model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
    )

    base_from_name.assert_not_called()
    name_from_base.assert_called_with(base_name)
    assert 2 == name_from_base.call_count


@patch("sagemaker.utils.name_from_base", return_value=ENDPOINT_NAME)
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
@patch("sagemaker.model.Model._create_sagemaker_model")
def test_deploy_tags(create_sagemaker_model, production_variant, name_from_base, sagemaker_session):
    model = Model(
        MODEL_IMAGE, MODEL_DATA, role=ROLE, name=MODEL_NAME, sagemaker_session=sagemaker_session
    )

    tags = [{"Key": "ModelName", "Value": "TestModel"}]
    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=INSTANCE_COUNT, tags=tags)

    create_sagemaker_model.assert_called_with(
        instance_type=INSTANCE_TYPE,
        accelerator_type=None,
        tags=tags,
        serverless_inference_config=None,
        accept_eula=None,
        model_reference_arn=None,
    )
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=ENDPOINT_NAME,
        production_variants=[BASE_PRODUCTION_VARIANT],
        tags=tags,
        kms_key=None,
        wait=True,
        explainer_config_dict=None,
        data_capture_config_dict=None,
        async_inference_config_dict=None,
        live_logging=False,
    )


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.utils.name_from_base", return_value=ENDPOINT_NAME)
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
def test_deploy_kms_key(production_variant, name_from_base, sagemaker_session):
    model = Model(
        MODEL_IMAGE, MODEL_DATA, role=ROLE, name=MODEL_NAME, sagemaker_session=sagemaker_session
    )

    key = "some-key-arn"
    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=INSTANCE_COUNT, kms_key=key)

    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=ENDPOINT_NAME,
        production_variants=[BASE_PRODUCTION_VARIANT],
        tags=None,
        kms_key=key,
        wait=True,
        explainer_config_dict=None,
        data_capture_config_dict=None,
        async_inference_config_dict=None,
        live_logging=False,
    )


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.utils.name_from_base", return_value=ENDPOINT_NAME)
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
def test_deploy_async(production_variant, name_from_base, sagemaker_session):
    model = Model(
        MODEL_IMAGE, MODEL_DATA, role=ROLE, name=MODEL_NAME, sagemaker_session=sagemaker_session
    )

    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=INSTANCE_COUNT, wait=False)

    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=ENDPOINT_NAME,
        production_variants=[BASE_PRODUCTION_VARIANT],
        tags=None,
        kms_key=None,
        wait=False,
        explainer_config_dict=None,
        data_capture_config_dict=None,
        async_inference_config_dict=None,
        live_logging=False,
    )


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.utils.name_from_base", return_value=ENDPOINT_NAME)
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
def test_deploy_data_capture_config(production_variant, name_from_base, sagemaker_session):
    model = Model(
        MODEL_IMAGE, MODEL_DATA, role=ROLE, name=MODEL_NAME, sagemaker_session=sagemaker_session
    )

    data_capture_config = Mock()
    data_capture_config_dict = {"EnableCapture": True}
    data_capture_config._to_request_dict.return_value = data_capture_config_dict
    model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
        data_capture_config=data_capture_config,
    )

    data_capture_config._to_request_dict.assert_called_with()
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=ENDPOINT_NAME,
        production_variants=[BASE_PRODUCTION_VARIANT],
        tags=None,
        kms_key=None,
        wait=True,
        explainer_config_dict=None,
        data_capture_config_dict=data_capture_config_dict,
        async_inference_config_dict=None,
        live_logging=False,
    )


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.utils.name_from_base", return_value=ENDPOINT_NAME)
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
def test_deploy_explainer_config(production_variant, name_from_base, sagemaker_session):
    model = Model(
        MODEL_IMAGE, MODEL_DATA, role=ROLE, name=MODEL_NAME, sagemaker_session=sagemaker_session
    )

    mock_clarify_explainer_config = Mock()
    mock_clarify_explainer_config_dict = {
        "EnableExplanations": "`true`",
    }
    mock_clarify_explainer_config._to_request_dict.return_value = mock_clarify_explainer_config_dict
    explainer_config = ExplainerConfig(clarify_explainer_config=mock_clarify_explainer_config)
    explainer_config_dict = {"ClarifyExplainerConfig": mock_clarify_explainer_config_dict}

    model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
        explainer_config=explainer_config,
    )

    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=ENDPOINT_NAME,
        production_variants=[BASE_PRODUCTION_VARIANT],
        tags=None,
        kms_key=None,
        wait=True,
        explainer_config_dict=explainer_config_dict,
        data_capture_config_dict=None,
        async_inference_config_dict=None,
        live_logging=False,
    )


def test_deploy_wrong_explainer_config(sagemaker_session):
    model = Model(MODEL_IMAGE, MODEL_DATA, sagemaker_session=sagemaker_session, role=ROLE)

    with pytest.raises(ValueError, match="explainer_config needs to be a ExplainerConfig object"):
        model.deploy(
            instance_type=INSTANCE_TYPE,
            initial_instance_count=INSTANCE_COUNT,
            explainer_config={},
        )


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.utils.name_from_base", return_value=ENDPOINT_NAME)
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
def test_deploy_async_inference(production_variant, name_from_base, sagemaker_session):
    S3_OUTPUT_PATH = "s3://some-output-path"
    S3_FAILURE_PATH = "s3://some-failure-path"

    model = Model(
        MODEL_IMAGE, MODEL_DATA, role=ROLE, name=MODEL_NAME, sagemaker_session=sagemaker_session
    )

    async_inference_config = AsyncInferenceConfig(
        output_path=S3_OUTPUT_PATH, failure_path=S3_FAILURE_PATH
    )

    async_inference_config_dict = {
        "OutputConfig": {"S3OutputPath": S3_OUTPUT_PATH, "S3FailurePath": S3_FAILURE_PATH},
    }

    model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
        async_inference_config=async_inference_config,
    )

    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=ENDPOINT_NAME,
        production_variants=[BASE_PRODUCTION_VARIANT],
        tags=None,
        kms_key=None,
        wait=True,
        explainer_config_dict=None,
        data_capture_config_dict=None,
        async_inference_config_dict=async_inference_config_dict,
        live_logging=False,
    )


@patch("sagemaker.utils.name_from_base", return_value=ENDPOINT_NAME)
@patch("sagemaker.model.Model._create_sagemaker_model")
@patch("sagemaker.production_variant")
def test_deploy_serverless_inference(production_variant, create_sagemaker_model, sagemaker_session):
    sagemaker_session.sagemaker_config = {}
    model = Model(
        MODEL_IMAGE, MODEL_DATA, role=ROLE, name=MODEL_NAME, sagemaker_session=sagemaker_session
    )

    production_variant_result = copy.deepcopy(BASE_PRODUCTION_VARIANT)
    production_variant.return_value = production_variant_result

    serverless_inference_config = ServerlessInferenceConfig()
    serverless_inference_config_dict = {
        "MemorySizeInMB": 2048,
        "MaxConcurrency": 5,
    }

    model.deploy(
        serverless_inference_config=serverless_inference_config,
    )

    create_sagemaker_model.assert_called_with(
        instance_type=None,
        accelerator_type=None,
        tags=None,
        serverless_inference_config=serverless_inference_config,
        accept_eula=None,
        model_reference_arn=None,
    )
    production_variant.assert_called_with(
        MODEL_NAME,
        None,
        None,
        accelerator_type=None,
        serverless_inference_config=serverless_inference_config_dict,
        volume_size=None,
        model_data_download_timeout=None,
        container_startup_health_check_timeout=None,
        routing_config=None,
        inference_ami_version=None,
    )

    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=ENDPOINT_NAME,
        production_variants=[production_variant_result],
        tags=None,
        kms_key=None,
        wait=True,
        explainer_config_dict=None,
        data_capture_config_dict=None,
        async_inference_config_dict=None,
        live_logging=False,
    )


def test_deploy_wrong_inference_type(sagemaker_session):
    model = Model(MODEL_IMAGE, MODEL_DATA, role=ROLE)

    bad_args = (
        {"instance_type": INSTANCE_TYPE},
        {"initial_instance_count": INSTANCE_COUNT},
        {"instance_type": None, "initial_instance_count": None},
    )
    for args in bad_args:
        with pytest.raises(
            ValueError,
            match="Must specify instance type and instance count unless using serverless inference",
        ):
            model.deploy(args)


def test_deploy_wrong_serverless_config(sagemaker_session):
    model = Model(MODEL_IMAGE, MODEL_DATA, role=ROLE)
    with pytest.raises(
        ValueError,
        match="serverless_inference_config needs to be a ServerlessInferenceConfig object",
    ):
        model.deploy(serverless_inference_config={})


@patch("sagemaker.session.Session")
@patch("sagemaker.local.LocalSession")
def test_deploy_creates_correct_session(local_session, session):
    local_session.return_value.sagemaker_config = {}
    session.return_value.sagemaker_config = {}
    # We expect a LocalSession when deploying to instance_type = 'local'
    model = Model(MODEL_IMAGE, MODEL_DATA, role=ROLE)
    model.deploy(endpoint_name="blah", instance_type="local", initial_instance_count=1)
    assert model.sagemaker_session == local_session.return_value

    # We expect a real Session when deploying to instance_type != local/local_gpu
    model = Model(MODEL_IMAGE, MODEL_DATA, role=ROLE)
    model.deploy(
        endpoint_name="remote_endpoint", instance_type="ml.m4.4xlarge", initial_instance_count=2
    )
    assert model.sagemaker_session == session.return_value


def test_deploy_no_role(sagemaker_session):
    model = Model(MODEL_IMAGE, MODEL_DATA, sagemaker_session=sagemaker_session)

    with pytest.raises(ValueError, match="Role can not be null for deploying a model"):
        model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=INSTANCE_COUNT)


def test_deploy_wrong_async_inferenc_config(sagemaker_session):
    model = Model(MODEL_IMAGE, MODEL_DATA, sagemaker_session=sagemaker_session, role=ROLE)

    with pytest.raises(
        ValueError, match="async_inference_config needs to be a AsyncInferenceConfig object"
    ):
        model.deploy(
            instance_type=INSTANCE_TYPE,
            initial_instance_count=INSTANCE_COUNT,
            async_inference_config={},
        )


def test_deploy_ir_with_incompatible_parameters(sagemaker_session):
    sagemaker_session.sagemaker_client.describe_inference_recommendations_job.return_value = (
        create_inference_recommendations_job_default_with_model_package_arn()
    )
    sagemaker_session.sagemaker_client.describe_model.return_value = None

    model = Model(MODEL_IMAGE, MODEL_DATA, sagemaker_session=sagemaker_session, role=ROLE)

    with pytest.raises(
        ValueError,
        match="instance_type and initial_instance_count are mutually exclusive with"
        "recommendation id since they are in recommendation."
        "Please specify both of them if you want to override the recommendation.",
    ):
        model.deploy(
            instance_type=INSTANCE_TYPE,
            inference_recommendation_id=RECOMMENDATION_ID,
        )

    with pytest.raises(
        ValueError,
        match="instance_type and initial_instance_count are mutually exclusive with"
        "recommendation id since they are in recommendation."
        "Please specify both of them if you want to override the recommendation.",
    ):
        model.deploy(
            initial_instance_count=INSTANCE_COUNT,
            inference_recommendation_id=RECOMMENDATION_ID,
        )

    with pytest.raises(
        ValueError, match="accelerator_type is not compatible with inference_recommendation_id"
    ):
        model.deploy(
            accelerator_type=ACCELERATOR_TYPE,
            inference_recommendation_id=INFERENCE_RECOMMENDATION_ID,
        )

    with pytest.raises(
        ValueError,
        match="async_inference_config is not compatible with inference_recommendation_id",
    ):
        model.deploy(
            async_inference_config=AsyncInferenceConfig(),
            inference_recommendation_id=INFERENCE_RECOMMENDATION_ID,
        )

    with pytest.raises(
        ValueError,
        match="serverless_inference_config is not compatible with inference_recommendation_id",
    ):
        model.deploy(
            serverless_inference_config=ServerlessInferenceConfig(),
            inference_recommendation_id=INFERENCE_RECOMMENDATION_ID,
        )


def test_deploy_with_wrong_recommendation_id(sagemaker_session):
    model = Model(MODEL_IMAGE, MODEL_DATA, sagemaker_session=sagemaker_session, role=ROLE)

    with pytest.raises(ValueError, match="inference_recommendation_id is not valid"):
        model.deploy(
            inference_recommendation_id=INVALID_RECOMMENDATION_ID,
        )


def mock_describe_model_package(ModelPackageName):
    if ModelPackageName == IR_MODEL_PACKAGE_VERSION_ARN:
        return DESCRIBE_MODEL_PACKAGE_RESPONSE


@patch("sagemaker.utils.name_from_base", return_value=IR_IMAGE)
def test_deploy_with_recommendation_id_with_model_pkg_arn(name_from_base, sagemaker_session):
    sagemaker_session.sagemaker_client.describe_inference_recommendations_job.return_value = (
        create_inference_recommendations_job_default_with_model_package_arn()
    )
    sagemaker_session.sagemaker_client.describe_model_package.side_effect = (
        mock_describe_model_package
    )
    sagemaker_session.sagemaker_client.describe_model.return_value = None

    model_package = ModelPackage(
        role=ROLE, model_data=IR_MODEL_DATA, model_package_arn=IR_MODEL_PACKAGE_VERSION_ARN
    )

    model_package.sagemaker_session = sagemaker_session

    model_package.deploy(
        inference_recommendation_id=RECOMMENDATION_ID,
    )

    sagemaker_session.create_model.assert_called_with(
        IR_IMAGE,
        ROLE,
        IR_MODEL_PACKAGE_CONTAINER_DEF,
        vpc_config=None,
        enable_network_isolation=False,
        tags=None,
    )

    assert model_package.model_package_arn == IR_MODEL_PACKAGE_VERSION_ARN
    assert model_package.model_data == IR_MODEL_DATA
    assert model_package.image_uri == IR_IMAGE
    assert model_package.env == IR_ENV


def mock_describe_model(ModelName):
    if ModelName == IR_MODEL_NAME:
        return DESCRIBE_MODEL_RESPONSE


@patch("sagemaker.utils.name_from_base", return_value=MODEL_IMAGE)
def test_deploy_with_recommendation_id_with_model_name(name_from_base, sagemaker_session):
    sagemaker_session.sagemaker_client.describe_inference_recommendations_job.return_value = (
        create_inference_recommendations_job_default_with_model_name()
    )
    sagemaker_session.sagemaker_client.describe_model.side_effect = mock_describe_model

    model = Model(MODEL_IMAGE, MODEL_DATA, sagemaker_session=sagemaker_session, role=ROLE)

    model.deploy(
        inference_recommendation_id=RECOMMENDATION_ID,
    )

    sagemaker_session.create_model.assert_called_with(
        name=MODEL_IMAGE,
        role=ROLE,
        container_defs=IR_CONTAINER_DEF,
        vpc_config=None,
        enable_network_isolation=False,
        tags=None,
    )

    assert model.model_data == IR_MODEL_DATA
    assert model.image_uri == IR_IMAGE
    assert model.env == IR_ENV


@patch("sagemaker.utils.name_from_base", return_value=IR_COMPILATION_IMAGE)
def test_deploy_with_recommendation_id_with_model_pkg_arn_and_compilation(
    name_from_base, sagemaker_session
):
    sagemaker_session.sagemaker_client.describe_inference_recommendations_job.return_value = (
        create_inference_recommendations_job_default_with_model_package_arn_and_compilation()
    )
    sagemaker_session.sagemaker_client.describe_model_package.side_effect = (
        mock_describe_model_package
    )
    sagemaker_session.sagemaker_client.describe_model.return_value = None

    model_package = ModelPackage(
        role=ROLE, model_data=MODEL_DATA, model_package_arn=IR_MODEL_PACKAGE_VERSION_ARN
    )

    model_package.sagemaker_session = sagemaker_session

    model_package.deploy(
        inference_recommendation_id=RECOMMENDATION_ID,
    )

    sagemaker_session.create_model.assert_called_with(
        IR_COMPILATION_IMAGE,
        ROLE,
        IR_COMPILATION_MODEL_PACKAGE_CONTAINER_DEF,
        vpc_config=None,
        enable_network_isolation=False,
        tags=None,
    )

    assert model_package.model_data == IR_COMPILATION_MODEL_DATA
    assert model_package.image_uri == IR_COMPILATION_IMAGE


@patch("sagemaker.utils.name_from_base", return_value=MODEL_IMAGE)
def test_deploy_with_recommendation_id_with_model_name_and_compilation(
    name_from_base, sagemaker_session
):
    def mock_describe_compilation_job(CompilationJobName):
        if CompilationJobName == IR_COMPILATION_JOB_NAME:
            return DESCRIBE_COMPILATION_JOB_RESPONSE

    sagemaker_session.sagemaker_client.describe_inference_recommendations_job.return_value = (
        create_inference_recommendations_job_default_with_model_name_and_compilation()
    )
    sagemaker_session.sagemaker_client.describe_compilation_job.side_effect = (
        mock_describe_compilation_job
    )
    sagemaker_session.sagemaker_client.describe_model.side_effect = mock_describe_model

    model = Model(MODEL_IMAGE, MODEL_DATA, sagemaker_session=sagemaker_session, role=ROLE)

    model.deploy(
        inference_recommendation_id=RECOMMENDATION_ID,
    )

    sagemaker_session.create_model.assert_called_with(
        name=MODEL_IMAGE,
        role=ROLE,
        container_defs=IR_COMPILATION_CONTAINER_DEF,
        vpc_config=None,
        enable_network_isolation=False,
        tags=None,
    )

    assert model.model_data == IR_COMPILATION_MODEL_DATA
    assert model.image_uri == IR_COMPILATION_IMAGE


def test_deploy_with_invalid_inference_recommendation_id(sagemaker_session):
    sagemaker_session.sagemaker_client.describe_inference_recommendations_job.return_value = (
        create_inference_recommendations_job_default_with_model_name_and_compilation()
    )
    sagemaker_session.sagemaker_client.describe_compilation_job.return_value = (
        DESCRIBE_COMPILATION_JOB_RESPONSE
    )
    sagemaker_session.sagemaker_client.describe_model.return_value = None

    model = Model(MODEL_IMAGE, MODEL_DATA, sagemaker_session=sagemaker_session, role=ROLE)

    with pytest.raises(
        ValueError,
        match="inference_recommendation_id is not valid",
    ):
        model.deploy(
            inference_recommendation_id=NOT_EXISTED_RECOMMENDATION_ID,
        )


def test_deploy_with_invalid_model_recommendation_id(sagemaker_session):
    sagemaker_session.sagemaker_client.describe_inference_recommendations_job.return_value = None
    sagemaker_session.sagemaker_client.describe_model.side_effect = mock_describe_model

    model = Model(MODEL_IMAGE, MODEL_DATA, sagemaker_session=sagemaker_session, role=ROLE)

    with pytest.raises(
        ValueError,
        match="inference_recommendation_id is not valid",
    ):
        model.deploy(
            inference_recommendation_id=NOT_EXISTED_MODEL_RECOMMENDATION_ID,
        )


@patch("sagemaker.utils.name_from_base", return_value=IR_IMAGE)
def test_deploy_with_valid_model_recommendation_id(name_from_base, sagemaker_session):
    sagemaker_session.sagemaker_client.describe_inference_recommendations_job.return_value = None
    sagemaker_session.sagemaker_client.describe_model.side_effect = mock_describe_model

    model = Model(IR_IMAGE, IR_MODEL_DATA, sagemaker_session=sagemaker_session, role=ROLE)
    model.deploy(
        inference_recommendation_id=MODEL_RECOMMENDATION_ID,
        initial_instance_count=INSTANCE_COUNT,
    )

    sagemaker_session.create_model.assert_called_with(
        name=IR_IMAGE,
        role=ROLE,
        container_defs=DEPLOYMENT_RECOMMENDATION_CONTAINER_DEF,
        vpc_config=None,
        enable_network_isolation=False,
        tags=None,
    )

    assert model.model_data == IR_MODEL_DATA
    assert model.image_uri == IR_IMAGE
    assert model.env == MODEL_RECOMMENDATION_ENV


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.predictor.Predictor._get_endpoint_config_name", Mock())
@patch("sagemaker.predictor.Predictor._get_model_names", Mock())
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
def test_deploy_predictor_cls(production_variant, sagemaker_session):
    model = Model(
        MODEL_IMAGE,
        MODEL_DATA,
        role=ROLE,
        name=MODEL_NAME,
        predictor_cls=sagemaker.predictor.Predictor,
        sagemaker_session=sagemaker_session,
    )

    endpoint_name = "foo"
    predictor = model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
        endpoint_name=endpoint_name,
    )

    assert isinstance(predictor, sagemaker.predictor.Predictor)
    assert predictor.endpoint_name == endpoint_name
    assert predictor.sagemaker_session == sagemaker_session

    endpoint_name_async = "foo-async"
    predictor_async = model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
        endpoint_name=endpoint_name_async,
        async_inference_config=AsyncInferenceConfig(),
    )

    assert isinstance(predictor_async, sagemaker.predictor_async.AsyncPredictor)
    assert predictor_async.name == model.name
    assert predictor_async.endpoint_name == endpoint_name_async
    assert predictor_async.sagemaker_session == sagemaker_session


@patch("sagemaker.production_variant")
@patch("sagemaker.model.Model.prepare_container_def")
@patch("sagemaker.utils.name_from_base", return_value=MODEL_NAME)
def test_deploy_customized_volume_size_and_timeout(
    name_from_base, prepare_container_def, production_variant, sagemaker_session
):
    volume_size_gb = 256
    model_data_download_timeout_sec = 1800
    startup_health_check_timeout_sec = 1800

    production_variant_result = copy.deepcopy(BASE_PRODUCTION_VARIANT)
    production_variant_result.update(
        {
            "VolumeSizeInGB": volume_size_gb,
            "ModelDataDownloadTimeoutInSeconds": model_data_download_timeout_sec,
            "ContainerStartupHealthCheckTimeoutInSeconds": startup_health_check_timeout_sec,
        }
    )
    production_variant.return_value = production_variant_result

    container_def = {"Image": MODEL_IMAGE, "Environment": {}, "ModelDataUrl": MODEL_DATA}
    prepare_container_def.return_value = container_def

    model = Model(MODEL_IMAGE, MODEL_DATA, role=ROLE, sagemaker_session=sagemaker_session)
    model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
        volume_size=volume_size_gb,
        model_data_download_timeout=model_data_download_timeout_sec,
        container_startup_health_check_timeout=startup_health_check_timeout_sec,
    )

    name_from_base.assert_called_with(MODEL_IMAGE)
    assert 2 == name_from_base.call_count

    prepare_container_def.assert_called_with(
        INSTANCE_TYPE,
        accelerator_type=None,
        serverless_inference_config=None,
        accept_eula=None,
        model_reference_arn=None,
    )
    production_variant.assert_called_with(
        MODEL_NAME,
        INSTANCE_TYPE,
        INSTANCE_COUNT,
        accelerator_type=None,
        serverless_inference_config=None,
        volume_size=volume_size_gb,
        model_data_download_timeout=model_data_download_timeout_sec,
        container_startup_health_check_timeout=startup_health_check_timeout_sec,
        routing_config=None,
        inference_ami_version=None,
    )

    sagemaker_session.create_model.assert_called_with(
        name=MODEL_NAME,
        role=ROLE,
        container_defs=container_def,
        vpc_config=None,
        enable_network_isolation=False,
        tags=None,
    )

    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=MODEL_NAME,
        production_variants=[production_variant_result],
        tags=None,
        kms_key=None,
        wait=True,
        explainer_config_dict=None,
        data_capture_config_dict=None,
        async_inference_config_dict=None,
        live_logging=False,
    )


@patch("sagemaker.production_variant")
@patch("sagemaker.utils.name_from_base", return_value=ENDPOINT_NAME)
@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
def test_deploy_with_resources(sagemaker_session, name_from_base, production_variant):
    production_variant.return_value = BASE_PRODUCTION_VARIANT
    sagemaker_session.sagemaker_config = {}
    model = Model(
        MODEL_IMAGE, MODEL_DATA, name=MODEL_NAME, role=ROLE, sagemaker_session=sagemaker_session
    )

    model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
        resource=RESOURCES,
    )
    production_variant.assert_called_with(
        MODEL_NAME,
        INSTANCE_TYPE,
        INSTANCE_COUNT,
        accelerator_type=None,
        serverless_inference_config=None,
        volume_size=None,
        model_data_download_timeout=None,
        container_startup_health_check_timeout=None,
        routing_config=None,
        inference_ami_version=None,
    )
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=name_from_base(MODEL_NAME),
        production_variants=[BASE_PRODUCTION_VARIANT],
        tags=None,
        kms_key=None,
        wait=True,
        explainer_config_dict=None,
        data_capture_config_dict=None,
        async_inference_config_dict=None,
        live_logging=False,
    )


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
def test_deploy_with_name_and_resources(sagemaker_session):
    sagemaker_session.sagemaker_config = {}
    model = Model(
        MODEL_IMAGE, MODEL_DATA, name=MODEL_NAME, role=ROLE, sagemaker_session=sagemaker_session
    )

    endpoint_name = "inference-component-based-endpoint-test"
    model.deploy(
        endpoint_name=endpoint_name,
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
        resource=RESOURCES,
    )
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=endpoint_name,
        production_variants=[BASE_PRODUCTION_VARIANT],
        tags=None,
        kms_key=None,
        wait=True,
        explainer_config_dict=None,
        data_capture_config_dict=None,
        async_inference_config_dict=None,
        live_logging=False,
    )


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.utils.name_from_base", return_value=ENDPOINT_NAME)
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
def test_deploy_with_update_endpoint(production_variant, name_from_base, sagemaker_session):
    model = Model(
        MODEL_IMAGE, MODEL_DATA, role=ROLE, name=MODEL_NAME, sagemaker_session=sagemaker_session
    )

    # Mock the create_endpoint_config to return a specific config name
    endpoint_config_name = "test-config-name"
    sagemaker_session.create_endpoint_config.return_value = endpoint_config_name

    # Test update_endpoint=True scenario
    endpoint_name = "existing-endpoint"
    model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
        endpoint_name=endpoint_name,
        update_endpoint=True,
    )

    # Verify create_endpoint_config is called with correct parameters
    sagemaker_session.create_endpoint_config.assert_called_with(
        name=MODEL_NAME,
        model_name=MODEL_NAME,
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        accelerator_type=None,
        tags=None,
        kms_key=None,
        data_capture_config_dict=None,
        volume_size=None,
        model_data_download_timeout=None,
        container_startup_health_check_timeout=None,
        explainer_config_dict=None,
        async_inference_config_dict=None,
        serverless_inference_config=None,
        routing_config=None,
        inference_ami_version=None,
    )

    # Verify update_endpoint is called with correct parameters
    sagemaker_session.update_endpoint.assert_called_with(endpoint_name, endpoint_config_name)

    # Test update_endpoint with serverless config
    serverless_inference_config = ServerlessInferenceConfig()
    serverless_inference_config_dict = {
        "MemorySizeInMB": 2048,
        "MaxConcurrency": 5,
    }
    model.deploy(
        endpoint_name=endpoint_name,
        update_endpoint=True,
        serverless_inference_config=serverless_inference_config,
    )

    sagemaker_session.create_endpoint_config.assert_called_with(
        name=MODEL_NAME,
        model_name=MODEL_NAME,
        initial_instance_count=None,
        instance_type=None,
        accelerator_type=None,
        tags=None,
        kms_key=None,
        data_capture_config_dict=None,
        volume_size=None,
        model_data_download_timeout=None,
        container_startup_health_check_timeout=None,
        explainer_config_dict=None,
        async_inference_config_dict=None,
        serverless_inference_config=serverless_inference_config_dict,
        routing_config=None,
        inference_ami_version=None,
    )

    # Verify update_endpoint is called with the new config
    sagemaker_session.update_endpoint.assert_called_with(endpoint_name, endpoint_config_name)

    # Test update_endpoint with async inference config
    async_inference_config = AsyncInferenceConfig(
        output_path="s3://bucket/output", failure_path="s3://bucket/failure"
    )
    async_inference_config_dict = {
        "OutputConfig": {
            "S3OutputPath": "s3://bucket/output",
            "S3FailurePath": "s3://bucket/failure",
        },
    }
    model.deploy(
        endpoint_name=endpoint_name,
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
        update_endpoint=True,
        async_inference_config=async_inference_config,
    )

    sagemaker_session.create_endpoint_config.assert_called_with(
        name=MODEL_NAME,
        model_name=MODEL_NAME,
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        accelerator_type=None,
        tags=None,
        kms_key=None,
        data_capture_config_dict=None,
        volume_size=None,
        model_data_download_timeout=None,
        container_startup_health_check_timeout=None,
        explainer_config_dict=None,
        async_inference_config_dict=async_inference_config_dict,
        serverless_inference_config=None,
        routing_config=None,
        inference_ami_version=None,
    )

    # Verify update_endpoint is called with the new config
    sagemaker_session.update_endpoint.assert_called_with(endpoint_name, endpoint_config_name)


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
@patch("sagemaker.production_variant", return_value=BASE_PRODUCTION_VARIANT)
def test_deploy_with_update_endpoint_inference_component(production_variant, sagemaker_session):
    model = Model(
        MODEL_IMAGE, MODEL_DATA, role=ROLE, name=MODEL_NAME, sagemaker_session=sagemaker_session
    )

    # Test that updating endpoint with inference component raises error
    with pytest.raises(
        ValueError, match="Currently update_endpoint is supported for single model endpoints"
    ):
        model.deploy(
            endpoint_name="test-endpoint",
            instance_type=INSTANCE_TYPE,
            initial_instance_count=INSTANCE_COUNT,
            update_endpoint=True,
            resources=RESOURCES,
            endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
        )
