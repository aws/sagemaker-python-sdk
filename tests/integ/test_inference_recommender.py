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
import time

import pytest

from botocore.exceptions import ClientError
from sagemaker import image_uris
from sagemaker.model import Model
from sagemaker.sklearn.model import SKLearnModel, SKLearnPredictor
from sagemaker.utils import unique_name_from_base
from tests.integ import DATA_DIR
from tests.integ.timeout import timeout
import pandas as pd
from sagemaker.inference_recommender.inference_recommender_mixin import Phase, ModelLatencyThreshold
from sagemaker.parameter import CategoricalParameter
import logging

logger = logging.getLogger(__name__)

# Running integration tests on SKLearn model
IR_DIR = os.path.join(DATA_DIR, "inference_recommender")
IR_SKLEARN_MODEL = os.path.join(IR_DIR, "sklearn-model.tar.gz")
IR_SKLEARN_ENTRY_POINT = os.path.join(IR_DIR, "inference.py")
IR_SKLEARN_PAYLOAD = os.path.join(IR_DIR, "sklearn-payload.tar.gz")
IR_SKLEARN_DATA = os.path.join(IR_DIR, "sample.csv")
IR_SKLEARN_CONTENT_TYPE = ["text/csv"]
IR_SKLEARN_FRAMEWORK = "SAGEMAKER-SCIKIT-LEARN"
IR_SKLEARN_FRAMEWORK_VERSION = "1.0-1"


def retry_and_back_off(right_size_fn):
    tot_retries = 3
    retries = 1
    while retries <= tot_retries:
        try:
            return right_size_fn
        except ClientError as e:
            if e.response["Error"]["Code"] == "ThrottlingException":
                retries += 1
                time.sleep(5 * retries)


@pytest.fixture(scope="module")
def default_right_sized_model(sagemaker_session, cpu_instance_type):
    with timeout(minutes=45):
        try:
            model_package_group_name = unique_name_from_base("test-ir-right-size-model-pkg-sklearn")
            ir_job_name = unique_name_from_base("test-ir-right-size-job-name")
            model_data = sagemaker_session.upload_data(path=IR_SKLEARN_MODEL)
            payload_data = sagemaker_session.upload_data(path=IR_SKLEARN_PAYLOAD)

            iam_client = sagemaker_session.boto_session.client("iam")
            role_arn = iam_client.get_role(RoleName="SageMakerRole")["Role"]["Arn"]

            sklearn_model = SKLearnModel(
                model_data=model_data,
                role=role_arn,
                entry_point=IR_SKLEARN_ENTRY_POINT,
                framework_version=IR_SKLEARN_FRAMEWORK_VERSION,
            )

            sklearn_model_package = sklearn_model.register(
                content_types=IR_SKLEARN_CONTENT_TYPE,
                response_types=IR_SKLEARN_CONTENT_TYPE,
                model_package_group_name=model_package_group_name,
                image_uri=sklearn_model.image_uri,
                approval_status="Approved",
            )

            return (
                retry_and_back_off(
                    sklearn_model_package.right_size(
                        job_name=ir_job_name,
                        sample_payload_url=payload_data,
                        supported_content_types=IR_SKLEARN_CONTENT_TYPE,
                        supported_instance_types=[cpu_instance_type],
                        framework=IR_SKLEARN_FRAMEWORK,
                        log_level="Quiet",
                    )
                ),
                model_package_group_name,
                ir_job_name,
            )
        except Exception:
            sagemaker_session.sagemaker_client.delete_model_package(
                ModelPackageName=sklearn_model_package.model_package_arn
            )
            sagemaker_session.sagemaker_client.delete_model_package_group(
                ModelPackageGroupName=model_package_group_name
            )


@pytest.fixture(scope="module")
def advanced_right_sized_model(sagemaker_session, cpu_instance_type):
    with timeout(minutes=45):
        try:
            model_package_group_name = unique_name_from_base("test-ir-right-size-model-pkg-sklearn")
            model_data = sagemaker_session.upload_data(path=IR_SKLEARN_MODEL)
            payload_data = sagemaker_session.upload_data(path=IR_SKLEARN_PAYLOAD)

            iam_client = sagemaker_session.boto_session.client("iam")
            role_arn = iam_client.get_role(RoleName="SageMakerRole")["Role"]["Arn"]

            sklearn_model = SKLearnModel(
                model_data=model_data,
                role=role_arn,
                entry_point=IR_SKLEARN_ENTRY_POINT,
                framework_version=IR_SKLEARN_FRAMEWORK_VERSION,
            )

            sklearn_model_package = sklearn_model.register(
                content_types=IR_SKLEARN_CONTENT_TYPE,
                response_types=IR_SKLEARN_CONTENT_TYPE,
                model_package_group_name=model_package_group_name,
                image_uri=sklearn_model.image_uri,
                approval_status="Approved",
            )

            hyperparameter_ranges = [
                {
                    "instance_types": CategoricalParameter([cpu_instance_type]),
                    "TEST_PARAM": CategoricalParameter(
                        ["TEST_PARAM_VALUE_1", "TEST_PARAM_VALUE_2"]
                    ),
                }
            ]

            phases = [
                Phase(duration_in_seconds=300, initial_number_of_users=2, spawn_rate=2),
                Phase(duration_in_seconds=300, initial_number_of_users=14, spawn_rate=2),
            ]

            model_latency_thresholds = [
                ModelLatencyThreshold(percentile="P95", value_in_milliseconds=100)
            ]

            return (
                retry_and_back_off(
                    sklearn_model_package.right_size(
                        sample_payload_url=payload_data,
                        supported_content_types=IR_SKLEARN_CONTENT_TYPE,
                        framework=IR_SKLEARN_FRAMEWORK,
                        job_duration_in_seconds=3600,
                        hyperparameter_ranges=hyperparameter_ranges,
                        phases=phases,
                        model_latency_thresholds=model_latency_thresholds,
                        max_invocations=100,
                        max_tests=5,
                        max_parallel_tests=5,
                    )
                ),
                model_package_group_name,
            )
        except Exception as e:
            sagemaker_session.sagemaker_client.delete_model_package(
                ModelPackageName=sklearn_model_package.model_package_arn
            )
            sagemaker_session.sagemaker_client.delete_model_package_group(
                ModelPackageGroupName=model_package_group_name
            )
            raise e


@pytest.fixture(scope="module")
def default_right_sized_unregistered_model(sagemaker_session, cpu_instance_type):
    with timeout(minutes=45):
        try:
            ir_job_name = unique_name_from_base("test-ir-right-size-job-name")
            model_data = sagemaker_session.upload_data(path=IR_SKLEARN_MODEL)
            payload_data = sagemaker_session.upload_data(path=IR_SKLEARN_PAYLOAD)

            iam_client = sagemaker_session.boto_session.client("iam")
            role_arn = iam_client.get_role(RoleName="SageMakerRole")["Role"]["Arn"]

            sklearn_model = SKLearnModel(
                model_data=model_data,
                role=role_arn,
                entry_point=IR_SKLEARN_ENTRY_POINT,
                framework_version=IR_SKLEARN_FRAMEWORK_VERSION,
            )

            return (
                retry_and_back_off(
                    sklearn_model.right_size(
                        job_name=ir_job_name,
                        sample_payload_url=payload_data,
                        supported_content_types=IR_SKLEARN_CONTENT_TYPE,
                        supported_instance_types=[cpu_instance_type],
                        framework=IR_SKLEARN_FRAMEWORK,
                        log_level="Quiet",
                    )
                ),
                ir_job_name,
            )
        except Exception:
            sagemaker_session.delete_model(model_name=sklearn_model.name)


@pytest.fixture(scope="module")
def advanced_right_sized_unregistered_model(sagemaker_session, cpu_instance_type):
    with timeout(minutes=45):
        try:
            model_data = sagemaker_session.upload_data(path=IR_SKLEARN_MODEL)
            payload_data = sagemaker_session.upload_data(path=IR_SKLEARN_PAYLOAD)

            iam_client = sagemaker_session.boto_session.client("iam")
            role_arn = iam_client.get_role(RoleName="SageMakerRole")["Role"]["Arn"]

            sklearn_model = SKLearnModel(
                model_data=model_data,
                role=role_arn,
                entry_point=IR_SKLEARN_ENTRY_POINT,
                framework_version=IR_SKLEARN_FRAMEWORK_VERSION,
            )

            hyperparameter_ranges = [
                {
                    "instance_types": CategoricalParameter([cpu_instance_type]),
                    "TEST_PARAM": CategoricalParameter(
                        ["TEST_PARAM_VALUE_1", "TEST_PARAM_VALUE_2"]
                    ),
                }
            ]

            phases = [
                Phase(duration_in_seconds=300, initial_number_of_users=2, spawn_rate=2),
                Phase(duration_in_seconds=300, initial_number_of_users=14, spawn_rate=2),
            ]

            model_latency_thresholds = [
                ModelLatencyThreshold(percentile="P95", value_in_milliseconds=100)
            ]

            return retry_and_back_off(
                sklearn_model.right_size(
                    sample_payload_url=payload_data,
                    supported_content_types=IR_SKLEARN_CONTENT_TYPE,
                    framework=IR_SKLEARN_FRAMEWORK,
                    job_duration_in_seconds=3600,
                    hyperparameter_ranges=hyperparameter_ranges,
                    phases=phases,
                    model_latency_thresholds=model_latency_thresholds,
                    max_invocations=100,
                    max_tests=5,
                    max_parallel_tests=5,
                    log_level="Quiet",
                )
            )

        except Exception:
            sagemaker_session.delete_model(model_name=sklearn_model.name)


@pytest.fixture(scope="module")
def default_right_sized_unregistered_base_model(sagemaker_session, cpu_instance_type):
    with timeout(minutes=45):
        try:
            ir_job_name = unique_name_from_base("test-ir-right-size-job-name")
            model_data = sagemaker_session.upload_data(path=IR_SKLEARN_MODEL)
            payload_data = sagemaker_session.upload_data(path=IR_SKLEARN_PAYLOAD)
            region = sagemaker_session._region_name
            image_uri = image_uris.retrieve(
                framework="sklearn", region=region, version="1.0-1", image_scope="inference"
            )

            iam_client = sagemaker_session.boto_session.client("iam")
            role_arn = iam_client.get_role(RoleName="SageMakerRole")["Role"]["Arn"]

            model = Model(
                model_data=model_data,
                role=role_arn,
                entry_point=IR_SKLEARN_ENTRY_POINT,
                image_uri=image_uri,
            )

            return (
                retry_and_back_off(
                    model.right_size(
                        job_name=ir_job_name,
                        sample_payload_url=payload_data,
                        supported_content_types=IR_SKLEARN_CONTENT_TYPE,
                        supported_instance_types=[cpu_instance_type],
                        framework=IR_SKLEARN_FRAMEWORK,
                        log_level="Quiet",
                    )
                ),
                ir_job_name,
            )
        except Exception:
            sagemaker_session.delete_model(model_name=model.name)


@pytest.fixture(scope="module")
def created_base_model(sagemaker_session, cpu_instance_type):
    model_data = sagemaker_session.upload_data(path=IR_SKLEARN_MODEL)
    region = sagemaker_session._region_name
    image_uri = image_uris.retrieve(
        framework="sklearn", region=region, version="1.0-1", image_scope="inference"
    )

    iam_client = sagemaker_session.boto_session.client("iam")
    role_arn = iam_client.get_role(RoleName="SageMakerRole")["Role"]["Arn"]

    model = Model(
        model_data=model_data,
        role=role_arn,
        entry_point=IR_SKLEARN_ENTRY_POINT,
        image_uri=image_uri,
        sagemaker_session=sagemaker_session,
    )

    model.create(instance_type=cpu_instance_type)

    return model


@pytest.mark.slow_test
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_default_right_size_and_deploy_registered_model_sklearn(
    default_right_sized_model, sagemaker_session
):
    endpoint_name = unique_name_from_base("test-ir-right-size-default-sklearn")

    right_size_model_package, model_package_group_name, ir_job_name = default_right_sized_model
    with timeout(minutes=45):
        try:
            right_size_model_package.predictor_cls = SKLearnPredictor
            predictor = right_size_model_package.deploy(endpoint_name=endpoint_name)

            payload = pd.read_csv(IR_SKLEARN_DATA, header=None)

            inference = predictor.predict(payload)
            assert inference is not None
            assert 26 == len(inference)
        finally:
            predictor.delete_model()
            predictor.delete_endpoint()


@pytest.mark.slow_test
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_default_right_size_and_deploy_unregistered_model_sklearn(
    default_right_sized_unregistered_model, sagemaker_session
):
    endpoint_name = unique_name_from_base("test-ir-right-size-default-unregistered-sklearn")

    right_size_model, ir_job_name = default_right_sized_unregistered_model
    with timeout(minutes=45):
        try:
            right_size_model.predictor_cls = SKLearnPredictor
            predictor = right_size_model.deploy(endpoint_name=endpoint_name)

            payload = pd.read_csv(IR_SKLEARN_DATA, header=None)

            inference = predictor.predict(payload)
            assert inference is not None
            assert 26 == len(inference)
        finally:
            predictor.delete_model()
            predictor.delete_endpoint()


@pytest.mark.slow_test
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_default_right_size_and_deploy_unregistered_base_model(
    default_right_sized_unregistered_base_model, sagemaker_session
):
    endpoint_name = unique_name_from_base("test-ir-right-size-default-unregistered-base")

    right_size_model, ir_job_name = default_right_sized_unregistered_base_model
    with timeout(minutes=45):
        try:
            right_size_model.predictor_cls = SKLearnPredictor
            predictor = right_size_model.deploy(endpoint_name=endpoint_name)

            payload = pd.read_csv(IR_SKLEARN_DATA, header=None)

            inference = predictor.predict(payload)
            assert inference is not None
            assert 26 == len(inference)
        finally:
            predictor.delete_model()
            predictor.delete_endpoint()


@pytest.mark.slow_test
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_advanced_right_size_and_deploy_unregistered_model_sklearn(
    advanced_right_sized_unregistered_model, sagemaker_session
):
    endpoint_name = unique_name_from_base("test-ir-right-size-advanced-sklearn")

    right_size_model = advanced_right_sized_unregistered_model
    with timeout(minutes=45):
        try:
            right_size_model.predictor_cls = SKLearnPredictor
            predictor = right_size_model.deploy(endpoint_name=endpoint_name)

            payload = pd.read_csv(IR_SKLEARN_DATA, header=None)

            inference = predictor.predict(payload)
            assert inference is not None
            assert 26 == len(inference)
        finally:
            predictor.delete_model()
            predictor.delete_endpoint()


@pytest.mark.skip(reason="Skipping this test class for now")
@pytest.mark.slow_test
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_advanced_right_size_and_deploy_registered_model_sklearn(
    advanced_right_sized_model, sagemaker_session
):
    endpoint_name = unique_name_from_base("test-ir-right-size-advanced-sklearn")

    right_size_model_package, model_package_group_name = advanced_right_sized_model
    with timeout(minutes=45):
        try:
            right_size_model_package.predictor_cls = SKLearnPredictor
            predictor = right_size_model_package.deploy(endpoint_name=endpoint_name)

            payload = pd.read_csv(IR_SKLEARN_DATA, header=None)

            inference = predictor.predict(payload)
            assert inference is not None
            assert 26 == len(inference)
        finally:
            sagemaker_session.sagemaker_client.delete_model_package(
                ModelPackageName=right_size_model_package.model_package_arn
            )
            sagemaker_session.sagemaker_client.delete_model_package_group(
                ModelPackageGroupName=model_package_group_name
            )
            predictor.delete_model()
            predictor.delete_endpoint()


# TODO when we've added support for inference_recommendation_id
# then add tests to test Framework models
@pytest.mark.slow_test
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_deploy_inference_recommendation_id_with_registered_model_sklearn(
    default_right_sized_model, sagemaker_session
):
    right_size_model_package, model_package_group_name, ir_job_name = default_right_sized_model
    endpoint_name = unique_name_from_base("test-rec-id-deployment-default-sklearn")
    rec_res = sagemaker_session.sagemaker_client.describe_inference_recommendations_job(
        JobName=ir_job_name
    )

    rec_id = get_realtime_recommendation_id(recommendation_list=rec_res["InferenceRecommendations"])

    with timeout(minutes=45):
        try:
            right_size_model_package.predictor_cls = SKLearnPredictor
            predictor = right_size_model_package.deploy(
                inference_recommendation_id=rec_id, endpoint_name=endpoint_name
            )

            payload = pd.read_csv(IR_SKLEARN_DATA, header=None)

            inference = predictor.predict(payload)
            assert inference is not None
            assert 26 == len(inference)
        finally:
            sagemaker_session.sagemaker_client.delete_model_package(
                ModelPackageName=right_size_model_package.model_package_arn
            )
            sagemaker_session.sagemaker_client.delete_model_package_group(
                ModelPackageGroupName=model_package_group_name
            )
            predictor.delete_model()
            predictor.delete_endpoint()


@pytest.mark.slow_test
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_deploy_deployment_recommendation_id_with_model(created_base_model, sagemaker_session):
    with timeout(minutes=20):
        try:
            deployment_recommendation = poll_for_deployment_recommendation(
                created_base_model, sagemaker_session
            )

            assert deployment_recommendation is not None

            real_time_recommendations = deployment_recommendation.get(
                "RealTimeInferenceRecommendations"
            )
            recommendation_id = real_time_recommendations[0].get("RecommendationId")

            endpoint_name = unique_name_from_base("test-rec-id-deployment-default-sklearn")
            created_base_model.predictor_cls = SKLearnPredictor
            predictor = created_base_model.deploy(
                inference_recommendation_id=recommendation_id,
                initial_instance_count=1,
                endpoint_name=endpoint_name,
            )

            payload = pd.read_csv(IR_SKLEARN_DATA, header=None)

            inference = predictor.predict(payload)
            assert inference is not None
            assert 26 == len(inference)
        finally:
            predictor.delete_model()
            predictor.delete_endpoint()


def poll_for_deployment_recommendation(created_base_model, sagemaker_session):
    with timeout(minutes=1):
        try:
            completed = False
            while not completed:
                describe_model_response = sagemaker_session.sagemaker_client.describe_model(
                    ModelName=created_base_model.name
                )
                deployment_recommendation = describe_model_response.get("DeploymentRecommendation")

                completed = (
                    deployment_recommendation is not None
                    and "COMPLETED" == deployment_recommendation.get("RecommendationStatus")
                )
            return deployment_recommendation
        except Exception as e:
            created_base_model.delete_model()
            raise e


def get_realtime_recommendation_id(recommendation_list):
    """Search recommendation based on recommendation id"""
    next(
        (rec["RecommendationId"] for rec in recommendation_list if "InstanceType" in rec),
        None,
    )
