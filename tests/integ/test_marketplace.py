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

import itertools
import os
import time
import requests

import pandas
import pytest
import docker

import sagemaker
import tests.integ
from tests.integ.utils import create_repository
from sagemaker import AlgorithmEstimator, ModelPackage, Model
from sagemaker.serializers import CSVSerializer
from sagemaker.tuner import IntegerParameter, HyperparameterTuner
from sagemaker.utils import sagemaker_timestamp, aws_partition, unique_name_from_base
from tests.integ import DATA_DIR
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name
from tests.integ.marketplace_utils import REGION_ACCOUNT_MAP
from tests.integ.test_multidatamodel import (
    _ecr_image_uri,
    _ecr_login,
    _delete_repository,
)
from tests.integ.retry import retries
import logging

logger = logging.getLogger(__name__)

# All these tests require a manual 1 time subscription to the following Marketplace items:
# Algorithm: Scikit Decision Trees
# https://aws.amazon.com/marketplace/pp/prodview-ha4f3kqugba3u
#
# Pre-Trained Model: Scikit Decision Trees - Pretrained Model
# https://aws.amazon.com/marketplace/pp/prodview-7qop4x5ahrdhe
#
# Both are written by Amazon and are free to subscribe.

ALGORITHM_ARN = (
    "arn:{partition}:sagemaker:{region}:{account}:algorithm/scikit-decision-trees-"
    "15423055-57b73412d2e93e9239e4e16f83298b8f"
)

MODEL_PACKAGE_ARN = (
    "arn:{partition}:sagemaker:{region}:{account}:model-package/scikit-iris-detector-"
    "154230595-8f00905c1f927a512b73ea29dd09ae30"
)


@pytest.mark.release
@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MARKET_PLACE_REGIONS,
    reason="Marketplace is not available in {}".format(tests.integ.test_region()),
)
@pytest.mark.skip(
    reason="This test has always failed, but the failure was masked by a bug. "
    "This test should be fixed. Details in https://github.com/aws/sagemaker-python-sdk/pull/968"
)
def test_marketplace_estimator(sagemaker_session, cpu_instance_type):
    with timeout(minutes=15):
        data_path = os.path.join(DATA_DIR, "marketplace", "training")
        region = sagemaker_session.boto_region_name
        account = REGION_ACCOUNT_MAP[region]
        algorithm_arn = ALGORITHM_ARN.format(
            partition=aws_partition(region), region=region, account=account
        )

        algo = AlgorithmEstimator(
            algorithm_arn=algorithm_arn,
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
        )

        train_input = algo.sagemaker_session.upload_data(
            path=data_path, key_prefix="integ-test-data/marketplace/train"
        )

        algo.fit({"training": train_input})

    endpoint_name = "test-marketplace-estimator{}".format(sagemaker_timestamp())
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session, minutes=20):
        predictor = algo.deploy(1, cpu_instance_type, endpoint_name=endpoint_name)
        shape = pandas.read_csv(os.path.join(data_path, "iris.csv"), header=None)

        a = [50 * i for i in range(3)]
        b = [40 + i for i in range(10)]
        indices = [i + j for i, j in itertools.product(a, b)]

        test_data = shape.iloc[indices[:-1]]
        test_x = test_data.iloc[:, 1:]

        print(predictor.predict(test_x.values).decode("utf-8"))


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MARKET_PLACE_REGIONS,
    reason="Marketplace is not available in {}".format(tests.integ.test_region()),
)
def test_marketplace_attach(sagemaker_session, cpu_instance_type):
    with timeout(minutes=15):
        data_path = os.path.join(DATA_DIR, "marketplace", "training")
        region = sagemaker_session.boto_region_name
        account = REGION_ACCOUNT_MAP[region]
        algorithm_arn = ALGORITHM_ARN.format(
            partition=aws_partition(region), region=region, account=account
        )

        mktplace = AlgorithmEstimator(
            algorithm_arn=algorithm_arn,
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            base_job_name=unique_name_from_base("test-marketplace"),
        )

        train_input = mktplace.sagemaker_session.upload_data(
            path=data_path, key_prefix="integ-test-data/marketplace/train"
        )

        mktplace.fit({"training": train_input}, wait=False)
        training_job_name = mktplace.latest_training_job.name

        print("Waiting to re-attach to the training job: %s" % training_job_name)
        time.sleep(20)
        endpoint_name = "test-marketplace-estimator{}".format(sagemaker_timestamp())

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session, minutes=20):
        print("Re-attaching now to: %s" % training_job_name)
        estimator = AlgorithmEstimator.attach(
            training_job_name=training_job_name, sagemaker_session=sagemaker_session
        )
        predictor = estimator.deploy(
            1, cpu_instance_type, endpoint_name=endpoint_name, serializer=CSVSerializer()
        )
        shape = pandas.read_csv(os.path.join(data_path, "iris.csv"), header=None)
        a = [50 * i for i in range(3)]
        b = [40 + i for i in range(10)]
        indices = [i + j for i, j in itertools.product(a, b)]

        test_data = shape.iloc[indices[:-1]]
        test_x = test_data.iloc[:, 1:]

        print(predictor.predict(test_x.values).decode("utf-8"))


@pytest.mark.release
@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MARKET_PLACE_REGIONS,
    reason="Marketplace is not available in {}".format(tests.integ.test_region()),
)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_marketplace_model(sagemaker_session, cpu_instance_type):
    region = sagemaker_session.boto_region_name
    account = REGION_ACCOUNT_MAP[region]
    model_package_arn = MODEL_PACKAGE_ARN.format(
        partition=aws_partition(region), region=region, account=account
    )

    def predict_wrapper(endpoint, session):
        return sagemaker.Predictor(endpoint, session, serializer=CSVSerializer())

    model = ModelPackage(
        role="SageMakerRole",
        model_package_arn=model_package_arn,
        sagemaker_session=sagemaker_session,
        predictor_cls=predict_wrapper,
    )

    endpoint_name = "test-marketplace-model-endpoint{}".format(sagemaker_timestamp())
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session, minutes=20):
        predictor = model.deploy(1, cpu_instance_type, endpoint_name=endpoint_name)
        data_path = os.path.join(DATA_DIR, "marketplace", "training")
        shape = pandas.read_csv(os.path.join(data_path, "iris.csv"), header=None)
        a = [50 * i for i in range(3)]
        b = [40 + i for i in range(10)]
        indices = [i + j for i, j in itertools.product(a, b)]

        test_data = shape.iloc[indices[:-1]]
        test_x = test_data.iloc[:, 1:]

        print(predictor.predict(test_x.values).decode("utf-8"))


@pytest.fixture(scope="module")
def iris_image(sagemaker_session):
    algorithm_name = unique_name_from_base("iris-classifier")
    ecr_image = _ecr_image_uri(sagemaker_session, algorithm_name)
    ecr_client = sagemaker_session.boto_session.client("ecr")
    username, password = _ecr_login(ecr_client)

    docker_client = docker.from_env()

    # Build and tag docker image locally
    path = os.path.join(DATA_DIR, "marketplace", "iris")
    image, build_logs = docker_client.images.build(
        path=path,
        tag=algorithm_name,
        rm=True,
    )
    image.tag(ecr_image, tag="latest")
    create_repository(ecr_client, algorithm_name)

    # Retry docker image push
    for _ in retries(3, "Upload docker image to ECR repo", seconds_to_sleep=10):
        try:
            docker_client.images.push(
                ecr_image, auth_config={"username": username, "password": password}
            )
            break
        except requests.exceptions.ConnectionError:
            # This can happen when we try to create multiple repositories in parallel, so we retry
            pass

    yield ecr_image

    # Delete repository after the marketplace integration tests complete
    _delete_repository(ecr_client, algorithm_name)


@pytest.mark.xfail(reason="marking this for xfail until we work on the test failure to be fixed")
def test_create_model_package(sagemaker_session, boto_session, iris_image):
    MODEL_NAME = "iris-classifier-mp"
    # Prepare
    s3_bucket = sagemaker_session.default_bucket()

    model_name = unique_name_from_base(MODEL_NAME)
    model_description = "This model accepts petal length, petal width, sepal length, sepal width and predicts whether \
    flower is of type setosa, versicolor, or virginica"

    supported_realtime_inference_instance_types = supported_batch_transform_instance_types = [
        "ml.m4.xlarge"
    ]
    supported_content_types = ["text/csv", "application/json", "application/jsonlines"]
    supported_response_MIME_types = ["application/json", "text/csv", "application/jsonlines"]

    validation_input_path = "s3://" + s3_bucket + "/validation-input-csv/"
    validation_output_path = "s3://" + s3_bucket + "/validation-output-csv/"

    iam = boto_session.resource("iam")
    role = iam.Role("SageMakerRole").arn
    sm_client = boto_session.client("sagemaker")
    s3_client = boto_session.client("s3")
    s3_client.put_object(
        Bucket=s3_bucket, Key="validation-input-csv/input.csv", Body="5.1, 3.5, 1.4, 0.2"
    )

    ValidationSpecification = {
        "ValidationRole": role,
        "ValidationProfiles": [
            {
                "ProfileName": "Validation-test",
                "TransformJobDefinition": {
                    "BatchStrategy": "SingleRecord",
                    "TransformInput": {
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": validation_input_path,
                            }
                        },
                        "ContentType": supported_content_types[0],
                    },
                    "TransformOutput": {
                        "S3OutputPath": validation_output_path,
                    },
                    "TransformResources": {
                        "InstanceType": supported_batch_transform_instance_types[0],
                        "InstanceCount": 1,
                    },
                },
            },
        ],
    }

    # get pre-existing model artifact stored in ECR
    model = Model(
        image_uri=iris_image,
        model_data=validation_input_path + "input.csv",
        role=role,
        sagemaker_session=sagemaker_session,
        enable_network_isolation=False,
    )

    # Call model.register() - the method under test - to create a model package
    model.register(
        supported_content_types,
        supported_response_MIME_types,
        supported_realtime_inference_instance_types,
        supported_batch_transform_instance_types,
        marketplace_cert=True,
        description=model_description,
        model_package_name=model_name,
        validation_specification=ValidationSpecification,
    )

    # wait for model execution to complete
    time.sleep(60 * 3)

    # query for all model packages with the name <MODEL_NAME>
    response = sm_client.list_model_packages(
        MaxResults=10,
        NameContains=MODEL_NAME,
        SortBy="CreationTime",
        SortOrder="Descending",
    )

    if len(response["ModelPackageSummaryList"]) > 0:
        sm_client.delete_model_package(ModelPackageName=model_name)

    # assert that response is non-empty
    assert len(response["ModelPackageSummaryList"]) > 0


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MARKET_PLACE_REGIONS,
    reason="Marketplace is not available in {}".format(tests.integ.test_region()),
)
def test_marketplace_tuning_job(sagemaker_session, cpu_instance_type):
    data_path = os.path.join(DATA_DIR, "marketplace", "training")
    region = sagemaker_session.boto_region_name
    account = REGION_ACCOUNT_MAP[region]
    algorithm_arn = ALGORITHM_ARN.format(
        partition=aws_partition(region), region=region, account=account
    )

    mktplace = AlgorithmEstimator(
        algorithm_arn=algorithm_arn,
        role="SageMakerRole",
        instance_count=1,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        base_job_name=unique_name_from_base("test-marketplace"),
    )

    train_input = mktplace.sagemaker_session.upload_data(
        path=data_path, key_prefix="integ-test-data/marketplace/train"
    )

    mktplace.set_hyperparameters(max_leaf_nodes=10)

    hyperparameter_ranges = {"max_leaf_nodes": IntegerParameter(1, 100000)}

    tuner = HyperparameterTuner(
        estimator=mktplace,
        base_tuning_job_name=unique_name_from_base("byo"),
        objective_metric_name="validation:accuracy",
        hyperparameter_ranges=hyperparameter_ranges,
        max_jobs=2,
        max_parallel_jobs=2,
    )

    tuner.fit({"training": train_input}, include_cls_metadata=False)
    time.sleep(15)
    tuner.wait()


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MARKET_PLACE_REGIONS,
    reason="Marketplace is not available in {}".format(tests.integ.test_region()),
)
def test_marketplace_transform_job(sagemaker_session, cpu_instance_type):
    data_path = os.path.join(DATA_DIR, "marketplace", "training")
    region = sagemaker_session.boto_region_name
    account = REGION_ACCOUNT_MAP[region]
    algorithm_arn = ALGORITHM_ARN.format(
        partition=aws_partition(region), region=region, account=account
    )

    algo = AlgorithmEstimator(
        algorithm_arn=algorithm_arn,
        role="SageMakerRole",
        instance_count=1,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        base_job_name=unique_name_from_base("test-marketplace"),
    )

    train_input = algo.sagemaker_session.upload_data(
        path=data_path, key_prefix="integ-test-data/marketplace/train"
    )

    shape = pandas.read_csv(data_path + "/iris.csv", header=None).drop([0], axis=1)

    transform_workdir = DATA_DIR + "/marketplace/transform"
    shape.to_csv(transform_workdir + "/batchtransform_test.csv", index=False, header=False)
    transform_input = algo.sagemaker_session.upload_data(
        transform_workdir, key_prefix="integ-test-data/marketplace/transform"
    )

    algo.fit({"training": train_input})

    transformer = algo.transformer(1, cpu_instance_type)
    transformer.transform(transform_input, content_type="text/csv")
    transformer.wait()


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MARKET_PLACE_REGIONS,
    reason="Marketplace is not available in {}".format(tests.integ.test_region()),
)
def test_marketplace_transform_job_from_model_package(sagemaker_session, cpu_instance_type):
    data_path = os.path.join(DATA_DIR, "marketplace", "training")
    shape = pandas.read_csv(data_path + "/iris.csv", header=None).drop([0], axis=1)

    TRANSFORM_WORKDIR = DATA_DIR + "/marketplace/transform"
    shape.to_csv(TRANSFORM_WORKDIR + "/batchtransform_test.csv", index=False, header=False)
    transform_input = sagemaker_session.upload_data(
        TRANSFORM_WORKDIR, key_prefix="integ-test-data/marketplace/transform"
    )

    region = sagemaker_session.boto_region_name
    account = REGION_ACCOUNT_MAP[region]
    model_package_arn = MODEL_PACKAGE_ARN.format(
        partition=aws_partition(region), region=region, account=account
    )

    model = ModelPackage(
        role="SageMakerRole",
        model_package_arn=model_package_arn,
        sagemaker_session=sagemaker_session,
    )

    transformer = model.transformer(1, cpu_instance_type)
    transformer.transform(transform_input, content_type="text/csv")
    transformer.wait()
