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

import base64
import os
import time
import requests

import docker
import numpy
import pytest

from tests.integ.utils import create_repository
from sagemaker import utils
from sagemaker.amazon.randomcutforest import RandomCutForest
from sagemaker.deserializers import StringDeserializer
from sagemaker.multidatamodel import MultiDataModel
from sagemaker.mxnet import MXNet
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.utils import sagemaker_timestamp, unique_name_from_base
from tests.integ import DATA_DIR, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.retry import retries
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name

ROLE = "SageMakerRole"
PRETRAINED_MODEL_PATH_1 = "customer_a/dummy_model.tar.gz"
PRETRAINED_MODEL_PATH_2 = "customer_b/dummy_model.tar.gz"
string_deserializer = StringDeserializer()


@pytest.fixture(scope="module")
def container_image(sagemaker_session):
    """Create a Multi-Model image since pre-built ones are not available yet."""
    algorithm_name = unique_name_from_base("sagemaker-multimodel-integ-test")
    ecr_image = _ecr_image_uri(sagemaker_session, algorithm_name)

    ecr_client = sagemaker_session.boto_session.client("ecr")
    username, password = _ecr_login(ecr_client)

    docker_client = docker.from_env()

    # Build and tag docker image locally
    image, build_log = docker_client.images.build(
        path=os.path.join(DATA_DIR, "multimodel", "container"),
        tag=algorithm_name,
        rm=True,
    )
    image.tag(ecr_image, tag="latest")

    # Create AWS ECR and push the local docker image to it
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

    # Delete repository after the multi model integration tests complete
    _delete_repository(ecr_client, algorithm_name)


def _ecr_image_uri(sagemaker_session, algorithm_name):
    region = sagemaker_session.boto_region_name

    sts_client = sagemaker_session.boto_session.client(
        "sts", region_name=region, endpoint_url=utils.sts_regional_endpoint(region)
    )
    account_id = sts_client.get_caller_identity()["Account"]

    endpoint_data = utils._botocore_resolver().construct_endpoint("ecr", region)
    if region == "il-central-1" and not endpoint_data:
        endpoint_data = {"hostname": "ecr.{}.amazonaws.com".format(region)}
    return "{}.dkr.{}/{}:latest".format(account_id, endpoint_data["hostname"], algorithm_name)


def _delete_repository(ecr_client, repository_name):
    """
    Deletes an ECS Repository (ECR). After the integration test completes
    we will remove the repository created during setup
    """
    try:
        ecr_client.describe_repositories(repositoryNames=[repository_name])
        ecr_client.delete_repository(repositoryName=repository_name, force=True)
    except ecr_client.exceptions.RepositoryNotFoundException:
        pass


def _ecr_login(ecr_client):
    """Get a login credentials for an ecr client."""
    login = ecr_client.get_authorization_token()
    b64token = login["authorizationData"][0]["authorizationToken"].encode("utf-8")
    username, password = base64.b64decode(b64token).decode("utf-8").split(":")
    return username, password


def test_multi_data_model_deploy_pretrained_models(
    container_image, sagemaker_session, cpu_instance_type
):
    timestamp = sagemaker_timestamp()
    endpoint_name = "test-multimodel-endpoint-{}".format(timestamp)
    model_name = "test-multimodel-{}".format(timestamp)

    # Define pretrained model local path
    pretrained_model_data_local_path = os.path.join(DATA_DIR, "sparkml_model", "mleap_model.tar.gz")

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        model_data_prefix = os.path.join(
            "s3://", sagemaker_session.default_bucket(), "multimodel-{}/".format(timestamp)
        )
        multi_data_model = MultiDataModel(
            name=model_name,
            model_data_prefix=model_data_prefix,
            image_uri=container_image,
            role=ROLE,
            sagemaker_session=sagemaker_session,
        )

        # Add model before deploy
        multi_data_model.add_model(pretrained_model_data_local_path, PRETRAINED_MODEL_PATH_1)
        # Deploy model to an endpoint
        multi_data_model.deploy(1, cpu_instance_type, endpoint_name=endpoint_name)
        time.sleep(30)
        # Add models after deploy
        multi_data_model.add_model(pretrained_model_data_local_path, PRETRAINED_MODEL_PATH_2)

        endpoint_models = []
        for model_path in multi_data_model.list_models():
            endpoint_models.append(model_path)
        assert PRETRAINED_MODEL_PATH_1 in endpoint_models
        assert PRETRAINED_MODEL_PATH_2 in endpoint_models

        predictor = Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=NumpySerializer(),
            deserializer=string_deserializer,
        )

        data = numpy.zeros(shape=(1, 1, 28, 28))
        result = predictor.predict(data, target_model=PRETRAINED_MODEL_PATH_1)
        assert result == "Invoked model: {}".format(PRETRAINED_MODEL_PATH_1)

        result = predictor.predict(data, target_model=PRETRAINED_MODEL_PATH_2)
        assert result == "Invoked model: {}".format(PRETRAINED_MODEL_PATH_2)

        # Cleanup
        sagemaker_session.sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        multi_data_model.delete_model()
    with pytest.raises(Exception) as exception:
        sagemaker_session.sagemaker_client.describe_model(ModelName=multi_data_model.name)
        assert "Could not find model" in str(exception.value)
        sagemaker_session.sagemaker_client.describe_endpoint_config(name=endpoint_name)
        assert "Could not find endpoint" in str(exception.value)


@pytest.mark.local_mode
def test_multi_data_model_deploy_pretrained_models_local_mode(container_image, sagemaker_session):
    timestamp = sagemaker_timestamp()
    endpoint_name = "test-multimodel-endpoint-{}".format(timestamp)
    model_name = "test-multimodel-{}".format(timestamp)

    # Define pretrained model local path
    pretrained_model_data_local_path = os.path.join(DATA_DIR, "sparkml_model", "mleap_model.tar.gz")

    with timeout(minutes=30):
        model_data_prefix = os.path.join(
            "s3://", sagemaker_session.default_bucket(), "multimodel-{}/".format(timestamp)
        )
        multi_data_model = MultiDataModel(
            name=model_name,
            model_data_prefix=model_data_prefix,
            image_uri=container_image,
            role=ROLE,
            sagemaker_session=sagemaker_session,
        )

        # Add model before deploy
        multi_data_model.add_model(pretrained_model_data_local_path, PRETRAINED_MODEL_PATH_1)
        # Deploy model to an endpoint
        multi_data_model.deploy(1, "local", endpoint_name=endpoint_name)
        # Add models after deploy
        multi_data_model.add_model(pretrained_model_data_local_path, PRETRAINED_MODEL_PATH_2)

        endpoint_models = []
        for model_path in multi_data_model.list_models():
            endpoint_models.append(model_path)
        assert PRETRAINED_MODEL_PATH_1 in endpoint_models
        assert PRETRAINED_MODEL_PATH_2 in endpoint_models

        predictor = Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=multi_data_model.sagemaker_session,
            serializer=NumpySerializer(),
            deserializer=string_deserializer,
        )

        data = numpy.zeros(shape=(1, 1, 28, 28))
        result = predictor.predict(data, target_model=PRETRAINED_MODEL_PATH_1)
        assert result == "Invoked model: {}".format(PRETRAINED_MODEL_PATH_1)

        result = predictor.predict(data, target_model=PRETRAINED_MODEL_PATH_2)
        assert result == "Invoked model: {}".format(PRETRAINED_MODEL_PATH_2)

        # Cleanup
        multi_data_model.sagemaker_session.sagemaker_client.delete_endpoint_config(
            EndpointConfigName=endpoint_name
        )
        multi_data_model.sagemaker_session.delete_endpoint(endpoint_name)
        multi_data_model.delete_model()
    with pytest.raises(Exception) as exception:
        sagemaker_session.sagemaker_client.describe_model(ModelName=multi_data_model.name)
        assert "Could not find model" in str(exception.value)
        sagemaker_session.sagemaker_client.describe_endpoint_config(name=endpoint_name)
        assert "Could not find endpoint" in str(exception.value)


def test_multi_data_model_deploy_trained_model_from_framework_estimator(
    container_image,
    sagemaker_session,
    cpu_instance_type,
    mxnet_inference_latest_version,
    mxnet_inference_latest_py_version,
):
    timestamp = sagemaker_timestamp()
    endpoint_name = "test-multimodel-endpoint-{}".format(timestamp)
    model_name = "test-multimodel-{}".format(timestamp)

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        mxnet_model_1 = _mxnet_training_job(
            sagemaker_session,
            container_image,
            mxnet_inference_latest_version,
            mxnet_inference_latest_py_version,
            cpu_instance_type,
            0.1,
        )
        model_data_prefix = os.path.join(
            "s3://", sagemaker_session.default_bucket(), "multimodel-{}/".format(timestamp)
        )
        multi_data_model = MultiDataModel(
            name=model_name,
            model_data_prefix=model_data_prefix,
            model=mxnet_model_1,
            sagemaker_session=sagemaker_session,
        )

        # Add model before deploy
        multi_data_model.add_model(mxnet_model_1.model_data, PRETRAINED_MODEL_PATH_1)
        # Deploy model to an endpoint
        multi_data_model.deploy(1, cpu_instance_type, endpoint_name=endpoint_name)
        time.sleep(30)

        # Train another model
        mxnet_model_2 = _mxnet_training_job(
            sagemaker_session,
            container_image,
            mxnet_inference_latest_version,
            mxnet_inference_latest_py_version,
            cpu_instance_type,
            0.01,
        )
        # Deploy newly trained model
        multi_data_model.add_model(mxnet_model_2.model_data, PRETRAINED_MODEL_PATH_2)

        endpoint_models = []
        for model_path in multi_data_model.list_models():
            endpoint_models.append(model_path)
        assert PRETRAINED_MODEL_PATH_1 in endpoint_models
        assert PRETRAINED_MODEL_PATH_2 in endpoint_models

        # Define a predictor to set `serializer` parameter with `NumpySerializer`
        # instead of `JSONSerializer` in the default predictor returned by `MXNetPredictor`
        # Since we are using a placeholder container image the prediction results are not accurate.
        predictor = Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=NumpySerializer(),
            deserializer=string_deserializer,
        )

        data = numpy.zeros(shape=(1, 1, 28, 28))
        # Prediction result for the first model
        result = predictor.predict(data, target_model=PRETRAINED_MODEL_PATH_1)
        assert result == "Invoked model: {}".format(PRETRAINED_MODEL_PATH_1)

        # Prediction result for the second model
        result = predictor.predict(data, target_model=PRETRAINED_MODEL_PATH_2)
        assert result == "Invoked model: {}".format(PRETRAINED_MODEL_PATH_2)

        # Cleanup
        sagemaker_session.sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        multi_data_model.delete_model()
    with pytest.raises(Exception) as exception:
        sagemaker_session.sagemaker_client.describe_model(ModelName=model_name)
        assert "Could not find model" in str(exception.value)
        sagemaker_session.sagemaker_client.describe_endpoint_config(name=endpoint_name)
        assert "Could not find endpoint" in str(exception.value)


def _mxnet_training_job(
    sagemaker_session, container_image, mxnet_version, py_version, cpu_instance_type, learning_rate
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        mx = MXNet(
            entry_point=script_path,
            role=ROLE,
            framework_version=mxnet_version,
            py_version=py_version,
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            hyperparameters={"learning-rate": learning_rate},
        )

        train_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
        )
        test_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
        )

        mx.fit({"train": train_input, "test": test_input})

        # Replace the container image value for now since the frameworks do not support
        # multi-model container image yet.
        return mx.create_model(image_uri=container_image)


@pytest.mark.slow_test
def test_multi_data_model_deploy_train_model_from_amazon_first_party_estimator(
    container_image, sagemaker_session, cpu_instance_type
):
    timestamp = sagemaker_timestamp()
    endpoint_name = "test-multimodel-endpoint-{}".format(timestamp)
    model_name = "test-multimodel-{}".format(timestamp)

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        rcf_model_v1 = __rcf_training_job(
            sagemaker_session, container_image, cpu_instance_type, 50, 20
        )

        model_data_prefix = os.path.join(
            "s3://", sagemaker_session.default_bucket(), "multimodel-{}/".format(timestamp)
        )
        multi_data_model = MultiDataModel(
            name=model_name,
            model_data_prefix=model_data_prefix,
            model=rcf_model_v1,
            sagemaker_session=sagemaker_session,
        )

        # Add model before deploy
        multi_data_model.add_model(rcf_model_v1.model_data, PRETRAINED_MODEL_PATH_1)
        # Deploy model to an endpoint
        multi_data_model.deploy(1, cpu_instance_type, endpoint_name=endpoint_name)
        time.sleep(30)
        # Train another model
        rcf_model_v2 = __rcf_training_job(
            sagemaker_session, container_image, cpu_instance_type, 70, 20
        )
        # Deploy newly trained model
        multi_data_model.add_model(rcf_model_v2.model_data, PRETRAINED_MODEL_PATH_2)

        # List model assertions
        endpoint_models = []
        for model_path in multi_data_model.list_models():
            endpoint_models.append(model_path)
        assert PRETRAINED_MODEL_PATH_1 in endpoint_models
        assert PRETRAINED_MODEL_PATH_2 in endpoint_models

        # Define a predictor to set `serializer` parameter with `NumpySerializer`
        # instead of `JSONSerializer` in the default predictor returned by `MXNetPredictor`
        # Since we are using a placeholder container image the prediction results are not accurate.
        predictor = Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=NumpySerializer(),
            deserializer=string_deserializer,
        )

        data = numpy.random.rand(1, 14)
        # Prediction result for the first model
        result = predictor.predict(data, target_model=PRETRAINED_MODEL_PATH_1)
        assert result == "Invoked model: {}".format(PRETRAINED_MODEL_PATH_1)

        # Prediction result for the second model
        result = predictor.predict(data, target_model=PRETRAINED_MODEL_PATH_2)
        assert result == "Invoked model: {}".format(PRETRAINED_MODEL_PATH_2)

        # Cleanup
        sagemaker_session.sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        multi_data_model.delete_model()
    with pytest.raises(Exception) as exception:
        sagemaker_session.sagemaker_client.describe_model(ModelName=model_name)
        assert "Could not find model" in str(exception.value)
        sagemaker_session.sagemaker_client.describe_endpoint_config(name=endpoint_name)
        assert "Could not find endpoint" in str(exception.value)


def __rcf_training_job(
    sagemaker_session, container_image, cpu_instance_type, num_trees, num_samples_per_tree
):
    job_name = unique_name_from_base("randomcutforest")
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        # Generate a thousand 14-dimensional datapoints.
        feature_num = 14
        train_input = numpy.random.rand(1000, feature_num)

        rcf = RandomCutForest(
            role=ROLE,
            instance_count=1,
            instance_type=cpu_instance_type,
            num_trees=num_trees,
            num_samples_per_tree=num_samples_per_tree,
            eval_metrics=["accuracy", "precision_recall_fscore"],
            sagemaker_session=sagemaker_session,
        )

        rcf.fit(records=rcf.record_set(train_input), job_name=job_name)

        # Replace the container image value with a multi-model container image for now since the
        # frameworks do not support multi-model container image yet.
        rcf_model = rcf.create_model()
        rcf_model.image_uri = container_image
        return rcf_model


def test_multi_data_model_deploy_pretrained_models_update_endpoint(
    container_image, sagemaker_session, cpu_instance_type, alternative_cpu_instance_type
):
    timestamp = sagemaker_timestamp()
    endpoint_name = "test-multimodel-endpoint-{}".format(timestamp)
    model_name = "test-multimodel-{}".format(timestamp)

    # Define pretrained model local path
    pretrained_model_data_local_path = os.path.join(DATA_DIR, "sparkml_model", "mleap_model.tar.gz")

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        model_data_prefix = os.path.join(
            "s3://", sagemaker_session.default_bucket(), "multimodel-{}/".format(timestamp)
        )
        multi_data_model = MultiDataModel(
            name=model_name,
            model_data_prefix=model_data_prefix,
            image_uri=container_image,
            role=ROLE,
            sagemaker_session=sagemaker_session,
        )

        # Add model before deploy
        multi_data_model.add_model(pretrained_model_data_local_path, PRETRAINED_MODEL_PATH_1)
        # Deploy model to an endpoint
        multi_data_model.deploy(1, cpu_instance_type, endpoint_name=endpoint_name)
        time.sleep(30)
        # Add model after deploy
        multi_data_model.add_model(pretrained_model_data_local_path, PRETRAINED_MODEL_PATH_2)

        # List model assertions
        endpoint_models = []
        for model_path in multi_data_model.list_models():
            endpoint_models.append(model_path)
        assert PRETRAINED_MODEL_PATH_1 in endpoint_models
        assert PRETRAINED_MODEL_PATH_2 in endpoint_models

        predictor = Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=NumpySerializer(),
            deserializer=string_deserializer,
        )

        data = numpy.zeros(shape=(1, 1, 28, 28))
        result = predictor.predict(data, target_model=PRETRAINED_MODEL_PATH_1)
        assert result == "Invoked model: {}".format(PRETRAINED_MODEL_PATH_1)

        result = predictor.predict(data, target_model=PRETRAINED_MODEL_PATH_2)
        assert result == "Invoked model: {}".format(PRETRAINED_MODEL_PATH_2)

        endpoint_desc = sagemaker_session.sagemaker_client.describe_endpoint(
            EndpointName=endpoint_name
        )
        old_config_name = endpoint_desc["EndpointConfigName"]

        # Update endpoint
        predictor.update_endpoint(
            initial_instance_count=1, instance_type=alternative_cpu_instance_type
        )

        endpoint_desc = sagemaker_session.sagemaker_client.describe_endpoint(
            EndpointName=endpoint_name
        )
        new_config_name = endpoint_desc["EndpointConfigName"]

        new_config = sagemaker_session.sagemaker_client.describe_endpoint_config(
            EndpointConfigName=new_config_name
        )
        assert old_config_name != new_config_name
        assert new_config["ProductionVariants"][0]["InstanceType"] == alternative_cpu_instance_type
        assert new_config["ProductionVariants"][0]["InitialInstanceCount"] == 1

        # Cleanup
        sagemaker_session.sagemaker_client.delete_endpoint_config(
            EndpointConfigName=old_config_name
        )
        sagemaker_session.sagemaker_client.delete_endpoint_config(
            EndpointConfigName=new_config_name
        )
        multi_data_model.delete_model()

    with pytest.raises(Exception) as exception:
        sagemaker_session.sagemaker_client.describe_model(ModelName=model_name)
        assert "Could not find model" in str(exception.value)
        sagemaker_session.sagemaker_client.describe_endpoint_config(name=old_config_name)
        assert "Could not find endpoint" in str(exception.value)
        sagemaker_session.sagemaker_client.describe_endpoint_config(name=new_config_name)
        assert "Could not find endpoint" in str(exception.value)
