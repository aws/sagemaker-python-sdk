# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import os

import pytest

import sagemaker
from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.serializers import SimpleBaseSerializer
from sagemaker.utils import unique_name_from_base
from tests.integ import DATA_DIR, TRAINING_DEFAULT_TIMEOUT_MINUTES, datasets
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


@pytest.fixture(scope="module")
def region(sagemaker_session):
    return sagemaker_session.boto_session.region_name


@pytest.fixture
def training_set():
    return datasets.one_p_mnist()


class _FactorizationMachineSerializer(SimpleBaseSerializer):
    # SimpleBaseSerializer already uses "application/json" CONTENT_TYPE by default

    def serialize(self, data):
        js = {"instances": []}
        for row in data:
            js["instances"].append({"features": row.tolist()})
        return json.dumps(js)


@pytest.mark.release
def test_byo_estimator(sagemaker_session, region, cpu_instance_type, training_set):
    """Use Factorization Machines algorithm as an example here.

    First we need to prepare data for training. We take standard data set, convert it to the
    format that the algorithm can process and upload it to S3.
    Then we create the Estimator and set hyperparamets as required by the algorithm.
    Next, we can call fit() with path to the S3.
    Later the trained model is deployed and prediction is called against the endpoint.
    Default predictor is updated with json serializer and deserializer.

    """
    image_uri = image_uris.retrieve("factorization-machines", region)
    training_data_path = os.path.join(DATA_DIR, "dummy_tensor")
    job_name = unique_name_from_base("byo")

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        prefix = "test_byo_estimator"
        key = "recordio-pb-data"

        s3_train_data = sagemaker_session.upload_data(
            path=training_data_path, key_prefix=os.path.join(prefix, "train", key)
        )

        estimator = Estimator(
            image_uri=image_uri,
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
        )

        estimator.set_hyperparameters(
            num_factors=10, feature_dim=784, mini_batch_size=100, predictor_type="binary_classifier"
        )

        # training labels must be 'float32'
        estimator.fit({"train": s3_train_data}, job_name=job_name)

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        model = estimator.create_model()
        predictor = model.deploy(
            1,
            cpu_instance_type,
            endpoint_name=job_name,
            serializer=_FactorizationMachineSerializer(),
            deserializer=sagemaker.deserializers.JSONDeserializer(),
        )

        result = predictor.predict(training_set[0][:10])

        assert len(result["predictions"]) == 10
        for prediction in result["predictions"]:
            assert prediction["score"] is not None


def test_async_byo_estimator(sagemaker_session, region, cpu_instance_type, training_set):
    image_uri = image_uris.retrieve("factorization-machines", region)
    endpoint_name = unique_name_from_base("byo")
    training_data_path = os.path.join(DATA_DIR, "dummy_tensor")
    job_name = unique_name_from_base("byo")

    with timeout(minutes=5):
        prefix = "test_byo_estimator"
        key = "recordio-pb-data"

        s3_train_data = sagemaker_session.upload_data(
            path=training_data_path, key_prefix=os.path.join(prefix, "train", key)
        )

        estimator = Estimator(
            image_uri=image_uri,
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
        )

        estimator.set_hyperparameters(
            num_factors=10, feature_dim=784, mini_batch_size=100, predictor_type="binary_classifier"
        )

        # training labels must be 'float32'
        estimator.fit({"train": s3_train_data}, wait=False, job_name=job_name)

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        estimator = Estimator.attach(
            training_job_name=job_name, sagemaker_session=sagemaker_session
        )
        model = estimator.create_model()
        predictor = model.deploy(
            1,
            cpu_instance_type,
            endpoint_name=endpoint_name,
            serializer=_FactorizationMachineSerializer(),
            deserializer=sagemaker.deserializers.JSONDeserializer(),
        )

        result = predictor.predict(training_set[0][:10])

        assert len(result["predictions"]) == 10
        for prediction in result["predictions"]:
            assert prediction["score"] is not None

        assert estimator.training_image_uri() == image_uri
