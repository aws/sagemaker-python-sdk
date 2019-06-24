# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import gzip
import json
import os
import pickle
import sys

import pytest

import sagemaker
from sagemaker.amazon.amazon_estimator import registry
from sagemaker.estimator import Estimator
from sagemaker.utils import unique_name_from_base
from tests.integ import DATA_DIR, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


@pytest.fixture(scope="module")
def region(sagemaker_session):
    return sagemaker_session.boto_session.region_name


def fm_serializer(data):
    js = {"instances": []}
    for row in data:
        js["instances"].append({"features": row.tolist()})
    return json.dumps(js)


@pytest.mark.canary_quick
def test_byo_estimator(sagemaker_session, region):
    """Use Factorization Machines algorithm as an example here.

    First we need to prepare data for training. We take standard data set, convert it to the
    format that the algorithm can process and upload it to S3.
    Then we create the Estimator and set hyperparamets as required by the algorithm.
    Next, we can call fit() with path to the S3.
    Later the trained model is deployed and prediction is called against the endpoint.
    Default predictor is updated with json serializer and deserializer.

    """
    image_name = registry(region) + "/factorization-machines:1"
    training_data_path = os.path.join(DATA_DIR, "dummy_tensor")
    job_name = unique_name_from_base("byo")

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        data_path = os.path.join(DATA_DIR, "one_p_mnist", "mnist.pkl.gz")
        pickle_args = {} if sys.version_info.major == 2 else {"encoding": "latin1"}

        with gzip.open(data_path, "rb") as f:
            train_set, _, _ = pickle.load(f, **pickle_args)

        prefix = "test_byo_estimator"
        key = "recordio-pb-data"

        s3_train_data = sagemaker_session.upload_data(
            path=training_data_path, key_prefix=os.path.join(prefix, "train", key)
        )

        estimator = Estimator(
            image_name=image_name,
            role="SageMakerRole",
            train_instance_count=1,
            train_instance_type="ml.c4.xlarge",
            sagemaker_session=sagemaker_session,
        )

        estimator.set_hyperparameters(
            num_factors=10, feature_dim=784, mini_batch_size=100, predictor_type="binary_classifier"
        )

        # training labels must be 'float32'
        estimator.fit({"train": s3_train_data}, job_name=job_name)

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        model = estimator.create_model()
        predictor = model.deploy(1, "ml.m4.xlarge", endpoint_name=job_name)
        predictor.serializer = fm_serializer
        predictor.content_type = "application/json"
        predictor.deserializer = sagemaker.predictor.json_deserializer

        result = predictor.predict(train_set[0][:10])

        assert len(result["predictions"]) == 10
        for prediction in result["predictions"]:
            assert prediction["score"] is not None


def test_async_byo_estimator(sagemaker_session, region):
    image_name = registry(region) + "/factorization-machines:1"
    endpoint_name = unique_name_from_base("byo")
    training_data_path = os.path.join(DATA_DIR, "dummy_tensor")
    job_name = unique_name_from_base("byo")

    with timeout(minutes=5):
        data_path = os.path.join(DATA_DIR, "one_p_mnist", "mnist.pkl.gz")
        pickle_args = {} if sys.version_info.major == 2 else {"encoding": "latin1"}

        with gzip.open(data_path, "rb") as f:
            train_set, _, _ = pickle.load(f, **pickle_args)

        prefix = "test_byo_estimator"
        key = "recordio-pb-data"

        s3_train_data = sagemaker_session.upload_data(
            path=training_data_path, key_prefix=os.path.join(prefix, "train", key)
        )

        estimator = Estimator(
            image_name=image_name,
            role="SageMakerRole",
            train_instance_count=1,
            train_instance_type="ml.c4.xlarge",
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
        predictor = model.deploy(1, "ml.m4.xlarge", endpoint_name=endpoint_name)
        predictor.serializer = fm_serializer
        predictor.content_type = "application/json"
        predictor.deserializer = sagemaker.predictor.json_deserializer

        result = predictor.predict(train_set[0][:10])

        assert len(result["predictions"]) == 10
        for prediction in result["predictions"]:
            assert prediction["score"] is not None

        assert estimator.train_image() == image_name
