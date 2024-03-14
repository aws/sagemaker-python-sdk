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
import math
import os

import pytest
import scipy.stats as st

from sagemaker import image_uris
from sagemaker.deserializers import CSVDeserializer
from sagemaker.s3 import S3Uploader
from sagemaker.session import production_variant
from sagemaker.sparkml import SparkMLModel
from sagemaker.utils import unique_name_from_base
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
import tests.integ


ROLE = "SageMakerRole"
MODEL_NAME = unique_name_from_base("test-xgboost-model")
DEFAULT_REGION = "us-west-2"
DEFAULT_INSTANCE_TYPE = "ml.m5.xlarge"
DEFAULT_INSTANCE_COUNT = 1
XG_BOOST_MODEL_LOCAL_PATH = os.path.join(tests.integ.DATA_DIR, "xgboost_model", "xgb_model.tar.gz")

TEST_VARIANT_1 = "Variant1"
TEST_VARIANT_1_WEIGHT = 0.3

TEST_VARIANT_2 = "Variant2"
TEST_VARIANT_2_WEIGHT = 0.7

VARIANT_TRAFFIC_SAMPLING_COUNT = 100
DESIRED_CONFIDENCE_FOR_VARIANT_TRAFFIC_DISTRIBUTION = 0.999

TEST_CSV_DATA = "42,42,42,42,42,42,42"

SPARK_ML_MODEL_LOCAL_PATH = os.path.join(
    tests.integ.DATA_DIR, "sparkml_model", "mleap_model.tar.gz"
)
SPARK_ML_DEFAULT_VARIANT_NAME = (
    "AllTraffic"  # default defined in src/sagemaker/session.py def production_variant
)
SPARK_ML_WRONG_VARIANT_NAME = "WRONG_VARIANT"
SPARK_ML_TEST_DATA = "1.0,C,38.0,71.5,1.0,female"
SPARK_ML_MODEL_SCHEMA = json.dumps(
    {
        "input": [
            {"name": "Pclass", "type": "float"},
            {"name": "Embarked", "type": "string"},
            {"name": "Age", "type": "float"},
            {"name": "Fare", "type": "float"},
            {"name": "SibSp", "type": "float"},
            {"name": "Sex", "type": "string"},
        ],
        "output": {"name": "features", "struct": "vector", "type": "double"},
    }
)


@pytest.fixture(scope="module")
def multi_variant_endpoint(sagemaker_session):
    """
    Sets up the multi variant endpoint before the integration tests run.
    Cleans up the multi variant endpoint after the integration tests run.
    """
    multi_variant_endpoint.endpoint_name = unique_name_from_base(
        "integ-test-multi-variant-endpoint"
    )
    with tests.integ.timeout.timeout_and_delete_endpoint_by_name(
        endpoint_name=multi_variant_endpoint.endpoint_name,
        sagemaker_session=sagemaker_session,
        hours=2,
    ):

        # Creating a model
        bucket = sagemaker_session.default_bucket()
        prefix = "sagemaker/DEMO-VariantTargeting"
        model_url = S3Uploader.upload(
            local_path=XG_BOOST_MODEL_LOCAL_PATH,
            desired_s3_uri="s3://{}/{}".format(bucket, prefix),
            sagemaker_session=sagemaker_session,
        )

        image_uri = image_uris.retrieve(
            "xgboost",
            sagemaker_session.boto_region_name,
            version="0.90-1",
            instance_type=DEFAULT_INSTANCE_TYPE,
            image_scope="inference",
        )
        multi_variant_endpoint_model = sagemaker_session.create_model(
            name=MODEL_NAME,
            role=ROLE,
            container_defs={"Image": image_uri, "ModelDataUrl": model_url},
        )

        # Creating a multi variant endpoint
        variant1 = production_variant(
            model_name=MODEL_NAME,
            instance_type=DEFAULT_INSTANCE_TYPE,
            initial_instance_count=DEFAULT_INSTANCE_COUNT,
            variant_name=TEST_VARIANT_1,
            initial_weight=TEST_VARIANT_1_WEIGHT,
        )
        variant2 = production_variant(
            model_name=MODEL_NAME,
            instance_type=DEFAULT_INSTANCE_TYPE,
            initial_instance_count=DEFAULT_INSTANCE_COUNT,
            variant_name=TEST_VARIANT_2,
            initial_weight=TEST_VARIANT_2_WEIGHT,
        )
        sagemaker_session.endpoint_from_production_variants(
            name=multi_variant_endpoint.endpoint_name, production_variants=[variant1, variant2]
        )

        # Yield to run the integration tests
        yield multi_variant_endpoint

        # Cleanup resources
        sagemaker_session.delete_model(multi_variant_endpoint_model)
        sagemaker_session.sagemaker_client.delete_endpoint_config(
            EndpointConfigName=multi_variant_endpoint.endpoint_name
        )

    # Validate resource cleanup
    with pytest.raises(Exception) as exception:
        sagemaker_session.sagemaker_client.describe_model(
            ModelName=multi_variant_endpoint_model.name
        )
        assert "Could not find model" in str(exception.value)
        sagemaker_session.sagemaker_client.describe_endpoint_config(
            name=multi_variant_endpoint.endpoint_name
        )
        assert "Could not find endpoint" in str(exception.value)


def test_target_variant_invocation(sagemaker_session, multi_variant_endpoint):

    response = sagemaker_session.sagemaker_runtime_client.invoke_endpoint(
        EndpointName=multi_variant_endpoint.endpoint_name,
        Body=TEST_CSV_DATA,
        ContentType="text/csv",
        Accept="text/csv",
        TargetVariant=TEST_VARIANT_1,
    )
    assert response["InvokedProductionVariant"] == TEST_VARIANT_1

    response = sagemaker_session.sagemaker_runtime_client.invoke_endpoint(
        EndpointName=multi_variant_endpoint.endpoint_name,
        Body=TEST_CSV_DATA,
        ContentType="text/csv",
        Accept="text/csv",
        TargetVariant=TEST_VARIANT_2,
    )
    assert response["InvokedProductionVariant"] == TEST_VARIANT_2


def test_predict_invocation_with_target_variant(sagemaker_session, multi_variant_endpoint):
    predictor = Predictor(
        endpoint_name=multi_variant_endpoint.endpoint_name,
        sagemaker_session=sagemaker_session,
        serializer=CSVSerializer(),
    )

    # Validate that no exception is raised when the target_variant is specified.
    predictor.predict(TEST_CSV_DATA, target_variant=TEST_VARIANT_1)
    predictor.predict(TEST_CSV_DATA, target_variant=TEST_VARIANT_2)


def test_variant_traffic_distribution(sagemaker_session, multi_variant_endpoint):
    variant_1_invocation_count = 0
    variant_2_invocation_count = 0

    for i in range(0, VARIANT_TRAFFIC_SAMPLING_COUNT):
        response = sagemaker_session.sagemaker_runtime_client.invoke_endpoint(
            EndpointName=multi_variant_endpoint.endpoint_name,
            Body=TEST_CSV_DATA,
            ContentType="text/csv",
            Accept="text/csv",
        )
        if response["InvokedProductionVariant"] == TEST_VARIANT_1:
            variant_1_invocation_count += 1
        elif response["InvokedProductionVariant"] == TEST_VARIANT_2:
            variant_2_invocation_count += 1

    assert variant_1_invocation_count + variant_2_invocation_count == VARIANT_TRAFFIC_SAMPLING_COUNT

    variant_1_invocation_percentage = float(variant_1_invocation_count) / float(
        VARIANT_TRAFFIC_SAMPLING_COUNT
    )
    variant_1_margin_of_error = _compute_and_retrieve_margin_of_error(TEST_VARIANT_1_WEIGHT)
    assert variant_1_invocation_percentage < TEST_VARIANT_1_WEIGHT + variant_1_margin_of_error
    assert variant_1_invocation_percentage > TEST_VARIANT_1_WEIGHT - variant_1_margin_of_error

    variant_2_invocation_percentage = float(variant_2_invocation_count) / float(
        VARIANT_TRAFFIC_SAMPLING_COUNT
    )
    variant_2_margin_of_error = _compute_and_retrieve_margin_of_error(TEST_VARIANT_2_WEIGHT)
    assert variant_2_invocation_percentage < TEST_VARIANT_2_WEIGHT + variant_2_margin_of_error
    assert variant_2_invocation_percentage > TEST_VARIANT_2_WEIGHT - variant_2_margin_of_error


def test_spark_ml_predict_invocation_with_target_variant(sagemaker_session):

    spark_ml_model_endpoint_name = unique_name_from_base("integ-test-target-variant-sparkml")

    model_data = sagemaker_session.upload_data(
        path=SPARK_ML_MODEL_LOCAL_PATH, key_prefix="integ-test-data/sparkml/model"
    )

    with tests.integ.timeout.timeout_and_delete_endpoint_by_name(
        spark_ml_model_endpoint_name, sagemaker_session
    ):
        spark_ml_model = SparkMLModel(
            model_data=model_data,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            env={"SAGEMAKER_SPARKML_SCHEMA": SPARK_ML_MODEL_SCHEMA},
        )

        predictor = spark_ml_model.deploy(
            DEFAULT_INSTANCE_COUNT,
            DEFAULT_INSTANCE_TYPE,
            endpoint_name=spark_ml_model_endpoint_name,
        )

        # Validate that no exception is raised when the target_variant is specified.
        predictor.predict(SPARK_ML_TEST_DATA, target_variant=SPARK_ML_DEFAULT_VARIANT_NAME)

        with pytest.raises(Exception) as exception_info:
            predictor.predict(SPARK_ML_TEST_DATA, target_variant=SPARK_ML_WRONG_VARIANT_NAME)

        assert "ValidationError" in str(exception_info.value)
        assert SPARK_ML_WRONG_VARIANT_NAME in str(exception_info.value)

        # cleanup resources
        spark_ml_model.delete_model()
        sagemaker_session.sagemaker_client.delete_endpoint_config(
            EndpointConfigName=spark_ml_model_endpoint_name
        )

    # Validate resource cleanup
    with pytest.raises(Exception) as exception:
        sagemaker_session.sagemaker_client.describe_model(ModelName=spark_ml_model.name)
        assert "Could not find model" in str(exception.value)
        sagemaker_session.sagemaker_client.describe_endpoint_config(
            name=spark_ml_model_endpoint_name
        )
        assert "Could not find endpoint" in str(exception.value)


@pytest.mark.local_mode
def test_target_variant_invocation_local_mode(sagemaker_session, multi_variant_endpoint):

    if sagemaker_session._region_name is None:
        sagemaker_session._region_name = DEFAULT_REGION

    response = sagemaker_session.sagemaker_runtime_client.invoke_endpoint(
        EndpointName=multi_variant_endpoint.endpoint_name,
        Body=TEST_CSV_DATA,
        ContentType="text/csv",
        Accept="text/csv",
        TargetVariant=TEST_VARIANT_1,
    )
    assert response["InvokedProductionVariant"] == TEST_VARIANT_1

    response = sagemaker_session.sagemaker_runtime_client.invoke_endpoint(
        EndpointName=multi_variant_endpoint.endpoint_name,
        Body=TEST_CSV_DATA,
        ContentType="text/csv",
        Accept="text/csv",
        TargetVariant=TEST_VARIANT_2,
    )
    assert response["InvokedProductionVariant"] == TEST_VARIANT_2


@pytest.mark.local_mode
def test_predict_invocation_with_target_variant_local_mode(
    sagemaker_session, multi_variant_endpoint
):

    if sagemaker_session._region_name is None:
        sagemaker_session._region_name = DEFAULT_REGION

    predictor = Predictor(
        endpoint_name=multi_variant_endpoint.endpoint_name,
        sagemaker_session=sagemaker_session,
        serializer=CSVSerializer(),
        deserializer=CSVDeserializer(),
    )

    # Validate that no exception is raised when the target_variant is specified.
    predictor.predict(TEST_CSV_DATA, target_variant=TEST_VARIANT_1)
    predictor.predict(TEST_CSV_DATA, target_variant=TEST_VARIANT_2)


def _compute_and_retrieve_margin_of_error(variant_weight):
    """
    Computes the margin of error using the Wald method for computing the confidence
    intervals of a binomial distribution.
    """
    z_value = st.norm.ppf(DESIRED_CONFIDENCE_FOR_VARIANT_TRAFFIC_DISTRIBUTION)
    margin_of_error = (variant_weight * (1 - variant_weight)) / VARIANT_TRAFFIC_SAMPLING_COUNT
    margin_of_error = z_value * math.sqrt(margin_of_error)
    return margin_of_error
