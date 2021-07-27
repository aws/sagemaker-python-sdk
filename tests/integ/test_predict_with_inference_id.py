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
import pytest

import tests.integ
import tests.integ.timeout

from sagemaker import image_uris
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.utils import unique_name_from_base

from tests.integ import DATA_DIR


ROLE = "SageMakerRole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c5.xlarge"
TEST_CSV_DATA = "42,42,42,42,42,42,42"
XGBOOST_DATA_PATH = os.path.join(DATA_DIR, "xgboost_model")


@pytest.yield_fixture(scope="module")
def endpoint_name(sagemaker_session):
    endpoint_name = unique_name_from_base("model-inference-id-integ")
    xgb_model_data = sagemaker_session.upload_data(
        path=os.path.join(XGBOOST_DATA_PATH, "xgb_model.tar.gz"),
        key_prefix="integ-test-data/xgboost/model",
    )

    xgb_image = image_uris.retrieve(
        "xgboost",
        sagemaker_session.boto_region_name,
        version="1",
        image_scope="inference",
    )

    with tests.integ.timeout.timeout_and_delete_endpoint_by_name(
        endpoint_name=endpoint_name, sagemaker_session=sagemaker_session, hours=2
    ):
        xgb_model = Model(
            model_data=xgb_model_data,
            image_uri=xgb_image,
            name=endpoint_name,  # model name
            role=ROLE,
            sagemaker_session=sagemaker_session,
        )
        xgb_model.deploy(INSTANCE_COUNT, INSTANCE_TYPE, endpoint_name=endpoint_name)
        yield endpoint_name


def test_predict_with_inference_id(sagemaker_session, endpoint_name):
    predictor = Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sagemaker_session,
        serializer=CSVSerializer(),
    )

    # Validate that no exception is raised when the target_variant is specified.
    response = predictor.predict(TEST_CSV_DATA, inference_id="foo")
    assert response


def test_invoke_endpoint_with_inference_id(sagemaker_session, endpoint_name):
    response = sagemaker_session.sagemaker_runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=TEST_CSV_DATA,
        ContentType="text/csv",
        Accept="text/csv",
        InferenceId="foo",
    )
    assert response
