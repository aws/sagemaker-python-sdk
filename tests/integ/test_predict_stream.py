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
import os
import pytest

import tests.integ
import tests.integ.timeout

from sagemaker import image_uris
from sagemaker.iterators import LineIterator
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.utils import unique_name_from_base

from tests.integ import DATA_DIR


ROLE = "SageMakerRole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.g5.2xlarge"
LMI_FALCON_7B_DATA_PATH = os.path.join(DATA_DIR, "lmi-model-falcon-7b")


@pytest.yield_fixture(scope="module")
def endpoint_name(sagemaker_session):
    lmi_endpoint_name = unique_name_from_base("lmi-model-falcon-7b")
    model_data = sagemaker_session.upload_data(
        path=os.path.join(LMI_FALCON_7B_DATA_PATH, "mymodel-7B.tar.gz"),
        key_prefix="large-model-lmi/code",
    )

    image_uri = image_uris.retrieve(
        framework="djl-deepspeed", region=sagemaker_session.boto_region_name, version="0.23.0"
    )

    with tests.integ.timeout.timeout_and_delete_endpoint_by_name(
        endpoint_name=lmi_endpoint_name, sagemaker_session=sagemaker_session, hours=2
    ):
        lmi_model = Model(
            sagemaker_session=sagemaker_session,
            model_data=model_data,
            image_uri=image_uri,
            name=lmi_endpoint_name,  # model name
            role=ROLE,
        )
        lmi_model.deploy(
            INSTANCE_COUNT,
            INSTANCE_TYPE,
            endpoint_name=lmi_endpoint_name,
            container_startup_health_check_timeout=900,
        )
        yield lmi_endpoint_name


def test_predict_stream(sagemaker_session, endpoint_name):
    data = {"inputs": "what does AWS stand for?", "parameters": {"max_new_tokens": 400}}
    initial_args = {"ContentType": "application/json"}
    predictor = Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sagemaker_session,
    )

    # Validate that no exception is raised when the target_variant is specified.
    stream_iterator = predictor.predict_stream(
        data=json.dumps(data),
        initial_args=initial_args,
        iterator=LineIterator,
    )

    response = ""
    for line in stream_iterator:
        resp = json.loads(line)
        response += resp.get("outputs")[0]

    assert "AWS stands for Amazon Web Services." in response

    data = {"inputs": "what does AWS stand for?", "parameters": {"max_new_tokens": 400}}
    initial_args = {"ContentType": "application/json"}
    predictor = Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sagemaker_session,
    )

    # Validate that no exception is raised when the target_variant is specified.
    # uses the default `sagemaker.iterator.ByteIterator`
    stream_iterator = predictor.predict_stream(
        data=json.dumps(data),
        initial_args=initial_args,
    )

    response = ""
    for line in stream_iterator:
        resp = json.loads(line)
        response += resp.get("outputs")[0]

    assert "AWS stands for Amazon Web Services." in response
