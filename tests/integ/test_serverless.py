# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from sagemaker.serverless import LambdaModel
from sagemaker.utils import unique_name_from_base

# See tests/data/serverless for the image source code.
IMAGE_URI = "142577830533.dkr.ecr.us-west-2.amazonaws.com/serverless-integ-test:latest"
ROLE = "arn:aws:iam::142577830533:role/lambda_basic_execution"
URL = "https://c.files.bbci.co.uk/12A9B/production/_111434467_gettyimages-1143489763.jpg"


@pytest.mark.skipif(
    "CODEBUILD_BUILD_ID" not in os.environ,
    reason="The container image is private to the CI account.",
)
def test_lambda():
    model = LambdaModel(image_uri=IMAGE_URI, role=ROLE)

    predictor = model.deploy(
        unique_name_from_base("my-lambda-function"), timeout=60, memory_size=4092
    )
    prediction = predictor.predict({"url": URL})

    assert prediction == {"class": "tabby"}

    model.destroy()
    predictor.destroy()
