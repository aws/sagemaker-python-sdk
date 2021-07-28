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

from botocore.errorfactory import ClientError

from sagemaker.sparkml.model import SparkMLModel
from sagemaker.utils import sagemaker_timestamp
from tests.integ import DATA_DIR
from tests.integ.timeout import timeout_and_delete_endpoint_by_name


@pytest.mark.release
def test_sparkml_model_deploy(sagemaker_session, cpu_instance_type):
    # Uploads an MLeap serialized MLeap model to S3 and use that to deploy
    # a SparkML model to perform inference
    data_path = os.path.join(DATA_DIR, "sparkml_model")
    endpoint_name = "test-sparkml-deploy-{}".format(sagemaker_timestamp())
    model_data = sagemaker_session.upload_data(
        path=os.path.join(data_path, "mleap_model.tar.gz"),
        key_prefix="integ-test-data/sparkml/model",
    )
    schema = json.dumps(
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
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        model = SparkMLModel(
            model_data=model_data,
            role="SageMakerRole",
            sagemaker_session=sagemaker_session,
            env={"SAGEMAKER_SPARKML_SCHEMA": schema},
        )
        predictor = model.deploy(1, cpu_instance_type, endpoint_name=endpoint_name)

        valid_data = "1.0,C,38.0,71.5,1.0,female"
        assert predictor.predict(valid_data) == b"1.0,0.0,38.0,1.0,71.5,0.0,1.0"

        invalid_data = "1.0,28.0,C,38.0,71.5,1.0"
        with pytest.raises(ClientError):
            predictor.predict(invalid_data)
