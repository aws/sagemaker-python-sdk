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

import json
import os

import pytest
from tests.integ import DATA_DIR
from tests.integ.timeout import timeout_and_delete_endpoint_by_name

from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.content_types import CONTENT_TYPE_CSV
from sagemaker.model import Model
from sagemaker.pipeline import PipelineModel
from sagemaker.predictor import RealTimePredictor, json_serializer
from sagemaker.sparkml.model import SparkMLModel
from sagemaker.utils import sagemaker_timestamp


@pytest.mark.continuous_testing
@pytest.mark.regional_testing
def test_inference_pipeline_model_deploy(sagemaker_session):
    sparkml_data_path = os.path.join(DATA_DIR, 'sparkml_model')
    xgboost_data_path = os.path.join(DATA_DIR, 'xgboost_model')
    endpoint_name = 'test-inference-pipeline-deploy-{}'.format(sagemaker_timestamp())
    sparkml_model_data = sagemaker_session.upload_data(
        path=os.path.join(sparkml_data_path, 'mleap_model.tar.gz'),
        key_prefix='integ-test-data/sparkml/model')
    xgb_model_data = sagemaker_session.upload_data(
        path=os.path.join(xgboost_data_path, 'xgb_model.tar.gz'),
        key_prefix='integ-test-data/xgboost/model')
    schema = json.dumps({
        "input": [
            {
                "name": "Pclass",
                "type": "float"
            },
            {
                "name": "Embarked",
                "type": "string"
            },
            {
                "name": "Age",
                "type": "float"
            },
            {
                "name": "Fare",
                "type": "float"
            },
            {
                "name": "SibSp",
                "type": "float"
            },
            {
                "name": "Sex",
                "type": "string"
            }
        ],
        "output": {
            "name": "features",
            "struct": "vector",
            "type": "double"
        }
    })
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        sparkml_model = SparkMLModel(model_data=sparkml_model_data,
                                     env={'SAGEMAKER_SPARKML_SCHEMA': schema},
                                     sagemaker_session=sagemaker_session)
        xgb_image = get_image_uri(sagemaker_session.boto_region_name, 'xgboost')
        xgb_model = Model(model_data=xgb_model_data, image=xgb_image,
                          sagemaker_session=sagemaker_session)
        model = PipelineModel(models=[sparkml_model, xgb_model], role='SageMakerRole',
                              sagemaker_session=sagemaker_session, name=endpoint_name)
        model.deploy(1, 'ml.m4.xlarge', endpoint_name=endpoint_name)
        predictor = RealTimePredictor(endpoint=endpoint_name, sagemaker_session=sagemaker_session,
                                      serializer=json_serializer, content_type=CONTENT_TYPE_CSV,
                                      accept=CONTENT_TYPE_CSV)

        valid_data = '1.0,C,38.0,71.5,1.0,female'
        assert predictor.predict(valid_data) == "0.714013934135"

        invalid_data = "1.0,28.0,C,38.0,71.5,1.0"
        assert (predictor.predict(invalid_data) is None)

    model.delete_model()
    with pytest.raises(Exception) as exception:
        sagemaker_session.sagemaker_client.describe_model(ModelName=model.name)
        assert 'Could not find model' in str(exception.value)
