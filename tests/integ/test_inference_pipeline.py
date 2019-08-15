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
import time

import pytest
from tests.integ import DATA_DIR, TRANSFORM_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import (
    timeout_and_delete_endpoint_by_name,
    timeout_and_delete_model_with_transformer,
)

from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.content_types import CONTENT_TYPE_CSV
from sagemaker.model import Model
from sagemaker.pipeline import PipelineModel
from sagemaker.predictor import RealTimePredictor, json_serializer
from sagemaker.sparkml.model import SparkMLModel
from sagemaker.utils import sagemaker_timestamp

SPARKML_DATA_PATH = os.path.join(DATA_DIR, "sparkml_model")
XGBOOST_DATA_PATH = os.path.join(DATA_DIR, "xgboost_model")
SPARKML_XGBOOST_DATA_DIR = "sparkml_xgboost_pipeline"
VALID_DATA_PATH = os.path.join(DATA_DIR, SPARKML_XGBOOST_DATA_DIR, "valid_input.csv")
INVALID_DATA_PATH = os.path.join(DATA_DIR, SPARKML_XGBOOST_DATA_DIR, "invalid_input.csv")
SCHEMA = json.dumps(
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


@pytest.mark.continuous_testing
@pytest.mark.regional_testing
def test_inference_pipeline_batch_transform(sagemaker_session, cpu_instance_type):
    sparkml_model_data = sagemaker_session.upload_data(
        path=os.path.join(SPARKML_DATA_PATH, "mleap_model.tar.gz"),
        key_prefix="integ-test-data/sparkml/model",
    )
    xgb_model_data = sagemaker_session.upload_data(
        path=os.path.join(XGBOOST_DATA_PATH, "xgb_model.tar.gz"),
        key_prefix="integ-test-data/xgboost/model",
    )
    batch_job_name = "test-inference-pipeline-batch-{}".format(sagemaker_timestamp())
    sparkml_model = SparkMLModel(
        model_data=sparkml_model_data,
        env={"SAGEMAKER_SPARKML_SCHEMA": SCHEMA},
        sagemaker_session=sagemaker_session,
    )
    xgb_image = get_image_uri(sagemaker_session.boto_region_name, "xgboost")
    xgb_model = Model(
        model_data=xgb_model_data, image=xgb_image, sagemaker_session=sagemaker_session
    )
    model = PipelineModel(
        models=[sparkml_model, xgb_model],
        role="SageMakerRole",
        sagemaker_session=sagemaker_session,
        name=batch_job_name,
    )
    transformer = model.transformer(1, cpu_instance_type)
    transform_input_key_prefix = "integ-test-data/sparkml_xgboost/transform"
    transform_input = transformer.sagemaker_session.upload_data(
        path=VALID_DATA_PATH, key_prefix=transform_input_key_prefix
    )

    with timeout_and_delete_model_with_transformer(
        transformer, sagemaker_session, minutes=TRANSFORM_DEFAULT_TIMEOUT_MINUTES
    ):
        transformer.transform(
            transform_input, content_type=CONTENT_TYPE_CSV, job_name=batch_job_name
        )
        transformer.wait()


@pytest.mark.canary_quick
@pytest.mark.regional_testing
@pytest.mark.skip(
    reason="This test has always failed, but the failure was masked by a bug. "
    "This test should be fixed. Details in https://github.com/aws/sagemaker-python-sdk/pull/968"
)
def test_inference_pipeline_model_deploy(sagemaker_session, cpu_instance_type):
    sparkml_data_path = os.path.join(DATA_DIR, "sparkml_model")
    xgboost_data_path = os.path.join(DATA_DIR, "xgboost_model")
    endpoint_name = "test-inference-pipeline-deploy-{}".format(sagemaker_timestamp())
    sparkml_model_data = sagemaker_session.upload_data(
        path=os.path.join(sparkml_data_path, "mleap_model.tar.gz"),
        key_prefix="integ-test-data/sparkml/model",
    )
    xgb_model_data = sagemaker_session.upload_data(
        path=os.path.join(xgboost_data_path, "xgb_model.tar.gz"),
        key_prefix="integ-test-data/xgboost/model",
    )

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        sparkml_model = SparkMLModel(
            model_data=sparkml_model_data,
            env={"SAGEMAKER_SPARKML_SCHEMA": SCHEMA},
            sagemaker_session=sagemaker_session,
        )
        xgb_image = get_image_uri(sagemaker_session.boto_region_name, "xgboost")
        xgb_model = Model(
            model_data=xgb_model_data, image=xgb_image, sagemaker_session=sagemaker_session
        )
        model = PipelineModel(
            models=[sparkml_model, xgb_model],
            role="SageMakerRole",
            sagemaker_session=sagemaker_session,
            name=endpoint_name,
        )
        model.deploy(1, cpu_instance_type, endpoint_name=endpoint_name)
        predictor = RealTimePredictor(
            endpoint=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=json_serializer,
            content_type=CONTENT_TYPE_CSV,
            accept=CONTENT_TYPE_CSV,
        )

        with open(VALID_DATA_PATH, "r") as f:
            valid_data = f.read()
            assert predictor.predict(valid_data) == "0.714013934135"

        with open(INVALID_DATA_PATH, "r") as f:
            invalid_data = f.read()
            assert predictor.predict(invalid_data) is None

    model.delete_model()
    with pytest.raises(Exception) as exception:
        sagemaker_session.sagemaker_client.describe_model(ModelName=model.name)
        assert "Could not find model" in str(exception.value)


def test_inference_pipeline_model_deploy_with_update_endpoint(
    sagemaker_session, cpu_instance_type, alternative_cpu_instance_type
):
    sparkml_data_path = os.path.join(DATA_DIR, "sparkml_model")
    xgboost_data_path = os.path.join(DATA_DIR, "xgboost_model")
    endpoint_name = "test-inference-pipeline-deploy-{}".format(sagemaker_timestamp())
    sparkml_model_data = sagemaker_session.upload_data(
        path=os.path.join(sparkml_data_path, "mleap_model.tar.gz"),
        key_prefix="integ-test-data/sparkml/model",
    )
    xgb_model_data = sagemaker_session.upload_data(
        path=os.path.join(xgboost_data_path, "xgb_model.tar.gz"),
        key_prefix="integ-test-data/xgboost/model",
    )

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        sparkml_model = SparkMLModel(
            model_data=sparkml_model_data,
            env={"SAGEMAKER_SPARKML_SCHEMA": SCHEMA},
            sagemaker_session=sagemaker_session,
        )
        xgb_image = get_image_uri(sagemaker_session.boto_region_name, "xgboost")
        xgb_model = Model(
            model_data=xgb_model_data, image=xgb_image, sagemaker_session=sagemaker_session
        )
        model = PipelineModel(
            models=[sparkml_model, xgb_model],
            role="SageMakerRole",
            sagemaker_session=sagemaker_session,
        )
        model.deploy(1, alternative_cpu_instance_type, endpoint_name=endpoint_name)
        old_endpoint = sagemaker_session.sagemaker_client.describe_endpoint(
            EndpointName=endpoint_name
        )
        old_config_name = old_endpoint["EndpointConfigName"]

        model.deploy(1, cpu_instance_type, update_endpoint=True, endpoint_name=endpoint_name)

        # Wait for endpoint to finish updating
        max_retry_count = 40  # Endpoint update takes ~7min. 40 retries * 30s sleeps = 20min timeout
        current_retry_count = 0
        while current_retry_count <= max_retry_count:
            if current_retry_count >= max_retry_count:
                raise Exception("Endpoint status not 'InService' within expected timeout.")
            time.sleep(30)
            new_endpoint = sagemaker_session.sagemaker_client.describe_endpoint(
                EndpointName=endpoint_name
            )
            current_retry_count += 1
            if new_endpoint["EndpointStatus"] == "InService":
                break

        new_config_name = new_endpoint["EndpointConfigName"]
        new_config = sagemaker_session.sagemaker_client.describe_endpoint_config(
            EndpointConfigName=new_config_name
        )

        assert old_config_name != new_config_name
        assert new_config["ProductionVariants"][0]["InstanceType"] == cpu_instance_type
        assert new_config["ProductionVariants"][0]["InitialInstanceCount"] == 1

    model.delete_model()
    with pytest.raises(Exception) as exception:
        sagemaker_session.sagemaker_client.describe_model(ModelName=model.name)
        assert "Could not find model" in str(exception.value)
