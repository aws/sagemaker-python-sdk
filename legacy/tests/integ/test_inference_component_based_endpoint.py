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
import sagemaker.predictor
import sagemaker.utils
import pytest

from sagemaker import image_uris

from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements
from sagemaker.enums import EndpointType
from sagemaker.model import Model
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer

from sagemaker.predictor import Predictor
from sagemaker.tensorflow.model import TensorFlowModel

from tests.integ import DATA_DIR

ROLE = "SageMakerRole"
XGBOOST_DATA_PATH = os.path.join(DATA_DIR, "xgboost_model")
TEST_CSV_DATA = "42,42,42,42,42,42,42"


@pytest.fixture(scope="module")
def model_name():
    return sagemaker.utils.unique_name_from_base("tensorflow-integ-test-model")


@pytest.fixture(scope="module")
def model_update_to_name():
    return sagemaker.utils.unique_name_from_base("xgboost-integ-test-model")


@pytest.fixture(scope="module")
def resources():
    resources = ResourceRequirements(
        requests={
            "num_cpus": 0.5,  # NumberOfCpuCoresRequired
            "memory": 512,  # MinMemoryRequiredInMb (required)
            "copies": 1,
        },
        limits={},
    )
    return resources


@pytest.fixture(scope="module")
def resources_update():
    resources = ResourceRequirements(
        requests={
            "num_cpus": 0.5,  # NumberOfCpuCoresRequired
            "memory": 512,  # MinMemoryRequiredInMb (required)
            "copies": 2,
        },
        limits={},
    )
    return resources


@pytest.fixture(scope="module")
def tfs_model(sagemaker_session, tensorflow_inference_latest_version, resources, model_name):
    model_data = sagemaker_session.upload_data(
        path=os.path.join(DATA_DIR, "tensorflow-serving-test-model.tar.gz"),
        key_prefix="tensorflow-serving/models",
    )
    model = TensorFlowModel(
        model_data=model_data,
        role=ROLE,
        framework_version=tensorflow_inference_latest_version,
        sagemaker_session=sagemaker_session,
        resources=resources,
        name=model_name,
    )
    return model


@pytest.fixture(scope="module")
def xgboost_model(sagemaker_session, resources, model_update_to_name):
    model_data = sagemaker_session.upload_data(
        path=os.path.join(XGBOOST_DATA_PATH, "xgb_model.tar.gz"),
        key_prefix="integ-test-data/xgboost/model",
    )

    xgb_image = image_uris.retrieve(
        "xgboost",
        sagemaker_session.boto_region_name,
        version="1",
        image_scope="inference",
    )

    xgb_model = Model(
        model_data=model_data,
        image_uri=xgb_image,
        name=model_update_to_name,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        predictor_cls=Predictor,
    )
    return xgb_model


@pytest.mark.skip(
    reason="This test is skipped temporarily due to failures. Need to re-enable later after fix."
)
def test_deploy_single_model_with_endpoint_name(tfs_model, resources):
    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-tensorflow-serving")
    predictor = tfs_model.deploy(
        1,
        "ml.m5.large",
        endpoint_name=endpoint_name,
        endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
        resources=resources,
    )

    input_data = {"instances": [1.0, 2.0, 5.0]}
    expected_result = {"predictions": [3.5, 4.0, 5.5]}

    result = predictor.predict(input_data)
    assert expected_result == result

    models, _ = predictor.list_related_models()
    assert models and len(models) > 0 and len(models) == 1

    # delete predictor
    predictor.delete_predictor(wait=True)

    # delete endpoint
    predictor.delete_endpoint()


@pytest.mark.skip(
    reason="This test is skipped temporarily due to failures. Need to re-enable later after fix."
)
def test_deploy_update_predictor_with_other_model(
    tfs_model,
    resources,
    resources_update,
    xgboost_model,
):
    endpoint_name = sagemaker.utils.unique_name_from_base("multi-different-model-endpoint")
    predictor_to_update = tfs_model.deploy(
        1,
        "ml.m5.4xlarge",
        endpoint_name=endpoint_name,
        endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
        resources=resources,
    )

    input_data = {"instances": [1.0, 2.0, 5.0]}
    expected_result = {"predictions": [3.5, 4.0, 5.5]}

    result = predictor_to_update.predict(input_data)
    assert expected_result == result

    models, _ = predictor_to_update.list_related_models()
    assert models and len(models) > 0 and len(models) == 1

    xgboost_predictor = xgboost_model.deploy(
        1,
        "ml.m5.4xlarge",
        endpoint_name=endpoint_name,
        endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
        resources=resources,
    )
    xgboost_predictor.serializer = CSVSerializer()
    xgboost_predictor.deserializer = CSVDeserializer()
    xgb_result = xgboost_predictor.predict(TEST_CSV_DATA)
    assert xgb_result

    # update Tensorflow predictor to Xgboost predictor
    predictor_to_update.update_predictor(
        model_name=xgboost_model.name,
        resources=resources_update,
    )

    # predictor_to_update is now a Xgboost predictor
    predictor_to_update.serializer = CSVSerializer()
    predictor_to_update.deserializer = CSVDeserializer()
    xgb_result_2 = predictor_to_update.predict(TEST_CSV_DATA)
    assert xgb_result == xgb_result_2

    # delete predictor
    predictor_to_update.delete_predictor(wait=True)
    xgboost_predictor.delete_predictor(wait=True)

    # delete endpoint
    predictor_to_update.delete_endpoint()


@pytest.mark.skip(
    reason="This test is skipped temporarily due to failures. Need to re-enable later after fix."
)
def test_deploy_multi_models_without_endpoint_name(tfs_model, resources):
    input_data = {"instances": [1.0, 2.0, 5.0]}
    expected_result = {"predictions": [3.5, 4.0, 5.5]}

    tfs_predictor1 = tfs_model.deploy(
        1,
        "ml.m5.large",
        endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
        resources=resources,
    )

    result = tfs_predictor1.predict(input_data)
    assert expected_result == result

    endpoint_name = tfs_predictor1.endpoint

    tfs_predictor2 = tfs_model.deploy(
        1,
        "ml.m5.large",
        endpoint_name=endpoint_name,
        endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
        resources=resources,
    )

    result_2 = tfs_predictor2.predict(input_data)
    assert expected_result == result_2

    models1, _ = tfs_predictor1.list_related_models()
    assert models1 and len(models1) > 0 and len(models1) == 2

    models2, _ = tfs_predictor2.list_related_models()
    assert models2 and len(models2) > 0 and len(models2) == 2

    # update endpoint(for instance scaling)
    tfs_predictor1.update_endpoint(max_instance_count=5)

    result = tfs_predictor1.predict(input_data)
    assert expected_result == result

    result_2 = tfs_predictor2.predict(input_data)
    assert expected_result == result_2

    # delete predictors
    tfs_predictor1.delete_predictor(wait=True)
    tfs_predictor2.delete_predictor(wait=True)

    # delete endpoint
    tfs_predictor1.delete_endpoint()
