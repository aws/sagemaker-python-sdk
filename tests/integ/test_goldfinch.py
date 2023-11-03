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

# import os
# import sagemaker
# import sagemaker.predictor
# import sagemaker.utils

# import tests.integ
# import tests.integ.timeout
import pytest

# import boto3
from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements

# from sagemaker.predictor import Predictor
# from sagemaker.enums import EndpointType
# from sagemaker.tensorflow.model import TensorFlowModel
# from sagemaker.model import Model
# from sagemaker.session import Session


@pytest.fixture(scope="module")
def resources():
    resources = ResourceRequirements(
        requests={
            "num_cpus": 1,  # NumberOfCpuCoresRequired
            "memory": 1024,  # MinMemoryRequiredInMb (required), differentiator for Goldfinch path
            "copies": 1,
        },
        limits={},
    )
    return resources


# @pytest.fixture(scope="module")
# def tfs_predictor(sagemaker_session, tensorflow_inference_latest_version, resources):
#     sagemaker_session = session_with_gamma_endpoint_override()
#     endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-tensorflow-serving")
#     model_data = sagemaker_session.upload_data(
#         path=os.path.join(tests.integ.DATA_DIR, "tensorflow-serving-test-model.tar.gz"),
#         key_prefix="tensorflow-serving/models",
#     )
#     model = TensorFlowModel(
#         model_data=model_data,
#         role="SageMakerRole",
#         framework_version=tensorflow_inference_latest_version,
#         sagemaker_session=sagemaker_session,
#         resources=resources,
#     )
#     predictor = model.deploy(
#         1,
#         "ml.m5.large",
#         endpoint_name=endpoint_name,
#         endpoint_type="goldfinch",
#         resources=resources,
#     )
#     yield predictor


# @pytest.mark.release
# def test_deploy_single_model_with_endpoint_name(tfs_predictor):
#     input_data = {"instances": [1.0, 2.0, 5.0]}
#     expected_result = {"predictions": [3.5, 4.0, 5.5]}

#     result = tfs_predictor.predict(input_data)
#     assert expected_result == result

#     models = tfs_predictor.list_colocated_models()
#     print(models)

#     print("endpoint_name is:", tfs_predictor.endpoint)

#     # update endpoint(instance scaling)
#     tfs_predictor.update_endpoint(max_instance_count=5)

#     # test
#     # tfs_predictor.update_predictor()

#     # delete predictor
#     tfs_predictor.delete_predictor()

#     # delete model, to fix
#     # tfs_predictor.delete_model()

#     # delete endpoint
#     tfs_predictor.delete_endpoint()


"""
    # [TODO]: save below commented integ test and need to come back add revisements
@pytest.mark.release(scope="module")
def test_quick_test():
    predictor = sagemaker.Predictor(
        endpoint_name="sagemaker-tensorflow-serving-1698553338-b847",
        sagemaker_session=session_with_gamma_endpoint_override(),
        serializer=IdentitySerializer(content_type="application/json"),
        deserializer=JSONDeserializer(),
        component_name="tensorflow-inference-2023-10-30-23-18-24-614-1698707906-920c",
    )

    print("endpoint_name is:",predictor.endpoint)

    predictor.update_endpoint(max_instance_count=5)

    #tfs_predictor.update_predictor()

    # delete predictor
    #tfs_predictor.delete_predictor()

    # delete model, to fix
    # tfs_predictor.delete_model()

    # delete endpoint
    # tfs_predictor.delete_endpoint()


def test_deploy_single_model_without_endpoint_name(tensorflow_inference_latest_version, resources):
    sagemaker_session = session_with_gamma_endpoint_override()

    model_data = sagemaker_session.upload_data(
        path=os.path.join(tests.integ.DATA_DIR, "tensorflow-serving-test-model.tar.gz"),
        key_prefix="tensorflow-serving/models",
    )

    model = TensorFlowModel(
        model_data=model_data,
        role="Admin",
        framework_version=tensorflow_inference_latest_version,
        sagemaker_session=sagemaker_session,
        resources=resources,
    )

    predictor_1 = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_type="goldfinch",
        resources=resources,
    )

    payload = bytes("hello", "utf-8")
    results = predictor_1.predict(payload)
    print(len(results))

    models = predictor_1.list_colocated_models()
    print(models)

    # delete model1
    predictor_1.delete_predictor()

    # delete endpoint
    predictor_1.delete_endpoint()


def test_deploy_multiple_model_to_one_endpoint(tensorflow_inference_latest_version, resources):
    sagemaker_session = session_with_gamma_endpoint_override()

    model_data = sagemaker_session.upload_data(
        path=os.path.join(tests.integ.DATA_DIR, "tensorflow-serving-test-model.tar.gz"),
        key_prefix="tensorflow-serving/models",
    )
    # with tests.integ.timeout.timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
    model1 = TensorFlowModel(
        model_data=model_data,
        role="Admin",
        framework_version=tensorflow_inference_latest_version,
        sagemaker_session=sagemaker_session,
        resources=resources,
    )

    model2 = TensorFlowModel(
        model_data=model_data,
        role="Admin",
        framework_version=tensorflow_inference_latest_version,
        sagemaker_session=sagemaker_session,
        resources=resources,
    )

    # deploy model1 and deploy model2 to same endpoint
    predictor_1 = model1.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_type="goldfinch",
        resources=resources,
    )

    predictor_2 = model2.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_type="goldfinch",
        endpoint_name=predictor_1.endpoint_name,
        resources=resources,
    )

    # predict
    payload = bytes("hello", "utf-8")
    results1 = predictor_1.predict(payload)
    results2 = predictor_2.predict(payload)
    print(len(results1))
    print(len(results2))

    models1 = predictor_1.list_colocated_models()
    models2 = predictor_2.list_colocated_models()
    print("list models from predictor1")
    print(models1)
    print("list models from predictor2")
    print(models2)

    # delete model1
    predictor_1.delete_predictor()

    # delete endpoint
    predictor_1.delete_endpoint()
"""


# def test_null_transformer(resources):
#     sagemaker_session = session_with_gamma_endpoint_override()
#
#     model = Model(
#         name="bhaoz-test-goldfinch-pysdk-1101-1538",
#         image_uri="971857754115.dkr.ecr.us-west-2.amazonaws.com/maevenulltransformerimage:latest",
#         role="arn:aws:iam::089965926474:role/SageMakerRole",
#         sagemaker_session=sagemaker_session,
#         resources=resources,
#         predictor_cls=Predictor,
#     )
#
#     print("Start deploying model")
#     predictor = model.deploy(
#         initial_instance_count=1,
#         instance_type="ml.m5.large",
#         endpoint_type=EndpointType.GOLDFINCH,
#         resources=resources,
#     )
#
#     print("Model deployed successfully\n")
#     payload = bytes("hello", "utf-8")
#     result = predictor.predict(payload)
#     print(result)
#     models = predictor.list_colocated_models()
#     print(models)
#
#
# def session_with_gamma_endpoint_override() -> Session:
#     boto_session = boto3.Session(region_name="us-west-2")
#     sagemaker = boto3.client(
#         service_name="sagemaker",
#         endpoint_url="https://sagemaker.beta.us-west-2.ml-platform.aws.a2z.com",
#     )
#     sagemaker_runtime = boto3.client(
#         service_name="sagemaker-runtime",
#         endpoint_url="https://maeveruntime.beta.us-west-2.ml-platform.aws.a2z.com",
#     )
#     return Session(
#         boto_session=boto_session,
#         sagemaker_client=sagemaker,
#         sagemaker_runtime_client=sagemaker_runtime,
#     )
