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

import tarfile

import botocore.exceptions
import os

import pytest
import sagemaker
import sagemaker.predictor
import sagemaker.utils
import tests.integ
import tests.integ.timeout
from sagemaker.deserializers import JSONDeserializer
from sagemaker.tensorflow.model import TensorFlowModel, TensorFlowPredictor
from sagemaker.serializers import CSVSerializer, IdentitySerializer


@pytest.fixture(scope="module")
def tfs_predictor(sagemaker_session, tensorflow_inference_latest_version):
    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-tensorflow-serving")
    model_data = sagemaker_session.upload_data(
        path=os.path.join(tests.integ.DATA_DIR, "tensorflow-serving-test-model.tar.gz"),
        key_prefix="tensorflow-serving/models",
    )
    with tests.integ.timeout.timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        model = TensorFlowModel(
            model_data=model_data,
            role="SageMakerRole",
            framework_version=tensorflow_inference_latest_version,
            sagemaker_session=sagemaker_session,
            name=endpoint_name,
        )
        predictor = model.deploy(1, "ml.c5.xlarge", endpoint_name=endpoint_name)
        yield predictor


def tar_dir(directory, tmpdir):
    target = os.path.join(str(tmpdir), "model.tar.gz")

    with tarfile.open(target, mode="w:gz") as t:
        t.add(directory, arcname=os.path.sep)
    return target


@pytest.fixture
def tfs_predictor_with_model_and_entry_point_same_tar(
    sagemaker_local_session, tensorflow_inference_latest_version, tmpdir
):
    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-tensorflow-serving")

    model_tar = tar_dir(
        os.path.join(tests.integ.DATA_DIR, "tfs/tfs-test-model-with-inference"), tmpdir
    )

    model = TensorFlowModel(
        model_data="file://" + model_tar,
        role="SageMakerRole",
        framework_version=tensorflow_inference_latest_version,
        sagemaker_session=sagemaker_local_session,
        name=endpoint_name,
    )
    predictor = model.deploy(1, "local", endpoint_name=endpoint_name)

    try:
        yield predictor
    finally:
        predictor.delete_endpoint()


@pytest.fixture(scope="module")
def tfs_predictor_with_model_and_entry_point_and_dependencies(
    sagemaker_local_session, tensorflow_inference_latest_version
):
    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-tensorflow-serving")

    entry_point = os.path.join(
        tests.integ.DATA_DIR, "tfs/tfs-test-entrypoint-and-dependencies/inference.py"
    )
    dependencies = [
        os.path.join(
            tests.integ.DATA_DIR,
            "tfs/tfs-test-entrypoint-and-dependencies/dependency.py",
        )
    ]

    model_data = "file://" + os.path.join(
        tests.integ.DATA_DIR, "tensorflow-serving-test-model.tar.gz"
    )

    model = TensorFlowModel(
        entry_point=entry_point,
        model_data=model_data,
        role="SageMakerRole",
        dependencies=dependencies,
        framework_version=tensorflow_inference_latest_version,
        sagemaker_session=sagemaker_local_session,
        name=endpoint_name,
    )

    predictor = model.deploy(1, "local", endpoint_name=endpoint_name)
    try:

        yield predictor
    finally:
        predictor.delete_endpoint()


@pytest.fixture(scope="module")
def tfs_predictor_with_accelerator(
    sagemaker_session, tensorflow_eia_latest_version, cpu_instance_type
):
    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-tensorflow-serving")
    model_data = sagemaker_session.upload_data(
        path=os.path.join(tests.integ.DATA_DIR, "tensorflow-serving-test-model.tar.gz"),
        key_prefix="tensorflow-serving/models",
    )
    with tests.integ.timeout.timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        model = TensorFlowModel(
            model_data=model_data,
            role="SageMakerRole",
            framework_version=tensorflow_eia_latest_version,
            sagemaker_session=sagemaker_session,
            name=endpoint_name,
        )
        predictor = model.deploy(
            1,
            cpu_instance_type,
            endpoint_name=endpoint_name,
            accelerator_type="ml.eia1.medium",
        )
        yield predictor


@pytest.mark.release
def test_predict(tfs_predictor):
    input_data = {"instances": [1.0, 2.0, 5.0]}
    expected_result = {"predictions": [3.5, 4.0, 5.5]}

    result = tfs_predictor.predict(input_data)
    assert expected_result == result


@pytest.mark.local_mode
def test_predict_with_entry_point(tfs_predictor_with_model_and_entry_point_same_tar):
    input_data = {"instances": [1.0, 2.0, 5.0]}
    expected_result = {"predictions": [4.0, 4.5, 6.0]}

    result = tfs_predictor_with_model_and_entry_point_same_tar.predict(input_data)
    assert expected_result == result


@pytest.mark.local_mode
def test_predict_with_model_and_entry_point_and_dependencies_separated(
    tfs_predictor_with_model_and_entry_point_and_dependencies,
):
    input_data = {"instances": [1.0, 2.0, 5.0]}
    expected_result = {"predictions": [4.0, 4.5, 6.0]}

    result = tfs_predictor_with_model_and_entry_point_and_dependencies.predict(input_data)
    assert expected_result == result


def test_predict_generic_json(tfs_predictor):
    input_data = [[1.0, 2.0, 5.0], [1.0, 2.0, 5.0]]
    expected_result = {"predictions": [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}

    result = tfs_predictor.predict(input_data)
    assert expected_result == result


def test_predict_jsons_json_content_type(tfs_predictor):
    input_data = "[1.0, 2.0, 5.0]\n[1.0, 2.0, 5.0]"
    expected_result = {"predictions": [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}

    predictor = sagemaker.Predictor(
        tfs_predictor.endpoint_name,
        tfs_predictor.sagemaker_session,
        serializer=IdentitySerializer(content_type="application/json"),
        deserializer=JSONDeserializer(),
    )

    result = predictor.predict(input_data)
    assert expected_result == result


def test_predict_jsons(tfs_predictor):
    input_data = "[1.0, 2.0, 5.0]\n[1.0, 2.0, 5.0]"
    expected_result = {"predictions": [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}

    predictor = sagemaker.Predictor(
        tfs_predictor.endpoint_name,
        tfs_predictor.sagemaker_session,
        serializer=IdentitySerializer(content_type="application/json"),
        deserializer=JSONDeserializer(),
    )

    result = predictor.predict(input_data)
    assert expected_result == result


def test_predict_jsonlines(tfs_predictor):
    input_data = "[1.0, 2.0, 5.0]\n[1.0, 2.0, 5.0]"
    expected_result = {"predictions": [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}

    predictor = sagemaker.Predictor(
        tfs_predictor.endpoint_name,
        tfs_predictor.sagemaker_session,
        serializer=IdentitySerializer(content_type="application/jsonlines"),
        deserializer=JSONDeserializer(),
    )

    result = predictor.predict(input_data)
    assert expected_result == result


def test_predict_csv(tfs_predictor):
    input_data = "1.0,2.0,5.0\n1.0,2.0,5.0"
    expected_result = {"predictions": [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}

    predictor = TensorFlowPredictor(
        tfs_predictor.endpoint_name,
        tfs_predictor.sagemaker_session,
        serializer=CSVSerializer(),
    )

    result = predictor.predict(input_data)
    assert expected_result == result


def test_predict_bad_input(tfs_predictor):
    input_data = {"junk": "data"}
    with pytest.raises(botocore.exceptions.ClientError):
        tfs_predictor.predict(input_data)
