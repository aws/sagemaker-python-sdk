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

import os
import time

import pytest

import tests.integ
from sagemaker.tensorflow import TensorFlow, TensorFlowModel
from sagemaker.utils import sagemaker_timestamp, unique_name_from_base
from tests.integ import DATA_DIR, TRAINING_DEFAULT_TIMEOUT_MINUTES, PYTHON_VERSION
from tests.integ.timeout import timeout_and_delete_endpoint_by_name, timeout
from tests.integ.vpc_test_utils import (
    get_or_create_vpc_resources,
    setup_security_group_for_encryption,
)

DATA_PATH = os.path.join(DATA_DIR, "iris", "data")


@pytest.fixture(scope="module")
@pytest.mark.skipif(PYTHON_VERSION != "py2", reason="TensorFlow image supports only python 2.")
def tf_training_job(sagemaker_session, tf_full_version):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, "iris", "iris-dnn-classifier.py")

        estimator = TensorFlow(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version=tf_full_version,
            training_steps=1,
            evaluation_steps=1,
            checkpoint_path="/opt/ml/model",
            hyperparameters={"input_tensor_name": "inputs"},
            train_instance_count=1,
            train_instance_type="ml.c4.xlarge",
            sagemaker_session=sagemaker_session,
            base_job_name="test-tf",
        )

        inputs = sagemaker_session.upload_data(path=DATA_PATH, key_prefix="integ-test-data/tf_iris")
        job_name = unique_name_from_base("test-tf-train")
        estimator.fit(inputs, job_name=job_name)
        print("job succeeded: {}".format(estimator.latest_training_job.name))

        return estimator.latest_training_job.name


@pytest.mark.canary_quick
@pytest.mark.regional_testing
@pytest.mark.skipif(PYTHON_VERSION != "py2", reason="TensorFlow image supports only python 2.")
def test_deploy_model(sagemaker_session, tf_training_job):
    endpoint_name = "test-tf-deploy-model-{}".format(sagemaker_timestamp())

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        desc = sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=tf_training_job
        )
        model_data = desc["ModelArtifacts"]["S3ModelArtifacts"]

        script_path = os.path.join(DATA_DIR, "iris", "iris-dnn-classifier.py")
        model = TensorFlowModel(
            model_data,
            "SageMakerRole",
            entry_point=script_path,
            sagemaker_session=sagemaker_session,
        )

        json_predictor = model.deploy(
            initial_instance_count=1, instance_type="ml.c4.xlarge", endpoint_name=endpoint_name
        )

        features = [6.4, 3.2, 4.5, 1.5]
        dict_result = json_predictor.predict({"inputs": features})
        print("predict result: {}".format(dict_result))
        list_result = json_predictor.predict(features)
        print("predict result: {}".format(list_result))

        assert dict_result == list_result


@pytest.mark.canary_quick
@pytest.mark.regional_testing
@pytest.mark.skipif(
    tests.integ.test_region() not in tests.integ.EI_SUPPORTED_REGIONS,
    reason="EI isn't supported in that specific region.",
)
@pytest.mark.skipif(PYTHON_VERSION != "py2", reason="TensorFlow image supports only python 2.")
def test_deploy_model_with_accelerator(sagemaker_session, tf_training_job, ei_tf_full_version):
    endpoint_name = "test-tf-deploy-model-ei-{}".format(sagemaker_timestamp())

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        desc = sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=tf_training_job
        )
        model_data = desc["ModelArtifacts"]["S3ModelArtifacts"]

        script_path = os.path.join(DATA_DIR, "iris", "iris-dnn-classifier.py")
        model = TensorFlowModel(
            model_data,
            "SageMakerRole",
            entry_point=script_path,
            framework_version=ei_tf_full_version,
            sagemaker_session=sagemaker_session,
        )

        json_predictor = model.deploy(
            initial_instance_count=1,
            instance_type="ml.c4.xlarge",
            endpoint_name=endpoint_name,
            accelerator_type="ml.eia1.medium",
        )

        features = [6.4, 3.2, 4.5, 1.5]
        dict_result = json_predictor.predict({"inputs": features})
        print("predict result: {}".format(dict_result))
        list_result = json_predictor.predict(features)
        print("predict result: {}".format(list_result))

        assert dict_result == list_result


@pytest.mark.skipif(PYTHON_VERSION != "py2", reason="TensorFlow image supports only python 2.")
def test_tf_async(sagemaker_session):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, "iris", "iris-dnn-classifier.py")

        estimator = TensorFlow(
            entry_point=script_path,
            role="SageMakerRole",
            training_steps=1,
            evaluation_steps=1,
            checkpoint_path="/opt/ml/model",
            hyperparameters={"input_tensor_name": "inputs"},
            train_instance_count=1,
            train_instance_type="ml.c4.xlarge",
            sagemaker_session=sagemaker_session,
            base_job_name="test-tf",
        )

        inputs = estimator.sagemaker_session.upload_data(
            path=DATA_PATH, key_prefix="integ-test-data/tf_iris"
        )
        job_name = unique_name_from_base("test-tf-async")
        estimator.fit(inputs, wait=False, job_name=job_name)
        training_job_name = estimator.latest_training_job.name
        time.sleep(20)

    endpoint_name = training_job_name
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        estimator = TensorFlow.attach(
            training_job_name=training_job_name, sagemaker_session=sagemaker_session
        )
        json_predictor = estimator.deploy(
            initial_instance_count=1, instance_type="ml.c4.xlarge", endpoint_name=endpoint_name
        )

        result = json_predictor.predict([6.4, 3.2, 4.5, 1.5])
        print("predict result: {}".format(result))


@pytest.mark.skipif(PYTHON_VERSION != "py2", reason="TensorFlow image supports only python 2.")
def test_tf_vpc_multi(sagemaker_session, tf_full_version):
    """Test Tensorflow multi-instance using the same VpcConfig for training and inference"""
    instance_type = "ml.c4.xlarge"
    instance_count = 2

    train_input = sagemaker_session.upload_data(
        path=os.path.join(DATA_DIR, "iris", "data"), key_prefix="integ-test-data/tf_iris"
    )
    script_path = os.path.join(DATA_DIR, "iris", "iris-dnn-classifier.py")

    ec2_client = sagemaker_session.boto_session.client("ec2")
    subnet_ids, security_group_id = get_or_create_vpc_resources(
        ec2_client, sagemaker_session.boto_session.region_name
    )

    setup_security_group_for_encryption(ec2_client, security_group_id)

    estimator = TensorFlow(
        entry_point=script_path,
        role="SageMakerRole",
        framework_version=tf_full_version,
        training_steps=1,
        evaluation_steps=1,
        hyperparameters={"input_tensor_name": "inputs"},
        train_instance_count=instance_count,
        train_instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        base_job_name="test-vpc-tf",
        subnets=subnet_ids,
        security_group_ids=[security_group_id],
        encrypt_inter_container_traffic=True,
    )
    job_name = unique_name_from_base("test-tf-vpc-multi")

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        estimator.fit(train_input, job_name=job_name)
        print("training job succeeded: {}".format(estimator.latest_training_job.name))

    job_desc = sagemaker_session.sagemaker_client.describe_training_job(
        TrainingJobName=estimator.latest_training_job.name
    )
    assert set(subnet_ids) == set(job_desc["VpcConfig"]["Subnets"])
    assert [security_group_id] == job_desc["VpcConfig"]["SecurityGroupIds"]
    assert job_desc["EnableInterContainerTrafficEncryption"] is True

    endpoint_name = estimator.latest_training_job.name
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        model = estimator.create_model()
        json_predictor = model.deploy(
            initial_instance_count=instance_count,
            instance_type="ml.c4.xlarge",
            endpoint_name=endpoint_name,
        )

        features = [6.4, 3.2, 4.5, 1.5]
        dict_result = json_predictor.predict({"inputs": features})
        print("predict result: {}".format(dict_result))
        list_result = json_predictor.predict(features)
        print("predict result: {}".format(list_result))

        assert dict_result == list_result

    model_desc = sagemaker_session.sagemaker_client.describe_model(ModelName=model.name)
    assert set(subnet_ids) == set(model_desc["VpcConfig"]["Subnets"])
    assert [security_group_id] == model_desc["VpcConfig"]["SecurityGroupIds"]


@pytest.mark.skipif(PYTHON_VERSION != "py2", reason="TensorFlow image supports only python 2.")
def test_failed_tf_training(sagemaker_session, tf_full_version):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, "iris", "failure_script.py")
        estimator = TensorFlow(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version=tf_full_version,
            training_steps=1,
            evaluation_steps=1,
            hyperparameters={"input_tensor_name": "inputs"},
            train_instance_count=1,
            train_instance_type="ml.c4.xlarge",
            sagemaker_session=sagemaker_session,
        )
        job_name = unique_name_from_base("test-tf-fail")

        with pytest.raises(ValueError) as e:
            estimator.fit(job_name=job_name)
        assert "This failure is expected" in str(e.value)
