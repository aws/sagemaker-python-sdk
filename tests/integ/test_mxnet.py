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
import time

import numpy
import pytest

from sagemaker import ModelPackage
from sagemaker.mxnet.estimator import MXNet
from sagemaker.mxnet.model import MXNetModel
from sagemaker.mxnet.processing import MXNetProcessor
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.utils import unique_name_from_base
from tests.integ import DATA_DIR, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.kms_utils import get_or_create_kms_key
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


@pytest.fixture(scope="module")
def mxnet_training_job(
    sagemaker_session,
    mxnet_training_latest_version,
    mxnet_training_latest_py_version,
    cpu_instance_type,
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        s3_prefix = "integ-test-data/mxnet_mnist"
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        s3_source = sagemaker_session.upload_data(
            path=os.path.join(data_path, "sourcedir.tar.gz"), key_prefix="{}/src".format(s3_prefix)
        )

        mx = MXNet(
            entry_point="mxnet_mnist/mnist.py",
            source_dir=s3_source,
            role="SageMakerRole",
            framework_version=mxnet_training_latest_version,
            py_version=mxnet_training_latest_py_version,
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
        )

        train_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="{}/train".format(s3_prefix)
        )
        test_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="{}/test".format(s3_prefix)
        )

        mx.fit({"train": train_input, "test": test_input})
        return mx.latest_training_job.name


@pytest.mark.release
def test_framework_processing_job_with_deps(
    sagemaker_session,
    mxnet_training_latest_version,
    mxnet_training_latest_py_version,
    cpu_instance_type,
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        code_path = os.path.join(DATA_DIR, "dummy_code_bundle_with_reqs")
        entry_point = "main_script.py"

        processor = MXNetProcessor(
            framework_version=mxnet_training_latest_version,
            py_version=mxnet_training_latest_py_version,
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            base_job_name="test-mxnet",
        )

        processor.run(
            code=entry_point,
            source_dir=code_path,
            inputs=[],
            wait=True,
        )


@pytest.mark.release
def test_attach_deploy(mxnet_training_job, sagemaker_session, cpu_instance_type):
    endpoint_name = unique_name_from_base("test-mxnet-attach-deploy")

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        estimator = MXNet.attach(mxnet_training_job, sagemaker_session=sagemaker_session)
        predictor = estimator.deploy(
            1,
            cpu_instance_type,
            entry_point="mnist.py",
            source_dir=os.path.join(DATA_DIR, "mxnet_mnist"),
            endpoint_name=endpoint_name,
        )
        data = numpy.zeros(shape=(1, 1, 28, 28))
        result = predictor.predict(data)
        assert result is not None


@pytest.mark.slow_test
def test_deploy_estimator_with_different_instance_types(
    mxnet_training_job,
    sagemaker_session,
    cpu_instance_type,
    alternative_cpu_instance_type,
):
    def _deploy_estimator_and_assert_instance_type(estimator, instance_type):
        # don't use timeout_and_delete_endpoint_by_name because this tests if
        # deploy() creates a new endpoint config/endpoint each time
        with timeout(minutes=45):
            try:
                predictor = estimator.deploy(1, instance_type)

                model_name = predictor._get_model_names()[0]
                config_name = sagemaker_session.sagemaker_client.describe_endpoint(
                    EndpointName=predictor.endpoint_name
                )["EndpointConfigName"]
                config = sagemaker_session.sagemaker_client.describe_endpoint_config(
                    EndpointConfigName=config_name
                )
            finally:
                predictor.delete_model()
                predictor.delete_endpoint()

        assert config["ProductionVariants"][0]["InstanceType"] == instance_type

        return (model_name, predictor.endpoint_name, config_name)

    estimator = MXNet.attach(mxnet_training_job, sagemaker_session)
    estimator.base_job_name = "test-mxnet-deploy-twice"

    old_model_name, old_endpoint_name, old_config_name = _deploy_estimator_and_assert_instance_type(
        estimator, cpu_instance_type
    )
    new_model_name, new_endpoint_name, new_config_name = _deploy_estimator_and_assert_instance_type(
        estimator, alternative_cpu_instance_type
    )

    assert old_model_name != new_model_name
    assert old_endpoint_name != new_endpoint_name
    assert old_config_name != new_config_name


def test_deploy_model(
    mxnet_training_job,
    sagemaker_session,
    mxnet_inference_latest_version,
    mxnet_inference_latest_py_version,
    cpu_instance_type,
):
    endpoint_name = unique_name_from_base("test-mxnet-deploy-model")

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        desc = sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=mxnet_training_job
        )
        model_data = desc["ModelArtifacts"]["S3ModelArtifacts"]
        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist.py")
        model = MXNetModel(
            model_data,
            "SageMakerRole",
            entry_point=script_path,
            py_version=mxnet_inference_latest_py_version,
            sagemaker_session=sagemaker_session,
            framework_version=mxnet_inference_latest_version,
        )
        predictor = model.deploy(1, cpu_instance_type, endpoint_name=endpoint_name)

        data = numpy.zeros(shape=(1, 1, 28, 28))
        result = predictor.predict(data)
        assert result is not None

    model.delete_model()
    with pytest.raises(Exception) as exception:
        sagemaker_session.sagemaker_client.describe_model(ModelName=model.name)
        assert "Could not find model" in str(exception.value)


def test_register_model_package(
    mxnet_training_job,
    sagemaker_session,
    mxnet_inference_latest_version,
    mxnet_inference_latest_py_version,
    cpu_instance_type,
):
    endpoint_name = unique_name_from_base("test-mxnet-deploy-model")

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        desc = sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=mxnet_training_job
        )
        model_data = desc["ModelArtifacts"]["S3ModelArtifacts"]
        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist.py")
        model = MXNetModel(
            model_data,
            "SageMakerRole",
            entry_point=script_path,
            py_version=mxnet_inference_latest_py_version,
            sagemaker_session=sagemaker_session,
            framework_version=mxnet_inference_latest_version,
        )
        model_package_name = unique_name_from_base("register-model-package")
        model_pkg = model.register(
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large"],
            transform_instances=["ml.m5.large"],
            model_package_name=model_package_name,
        )
        assert isinstance(model_pkg, ModelPackage)
        predictor = model.deploy(1, cpu_instance_type, endpoint_name=endpoint_name)
        data = numpy.zeros(shape=(1, 1, 28, 28))
        result = predictor.predict(data)
        assert result is not None
        sagemaker_session.sagemaker_client.delete_model_package(ModelPackageName=model_package_name)


def test_register_model_package_versioned(
    mxnet_training_job,
    sagemaker_session,
    mxnet_inference_latest_version,
    mxnet_inference_latest_py_version,
    cpu_instance_type,
):
    endpoint_name = unique_name_from_base("test-mxnet-deploy-model")

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        desc = sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=mxnet_training_job
        )
        model_package_group_name = unique_name_from_base("register-model-package")
        sagemaker_session.sagemaker_client.create_model_package_group(
            ModelPackageGroupName=model_package_group_name
        )
        model_data = desc["ModelArtifacts"]["S3ModelArtifacts"]
        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist.py")
        model = MXNetModel(
            model_data,
            "SageMakerRole",
            entry_point=script_path,
            py_version=mxnet_inference_latest_py_version,
            sagemaker_session=sagemaker_session,
            framework_version=mxnet_inference_latest_version,
        )
        model_pkg = model.register(
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large"],
            transform_instances=["ml.m5.large"],
            model_package_group_name=model_package_group_name,
            approval_status="Approved",
        )
        assert isinstance(model_pkg, ModelPackage)
        predictor = model.deploy(1, cpu_instance_type, endpoint_name=endpoint_name)
        data = numpy.zeros(shape=(1, 1, 28, 28))
        result = predictor.predict(data)
        assert result is not None
        sagemaker_session.sagemaker_client.delete_model_package(
            ModelPackageName=model_pkg.model_package_arn
        )
        sagemaker_session.sagemaker_client.delete_model_package_group(
            ModelPackageGroupName=model_package_group_name
        )


def test_deploy_model_with_tags_and_kms(
    mxnet_training_job,
    sagemaker_session,
    mxnet_inference_latest_version,
    mxnet_inference_latest_py_version,
    cpu_instance_type,
):
    endpoint_name = unique_name_from_base("test-mxnet-deploy-model")

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        desc = sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=mxnet_training_job
        )
        model_data = desc["ModelArtifacts"]["S3ModelArtifacts"]
        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist.py")
        model = MXNetModel(
            model_data,
            "SageMakerRole",
            entry_point=script_path,
            py_version=mxnet_inference_latest_py_version,
            sagemaker_session=sagemaker_session,
            framework_version=mxnet_inference_latest_version,
        )

        tags = [{"Key": "TagtestKey", "Value": "TagtestValue"}]
        kms_key_arn = get_or_create_kms_key(sagemaker_session)

        model.deploy(
            1, cpu_instance_type, endpoint_name=endpoint_name, tags=tags, kms_key=kms_key_arn
        )

        returned_model = sagemaker_session.sagemaker_client.describe_model(ModelName=model.name)
        returned_model_tags = sagemaker_session.sagemaker_client.list_tags(
            ResourceArn=returned_model["ModelArn"]
        )["Tags"]

        endpoint = sagemaker_session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_tags = sagemaker_session.sagemaker_client.list_tags(
            ResourceArn=endpoint["EndpointArn"]
        )["Tags"]

        endpoint_config = sagemaker_session.sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint["EndpointConfigName"]
        )
        endpoint_config_tags = sagemaker_session.sagemaker_client.list_tags(
            ResourceArn=endpoint_config["EndpointConfigArn"]
        )["Tags"]

        production_variants = endpoint_config["ProductionVariants"]

        assert returned_model_tags == tags
        assert endpoint_config_tags == tags
        assert endpoint_tags == tags
        assert production_variants[0]["InstanceType"] == cpu_instance_type
        assert production_variants[0]["InitialInstanceCount"] == 1
        assert endpoint_config["KmsKeyId"] == kms_key_arn


@pytest.mark.slow_test
def test_deploy_model_and_update_endpoint(
    mxnet_training_job,
    sagemaker_session,
    mxnet_inference_latest_version,
    mxnet_inference_latest_py_version,
    cpu_instance_type,
    alternative_cpu_instance_type,
):
    endpoint_name = unique_name_from_base("test-mxnet-deploy-model")

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        desc = sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=mxnet_training_job
        )
        model_data = desc["ModelArtifacts"]["S3ModelArtifacts"]
        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist.py")
        model = MXNetModel(
            model_data,
            "SageMakerRole",
            entry_point=script_path,
            py_version=mxnet_inference_latest_py_version,
            sagemaker_session=sagemaker_session,
            framework_version=mxnet_inference_latest_version,
        )
        predictor = model.deploy(1, alternative_cpu_instance_type, endpoint_name=endpoint_name)
        endpoint_desc = sagemaker_session.sagemaker_client.describe_endpoint(
            EndpointName=endpoint_name
        )
        old_config_name = endpoint_desc["EndpointConfigName"]

        predictor.update_endpoint(initial_instance_count=1, instance_type=cpu_instance_type)

        endpoint_desc = sagemaker_session.sagemaker_client.describe_endpoint(
            EndpointName=endpoint_name
        )
        new_config_name = endpoint_desc["EndpointConfigName"]
        new_config = sagemaker_session.sagemaker_client.describe_endpoint_config(
            EndpointConfigName=new_config_name
        )

        assert old_config_name != new_config_name
        assert new_config["ProductionVariants"][0]["InstanceType"] == cpu_instance_type
        assert new_config["ProductionVariants"][0]["InitialInstanceCount"] == 1


def test_deploy_model_with_serverless_inference_config(
    mxnet_training_job,
    sagemaker_session,
    mxnet_inference_latest_version,
    mxnet_inference_latest_py_version,
):
    endpoint_name = unique_name_from_base("test-mxnet-deploy-model-serverless")

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        desc = sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=mxnet_training_job
        )
        model_data = desc["ModelArtifacts"]["S3ModelArtifacts"]
        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist.py")
        model = MXNetModel(
            model_data,
            "SageMakerRole",
            entry_point=script_path,
            py_version=mxnet_inference_latest_py_version,
            sagemaker_session=sagemaker_session,
            framework_version=mxnet_inference_latest_version,
        )
        predictor = model.deploy(
            serverless_inference_config=ServerlessInferenceConfig(), endpoint_name=endpoint_name
        )

        data = numpy.zeros(shape=(1, 1, 28, 28))
        result = predictor.predict(data)

        print("==========Result is===========")
        print(result)
        assert result is not None

    model.delete_model()
    with pytest.raises(Exception) as exception:
        sagemaker_session.sagemaker_client.describe_model(ModelName=model.name)
        assert "Could not find model" in str(exception.value)


def test_async_fit(
    sagemaker_session,
    mxnet_training_latest_version,
    mxnet_inference_latest_py_version,
    cpu_instance_type,
):
    endpoint_name = unique_name_from_base("test-mxnet-attach-deploy")

    with timeout(minutes=5):
        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        mx = MXNet(
            entry_point=script_path,
            role="SageMakerRole",
            py_version=mxnet_inference_latest_py_version,
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            framework_version=mxnet_training_latest_version,
            distribution={"parameter_server": {"enabled": True}},
        )

        train_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
        )
        test_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
        )

        mx.fit({"train": train_input, "test": test_input}, wait=False)
        training_job_name = mx.latest_training_job.name

        print("Waiting to re-attach to the training job: %s" % training_job_name)
        time.sleep(20)

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        print("Re-attaching now to: %s" % training_job_name)
        estimator = MXNet.attach(
            training_job_name=training_job_name, sagemaker_session=sagemaker_session
        )
        predictor = estimator.deploy(1, cpu_instance_type, endpoint_name=endpoint_name)
        data = numpy.zeros(shape=(1, 1, 28, 28))
        result = predictor.predict(data)
        assert result is not None
