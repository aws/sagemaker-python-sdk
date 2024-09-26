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

import numpy as np
import os
import time

import pytest

from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.tensorflow import TensorFlow, TensorFlowProcessor, TensorFlowModel
from sagemaker.utils import unique_name_from_base, sagemaker_timestamp

import tests.integ
from tests.integ import DATA_DIR, TRAINING_DEFAULT_TIMEOUT_MINUTES, kms_utils, timeout
from tests.integ.retry import retries
from tests.integ.utils import gpu_list, retry_with_instance_list
from tests.integ.s3_utils import assert_s3_file_patterns_exist

from packaging.version import Version

ROLE = "SageMakerRole"

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
MNIST_RESOURCE_PATH = os.path.join(RESOURCE_PATH, "tensorflow_mnist")
TFS_RESOURCE_PATH = os.path.join(RESOURCE_PATH, "tfs", "tfs-test-entrypoint-with-handler")

SCRIPT = "mnist.py"
PARAMETER_SERVER_DISTRIBUTION = {"parameter_server": {"enabled": True}}
MPI_DISTRIBUTION = {"mpi": {"enabled": True}}
MWMS_DISTRIBUTION = {"multi_worker_mirrored_strategy": {"enabled": True}}
TAGS = [{"Key": "some-key", "Value": "some-value"}]
ENV_INPUT = {"env_key1": "env_val1", "env_key2": "env_val2", "env_key3": "env_val3"}


@pytest.mark.release
@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.TRAINING_NO_P2_REGIONS
    and tests.integ.test_region() in tests.integ.TRAINING_NO_P3_REGIONS,
    reason="no ml.p2 or ml.p3 instances in this region",
)
@retry_with_instance_list(gpu_list(tests.integ.test_region()))
def test_framework_processing_job_with_deps(
    sagemaker_session,
    tensorflow_training_latest_version,
    tensorflow_training_latest_py_version,
    **kwargs,
):
    if (
        Version(tensorflow_training_latest_version) >= Version("2.12")
        and kwargs["instance_type"] == "ml.p2.xlarge"
    ):
        pytest.skip("P2 instances have been deprecated for sagemaker jobs starting TensorFlow 2.12")

    with timeout.timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        code_path = os.path.join(DATA_DIR, "dummy_code_bundle_with_reqs")
        entry_point = "main_script.py"

        processor = TensorFlowProcessor(
            framework_version=tensorflow_training_latest_version,
            py_version=tensorflow_training_latest_py_version,
            role=ROLE,
            instance_count=1,
            instance_type=kwargs["instance_type"],
            sagemaker_session=sagemaker_session,
            base_job_name="test-tensorflow",
        )
        processor.run(code=entry_point, source_dir=code_path, inputs=[], wait=True)


def test_mnist_with_checkpoint_config(
    sagemaker_session,
    instance_type,
    tensorflow_training_latest_version,
    tensorflow_training_latest_py_version,
):
    if Version(tensorflow_training_latest_version) >= Version("2.16"):
        pytest.skip(
            "This test is failing in TensorFlow 2.16 beacuse of an upstream bug: "
            "https://github.com/tensorflow/io/issues/2039"
        )
    checkpoint_s3_uri = "s3://{}/checkpoints/tf-{}".format(
        sagemaker_session.default_bucket(), sagemaker_timestamp()
    )
    checkpoint_local_path = "/test/checkpoint/path"
    estimator = TensorFlow(
        entry_point=SCRIPT,
        source_dir=MNIST_RESOURCE_PATH,
        role=ROLE,
        instance_count=1,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        framework_version=tensorflow_training_latest_version,
        py_version=tensorflow_training_latest_py_version,
        metric_definitions=[{"Name": "train:global_steps", "Regex": r"global_step\/sec:\s(.*)"}],
        checkpoint_s3_uri=checkpoint_s3_uri,
        checkpoint_local_path=checkpoint_local_path,
        environment=ENV_INPUT,
        max_wait=24 * 60 * 60,
        max_retry_attempts=2,
    )
    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(MNIST_RESOURCE_PATH, "data"), key_prefix="scriptmode/mnist"
    )

    training_job_name = unique_name_from_base("test-tf-sm-mnist")
    with tests.integ.timeout.timeout(minutes=tests.integ.TRAINING_DEFAULT_TIMEOUT_MINUTES):
        estimator.fit(inputs=inputs, job_name=training_job_name)
    assert_s3_file_patterns_exist(
        sagemaker_session, estimator.model_dir, [r"model\.ckpt-\d+\.index", r"checkpoint"]
    )
    # remove dataframe assertion to unblock PR build
    # TODO: add independent integration test for `training_job_analytics`

    expected_training_checkpoint_config = {
        "S3Uri": checkpoint_s3_uri,
        "LocalPath": checkpoint_local_path,
    }
    actual_training_checkpoint_config = sagemaker_session.sagemaker_client.describe_training_job(
        TrainingJobName=training_job_name
    )["CheckpointConfig"]
    actual_training_environment_variable_config = (
        sagemaker_session.sagemaker_client.describe_training_job(TrainingJobName=training_job_name)[
            "Environment"
        ]
    )

    expected_retry_strategy = {"MaximumRetryAttempts": 2}
    actual_retry_strategy = sagemaker_session.sagemaker_client.describe_training_job(
        TrainingJobName=training_job_name
    )["RetryStrategy"]
    assert actual_training_checkpoint_config == expected_training_checkpoint_config
    assert actual_training_environment_variable_config == ENV_INPUT
    assert actual_retry_strategy == expected_retry_strategy


def test_server_side_encryption(sagemaker_session, tf_full_version, tf_full_py_version):
    with kms_utils.bucket_with_encryption(sagemaker_session, ROLE) as (bucket_with_kms, kms_key):
        output_path = os.path.join(
            bucket_with_kms, "test-server-side-encryption", time.strftime("%y%m%d-%H%M")
        )

        estimator = TensorFlow(
            entry_point="training.py",
            source_dir=TFS_RESOURCE_PATH,
            role=ROLE,
            instance_count=1,
            instance_type="ml.c5.xlarge",
            sagemaker_session=sagemaker_session,
            framework_version=tf_full_version,
            py_version=tf_full_py_version,
            code_location=output_path,
            output_path=output_path,
            model_dir="/opt/ml/model",
            output_kms_key=kms_key,
        )

        inputs = estimator.sagemaker_session.upload_data(
            path=os.path.join(MNIST_RESOURCE_PATH, "data"), key_prefix="scriptmode/mnist"
        )

        with tests.integ.timeout.timeout(minutes=tests.integ.TRAINING_DEFAULT_TIMEOUT_MINUTES):
            estimator.fit(
                inputs=inputs, job_name=unique_name_from_base("test-server-side-encryption")
            )

        endpoint_name = unique_name_from_base("test-server-side-encryption")
        with timeout.timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
            estimator.deploy(
                initial_instance_count=1,
                instance_type="ml.c5.xlarge",
                endpoint_name=endpoint_name,
                entry_point=os.path.join(TFS_RESOURCE_PATH, "inference.py"),
            )


@pytest.mark.release
@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.TRAINING_NO_P2_REGIONS
    and tests.integ.test_region() in tests.integ.TRAINING_NO_P3_REGIONS,
    reason="no ml.p2 or ml.p3 instances in this region",
)
@retry_with_instance_list(gpu_list(tests.integ.test_region()))
def test_mwms_gpu(
    sagemaker_session,
    tensorflow_training_latest_version,
    tensorflow_training_latest_py_version,
    capsys,
    **kwargs,
):
    if (
        Version(tensorflow_training_latest_version) >= Version("2.12")
        and kwargs["instance_type"] == "ml.p2.xlarge"
    ):
        pytest.skip("P2 instances have been deprecated for sagemaker jobs starting TensorFlow 2.12")

    instance_count = 2
    estimator = TensorFlow(
        source_dir=os.path.join(RESOURCE_PATH, "tensorflow_mnist"),
        entry_point="mnist_mwms.py",
        model_dir=False,
        instance_type=kwargs["instance_type"],
        instance_count=instance_count,
        framework_version=tensorflow_training_latest_version,
        py_version=tensorflow_training_latest_py_version,
        distribution=MWMS_DISTRIBUTION,
        environment={"NCCL_DEBUG": "INFO"},
        max_run=60 * 60 * 1,  # 1 hour
        role=ROLE,
        volume_size=400,
        sagemaker_session=sagemaker_session,
        disable_profiler=True,
    )

    with tests.integ.timeout.timeout(minutes=tests.integ.TRAINING_DEFAULT_TIMEOUT_MINUTES):
        estimator.fit(job_name=unique_name_from_base("test-tf-mwms"))

    captured = capsys.readouterr()
    logs = captured.out + captured.err
    print(logs)
    assert "Running distributed training job with multi_worker_mirrored_strategy setup" in logs
    assert f"strategy.num_replicas_in_sync={instance_count}" in logs


@pytest.mark.release
def test_mnist_distributed_cpu(
    sagemaker_session,
    cpu_instance_type,
    tensorflow_training_latest_version,
    tensorflow_training_latest_py_version,
):
    if Version(tensorflow_training_latest_version) >= Version("2.16"):
        pytest.skip(
            "This test is failing in TensorFlow 2.16 beacuse of an upstream bug: "
            "https://github.com/tensorflow/io/issues/2039"
        )
    _create_and_fit_estimator(
        sagemaker_session,
        tensorflow_training_latest_version,
        tensorflow_training_latest_py_version,
        cpu_instance_type,
    )


@pytest.mark.release
@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.TRAINING_NO_P2_REGIONS
    and tests.integ.test_region() in tests.integ.TRAINING_NO_P3_REGIONS,
    reason="no ml.p2 or ml.p3 instances in this region",
)
@retry_with_instance_list(gpu_list(tests.integ.test_region()))
def test_mnist_distributed_gpu(
    sagemaker_session,
    tensorflow_training_latest_version,
    tensorflow_training_latest_py_version,
    **kwargs,
):
    if (
        Version(tensorflow_training_latest_version) >= Version("2.12")
        and kwargs["instance_type"] == "ml.p2.xlarge"
    ):
        pytest.skip("P2 instances have been deprecated for sagemaker jobs starting TensorFlow 2.12")

    _create_and_fit_estimator(
        sagemaker_session,
        tensorflow_training_latest_version,
        tensorflow_training_latest_py_version,
        kwargs["instance_type"],
    )


def _create_and_fit_estimator(sagemaker_session, tf_version, py_version, instance_type):
    estimator = TensorFlow(
        entry_point=SCRIPT,
        source_dir=MNIST_RESOURCE_PATH,
        role=ROLE,
        instance_count=2,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        framework_version=tf_version,
        py_version=py_version,
        distribution=PARAMETER_SERVER_DISTRIBUTION,
        disable_profiler=True,
    )
    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(MNIST_RESOURCE_PATH, "data"), key_prefix="scriptmode/distributed_mnist"
    )

    with tests.integ.timeout.timeout(minutes=tests.integ.TRAINING_DEFAULT_TIMEOUT_MINUTES):
        estimator.fit(inputs=inputs, job_name=unique_name_from_base("test-tf-sm-distributed"))
    assert_s3_file_patterns_exist(
        sagemaker_session, estimator.model_dir, [r"model\.ckpt-\d+\.index", r"checkpoint"]
    )


@pytest.mark.slow_test
def test_mnist_async(sagemaker_session, cpu_instance_type, tf_full_version, tf_full_py_version):
    if Version(tf_full_version) >= Version("2.16"):
        pytest.skip(
            "This test is failing in TensorFlow 2.16 beacuse of an upstream bug: "
            "https://github.com/tensorflow/io/issues/2039"
        )
    if tf_full_version == "2.7.0":
        tf_full_version = "2.7"

    estimator = TensorFlow(
        entry_point=SCRIPT,
        source_dir=MNIST_RESOURCE_PATH,
        role=ROLE,
        instance_count=1,
        instance_type="ml.c5.4xlarge",
        sagemaker_session=sagemaker_session,
        framework_version=tf_full_version,
        py_version=tf_full_py_version,
        tags=TAGS,
    )
    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(MNIST_RESOURCE_PATH, "data"), key_prefix="scriptmode/mnist"
    )
    estimator.fit(inputs=inputs, wait=False, job_name=unique_name_from_base("test-tf-sm-async"))
    training_job_name = estimator.latest_training_job.name
    time.sleep(20)
    endpoint_name = training_job_name
    _assert_training_job_tags_match(
        sagemaker_session.sagemaker_client, estimator.latest_training_job.name, TAGS
    )
    with tests.integ.timeout.timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        estimator = TensorFlow.attach(
            training_job_name=training_job_name, sagemaker_session=sagemaker_session
        )
        model_name = "model-mnist-async"
        predictor = estimator.deploy(
            initial_instance_count=1,
            instance_type=cpu_instance_type,
            endpoint_name=endpoint_name,
            model_name=model_name,
        )

        result = predictor.predict(np.zeros(784))
        print("predict result: {}".format(result))
        _assert_endpoint_tags_match(
            sagemaker_session.sagemaker_client, predictor.endpoint_name, TAGS
        )
        _assert_model_tags_match(sagemaker_session.sagemaker_client, model_name, TAGS)
        _assert_model_name_match(sagemaker_session.sagemaker_client, endpoint_name, model_name)


def test_deploy_with_input_handlers(
    sagemaker_session, instance_type, tf_full_version, tf_full_py_version
):
    estimator = TensorFlow(
        entry_point="training.py",
        source_dir=TFS_RESOURCE_PATH,
        role=ROLE,
        instance_count=1,
        instance_type=instance_type,
        framework_version=tf_full_version,
        py_version=tf_full_py_version,
        sagemaker_session=sagemaker_session,
        tags=TAGS,
    )

    estimator.fit(job_name=unique_name_from_base("test-tf-tfs-deploy"))

    endpoint_name = estimator.latest_training_job.name

    with timeout.timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        predictor = estimator.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            entry_point=os.path.join(TFS_RESOURCE_PATH, "inference.py"),
        )

        input_data = {"instances": [1.0, 2.0, 5.0]}
        expected_result = {"predictions": [4.0, 4.5, 6.0]}

        result = predictor.predict(input_data)
        assert expected_result == result


def test_model_deploy_with_serverless_inference_config(
    sagemaker_session, tf_full_version, tf_full_py_version
):
    endpoint_name = unique_name_from_base("sagemaker-tensorflow-serverless")
    model_data = sagemaker_session.upload_data(
        path=os.path.join(tests.integ.DATA_DIR, "tensorflow-serving-test-model.tar.gz"),
        key_prefix="tensorflow-serving/models",
    )
    with tests.integ.timeout.timeout_and_delete_endpoint_by_name(
        endpoint_name=endpoint_name,
        sagemaker_session=sagemaker_session,
        hours=2,
        sleep_between_cleanup_attempts=20,
        exponential_sleep=True,
    ):
        model = TensorFlowModel(
            model_data=model_data,
            role=ROLE,
            framework_version=tf_full_version,
            sagemaker_session=sagemaker_session,
        )
        predictor = model.deploy(
            serverless_inference_config=ServerlessInferenceConfig(), endpoint_name=endpoint_name
        )

        input_data = {"instances": [1.0, 2.0, 5.0]}
        expected_result = {"predictions": [3.5, 4.0, 5.5]}

        result = predictor.predict(input_data)
        assert expected_result == result


def _assert_tags_match(sagemaker_client, resource_arn, tags, retry_count=15):
    # endpoint and training tags might take minutes to propagate.
    for _ in retries(retry_count, "Getting endpoint tags", seconds_to_sleep=30):
        actual_tags = sagemaker_client.list_tags(ResourceArn=resource_arn)["Tags"]
        if actual_tags:
            break

    assert actual_tags == tags


def _assert_model_tags_match(sagemaker_client, model_name, tags):
    model_description = sagemaker_client.describe_model(ModelName=model_name)
    _assert_tags_match(sagemaker_client, model_description["ModelArn"], tags)


def _assert_endpoint_tags_match(sagemaker_client, endpoint_name, tags):
    endpoint_description = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)

    _assert_tags_match(sagemaker_client, endpoint_description["EndpointArn"], tags)


def _assert_training_job_tags_match(sagemaker_client, training_job_name, tags):
    training_job_description = sagemaker_client.describe_training_job(
        TrainingJobName=training_job_name
    )
    _assert_tags_match(sagemaker_client, training_job_description["TrainingJobArn"], tags)


def _assert_model_name_match(sagemaker_client, endpoint_config_name, model_name):
    endpoint_config_description = sagemaker_client.describe_endpoint_config(
        EndpointConfigName=endpoint_config_name
    )
    assert model_name == endpoint_config_description["ProductionVariants"][0]["ModelName"]
