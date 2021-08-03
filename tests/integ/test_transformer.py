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
import time

import pytest

from sagemaker import KMeans, s3
from sagemaker.mxnet import MXNet
from sagemaker.pytorch import PyTorchModel
from sagemaker.tensorflow import TensorFlow
from sagemaker.transformer import Transformer
from sagemaker.estimator import Estimator
from sagemaker.utils import unique_name_from_base
from tests.integ import (
    datasets,
    DATA_DIR,
    TRAINING_DEFAULT_TIMEOUT_MINUTES,
    TRANSFORM_DEFAULT_TIMEOUT_MINUTES,
)
from tests.integ.kms_utils import bucket_with_encryption, get_or_create_kms_key
from tests.integ.timeout import timeout, timeout_and_delete_model_with_transformer
from tests.integ.vpc_test_utils import get_or_create_vpc_resources

MXNET_MNIST_PATH = os.path.join(DATA_DIR, "mxnet_mnist")


@pytest.fixture(scope="module")
def mxnet_estimator(
    sagemaker_session,
    mxnet_inference_latest_version,
    mxnet_inference_latest_py_version,
    cpu_instance_type,
):
    mx = MXNet(
        entry_point=os.path.join(MXNET_MNIST_PATH, "mnist.py"),
        role="SageMakerRole",
        instance_count=1,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        framework_version=mxnet_inference_latest_version,
        py_version=mxnet_inference_latest_py_version,
    )

    train_input = mx.sagemaker_session.upload_data(
        path=os.path.join(MXNET_MNIST_PATH, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
    )
    test_input = mx.sagemaker_session.upload_data(
        path=os.path.join(MXNET_MNIST_PATH, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
    )

    job_name = unique_name_from_base("test-mxnet-transform")
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        mx.fit({"train": train_input, "test": test_input}, job_name=job_name)

    return mx


@pytest.fixture(scope="module")
def mxnet_transform_input(sagemaker_session):
    transform_input_path = os.path.join(MXNET_MNIST_PATH, "transform", "data.csv")
    transform_input_key_prefix = "integ-test-data/mxnet_mnist/transform"
    return sagemaker_session.upload_data(
        path=transform_input_path, key_prefix=transform_input_key_prefix
    )


@pytest.mark.release
def test_transform_mxnet(
    mxnet_estimator, mxnet_transform_input, sagemaker_session, cpu_instance_type
):
    kms_key_arn = get_or_create_kms_key(sagemaker_session)
    output_filter = "$"
    input_filter = "$"

    transformer = _create_transformer_and_transform_job(
        mxnet_estimator,
        mxnet_transform_input,
        cpu_instance_type,
        kms_key_arn,
        input_filter=input_filter,
        output_filter=output_filter,
        join_source=None,
    )
    with timeout_and_delete_model_with_transformer(
        transformer, sagemaker_session, minutes=TRANSFORM_DEFAULT_TIMEOUT_MINUTES
    ):
        transformer.wait()

    job_desc = transformer.sagemaker_session.describe_transform_job(
        job_name=transformer.latest_transform_job.name
    )
    assert kms_key_arn == job_desc["TransformResources"]["VolumeKmsKeyId"]
    assert output_filter == job_desc["DataProcessing"]["OutputFilter"]
    assert input_filter == job_desc["DataProcessing"]["InputFilter"]


@pytest.mark.release
def test_attach_transform_kmeans(sagemaker_session, cpu_instance_type):
    kmeans = KMeans(
        role="SageMakerRole",
        instance_count=1,
        instance_type=cpu_instance_type,
        k=10,
        sagemaker_session=sagemaker_session,
        output_path="s3://{}/".format(sagemaker_session.default_bucket()),
    )

    # set kmeans specific hp
    kmeans.init_method = "random"
    kmeans.max_iterators = 1
    kmeans.tol = 1
    kmeans.num_trials = 1
    kmeans.local_init_method = "kmeans++"
    kmeans.half_life_time_size = 1
    kmeans.epochs = 1

    records = kmeans.record_set(datasets.one_p_mnist()[0][:100])

    job_name = unique_name_from_base("test-kmeans-attach")

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        kmeans.fit(records, job_name=job_name)

    transform_input_path = os.path.join(DATA_DIR, "one_p_mnist", "transform_input.csv")
    transform_input_key_prefix = "integ-test-data/one_p_mnist/transform"
    transform_input = kmeans.sagemaker_session.upload_data(
        path=transform_input_path, key_prefix=transform_input_key_prefix
    )

    transformer = _create_transformer_and_transform_job(kmeans, transform_input, cpu_instance_type)

    attached_transformer = Transformer.attach(
        transformer.latest_transform_job.name, sagemaker_session=sagemaker_session
    )
    with timeout_and_delete_model_with_transformer(
        transformer, sagemaker_session, minutes=TRANSFORM_DEFAULT_TIMEOUT_MINUTES
    ):
        attached_transformer.wait()


def test_transform_pytorch_vpc_custom_model_bucket(
    sagemaker_session,
    pytorch_inference_latest_version,
    pytorch_inference_latest_py_version,
    cpu_instance_type,
    custom_bucket_name,
):
    data_dir = os.path.join(DATA_DIR, "pytorch_mnist")

    ec2_client = sagemaker_session.boto_session.client("ec2")
    subnet_ids, security_group_id = get_or_create_vpc_resources(ec2_client)

    model_data = sagemaker_session.upload_data(
        path=os.path.join(data_dir, "model.tar.gz"),
        bucket=custom_bucket_name,
        key_prefix="integ-test-data/pytorch_mnist/model",
    )

    model = PyTorchModel(
        model_data=model_data,
        entry_point=os.path.join(data_dir, "mnist.py"),
        role="SageMakerRole",
        framework_version=pytorch_inference_latest_version,
        py_version=pytorch_inference_latest_py_version,
        sagemaker_session=sagemaker_session,
        vpc_config={"Subnets": subnet_ids, "SecurityGroupIds": [security_group_id]},
        code_location="s3://{}".format(custom_bucket_name),
    )

    transform_input = sagemaker_session.upload_data(
        path=os.path.join(data_dir, "transform", "data.npy"),
        key_prefix="integ-test-data/pytorch_mnist/transform",
    )

    transformer = model.transformer(1, cpu_instance_type)
    transformer.transform(
        transform_input,
        content_type="application/x-npy",
        job_name=unique_name_from_base("test-transform-vpc"),
    )

    with timeout_and_delete_model_with_transformer(
        transformer, sagemaker_session, minutes=TRANSFORM_DEFAULT_TIMEOUT_MINUTES
    ):
        transformer.wait()
        model_desc = sagemaker_session.sagemaker_client.describe_model(
            ModelName=transformer.model_name
        )
        assert set(subnet_ids) == set(model_desc["VpcConfig"]["Subnets"])
        assert [security_group_id] == model_desc["VpcConfig"]["SecurityGroupIds"]

        model_bucket, _ = s3.parse_s3_url(model_desc["PrimaryContainer"]["ModelDataUrl"])
        assert custom_bucket_name == model_bucket


def test_transform_mxnet_tags(
    mxnet_estimator, mxnet_transform_input, sagemaker_session, cpu_instance_type
):
    tags = [{"Key": "some-tag", "Value": "value-for-tag"}]

    transformer = mxnet_estimator.transformer(1, cpu_instance_type, tags=tags)
    transformer.transform(mxnet_transform_input, content_type="text/csv")

    with timeout_and_delete_model_with_transformer(
        transformer, sagemaker_session, minutes=TRANSFORM_DEFAULT_TIMEOUT_MINUTES
    ):
        transformer.wait()
        model_desc = sagemaker_session.sagemaker_client.describe_model(
            ModelName=transformer.model_name
        )
        model_tags = sagemaker_session.sagemaker_client.list_tags(
            ResourceArn=model_desc["ModelArn"]
        )["Tags"]
        assert tags == model_tags


def test_transform_model_client_config(
    mxnet_estimator, mxnet_transform_input, sagemaker_session, cpu_instance_type
):
    model_client_config = {"InvocationsTimeoutInSeconds": 60, "InvocationsMaxRetries": 2}
    transformer = mxnet_estimator.transformer(1, cpu_instance_type)
    transformer.transform(
        mxnet_transform_input, content_type="text/csv", model_client_config=model_client_config
    )

    with timeout_and_delete_model_with_transformer(
        transformer, sagemaker_session, minutes=TRANSFORM_DEFAULT_TIMEOUT_MINUTES
    ):
        transformer.wait()
        transform_job_desc = sagemaker_session.sagemaker_client.describe_transform_job(
            TransformJobName=transformer.latest_transform_job.name
        )

        assert model_client_config == transform_job_desc["ModelClientConfig"]


def test_transform_byo_estimator(sagemaker_session, cpu_instance_type):
    tags = [{"Key": "some-tag", "Value": "value-for-tag"}]

    kmeans = KMeans(
        role="SageMakerRole",
        instance_count=1,
        instance_type=cpu_instance_type,
        k=10,
        sagemaker_session=sagemaker_session,
        output_path="s3://{}/".format(sagemaker_session.default_bucket()),
    )

    # set kmeans specific hp
    kmeans.init_method = "random"
    kmeans.max_iterators = 1
    kmeans.tol = 1
    kmeans.num_trials = 1
    kmeans.local_init_method = "kmeans++"
    kmeans.half_life_time_size = 1
    kmeans.epochs = 1

    records = kmeans.record_set(datasets.one_p_mnist()[0][:100])

    job_name = unique_name_from_base("test-kmeans-attach")

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        kmeans.fit(records, job_name=job_name)

    estimator = Estimator.attach(training_job_name=job_name, sagemaker_session=sagemaker_session)
    estimator._enable_network_isolation = True

    transform_input_path = os.path.join(DATA_DIR, "one_p_mnist", "transform_input.csv")
    transform_input_key_prefix = "integ-test-data/one_p_mnist/transform"
    transform_input = kmeans.sagemaker_session.upload_data(
        path=transform_input_path, key_prefix=transform_input_key_prefix
    )

    transformer = estimator.transformer(1, cpu_instance_type, tags=tags)
    transformer.transform(transform_input, content_type="text/csv")

    with timeout_and_delete_model_with_transformer(
        transformer, sagemaker_session, minutes=TRANSFORM_DEFAULT_TIMEOUT_MINUTES
    ):
        transformer.wait()
        model_desc = sagemaker_session.sagemaker_client.describe_model(
            ModelName=transformer.model_name
        )
        assert model_desc["EnableNetworkIsolation"]

        model_tags = sagemaker_session.sagemaker_client.list_tags(
            ResourceArn=model_desc["ModelArn"]
        )["Tags"]
        assert tags == model_tags


def test_single_transformer_multiple_jobs(
    mxnet_estimator, mxnet_transform_input, sagemaker_session, cpu_instance_type
):
    transformer = mxnet_estimator.transformer(1, cpu_instance_type)

    job_name = unique_name_from_base("test-mxnet-transform")
    transformer.transform(mxnet_transform_input, content_type="text/csv", job_name=job_name)
    with timeout_and_delete_model_with_transformer(
        transformer, sagemaker_session, minutes=TRANSFORM_DEFAULT_TIMEOUT_MINUTES
    ):
        assert transformer.output_path == "s3://{}/{}".format(
            sagemaker_session.default_bucket(), job_name
        )
        job_name = unique_name_from_base("test-mxnet-transform")
        transformer.transform(mxnet_transform_input, content_type="text/csv", job_name=job_name)
        assert transformer.output_path == "s3://{}/{}".format(
            sagemaker_session.default_bucket(), job_name
        )


def test_stop_transform_job(mxnet_estimator, mxnet_transform_input, cpu_instance_type):
    transformer = mxnet_estimator.transformer(1, cpu_instance_type)
    transformer.transform(mxnet_transform_input, content_type="text/csv", wait=False)

    time.sleep(15)

    latest_transform_job_name = transformer.latest_transform_job.name

    print("Attempting to stop {}".format(latest_transform_job_name))

    transformer.stop_transform_job()

    desc = transformer.latest_transform_job.sagemaker_session.describe_transform_job(
        job_name=latest_transform_job_name
    )
    assert desc["TransformJobStatus"] == "Stopped"


def test_transform_mxnet_logs(
    mxnet_estimator, mxnet_transform_input, sagemaker_session, cpu_instance_type
):
    with timeout(minutes=45):
        transformer = _create_transformer_and_transform_job(
            mxnet_estimator, mxnet_transform_input, cpu_instance_type, wait=True, logs=True
        )

    with timeout_and_delete_model_with_transformer(
        transformer, sagemaker_session, minutes=TRANSFORM_DEFAULT_TIMEOUT_MINUTES
    ):
        transformer.wait()


def test_transform_tf_kms_network_isolation(
    sagemaker_session, cpu_instance_type, tmpdir, tf_full_version, tf_full_py_version
):
    data_path = os.path.join(DATA_DIR, "tensorflow_mnist")

    tf = TensorFlow(
        entry_point=os.path.join(data_path, "mnist.py"),
        role="SageMakerRole",
        instance_count=1,
        instance_type=cpu_instance_type,
        framework_version=tf_full_version,
        py_version=tf_full_py_version,
        sagemaker_session=sagemaker_session,
    )

    s3_prefix = "integ-test-data/tf-scriptmode/mnist"
    training_input = sagemaker_session.upload_data(
        path=os.path.join(data_path, "data"), key_prefix="{}/training".format(s3_prefix)
    )

    job_name = unique_name_from_base("test-tf-transform")
    tf.fit(inputs=training_input, job_name=job_name)

    transform_input = sagemaker_session.upload_data(
        path=os.path.join(data_path, "transform"), key_prefix="{}/transform".format(s3_prefix)
    )

    with bucket_with_encryption(sagemaker_session, "SageMakerRole") as (bucket_with_kms, kms_key):
        output_path = "{}/{}/output".format(bucket_with_kms, job_name)

        transformer = tf.transformer(
            instance_count=1,
            instance_type=cpu_instance_type,
            output_path=output_path,
            output_kms_key=kms_key,
            volume_kms_key=kms_key,
            enable_network_isolation=True,
        )

        with timeout_and_delete_model_with_transformer(
            transformer, sagemaker_session, minutes=TRANSFORM_DEFAULT_TIMEOUT_MINUTES
        ):
            transformer.transform(
                transform_input, job_name=job_name, content_type="text/csv", wait=True
            )

            model_desc = sagemaker_session.sagemaker_client.describe_model(
                ModelName=transformer.model_name
            )
            assert model_desc["EnableNetworkIsolation"]

        job_desc = sagemaker_session.describe_transform_job(job_name=job_name)
        assert job_desc["TransformOutput"]["S3OutputPath"] == output_path
        assert job_desc["TransformOutput"]["KmsKeyId"] == kms_key
        assert job_desc["TransformResources"]["VolumeKmsKeyId"] == kms_key

        s3.S3Downloader.download(
            s3_uri=output_path,
            local_path=os.path.join(tmpdir, "tf-batch-output"),
            sagemaker_session=sagemaker_session,
        )

        with open(os.path.join(tmpdir, "tf-batch-output", "data.csv.out")) as f:
            result = json.load(f)
            assert len(result["predictions"][0]["probabilities"]) == 10
            assert result["predictions"][0]["classes"] >= 1


def _create_transformer_and_transform_job(
    estimator,
    transform_input,
    instance_type,
    volume_kms_key=None,
    input_filter=None,
    output_filter=None,
    join_source=None,
    wait=False,
    logs=False,
):
    transformer = estimator.transformer(1, instance_type, volume_kms_key=volume_kms_key)
    transformer.transform(
        transform_input,
        content_type="text/csv",
        input_filter=input_filter,
        output_filter=output_filter,
        join_source=join_source,
        wait=wait,
        logs=logs,
        job_name=unique_name_from_base("test-transform"),
    )
    return transformer
