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

from sagemaker import KMeans, s3, get_execution_role
from sagemaker.mxnet import MXNet
from sagemaker.pytorch import PyTorchModel
from sagemaker.tensorflow import TensorFlow
from sagemaker.transformer import Transformer
from sagemaker.estimator import Estimator
from sagemaker.inputs import BatchDataCaptureConfig
from sagemaker.xgboost import XGBoostModel
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

from sagemaker.model_monitor import DatasetFormat, Statistics, Constraints

from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
)
from sagemaker.workflow.parameters import ParameterString
from sagemaker.s3 import S3Uploader
from sagemaker.clarify import (
    BiasConfig,
    DataConfig,
)
from sagemaker.workflow.clarify_check_step import (
    DataBiasCheckConfig,
)

_INSTANCE_COUNT = 1
_INSTANCE_TYPE = "ml.c5.xlarge"
_HEADERS = ["Label", "F1", "F2", "F3", "F4"]
_CHECK_FAIL_ERROR_MSG_CLARIFY = "ClientError: Clarify check failed. See violation report"
_PROBLEM_TYPE = "Regression"
_HEADER_OF_LABEL = "Label"
_HEADER_OF_PREDICTED_LABEL = "Prediction"
_CHECK_FAIL_ERROR_MSG_QUALITY = "ClientError: Quality check failed. See violation report"


MXNET_MNIST_PATH = os.path.join(DATA_DIR, "mxnet_mnist")


@pytest.fixture(scope="module")
def role(sagemaker_session):
    return get_execution_role(sagemaker_session)


@pytest.fixture
def pipeline_name():
    return unique_name_from_base("my-pipeline-transform")


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


@pytest.fixture
def check_job_config(role, pipeline_session):
    return CheckJobConfig(
        role=role,
        instance_count=_INSTANCE_COUNT,
        instance_type=_INSTANCE_TYPE,
        volume_size_in_gb=60,
        sagemaker_session=pipeline_session,
    )


@pytest.fixture
def supplied_baseline_statistics_uri_param():
    return ParameterString(name="SuppliedBaselineStatisticsUri", default_value="")


@pytest.fixture
def supplied_baseline_constraints_uri_param():
    return ParameterString(name="SuppliedBaselineConstraintsUri", default_value="")


@pytest.fixture
def dataset(pipeline_session):
    dataset_local_path = os.path.join(DATA_DIR, "pipeline/clarify_check_step/dataset.csv")
    dataset_s3_uri = "s3://{}/{}/{}/{}/{}".format(
        pipeline_session.default_bucket(),
        "clarify_check_step",
        "input",
        "dataset",
        unique_name_from_base("dataset"),
    )
    return S3Uploader.upload(dataset_local_path, dataset_s3_uri, sagemaker_session=pipeline_session)


@pytest.fixture
def data_config(pipeline_session, dataset):
    output_path = "s3://{}/{}/{}/{}".format(
        pipeline_session.default_bucket(),
        "clarify_check_step",
        "analysis_result",
        unique_name_from_base("result"),
    )
    analysis_cfg_output_path = "s3://{}/{}/{}/{}".format(
        pipeline_session.default_bucket(),
        "clarify_check_step",
        "analysis_cfg",
        unique_name_from_base("analysis_cfg"),
    )
    return DataConfig(
        s3_data_input_path=dataset,
        s3_output_path=output_path,
        s3_analysis_config_output_path=analysis_cfg_output_path,
        label="Label",
        headers=_HEADERS,
        dataset_type="text/csv",
    )


@pytest.fixture
def bias_config():
    return BiasConfig(
        label_values_or_threshold=[1],
        facet_name="F1",
        facet_values_or_threshold=[0.5],
        group_name="F2",
    )


@pytest.fixture
def data_bias_check_config(data_config, bias_config):
    return DataBiasCheckConfig(
        data_config=data_config,
        data_bias_config=bias_config,
    )


@pytest.fixture
def data_quality_baseline_dataset():
    return os.path.join(DATA_DIR, "pipeline/quality_check_step/data_quality/baseline_dataset.csv")


@pytest.fixture
def data_quality_check_config(data_quality_baseline_dataset):
    return DataQualityCheckConfig(
        baseline_dataset=data_quality_baseline_dataset,
        dataset_format=DatasetFormat.csv(header=False),
    )


@pytest.fixture
def data_quality_supplied_baseline_statistics(sagemaker_session):
    return Statistics.from_file_path(
        statistics_file_path=os.path.join(
            DATA_DIR, "pipeline/quality_check_step/data_quality/statistics.json"
        ),
        sagemaker_session=sagemaker_session,
    ).file_s3_uri


@pytest.fixture
def model_quality_baseline_dataset():
    return os.path.join(DATA_DIR, "pipeline/quality_check_step/model_quality/baseline_dataset.csv")


@pytest.fixture
def model_quality_check_config(model_quality_baseline_dataset):
    return ModelQualityCheckConfig(
        baseline_dataset=model_quality_baseline_dataset,
        dataset_format=DatasetFormat.csv(),
        problem_type=_PROBLEM_TYPE,
        inference_attribute=_HEADER_OF_LABEL,
        ground_truth_attribute=_HEADER_OF_PREDICTED_LABEL,
    )


@pytest.fixture
def model_quality_supplied_baseline_statistics(sagemaker_session):
    return Statistics.from_file_path(
        statistics_file_path=os.path.join(
            DATA_DIR, "pipeline/quality_check_step/model_quality/statistics.json"
        ),
        sagemaker_session=sagemaker_session,
    ).file_s3_uri


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


def test_transform_data_capture_config(
    mxnet_estimator, mxnet_transform_input, sagemaker_session, cpu_instance_type
):
    destination_s3_uri = os.path.join("s3://", sagemaker_session.default_bucket(), "data_capture")
    batch_data_capture_config = BatchDataCaptureConfig(
        destination_s3_uri=destination_s3_uri, kms_key_id="", generate_inference_id=False
    )
    transformer = mxnet_estimator.transformer(1, cpu_instance_type)

    # we extract the S3Prefix from the input
    filename = mxnet_transform_input.split("/")[-1]
    input_prefix = mxnet_transform_input.replace(f"/{filename}", "")
    transformer.transform(
        input_prefix,
        content_type="text/csv",
        batch_data_capture_config=batch_data_capture_config,
    )

    with timeout_and_delete_model_with_transformer(
        transformer, sagemaker_session, minutes=TRANSFORM_DEFAULT_TIMEOUT_MINUTES
    ):
        transformer.wait()
        transform_job_desc = sagemaker_session.sagemaker_client.describe_transform_job(
            TransformJobName=transformer.latest_transform_job.name
        )

        assert (
            batch_data_capture_config._to_request_dict() == transform_job_desc["DataCaptureConfig"]
        )


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
        assert transformer.output_path == "s3://{}/{}/{}".format(
            sagemaker_session.default_bucket(), sagemaker_session.default_bucket_prefix, job_name
        )
        job_name = unique_name_from_base("test-mxnet-transform")
        transformer.transform(mxnet_transform_input, content_type="text/csv", job_name=job_name)
        assert transformer.output_path == "s3://{}/{}/{}".format(
            sagemaker_session.default_bucket(), sagemaker_session.default_bucket_prefix, job_name
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
        entry_point="mnist.py",
        source_dir=data_path,
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
            prediction_0 = result["predictions"][0]
            if type(prediction_0) is dict:
                assert len(result["predictions"][0]["probabilities"]) == 10
                assert result["predictions"][0]["classes"] >= 1
            else:
                assert len(result["predictions"][0]) == 10


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


def test_transformer_and_monitoring_job(
    pipeline_session,
    sagemaker_session,
    role,
    pipeline_name,
    check_job_config,
    data_bias_check_config,
):
    xgb_model_data_s3 = pipeline_session.upload_data(
        path=os.path.join(os.path.join(DATA_DIR, "xgboost_abalone"), "xgb_model.tar.gz"),
        key_prefix="integ-test-data/xgboost/model",
    )
    data_bias_supplied_baseline_constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(
            DATA_DIR, "pipeline/clarify_check_step/data_bias/good_cases/analysis.json"
        ),
        sagemaker_session=sagemaker_session,
    ).file_s3_uri

    xgb_model = XGBoostModel(
        model_data=xgb_model_data_s3,
        framework_version="1.3-1",
        role=role,
        sagemaker_session=sagemaker_session,
        entry_point=os.path.join(os.path.join(DATA_DIR, "xgboost_abalone"), "inference.py"),
        enable_network_isolation=True,
    )

    xgb_model.deploy(_INSTANCE_COUNT, _INSTANCE_TYPE)

    transform_output = f"s3://{sagemaker_session.default_bucket()}/{pipeline_name}Transform"
    transformer = Transformer(
        model_name=xgb_model.name,
        strategy="SingleRecord",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        output_path=transform_output,
        sagemaker_session=pipeline_session,
    )

    transform_input = pipeline_session.upload_data(
        path=os.path.join(DATA_DIR, "xgboost_abalone", "abalone"),
        key_prefix="integ-test-data/xgboost_abalone/abalone",
    )

    execution = transformer.transform_with_monitoring(
        monitoring_config=data_bias_check_config,
        monitoring_resource_config=check_job_config,
        data=transform_input,
        content_type="text/libsvm",
        supplied_baseline_constraints=data_bias_supplied_baseline_constraints,
        role=role,
    )

    execution_steps = execution.list_steps()
    assert len(execution_steps) == 2

    for execution_step in execution_steps:
        assert execution_step["StepStatus"] == "Succeeded"

    xgb_model.delete_model()
