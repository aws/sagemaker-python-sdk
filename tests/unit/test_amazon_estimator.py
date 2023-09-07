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
import pytest
from mock import ANY, Mock, patch, call

from sagemaker import image_uris
from sagemaker.amazon.pca import PCA  # Use PCA as a test implementation of AmazonAlgorithmEstimator
from sagemaker.amazon.amazon_estimator import (
    upload_numpy_to_s3_shards,
    _build_shards,
    FileSystemRecordSet,
)
from sagemaker.session_settings import SessionSettings
from tests.unit import (
    DEFAULT_S3_OBJECT_KEY_PREFIX_NAME,
    DEFAULT_S3_BUCKET_NAME,
    _test_default_bucket_and_prefix_combinations,
)

COMMON_ARGS = {"role": "myrole", "instance_count": 1, "instance_type": "ml.c4.xlarge"}

REGION = "us-west-2"
BUCKET_NAME = "Some-Bucket"
DEFAULT_PREFIX_NAME = "Some-Prefix"
TIMESTAMP = "2017-11-06-14:14:15.671"


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    sms = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        region_name=REGION,
        config=None,
        local_mode=False,
        settings=SessionSettings(),
        default_bucket_prefix=None,
    )
    sms.boto_region_name = REGION
    sms.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    returned_job_description = {
        "AlgorithmSpecification": {
            "TrainingInputMode": "File",
            "TrainingImage": image_uris.retrieve("pca", "us-west-2"),
        },
        "ModelArtifacts": {"S3ModelArtifacts": "s3://some-bucket/model.tar.gz"},
        "HyperParameters": {
            "sagemaker_submit_directory": '"s3://some/sourcedir.tar.gz"',
            "checkpoint_path": '"s3://other/1508872349"',
            "sagemaker_program": '"iris-dnn-classifier.py"',
            "sagemaker_container_log_level": '"logging.INFO"',
            "sagemaker_job_name": '"neo"',
            "training_steps": "100",
        },
        "RoleArn": "arn:aws:iam::366:role/IMRole",
        "ResourceConfig": {
            "VolumeSizeInGB": 30,
            "InstanceCount": 1,
            "InstanceType": "ml.c4.xlarge",
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 24 * 60 * 60},
        "TrainingJobName": "neo",
        "TrainingJobStatus": "Completed",
        "OutputDataConfig": {"KmsKeyId": "", "S3OutputPath": "s3://place/output/neo"},
        "TrainingJobOutput": {"S3TrainingJobOutput": "s3://here/output.tar.gz"},
    }
    sms.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job", return_value=returned_job_description
    )
    # For tests which doesn't verify config file injection, operate with empty config
    sms.sagemaker_config = {}
    return sms


def test_init(sagemaker_session):
    pca = PCA(num_components=55, sagemaker_session=sagemaker_session, **COMMON_ARGS)
    assert pca.num_components == 55
    assert pca.enable_network_isolation() is False


def test_init_enable_network_isolation(sagemaker_session):
    pca = PCA(
        num_components=55,
        sagemaker_session=sagemaker_session,
        enable_network_isolation=True,
        **COMMON_ARGS,
    )
    assert pca.num_components == 55
    assert pca.enable_network_isolation() is True


def test_init_all_pca_hyperparameters(sagemaker_session):
    pca = PCA(
        num_components=55,
        algorithm_mode="randomized",
        subtract_mean=True,
        extra_components=33,
        sagemaker_session=sagemaker_session,
        **COMMON_ARGS,
    )
    assert pca.num_components == 55
    assert pca.algorithm_mode == "randomized"
    assert pca.extra_components == 33


def test_init_estimator_args(sagemaker_session):
    pca = PCA(
        num_components=1,
        max_run=1234,
        sagemaker_session=sagemaker_session,
        data_location="s3://some-bucket/some-key/",
        **COMMON_ARGS,
    )
    assert pca.instance_type == COMMON_ARGS["instance_type"]
    assert pca.instance_count == COMMON_ARGS["instance_count"]
    assert pca.role == COMMON_ARGS["role"]
    assert pca.max_run == 1234
    assert pca.data_location == "s3://some-bucket/some-key/"


def test_data_location_validation(sagemaker_session):
    pca = PCA(num_components=2, sagemaker_session=sagemaker_session, **COMMON_ARGS)
    with pytest.raises(ValueError):
        pca.data_location = "nots3://abcd/efgh"


def test_data_location_does_not_call_default_bucket(sagemaker_session):
    data_location = "s3://my-bucket/path/"
    pca = PCA(
        num_components=2,
        sagemaker_session=sagemaker_session,
        data_location=data_location,
        **COMMON_ARGS,
    )
    assert pca.data_location == data_location
    assert not sagemaker_session.default_bucket.called


def test_data_location_default_bucket_and_prefix_combinations():
    actual, expected = _test_default_bucket_and_prefix_combinations(
        function_with_user_input=(
            lambda sess: PCA(
                num_components=2,
                sagemaker_session=sess,
                data_location="s3://test",
                **COMMON_ARGS,
            ).data_location
        ),
        function_without_user_input=(
            lambda sess: PCA(num_components=2, sagemaker_session=sess, **COMMON_ARGS).data_location
        ),
        expected__without_user_input__with_default_bucket_and_default_prefix=(
            f"s3://{DEFAULT_S3_BUCKET_NAME}/{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/sagemaker-record-sets/"
        ),
        expected__without_user_input__with_default_bucket_only=f"s3://{DEFAULT_S3_BUCKET_NAME}/sagemaker-record-sets/",
        expected__with_user_input__with_default_bucket_and_prefix="s3://test",
        expected__with_user_input__with_default_bucket_only="s3://test",
    )
    assert actual == expected


def test_prepare_for_training(sagemaker_session):
    pca = PCA(num_components=55, sagemaker_session=sagemaker_session, **COMMON_ARGS)

    train = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 8.0], [44.0, 55.0, 66.0]]
    labels = [99, 85, 87, 2]
    records = pca.record_set(np.array(train), np.array(labels))

    pca._prepare_for_training(records, mini_batch_size=1)
    assert pca.feature_dim == 3
    assert pca.mini_batch_size == 1


def test_prepare_for_training_list(sagemaker_session):
    pca = PCA(num_components=55, sagemaker_session=sagemaker_session, **COMMON_ARGS)

    train = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 8.0], [44.0, 55.0, 66.0]]
    labels = [99, 85, 87, 2]
    records = [pca.record_set(np.array(train), np.array(labels))]

    pca._prepare_for_training(records, mini_batch_size=1)
    assert pca.feature_dim == 3
    assert pca.mini_batch_size == 1


def test_prepare_for_training_list_no_train_channel(sagemaker_session):
    pca = PCA(num_components=55, sagemaker_session=sagemaker_session, **COMMON_ARGS)

    train = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 8.0], [44.0, 55.0, 66.0]]
    labels = [99, 85, 87, 2]
    records = [pca.record_set(np.array(train), np.array(labels), "test")]

    with pytest.raises(ValueError) as ex:
        pca._prepare_for_training(records, mini_batch_size=1)

    assert "Must provide train channel." in str(ex)


def test_prepare_for_training_encrypt(sagemaker_session):
    pca = PCA(num_components=55, sagemaker_session=sagemaker_session, **COMMON_ARGS)

    train = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 8.0], [44.0, 55.0, 66.0]]
    labels = [99, 85, 87, 2]
    with patch(
        "sagemaker.amazon.amazon_estimator.upload_numpy_to_s3_shards", return_value="manfiest_file"
    ) as mock_upload:
        pca.record_set(np.array(train), np.array(labels))
        pca.record_set(np.array(train), np.array(labels), encrypt=True)

    def make_upload_call(encrypt):
        return call(ANY, ANY, ANY, ANY, ANY, ANY, encrypt)

    mock_upload.assert_has_calls([make_upload_call(False), make_upload_call(True)])


@patch("time.strftime", return_value=TIMESTAMP)
def test_fit_ndarray(time, sagemaker_session):
    mock_s3 = Mock()
    mock_object = Mock()
    mock_s3.Object = Mock(return_value=mock_object)
    sagemaker_session.boto_session.resource = Mock(return_value=mock_s3)
    kwargs = dict(COMMON_ARGS)
    kwargs["instance_count"] = 3
    pca = PCA(
        num_components=55,
        sagemaker_session=sagemaker_session,
        data_location="s3://{}/key-prefix/".format(BUCKET_NAME),
        **kwargs,
    )
    train = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 8.0], [44.0, 55.0, 66.0]]
    labels = [99, 85, 87, 2]
    pca.fit(pca.record_set(np.array(train), np.array(labels)))
    mock_s3.Object.assert_any_call(
        BUCKET_NAME, "key-prefix/PCA-2017-11-06-14:14:15.671/matrix_0.pbr"
    )
    mock_s3.Object.assert_any_call(
        BUCKET_NAME, "key-prefix/PCA-2017-11-06-14:14:15.671/matrix_1.pbr"
    )
    mock_s3.Object.assert_any_call(
        BUCKET_NAME, "key-prefix/PCA-2017-11-06-14:14:15.671/matrix_2.pbr"
    )
    mock_s3.Object.assert_any_call(
        BUCKET_NAME, "key-prefix/PCA-2017-11-06-14:14:15.671/.amazon.manifest"
    )

    assert mock_object.put.call_count == 4

    called_args = sagemaker_session.train.call_args
    assert not called_args[1]["experiment_config"]


def test_fit_pass_experiment_config(sagemaker_session):
    kwargs = dict(COMMON_ARGS)
    kwargs["instance_count"] = 3
    pca = PCA(
        num_components=55,
        sagemaker_session=sagemaker_session,
        data_location="s3://{}/key-prefix/".format(BUCKET_NAME),
        **kwargs,
    )
    train = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 8.0], [44.0, 55.0, 66.0]]
    labels = [99, 85, 87, 2]
    pca.fit(
        pca.record_set(np.array(train), np.array(labels)),
        experiment_config={
            "ExperimentName": "exp",
            "RunName": "rn",
        },
    )

    called_args = sagemaker_session.train.call_args

    assert called_args[1]["experiment_config"] == {
        "ExperimentName": "exp",
        "RunName": "rn",
    }


def test_build_shards():
    array = np.array([1, 2, 3, 4])
    shards = _build_shards(4, array)
    assert shards == [np.array([1]), np.array([2]), np.array([3]), np.array([4])]

    shards = _build_shards(3, array)
    for out, expected in zip(shards, map(np.array, [[1], [2], [3, 4]])):
        assert np.array_equal(out, expected)

    with pytest.raises(ValueError):
        shards = _build_shards(5, array)


def test_upload_numpy_to_s3_shards():
    mock_s3 = Mock()
    mock_object = Mock()
    mock_s3.Object = Mock(return_value=mock_object)
    mock_put = mock_s3.Object.return_value.put
    array = np.array([[j for j in range(10)] for i in range(10)])
    labels = np.array([i for i in range(10)])
    num_shards = 3
    num_objects = num_shards + 1  # Account for the manifest file.

    def make_all_put_calls(**kwargs):
        return [call(Body=ANY, **kwargs) for i in range(num_objects)]

    upload_numpy_to_s3_shards(num_shards, mock_s3, BUCKET_NAME, "key-prefix", array, labels)
    mock_s3.Object.assert_has_calls([call(BUCKET_NAME, "key-prefix/matrix_0.pbr")])
    mock_s3.Object.assert_has_calls([call(BUCKET_NAME, "key-prefix/matrix_1.pbr")])
    mock_s3.Object.assert_has_calls([call(BUCKET_NAME, "key-prefix/matrix_2.pbr")])
    mock_put.assert_has_calls(make_all_put_calls())

    mock_put.reset()
    upload_numpy_to_s3_shards(3, mock_s3, BUCKET_NAME, "key-prefix", array, labels, encrypt=True)
    mock_put.assert_has_calls(make_all_put_calls(ServerSideEncryption="AES256"))


def test_file_system_record_set_efs_default_parameters():
    file_system_id = "fs-0a48d2a1"
    file_system_type = "EFS"
    directory_path = "ipinsights"
    num_records = 1
    feature_dim = 1

    actual = FileSystemRecordSet(
        file_system_id=file_system_id,
        file_system_type=file_system_type,
        directory_path=directory_path,
        num_records=num_records,
        feature_dim=feature_dim,
    )

    expected_input_config = {
        "DataSource": {
            "FileSystemDataSource": {
                "DirectoryPath": "ipinsights",
                "FileSystemId": "fs-0a48d2a1",
                "FileSystemType": "EFS",
                "FileSystemAccessMode": "ro",
            }
        }
    }
    assert actual.file_system_input.config == expected_input_config
    assert actual.num_records == num_records
    assert actual.feature_dim == feature_dim
    assert actual.channel == "train"


def test_file_system_record_set_efs_customized_parameters():
    file_system_id = "fs-0a48d2a1"
    file_system_type = "EFS"
    directory_path = "ipinsights"
    num_records = 1
    feature_dim = 1

    actual = FileSystemRecordSet(
        file_system_id=file_system_id,
        file_system_type=file_system_type,
        directory_path=directory_path,
        num_records=num_records,
        feature_dim=feature_dim,
        file_system_access_mode="rw",
        channel="test",
    )

    expected_input_config = {
        "DataSource": {
            "FileSystemDataSource": {
                "DirectoryPath": "ipinsights",
                "FileSystemId": "fs-0a48d2a1",
                "FileSystemType": "EFS",
                "FileSystemAccessMode": "rw",
            }
        }
    }
    assert actual.file_system_input.config == expected_input_config
    assert actual.num_records == num_records
    assert actual.feature_dim == feature_dim
    assert actual.channel == "test"


def test_file_system_record_set_fsx_default_parameters():
    file_system_id = "fs-0a48d2a1"
    file_system_type = "FSxLustre"
    directory_path = "ipinsights"
    num_records = 1
    feature_dim = 1

    actual = FileSystemRecordSet(
        file_system_id=file_system_id,
        file_system_type=file_system_type,
        directory_path=directory_path,
        num_records=num_records,
        feature_dim=feature_dim,
    )
    expected_input_config = {
        "DataSource": {
            "FileSystemDataSource": {
                "DirectoryPath": "ipinsights",
                "FileSystemId": "fs-0a48d2a1",
                "FileSystemType": "FSxLustre",
                "FileSystemAccessMode": "ro",
            }
        }
    }
    assert actual.file_system_input.config == expected_input_config
    assert actual.num_records == num_records
    assert actual.feature_dim == feature_dim
    assert actual.channel == "train"


def test_file_system_record_set_fsx_customized_parameters():
    file_system_id = "fs-0a48d2a1"
    file_system_type = "FSxLustre"
    directory_path = "ipinsights"
    num_records = 1
    feature_dim = 1

    actual = FileSystemRecordSet(
        file_system_id=file_system_id,
        file_system_type=file_system_type,
        directory_path=directory_path,
        num_records=num_records,
        feature_dim=feature_dim,
        file_system_access_mode="rw",
        channel="test",
    )

    expected_input_config = {
        "DataSource": {
            "FileSystemDataSource": {
                "DirectoryPath": "ipinsights",
                "FileSystemId": "fs-0a48d2a1",
                "FileSystemType": "FSxLustre",
                "FileSystemAccessMode": "rw",
            }
        }
    }
    assert actual.file_system_input.config == expected_input_config
    assert actual.num_records == num_records
    assert actual.feature_dim == feature_dim
    assert actual.channel == "test"


def test_file_system_record_set_data_channel():
    file_system_id = "fs-0a48d2a1"
    file_system_type = "EFS"
    directory_path = "ipinsights"
    num_records = 1
    feature_dim = 1
    record_set = FileSystemRecordSet(
        file_system_id=file_system_id,
        file_system_type=file_system_type,
        directory_path=directory_path,
        num_records=num_records,
        feature_dim=feature_dim,
    )

    file_system_input = Mock()
    record_set.file_system_input = file_system_input
    actual = record_set.data_channel()
    expected = {"train": file_system_input}
    assert actual == expected
