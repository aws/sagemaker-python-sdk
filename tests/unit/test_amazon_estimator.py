# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import pytest
from mock import Mock, patch, call
import numpy as np

# Use PCA as a test implementation of AmazonAlgorithmEstimator
from sagemaker.amazon.pca import PCA
from sagemaker.amazon.amazon_estimator import upload_numpy_to_s3_shards, _build_shards, registry

COMMON_ARGS = {'role': 'myrole', 'train_instance_count': 1, 'train_instance_type': 'ml.c4.xlarge'}

REGION = "us-west-2"
BUCKET_NAME = "Some-Bucket"
TIMESTAMP = '2017-11-06-14:14:15.671'


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name='boto_session', region_name=REGION)
    sms = Mock(name='sagemaker_session', boto_session=boto_mock)
    sms.boto_region_name = REGION
    sms.default_bucket = Mock(name='default_bucket', return_value=BUCKET_NAME)
    returned_job_description = {'AlgorithmSpecification': {'TrainingInputMode': 'File',
                                                           'TrainingImage': registry("us-west-2") + "/pca:1"},
                                'ModelArtifacts': {'S3ModelArtifacts': "s3://some-bucket/model.tar.gz"},
                                'HyperParameters':
                                    {'sagemaker_submit_directory': '"s3://some/sourcedir.tar.gz"',
                                     'checkpoint_path': '"s3://other/1508872349"',
                                     'sagemaker_program': '"iris-dnn-classifier.py"',
                                     'sagemaker_enable_cloudwatch_metrics': 'false',
                                     'sagemaker_container_log_level': '"logging.INFO"',
                                     'sagemaker_job_name': '"neo"',
                                     'training_steps': '100'},
                                'RoleArn': 'arn:aws:iam::366:role/IMRole',
                                'ResourceConfig':
                                    {'VolumeSizeInGB': 30,
                                     'InstanceCount': 1,
                                     'InstanceType': 'ml.c4.xlarge'},
                                'StoppingCondition': {'MaxRuntimeInSeconds': 24 * 60 * 60},
                                'TrainingJobName': 'neo',
                                'TrainingJobStatus': 'Completed',
                                'OutputDataConfig': {'KmsKeyId': '',
                                                     'S3OutputPath': 's3://place/output/neo'},
                                'TrainingJobOutput': {'S3TrainingJobOutput': 's3://here/output.tar.gz'}}
    sms.sagemaker_client.describe_training_job = Mock(name='describe_training_job',
                                                      return_value=returned_job_description)
    return sms


def test_init(sagemaker_session):
    pca = PCA(num_components=55, sagemaker_session=sagemaker_session, **COMMON_ARGS)
    assert pca.num_components == 55


def test_init_all_pca_hyperparameters(sagemaker_session):
    pca = PCA(num_components=55, algorithm_mode='randomized',
              subtract_mean=True, extra_components=33, sagemaker_session=sagemaker_session,
              **COMMON_ARGS)
    assert pca.num_components == 55
    assert pca.algorithm_mode == 'randomized'
    assert pca.extra_components == 33


def test_init_estimator_args(sagemaker_session):
    pca = PCA(num_components=1, train_max_run=1234, sagemaker_session=sagemaker_session,
              data_location='s3://some-bucket/some-key/', **COMMON_ARGS)
    assert pca.train_instance_type == COMMON_ARGS['train_instance_type']
    assert pca.train_instance_count == COMMON_ARGS['train_instance_count']
    assert pca.role == COMMON_ARGS['role']
    assert pca.train_max_run == 1234
    assert pca.data_location == 's3://some-bucket/some-key/'


def test_data_location_validation(sagemaker_session):
    pca = PCA(num_components=2, sagemaker_session=sagemaker_session, **COMMON_ARGS)
    with pytest.raises(ValueError):
        pca.data_location = "nots3://abcd/efgh"


def test_pca_hyperparameters(sagemaker_session):
    pca = PCA(num_components=55, algorithm_mode='randomized',
              subtract_mean=True, extra_components=33, sagemaker_session=sagemaker_session,
              **COMMON_ARGS)
    assert pca.hyperparameters() == dict(
        num_components='55',
        extra_components='33',
        subtract_mean='True',
        algorithm_mode='randomized')


def test_image(sagemaker_session):
    pca = PCA(num_components=55, sagemaker_session=sagemaker_session, **COMMON_ARGS)
    assert pca.train_image() == registry('us-west-2') + '/pca:1'


@patch('time.strftime', return_value=TIMESTAMP)
def test_fit_ndarray(time, sagemaker_session):
    mock_s3 = Mock()
    mock_object = Mock()
    mock_s3.Object = Mock(return_value=mock_object)
    sagemaker_session.boto_session.resource = Mock(return_value=mock_s3)
    kwargs = dict(COMMON_ARGS)
    kwargs['train_instance_count'] = 3
    pca = PCA(num_components=55, sagemaker_session=sagemaker_session,
              data_location='s3://{}/key-prefix/'.format(BUCKET_NAME), **kwargs)
    train = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 8.0], [44.0, 55.0, 66.0]]
    labels = [99, 85, 87, 2]
    pca.fit(pca.record_set(np.array(train), np.array(labels)))
    mock_s3.Object.assert_any_call(
        BUCKET_NAME, 'key-prefix/PCA-2017-11-06-14:14:15.671/matrix_0.pbr'.format(TIMESTAMP))
    mock_s3.Object.assert_any_call(
        BUCKET_NAME, 'key-prefix/PCA-2017-11-06-14:14:15.671/matrix_1.pbr'.format(TIMESTAMP))
    mock_s3.Object.assert_any_call(
        BUCKET_NAME, 'key-prefix/PCA-2017-11-06-14:14:15.671/matrix_2.pbr'.format(TIMESTAMP))
    mock_s3.Object.assert_any_call(
        BUCKET_NAME, 'key-prefix/PCA-2017-11-06-14:14:15.671/.amazon.manifest'.format(TIMESTAMP))

    assert mock_object.put.call_count == 4


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
    array = np.array([[j for j in range(10)] for i in range(10)])
    labels = np.array([i for i in range(10)])
    upload_numpy_to_s3_shards(3, mock_s3, BUCKET_NAME, "key-prefix", array, labels)
    mock_s3.Object.assert_has_calls([call(BUCKET_NAME, 'key-prefix/matrix_0.pbr')])
    mock_s3.Object.assert_has_calls([call(BUCKET_NAME, 'key-prefix/matrix_1.pbr')])
    mock_s3.Object.assert_has_calls([call(BUCKET_NAME, 'key-prefix/matrix_2.pbr')])
