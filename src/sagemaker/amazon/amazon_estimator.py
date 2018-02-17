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
import json
import logging
import tempfile
from six.moves.urllib.parse import urlparse
from sagemaker.amazon import validation
from sagemaker.amazon.hyperparameter import Hyperparameter as hp  # noqa
from sagemaker.amazon.common import write_numpy_to_dense_tensor
from sagemaker.estimator import EstimatorBase
from sagemaker.session import s3_input
from sagemaker.utils import sagemaker_timestamp

logger = logging.getLogger(__name__)


class AmazonAlgorithmEstimatorBase(EstimatorBase):
    """Base class for Amazon first-party Estimator implementations. This class isn't intended
    to be instantiated directly."""

    feature_dim = hp('feature_dim', validation.gt(0), data_type=int)
    mini_batch_size = hp('mini_batch_size', validation.gt(0), data_type=int)

    def __init__(self, role, train_instance_count, train_instance_type, data_location=None, **kwargs):
        """Initialize an AmazonAlgorithmEstimatorBase.

        Args:
            data_location (str or None): The s3 prefix to upload RecordSet objects to, expressed as an
                S3 url. For example "s3://example-bucket/some-key-prefix/". Objects will be
                saved in a unique sub-directory of the specified location. If None, a default
                data location will be used."""
        super(AmazonAlgorithmEstimatorBase, self).__init__(role, train_instance_count, train_instance_type,
                                                           **kwargs)

        default_location = "s3://{}/sagemaker-record-sets/".format(self.sagemaker_session.default_bucket())
        data_location = data_location or default_location
        self.data_location = data_location

    def train_image(self):
        repo = '{}:{}'.format(type(self).repo_name, type(self).repo_version)
        return '{}/{}'.format(registry(self.sagemaker_session.boto_region_name, type(self).repo_name), repo)

    def hyperparameters(self):
        return hp.serialize_all(self)

    @property
    def data_location(self):
        return self._data_location

    @data_location.setter
    def data_location(self, data_location):
        if not data_location.startswith('s3://'):
            raise ValueError('Expecting an S3 URL beginning with "s3://". Got "{}"'.format(data_location))
        if data_location[-1] != '/':
            data_location = data_location + '/'
        self._data_location = data_location

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details):
        """Convert the job description to init params that can be handled by the class constructor

        Args:
            job_details: the returned job details from a describe_training_job API call.

        Returns:
             dictionary: The transformed init_params

        """
        init_params = super(AmazonAlgorithmEstimatorBase, cls)._prepare_init_params_from_job_description(job_details)

        # The hyperparam names may not be the same as the class attribute that holds them,
        # for instance: local_lloyd_init_method is called local_init_method. We need to map these
        # and pass the correct name to the constructor.
        for attribute, value in cls.__dict__.items():
            if isinstance(value, hp):
                if value.name in init_params['hyperparameters']:
                    init_params[attribute] = init_params['hyperparameters'][value.name]

        del init_params['hyperparameters']
        del init_params['image']
        return init_params

    def fit(self, records, mini_batch_size=None, **kwargs):
        """Fit this Estimator on serialized Record objects, stored in S3.

        ``records`` should be an instance of :class:`~RecordSet`. This defines a collection of
        s3 data files to train this ``Estimator`` on.

        Training data is expected to be encoded as dense or sparse vectors in the "values" feature
        on each Record. If the data is labeled, the label is expected to be encoded as a list of
        scalas in the "values" feature of the Record label.

        More information on the Amazon Record format is available at:
        https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html

        See :meth:`~AmazonAlgorithmEstimatorBase.record_set` to construct a ``RecordSet`` object
        from :class:`~numpy.ndarray` arrays.

        Args:
            records (:class:`~RecordSet`): The records to train this ``Estimator`` on
            mini_batch_size (int or None): The size of each mini-batch to use when training. If None, a
                default value will be used.
        """
        self.feature_dim = records.feature_dim
        self.mini_batch_size = mini_batch_size

        data = {records.channel: s3_input(records.s3_data, distribution='ShardedByS3Key',
                                          s3_data_type=records.s3_data_type)}
        super(AmazonAlgorithmEstimatorBase, self).fit(data, **kwargs)

    def record_set(self, train, labels=None, channel="train"):
        """Build a :class:`~RecordSet` from a numpy :class:`~ndarray` matrix and label vector.

        For the 2D ``ndarray`` ``train``, each row is converted to a :class:`~Record` object.
        The vector is stored in the "values" entry of the ``features`` property of each Record.
        If ``labels`` is not None, each corresponding label is assigned to the "values" entry
        of the ``labels`` property of each Record.

        The collection of ``Record`` objects are protobuf serialized and uploaded to new
        S3 locations. A manifest file is generated containing the list of objects created and
        also stored in S3.

        The number of S3 objects created is controlled by the ``train_instance_count`` property
        on this Estimator. One S3 object is created per training instance.

        Args:
            train (numpy.ndarray): A 2D numpy array of training data.
            labels (numpy.ndarray): A 1D numpy array of labels. Its length must be equal to the
               number of rows in ``train``.
            channel (str): The SageMaker TrainingJob channel this RecordSet should be assigned to.
        Returns:
            RecordSet: A RecordSet referencing the encoded, uploading training and label data.
        """
        s3 = self.sagemaker_session.boto_session.resource('s3')
        parsed_s3_url = urlparse(self.data_location)
        bucket, key_prefix = parsed_s3_url.netloc, parsed_s3_url.path
        key_prefix = key_prefix + '{}-{}/'.format(type(self).__name__, sagemaker_timestamp())
        key_prefix = key_prefix.lstrip('/')
        logger.debug('Uploading to bucket {} and key_prefix {}'.format(bucket, key_prefix))
        manifest_s3_file = upload_numpy_to_s3_shards(self.train_instance_count, s3, bucket, key_prefix, train, labels)
        logger.debug("Created manifest file {}".format(manifest_s3_file))
        return RecordSet(manifest_s3_file, num_records=train.shape[0], feature_dim=train.shape[1], channel=channel)


class RecordSet(object):

    def __init__(self, s3_data, num_records, feature_dim, s3_data_type='ManifestFile', channel='train'):
        """A collection of Amazon :class:~`Record` objects serialized and stored in S3.

        Args:
            s3_data (str): The S3 location of the training data
            num_records (int): The number of records in the set.
            feature_dim (int): The dimensionality of "values" arrays in the Record features,
                and label (if each Record is labeled).
            s3_data_type (str): Valid values: 'S3Prefix', 'ManifestFile'. If 'S3Prefix', ``s3_data`` defines
                a prefix of s3 objects to train on. All objects with s3 keys beginning with ``s3_data`` will
                be used to train. If 'ManifestFile', then ``s3_data`` defines a single s3 manifest file, listing
                each s3 object to train on.
            channel (str): The SageMaker Training Job channel this RecordSet should be bound to
        """
        self.s3_data = s3_data
        self.feature_dim = feature_dim
        self.num_records = num_records
        self.s3_data_type = s3_data_type
        self.channel = channel

    def __repr__(self):
        """Return an unambiguous representation of this RecordSet"""
        return str((RecordSet, self.__dict__))


def _build_shards(num_shards, array):
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    shard_size = int(array.shape[0] / num_shards)
    if shard_size == 0:
        raise ValueError("Array length is less than num shards")
    shards = [array[i * shard_size:i * shard_size + shard_size] for i in range(num_shards - 1)]
    shards.append(array[(num_shards - 1) * shard_size:])
    return shards


def upload_numpy_to_s3_shards(num_shards, s3, bucket, key_prefix, array, labels=None):
    """Upload the training ``array`` and ``labels`` arrays to ``num_shards`` s3 objects,
    stored in "s3://``bucket``/``key_prefix``/"."""
    shards = _build_shards(num_shards, array)
    if labels is not None:
        label_shards = _build_shards(num_shards, labels)
    uploaded_files = []
    if key_prefix[-1] != '/':
        key_prefix = key_prefix + '/'
    try:
        for shard_index, shard in enumerate(shards):
            with tempfile.TemporaryFile() as file:
                if labels is not None:
                    write_numpy_to_dense_tensor(file, shard, label_shards[shard_index])
                else:
                    write_numpy_to_dense_tensor(file, shard)
                file.seek(0)
                shard_index_string = str(shard_index).zfill(len(str(len(shards))))
                file_name = "matrix_{}.pbr".format(shard_index_string)
                key = key_prefix + file_name
                logger.debug("Creating object {} in bucket {}".format(key, bucket))
                s3.Object(bucket, key).put(Body=file)
                uploaded_files.append(file_name)
        manifest_key = key_prefix + ".amazon.manifest"
        manifest_str = json.dumps(
            [{'prefix': 's3://{}/{}'.format(bucket, key_prefix)}] + uploaded_files)
        s3.Object(bucket, manifest_key).put(Body=manifest_str.encode('utf-8'))
        return "s3://{}/{}".format(bucket, manifest_key)
    except Exception as ex:
        try:
            for file in uploaded_files:
                s3.Object(bucket, key_prefix + file).delete()
        finally:
            raise ex


def registry(region_name, algorithm=None):
    """Return docker registry for the given AWS region"""
    if algorithm in [None, "pca", "kmeans", "linear-learner", "factorization-machines", "ntm"]:
        account_id = {
            "us-east-1": "382416733822",
            "us-east-2": "404615174143",
            "us-west-2": "174872318107",
            "eu-west-1": "438346466558"
        }[region_name]
    elif algorithm in ["lda"]:
        account_id = {
            "us-east-1": "766337827248",
            "us-east-2": "999911452149",
            "us-west-2": "266724342769",
            "eu-west-1": "999678624901"
        }[region_name]
    else:
        raise ValueError("Algorithm class:{} doesn't have mapping to account_id with images".format(algorithm))
    return "{}.dkr.ecr.{}.amazonaws.com".format(account_id, region_name)
