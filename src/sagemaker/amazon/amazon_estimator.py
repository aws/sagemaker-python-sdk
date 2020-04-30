# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Placeholder docstring"""
from __future__ import absolute_import

import json
import logging
import tempfile

from six.moves.urllib.parse import urlparse

from sagemaker.amazon import validation
from sagemaker.amazon.hyperparameter import Hyperparameter as hp  # noqa
from sagemaker.amazon.common import write_numpy_to_dense_tensor
from sagemaker.estimator import EstimatorBase, _TrainingJob
from sagemaker.inputs import FileSystemInput
from sagemaker.model import NEO_IMAGE_ACCOUNT
from sagemaker.session import s3_input
from sagemaker.utils import sagemaker_timestamp, get_ecr_image_uri_prefix
from sagemaker.xgboost.defaults import (
    XGBOOST_LATEST_VERSION,
    XGBOOST_SUPPORTED_VERSIONS,
    XGBOOST_VERSION_1,
    XGBOOST_VERSION_EQUIVALENTS,
)
from sagemaker.xgboost.estimator import get_xgboost_image_uri

logger = logging.getLogger(__name__)


class AmazonAlgorithmEstimatorBase(EstimatorBase):
    """Base class for Amazon first-party Estimator implementations. This class
    isn't intended to be instantiated directly.
    """

    feature_dim = hp("feature_dim", validation.gt(0), data_type=int)
    mini_batch_size = hp("mini_batch_size", validation.gt(0), data_type=int)
    repo_name = None
    repo_version = None

    def __init__(
        self,
        role,
        train_instance_count,
        train_instance_type,
        data_location=None,
        enable_network_isolation=False,
        **kwargs
    ):
        """Initialize an AmazonAlgorithmEstimatorBase.

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role, if it needs to access an AWS resource.
            train_instance_count (int): Number of Amazon EC2 instances to use
                for training.
            train_instance_type (str): Type of EC2 instance to use for training,
                for example, 'ml.c4.xlarge'.
            data_location (str or None): The s3 prefix to upload RecordSet
                objects to, expressed as an S3 url. For example
                "s3://example-bucket/some-key-prefix/". Objects will be saved in
                a unique sub-directory of the specified location. If None, a
                default data location will be used.
            enable_network_isolation (bool): Specifies whether container will
                run in network isolation mode. Network isolation mode restricts
                the container access to outside networks (such as the internet).
                Also known as internet-free mode (default: ``False``).
            **kwargs: Additional parameters passed to
                :class:`~sagemaker.estimator.EstimatorBase`.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.estimator.EstimatorBase`.
        """
        super(AmazonAlgorithmEstimatorBase, self).__init__(
            role,
            train_instance_count,
            train_instance_type,
            enable_network_isolation=enable_network_isolation,
            **kwargs
        )

        data_location = data_location or "s3://{}/sagemaker-record-sets/".format(
            self.sagemaker_session.default_bucket()
        )
        self._data_location = data_location

    def train_image(self):
        """Placeholder docstring"""
        return get_image_uri(
            self.sagemaker_session.boto_region_name, type(self).repo_name, type(self).repo_version
        )

    def hyperparameters(self):
        """Placeholder docstring"""
        return hp.serialize_all(self)

    @property
    def data_location(self):
        """Placeholder docstring"""
        return self._data_location

    @data_location.setter
    def data_location(self, data_location):
        """
        Args:
            data_location:
        """
        if not data_location.startswith("s3://"):
            raise ValueError(
                'Expecting an S3 URL beginning with "s3://". Got "{}"'.format(data_location)
            )
        if data_location[-1] != "/":
            data_location = data_location + "/"
        self._data_location = data_location

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        """Convert the job description to init params that can be handled by the
        class constructor

        Args:
            job_details: the returned job details from a describe_training_job
                API call.
            model_channel_name (str): Name of the channel where pre-trained
                model data will be downloaded.

        Returns:
            dictionary: The transformed init_params
        """
        init_params = super(
            AmazonAlgorithmEstimatorBase, cls
        )._prepare_init_params_from_job_description(job_details, model_channel_name)

        # The hyperparam names may not be the same as the class attribute that holds them,
        # for instance: local_lloyd_init_method is called local_init_method. We need to map these
        # and pass the correct name to the constructor.
        for attribute, value in cls.__dict__.items():
            if isinstance(value, hp):
                if value.name in init_params["hyperparameters"]:
                    init_params[attribute] = init_params["hyperparameters"][value.name]

        del init_params["hyperparameters"]
        del init_params["image"]
        return init_params

    def prepare_workflow_for_training(self, records=None, mini_batch_size=None, job_name=None):
        """Calls _prepare_for_training. Used when setting up a workflow.

        Args:
            records (:class:`~RecordSet`): The records to train this ``Estimator`` on.
            mini_batch_size (int or None): The size of each mini-batch to use when
                training. If ``None``, a default value will be used.
            job_name (str): Name of the training job to be created. If not
                specified, one is generated, using the base name given to the
                constructor if applicable.
        """
        self._prepare_for_training(
            records=records, mini_batch_size=mini_batch_size, job_name=job_name
        )

    def _prepare_for_training(self, records, mini_batch_size=None, job_name=None):
        """Set hyperparameters needed for training.

        Args:
            records (:class:`~RecordSet`): The records to train this ``Estimator`` on.
            mini_batch_size (int or None): The size of each mini-batch to use when
                training. If ``None``, a default value will be used.
            job_name (str): Name of the training job to be created. If not
                specified, one is generated, using the base name given to the
                constructor if applicable.
        """
        super(AmazonAlgorithmEstimatorBase, self)._prepare_for_training(job_name=job_name)

        feature_dim = None

        if isinstance(records, list):
            for record in records:
                if record.channel == "train":
                    feature_dim = record.feature_dim
                    break
            if feature_dim is None:
                raise ValueError("Must provide train channel.")
        else:
            feature_dim = records.feature_dim

        self.feature_dim = feature_dim
        self.mini_batch_size = mini_batch_size

    def fit(
        self,
        records,
        mini_batch_size=None,
        wait=True,
        logs=True,
        job_name=None,
        experiment_config=None,
    ):
        """Fit this Estimator on serialized Record objects, stored in S3.

        ``records`` should be an instance of :class:`~RecordSet`. This
        defines a collection of S3 data files to train this ``Estimator`` on.

        Training data is expected to be encoded as dense or sparse vectors in
        the "values" feature on each Record. If the data is labeled, the label
        is expected to be encoded as a list of scalas in the "values" feature of
        the Record label.

        More information on the Amazon Record format is available at:
        https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html

        See :meth:`~AmazonAlgorithmEstimatorBase.record_set` to construct a
        ``RecordSet`` object from :class:`~numpy.ndarray` arrays.

        Args:
            records (:class:`~RecordSet`): The records to train this ``Estimator`` on
            mini_batch_size (int or None): The size of each mini-batch to use
                when training. If ``None``, a default value will be used.
            wait (bool): Whether the call should wait until the job completes
                (default: True).
            logs (bool): Whether to show the logs produced by the job. Only
                meaningful when wait is True (default: True).
            job_name (str): Training job name. If not specified, the estimator
                generates a default job name, based on the training image name
                and current timestamp.
            experiment_config (dict[str, str]): Experiment management configuration.
                Dictionary contains three optional keys, 'ExperimentName',
                'TrialName', and 'TrialComponentName'
                (default: ``None``).
        """
        self._prepare_for_training(records, job_name=job_name, mini_batch_size=mini_batch_size)

        self.latest_training_job = _TrainingJob.start_new(
            self, records, experiment_config=experiment_config
        )
        if wait:
            self.latest_training_job.wait(logs=logs)

    def record_set(self, train, labels=None, channel="train", encrypt=False):
        """Build a :class:`~RecordSet` from a numpy :class:`~ndarray` matrix and
        label vector.

        For the 2D ``ndarray`` ``train``, each row is converted to a
        :class:`~Record` object. The vector is stored in the "values" entry of
        the ``features`` property of each Record. If ``labels`` is not None,
        each corresponding label is assigned to the "values" entry of the
        ``labels`` property of each Record.

        The collection of ``Record`` objects are protobuf serialized and
        uploaded to new S3 locations. A manifest file is generated containing
        the list of objects created and also stored in S3.

        The number of S3 objects created is controlled by the
        ``train_instance_count`` property on this Estimator. One S3 object is
        created per training instance.

        Args:
            train (numpy.ndarray): A 2D numpy array of training data.
            labels (numpy.ndarray): A 1D numpy array of labels. Its length must
                be equal to the number of rows in ``train``.
            channel (str): The SageMaker TrainingJob channel this RecordSet
                should be assigned to.
            encrypt (bool): Specifies whether the objects uploaded to S3 are
                encrypted on the server side using AES-256 (default: ``False``).

        Returns:
            RecordSet: A RecordSet referencing the encoded, uploading training
            and label data.
        """
        s3 = self.sagemaker_session.boto_session.resource(
            "s3", region_name=self.sagemaker_session.boto_region_name
        )
        parsed_s3_url = urlparse(self.data_location)
        bucket, key_prefix = parsed_s3_url.netloc, parsed_s3_url.path
        key_prefix = key_prefix + "{}-{}/".format(type(self).__name__, sagemaker_timestamp())
        key_prefix = key_prefix.lstrip("/")
        logger.debug("Uploading to bucket %s and key_prefix %s", bucket, key_prefix)
        manifest_s3_file = upload_numpy_to_s3_shards(
            self.train_instance_count, s3, bucket, key_prefix, train, labels, encrypt
        )
        logger.debug("Created manifest file %s", manifest_s3_file)
        return RecordSet(
            manifest_s3_file,
            num_records=train.shape[0],
            feature_dim=train.shape[1],
            channel=channel,
        )


class RecordSet(object):
    """Placeholder docstring"""

    def __init__(
        self, s3_data, num_records, feature_dim, s3_data_type="ManifestFile", channel="train"
    ):
        """A collection of Amazon :class:~`Record` objects serialized and stored
        in S3.

        Args:
            s3_data (str): The S3 location of the training data
            num_records (int): The number of records in the set.
            feature_dim (int): The dimensionality of "values" arrays in the
                Record features, and label (if each Record is labeled).
            s3_data_type (str): Valid values: 'S3Prefix', 'ManifestFile'. If
                'S3Prefix', ``s3_data`` defines a prefix of s3 objects to train
                on. All objects with s3 keys beginning with ``s3_data`` will be
                used to train. If 'ManifestFile', then ``s3_data`` defines a
                single s3 manifest file, listing each s3 object to train on.
            channel (str): The SageMaker Training Job channel this RecordSet
                should be bound to
        """
        self.s3_data = s3_data
        self.feature_dim = feature_dim
        self.num_records = num_records
        self.s3_data_type = s3_data_type
        self.channel = channel

    def __repr__(self):
        """Return an unambiguous representation of this RecordSet"""
        return str((RecordSet, self.__dict__))

    def data_channel(self):
        """Return a dictionary to represent the training data in a channel for
        use with ``fit()``
        """
        return {self.channel: self.records_s3_input()}

    def records_s3_input(self):
        """Return a s3_input to represent the training data"""
        return s3_input(self.s3_data, distribution="ShardedByS3Key", s3_data_type=self.s3_data_type)


class FileSystemRecordSet(object):
    """Amazon SageMaker channel configuration for a file system data source
    for Amazon algorithms.
    """

    def __init__(
        self,
        file_system_id,
        file_system_type,
        directory_path,
        num_records,
        feature_dim,
        file_system_access_mode="ro",
        channel="train",
    ):
        """Initialize a ``FileSystemRecordSet`` object.

        Args:
            file_system_id (str): An Amazon file system ID starting with 'fs-'.
            file_system_type (str): The type of file system used for the input.
                Valid values: 'EFS', 'FSxLustre'.
            directory_path (str): Absolute or normalized path to the root directory (mount point) in
                the file system. Reference:
                https://docs.aws.amazon.com/efs/latest/ug/mounting-fs.html and
                https://docs.aws.amazon.com/efs/latest/ug/wt1-test.html
            num_records (int): The number of records in the set.
            feature_dim (int): The dimensionality of "values" arrays in the Record features,
                and label (if each Record is labeled).
            file_system_access_mode (str): Permissions for read and write.
                Valid values: 'ro' or 'rw'. Defaults to 'ro'.
            channel (str): The SageMaker Training Job channel this RecordSet should be bound to
        """

        self.file_system_input = FileSystemInput(
            file_system_id, file_system_type, directory_path, file_system_access_mode
        )
        self.feature_dim = feature_dim
        self.num_records = num_records
        self.channel = channel

    def __repr__(self):
        """Return an unambiguous representation of this RecordSet"""
        return str((FileSystemRecordSet, self.__dict__))

    def data_channel(self):
        """Return a dictionary to represent the training data in a channel for use with ``fit()``"""
        return {self.channel: self.file_system_input}


def _build_shards(num_shards, array):
    """
    Args:
        num_shards:
        array:
    """
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    shard_size = int(array.shape[0] / num_shards)
    if shard_size == 0:
        raise ValueError("Array length is less than num shards")
    shards = [array[i * shard_size : i * shard_size + shard_size] for i in range(num_shards - 1)]
    shards.append(array[(num_shards - 1) * shard_size :])
    return shards


def upload_numpy_to_s3_shards(
    num_shards, s3, bucket, key_prefix, array, labels=None, encrypt=False
):
    """Upload the training ``array`` and ``labels`` arrays to ``num_shards`` S3
    objects, stored in "s3:// ``bucket`` / ``key_prefix`` /". Optionally
    ``encrypt`` the S3 objects using AES-256.

    Args:
        num_shards:
        s3:
        bucket:
        key_prefix:
        array:
        labels:
        encrypt:
    """
    shards = _build_shards(num_shards, array)
    if labels is not None:
        label_shards = _build_shards(num_shards, labels)
    uploaded_files = []
    if key_prefix[-1] != "/":
        key_prefix = key_prefix + "/"
    extra_put_kwargs = {"ServerSideEncryption": "AES256"} if encrypt else {}
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
                logger.debug("Creating object %s in bucket %s", key, bucket)
                s3.Object(bucket, key).put(Body=file, **extra_put_kwargs)
                uploaded_files.append(file_name)
        manifest_key = key_prefix + ".amazon.manifest"
        manifest_str = json.dumps(
            [{"prefix": "s3://{}/{}".format(bucket, key_prefix)}] + uploaded_files
        )
        s3.Object(bucket, manifest_key).put(Body=manifest_str.encode("utf-8"), **extra_put_kwargs)
        return "s3://{}/{}".format(bucket, manifest_key)
    except Exception as ex:  # pylint: disable=broad-except
        try:
            for file in uploaded_files:
                s3.Object(bucket, key_prefix + file).delete()
        finally:
            raise ex


def registry(region_name, algorithm=None):
    """Return docker registry for the given AWS region

    Note: Not all the algorithms listed below have an Amazon Estimator
    implemented. For full list of pre-implemented Estimators, look at:

    https://github.com/aws/sagemaker-python-sdk/tree/master/src/sagemaker/amazon

    Args:
        region_name (str): The region name for the account.
        algorithm (str): The algorithm for the account.

    Raises:
        ValueError: If invalid algorithm passed in or if mapping does not exist for given algorithm
            and region.
    """
    region_to_accounts = {}
    if algorithm in [
        None,
        "pca",
        "kmeans",
        "linear-learner",
        "factorization-machines",
        "ntm",
        "randomcutforest",
        "knn",
        "object2vec",
        "ipinsights",
    ]:
        region_to_accounts = {
            "us-east-1": "382416733822",
            "us-east-2": "404615174143",
            "us-west-2": "174872318107",
            "eu-west-1": "438346466558",
            "eu-central-1": "664544806723",
            "ap-northeast-1": "351501993468",
            "ap-northeast-2": "835164637446",
            "ap-southeast-2": "712309505854",
            "us-gov-west-1": "226302683700",
            "ap-southeast-1": "475088953585",
            "ap-south-1": "991648021394",
            "ca-central-1": "469771592824",
            "eu-west-2": "644912444149",
            "us-west-1": "632365934929",
            "us-iso-east-1": "490574956308",
            "ap-east-1": "286214385809",
            "eu-north-1": "669576153137",
            "eu-west-3": "749696950732",
            "sa-east-1": "855470959533",
            "me-south-1": "249704162688",
            "cn-north-1": "390948362332",
            "cn-northwest-1": "387376663083",
        }
    elif algorithm in ["lda"]:
        region_to_accounts = {
            "us-east-1": "766337827248",
            "us-east-2": "999911452149",
            "us-west-2": "266724342769",
            "eu-west-1": "999678624901",
            "eu-central-1": "353608530281",
            "ap-northeast-1": "258307448986",
            "ap-northeast-2": "293181348795",
            "ap-southeast-2": "297031611018",
            "us-gov-west-1": "226302683700",
            "ap-southeast-1": "475088953585",
            "ap-south-1": "991648021394",
            "ca-central-1": "469771592824",
            "eu-west-2": "644912444149",
            "us-west-1": "632365934929",
            "us-iso-east-1": "490574956308",
        }
    elif algorithm in ["forecasting-deepar"]:
        region_to_accounts = {
            "us-east-1": "522234722520",
            "us-east-2": "566113047672",
            "us-west-2": "156387875391",
            "eu-west-1": "224300973850",
            "eu-central-1": "495149712605",
            "ap-northeast-1": "633353088612",
            "ap-northeast-2": "204372634319",
            "ap-southeast-2": "514117268639",
            "us-gov-west-1": "226302683700",
            "ap-southeast-1": "475088953585",
            "ap-south-1": "991648021394",
            "ca-central-1": "469771592824",
            "eu-west-2": "644912444149",
            "us-west-1": "632365934929",
            "us-iso-east-1": "490574956308",
            "ap-east-1": "286214385809",
            "eu-north-1": "669576153137",
            "eu-west-3": "749696950732",
            "sa-east-1": "855470959533",
            "me-south-1": "249704162688",
            "cn-north-1": "390948362332",
            "cn-northwest-1": "387376663083",
        }
    elif algorithm in [
        "xgboost",
        "seq2seq",
        "image-classification",
        "blazingtext",
        "object-detection",
        "semantic-segmentation",
    ]:
        region_to_accounts = {
            "us-east-1": "811284229777",
            "us-east-2": "825641698319",
            "us-west-2": "433757028032",
            "eu-west-1": "685385470294",
            "eu-central-1": "813361260812",
            "ap-northeast-1": "501404015308",
            "ap-northeast-2": "306986355934",
            "ap-southeast-2": "544295431143",
            "us-gov-west-1": "226302683700",
            "ap-southeast-1": "475088953585",
            "ap-south-1": "991648021394",
            "ca-central-1": "469771592824",
            "eu-west-2": "644912444149",
            "us-west-1": "632365934929",
            "us-iso-east-1": "490574956308",
            "ap-east-1": "286214385809",
            "eu-north-1": "669576153137",
            "eu-west-3": "749696950732",
            "sa-east-1": "855470959533",
            "me-south-1": "249704162688",
            "cn-north-1": "390948362332",
            "cn-northwest-1": "387376663083",
        }
    elif algorithm in ["image-classification-neo", "xgboost-neo"]:
        region_to_accounts = NEO_IMAGE_ACCOUNT
    else:
        raise ValueError(
            "Algorithm class:{} does not have mapping to account_id with images".format(algorithm)
        )

    if region_name in region_to_accounts:
        account_id = region_to_accounts[region_name]
        return get_ecr_image_uri_prefix(account_id, region_name)

    raise ValueError(
        "Algorithm ({algorithm}) is unsupported for region ({region_name}).".format(
            algorithm=algorithm, region_name=region_name
        )
    )


def get_image_uri(region_name, repo_name, repo_version=1):
    """Return algorithm image URI for the given AWS region, repository name, and
    repository version

    Args:
        region_name:
        repo_name:
        repo_version:
    """
    if repo_name == "xgboost":
        if not _is_latest_xgboost_version(repo_version):
            logging.warning(
                "There is a more up to date SageMaker XGBoost image. "
                "To use the newer image, please set 'repo_version'="
                "'%s'. For example:\n"
                "\tget_image_uri(region, 'xgboost', '%s').",
                XGBOOST_LATEST_VERSION,
                XGBOOST_LATEST_VERSION,
            )

        if repo_version in ["0.90", "0.90-1", "0.90-1-cpu-py3"]:
            return get_xgboost_image_uri(region_name, XGBOOST_VERSION_1)

        supported_version = [
            version
            for version in XGBOOST_SUPPORTED_VERSIONS
            if repo_version in _generate_version_equivalents(version)
        ]
        if supported_version:
            return get_xgboost_image_uri(region_name, supported_version[0])

    repo = "{}:{}".format(repo_name, repo_version)
    return "{}/{}".format(registry(region_name, repo_name), repo)


def _is_latest_xgboost_version(repo_version):
    """Compare xgboost image version with latest version

    Args:
        repo_version:
    """
    if repo_version in (1, "latest"):
        return False
    return repo_version in _generate_version_equivalents(XGBOOST_LATEST_VERSION)


def _generate_version_equivalents(version):
    """Returns a list of version equivalents for XGBoost

    Args:
        version:
    """
    return [version + suffix for suffix in XGBOOST_VERSION_EQUIVALENTS] + [version]
