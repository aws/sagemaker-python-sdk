from six.moves.urllib.parse import urlparse

from sagemaker.amazon.amazon_estimator import RecordSet
from sagemaker.utils import sagemaker_timestamp


def prepare_record_set_from_local_files(dir_path, destination, num_records, feature_dim, sagemaker_session):
    """Build a :class:`~RecordSet` by pointing to local files.

    Args:
        dir_path (string): Path to local directory from where the files shall be uploaded.
        destination (string): S3 path to upload the file to.
        num_records (int): Number of records in all the files
        feature_dim (int): Number of features in the data set
        sagemaker_session (sagemaker.session.Session): Session object to manage interactions with Amazon SageMaker APIs.
    Returns:
        RecordSet: A RecordSet specified by S3Prefix to to be used in training.
    """
    key_prefix = urlparse(destination).path
    key_prefix = key_prefix + '{}-{}'.format("testfiles", sagemaker_timestamp())
    key_prefix = key_prefix.lstrip('/')
    uploaded_location = sagemaker_session.upload_data(path=dir_path, key_prefix=key_prefix)
    return RecordSet(uploaded_location, num_records, feature_dim, s3_data_type='S3Prefix')
