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

from six.moves.urllib.parse import urlparse

from sagemaker.amazon.amazon_estimator import RecordSet
from sagemaker.utils import sagemaker_timestamp


def prepare_record_set_from_local_files(
    dir_path, destination, num_records, feature_dim, sagemaker_session
):
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
    key_prefix = key_prefix + "{}-{}".format("testfiles", sagemaker_timestamp())
    key_prefix = key_prefix.lstrip("/")
    uploaded_location = sagemaker_session.upload_data(path=dir_path, key_prefix=key_prefix)
    return RecordSet(uploaded_location, num_records, feature_dim, s3_data_type="S3Prefix")
