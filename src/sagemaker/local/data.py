# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import sys
import tempfile
from six.moves.urllib.parse import urlparse

from sagemaker.amazon.common import read_recordio
from sagemaker.local.utils import download_folder
from sagemaker.utils import get_config_value


class DataSourceFactory(object):

    @staticmethod
    def get_instance(data_source, sagemaker_session):
        parsed_uri = urlparse(data_source)
        if parsed_uri.scheme == 'file':
            return LocalFileDataSource(parsed_uri.path)
        elif parsed_uri.scheme == 's3':
            return S3DataSource(parsed_uri.netloc, parsed_uri.path, sagemaker_session)


class DataSource(object):

    def get_file_list(self):
        pass

    def get_root_dir(self):
        pass


class LocalFileDataSource(DataSource):
    """
    Represents a data source within the local filesystem.
    """

    def __init__(self, root_path):
        self.root_path = os.path.abspath(root_path)
        if not os.path.exists(self.root_path):
            raise RuntimeError('Invalid data source: %s Does not exist.' % self.root_path)

    def get_file_list(self):
        """Retrieve the list of absolute paths to all the files in this data source.

        Returns:
             List[string] List of absolute paths.
        """
        if os.path.isdir(self.root_path):
            files = [os.path.join(self.root_path, f) for f in os.listdir(self.root_path)
                     if os.path.isfile(os.path.join(self.root_path, f))]
        else:
            files = [self.root_path]

        return files

    def get_root_dir(self):
        """Retrieve the absolute path to the root directory of this data source.

        Returns:
            string: absolute path to the root directory of this data source.
        """
        if os.path.isdir(self.root_path):
            return self.root_path
        else:
            return os.path.dirname(self.root_path)


class S3DataSource(DataSource):
    """Defines a data source given by a bucket and s3 prefix. The contents will be downloaded
    and then processed as local data.
    """

    def __init__(self, bucket, prefix, sagemaker_session):
        """Create an S3DataSource instance

        Args:
            bucket (str): s3 bucket name
            prefix (str): s3 prefix path to the data
            sagemaker_session (sagemaker.Session): a sagemaker_session with the desired settings to talk to s3

        """

        # Create a temporary dir to store the S3 contents
        root_dir = get_config_value('local.container_root', sagemaker_session.config)
        if root_dir:
            root_dir = os.path.abspath(root_dir)

        working_dir = tempfile.mkdtemp(dir=root_dir)
        download_folder(bucket, prefix, working_dir, sagemaker_session)
        self.files = LocalFileDataSource(working_dir)

    def get_file_list(self):
        return self.files.get_file_list()

    def get_root_dir(self):
        return self.files.get_root_dir()


class SplitterFactory(object):

    @staticmethod
    def get_instance(split_type):
        if split_type is None:
            return NoneSplitter()
        elif split_type == 'Line':
            return LineSplitter()
        elif split_type == 'RecordIO':
            return RecordIOSplitter()
        else:
            raise ValueError('Invalid Split Type: %s' % split_type)


class Splitter(object):

    def split(self, file):
        pass


class NoneSplitter(Splitter):
    """Does not split records, essentially reads the whole file.
    """

    def split(self, file):
        with open(file, 'r') as f:
            yield f.read()


class LineSplitter(Splitter):
    """Split records by new line.

    """

    def split(self, file):
        with open(file, 'r') as f:
            for line in f:
                yield line


class RecordIOSplitter(Splitter):
    """Split using Amazon Recordio.

    Not useful for string content.

    """
    def split(self, file):
        with open(file, 'rb') as f:
            for record in read_recordio(f):
                yield record


class BatchStrategyFactory(object):

    @staticmethod
    def get_instance(strategy, splitter):
        if strategy == 'SingleRecord':
            return SingleRecordStrategy(splitter)
        elif strategy == 'MultiRecord':
            return MultiRecordStrategy(splitter)
        else:
            raise ValueError('Invalid Batch Strategy: %s - Valid Strategies: "SingleRecord", "MultiRecord"')


class BatchStrategy(object):

    def pad(self, file, size):
        pass


class MultiRecordStrategy(BatchStrategy):
    """Feed multiple records at a time for batch inference.

    Will group up as many records as possible within the payload specified.

    """
    def __init__(self, splitter):
        self.splitter = splitter

    def pad(self, file, size=6):
        buffer = ''
        for element in self.splitter.split(file):
            if _payload_size_within_limit(buffer + element, size):
                buffer += element
            else:
                tmp = buffer
                buffer = element
                yield tmp
        if _validate_payload_size(buffer, size):
            yield buffer


class SingleRecordStrategy(BatchStrategy):
    """Feed a single record at a time for batch inference.

    If a single record does not fit within the payload specified it will throw a Runtime error.
    """
    def __init__(self, splitter):
        self.splitter = splitter

    def pad(self, file, size=6):
        for element in self.splitter.split(file):
            if _validate_payload_size(element, size):
                yield element


def _payload_size_within_limit(payload, size):
    """

    Args:
        payload:
        size:

    Returns:

    """
    size_in_bytes = size * 1024 * 1024
    if size == 0:
        return True
    else:
        return sys.getsizeof(payload) < size_in_bytes


def _validate_payload_size(payload, size):
    """Check if a payload is within the size in MB threshold. Raise an exception otherwise.

    Args:
        payload: data that will be checked
        size (int): max size in MB

    Returns (bool): True if within bounds. if size=0 it will always return True
    Raises:
        RuntimeError: If the payload is larger a runtime error is thrown.
    """

    if not _payload_size_within_limit(payload, size):
        raise RuntimeError('Record is larger than %sMB. Please increase your max_payload' % size)
    return True
