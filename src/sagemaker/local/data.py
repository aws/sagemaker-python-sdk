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
from six.moves.urllib.parse import urlparse


class DataSourceFactory(object):

    @staticmethod
    def get_instance(data_source):
        parsed_uri = urlparse(data_source)
        if parsed_uri.scheme == 'file':
            return LocalFileDataSource(parsed_uri.path)
        else:
            # TODO Figure S3 and S3Manifest.
            return None

class DataSource(object):

    def get_file_list(self):
        pass


class LocalFileDataSource(DataSource):

    def __init__(self, root_path):
        self.root_path = root_path

    def get_file_list(self):
        if not os.path.exists(self.root_path):
            raise RuntimeError('Invalid data source: %s Does not exist.' % self.root_path)

        files = []
        if os.path.isdir(self.root_path):
            files = [os.path.join(self.root_path, f) for f in os.listdir(self.root_path)
                     if os.path.isfile(os.path.join(self.root_path, f))]
        else:
            files = [self.root_path]

        return files

class S3DataSource(DataSource):
    pass


class S3ManifestDataSource(DataSource):
    pass


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

    def split(self, file):
        with open(file, 'r') as f:
            yield f.read()

class LineSplitter(Splitter):

    def split(self, file):
        with open(file, 'r') as f:
            for line in f:
                yield line

class RecordIOSplitter(Splitter):

    def split(self, file):
        pass


class BatchStrategyFactory(object):

    @staticmethod
    def get_instance(strategy, splitter):
        if strategy == 'SingleRecord':
            return SingleRecordStrategy(splitter)
        elif strategy == 'MultiRecord':
            return MultiRecordStrategy(splitter)
        else:
            return None

class BatchStrategy(object):

    def pad(self, file, size):
        pass

class MultiRecordStrategy(BatchStrategy):

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

    def __init__(self, splitter):
        self.splitter = splitter

    def pad(self, file, size=6):
        for element in self.splitter.split(file):
            if _validate_payload_size(element, size):
                yield element


def _payload_size_within_limit(payload, size):
    size_in_bytes = size * 1024 * 1024
    if size == 0:
        return True
    else:
        print('size_of_payload: %s > %s' % (sys.getsizeof(payload), size_in_bytes))
        return sys.getsizeof(payload) < size_in_bytes

def _validate_payload_size(payload, size):
        if not _payload_size_within_limit(payload, size):
            raise RuntimeError('Record is larger than %sMB. Please increase your max_payload' % size)
        return True
