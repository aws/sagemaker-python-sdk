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


def recordio_iterator(file):
    # this isn't that simple
    pass
