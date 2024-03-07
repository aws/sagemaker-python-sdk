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
from typing import List, Optional
from datetime import datetime

import pytest
from sagemaker.jumpstart.curated_hub.sync.comparator import SizeAndLastUpdatedComparator

from sagemaker.jumpstart.curated_hub.sync.request import HubSyncRequestFactory
from sagemaker.jumpstart.curated_hub.types import (
    FileInfo,
    HubContentDependencyType,
    S3ObjectLocation,
)

COMPARATOR = SizeAndLastUpdatedComparator()


def _helper_generate_fileinfos(
    num_infos: int,
    bucket: Optional[str] = None,
    key_prefix: Optional[str] = None,
    size: Optional[int] = None,
    last_updated: Optional[datetime] = None,
    dependecy_type: Optional[HubContentDependencyType] = None,
) -> List[FileInfo]:

    file_infos = []
    for i in range(num_infos):
        bucket = bucket or "default-bucket"
        key_prefix = key_prefix or "mock-key"
        size = size or 123456
        last_updated = last_updated or datetime.today()

        file_infos.append(
            FileInfo(
                bucket=bucket,
                key=f"{key_prefix}-{i}",
                size=size,
                last_updated=last_updated,
                dependecy_type={dependecy_type},
            )
        )
    return file_infos


@pytest.mark.parametrize(
    ("src_files,dest_files"),
    [
        pytest.param(_helper_generate_fileinfos(8), []),
        pytest.param([], _helper_generate_fileinfos(8)),
        pytest.param([], []),
    ],
)
def test_sync_request_factory_edge_cases(src_files, dest_files):
    dest_location = S3ObjectLocation("mock-bucket-123", "mock-prefix")
    factory = HubSyncRequestFactory(src_files, dest_files, dest_location, COMPARATOR)

    req = factory.create()

    assert req.files == src_files
    assert req.destination == dest_location


def test_passes_existing_files_in_dest():
    files = _helper_generate_fileinfos(4, key_prefix="aafile.py")
    tarballs = _helper_generate_fileinfos(3, key_prefix="bb.tar.gz")
    extra_files = _helper_generate_fileinfos(2, key_prefix="ccextrafiles.py")

    src_files = [*tarballs, *files, *extra_files]
    dest_files = [files[1], files[2], tarballs[1]]

    expected_response = [files[0], files[3], tarballs[0], tarballs[2], *extra_files]

    dest_location = S3ObjectLocation("mock-bucket-123", "mock-prefix")
    factory = HubSyncRequestFactory(src_files, dest_files, dest_location, COMPARATOR)

    req = factory.create()

    assert req.files == expected_response


def test_adds_files_with_same_name_diff_size():
    file_one = _helper_generate_fileinfos(1, key_prefix="file.py", size=101010)[0]
    file_two = _helper_generate_fileinfos(1, key_prefix="file.py", size=123456)[0]

    src_files = [file_one]
    dest_files = [file_two]

    dest_location = S3ObjectLocation("mock-bucket-123", "mock-prefix")
    factory = HubSyncRequestFactory(src_files, dest_files, dest_location, COMPARATOR)

    req = factory.create()

    assert req.files == src_files


def test_adds_files_with_same_name_dest_older_time():
    file_dest = _helper_generate_fileinfos(1, key_prefix="file.py", last_updated=datetime.today())[
        0
    ]
    file_src = _helper_generate_fileinfos(1, key_prefix="file.py", size=datetime.today())[0]

    src_files = [file_src]
    dest_files = [file_dest]

    dest_location = S3ObjectLocation("mock-bucket-123", "mock-prefix")
    factory = HubSyncRequestFactory(src_files, dest_files, dest_location, COMPARATOR)

    req = factory.create()

    assert req.files == src_files


def test_does_not_add_files_with_same_name_src_older_time():
    file_src = _helper_generate_fileinfos(1, key_prefix="file.py", last_updated=datetime.today())[0]
    file_dest = _helper_generate_fileinfos(1, key_prefix="file.py", size=datetime.today())[0]

    src_files = [file_src]
    dest_files = [file_dest]

    dest_location = S3ObjectLocation("mock-bucket-123", "mock-prefix")
    factory = HubSyncRequestFactory(src_files, dest_files, dest_location, COMPARATOR)

    req = factory.create()

    assert req.files == src_files
