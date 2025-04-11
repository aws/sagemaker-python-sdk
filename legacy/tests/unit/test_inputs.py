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

import pytest

from sagemaker import TrainingInput
from sagemaker.inputs import FileSystemInput


def test_training_input_all_defaults(caplog):
    prefix = "pre"
    actual = TrainingInput(s3_data=prefix)
    expected = {
        "DataSource": {
            "S3DataSource": {
                "S3DataDistributionType": "FullyReplicated",
                "S3DataType": "S3Prefix",
                "S3Uri": prefix,
            }
        }
    }
    assert actual.config == expected


def test_training_input_all_arguments():
    prefix = "pre"
    distribution = "FullyReplicated"
    compression = "Gzip"
    content_type = "text/csv"
    record_wrapping = "RecordIO"
    s3_data_type = "Manifestfile"
    input_mode = "Pipe"
    hub_access_config = {"HubContentArn": "some-hub-content-arn"}
    model_access_config = {"AcceptEula": True}
    result = TrainingInput(
        s3_data=prefix,
        distribution=distribution,
        compression=compression,
        input_mode=input_mode,
        content_type=content_type,
        record_wrapping=record_wrapping,
        s3_data_type=s3_data_type,
        hub_access_config=hub_access_config,
        model_access_config=model_access_config,
    )
    expected = {
        "DataSource": {
            "S3DataSource": {
                "S3DataDistributionType": distribution,
                "S3DataType": s3_data_type,
                "S3Uri": prefix,
                "ModelAccessConfig": model_access_config,
                "HubAccessConfig": hub_access_config,
            }
        },
        "CompressionType": compression,
        "ContentType": content_type,
        "RecordWrapperType": record_wrapping,
        "InputMode": input_mode,
    }

    assert result.config == expected


def test_training_input_all_arguments_heterogeneous_cluster():
    prefix = "pre"
    distribution = "FullyReplicated"
    compression = "Gzip"
    content_type = "text/csv"
    record_wrapping = "RecordIO"
    s3_data_type = "Manifestfile"
    instance_groups = ["data-server"]
    input_mode = "Pipe"
    hub_access_config = {"HubContentArn": "some-hub-content-arn"}
    model_access_config = {"AcceptEula": True}
    result = TrainingInput(
        s3_data=prefix,
        distribution=distribution,
        compression=compression,
        input_mode=input_mode,
        content_type=content_type,
        record_wrapping=record_wrapping,
        s3_data_type=s3_data_type,
        instance_groups=instance_groups,
        hub_access_config=hub_access_config,
        model_access_config=model_access_config,
    )

    expected = {
        "DataSource": {
            "S3DataSource": {
                "S3DataDistributionType": distribution,
                "S3DataType": s3_data_type,
                "S3Uri": prefix,
                "InstanceGroupNames": instance_groups,
                "ModelAccessConfig": model_access_config,
                "HubAccessConfig": hub_access_config,
            }
        },
        "CompressionType": compression,
        "ContentType": content_type,
        "RecordWrapperType": record_wrapping,
        "InputMode": input_mode,
    }

    assert result.config == expected


def test_file_system_input_default_access_mode():
    file_system_id = "fs-0a48d2a1"
    file_system_type = "EFS"
    directory_path = "tensorflow"
    actual = FileSystemInput(
        file_system_id=file_system_id,
        file_system_type=file_system_type,
        directory_path=directory_path,
    )
    expected = {
        "DataSource": {
            "FileSystemDataSource": {
                "FileSystemId": file_system_id,
                "FileSystemType": file_system_type,
                "DirectoryPath": directory_path,
                "FileSystemAccessMode": "ro",
            }
        }
    }
    assert actual.config == expected


def test_file_system_input_all_arguments():
    file_system_id = "fs-0a48d2a1"
    file_system_type = "FSxLustre"
    directory_path = "tensorflow"
    file_system_access_mode = "rw"
    actual = FileSystemInput(
        file_system_id=file_system_id,
        file_system_type=file_system_type,
        directory_path=directory_path,
        file_system_access_mode=file_system_access_mode,
    )
    expected = {
        "DataSource": {
            "FileSystemDataSource": {
                "FileSystemId": file_system_id,
                "FileSystemType": file_system_type,
                "DirectoryPath": directory_path,
                "FileSystemAccessMode": "rw",
            }
        }
    }
    assert actual.config == expected


def test_file_system_input_content_type():
    file_system_id = "fs-0a48d2a1"
    file_system_type = "FSxLustre"
    directory_path = "tensorflow"
    file_system_access_mode = "rw"
    content_type = "application/json"
    actual = FileSystemInput(
        file_system_id=file_system_id,
        file_system_type=file_system_type,
        directory_path=directory_path,
        file_system_access_mode=file_system_access_mode,
        content_type=content_type,
    )
    expected = {
        "DataSource": {
            "FileSystemDataSource": {
                "FileSystemId": file_system_id,
                "FileSystemType": file_system_type,
                "DirectoryPath": directory_path,
                "FileSystemAccessMode": "rw",
            }
        },
        "ContentType": content_type,
    }
    assert actual.config == expected


def test_file_system_input_type_invalid():
    with pytest.raises(ValueError) as excinfo:
        file_system_id = "fs-0a48d2a1"
        file_system_type = "ABC"
        directory_path = "tensorflow"
        FileSystemInput(
            file_system_id=file_system_id,
            file_system_type=file_system_type,
            directory_path=directory_path,
        )
    assert str(excinfo.value) == "Unrecognized file system type: ABC. Valid values: FSxLustre, EFS."


def test_file_system_input_mode_invalid():
    with pytest.raises(ValueError) as excinfo:
        file_system_id = "fs-0a48d2a1"
        file_system_type = "EFS"
        directory_path = "tensorflow"
        file_system_access_mode = "p"
        FileSystemInput(
            file_system_id=file_system_id,
            file_system_type=file_system_type,
            directory_path=directory_path,
            file_system_access_mode=file_system_access_mode,
        )
    assert str(excinfo.value) == "Unrecognized file system access mode: p. Valid values: ro, rw."
