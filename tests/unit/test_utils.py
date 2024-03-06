# -*- coding: utf-8 -*-

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

import copy
import shutil
import tarfile
from datetime import datetime
import os
import re
import time
import json
from unittest import TestCase

from boto3 import exceptions
import botocore
import pytest
from mock import call, patch, Mock, MagicMock, PropertyMock

import sagemaker
from sagemaker.experiments._run_context import _RunContext
from sagemaker.session_settings import SessionSettings
from sagemaker.utils import (
    get_instance_type_family,
    retry_with_backoff,
    check_and_get_run_experiment_config,
    get_sagemaker_config_value,
    resolve_value_from_config,
    resolve_class_attribute_from_config,
    resolve_nested_dict_value_from_config,
    update_list_of_dicts_with_values_from_config,
    volume_size_supported,
    _get_resolved_path,
    _is_bad_path,
    _is_bad_link,
    custom_extractall_tarfile,
)
from tests.unit.sagemaker.workflow.helpers import CustomStep
from sagemaker.workflow.parameters import ParameterString, ParameterInteger

BUCKET_WITHOUT_WRITING_PERMISSION = "s3://bucket-without-writing-permission"

NAME = "base_name"
BUCKET_NAME = "some_bucket"


def test_get_config_value():
    config = {"local": {"region_name": "us-west-2", "port": "123"}, "other": {"key": 1}}

    assert sagemaker.utils.get_config_value("local.region_name", config) == "us-west-2"
    assert sagemaker.utils.get_config_value("local", config) == {
        "region_name": "us-west-2",
        "port": "123",
    }

    assert sagemaker.utils.get_config_value("does_not.exist", config) is None
    assert sagemaker.utils.get_config_value("other.key", None) is None


def test_get_nested_value():
    dictionary = {
        "local": {"region_name": "us-west-2", "port": "123"},
        "other": {"key": 1},
        "nest1": {"nest2": {"nest3": {"nest4": {"nest5a": "value", "nest5b": None}}}},
    }

    # happy cases: keys and values exist
    assert sagemaker.utils.get_nested_value(dictionary, ["local", "region_name"]) == "us-west-2"
    assert sagemaker.utils.get_nested_value(dictionary, ["local"]) == {
        "region_name": "us-west-2",
        "port": "123",
    }
    assert (
        sagemaker.utils.get_nested_value(dictionary, ["nest1", "nest2", "nest3", "nest4", "nest5a"])
        == "value"
    )

    # edge cases: non-existing keys
    assert sagemaker.utils.get_nested_value(dictionary, ["local", "new_depth_1_key"]) is None
    assert sagemaker.utils.get_nested_value(dictionary, ["new_depth_0_key"]) is None
    assert (
        sagemaker.utils.get_nested_value(dictionary, ["new_depth_0_key", "new_depth_1_key"]) is None
    )
    assert (
        sagemaker.utils.get_nested_value(
            dictionary, ["nest1", "nest2", "nest3", "nest4", "nest5b", "does_not", "exist"]
        )
        is None
    )

    # edge case: specified nested_keys contradict structure of dict
    with pytest.raises(ValueError):
        sagemaker.utils.get_nested_value(
            dictionary, ["nest1", "nest2", "nest3", "nest4", "nest5a", "does_not", "exist"]
        )

    # edge cases: non-actionable inputs
    assert sagemaker.utils.get_nested_value(None, ["other", "key"]) is None
    assert sagemaker.utils.get_nested_value("not_a_dict", ["other", "key"]) is None
    assert sagemaker.utils.get_nested_value(dictionary, None) is None
    assert sagemaker.utils.get_nested_value(dictionary, []) is None


@patch("jsonschema.validate")
def test_update_list_of_dicts_with_values_from_config(mock_json_schema_validation):
    input_list = [{"a": 1, "b": 2}]
    input_config_list = [
        {
            "a": 4,  # This should not be used. Use values from Input.
            "c": 3,
        }
    ]
    # Using short form for sagemaker_session
    ss = MagicMock(settings=SessionSettings())

    ss.sagemaker_config = {"DUMMY": {"CONFIG": {"PATH": input_config_list}}}
    config_path = "DUMMY.CONFIG.PATH"
    # happy case - both inputs and config have same number of elements
    update_list_of_dicts_with_values_from_config(input_list, config_path, sagemaker_session=ss)
    assert input_list == [{"a": 1, "b": 2, "c": 3}]
    # Case where Input has more entries compared to Config
    input_list = [
        {"a": 1, "b": 2},
        {"a": 5, "b": 6},
    ]
    input_config_list = [
        {
            "a": 4,  # This should not be used. Use values from Input.
            "c": 3,
        }
    ]
    ss.sagemaker_config = {"DUMMY": {"CONFIG": {"PATH": input_config_list}}}
    update_list_of_dicts_with_values_from_config(input_list, config_path, sagemaker_session=ss)
    assert input_list == [
        {"a": 1, "b": 2, "c": 3},
        {"a": 5, "b": 6},
    ]
    # Case where Config has more entries when compared to the input
    input_list = [{"a": 1, "b": 2}]
    input_config_list = [
        {
            "a": 4,  # This should not be used. Use values from Input.
            "c": 3,
        },
        {"a": 5, "b": 6},
    ]
    ss.sagemaker_config = {"DUMMY": {"CONFIG": {"PATH": input_config_list}}}
    update_list_of_dicts_with_values_from_config(input_list, config_path, sagemaker_session=ss)
    assert input_list == [{"a": 1, "b": 2, "c": 3}]
    # Testing required parameters. If required parameters are not present, don't do the merge
    input_list = [{"a": 1, "b": 2}]
    input_config_list = [
        {
            "a": 4,  # This should not be used. Use values from Input.
            "c": 3,
        },
    ]
    ss.sagemaker_config = {"DUMMY": {"CONFIG": {"PATH": input_config_list}}}
    update_list_of_dicts_with_values_from_config(
        input_list, config_path, required_key_paths=["d"], sagemaker_session=ss
    )
    # since 'd' is not there , merge shouldn't have happened
    assert input_list == [{"a": 1, "b": 2}]
    # Testing required parameters. If required parameters are present, do the merge
    input_list = [{"a": 1, "b": 2}, {"a": 5, "c": 6}]
    input_config_list = [
        {
            "a": 4,  # This should not be used. Use values from Input.
            "c": 3,
        },
        {
            "a": 7,  # This should not be used. Use values from Input.
            "b": 8,
        },
    ]
    ss.sagemaker_config = {"DUMMY": {"CONFIG": {"PATH": input_config_list}}}
    update_list_of_dicts_with_values_from_config(
        input_list, config_path, required_key_paths=["c"], sagemaker_session=ss
    )
    assert input_list == [
        {"a": 1, "b": 2, "c": 3},
        {"a": 5, "b": 8, "c": 6},
    ]
    # Testing union parameters: If both parameters are present don't do the merge
    input_list = [{"a": 1, "b": 2}, {"a": 5, "c": 6}]
    input_config_list = [
        {
            "a": 4,  # This should not be used. Use values from Input.
            "c": 3,
        },
        {
            "a": 7,  # This should not be used. Use values from Input.
            "d": 8,  # c is present in the original list and d is present in this list.
        },
    ]
    ss.sagemaker_config = {"DUMMY": {"CONFIG": {"PATH": input_config_list}}}
    update_list_of_dicts_with_values_from_config(
        input_list, config_path, union_key_paths=[["c", "d"]], sagemaker_session=ss
    )
    assert input_list == [
        {"a": 1, "b": 2, "c": 3},
        {"a": 5, "c": 6},  # merge didn't happen
    ]
    # Testing union parameters: Happy case
    input_list = [{"a": 1, "b": 2}, {"a": 5, "c": 6}]
    input_config_list = [
        {
            "a": 4,  # This should not be used. Use values from Input.
            "c": 3,
        },
        {
            "a": 7,  # This should not be used. Use values from Input.
            "d": 8,
        },
    ]
    ss.sagemaker_config = {"DUMMY": {"CONFIG": {"PATH": input_config_list}}}
    update_list_of_dicts_with_values_from_config(
        input_list, config_path, union_key_paths=[["c", "e"], ["d", "e"]], sagemaker_session=ss
    )
    assert input_list == [
        {"a": 1, "b": 2, "c": 3},
        {"a": 5, "c": 6, "d": 8},
    ]
    # Same happy case with different order of items in union_key_paths
    input_list = [{"a": 1, "b": 2}, {"a": 5, "c": 6}]
    input_config_list = [
        {
            "a": 4,  # This should not be used. Use values from Input.
            "c": 3,
        },
        {
            "a": 7,  # This should not be used. Use values from Input.
            "d": 8,
        },
    ]
    ss.sagemaker_config = {"DUMMY": {"CONFIG": {"PATH": input_config_list}}}
    update_list_of_dicts_with_values_from_config(
        input_list, config_path, union_key_paths=[["d", "e"], ["c", "e"]], sagemaker_session=ss
    )
    assert input_list == [
        {"a": 1, "b": 2, "c": 3},
        {"a": 5, "c": 6, "d": 8},
    ]
    # Testing the combination of union parameter and required parameter. i.e. A parameter is both
    # required and part of Union.
    input_list = [{"a": 1, "b": 2}, {"a": 5, "c": 6}]
    input_config_list = [
        {
            "a": 4,  # This should not be used. Use values from Input.
            "c": 3,
        },
        {
            "a": 7,  # This should not be used. Use values from Input.
            "d": 8,
        },
    ]
    ss.sagemaker_config = {"DUMMY": {"CONFIG": {"PATH": input_config_list}}}
    update_list_of_dicts_with_values_from_config(
        input_list,
        config_path,
        required_key_paths=["e"],
        union_key_paths=[["d", "e"], ["c", "e"]],
        sagemaker_session=ss,
    )
    # No merge should happen since 'e' is not present, even though union is obeyed.
    assert input_list == [{"a": 1, "b": 2}, {"a": 5, "c": 6}]
    # Same test but the required parameter is present.
    input_list = [{"a": 1, "e": 2}, {"a": 5, "e": 6}]
    input_config_list = [
        {
            "a": 4,  # This should not be used. Use values from Input.
            "f": 3,
        },
        {
            "a": 7,  # This should not be used. Use values from Input.
            "g": 8,
        },
    ]
    ss.sagemaker_config = {"DUMMY": {"CONFIG": {"PATH": input_config_list}}}
    update_list_of_dicts_with_values_from_config(
        input_list,
        config_path,
        required_key_paths=["e"],
        union_key_paths=[["d", "e"], ["c", "e"]],
        sagemaker_session=ss,
    )
    assert input_list == [
        {"a": 1, "e": 2, "f": 3},
        {"a": 5, "e": 6, "g": 8},
    ]


def test_set_nested_value():
    dictionary = {
        "local": {"region_name": "us-west-2", "port": "123"},
        "other": {"key": 1},
        "nest1": {"nest2": {"nest3": {"nest4": {"nest5a": "value", "nest5b": None}}}},
        "existing_depth_0_key": None,
    }
    dictionary_copy = copy.deepcopy(dictionary)

    # happy cases: change existing values
    dictionary_copy["local"]["region_name"] = "region1"
    assert (
        sagemaker.utils.set_nested_value(dictionary, ["local", "region_name"], "region1")
        == dictionary_copy
    )

    dictionary_copy["existing_depth_0_key"] = {"new_key": "new_value"}
    assert (
        sagemaker.utils.set_nested_value(
            dictionary, ["existing_depth_0_key"], {"new_key": "new_value"}
        )
        == dictionary_copy
    )

    dictionary_copy["nest1"]["nest2"]["nest3"]["nest4"]["nest5a"] = "value2"
    assert (
        sagemaker.utils.set_nested_value(
            dictionary, ["nest1", "nest2", "nest3", "nest4", "nest5a"], "value2"
        )
        == dictionary_copy
    )

    # happy cases: add new keys and values
    dictionary_copy["local"]["new_depth_1_key"] = "value"
    assert (
        sagemaker.utils.set_nested_value(dictionary, ["local", "new_depth_1_key"], "value")
        == dictionary_copy
    )

    dictionary_copy["new_depth_0_key"] = "value"
    assert (
        sagemaker.utils.set_nested_value(dictionary, ["new_depth_0_key"], "value")
        == dictionary_copy
    )

    dictionary_copy["new_depth_0_key_2"] = {"new_depth_1_key_2": "value"}
    assert (
        sagemaker.utils.set_nested_value(
            dictionary, ["new_depth_0_key_2", "new_depth_1_key_2"], "value"
        )
        == dictionary_copy
    )

    dictionary_copy["nest1"]["nest2"]["nest3"]["nest4"]["nest5b"] = {"does_not": {"exist": "value"}}
    assert (
        sagemaker.utils.set_nested_value(
            dictionary, ["nest1", "nest2", "nest3", "nest4", "nest5b", "does_not", "exist"], "value"
        )
        == dictionary_copy
    )

    # edge case: overwrite non-dict value
    dictionary["nest1"]["nest2"]["nest3"]["nest4"]["nest5a"] = "value2"
    dictionary_copy["nest1"]["nest2"]["nest3"]["nest4"]["nest5a"] = {"does_not": {"exist": "value"}}
    assert (
        sagemaker.utils.set_nested_value(
            dictionary, ["nest1", "nest2", "nest3", "nest4", "nest5a", "does_not", "exist"], "value"
        )
        == dictionary_copy
    )

    # edge case: dict does not exist
    assert sagemaker.utils.set_nested_value(None, ["other", "key"], "value") == {
        "other": {"key": "value"}
    }

    # edge cases: non-actionable inputs
    dictionary_copy_2 = copy.deepcopy(dictionary)
    assert sagemaker.utils.set_nested_value("not_a_dict", ["other", "key"], "value") == "not_a_dict"
    assert sagemaker.utils.set_nested_value(dictionary, None, "value") == dictionary_copy_2
    assert sagemaker.utils.set_nested_value(dictionary, [], "value") == dictionary_copy_2


def test_get_short_version():
    assert sagemaker.utils.get_short_version("2.2.0") == "2.2"
    assert sagemaker.utils.get_short_version("2.2") == "2.2"
    assert sagemaker.utils.get_short_version("2.1.0") == "2.1"
    assert sagemaker.utils.get_short_version("2.1") == "2.1"
    assert sagemaker.utils.get_short_version("2.0.1") == "2.0"
    assert sagemaker.utils.get_short_version("2.0.0") == "2.0"
    assert sagemaker.utils.get_short_version("2.0") == "2.0"


def test_deferred_error():
    de = sagemaker.utils.DeferredError(ImportError("pretend the import failed"))
    with pytest.raises(ImportError) as _:  # noqa: F841
        de.something()


def test_bad_import():
    try:
        import pandas_is_not_installed as pd
    except ImportError as e:
        pd = sagemaker.utils.DeferredError(e)
    assert pd is not None
    with pytest.raises(ImportError) as _:  # noqa: F841
        pd.DataFrame()


@patch("sagemaker.utils.name_from_base")
@patch("sagemaker.utils.base_name_from_image")
def test_name_from_image(base_name_from_image, name_from_base):
    image = "image:latest"
    max_length = 32

    sagemaker.utils.name_from_image(image, max_length=max_length)
    base_name_from_image.assert_called_with(image)
    name_from_base.assert_called_with(base_name_from_image.return_value, max_length=max_length)


@pytest.mark.parametrize(
    "inputs",
    [
        (
            CustomStep(name="test-custom-step").properties.OutputDataConfig.S3OutputPath,
            None,
            "base_name",
        ),
        (
            CustomStep(name="test-custom-step").properties.OutputDataConfig.S3OutputPath,
            "whatever",
            "whatever",
        ),
        (ParameterString(name="image_uri"), None, "base_name"),
        (ParameterString(name="image_uri"), "whatever", "whatever"),
        (
            ParameterString(
                name="image_uri",
                default_value="922956235488.dkr.ecr.us-west-2.amazonaws.com/analyzer",
            ),
            None,
            "analyzer",
        ),
        (
            ParameterString(
                name="image_uri",
                default_value="922956235488.dkr.ecr.us-west-2.amazonaws.com/analyzer",
            ),
            "whatever",
            "analyzer",
        ),
    ],
)
def test_base_name_from_image_with_pipeline_param(inputs):
    image, default_base_name, expected = inputs
    assert expected == sagemaker.utils.base_name_from_image(
        image=image, default_base_name=default_base_name
    )


@patch("sagemaker.utils.sagemaker_timestamp")
def test_name_from_base(sagemaker_timestamp):
    sagemaker.utils.name_from_base(NAME, short=False)
    sagemaker_timestamp.assert_called_once


@patch("sagemaker.utils.sagemaker_short_timestamp")
def test_name_from_base_short(sagemaker_short_timestamp):
    sagemaker.utils.name_from_base(NAME, short=True)
    sagemaker_short_timestamp.assert_called_once


def test_unique_name_from_base():
    assert re.match(r"base-\d{10}-[a-f0-9]{4}", sagemaker.utils.unique_name_from_base("base"))


def test_unique_name_from_base_uuid4():
    assert re.match(
        r"base-([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})",
        sagemaker.utils.unique_name_from_base_uuid4("base"),
    )


def test_unique_name_from_base_uuid4_truncated():
    assert re.match(
        r"a-really-long-([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})",
        sagemaker.utils.unique_name_from_base_uuid4("a-really-long-base-name", max_length=50),
    )


def test_unique_name_from_base_truncated():
    assert re.match(
        r"real-\d{10}-[a-f0-9]{4}",
        sagemaker.utils.unique_name_from_base("really-long-name", max_length=20),
    )


def test_base_from_name():
    name = "mxnet-training-2020-06-29-15-19-25-475"
    assert "mxnet-training" == sagemaker.utils.base_from_name(name)

    name = "sagemaker-pytorch-200629-1611"
    assert "sagemaker-pytorch" == sagemaker.utils.base_from_name(name)


MESSAGE = "message"
STATUS = "status"
TRAINING_JOB_DESCRIPTION_1 = {
    "SecondaryStatusTransitions": [{"StatusMessage": MESSAGE, "Status": STATUS}]
}
TRAINING_JOB_DESCRIPTION_2 = {
    "SecondaryStatusTransitions": [{"StatusMessage": "different message", "Status": STATUS}]
}

TRAINING_JOB_DESCRIPTION_EMPTY = {"SecondaryStatusTransitions": []}


def test_secondary_training_status_changed_true():
    changed = sagemaker.utils.secondary_training_status_changed(
        TRAINING_JOB_DESCRIPTION_1, TRAINING_JOB_DESCRIPTION_2
    )
    assert changed is True


def test_secondary_training_status_changed_false():
    changed = sagemaker.utils.secondary_training_status_changed(
        TRAINING_JOB_DESCRIPTION_1, TRAINING_JOB_DESCRIPTION_1
    )
    assert changed is False


def test_secondary_training_status_changed_prev_missing():
    changed = sagemaker.utils.secondary_training_status_changed(TRAINING_JOB_DESCRIPTION_1, {})
    assert changed is True


def test_secondary_training_status_changed_prev_none():
    changed = sagemaker.utils.secondary_training_status_changed(TRAINING_JOB_DESCRIPTION_1, None)
    assert changed is True


def test_secondary_training_status_changed_current_missing():
    changed = sagemaker.utils.secondary_training_status_changed({}, TRAINING_JOB_DESCRIPTION_1)
    assert changed is False


def test_secondary_training_status_changed_empty():
    changed = sagemaker.utils.secondary_training_status_changed(
        TRAINING_JOB_DESCRIPTION_EMPTY, TRAINING_JOB_DESCRIPTION_1
    )
    assert changed is False


def test_secondary_training_status_message_status_changed():
    now = datetime.now()
    TRAINING_JOB_DESCRIPTION_1["LastModifiedTime"] = now
    expected = "{} {} - {}".format(
        datetime.utcfromtimestamp(time.mktime(now.timetuple())).strftime("%Y-%m-%d %H:%M:%S"),
        STATUS,
        MESSAGE,
    )
    assert (
        sagemaker.utils.secondary_training_status_message(
            TRAINING_JOB_DESCRIPTION_1, TRAINING_JOB_DESCRIPTION_EMPTY
        )
        == expected
    )


def test_secondary_training_status_message_status_not_changed():
    now = datetime.now()
    TRAINING_JOB_DESCRIPTION_1["LastModifiedTime"] = now
    expected = "{} {} - {}".format(
        datetime.utcfromtimestamp(time.mktime(now.timetuple())).strftime("%Y-%m-%d %H:%M:%S"),
        STATUS,
        MESSAGE,
    )
    assert (
        sagemaker.utils.secondary_training_status_message(
            TRAINING_JOB_DESCRIPTION_1, TRAINING_JOB_DESCRIPTION_2
        )
        == expected
    )


def test_secondary_training_status_message_prev_missing():
    now = datetime.now()
    TRAINING_JOB_DESCRIPTION_1["LastModifiedTime"] = now
    expected = "{} {} - {}".format(
        datetime.utcfromtimestamp(time.mktime(now.timetuple())).strftime("%Y-%m-%d %H:%M:%S"),
        STATUS,
        MESSAGE,
    )
    assert (
        sagemaker.utils.secondary_training_status_message(TRAINING_JOB_DESCRIPTION_1, {})
        == expected
    )


SAMPLE_DATA_CONFIG = {"us-west-2": "sagemaker-hosted-datasets", "default": "sagemaker-sample-files"}


def test_notebooks_data_config_if_region_not_present():
    sample_data_config = json.dumps(SAMPLE_DATA_CONFIG)

    boto_mock = MagicMock(name="boto_session", region_name="ap-northeast-1")
    session = sagemaker.Session(boto_session=boto_mock, sagemaker_client=MagicMock())
    session.read_s3_file = Mock(return_value=sample_data_config)
    assert (
        sagemaker.utils.S3DataConfig(
            session, "example-notebooks-data-config", "config/data_config.json"
        ).get_data_bucket()
        == "sagemaker-sample-files"
    )


def test_notebooks_data_config_if_region_present():
    sample_data_config = json.dumps(SAMPLE_DATA_CONFIG)

    boto_mock = MagicMock(name="boto_session", region_name="us-west-2")
    session = sagemaker.Session(boto_session=boto_mock, sagemaker_client=MagicMock())
    session.read_s3_file = Mock(return_value=sample_data_config)
    assert (
        sagemaker.utils.S3DataConfig(
            session, "example-notebooks-data-config", "config/data_config.json"
        ).get_data_bucket()
        == "sagemaker-hosted-datasets"
    )


@patch("os.makedirs")
def test_download_folder(makedirs):
    boto_mock = MagicMock(name="boto_session")
    session = sagemaker.Session(boto_session=boto_mock, sagemaker_client=MagicMock())
    s3_mock = boto_mock.resource("s3")

    obj_mock = Mock()
    s3_mock.Object.return_value = obj_mock

    def obj_mock_download(path):
        # Mock the S3 object to raise an error when the input to download_file
        # is a "folder"
        if path in ("/tmp/", os.path.join("/tmp", "prefix")):
            raise botocore.exceptions.ClientError(
                error_response={"Error": {"Code": "404", "Message": "Not Found"}},
                operation_name="HeadObject",
            )
        else:
            return Mock()

    obj_mock.download_file.side_effect = obj_mock_download

    train_data = Mock()
    validation_data = Mock()

    train_data.bucket_name.return_value = BUCKET_NAME
    train_data.key = "prefix/train/train_data.csv"
    validation_data.bucket_name.return_value = BUCKET_NAME
    validation_data.key = "prefix/train/validation_data.csv"

    s3_files = [train_data, validation_data]
    s3_mock.Bucket(BUCKET_NAME).objects.filter.return_value = s3_files

    # all the S3 mocks are set, the test itself begins now.
    sagemaker.utils.download_folder(BUCKET_NAME, "/prefix", "/tmp", session)

    obj_mock.download_file.assert_called()
    calls = [
        call(os.path.join("/tmp", "train", "train_data.csv")),
        call(os.path.join("/tmp", "train", "validation_data.csv")),
    ]
    obj_mock.download_file.assert_has_calls(calls)
    assert s3_mock.Object.call_count == 3

    s3_mock.reset_mock()
    obj_mock.reset_mock()

    # Test with a trailing slash for the prefix.
    sagemaker.utils.download_folder(BUCKET_NAME, "/prefix/", "/tmp", session)
    obj_mock.download_file.assert_called()
    obj_mock.download_file.assert_has_calls(calls)
    assert s3_mock.Object.call_count == 2


@patch("os.makedirs")
def test_download_folder_points_to_single_file(makedirs):
    boto_mock = MagicMock(name="boto_session")
    boto_mock.client("sts").get_caller_identity.return_value = {"Account": "123"}

    session = sagemaker.Session(boto_session=boto_mock, sagemaker_client=MagicMock())

    train_data = Mock()

    train_data.bucket_name.return_value = BUCKET_NAME
    train_data.key = "prefix/train/train_data.csv"

    s3_files = [train_data]
    boto_mock.resource("s3").Bucket(BUCKET_NAME).objects.filter.return_value = s3_files

    obj_mock = Mock()
    boto_mock.resource("s3").Object.return_value = obj_mock

    # all the S3 mocks are set, the test itself begins now.
    sagemaker.utils.download_folder(BUCKET_NAME, "/prefix/train/train_data.csv", "/tmp", session)

    obj_mock.download_file.assert_called()
    calls = [call(os.path.join("/tmp", "train_data.csv"))]
    obj_mock.download_file.assert_has_calls(calls)
    boto_mock.resource("s3").Bucket(BUCKET_NAME).objects.filter.assert_not_called()
    obj_mock.reset_mock()


def test_download_file():
    boto_mock = MagicMock(name="boto_session")
    boto_mock.client("sts").get_caller_identity.return_value = {"Account": "123"}
    bucket_mock = Mock()
    boto_mock.resource("s3").Bucket.return_value = bucket_mock
    session = sagemaker.Session(boto_session=boto_mock, sagemaker_client=MagicMock())

    sagemaker.utils.download_file(
        BUCKET_NAME, "/prefix/path/file.tar.gz", "/tmp/file.tar.gz", session
    )

    bucket_mock.download_file.assert_called_with("prefix/path/file.tar.gz", "/tmp/file.tar.gz")


@patch("tarfile.open")
def test_create_tar_file_with_provided_path(open):
    files = mock_tarfile(open)

    file_list = ["/tmp/a", "/tmp/b"]

    path = sagemaker.utils.create_tar_file(file_list, target="/my/custom/path.tar.gz")
    assert path == "/my/custom/path.tar.gz"
    assert files == [["/tmp/a", "a"], ["/tmp/b", "b"]]


def mock_tarfile(open):
    open.return_value = open
    files = []

    def add_files(filename, arcname):
        files.append([filename, arcname])

    open.__enter__ = Mock()
    open.__enter__().add = add_files
    open.__exit__ = Mock(return_value=None)
    return files


@patch("tarfile.open")
@patch("tempfile.mkstemp", Mock(return_value=(None, "/auto/generated/path")))
def test_create_tar_file_with_auto_generated_path(open):
    files = mock_tarfile(open)

    path = sagemaker.utils.create_tar_file(["/tmp/a", "/tmp/b"])
    assert path == "/auto/generated/path"
    assert files == [["/tmp/a", "a"], ["/tmp/b", "b"]]


def create_file_tree(root, tree):
    for file in tree:
        try:
            os.makedirs(os.path.join(root, os.path.dirname(file)))
        except:  # noqa: E722 Using bare except because p2/3 incompatibility issues.
            pass
        with open(os.path.join(root, file), "a") as f:
            f.write(file)


@pytest.fixture()
def tmp(tmpdir):
    yield str(tmpdir)


def test_repack_model_without_source_dir(tmp, fake_s3):
    create_file_tree(
        tmp,
        [
            "model-dir/model",
            "dependencies/a",
            "dependencies/some/dir/b",
            "aa",
            "bb",
            "source-dir/inference.py",
            "source-dir/this-file-should-not-be-included.py",
        ],
    )

    fake_s3.tar_and_upload("model-dir", "s3://fake/location")

    sagemaker.utils.repack_model(
        inference_script=os.path.join(tmp, "source-dir/inference.py"),
        source_directory=None,
        dependencies=[
            os.path.join(tmp, "dependencies/a"),
            os.path.join(tmp, "dependencies/some/dir"),
            os.path.join(tmp, "aa"),
            os.path.join(tmp, "bb"),
        ],
        model_uri="s3://fake/location",
        repacked_model_uri="s3://destination-bucket/model.tar.gz",
        sagemaker_session=fake_s3.sagemaker_session,
    )

    assert list_tar_files(fake_s3.fake_upload_path, tmp) == {
        "/model",
        "/code/lib/a",
        "/code/lib/aa",
        "/code/lib/bb",
        "/code/lib/dir/b",
        "/code/inference.py",
    }

    extra_args = {"ServerSideEncryption": "aws:kms"}
    object_mock = fake_s3.object_mock
    _, _, kwargs = object_mock.mock_calls[0]

    assert "ExtraArgs" in kwargs
    assert kwargs["ExtraArgs"] == extra_args


def test_repack_model_with_entry_point_without_path_without_source_dir(tmp, fake_s3):
    create_file_tree(
        tmp,
        [
            "model-dir/model",
            "source-dir/inference.py",
            "source-dir/this-file-should-not-be-included.py",
        ],
    )

    fake_s3.tar_and_upload("model-dir", "s3://fake/location")

    cwd = os.getcwd()
    try:
        os.chdir(os.path.join(tmp, "source-dir"))

        sagemaker.utils.repack_model(
            "inference.py",
            None,
            None,
            "s3://fake/location",
            "s3://destination-bucket/model.tar.gz",
            fake_s3.sagemaker_session,
            kms_key="kms_key",
        )
    finally:
        os.chdir(cwd)

    assert list_tar_files(fake_s3.fake_upload_path, tmp) == {"/code/inference.py", "/model"}

    extra_args = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": "kms_key"}
    object_mock = fake_s3.object_mock
    _, _, kwargs = object_mock.mock_calls[0]

    assert "ExtraArgs" in kwargs
    assert kwargs["ExtraArgs"] == extra_args


def test_repack_model_from_s3_to_s3(tmp, fake_s3):
    create_file_tree(
        tmp,
        [
            "model-dir/model",
            "source-dir/inference.py",
            "source-dir/this-file-should-be-included.py",
        ],
    )

    fake_s3.tar_and_upload("model-dir", "s3://fake/location")
    fake_s3.sagemaker_session.settings = SessionSettings(encrypt_repacked_artifacts=False)

    sagemaker.utils.repack_model(
        "inference.py",
        os.path.join(tmp, "source-dir"),
        None,
        "s3://fake/location",
        "s3://destination-bucket/model.tar.gz",
        fake_s3.sagemaker_session,
    )

    assert list_tar_files(fake_s3.fake_upload_path, tmp) == {
        "/code/this-file-should-be-included.py",
        "/code/inference.py",
        "/model",
    }

    object_mock = fake_s3.object_mock
    _, _, kwargs = object_mock.mock_calls[0]
    assert "ExtraArgs" in kwargs
    assert kwargs["ExtraArgs"] is None


def test_repack_model_from_file_to_file(tmp):
    create_file_tree(tmp, ["model", "dependencies/a", "source-dir/inference.py"])

    model_tar_path = os.path.join(tmp, "model.tar.gz")
    sagemaker.utils.create_tar_file([os.path.join(tmp, "model")], model_tar_path)

    sagemaker_session = MagicMock(settings=SessionSettings())

    file_mode_path = "file://%s" % model_tar_path
    destination_path = "file://%s" % os.path.join(tmp, "repacked-model.tar.gz")

    sagemaker.utils.repack_model(
        "inference.py",
        os.path.join(tmp, "source-dir"),
        [os.path.join(tmp, "dependencies/a")],
        file_mode_path,
        destination_path,
        sagemaker_session,
    )

    assert list_tar_files(destination_path, tmp) == {"/code/lib/a", "/code/inference.py", "/model"}


def test_repack_model_with_inference_code_should_replace_the_code(tmp, fake_s3):
    create_file_tree(
        tmp, ["model-dir/model", "source-dir/new-inference.py", "model-dir/code/old-inference.py"]
    )

    fake_s3.tar_and_upload("model-dir", "s3://fake/location")

    sagemaker.utils.repack_model(
        "inference.py",
        os.path.join(tmp, "source-dir"),
        None,
        "s3://fake/location",
        "s3://destination-bucket/repacked-model",
        fake_s3.sagemaker_session,
    )

    assert list_tar_files(fake_s3.fake_upload_path, tmp) == {"/code/new-inference.py", "/model"}


def test_repack_model_from_file_to_folder(tmp):
    create_file_tree(tmp, ["model", "source-dir/inference.py"])

    model_tar_path = os.path.join(tmp, "model.tar.gz")
    sagemaker.utils.create_tar_file([os.path.join(tmp, "model")], model_tar_path)

    file_mode_path = "file://%s" % model_tar_path

    sagemaker.utils.repack_model(
        "inference.py",
        os.path.join(tmp, "source-dir"),
        [],
        file_mode_path,
        "file://%s/repacked-model.tar.gz" % tmp,
        MagicMock(settings=SessionSettings()),
    )

    assert list_tar_files("file://%s/repacked-model.tar.gz" % tmp, tmp) == {
        "/code/inference.py",
        "/model",
    }


def test_repack_model_with_inference_code_and_requirements(tmp, fake_s3):
    create_file_tree(
        tmp,
        [
            "new-inference.py",
            "model-dir/model",
            "model-dir/code/old-inference.py",
            "model-dir/code/requirements.txt",
        ],
    )

    fake_s3.tar_and_upload("model-dir", "s3://fake/location")

    sagemaker.utils.repack_model(
        os.path.join(tmp, "new-inference.py"),
        None,
        None,
        "s3://fake/location",
        "s3://destination-bucket/repacked-model",
        fake_s3.sagemaker_session,
    )

    assert list_tar_files(fake_s3.fake_upload_path, tmp) == {
        "/code/requirements.txt",
        "/code/new-inference.py",
        "/code/old-inference.py",
        "/model",
    }


def test_repack_model_with_same_inference_file_name(tmp, fake_s3):
    create_file_tree(
        tmp,
        [
            "inference.py",
            "model-dir/model",
            "model-dir/code/inference.py",
            "model-dir/code/requirements.txt",
        ],
    )

    fake_s3.tar_and_upload("model-dir", "s3://fake/location")

    sagemaker.utils.repack_model(
        os.path.join(tmp, "inference.py"),
        None,
        None,
        "s3://fake/location",
        "s3://destination-bucket/repacked-model",
        fake_s3.sagemaker_session,
    )

    assert list_tar_files(fake_s3.fake_upload_path, tmp) == {
        "/code/requirements.txt",
        "/code/inference.py",
        "/model",
    }


class FakeS3(object):
    def __init__(self, tmp):
        self.tmp = tmp
        self.sagemaker_session = MagicMock(settings=SessionSettings())
        self.location_map = {}
        self.current_bucket = None
        self.object_mock = MagicMock()

        self.sagemaker_session.boto_session.resource().Bucket().download_file.side_effect = (
            self.download_file
        )
        self.sagemaker_session.boto_session.resource().Bucket.side_effect = self.bucket
        self.fake_upload_path = self.mock_s3_upload()

    def bucket(self, name):
        self.current_bucket = name
        return self

    def download_file(self, path, target):
        key = "%s/%s" % (self.current_bucket, path)
        shutil.copy2(self.location_map[key], target)

    def tar_and_upload(self, path, fake_location):
        tar_location = os.path.join(self.tmp, "model-%s.tar.gz" % time.time())
        with tarfile.open(tar_location, mode="w:gz") as t:
            t.add(os.path.join(self.tmp, path), arcname=os.path.sep)

        self.location_map[fake_location.replace("s3://", "")] = tar_location
        return tar_location

    def mock_s3_upload(self):
        dst = os.path.join(self.tmp, "dst")
        object_mock = self.object_mock

        class MockS3Object(object):
            def __init__(self, bucket, key):
                self.bucket = bucket
                self.key = key

            def upload_file(self, target, **kwargs):
                if self.bucket in BUCKET_WITHOUT_WRITING_PERMISSION:
                    raise exceptions.S3UploadFailedError()
                shutil.copy2(target, dst)
                object_mock.upload_file(target, **kwargs)

        self.sagemaker_session.boto_session.resource().Object = MockS3Object
        return dst


@pytest.fixture()
def fake_s3(tmp):
    return FakeS3(tmp)


def list_tar_files(tar_ball, tmp):
    tar_ball = tar_ball.replace("file://", "")
    startpath = os.path.join(tmp, "startpath")
    os.mkdir(startpath)

    with tarfile.open(name=tar_ball, mode="r:gz") as t:
        custom_extractall_tarfile(t, startpath)

    def walk():
        for root, dirs, files in os.walk(startpath):
            path = root.replace(startpath, "")
            for f in files:
                yield "%s/%s" % (path, f)

    result = set(walk())
    return result if result else {}


def test_sts_regional_endpoint():
    endpoint = sagemaker.utils.sts_regional_endpoint("us-west-2")
    assert endpoint == "https://sts.us-west-2.amazonaws.com"
    assert botocore.utils.is_valid_endpoint_url(endpoint)

    endpoint = sagemaker.utils.sts_regional_endpoint("us-iso-east-1")
    assert endpoint == "https://sts.us-iso-east-1.c2s.ic.gov"
    assert botocore.utils.is_valid_endpoint_url(endpoint)

    endpoint = sagemaker.utils.sts_regional_endpoint("us-isob-east-1")
    assert endpoint == "https://sts.us-isob-east-1.sc2s.sgov.gov"
    assert botocore.utils.is_valid_endpoint_url(endpoint)


def test_partition_by_region():
    assert sagemaker.utils.aws_partition("us-west-2") == "aws"
    assert sagemaker.utils.aws_partition("cn-north-1") == "aws-cn"
    assert sagemaker.utils.aws_partition("us-gov-east-1") == "aws-us-gov"
    assert sagemaker.utils.aws_partition("us-iso-east-1") == "aws-iso"
    assert sagemaker.utils.aws_partition("us-isob-east-1") == "aws-iso-b"


def test_pop_out_unused_kwarg():
    # The given arg_name is in kwargs
    kwargs = dict(arg1=1, arg2=2)
    sagemaker.utils.pop_out_unused_kwarg("arg1", kwargs)
    assert "arg1" not in kwargs

    # The given arg_name is not in kwargs
    kwargs = dict(arg1=1, arg2=2)
    sagemaker.utils.pop_out_unused_kwarg("arg3", kwargs)
    assert len(kwargs) == 2


def test_to_string():
    var = 1
    assert sagemaker.utils.to_string(var) == "1"

    var = ParameterInteger(name="MyInt")
    assert sagemaker.utils.to_string(var).expr == {
        "Std:Join": {
            "On": "",
            "Values": [{"Get": "Parameters.MyInt"}],
        },
    }


@patch("time.sleep", return_value=None)
def test_start_waiting(patched_sleep, capfd):
    waiting_time = 1
    sagemaker.utils._start_waiting(waiting_time)
    out, _ = capfd.readouterr()

    assert "." * sagemaker.utils.WAITING_DOT_NUMBER in out


@patch("time.sleep", return_value=None)
def test_retry_with_backoff(patched_sleep):
    callable_func = Mock()

    # Invalid input
    with pytest.raises(ValueError) as value_err:
        retry_with_backoff(callable_func, 0)
    assert "The num_attempts must be >= 1" in str(value_err)
    callable_func.assert_not_called()

    # All retries fail
    run_err_msg = "Test Retry Error"
    callable_func.side_effect = RuntimeError(run_err_msg)
    with pytest.raises(RuntimeError) as run_err:
        retry_with_backoff(callable_func, 2)
    assert run_err_msg in str(run_err)

    # One retry passes
    func_return_val = "Test Return"
    callable_func.side_effect = [RuntimeError(run_err_msg), func_return_val]
    assert retry_with_backoff(callable_func, 2) == func_return_val

    # when retry on specific error, fail for other error on 1st try
    func_return_val = "Test Return"
    response = {"Error": {"Code": "ValidationException", "Message": "Could not find entity."}}
    error = botocore.exceptions.ClientError(error_response=response, operation_name="foo")
    callable_func.side_effect = [error, func_return_val]
    with pytest.raises(botocore.exceptions.ClientError) as run_err:
        retry_with_backoff(callable_func, 2, botocore_client_error_code="AccessDeniedException")
    assert "ValidationException" in str(run_err)

    # when retry on specific error, One retry passes
    func_return_val = "Test Return"
    response = {"Error": {"Code": "AccessDeniedException", "Message": "Access denied."}}
    error = botocore.exceptions.ClientError(error_response=response, operation_name="foo")
    callable_func.side_effect = [error, func_return_val]
    assert (
        retry_with_backoff(callable_func, 2, botocore_client_error_code="AccessDeniedException")
        == func_return_val
    )

    # No retry
    callable_func.side_effect = None
    callable_func.return_value = func_return_val
    assert retry_with_backoff(callable_func, 2) == func_return_val


def test_resolve_value_from_config():
    mock_config_logger = Mock()

    mock_info_logger = Mock()
    mock_config_logger.info = mock_info_logger
    # using a shorter name for inside the test
    sagemaker_session = MagicMock()
    sagemaker_session.sagemaker_config = {"SchemaVersion": "1.0"}
    config_key_path = "SageMaker.EndpointConfig.KmsKeyId"
    sagemaker_session.sagemaker_config.update(
        {"SageMaker": {"EndpointConfig": {"KmsKeyId": "CONFIG_VALUE"}}}
    )
    sagemaker_config = {
        "SchemaVersion": "1.0",
        "SageMaker": {"EndpointConfig": {"KmsKeyId": "CONFIG_VALUE"}},
    }

    # direct_input should be respected
    assert (
        resolve_value_from_config("INPUT", config_key_path, "DEFAULT_VALUE", sagemaker_session)
        == "INPUT"
    )

    assert resolve_value_from_config("INPUT", config_key_path, None, sagemaker_session) == "INPUT"

    assert (
        resolve_value_from_config("INPUT", "SageMaker.EndpointConfig.Tags", None, sagemaker_session)
        == "INPUT"
    )

    # Config or default values should be returned if no direct_input
    assert (
        resolve_value_from_config(None, None, "DEFAULT_VALUE", sagemaker_session) == "DEFAULT_VALUE"
    )

    assert (
        resolve_value_from_config(
            None, "SageMaker.EndpointConfig.Tags", "DEFAULT_VALUE", sagemaker_session
        )
        == "DEFAULT_VALUE"
    )

    assert (
        resolve_value_from_config(None, config_key_path, "DEFAULT_VALUE", sagemaker_session)
        == "CONFIG_VALUE"
    )

    assert resolve_value_from_config(None, None, None, sagemaker_session) is None

    # Config value from sagemaker_config should be returned
    # if no direct_input and sagemaker_session is None
    assert (
        resolve_value_from_config(None, config_key_path, None, None, sagemaker_config)
        == "CONFIG_VALUE"
    )

    # Different falsy direct_inputs
    assert resolve_value_from_config("", config_key_path, None, sagemaker_session) == ""

    assert resolve_value_from_config([], config_key_path, None, sagemaker_session) == []

    assert resolve_value_from_config(False, config_key_path, None, sagemaker_session) is False

    assert resolve_value_from_config({}, config_key_path, None, sagemaker_session) == {}

    # Different falsy config_values
    sagemaker_session.sagemaker_config.update({"SageMaker": {"EndpointConfig": {"KmsKeyId": ""}}})
    assert resolve_value_from_config(None, config_key_path, None, sagemaker_session) == ""

    mock_info_logger.reset_mock()


def test_get_sagemaker_config_value():
    mock_config_logger = Mock()

    mock_info_logger = Mock()
    mock_config_logger.info = mock_info_logger
    # using a shorter name for inside the test
    sagemaker_session = MagicMock()
    sagemaker_session.sagemaker_config = {"SchemaVersion": "1.0"}
    config_key_path = "SageMaker.EndpointConfig.KmsKeyId"
    sagemaker_session.sagemaker_config.update(
        {"SageMaker": {"EndpointConfig": {"KmsKeyId": "CONFIG_VALUE"}}}
    )
    sagemaker_config = {
        "SchemaVersion": "1.0",
        "SageMaker": {"EndpointConfig": {"KmsKeyId": "CONFIG_VALUE"}},
    }

    # Tests that the function returns the correct value when the key exists in the sagemaker_session configuration.
    assert (
        get_sagemaker_config_value(
            sagemaker_session=sagemaker_session, key=config_key_path, sagemaker_config=None
        )
        == "CONFIG_VALUE"
    )

    # Tests that the function correctly uses the sagemaker_config to get value for the requested
    # config_key_path when sagemaker_session is None.
    assert (
        get_sagemaker_config_value(
            sagemaker_session=None, key=config_key_path, sagemaker_config=sagemaker_config
        )
        == "CONFIG_VALUE"
    )

    # Tests that the function returns None when the key does not exist in the configuration.
    invalid_key = "inavlid_key"
    assert (
        get_sagemaker_config_value(
            sagemaker_session=sagemaker_session, key=invalid_key, sagemaker_config=sagemaker_config
        )
        is None
    )

    # Tests that the function returns None when sagemaker_session and sagemaker_config are None.
    assert (
        get_sagemaker_config_value(
            sagemaker_session=None, key=config_key_path, sagemaker_config=None
        )
        is None
    )


@patch("jsonschema.validate")
@pytest.mark.parametrize(
    "existing_value, config_value, default_value",
    [
        ("EXISTING_VALUE", "CONFIG_VALUE", "DEFAULT_VALUE"),
        (False, True, False),
        (False, False, True),
        (0, 1, 2),
    ],
)
def test_resolve_class_attribute_from_config(
    mock_validate, existing_value, config_value, default_value
):
    # using a shorter name for inside the test
    ss = MagicMock(settings=SessionSettings())

    class TestClass(object):
        def __init__(self, test_attribute=None, extra=None):
            self.test_attribute = test_attribute
            # the presence of an extra value that is set to None by default helps make sure a brand new
            # TestClass object is being created only in the right scenarios
            self.extra_attribute = extra

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                return self.__dict__ == other.__dict__
            else:
                return False

    dummy_config_path = "DUMMY.CONFIG.PATH"

    # with an existing config value

    ss.sagemaker_config = {"DUMMY": {"CONFIG": {"PATH": config_value}}}

    # instance exists and has value; config has value
    test_instance = TestClass(test_attribute=existing_value, extra="EXTRA_VALUE")
    assert resolve_class_attribute_from_config(
        TestClass, test_instance, "test_attribute", dummy_config_path, sagemaker_session=ss
    ) == TestClass(test_attribute=existing_value, extra="EXTRA_VALUE")

    # instance exists but doesnt have value; config has value
    test_instance = TestClass(extra="EXTRA_VALUE")
    assert resolve_class_attribute_from_config(
        TestClass, test_instance, "test_attribute", dummy_config_path, sagemaker_session=ss
    ) == TestClass(test_attribute=config_value, extra="EXTRA_VALUE")

    # instance doesnt exist; config has value
    test_instance = None
    assert resolve_class_attribute_from_config(
        TestClass, test_instance, "test_attribute", dummy_config_path, sagemaker_session=ss
    ) == TestClass(test_attribute=config_value, extra=None)

    # wrong attribute used
    test_instance = TestClass()
    with pytest.raises(TypeError):
        resolve_class_attribute_from_config(
            TestClass, test_instance, "other_attribute", dummy_config_path, sagemaker_session=ss
        )

    # instance doesnt exist; clazz doesnt exist
    test_instance = None
    assert (
        resolve_class_attribute_from_config(
            None, test_instance, "test_attribute", dummy_config_path, sagemaker_session=ss
        )
        is None
    )

    # instance doesnt exist; clazz isnt a class
    test_instance = None
    assert (
        resolve_class_attribute_from_config(
            "CLASS", test_instance, "test_attribute", dummy_config_path, sagemaker_session=ss
        )
        is None
    )

    # without an existing config value
    ss.sagemaker_config = {"DUMMY": {"CONFIG": {"SOMEOTHERPATH": config_value}}}
    # instance exists but doesnt have value; config doesnt have value
    test_instance = TestClass(extra="EXTRA_VALUE")
    assert resolve_class_attribute_from_config(
        TestClass, test_instance, "test_attribute", dummy_config_path, sagemaker_session=ss
    ) == TestClass(test_attribute=None, extra="EXTRA_VALUE")

    # instance exists but doesnt have value; config doesnt have value; default_value passed in
    test_instance = TestClass(extra="EXTRA_VALUE")
    assert resolve_class_attribute_from_config(
        TestClass,
        test_instance,
        "test_attribute",
        dummy_config_path,
        default_value=default_value,
        sagemaker_session=ss,
    ) == TestClass(test_attribute=default_value, extra="EXTRA_VALUE")

    # instance doesnt exist; config doesnt have value
    test_instance = None
    assert (
        resolve_class_attribute_from_config(
            TestClass, test_instance, "test_attribute", dummy_config_path, sagemaker_session=ss
        )
        is None
    )

    # instance doesnt exist; config doesnt have value; default_value passed in
    test_instance = None
    assert resolve_class_attribute_from_config(
        TestClass,
        test_instance,
        "test_attribute",
        dummy_config_path,
        default_value=default_value,
        sagemaker_session=ss,
    ) == TestClass(test_attribute=default_value, extra=None)


@patch("jsonschema.validate")
def test_resolve_nested_dict_value_from_config(mock_validate):
    # using a shorter name for inside the test
    ss = MagicMock(settings=SessionSettings())

    dummy_config_path = "DUMMY.CONFIG.PATH"
    # happy cases: return existing dict with existing values
    assert resolve_nested_dict_value_from_config(
        {"local": {"region_name": "us-west-2", "port": "123"}},
        ["local", "region_name"],
        dummy_config_path,
        default_value="DEFAULT_VALUE",
        sagemaker_session=ss,
    ) == {"local": {"region_name": "us-west-2", "port": "123"}}
    assert resolve_nested_dict_value_from_config(
        {"local": {"region_name": "us-west-2", "port": "123"}},
        ["local", "region_name"],
        dummy_config_path,
        default_value=None,
        sagemaker_session=ss,
    ) == {"local": {"region_name": "us-west-2", "port": "123"}}

    # happy case: return dict with config_value when it wasnt set in dict or was None

    ss.sagemaker_config = {"DUMMY": {"CONFIG": {"PATH": "CONFIG_VALUE"}}}
    assert resolve_nested_dict_value_from_config(
        {"local": {"port": "123"}},
        ["local", "region_name"],
        dummy_config_path,
        default_value="DEFAULT_VALUE",
        sagemaker_session=ss,
    ) == {"local": {"region_name": "CONFIG_VALUE", "port": "123"}}
    assert resolve_nested_dict_value_from_config(
        {}, ["local", "region_name"], dummy_config_path, default_value=None, sagemaker_session=ss
    ) == {"local": {"region_name": "CONFIG_VALUE"}}
    assert resolve_nested_dict_value_from_config(
        None, ["local", "region_name"], dummy_config_path, default_value=None, sagemaker_session=ss
    ) == {"local": {"region_name": "CONFIG_VALUE"}}
    assert resolve_nested_dict_value_from_config(
        {
            "local": {"region_name": "us-west-2", "port": "123"},
            "other": {"key": 1},
            "nest1": {"nest2": {"nest3": {"nest4a": "value", "nest4b": None}}},
        },
        ["nest1", "nest2", "nest3", "nest4b", "does_not", "exist"],
        dummy_config_path,
        default_value="DEFAULT_VALUE",
        sagemaker_session=ss,
    ) == {
        "local": {"region_name": "us-west-2", "port": "123"},
        "other": {"key": 1},
        "nest1": {
            "nest2": {
                "nest3": {"nest4a": "value", "nest4b": {"does_not": {"exist": "CONFIG_VALUE"}}}
            }
        },
    }

    # edge case: doesnt overwrite non-None and non-dict values
    dictionary = {
        "local": {"region_name": "us-west-2", "port": "123"},
        "other": {"key": 1},
        "nest1": {"nest2": {"nest3": {"nest4a": "value", "nest4b": None}}},
    }
    dictionary_copy = copy.deepcopy(dictionary)
    assert (
        resolve_nested_dict_value_from_config(
            dictionary,
            ["nest1", "nest2", "nest3", "nest4a", "does_not", "exist"],
            dummy_config_path,
            default_value="DEFAULT_VALUE",
            sagemaker_session=ss,
        )
        == dictionary_copy
    )
    assert (
        resolve_nested_dict_value_from_config(
            dictionary,
            ["other", "key"],
            dummy_config_path,
            default_value="DEFAULT_VALUE",
            sagemaker_session=ss,
        )
        == dictionary_copy
    )

    # without an existing config value
    ss.sagemaker_config = {"DUMMY": {"CONFIG": {"ANOTHER_PATH": "CONFIG_VALUE"}}}

    # happy case: return dict with default_value when it wasnt set in dict and in config
    assert resolve_nested_dict_value_from_config(
        {"local": {"port": "123"}},
        ["local", "region_name"],
        dummy_config_path,
        default_value="DEFAULT_VALUE",
        sagemaker_session=ss,
    ) == {"local": {"region_name": "DEFAULT_VALUE", "port": "123"}}

    # happy case: return dict as-is when value wasnt set in dict, in config, and as default
    assert resolve_nested_dict_value_from_config(
        {"local": {"port": "123"}},
        ["local", "region_name"],
        dummy_config_path,
        default_value=None,
        sagemaker_session=ss,
    ) == {"local": {"port": "123"}}
    assert (
        resolve_nested_dict_value_from_config(
            {},
            ["local", "region_name"],
            dummy_config_path,
            default_value=None,
            sagemaker_session=ss,
        )
        == {}
    )
    assert (
        resolve_nested_dict_value_from_config(
            None,
            ["local", "region_name"],
            dummy_config_path,
            default_value=None,
            sagemaker_session=ss,
        )
        is None
    )


def test_check_and_get_run_experiment_config():
    supplied_exp_cfg = {"ExperimentName": "my-supplied-exp-name", "RunName": "my-supplied-run-name"}
    run_exp_cfg = {"ExperimentName": "my-run-exp-name", "RunName": "my-run-run-name"}

    # No user supplied exp config and no current Run
    assert not _RunContext.get_current_run()
    exp_cfg1 = check_and_get_run_experiment_config(None)
    assert exp_cfg1 is None

    # With user supplied exp config and no current Run
    assert not _RunContext.get_current_run()
    exp_cfg2 = check_and_get_run_experiment_config(supplied_exp_cfg)
    assert exp_cfg2 == supplied_exp_cfg

    run = Mock()
    type(run).experiment_config = PropertyMock(return_value=run_exp_cfg)
    _RunContext.add_run_object(run)

    try:
        # No user supplied exp config and with current Run
        assert _RunContext.get_current_run().experiment_config == run_exp_cfg
        exp_cfg3 = check_and_get_run_experiment_config(None)
        assert exp_cfg3 == run_exp_cfg

        # With user supplied exp config and current Run
        assert _RunContext.get_current_run().experiment_config == run_exp_cfg
        exp_cfg4 = check_and_get_run_experiment_config(supplied_exp_cfg)
        assert exp_cfg4 == supplied_exp_cfg
    finally:
        # Clean up the global static variable in case it affects other tests
        _RunContext.drop_current_run()


def test_stringify_object():
    class MyTestClass:
        def __init__(self):
            self.blah = "blah"
            self.wtafigo = "eiifccreeeiuclkftdvttufbkhirtvvbhrieclghjiru"
            self.none_field = None
            self.dict_field = {"my": "dict"}
            self.list_field = ["1", 2, 3.0]
            self.list_dict_field = [{"hello": {"world": {"hello"}}}]

    stringified_class = (
        b"MyTestClass: {'blah': 'blah', 'wtafigo': 'eiifccreeeiuc"
        b"lkftdvttufbkhirtvvbhrieclghjiru', 'dict_field': {'my': 'dict'}, 'list_field'"
        b": ['1', 2, 3.0], 'list_dict_field': [{'hello': {'world': {'hello'}}}]}"
    )

    assert sagemaker.utils.stringify_object(MyTestClass()).encode() == stringified_class


class TestVolumeSizeSupported(TestCase):
    def test_volume_size_supported(self):
        instances_that_support_volume_size = [
            "ml.inf1.xlarge",
            "ml.inf1.2xlarge",
            "ml.inf1.6xlarge",
            "ml.inf1.24xlarge",
            "ml.inf2.xlarge",
            "ml.inf2.8xlarge",
            "ml.inf2.24xlarge",
            "ml.inf2.48xlarge",
            "ml.m5.large",
            "ml.m5.xlarge",
            "ml.m5.2xlarge",
            "ml.m5.4xlarge",
            "ml.m5.8xlarge",
            "ml.m5.12xlarge",
            "ml.m5.16xlarge",
            "ml.m5.24xlarge",
            "ml.m5.metal",
            "ml.c5.large",
            "ml.c5.xlarge",
            "ml.c5.2xlarge",
            "ml.c5.4xlarge",
            "ml.c5.9xlarge",
            "ml.c5.12xlarge",
            "ml.c5.18xlarge",
            "ml.c5.24xlarge",
            "ml.c5.metal",
            "ml.p3.2xlarge",
            "ml.p3.8xlarge",
            "ml.p3.16xlarge",
            "inf1.xlarge",
            "inf1.2xlarge",
            "inf1.6xlarge",
            "inf1.24xlarge",
            "inf2.xlarge",
            "inf2.8xlarge",
            "inf2.24xlarge",
            "inf2.48xlarge",
            "m5.large",
            "m5.xlarge",
            "m5.2xlarge",
            "m5.4xlarge",
            "m5.8xlarge",
            "m5.12xlarge",
            "m5.16xlarge",
            "m5.24xlarge",
            "m5.metal",
            "c5.large",
            "c5.xlarge",
            "c5.2xlarge",
            "c5.4xlarge",
            "c5.9xlarge",
            "c5.12xlarge",
            "c5.18xlarge",
            "c5.24xlarge",
            "c5.metal",
            "p3.2xlarge",
            "p3.8xlarge",
            "p3.16xlarge",
        ]

        for instance in instances_that_support_volume_size:
            self.assertTrue(volume_size_supported(instance))

    def test_volume_size_not_supported(self):
        instances_that_dont_support_volume_size = [
            "ml.p4d.xlarge",
            "ml.p4d.2xlarge",
            "ml.p4d.4xlarge",
            "ml.p4d.8xlarge",
            "ml.p4de.xlarge",
            "ml.p4de.2xlarge",
            "ml.p4de.4xlarge",
            "ml.p4de.8xlarge",
            "ml.g4dn.xlarge",
            "ml.g4dn.2xlarge",
            "ml.g4dn.4xlarge",
            "ml.g4dn.8xlarge",
            "ml.g5.xlarge",
            "ml.g5.2xlarge",
            "ml.g5.4xlarge",
            "ml.g5.8xlarge",
            "p4d.xlarge",
            "p4d.2xlarge",
            "p4d.4xlarge",
            "p4d.8xlarge",
            "p4de.xlarge",
            "p4de.2xlarge",
            "p4de.4xlarge",
            "p4de.8xlarge",
            "g4dn.xlarge",
            "g4dn.2xlarge",
            "g4dn.4xlarge",
            "g4dn.8xlarge",
            "g5.xlarge",
            "g5.2xlarge",
            "g5.4xlarge",
            "g5.8xlarge",
            "local",
            "local_gpu",
            ParameterString(name="InstanceType", default_value="ml.m4.xlarge"),
        ]

        for instance in instances_that_dont_support_volume_size:
            self.assertFalse(volume_size_supported(instance))

    def test_volume_size_badly_formatted(self):
        with pytest.raises(ValueError):
            volume_size_supported("blah")

        with pytest.raises(ValueError):
            volume_size_supported(float("inf"))

        with pytest.raises(ValueError):
            volume_size_supported("p2")

        with pytest.raises(ValueError):
            volume_size_supported({})

    def test_instance_family_from_full_instance_type(self):

        instance_type_to_family_test_dict = {
            "ml.p3.xlarge": "p3",
            "ml.inf1.4xlarge": "inf1",
            "ml.afbsadjfbasfb.sdkjfnsa": "afbsadjfbasfb",
            "ml_fdsfsdf.xlarge": "fdsfsdf",
            "ml_c2.4xlarge": "c2",
            "sdfasfdda": "",
            "local": "",
            "c2.xlarge": "",
            "": "",
        }

        for instance_type, family in instance_type_to_family_test_dict.items():
            self.assertEqual(family, get_instance_type_family(instance_type))


@pytest.fixture
def mock_custom_tarfile():
    class MockTarfile:
        def __init__(self, data_filter=False):
            self.data_filter = data_filter

        def extractall(self, path, members=None, filter=None):
            assert path == "/extract/path"
            if members is not None:
                assert next(members).name == "file.txt"

    return MockTarfile


def test_get_resolved_path():
    assert _get_resolved_path("path/to/file") == os.path.normpath(
        os.path.realpath(os.path.abspath("path/to/file"))
    )


@pytest.mark.parametrize("file_path, base, expected", [("file.txt", "/path/to/base", False)])
def test_is_bad_path(file_path, base, expected):
    assert _is_bad_path(file_path, base) == expected


@pytest.mark.parametrize(
    "link_name, base, expected", [("link_to_file.txt", "/path/to/base", False)]
)
def test_is_bad_link(link_name, base, expected):
    dummy_info = tarfile.TarInfo(name="dummy.txt")
    dummy_info.linkname = link_name
    assert _is_bad_link(dummy_info, base) == expected


@pytest.mark.parametrize(
    "data_filter, expected_extract_path", [(True, "/extract/path"), (False, "/extract/path")]
)
def test_custom_extractall_tarfile(mock_custom_tarfile, data_filter, expected_extract_path):
    tar = mock_custom_tarfile(data_filter)
    custom_extractall_tarfile(tar, "/extract/path")
