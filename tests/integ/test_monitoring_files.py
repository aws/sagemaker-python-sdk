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

import os
import uuid

import pytest

import tests.integ
import tests.integ.timeout
from sagemaker.model_monitor import Statistics, Constraints, ConstraintViolations
from sagemaker.s3 import S3Uploader
from tests.integ.kms_utils import get_or_create_kms_key


@pytest.fixture(scope="module")
def monitoring_files_kms_key(sagemaker_session):
    return get_or_create_kms_key(sagemaker_session=sagemaker_session)


def test_statistics_object_creation_from_file_path_with_customizations(
    sagemaker_session, monitoring_files_kms_key
):
    statistics = Statistics.from_file_path(
        statistics_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/statistics.json"),
        kms_key=monitoring_files_kms_key,
        sagemaker_session=sagemaker_session,
    )

    assert statistics.file_s3_uri.startswith("s3://")
    assert statistics.file_s3_uri.endswith("statistics.json")

    assert statistics.body_dict["dataset"]["item_count"] == 418


def test_statistics_object_creation_from_file_path_without_customizations(sagemaker_session):
    statistics = Statistics.from_file_path(
        statistics_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/statistics.json"),
        sagemaker_session=sagemaker_session,
    )

    assert statistics.file_s3_uri.startswith("s3://")
    assert statistics.file_s3_uri.endswith("statistics.json")

    assert statistics.body_dict["dataset"]["item_count"] == 418


def test_statistics_object_creation_from_string_with_customizations(
    sagemaker_session, monitoring_files_kms_key
):
    with open(os.path.join(tests.integ.DATA_DIR, "monitor/statistics.json"), "r") as f:
        file_body = f.read()

    statistics = Statistics.from_string(
        statistics_file_string=file_body,
        kms_key=monitoring_files_kms_key,
        file_name="statistics.json",
        sagemaker_session=sagemaker_session,
    )

    assert statistics.file_s3_uri.startswith("s3://")
    assert statistics.file_s3_uri.endswith("statistics.json")

    assert statistics.body_dict["dataset"]["item_count"] == 418


def test_statistics_object_creation_from_string_without_customizations(sagemaker_session):
    with open(os.path.join(tests.integ.DATA_DIR, "monitor/statistics.json"), "r") as f:
        file_body = f.read()

    statistics = Statistics.from_string(
        statistics_file_string=file_body, sagemaker_session=sagemaker_session
    )

    assert statistics.file_s3_uri.startswith("s3://")
    assert statistics.file_s3_uri.endswith("statistics.json")

    assert statistics.body_dict["dataset"]["item_count"] == 418


def test_statistics_object_creation_from_s3_uri_with_customizations(
    sagemaker_session, monitoring_files_kms_key
):
    with open(os.path.join(tests.integ.DATA_DIR, "monitor/statistics.json"), "r") as f:
        file_body = f.read()

    file_name = "statistics.json"
    desired_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "integ-test-test-monitoring-files",
        str(uuid.uuid4()),
        file_name,
    )

    s3_uri = S3Uploader.upload_string_as_file_body(
        body=file_body,
        desired_s3_uri=desired_s3_uri,
        kms_key=monitoring_files_kms_key,
        sagemaker_session=sagemaker_session,
    )

    statistics = Statistics.from_s3_uri(
        statistics_file_s3_uri=s3_uri,
        kms_key=monitoring_files_kms_key,
        sagemaker_session=sagemaker_session,
    )

    assert statistics.file_s3_uri.startswith("s3://")
    assert statistics.file_s3_uri.endswith("statistics.json")

    assert statistics.body_dict["dataset"]["item_count"] == 418


def test_statistics_object_creation_from_s3_uri_without_customizations(sagemaker_session):
    with open(os.path.join(tests.integ.DATA_DIR, "monitor/statistics.json"), "r") as f:
        file_body = f.read()

    file_name = "statistics.json"
    desired_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "integ-test-test-monitoring-files",
        str(uuid.uuid4()),
        file_name,
    )

    s3_uri = S3Uploader.upload_string_as_file_body(
        body=file_body, desired_s3_uri=desired_s3_uri, sagemaker_session=sagemaker_session
    )

    statistics = Statistics.from_s3_uri(
        statistics_file_s3_uri=s3_uri, sagemaker_session=sagemaker_session
    )

    assert statistics.file_s3_uri.startswith("s3://")
    assert statistics.file_s3_uri.endswith("statistics.json")

    assert statistics.body_dict["dataset"]["item_count"] == 418


def test_constraints_object_creation_from_file_path_with_customizations(
    sagemaker_session, monitoring_files_kms_key
):
    constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/constraints.json"),
        kms_key=monitoring_files_kms_key,
        sagemaker_session=sagemaker_session,
    )

    assert constraints.file_s3_uri.startswith("s3://")
    assert constraints.file_s3_uri.endswith("constraints.json")

    assert constraints.body_dict["monitoring_config"]["evaluate_constraints"] == "Enabled"

    constraints.set_monitoring(False)

    assert constraints.body_dict["monitoring_config"]["evaluate_constraints"] == "Disabled"

    constraints.set_monitoring(True, "message")

    assert (
        constraints.body_dict["features"][0]["string_constraints"]["monitoring_config_overrides"][
            "evaluate_constraints"
        ]
        == "Enabled"
    )

    constraints.set_monitoring(True, "second_message")

    assert (
        constraints.body_dict["features"][0]["string_constraints"]["monitoring_config_overrides"][
            "evaluate_constraints"
        ]
        == "Enabled"
    )

    constraints.save()

    new_constraints = Constraints.from_s3_uri(
        constraints.file_s3_uri, sagemaker_session=sagemaker_session
    )

    assert new_constraints.body_dict["monitoring_config"]["evaluate_constraints"] == "Disabled"


def test_constraints_object_creation_from_file_path_without_customizations(sagemaker_session):
    constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/constraints.json"),
        sagemaker_session=sagemaker_session,
    )

    assert constraints.file_s3_uri.startswith("s3://")
    assert constraints.file_s3_uri.endswith("constraints.json")

    assert constraints.body_dict["monitoring_config"]["evaluate_constraints"] == "Enabled"


def test_constraints_object_creation_from_string_with_customizations(
    sagemaker_session, monitoring_files_kms_key
):
    with open(os.path.join(tests.integ.DATA_DIR, "monitor/constraints.json"), "r") as f:
        file_body = f.read()

    constraints = Constraints.from_string(
        constraints_file_string=file_body,
        kms_key=monitoring_files_kms_key,
        file_name="constraints.json",
        sagemaker_session=sagemaker_session,
    )

    assert constraints.file_s3_uri.startswith("s3://")
    assert constraints.file_s3_uri.endswith("constraints.json")

    assert constraints.body_dict["monitoring_config"]["evaluate_constraints"] == "Enabled"


def test_constraints_object_creation_from_string_without_customizations(sagemaker_session):
    with open(os.path.join(tests.integ.DATA_DIR, "monitor/constraints.json"), "r") as f:
        file_body = f.read()

    constraints = Constraints.from_string(
        constraints_file_string=file_body, sagemaker_session=sagemaker_session
    )

    assert constraints.file_s3_uri.startswith("s3://")
    assert constraints.file_s3_uri.endswith("constraints.json")

    assert constraints.body_dict["monitoring_config"]["evaluate_constraints"] == "Enabled"


def test_constraints_object_creation_from_s3_uri_with_customizations(
    sagemaker_session, monitoring_files_kms_key
):
    with open(os.path.join(tests.integ.DATA_DIR, "monitor/constraints.json"), "r") as f:
        file_body = f.read()

    file_name = "constraints.json"
    desired_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "integ-test-test-monitoring-files",
        str(uuid.uuid4()),
        file_name,
    )

    s3_uri = S3Uploader.upload_string_as_file_body(
        body=file_body,
        desired_s3_uri=desired_s3_uri,
        kms_key=monitoring_files_kms_key,
        sagemaker_session=sagemaker_session,
    )

    constraints = Constraints.from_s3_uri(
        constraints_file_s3_uri=s3_uri,
        kms_key=monitoring_files_kms_key,
        sagemaker_session=sagemaker_session,
    )

    assert constraints.file_s3_uri.startswith("s3://")
    assert constraints.file_s3_uri.endswith("constraints.json")

    assert constraints.body_dict["monitoring_config"]["evaluate_constraints"] == "Enabled"


def test_constraints_object_creation_from_s3_uri_without_customizations(sagemaker_session):
    with open(os.path.join(tests.integ.DATA_DIR, "monitor/constraints.json"), "r") as f:
        file_body = f.read()

    file_name = "constraints.json"
    desired_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "integ-test-test-monitoring-files",
        str(uuid.uuid4()),
        file_name,
    )

    s3_uri = S3Uploader.upload_string_as_file_body(
        body=file_body, desired_s3_uri=desired_s3_uri, sagemaker_session=sagemaker_session
    )

    constraints = Constraints.from_s3_uri(
        constraints_file_s3_uri=s3_uri, sagemaker_session=sagemaker_session
    )

    assert constraints.file_s3_uri.startswith("s3://")
    assert constraints.file_s3_uri.endswith("constraints.json")

    assert constraints.body_dict["monitoring_config"]["evaluate_constraints"] == "Enabled"


def test_constraint_violations_object_creation_from_file_path_with_customizations(
    sagemaker_session, monitoring_files_kms_key
):
    constraint_violations = ConstraintViolations.from_file_path(
        constraint_violations_file_path=os.path.join(
            tests.integ.DATA_DIR, "monitor/constraint_violations.json"
        ),
        kms_key=monitoring_files_kms_key,
        sagemaker_session=sagemaker_session,
    )

    assert constraint_violations.file_s3_uri.startswith("s3://")
    assert constraint_violations.file_s3_uri.endswith("constraint_violations.json")

    assert constraint_violations.body_dict["violations"][0]["feature_name"] == "store_and_fwd_flag"


def test_constraint_violations_object_creation_from_file_path_without_customizations(
    sagemaker_session,
):
    constraint_violations = ConstraintViolations.from_file_path(
        constraint_violations_file_path=os.path.join(
            tests.integ.DATA_DIR, "monitor/constraint_violations.json"
        ),
        sagemaker_session=sagemaker_session,
    )

    assert constraint_violations.file_s3_uri.startswith("s3://")
    assert constraint_violations.file_s3_uri.endswith("constraint_violations.json")

    assert constraint_violations.body_dict["violations"][0]["feature_name"] == "store_and_fwd_flag"


def test_constraint_violations_object_creation_from_string_with_customizations(
    sagemaker_session, monitoring_files_kms_key
):
    with open(os.path.join(tests.integ.DATA_DIR, "monitor/constraint_violations.json"), "r") as f:
        file_body = f.read()

    constraint_violations = ConstraintViolations.from_string(
        constraint_violations_file_string=file_body,
        kms_key=monitoring_files_kms_key,
        file_name="constraint_violations.json",
        sagemaker_session=sagemaker_session,
    )

    assert constraint_violations.file_s3_uri.startswith("s3://")
    assert constraint_violations.file_s3_uri.endswith("constraint_violations.json")

    assert constraint_violations.body_dict["violations"][0]["feature_name"] == "store_and_fwd_flag"


def test_constraint_violations_object_creation_from_string_without_customizations(
    sagemaker_session,
):
    with open(os.path.join(tests.integ.DATA_DIR, "monitor/constraint_violations.json"), "r") as f:
        file_body = f.read()

    constraint_violations = ConstraintViolations.from_string(
        constraint_violations_file_string=file_body, sagemaker_session=sagemaker_session
    )

    assert constraint_violations.file_s3_uri.startswith("s3://")
    assert constraint_violations.file_s3_uri.endswith("constraint_violations.json")

    assert constraint_violations.body_dict["violations"][0]["feature_name"] == "store_and_fwd_flag"


def test_constraint_violations_object_creation_from_s3_uri_with_customizations(
    sagemaker_session, monitoring_files_kms_key
):
    with open(os.path.join(tests.integ.DATA_DIR, "monitor/constraint_violations.json"), "r") as f:
        file_body = f.read()

    file_name = "constraint_violations.json"
    desired_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "integ-test-test-monitoring-files",
        str(uuid.uuid4()),
        file_name,
    )

    s3_uri = S3Uploader.upload_string_as_file_body(
        body=file_body,
        desired_s3_uri=desired_s3_uri,
        kms_key=monitoring_files_kms_key,
        sagemaker_session=sagemaker_session,
    )

    constraint_violations = ConstraintViolations.from_s3_uri(
        constraint_violations_file_s3_uri=s3_uri,
        kms_key=monitoring_files_kms_key,
        sagemaker_session=sagemaker_session,
    )

    assert constraint_violations.file_s3_uri.startswith("s3://")
    assert constraint_violations.file_s3_uri.endswith("constraint_violations.json")

    assert constraint_violations.body_dict["violations"][0]["feature_name"] == "store_and_fwd_flag"


def test_constraint_violations_object_creation_from_s3_uri_without_customizations(
    sagemaker_session,
):
    with open(os.path.join(tests.integ.DATA_DIR, "monitor/constraint_violations.json"), "r") as f:
        file_body = f.read()

    file_name = "constraint_violations.json"
    desired_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "integ-test-test-monitoring-files",
        str(uuid.uuid4()),
        file_name,
    )

    s3_uri = S3Uploader.upload_string_as_file_body(
        body=file_body, desired_s3_uri=desired_s3_uri, sagemaker_session=sagemaker_session
    )

    constraint_violations = ConstraintViolations.from_s3_uri(
        constraint_violations_file_s3_uri=s3_uri, sagemaker_session=sagemaker_session
    )

    assert constraint_violations.file_s3_uri.startswith("s3://")
    assert constraint_violations.file_s3_uri.endswith("constraint_violations.json")

    assert constraint_violations.body_dict["violations"][0]["feature_name"] == "store_and_fwd_flag"
