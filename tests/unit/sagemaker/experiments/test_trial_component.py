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

import datetime
import unittest.mock

from unittest.mock import patch

import botocore

from sagemaker import Session
from sagemaker.experiments import _api_types
from sagemaker.experiments._api_types import (
    TrialComponentSearchResult,
    Parent,
    _TrialComponentStatusType,
)
from sagemaker.experiments.trial_component import _TrialComponent


def test_create(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    client.create_trial_component.return_value = {
        "TrialComponentArn": "bazz",
    }
    obj = _TrialComponent.create(
        trial_component_name="foo", display_name="bar", sagemaker_session=sagemaker_session
    )
    client.create_trial_component.assert_called_with(TrialComponentName="foo", DisplayName="bar")
    assert "foo" == obj.trial_component_name
    assert "bar" == obj.display_name
    assert "bazz" == obj.trial_component_arn


def test_create_with_tags(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    client.create_trial_component.return_value = {
        "TrialComponentArn": "bazz",
    }
    tags = [{"Key": "foo", "Value": "bar"}]
    _TrialComponent.create(
        trial_component_name="foo",
        display_name="bar",
        sagemaker_session=sagemaker_session,
        tags=tags,
    )
    client.create_trial_component.assert_called_with(
        TrialComponentName="foo", DisplayName="bar", Tags=tags
    )


def test_load(sagemaker_session):
    now = datetime.datetime.now(datetime.timezone.utc)
    client = sagemaker_session.sagemaker_client
    client.describe_trial_component.return_value = {
        "TrialComponentArn": "A",
        "TrialComponentName": "B",
        "DisplayName": "C",
        "Status": {"PrimaryStatus": _TrialComponentStatusType.InProgress.value, "Message": "D"},
        "Parameters": {"E": {"NumberValue": 1.0}, "F": {"StringValue": "G"}},
        "InputArtifacts": {"H": {"Value": "s3://foo/bar", "MediaType": "text/plain"}},
        "OutputArtifacts": {"I": {"Value": "s3://whizz/bang", "MediaType": "text/plain"}},
        "Metrics": [
            {
                "MetricName": "J",
                "Count": 1,
                "Min": 1.0,
                "Max": 2.0,
                "Avg": 3.0,
                "StdDev": 4.0,
                "SourceArn": "K",
                "Timestamp": now,
            }
        ],
    }
    obj = _TrialComponent.load(trial_component_name="foo", sagemaker_session=sagemaker_session)
    client.describe_trial_component.assert_called_with(TrialComponentName="foo")
    assert "A" == obj.trial_component_arn
    assert "B" == obj.trial_component_name
    assert "C" == obj.display_name
    assert (
        _api_types.TrialComponentStatus(
            primary_status=_TrialComponentStatusType.InProgress.value, message="D"
        )
        == obj.status
    )
    assert {"E": 1.0, "F": "G"} == obj.parameters
    assert {"H": _api_types.TrialComponentArtifact(value="s3://foo/bar", media_type="text/plain")}
    assert {
        "I": _api_types.TrialComponentArtifact(value="s3://whizz/bang", media_type="text/plain")
    }
    assert [
        _api_types.TrialComponentMetricSummary(
            metric_name="J",
            count=1,
            min=1.0,
            max=2.0,
            avg=3.0,
            std_dev=4.0,
            source_arn="K",
            timestamp=now,
        )
    ]


def test_save(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    obj = _TrialComponent(
        sagemaker_session,
        trial_component_name="foo",
        display_name="bar",
        parameters_to_remove=["E"],
        input_artifacts_to_remove=["F"],
        output_artifacts_to_remove=["G"],
    )
    client.update_trial_component.return_value = {}
    obj.save()

    client.update_trial_component.assert_called_with(
        TrialComponentName="foo",
        DisplayName="bar",
        Parameters={},
        ParametersToRemove=["E"],
        InputArtifacts={},
        InputArtifactsToRemove=["F"],
        OutputArtifacts={},
        OutputArtifactsToRemove=["G"],
    )


def test_delete(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    obj = _TrialComponent(sagemaker_session, trial_component_name="foo", display_name="bar")
    client.delete_trial_component.return_value = {}
    obj.delete()
    client.delete_trial_component.assert_called_with(TrialComponentName="foo")


def test_delete_with_force_disassociate(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    obj = _TrialComponent(sagemaker_session, trial_component_name="foo", display_name="bar")
    client.delete_trial_component.return_value = {}

    client.list_trials.side_effect = [
        {"TrialSummaries": [{"TrialName": "trial-1"}, {"TrialName": "trial-2"}], "NextToken": "a"},
        {"TrialSummaries": [{"TrialName": "trial-3"}, {"TrialName": "trial-4"}]},
    ]

    obj.delete(force_disassociate=True)
    expected_calls = [
        unittest.mock.call(TrialName="trial-1", TrialComponentName="foo"),
        unittest.mock.call(TrialName="trial-2", TrialComponentName="foo"),
        unittest.mock.call(TrialName="trial-3", TrialComponentName="foo"),
        unittest.mock.call(TrialName="trial-4", TrialComponentName="foo"),
    ]
    assert expected_calls == client.disassociate_trial_component.mock_calls
    client.delete_trial_component.assert_called_with(TrialComponentName="foo")


def test_list(sagemaker_session):
    start_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
    end_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=2)
    creation_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=3)
    last_modified_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=4)

    client = sagemaker_session.sagemaker_client
    client.list_trial_components.side_effect = [
        {
            "TrialComponentSummaries": [
                {
                    "TrialComponentName": "A" + str(i),
                    "TrialComponentArn": "B" + str(i),
                    "DisplayName": "C" + str(i),
                    "SourceArn": "D" + str(i),
                    "Status": {
                        "PrimaryStatus": _TrialComponentStatusType.InProgress.value,
                        "Message": "E" + str(i),
                    },
                    "StartTime": start_time + datetime.timedelta(hours=i),
                    "EndTime": end_time + datetime.timedelta(hours=i),
                    "CreationTime": creation_time + datetime.timedelta(hours=i),
                    "LastModifiedTime": last_modified_time + datetime.timedelta(hours=i),
                    "LastModifiedBy": {},
                }
                for i in range(10)
            ],
            "NextToken": "100",
        },
        {
            "TrialComponentSummaries": [
                {
                    "TrialComponentName": "A" + str(i),
                    "TrialComponentArn": "B" + str(i),
                    "DisplayName": "C" + str(i),
                    "SourceArn": "D" + str(i),
                    "Status": {
                        "PrimaryStatus": _TrialComponentStatusType.InProgress.value,
                        "Message": "E" + str(i),
                    },
                    "StartTime": start_time + datetime.timedelta(hours=i),
                    "EndTime": end_time + datetime.timedelta(hours=i),
                    "CreationTime": creation_time + datetime.timedelta(hours=i),
                    "LastModifiedTime": last_modified_time + datetime.timedelta(hours=i),
                    "LastModifiedBy": {},
                }
                for i in range(10, 20)
            ]
        },
    ]

    expected = [
        _api_types.TrialComponentSummary(
            trial_component_name="A" + str(i),
            trial_component_arn="B" + str(i),
            display_name="C" + str(i),
            source_arn="D" + str(i),
            status=_api_types.TrialComponentStatus(
                primary_status=_TrialComponentStatusType.InProgress.value, message="E" + str(i)
            ),
            start_time=start_time + datetime.timedelta(hours=i),
            end_time=end_time + datetime.timedelta(hours=i),
            creation_time=creation_time + datetime.timedelta(hours=i),
            last_modified_time=last_modified_time + datetime.timedelta(hours=i),
            last_modified_by={},
        )
        for i in range(20)
    ]
    result = list(
        _TrialComponent.list(
            sagemaker_session=sagemaker_session,
            source_arn="foo",
            sort_by="CreationTime",
            sort_order="Ascending",
        )
    )

    assert expected == result
    expected_calls = [
        unittest.mock.call(SortBy="CreationTime", SortOrder="Ascending", SourceArn="foo"),
        unittest.mock.call(
            NextToken="100", SortBy="CreationTime", SortOrder="Ascending", SourceArn="foo"
        ),
    ]
    assert expected_calls == client.list_trial_components.mock_calls


def test_list_empty(sagemaker_session):
    sagemaker_session.sagemaker_client.list_trial_components.return_value = {
        "TrialComponentSummaries": []
    }
    assert [] == list(_TrialComponent.list(sagemaker_session=sagemaker_session))


def test_list_trial_components_call_args(sagemaker_session):
    created_before = datetime.datetime(1999, 10, 12, 0, 0, 0)
    created_after = datetime.datetime(1990, 10, 12, 0, 0, 0)
    trial_name = "foo-trial"
    experiment_name = "foo-experiment"
    next_token = "thetoken"
    max_results = 99

    client = sagemaker_session.sagemaker_client
    client.list_trial_components.return_value = {}
    assert [] == list(
        _TrialComponent.list(
            sagemaker_session=sagemaker_session,
            trial_name=trial_name,
            experiment_name=experiment_name,
            created_before=created_before,
            created_after=created_after,
            next_token=next_token,
            max_results=max_results,
            sort_by="CreationTime",
            sort_order="Ascending",
        )
    )

    expected_calls = [
        unittest.mock.call(
            TrialName="foo-trial",
            ExperimentName="foo-experiment",
            CreatedBefore=created_before,
            CreatedAfter=created_after,
            SortBy="CreationTime",
            SortOrder="Ascending",
            NextToken="thetoken",
            MaxResults=99,
        )
    ]
    assert expected_calls == client.list_trial_components.mock_calls


@patch("sagemaker.experiments.trial_component._TrialComponent.load")
@patch("sagemaker.experiments.trial_component._TrialComponent.create")
def test_load_or_create_when_exist(mock_create, mock_load, sagemaker_session):
    tc_name = "tc_name"
    exists_error = botocore.exceptions.ClientError(
        error_response={
            "Error": {
                "Code": "ValidationException",
                "Message": "Experiment with name (experiment-xyz) already exists.",
            }
        },
        operation_name="foo",
    )
    mock_create.side_effect = exists_error
    _, is_existed = _TrialComponent._load_or_create(
        trial_component_name=tc_name, sagemaker_session=sagemaker_session
    )
    mock_create.assert_called_once_with(
        trial_component_name=tc_name,
        display_name=None,
        tags=None,
        sagemaker_session=sagemaker_session,
    )
    assert is_existed
    mock_load.assert_called_once_with(
        tc_name,
        sagemaker_session,
    )


@patch("sagemaker.experiments.trial_component._TrialComponent.load")
@patch("sagemaker.experiments.trial_component._TrialComponent.create")
def test_load_or_create_when_not_exist(mock_create, mock_load):
    sagemaker_session = Session()
    tc_name = "tc_name"

    _, is_existed = _TrialComponent._load_or_create(
        trial_component_name=tc_name, sagemaker_session=sagemaker_session
    )

    assert not is_existed
    mock_create.assert_called_once_with(
        trial_component_name=tc_name,
        display_name=None,
        tags=None,
        sagemaker_session=sagemaker_session,
    )
    mock_load.assert_not_called()


def test_search(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    client.search.return_value = {
        "Results": [
            {
                "TrialComponent": {
                    "TrialComponentName": "tc-1",
                    "TrialComponentArn": "arn::tc-1",
                    "DisplayName": "TC1",
                    "Parents": [
                        {
                            "ExperimentName": "e-1",
                            "TrialName": "t-1",
                        },
                        {
                            "ExperimentName": "e-2",
                            "TrialName": "t-2",
                        },
                    ],
                }
            },
            {
                "TrialComponent": {
                    "TrialComponentName": "tc-2",
                    "TrialComponentArn": "arn::tc-2",
                    "DisplayName": "TC2",
                }
            },
        ]
    }
    expected = [
        TrialComponentSearchResult(
            trial_component_name="tc-1",
            trial_component_arn="arn::tc-1",
            display_name="TC1",
            parents=[
                Parent(experiment_name="e-1", trial_name="t-1"),
                Parent(experiment_name="e-2", trial_name="t-2"),
            ],
        ),
        TrialComponentSearchResult(
            trial_component_name="tc-2", trial_component_arn="arn::tc-2", display_name="TC2"
        ),
    ]
    assert expected == list(_TrialComponent.search(sagemaker_session=sagemaker_session))


def test_trial_component_is_associated_to_trial(sagemaker_session):
    obj = _TrialComponent(sagemaker_session, trial_component_name="tc-1")
    sagemaker_session.sagemaker_client.search.return_value = {
        "Results": [
            {
                "TrialComponent": {
                    "Parents": [{"ExperimentName": "e-1", "TrialName": "t-1"}],
                    "TrialComponentName": "tc-1",
                }
            }
        ]
    }

    assert obj._trial_component_is_associated_to_trial("tc-1", "t-1", sagemaker_session) is True


def test_trial_component_is_not_associated_to_trial(sagemaker_session):
    obj = _TrialComponent(sagemaker_session, trial_component_name="tc-1")
    sagemaker_session.sagemaker_client.search.return_value = {"Results": []}

    assert obj._trial_component_is_associated_to_trial("tc-1", "t-1", sagemaker_session) is False
