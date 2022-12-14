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
import uuid

from sagemaker.experiments._api_types import _TrialComponentStatusType
from tests.integ.sagemaker.experiments.helpers import EXP_INTEG_TEST_NAME_PREFIX
from sagemaker.experiments import _api_types, trial_component
from sagemaker.utilities.search_expression import Filter, Operator, SearchExpression


def test_create_delete(trial_component_obj):
    # Fixture does create / delete, just need to ensure called at least once
    assert trial_component_obj.trial_component_name
    assert trial_component_obj.input_artifacts == {}
    assert trial_component_obj.parameters == {}
    assert trial_component_obj.output_artifacts == {}


def test_create_tags(trial_component_obj, sagemaker_session):
    client = sagemaker_session.sagemaker_client
    while True:
        actual_tags = client.list_tags(ResourceArn=trial_component_obj.trial_component_arn)["Tags"]
        if actual_tags:
            break
    for tag in actual_tags:
        if "aws:tag" in tag.get("Key"):
            actual_tags.remove(tag)
    assert actual_tags == trial_component_obj.tags


def test_delete_with_force_disassociate(
    trial_component_with_force_disassociation_obj, sagemaker_session
):
    assert trial_component_with_force_disassociation_obj.trial_component_name
    trials = sagemaker_session.sagemaker_client.list_trials(
        TrialComponentName=trial_component_with_force_disassociation_obj.trial_component_name
    )["TrialSummaries"]
    assert len(trials) == 3


def test_save(trial_component_obj, sagemaker_session):
    trial_component_obj.display_name = str(uuid.uuid4())
    trial_component_obj.status = _api_types.TrialComponentStatus(
        primary_status=_TrialComponentStatusType.InProgress.value, message="Message"
    )
    trial_component_obj.start_time = datetime.datetime.now(
        datetime.timezone.utc
    ) - datetime.timedelta(days=1)
    trial_component_obj.end_time = datetime.datetime.now(datetime.timezone.utc)
    trial_component_obj.parameters = {"foo": "bar", "whizz": 100.1}
    trial_component_obj.input_artifacts = {
        "snizz": _api_types.TrialComponentArtifact(value="s3:/foo/bar", media_type="text/plain"),
        "snizz1": _api_types.TrialComponentArtifact(value="s3:/foo/bar2", media_type="text/plain2"),
    }
    trial_component_obj.output_artifacts = {
        "fly": _api_types.TrialComponentArtifact(value="s3:/sky/far", media_type="away/tomorrow"),
        "fly2": _api_types.TrialComponentArtifact(
            value="s3:/sky/far2", media_type="away/tomorrow2"
        ),
    }
    trial_component_obj.parameters_to_remove = ["foo"]
    trial_component_obj.input_artifacts_to_remove = ["snizz"]
    trial_component_obj.output_artifacts_to_remove = ["fly2"]

    trial_component_obj.save()

    loaded = trial_component._TrialComponent.load(
        trial_component_name=trial_component_obj.trial_component_name,
        sagemaker_session=sagemaker_session,
    )

    assert trial_component_obj.trial_component_name == loaded.trial_component_name
    assert trial_component_obj.status == loaded.status

    assert trial_component_obj.start_time - loaded.start_time < datetime.timedelta(seconds=1)
    assert trial_component_obj.end_time - loaded.end_time < datetime.timedelta(seconds=1)

    assert loaded.parameters == {"whizz": 100.1}
    assert loaded.input_artifacts == {
        "snizz1": _api_types.TrialComponentArtifact(value="s3:/foo/bar2", media_type="text/plain2")
    }
    assert loaded.output_artifacts == {
        "fly": _api_types.TrialComponentArtifact(value="s3:/sky/far", media_type="away/tomorrow")
    }


def test_load(trial_component_obj, sagemaker_session):
    loaded = trial_component._TrialComponent.load(
        trial_component_name=trial_component_obj.trial_component_name,
        sagemaker_session=sagemaker_session,
    )
    assert trial_component_obj.trial_component_arn == loaded.trial_component_arn


def test_list_sort(trial_components, sagemaker_session):
    slack = datetime.timedelta(minutes=1)
    now = datetime.datetime.now(datetime.timezone.utc)
    trial_component_names = [tc.trial_component_name for tc in trial_components]

    for sort_order in ["Ascending", "Descending"]:
        trial_component_names_listed = [
            s.trial_component_name
            for s in trial_component._TrialComponent.list(
                created_after=now - slack,
                created_before=now + slack,
                sort_by="CreationTime",
                sort_order=sort_order,
                sagemaker_session=sagemaker_session,
            )
            if s.trial_component_name in trial_component_names
        ]

    if sort_order == "Descending":
        trial_component_names_listed = trial_component_names_listed[::-1]
    assert trial_component_names == trial_component_names_listed
    assert trial_component_names  # sanity test


def test_search(sagemaker_session):
    trial_component_names_searched = []
    search_filter = Filter(
        name="TrialComponentName", operator=Operator.CONTAINS, value=EXP_INTEG_TEST_NAME_PREFIX
    )
    search_expression = SearchExpression(filters=[search_filter])
    for s in trial_component._TrialComponent.search(
        search_expression=search_expression, max_results=10, sagemaker_session=sagemaker_session
    ):
        trial_component_names_searched.append(s.trial_component_name)

    assert len(trial_component_names_searched) > 0
    assert trial_component_names_searched  # sanity test
