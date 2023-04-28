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
"""This module contains code to test SageMaker ``Artifacts``"""
from __future__ import absolute_import

import datetime
import logging
import time

import pytest

from sagemaker.lineage import artifact
from sagemaker.utils import retry_with_backoff

CREATION_VERIFICATION_WINDOW_MINUTES = 2


def test_create_delete(artifact_obj):
    # fixture does create and then delete, this test ensures it happens at least once
    assert artifact_obj.artifact_arn


def test_create_delete_with_association(artifact_obj_with_association):
    # fixture does create and then delete, this test ensures it happens at least once
    assert artifact_obj_with_association.artifact_arn


def test_save(artifact_obj, sagemaker_session):
    artifact_obj.properties = {"k3": "v3"}
    artifact_obj.properties_to_remove = ["k1"]

    artifact_obj.save()

    loaded = artifact.Artifact.load(
        artifact_arn=artifact_obj.artifact_arn, sagemaker_session=sagemaker_session
    )

    assert {"k3": "v3"} == loaded.properties


def test_load(artifact_obj, sagemaker_session):
    assert artifact_obj.artifact_name
    logging.info(f"loading {artifact_obj.artifact_name}")
    loaded = artifact.Artifact.load(
        artifact_arn=artifact_obj.artifact_arn, sagemaker_session=sagemaker_session
    )
    assert artifact_obj.artifact_arn == loaded.artifact_arn


def test_list(artifact_objs, sagemaker_session):
    slack = datetime.timedelta(minutes=1)
    now = datetime.datetime.now(datetime.timezone.utc)
    artifact_names = [art.artifact_name for art in artifact_objs]

    for sort_order in ["Ascending", "Descending"]:
        artifact_names_listed = [
            artifact_listed.artifact_name
            for artifact_listed in artifact.Artifact.list(
                created_after=now - slack,
                created_before=now + slack,
                sort_by="CreationTime",
                sort_order=sort_order,
                sagemaker_session=sagemaker_session,
            )
            if artifact_listed.artifact_name in artifact_names
        ]

    if sort_order == "Descending":
        artifact_names_listed = artifact_names_listed[::-1]
    assert artifact_names == artifact_names_listed
    # sanity check
    assert artifact_names


def test_list_by_type(artifact_objs, sagemaker_session):
    slack = datetime.timedelta(minutes=CREATION_VERIFICATION_WINDOW_MINUTES)
    now = datetime.datetime.now(datetime.timezone.utc)
    expected_name = list(
        filter(lambda x: x.artifact_type == "SDKIntegrationTestType2", artifact_objs)
    )[0].artifact_name
    artifact_names = [art.artifact_name for art in artifact_objs]

    artifact_names_listed = [
        artifact_listed.artifact_name
        for artifact_listed in artifact.Artifact.list(
            created_after=now - slack,
            artifact_type="SDKIntegrationTestType2",
            sagemaker_session=sagemaker_session,
        )
        if artifact_listed.artifact_name in artifact_names
    ]

    assert len(artifact_names_listed) == 1
    assert artifact_names_listed[0] == expected_name


@pytest.mark.skip("data inconsistency P61661075")
def test_get_artifact(static_dataset_artifact):
    s3_uri = static_dataset_artifact.source.source_uri
    expected_artifact = static_dataset_artifact.s3_uri_artifacts(s3_uri=s3_uri)
    for ar in expected_artifact["ArtifactSummaries"]:
        assert ar.get("Source")["SourceUri"] == s3_uri


def test_downstream_trials(trial_associated_artifact, trial_obj, sagemaker_session):
    # allow trial components to index, 30 seconds max
    def validate():
        for i in range(3):
            time.sleep(10)
            trials = trial_associated_artifact.downstream_trials(
                sagemaker_session=sagemaker_session
            )
            logging.info(f"Found {len(trials)} downstream trials.")
            if len(trials) > 0:
                break

        assert len(trials) == 1
        assert trial_obj.trial_name in trials

    retry_with_backoff(validate, num_attempts=3)


def test_downstream_trials_v2(trial_associated_artifact, trial_obj, sagemaker_session):
    trials = trial_associated_artifact.downstream_trials_v2()
    assert len(trials) == 1
    assert trial_obj.trial_name in trials


def test_upstream_trials(upstream_trial_associated_artifact, trial_obj, sagemaker_session):
    trials = upstream_trial_associated_artifact.upstream_trials()
    assert len(trials) == 1
    assert trial_obj.trial_name in trials


@pytest.mark.timeout(30)
def test_tag(artifact_obj, sagemaker_session):
    tag = {"Key": "foo", "Value": "bar"}
    artifact_obj.set_tag(tag)

    while True:
        actual_tags = sagemaker_session.sagemaker_client.list_tags(
            ResourceArn=artifact_obj.artifact_arn
        )["Tags"]
        if actual_tags:
            break
        time.sleep(5)
    # When sagemaker-client-config endpoint-url is passed as argument to hit some endpoints,
    # length of actual tags will be greater than 1
    assert len(actual_tags) > 0
    assert actual_tags[0] == tag


@pytest.mark.timeout(30)
def test_tags(artifact_obj, sagemaker_session):
    tags = [{"Key": "foo1", "Value": "bar1"}]
    artifact_obj.set_tags(tags)

    while True:
        actual_tags = sagemaker_session.sagemaker_client.list_tags(
            ResourceArn=artifact_obj.artifact_arn
        )["Tags"]
        if actual_tags:
            break
        time.sleep(5)
    # When sagemaker-client-config endpoint-url is passed as argument to hit some endpoints,
    # length of actual tags will be greater than 1
    assert len(actual_tags) > 0
    assert [actual_tags[-1]] == tags
