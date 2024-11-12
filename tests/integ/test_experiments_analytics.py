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

import time
import uuid
from contextlib import contextmanager

import pytest

from sagemaker.analytics import ExperimentAnalytics
from tests import integ


@contextmanager
def experiment(sagemaker_session):
    sm = sagemaker_session.sagemaker_client
    trials = {}  # for resource cleanup

    experiment_name = "experiment-" + str(uuid.uuid4())
    try:
        sm.create_experiment(ExperimentName=experiment_name)

        # Search returns 10 results by default. Add 20 trials to verify pagination.
        for i in range(20):
            trial_name = "trial-" + str(uuid.uuid4())
            sm.create_trial(TrialName=trial_name, ExperimentName=experiment_name)

            trial_component_name = "tc-" + str(uuid.uuid4())
            trials[trial_name] = trial_component_name

            sm.create_trial_component(
                TrialComponentName=trial_component_name, DisplayName="Training"
            )
            sm.update_trial_component(
                TrialComponentName=trial_component_name, Parameters={"hp1": {"NumberValue": i}}
            )
            sm.associate_trial_component(
                TrialComponentName=trial_component_name, TrialName=trial_name
            )
            time.sleep(1)

        time.sleep(15)  # wait for search to get updated

        # allow search time thrice
        for _ in range(3):
            analytics = ExperimentAnalytics(
                experiment_name=experiment_name, sagemaker_session=sagemaker_session
            )

            if len(analytics.dataframe().columns) > 0:
                break

            time.sleep(15)

        yield experiment_name
    finally:
        _delete_resources(sm, experiment_name, trials)


@contextmanager
def experiment_with_artifacts(sagemaker_session):
    sm = sagemaker_session.sagemaker_client
    trials = {}  # for resource cleanup

    experiment_name = "experiment-" + str(uuid.uuid4())
    try:
        sm.create_experiment(ExperimentName=experiment_name)

        # Search returns 10 results by default. Add 20 trials to verify pagination.
        for i in range(20):
            trial_name = "trial-" + str(uuid.uuid4())
            sm.create_trial(TrialName=trial_name, ExperimentName=experiment_name)

            trial_component_name = "tc-" + str(uuid.uuid4())
            trials[trial_name] = trial_component_name

            sm.create_trial_component(
                TrialComponentName=trial_component_name, DisplayName="Training"
            )
            sm.update_trial_component(
                TrialComponentName=trial_component_name,
                Parameters={"hp1": {"NumberValue": i}},
                InputArtifacts={
                    "inputArtifacts1": {"MediaType": "text/csv", "Value": "s3:/foo/bar1"}
                },
                OutputArtifacts={
                    "outputArtifacts1": {"MediaType": "text/plain", "Value": "s3:/foo/bar2"}
                },
            )
            sm.associate_trial_component(
                TrialComponentName=trial_component_name, TrialName=trial_name
            )
            time.sleep(1)

        time.sleep(15)  # wait for search to get updated

        # allow search time thrice
        for _ in range(3):
            analytics = ExperimentAnalytics(
                experiment_name=experiment_name, sagemaker_session=sagemaker_session
            )

            if len(analytics.dataframe().columns) > 0:
                break

            time.sleep(15)

        yield experiment_name
    finally:
        _delete_resources(sm, experiment_name, trials)


@pytest.mark.release
@pytest.mark.skipif(
    integ.test_region() == "us-east-2", reason="Currently issues in this region NonSDK related"
)
def test_experiment_analytics_artifacts(sagemaker_session):
    with experiment_with_artifacts(sagemaker_session) as experiment_name:
        analytics = ExperimentAnalytics(
            experiment_name=experiment_name, sagemaker_session=sagemaker_session
        )

        assert list(analytics.dataframe().columns) == [
            "TrialComponentName",
            "DisplayName",
            "hp1",
            "inputArtifacts1 - MediaType",
            "inputArtifacts1 - Value",
            "outputArtifacts1 - MediaType",
            "outputArtifacts1 - Value",
            "Trials",
            "Experiments",
        ]


def test_experiment_analytics_pagination(sagemaker_session):
    with experiment(sagemaker_session) as experiment_name:
        analytics = ExperimentAnalytics(
            experiment_name=experiment_name, sagemaker_session=sagemaker_session
        )

        assert list(analytics.dataframe().columns) == [
            "TrialComponentName",
            "DisplayName",
            "hp1",
            "Trials",
            "Experiments",
        ]
        assert (
            len(analytics.dataframe()) > 10
        )  # TODO [owen-t] Replace with == 20 and put test in retry block


def test_experiment_analytics_search_by_nested_filter(sagemaker_session):
    with experiment(sagemaker_session) as experiment_name:
        search_exp = {
            "Filters": [
                {"Name": "Parents.ExperimentName", "Operator": "Equals", "Value": experiment_name},
                {"Name": "Parameters.hp1", "Operator": "GreaterThanOrEqualTo", "Value": "10"},
            ]
        }

        analytics = ExperimentAnalytics(
            sagemaker_session=sagemaker_session, search_expression=search_exp
        )

        assert list(analytics.dataframe().columns) == [
            "TrialComponentName",
            "DisplayName",
            "hp1",
            "Trials",
            "Experiments",
        ]
        assert (
            len(analytics.dataframe()) > 5
        )  # TODO [owen-t] Replace with == 10 and put test in retry block


def test_experiment_analytics_search_by_nested_filter_sort_ascending(sagemaker_session):
    with experiment(sagemaker_session) as experiment_name:
        search_exp = {
            "Filters": [
                {"Name": "Parents.ExperimentName", "Operator": "Equals", "Value": experiment_name},
                {"Name": "Parameters.hp1", "Operator": "GreaterThanOrEqualTo", "Value": "10"},
            ]
        }

        analytics = ExperimentAnalytics(
            sagemaker_session=sagemaker_session,
            search_expression=search_exp,
            sort_by="Parameters.hp1",
            sort_order="Ascending",
        )

        assert list(analytics.dataframe().columns) == [
            "TrialComponentName",
            "DisplayName",
            "hp1",
            "Trials",
            "Experiments",
        ]
        assert (
            len(analytics.dataframe()) > 5
        )  # TODO [owen-t] Replace with == 10 and put test in retry block
        assert list(analytics.dataframe()["hp1"].values) == sorted(
            analytics.dataframe()["hp1"].values
        )


def test_experiment_analytics_search_by_nested_filter_sort_descending(sagemaker_session):
    with experiment(sagemaker_session) as experiment_name:
        search_exp = {
            "Filters": [
                {"Name": "Parents.ExperimentName", "Operator": "Equals", "Value": experiment_name},
                {"Name": "Parameters.hp1", "Operator": "GreaterThanOrEqualTo", "Value": "10"},
            ]
        }

        analytics = ExperimentAnalytics(
            sagemaker_session=sagemaker_session,
            search_expression=search_exp,
            sort_by="Parameters.hp1",
        )

        assert list(analytics.dataframe().columns) == [
            "TrialComponentName",
            "DisplayName",
            "hp1",
            "Trials",
            "Experiments",
        ]
        assert (
            len(analytics.dataframe()) > 5
        )  # TODO [owen-t] Replace with == 10 and put test in retry block
        assert (
            list(analytics.dataframe()["hp1"].values)
            == sorted(analytics.dataframe()["hp1"].values)[::-1]
        )


def _delete_resources(sagemaker_client, experiment_name, trials):
    for trial, tc in trials.items():
        with _ignore_resource_not_found(sagemaker_client):
            sagemaker_client.disassociate_trial_component(TrialName=trial, TrialComponentName=tc)
            _wait_for_trial_component_disassociation(sagemaker_client, tc)

        with _ignore_resource_not_found(sagemaker_client):
            sagemaker_client.delete_trial_component(TrialComponentName=tc)

        with _ignore_resource_not_found(sagemaker_client):
            sagemaker_client.delete_trial(TrialName=trial)

    with _ignore_resource_not_found(sagemaker_client):
        sagemaker_client.delete_experiment(ExperimentName=experiment_name)


@contextmanager
def _ignore_resource_not_found(sagemaker_client):
    try:
        yield
    except sagemaker_client.exceptions.ResourceNotFound:
        pass


def _wait_for_trial_component_disassociation(sagemaker_client, tc):
    # Sometimes it can take a bit of waiting for the trial component to be disassociated
    for _ in range(5):
        # Check that the trial component has been disassociated from the trial
        trials = sagemaker_client.list_trials(TrialComponentName=tc)["TrialSummaries"]
        if len(trials) == 0:
            break

        time.sleep(1)
