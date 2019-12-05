from __future__ import absolute_import

import uuid
import time

from sagemaker.analytics import ExperimentAnalytics


def test_experiment_analytics(sagemaker_session):
    sm = sagemaker_session.sagemaker_client

    experiment_name = "experiment-" + str(uuid.uuid4())
    sm.create_experiment(ExperimentName=experiment_name)

    for i in range(5):
        trial_name = "trial-" + str(uuid.uuid4())
        sm.create_trial(TrialName=trial_name, ExperimentName=experiment_name)
        trial_component_name = "tc-" + str(uuid.uuid4())
        sm.create_trial_component(TrialComponentName=trial_component_name, DisplayName="Training")
        sm.update_trial_component(
            TrialComponentName=trial_component_name, Parameters={"hp1": {"NumberValue": i}}
        )
        sm.associate_trial_component(TrialComponentName=trial_component_name, TrialName=trial_name)

    time.sleep(15)  # wait for search to get updated

    analytics = ExperimentAnalytics(
        experiment_name=experiment_name, sagemaker_session=sagemaker_session
    )

    assert list(analytics.dataframe().columns) == ["TrialComponentName", "DisplayName", "hp1"]


def test_experiment_analytics_pagination(sagemaker_session):
    sm = sagemaker_session.sagemaker_client

    experiment_name = "experiment" + str(uuid.uuid4())
    sm.create_experiment(ExperimentName=experiment_name)

    # Search returns 10 results by default. Add 20 trials to verify pagination,
    for i in range(20):
        trial_name = "trial-" + str(uuid.uuid4())
        sm.create_trial(TrialName=trial_name, ExperimentName=experiment_name)
        trial_component_name = "tc-" + str(uuid.uuid4())
        sm.create_trial_component(TrialComponentName=trial_component_name, DisplayName="Training")
        sm.update_trial_component(
            TrialComponentName=trial_component_name, Parameters={"hp1": {"NumberValue": i}}
        )
        sm.associate_trial_component(TrialComponentName=trial_component_name, TrialName=trial_name)

    time.sleep(15)  # wait for search to get updated  TODO [owen-t]: Replace with retry

    analytics = ExperimentAnalytics(
        experiment_name=experiment_name, sagemaker_session=sagemaker_session
    )

    assert list(analytics.dataframe().columns) == ["TrialComponentName", "DisplayName", "hp1"]
    assert (
        len(analytics.dataframe()) > 10
    )  # TODO [owen-t] Replace with == 20 and put test in retry block


def test_experiment_analytics_search_by_nested_filter(sagemaker_session):
    sm = sagemaker_session.sagemaker_client

    experiment_name = "experiment" + str(uuid.uuid4())
    sm.create_experiment(ExperimentName=experiment_name)

    for i in range(20):
        trial_name = "trial-" + str(uuid.uuid4())
        sm.create_trial(TrialName=trial_name, ExperimentName=experiment_name)
        trial_component_name = "tc-" + str(uuid.uuid4())
        sm.create_trial_component(TrialComponentName=trial_component_name, DisplayName="Training")
        sm.update_trial_component(
            TrialComponentName=trial_component_name, Parameters={"hp1": {"NumberValue": i}}
        )
        sm.associate_trial_component(TrialComponentName=trial_component_name, TrialName=trial_name)

    time.sleep(15)  # wait for search to get updated  TODO [owen-t]: Replace with retry

    search_exp = {
        "Filters": [
            {"Name": "Parents.ExperimentName", "Operator": "Equals", "Value": experiment_name},
            {"Name": "Parameters.hp1", "Operator": "GreaterThanOrEqualTo", "Value": "10"},
        ]
    }

    analytics = ExperimentAnalytics(
        sagemaker_session=sagemaker_session, search_expression=search_exp
    )

    assert list(analytics.dataframe().columns) == ["TrialComponentName", "DisplayName", "hp1"]
    assert (
        len(analytics.dataframe()) > 5
    )  # TODO [owen-t] Replace with == 10 and put test in retry block


def test_experiment_analytics_search_by_nested_filter_sort_ascending(sagemaker_session):
    sm = sagemaker_session.sagemaker_client

    experiment_name = "experiment" + str(uuid.uuid4())
    sm.create_experiment(ExperimentName=experiment_name)

    for i in range(20):
        trial_name = "trial-" + str(uuid.uuid4())
        sm.create_trial(TrialName=trial_name, ExperimentName=experiment_name)
        trial_component_name = "tc-" + str(uuid.uuid4())
        sm.create_trial_component(TrialComponentName=trial_component_name, DisplayName="Training")
        sm.update_trial_component(
            TrialComponentName=trial_component_name, Parameters={"hp1": {"NumberValue": i}}
        )
        sm.associate_trial_component(TrialComponentName=trial_component_name, TrialName=trial_name)

    time.sleep(15)  # wait for search to get updated  TODO [owen-t]: Replace with retry

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

    assert list(analytics.dataframe().columns) == ["TrialComponentName", "DisplayName", "hp1"]
    assert (
        len(analytics.dataframe()) > 5
    )  # TODO [owen-t] Replace with == 10 and put test in retry block
    assert list(analytics.dataframe()["hp1"].values) == sorted(analytics.dataframe()["hp1"].values)


def test_experiment_analytics_search_by_nested_filter_sort_descending(sagemaker_session):
    sm = sagemaker_session.sagemaker_client

    experiment_name = "experiment" + str(uuid.uuid4())
    sm.create_experiment(ExperimentName=experiment_name)

    for i in range(20):
        trial_name = "trial-" + str(uuid.uuid4())
        sm.create_trial(TrialName=trial_name, ExperimentName=experiment_name)
        trial_component_name = "tc-" + str(uuid.uuid4())
        sm.create_trial_component(TrialComponentName=trial_component_name, DisplayName="Training")
        sm.update_trial_component(
            TrialComponentName=trial_component_name, Parameters={"hp1": {"NumberValue": i}}
        )
        sm.associate_trial_component(TrialComponentName=trial_component_name, TrialName=trial_name)

    time.sleep(15)  # wait for search to get updated  TODO [owen-t]: Replace with retry

    search_exp = {
        "Filters": [
            {"Name": "Parents.ExperimentName", "Operator": "Equals", "Value": experiment_name},
            {"Name": "Parameters.hp1", "Operator": "GreaterThanOrEqualTo", "Value": "10"},
        ]
    }

    analytics = ExperimentAnalytics(
        sagemaker_session=sagemaker_session, search_expression=search_exp, sort_by="Parameters.hp1"
    )

    assert list(analytics.dataframe().columns) == ["TrialComponentName", "DisplayName", "hp1"]
    assert (
        len(analytics.dataframe()) > 5
    )  # TODO [owen-t] Replace with == 10 and put test in retry block
    assert (
        list(analytics.dataframe()["hp1"].values)
        == sorted(analytics.dataframe()["hp1"].values)[::-1]
    )
