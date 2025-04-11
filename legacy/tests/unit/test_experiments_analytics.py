from __future__ import absolute_import

import mock
import pytest
import pandas as pd

from collections import OrderedDict

from sagemaker.analytics import ExperimentAnalytics


@pytest.fixture
def mock_session():
    return mock.Mock()


def trial_component(trial_component_name):
    return {
        "TrialComponentName": trial_component_name,
        "DisplayName": "Training",
        "Source": {"SourceArn": "some-source-arn"},
        "Parameters": {"hp1": {"NumberValue": 1.0}, "hp2": {"StringValue": "abc"}},
        "Metrics": [
            {
                "MetricName": "metric1",
                "Max": 5.0,
                "Min": 3.0,
                "Avg": 4.0,
                "StdDev": 1.0,
                "Last": 2.0,
                "Count": 2.0,
            },
            {
                "MetricName": "metric2",
                "Max": 10.0,
                "Min": 8.0,
                "Avg": 9.0,
                "StdDev": 0.05,
                "Last": 7.0,
                "Count": 2.0,
            },
        ],
        "InputArtifacts": {
            "inputArtifacts1": {"MediaType": "text/plain", "Value": "s3:/foo/bar1"},
            "inputArtifacts2": {"MediaType": "text/plain", "Value": "s3:/foo/bar2"},
        },
        "OutputArtifacts": {
            "outputArtifacts1": {"MediaType": "text/csv", "Value": "s3:/sky/far1"},
            "outputArtifacts2": {"MediaType": "text/csv", "Value": "s3:/sky/far2"},
        },
        "Parents": [{"TrialName": "trial1", "ExperimentName": "experiment1"}],
    }


def test_trial_analytics_dataframe_all(mock_session):
    mock_session.sagemaker_client.search.return_value = {
        "Results": [
            {"TrialComponent": trial_component("trial-1")},
            {"TrialComponent": trial_component("trial-2")},
        ]
    }
    analytics = ExperimentAnalytics(experiment_name="experiment1", sagemaker_session=mock_session)

    expected_dataframe = pd.DataFrame.from_dict(
        OrderedDict(
            [
                ("TrialComponentName", ["trial-1", "trial-2"]),
                ("DisplayName", ["Training", "Training"]),
                ("SourceArn", ["some-source-arn", "some-source-arn"]),
                ("hp1", [1.0, 1.0]),
                ("hp2", ["abc", "abc"]),
                ("metric1 - Min", [3.0, 3.0]),
                ("metric1 - Max", [5.0, 5.0]),
                ("metric1 - Avg", [4.0, 4.0]),
                ("metric1 - StdDev", [1.0, 1.0]),
                ("metric1 - Last", [2.0, 2.0]),
                ("metric1 - Count", [2.0, 2.0]),
                ("metric2 - Min", [8.0, 8.0]),
                ("metric2 - Max", [10.0, 10.0]),
                ("metric2 - Avg", [9.0, 9.0]),
                ("metric2 - StdDev", [0.05, 0.05]),
                ("metric2 - Last", [7.0, 7.0]),
                ("metric2 - Count", [2.0, 2.0]),
                ("inputArtifacts1 - MediaType", ["text/plain", "text/plain"]),
                ("inputArtifacts1 - Value", ["s3:/foo/bar1", "s3:/foo/bar1"]),
                ("inputArtifacts2 - MediaType", ["text/plain", "text/plain"]),
                ("inputArtifacts2 - Value", ["s3:/foo/bar2", "s3:/foo/bar2"]),
                ("outputArtifacts1 - MediaType", ["text/csv", "text/csv"]),
                ("outputArtifacts1 - Value", ["s3:/sky/far1", "s3:/sky/far1"]),
                ("outputArtifacts2 - MediaType", ["text/csv", "text/csv"]),
                ("outputArtifacts2 - Value", ["s3:/sky/far2", "s3:/sky/far2"]),
                ("Trials", [["trial1"], ["trial1"]]),
                ("Experiments", [["experiment1"], ["experiment1"]]),
            ]
        )
    )

    pd.testing.assert_frame_equal(expected_dataframe, analytics.dataframe())
    expected_search_exp = {
        "Filters": [
            {"Name": "Parents.ExperimentName", "Operator": "Equals", "Value": "experiment1"}
        ]
    }
    mock_session.sagemaker_client.search.assert_called_with(
        Resource="ExperimentTrialComponent", SearchExpression=expected_search_exp
    )


def test_trial_analytics_dataframe_selected_hyperparams(mock_session):
    mock_session.sagemaker_client.search.return_value = {
        "Results": [
            {"TrialComponent": trial_component("trial-1")},
            {"TrialComponent": trial_component("trial-2")},
        ]
    }
    analytics = ExperimentAnalytics(
        experiment_name="experiment1", parameter_names=["hp2"], sagemaker_session=mock_session
    )

    expected_dataframe = pd.DataFrame.from_dict(
        OrderedDict(
            [
                ("TrialComponentName", ["trial-1", "trial-2"]),
                ("DisplayName", ["Training", "Training"]),
                ("SourceArn", ["some-source-arn", "some-source-arn"]),
                ("hp2", ["abc", "abc"]),
                ("metric1 - Min", [3.0, 3.0]),
                ("metric1 - Max", [5.0, 5.0]),
                ("metric1 - Avg", [4.0, 4.0]),
                ("metric1 - StdDev", [1.0, 1.0]),
                ("metric1 - Last", [2.0, 2.0]),
                ("metric1 - Count", [2.0, 2.0]),
                ("metric2 - Min", [8.0, 8.0]),
                ("metric2 - Max", [10.0, 10.0]),
                ("metric2 - Avg", [9.0, 9.0]),
                ("metric2 - StdDev", [0.05, 0.05]),
                ("metric2 - Last", [7.0, 7.0]),
                ("metric2 - Count", [2.0, 2.0]),
                ("inputArtifacts1 - MediaType", ["text/plain", "text/plain"]),
                ("inputArtifacts1 - Value", ["s3:/foo/bar1", "s3:/foo/bar1"]),
                ("inputArtifacts2 - MediaType", ["text/plain", "text/plain"]),
                ("inputArtifacts2 - Value", ["s3:/foo/bar2", "s3:/foo/bar2"]),
                ("outputArtifacts1 - MediaType", ["text/csv", "text/csv"]),
                ("outputArtifacts1 - Value", ["s3:/sky/far1", "s3:/sky/far1"]),
                ("outputArtifacts2 - MediaType", ["text/csv", "text/csv"]),
                ("outputArtifacts2 - Value", ["s3:/sky/far2", "s3:/sky/far2"]),
                ("Trials", [["trial1"], ["trial1"]]),
                ("Experiments", [["experiment1"], ["experiment1"]]),
            ]
        )
    )

    pd.testing.assert_frame_equal(expected_dataframe, analytics.dataframe())
    expected_search_exp = {
        "Filters": [
            {"Name": "Parents.ExperimentName", "Operator": "Equals", "Value": "experiment1"}
        ]
    }
    mock_session.sagemaker_client.search.assert_called_with(
        Resource="ExperimentTrialComponent", SearchExpression=expected_search_exp
    )


def test_trial_analytics_dataframe_selected_metrics(mock_session):
    mock_session.sagemaker_client.search.return_value = {
        "Results": [
            {"TrialComponent": trial_component("trial-1")},
            {"TrialComponent": trial_component("trial-2")},
        ]
    }
    analytics = ExperimentAnalytics(
        experiment_name="experiment1", metric_names=["metric1"], sagemaker_session=mock_session
    )

    expected_dataframe = pd.DataFrame.from_dict(
        OrderedDict(
            [
                ("TrialComponentName", ["trial-1", "trial-2"]),
                ("DisplayName", ["Training", "Training"]),
                ("SourceArn", ["some-source-arn", "some-source-arn"]),
                ("hp1", [1.0, 1.0]),
                ("hp2", ["abc", "abc"]),
                ("metric1 - Min", [3.0, 3.0]),
                ("metric1 - Max", [5.0, 5.0]),
                ("metric1 - Avg", [4.0, 4.0]),
                ("metric1 - StdDev", [1.0, 1.0]),
                ("metric1 - Last", [2.0, 2.0]),
                ("metric1 - Count", [2.0, 2.0]),
                ("inputArtifacts1 - MediaType", ["text/plain", "text/plain"]),
                ("inputArtifacts1 - Value", ["s3:/foo/bar1", "s3:/foo/bar1"]),
                ("inputArtifacts2 - MediaType", ["text/plain", "text/plain"]),
                ("inputArtifacts2 - Value", ["s3:/foo/bar2", "s3:/foo/bar2"]),
                ("outputArtifacts1 - MediaType", ["text/csv", "text/csv"]),
                ("outputArtifacts1 - Value", ["s3:/sky/far1", "s3:/sky/far1"]),
                ("outputArtifacts2 - MediaType", ["text/csv", "text/csv"]),
                ("outputArtifacts2 - Value", ["s3:/sky/far2", "s3:/sky/far2"]),
                ("Trials", [["trial1"], ["trial1"]]),
                ("Experiments", [["experiment1"], ["experiment1"]]),
            ]
        )
    )

    pd.testing.assert_frame_equal(expected_dataframe, analytics.dataframe())
    expected_search_exp = {
        "Filters": [
            {"Name": "Parents.ExperimentName", "Operator": "Equals", "Value": "experiment1"}
        ]
    }
    mock_session.sagemaker_client.search.assert_called_with(
        Resource="ExperimentTrialComponent", SearchExpression=expected_search_exp
    )


def test_trial_analytics_dataframe_search_pagination(mock_session):
    result_page_1 = {
        "Results": [{"TrialComponent": trial_component("trial-1")}],
        "NextToken": "nextToken",
    }

    result_page_2 = {"Results": [{"TrialComponent": trial_component("trial-2")}]}

    mock_session.sagemaker_client.search.side_effect = [result_page_1, result_page_2]
    analytics = ExperimentAnalytics(experiment_name="experiment1", sagemaker_session=mock_session)

    expected_dataframe = pd.DataFrame.from_dict(
        OrderedDict(
            [
                ("TrialComponentName", ["trial-1", "trial-2"]),
                ("DisplayName", ["Training", "Training"]),
                ("SourceArn", ["some-source-arn", "some-source-arn"]),
                ("hp1", [1.0, 1.0]),
                ("hp2", ["abc", "abc"]),
                ("metric1 - Min", [3.0, 3.0]),
                ("metric1 - Max", [5.0, 5.0]),
                ("metric1 - Avg", [4.0, 4.0]),
                ("metric1 - StdDev", [1.0, 1.0]),
                ("metric1 - Last", [2.0, 2.0]),
                ("metric1 - Count", [2.0, 2.0]),
                ("metric2 - Min", [8.0, 8.0]),
                ("metric2 - Max", [10.0, 10.0]),
                ("metric2 - Avg", [9.0, 9.0]),
                ("metric2 - StdDev", [0.05, 0.05]),
                ("metric2 - Last", [7.0, 7.0]),
                ("metric2 - Count", [2.0, 2.0]),
                ("inputArtifacts1 - MediaType", ["text/plain", "text/plain"]),
                ("inputArtifacts1 - Value", ["s3:/foo/bar1", "s3:/foo/bar1"]),
                ("inputArtifacts2 - MediaType", ["text/plain", "text/plain"]),
                ("inputArtifacts2 - Value", ["s3:/foo/bar2", "s3:/foo/bar2"]),
                ("outputArtifacts1 - MediaType", ["text/csv", "text/csv"]),
                ("outputArtifacts1 - Value", ["s3:/sky/far1", "s3:/sky/far1"]),
                ("outputArtifacts2 - MediaType", ["text/csv", "text/csv"]),
                ("outputArtifacts2 - Value", ["s3:/sky/far2", "s3:/sky/far2"]),
                ("Trials", [["trial1"], ["trial1"]]),
                ("Experiments", [["experiment1"], ["experiment1"]]),
            ]
        )
    )

    pd.testing.assert_frame_equal(expected_dataframe, analytics.dataframe())
    expected_search_exp = {
        "Filters": [
            {"Name": "Parents.ExperimentName", "Operator": "Equals", "Value": "experiment1"}
        ]
    }
    mock_session.sagemaker_client.search.assert_has_calls(
        [
            mock.call(Resource="ExperimentTrialComponent", SearchExpression=expected_search_exp),
            mock.call(
                Resource="ExperimentTrialComponent",
                SearchExpression=expected_search_exp,
                NextToken="nextToken",
            ),
        ]
    )


def test_trial_analytics_dataframe_filter_trials_search_exp_only(mock_session):
    mock_session.sagemaker_client.search.return_value = {"Results": []}

    search_exp = {"Filters": [{"Name": "Tags.someTag", "Operator": "Equals", "Value": "someValue"}]}
    analytics = ExperimentAnalytics(search_expression=search_exp, sagemaker_session=mock_session)

    analytics.dataframe()

    mock_session.sagemaker_client.search.assert_called_with(
        Resource="ExperimentTrialComponent", SearchExpression=search_exp
    )


def test_trial_analytics_dataframe_filter_trials_search_exp_with_experiment(mock_session):
    mock_session.sagemaker_client.search.return_value = {"Results": []}

    search_exp = {"Filters": [{"Name": "Tags.someTag", "Operator": "Equals", "Value": "someValue"}]}
    analytics = ExperimentAnalytics(
        experiment_name="someExperiment",
        search_expression=search_exp,
        sagemaker_session=mock_session,
    )

    analytics.dataframe()

    expected_search_exp = {
        "Filters": [
            {"Name": "Tags.someTag", "Operator": "Equals", "Value": "someValue"},
            {"Name": "Parents.ExperimentName", "Operator": "Equals", "Value": "someExperiment"},
        ]
    }

    mock_session.sagemaker_client.search.assert_called_with(
        Resource="ExperimentTrialComponent", SearchExpression=expected_search_exp
    )


def test_trial_analytics_dataframe_throws_error_if_no_filter_specified(mock_session):
    with pytest.raises(ValueError):
        ExperimentAnalytics(sagemaker_session=mock_session)


def test_trial_analytics_dataframe_filter_trials_search_exp_with_sort(mock_session):
    mock_session.sagemaker_client.search.return_value = {"Results": []}

    search_exp = {"Filters": [{"Name": "Tags.someTag", "Operator": "Equals", "Value": "someValue"}]}
    analytics = ExperimentAnalytics(
        experiment_name="someExperiment",
        search_expression=search_exp,
        sort_by="Tags.someTag",
        sort_order="Ascending",
        sagemaker_session=mock_session,
    )

    analytics.dataframe()

    expected_search_exp = {
        "Filters": [
            {"Name": "Tags.someTag", "Operator": "Equals", "Value": "someValue"},
            {"Name": "Parents.ExperimentName", "Operator": "Equals", "Value": "someExperiment"},
        ]
    }

    mock_session.sagemaker_client.search.assert_called_with(
        Resource="ExperimentTrialComponent",
        SearchExpression=expected_search_exp,
        SortBy="Tags.someTag",
        SortOrder="Ascending",
    )
