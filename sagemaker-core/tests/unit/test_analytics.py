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
"""Unit tests for sagemaker.core.analytics module."""
from __future__ import absolute_import

import datetime
import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from collections import OrderedDict

# Mock pandas before importing analytics
import sys

sys.modules["pandas"] = MagicMock()

from sagemaker.core.analytics import (
    AnalyticsMetricsBase,
    HyperparameterTuningJobAnalytics,
    TrainingJobAnalytics,
    ArtifactAnalytics,
    ExperimentAnalytics,
    METRICS_PERIOD_DEFAULT,
)


class TestAnalyticsMetricsBase:
    """Test AnalyticsMetricsBase abstract class."""

    def test_init(self):
        """Test initialization of base class."""

        # Create a concrete implementation for testing
        class ConcreteAnalytics(AnalyticsMetricsBase):
            def _fetch_dataframe(self):
                return Mock()

        analytics = ConcreteAnalytics()
        assert analytics._dataframe is None

    def test_clear_cache(self):
        """Test clear_cache method."""

        class ConcreteAnalytics(AnalyticsMetricsBase):
            def _fetch_dataframe(self):
                return Mock()

        analytics = ConcreteAnalytics()
        analytics._dataframe = Mock()
        analytics.clear_cache()
        assert analytics._dataframe is None

    @patch("sagemaker.core.analytics.pd")
    def test_export_csv(self, mock_pd):
        """Test export_csv method."""

        class ConcreteAnalytics(AnalyticsMetricsBase):
            def _fetch_dataframe(self):
                return mock_pd.DataFrame()

        analytics = ConcreteAnalytics()
        mock_df = Mock()
        analytics._dataframe = mock_df

        analytics.export_csv("test.csv")
        mock_df.to_csv.assert_called_once_with("test.csv")

    @patch("sagemaker.core.analytics.pd")
    def test_dataframe_cached(self, mock_pd):
        """Test dataframe method with caching."""

        class ConcreteAnalytics(AnalyticsMetricsBase):
            def _fetch_dataframe(self):
                return mock_pd.DataFrame()

        analytics = ConcreteAnalytics()
        mock_df = Mock()
        analytics._dataframe = mock_df

        result = analytics.dataframe()
        assert result == mock_df

    @patch("sagemaker.core.analytics.pd")
    def test_dataframe_force_refresh(self, mock_pd):
        """Test dataframe method with force_refresh."""
        mock_new_df = Mock()

        class ConcreteAnalytics(AnalyticsMetricsBase):
            def _fetch_dataframe(self):
                return mock_new_df

        analytics = ConcreteAnalytics()
        analytics._dataframe = Mock()

        result = analytics.dataframe(force_refresh=True)
        assert result == mock_new_df


class TestHyperparameterTuningJobAnalytics:
    """Test HyperparameterTuningJobAnalytics class."""

    @patch("sagemaker.core.analytics.Session")
    def test_init(self, mock_session_class):
        """Test initialization."""
        mock_session = Mock()
        mock_session.sagemaker_client = Mock()
        mock_session_class.return_value = mock_session

        analytics = HyperparameterTuningJobAnalytics("test-tuning-job")

        assert analytics.name == "test-tuning-job"
        assert analytics._tuning_job_name == "test-tuning-job"
        assert analytics._tuning_job_describe_result is None
        assert analytics._training_job_summaries is None

    @patch("sagemaker.core.analytics.Session")
    def test_init_with_session(self, mock_session_class):
        """Test initialization with provided session."""
        mock_session = Mock()
        mock_session.sagemaker_client = Mock()

        analytics = HyperparameterTuningJobAnalytics(
            "test-tuning-job", sagemaker_session=mock_session
        )

        assert analytics._sage_client == mock_session.sagemaker_client

    @patch("sagemaker.core.analytics.Session")
    def test_repr(self, mock_session_class):
        """Test string representation."""
        mock_session = Mock()
        mock_session.sagemaker_client = Mock()
        mock_session_class.return_value = mock_session

        analytics = HyperparameterTuningJobAnalytics("test-job")
        result = repr(analytics)

        assert "HyperparameterTuningJobAnalytics" in result
        assert "test-job" in result

    @patch("sagemaker.core.analytics.Session")
    def test_clear_cache(self, mock_session_class):
        """Test clear_cache method."""
        mock_session = Mock()
        mock_session.sagemaker_client = Mock()
        mock_session_class.return_value = mock_session

        analytics = HyperparameterTuningJobAnalytics("test-job")
        analytics._tuning_job_describe_result = {"test": "data"}
        analytics._training_job_summaries = [{"job": "summary"}]
        analytics._dataframe = Mock()

        analytics.clear_cache()

        assert analytics._tuning_job_describe_result is None
        assert analytics._training_job_summaries is None
        assert analytics._dataframe is None

    @patch("sagemaker.core.analytics.Session")
    def test_description(self, mock_session_class):
        """Test description method."""
        mock_session = Mock()
        mock_client = Mock()
        mock_client.describe_hyper_parameter_tuning_job.return_value = {"JobName": "test"}
        mock_session.sagemaker_client = mock_client
        mock_session_class.return_value = mock_session

        analytics = HyperparameterTuningJobAnalytics("test-job")
        result = analytics.description()

        assert result == {"JobName": "test"}
        mock_client.describe_hyper_parameter_tuning_job.assert_called_once()

    @patch("sagemaker.core.analytics.Session")
    def test_description_cached(self, mock_session_class):
        """Test description method with caching."""
        mock_session = Mock()
        mock_client = Mock()
        mock_session.sagemaker_client = mock_client
        mock_session_class.return_value = mock_session

        analytics = HyperparameterTuningJobAnalytics("test-job")
        analytics._tuning_job_describe_result = {"cached": "data"}

        result = analytics.description()

        assert result == {"cached": "data"}
        mock_client.describe_hyper_parameter_tuning_job.assert_not_called()

    @patch("sagemaker.core.analytics.Session")
    @patch("sagemaker.core.analytics.pd")
    def test_fetch_dataframe(self, mock_pd, mock_session_class):
        """Test _fetch_dataframe method."""
        mock_session = Mock()
        mock_session.sagemaker_client = Mock()
        mock_session_class.return_value = mock_session

        analytics = HyperparameterTuningJobAnalytics("test-job")
        analytics._training_job_summaries = [
            {
                "TunedHyperParameters": {"lr": "0.01", "epochs": "10"},
                "TrainingJobName": "job-1",
                "TrainingJobStatus": "Completed",
                "FinalHyperParameterTuningJobObjectiveMetric": {"Value": 0.95},
                "TrainingStartTime": datetime.datetime(2023, 1, 1, 10, 0),
                "TrainingEndTime": datetime.datetime(2023, 1, 1, 11, 0),
            }
        ]

        mock_df = Mock()
        mock_pd.DataFrame.return_value = mock_df

        result = analytics._fetch_dataframe()

        assert result == mock_df
        mock_pd.DataFrame.assert_called_once()


class TestTrainingJobAnalytics:
    """Test TrainingJobAnalytics class."""

    @patch("sagemaker.core.analytics.Session")
    def test_init_with_metric_names(self, mock_session_class):
        """Test initialization with metric names."""
        mock_session = Mock()
        mock_client = Mock()
        mock_client.describe_training_job.return_value = {
            "TrainingStartTime": datetime.datetime(2023, 1, 1, 10, 0),
            "TrainingEndTime": datetime.datetime(2023, 1, 1, 11, 0),
        }
        mock_session.sagemaker_client = mock_client
        mock_session.boto_session.client.return_value = Mock()
        mock_session_class.return_value = mock_session

        analytics = TrainingJobAnalytics("test-job", metric_names=["accuracy", "loss"])

        assert analytics.name == "test-job"
        assert analytics._metric_names == ["accuracy", "loss"]
        assert analytics._period == METRICS_PERIOD_DEFAULT

    @patch("sagemaker.core.analytics.Session")
    def test_init_with_custom_period(self, mock_session_class):
        """Test initialization with custom period."""
        mock_session = Mock()
        mock_client = Mock()
        mock_client.describe_training_job.return_value = {
            "TrainingStartTime": datetime.datetime(2023, 1, 1, 10, 0),
            "TrainingEndTime": datetime.datetime(2023, 1, 1, 11, 0),
        }
        mock_session.sagemaker_client = mock_client
        mock_session.boto_session.client.return_value = Mock()
        mock_session_class.return_value = mock_session

        analytics = TrainingJobAnalytics("test-job", metric_names=["accuracy"], period=120)

        assert analytics._period == 120

    @patch("sagemaker.core.analytics.Session")
    def test_repr(self, mock_session_class):
        """Test string representation."""
        mock_session = Mock()
        mock_client = Mock()
        mock_client.describe_training_job.return_value = {
            "TrainingStartTime": datetime.datetime(2023, 1, 1, 10, 0),
            "TrainingEndTime": datetime.datetime(2023, 1, 1, 11, 0),
        }
        mock_session.sagemaker_client = mock_client
        mock_session.boto_session.client.return_value = Mock()
        mock_session_class.return_value = mock_session

        analytics = TrainingJobAnalytics("test-job", metric_names=["accuracy"])
        result = repr(analytics)

        assert "TrainingJobAnalytics" in result
        assert "test-job" in result

    @patch("sagemaker.core.analytics.Session")
    @patch("sagemaker.core.analytics.datetime")
    def test_determine_timeinterval(self, mock_datetime, mock_session_class):
        """Test _determine_timeinterval method."""
        mock_session = Mock()
        mock_client = Mock()
        start_time = datetime.datetime(2023, 1, 1, 10, 0)
        end_time = datetime.datetime(2023, 1, 1, 11, 0)

        mock_client.describe_training_job.return_value = {
            "TrainingStartTime": start_time,
            "TrainingEndTime": end_time,
        }
        mock_session.sagemaker_client = mock_client
        mock_session.boto_session.client.return_value = Mock()
        mock_session_class.return_value = mock_session

        analytics = TrainingJobAnalytics("test-job", metric_names=["accuracy"])
        result = analytics._time_interval

        assert "start_time" in result
        assert "end_time" in result


class TestArtifactAnalytics:
    """Test ArtifactAnalytics class."""

    def test_init_default(self):
        """Test initialization with defaults."""
        analytics = ArtifactAnalytics()

        assert analytics._sort_by is None
        assert analytics._sort_order is None
        assert analytics._source_uri is None
        assert analytics._artifact_type is None

    def test_init_with_sort_by_name(self):
        """Test initialization with sort_by Name."""
        analytics = ArtifactAnalytics(sort_by="Name", sort_order="Ascending")

        assert analytics._sort_by == "Name"
        assert analytics._sort_order == "Ascending"

    def test_init_with_invalid_sort_by(self):
        """Test initialization with invalid sort_by."""
        analytics = ArtifactAnalytics(sort_by="InvalidField")

        assert analytics._sort_by is None

    def test_repr(self):
        """Test string representation."""
        analytics = ArtifactAnalytics()
        result = repr(analytics)

        assert "ArtifactAnalytics" in result

    def test_reshape_source_type(self):
        """Test _reshape_source_type method."""
        analytics = ArtifactAnalytics()
        result = analytics._reshape_source_type(["type1", "type2"])

        assert isinstance(result, OrderedDict)
        assert "ArtifactSourceType" in result

    def test_reshape(self):
        """Test _reshape method."""
        analytics = ArtifactAnalytics()

        mock_artifact = Mock()
        mock_artifact.artifact_name = "test-artifact"
        mock_artifact.artifact_arn = "arn:aws:sagemaker:us-west-2:123456789012:artifact/test"
        mock_artifact.artifact_type = "Model"
        mock_artifact.source.source_uri = "s3://bucket/model"
        mock_artifact.creation_time = datetime.datetime(2023, 1, 1)
        mock_artifact.last_modified_time = datetime.datetime(2023, 1, 2)

        result = analytics._reshape(mock_artifact)

        assert result["ArtifactName"] == "test-artifact"
        assert result["ArtifactType"] == "Model"
        assert result["ArtifactSourceUri"] == "s3://bucket/model"


class TestExperimentAnalytics:
    """Test ExperimentAnalytics class."""

    @patch("sagemaker.core.analytics.Session")
    def test_init_with_experiment_name(self, mock_session_class):
        """Test initialization with experiment name."""
        mock_session = Mock()
        mock_session.sagemaker_client = Mock()
        mock_session_class.return_value = mock_session

        analytics = ExperimentAnalytics(experiment_name="test-experiment")

        assert analytics.name == "test-experiment"
        assert analytics._experiment_name == "test-experiment"

    @patch("sagemaker.core.analytics.Session")
    def test_init_with_search_expression(self, mock_session_class):
        """Test initialization with search expression."""
        mock_session = Mock()
        mock_session.sagemaker_client = Mock()
        mock_session_class.return_value = mock_session

        search_expr = {"Filters": [{"Name": "Status", "Value": "Completed"}]}
        analytics = ExperimentAnalytics(search_expression=search_expr)

        assert analytics._search_expression == search_expr

    def test_init_missing_required_params(self):
        """Test initialization without required parameters."""
        with pytest.raises(ValueError, match="Either experiment_name or search_expression"):
            ExperimentAnalytics()

    @patch("sagemaker.core.analytics.Session")
    def test_repr(self, mock_session_class):
        """Test string representation."""
        mock_session = Mock()
        mock_session.sagemaker_client = Mock()
        mock_session_class.return_value = mock_session

        analytics = ExperimentAnalytics(experiment_name="test-exp")
        result = repr(analytics)

        assert "ExperimentAnalytics" in result
        assert "test-exp" in result

    @patch("sagemaker.core.analytics.Session")
    def test_reshape_parameters(self, mock_session_class):
        """Test _reshape_parameters method."""
        mock_session = Mock()
        mock_session.sagemaker_client = Mock()
        mock_session_class.return_value = mock_session

        analytics = ExperimentAnalytics(experiment_name="test")

        parameters = {
            "learning_rate": {"NumberValue": 0.01},
            "batch_size": {"NumberValue": 32},
            "optimizer": {"StringValue": "adam"},
        }

        result = analytics._reshape_parameters(parameters)

        assert result["learning_rate"] == 0.01
        assert result["batch_size"] == 32
        assert result["optimizer"] == "adam"

    @patch("sagemaker.core.analytics.Session")
    def test_reshape_metrics(self, mock_session_class):
        """Test _reshape_metrics method."""
        mock_session = Mock()
        mock_session.sagemaker_client = Mock()
        mock_session_class.return_value = mock_session

        analytics = ExperimentAnalytics(experiment_name="test")

        metrics = [{"MetricName": "accuracy", "Min": 0.8, "Max": 0.95, "Avg": 0.9, "Last": 0.93}]

        result = analytics._reshape_metrics(metrics)

        assert "accuracy - Min" in result
        assert result["accuracy - Min"] == 0.8
        assert result["accuracy - Max"] == 0.95

    @patch("sagemaker.core.analytics.Session")
    def test_reshape_artifacts(self, mock_session_class):
        """Test _reshape_artifacts method."""
        mock_session = Mock()
        mock_session.sagemaker_client = Mock()
        mock_session_class.return_value = mock_session

        analytics = ExperimentAnalytics(experiment_name="test")

        artifacts = {"dataset": {"MediaType": "text/csv", "Value": "s3://bucket/data.csv"}}

        result = analytics._reshape_artifacts(artifacts, None)

        assert "dataset - MediaType" in result
        assert result["dataset - MediaType"] == "text/csv"
        assert result["dataset - Value"] == "s3://bucket/data.csv"

    @patch("sagemaker.core.analytics.Session")
    def test_reshape_parents(self, mock_session_class):
        """Test _reshape_parents method."""
        mock_session = Mock()
        mock_session.sagemaker_client = Mock()
        mock_session_class.return_value = mock_session

        analytics = ExperimentAnalytics(experiment_name="test")

        parents = [
            {"TrialName": "trial-1", "ExperimentName": "exp-1"},
            {"TrialName": "trial-2", "ExperimentName": "exp-1"},
        ]

        result = analytics._reshape_parents(parents)

        assert "Trials" in result
        assert "Experiments" in result
        assert result["Trials"] == ["trial-1", "trial-2"]
        assert result["Experiments"] == ["exp-1", "exp-1"]


class TestConstants:
    """Test module constants."""

    def test_metrics_period_default(self):
        """Test METRICS_PERIOD_DEFAULT constant."""
        assert METRICS_PERIOD_DEFAULT == 60
