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
import pandas as pd
import pytest
from mock import Mock, patch, MagicMock
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.session_settings import SessionSettings
from sagemaker.tuner import (
    HyperparameterTuner
)
from tests.unit.tuner_test_utils import (
    OBJECTIVE_METRIC_NAME,
    HYPERPARAMETER_RANGES,
    METRIC_DEFINITIONS
)
from sagemaker.session_settings import SessionSettings
# Visualization specific imports
from sagemaker.amtviz.visualization import visualize_tuning_job, get_job_analytics_data
from tests.unit.tuner_visualize_test_utils import (
    TUNING_JOB_NAMES,
    TUNED_PARAMETERS,
    OBJECTIVE_NAME,
    TRIALS_DF_DATA,
    FULL_DF_DATA,
    TUNING_JOB_NAME_1,
    TUNING_JOB_NAME_2,
    TUNING_JOB_RESULT,
    TRIALS_DF_COLUMNS,
    FULL_DF_COLUMNS,
    TRIALS_DF_TRAINING_JOB_NAMES,
    TRIALS_DF_TRAINING_JOB_STATUSES,
    TUNING_JOB_NAMES,
    TRIALS_DF_VALID_F1_VALUES,
    FILTERED_TUNING_JOB_DF_DATA,
    TUNING_RANGES
)
import altair as alt

def create_sagemaker_session():
    boto_mock = Mock(name="boto_session")
    sms = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        config=None,
        local_mode=False,
        settings=SessionSettings()
    )
    sms.sagemaker_config = {}
    return sms

@pytest.fixture()
def sagemaker_session():
    return create_sagemaker_session()


@pytest.fixture()
def estimator(sagemaker_session):
    return Estimator(
        "image",
        "role",
        1,
        "ml.c4.xlarge",
        output_path="s3://bucket/prefix",
        sagemaker_session=sagemaker_session,
    )


@pytest.fixture()
def tuner(estimator):
    return HyperparameterTuner(
        estimator, OBJECTIVE_METRIC_NAME, HYPERPARAMETER_RANGES, METRIC_DEFINITIONS
    )

@pytest.fixture()
def tuner2(estimator):
    return HyperparameterTuner(
        estimator, OBJECTIVE_METRIC_NAME, HYPERPARAMETER_RANGES, METRIC_DEFINITIONS
    )


@pytest.fixture
def mock_visualize_tuning_job():
    with patch("sagemaker.amtviz.visualize_tuning_job") as mock_visualize:
        mock_visualize.return_value = "mock_chart"
        yield mock_visualize


@pytest.fixture
def mock_get_job_analytics_data():
    with patch("sagemaker.amtviz.visualization.get_job_analytics_data") as mock:
        mock.return_value = (
            pd.DataFrame(TRIALS_DF_DATA),
            TUNED_PARAMETERS,
            OBJECTIVE_NAME,
            True
        )
        yield mock


@pytest.fixture
def mock_prepare_consolidated_df():
    with patch("sagemaker.amtviz.visualization._prepare_consolidated_df") as mock:
        mock.return_value = pd.DataFrame(FULL_DF_DATA)
        yield mock


# Test graceful handling if the required altair library is not installed
def test_visualize_jobs_altair_not_installed(capsys):
    # Mock importlib.import_module to raise ImportError for 'altair'
    with patch("importlib.import_module") as mock_import:
        mock_import.side_effect = ImportError("No module named 'altair'")
        result = HyperparameterTuner.visualize_jobs(TUNING_JOB_NAMES)
        assert result is None
        captured = capsys.readouterr()
        assert "Altair is not installed." in captured.out
        assert "pip install altair" in captured.out


# Test basic method call if altair is installed
def test_visualize_jobs_altair_installed(mock_visualize_tuning_job):
    # Mock successful import of altair
    with patch("importlib.import_module") as mock_import:
        result = HyperparameterTuner.visualize_jobs(TUNING_JOB_NAMES)
        assert result == "mock_chart"


# Test for static method visualize_jobs()
def test_visualize_jobs(mock_visualize_tuning_job):
    result = HyperparameterTuner.visualize_jobs(TUNING_JOB_NAMES)
    assert result == "mock_chart"
    mock_visualize_tuning_job.assert_called_once_with(
        TUNING_JOB_NAMES,
        return_dfs=False,
        job_metrics=None,
        trials_only=False,
        advanced=False
    )
    # Vary the parameters and check if they have been passed correctly
    result = HyperparameterTuner.visualize_jobs(
        [TUNING_JOB_NAME_1], return_dfs=True, job_metrics="job_metrics", trials_only=True, advanced=True)
    mock_visualize_tuning_job.assert_called_with(
        [TUNING_JOB_NAME_1],
        return_dfs=True,
        job_metrics="job_metrics",
        trials_only=True,
        advanced=True
    )

# Test the instance method visualize_job() on a stubbed tuner object
def test_visualize_job(tuner, mock_visualize_tuning_job):
    # With default parameters
    result = tuner.visualize_job()
    assert result == "mock_chart"
    mock_visualize_tuning_job.assert_called_once_with(
        tuner,
        return_dfs=False,
        job_metrics=None,
        trials_only=False,
        advanced=False
    )
    # With varying parameters
    result = tuner.visualize_job(return_dfs=True, job_metrics="job_metrics", trials_only=True, advanced=True)
    assert result == "mock_chart"
    mock_visualize_tuning_job.assert_called_with(
        tuner,
        return_dfs=True,
        job_metrics="job_metrics",
        trials_only=True,
        advanced=True
    )

# Test the static method visualize_jobs() on multiple stubbed tuner objects
def test_visualize_multiple_jobs(tuner, tuner2, mock_visualize_tuning_job):
    result = HyperparameterTuner.visualize_jobs([tuner, tuner2])
    assert result == "mock_chart"
    mock_visualize_tuning_job.assert_called_once_with(
        [tuner, tuner2],
        return_dfs=False,
        job_metrics=None,
        trials_only=False,
        advanced=False
    )
    # Vary the parameters and check if they have been passed correctly
    result = HyperparameterTuner.visualize_jobs(
        [[tuner, tuner2]], return_dfs=True, job_metrics="job_metrics", trials_only=True, advanced=True)
    mock_visualize_tuning_job.assert_called_with(
        [[tuner, tuner2]],
        return_dfs=True,
        job_metrics="job_metrics",
        trials_only=True,
        advanced=True
    )

# Test direct method call for basic chart return type and default render settings
def test_visualize_tuning_job_analytics_data_results_in_altair_chart(mock_get_job_analytics_data):
    result = visualize_tuning_job("mock_job")
    assert alt.renderers.active == "default"
    assert isinstance(result, alt.VConcatChart)


# Test the size and structure of the returned dataframes (trials_df and full_df)
def test_visualize_tuning_job_return_dfs(mock_get_job_analytics_data, mock_prepare_consolidated_df):
    charts, trials_df, full_df = visualize_tuning_job("mock_job", return_dfs=True)
    # Basic assertion for the charts
    assert isinstance(charts, alt.VConcatChart)

    # Assertions for trials_df
    assert isinstance(trials_df, pd.DataFrame)
    assert trials_df.shape == (2, len(TRIALS_DF_COLUMNS))
    assert trials_df.columns.tolist() == TRIALS_DF_COLUMNS
    assert trials_df['TrainingJobName'].tolist() == TRIALS_DF_TRAINING_JOB_NAMES
    assert trials_df['TrainingJobStatus'].tolist() == TRIALS_DF_TRAINING_JOB_STATUSES
    assert trials_df['TuningJobName'].tolist() == TUNING_JOB_NAMES
    assert trials_df['valid-f1'].tolist() == TRIALS_DF_VALID_F1_VALUES

    # Assertions for full_df
    assert isinstance(full_df, pd.DataFrame)
    assert full_df.shape == (2, 16)
    assert full_df.columns.tolist() == FULL_DF_COLUMNS


# Test the handling of an an empty trials dataframe
@patch("sagemaker.amtviz.visualization.get_job_analytics_data")
def test_visualize_tuning_job_empty_trials(mock_get_job_analytics_data):
    mock_get_job_analytics_data.return_value = (
        pd.DataFrame(),  # empty dataframe
        TUNED_PARAMETERS,
        OBJECTIVE_NAME,
        True
    )
    charts = visualize_tuning_job("empty_job")
    assert charts.empty


# Test handling of return_dfs and trials_only parameter
def test_visualize_tuning_job_trials_only(mock_get_job_analytics_data):
    # If return_dfs is set to False, then only charts should be returned
    result = visualize_tuning_job("mock_job", return_dfs=False, trials_only=True)
    assert isinstance(result, alt.VConcatChart)
    # Trials_only controls the content of the two returned dataframes (trials_df, full_df)
    result, df1, df2 = visualize_tuning_job("mock_job", return_dfs=True, trials_only=True)
    assert isinstance(df1, pd.DataFrame)
    assert df1.shape == (2, len(TRIALS_DF_COLUMNS))
    assert isinstance(df2, pd.DataFrame)
    assert df2.empty
    # The combination of return_dfs and trials_only=False is covered in 'test_visualize_tuning_job_return_dfs'


# Check if all parameters are correctly passed to the (mocked) create_charts method
@patch("sagemaker.amtviz.visualization.create_charts")
def test_visualize_tuning_job_with_full_df(mock_create_charts, mock_get_job_analytics_data, mock_prepare_consolidated_df):
    mock_create_charts.return_value = alt.Chart()
    visualize_tuning_job("dummy_job")

    # Check the create_charts call arguments
    call_args = mock_create_charts.call_args[0]
    call_kwargs = mock_create_charts.call_args[1]
    assert isinstance(call_args[0], pd.DataFrame)  # trials_df
    assert isinstance(call_args[1], list)  # tuned_parameters
    assert isinstance(call_args[2], pd.DataFrame)  # full_df
    assert isinstance(call_args[3], str)  # objective_name
    assert call_kwargs.get("minimize_objective")

    # Check the details of the passed arguments
    trials_df = call_args[0]
    assert trials_df.columns.tolist() == TRIALS_DF_COLUMNS
    tuned_parameters = call_args[1]
    assert tuned_parameters == TUNED_PARAMETERS
    objective_name = call_args[3]
    assert objective_name == OBJECTIVE_NAME
    full_df = call_args[2]
    assert full_df.columns.tolist() == FULL_DF_COLUMNS


# Test the dataframe produced by get_job_analytics_data()
@patch("sagemaker.HyperparameterTuningJobAnalytics")
def test_get_job_analytics_data(mock_hyperparameter_tuning_job_analytics):
    # Mock sagemaker's describe_hyper_parameter_tuning_job and some internal methods
    sagemaker.amtviz.visualization.sm.describe_hyper_parameter_tuning_job = Mock(return_value=TUNING_JOB_RESULT)
    sagemaker.amtviz.visualization._get_tuning_job_names_with_parents = Mock(
        return_value=[TUNING_JOB_NAME_1, TUNING_JOB_NAME_2])
    sagemaker.amtviz.visualization._get_df = Mock(return_value=pd.DataFrame(FILTERED_TUNING_JOB_DF_DATA))
    mock_tuning_job_instance = MagicMock()
    mock_hyperparameter_tuning_job_analytics.return_value = mock_tuning_job_instance
    mock_tuning_job_instance.tuning_ranges.values.return_value = TUNING_RANGES

    df, tuned_parameters, objective_name, is_minimize = get_job_analytics_data([TUNING_JOB_NAME_1])
    assert df.shape == (4, 12)
    assert df.columns.tolist() == TRIALS_DF_COLUMNS
    assert tuned_parameters == TUNED_PARAMETERS
    assert objective_name == OBJECTIVE_NAME
    assert is_minimize is False