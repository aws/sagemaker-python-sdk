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
import os
import uuid

import pytest
from mock import Mock

from sagemaker.analytics import (
    AnalyticsMetricsBase,
    HyperparameterTuningJobAnalytics,
    TrainingJobAnalytics,
)
from sagemaker.session_settings import SessionSettings

BUCKET_NAME = "mybucket"
REGION = "us-west-2"


@pytest.fixture()
def sagemaker_session():
    return create_sagemaker_session()


def create_sagemaker_session(
    describe_training_result=None,
    list_training_results=None,
    metric_stats_results=None,
    describe_tuning_result=None,
):
    boto_mock = Mock(name="boto_session", region_name=REGION)
    sms = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
        settings=SessionSettings(),
        default_bucket_prefix=None,
    )
    sms.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    sms.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_hyper_parameter_tuning_job", return_value=describe_tuning_result
    )
    sms.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job", return_value=describe_training_result
    )
    sms.sagemaker_client.list_training_jobs_for_hyper_parameter_tuning_job = Mock(
        name="list_training_jobs_for_hyper_parameter_tuning_job", return_value=list_training_results
    )
    cwm_mock = Mock(name="cloudwatch_client")
    boto_mock.client = Mock(return_value=cwm_mock)
    cwm_mock.get_metric_statistics = Mock(name="get_metric_statistics")
    cwm_mock.get_metric_statistics.side_effect = cw_request_side_effect
    return sms


def cw_request_side_effect(
    Namespace, MetricName, Dimensions, StartTime, EndTime, Period, Statistics
):
    if _is_valid_request(Namespace, MetricName, Dimensions, StartTime, EndTime, Period, Statistics):
        return _metric_stats_results()


def _is_valid_request(Namespace, MetricName, Dimensions, StartTime, EndTime, Period, Statistics):
    could_watch_request = {
        "Namespace": Namespace,
        "MetricName": MetricName,
        "Dimensions": Dimensions,
        "StartTime": StartTime,
        "EndTime": EndTime,
        "Period": Period,
        "Statistics": Statistics,
    }
    print(could_watch_request)
    return could_watch_request == cw_request()


def cw_request():
    describe_training_result = _describe_training_result()
    return {
        "Namespace": "/aws/sagemaker/TrainingJobs",
        "MetricName": "train:acc",
        "Dimensions": [{"Name": "TrainingJobName", "Value": "my-training-job"}],
        "StartTime": describe_training_result["TrainingStartTime"],
        "EndTime": describe_training_result["TrainingEndTime"] + datetime.timedelta(minutes=1),
        "Period": 60,
        "Statistics": ["Average"],
    }


def test_abstract_base_class():
    # confirm that the abstract base class can't be instantiated directly
    with pytest.raises(TypeError) as _:  # noqa: F841
        AnalyticsMetricsBase()


def test_tuner_name(sagemaker_session):
    tuner = HyperparameterTuningJobAnalytics("my-tuning-job", sagemaker_session=sagemaker_session)
    assert tuner.name == "my-tuning-job"
    assert str(tuner).find("my-tuning-job") != -1


@pytest.mark.parametrize("has_training_job_definition_name", [True, False])
def test_tuner_dataframe(has_training_job_definition_name):
    training_job_definition_name = "training_def_1"

    def mock_summary(name="job-name", value=0.9):
        summary = {
            "TrainingJobName": name,
            "TrainingJobStatus": "Completed",
            "FinalHyperParameterTuningJobObjectiveMetric": {"Name": "awesomeness", "Value": value},
            "TrainingStartTime": datetime.datetime(2018, 5, 16, 1, 2, 3),
            "TrainingEndTime": datetime.datetime(2018, 5, 16, 5, 6, 7),
            "TunedHyperParameters": {"learning_rate": 0.1, "layers": 137},
        }

        if has_training_job_definition_name:
            summary["TrainingJobDefinitionName"] = training_job_definition_name
        return summary

    session = create_sagemaker_session(
        list_training_results={
            "TrainingJobSummaries": [
                mock_summary(),
                mock_summary(),
                mock_summary(),
                mock_summary(),
                mock_summary(),
            ]
        }
    )

    tuner = HyperparameterTuningJobAnalytics("my-tuning-job", sagemaker_session=session)
    df = tuner.dataframe()
    assert df is not None
    assert len(df) == 5
    assert (
        len(session.sagemaker_client.list_training_jobs_for_hyper_parameter_tuning_job.mock_calls)
        == 1
    )

    # Clear the cache, check that it calls the service again.
    tuner.clear_cache()
    df = tuner.dataframe()
    assert (
        len(session.sagemaker_client.list_training_jobs_for_hyper_parameter_tuning_job.mock_calls)
        == 2
    )
    df = tuner.dataframe(force_refresh=True)
    assert (
        len(session.sagemaker_client.list_training_jobs_for_hyper_parameter_tuning_job.mock_calls)
        == 3
    )

    # check that the hyperparameter is in the dataframe
    assert len(df["layers"]) == 5
    assert min(df["layers"]) == 137

    # Check that the training time calculation is returning something sane.
    assert min(df["TrainingElapsedTimeSeconds"]) > 5
    assert max(df["TrainingElapsedTimeSeconds"]) < 86400

    if has_training_job_definition_name:
        for index in range(0, 5):
            assert df["TrainingJobDefinitionName"][index] == training_job_definition_name
    else:
        assert "TrainingJobDefinitionName" not in df

    # Export to CSV and check that file exists
    tmp_name = "/tmp/unit-test-%s.csv" % uuid.uuid4()
    assert not os.path.isfile(tmp_name)
    tuner.export_csv(tmp_name)
    assert os.path.isfile(tmp_name)
    os.unlink(tmp_name)


def test_description():
    session = create_sagemaker_session(
        describe_tuning_result={
            "HyperParameterTuningJobConfig": {
                "ParameterRanges": {
                    "CategoricalParameterRanges": [],
                    "ContinuousParameterRanges": [
                        {"MaxValue": "1", "MinValue": "0", "Name": "eta"},
                        {"MaxValue": "10", "MinValue": "0", "Name": "gamma"},
                    ],
                    "IntegerParameterRanges": [
                        {"MaxValue": "30", "MinValue": "5", "Name": "num_layers"},
                        {"MaxValue": "100", "MinValue": "50", "Name": "iterations"},
                    ],
                }
            },
            "TrainingJobDefinition": {
                "AlgorithmSpecification": {
                    "TrainingImage": "training_image",
                    "TrainingInputMode": "File",
                }
            },
        }
    )

    tuner = HyperparameterTuningJobAnalytics("my-tuning-job", sagemaker_session=session)

    d = tuner.description()
    assert len(session.sagemaker_client.describe_hyper_parameter_tuning_job.mock_calls) == 1
    assert d is not None
    assert d["HyperParameterTuningJobConfig"] is not None
    tuner.clear_cache()
    d = tuner.description()
    assert len(session.sagemaker_client.describe_hyper_parameter_tuning_job.mock_calls) == 2
    d = tuner.description()
    assert len(session.sagemaker_client.describe_hyper_parameter_tuning_job.mock_calls) == 2
    d = tuner.description(force_refresh=True)
    assert len(session.sagemaker_client.describe_hyper_parameter_tuning_job.mock_calls) == 3

    # Check that the ranges work.
    r = tuner.tuning_ranges
    assert len(r) == 4


def test_tuning_ranges_multi_training_job_definitions():
    session = create_sagemaker_session(
        describe_tuning_result={
            "HyperParameterTuningJobConfig": {},
            "TrainingJobDefinitions": [
                {
                    "DefinitionName": "estimator_1",
                    "HyperParameterRanges": {
                        "CategoricalParameterRanges": [],
                        "ContinuousParameterRanges": [
                            {"MaxValue": "1", "MinValue": "0", "Name": "eta"},
                            {"MaxValue": "10", "MinValue": "0", "Name": "gamma"},
                        ],
                        "IntegerParameterRanges": [
                            {"MaxValue": "30", "MinValue": "5", "Name": "num_layers"},
                            {"MaxValue": "100", "MinValue": "50", "Name": "iterations"},
                        ],
                    },
                    "AlgorithmSpecification": {
                        "TrainingImage": "training_image_1",
                        "TrainingInputMode": "File",
                    },
                },
                {
                    "DefinitionName": "estimator_2",
                    "HyperParameterRanges": {
                        "CategoricalParameterRanges": [
                            {"Values": ["TF", "MXNet"], "Name": "framework"}
                        ],
                        "ContinuousParameterRanges": [
                            {"MaxValue": "1.0", "MinValue": "0.2", "Name": "gamma"}
                        ],
                        "IntegerParameterRanges": [],
                    },
                    "AlgorithmSpecification": {
                        "TrainingImage": "training_image_2",
                        "TrainingInputMode": "File",
                    },
                },
            ],
        }
    )

    expected_result = {
        "estimator_1": {
            "eta": {"MaxValue": "1", "MinValue": "0", "Name": "eta"},
            "gamma": {"MaxValue": "10", "MinValue": "0", "Name": "gamma"},
            "iterations": {"MaxValue": "100", "MinValue": "50", "Name": "iterations"},
            "num_layers": {"MaxValue": "30", "MinValue": "5", "Name": "num_layers"},
        },
        "estimator_2": {
            "framework": {"Values": ["TF", "MXNet"], "Name": "framework"},
            "gamma": {"MaxValue": "1.0", "MinValue": "0.2", "Name": "gamma"},
        },
    }

    tuner = HyperparameterTuningJobAnalytics("my-tuning-job", sagemaker_session=session)

    assert expected_result == tuner.tuning_ranges


def test_trainer_name():
    describe_training_result = {
        "TrainingStartTime": datetime.datetime(2018, 5, 16, 1, 2, 3),
        "TrainingEndTime": datetime.datetime(2018, 5, 16, 5, 6, 7),
    }
    session = create_sagemaker_session(describe_training_result)
    trainer = TrainingJobAnalytics("my-training-job", ["metric"], sagemaker_session=session)
    assert trainer.name == "my-training-job"
    assert str(trainer).find("my-training-job") != -1


def _describe_training_result():
    return {
        "TrainingStartTime": datetime.datetime(2018, 5, 16, 1, 2, 3),
        "TrainingEndTime": datetime.datetime(2018, 5, 16, 5, 6, 7),
    }


def _metric_stats_results():
    return {
        "Datapoints": [
            {"Average": 77.1, "Timestamp": datetime.datetime(2018, 5, 16, 1, 3, 3)},
            {"Average": 87.1, "Timestamp": datetime.datetime(2018, 5, 16, 1, 8, 3)},
            {"Average": 97.1, "Timestamp": datetime.datetime(2018, 5, 16, 2, 3, 3)},
        ]
    }


def test_trainer_dataframe():
    session = create_sagemaker_session(
        describe_training_result=_describe_training_result(),
        metric_stats_results=_metric_stats_results(),
    )
    trainer = TrainingJobAnalytics("my-training-job", ["train:acc"], sagemaker_session=session)

    df = trainer.dataframe()
    assert df is not None
    assert len(df) == 3
    assert min(df["value"]) == 77.1
    assert max(df["value"]) == 97.1

    # Export to CSV and check that file exists
    tmp_name = "/tmp/unit-test-%s.csv" % uuid.uuid4()
    assert not os.path.isfile(tmp_name)
    trainer.export_csv(tmp_name)
    assert os.path.isfile(tmp_name)
    os.unlink(tmp_name)


def test_start_time_end_time_and_period_specified():
    describe_training_result = {
        "TrainingStartTime": datetime.datetime(2018, 5, 16, 1, 2, 3),
        "TrainingEndTime": datetime.datetime(2018, 5, 16, 5, 6, 7),
    }
    session = create_sagemaker_session(describe_training_result)
    start_time = datetime.datetime(2018, 5, 16, 1, 3, 4)
    end_time = datetime.datetime(2018, 5, 16, 5, 1, 1)
    period = 300
    trainer = TrainingJobAnalytics(
        "my-training-job",
        ["metric"],
        sagemaker_session=session,
        start_time=start_time,
        end_time=end_time,
        period=period,
    )

    assert trainer._time_interval["start_time"] == start_time
    assert trainer._time_interval["end_time"] == end_time
    assert trainer._period == period
