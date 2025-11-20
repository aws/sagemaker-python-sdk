import pytest
import datetime
import logging
from unittest.mock import Mock, patch, call
from sagemaker.core.resources import TrainingJob, DataQualityJobDefinition
from sagemaker.core.shapes import (
    AdditionalS3DataSource,
    TrialComponent,
    TrialComponentParameterValue,
)
from sagemaker.core.utils.utils import *


LIST_TRAINING_JOB_RESPONSE_WITH_NEXT_TOKEN = {
    "TrainingJobSummaries": [
        {
            "TrainingJobName": "xgboost-iris-1",
            "TrainingJobArn": "arn:aws:sagemaker:us-west-2:111111111111:training-job/xgboost-iris-1",
            "CreationTime": datetime.datetime.now(),
            "TrainingEndTime": datetime.datetime.now(),
            "LastModifiedTime": datetime.datetime.now(),
            "TrainingJobStatus": "Completed",
        },
        {
            "TrainingJobName": "xgboost-iris-2",
            "TrainingJobArn": "arn:aws:sagemaker:us-west-2:111111111111:training-job/xgboost-iris-2",
            "CreationTime": datetime.datetime.now(),
            "TrainingEndTime": datetime.datetime.now(),
            "LastModifiedTime": datetime.datetime.now(),
            "TrainingJobStatus": "Completed",
        },
    ],
    "NextToken": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
}

LIST_TRAINING_JOB_RESPONSE_WITHOUT_NEXT_TOKEN = {
    "TrainingJobSummaries": [
        {
            "TrainingJobName": "xgboost-iris-1",
            "TrainingJobArn": "arn:aws:sagemaker:us-west-2:111111111111:training-job/xgboost-iris-1",
            "CreationTime": datetime.datetime.now(),
            "TrainingEndTime": datetime.datetime.now(),
            "LastModifiedTime": datetime.datetime.now(),
            "TrainingJobStatus": "Completed",
        },
        {
            "TrainingJobName": "xgboost-iris-2",
            "TrainingJobArn": "arn:aws:sagemaker:us-west-2:111111111111:training-job/xgboost-iris-2",
            "CreationTime": datetime.datetime.now(),
            "TrainingEndTime": datetime.datetime.now(),
            "LastModifiedTime": datetime.datetime.now(),
            "TrainingJobStatus": "Completed",
        },
    ],
}

LIST_DATA_QUALITY_JOB_DEFINITION_RESPONSE_WITHOUT_NEXT_TOKEN = {
    "JobDefinitionSummaries": [
        {
            "MonitoringJobDefinitionName": "data-quality-job-definition-1",
            "MonitoringJobDefinitionArn": "arn:aws:sagemaker:us-west-2:111111111111:data-quality-job-definition/data-quality-job-definition-1",
            "CreationTime": datetime.datetime.now(),
            "EndpointName": "sagemaker-tensorflow-serving-1",
        },
        {
            "MonitoringJobDefinitionName": "data-quality-job-definition-2",
            "MonitoringJobDefinitionArn": "arn:aws:sagemaker:us-west-2:111111111111:data-quality-job-definition/data-quality-job-definition-2",
            "CreationTime": datetime.datetime.now(),
            "EndpointName": "sagemaker-tensorflow-serving-2",
        },
    ],
}

LIST_ALIASES_RESPONSE_WITHOUT_NEXT_TOKEN = {
    "SageMakerImageVersionAliases": [
        {
            "SageMakerImageVersionAlias": "alias-1",
        },
        {
            "SageMakerImageVersionAlias": "alias-2",
        },
    ],
}


@pytest.fixture
def resource_iterator():
    client = Mock()
    resource_cls = TrainingJob
    iterator = ResourceIterator(
        client=client,
        summaries_key="TrainingJobSummaries",
        summary_name="TrainingJobSummary",
        resource_cls=resource_cls,
        list_method="list_training_jobs",
        list_method_kwargs={},
        custom_key_mapping=None,
    )

    return iterator, client, resource_cls


@pytest.fixture
def resource_iterator_with_custom_key_mapping():
    client = Mock()
    resource_cls = DataQualityJobDefinition
    custom_key_mapping = {
        "monitoring_job_definition_name": "job_definition_name",
        "monitoring_job_definition_arn": "job_definition_arn",
    }
    iterator = ResourceIterator(
        client=client,
        list_method="list_data_quality_job_definitions",
        summaries_key="JobDefinitionSummaries",
        summary_name="MonitoringJobDefinitionSummary",
        resource_cls=DataQualityJobDefinition,
        custom_key_mapping=custom_key_mapping,
        list_method_kwargs={},
    )
    return iterator, client, resource_cls


@pytest.fixture
def resource_iterator_with_primitive_class():
    client = Mock()
    resource_cls = str
    iterator = ResourceIterator(
        client=client,
        summaries_key="SageMakerImageVersionAliases",
        summary_name="SageMakerImageVersionAlias",
        resource_cls=resource_cls,
        list_method="list_aliases",
        list_method_kwargs={},
        custom_key_mapping=None,
    )

    return iterator, client, resource_cls


def test_next_with_summaries_in_summary_list(resource_iterator):
    iterator, _, _ = resource_iterator
    iterator.index = 0
    iterator.summary_list = LIST_TRAINING_JOB_RESPONSE_WITHOUT_NEXT_TOKEN["TrainingJobSummaries"]
    expected_training_job_data = LIST_TRAINING_JOB_RESPONSE_WITHOUT_NEXT_TOKEN[
        "TrainingJobSummaries"
    ][0]

    with patch.object(TrainingJob, "refresh") as mock_refresh:
        next_item = next(iterator)
        assert isinstance(next_item, TrainingJob)
        assert mock_refresh.call_count == 1

        assert next_item.training_job_name == expected_training_job_data["TrainingJobName"]
        assert next_item.training_job_arn == expected_training_job_data["TrainingJobArn"]
        assert next_item.creation_time == expected_training_job_data["CreationTime"]
        assert next_item.training_end_time == expected_training_job_data["TrainingEndTime"]
        assert next_item.last_modified_time == expected_training_job_data["LastModifiedTime"]
        assert next_item.training_job_status == expected_training_job_data["TrainingJobStatus"]


def test_next_reached_end_of_summary_list_without_next_token(resource_iterator):
    iterator, _, _ = resource_iterator
    iterator.summary_list = LIST_TRAINING_JOB_RESPONSE_WITHOUT_NEXT_TOKEN["TrainingJobSummaries"]
    iterator.index = len(iterator.summary_list)

    with pytest.raises(StopIteration):
        next(iterator)


def test_next_client_returns_empty_list(resource_iterator):
    iterator, client, _ = resource_iterator
    client.list_training_jobs.return_value = {"TrainingJobSummaries": []}

    with pytest.raises(StopIteration):
        next(iterator)


def test_next_without_next_token(resource_iterator):
    iterator, client, _ = resource_iterator
    client.list_training_jobs.return_value = LIST_TRAINING_JOB_RESPONSE_WITHOUT_NEXT_TOKEN

    with patch.object(TrainingJob, "refresh") as mock_refresh:
        index = 0
        while True:
            try:
                next_item = next(iterator)
                assert isinstance(next_item, TrainingJob)

                expected_training_job_data = LIST_TRAINING_JOB_RESPONSE_WITHOUT_NEXT_TOKEN[
                    "TrainingJobSummaries"
                ][index]
                assert next_item.training_job_name == expected_training_job_data["TrainingJobName"]
                assert next_item.training_job_arn == expected_training_job_data["TrainingJobArn"]
                assert next_item.creation_time == expected_training_job_data["CreationTime"]
                assert next_item.training_end_time == expected_training_job_data["TrainingEndTime"]
                assert (
                    next_item.last_modified_time == expected_training_job_data["LastModifiedTime"]
                )
                assert (
                    next_item.training_job_status == expected_training_job_data["TrainingJobStatus"]
                )
                index += 1
            except StopIteration:
                break

        assert mock_refresh.call_count == 2
        assert client.list_training_jobs.call_args_list == [call()]


def test_next_with_next_token(resource_iterator):
    iterator, client, _ = resource_iterator
    client.list_training_jobs.side_effect = [
        LIST_TRAINING_JOB_RESPONSE_WITH_NEXT_TOKEN,
        LIST_TRAINING_JOB_RESPONSE_WITHOUT_NEXT_TOKEN,
    ]

    with patch.object(TrainingJob, "refresh") as mock_refresh:
        index = 0
        while True:
            try:
                next_item = next(iterator)
                assert isinstance(next_item, TrainingJob)

                if index < len(LIST_TRAINING_JOB_RESPONSE_WITH_NEXT_TOKEN["TrainingJobSummaries"]):
                    expected_training_job_data = LIST_TRAINING_JOB_RESPONSE_WITH_NEXT_TOKEN[
                        "TrainingJobSummaries"
                    ][index]
                else:
                    expected_training_job_data = LIST_TRAINING_JOB_RESPONSE_WITHOUT_NEXT_TOKEN[
                        "TrainingJobSummaries"
                    ][
                        index
                        - len(LIST_TRAINING_JOB_RESPONSE_WITH_NEXT_TOKEN["TrainingJobSummaries"])
                    ]

                assert next_item.training_job_name == expected_training_job_data["TrainingJobName"]
                assert next_item.training_job_arn == expected_training_job_data["TrainingJobArn"]
                assert next_item.creation_time == expected_training_job_data["CreationTime"]
                assert next_item.training_end_time == expected_training_job_data["TrainingEndTime"]
                assert (
                    next_item.last_modified_time == expected_training_job_data["LastModifiedTime"]
                )
                assert (
                    next_item.training_job_status == expected_training_job_data["TrainingJobStatus"]
                )
                index += 1
            except StopIteration:
                break

        assert mock_refresh.call_count == 4
        assert client.list_training_jobs.call_args_list == [
            call(),
            call(NextToken=LIST_TRAINING_JOB_RESPONSE_WITH_NEXT_TOKEN["NextToken"]),
        ]


def test_next_with_custom_key_mapping(resource_iterator_with_custom_key_mapping):
    iterator, client, _ = resource_iterator_with_custom_key_mapping
    client.list_data_quality_job_definitions.return_value = (
        LIST_DATA_QUALITY_JOB_DEFINITION_RESPONSE_WITHOUT_NEXT_TOKEN
    )
    iterator.index = 0
    with patch.object(DataQualityJobDefinition, "refresh") as mock_refresh:
        index = 0
        while True:
            try:
                next_item = next(iterator)
                assert isinstance(next_item, DataQualityJobDefinition)
                print(next_item)
                expected_data_quality_job_definition_data = (
                    LIST_DATA_QUALITY_JOB_DEFINITION_RESPONSE_WITHOUT_NEXT_TOKEN[
                        "JobDefinitionSummaries"
                    ][index]
                )

                assert (
                    next_item.job_definition_name
                    == expected_data_quality_job_definition_data["MonitoringJobDefinitionName"]
                )
                assert (
                    next_item.job_definition_arn
                    == expected_data_quality_job_definition_data["MonitoringJobDefinitionArn"]
                )
                assert (
                    next_item.creation_time
                    == expected_data_quality_job_definition_data["CreationTime"]
                )
                assert not hasattr(next_item, "endpoint_name")
                index += 1
            except StopIteration:
                break

        assert mock_refresh.call_count == 2
        assert client.list_data_quality_job_definitions.call_args_list == [call()]


def test_next_with_primitive_class(resource_iterator_with_primitive_class):
    iterator, client, _ = resource_iterator_with_primitive_class
    client.list_aliases.return_value = LIST_ALIASES_RESPONSE_WITHOUT_NEXT_TOKEN

    index = 0
    while True:
        try:
            next_item = next(iterator)
            assert isinstance(next_item, str)
            print(next_item)
            expected_image_version_alias_data = LIST_ALIASES_RESPONSE_WITHOUT_NEXT_TOKEN[
                "SageMakerImageVersionAliases"
            ][index]

            assert next_item == expected_image_version_alias_data["SageMakerImageVersionAlias"]

            assert not hasattr(next_item, "endpoint_name")
            index += 1
        except StopIteration:
            break

        assert client.list_aliases.call_args_list == [call()]


def test_configure_logging_with_default_log_level(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    configure_logging()
    assert logging.getLogger().level == logging.INFO


def test_configure_logging_with_debug_log_level(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    configure_logging()
    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_with_invalid_log_level():
    with pytest.raises(AttributeError):
        configure_logging("INVALID_LOG_LEVEL")


def test_configure_logging_with_explicit_log_level():
    configure_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING


def test_serialize_method_returns_dict():
    additional_s3_data_source = AdditionalS3DataSource(s3_data_type="filestring", s3_uri="s3/uri")
    serialized_data = serialize(additional_s3_data_source)
    assert isinstance(serialized_data, dict)


def test_serialize_method_returns_correct_data():
    additional_s3_data_source = AdditionalS3DataSource(s3_data_type="filestring", s3_uri="s3/uri")
    serialized_data = serialize(additional_s3_data_source)
    assert serialized_data["S3DataType"] == "filestring"
    assert serialized_data["S3Uri"] == "s3/uri"


def test_serialize_method_nested_shape():
    trial_component_parameters = {
        "test_num_value": TrialComponentParameterValue(number_value=1),
        "test_str_value": TrialComponentParameterValue(string_value="string"),
    }
    trial_component = TrialComponent(
        trial_component_name="test", parameters=trial_component_parameters
    )
    serialized_data = serialize(trial_component)
    assert serialized_data["TrialComponentName"] == "test"
    assert serialized_data["Parameters"] == {
        "test_num_value": {
            "NumberValue": 1,
        },
        "test_str_value": {
            "StringValue": "string",
        },
    }
