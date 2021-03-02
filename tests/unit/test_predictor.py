# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import io
import json

import pytest
from mock import Mock, call, patch

from sagemaker.deserializers import CSVDeserializer, PandasDeserializer
from sagemaker.model_monitor.model_monitoring import DEFAULT_REPOSITORY_NAME
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer, CSVSerializer

ENDPOINT = "mxnet_endpoint"
BUCKET_NAME = "mxnet_endpoint"
DEFAULT_CONTENT_TYPE = "application/octet-stream"
CSV_CONTENT_TYPE = "text/csv"
DEFAULT_ACCEPT = "*/*"
RETURN_VALUE = 0
CSV_RETURN_VALUE = "1,2,3\r\n"
PRODUCTION_VARIANT_1 = "PRODUCTION_VARIANT_1"
INFERENCE_ID = "inference-id"

ENDPOINT_DESC = {"EndpointArn": "foo", "EndpointConfigName": ENDPOINT}

ENDPOINT_CONFIG_DESC = {"ProductionVariants": [{"ModelName": "model-1"}, {"ModelName": "model-2"}]}


def empty_sagemaker_session():
    ims = Mock(name="sagemaker_session")
    ims.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    ims.sagemaker_runtime_client = Mock(name="sagemaker_runtime")
    ims.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    ims.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)

    response_body = Mock("body")
    response_body.read = Mock("read", return_value=RETURN_VALUE)
    response_body.close = Mock("close", return_value=None)
    ims.sagemaker_runtime_client.invoke_endpoint = Mock(
        name="invoke_endpoint", return_value={"Body": response_body}
    )
    return ims


def test_predict_call_pass_through():
    sagemaker_session = empty_sagemaker_session()
    predictor = Predictor(ENDPOINT, sagemaker_session)

    data = "untouched"
    result = predictor.predict(data)

    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint.called

    expected_request_args = {
        "Accept": DEFAULT_ACCEPT,
        "Body": data,
        "ContentType": DEFAULT_CONTENT_TYPE,
        "EndpointName": ENDPOINT,
    }

    call_args, kwargs = sagemaker_session.sagemaker_runtime_client.invoke_endpoint.call_args
    assert kwargs == expected_request_args

    assert result == RETURN_VALUE


def test_predict_call_with_target_variant():
    sagemaker_session = empty_sagemaker_session()
    predictor = Predictor(ENDPOINT, sagemaker_session)

    data = "untouched"
    result = predictor.predict(data, target_variant=PRODUCTION_VARIANT_1)

    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint.called

    expected_request_args = {
        "Accept": DEFAULT_ACCEPT,
        "Body": data,
        "ContentType": DEFAULT_CONTENT_TYPE,
        "EndpointName": ENDPOINT,
        "TargetVariant": PRODUCTION_VARIANT_1,
    }

    call_args, kwargs = sagemaker_session.sagemaker_runtime_client.invoke_endpoint.call_args
    assert kwargs == expected_request_args

    assert result == RETURN_VALUE


def test_predict_call_with_inference_id():
    sagemaker_session = empty_sagemaker_session()
    predictor = Predictor(ENDPOINT, sagemaker_session)

    data = "untouched"
    result = predictor.predict(data, inference_id=INFERENCE_ID)

    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint.called

    expected_request_args = {
        "Accept": DEFAULT_ACCEPT,
        "Body": data,
        "ContentType": DEFAULT_CONTENT_TYPE,
        "EndpointName": ENDPOINT,
        "InferenceId": INFERENCE_ID,
    }

    call_args, kwargs = sagemaker_session.sagemaker_runtime_client.invoke_endpoint.call_args
    assert kwargs == expected_request_args

    assert result == RETURN_VALUE


def test_multi_model_predict_call():
    sagemaker_session = empty_sagemaker_session()
    predictor = Predictor(ENDPOINT, sagemaker_session)

    data = "untouched"
    result = predictor.predict(data, target_model="model.tar.gz")

    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint.called

    expected_request_args = {
        "Accept": DEFAULT_ACCEPT,
        "Body": data,
        "ContentType": DEFAULT_CONTENT_TYPE,
        "EndpointName": ENDPOINT,
        "TargetModel": "model.tar.gz",
    }
    call_args, kwargs = sagemaker_session.sagemaker_runtime_client.invoke_endpoint.call_args
    assert kwargs == expected_request_args

    assert result == RETURN_VALUE


def json_sagemaker_session():
    ims = Mock(name="sagemaker_session")
    ims.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    ims.sagemaker_runtime_client = Mock(name="sagemaker_runtime")
    ims.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    ims.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)

    ims.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    ims.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)

    response_body = Mock("body")
    response_body.read = Mock("read", return_value=json.dumps([RETURN_VALUE]))
    response_body.close = Mock("close", return_value=None)
    ims.sagemaker_runtime_client.invoke_endpoint = Mock(
        name="invoke_endpoint",
        return_value={"Body": response_body, "ContentType": "application/json"},
    )
    return ims


def test_predict_call_with_json():
    sagemaker_session = json_sagemaker_session()
    predictor = Predictor(ENDPOINT, sagemaker_session, serializer=JSONSerializer())

    data = [1, 2]
    result = predictor.predict(data)

    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint.called

    expected_request_args = {
        "Accept": DEFAULT_ACCEPT,
        "Body": json.dumps(data),
        "ContentType": "application/json",
        "EndpointName": ENDPOINT,
    }
    call_args, kwargs = sagemaker_session.sagemaker_runtime_client.invoke_endpoint.call_args
    assert kwargs == expected_request_args

    assert result == json.dumps([RETURN_VALUE])


def ret_csv_sagemaker_session():
    ims = Mock(name="sagemaker_session")
    ims.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    ims.sagemaker_runtime_client = Mock(name="sagemaker_runtime")
    ims.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    ims.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)

    ims.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    ims.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)

    response_body = io.BytesIO(bytes(CSV_RETURN_VALUE, "utf-8"))
    ims.sagemaker_runtime_client.invoke_endpoint = Mock(
        name="invoke_endpoint",
        return_value={"Body": response_body, "ContentType": CSV_CONTENT_TYPE},
    )
    return ims


def test_predict_call_with_csv():
    sagemaker_session = ret_csv_sagemaker_session()
    predictor = Predictor(
        ENDPOINT, sagemaker_session, serializer=CSVSerializer(), deserializer=CSVDeserializer()
    )

    data = [1, 2]
    result = predictor.predict(data)

    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint.called

    expected_request_args = {
        "Accept": CSV_CONTENT_TYPE,
        "Body": "1,2",
        "ContentType": CSV_CONTENT_TYPE,
        "EndpointName": ENDPOINT,
    }
    call_args, kwargs = sagemaker_session.sagemaker_runtime_client.invoke_endpoint.call_args
    assert kwargs == expected_request_args

    assert result == [["1", "2", "3"]]


def test_predict_call_with_multiple_accept_types():
    sagemaker_session = ret_csv_sagemaker_session()
    predictor = Predictor(
        ENDPOINT, sagemaker_session, serializer=CSVSerializer(), deserializer=PandasDeserializer()
    )

    data = [1, 2]
    predictor.predict(data)

    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint.called

    expected_request_args = {
        "Accept": "text/csv, application/json",
        "Body": "1,2",
        "ContentType": CSV_CONTENT_TYPE,
        "EndpointName": ENDPOINT,
    }
    call_args, kwargs = sagemaker_session.sagemaker_runtime_client.invoke_endpoint.call_args
    assert kwargs == expected_request_args


@patch("sagemaker.predictor.name_from_base")
def test_update_endpoint_no_args(name_from_base):
    new_endpoint_config_name = "new-endpoint-config"
    name_from_base.return_value = new_endpoint_config_name

    sagemaker_session = empty_sagemaker_session()
    existing_endpoint_config_name = "existing-endpoint-config"

    predictor = Predictor(ENDPOINT, sagemaker_session=sagemaker_session)
    predictor._endpoint_config_name = existing_endpoint_config_name

    predictor.update_endpoint()

    assert ["model-1", "model-2"] == predictor._model_names
    assert new_endpoint_config_name == predictor._endpoint_config_name

    name_from_base.assert_called_with(existing_endpoint_config_name)
    sagemaker_session.create_endpoint_config_from_existing.assert_called_with(
        existing_endpoint_config_name,
        new_endpoint_config_name,
        new_tags=None,
        new_kms_key=None,
        new_data_capture_config_dict=None,
        new_production_variants=None,
    )
    sagemaker_session.update_endpoint.assert_called_with(
        ENDPOINT, new_endpoint_config_name, wait=True
    )


@patch("sagemaker.predictor.production_variant")
@patch("sagemaker.predictor.name_from_base")
def test_update_endpoint_all_args(name_from_base, production_variant):
    new_endpoint_config_name = "new-endpoint-config"
    name_from_base.return_value = new_endpoint_config_name

    sagemaker_session = empty_sagemaker_session()
    existing_endpoint_config_name = "existing-endpoint-config"

    predictor = Predictor(ENDPOINT, sagemaker_session=sagemaker_session)
    predictor._endpoint_config_name = existing_endpoint_config_name

    new_instance_count = 2
    new_instance_type = "ml.c4.xlarge"
    new_accelerator_type = "ml.eia1.medium"
    new_model_name = "new-model"
    new_tags = {"Key": "foo", "Value": "bar"}
    new_kms_key = "new-key"
    new_data_capture_config_dict = {}

    predictor.update_endpoint(
        initial_instance_count=new_instance_count,
        instance_type=new_instance_type,
        accelerator_type=new_accelerator_type,
        model_name=new_model_name,
        tags=new_tags,
        kms_key=new_kms_key,
        data_capture_config_dict=new_data_capture_config_dict,
        wait=False,
    )

    assert [new_model_name] == predictor._model_names
    assert new_endpoint_config_name == predictor._endpoint_config_name

    production_variant.assert_called_with(
        new_model_name,
        new_instance_type,
        initial_instance_count=new_instance_count,
        accelerator_type=new_accelerator_type,
    )
    sagemaker_session.create_endpoint_config_from_existing.assert_called_with(
        existing_endpoint_config_name,
        new_endpoint_config_name,
        new_tags=new_tags,
        new_kms_key=new_kms_key,
        new_data_capture_config_dict=new_data_capture_config_dict,
        new_production_variants=[production_variant.return_value],
    )
    sagemaker_session.update_endpoint.assert_called_with(
        ENDPOINT, new_endpoint_config_name, wait=False
    )


@patch("sagemaker.predictor.production_variant")
@patch("sagemaker.predictor.name_from_base")
def test_update_endpoint_instance_type_and_count(name_from_base, production_variant):
    new_endpoint_config_name = "new-endpoint-config"
    name_from_base.return_value = new_endpoint_config_name

    sagemaker_session = empty_sagemaker_session()
    existing_endpoint_config_name = "existing-endpoint-config"
    existing_model_name = "existing-model"

    predictor = Predictor(ENDPOINT, sagemaker_session=sagemaker_session)
    predictor._endpoint_config_name = existing_endpoint_config_name
    predictor._model_names = [existing_model_name]

    new_instance_count = 2
    new_instance_type = "ml.c4.xlarge"

    predictor.update_endpoint(
        initial_instance_count=new_instance_count,
        instance_type=new_instance_type,
    )

    assert [existing_model_name] == predictor._model_names
    assert new_endpoint_config_name == predictor._endpoint_config_name

    production_variant.assert_called_with(
        existing_model_name,
        new_instance_type,
        initial_instance_count=new_instance_count,
        accelerator_type=None,
    )
    sagemaker_session.create_endpoint_config_from_existing.assert_called_with(
        existing_endpoint_config_name,
        new_endpoint_config_name,
        new_tags=None,
        new_kms_key=None,
        new_data_capture_config_dict=None,
        new_production_variants=[production_variant.return_value],
    )
    sagemaker_session.update_endpoint.assert_called_with(
        ENDPOINT, new_endpoint_config_name, wait=True
    )


def test_update_endpoint_no_instance_type_or_no_instance_count():
    sagemaker_session = empty_sagemaker_session()
    predictor = Predictor(ENDPOINT, sagemaker_session=sagemaker_session)

    bad_args = ({"instance_type": "ml.c4.xlarge"}, {"initial_instance_count": 2})
    for args in bad_args:
        with pytest.raises(ValueError) as exception:
            predictor.update_endpoint(**args)

        expected_msg = "Missing initial_instance_count and/or instance_type."
        assert expected_msg in str(exception.value)


def test_update_endpoint_no_one_default_model_name_with_instance_type_and_count():
    sagemaker_session = empty_sagemaker_session()
    predictor = Predictor(ENDPOINT, sagemaker_session=sagemaker_session)

    with pytest.raises(ValueError) as exception:
        predictor.update_endpoint(initial_instance_count=2, instance_type="ml.c4.xlarge")

    assert "Unable to choose a default model for a new EndpointConfig" in str(exception.value)


def test_delete_endpoint_with_config():
    sagemaker_session = empty_sagemaker_session()
    sagemaker_session.sagemaker_client.describe_endpoint = Mock(
        return_value={"EndpointConfigName": "endpoint-config"}
    )
    predictor = Predictor(ENDPOINT, sagemaker_session=sagemaker_session)
    predictor.delete_endpoint()

    sagemaker_session.delete_endpoint.assert_called_with(ENDPOINT)
    sagemaker_session.delete_endpoint_config.assert_called_with("endpoint-config")


def test_delete_endpoint_only():
    sagemaker_session = empty_sagemaker_session()
    predictor = Predictor(ENDPOINT, sagemaker_session=sagemaker_session)
    predictor.delete_endpoint(delete_endpoint_config=False)

    sagemaker_session.delete_endpoint.assert_called_with(ENDPOINT)
    sagemaker_session.delete_endpoint_config.assert_not_called()


def test_delete_model():
    sagemaker_session = empty_sagemaker_session()
    predictor = Predictor(ENDPOINT, sagemaker_session=sagemaker_session)

    predictor.delete_model()

    expected_call_count = 2
    expected_call_args_list = [call("model-1"), call("model-2")]
    assert sagemaker_session.delete_model.call_count == expected_call_count
    assert sagemaker_session.delete_model.call_args_list == expected_call_args_list


def test_delete_model_fail():
    sagemaker_session = empty_sagemaker_session()
    sagemaker_session.sagemaker_client.delete_model = Mock(
        side_effect=Exception("Could not find model.")
    )
    expected_error_message = "One or more models cannot be deleted, please retry."

    predictor = Predictor(ENDPOINT, sagemaker_session=sagemaker_session)

    with pytest.raises(Exception) as exception:
        predictor.delete_model()
        assert expected_error_message in str(exception.val)


def context_sagemaker_session(summaries=True):
    ims = Mock(name="sagemaker_session")
    ims.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    ims.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)

    if summaries:
        ims.sagemaker_client.list_contexts = Mock(
            return_value={"ContextSummaries": [{"ContextName": "bar"}]}
        )
    else:
        ims.sagemaker_client.list_contexts = Mock(return_value={"ContextSummaries": []})

    ims.sagemaker_client.describe_context = Mock(
        return_value={
            "ContextArn": "foo",
            "ContextName": "bar",
        }
    )

    response_body = Mock("body")
    response_body.read = Mock("read", return_value=json.dumps([RETURN_VALUE]))
    response_body.close = Mock("close", return_value=None)
    ims.sagemaker_runtime_client.invoke_endpoint = Mock(
        name="invoke_endpoint",
        return_value={"Body": response_body, "ContentType": "application/json"},
    )
    return ims


def test_endpoint_context_success():
    session = context_sagemaker_session()
    pdctr = Predictor(ENDPOINT, sagemaker_session=session)

    context = pdctr.endpoint_context()

    assert context


def test_endpoint_context_fail():
    session = context_sagemaker_session(summaries=False)
    pdctr = Predictor(ENDPOINT, sagemaker_session=session)

    context = pdctr.endpoint_context()

    assert not context


@patch("sagemaker.predictor.ModelExplainabilityMonitor.attach")
@patch("sagemaker.predictor.ModelBiasMonitor.attach")
@patch("sagemaker.predictor.ModelQualityMonitor.attach")
@patch("sagemaker.predictor.ModelMonitor.attach")
@patch("sagemaker.predictor.DefaultModelMonitor.attach")
def test_list_monitors(default_model_monitor_attach, *attach_methods):
    sagemaker_session = empty_sagemaker_session()
    sagemaker_session.list_monitoring_schedules = Mock(
        return_value={
            "MonitoringScheduleSummaries": [
                {
                    "MonitoringScheduleName": "default-monitor",
                },
                {
                    "MonitoringScheduleName": "byoc-monitor",
                },
                {
                    "MonitoringScheduleName": "data-quality-monitor",
                    "MonitoringType": "DataQuality",
                },
                {
                    "MonitoringScheduleName": "model-quality-monitor",
                    "MonitoringType": "ModelQuality",
                },
                {
                    "MonitoringScheduleName": "model-bias-monitor",
                    "MonitoringType": "ModelBias",
                },
                {
                    "MonitoringScheduleName": "model-explainability-monitor",
                    "MonitoringType": "ModelExplainability",
                },
            ]
        }
    )
    sagemaker_session.describe_monitoring_schedule = Mock(
        side_effect=[
            {
                "MonitoringScheduleConfig": {
                    "MonitoringJobDefinition": {
                        "MonitoringAppSpecification": {
                            "ImageUri": DEFAULT_REPOSITORY_NAME,
                        }
                    }
                }
            },
            {
                "MonitoringScheduleConfig": {
                    "MonitoringJobDefinition": {
                        "MonitoringAppSpecification": {
                            "ImageUri": "byoc-image",
                        }
                    }
                }
            },
            {
                "MonitoringScheduleConfig": {
                    "MonitoringType": "DataQuality",
                    "MonitoringJobDefinitionName": "data-quality-job-definition",
                }
            },
            {
                "MonitoringScheduleConfig": {
                    "MonitoringType": "ModelQuality",
                    "MonitoringJobDefinitionName": "model-quality-job-definition",
                }
            },
        ]
    )
    predictor = Predictor(ENDPOINT, sagemaker_session=sagemaker_session)
    predictor.list_monitors()
    for attach_method in attach_methods:
        attach_method.assert_called_once()
    assert default_model_monitor_attach.call_count == 2


def test_list_monitors_unknown_monitoring_type():
    sagemaker_session = empty_sagemaker_session()
    sagemaker_session.list_monitoring_schedules = Mock(
        return_value={
            "MonitoringScheduleSummaries": [
                {
                    "MonitoringScheduleName": "model-explainability-monitor",
                    "MonitoringType": "UnknownType",
                },
            ]
        }
    )
    sagemaker_session.describe_monitoring_schedule = Mock(
        side_effect=[
            {
                "MonitoringScheduleConfig": {
                    "MonitoringType": "UnknownType",
                    "MonitoringJobDefinitionName": "unknown-job-definition",
                }
            },
        ]
    )
    predictor = Predictor(ENDPOINT, sagemaker_session=sagemaker_session)
    with pytest.raises(TypeError):
        predictor.list_monitors()
