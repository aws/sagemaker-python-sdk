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

import pytest
import random
import string
from mock import MagicMock, Mock, patch
from sagemaker.experiments.experiment import Experiment
from sagemaker.experiments.run import Run
from sagemaker.experiments.trial import _Trial
from sagemaker.experiments.trial_component import _TrialComponent

from sagemaker.remote_function.core.stored_function import (
    StoredFunction,
    JSON_SERIALIZED_RESULT_KEY,
    _SerializedData,
)
from sagemaker.remote_function.core.serialization import (
    deserialize_obj_from_s3,
    serialize_obj_to_s3,
    CloudpickleSerializer,
)
from sagemaker.remote_function.core.pipeline_variables import (
    Context,
    convert_pipeline_variables_to_pickleable,
)
from sagemaker.remote_function.errors import SerializationError

from sagemaker.workflow.function_step import _FunctionStep, DelayedReturn
from sagemaker.workflow.parameters import ParameterFloat

from tests.unit.sagemaker.experiments.helpers import (
    TEST_EXP_DISPLAY_NAME,
    TEST_EXP_NAME,
    TEST_RUN_DISPLAY_NAME,
    TEST_RUN_NAME,
    mock_tc_load_or_create_func,
    mock_trial_load_or_create_func,
)

KMS_KEY = "kms-key"
HMAC_KEY = "some-hmac-key"
FUNCTION_FOLDER = "function"
ARGUMENT_FOLDER = "arguments"
RESULT_FOLDER = "results"
PIPELINE_BUILD_TIME = "2022-05-10T17:30:20Z"

mock_s3 = {}


def random_s3_uri():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=10))


def upload_bytes(b, s3_uri, kms_key=None, sagemaker_session=None):
    assert kms_key == KMS_KEY
    mock_s3[s3_uri] = b


def read_bytes(s3_uri, sagemaker_session=None):
    return mock_s3[s3_uri]


def quadratic(x=2, *, a=1, b=0, c=0):
    return a * x * x + b * x + c


def log_bigger(a, b, run: Run):
    if a >= b:
        run.log_metric("bigger", a)
    else:
        run.log_metric("bigger", b)


@pytest.mark.parametrize(
    "args, kwargs",
    [([], {}), ([3], {}), ([], {"a": 2, "b": 1, "c": 1})],
)
@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload_bytes)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read_bytes)
@patch("sagemaker.s3.S3Uploader.upload")
@patch("sagemaker.s3.S3Downloader.download")
def test_save_and_load(s3_source_dir_download, s3_source_dir_upload, args, kwargs):
    session = Mock()
    s3_base_uri = random_s3_uri()

    stored_function = StoredFunction(
        sagemaker_session=session, s3_base_uri=s3_base_uri, s3_kms_key=KMS_KEY, hmac_key=HMAC_KEY
    )
    stored_function.save(quadratic, *args, **kwargs)
    stored_function.load_and_invoke()

    assert deserialize_obj_from_s3(
        session, s3_uri=f"{s3_base_uri}/results", hmac_key=HMAC_KEY
    ) == quadratic(*args, **kwargs)


@patch(
    "sagemaker.experiments.run.Experiment._load_or_create",
    MagicMock(return_value=Experiment(experiment_name=TEST_EXP_NAME)),
)
@patch(
    "sagemaker.experiments.run._Trial._load_or_create",
    MagicMock(side_effect=mock_trial_load_or_create_func),
)
@patch.object(_Trial, "add_trial_component", MagicMock(return_value=None))
@patch(
    "sagemaker.experiments.run._TrialComponent._load_or_create",
    MagicMock(side_effect=mock_tc_load_or_create_func),
)
@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload_bytes)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read_bytes)
@patch.object(_TrialComponent, "save")
@patch("sagemaker.s3.S3Uploader.upload")
@patch("sagemaker.s3.S3Downloader.download")
def test_save_with_parameter_of_run_type(
    s3_source_dir_download, s3_source_dir_upload, mock_tc_save
):
    session = Mock()
    s3_base_uri = random_s3_uri()
    session.sagemaker_client.search.return_value = {"Results": []}

    run = Run(
        experiment_name=TEST_EXP_NAME,
        run_name=TEST_RUN_NAME,
        experiment_display_name=TEST_EXP_DISPLAY_NAME,
        run_display_name=TEST_RUN_DISPLAY_NAME,
        sagemaker_session=session,
    )
    stored_function = StoredFunction(
        sagemaker_session=session, s3_base_uri=s3_base_uri, s3_kms_key=KMS_KEY, hmac_key=HMAC_KEY
    )
    with pytest.raises(SerializationError) as e:
        stored_function.save(log_bigger, 1, 2, run)
        assert isinstance(e.__cause__, NotImplementedError)


@pytest.mark.parametrize(
    "step_name, execution_id, upload_path",
    [
        (None, None, "s3://base_uri/"),
        ("step-name", "execution-id", "s3://base_uri/step-name/" + PIPELINE_BUILD_TIME + "/"),
    ],
)
@patch("sagemaker.remote_function.core.serialization.serialize_obj_to_s3")
@patch("sagemaker.remote_function.core.serialization.serialize_func_to_s3")
def test_save_s3_paths_verification(
    serialize_func, serialize_obj, step_name, execution_id, upload_path
):
    session = Mock()
    s3_base_uri = "s3://base_uri/"

    stored_function = StoredFunction(
        sagemaker_session=session,
        s3_base_uri=s3_base_uri,
        s3_kms_key=KMS_KEY,
        hmac_key=HMAC_KEY,
        context=Context(
            step_name=step_name,
            execution_id=execution_id,
            func_step_s3_dir=PIPELINE_BUILD_TIME if step_name else None,
        ),
    )

    stored_function.save(quadratic, (3))

    serialize_func.assert_called_once_with(
        func=quadratic,
        sagemaker_session=session,
        s3_uri=(upload_path + FUNCTION_FOLDER),
        s3_kms_key=KMS_KEY,
        hmac_key=HMAC_KEY,
    )
    serialize_obj.assert_called_once_with(
        obj=((3,), {}),
        sagemaker_session=session,
        s3_uri=(upload_path + ARGUMENT_FOLDER),
        hmac_key=HMAC_KEY,
        s3_kms_key=KMS_KEY,
    )


@pytest.mark.parametrize(
    "step_name, execution_id, upload_path, download_path",
    [
        (None, None, "s3://base_uri/", "s3://base_uri/"),
        (
            "step-name",
            "execution-id",
            "s3://base_uri/execution-id/step-name/",
            "s3://base_uri/step-name/" + PIPELINE_BUILD_TIME + "/",
        ),
    ],
)
@patch("sagemaker.remote_function.core.serialization.serialize_obj_to_s3")
@patch(
    "sagemaker.remote_function.core.serialization.deserialize_obj_from_s3", return_value=([3], {})
)
@patch(
    "sagemaker.remote_function.core.serialization.deserialize_func_from_s3", return_value=quadratic
)
def test_load_and_invoke_s3_paths_verification(
    deserialize_func,
    deserialize_obj,
    serialize_obj,
    step_name,
    execution_id,
    upload_path,
    download_path,
):
    session = Mock()
    s3_base_uri = "s3://base_uri/"

    stored_function = StoredFunction(
        sagemaker_session=session,
        s3_base_uri=s3_base_uri,
        s3_kms_key=KMS_KEY,
        hmac_key=HMAC_KEY,
        context=Context(
            step_name=step_name,
            execution_id=execution_id,
            func_step_s3_dir=PIPELINE_BUILD_TIME if step_name else None,
        ),
    )

    stored_function.load_and_invoke()

    deserialize_func.assert_called_once_with(
        sagemaker_session=session, s3_uri=(download_path + FUNCTION_FOLDER), hmac_key=HMAC_KEY
    )
    deserialize_obj.assert_called_once_with(
        sagemaker_session=session,
        s3_uri=(download_path + ARGUMENT_FOLDER),
        hmac_key=HMAC_KEY,
    )

    result = deserialize_func.return_value(
        *deserialize_obj.return_value[0], **deserialize_obj.return_value[1]
    )

    serialize_obj.assert_called_once_with(
        obj=result,
        sagemaker_session=session,
        s3_uri=(upload_path + RESULT_FOLDER),
        hmac_key=HMAC_KEY,
        s3_kms_key=KMS_KEY,
    )


@pytest.mark.parametrize(
    "serialize_output_to_json",
    [False, True],
)
@patch("sagemaker.remote_function.core.serialization.json_serialize_obj_to_s3")
@patch("sagemaker.remote_function.core.serialization.serialize_obj_to_s3", Mock())
@patch(
    "sagemaker.remote_function.core.serialization.deserialize_obj_from_s3", return_value=([3], {})
)
@patch(
    "sagemaker.remote_function.core.serialization.deserialize_func_from_s3", return_value=quadratic
)
def test_load_and_invoke_json_serialization(
    deserialize_func,
    deserialize_obj,
    json_serialize_obj,
    serialize_output_to_json,
):
    session = Mock()
    s3_base_uri = "s3://base_uri/"

    stored_function = StoredFunction(
        sagemaker_session=session,
        s3_base_uri=s3_base_uri,
        s3_kms_key=KMS_KEY,
        hmac_key=HMAC_KEY,
        context=Context(
            serialize_output_to_json=serialize_output_to_json,
        ),
    )

    stored_function.load_and_invoke()

    result = deserialize_func.return_value(
        *deserialize_obj.return_value[0], **deserialize_obj.return_value[1]
    )

    if serialize_output_to_json:
        json_serialize_obj.assert_called_once_with(
            obj=result,
            json_key=JSON_SERIALIZED_RESULT_KEY,
            sagemaker_session=session,
            s3_uri=(s3_base_uri + RESULT_FOLDER + "/results.json"),
            s3_kms_key=KMS_KEY,
        )
    else:
        json_serialize_obj.assert_not_called()


@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload_bytes)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read_bytes)
@patch("sagemaker.s3.S3Uploader.upload", MagicMock())
@patch("sagemaker.s3.S3Downloader.download", MagicMock())
def test_save_and_load_with_pipeline_variable(monkeypatch):
    session = Mock()
    s3_base_uri = random_s3_uri()
    func1_result_path = f"{s3_base_uri}/execution-id/func_1/results"

    function_step = _FunctionStep(name="func_1", display_name=None, description=None)
    x = DelayedReturn(function_step=function_step)
    serialize_obj_to_s3(3.0, session, func1_result_path, HMAC_KEY, KMS_KEY)

    stored_function = StoredFunction(
        sagemaker_session=session,
        s3_base_uri=s3_base_uri,
        s3_kms_key=KMS_KEY,
        hmac_key=HMAC_KEY,
        context=Context(
            property_references={
                "Parameters.a": "1.0",
                "Parameters.b": "2.0",
                "Parameters.c": "3.0",
                "Steps.func_1.OutputDataConfig.S3OutputPath": func1_result_path,
            },
            execution_id="execution-id",
            step_name="func_2",
        ),
    )

    func_args, func_kwargs = convert_pipeline_variables_to_pickleable(
        func_args=(x,),
        func_kwargs={
            "a": ParameterFloat("a"),
            "b": ParameterFloat("b"),
            "c": ParameterFloat("c"),
        },
    )

    test_serialized_data = _SerializedData(
        func=CloudpickleSerializer.serialize(quadratic),
        args=CloudpickleSerializer.serialize((func_args, func_kwargs)),
    )

    stored_function.save_pipeline_step_function(test_serialized_data)
    stored_function.load_and_invoke()

    func2_result_path = f"{s3_base_uri}/execution-id/func_2/results"
    assert deserialize_obj_from_s3(
        session, s3_uri=func2_result_path, hmac_key=HMAC_KEY
    ) == quadratic(3.0, a=1.0, b=2.0, c=3.0)


@patch("sagemaker.remote_function.core.serialization._upload_payload_and_metadata_to_s3")
@patch("sagemaker.remote_function.job._JobSettings")
def test_save_pipeline_step_function(mock_job_settings, upload_payload):
    session = Mock()
    s3_base_uri = random_s3_uri()
    mock_job_settings.s3_root_uri = s3_base_uri

    stored_function = StoredFunction(
        sagemaker_session=session,
        s3_base_uri=s3_base_uri,
        s3_kms_key=KMS_KEY,
        hmac_key=HMAC_KEY,
        context=Context(
            step_name="step_name",
            execution_id="execution_id",
        ),
    )

    func_args, func_kwargs = convert_pipeline_variables_to_pickleable(
        func_args=(1,),
        func_kwargs={
            "a": 2,
            "b": 3,
        },
    )

    test_serialized_data = _SerializedData(
        func=CloudpickleSerializer.serialize(quadratic),
        args=CloudpickleSerializer.serialize((func_args, func_kwargs)),
    )
    stored_function.save_pipeline_step_function(test_serialized_data)

    assert upload_payload.call_count == 2
