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

import random
import string
import pytest

from mock import patch, Mock
from sagemaker.experiments.run import Run
from sagemaker.workflow.parameters import ParameterInteger
from sagemaker.workflow.function_step import DelayedReturn
from sagemaker.remote_function.core.serialization import (
    serialize_func_to_s3,
    deserialize_func_from_s3,
    serialize_obj_to_s3,
    deserialize_obj_from_s3,
    serialize_exception_to_s3,
    deserialize_exception_from_s3,
)
from sagemaker.remote_function.errors import ServiceError, SerializationError, DeserializationError
from tblib import pickling_support

KMS_KEY = "kms-key"
HMAC_KEY = "some-hmac-key"


mock_s3 = {}


def random_s3_uri():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=10))


def upload(b, s3_uri, kms_key=None, sagemaker_session=None):
    assert kms_key == KMS_KEY
    mock_s3[s3_uri] = b


def read(s3_uri, sagemaker_session=None):
    return mock_s3[s3_uri]


def upload_error(b, s3_uri, kms_key=None, sagemaker_session=None):
    raise RuntimeError("some failure when upload_bytes")


def read_error(s3_uri, sagemaker_session=None):
    raise RuntimeError("some failure when read_bytes")


@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read)
def test_serialize_deserialize_func():
    def square(x):
        return x * x

    s3_uri = random_s3_uri()
    serialize_func_to_s3(
        func=square, sagemaker_session=Mock(), s3_uri=s3_uri, s3_kms_key=KMS_KEY, hmac_key=HMAC_KEY
    )

    del square

    deserialized = deserialize_func_from_s3(
        sagemaker_session=Mock(), s3_uri=s3_uri, hmac_key=HMAC_KEY
    )

    assert deserialized(3) == 9


@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read)
def test_serialize_deserialize_lambda():

    s3_uri = random_s3_uri()
    serialize_func_to_s3(
        func=lambda x: x * x,
        sagemaker_session=Mock(),
        s3_uri=s3_uri,
        s3_kms_key=KMS_KEY,
        hmac_key=HMAC_KEY,
    )

    deserialized = deserialize_func_from_s3(
        sagemaker_session=Mock(), s3_uri=s3_uri, hmac_key=HMAC_KEY
    )

    assert deserialized(3) == 9


@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read)
@patch("sagemaker.experiments.run.Experiment")
@patch("sagemaker.experiments.run._Trial")
@patch("sagemaker.experiments.run._TrialComponent._load_or_create", return_value=(Mock(), False))
@patch("sagemaker.experiments.run._MetricsManager")
@patch("sagemaker.remote_function.job.Session")
def test_serialize_func_referencing_to_run(*args, **kwargs):

    with Run(experiment_name="exp_name", run_name="run_name") as run:

        def train(x):
            return run.log_metric()

    s3_uri = random_s3_uri()
    with pytest.raises(
        SerializationError,
        match="or instantiate a new Run in the function.",
    ):
        serialize_func_to_s3(
            func=train,
            sagemaker_session=Mock(),
            s3_uri=s3_uri,
            s3_kms_key=KMS_KEY,
            hmac_key=HMAC_KEY,
        )


@pytest.mark.parametrize(
    "pipeline_variable",
    [
        ParameterInteger(name="var1", default_value=1),
        DelayedReturn(function_step=Mock()),
    ],
)
@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read)
def test_serialize_func_referencing_to_pipeline_variables(pipeline_variable):
    def func(x):
        print(pipeline_variable)

    s3_uri = random_s3_uri()
    with pytest.raises(
        SerializationError,
        match="Please pass the pipeline variable to the function decorated with @step as an argument",
    ):
        serialize_func_to_s3(
            func=func,
            sagemaker_session=Mock(),
            s3_uri=s3_uri,
            s3_kms_key=KMS_KEY,
            hmac_key=HMAC_KEY,
        )


@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read)
@patch("cloudpickle.CloudPickler")
def test_serialize_func_serialization_error(mock_cloudpickler):
    mock_cloudpickler.side_effect = RuntimeError("some failure when dumps")

    def square(x):
        return x * x

    s3_uri = random_s3_uri()

    with pytest.raises(
        SerializationError,
        match=r"Error when serializing object of type \[function\]: RuntimeError\('some failure when dumps'\)",
    ):
        serialize_func_to_s3(
            func=square,
            sagemaker_session=Mock(),
            s3_uri=s3_uri,
            s3_kms_key=KMS_KEY,
            hmac_key=HMAC_KEY,
        )


@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read)
@patch("cloudpickle.loads")
def test_deserialize_func_deserialization_error(mock_cloudpickle_loads):
    mock_cloudpickle_loads.side_effect = RuntimeError("some failure when loads")

    def square(x):
        return x * x

    s3_uri = random_s3_uri()

    serialize_func_to_s3(
        func=square, sagemaker_session=Mock(), s3_uri=s3_uri, s3_kms_key=KMS_KEY, hmac_key=HMAC_KEY
    )

    del square

    with pytest.raises(
        DeserializationError,
        match=rf"Error when deserializing bytes downloaded from {s3_uri}/payload.pkl: "
        + r"RuntimeError\('some failure when loads'\). "
        + r"NOTE: this may be caused by inconsistent sagemaker python sdk versions",
    ):
        deserialize_func_from_s3(sagemaker_session=Mock(), s3_uri=s3_uri, hmac_key=HMAC_KEY)


@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read)
def test_deserialize_func_corrupt_metadata():
    def square(x):
        return x * x

    s3_uri = random_s3_uri()

    serialize_func_to_s3(
        func=square, sagemaker_session=Mock(), s3_uri=s3_uri, s3_kms_key=KMS_KEY, hmac_key=HMAC_KEY
    )
    mock_s3[f"{s3_uri}/metadata.json"] = b"not json serializable"

    del square

    with pytest.raises(DeserializationError, match=r"Corrupt metadata file."):
        deserialize_func_from_s3(sagemaker_session=Mock(), s3_uri=s3_uri, hmac_key=HMAC_KEY)


@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read)
def test_deserialize_integrity_check_failed():
    def square(x):
        return x * x

    s3_uri = random_s3_uri()
    serialize_func_to_s3(
        func=square, sagemaker_session=Mock(), s3_uri=s3_uri, s3_kms_key=KMS_KEY, hmac_key=HMAC_KEY
    )

    del square

    with pytest.raises(
        DeserializationError, match=r"Integrity check for the serialized function or data failed."
    ):
        deserialize_func_from_s3(sagemaker_session=Mock(), s3_uri=s3_uri, hmac_key="invalid_key")


@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read)
def test_serialize_deserialize_custom_class_data():
    class MyData:
        def __init__(self, x):
            self.x = x

    my_data = MyData(10)

    s3_uri = random_s3_uri()
    serialize_obj_to_s3(
        my_data, sagemaker_session=Mock(), s3_uri=s3_uri, s3_kms_key=KMS_KEY, hmac_key=HMAC_KEY
    )

    del my_data
    del MyData

    deserialized = deserialize_obj_from_s3(
        sagemaker_session=Mock(), s3_uri=s3_uri, hmac_key=HMAC_KEY
    )

    assert deserialized.x == 10


@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read)
def test_serialize_deserialize_data_built_in_types():

    my_data = {"a": [10]}

    s3_uri = random_s3_uri()
    serialize_obj_to_s3(
        my_data, sagemaker_session=Mock(), s3_uri=s3_uri, s3_kms_key=KMS_KEY, hmac_key=HMAC_KEY
    )

    del my_data

    deserialized = deserialize_obj_from_s3(
        sagemaker_session=Mock(), s3_uri=s3_uri, hmac_key=HMAC_KEY
    )

    assert deserialized == {"a": [10]}


@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read)
def test_serialize_deserialize_none():

    s3_uri = random_s3_uri()
    serialize_obj_to_s3(
        None, sagemaker_session=Mock(), s3_uri=s3_uri, s3_kms_key=KMS_KEY, hmac_key=HMAC_KEY
    )

    deserialized = deserialize_obj_from_s3(
        sagemaker_session=Mock(), s3_uri=s3_uri, hmac_key=HMAC_KEY
    )

    assert deserialized is None


@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read)
@patch("sagemaker.experiments.run.Experiment")
@patch("sagemaker.experiments.run._Trial")
@patch("sagemaker.experiments.run._TrialComponent._load_or_create", return_value=(Mock(), False))
@patch("sagemaker.experiments.run._MetricsManager")
@patch("sagemaker.remote_function.job.Session")
def test_serialize_run(*args, **kwargs):
    with Run(experiment_name="exp_name", run_name="run_name") as run:
        s3_uri = random_s3_uri()
        with pytest.raises(
            SerializationError,
            match="or instantiate a new Run in the function.",
        ):
            serialize_obj_to_s3(
                obj=run,
                sagemaker_session=Mock(),
                s3_uri=s3_uri,
                s3_kms_key=KMS_KEY,
                hmac_key=HMAC_KEY,
            )


@pytest.mark.parametrize(
    "pipeline_variable",
    [
        ParameterInteger(name="var1", default_value=1),
        DelayedReturn(function_step=Mock()),
    ],
)
@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read)
def test_serialize_pipeline_variables(pipeline_variable):
    s3_uri = random_s3_uri()
    with pytest.raises(
        SerializationError,
        match="Please pass the pipeline variable to the function decorated with @step as an argument",
    ):
        serialize_obj_to_s3(
            obj=(pipeline_variable,),
            sagemaker_session=Mock(),
            s3_uri=s3_uri,
            s3_kms_key=KMS_KEY,
            hmac_key=HMAC_KEY,
        )


@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read)
@patch("cloudpickle.CloudPickler")
def test_serialize_obj_serialization_error(mock_cloudpickler):
    mock_cloudpickler.side_effect = RuntimeError("some failure when dumps")

    class MyData:
        def __init__(self, x):
            self.x = x

    my_data = MyData(10)
    s3_uri = random_s3_uri()

    with pytest.raises(
        SerializationError,
        match=r"Error when serializing object of type \[MyData\]: RuntimeError\('some failure when dumps'\)",
    ):
        serialize_obj_to_s3(
            obj=my_data,
            sagemaker_session=Mock(),
            s3_uri=s3_uri,
            s3_kms_key=KMS_KEY,
            hmac_key=HMAC_KEY,
        )


@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read)
@patch("cloudpickle.loads")
def test_deserialize_obj_deserialization_error(mock_cloudpickle_loads):
    mock_cloudpickle_loads.side_effect = RuntimeError("some failure when loads")

    class MyData:
        def __init__(self, x):
            self.x = x

    my_data = MyData(10)
    s3_uri = random_s3_uri()

    serialize_obj_to_s3(
        obj=my_data, sagemaker_session=Mock(), s3_uri=s3_uri, s3_kms_key=KMS_KEY, hmac_key=HMAC_KEY
    )

    del my_data
    del MyData

    with pytest.raises(
        DeserializationError,
        match=rf"Error when deserializing bytes downloaded from {s3_uri}/payload.pkl: "
        + r"RuntimeError\('some failure when loads'\). "
        + r"NOTE: this may be caused by inconsistent sagemaker python sdk versions",
    ):
        deserialize_obj_from_s3(sagemaker_session=Mock(), s3_uri=s3_uri, hmac_key=HMAC_KEY)


@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload_error)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read_error)
def test_serialize_deserialize_service_error():

    my_func = lambda a: a + 10  # noqa: E731

    s3_uri = random_s3_uri()
    with pytest.raises(
        ServiceError,
        match=rf"Failed to upload serialized bytes to {s3_uri}/payload.pkl: "
        + r"RuntimeError\('some failure when upload_bytes'\)",
    ):
        serialize_func_to_s3(
            func=my_func,
            sagemaker_session=Mock(),
            s3_uri=s3_uri,
            s3_kms_key=KMS_KEY,
            hmac_key=HMAC_KEY,
        )

    del my_func

    with pytest.raises(
        ServiceError,
        match=rf"Failed to read serialized bytes from {s3_uri}/metadata.json: "
        + r"RuntimeError\('some failure when read_bytes'\)",
    ):
        deserialize_func_from_s3(sagemaker_session=Mock(), s3_uri=s3_uri, hmac_key=HMAC_KEY)


@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read)
def test_serialize_deserialize_exception_with_traceback():
    s3_uri = random_s3_uri()

    class CustomError(Exception):
        ...

    def func_a():
        raise TypeError

    def func_b():
        try:
            func_a()
        except TypeError as first_exception:
            raise CustomError("Some error") from first_exception

    try:
        func_b()
    except Exception as e:
        pickling_support.install()
        serialize_obj_to_s3(
            e, sagemaker_session=Mock(), s3_uri=s3_uri, s3_kms_key=KMS_KEY, hmac_key=HMAC_KEY
        )

    with pytest.raises(CustomError, match="Some error") as exc_info:
        raise deserialize_obj_from_s3(sagemaker_session=Mock(), s3_uri=s3_uri, hmac_key=HMAC_KEY)
    assert type(exc_info.value.__cause__) is TypeError


@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read)
def test_serialize_deserialize_custom_exception_with_traceback():
    s3_uri = random_s3_uri()

    class CustomError(Exception):
        ...

    def func_a():
        raise TypeError

    def func_b():
        try:
            func_a()
        except TypeError as first_exception:
            raise CustomError("Some error") from first_exception

    try:
        func_b()
    except Exception as e:
        serialize_exception_to_s3(
            e, sagemaker_session=Mock(), s3_uri=s3_uri, s3_kms_key=KMS_KEY, hmac_key=HMAC_KEY
        )

    with pytest.raises(CustomError, match="Some error") as exc_info:
        raise deserialize_exception_from_s3(
            sagemaker_session=Mock(), s3_uri=s3_uri, hmac_key=HMAC_KEY
        )
    assert type(exc_info.value.__cause__) is TypeError


@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read)
def test_serialize_deserialize_remote_function_error_with_traceback():
    s3_uri = random_s3_uri()

    class CustomError(Exception):
        ...

    def func_a():
        raise TypeError

    def func_b():
        try:
            func_a()
        except TypeError as first_exception:
            raise ServiceError("Some error") from first_exception

    try:
        func_b()
    except Exception as e:
        serialize_exception_to_s3(
            e, sagemaker_session=Mock(), s3_uri=s3_uri, s3_kms_key=KMS_KEY, hmac_key=HMAC_KEY
        )

    with pytest.raises(ServiceError, match="Some error") as exc_info:
        raise deserialize_exception_from_s3(
            sagemaker_session=Mock(), s3_uri=s3_uri, hmac_key=HMAC_KEY
        )
    assert type(exc_info.value.__cause__) is TypeError
