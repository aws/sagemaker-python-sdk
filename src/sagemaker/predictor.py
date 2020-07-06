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
"""Placeholder docstring"""
from __future__ import print_function, absolute_import

import codecs
import csv
import json
import six
from six import StringIO, BytesIO
import numpy as np

from sagemaker.content_types import CONTENT_TYPE_JSON, CONTENT_TYPE_CSV, CONTENT_TYPE_NPY
from sagemaker.deserializers import BaseDeserializer
from sagemaker.model_monitor import DataCaptureConfig
from sagemaker.serializers import BaseSerializer
from sagemaker.session import production_variant, Session
from sagemaker.utils import name_from_base

from sagemaker.model_monitor.model_monitoring import (
    _DEFAULT_MONITOR_IMAGE_URI_WITH_PLACEHOLDERS,
    ModelMonitor,
    DefaultModelMonitor,
)


class Predictor(object):
    """Make prediction requests to an Amazon SageMaker endpoint."""

    def __init__(
        self,
        endpoint_name,
        sagemaker_session=None,
        serializer=None,
        deserializer=None,
        content_type=None,
        accept=None,
    ):
        """Initialize a ``Predictor``.

        Behavior for serialization of input data and deserialization of
        result data can be configured through initializer arguments. If not
        specified, a sequence of bytes is expected and the API sends it in the
        request body without modifications. In response, the API returns the
        sequence of bytes from the prediction result without any modifications.

        Args:
            endpoint_name (str): Name of the Amazon SageMaker endpoint to which
                requests are sent.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.
            serializer (sagemaker.serializers.BaseSerializer): A serializer
                object, used to encode data for an inference endpoint
                (default: None).
            deserializer (sagemaker.deserializers.BaseDeserializer): A
                deserializer object, used to decode data from an inference
                endpoint (default: None).
            content_type (str): The invocation's "ContentType", overriding any
                ``CONTENT_TYPE`` from the serializer (default: None).
            accept (str): The invocation's "Accept", overriding any accept from
                the deserializer (default: None).
        """
        if serializer is not None and not isinstance(serializer, BaseSerializer):
            serializer = LegacySerializer(serializer)
        if deserializer is not None and not isinstance(deserializer, BaseDeserializer):
            deserializer = LegacyDeserializer(deserializer)

        self.endpoint_name = endpoint_name
        self.sagemaker_session = sagemaker_session or Session()
        self.serializer = serializer
        self.deserializer = deserializer
        self.content_type = content_type or getattr(serializer, "CONTENT_TYPE", None)
        self.accept = accept or getattr(deserializer, "ACCEPT", None)
        self._endpoint_config_name = self._get_endpoint_config_name()
        self._model_names = self._get_model_names()

    def predict(self, data, initial_args=None, target_model=None, target_variant=None):
        """Return the inference from the specified endpoint.

        Args:
            data (object): Input data for which you want the model to provide
                inference. If a serializer was specified when creating the
                Predictor, the result of the serializer is sent as input
                data. Otherwise the data must be sequence of bytes, and the
                predict method then sends the bytes in the request body as is.
            initial_args (dict[str,str]): Optional. Default arguments for boto3
                ``invoke_endpoint`` call. Default is None (no default
                arguments).
            target_model (str): S3 model artifact path to run an inference request on,
                in case of a multi model endpoint. Does not apply to endpoints hosting
                single model (Default: None)
            target_variant (str): The name of the production variant to run an inference
            request on (Default: None). Note that the ProductionVariant identifies the model
            you want to host and the resources you want to deploy for hosting it.

        Returns:
            object: Inference for the given input. If a deserializer was specified when creating
                the Predictor, the result of the deserializer is
                returned. Otherwise the response returns the sequence of bytes
                as is.
        """

        request_args = self._create_request_args(data, initial_args, target_model, target_variant)
        response = self.sagemaker_session.sagemaker_runtime_client.invoke_endpoint(**request_args)
        return self._handle_response(response)

    def _handle_response(self, response):
        """
        Args:
            response:
        """
        response_body = response["Body"]
        if self.deserializer is not None:
            # It's the deserializer's responsibility to close the stream
            return self.deserializer.deserialize(response_body, response["ContentType"])
        data = response_body.read()
        response_body.close()
        return data

    def _create_request_args(self, data, initial_args=None, target_model=None, target_variant=None):
        """
        Args:
            data:
            initial_args:
            target_model:
            target_variant:
        """
        args = dict(initial_args) if initial_args else {}

        if "EndpointName" not in args:
            args["EndpointName"] = self.endpoint_name

        if self.content_type and "ContentType" not in args:
            args["ContentType"] = self.content_type

        if self.accept and "Accept" not in args:
            args["Accept"] = self.accept

        if target_model:
            args["TargetModel"] = target_model

        if target_variant:
            args["TargetVariant"] = target_variant

        if self.serializer is not None:
            data = self.serializer.serialize(data)

        args["Body"] = data
        return args

    def update_endpoint(
        self,
        initial_instance_count=None,
        instance_type=None,
        accelerator_type=None,
        model_name=None,
        tags=None,
        kms_key=None,
        data_capture_config_dict=None,
        wait=True,
    ):
        """Update the existing endpoint with the provided attributes.

        This creates a new EndpointConfig in the process. If ``initial_instance_count``,
        ``instance_type``, ``accelerator_type``, or ``model_name`` is specified, then a new
        ProductionVariant configuration is created; values from the existing configuration
        are not preserved if any of those parameters are specified.

        Args:
            initial_instance_count (int): The initial number of instances to run in the endpoint.
                This is required if ``instance_type``, ``accelerator_type``, or ``model_name`` is
                specified. Otherwise, the values from the existing endpoint configuration's
                ProductionVariants are used.
            instance_type (str): The EC2 instance type to deploy the endpoint to.
                This is required if ``initial_instance_count`` or ``accelerator_type`` is specified.
                Otherwise, the values from the existing endpoint configuration's
                ``ProductionVariants`` are used.
            accelerator_type (str): The type of Elastic Inference accelerator to attach to
                the endpoint, e.g. "ml.eia1.medium". If not specified, and
                ``initial_instance_count``, ``instance_type``, and ``model_name`` are also ``None``,
                the values from the existing endpoint configuration's ``ProductionVariants`` are
                used. Otherwise, no Elastic Inference accelerator is attached to the endpoint.
            model_name (str): The name of the model to be associated with the endpoint.
                This is required if ``initial_instance_count``, ``instance_type``, or
                ``accelerator_type`` is specified and if there is more than one model associated
                with the endpoint. Otherwise, the existing model for the endpoint is used.
            tags (list[dict[str, str]]): The list of tags to add to the endpoint
                config. If not specified, the tags of the existing endpoint configuration are used.
                If any of the existing tags are reserved AWS ones (i.e. begin with "aws"),
                they are not carried over to the new endpoint configuration.
            kms_key (str): The KMS key that is used to encrypt the data on the storage volume
                attached to the instance hosting the endpoint If not specified,
                the KMS key of the existing endpoint configuration is used.
            data_capture_config_dict (dict): The endpoint data capture configuration
                for use with Amazon SageMaker Model Monitoring. If not specified,
                the data capture configuration of the existing endpoint configuration is used.

        Raises:
            ValueError: If there is not enough information to create a new ``ProductionVariant``:

                - If ``initial_instance_count``, ``accelerator_type``, or ``model_name`` is
                  specified, but ``instance_type`` is ``None``.
                - If ``initial_instance_count``, ``instance_type``, or ``accelerator_type`` is
                  specified and either ``model_name`` is ``None`` or there are multiple models
                  associated with the endpoint.
        """
        production_variants = None

        if initial_instance_count or instance_type or accelerator_type or model_name:
            if instance_type is None or initial_instance_count is None:
                raise ValueError(
                    "Missing initial_instance_count and/or instance_type. Provided values: "
                    "initial_instance_count={}, instance_type={}, accelerator_type={}, "
                    "model_name={}.".format(
                        initial_instance_count, instance_type, accelerator_type, model_name
                    )
                )

            if model_name is None:
                if len(self._model_names) > 1:
                    raise ValueError(
                        "Unable to choose a default model for a new EndpointConfig because "
                        "the endpoint has multiple models: {}".format(", ".join(self._model_names))
                    )
                model_name = self._model_names[0]
            else:
                self._model_names = [model_name]

            production_variant_config = production_variant(
                model_name,
                instance_type,
                initial_instance_count=initial_instance_count,
                accelerator_type=accelerator_type,
            )
            production_variants = [production_variant_config]

        new_endpoint_config_name = name_from_base(self._endpoint_config_name)
        self.sagemaker_session.create_endpoint_config_from_existing(
            self._endpoint_config_name,
            new_endpoint_config_name,
            new_tags=tags,
            new_kms_key=kms_key,
            new_data_capture_config_dict=data_capture_config_dict,
            new_production_variants=production_variants,
        )
        self.sagemaker_session.update_endpoint(
            self.endpoint_name, new_endpoint_config_name, wait=wait
        )
        self._endpoint_config_name = new_endpoint_config_name

    def _delete_endpoint_config(self):
        """Delete the Amazon SageMaker endpoint configuration"""
        self.sagemaker_session.delete_endpoint_config(self._endpoint_config_name)

    def delete_endpoint(self, delete_endpoint_config=True):
        """Delete the Amazon SageMaker endpoint backing this predictor. Also
        delete the endpoint configuration attached to it if
        delete_endpoint_config is True.

        Args:
            delete_endpoint_config (bool, optional): Flag to indicate whether to
                delete endpoint configuration together with endpoint. Defaults
                to True. If True, both endpoint and endpoint configuration will
                be deleted. If False, only endpoint will be deleted.
        """
        if delete_endpoint_config:
            self._delete_endpoint_config()

        self.sagemaker_session.delete_endpoint(self.endpoint_name)

    def delete_model(self):
        """Deletes the Amazon SageMaker models backing this predictor."""
        request_failed = False
        failed_models = []
        for model_name in self._model_names:
            try:
                self.sagemaker_session.delete_model(model_name)
            except Exception:  # pylint: disable=broad-except
                request_failed = True
                failed_models.append(model_name)

        if request_failed:
            raise Exception(
                "One or more models cannot be deleted, please retry. \n"
                "Failed models: {}".format(", ".join(failed_models))
            )

    def enable_data_capture(self):
        """Updates the DataCaptureConfig for the Predictor's associated Amazon SageMaker Endpoint
        to enable data capture. For a more customized experience, refer to
        update_data_capture_config, instead.
        """
        self.update_data_capture_config(
            data_capture_config=DataCaptureConfig(
                enable_capture=True, sagemaker_session=self.sagemaker_session
            )
        )

    def disable_data_capture(self):
        """Updates the DataCaptureConfig for the Predictor's associated Amazon SageMaker Endpoint
        to disable data capture. For a more customized experience, refer to
        update_data_capture_config, instead.
        """
        self.update_data_capture_config(
            data_capture_config=DataCaptureConfig(
                enable_capture=False, sagemaker_session=self.sagemaker_session
            )
        )

    def update_data_capture_config(self, data_capture_config):
        """Updates the DataCaptureConfig for the Predictor's associated Amazon SageMaker Endpoint
        with the provided DataCaptureConfig.

        Args:
            data_capture_config (sagemaker.model_monitor.DataCaptureConfig): The
                DataCaptureConfig to update the predictor's endpoint to use.
        """
        endpoint_desc = self.sagemaker_session.sagemaker_client.describe_endpoint(
            EndpointName=self.endpoint_name
        )

        new_config_name = name_from_base(base=self.endpoint_name)

        data_capture_config_dict = None
        if data_capture_config is not None:
            data_capture_config_dict = data_capture_config._to_request_dict()

        self.sagemaker_session.create_endpoint_config_from_existing(
            existing_config_name=endpoint_desc["EndpointConfigName"],
            new_config_name=new_config_name,
            new_data_capture_config_dict=data_capture_config_dict,
        )

        self.sagemaker_session.update_endpoint(
            endpoint_name=self.endpoint_name, endpoint_config_name=new_config_name
        )

    def list_monitors(self):
        """Generates ModelMonitor objects (or DefaultModelMonitors) based on the schedule(s)
        associated with the endpoint that this predictor refers to.

        Returns:
            [sagemaker.model_monitor.model_monitoring.ModelMonitor]: A list of
                ModelMonitor (or DefaultModelMonitor) objects.

        """
        monitoring_schedules_dict = self.sagemaker_session.list_monitoring_schedules(
            endpoint_name=self.endpoint_name
        )
        if len(monitoring_schedules_dict["MonitoringScheduleSummaries"]) == 0:
            print("No monitors found for endpoint. endpoint: {}".format(self.endpoint_name))
            return []

        monitors = []
        for schedule_dict in monitoring_schedules_dict["MonitoringScheduleSummaries"]:
            schedule_name = schedule_dict["MonitoringScheduleName"]
            schedule = self.sagemaker_session.describe_monitoring_schedule(
                monitoring_schedule_name=schedule_name
            )
            image_uri = schedule["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "MonitoringAppSpecification"
            ]["ImageUri"]
            index_after_placeholders = _DEFAULT_MONITOR_IMAGE_URI_WITH_PLACEHOLDERS.rfind("{}")
            if image_uri.endswith(
                _DEFAULT_MONITOR_IMAGE_URI_WITH_PLACEHOLDERS[index_after_placeholders + len("{}") :]
            ):
                monitors.append(
                    DefaultModelMonitor.attach(
                        monitor_schedule_name=schedule_name,
                        sagemaker_session=self.sagemaker_session,
                    )
                )
            else:
                monitors.append(
                    ModelMonitor.attach(
                        monitor_schedule_name=schedule_name,
                        sagemaker_session=self.sagemaker_session,
                    )
                )

        return monitors

    def _get_endpoint_config_name(self):
        """Placeholder docstring"""
        endpoint_desc = self.sagemaker_session.sagemaker_client.describe_endpoint(
            EndpointName=self.endpoint_name
        )
        endpoint_config_name = endpoint_desc["EndpointConfigName"]
        return endpoint_config_name

    def _get_model_names(self):
        """Placeholder docstring"""
        endpoint_config = self.sagemaker_session.sagemaker_client.describe_endpoint_config(
            EndpointConfigName=self._endpoint_config_name
        )
        production_variants = endpoint_config["ProductionVariants"]
        return [d["ModelName"] for d in production_variants]


class LegacySerializer(BaseSerializer):
    """Wrapper that makes legacy serializers forward compatibile."""

    def __init__(self, serializer):
        """Placeholder docstring.

        Args:
            serializer (callable): A legacy serializer.
        """
        self.serializer = serializer
        self.content_type = getattr(serializer, "content_type", None)

    def __call__(self, *args, **kwargs):
        """Wraps the call method of the legacy serializer.

        Args:
            data (object): Data to be serialized.

        Returns:
            object: Serialized data used for a request.
        """
        return self.serializer(*args, **kwargs)

    def serialize(self, data):
        """Wraps the call method of the legacy serializer.

        Args:
            data (object): Data to be serialized.

        Returns:
            object: Serialized data used for a request.
        """
        return self.serializer(data)

    @property
    def CONTENT_TYPE(self):
        """The MIME type of the data sent to the inference endpoint."""
        return self.content_type


class LegacyDeserializer(BaseDeserializer):
    """Wrapper that makes legacy deserializers forward compatibile."""

    def __init__(self, deserializer):
        """Placeholder docstring.

        Args:
            deserializer (callable): A legacy deserializer.
        """
        self.deserializer = deserializer
        self.accept = getattr(deserializer, "accept", None)

    def __call__(self, *args, **kwargs):
        """Wraps the call method of the legacy deserializer.

        Args:
            data (object): Data to be deserialized.
            content_type (str): The MIME type of the data.

        Returns:
            object: The data deserialized into an object.
        """
        return self.deserializer(*args, **kwargs)

    def deserialize(self, data, content_type):
        """Wraps the call method of the legacy deserializer.

        Args:
            data (object): Data to be deserialized.
            content_type (str): The MIME type of the data.

        Returns:
            object: The data deserialized into an object.
        """
        return self.deserializer(data, content_type)

    @property
    def ACCEPT(self):
        """The content type that is expected from the inference endpoint."""
        return self.accept


class _CsvSerializer(object):
    """Placeholder docstring"""

    def __init__(self):
        """Placeholder docstring"""
        self.content_type = CONTENT_TYPE_CSV

    def __call__(self, data):
        """Take data of various data formats and serialize them into CSV.

        Args:
            data (object): Data to be serialized.

        Returns:
            object: Sequence of bytes to be used for the request body.
        """
        # For inputs which represent multiple "rows", the result should be newline-separated CSV
        # rows
        if _is_mutable_sequence_like(data) and len(data) > 0 and _is_sequence_like(data[0]):
            return "\n".join([_CsvSerializer._serialize_row(row) for row in data])
        return _CsvSerializer._serialize_row(data)

    @staticmethod
    def _serialize_row(data):
        # Don't attempt to re-serialize a string
        """
        Args:
            data:
        """
        if isinstance(data, str):
            return data
        if isinstance(data, np.ndarray):
            data = np.ndarray.flatten(data)
        if hasattr(data, "__len__"):
            if len(data) == 0:
                raise ValueError("Cannot serialize empty array")
            return _csv_serialize_python_array(data)

        # files and buffers
        if hasattr(data, "read"):
            return _csv_serialize_from_buffer(data)

        raise ValueError("Unable to handle input format: ", type(data))


def _csv_serialize_python_array(data):
    """
    Args:
        data:
    """
    return _csv_serialize_object(data)


def _csv_serialize_from_buffer(buff):
    """
    Args:
        buff:
    """
    return buff.read()


def _csv_serialize_object(data):
    """
    Args:
        data:
    """
    csv_buffer = StringIO()

    csv_writer = csv.writer(csv_buffer, delimiter=",")
    csv_writer.writerow(data)
    return csv_buffer.getvalue().rstrip("\r\n")


csv_serializer = _CsvSerializer()


def _is_mutable_sequence_like(obj):
    """
    Args:
        obj:
    """
    return _is_sequence_like(obj) and hasattr(obj, "__setitem__")


def _is_sequence_like(obj):
    """
    Args:
        obj:
    """
    return hasattr(obj, "__iter__") and hasattr(obj, "__getitem__")


def _row_to_csv(obj):
    """
    Args:
        obj:
    """
    if isinstance(obj, str):
        return obj
    return ",".join(obj)


class _CsvDeserializer(object):
    """Placeholder docstring"""

    def __init__(self, encoding="utf-8"):
        """
        Args:
            encoding:
        """
        self.accept = CONTENT_TYPE_CSV
        self.encoding = encoding

    def __call__(self, stream, content_type):
        """
        Args:
            stream:
            content_type:
        """
        try:
            return list(csv.reader(stream.read().decode(self.encoding).splitlines()))
        finally:
            stream.close()


csv_deserializer = _CsvDeserializer()


class BytesDeserializer(object):
    """Return the response as an undecoded array of bytes.

    Args:
        accept (str): The Accept header to send to the server (optional).
    """

    def __init__(self, accept=None):
        """
        Args:
            accept:
        """
        self.accept = accept

    def __call__(self, stream, content_type):
        """
        Args:
            stream:
            content_type:
        """
        try:
            return stream.read()
        finally:
            stream.close()


class StringDeserializer(object):
    """Return the response as a decoded string.

    Args:
        encoding (str): The string encoding to use (default=utf-8).
        accept (str): The Accept header to send to the server (optional).
    """

    def __init__(self, encoding="utf-8", accept=None):
        """
        Args:
            encoding:
            accept:
        """
        self.encoding = encoding
        self.accept = accept

    def __call__(self, stream, content_type):
        """
        Args:
            stream:
            content_type:
        """
        try:
            return stream.read().decode(self.encoding)
        finally:
            stream.close()


class StreamDeserializer(object):
    """Returns the tuple of the response stream and the content-type of the response.
       It is the receivers responsibility to close the stream when they're done
       reading the stream.

    Args:
        accept (str): The Accept header to send to the server (optional).
    """

    def __init__(self, accept=None):
        """
        Args:
            accept:
        """
        self.accept = accept

    def __call__(self, stream, content_type):
        """
        Args:
            stream:
            content_type:
        """
        return (stream, content_type)


class _JsonSerializer(object):
    """Placeholder docstring"""

    def __init__(self):
        """Placeholder docstring"""
        self.content_type = CONTENT_TYPE_JSON

    def __call__(self, data):
        """Take data of various formats and serialize them into the expected
        request body. This uses information about supported input formats for
        the deployed model.

        Args:
            data (object): Data to be serialized.

        Returns:
            object: Serialized data used for the request.
        """
        if isinstance(data, dict):
            # convert each value in dict from a numpy array to a list if necessary, so they can be
            # json serialized
            return json.dumps({k: _ndarray_to_list(v) for k, v in six.iteritems(data)})

        # files and buffers
        if hasattr(data, "read"):
            return _json_serialize_from_buffer(data)

        return json.dumps(_ndarray_to_list(data))


json_serializer = _JsonSerializer()


def _ndarray_to_list(data):
    """
    Args:
        data:
    """
    return data.tolist() if isinstance(data, np.ndarray) else data


def _json_serialize_from_buffer(buff):
    """
    Args:
        buff:
    """
    return buff.read()


class _JsonDeserializer(object):
    """Placeholder docstring"""

    def __init__(self):
        """Placeholder docstring"""
        self.accept = CONTENT_TYPE_JSON

    def __call__(self, stream, content_type):
        """Decode a JSON object into the corresponding Python object.

        Args:
            stream (stream): The response stream to be deserialized.
            content_type (str): The content type of the response.

        Returns:
            object: Body of the response deserialized into a JSON object.
        """
        try:
            return json.load(codecs.getreader("utf-8")(stream))
        finally:
            stream.close()


json_deserializer = _JsonDeserializer()


class _NumpyDeserializer(object):
    """Placeholder docstring"""

    def __init__(self, accept=CONTENT_TYPE_NPY, dtype=None):
        """
        Args:
            accept:
            dtype:
        """
        self.accept = accept
        self.dtype = dtype

    def __call__(self, stream, content_type=CONTENT_TYPE_NPY):
        """Decode from serialized data into a Numpy array.

        Args:
            stream (stream): The response stream to be deserialized.
            content_type (str): The content type of the response. Can accept
                CSV, JSON, or NPY data.

        Returns:
            object: Body of the response deserialized into a Numpy array.
        """
        try:
            if content_type == CONTENT_TYPE_CSV:
                return np.genfromtxt(
                    codecs.getreader("utf-8")(stream), delimiter=",", dtype=self.dtype
                )
            if content_type == CONTENT_TYPE_JSON:
                return np.array(json.load(codecs.getreader("utf-8")(stream)), dtype=self.dtype)
            if content_type == CONTENT_TYPE_NPY:
                return np.load(BytesIO(stream.read()))
        finally:
            stream.close()
        raise ValueError(
            "content_type must be one of the following: CSV, JSON, NPY. content_type: {}".format(
                content_type
            )
        )


numpy_deserializer = _NumpyDeserializer()


class _NPYSerializer(object):
    """Placeholder docstring"""

    def __init__(self):
        """Placeholder docstring"""
        self.content_type = CONTENT_TYPE_NPY

    def __call__(self, data, dtype=None):
        """Serialize data into the request body in NPY format.

        Args:
            data (object): Data to be serialized. Can be a numpy array, list,
                file, or buffer.
            dtype:

        Returns:
            object: NPY serialized data used for the request.
        """
        if isinstance(data, np.ndarray):
            if not data.size > 0:
                raise ValueError("empty array can't be serialized")
            return _npy_serialize(data)

        if isinstance(data, list):
            if not len(data) > 0:
                raise ValueError("empty array can't be serialized")
            return _npy_serialize(np.array(data, dtype))

        # files and buffers. Assumed to hold npy-formatted data.
        if hasattr(data, "read"):
            return data.read()

        return _npy_serialize(np.array(data))


def _npy_serialize(data):
    """
    Args:
        data:
    """
    buffer = BytesIO()
    np.save(buffer, data)
    return buffer.getvalue()


npy_serializer = _NPYSerializer()
