"""Placeholder docstring"""

from __future__ import absolute_import
import io
import logging
from pathlib import Path
import numpy as np
from pandas import DataFrame

from sagemaker.deserializers import (
    BaseDeserializer,
    BytesDeserializer,
    NumpyDeserializer,
    JSONDeserializer,
    PandasDeserializer,
    TorchTensorDeserializer,
    StringDeserializer,
)
from sagemaker.serializers import (
    DataSerializer,
    NumpySerializer,
    JSONSerializer,
    CSVSerializer,
    TorchTensorSerializer,
    StringSerializer,
)

from sagemaker.serve.marshalling.custom_payload_translator import CustomPayloadTranslator

from sagemaker.serve.builder.triton_schema_builder import TritonSchemaBuilder

logger = logging.getLogger(__name__)


class JSONSerializerWrapper(JSONSerializer):
    """Wraps the JSONSerializer because it does not convert jsonable to bytes"""

    def serialize(self, data) -> bytes:
        """Placeholder docstring"""

        return super().serialize(data).encode("utf-8")


class CSVSerializerWrapper(CSVSerializer):
    """Wraps the CSVSerializer because it does not convert dataframe to bytes"""

    def serialize(self, data) -> bytes:
        """Placeholder docstring"""
        return super().serialize(data).encode("utf-8")


translation_mapping = {
    NumpySerializer: NumpyDeserializer,
    NumpyDeserializer: NumpySerializer,
    JSONSerializerWrapper: JSONDeserializer,
    JSONDeserializer: JSONSerializerWrapper,
    TorchTensorSerializer: TorchTensorDeserializer,
    TorchTensorDeserializer: TorchTensorSerializer,
    DataSerializer: BytesDeserializer,
    BytesDeserializer: DataSerializer,
    CSVSerializerWrapper: PandasDeserializer,
    PandasDeserializer: CSVSerializerWrapper,
    StringSerializer: StringDeserializer,
    StringDeserializer: StringSerializer,
}


class DeserializerWrapper(BaseDeserializer):
    """Wraps the deserializer to comply with the function signature."""

    def __init__(self, deserializer, accept):
        self._deserializer = deserializer
        self._accept = accept

    def deserialize(self, stream, content_type: str = None):
        """Deserialize stream into object"""
        return self._deserializer.deserialize(
            stream,
            # We need to overwrite the accept type because model
            # servers like XGBOOST always returns "text/html"
            self._accept[0],
        )

    @property
    def ACCEPT(self):
        """Placeholder docstring"""
        return self._accept[0]


class SchemaBuilder(TritonSchemaBuilder):
    """Automatically detects the serializer and deserializer for your model.

    This is done by inspecting the `sample_input` and `sample_output` object.
    Alternatively, provide your custom serializer and deserializer
    for your request or response by creating a class that inherits
    ``CustomPayloadTranslator`` and provide it to ``SchemaBuilder``.

    Args:
       sample_input (object): Sample input to the model which can be used
           for testing. The schema builder internally generates the content
           type and corresponding serializing functions.
       sample_output (object): Sample output to the model which can be
           used for testing. The schema builder internally generates
           the accept type and corresponding serializing functions.
       input_translator (Optional[CustomPayloadTranslator]): If you
           want to define your own serialization method for the payload,
           you can implement your functions for translation.
       output_translator (Optional[CustomPayloadTranslator]): If
           you want to define your own serialization method for the output,
           you can implement your functions for translation.
    """

    def __init__(
        self,
        sample_input,
        sample_output,
        input_translator: CustomPayloadTranslator = None,
        output_translator: CustomPayloadTranslator = None,
    ):
        super().__init__()

        self.sample_input = sample_input
        self.sample_output = sample_output
        if input_translator:
            _validate_translations(
                payload=sample_input,
                serialize_callable=input_translator.serialize,
                deserialize_callable=input_translator.deserialize,
            )
            self.custom_input_translator = input_translator
        else:
            self.input_serializer = self._get_serializer(sample_input)
            self._input_deserializer = self._get_inverse(self.input_serializer)
            self.input_deserializer = DeserializerWrapper(
                self._input_deserializer, self._input_deserializer.ACCEPT
            )

        if output_translator:
            _validate_translations(
                payload=sample_output,
                serialize_callable=output_translator.serialize,
                deserialize_callable=output_translator.deserialize,
            )
            self.custom_output_translator = output_translator
        else:
            self._output_deserializer = self._get_deserializer(sample_output)
            self.output_serializer = self._get_inverse(self._output_deserializer)

            self.output_deserializer = DeserializerWrapper(
                self._output_deserializer, self._output_deserializer.ACCEPT
            )

    def _get_serializer(self, obj):
        # pylint: disable=too-many-return-statements
        """Placeholder docstring"""
        if isinstance(obj, np.ndarray):
            return NumpySerializer()
        if isinstance(obj, DataFrame):
            return CSVSerializerWrapper()
        if isinstance(obj, bytes) or _is_path_to_file(obj):
            return DataSerializer()
        if _is_torch_tensor(obj):
            return TorchTensorSerializer()
        if isinstance(obj, str):
            return StringSerializer()
        if _is_jsonable(obj):
            return JSONSerializerWrapper()
        if isinstance(obj, dict) and "content_type" in obj:
            try:
                return DataSerializer(content_type=obj["content_type"])
            except ValueError as e:
                logger.error(e)

        raise ValueError(
            (
                "SchemaBuilder cannot determine the serializer of type %s "
                "Please provide your own marshalling code"
                "to SchemaBuilder via CustomPayloadTranslator"
            )
            % type(obj)
        )

    def _get_deserializer(self, obj):
        # pylint: disable=too-many-return-statements
        """Placeholder docstring"""
        if isinstance(obj, np.ndarray):
            return NumpyDeserializer()
        if isinstance(obj, DataFrame):
            return PandasDeserializer()
        if isinstance(obj, bytes):
            return BytesDeserializer()
        if _is_torch_tensor(obj):
            return TorchTensorDeserializer()
        if isinstance(obj, str):
            return StringDeserializer()
        if _is_jsonable(obj):
            return JSONDeserializer()

        raise ValueError(
            (
                "SchemaBuilder cannot determine deserializer of type %s "
                "Please provide your own marshalling code"
                "to SchemaBuilder via CustomPayloadTranslator"
            )
            % type(obj)
        )

    def _get_inverse(self, obj):
        """Placeholder docstring"""
        try:
            return translation_mapping.get(obj.__class__)()
        except KeyError:
            raise Exception("Unable to serialize")

    def __repr__(self):
        """Placeholder docstring"""
        if hasattr(self, "input_serializer") and hasattr(self, "output_serializer"):
            return (
                f"SchemaBuilder(\n"
                f"input_serializer={self.input_serializer}\n"
                f"output_serializer={self.output_serializer}\n"
                f"input_deserializer={self.input_deserializer._deserializer}\n"
                f"output_deserializer={self.output_deserializer._deserializer})"
            )
        return (
            f"SchemaBuilder(\n"
            f"custom_input_translator={self.custom_input_translator}\n"
            f"custom_output_translator={self.custom_output_translator}\n"
        )

    def generate_marshalling_map(self) -> dict:
        """Generate marshalling map for the schema builder"""
        return {
            "input_serializer": (
                self.input_serializer.__class__.__name__
                if hasattr(self, "input_serializer")
                else None
            ),
            "output_serializer": (
                self.output_serializer.__class__.__name__
                if hasattr(self, "output_serializer")
                else None
            ),
            "input_deserializer": (
                self._input_deserializer.__class__.__name__
                if hasattr(self, "_input_deserializer")
                else None
            ),
            "output_deserializer": (
                self._output_deserializer.__class__.__name__
                if hasattr(self, "_output_deserializer")
                else None
            ),
            "custom_input_translator": hasattr(self, "custom_input_translator"),
            "custom_output_translator": hasattr(self, "custom_output_translator"),
        }

    def get_input_sample(self) -> object:
        """Get input sample for the schema builder"""
        return self.sample_input


def _is_torch_tensor(data: object) -> bool:
    """Placeholder docstring"""
    try:
        from torch import Tensor

        return isinstance(data, Tensor)
    except ModuleNotFoundError:
        return False


def _is_jsonable(data: object) -> bool:
    # pylint: disable=broad-except
    """Placeholder docstring"""
    try:
        JSONSerializerWrapper().serialize(data)
        return True
    except Exception:
        return False


def _is_path_to_file(data: object) -> bool:
    """Placeholder docstring"""
    return isinstance(data, str) and Path(data).resolve().is_file()


def _validate_translations(
    payload: object, serialize_callable: callable, deserialize_callable: callable
) -> None:
    """Placeholder docstring"""
    try:
        b = serialize_callable(payload=payload, content_type="application/custom")
        stream = io.BytesIO(b)
        deserialize_callable(stream=stream, content_type="application/custom")
    except Exception as e:
        raise ValueError("Error when validating payload serialization and deserialization.", e)
