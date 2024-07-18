"""Defines the predictors used in local container mode"""

from __future__ import absolute_import
import io
from typing import Type

from sagemaker import Session
from sagemaker.serve.mode.local_container_mode import LocalContainerMode
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serializers import IdentitySerializer, JSONSerializer
from sagemaker.deserializers import BytesDeserializer, JSONDeserializer
from sagemaker.base_predictor import PredictorBase, Predictor
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.model_server.triton.triton_builder import TritonSerializer

APPLICATION_X_NPY = "application/x-npy"


class TorchServeLocalPredictor(PredictorBase):
    """Lightweight predictor for local deployment in IN_PROCESS and LOCAL_CONTAINER modes"""

    # TODO: change mode_obj to union of IN_PROCESS and LOCAL_CONTAINER objs
    def __init__(
        self,
        mode_obj: Type[LocalContainerMode],
        serializer=IdentitySerializer(),
        deserializer=BytesDeserializer(),
    ):
        self._mode_obj = mode_obj
        self.serializer = serializer
        self.deserializer = deserializer

    def predict(self, data):
        """Placeholder docstring"""
        return self.deserializer.deserialize(
            io.BytesIO(
                self._mode_obj._invoke_torch_serve(
                    self.serializer.serialize(data),
                    self.content_type,
                    self.accept[0],
                )
            )
        )

    @property
    def content_type(self):
        """The MIME type of the data sent to the inference endpoint."""
        return self.serializer.CONTENT_TYPE

    @property
    def accept(self):
        """The content type(s) that are expected from the inference endpoint."""
        return self.deserializer.ACCEPT

    def delete_predictor(self):
        """Shut down and remove the container that you created in LOCAL_CONTAINER mode"""
        self._mode_obj.destroy_server()


class TritonLocalPredictor(PredictorBase):
    """Lightweight predictor for Triton in LOCAL_CONTAINER modes"""

    def __init__(self, mode_obj: Type[LocalContainerMode]) -> None:
        self._mode_obj = mode_obj

    def predict(self, data):
        """Placeholder docstring"""
        return self._mode_obj._invoke_triton_server(data, APPLICATION_X_NPY, APPLICATION_X_NPY)

    @property
    def content_type(self):
        """Triton expects request and response payload to be numpy"""
        return APPLICATION_X_NPY

    @property
    def accept(self):
        """Triton expects request and response payload to be numpy"""
        return APPLICATION_X_NPY

    def delete_predictor(self):
        """Shut down and remove the container that you created in LOCAL_CONTAINER mode"""
        self._mode_obj.destroy_server()


class DjlLocalModePredictor(PredictorBase):
    """Lightweight Djl predictor for local deployment in IN_PROCESS and LOCAL_CONTAINER modes"""

    def __init__(
        self,
        mode_obj: Type[LocalContainerMode],
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    ):
        self._mode_obj = mode_obj
        self.serializer = serializer
        self.deserializer = deserializer

    def predict(self, data):
        """Placeholder docstring"""
        return self.deserializer.deserialize(
            io.BytesIO(
                self._mode_obj._invoke_djl_serving(
                    self.serializer.serialize(data),
                    self.content_type,
                    self.deserializer.ACCEPT[0],
                )
            ),
            self.content_type,
        )

    @property
    def content_type(self):
        """The MIME type of the data sent to the inference endpoint."""
        return self.serializer.CONTENT_TYPE

    @property
    def accept(self):
        """The content type(s) that are expected from the inference endpoint."""
        return self.deserializer.ACCEPT

    def delete_predictor(self):
        """Shut down and remove the container that you created in LOCAL_CONTAINER mode"""
        self._mode_obj.destroy_server()


class TgiLocalModePredictor(PredictorBase):
    """Lightweight Tgi predictor for local deployment in IN_PROCESS and LOCAL_CONTAINER modes"""

    def __init__(
        self,
        mode_obj: Type[LocalContainerMode],
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    ):
        self._mode_obj = mode_obj
        self.serializer = serializer
        self.deserializer = deserializer

    def predict(self, data):
        """Placeholder docstring"""
        return [
            self.deserializer.deserialize(
                io.BytesIO(
                    self._mode_obj._invoke_tgi_serving(
                        self.serializer.serialize(data),
                        self.content_type,
                        self.deserializer.ACCEPT[0],
                    )
                ),
                self.content_type,
            )
        ]

    @property
    def content_type(self):
        """The MIME type of the data sent to the inference endpoint."""
        return self.serializer.CONTENT_TYPE

    @property
    def accept(self):
        """The content type(s) that are expected from the inference endpoint."""
        return self.deserializer.ACCEPT

    def delete_predictor(self):
        """Shut down and remove the container that you created in LOCAL_CONTAINER mode"""
        self._mode_obj.destroy_server()


class TransformersLocalModePredictor(PredictorBase):
    """Lightweight Transformers predictor for local deployment"""

    def __init__(
        self,
        mode_obj: Type[LocalContainerMode],
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    ):
        self._mode_obj = mode_obj
        self.serializer = serializer
        self.deserializer = deserializer

    def predict(self, data):
        """Placeholder docstring"""
        return [
            self.deserializer.deserialize(
                io.BytesIO(
                    self._mode_obj._invoke_multi_model_server_serving(
                        self.serializer.serialize(data),
                        self.content_type,
                        self.deserializer.ACCEPT[0],
                    )
                ),
                self.content_type,
            )
        ]

    @property
    def content_type(self):
        """The MIME type of the data sent to the inference endpoint."""
        return self.serializer.CONTENT_TYPE

    @property
    def accept(self):
        """The content type(s) that are expected from the inference endpoint."""
        return self.deserializer.ACCEPT

    def delete_predictor(self):
        """Shut down and remove the container that you created in LOCAL_CONTAINER mode"""
        self._mode_obj.destroy_server()


class TeiLocalModePredictor(PredictorBase):
    """Lightweight Tei predictor for local deployment in IN_PROCESS and LOCAL_CONTAINER modes"""

    def __init__(
        self,
        mode_obj: Type[LocalContainerMode],
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    ):
        self._mode_obj = mode_obj
        self.serializer = serializer
        self.deserializer = deserializer

    def predict(self, data):
        """Placeholder docstring"""
        return [
            self.deserializer.deserialize(
                io.BytesIO(
                    self._mode_obj._invoke_serving(
                        self.serializer.serialize(data),
                        self.content_type,
                        self.deserializer.ACCEPT[0],
                    )
                ),
                self.content_type,
            )
        ]

    @property
    def content_type(self):
        """The MIME type of the data sent to the inference endpoint."""
        return self.serializer.CONTENT_TYPE

    @property
    def accept(self):
        """The content type(s) that are expected from the inference endpoint."""
        return self.deserializer.ACCEPT

    def delete_predictor(self):
        """Shut down and remove the container that you created in LOCAL_CONTAINER mode"""
        self._mode_obj.destroy_server()


class TensorflowServingLocalPredictor(PredictorBase):
    """Lightweight predictor for local deployment in LOCAL_CONTAINER modes"""

    # TODO: change mode_obj to union of IN_PROCESS and LOCAL_CONTAINER objs
    def __init__(
        self,
        mode_obj: Type[LocalContainerMode],
        serializer=IdentitySerializer(),
        deserializer=BytesDeserializer(),
    ):
        self._mode_obj = mode_obj
        self.serializer = serializer
        self.deserializer = deserializer

    def predict(self, data):
        """Placeholder docstring"""
        return self.deserializer.deserialize(
            io.BytesIO(
                self._mode_obj._invoke_tensorflow_serving(
                    self.serializer.serialize(data),
                    self.content_type,
                    self.accept[0],
                )
            )
        )

    @property
    def content_type(self):
        """The MIME type of the data sent to the inference endpoint."""
        return self.serializer.CONTENT_TYPE

    @property
    def accept(self):
        """The content type(s) that are expected from the inference endpoint."""
        return self.deserializer.ACCEPT

    def delete_predictor(self):
        """Shut down and remove the container that you created in LOCAL_CONTAINER mode"""
        self._mode_obj.destroy_server()


def _get_local_mode_predictor(
    model_server: ModelServer,
    mode_obj: Type[LocalContainerMode],
    serializer=IdentitySerializer(),
    deserializer=BytesDeserializer(),
) -> Type[PredictorBase]:
    """Placeholder docstring"""
    if model_server == ModelServer.TORCHSERVE:
        return TorchServeLocalPredictor(
            mode_obj=mode_obj, serializer=serializer, deserializer=deserializer
        )
    if model_server == ModelServer.TRITON:
        return TritonLocalPredictor(mode_obj=mode_obj)

    if model_server == ModelServer.TENSORFLOW_SERVING:
        return TensorflowServingLocalPredictor(
            mode_obj=mode_obj, serializer=serializer, deserializer=deserializer
        )

    raise ValueError("%s model server is not supported yet!" % model_server)


def retrieve_predictor(
    endpoint_name: str,
    schema_builder: SchemaBuilder,
    sagemaker_session: Session,
    model_server: ModelServer = ModelServer.TORCHSERVE,
) -> Predictor:
    """Retrieve the predictor for existing sagemaker endpoint"""
    # TODO: extend to LOCAL_CONTAINER mode
    if model_server == ModelServer.TRITON:
        schema_builder._update_serializer_deserializer_for_triton()
        schema_builder._detect_dtype_for_triton()
        dtype = schema_builder._input_triton_dtype.split("_")[-1]
        serializer = TritonSerializer(input_serializer=schema_builder.input_serializer, dtype=dtype)
        deserializer = JSONDeserializer()

    elif model_server == ModelServer.TORCHSERVE:
        serializer = (
            schema_builder.custom_input_translator
            if hasattr(schema_builder, "custom_input_translator")
            else schema_builder.input_serializer
        )
        deserializer = (
            schema_builder.custom_output_translator
            if hasattr(schema_builder, "custom_output_translator")
            else schema_builder.output_deserializer
        )

    else:
        raise ValueError("Unsupproted model_server: %s" % model_server)

    return Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sagemaker_session,
        serializer=serializer,
        deserializer=deserializer,
    )
