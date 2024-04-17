"""Placeholder docstring"""

from __future__ import absolute_import

from sagemaker.serve.marshalling.triton_translator import (
    TorchTensorTranslator,
    TensorflowTensorTranslator,
    NumpyTranslator,
    ListTranslator,
)

# class names supported by triton
TORCH_TENSOR = "torch"
TF_TENSOR = "tensorflow"
NUMPY_ARRAY = "ndarray"
PYTHON_LIST = "list"
SUPPORTED_TYPES = set([TORCH_TENSOR, TF_TENSOR, NUMPY_ARRAY])

CLASS_TO_TRANSLATOR_MAP = {
    TORCH_TENSOR: TorchTensorTranslator,
    TF_TENSOR: TensorflowTensorTranslator,
    NUMPY_ARRAY: NumpyTranslator,
    PYTHON_LIST: ListTranslator,
}

# https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype
PYTORCH_TENSOR_TO_TRITON_DTYPE_MAP = {
    "torch.float16": "TYPE_FP16",
    "torch.half": "TYPE_FP16",
    "torch.bfloat16": "TYPE_BF16",
    "torch.float32": "TYPE_FP32",
    "torch.float": "TYPE_FP32",
    "torch.float64": "TYPE_FP64",
    "torch.double": "TYPE_FP64",
    "torch.uint8": "TYPE_UINT8",
    "torch.int8": "TYPE_INT8",
    "torch.int16": "TYPE_INT16",
    "torch.short": "TYPE_INT16",
    "torch.int32": "TYPE_INT32",
    "torch.int": "TYPE_INT32",
    "torch.int64": "TYPE_INT64",
    "torch.long": "TYPE_INT64",
    "torch.bool": "TYPE_BOOL",
}

# https://www.tensorflow.org/api_docs/python/tf/dtypes
TENSORFLOW_TO_TRITON_DTYPE_MAP = {
    "float16": "TYPE_FP16",
    "half": "TYPE_FP16",
    "bfloat16": "TYPE_BF16",
    "float32": "TYPE_FP32",
    "float64": "TYPE_FP64",
    "double": "TYPE_FP64",
    "uint8": "TYPE_UINT8",
    "int8": "TYPE_INT8",
    "int16": "TYPE_INT16",
    "int32": "TYPE_INT32",
    "int": "TYPE_INT32",
    "int64": "TYPE_INT64",
    "bool": "TYPE_BOOL",
}


NUMPY_ARRAY_TRITON_DTYPE_MAP = {
    "bool": "TYPE_BOOL",
    "uint8": "TYPE_UINT8",
    "uint16": "TYPE_UINT16",
    "uint32": "TYPE_UINT32",
    "uint64": "TYPE_UINT64",
    "int8": "TYPE_INT8",
    "int16": "TYPE_INT16",
    "int32": "TYPE_INT32",
    "int64": "TYPE_INT64",
    "float16": "TYPE_FP16",
    "float32": "TYPE_FP32",
    "float64": "TYPE_FP64",
    "object_": "TYPE_STRING",
}

DEFAULT_DTYPE = "TYPE_FP32"


class TritonSchemaBuilder:
    """Mixin class for SchemaBuilder that holds Triton specific methods"""

    # pylint: disable=no-member, attribute-defined-outside-init

    def __init__(self) -> None:
        self._input_class_name = None
        self._output_class_name = None

        self._input_triton_dtype = None
        self._output_triton_dtype = None

        self._sample_input_ndarray = None
        self._sample_output_ndarray = None

    def _update_serializer_deserializer_for_triton(self) -> None:
        """Update serializer and deserializer method for triton

        Update input_serializer, input_deserializer, output_serializer
        and output_deserializer to use Triton specific converter.
        This method is only meant to be called during ModelBuilder().build() for Triton.
        """
        # Update for input
        self._detect_class_of_sample_input_and_output()
        self.input_serializer = CLASS_TO_TRANSLATOR_MAP.get(self._input_class_name)()
        self.input_deserializer = CLASS_TO_TRANSLATOR_MAP.get(self._input_class_name)()
        self.output_serializer = CLASS_TO_TRANSLATOR_MAP.get(self._output_class_name)()
        self.output_deserializer = CLASS_TO_TRANSLATOR_MAP.get(self._output_class_name)()

        # Validate translation
        try:
            self._sample_input_ndarray = self.input_serializer.serialize(self.sample_input)
            self._sample_output_ndarray = self.output_serializer.serialize(self.sample_output)
            self.input_deserializer.deserialize(self._sample_input_ndarray)
            self.output_deserializer.deserialize(self._sample_output_ndarray)

        except Exception as e:
            raise ValueError(
                (
                    "Validation of serialization and deserialization failed: %s,"
                    "please verify your sample_input and sample_output."
                )
                % e
            )

    def _detect_class_of_sample_input_and_output(self):
        """Detect the class of sample_input and sample_output"""
        input_class_name = str(self.sample_input.__class__)
        for supported_type in SUPPORTED_TYPES:
            if supported_type in input_class_name:
                self._input_class_name = supported_type
                break

        if not self._input_class_name:
            raise ValueError(
                (
                    "Unable to update input serializer and deserializer for type %s for Triton. "
                    "Please provide sample_input of the following type: %s to SchemaBuilder."
                )
                % (type(self.sample_input), SUPPORTED_TYPES)
            )

        # Update for Output
        output_class_name = str(self.sample_output.__class__)
        for supported_type in SUPPORTED_TYPES:
            if supported_type in output_class_name:
                self._output_class_name = supported_type
                break

        if not self._output_class_name:
            raise ValueError(
                (
                    "Unable to update output serializer and deserializer for type %s for Triton. "
                    "Please provide sample_output of the following type: %s to SchemaBuilder."
                )
                % (type(self.sample_output), SUPPORTED_TYPES)
            )

    def _detect_dtype_for_triton(self):
        """Map sample_input and sample_output data type to Triton data type"""
        # detect for input
        if self._input_class_name == TORCH_TENSOR:
            self._input_triton_dtype = self._detect_dtype_for_pytorch_tensor(data=self.sample_input)
        elif self._input_class_name == NUMPY_ARRAY:
            self._input_triton_dtype = self._detect_dtype_for_numpy(data=self.sample_input)
        elif self._input_class_name == TF_TENSOR:
            self._input_triton_dtype = self._detect_dtype_for_tensorflow(data=self.sample_input)
        else:
            self._input_triton_dtype = DEFAULT_DTYPE

        # detect for output
        if self._output_class_name == TORCH_TENSOR:
            self._output_triton_dtype = self._detect_dtype_for_pytorch_tensor(
                data=self.sample_output
            )
        elif self._output_class_name == NUMPY_ARRAY:
            self._output_triton_dtype = self._detect_dtype_for_numpy(data=self.sample_output)
        elif self._output_class_name == TF_TENSOR:
            self._output_triton_dtype = self._detect_dtype_for_tensorflow(data=self.sample_output)
        else:
            self._output_triton_dtype = DEFAULT_DTYPE

    def _detect_dtype_for_pytorch_tensor(self, data):
        """Placeholder docstring"""
        return PYTORCH_TENSOR_TO_TRITON_DTYPE_MAP.get(str(data.dtype), DEFAULT_DTYPE)

    def _detect_dtype_for_numpy(self, data):
        """Placeholder docstring"""
        return NUMPY_ARRAY_TRITON_DTYPE_MAP.get(data.dtype.name, DEFAULT_DTYPE)

    def _detect_dtype_for_tensorflow(self, data):
        """Placeholder docstring"""
        return TENSORFLOW_TO_TRITON_DTYPE_MAP.get(data.dtype.name, DEFAULT_DTYPE)
