from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Float32Tensor(_message.Message):
    __slots__ = ["values", "keys", "shape"]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    keys: _containers.RepeatedScalarFieldContainer[int]
    shape: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, values: _Optional[_Iterable[float]] = ..., keys: _Optional[_Iterable[int]] = ..., shape: _Optional[_Iterable[int]] = ...) -> None: ...

class Float64Tensor(_message.Message):
    __slots__ = ["values", "keys", "shape"]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    keys: _containers.RepeatedScalarFieldContainer[int]
    shape: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, values: _Optional[_Iterable[float]] = ..., keys: _Optional[_Iterable[int]] = ..., shape: _Optional[_Iterable[int]] = ...) -> None: ...

class Int32Tensor(_message.Message):
    __slots__ = ["values", "keys", "shape"]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[int]
    keys: _containers.RepeatedScalarFieldContainer[int]
    shape: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, values: _Optional[_Iterable[int]] = ..., keys: _Optional[_Iterable[int]] = ..., shape: _Optional[_Iterable[int]] = ...) -> None: ...

class Bytes(_message.Message):
    __slots__ = ["value", "content_type"]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    value: _containers.RepeatedScalarFieldContainer[bytes]
    content_type: str
    def __init__(self, value: _Optional[_Iterable[bytes]] = ..., content_type: _Optional[str] = ...) -> None: ...

class Value(_message.Message):
    __slots__ = ["float32_tensor", "float64_tensor", "int32_tensor", "bytes"]
    FLOAT32_TENSOR_FIELD_NUMBER: _ClassVar[int]
    FLOAT64_TENSOR_FIELD_NUMBER: _ClassVar[int]
    INT32_TENSOR_FIELD_NUMBER: _ClassVar[int]
    BYTES_FIELD_NUMBER: _ClassVar[int]
    float32_tensor: Float32Tensor
    float64_tensor: Float64Tensor
    int32_tensor: Int32Tensor
    bytes: Bytes
    def __init__(self, float32_tensor: _Optional[_Union[Float32Tensor, _Mapping]] = ..., float64_tensor: _Optional[_Union[Float64Tensor, _Mapping]] = ..., int32_tensor: _Optional[_Union[Int32Tensor, _Mapping]] = ..., bytes: _Optional[_Union[Bytes, _Mapping]] = ...) -> None: ...

class Record(_message.Message):
    __slots__ = ["features", "label", "uid", "metadata", "configuration"]
    class FeaturesEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Value, _Mapping]] = ...) -> None: ...
    class LabelEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Value, _Mapping]] = ...) -> None: ...
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    UID_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    CONFIGURATION_FIELD_NUMBER: _ClassVar[int]
    features: _containers.MessageMap[str, Value]
    label: _containers.MessageMap[str, Value]
    uid: str
    metadata: str
    configuration: str
    def __init__(self, features: _Optional[_Mapping[str, Value]] = ..., label: _Optional[_Mapping[str, Value]] = ..., uid: _Optional[str] = ..., metadata: _Optional[str] = ..., configuration: _Optional[str] = ...) -> None: ...
