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
"""Configuration and result types for the training_session client interface."""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_serializer,
    model_validator,
)

logger = logging.getLogger(__name__)

# Default LoRA rank for the SDK (the service's native default is 8).
DEFAULT_LORA_RANK = 32
DEFAULT_LORA_ALPHA = 64


class _ForbidExtraBase(BaseModel):
    """Internal shared base for customer-authored types: unknown fields are rejected.

    Rejecting extras catches construction typos (e.g. ``max_token`` for
    ``max_tokens``) at object-creation time instead of silently dropping them.
    Not exported: an implementation detail, not API surface.
    """

    model_config = ConfigDict(extra="forbid")


class _AllowExtraBase(BaseModel):
    """Internal shared base for permissive types: unknown fields are kept.

    Used by the types constructed from external objects, where fields we
    don't model must not be an error. Not exported: an implementation detail,
    not API surface.
    """

    model_config = ConfigDict(extra="allow")


def _params_from_attrs(cls: Any, value: Any) -> Any:
    """Build an all-optional params type from an arbitrary object's attributes.

    The types this serves (:class:`AdamParams`, :class:`SamplingParams`) have a
    default for every field, so there is no required attribute to read and fail
    on the way :meth:`VpcConfig.from_any` does. Reading each field with a default
    fallback on its own would therefore accept *any* object at all and quietly
    substitute the defaults, so an object using different names (``maxTokens``,
    ``temp``) would look like it had been honoured while none of its values were
    used.

    Sharing at least one field name is what distinguishes a params-like object
    from an unrelated one. Fields the object omits still fall back to the
    defaults, which is what lets a caller pass a type modelling fewer knobs.
    """
    present = [field for field in cls.model_fields if hasattr(value, field)]
    if not present:
        raise TypeError(
            f"cannot read {cls.__name__} fields from {type(value).__name__}: none of "
            f"{sorted(cls.model_fields)} are present. Pass a {cls.__name__}, a dict, "
            f"or an object using these field names."
        )
    return cls(**_drop_unset({field: getattr(value, field) for field in present}))


def _drop_unset(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Drop ``None`` entries so the field's own default applies instead.

    A params type that models a knob as optional carries ``None`` for one the
    caller never set. Forwarding that ``None`` would fail validation on a field
    of ours that is required with a default; ``max_tokens`` is the one that
    bites, since the service requires it. Treating "not set" as "use the
    default" is the same outcome the caller would get by omitting the field.
    """
    return {key: val for key, val in fields.items() if val is not None}


class SessionStatus(str, Enum):
    """Status of a training session.

    Use with :meth:`ServiceClient.list_sessions` to filter by status without
    memorising the service's raw PascalCase strings.
    """

    CREATED = "Created"
    SCHEDULED = "Scheduled"
    RUNNING = "Running"
    STOPPING = "Stopping"
    STOPPED = "Stopped"
    FAILED = "Failed"
    BLOCKED = "Blocked"
    EXPIRED = "Expired"
    DELETING = "Deleting"
    DELETION_FAILED = "DeletionFailed"


class ModelPackageKind(str, Enum):
    """Kind of model package a save operation registers.

    A session registers one of two kinds of model package:

    * ``CHECKPOINT``: a ``save_state`` checkpoint: sharded LoRA weights plus
      optimizer/FSDP state. Use it to resume an interrupted training run via
      :meth:`TrainingClient.load_state_with_optimizer` (weights + optimizer) or
      :meth:`TrainingClient.load_state` (weights only, optimizer resets).
    * ``MODEL``: a ``save_weights_for_sampler`` model: the full (unsharded) LoRA
      weights. This is the deployable artifact (no optimizer state), so use it
      for downstream tasks like inference/sampling or to start a new training
      run.
    """

    CHECKPOINT = "checkpoint"
    MODEL = "model"


class ModelPackageInfo(_ForbidExtraBase):
    """Summary of one model package version in the session's group.

    A lightweight view (ARN, version, creation time, kind), distinct from the
    full ``sagemaker.core.resources.ModelPackage`` resource. Returned by
    :meth:`ServiceClient.list_model_packages` and
    :meth:`ServiceClient.get_latest_model_package`. Pass :attr:`model_package_arn`
    straight to :meth:`TrainingClient.load_state` /
    :meth:`TrainingClient.load_state_with_optimizer` (resume training), or to
    :meth:`ServiceClient.create_sampling_client` (inference).

    Args:
        model_package_arn: ARN of the model package version.
        version: Model package version number (monotonic within the group).
        creation_time: When the model package was registered.
        description: The model package description set when it was saved
            (a customer-provided string, or a platform default).
        kind: Whether this is a ``CHECKPOINT`` (save_state) or ``MODEL``
            (save_weights_for_sampler) package. ``None`` when not resolved: the field
            is only available from ``DescribeModelPackage``, so
            :meth:`ServiceClient.list_model_packages` leaves it unset (it lists
            without describing); the ``get_latest_*`` helpers always populate it.
    """

    model_package_arn: str
    version: Optional[int] = None
    creation_time: Optional[datetime] = None
    description: Optional[str] = None
    kind: Optional[ModelPackageKind] = None

    @classmethod
    def _from_summary(cls, summary: Dict[str, Any]) -> "ModelPackageInfo":
        """Build from a ``ListModelPackages`` summary (no ``kind``, which the summary omits)."""
        return cls(
            model_package_arn=summary["ModelPackageArn"],
            version=summary.get("ModelPackageVersion"),
            creation_time=summary.get("CreationTime"),
            description=summary.get("ModelPackageDescription"),
        )


class TrainingSessionSummary(_ForbidExtraBase):
    """Summary of one training session, as listed.

    Returned by :meth:`ServiceClient.list_sessions`. Carries the fields the list
    reports; anything else about a session comes from describing it.

    Args:
        training_session_arn: ARN of the session.
        status: Current session status.
        creation_time: When the session was created.
        last_modified_time: When the session last changed state.
    """

    training_session_arn: str
    #: ``str`` is part of the type so a status this version does not know about
    #: still reaches the caller rather than failing the whole list call.
    status: Optional[Union[SessionStatus, str]] = None
    creation_time: Optional[datetime] = None
    last_modified_time: Optional[datetime] = None

    @property
    def training_session_name(self) -> str:
        """The session name, taken from the tail of :attr:`training_session_arn`."""
        return self.training_session_arn.rsplit("/", 1)[-1]

    @classmethod
    def _from_summary(cls, summary: Dict[str, Any]) -> "TrainingSessionSummary":
        """Build from a ``ListTrainingSessions`` summary.

        An unrecognised status is kept as its raw string rather than dropped, so
        a status this version does not know about still reaches the caller.
        """
        raw_status = summary.get("TrainingSessionStatus")
        try:
            status = SessionStatus(raw_status) if raw_status is not None else None
        except ValueError:
            status = raw_status
        return cls(
            training_session_arn=summary["TrainingSessionArn"],
            status=status,
            creation_time=summary.get("CreationTime"),
            last_modified_time=summary.get("LastModifiedTime"),
        )


class ModelPackageDeletion(_ForbidExtraBase):
    """Outcome of :meth:`ServiceClient.delete_model_package`.

    Args:
        model_package_arn: ARN of the model package that was deleted.
        model_package_deleted: True when this call deleted the package. False
            means it was already gone, so there was nothing left to delete.
        s3_uri: The S3 location the model package pointed at, or ``None`` when
            the package exposes none. When the artifacts were kept, this is the
            only remaining record of where they are.
        objects_deleted: Number of S3 objects deleted under :attr:`s3_uri`. Zero
            when the artifacts were kept, or when the location was already
            empty. On a versioned bucket a delete adds a marker instead of
            reclaiming storage, so this counts objects, not bytes freed.
        artifacts_retained: True when this call did not attempt to delete the
            artifacts, either because ``delete_artifacts`` was False or
            because they are not in a location you manage. False with
            :attr:`objects_deleted` at zero means the location was already
            empty.
        retained_reason: Human-readable reason the artifacts were kept, or
            ``None`` when they were deleted. Diagnostic text for a log; read
            :attr:`artifacts_retained` to branch on.
    """

    model_package_arn: str
    model_package_deleted: bool
    s3_uri: Optional[str] = None
    objects_deleted: int = 0
    artifacts_retained: bool = True
    retained_reason: Optional[str] = None


class VpcConfig(_ForbidExtraBase):
    """VPC configuration for a training session.

    When set, the session's compute runs inside your VPC: traffic to your
    Amazon S3 output location and to other resources flows through the
    subnets and security groups you specify, so make sure they allow the
    required access (for example an S3 gateway endpoint).

    Args:
        subnets: IDs of the subnets to run the session in.
        security_group_ids: IDs of the security groups to apply.
    """

    subnets: List[str]
    security_group_ids: List[str]

    @classmethod
    def from_any(cls, value: Any) -> "VpcConfig":
        """Coerce a VPC-config-like value to VpcConfig.

        Accepts a :class:`VpcConfig` (passthrough), a dict of fields, or any
        object with ``subnets`` / ``security_group_ids`` attributes.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        return cls(
            subnets=list(getattr(value, "subnets")),
            security_group_ids=list(getattr(value, "security_group_ids")),
        )

    def _to_api(self) -> Dict[str, Any]:
        """Convert to the CreateTrainingSession VpcConfig shape."""
        return {
            "Subnets": list(self.subnets),
            "SecurityGroupIds": list(self.security_group_ids),
        }


class TensorDtype(str, Enum):
    """Data type for tensor values in loss function inputs.

    Maps to the dtypes the runtime accepts in ``TensorData.dtype``.
    """

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT32 = "int32"
    INT64 = "int64"


#: Dtype spellings that name a wire dtype the enum does not answer to.
_DTYPE_ALIASES = {"half": TensorDtype.FLOAT16}

#: Float dtypes with no wire slot, sent as ``float32``. Narrowing, so it warns.
#: Any name starting with ``float`` also lands here, which covers the widths
#: that keep being added (``float8_e4m3fn``, ``float128``).
_NARROWED_TO_FLOAT32 = frozenset({"double", "float", "longdouble"})

#: Integer and boolean dtypes with no wire slot, sent as ``int64``. Widening or
#: exact, so it is silent. Booleans land here rather than on a float dtype:
#: a mask reaches the loss function as the 0/1 integers it is.
_WIDENED_TO_INT64 = frozenset(
    {
        "bool",
        "int8",
        "int16",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int",
        "long",
        "short",
    }
)


def _dtype_name(dtype: Any) -> str:
    """Reduce a dtype spec to a bare type name.

    Covers all three spellings: ``float32`` (numpy dtype or wire string),
    ``torch.float32``, and ``<class 'numpy.float32'>`` (a scalar *type*, not a
    dtype instance). The last is unwrapped before the name is taken, so the
    result is always a bare name the dtype tables can match.
    """
    name = str(dtype)
    if name.startswith("<class '") and name.endswith("'>"):
        name = name[len("<class '") : -len("'>")]
    return name.rsplit(".", 1)[-1]


def _dtype_kind(dtype: Any, name: str) -> Optional[str]:
    """Classify a dtype as ``"f"`` (float), ``"i"`` (integral), or ``None``.

    Asks the dtype object itself before falling back to its spelling, so a
    width the name tables never heard of still classifies correctly: torch
    keeps adding float types (``float8_e4m3fn``), which a fixed name list
    would reject.
    """
    # torch: torch.dtype carries these as plain bools.
    is_float = getattr(dtype, "is_floating_point", None)
    if isinstance(is_float, bool):
        if is_float:
            return "f"
        # Complex is neither, and must not fall through to the integer branch:
        # torch reports is_floating_point False for it.
        return None if getattr(dtype, "is_complex", False) else "i"
    # numpy: dtype.kind is a one-character code.
    kind = getattr(dtype, "kind", None)
    if isinstance(kind, str) and kind:
        if kind == "f":
            return "f"
        return "i" if kind in ("i", "u", "b") else None
    # A bare spelling, e.g. from a wire dict's "dtype" field.
    if name in _NARROWED_TO_FLOAT32 or name.startswith("float"):
        return "f"
    return "i" if name in _WIDENED_TO_INT64 else None


def _flatten_nested(values: List[Any]) -> "tuple[List[Any], List[int]]":
    """Fully flatten a rectangular nested list, returning the values and shape.

    Serves the tensor-materialization path, where the rank is whatever the
    source tensor had. Raw ``loss_fn_inputs`` lists go through
    :func:`_flatten_with_shape` instead, which caps the rank and names the
    offending key.
    """
    shape: List[int] = []
    probe: Any = values
    while isinstance(probe, list):
        shape.append(len(probe))
        probe = probe[0] if probe else None
    flat: List[Any] = []

    def walk(node: Any, depth: int) -> None:
        if isinstance(node, list):
            if depth >= len(shape) or len(node) != shape[depth]:
                raise ValueError(
                    f"tensor data is ragged: expected {shape} but a row at depth "
                    f"{depth} has {len(node)} entries; rows must all be the same length"
                )
            for child in node:
                walk(child, depth + 1)
        elif depth != len(shape):
            raise ValueError(
                f"tensor data mixes nesting depths: expected {shape} but found a "
                f"value {len(shape) - depth} level(s) above the deepest row"
            )
        else:
            flat.append(node)

    walk(values, 0)
    return flat, shape


class TensorData(_ForbidExtraBase):
    """A named tensor with dtype and shape metadata.

    Use in ``loss_fn_inputs`` to pass typed tensor data (advantages, weights,
    logprobs, etc.) alongside training datums.

    Tensor-like values (torch tensors, numpy arrays, anything with
    ``tolist()``) are held as-is and only flattened to a plain list when the
    request is serialized or :attr:`data` is first read, so building datums
    from large tensors does not copy them up front.

    Args:
        data: Flat list of numeric values, or a tensor-like object converted
            lazily.
        dtype: Tensor element type.
        shape: Tensor dimensions (e.g. ``[batch_size, seq_len]``).
    """

    dtype: TensorDtype = TensorDtype.FLOAT32
    shape: Optional[List[int]] = None

    _data: Optional[List[Any]] = PrivateAttr(default=None)
    _source: Any = PrivateAttr(default=None)

    def __init__(self, data: Any = None, **kwargs: Any) -> None:
        if data is None:
            raise ValueError("TensorData requires 'data'")
        super().__init__(**kwargs)
        if isinstance(data, list):
            self._data = data
        else:
            self._source = data

    @property
    def data(self) -> List[Any]:
        """Tensor values as a flat list (materialized from the source on first access)."""
        if self._data is None:
            source = self._source
            if hasattr(source, "tolist"):  # torch.Tensor / numpy.ndarray
                raw = source.tolist()
                self._data = raw if isinstance(raw, list) else [raw]
            else:
                self._data = list(source)
        if any(isinstance(v, list) for v in self._data):
            # The wire carries a flat ``data`` array alongside ``shape``, so a
            # multi-dimensional source has to be flattened. Left nested, it was
            # sent as a list of lists and rejected: the runtime types ``data`` as
            # an array of numbers. Done once, in place of the nested value.
            flat, nested_shape = _flatten_nested(self._data)
            self._data = flat
            if self.shape is None:
                self.shape = nested_shape
        return self._data

    def __eq__(self, other: Any) -> bool:
        """Compare on values, dtype and shape, whichever form each was built in.

        The values live outside the model fields and are materialized lazily, so
        the inherited comparison saw ``_source`` on one side and ``_data`` on the
        other and called two equal tensors different until both had been read.
        Reading through :attr:`data` normalizes that, and also settles ``shape``.
        """
        if not isinstance(other, TensorData):
            return NotImplemented
        return self.data == other.data and self.dtype == other.dtype and self.shape == other.shape

    # No __hash__ here on purpose. Defining __eq__ already makes this
    # unhashable, which is what the mutable values call for and what every other
    # model here does. Supplying one that raises instead would report the type as
    # hashable and then fail, sending any check for hashability down the wrong
    # branch.

    @model_serializer(mode="wrap")
    def _serialize_with_data(self, handler: Any) -> Dict[str, Any]:
        """Include the tensor values in ``model_dump()``.

        The values are held outside the model fields so a large tensor is not
        copied when a datum is built. That kept them out of the dump entirely,
        which made a dumped tensor unusable: it round-tripped to a
        ``TensorData`` with no values. Materializing here is the first point
        where the caller has asked to see them.
        """
        dumped = dict(handler(self))
        dumped["data"] = list(self.data)
        return dumped

    @field_validator("dtype", mode="before")
    @classmethod
    def _validate_dtype(cls, dtype: Any) -> Any:
        """Normalize whatever was passed as ``dtype`` before pydantic checks it.

        Runs on the field, so direct construction,
        ``TensorData(data=t.tolist(), dtype=t.dtype)``, gets the same coercion
        as :meth:`from_any` instead of failing pydantic enum validation because
        the field itself names only the wire spellings.
        """
        return cls._coerce_dtype(dtype) if dtype is not None else dtype

    @classmethod
    def _coerce_dtype(cls, dtype: Any) -> TensorDtype:
        """Normalize a dtype spec (enum, string, or torch/numpy dtype) to TensorDtype.

        A dtype the wire has no slot for is widened to one it does: a float to
        ``float32`` (narrowing ``float64``, which warns), an integer or boolean
        to ``int64``. A dtype that is not numeric at all raises rather than
        going out as ``float32``, which the service would reject or, worse,
        accept with reinterpreted values.
        """
        if isinstance(dtype, TensorDtype):
            return dtype
        name = _dtype_name(dtype)
        try:
            return TensorDtype(name)
        except ValueError:
            pass
        aliased = _DTYPE_ALIASES.get(name)
        if aliased is not None:
            return aliased
        kind = _dtype_kind(dtype, name)
        if kind == "f":
            logger.warning(
                "dtype %s is not supported on the wire; sending as float32. "
                "Cast to float32 yourself to control the rounding.",
                name,
            )
            return TensorDtype.FLOAT32
        if kind == "i":
            return TensorDtype.INT64
        raise ValueError(
            f"unsupported tensor dtype {name!r}: expected a float or integer "
            f"dtype (one of {', '.join(d.value for d in TensorDtype)}, or a "
            f"numpy/torch dtype convertible to one)"
        )

    @classmethod
    def from_any(cls, value: Any) -> "TensorData":
        """Coerce a tensor-like value to a TensorData.

        Accepts a :class:`TensorData` (passthrough), a wire-format dict, a
        torch tensor / numpy array (anything with ``tolist()``), a plain list,
        or any object with ``data``/``dtype`` attributes.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(
                data=value["data"],
                dtype=cls._coerce_dtype(value.get("dtype", TensorDtype.FLOAT32)),
                shape=value.get("shape"),
            )
        shape = getattr(value, "shape", None)
        if shape is not None and not isinstance(shape, list):
            shape = list(shape)  # torch.Size / numpy tuple
        if hasattr(value, "tolist"):  # torch.Tensor / numpy.ndarray, converted lazily
            raw: Any = value
        else:
            raw = getattr(value, "data", value)
        if not isinstance(raw, list) and not hasattr(raw, "tolist"):
            raw = list(raw)
        return cls(
            data=raw,
            dtype=cls._coerce_dtype(getattr(value, "dtype", TensorDtype.FLOAT32)),
            shape=shape,
        )

    def tolist(self) -> List[Any]:
        """Return the tensor values as a plain list."""
        return self.data

    def to_torch(self) -> Any:
        """Convert to a ``torch.Tensor`` (requires torch)."""
        import torch

        dtype_map = {
            TensorDtype.FLOAT32: torch.float32,
            TensorDtype.FLOAT16: torch.float16,
            TensorDtype.BFLOAT16: torch.bfloat16,
            TensorDtype.INT32: torch.int32,
            TensorDtype.INT64: torch.int64,
        }
        tensor = torch.tensor(self.data, dtype=dtype_map[self.dtype])
        return tensor.reshape(self.shape) if self.shape is not None else tensor

    def to_numpy(self) -> Any:
        """Convert to a ``numpy.ndarray`` (requires numpy)."""
        import numpy as np

        # numpy has no bfloat16; fall back to float32 for it.
        dtype_name = "float32" if self.dtype is TensorDtype.BFLOAT16 else self.dtype.value
        arr = np.asarray(self.data, dtype=np.dtype(dtype_name))
        return arr.reshape(self.shape) if self.shape is not None else arr

    def _to_api(self) -> Dict[str, Any]:
        """Convert to the runtime TensorData shape."""
        # Read first, not inline below: materializing can fill in ``shape`` from
        # the nesting, and that has to happen before ``shape`` is checked.
        data = self.data
        d: Dict[str, Any] = {"data": data, "dtype": self.dtype.value}
        if self.shape is not None:
            d["shape"] = self.shape
        return d


#: Wire dtype for a raw numeric list passed under a known loss-fn input name.
#: A list carries no dtype of its own, and the choice is not inferable from the
#: values: ``[0, 1, 1]`` is a valid weight mask and a valid token list.
_LOSS_FN_INPUT_DTYPES = {
    "target_tokens": TensorDtype.INT64,
    "weights": TensorDtype.FLOAT32,
    "advantages": TensorDtype.FLOAT32,
    "logprobs": TensorDtype.FLOAT32,
    "old_logprobs": TensorDtype.FLOAT32,
    "ref_log_prob": TensorDtype.FLOAT32,
    "temperature": TensorDtype.FLOAT32,
}


def _flatten_with_shape(key: str, values: List[Any]) -> "tuple[List[Any], List[int]]":
    """Flatten a 1-D or rectangular 2-D list, returning the values and shape.

    The wire carries a flat ``data`` array plus a ``shape``, so a nested list has
    to be flattened here rather than sent as-is.
    """
    rows = [v for v in values if isinstance(v, list)]
    if not rows:
        return values, [len(values)]
    if len(rows) != len(values):
        raise ValueError(
            f"loss_fn_inputs[{key!r}] mixes numbers and lists; pass either a flat "
            f"list of values or a list of equal-length rows"
        )
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(
            f"loss_fn_inputs[{key!r}] is a ragged nested list (row lengths "
            f"{sorted(len(r) for r in rows)}); rows must all be the same length"
        )
    if any(isinstance(v, list) for r in rows for v in r):
        raise ValueError(
            f"loss_fn_inputs[{key!r}] nests more than two levels deep; pass a "
            f"tensor or a flat list instead"
        )
    return [v for r in rows for v in r], [len(rows), widths.pop()]


def _loss_fn_input_dtype(key: str, flat: List[Any]) -> TensorDtype:
    """Wire dtype for a raw list under ``key``: by name, else inferred."""
    known = _LOSS_FN_INPUT_DTYPES.get(key)
    if known is not None:
        return known
    # bool is a subclass of int, and a mask of booleans is integral.
    if flat and all(isinstance(v, (int, bool)) for v in flat):
        return TensorDtype.INT64
    return TensorDtype.FLOAT32


def _serialize_loss_fn_inputs(loss_fn_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize ``loss_fn_inputs`` values to wire shapes.

    :class:`TensorData` and tensor-like values (torch tensors, numpy arrays,
    objects with ``data``/``dtype`` attributes) become the runtime TensorData
    dict. A raw list of numbers becomes one too, taking its dtype from the input
    name: ``target_tokens`` is integral, the rest are floating. A dict already
    in the runtime TensorData shape passes through unchanged.
    """
    out: Dict[str, Any] = {}
    for key, value in loss_fn_inputs.items():
        if isinstance(value, TensorData):
            out[key] = value._to_api()
        elif hasattr(value, "tolist") and hasattr(value, "shape"):
            # torch.Tensor / numpy.ndarray, detected without converting
            out[key] = TensorData.from_any(value)._to_api()
        elif not isinstance(value, dict) and hasattr(value, "data") and hasattr(value, "dtype"):
            out[key] = TensorData.from_any(value)._to_api()
        elif isinstance(value, list):
            # The wire wants a TensorData here, never a bare array. Left to pass
            # through, a plain list was rejected before the request went out.
            flat, shape = _flatten_with_shape(key, value)
            out[key] = TensorData(
                data=flat, dtype=_loss_fn_input_dtype(key, flat), shape=shape
            )._to_api()
        else:
            out[key] = value
    return out


class EncodedTextChunk(_ForbidExtraBase):
    """A chunk of token IDs representing encoded text.

    This is the fundamental building block for model inputs: a sequence of
    token IDs. Multiple chunks can be combined into a :class:`ModelInputChunk`
    list to represent structured prompts.

    Args:
        tokens: List of token IDs.
    """

    tokens: List[int]

    @property
    def length(self) -> int:
        """Number of tokens in this chunk."""
        return len(self.tokens)

    @classmethod
    def from_any(cls, value: Any) -> "EncodedTextChunk":
        """Coerce a chunk-like value to an EncodedTextChunk.

        Accepts an :class:`EncodedTextChunk` (passthrough), a
        :class:`ModelInputChunk` wrapper (or any object with the same attributes),
        a wire dict, a plain token ID list, or any object with ``tokens``.

        Raises:
            TypeError: If the value carries no token IDs. Rejected instead of
                coerced to an empty chunk, which would send a prompt the model
                never sees and report success.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            raise TypeError(
                "a chunk holds token IDs, not text; encode the text with the "
                f"model's tokenizer first (got {value!r})"
            )
        if isinstance(value, dict):
            if "chunk" in value:
                return cls.from_any(value["chunk"])
            if "tokens" not in value:
                raise TypeError(f"chunk dict carries no 'tokens': {value!r}")
            return cls(tokens=list(value["tokens"]))
        if isinstance(value, list):
            return cls(tokens=list(value))
        inner = getattr(value, "chunk", None)  # ModelInputChunk wrapper
        if inner is not None:
            return cls.from_any(inner)
        tokens = getattr(value, "tokens", None)
        if tokens is None:
            raise TypeError(
                f"cannot read token IDs from {type(value).__name__}; pass a "
                "list of token IDs or an EncodedTextChunk"
            )
        return cls(tokens=list(tokens))

    def _to_api(self) -> Dict[str, Any]:
        """Convert to the runtime chunk shape."""
        return {"tokens": self.tokens, "type": "encoded_text"}


class ModelInputChunk(_AllowExtraBase):
    """A single chunk within a structured model input.

    Wraps an :class:`EncodedTextChunk` (the only chunk type today). This
    layer exists so the type system can accommodate future chunk types
    (e.g. image embeddings) without breaking existing code.

    Args:
        chunk: The encoded text chunk.
    """

    chunk: EncodedTextChunk

    def _to_api(self) -> Dict[str, Any]:
        """Convert to the runtime chunk shape."""
        return self.chunk._to_api()


class ModelInput(_AllowExtraBase):
    """Structured model input composed of one or more chunks.

    Use as the ``prompt`` argument to :meth:`SamplingClient.sample` or to
    build typed training datums. Accepts either pre-built chunks or a
    simple prompt (token IDs or text string) via the convenience
    constructors :meth:`from_ints` and :meth:`from_text`.

    Args:
        chunks: List of input chunks. Each chunk is either a
            :class:`ModelInputChunk`, an :class:`EncodedTextChunk`, or a
            raw dict matching the wire format.
    """

    chunks: Optional[List[Any]] = None
    prompt: Optional[Any] = None

    @model_validator(mode="after")
    def _reject_both_inputs(self) -> "ModelInput":
        """One input or the other, for the reason a Datum takes one or the other.

        The wire shape is built from ``chunks`` when both are set, so a prompt
        passed alongside them goes nowhere. An empty prompt carries nothing and
        is not a second input.
        """
        if self.chunks is not None and self.prompt:
            raise ValueError(
                "a ModelInput takes either chunks or prompt, not both. Pass "
                "chunks for pre-built input, or prompt for token IDs or text."
            )
        return self

    @classmethod
    def from_ints(cls, tokens: List[int]) -> "ModelInput":
        """Build a ModelInput from a flat list of token IDs."""
        return cls(chunks=[EncodedTextChunk(tokens=list(tokens))])

    @classmethod
    def from_text(cls, text: str) -> "ModelInput":
        """Build a ModelInput from text.

        The text is tokenized when the input is submitted, using the session's
        tokenizer, the same handling a bare string gets. Passing one to a
        client that cannot resolve a tokenizer raises there.
        """
        return cls(prompt=text)

    @classmethod
    def empty(cls) -> "ModelInput":
        """Build a ModelInput with no chunks."""
        return cls(chunks=[])

    @classmethod
    def from_any(cls, value: Any) -> "ModelInput":
        """Coerce a prompt-like value to a ModelInput.

        Accepts a :class:`ModelInput` (passthrough), an
        :class:`EncodedTextChunk`, a wire dict (``chunks``, ``prompt``, or a
        single chunk's ``tokens``), a token ID list, or any object with
        ``chunks``.

        Raises:
            ValueError: A dict naming none of those keys. Such a dict once became
                an input with no chunks, which the service accepts, so a training
                example was dropped without anything to see.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, (EncodedTextChunk, ModelInputChunk)):
            return cls(chunks=[EncodedTextChunk.from_any(value)])
        if isinstance(value, dict):
            if value.get("chunks") is not None and value.get("prompt"):
                raise ValueError(
                    "a ModelInput takes either chunks or prompt, not both. Pass "
                    "chunks for pre-built input, or prompt for token IDs or text."
                )
            if value.get("chunks") is not None:
                return cls(chunks=[EncodedTextChunk.from_any(c) for c in value["chunks"]])
            if "prompt" in value:
                return cls(prompt=value["prompt"])
            if "tokens" in value:
                # A single chunk rather than a whole input.
                return cls(chunks=[EncodedTextChunk.from_any(value)])
            raise ValueError(
                f"cannot build a model input from a dict with keys {sorted(value)}: "
                "pass 'chunks', 'prompt', or a chunk's 'tokens'"
            )
        if isinstance(value, list):
            return cls.from_ints(value)
        raw_chunks = getattr(value, "chunks", None)
        if raw_chunks is not None:
            return cls(chunks=[EncodedTextChunk.from_any(c) for c in raw_chunks])
        return cls(prompt=value)

    @property
    def length(self) -> int:
        """Total number of tokens across all chunks.

        Raises:
            ValueError: This input has no token IDs to count, for the reasons
                :meth:`to_ints` gives.
        """
        return len(self.to_ints())

    def to_ints(self) -> List[int]:
        """Flatten all chunks to a single token ID list.

        Raises:
            ValueError: This input carries a prompt that is not token IDs, so
                it has none to return yet. Returning an empty list instead
                would read a full prompt as empty wherever a token budget is
                counted.
        """
        if self._pending_text is not None:
            raise ValueError(
                "this ModelInput holds text that has not been tokenized yet, so "
                "it has no token IDs. Build it with "
                "ModelInput.from_ints(tokenizer.encode(text)) first."
            )
        if self.chunks is not None:
            out: List[int] = []
            for c in self.chunks:
                out.extend(EncodedTextChunk.from_any(c).tokens)
            return out
        if isinstance(self.prompt, list):
            return list(self.prompt)
        if self.prompt is not None:
            raise ValueError(
                f"this ModelInput holds a prompt of type "
                f"{type(self.prompt).__name__}, which is not token IDs, so it "
                f"has none to return. Pass token IDs or text."
            )
        return []

    def append(self, chunk: Any) -> "ModelInput":
        """Return a new ModelInput with ``chunk`` appended (does not mutate).

        An input built from a prompt of token IDs carries those tokens into the
        chunks first, so appending extends it rather than replacing it.

        Raises:
            ValueError: This input still holds untokenized text. Appending built
                the result from ``chunks`` alone, so the text was dropped and the
                appended chunk became the whole input.
        """
        if self._pending_text is not None:
            raise ValueError(
                "cannot append to a ModelInput built from text: the text has not "
                "been tokenized yet, so there are no chunks to append to. Build it "
                "with ModelInput.from_ints(tokenizer.encode(text)) first."
            )
        existing = list(self.chunks or [])
        if not existing and self.prompt is not None:
            # A token-ID prompt is this input's whole content, the same way
            # ``_to_api`` reads it, so it has to survive the append.
            existing = [EncodedTextChunk.from_any(self.prompt)]
        return ModelInput(chunks=existing + [EncodedTextChunk.from_any(chunk)])

    def append_int(self, token: int) -> "ModelInput":
        """Return a new ModelInput with a single-token chunk appended."""
        return self.append(EncodedTextChunk(tokens=[token]))

    @property
    def _pending_text(self) -> Optional[str]:
        """The text awaiting tokenization, or ``None`` if there is none."""
        return self.prompt if isinstance(self.prompt, str) else None

    def _tokenized_with(self, tokenizer: Any) -> "ModelInput":
        """Return an equivalent ModelInput with any pending text tokenized."""
        text = self._pending_text
        if text is None:
            return self
        return ModelInput.from_ints(tokenizer.encode(text))

    def _to_api(self) -> Dict[str, Any]:
        """Convert to the runtime ModelInput shape (``{"chunks": [...]}``).

        If constructed with ``chunks``, serializes them. If constructed with a
        raw ``prompt`` of token IDs, wraps it in the chunks format.

        Raises:
            ValueError: The input still holds untokenized text. Only a tokenizer
                can turn it into chunks, and this type has none. Reaching the
                wire, the text was sent as a field the runtime does not define
                and the request was rejected.
        """
        if self.chunks is not None:
            api_chunks = []
            for c in self.chunks:
                if isinstance(c, (ModelInputChunk, EncodedTextChunk)):
                    api_chunks.append(c._to_api())
                elif isinstance(c, dict):
                    api_chunks.append(c)
                else:
                    api_chunks.append(EncodedTextChunk.from_any(c)._to_api())
            return {"chunks": api_chunks}
        if self.prompt is not None:
            if isinstance(self.prompt, list):
                return {"chunks": [{"tokens": self.prompt, "type": "encoded_text"}]}
            if self._pending_text is not None:
                raise ValueError(
                    "this ModelInput holds text that was never tokenized. Submit "
                    "it through a client method (which tokenizes for you), or "
                    "build it from token IDs with ModelInput.from_ints(...)."
                )
            raise TypeError(
                f"cannot build a model input from {type(self.prompt).__name__}: "
                f"pass token IDs, text, or pre-built chunks"
            )
        return {"chunks": []}


class Datum(_AllowExtraBase):
    """A single training datum (model input + optional loss function inputs).

    Use as the batch element for :meth:`TrainingClient.forward_backward` and
    :meth:`TrainingClient.forward`.

    Can be constructed from:
    - A :class:`ModelInput` with optional loss_fn_inputs
    - Simple token IDs (via ``prompt``) for convenience

    Args:
        model_input: Structured input (chunks-based). Mutually exclusive with
            ``prompt``.
        prompt: Convenience shorthand for token IDs or text. Converted to a
            :class:`ModelInput` internally.
        loss_fn_inputs: Per-datum loss function inputs, keyed by name:
            ``target_tokens``, ``weights``, ``advantages``, ``logprobs``,
            ``ref_log_prob``, ``temperature``. Which are required depends on the
            loss function. Values can be :class:`TensorData` objects, tensors, or
            raw lists. Per-token entries must be as long as the token sequence.
    """

    # Merges with the base config; clears the pydantic "model_" prefix guard
    # so the ``model_input`` field name is allowed.
    model_config = ConfigDict(protected_namespaces=())

    model_input: Optional[ModelInput] = None
    prompt: Optional[Any] = None
    loss_fn_inputs: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _reject_both_inputs(self) -> "Datum":
        """Enforce the documented exclusivity of ``model_input`` and ``prompt``.

        Both set, the wire shape was built from ``model_input`` and the prompt
        went nowhere, so a caller who meant to send it trained on something else.

        Raises:
            ValueError: Both are set.
        """
        # An empty prompt carries no input, so it is not a second one. Objects
        # coerced through ``from_any`` commonly declare ``prompt = ""`` as a
        # default, and rejecting those would turn a working call into an error
        # while dropping nothing.
        if self.model_input is not None and self.prompt:
            raise ValueError(
                "a Datum takes either model_input or prompt, not both. Pass "
                "model_input for chunked input, or prompt for token IDs or text."
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def _reject_datum_weight(cls, data: Any) -> Any:
        """Reject a ``weight``, pointing at the loss input that carries one.

        A datum weight is applied through ``loss_fn_inputs["weights"]``, one
        value per token, not as a scalar on the datum. This type accepts extra
        fields, so an unchecked ``weight`` would sit on the object and never be
        sent, leaving out the weighting the caller asked for.
        """
        if isinstance(data, dict) and "weight" in data:
            raise ValueError(
                "Datum has no 'weight' field. Weight a datum through "
                "loss_fn_inputs={'weights': [...]}, one value per token of "
                "model_input, e.g. 0.0 to mask a token out of the loss."
            )
        return data

    @classmethod
    def from_any(cls, value: Any) -> "Datum":
        """Coerce a datum-like value to a Datum.

        Accepts a :class:`Datum` (passthrough), a :class:`ModelInput` or a bare
        chunk, a token ID list or text (the ``prompt`` shorthand), a wire dict
        (``modelInput``/``lossFnInputs`` or snake_case keys), or any object with
        ``model_input`` / ``prompt`` / ``loss_fn_inputs`` attributes.

        Raises ``TypeError`` for a value carrying none of those. Every field on
        this type is optional, so an unrecognised value would otherwise become
        an empty ``Datum`` serializing to ``{}``, dropping a training example
        from the batch with nothing raised. A mixed batch reaches this path:
        the element kind is decided by the *first* entry, and later entries
        still come through here one at a time.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, (ModelInput, EncodedTextChunk, ModelInputChunk)):
            # ModelInput.from_any wraps a bare chunk; a ModelInput passes through.
            return cls(model_input=ModelInput.from_any(value))
        if isinstance(value, dict):
            mi = value.get("modelInput", value.get("model_input"))
            lfi = value.get("lossFnInputs", value.get("loss_fn_inputs"))
            prompt = value.get("prompt")
            cls._reject_datum_weight(value)
            if mi is None and lfi is None and prompt is None:
                raise TypeError(
                    f"cannot read Datum fields from dict with keys "
                    f"{sorted(map(str, value))}: none of 'modelInput', 'prompt', "
                    f"'lossFnInputs' are present (snake_case also accepted)."
                )
            if mi is not None:
                # ``prompt`` is passed through rather than ignored, so a dict
                # naming both is rejected by the same rule the constructor
                # applies instead of quietly losing one of them.
                return cls(
                    model_input=ModelInput.from_any(mi),
                    prompt=prompt,
                    loss_fn_inputs=lfi or None,
                )
            return cls(prompt=prompt, loss_fn_inputs=lfi or None)
        if isinstance(value, (str, list, tuple)):
            # The prompt shorthand documented on this type: token IDs or text.
            return cls(prompt=list(value) if isinstance(value, tuple) else value)
        mi = getattr(value, "model_input", None)
        # ``is not None``, not ``hasattr``: an object declaring ``chunks = None``
        # is not a ModelInput-like value, and treating it as one would wrap the
        # object itself as a prompt instead of reaching the TypeError below.
        if mi is None and getattr(value, "chunks", None) is not None:
            # A bare ModelInput-like object (ours already handled above).
            return cls(model_input=ModelInput.from_any(value))
        lfi = getattr(value, "loss_fn_inputs", None)
        prompt = getattr(value, "prompt", None)
        if mi is None and lfi is None and prompt is None:
            raise TypeError(
                f"cannot read Datum fields from {type(value).__name__}: none of "
                f"'model_input', 'chunks', 'prompt', 'loss_fn_inputs' are present. "
                f"Pass a Datum, a ModelInput, a token ID list, text, or a dict."
            )
        # A caller's own datum type that models a weight is caught rather than
        # having it silently dropped: there is nowhere on the wire to put one.
        if getattr(value, "weight", None) is not None:
            cls._reject_datum_weight({"weight": getattr(value, "weight")})
        return cls(
            model_input=ModelInput.from_any(mi) if mi is not None else None,
            prompt=prompt,
            loss_fn_inputs=dict(lfi) if lfi else None,
        )

    @property
    def _pending_text(self) -> Optional[str]:
        """The text awaiting tokenization, or ``None`` if there is none."""
        if isinstance(self.prompt, str):
            return self.prompt
        return self.model_input._pending_text if self.model_input is not None else None

    def _tokenized_with(self, tokenizer: Any) -> "Datum":
        """Return an equivalent Datum with any pending text tokenized."""
        text = self._pending_text
        if text is None:
            return self
        return Datum(
            model_input=ModelInput.from_ints(tokenizer.encode(text)),
            loss_fn_inputs=self.loss_fn_inputs,
        )

    def _to_api(self) -> Dict[str, Any]:
        """Convert to the runtime Datum shape.

        Raises:
            ValueError: The datum still holds untokenized text; see
                :meth:`ModelInput._to_api`.
        """
        if self.model_input is not None:
            d: Dict[str, Any] = {"modelInput": self.model_input._to_api()}
        elif self.prompt is not None:
            # Routed through ModelInput so token IDs, text and unusable values
            # are each handled the one way, here and for a bare model input.
            d = {"modelInput": ModelInput.from_any(self.prompt)._to_api()}
        else:
            d = {}
        if self.loss_fn_inputs is not None:
            d["lossFnInputs"] = _serialize_loss_fn_inputs(self.loss_fn_inputs)
        return d


class LossFunction(str, Enum):
    """Built-in loss functions for training operations.

    Supported values: cross_entropy, importance_sampling, ppo, cispo.
    """

    CROSS_ENTROPY = "cross_entropy"
    IMPORTANCE_SAMPLING = "importance_sampling"
    PPO = "ppo"
    CISPO = "cispo"


class AdamParams(_ForbidExtraBase):
    """Optimizer parameters for Adam.

    Args:
        learning_rate: Step size for parameter updates.
        beta1: Exponential decay rate for the first moment estimates.
        beta2: Exponential decay rate for the second moment estimates.
        eps: Small constant for numerical stability.
        weight_decay: L2 regularization coefficient.
        grad_clip_norm: Gradient clipping norm.
        lora_plus_lr_ratio: Learning-rate multiplier for LoRA+ B matrices
            relative to A matrices. Typical values are 8-32. When set, the
            service applies ``learning_rate * lora_plus_lr_ratio`` to B
            matrices while A matrices use ``learning_rate``.
    """

    model_config = ConfigDict(validate_assignment=True)

    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01
    grad_clip_norm: Optional[float] = None
    lora_plus_lr_ratio: Optional[float] = None

    @classmethod
    def from_any(cls, value: Any) -> "AdamParams":
        """Coerce a params-like value to AdamParams.

        Accepts an :class:`AdamParams` (passthrough), a dict of fields, or any
        object sharing at least one field name. Raises ``TypeError`` for an
        object that shares none, instead of silently returning all defaults.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**_drop_unset(value))
        return _params_from_attrs(cls, value)

    def _to_api(self) -> Dict[str, Any]:
        """Convert to the runtime AdamParamsConfig shape."""
        d: Dict[str, Any] = {
            "learningRate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weightDecay": self.weight_decay,
        }
        if self.grad_clip_norm is not None:
            d["gradClipNorm"] = self.grad_clip_norm
        if self.lora_plus_lr_ratio is not None:
            d["loraPlusLrRatio"] = self.lora_plus_lr_ratio
        return d


class SamplingParams(_ForbidExtraBase):
    """Parameters for sampling / inference operations.

    Args:
        max_tokens: Maximum number of tokens to generate. The service requires
            a minimum of 512.
        temperature: Sampling temperature. Must not be negative; 0 is greedy
            decoding. The service applies its own upper limit.
        top_p: Nucleus sampling probability threshold, between 0 and 1. It is a
            cumulative probability, so a percentage such as ``95`` is rejected
            here rather than at the service.
        top_k: Top-k sampling parameter. ``-1`` disables it.
        seed: Random seed for reproducibility.
        stop: Stop sequences.
    """

    model_config = ConfigDict(validate_assignment=True)

    # Only the two fractional parameters are bounded here, and only where the
    # range is a property of the value rather than a service limit. A whole
    # number outside its range is refused before the request is sent, so
    # max_tokens needs no guard; its ceiling, like the rest of the service's
    # limits, is the service's to raise and not worth a stale copy of. top_k
    # takes -1 to mean "off", so a floor would refuse the documented way to
    # disable it.
    max_tokens: int = 512
    temperature: Optional[float] = Field(default=None, ge=0)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    top_k: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[List[str]] = None

    @field_validator("stop", mode="before")
    @classmethod
    def _coerce_stop(cls, value: Any) -> Any:
        """Accept a single stop sequence as a bare string.

        The service models ``stop`` as a list of text sequences. A lone
        sequence is naturally written without the enclosing list, so wrap it
        without making the caller remember the difference. Token IDs are
        rejected with the reason, because the service takes text only.
        """
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)) and any(isinstance(v, int) for v in value):
            raise ValueError(
                "stop takes text sequences, not token IDs; the service matches "
                "stop sequences on decoded text. Pass the text form instead."
            )
        return value

    @classmethod
    def from_any(cls, value: Any) -> "SamplingParams":
        """Coerce a params-like value to SamplingParams.

        Accepts a :class:`SamplingParams` (passthrough), a dict of fields, or any
        object sharing at least one field name. Raises ``TypeError`` for an
        object that shares none, instead of silently returning all defaults.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**_drop_unset(value))
        return _params_from_attrs(cls, value)

    def _to_api(self) -> Dict[str, Any]:
        """Convert to the runtime SamplingParams shape."""
        d: Dict[str, Any] = {"maxTokens": self.max_tokens}
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.top_p is not None:
            d["topP"] = self.top_p
        if self.top_k is not None:
            d["topK"] = self.top_k
        if self.seed is not None:
            d["seed"] = self.seed
        if self.stop is not None:
            d["stop"] = self.stop
        return d


class StopReason(str, Enum):
    """Reason why text generation stopped."""

    STOP = "stop"
    LENGTH = "length"


class SampledSequence(_ForbidExtraBase):
    """A single generated sequence from a sampling operation.

    Args:
        stop_reason: Why generation stopped.
        tokens: Generated token IDs.
        logprobs: Per-token log probabilities (if requested).
    """

    stop_reason: StopReason
    tokens: List[int]
    logprobs: Optional[List[float]] = None

    @classmethod
    def _from_api(cls, raw: Dict[str, Any]) -> "SampledSequence":
        """Construct from a service response SampledSequence shape."""
        return cls(
            stop_reason=StopReason(raw["stopReason"]),
            tokens=raw["tokens"],
            logprobs=raw.get("logprobs"),
        )


class TokenLogprob(_ForbidExtraBase):
    """A token ID and its log probability."""

    token_id: int
    logprob: float

    @classmethod
    def _from_api(cls, raw: Dict[str, Any]) -> "TokenLogprob":
        """Construct from a service response TokenLogprob shape."""
        return cls(token_id=raw["tokenId"], logprob=raw["logprob"])


class _OperationResult(_ForbidExtraBase):
    """Base accessor over the ``operationResult`` of a completed operation.

    Wraps the raw ``OperationResult`` shape returned by
    ``GetTrainingOperationResult`` and keeps it internally. Each operation
    resolves to a dedicated subclass that derives its accessors from the
    sub-result key the service populates for that operation, so callers never
    probe the union themselves. Remains a :class:`~pydantic.BaseModel` for
    consistency with the rest of the surface.
    """

    _raw: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @classmethod
    def _from_api(cls, raw: Dict[str, Any]):
        """Construct from a service ``GetTrainingOperationResult`` operationResult."""
        obj = cls()
        obj._raw = raw
        return obj

    def _sub_result(self, *keys: str) -> Dict[str, Any]:
        """Return the first populated sub-result among ``keys``, or an empty dict."""
        for key in keys:
            if key in self._raw:
                return self._raw[key] or {}
        return {}


class ForwardBackwardOutput(_OperationResult):
    """Result of a completed ``forward_backward`` / ``forward`` operation."""

    @property
    def metrics(self) -> Dict[str, float]:
        """Return scalar training metrics for the pass.

        Contains the throughput metrics (``promised_flops``,
        ``global_token_num``, ``mfu``, ``estimated_flops``) plus scalar
        mirrors of the loss diagnostics (``actor/pg_loss``,
        ``actor/grad_norm``, ...); the tensor-form diagnostics are in
        :attr:`loss_fn_outputs`.
        """
        return self._sub_result("forward", "forwardBackward").get("metrics", {})

    @property
    def loss_fn_outputs(self) -> Optional[List[Dict[str, Any]]]:
        """Return loss-function tensor outputs from the pass, one entry per datum.

        Entries carry the loss outputs under ``actor/*`` keys
        (e.g. ``actor/pg_loss``, ``actor/ppo_kl``, ``actor/pg_clipfrac``,
        ``actor/grad_norm``).
        """
        return self._sub_result("forward", "forwardBackward").get("lossFnOutputs")


class OptimStepResponse(_OperationResult):
    """Result of a completed ``optim_step`` operation."""

    @property
    def metrics(self) -> Dict[str, float]:
        """Return the optimizer-step metrics, such as the learning rate."""
        return self._sub_result("optimizerStep").get("metrics", {})


class SaveStateResponse(_OperationResult):
    """Result of a completed ``save_state`` operation."""

    @property
    def model_package_arn(self) -> Optional[str]:
        """Return the ARN of the model package the checkpoint was registered as."""
        return self._sub_result("saveState").get("modelPackageArn")


class SaveWeightsResponse(_OperationResult):
    """Result of a completed ``save_weights_for_sampler`` operation."""

    @property
    def model_package_arn(self) -> Optional[str]:
        """Return the ARN of the model package the weights were registered as."""
        return self._sub_result("saveWeights").get("modelPackageArn")


class LoadStateResponse(_OperationResult):
    """Result of a completed ``load_state`` / ``load_state_with_optimizer`` operation.

    Successful resolution of the future is the signal that the state was
    loaded; the service returns no payload fields for load operations.
    """


class SampleResponse(_ForbidExtraBase):
    """Result of a completed sampling operation.

    Args:
        sequences: Generated sequences.
        prompt_logprobs: Log probabilities for prompt tokens (if requested).
        topk_prompt_logprobs: Top-k log probabilities per prompt position
            (if requested).
    """

    sequences: List[SampledSequence]
    prompt_logprobs: Optional[List[float]] = None
    topk_prompt_logprobs: Optional[List[List[TokenLogprob]]] = None

    @classmethod
    def _from_api(cls, raw: Dict[str, Any]) -> "SampleResponse":
        """Construct from a service sampling-operation ``sample`` result."""
        sequences = [SampledSequence._from_api(s) for s in raw.get("sequences") or []]
        topk_raw = raw.get("topkPromptLogprobs")
        topk = [[TokenLogprob._from_api(t) for t in pos] for pos in topk_raw] if topk_raw else None
        return cls(
            sequences=sequences,
            prompt_logprobs=raw.get("promptLogprobs"),
            topk_prompt_logprobs=topk,
        )
