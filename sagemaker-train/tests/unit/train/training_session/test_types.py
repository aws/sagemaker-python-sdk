# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
"""Unit tests for sagemaker.train.training_session.types."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from sagemaker.train.training_session.types import (
    AdamParams,
    Datum,
    EncodedTextChunk,
    ForwardBackwardOutput,
    LoadStateResponse,
    LossFunction,
    ModelInput,
    ModelInputChunk,
    ModelPackageDeletion,
    ModelPackageInfo,
    ModelPackageKind,
    OptimStepResponse,
    SampleResponse,
    SampledSequence,
    SamplingParams,
    SaveWeightsResponse,
    SaveStateResponse,
    SessionStatus,
    StopReason,
    TensorData,
    TensorDtype,
    TokenLogprob,
    VpcConfig,
    _serialize_loss_fn_inputs,
)


class TestLossFunction:
    def test_enum_values(self):
        assert LossFunction.PPO.value == "ppo"
        assert LossFunction.IMPORTANCE_SAMPLING.value == "importance_sampling"
        assert LossFunction.CISPO.value == "cispo"

    def test_from_string(self):
        assert LossFunction("ppo") == LossFunction.PPO

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            LossFunction("invalid")


class TestAdamParams:
    def test_defaults(self):
        p = AdamParams()
        assert p.learning_rate == 1e-4
        assert p.beta1 == 0.9
        assert p.beta2 == 0.999

    def test_to_api_minimal(self):
        api = AdamParams()._to_api()
        assert api["learningRate"] == 1e-4
        assert "gradClipNorm" not in api

    def test_to_api_with_grad_clip(self):
        api = AdamParams(grad_clip_norm=1.0)._to_api()
        assert api["gradClipNorm"] == 1.0

    def test_to_api_with_lora_plus_lr_ratio(self):
        api = AdamParams(lora_plus_lr_ratio=16.0)._to_api()
        assert api["loraPlusLrRatio"] == 16.0

    def test_to_api_omits_lora_plus_lr_ratio_when_none(self):
        api = AdamParams()._to_api()
        assert "loraPlusLrRatio" not in api

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            AdamParams(unknown=1)


class TestSamplingParams:
    def test_defaults(self):
        # Service requires max_tokens >= 512; default must satisfy it.
        p = SamplingParams()
        assert p.max_tokens == 512

    def test_to_api(self):
        p = SamplingParams(max_tokens=512, temperature=0.7, top_k=50)
        api = p._to_api()
        assert api == {"maxTokens": 512, "temperature": 0.7, "topK": 50}

    def test_to_api_minimal(self):
        api = SamplingParams()._to_api()
        assert api == {"maxTokens": 512}

    @pytest.mark.parametrize(
        "field,value",
        [
            ("temperature", -0.1),
            ("top_p", 95),  # a percentage, the mistake this guard exists for
            ("top_p", 1.5),
            ("top_p", -0.1),
        ],
    )
    def test_a_probability_outside_its_own_range_is_refused(self, field, value):
        """These two carry no meaning outside their range, and nothing else stops them.

        The wire model bounds both, but that bound is not applied before the
        request is sent, so an out-of-range value would otherwise travel to the
        service and fail there, one background submit away from the call that
        made it.
        """
        with pytest.raises(ValidationError):
            SamplingParams(max_tokens=512, **{field: value})

    @pytest.mark.parametrize(
        "field,value",
        [("temperature", 0), ("temperature", 2), ("top_p", 0), ("top_p", 1)],
    )
    def test_the_range_ends_are_usable(self, field, value):
        """Zero temperature is greedy decoding and top_p of 1 keeps everything."""
        p = SamplingParams(max_tokens=512, **{field: value})

        assert getattr(p, field) == value


class TestParamsFromAny:
    """The attribute path of from_any on the all-optional params types.

    Both types default every field, so an object that shares no field names
    would otherwise be accepted and silently replaced by our defaults.
    """

    class TheirParams:
        """A caller's own params class: same field names, no inheritance."""

        max_tokens = 1024
        temperature = 0.7
        learning_rate = 0.5

    class Partial:
        """Models fewer knobs than we do; the rest must take our defaults."""

        temperature = 0.3
        beta1 = 0.5

    class Unrelated:
        """camelCase / misspelt: shares no field name with either type."""

        maxTokens = 1024
        temp = 0.7
        lr = 0.5

    @pytest.mark.parametrize("cls", [SamplingParams, AdamParams])
    def test_passthrough(self, cls):
        p = cls()
        assert cls.from_any(p) is p

    def test_sampling_params_from_duck_typed_object(self):
        p = SamplingParams.from_any(self.TheirParams())
        assert p.max_tokens == 1024
        assert p.temperature == 0.7

    def test_adam_params_from_duck_typed_object(self):
        p = AdamParams.from_any(self.TheirParams())
        assert p.learning_rate == 0.5

    def test_partial_object_keeps_our_defaults_for_the_rest(self):
        assert SamplingParams.from_any(self.Partial()).temperature == 0.3
        assert SamplingParams.from_any(self.Partial()).max_tokens == 512
        assert AdamParams.from_any(self.Partial()).beta1 == 0.5
        assert AdamParams.from_any(self.Partial()).learning_rate == 1e-4

    @pytest.mark.parametrize("cls", [SamplingParams, AdamParams])
    def test_object_sharing_no_field_names_is_rejected(self, cls):
        # Regression: this used to return all-defaults, so a caller's values
        # were dropped without a word.
        with pytest.raises(TypeError, match="none of"):
            cls.from_any(self.Unrelated())

    @pytest.mark.parametrize("cls", [SamplingParams, AdamParams])
    def test_dict_with_an_unknown_key_is_rejected(self, cls):
        with pytest.raises(ValidationError):
            cls.from_any({"definitely_not_a_field": 1})


class TestOperationResults:
    def test_forward_metrics_and_loss_fn_outputs(self):
        r = ForwardBackwardOutput._from_api(
            {"forward": {"metrics": {"loss": 1.5}, "lossFnOutputs": [{"logprobs": [0.1]}]}}
        )
        assert r.metrics == {"loss": 1.5}
        assert r.loss_fn_outputs == [{"logprobs": [0.1]}]

    def test_forward_backward_metrics(self):
        r = ForwardBackwardOutput._from_api({"forwardBackward": {"metrics": {"grad_norm": 0.2}}})
        assert r.metrics == {"grad_norm": 0.2}


class TestLossFnOutputsWireShapes:
    """``lossFnOutputs`` is a per-datum list (``TensorDataMapList``).

    The field is either a list, an empty list, or absent; a mapping is not one
    of the shapes the response can carry.
    """

    def _fb(self, body):
        return ForwardBackwardOutput._from_api({"forwardBackward": body})

    def test_per_datum_entries(self):
        r = self._fb({"lossFnOutputs": [{"actor/pg_loss": [1.0]}, {"actor/pg_loss": [2.0]}]})
        assert r.loss_fn_outputs == [{"actor/pg_loss": [1.0]}, {"actor/pg_loss": [2.0]}]

    def test_absent_is_none_not_empty(self):
        """Absent stays distinguishable from an empty result.

        An operation that produced no per-datum outputs omits the field
        altogether, which is not the same answer as an empty list.
        """
        r = self._fb({"metrics": {"grad_norm": 0.2}})
        assert r.loss_fn_outputs is None

    def test_empty_list_preserved(self):
        r = self._fb({"lossFnOutputs": []})
        assert r.loss_fn_outputs == []

    def test_optimizer_step_metrics(self):
        r = OptimStepResponse._from_api({"optimizerStep": {"metrics": {"lr": 1e-4}}})
        assert r.metrics == {"lr": 1e-4}

    def test_model_package_arn_from_save_ops(self):
        assert (
            SaveStateResponse._from_api(
                {"saveState": {"modelPackageArn": "arn:x"}}
            ).model_package_arn
            == "arn:x"
        )
        assert (
            SaveWeightsResponse._from_api(
                {"saveWeights": {"modelPackageArn": "arn:y"}}
            ).model_package_arn
            == "arn:y"
        )

    def test_empty_result_defaults(self):
        r = ForwardBackwardOutput._from_api({})
        assert r.metrics == {}
        assert r.loss_fn_outputs is None
        assert OptimStepResponse._from_api({}).metrics == {}
        assert SaveStateResponse._from_api({}).model_package_arn is None
        assert SaveWeightsResponse._from_api({}).model_package_arn is None
        assert LoadStateResponse._from_api({}) is not None

    def test_is_basemodel_raw_not_public(self):
        raw = {"saveWeights": {"modelPackageArn": "arn:z"}}
        r = SaveWeightsResponse._from_api(raw)
        assert isinstance(r, BaseModel)
        assert r.model_package_arn == "arn:z"
        assert "raw" not in type(r).model_fields

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ForwardBackwardOutput(unexpected=1)


class _CompatTensor:
    """External-object stand-in with TensorData-shaped attributes."""

    def __init__(self, data, dtype="float32"):
        self.data = data
        self.dtype = dtype


class _CompatChunk:
    """External-object stand-in with EncodedTextChunk-shaped attributes."""

    def __init__(self, tokens):
        self.tokens = tokens


class _CompatModelInput:
    """External-object stand-in with ModelInput-shaped attributes."""

    def __init__(self, chunks):
        self.chunks = chunks


class _CompatDatum:
    """External-object stand-in with Datum-shaped attributes."""

    def __init__(self, model_input, loss_fn_inputs=None):
        self.model_input = model_input
        self.loss_fn_inputs = loss_fn_inputs or {}


class _CompatCompressedTensor:
    """External tensor whose ``data`` holds a compressed encoding.

    Stands in for the tensor libraries that store a sparse encoding in ``data``
    and expand it in ``tolist()``, with ``shape`` describing the *dense* tensor.
    Reading ``data`` here sends the non-zero values alone against a dense shape,
    a mask silently missing most of its entries, so ``tolist()`` has to win.
    """

    def __init__(self, dense_rows, dtype="float32"):
        self._dense_rows = dense_rows
        self.shape = [len(dense_rows), len(dense_rows[0])]
        self.dtype = dtype
        # The compressed payload: non-zero values only.
        self.data = [v for row in dense_rows for v in row if v]

    def tolist(self):
        return [list(row) for row in self._dense_rows]


class TestTensorData:
    def test_to_api(self):
        api = TensorData(data=[1.0, 2.0], dtype=TensorDtype.FLOAT32)._to_api()
        assert api == {"data": [1.0, 2.0], "dtype": "float32"}

    def test_to_api_with_shape(self):
        api = TensorData(data=[1, 2, 3, 4], dtype=TensorDtype.INT64, shape=[2, 2])._to_api()
        assert api["shape"] == [2, 2]
        assert api["dtype"] == "int64"

    def test_from_any_passthrough(self):
        td = TensorData(data=[1.0])
        assert TensorData.from_any(td) is td

    def test_from_any_dict(self):
        td = TensorData.from_any({"data": [1, 2], "dtype": "int32", "shape": [2]})
        assert td.data == [1, 2]
        assert td.dtype == TensorDtype.INT32
        assert td.shape == [2]

    def test_from_any_plain_list(self):
        td = TensorData.from_any([0.5, 0.7])
        assert td.data == [0.5, 0.7]
        assert td.dtype == TensorDtype.FLOAT32

    def test_from_any_compatible(self):
        td = TensorData.from_any(_CompatTensor([1.0, 2.0], dtype="bfloat16"))
        assert td.data == [1.0, 2.0]
        assert td.dtype == TensorDtype.BFLOAT16

    def test_from_any_torch_style_dtype_string(self):
        td = TensorData.from_any(_CompatTensor([1], dtype="torch.int64"))
        assert td.dtype == TensorDtype.INT64

    def test_from_any_non_numeric_dtype_raises(self):
        """A dtype the wire cannot carry is refused, not sent as float32.

        Reinterpreting complex values as float32 produced a request the service
        either rejected or trained on with the wrong numbers.
        """
        with pytest.raises(ValueError, match="unsupported tensor dtype 'complex128'"):
            TensorData.from_any(_CompatTensor([1.0], dtype="complex128"))

    def test_from_any_float64_narrows_to_float32(self):
        td = TensorData.from_any(_CompatTensor([1.0], dtype="torch.float64"))
        assert td.dtype == TensorDtype.FLOAT32

    def test_from_any_bool_widens_to_int64(self):
        """A boolean mask reaches the loss function as 0/1 integers, not floats."""
        td = TensorData.from_any(_CompatTensor([1], dtype="torch.bool"))
        assert td.dtype == TensorDtype.INT64

    def test_direct_construction_coerces_a_library_dtype(self):
        """``TensorData(data=..., dtype=t.dtype)`` is the obvious way to write it.

        The coercion used to live only on ``from_any``, so passing a tensor
        library's dtype object, or its plain name, failed enum validation
        with no hint that a conversion existed.
        """
        np = pytest.importorskip("numpy")
        assert TensorData(data=[1.0], dtype=np.dtype("float64")).dtype == TensorDtype.FLOAT32
        assert TensorData(data=[1], dtype=np.dtype("bool")).dtype == TensorDtype.INT64
        assert TensorData(data=[1.0], dtype="float64").dtype == TensorDtype.FLOAT32

    def test_two_dimensional_source_is_flattened_with_shape(self):
        """The wire carries a flat ``data`` array plus ``shape``, never nested rows.

        A 2-D tensor reached the wire as a list of lists, which the runtime types
        as an array of numbers and rejects.
        """
        np = pytest.importorskip("numpy")
        api = TensorData.from_any(np.arange(6, dtype=np.float32).reshape(2, 3))._to_api()
        assert api == {"data": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "dtype": "float32", "shape": [2, 3]}

    def test_nested_data_without_a_shape_infers_one(self):
        api = TensorData(data=[[1.0, 2.0], [3.0, 4.0]])._to_api()
        assert api == {"data": [1.0, 2.0, 3.0, 4.0], "dtype": "float32", "shape": [2, 2]}

    def test_ragged_nested_data_raises(self):
        with pytest.raises(ValueError, match="ragged"):
            TensorData(data=[[1.0, 2.0], [3.0]])._to_api()

    def test_unknown_float_width_narrows_to_float32(self):
        """Float widths keep being added; a name table silently rejects each new one."""
        assert TensorData._coerce_dtype("float8_e4m3fn") == TensorDtype.FLOAT32
        assert TensorData._coerce_dtype("float128") == TensorDtype.FLOAT32

    def test_half_is_float16_not_float32(self):
        """``half`` names a dtype the wire carries, so it must not be narrowed."""
        assert TensorData._coerce_dtype("half") == TensorDtype.FLOAT16

    def test_every_dtype_spelling_agrees(self):
        """A scalar type, a dtype instance and a plain name must reach one answer.

        A scalar type stringifies as ``<class 'numpy.int32'>``; unwrapping that
        badly left ``int32'>``, which matched nothing, so a dtype the wire
        carries was refused, and ``float32`` warned about narrowing to itself.
        """
        np = pytest.importorskip("numpy")
        for spelling in (np.int32, np.dtype("int32"), "int32"):
            assert TensorData._coerce_dtype(spelling) == TensorDtype.INT32
        for spelling in (np.float32, np.dtype("float32"), "float32"):
            assert TensorData._coerce_dtype(spelling) == TensorDtype.FLOAT32
        with pytest.raises(ValueError, match="'complex128'"):
            TensorData._coerce_dtype(np.complex128)

    def test_tolist(self):
        assert TensorData(data=[1, 2]).tolist() == [1, 2]

    def test_to_numpy(self):
        np = pytest.importorskip("numpy")
        arr = TensorData(data=[1.0, 2.0], dtype=TensorDtype.FLOAT32).to_numpy()
        assert arr.dtype == np.float32
        assert arr.tolist() == [1.0, 2.0]

    def test_to_numpy_respects_shape(self):
        pytest.importorskip("numpy")
        arr = TensorData(data=[1, 2, 3, 4], dtype=TensorDtype.INT32, shape=[2, 2]).to_numpy()
        assert arr.shape == (2, 2)


class TestSerializeLossFnInputs:
    def test_tensor_data_serialized(self):
        out = _serialize_loss_fn_inputs({"advantages": TensorData(data=[0.1, 0.2])})
        assert out["advantages"] == {"data": [0.1, 0.2], "dtype": "float32"}

    def test_compatible_tensor_serialized(self):
        out = _serialize_loss_fn_inputs({"weights": _CompatTensor([1.0], dtype="float16")})
        assert out["weights"] == {"data": [1.0], "dtype": "float16"}

    def test_raw_list_becomes_tensor_data(self):
        """The wire wants a TensorData here; a bare list was rejected locally.

        dtype comes from the input name, which a list cannot carry: ``[1, 2, 3]``
        is a valid token list and a valid weight mask.
        """
        out = _serialize_loss_fn_inputs({"target_tokens": [1, 2, 3], "weights": [1, 0, 1]})
        assert out["target_tokens"] == {"data": [1, 2, 3], "dtype": "int64", "shape": [3]}
        assert out["weights"] == {"data": [1, 0, 1], "dtype": "float32", "shape": [3]}

    def test_raw_list_under_unknown_key_infers_dtype(self):
        out = _serialize_loss_fn_inputs({"custom": [1, 2], "custom_f": [1.5]})
        assert out["custom"]["dtype"] == "int64"
        assert out["custom_f"]["dtype"] == "float32"

    def test_nested_list_is_flattened_with_shape(self):
        out = _serialize_loss_fn_inputs({"weights": [[1.0, 0.0], [0.5, 1.0]]})
        assert out["weights"] == {
            "data": [1.0, 0.0, 0.5, 1.0],
            "dtype": "float32",
            "shape": [2, 2],
        }

    def test_ragged_nested_list_raises(self):
        with pytest.raises(ValueError, match="ragged nested list"):
            _serialize_loss_fn_inputs({"weights": [[1.0, 0.0], [1.0]]})

    def test_non_list_values_pass_through(self):
        """A dict is the wire shape already, or an advanced escape hatch."""
        raw = {"config": {"k": "v"}}
        assert _serialize_loss_fn_inputs(raw) == raw

    def test_numpy_array_serialized(self):
        np = pytest.importorskip("numpy")
        out = _serialize_loss_fn_inputs({"advantages": np.array([0.5, 0.6], dtype=np.float32)})
        assert out["advantages"]["data"] == pytest.approx([0.5, 0.6])
        assert out["advantages"]["dtype"] == "float32"

    def test_two_dimensional_array_is_flattened(self):
        np = pytest.importorskip("numpy")
        out = _serialize_loss_fn_inputs({"weights": np.ones((2, 3), dtype=np.float32)})
        assert out["weights"] == {"data": [1.0] * 6, "dtype": "float32", "shape": [2, 3]}


class TestExternalTensorObjects:
    """Values built by another tensor library must serialize to the same wire shape.

    A caller porting an existing training loop passes that library's objects
    straight through, so these paths carry real data and cannot be allowed to
    reinterpret it.
    """

    def test_expanded_values_win_over_a_compressed_data_attribute(self):
        """``tolist()`` is the authority on the values, not ``data``.

        For a compressed encoding the two disagree: ``data`` holds only the
        non-zero values while ``shape`` describes the dense tensor. Reading
        ``data`` sent a mask with most of its entries missing, against a shape
        claiming they were there: accepted, and trained on the wrong numbers.
        """
        weights = _CompatCompressedTensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        assert weights.data == [1.0, 2.0]  # the compressed payload
        out = _serialize_loss_fn_inputs({"weights": weights})
        assert out["weights"] == {
            "data": [1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            "dtype": "float32",
            "shape": [2, 3],
        }

    def test_external_datum_of_external_tensors_round_trips(self):
        datum = _CompatDatum(
            model_input=_CompatModelInput([_CompatChunk([10, 11, 12])]),
            loss_fn_inputs={
                "target_tokens": _CompatTensor([11, 12, 13], dtype="int64"),
                "weights": _CompatCompressedTensor([[0.0, 1.0, 1.0]]),
            },
        )
        assert Datum.from_any(datum)._to_api() == {
            "modelInput": {"chunks": [{"tokens": [10, 11, 12], "type": "encoded_text"}]},
            "lossFnInputs": {
                "target_tokens": {"data": [11, 12, 13], "dtype": "int64"},
                "weights": {"data": [0.0, 1.0, 1.0], "dtype": "float32", "shape": [1, 3]},
            },
        }

    def test_raw_lists_and_external_tensors_agree_on_the_wire(self):
        """The same values reach the wire identically whichever form they arrive in."""
        np = pytest.importorskip("numpy")
        rows = [[0.0, 1.0], [1.0, 0.0]]
        from_list = _serialize_loss_fn_inputs({"weights": rows})
        from_array = _serialize_loss_fn_inputs({"weights": np.array(rows, dtype=np.float32)})
        assert from_list == from_array


class TestEncodedTextChunk:
    def test_length(self):
        assert EncodedTextChunk(tokens=[1, 2, 3]).length == 3

    def test_from_any_passthrough(self):
        c = EncodedTextChunk(tokens=[1])
        assert EncodedTextChunk.from_any(c) is c

    def test_from_any_dict(self):
        assert EncodedTextChunk.from_any({"tokens": [1, 2]}).tokens == [1, 2]

    def test_from_any_list(self):
        assert EncodedTextChunk.from_any([3, 4]).tokens == [3, 4]

    def test_from_any_model_input_chunk_wrapper(self):
        wrapper = ModelInputChunk(chunk=EncodedTextChunk(tokens=[5]))
        assert EncodedTextChunk.from_any(wrapper).tokens == [5]

    def test_from_any_compatible(self):
        assert EncodedTextChunk.from_any(_CompatChunk([7, 8])).tokens == [7, 8]


class TestModelInput:
    def test_from_ints(self):
        mi = ModelInput.from_ints([1, 2, 3])
        assert mi._to_api() == {"chunks": [{"tokens": [1, 2, 3], "type": "encoded_text"}]}

    def test_empty(self):
        mi = ModelInput.empty()
        assert mi.length == 0
        assert mi._to_api() == {"chunks": []}

    def test_to_ints_flattens_chunks(self):
        mi = ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2]), EncodedTextChunk(tokens=[3])])
        assert mi.to_ints() == [1, 2, 3]

    def test_length(self):
        assert ModelInput.from_ints([1, 2, 3]).length == 3

    def test_append_returns_new_input(self):
        mi = ModelInput.from_ints([1])
        mi2 = mi.append([2, 3])
        assert mi.to_ints() == [1]
        assert mi2.to_ints() == [1, 2, 3]

    def test_append_int(self):
        assert ModelInput.from_ints([1]).append_int(9).to_ints() == [1, 9]

    def test_from_any_passthrough(self):
        mi = ModelInput.from_ints([1])
        assert ModelInput.from_any(mi) is mi

    def test_from_any_token_list(self):
        assert ModelInput.from_any([1, 2]).to_ints() == [1, 2]

    def test_from_any_wire_dict(self):
        mi = ModelInput.from_any({"chunks": [{"tokens": [1], "type": "encoded_text"}]})
        assert mi.to_ints() == [1]

    def test_from_any_compatible(self):
        compat = _CompatModelInput(chunks=[_CompatChunk([1, 2]), _CompatChunk([3])])
        assert ModelInput.from_any(compat).to_ints() == [1, 2, 3]

    def test_to_api_coerces_compatible_chunk(self):
        # A compatible chunk passed directly (bypassing from_any) must still be
        # serialized to the wire format rather than sent as-is.
        mi = ModelInput(chunks=[_CompatChunk([4, 5])])
        assert mi._to_api() == {"chunks": [{"tokens": [4, 5], "type": "encoded_text"}]}


class TestDatum:
    def test_to_api_with_model_input(self):
        d = Datum(model_input=ModelInput.from_ints([1, 2]))
        assert d._to_api() == {
            "modelInput": {"chunks": [{"tokens": [1, 2], "type": "encoded_text"}]}
        }

    def test_to_api_serializes_tensor_loss_inputs(self):
        d = Datum(
            model_input=ModelInput.from_ints([1]),
            loss_fn_inputs={"advantages": TensorData(data=[0.5])},
        )
        assert d._to_api()["lossFnInputs"]["advantages"] == {"data": [0.5], "dtype": "float32"}

    def test_from_any_passthrough(self):
        d = Datum(model_input=ModelInput.from_ints([1]))
        assert Datum.from_any(d) is d

    def test_from_any_model_input(self):
        d = Datum.from_any(ModelInput.from_ints([1, 2]))
        assert d.model_input.to_ints() == [1, 2]

    def test_from_any_wire_dict(self):
        d = Datum.from_any(
            {"modelInput": {"chunks": [{"tokens": [1]}]}, "lossFnInputs": {"w": [1.0]}}
        )
        assert d.model_input.to_ints() == [1]
        assert d.loss_fn_inputs == {"w": [1.0]}

    def test_from_any_compatible(self):
        compat = _CompatDatum(
            model_input=_CompatModelInput([_CompatChunk([1, 2])]),
            loss_fn_inputs={"advantages": _CompatTensor([0.1])},
        )
        d = Datum.from_any(compat)
        api = d._to_api()
        assert api["modelInput"]["chunks"][0]["tokens"] == [1, 2]
        assert api["lossFnInputs"]["advantages"] == {"data": [0.1], "dtype": "float32"}

    def test_from_any_empty_loss_inputs_omitted(self):
        compat = _CompatDatum(model_input=_CompatModelInput([_CompatChunk([1])]))
        assert "lossFnInputs" not in Datum.from_any(compat)._to_api()


class TestDatumFromAnyNeverSilentlyEmpty:
    """Every field on Datum is optional, so an unhandled value used to coerce to
    an empty Datum and serialize to ``{}``: a training example dropped from
    the batch with nothing raised.
    """

    def test_token_id_list_is_the_prompt_shorthand(self):
        assert Datum.from_any([4, 5, 6])._to_api() == {
            "modelInput": {"chunks": [{"tokens": [4, 5, 6], "type": "encoded_text"}]}
        }

    def test_text_is_the_prompt_shorthand(self):
        """Text is preserved for the client to tokenize, not serialized as-is.

        The runtime Datum has no text member, so serializing text raises here and
        the tokenizing happens at the client boundary instead.
        """
        d = Datum.from_any("hello")
        assert d._pending_text == "hello"
        with pytest.raises(ValueError, match="never tokenized"):
            d._to_api()

    def test_pending_text_tokenizes_at_the_boundary(self):
        class _Tok:
            def encode(self, text):
                return [11, 12]

        assert Datum.from_any("hello")._tokenized_with(_Tok())._to_api() == {
            "modelInput": {"chunks": [{"tokens": [11, 12], "type": "encoded_text"}]}
        }

    def test_tuple_of_token_ids(self):
        assert Datum.from_any((4, 5))._to_api() == {
            "modelInput": {"chunks": [{"tokens": [4, 5], "type": "encoded_text"}]}
        }

    def test_bare_chunk_is_wrapped(self):
        assert Datum.from_any(EncodedTextChunk(tokens=[7, 8]))._to_api() == {
            "modelInput": {"chunks": [{"tokens": [7, 8], "type": "encoded_text"}]}
        }

    def test_duck_typed_weight_is_rejected(self):
        """A scalar datum weight has no wire member, so it is refused not dropped.

        Accepted and ignored, the weighting the caller asked for was silently
        absent from training.
        """

        class TheirDatum:
            model_input = ModelInput.from_ints([1, 2])
            weight = 2.5

        with pytest.raises(ValueError, match="no 'weight' field"):
            Datum.from_any(TheirDatum())

    def test_duck_typed_prompt_is_honoured(self):
        class TheirDatum:
            prompt = "hi"

        assert Datum.from_any(TheirDatum())._pending_text == "hi"

    def test_unrelated_object_is_rejected(self):
        class Unrelated:
            foo = 1

        with pytest.raises(TypeError, match="none of"):
            Datum.from_any(Unrelated())

    def test_object_declaring_empty_fields_is_rejected(self):
        """Declaring the attributes is not the same as carrying values.

        A ``chunks`` attribute set to ``None`` must not read as a
        ModelInput-like object: that would wrap the object itself as a prompt
        and put ``{'prompt': <object>}`` on the wire instead of raising.
        """

        class AllNone:
            chunks = None
            model_input = None
            prompt = None
            loss_fn_inputs = None

        with pytest.raises(TypeError, match="none of"):
            Datum.from_any(AllNone())

    def test_object_with_populated_chunks_still_accepted(self):
        """The guard must not reject the ModelInput-like objects it exists for."""

        class TheirModelInput:
            chunks = [EncodedTextChunk(tokens=[1, 2])]

        assert Datum.from_any(TheirModelInput())._to_api() == {
            "modelInput": {"chunks": [{"tokens": [1, 2], "type": "encoded_text"}]}
        }

    def test_unrecognised_dict_is_rejected(self):
        with pytest.raises(TypeError, match="none of"):
            Datum.from_any({"foo": 1})

    def test_dict_weight_is_rejected(self):
        with pytest.raises(ValueError, match="no 'weight' field"):
            Datum.from_any({"modelInput": {"chunks": [{"tokens": [1]}]}, "weight": 2.0})

    def test_constructor_weight_is_rejected(self):
        with pytest.raises(ValueError, match="no 'weight' field"):
            Datum(model_input=ModelInput.from_ints([1]), weight=2.0)

    def test_prompt_only_dict(self):
        assert Datum.from_any({"prompt": "hi"})._pending_text == "hi"

    def test_loss_inputs_only_dict_is_allowed(self):
        """An explicit wire dict is the advanced escape hatch; only a dict with
        no recognised key at all is a mistake.
        """
        assert Datum.from_any({"lossFnInputs": {"w": [1.0]}})._to_api() == {
            "lossFnInputs": {"w": {"data": [1.0], "dtype": "float32", "shape": [1]}}
        }

    @pytest.mark.parametrize(
        "batch",
        [
            [Datum(model_input=ModelInput.from_ints([1, 2, 3])), [4, 5, 6]],
            [[4, 5, 6], Datum(model_input=ModelInput.from_ints([1, 2, 3]))],
            [
                {"modelInput": {"chunks": [{"tokens": [4, 5, 6]}]}},
                Datum(model_input=ModelInput.from_ints([1, 2, 3])),
            ],
        ],
        ids=["typed-first", "tokens-first", "dict-first"],
    )
    def test_mixed_batch_keeps_every_element(self, batch):
        """A batch mixing kinds is coerced element by element, in either order.

        The homogeneous fast paths read the whole batch through the one
        interpretation chosen for ``batch[0]``, which silently emptied later
        elements one way round and serialized a raw Datum object the other.
        """
        from sagemaker.train.training_session._utils import (
            batch_to_datum_list,
        )

        datums = batch_to_datum_list(batch)
        assert len(datums) == 2
        assert all(d.get("modelInput", {}).get("chunks") for d in datums)
        assert sorted(d["modelInput"]["chunks"][0]["tokens"] for d in datums) == [
            [1, 2, 3],
            [4, 5, 6],
        ]

    def test_mixed_batch_still_tokenizes_text(self):
        """Text is the one kind needing outside context, so it keeps the
        tokenizer path rather than passing through as a raw prompt.
        """
        from sagemaker.train.training_session._utils import (
            batch_to_datum_list,
        )

        class FakeTokenizer:
            def encode(self, text):
                return [len(text)]

        datums = batch_to_datum_list(
            [Datum(model_input=ModelInput.from_ints([1, 2])), "hello"],
            tokenizer=FakeTokenizer(),
        )
        assert datums[1]["modelInput"]["chunks"][0]["tokens"] == [5]


class TestFromAnyParams:
    def test_adam_params_passthrough(self):
        p = AdamParams(learning_rate=3e-4)
        assert AdamParams.from_any(p) is p

    def test_adam_params_from_dict(self):
        assert AdamParams.from_any({"learning_rate": 2e-4}).learning_rate == 2e-4

    def test_adam_params_compatible(self):
        class _CompatAdam:
            learning_rate = 5e-5
            beta1 = 0.8

        p = AdamParams.from_any(_CompatAdam())
        assert p.learning_rate == 5e-5
        assert p.beta1 == 0.8
        assert p.beta2 == 0.999  # default filled in

    def test_sampling_params_from_dict(self):
        assert SamplingParams.from_any({"max_tokens": 1024}).max_tokens == 1024

    def test_sampling_params_compatible(self):
        class _CompatSampling:
            max_tokens = 700
            temperature = 0.9

        p = SamplingParams.from_any(_CompatSampling())
        assert p.max_tokens == 700
        assert p.temperature == 0.9


class TestParamsUnsetFieldsUseOurDefaults:
    """An external params object leaves unset knobs as ``None``.

    Forwarding that ``None`` would fail validation on a field of ours that is
    required with a default, so "not set" has to mean "use the default": the
    same outcome as omitting the field.
    """

    def test_sampling_unset_max_tokens_falls_back(self):
        class _Unset:
            max_tokens = None
            temperature = 0.7

        p = SamplingParams.from_any(_Unset())
        assert p.max_tokens == 512  # our default, not a validation error
        assert p.temperature == 0.7

    def test_sampling_unset_max_tokens_in_dict(self):
        p = SamplingParams.from_any({"max_tokens": None, "temperature": 0.7})
        assert p.max_tokens == 512
        assert p.temperature == 0.7

    def test_adam_unset_fields_fall_back(self):
        class _Unset:
            learning_rate = 5e-5
            beta2 = None

        p = AdamParams.from_any(_Unset())
        assert p.learning_rate == 5e-5
        assert p.beta2 == 0.999  # ours, not None

    def test_explicit_sentinels_still_pass_through(self):
        """A deliberate value is honoured, including the service's own defaults."""

        class _Explicit:
            max_tokens = 512
            temperature = 1.0
            top_p = 1.0
            top_k = -1  # the service's documented "no limit"

        api = SamplingParams.from_any(_Explicit())._to_api()
        assert api == {
            "maxTokens": 512,
            "temperature": 1.0,
            "topP": 1.0,
            "topK": -1,
        }


class TestSamplingParamsStopCoercion:
    def test_bare_string_is_wrapped(self):
        assert SamplingParams(max_tokens=512, stop="END").stop == ["END"]

    def test_bare_string_via_from_any(self):
        p = SamplingParams.from_any({"max_tokens": 512, "stop": "END"})
        assert p._to_api()["stop"] == ["END"]

    def test_list_of_strings_unchanged(self):
        assert SamplingParams(max_tokens=512, stop=["a", "b"]).stop == ["a", "b"]

    def test_token_ids_rejected_with_reason(self):
        """The service matches stop sequences on decoded text, so IDs cannot work."""
        with pytest.raises(ValidationError, match="not token IDs"):
            SamplingParams(max_tokens=512, stop=[1, 2])


class _CountingTensor:
    """Tensor-like object that counts tolist() calls (torch/numpy stand-in)."""

    def __init__(self, values, dtype="float32"):
        self._values = values
        self.dtype = dtype
        self.shape = (len(values),)
        self.tolist_calls = 0

    def tolist(self):
        self.tolist_calls += 1
        return list(self._values)


class TestTensorDataLazyMaterialization:
    def test_from_any_defers_tolist(self):
        src = _CountingTensor([1.0, 2.0])
        td = TensorData.from_any(src)
        assert src.tolist_calls == 0
        assert td.shape == [2]
        assert td.dtype == TensorDtype.FLOAT32

    def test_data_access_materializes_once(self):
        src = _CountingTensor([1.0, 2.0])
        td = TensorData.from_any(src)
        assert td.data == [1.0, 2.0]
        assert td.data == [1.0, 2.0]
        assert src.tolist_calls == 1

    def test_to_api_materializes_lazily(self):
        src = _CountingTensor([1, 2, 3], dtype="int32")
        td = TensorData.from_any(src)
        assert src.tolist_calls == 0
        api = td._to_api()
        assert src.tolist_calls == 1
        assert api == {"data": [1, 2, 3], "dtype": "int32", "shape": [3]}

    def test_serialize_loss_fn_inputs_converts_once(self):
        src = _CountingTensor([0.1, 0.2])
        out = _serialize_loss_fn_inputs({"advantages": src})
        assert out["advantages"]["data"] == [0.1, 0.2]
        assert src.tolist_calls == 1

    def test_data_is_required(self):
        with pytest.raises(ValueError):
            TensorData(dtype=TensorDtype.FLOAT32)

    def test_extra_fields_still_rejected(self):
        with pytest.raises(ValidationError):
            TensorData(data=[1.0], not_a_field=1)


class TestSharedBaseConfigInheritance:
    """The shared bases must preserve each type's original ConfigDict behavior."""

    def test_adam_params_still_forbids_extras(self):
        with pytest.raises(ValidationError):
            AdamParams(learning_rte=1e-4)

    def test_adam_params_still_validates_assignment(self):
        p = AdamParams()
        with pytest.raises(ValidationError):
            p.learning_rate = "not-a-float"

    def test_sampling_params_still_forbids_extras(self):
        with pytest.raises(ValidationError):
            SamplingParams(max_token=100)

    def test_datum_still_allows_extras(self):
        d = Datum(prompt=[1, 2], custom_field="kept")
        assert d.custom_field == "kept"

    def test_datum_model_input_name_allowed(self):
        d = Datum(model_input=ModelInput.from_ints([1, 2]))
        assert d.model_input.to_ints() == [1, 2]


class TestStatusAndKindEnums:
    """The public enums must keep the service's raw wire values."""

    def test_session_status_values(self):
        assert SessionStatus.RUNNING.value == "Running"
        assert SessionStatus("Stopped") is SessionStatus.STOPPED
        assert {s.value for s in SessionStatus} == {
            "Created",
            "Scheduled",
            "Running",
            "Stopping",
            "Stopped",
            "Failed",
            "Blocked",
            "Expired",
            "Deleting",
            "DeletionFailed",
        }

    def test_model_package_kind_values(self):
        assert ModelPackageKind.CHECKPOINT.value == "checkpoint"
        assert ModelPackageKind.MODEL.value == "model"
        assert ModelPackageKind("model") is ModelPackageKind.MODEL

    def test_stop_reason_values(self):
        assert StopReason("length") is StopReason.LENGTH
        assert StopReason("stop") is StopReason.STOP

    def test_tensor_dtype_values(self):
        assert {d.value for d in TensorDtype} == {
            "float32",
            "float16",
            "bfloat16",
            "int32",
            "int64",
        }


class TestModelPackageInfo:
    def test_from_summary(self):
        info = ModelPackageInfo._from_summary(
            {
                "ModelPackageArn": "arn:aws:sagemaker:us-west-2:123:model-package/g/3",
                "ModelPackageVersion": 3,
                "ModelPackageDescription": "after epoch 2",
            }
        )
        assert info.model_package_arn == "arn:aws:sagemaker:us-west-2:123:model-package/g/3"
        assert info.version == 3
        assert info.description == "after epoch 2"
        # ListModelPackages summaries carry no kind — it stays unresolved.
        assert info.kind is None

    def test_kind_accepts_enum(self):
        info = ModelPackageInfo(model_package_arn="arn:x", kind=ModelPackageKind.CHECKPOINT)
        assert info.kind is ModelPackageKind.CHECKPOINT

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ModelPackageInfo(model_package_arn="arn:x", not_a_field=1)


class TestModelPackageDeletion:
    def test_defaults_describe_a_retained_location(self):
        """The default shape is the common one: package gone, objects kept."""
        result = ModelPackageDeletion(model_package_arn="arn:x", model_package_deleted=True)
        assert result.s3_uri is None
        assert result.objects_deleted == 0
        assert result.artifacts_retained is True
        assert result.retained_reason is None

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ModelPackageDeletion(
                model_package_arn="arn:x", model_package_deleted=True, not_a_field=1
            )


class TestTokenLogprob:
    def test_from_api(self):
        tl = TokenLogprob._from_api({"tokenId": 25, "logprob": -4.22})
        assert tl.token_id == 25
        assert tl.logprob == -4.22

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            TokenLogprob(token_id=1, logprob=-0.5, not_a_field=1)


class TestSampledSequence:
    def test_from_api(self):
        seq = SampledSequence._from_api(
            {"stopReason": "length", "tokens": [4, 5, 6], "logprobs": [-0.7, -0.5, -0.3]}
        )
        assert seq.stop_reason is StopReason.LENGTH
        assert seq.tokens == [4, 5, 6]
        assert seq.logprobs == [-0.7, -0.5, -0.3]

    def test_from_api_without_logprobs(self):
        seq = SampledSequence._from_api({"stopReason": "stop", "tokens": [1]})
        assert seq.logprobs is None

    def test_from_api_ignores_token_usage(self):
        """tokenUsage is on the wire but deliberately not surfaced.

        The service returns it per sequence; the SDK drops it rather than
        committing to public surface it would then have to keep.
        """
        seq = SampledSequence._from_api(
            {
                "stopReason": "stop",
                "tokens": [1, 2],
                "tokenUsage": {"prefill": 7, "decode": 2},
            }
        )
        assert not hasattr(seq, "token_usage")


class TestSampleResponseFromApi:
    def test_full_response(self):
        # Wire shape: topkPromptLogprobs is a list (per prompt position) of
        # TokenLogprob objects, converted server-side from [tokenId, logprob]
        # pairs.
        r = SampleResponse._from_api(
            {
                "sequences": [{"stopReason": "length", "tokens": [4, 5]}],
                "promptLogprobs": [-8.25, -11.30],
                "topkPromptLogprobs": [
                    [{"tokenId": 25, "logprob": -4.22}, {"tokenId": 654, "logprob": -4.47}]
                ],
            }
        )
        assert len(r.sequences) == 1
        assert r.sequences[0].tokens == [4, 5]
        assert r.prompt_logprobs == [-8.25, -11.30]
        assert r.topk_prompt_logprobs[0][0].token_id == 25
        assert r.topk_prompt_logprobs[0][1].logprob == -4.47

    def test_minimal_response(self):
        r = SampleResponse._from_api({"sequences": []})
        assert r.sequences == []
        assert r.prompt_logprobs is None
        assert r.topk_prompt_logprobs is None

    def test_stop_reason_is_parsed(self):
        r = SampleResponse._from_api(
            {"sequences": [{"stopReason": "stop", "tokens": [10, 20, 30]}]}
        )
        assert r.sequences[0].tokens == [10, 20, 30]
        assert r.sequences[0].stop_reason.value == "stop"

    def test_empty_response_no_field(self):
        """``sequences`` absent — returns empty list rather than raising."""
        r = SampleResponse._from_api({})
        assert r.sequences == []


class TestVpcConfig:
    def test_to_api_shape(self):
        vc = VpcConfig(subnets=["subnet-1", "subnet-2"], security_group_ids=["sg-1"])
        assert vc._to_api() == {
            "Subnets": ["subnet-1", "subnet-2"],
            "SecurityGroupIds": ["sg-1"],
        }

    def test_from_any_passthrough(self):
        vc = VpcConfig(subnets=["subnet-1"], security_group_ids=["sg-1"])
        assert VpcConfig.from_any(vc) is vc

    def test_from_any_dict(self):
        vc = VpcConfig.from_any({"subnets": ["subnet-1"], "security_group_ids": ["sg-1", "sg-2"]})
        assert vc.subnets == ["subnet-1"]
        assert vc.security_group_ids == ["sg-1", "sg-2"]

    def test_from_any_compatible(self):
        class _Compat:
            subnets = ("subnet-1",)
            security_group_ids = ("sg-1",)

        vc = VpcConfig.from_any(_Compat())
        assert vc.subnets == ["subnet-1"]
        assert vc.security_group_ids == ["sg-1"]

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            VpcConfig(subnets=["s"], security_group_ids=["g"], vpc_id="vpc-1")


class TestResponseTypeNaming:
    """The per-operation response-type names ARE the canonical public names."""

    def test_legacy_names_removed(self):
        import sagemaker.train.training_session as ts
        import sagemaker.train.training_session.types as types_mod

        for legacy in ("TrainingResult", "SampleResult", "ComputeLogprobsResult"):
            assert not hasattr(ts, legacy)
            assert not hasattr(types_mod, legacy)

    def test_response_types_exported_from_package_root(self):
        import sagemaker.train.training_session as ts

        assert ts.ForwardBackwardOutput is ForwardBackwardOutput
        assert ts.OptimStepResponse is OptimStepResponse
        assert ts.SaveStateResponse is SaveStateResponse
        assert ts.SaveWeightsResponse is SaveWeightsResponse
        assert ts.LoadStateResponse is LoadStateResponse
        assert ts.SampleResponse is SampleResponse

    def test_each_save_type_reads_only_its_own_operation_key(self):
        """The name of each response type is the operation it reads.

        Crossing the two keys must yield nothing rather than the other's ARN.
        """
        assert (
            SaveStateResponse._from_api(
                {"saveWeights": {"modelPackageArn": "arn:wrong"}}
            ).model_package_arn
            is None
        )
        assert (
            SaveWeightsResponse._from_api(
                {"saveState": {"modelPackageArn": "arn:wrong"}}
            ).model_package_arn
            is None
        )

    def test_retired_weights_era_names_are_gone(self):
        """The old names must not linger as aliases: one name per operation."""
        import sagemaker.train.training_session as ts
        import sagemaker.train.training_session.types as types_mod

        for retired in ("SaveWeightsForSamplerResponse", "LoadWeightsResponse"):
            assert not hasattr(ts, retired)
            assert not hasattr(types_mod, retired)


class TestModelInputFromAnyRejectsUnknownDicts:
    """An unrecognized dict must not become an input with no chunks.

    That shape is valid on the wire, so a dropped training example produced no
    error anywhere: the operation succeeded and simply trained on nothing.
    """

    def test_an_unrecognized_dict_raises(self):
        with pytest.raises(ValueError, match="cannot build a model input"):
            ModelInput.from_any({"totally": "unknown"})

    def test_the_message_names_the_keys_it_accepts(self):
        with pytest.raises(ValueError, match="chunks"):
            ModelInput.from_any({"totally": "unknown"})

    def test_a_bare_chunk_is_read_as_one_chunk(self):
        """``{"tokens": [...]}`` is a chunk, and used to silently become empty."""
        assert ModelInput.from_any({"tokens": [1, 2, 3]}).to_ints() == [1, 2, 3]

    def test_the_recognized_shapes_still_work(self):
        assert ModelInput.from_any({"chunks": [{"tokens": [4, 5]}]}).to_ints() == [4, 5]
        assert ModelInput.from_any({"prompt": "hello"}).prompt == "hello"


class TestTensorDataRoundTrip:
    """Values are held outside the model fields, but a dump must still carry them."""

    def test_model_dump_includes_the_data(self):
        tensor = TensorData([1.0, 2.0], dtype=TensorDtype.FLOAT32, shape=[2])

        assert tensor.model_dump()["data"] == [1.0, 2.0]

    def test_a_dump_round_trips(self):
        tensor = TensorData([1, 2, 3], dtype=TensorDtype.INT64, shape=[3])

        restored = TensorData.model_validate(tensor.model_dump())

        assert restored.data == [1, 2, 3]
        assert restored.dtype is TensorDtype.INT64
        assert restored.shape == [3]

    def test_a_tensor_like_source_is_not_materialized_by_construction(self):
        """The laziness the private storage exists for has to survive."""

        class Lazy:
            def __init__(self):
                self.reads = 0

            def tolist(self):
                self.reads += 1
                return [7, 8]

        source = Lazy()
        tensor = TensorData(source, dtype=TensorDtype.INT64, shape=[2])
        assert source.reads == 0

        assert tensor.data == [7, 8]
        assert source.reads == 1


class TestDatumRejectsBothInputs:
    """``model_input`` and ``prompt`` are documented as exclusive; enforce it.

    With both set the wire shape came from ``model_input`` and the prompt went
    nowhere, so a caller who meant to send it trained on something else.
    """

    def test_both_set_raises(self):
        with pytest.raises(ValidationError, match="not both"):
            Datum(model_input=ModelInput.from_ints([1, 2]), prompt=[3, 4])

    def test_either_alone_is_accepted(self):
        assert Datum(model_input=ModelInput.from_ints([1, 2])).prompt is None
        assert Datum(prompt=[3, 4]).model_input is None


class TestModelInputAppendRejectsPendingText:
    """Appending to a text-built input dropped the text and kept the chunk."""

    def test_appending_to_text_raises(self):
        with pytest.raises(ValueError, match="has not been tokenized"):
            ModelInput.from_text("hello").append(EncodedTextChunk(tokens=[1]))

    def test_appending_to_token_input_still_works(self):
        appended = ModelInput.from_ints([1, 2]).append(EncodedTextChunk(tokens=[3]))

        assert appended.to_ints() == [1, 2, 3]

    def test_appending_keeps_a_token_id_prompt(self):
        """A prompt of token IDs is content, not pending text, so it survives.

        Only text needs a tokenizer; token IDs are already the input's chunks as
        far as the wire is concerned, and dropping them loses training tokens
        with nothing raised.
        """
        appended = ModelInput(prompt=[1, 2, 3]).append(EncodedTextChunk(tokens=[9]))

        assert appended.to_ints() == [1, 2, 3, 9]
        assert appended._to_api() == {
            "chunks": [
                {"tokens": [1, 2, 3], "type": "encoded_text"},
                {"tokens": [9], "type": "encoded_text"},
            ]
        }


class TestDatumFromAnyAppliesTheSameExclusivity:
    """``from_any`` is the path every batch element takes, so it must agree.

    The constructor rejected both inputs while this path kept ``model_input``
    and dropped ``prompt``, which is the same silent loss one layer down.
    """

    def test_a_dict_naming_both_is_rejected(self):
        with pytest.raises(ValidationError, match="not both"):
            Datum.from_any({"model_input": {"chunks": [{"tokens": [1, 2]}]}, "prompt": "hi"})

    def test_an_empty_prompt_is_not_a_second_input(self):
        """``prompt=""`` carries nothing, so it is not a conflict to reject.

        Objects coerced through this path commonly default their prompt to an
        empty string, and failing those would break a call that loses nothing.
        """
        datum = Datum.from_any({"model_input": {"chunks": [{"tokens": [1, 2]}]}, "prompt": ""})

        assert datum.model_input is not None
        assert datum.model_input.to_ints() == [1, 2]

    def test_an_object_with_an_empty_prompt_and_model_input_is_accepted(self):
        class Carrier:
            model_input = ModelInput.from_ints([4, 5])
            prompt = ""
            loss_fn_inputs = None

        datum = Datum.from_any(Carrier())

        assert datum.model_input is not None
        assert datum.model_input.to_ints() == [4, 5]


class TestTensorDataFollowsTheHashProtocol:
    """Mutable values mean unhashable, and that has to be reported honestly.

    Supplying a ``__hash__`` that raises registers the type as hashable, so a
    caller branching on ``isinstance(x, Hashable)`` takes the hashable path and
    then crashes.
    """

    def test_it_reports_itself_unhashable(self):
        import collections.abc

        assert not isinstance(TensorData([1, 2]), collections.abc.Hashable)

    def test_hashing_it_raises_type_error(self):
        with pytest.raises(TypeError, match="unhashable"):
            hash(TensorData([1, 2]))


class TestTensorDataEqualityIgnoresAccessOrder:
    """Values are materialized lazily, and equality must not depend on that."""

    def test_a_tensor_like_source_equals_the_same_list(self):
        class Lazy:
            def tolist(self):
                return [1.0, 2.0]

        from_source = TensorData(Lazy(), dtype=TensorDtype.FLOAT32, shape=[2])
        from_list = TensorData([1.0, 2.0], dtype=TensorDtype.FLOAT32, shape=[2])

        assert from_source == from_list

    def test_different_values_are_still_unequal(self):
        assert TensorData([1, 2], dtype=TensorDtype.INT64, shape=[2]) != TensorData(
            [9, 9], dtype=TensorDtype.INT64, shape=[2]
        )

    def test_a_different_dtype_is_unequal(self):
        assert TensorData([1, 2], dtype=TensorDtype.INT64, shape=[2]) != TensorData(
            [1, 2], dtype=TensorDtype.INT32, shape=[2]
        )


class TestOldLogprobsIsPinnedLikeLogprobs:
    """Both logprob inputs are floats, so neither may be inferred from its values.

    ``old_logprobs`` was absent from the dtype table, so a batch whose values
    happened to be whole numbers went out as int64 while ``logprobs`` stayed
    float32 -- the same quantity described two ways in one request.
    """

    def test_whole_numbers_still_serialize_as_float32(self):
        datum = Datum(
            model_input=ModelInput.from_ints([1, 2, 3]),
            loss_fn_inputs={"old_logprobs": [0, -1, -2], "logprobs": [0, -1, -2]},
        )

        sent = datum._to_api()["lossFnInputs"]

        assert sent["old_logprobs"]["dtype"] == "float32"
        assert sent["logprobs"]["dtype"] == "float32"


class TestModelInputFromAnyRejectsBothInputs:
    """``from_any`` must answer chunks-vs-prompt the way the constructor does.

    It took ``chunks`` and dropped ``prompt``, so a dict carrying an empty chunk
    list and a real prompt built an empty input that the service accepts,
    dropping the example with nothing raised.
    """

    def test_both_keys_raises(self):
        with pytest.raises(ValueError, match="not both"):
            ModelInput.from_any({"chunks": [{"tokens": [1]}], "prompt": [2, 3]})

    def test_an_empty_chunk_list_with_a_prompt_also_raises(self):
        with pytest.raises(ValueError, match="not both"):
            ModelInput.from_any({"chunks": [], "prompt": [2, 3]})

    def test_a_dumped_chunk_input_still_round_trips(self):
        """``model_dump()`` carries ``prompt=None``, which is not a second input."""
        dumped = ModelInput.from_ints([1, 2]).model_dump()

        assert ModelInput.from_any(dumped).to_ints() == [1, 2]

    def test_either_key_alone_is_accepted(self):
        assert ModelInput.from_any({"chunks": [{"tokens": [1]}]}).to_ints() == [1]
        assert ModelInput.from_any({"prompt": [2, 3]}).to_ints() == [2, 3]

    def test_a_text_built_dump_still_round_trips(self):
        """``from_text`` dumps carry ``chunks=None``, which is not a second input.

        A dump names every field, so the exclusivity rule has to read the values
        rather than the keys.
        """
        dumped = ModelInput.from_text("hello").model_dump()

        assert dumped["chunks"] is None
        assert ModelInput.from_any(dumped)._pending_text == "hello"

    def test_a_null_chunks_key_alongside_a_prompt_is_accepted(self):
        assert ModelInput.from_any({"chunks": None, "prompt": [1, 2]}).to_ints() == [1, 2]


class TestTextInputHasNoTokenIdsYet:
    """``to_ints``/``length`` reported a full prompt as empty.

    ``_to_api`` already refuses a text-built input; these two returned ``[]`` and
    ``0``, so a token budget counted the prompt as absent instead of failing.
    """

    def test_to_ints_raises(self):
        with pytest.raises(ValueError, match="has not been tokenized"):
            ModelInput.from_text("hello world").to_ints()

    def test_length_raises(self):
        with pytest.raises(ValueError, match="has not been tokenized"):
            _ = ModelInput.from_text("hello world").length

    def test_a_prompt_that_is_not_token_ids_also_raises(self):
        """``_to_api`` already refuses these, so reading them as empty disagreed.

        Any prompt that is not a token ID list has none to return, not zero of
        them, and a token budget counted the difference.
        """
        with pytest.raises(ValueError, match="not token IDs"):
            ModelInput(prompt=object()).to_ints()

    def test_a_token_input_is_unaffected(self):
        assert ModelInput.from_ints([1, 2, 3]).to_ints() == [1, 2, 3]
        assert ModelInput.from_ints([1, 2, 3]).length == 3

    def test_an_empty_input_is_still_empty_rather_than_an_error(self):
        assert ModelInput.empty().to_ints() == []
        assert ModelInput.empty().length == 0
