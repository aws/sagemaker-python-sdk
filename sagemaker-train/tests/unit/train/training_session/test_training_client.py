# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
"""Unit tests for sagemaker.train.training_session.training_client.

The runtime client is faked; no test performs a real sleep or network call.
Here we assert request shaping, sequence numbering, and error propagation.
"""

from __future__ import annotations

import asyncio

import pytest
from botocore.exceptions import ClientError

from sagemaker.train.training_session.training_client import TrainingClient
from sagemaker.train.training_session.types import (
    AdamParams,
    ForwardBackwardOutput,
    LoadStateResponse,
    ModelPackageKind,
    OptimStepResponse,
    SaveWeightsResponse,
    SaveStateResponse,
)


def _client_error(code, *, status=400, message="boom", op="ExecuteTrainingOperation"):
    """A botocore error shaped the way the runtime client raises one."""
    return ClientError(
        {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        op,
    )


class FakeRuntime:
    """Records execute_training_operation calls and replays canned responses.

    Stands in for the botocore runtime client the training client calls
    directly, so ``execute_error`` is a botocore error, exactly what the wire
    raises.
    """

    def __init__(self, *, execute_error: Exception | None = None):
        self.execute_calls: list[dict] = []
        self.get_calls: list[dict] = []
        self.sample_execute_calls: list[dict] = []
        self.sample_get_calls: list[dict] = []
        self._execute_error = execute_error
        self._op_counter = 0

    def execute_training_operation(self, **kwargs):
        self.execute_calls.append(kwargs)
        if self._execute_error is not None:
            raise self._execute_error
        self._op_counter += 1
        return {"operationId": f"op-{self._op_counter}"}

    def get_training_operation_result(self, **kwargs):
        self.get_calls.append(kwargs)
        return {
            "status": "Completed",
            "operationResult": {"optimizerStep": {"metrics": {"loss": 0.25}}},
        }

    def execute_sampling_operation(self, **kwargs):
        self.sample_execute_calls.append(kwargs)
        self._op_counter += 1
        return {"operationId": f"sample-op-{self._op_counter}"}

    def get_sampling_operation_result(self, **kwargs):
        self.sample_get_calls.append(kwargs)
        return {
            "status": "Completed",
            "operationResult": {},
        }


def _client(**kwargs) -> tuple[TrainingClient, FakeRuntime]:
    rt = FakeRuntime(**kwargs)
    return TrainingClient("sess-1", rt), rt


def _ce(length, loss_fn="cross_entropy"):
    """The loss inputs ``loss_fn`` requires, sized for datums of ``length`` tokens.

    An operation whose datums lack them is refused, so a test about chunking or
    sequencing still has to carry them.
    """
    inputs = {
        "target_tokens": list(range(length)),
        "weights": [1.0] * length,
    }
    if loss_fn in ("ppo", "cispo", "importance_sampling"):
        inputs["logprobs"] = [-0.1] * length
        inputs["advantages"] = [1.0] * length
    return inputs


# --- request shaping --------------------------------------------------------


def test_forward_backward_shapes_request_from_token_ids():
    tc, rt = _client()
    tc.forward_backward([[1, 2, 3]], loss_fn="cross_entropy", loss_fn_inputs=_ce(3)).result()
    call = rt.execute_calls[0]
    assert call["trainingSessionName"] == "sess-1"
    fb = call["operationConfig"]["forwardBackward"]
    assert fb["lossFnName"] == "cross_entropy"
    assert fb["datumList"][0]["modelInput"]["chunks"][0]["tokens"] == [1, 2, 3]


def test_forward_backward_accepts_enum_and_attaches_loss_fn_inputs():
    from sagemaker.train.training_session.types import LossFunction

    tc, rt = _client()
    # The full set ppo reads, each per-token entry as long as the 2-token
    # datum: a partial set is refused, so a plumbing check passes a whole one.
    inputs = {
        "target_tokens": {"data": [6, 0], "dtype": "int64", "shape": [2]},
        "weights": {"data": [0.0, 1.0], "dtype": "float32", "shape": [2]},
        "logprobs": {"data": [-0.1, -0.2], "dtype": "float32", "shape": [2]},
        "advantages": {"data": [1.0, 1.0], "dtype": "float32", "shape": [2]},
    }
    tc.forward_backward([[5, 6]], loss_fn=LossFunction.PPO, loss_fn_inputs=inputs).result()
    fb = rt.execute_calls[0]["operationConfig"]["forwardBackward"]
    assert fb["lossFnName"] == "ppo"
    assert fb["datumList"][0]["lossFnInputs"] == inputs


def test_optim_step_defaults_and_explicit_params():
    tc, rt = _client()
    tc.optim_step().result()
    cfg = rt.execute_calls[0]["operationConfig"]["optimizerStep"]["adamParamsConfig"]
    assert cfg["learningRate"] == AdamParams().learning_rate

    tc.optim_step(AdamParams(learning_rate=3e-4, grad_clip_norm=1.0)).result()
    cfg2 = rt.execute_calls[1]["operationConfig"]["optimizerStep"]["adamParamsConfig"]
    assert cfg2["learningRate"] == 3e-4
    assert cfg2["gradClipNorm"] == 1.0


def test_optim_step_lora_plus_lr_ratio():
    tc, rt = _client()
    tc.optim_step(AdamParams(learning_rate=1e-4, lora_plus_lr_ratio=16.0)).result()
    cfg = rt.execute_calls[0]["operationConfig"]["optimizerStep"]["adamParamsConfig"]
    assert cfg["loraPlusLrRatio"] == 16.0
    assert cfg["learningRate"] == 1e-4


def test_optim_step_omits_lora_plus_lr_ratio_when_none():
    tc, rt = _client()
    tc.optim_step(AdamParams(learning_rate=1e-4)).result()
    cfg = rt.execute_calls[0]["operationConfig"]["optimizerStep"]["adamParamsConfig"]
    assert "loraPlusLrRatio" not in cfg


def test_save_weights_for_sampler_save_state_load_state_shapes():
    tc, rt = _client()
    tc.save_weights_for_sampler().result()
    assert rt.execute_calls[0]["operationConfig"]["saveWeights"] == {}

    tc.save_state().result()
    assert rt.execute_calls[1]["operationConfig"]["saveState"] == {}

    tc.load_state("arn:aws:sagemaker:::model-package/x").result()
    ls = rt.execute_calls[2]["operationConfig"]["loadState"]
    assert ls == {
        "modelPackageArn": "arn:aws:sagemaker:::model-package/x",
        "loadOptimizer": False,
    }


def test_save_state_sends_description_when_provided():
    tc, rt = _client()
    tc.save_state(description="step 100 checkpoint").result()
    assert rt.execute_calls[0]["operationConfig"]["saveState"] == {
        "description": "step 100 checkpoint"
    }


def test_save_state_omits_description_by_default():
    tc, rt = _client()
    tc.save_state().result()
    assert rt.execute_calls[0]["operationConfig"]["saveState"] == {}


def test_save_state_async_sends_description():
    tc, rt = _client()
    asyncio.run(tc.save_state_async(description="async save at step 50")).result()
    assert rt.execute_calls[0]["operationConfig"]["saveState"] == {
        "description": "async save at step 50"
    }


def test_save_weights_for_sampler_sends_description_when_provided():
    tc, rt = _client()
    tc.save_weights_for_sampler(description="weights at step 200").result()
    assert rt.execute_calls[0]["operationConfig"]["saveWeights"] == {
        "description": "weights at step 200",
    }


def test_save_weights_for_sampler_omits_description_by_default():
    tc, rt = _client()
    tc.save_weights_for_sampler().result()
    assert rt.execute_calls[0]["operationConfig"]["saveWeights"] == {}


def test_save_weights_for_sampler_async_sends_description():
    tc, rt = _client()
    asyncio.run(tc.save_weights_for_sampler_async(description="async weights step 50")).result()
    assert rt.execute_calls[0]["operationConfig"]["saveWeights"] == {
        "description": "async weights step 50",
    }


def test_save_weights_for_sampler_omits_merge_flag_by_default():
    tc, rt = _client()
    tc.save_weights_for_sampler().result()
    assert rt.execute_calls[0]["operationConfig"]["saveWeights"] == {}


def test_save_weights_for_sampler_sends_merge_flag_when_requested():
    tc, rt = _client()
    tc.save_weights_for_sampler(merge_weights=True).result()
    assert rt.execute_calls[0]["operationConfig"]["saveWeights"] == {
        "mergeWeights": True,
    }
    asyncio.run(tc.save_weights_for_sampler_async(merge_weights=True)).result()
    assert rt.execute_calls[1]["operationConfig"]["saveWeights"] == {
        "mergeWeights": True,
    }


# --- forward-backward chunking ----------------------------------------------


def _token_datum(n_tokens: int):
    return {"modelInput": {"chunks": [{"tokens": list(range(n_tokens)), "type": "encoded_text"}]}}


def test_forward_backward_single_chunk_sends_one_op_unchanged():
    # A batch that fits in one chunk submits exactly one op, unchanged.
    tc, rt = _client()
    tc.forward_backward(
        [[1, 2, 3], [4, 5, 6]], loss_fn="cross_entropy", loss_fn_inputs=_ce(3)
    ).result()
    assert len(rt.execute_calls) == 1
    fb = rt.execute_calls[0]["operationConfig"]["forwardBackward"]
    assert fb["lossFnName"] == "cross_entropy"
    assert len(fb["datumList"]) == 2


def test_forward_backward_splits_batch_over_count_limit():
    # More datums than the chunk-length cap -> multiple ops, order preserved,
    # every datum sent exactly once.
    from sagemaker.train.training_session import training_client as tc_mod

    tc, rt = _client()
    original = tc_mod.chunk_datum_list

    # Force a tiny count cap so a small batch chunks deterministically.
    def tc_mod_chunk(dl, **kw):
        return original(dl, max_len=2, max_bytes=10**9)

    try:
        tc_mod.chunk_datum_list = tc_mod_chunk
        tc.forward_backward(
            [[1], [2], [3], [4], [5]], loss_fn="ppo", loss_fn_inputs=_ce(1, "ppo")
        ).result()
    finally:
        tc_mod.chunk_datum_list = original

    assert len(rt.execute_calls) == 3  # 2 + 2 + 1
    sent = [
        d["modelInput"]["chunks"][0]["tokens"][0]
        for call in rt.execute_calls
        for d in call["operationConfig"]["forwardBackward"]["datumList"]
    ]
    assert sent == [1, 2, 3, 4, 5]  # order preserved, no datum dropped/duplicated
    # Sequence numbers are contiguous across the chunk ops, consecutive with
    # no gap. The service accumulates gradients over these ops and blocks on any
    # missing sequence number, so contiguity (not merely strict increase) is the
    # property that keeps a chunked batch equivalent to one un-chunked op.
    seqs = [c["sequenceNumber"] for c in rt.execute_calls]
    assert seqs == list(range(seqs[0], seqs[0] + len(seqs)))


def test_forward_splits_batch_over_count_limit():
    # forward chunks too. It carries the same batch as forward_backward, so a
    # batch one of them can send and the other cannot is a surprise: the
    # forward-only pass failed on a size the training pass handled.
    from sagemaker.train.training_session import training_client as tc_mod

    tc, rt = _client()
    original = tc_mod.chunk_datum_list
    try:
        tc_mod.chunk_datum_list = lambda dl, **kw: original(dl, max_len=2, max_bytes=10**9)
        tc.forward([[1], [2], [3], [4], [5]], loss_fn="ppo", loss_fn_inputs=_ce(1, "ppo")).result()
    finally:
        tc_mod.chunk_datum_list = original

    assert len(rt.execute_calls) == 3  # 2 + 2 + 1
    sent = [
        d["modelInput"]["chunks"][0]["tokens"][0]
        for call in rt.execute_calls
        for d in call["operationConfig"]["forward"]["datumList"]
    ]
    assert sent == [1, 2, 3, 4, 5]  # order preserved, no datum dropped/duplicated


def test_forward_single_chunk_sends_one_op_unchanged():
    tc, rt = _client()
    tc.forward([[1, 2, 3], [4, 5, 6]], loss_fn="cross_entropy", loss_fn_inputs=_ce(3)).result()
    assert len(rt.execute_calls) == 1
    fwd = rt.execute_calls[0]["operationConfig"]["forward"]
    assert fwd["lossFnName"] == "cross_entropy"
    assert len(fwd["datumList"]) == 2


def test_small_batch_sends_single_op():
    # A batch comfortably under the module chunk bounds is sent as exactly one op.
    rt = FakeRuntime()
    tc = TrainingClient("sess-1", rt)
    tc.forward_backward(
        [[1, 2, 3], [4, 5, 6]], loss_fn="cross_entropy", loss_fn_inputs=_ce(3)
    ).result()
    assert len(rt.execute_calls) == 1


def test_chunk_datum_list_bounds():
    from sagemaker.train.training_session._utils import chunk_datum_list

    # Empty -> no chunks.
    assert chunk_datum_list([]) == []
    # Count bound.
    data = [_token_datum(1) for _ in range(5)]
    assert [len(c) for c in chunk_datum_list(data, max_len=2, max_bytes=10**9)] == [2, 2, 1]
    # Byte bound: each datum ~ 10 tokens * 10 = ~100 bytes; cap at 250 -> 2 per chunk.
    big = [_token_datum(10) for _ in range(5)]
    assert [len(c) for c in chunk_datum_list(big, max_len=1000, max_bytes=250)] == [2, 2, 1]
    # A single oversized datum still gets its own chunk (never dropped).
    huge = [_token_datum(1000)]
    assert chunk_datum_list(huge, max_len=1000, max_bytes=1) == [huge]


def test_combine_forward_backward_results_weights_and_concatenates():
    from sagemaker.train.training_session._utils import (
        combine_forward_backward_results,
    )

    # Single result passes through unchanged.
    solo = ForwardBackwardOutput._from_api({"forwardBackward": {"metrics": {"loss": 0.5}}})
    assert combine_forward_backward_results([solo]) is solo

    # Chunks carrying no loss outputs still each count once, so the weighted
    # mean does not collapse to the last chunk.
    r1 = ForwardBackwardOutput._from_api(
        {"forwardBackward": {"metrics": {"loss": 1.0, "grad_norm": 9.9}}}
    )
    r2 = ForwardBackwardOutput._from_api(
        {"forwardBackward": {"metrics": {"loss": 4.0, "grad_norm": 2.0}}}
    )
    combined = combine_forward_backward_results([r1, r2])
    assert abs(combined.metrics["loss"] - 2.5) < 1e-9  # (1.0*1 + 4.0*1)/2
    assert combined.metrics["grad_norm"] == 2.0  # last chunk (accumulated grad)
    assert combined.loss_fn_outputs is None


def test_combine_forward_backward_results_concatenates_per_datum_entries():
    """Per-datum entries concatenate in chunk order, weighting by entry count."""
    from sagemaker.train.training_session._utils import (
        combine_forward_backward_results,
    )

    r1 = ForwardBackwardOutput._from_api(
        {
            "forwardBackward": {
                "metrics": {"loss": 1.0, "grad_norm": 9.9},
                "lossFnOutputs": [{"loss": [0.1]}, {"loss": [0.2]}],
            }
        }
    )
    r2 = ForwardBackwardOutput._from_api(
        {
            "forwardBackward": {
                "metrics": {"loss": 4.0, "grad_norm": 2.0},
                "lossFnOutputs": [{"loss": [0.3]}],
            }
        }
    )
    combined = combine_forward_backward_results([r1, r2])
    # Datum counts come from the entry counts: 2 and 1 -> (1.0*2 + 4.0*1)/3
    assert abs(combined.metrics["loss"] - 2.0) < 1e-9
    assert combined.metrics["grad_norm"] == 2.0
    assert combined.loss_fn_outputs == [{"loss": [0.1]}, {"loss": [0.2]}, {"loss": [0.3]}]


# --- base returns future / async twins are coroutines ------------------------


def test_base_ops_return_future_resolving_to_parsed_result():
    tc, _ = _client()
    # Base methods return an unresolved future immediately; .result() (or
    # await) yields the parsed per-op result.
    assert isinstance(
        tc.forward_backward([[1]], loss_fn_inputs=_ce(1)).result(), ForwardBackwardOutput
    )
    assert isinstance(tc.optim_step().result(), OptimStepResponse)
    assert isinstance(tc.save_state().result(), SaveStateResponse)
    assert isinstance(tc.forward([[1]], loss_fn_inputs=_ce(1)).result(), ForwardBackwardOutput)


def test_async_ops_are_coroutines_yielding_futures():
    tc, _ = _client()
    # *_async methods are coroutines: awaiting one completes the enqueue and
    # yields the same APIFuture the base method returns.
    for coro_fn in (
        tc.forward_backward_async,
        tc.optim_step_async,
        tc.save_state_async,
        tc.forward_async,
        tc.save_weights_for_sampler_async,
        tc.load_state_async,
    ):
        assert asyncio.iscoroutinefunction(coro_fn)
    for coro, expected_cls in (
        (tc.forward_backward_async([[1]], loss_fn_inputs=_ce(1)), ForwardBackwardOutput),
        (tc.optim_step_async(), OptimStepResponse),
        (tc.save_state_async(), SaveStateResponse),
        (tc.forward_async([[1]], loss_fn_inputs=_ce(1)), ForwardBackwardOutput),
        (tc.save_weights_for_sampler_async(), SaveWeightsResponse),
        (tc.load_state_async("arn:mp/1"), LoadStateResponse),
    ):
        fut = asyncio.run(coro)
        assert hasattr(fut, "result")
        assert isinstance(fut.result(), expected_cls)
    # The internal LWFS helper stays sync and returns its future directly.
    fut = tc._submit_load_weights_for_sampling("arn:mp/1")
    assert isinstance(fut.result(), LoadStateResponse)


def test_load_state_with_optimizer_sets_flag():
    tc, rt = _client()
    tc.load_state_with_optimizer("arn:mp/7").result()
    cfg = rt.execute_calls[0]["operationConfig"]["loadState"]
    assert cfg["loadOptimizer"] is True
    assert cfg["modelPackageArn"] == "arn:mp/7"


def test_load_state_is_weights_only():
    """Bare load_state must NOT restore the optimizer.

    The negative twin of the test above. Without it the two methods can
    silently collapse into the same call, which is what happened when
    load_state defaulted load_optimizer=True: load_state_with_optimizer
    became unreachable and a caller expecting weights-only resume got the
    optimizer restored. Engine-side and client-side both split this by
    method name, not by a flag.
    """
    tc, rt = _client()
    tc.load_state("arn:mp/7").result()
    cfg = rt.execute_calls[0]["operationConfig"]["loadState"]
    assert cfg["loadOptimizer"] is False
    assert cfg["modelPackageArn"] == "arn:mp/7"


def test_load_state_exposes_no_load_optimizer_knob():
    """Neither load method takes a load_optimizer parameter.

    ``load_state(path)`` / ``load_state_with_optimizer(path)`` carry the
    choice in the method name. A flag re-introduces the ambiguity this pair
    exists to remove.
    """
    import inspect

    for fn in (
        TrainingClient.load_state,
        TrainingClient.load_state_async,
        TrainingClient.load_state_with_optimizer,
        TrainingClient.load_state_with_optimizer_async,
    ):
        assert "load_optimizer" not in inspect.signature(fn).parameters, fn.__name__


def test_load_weights_for_sampling_uses_sample_endpoint():
    tc, rt = _client()
    tc._load_weights_for_sampling("arn:mp/42")
    assert len(rt.sample_execute_calls) == 1
    call = rt.sample_execute_calls[0]
    assert call["operationConfig"] == {"loadWeightsForSampling": {"modelPackageArn": "arn:mp/42"}}
    assert call["trainingSessionName"] == "sess-1"
    assert len(rt.execute_calls) == 0, "LWFS must NOT go through training path"


def test_load_weights_for_sampling_does_not_consume_sequence():
    tc, rt = _client()
    tc.forward_backward([[1]], loss_fn_inputs=_ce(1)).result()
    seq_before = tc.current_sequence_number
    tc._load_weights_for_sampling("arn:mp/5")
    assert (
        tc.current_sequence_number == seq_before
    ), "LWFS must not advance the training sequence counter"
    tc.optim_step().result()
    seqs = [c["sequenceNumber"] for c in rt.execute_calls]
    assert seqs == [1, 2], "Training ops should be 1,2 (no gap from LWFS)"


# --- sequence numbering -----------------------------------------------------


def test_sequence_numbers_are_monotonic_and_start_at_one():
    tc, rt = _client()
    tc.forward_backward([[1]], loss_fn_inputs=_ce(1)).result()
    tc.optim_step().result()
    tc.save_state().result()
    seqs = [c["sequenceNumber"] for c in rt.execute_calls]
    assert seqs == [1, 2, 3]


def test_each_submit_uses_unique_client_token():
    tc, rt = _client()
    tc.optim_step().result()
    tc.optim_step().result()
    tokens = {c["clientToken"] for c in rt.execute_calls}
    assert len(tokens) == 2


# --- error propagation ------------------------------------------------------
# The service's own error reaches the caller unchanged: the client neither
# renames it nor hides the code and message the service sent. Submits are queued
# on a background thread, so a submit error surfaces when the returned future is
# resolved; tests resolve via .result().


@pytest.mark.parametrize(
    "code",
    ["ConflictException", "ResourceNotFoundException", "ValidationException"],
)
def test_service_error_propagates_unchanged(code):
    tc, _ = _client(execute_error=_client_error(code, message="from the service"))
    with pytest.raises(ClientError) as ei:
        tc.optim_step().result()
    assert ei.value.response["Error"]["Code"] == code
    assert "from the service" in str(ei.value)


# --- runtime-warmup transient retry on submit --------------------------------
# The control plane reports Running before the training runtime endpoint accepts
# traffic, so the first submit can 503 at the runtime front door. The submit is
# idempotent (stable clientToken + sequenceNumber), so it retries through the
# warmup window. Sleeps are patched out so the test stays sub-second.


class _FlakyRuntime(FakeRuntime):
    """Raises a retryable botocore error ``fail_n`` times, then succeeds."""

    def __init__(self, *, fail_n: int):
        super().__init__()
        self._fail_n = fail_n
        self._attempts = 0

    def execute_training_operation(self, **kwargs):
        self._attempts += 1
        if self._attempts <= self._fail_n:
            raise _client_error("InternalServerException", status=503, message="warming up")
        return super().execute_training_operation(**kwargs)


def _patch_retry_no_sleep(monkeypatch):
    """Make _submit's transient retry sleep instantly (no real waits in tests).

    ``call_with_transient_retry`` captures ``time.sleep`` as a default arg at
    import time, so patching the time module doesn't reach it — instead replace
    the symbol the client calls with a no-sleep partial of the real function.
    """
    import functools
    import sagemaker.train.training_session.training_client as tc_mod
    from sagemaker.train.training_session.futures import call_with_transient_retry

    no_sleep = functools.partial(call_with_transient_retry, sleep_fn=lambda _s: None)
    monkeypatch.setattr(tc_mod, "call_with_transient_retry", no_sleep)


def test_submit_retries_transient_then_succeeds(monkeypatch):
    _patch_retry_no_sleep(monkeypatch)
    rt = _FlakyRuntime(fail_n=8)  # > the old 5-attempt budget, < the 30 budget
    tc = TrainingClient("sess-1", rt)

    fut = tc.optim_step()  # must not raise — rides through the warmup 503s

    assert fut.operation_id == "op-1"  # blocks until the submit lands
    assert rt._attempts == 9  # 8 failures + 1 success


def test_submit_gives_up_when_transient_never_clears(monkeypatch):
    from sagemaker.train.training_session.training_client import (
        _TRAINING_SUBMIT_MAX_ATTEMPTS,
    )

    _patch_retry_no_sleep(monkeypatch)
    rt = _FlakyRuntime(fail_n=10_000)  # never clears
    tc = TrainingClient("sess-1", rt)

    with pytest.raises(ClientError):
        tc.optim_step().result()
    assert rt._attempts == _TRAINING_SUBMIT_MAX_ATTEMPTS


# --- result wiring ----------------------------------------------------------


def test_future_result_parses_training_result():
    tc, rt = _client()
    fut = tc.optim_step()
    result = fut.result()
    assert isinstance(result, OptimStepResponse)
    assert result.metrics == {"loss": 0.25}
    # poll hit the right operation id
    assert rt.get_calls[0]["operationId"] == fut.operation_id


# --- forward (forward-only op) ----------------------------------------------


def test_forward_shapes_request_from_token_ids():
    tc, rt = _client()
    tc.forward([[1, 2, 3]], loss_fn="cross_entropy", loss_fn_inputs=_ce(3)).result()
    call = rt.execute_calls[0]
    assert call["trainingSessionName"] == "sess-1"
    fwd = call["operationConfig"]["forward"]
    assert fwd["lossFnName"] == "cross_entropy"
    assert fwd["datumList"][0]["modelInput"]["chunks"][0]["tokens"] == [1, 2, 3]
    # forward-only must NOT emit a forwardBackward config.
    assert "forwardBackward" not in call["operationConfig"]


def test_forward_accepts_enum_and_attaches_loss_fn_inputs():
    from sagemaker.train.training_session.types import LossFunction

    tc, rt = _client()
    # The full set ppo reads, each per-token entry as long as the 2-token
    # datum: a partial set is refused, so a plumbing check passes a whole one.
    inputs = {
        "target_tokens": {"data": [6, 0], "dtype": "int64", "shape": [2]},
        "weights": {"data": [0.0, 1.0], "dtype": "float32", "shape": [2]},
        "logprobs": {"data": [-0.1, -0.2], "dtype": "float32", "shape": [2]},
        "advantages": {"data": [1.0, 1.0], "dtype": "float32", "shape": [2]},
    }
    tc.forward([[5, 6]], loss_fn=LossFunction.PPO, loss_fn_inputs=inputs).result()
    fwd = rt.execute_calls[0]["operationConfig"]["forward"]
    assert fwd["lossFnName"] == "ppo"
    assert fwd["datumList"][0]["lossFnInputs"] == inputs


# --- loss_fn_config -----------------------------------------------------------


def test_forward_backward_sends_loss_fn_config():
    tc, rt = _client()
    cfg = {"clip_low_threshold": 0.2, "clip_high_threshold": 5.0}
    tc.forward_backward(
        [[1, 2]], loss_fn="cispo", loss_fn_config=cfg, loss_fn_inputs=_ce(2, "cispo")
    ).result()
    fb = rt.execute_calls[0]["operationConfig"]["forwardBackward"]
    assert fb["lossFnConfig"] == cfg


def test_forward_backward_omits_loss_fn_config_when_none():
    tc, rt = _client()
    tc.forward_backward([[1, 2]], loss_fn_inputs=_ce(2)).result()
    fb = rt.execute_calls[0]["operationConfig"]["forwardBackward"]
    assert "lossFnConfig" not in fb


def test_forward_sends_loss_fn_config():
    tc, rt = _client()
    cfg = {"clip_low_threshold": 0.1}
    tc.forward([[1, 2]], loss_fn="ppo", loss_fn_config=cfg, loss_fn_inputs=_ce(2, "ppo")).result()
    fwd = rt.execute_calls[0]["operationConfig"]["forward"]
    assert fwd["lossFnConfig"] == cfg


def test_forward_omits_loss_fn_config_when_none():
    tc, rt = _client()
    tc.forward([[1, 2]], loss_fn_inputs=_ce(2)).result()
    fwd = rt.execute_calls[0]["operationConfig"]["forward"]
    assert "lossFnConfig" not in fwd


def test_forward_backward_tokenizes_a_model_input_built_from_text():
    """A typed batch element carrying text is tokenized like a bare string."""
    from sagemaker.train.training_session.types import Datum, ModelInput

    class _Tok:
        def encode(self, text):
            return [4, 5]

    rt = FakeRuntime()
    tc = TrainingClient("sess-1", rt, tokenizer=_Tok())
    tc.forward_backward(
        [Datum(model_input=ModelInput.from_text("hi"))], loss_fn_inputs=_ce(2)
    ).result()
    datum = rt.execute_calls[0]["operationConfig"]["forwardBackward"]["datumList"][0]
    assert datum["modelInput"] == {"chunks": [{"tokens": [4, 5], "type": "encoded_text"}]}


# --- get_tokenizer -----------------------------------------------------------


def test_get_tokenizer_lazy_loads_and_caches(monkeypatch):
    import sagemaker.train.training_session._utils as utils

    calls = []
    sentinel = object()

    def fake_load(base_model, hub_client, hf_model_id=None):
        calls.append(base_model)
        return sentinel

    monkeypatch.setattr(utils, "load_tokenizer", fake_load)
    tc = TrainingClient("sess-1", FakeRuntime(), base_model="Qwen/Qwen3-32B")
    tok1 = tc.get_tokenizer()
    tok2 = tc.get_tokenizer()
    assert tok1 is sentinel and tok2 is sentinel
    # Loaded once for the right base model, then cached.
    assert calls == ["Qwen/Qwen3-32B"]


def test_get_tokenizer_unsupported_model_raises(monkeypatch):
    import sagemaker.train.training_session._utils as utils

    def fake_load(base_model, hub_client, hf_model_id=None):
        raise ValueError("No public HuggingFace tokenizer is available")

    monkeypatch.setattr(utils, "load_tokenizer", fake_load)
    tc = TrainingClient("sess-1", FakeRuntime(), base_model="arn:.../private/1.0")
    with pytest.raises(ValueError, match="No public HuggingFace tokenizer"):
        tc.get_tokenizer()


# --- checkpoint discovery -----------------------------------------------------
# list_model_packages delegates to the owning ServiceClient with the session's own
# group. latest_*_checkpoint are CACHE-ONLY (see the save-driven cache tests
# below) — they never query the group, so a fresh/resumed client returns None.


class _RecordingServiceClient:
    """Records the kwargs each checkpoint helper is called with."""

    def __init__(self):
        self.calls: list[tuple] = []

    def list_model_packages(self, **kwargs):
        self.calls.append(("list", kwargs))
        return ["ckpt-list"]

    def _stop_session_if_active(self, training_session_name: str) -> bool:
        # Reached only when a client is dropped without an explicit stop; the
        # automatic path calls this rather than stop_session.
        self.calls.append(("stop", training_session_name))
        return True


def test_list_checkpoints_forwards_max_results():
    svc = _RecordingServiceClient()
    tc = TrainingClient("sess-1", FakeRuntime(), model_package_group_arn="grp", service_client=svc)

    tc.list_model_packages(max_results=5)
    kind, kwargs = svc.calls[0]
    assert kind == "list"
    assert kwargs == {"model_package_group_arn": "grp", "max_results": 5}

    # Default (None) omits max_results so the ServiceClient default applies.
    svc.calls.clear()
    tc.list_model_packages()
    assert svc.calls[0][1] == {"model_package_group_arn": "grp"}


def test_list_checkpoints_requires_group():
    # No group on the client -> clear error pointing to ServiceClient.
    tc = TrainingClient("sess-1", FakeRuntime(), service_client=_RecordingServiceClient())
    with pytest.raises(ValueError, match="no model package group"):
        tc.list_model_packages()


def test_list_checkpoints_requires_service_client():
    tc = TrainingClient("sess-1", FakeRuntime(), model_package_group_arn="grp")
    with pytest.raises(ValueError, match="not created by a ServiceClient"):
        tc.list_model_packages()


def test_latest_by_kind_are_none_without_a_save():
    # Cache-only: a client that has not saved returns None and NEVER queries the
    # group, even when both a group and a ServiceClient are available.
    svc = _RecordingServiceClient()
    tc = TrainingClient("sess-1", FakeRuntime(), model_package_group_arn="grp", service_client=svc)
    assert tc.get_latest_model_package(ModelPackageKind.CHECKPOINT) is None
    assert tc.get_latest_model_package(ModelPackageKind.MODEL) is None
    assert svc.calls == []  # no fallback to the group


# --- save-driven checkpoint cache ------------------------------------------


class _SaveRuntime(FakeRuntime):
    """FakeRuntime whose ops complete carrying a save result with an ARN.

    ``op_kind`` selects whether the completed op reports ``saveState`` or
    ``saveWeights`` (so the save response carries model_package_arn). Each
    completed op gets a unique ARN so the cache can be checked for recency.
    """

    def __init__(self, op_kind: str = "saveState", **kwargs):
        super().__init__(**kwargs)
        self._op_kind = op_kind

    def get_training_operation_result(self, **kwargs):
        self.get_calls.append(kwargs)
        op_id = kwargs["operationId"]
        return {
            "status": "Completed",
            "operationResult": {self._op_kind: {"modelPackageArn": f"arn-{op_id}"}},
        }


def test_save_state_caches_checkpoint_no_api_call():
    svc = _RecordingServiceClient()
    tc = TrainingClient(
        "sess-1",
        _SaveRuntime("saveState"),
        model_package_group_arn="grp",
        service_client=svc,
    )

    tc.save_state().result()
    ckpt = tc.get_latest_model_package(ModelPackageKind.CHECKPOINT)

    assert ckpt.model_package_arn == "arn-op-1"
    assert ckpt.kind.value == "checkpoint"
    # A resumable save is not exposed as a model checkpoint.
    assert tc.get_latest_model_package(ModelPackageKind.MODEL) is None
    # Cache hit -> ServiceClient was never queried.
    assert svc.calls == []


def test_save_weights_for_sampler_caches_model_checkpoint():
    svc = _RecordingServiceClient()
    tc = TrainingClient(
        "sess-1",
        _SaveRuntime("saveWeights"),
        model_package_group_arn="grp",
        service_client=svc,
    )

    tc.save_weights_for_sampler().result()
    ckpt = tc.get_latest_model_package(ModelPackageKind.MODEL)

    assert ckpt.model_package_arn == "arn-op-1"
    assert ckpt.kind.value == "model"
    assert svc.calls == []


def test_async_save_populates_cache_on_result():
    # The cache fills on completion, which for the async path happens when
    # parse_fn runs at .result() time, not at submit time.
    svc = _RecordingServiceClient()
    tc = TrainingClient(
        "sess-1",
        _SaveRuntime("saveState"),
        model_package_group_arn="grp",
        service_client=svc,
    )

    fut = tc.save_state()
    # Not resolved yet -> cache-only helper returns None (no group fallback).
    assert tc.get_latest_model_package(ModelPackageKind.CHECKPOINT) is None
    assert svc.calls == []

    fut.result()
    assert tc.get_latest_model_package(ModelPackageKind.CHECKPOINT).model_package_arn == "arn-op-1"
    assert svc.calls == []  # served from cache


def test_latest_returns_most_recent_cached_save():
    tc = TrainingClient(
        "sess-1",
        _SaveRuntime("saveState"),
        model_package_group_arn="grp",
        service_client=_RecordingServiceClient(),
    )

    tc.save_state().result()  # arn-op-1
    tc.save_state().result()  # arn-op-2
    assert tc.get_latest_model_package(ModelPackageKind.CHECKPOINT).model_package_arn == "arn-op-2"


def test_latest_per_kind_are_tracked_independently():
    # The two kinds occupy separate cache slots: a model save must not evict
    # the resumable entry, and each helper returns the latest of its own kind.
    tc = TrainingClient(
        "sess-1",
        _SaveRuntime("saveState"),
        model_package_group_arn="grp",
        service_client=_RecordingServiceClient(),
    )

    tc.save_state().result()  # arn-op-1 -> resumable
    # Flip the same fake to report a model save so the op counter keeps
    # advancing (a fresh one would reset it to op-1).
    tc._runtime._op_kind = "saveWeights"
    tc.save_weights_for_sampler().result()  # arn-op-2 -> model

    assert tc.get_latest_model_package(ModelPackageKind.CHECKPOINT).model_package_arn == "arn-op-1"
    assert tc.get_latest_model_package(ModelPackageKind.MODEL).model_package_arn == "arn-op-2"


# --- session lifecycle sugar -----------------------------------------------


def test_stop_delegates_to_service_client():
    calls = []

    class _Svc(_RecordingServiceClient):
        def stop_session(self, name):
            calls.append(("stop", name))

    tc = TrainingClient("sess-1", FakeRuntime(), service_client=_Svc())
    tc.stop()
    assert calls == [("stop", "sess-1")]


def test_delete_delegates_with_stop_flag():
    calls = []

    class _Svc(_RecordingServiceClient):
        def delete_session(self, name, *, stop_if_active=False):
            calls.append(("delete", name, stop_if_active))

    tc = TrainingClient("sess-1", FakeRuntime(), service_client=_Svc())
    tc.delete(stop_if_active=True)
    assert calls == [("delete", "sess-1", True)]


def test_stop_requires_service_client():
    tc = TrainingClient("sess-1", FakeRuntime())
    with pytest.raises(ValueError, match="not created by a ServiceClient"):
        tc.stop()


# --- calling convention -----------------------------------------------------


def test_the_parameters_that_do_nothing_cannot_be_passed_positionally():
    """A parameter that is ignored must not occupy a position.

    Each of these is accepted for compatibility and does nothing. Positionally
    they would be indistinguishable from the parameters that do something, so a
    value meant for one would land silently in another; requiring the keyword
    makes such a call a ``TypeError``. It also leaves them removable later,
    since no caller can depend on their position.
    """
    import inspect

    from sagemaker.train.training_session.sampling_client import SamplingClient
    from sagemaker.train.training_session.service_client import ServiceClient

    inert = {
        "seed",
        "train_mlp",
        "train_attn",
        "train_unembed",
        "user_metadata",
        "retry_config",
        "ttl_seconds",
        "name",
        "overwrite",
    }
    positional = []
    for cls in (ServiceClient, TrainingClient, SamplingClient):
        for method_name, method in vars(cls).items():
            if method_name.startswith("_") or not callable(method):
                continue
            try:
                signature = inspect.signature(method)
            except (TypeError, ValueError):  # pragma: no cover - not a function
                continue
            positional += [
                f"{cls.__name__}.{method_name}({parameter})"
                for parameter, spec in signature.parameters.items()
                if parameter in inert and spec.kind is spec.POSITIONAL_OR_KEYWORD
            ]

    assert positional == [], f"inert parameters accepting a position: {positional}"


def test_the_parameters_that_do_something_stay_positional():
    """The guard above must not have swept up the real parameters with them."""
    import inspect

    expected = {
        "save_state": ["description"],
        "save_weights_for_sampler": ["description", "merge_weights"],
        "save_weights_and_get_sampling_client": ["description", "merge_weights"],
        "create_sampling_client": ["model_package_arn"],
        "load_state": ["model_package_arn"],
    }
    for method_name, wanted in expected.items():
        signature = inspect.signature(getattr(TrainingClient, method_name))
        actual = [
            parameter
            for parameter, spec in signature.parameters.items()
            if spec.kind is spec.POSITIONAL_OR_KEYWORD and parameter != "self"
        ]
        assert actual == wanted, f"{method_name} takes {actual}, expected {wanted}"


# --- sequence number accessor ----------------------------------------------


def test_current_sequence_number_is_read_only():
    # The public accessor reports the counter WITHOUT advancing it; reading it
    # repeatedly must not consume a sequence (the bug that burned seq 6 in e2e).
    tc = TrainingClient("sess-1", FakeRuntime())
    assert tc.current_sequence_number == 0
    assert tc.current_sequence_number == 0  # idempotent — no increment
    # A real op advances it by exactly one.
    tc.optim_step()
    assert tc.current_sequence_number == 1


def test_each_submit_advances_sequence_by_one():
    # 1 increment == 1 dispatched request: the seq sent on the wire matches the
    # accessor after each op, with no gaps.
    tc, rt = _client()
    tc.optim_step().result()
    tc.optim_step().result()
    seqs = [c["sequenceNumber"] for c in rt.execute_calls]
    assert seqs == [1, 2]
    assert tc.current_sequence_number == 2


# --- non-blocking submit ------------------------------------------------------


def test_base_methods_return_before_submit_completes():
    import threading
    import time

    class _GatedRuntime(FakeRuntime):
        """Blocks every execute until released, like a slow wire."""

        def __init__(self):
            super().__init__()
            self.release = threading.Event()

        def execute_training_operation(self, **kwargs):
            assert self.release.wait(timeout=5)
            return super().execute_training_operation(**kwargs)

    rt = _GatedRuntime()
    tc = TrainingClient("sess-1", rt)
    start = time.monotonic()
    futs = [tc.optim_step() for _ in range(10)]
    elapsed = time.monotonic() - start
    # All ten submits queued without a single wire round trip completing.
    assert elapsed < 0.5
    assert rt.execute_calls == []
    rt.release.set()
    for f in futs:
        assert isinstance(f.result(), OptimStepResponse)
    # Wire order matches call order despite the background handoff.
    assert [c["sequenceNumber"] for c in rt.execute_calls] == list(range(1, 11))


def test_stop_drains_queued_submits_before_stopping():
    from sagemaker.train.training_session.exceptions import TrainingSessionError

    order = []

    class _Svc(_RecordingServiceClient):
        def stop_session(self, name):
            order.append("stop")

    class _OrderRuntime(FakeRuntime):
        def execute_training_operation(self, **kwargs):
            order.append("submit")
            return super().execute_training_operation(**kwargs)

    tc = TrainingClient("sess-1", _OrderRuntime(), service_client=_Svc())
    tc.optim_step()
    tc.stop()
    assert order == ["submit", "stop"]
    # The client accepts no new operations after stop.
    with pytest.raises(TrainingSessionError, match="closed and accepts no new operations"):
        tc.optim_step()


# --- the chunk weighting is wired, not just implemented ------------------------
# The combiner's arithmetic is covered where it lives. What is only observable
# here is whether the client hands it the chunk sizes: without them an uneven
# split reports a plain mean over chunks, understating whichever chunk carried
# the most datums, and nothing raises.


class _PerChunkLossRuntime(FakeRuntime):
    """Replays a different loss for each forward/forward_backward op, in order."""

    def __init__(self, losses):
        super().__init__()
        self._losses = list(losses)
        self._served = 0

    def get_training_operation_result(self, **kwargs):
        self.get_calls.append(kwargs)
        loss = self._losses[min(self._served, len(self._losses) - 1)]
        self._served += 1
        return {
            "status": "Completed",
            "operationResult": {"forwardBackward": {"metrics": {"loss": loss}}},
        }


def _chunked_loss(method_name, losses, batch, loss_fn_inputs):
    """Run one chunked call whose chunks report ``losses``; return the combined loss."""
    from sagemaker.train.training_session import training_client as tc_mod

    rt = _PerChunkLossRuntime(losses)
    tc = TrainingClient("sess-1", rt)
    original = tc_mod.chunk_datum_list
    try:
        # Two datums per chunk, so a 5-datum batch splits 2 + 2 + 1.
        tc_mod.chunk_datum_list = lambda dl, **kw: original(dl, max_len=2, max_bytes=10**9)
        result = getattr(tc, method_name)(
            batch, loss_fn="cross_entropy", loss_fn_inputs=loss_fn_inputs
        ).result()
    finally:
        tc_mod.chunk_datum_list = original
    assert len(rt.execute_calls) == 3, "the batch did not chunk as expected"
    return result.metrics["loss"]


@pytest.mark.parametrize("method_name", ["forward_backward", "forward"])
def test_chunked_loss_is_weighted_by_chunk_size(method_name):
    """Both chunked paths weight by datum count, so an uneven split is honest.

    Chunks of 2, 2 and 1 reporting 1.0, 1.0 and 11.0 average to
    (2 + 2 + 11) / 5 = 3.0 by datum count, against 4.333 as a plain mean over
    chunks. The plain mean is what a missing ``datum_counts`` produces.
    """
    combined = _chunked_loss(
        method_name,
        losses=[1.0, 1.0, 11.0],
        batch=[[1], [2], [3], [4], [5]],
        loss_fn_inputs=_ce(1),
    )

    assert combined == pytest.approx(3.0), (
        f"{method_name} reported {combined}; 4.333 means the chunk sizes never "
        f"reached the combiner"
    )
