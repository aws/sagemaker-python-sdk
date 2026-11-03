# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
"""Unit tests for sagemaker.train.training_session.sampling_client.

The runtime client is faked; no test performs a real sleep or network call.
Covers request shaping, prompt encoding, optional logprob flags, result
parsing, error propagation, and the TrainingClient factory wiring.
"""

from __future__ import annotations

import asyncio
import gc
import os
import threading

import pytest
from botocore.exceptions import ClientError

from sagemaker.train.training_session._session_handle import _SessionHandle
from sagemaker.train.training_session.exceptions import (
    ResultMismatchError,
    TrainingSessionError,
)
from sagemaker.train.training_session.sampling_client import SamplingClient
from sagemaker.train.training_session.training_client import TrainingClient
from sagemaker.train.training_session.types import (
    SampleResponse,
    SampledSequence,
    SamplingParams,
    StopReason,
)


class FakeRuntime:
    """Records execute_sampling_operation calls and replays canned responses.

    Stands in for the botocore runtime client the sampling client calls
    directly, so ``execute_error`` is a botocore error, exactly what the wire
    raises.
    """

    def __init__(self, *, execute_error: Exception | None = None, sample_response=None):
        self.execute_calls: list[dict] = []
        self.training_calls: list[dict] = []
        self.get_calls: list[dict] = []
        self._execute_error = execute_error
        self._op_counter = 0
        self._sample_response = sample_response or {
            "sequences": [
                {"stopReason": "stop", "tokens": [10, 11, 12], "logprobs": [-0.1, -0.2, -0.3]}
            ]
        }

    # Mirrors the current ExecuteSamplingOperation input shape: trainingSessionName
    # + the SampleOperationConfig union. Reject unknown params the way the
    # runtime model would, so a stray field (e.g. clientToken or
    # the now-deprecated flat prompt/numSamples) fails in unit tests too.
    _ALLOWED_SAMPLE_PARAMS = frozenset(
        {
            "trainingSessionName",
            "operationConfig",
        }
    )

    def execute_sampling_operation(self, **kwargs):
        unknown = set(kwargs) - self._ALLOWED_SAMPLE_PARAMS
        if unknown:
            raise TypeError(f"Unknown parameter(s) in ExecuteSamplingOperation: {unknown}")
        self.execute_calls.append(kwargs)
        if self._execute_error is not None:
            raise self._execute_error
        self._op_counter += 1
        return {"operationId": f"sop-{self._op_counter}"}

    def get_sampling_operation_result(self, **kwargs):
        self.get_calls.append(kwargs)
        # Mirror the union variant of the most recent submit: route result by
        # the operation config type of the corresponding execute call.
        last_cfg = self.execute_calls[-1]["operationConfig"] if self.execute_calls else {}
        if "computeLogprobs" in last_cfg:
            # One logprob per prompt token, first entry None, as the real
            # contract. A fixed-length reply would trip the SDK's length guard.
            sent = sum(len(c["tokens"]) for c in last_cfg["computeLogprobs"]["prompt"]["chunks"])
            logprobs = [None] + [round(-0.1 * (i + 1), 1) for i in range(1, sent)]
            return {
                "status": "Completed",
                "operationResult": {"computeLogprobs": {"logprobs": logprobs}},
            }
        if "loadWeightsForSampling" in last_cfg:
            return {
                "status": "Completed",
                "operationResult": {},
            }
        return {
            "status": "Completed",
            "operationResult": {"sample": self._sample_response},
        }

    # training side — only used to drive save_weights_and_get_sampling_client
    def execute_training_operation(self, **kwargs):
        self.training_calls.append(kwargs)
        self._op_counter += 1
        return {"operationId": f"op-{self._op_counter}"}

    def get_training_operation_result(self, **kwargs):
        return {
            "status": "Completed",
            "operationResult": {"saveWeights": {"modelPackageArn": "arn:mp/7"}},
        }


def _client(**kwargs) -> tuple[SamplingClient, FakeRuntime]:
    rt = FakeRuntime(**kwargs)
    return SamplingClient("sess-1", rt), rt


def _sample_cfg(call: dict) -> dict:
    """Return the ``sample`` variant of a recorded execute_sampling_operation call."""
    return call["operationConfig"]["sample"]


# --- request shaping --------------------------------------------------------


def test_sample_shapes_request_from_token_ids():
    sc, rt = _client()
    sc.sample([1, 2, 3], num_samples=4, sampling_params=SamplingParams(max_tokens=64)).result()
    call = rt.execute_calls[0]
    cfg = _sample_cfg(call)
    assert call["trainingSessionName"] == "sess-1"
    assert cfg["numSamples"] == 4
    assert cfg["prompt"]["chunks"][0]["tokens"] == [1, 2, 3]
    assert cfg["samplingParams"]["maxTokens"] == 64
    # ExecuteSamplingOperation has no clientToken field (sampling is not deduped)
    assert "clientToken" not in call
    # optional flags omitted unless requested
    assert "includePromptLogprobs" not in cfg
    assert "topkPromptLogprobs" not in cfg


def test_sample_defaults_to_one_sample_and_default_params():
    sc, rt = _client()
    sc.sample([5, 6]).result()
    cfg = _sample_cfg(rt.execute_calls[0])
    assert cfg["numSamples"] == 1
    assert cfg["samplingParams"]["maxTokens"] == SamplingParams().max_tokens


def test_sample_passes_through_prebuilt_model_input():
    sc, rt = _client()
    mi = {"chunks": [{"tokens": [9, 9], "type": "encoded_text"}]}
    sc.sample(mi).result()
    assert _sample_cfg(rt.execute_calls[0])["prompt"] is mi


def test_sample_string_prompt_requires_tokenizer():
    sc, _ = _client()
    with pytest.raises(ValueError, match="tokenizer"):
        sc.sample("hello world")


def test_sample_string_prompt_uses_tokenizer():
    rt = FakeRuntime()

    class _Tok:
        def encode(self, text):
            return [ord(c) % 7 for c in text]

    sc = SamplingClient("sess-1", rt, tokenizer=_Tok())
    sc.sample("hi").result()
    assert _sample_cfg(rt.execute_calls[0])["prompt"]["chunks"][0]["tokens"] == [
        ord("h") % 7,
        ord("i") % 7,
    ]


def test_sample_tokenizes_a_model_input_built_from_text():
    """``ModelInput.from_text`` must reach the wire as tokens, like a bare string.

    The runtime ModelInput has only a ``chunks`` member, so the typed path has to
    tokenize as well; otherwise passing text through the type was strictly
    worse than passing it untyped.
    """
    from sagemaker.train.training_session.types import ModelInput

    rt = FakeRuntime()

    class _Tok:
        def encode(self, text):
            return [1, 2, 3]

    sc = SamplingClient("sess-1", rt, tokenizer=_Tok())
    sc.sample(ModelInput.from_text("hi")).result()
    cfg = _sample_cfg(rt.execute_calls[0])
    assert cfg["prompt"] == {"chunks": [{"tokens": [1, 2, 3], "type": "encoded_text"}]}
    assert "prompt" not in cfg["prompt"]


def test_sample_model_input_from_text_requires_tokenizer():
    from sagemaker.train.training_session.types import ModelInput

    sc, _ = _client()
    with pytest.raises(ValueError, match="tokenizer"):
        sc.sample(ModelInput.from_text("hello world"))


def test_sample_rejects_unsupported_prompt_type():
    sc, _ = _client()
    with pytest.raises(TypeError, match="Unsupported prompt type"):
        sc.sample(12.5)


def test_logprob_flags_attached_when_requested():
    sc, rt = _client()
    sc.sample([1, 2], include_prompt_logprobs=True, topk_prompt_logprobs=5).result()
    cfg = _sample_cfg(rt.execute_calls[0])
    assert cfg["includePromptLogprobs"] is True
    assert cfg["topkPromptLogprobs"] == 5


# --- checkpoint selection (modelPackageArn on the wire) ---------------------
# model_package_arn selects the checkpoint, sent on the wire as modelPackageArn.
# Empty => the field is omitted => the base model is sampled.


def test_sample_sends_model_package_arn_when_set():
    rt = FakeRuntime()
    sc = SamplingClient("sess-1", rt, model_package_arn="arn:mp/42")
    sc.sample([1, 2, 3]).result()
    assert _sample_cfg(rt.execute_calls[0])["modelPackageArn"] == "arn:mp/42"


def test_sample_omits_model_package_arn_for_base_model():
    sc, rt = _client()  # default model_package_arn == "" (base model)
    sc.sample([1, 2, 3]).result()
    # Field must be ABSENT (not an empty string) so the base model is sampled.
    assert "modelPackageArn" not in _sample_cfg(rt.execute_calls[0])


def test_sample_async_sends_model_package_arn_when_set():
    rt = FakeRuntime()
    sc = SamplingClient("sess-1", rt, model_package_arn="arn:mp/7")
    asyncio.run(sc.sample_async([4, 5]))
    assert _sample_cfg(rt.execute_calls[0])["modelPackageArn"] == "arn:mp/7"


def test_compute_logprobs_sends_model_package_arn_when_set():
    rt = FakeRuntime()
    sc = SamplingClient("sess-1", rt, model_package_arn="arn:mp/9")
    sc.compute_logprobs([1, 2, 3]).result()
    cfg = rt.execute_calls[0]["operationConfig"]["computeLogprobs"]
    assert cfg["modelPackageArn"] == "arn:mp/9"


def test_compute_logprobs_omits_model_package_arn_for_base_model():
    sc, rt = _client()  # base model
    sc.compute_logprobs([1, 2, 3]).result()
    cfg = rt.execute_calls[0]["operationConfig"]["computeLogprobs"]
    assert "modelPackageArn" not in cfg


def test_created_sampling_client_sends_its_model_package_arn_on_sample():
    # End-to-end factory wiring: a client created for a specific checkpoint
    # actually puts that ARN on the wire (regression guard — the selector used
    # to be stored but silently dropped from the request).
    rt = FakeRuntime()
    tc = TrainingClient("sess-1", rt)
    sc = tc.create_sampling_client(model_package_arn="arn:mp/3")
    sc.sample([1]).result()
    assert _sample_cfg(rt.execute_calls[0])["modelPackageArn"] == "arn:mp/3"


def test_compute_logprobs_sends_union_variant_and_parses_result():
    sc, rt = _client()
    result = sc.compute_logprobs([1, 2, 3]).result()
    # Sent the computeLogprobs union variant with the tokenized prompt, and no
    # sample-only fields.
    cfg = rt.execute_calls[0]["operationConfig"]
    assert set(cfg) == {"computeLogprobs"}
    assert cfg["computeLogprobs"]["prompt"]["chunks"][0]["tokens"] == [1, 2, 3]
    # Parsed from operationResult.computeLogprobs.logprobs; the first entry is
    # None (first token has no preceding context).
    assert result == [None, -0.2, -0.3]


def test_compute_logprobs_rejects_result_of_the_wrong_length():
    """A per-token result must be as long as the sequence sent.

    A shorter or longer list belongs to a different request, so it is raised
    rather than returned; otherwise the caller consumes another request's
    values as its own.
    """

    class _WrongLength(FakeRuntime):
        def get_sampling_operation_result(self, **kwargs):
            return {
                "status": "Completed",
                "operationResult": {"computeLogprobs": {"logprobs": [None, -0.2]}},
            }

    rt = _WrongLength()
    sc = SamplingClient("sess-1", rt)
    with pytest.raises(ResultMismatchError, match="2 values for a 3-token sequence"):
        sc.compute_logprobs([1, 2, 3]).result()


def test_compute_logprobs_accepts_result_matching_the_prompt():
    """The guard does not fire on the correct length (regression guard)."""
    sc, _ = _client()
    assert len(sc.compute_logprobs([1, 2, 3, 4]).result()) == 4


def test_compute_logprobs_skips_guard_when_count_unknown():
    """A prompt the service tokenizes has no client-side count to check."""
    from sagemaker.train.training_session.sampling_client import (
        _make_compute_logprobs_parser,
    )

    parse = _make_compute_logprobs_parser(None)
    raw = {"operationResult": {"computeLogprobs": {"logprobs": [None, -0.2]}}}
    assert parse(raw) == [None, -0.2]


def test_compute_logprobs_string_prompt_uses_tokenizer():
    rt = FakeRuntime()

    class _Tok:
        def encode(self, text):
            return [ord(c) % 7 for c in text]

    sc = SamplingClient("sess-1", rt, tokenizer=_Tok())
    sc.compute_logprobs("hi").result()
    cfg = rt.execute_calls[0]["operationConfig"]["computeLogprobs"]
    assert cfg["prompt"]["chunks"][0]["tokens"] == [ord("h") % 7, ord("i") % 7]


def test_each_call_records_separately():
    sc, rt = _client()
    f1 = sc.sample([1])
    f2 = sc.sample([2])
    f1.result()
    f2.result()
    assert len(rt.execute_calls) == 2
    # Sample submits are unordered (order-independent ops), so assert on the
    # set of prompts sent rather than arrival order.
    sent = sorted(_sample_cfg(c)["prompt"]["chunks"][0]["tokens"][0] for c in rt.execute_calls)
    assert sent == [1, 2]


# --- result parsing ---------------------------------------------------------


def test_result_parses_sample_response():
    sc, _ = _client()
    result = sc.sample([1, 2, 3]).result()
    assert isinstance(result, SampleResponse)
    assert len(result.sequences) == 1
    seq = result.sequences[0]
    assert isinstance(seq, SampledSequence)
    assert seq.stop_reason == StopReason.STOP
    assert seq.tokens == [10, 11, 12]
    assert seq.logprobs == [-0.1, -0.2, -0.3]


def test_result_parses_prompt_and_topk_logprobs():
    resp = {
        "sequences": [{"stopReason": "length", "tokens": [1]}],
        "promptLogprobs": [-0.5, -0.6],
        "topkPromptLogprobs": [
            [{"tokenId": 1, "logprob": -0.5}, {"tokenId": 2, "logprob": -1.0}],
            [{"tokenId": 3, "logprob": -0.6}],
        ],
    }
    sc, _ = _client(sample_response=resp)
    result = sc.sample([1, 2]).result()
    assert result.prompt_logprobs == [-0.5, -0.6]
    assert result.topk_prompt_logprobs[0][0].token_id == 1
    assert result.topk_prompt_logprobs[0][1].logprob == -1.0
    assert result.sequences[0].stop_reason == StopReason.LENGTH


def test_poll_uses_sampling_operation_id():
    sc, rt = _client()
    sc.sample([1]).result()
    assert rt.get_calls[0]["operationId"] == "sop-1"
    assert rt.get_calls[0]["trainingSessionName"] == "sess-1"


# --- error propagation ------------------------------------------------------
# The service's own error reaches the caller unchanged, code and message intact.


def test_service_error_propagates_unchanged():
    error = ClientError(
        {
            "Error": {"Code": "ConflictException", "Message": "session is Stopping"},
            "ResponseMetadata": {"HTTPStatusCode": 409},
        },
        "ExecuteSamplingOperation",
    )
    sc, _ = _client(execute_error=error)
    with pytest.raises(ClientError) as ei:
        sc.sample([1, 2]).result()
    assert ei.value.response["Error"]["Code"] == "ConflictException"
    assert "session is Stopping" in str(ei.value)


# --- future contract (sample / sample_async / compute_logprobs) -------------


def test_sample_async_awaits_response_directly():
    sc, rt = _client()
    # The async variant is a coroutine: a single await yields the response.
    assert asyncio.iscoroutinefunction(sc.sample_async)
    resp = asyncio.run(sc.sample_async([7, 8], num_samples=2))
    cfg = _sample_cfg(rt.execute_calls[0])
    assert cfg["numSamples"] == 2
    assert cfg["prompt"]["chunks"][0]["tokens"] == [7, 8]
    assert isinstance(resp, SampleResponse)
    assert resp.sequences[0].tokens == [10, 11, 12]


def test_sample_returns_unresolved_future():
    # Drop-in contract: sample() returns the future itself; callers unwrap
    # with .result() (or await).
    sc, rt = _client()
    fut = sc.sample([7, 8])
    assert hasattr(fut, "result")
    assert rt.get_calls == []
    assert isinstance(fut.result(), SampleResponse)
    assert len(rt.get_calls) == 1


def test_compute_logprobs_returns_unresolved_future():
    sc, rt = _client()
    fut = sc.compute_logprobs([1, 2, 3])
    assert hasattr(fut, "result")
    assert rt.get_calls == []
    assert fut.result() == [None, -0.2, -0.3]


def test_compute_logprobs_async_awaits_logprobs_directly():
    sc, _ = _client()
    assert asyncio.iscoroutinefunction(sc.compute_logprobs_async)
    assert asyncio.run(sc.compute_logprobs_async([1, 2, 3])) == [None, -0.2, -0.3]


def test_sample_submits_overlap_and_return_before_wire_completes():
    import threading
    import time

    class _GatedRuntime(FakeRuntime):
        """Blocks every execute until released, like a slow wire."""

        def __init__(self):
            super().__init__()
            self.release = threading.Event()
            self.started = []

        def execute_sampling_operation(self, **kwargs):
            self.started.append(1)
            assert self.release.wait(timeout=5)
            return super().execute_sampling_operation(**kwargs)

    rt = _GatedRuntime()
    sc = SamplingClient("sess-1", rt)
    futs = [sc.sample([i]) for i in range(4)]
    # Unordered submits overlap: all four reach the wire before any completes.
    deadline = time.monotonic() + 2
    while len(rt.started) < 4 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert len(rt.started) == 4
    assert rt.execute_calls == []
    rt.release.set()
    for f in futs:
        assert isinstance(f.result(), SampleResponse)


# --- TrainingClient factory wiring ------------------------------------------


def test_create_sampling_client_binds_session_and_ref():
    rt = FakeRuntime()
    tc = TrainingClient("sess-1", rt)
    sc = tc.create_sampling_client(model_package_arn="arn:mp/3")
    assert isinstance(sc, SamplingClient)
    assert sc.training_session_name == "sess-1"
    assert sc.model_package_arn == "arn:mp/3"


def test_create_sampling_client_defaults_to_base_model():
    rt = FakeRuntime()
    tc = TrainingClient("sess-1", rt)
    assert tc.create_sampling_client().model_package_arn == ""


def test_save_weights_and_get_sampling_client_uses_model_package_arn():
    rt = FakeRuntime()
    tc = TrainingClient("sess-1", rt)
    sc = tc.save_weights_and_get_sampling_client()
    assert isinstance(sc, SamplingClient)
    assert sc.model_package_arn == "arn:mp/7"  # from the canned saveWeights result
    # save_weights_for_sampler goes through the training path...
    training_ops = [next(iter(c["operationConfig"])) for c in rt.training_calls]
    assert training_ops == ["saveWeights"]
    # ...but LWFS routes through the sampling-operations path (no seq consumed).
    assert len(rt.execute_calls) == 1
    lwfs = rt.execute_calls[0]["operationConfig"]["loadWeightsForSampling"]
    assert lwfs["modelPackageArn"] == "arn:mp/7"


def test_load_weights_for_sampling_shapes_request():
    rt = FakeRuntime()
    tc = TrainingClient("sess-1", rt)
    tc._load_weights_for_sampling("arn:mp/9")
    assert len(rt.execute_calls) == 1
    cfg = rt.execute_calls[0]["operationConfig"]["loadWeightsForSampling"]
    assert cfg == {"modelPackageArn": "arn:mp/9"}
    assert len(rt.training_calls) == 0, "LWFS must not use training path"


def test_create_sampling_client_propagates_base_model(monkeypatch):
    import sagemaker.train.training_session._utils as utils

    sentinel = object()
    monkeypatch.setattr(
        utils,
        "load_tokenizer",
        lambda base_model, hub_client, hf_model_id=None: sentinel,
    )
    tc = TrainingClient("sess-1", FakeRuntime(), base_model="Qwen/Qwen3-32B")
    sc = tc.create_sampling_client(model_package_arn="arn:mp/3")
    # SamplingClient can resolve a tokenizer from the inherited base model.
    assert sc.get_tokenizer() is sentinel


def test_sampling_client_get_tokenizer_unsupported_raises(monkeypatch):
    import sagemaker.train.training_session._utils as utils

    def fake_load(base_model, hub_client, hf_model_id=None):
        raise ValueError("No public HuggingFace tokenizer is available")

    monkeypatch.setattr(utils, "load_tokenizer", fake_load)
    sc = SamplingClient("sess-1", FakeRuntime(), base_model="arn:.../private/1.0")
    with pytest.raises(ValueError, match="No public HuggingFace tokenizer"):
        sc.get_tokenizer()


def test_save_weights_and_get_sampling_client_raises_without_a_checkpoint():
    """A save with no ARN must raise, not hand back a base-model sampler.

    An empty ``model_package_arn`` means the base model to ``SamplingClient``, so
    returning one here sampled untrained weights while the caller believed it was
    sampling what had just been saved.
    """
    rt = FakeRuntime()

    def _no_mp_result(**kwargs):
        return {"status": "Completed", "operationResult": {"saveWeights": {}}}

    rt.get_training_operation_result = _no_mp_result
    tc = TrainingClient("sess-1", rt)

    with pytest.raises(TrainingSessionError, match="no checkpoint ARN"):
        tc.save_weights_and_get_sampling_client()

    # The save itself still went out; nothing was loaded for sampling.
    op_types = [next(iter(c["operationConfig"])) for c in rt.training_calls]
    assert op_types == ["saveWeights"]


# --- session ownership / context-manager lifecycle --------------------------
#
# Only ServiceClient.create_sampling_client creates the backing session, so only
# that client stops it. A client from TrainingClient.create_sampling_client
# shares the training session and must leave it running.


class FakeServiceClient:
    """Records stop_session calls; optionally raises to exercise the error path."""

    def __init__(self, *, stop_error: Exception | None = None):
        self.stopped: list[str] = []
        self._stop_error = stop_error

    def _get_hub_client(self):
        # TrainingClient.create_sampling_client lazily resolves a Hub client
        # through its ServiceClient; tests never need a real one.
        return None

    def stop_session(self, training_session_name: str) -> None:
        self._stop_session_if_active(training_session_name)

    def _stop_session_if_active(self, training_session_name: str) -> bool:
        self.stopped.append(training_session_name)
        if self._stop_error is not None:
            raise self._stop_error
        return True


def test_owned_session_is_stopped_on_context_exit():
    svc = FakeServiceClient()
    sc = SamplingClient("sess-owned", FakeRuntime(), service_client=svc)
    with sc as entered:
        assert entered is sc
        assert svc.stopped == [], "must not stop while the block is running"
    assert svc.stopped == ["sess-owned"]


def test_unowned_session_is_not_stopped_on_context_exit():
    # Workflow A: tc.create_sampling_client() shares the TrainingClient's
    # session. Leaving the SamplingClient block must not kill it.
    svc = FakeServiceClient()
    tc = TrainingClient("sess-1", FakeRuntime(), service_client=svc)
    sc = tc.create_sampling_client()
    assert sc._owns_session is False
    with sc:
        pass
    assert svc.stopped == []


def test_service_client_sampling_client_owns_its_session():
    sc = SamplingClient("sess-x", FakeRuntime(), service_client=FakeServiceClient())
    assert sc._owns_session is True


def test_stop_on_owned_session_calls_service_client():
    svc = FakeServiceClient()
    sc = SamplingClient("sess-owned", FakeRuntime(), service_client=svc)
    sc.stop()
    assert svc.stopped == ["sess-owned"]


def test_stop_on_unowned_session_is_a_noop():
    # Sharing a handle is what makes it unowned — the same thing
    # TrainingClient.create_sampling_client does.
    svc = FakeServiceClient()
    sc = SamplingClient(
        "sess-shared",
        FakeRuntime(),
        service_client=svc,
        session_handle=_SessionHandle("sess-shared", svc),
    )
    sc.stop()
    assert svc.stopped == []


def test_stop_on_owned_session_without_service_client_warns(caplog):
    sc = SamplingClient("sess-orphan", FakeRuntime())
    with caplog.at_level("WARNING"):
        sc.stop()
    assert "sess-orphan" in caplog.text


def test_owned_session_stopped_even_when_block_raises():
    svc = FakeServiceClient()
    sc = SamplingClient("sess-owned", FakeRuntime(), service_client=svc)
    with pytest.raises(RuntimeError, match="boom"):
        with sc:
            raise RuntimeError("boom")
    # The session is still reclaimed, and __exit__ does not swallow the error.
    assert svc.stopped == ["sess-owned"]


def test_stop_failure_on_exit_is_logged_not_raised(caplog):
    svc = FakeServiceClient(stop_error=RuntimeError("throttled"))
    sc = SamplingClient("sess-owned", FakeRuntime(), service_client=svc)
    with caplog.at_level("WARNING"):
        with sc:
            pass
    assert "sess-owned" in caplog.text


def test_context_exit_drains_submitter_in_both_ownership_modes():
    owned = SamplingClient("s1", FakeRuntime(), service_client=FakeServiceClient())
    with owned:
        pass
    shared = SamplingClient("s2", FakeRuntime())
    with shared:
        pass
    # A drained submitter accepts no further work.
    for client in (owned, shared):
        with pytest.raises(Exception):
            client.sample([1, 2, 3]).result()


# --- submitter sharing / per-refresh cost -----------------------------------
#
# A submitter is an event loop on a daemon thread, and the TrainingClient
# factory is called once per sampler refresh. These tests pin the sharing that
# keeps that off the per-step path, and the ownership rules it needs.


def _live_fds():
    """Descriptor count for this process; Linux-only, so the caller skips."""
    return len(os.listdir(f"/proc/{os.getpid()}/fd"))


def test_derived_sampling_clients_share_the_training_clients_submitter():
    tc = TrainingClient("sess-1", FakeRuntime())
    clients = [tc.create_sampling_client() for _ in range(5)]
    assert all(sc._submitter is tc._submitter for sc in clients)
    assert all(sc._owns_submitter is False for sc in clients)


def test_a_standalone_sampling_client_builds_and_owns_its_own_submitter():
    """The session-owning path has no parent to borrow from."""
    sc = SamplingClient("sess-x", FakeRuntime(), service_client=FakeServiceClient())
    assert sc._owns_submitter is True
    assert sc._submitter is not None
    sc.stop()


def test_refreshing_the_sampler_costs_no_thread_and_no_descriptor_per_refresh():
    """The leak this sharing exists to remove: +1 thread and +3 fds per step.

    An RL loop refreshes its sampler every step, so an unshared submitter made
    the cost per step rather than per session; ~20 refreshes exhausted the
    default macOS descriptor budget on top of a socket-heavy baseline.
    """
    if not os.path.isdir(f"/proc/{os.getpid()}/fd"):
        pytest.skip("descriptor accounting via /proc is Linux-only")

    tc = TrainingClient("sess-1", FakeRuntime())
    # Force the parent's loop into existence first, so it is part of the
    # baseline and the measurement is purely what the refreshes add.
    tc.create_sampling_client().sample([1, 2, 3]).result()
    # Settle cyclic garbage first: an earlier test's parked loop collected
    # mid-measurement would drop the count and read as a refresh taking one.
    gc.collect()
    fds_before, threads_before = _live_fds(), threading.active_count()

    clients = [tc.create_sampling_client() for _ in range(20)]
    for sc in clients:
        sc.sample([1, 2, 3]).result()

    assert threading.active_count() == threads_before, "a refresh started a thread"
    assert _live_fds() == fds_before, "a refresh took a descriptor"


def test_a_retired_sampling_client_does_not_close_its_parents_submitter():
    """Without the ownership gate, retiring one sampler ends the training run.

    ``stop`` and ``__exit__`` closed the submitter unconditionally, which under
    sharing is the parent's. Every later training operation would then fail with
    a closed-submitter error.
    """
    tc = TrainingClient("sess-1", FakeRuntime())
    retired = tc.create_sampling_client()

    retired.stop()
    with tc.create_sampling_client() as also_retired:
        assert also_retired is not None

    # The parent is untouched: it still submits, and its submitter is still open.
    assert tc._submitter._closed is False
    assert tc.create_sampling_client().sample([1, 2, 3]).result() is not None


@pytest.mark.parametrize("retire", [lambda sc: sc.stop(), lambda sc: sc.__exit__(*(None,) * 3)])
def test_a_retired_sampling_client_refuses_work_even_though_it_shares_a_submitter(retire):
    """Retirement has to fail fast, not silently keep submitting.

    Leaving the shared submitter open for its owner means closure cannot be read
    off the submitter, so the client records it. Without that, a caller who
    retires a sampler and keeps the reference would go on reaching the service
    against a session it no longer holds.
    """
    tc = TrainingClient("sess-1", FakeRuntime())
    retired = tc.create_sampling_client()

    retire(retired)

    for op in (lambda: retired.sample([1, 2, 3]), lambda: retired.compute_logprobs([1, 2, 3])):
        with pytest.raises(TrainingSessionError, match="closed and accepts no new"):
            op()
    # Refusing is this client's own state; the shared submitter stays usable.
    assert tc._submitter._closed is False


def test_a_shared_submitter_is_registered_once_not_once_per_refresh():
    """The session handle holds submitters strongly for the whole session.

    Registering a shared submitter per refresh would pile up references to the
    single submitter and drain it once per entry at teardown.
    """
    tc = TrainingClient("sess-1", FakeRuntime())
    for _ in range(10):
        tc.create_sampling_client()
    assert tc._handle._submitters == [tc._submitter]


def test_a_standalone_client_still_registers_its_own_submitter():
    """Skipping registration for a shared submitter must not skip owned ones."""
    sc = SamplingClient("sess-x", FakeRuntime(), service_client=FakeServiceClient())
    assert sc._handle._submitters == [sc._submitter]


def test_stopping_the_training_client_closes_its_derived_samplers_too():
    """The shared submitter runs both, so the parent's teardown ends both.

    A derived client used to hold a private submitter and so stayed submittable
    after its parent stopped, but the session it samples is gone by then, so the
    submits only failed later and further away. Failing at the client with the
    closed-client message is the better of the two.
    """
    svc = FakeServiceClient()
    tc = TrainingClient("sess-1", FakeRuntime(), service_client=svc)
    sc = tc.create_sampling_client()

    tc.stop()

    assert svc.stopped == ["sess-1"]
    with pytest.raises(TrainingSessionError, match="closed and accepts no new"):
        sc.sample([1, 2, 3])
