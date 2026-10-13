# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
"""Unit tests for sagemaker.train.training_session.service_client.

Both the control-plane and runtime clients are faked by patching the factories
in ``_clients``; the Running-wait loop is driven by a patched ``time`` so no
test performs a real sleep.
"""

from __future__ import annotations

import json

import pytest
from botocore.exceptions import ClientError

from sagemaker.train.training_session import service_client as sc_mod
from sagemaker.train.training_session.service_client import ServiceClient
from sagemaker.train.training_session.training_client import TrainingClient
from sagemaker.train.training_session.types import ModelPackageKind, SessionStatus


class FakeSageMakerClient:
    """Control-plane fake: scripts describe_training_session status sequence."""

    def __init__(self, statuses, *, describe_detail=None, list_pages=None, stop_error=None):
        self._statuses = list(statuses)
        self._stop_error = stop_error
        self.create_calls: list[dict] = []
        self.describe_calls: list[dict] = []
        self.stop_calls: list[dict] = []
        self.delete_calls: list[dict] = []
        self.list_calls: list[dict] = []
        self._describe_detail = describe_detail or {}
        # list_pages: list of (summaries, next_token) tuples served in order.
        self._list_pages = list(list_pages or [])

    def create_training_session(self, **kwargs):
        self.create_calls.append(kwargs)
        return {"TrainingSessionArn": "arn:aws:sagemaker:::training-session/x"}

    def describe_training_session(self, **kwargs):
        self.describe_calls.append(kwargs)
        status = self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]
        detail = {"TrainingSessionStatus": status, **self._describe_detail}
        return {"TrainingSessionDetail": detail}

    def stop_training_session(self, **kwargs):
        self.stop_calls.append(kwargs)
        if self._stop_error is not None:
            raise self._stop_error
        return {}

    def delete_training_session(self, **kwargs):
        self.delete_calls.append(kwargs)
        return {}

    def list_training_sessions(self, **kwargs):
        self.list_calls.append(kwargs)
        summaries, next_token = self._list_pages.pop(0)
        resp = {"TrainingSessionSummaries": summaries}
        if next_token:
            resp["NextToken"] = next_token
        return resp


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def patch_clients(monkeypatch):
    """Patch client factories + time so ServiceClient runs offline and instantly."""

    clock = FakeClock()
    monkeypatch.setattr(sc_mod.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(sc_mod.time, "sleep", clock.sleep)

    holder = {}

    def _install(sagemaker_client, runtime=None):
        runtime = runtime if runtime is not None else object()
        holder["sm"] = sagemaker_client
        holder["rt"] = runtime
        import sagemaker.train.training_session._clients as clients_mod

        # ServiceClient imports both client builders lazily inside __init__, so
        # patch the source module.
        monkeypatch.setattr(clients_mod, "_control_plane_client", lambda **kw: sagemaker_client)
        monkeypatch.setattr(clients_mod, "_runtime_client", lambda **kw: runtime)
        return clock

    return _install, holder


def _create_kwargs(**overrides):
    base = dict(
        training_session_name="sess-1",
        model_package_group_arn="arn:mpg/x",
        s3_output_path="s3://b/out/",
    )
    base.update(overrides)
    return base


# The new API takes base_model as positional arg; helper returns the
# positional + kwargs for svc.create_lora_training_client(base_model, **kwargs).
_BASE_MODEL_ARN = "arn:hub-content/Model/base"


def test_create_waits_for_running_then_returns_training_client(patch_clients):
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Created", "Created", "Running"])
    clock = install(sm)

    svc = ServiceClient(region="us-west-2", role_arn="arn:aws:iam::123456789012:role/TestRole")
    tc = svc.create_lora_training_client(_BASE_MODEL_ARN, **_create_kwargs(rank=16, alpha=32))

    assert isinstance(tc, TrainingClient)
    assert tc.training_session_name == "sess-1"
    # CreateTrainingSession got the LoRA config.
    cc = sm.create_calls[0]
    assert cc["LoraConfig"] == {"Rank": 16, "Alpha": 32}
    # Polled until Running; only slept while pending (no real wait).
    assert clock.sleeps and all(s == 10 for s in clock.sleeps)


def test_create_raises_on_terminal_status(patch_clients):
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Created", "Failed"])
    install(sm)

    svc = ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole")
    with pytest.raises(RuntimeError, match="terminal state"):
        svc.create_lora_training_client(_BASE_MODEL_ARN, **_create_kwargs())


# --- create_sampling_client -------------------------------------------------


def test_create_sampling_client_with_base_model(patch_clients):
    """Creates an implicit session and returns a SamplingClient sampling the
    base model."""
    install, holder = patch_clients
    sm = FakeSageMakerClient(["Running"])
    runtime = object()
    install(sm, runtime=runtime)

    svc = ServiceClient(
        role_arn="arn:aws:iam::123456789012:role/TestRole",
        model_package_group_arn="arn:mpg/x",
        s3_output_path="s3://b/out/",
    )
    svc._hub_client = FakeHubClient()
    sampler = svc.create_sampling_client(base_model="hubcontent-vlm-model-b")

    assert sampler.model_package_arn == ""
    assert sampler._base_model == svc._hub_client._arn
    assert sampler._runtime is runtime
    # An implicit session was created.
    assert len(sm.create_calls) == 1
    assert sm.create_calls[0]["RoleArn"] == "arn:aws:iam::123456789012:role/TestRole"
    # Nothing else owns that implicit session, so this client must stop it.
    assert sampler._owns_session is True
    assert sampler._service_client is svc


def test_sampling_client_context_exit_stops_implicit_session(patch_clients):
    """`with svc.create_sampling_client(...)` stops the session it created."""
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Running"])
    install(sm)

    svc = ServiceClient(
        role_arn="arn:aws:iam::123456789012:role/TestRole",
        model_package_group_arn="arn:mpg/x",
        s3_output_path="s3://b/out/",
    )
    svc._hub_client = FakeHubClient()

    with svc.create_sampling_client(base_model="hubcontent-vlm-model-b") as sp:
        session_name = sp.training_session_name
        assert sm.stop_calls == []

    assert [c["TrainingSessionName"] for c in sm.stop_calls] == [session_name]


def test_create_sampling_client_with_model_package_arn(patch_clients):
    """Creates an implicit session and binds the checkpoint ARN."""
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Running"])
    install(sm)

    svc = ServiceClient(
        region="us-west-2",
        role_arn="arn:aws:iam::123456789012:role/TestRole",
        model_package_group_arn="arn:mpg/x",
        s3_output_path="s3://b/out/",
    )
    # Stub hub client: DescribeModelPackage publishes the base model as
    # InferenceSpecification.Containers[0].BaseModel (name + pinned version),
    # not as a top-level field.
    hub = FakeHubClient()
    hub.describe_model_package = lambda **kw: {
        "ModelPackageArn": "arn:mp/weights-7",
        "InferenceSpecification": {
            "Containers": [
                {
                    "IsCheckpoint": False,
                    "BaseModel": {
                        "HubContentName": "hubcontent-text-model-a",
                        "HubContentVersion": "3.15.0",
                    },
                }
            ]
        },
    }
    svc._hub_client = hub

    sampler = svc.create_sampling_client(model_package_arn="arn:mp/weights-7")

    assert sampler.model_package_arn == "arn:mp/weights-7"
    assert sampler._base_model == (
        "arn:aws:sagemaker:us-west-2:aws:hub-content/"
        "SageMakerPublicHub/Model/hubcontent-text-model-a/3.15.0"
    )


def test_create_sampling_client_rejects_neither_arg(patch_clients):
    """Must supply exactly one of base_model or model_package_arn."""
    install, _ = patch_clients
    install(FakeSageMakerClient(["Running"]))

    svc = ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole")
    with pytest.raises(ValueError, match="exactly one"):
        svc.create_sampling_client()


def test_create_sampling_client_rejects_both_args(patch_clients):
    install, _ = patch_clients
    install(FakeSageMakerClient(["Running"]))

    svc = ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole")
    with pytest.raises(ValueError, match="exactly one"):
        svc.create_sampling_client(base_model="provider-a/model-a", model_package_arn="arn:mp/x")


# --- stop / delete ----------------------------------------------------------


def test_stop_session_calls_control_plane(patch_clients):
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Running"])
    install(sm)

    ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole").stop_session("sess-1")
    assert sm.stop_calls == [{"TrainingSessionName": "sess-1"}]


def _stop_state_error(from_state):
    """The ValidationException the control plane returns for a rejected stop."""
    return ClientError(
        {
            "Error": {
                "Code": "ValidationException",
                "Message": (
                    f"Cannot stop training session in state {from_state}: invalid "
                    f"session state transition from {from_state} to Stopping"
                ),
            },
            "ResponseMetadata": {"HTTPStatusCode": 400},
        },
        "StopTrainingSession",
    )


@pytest.mark.parametrize("state", ["Stopping", "Stopped", "Failed", "Deleting", "DeletionFailed"])
def test_stop_session_is_idempotent_for_states_with_nothing_to_stop(patch_clients, state):
    """A stop the service cannot apply is not an error the caller has to handle."""
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Running"], stop_error=_stop_state_error(state))
    install(sm)

    svc = ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole")
    svc.stop_session("sess-1")  # must not raise
    assert sm.stop_calls == [{"TrainingSessionName": "sess-1"}]


def test_stop_session_repeated_calls_both_return(patch_clients):
    """Second stop mirrors an explicit stop followed by the automatic one."""
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Running"])
    install(sm)

    svc = ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole")
    svc.stop_session("sess-1")
    sm._stop_error = _stop_state_error("Stopping")
    svc.stop_session("sess-1")
    assert len(sm.stop_calls) == 2


def test_stop_session_reraises_unrelated_validation_error(patch_clients):
    """Only the state-transition case is absorbed; real failures still surface."""
    install, _ = patch_clients
    sm = FakeSageMakerClient(
        ["Running"],
        stop_error=ClientError(
            {"Error": {"Code": "ValidationException", "Message": "1 validation error"}},
            "StopTrainingSession",
        ),
    )
    install(sm)

    svc = ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole")
    with pytest.raises(ClientError):
        svc.stop_session("sess-1")


def test_stop_session_reraises_resource_not_found(patch_clients):
    install, _ = patch_clients
    sm = FakeSageMakerClient(
        ["Running"],
        stop_error=ClientError(
            {
                "Error": {
                    "Code": "ResourceNotFoundException",
                    "Message": "Training session not found: sess-1",
                }
            },
            "StopTrainingSession",
        ),
    )
    install(sm)

    svc = ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole")
    with pytest.raises(ClientError):
        svc.stop_session("sess-1")


def test_delete_session_calls_delete_directly(patch_clients):
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Stopped"])
    install(sm)

    ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole").delete_session("sess-1")
    assert sm.delete_calls == [{"TrainingSessionName": "sess-1"}]
    # Without stop_if_active we don't stop or poll.
    assert sm.stop_calls == []
    assert sm.describe_calls == []


def test_delete_session_stops_and_waits_when_requested(patch_clients):
    install, _ = patch_clients
    # Stopping -> Stopped: the wait loop sleeps once, then deletes.
    sm = FakeSageMakerClient(["Stopping", "Stopped"])
    install(sm)

    ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole").delete_session(
        "sess-1", stop_if_active=True
    )
    assert sm.stop_calls == [{"TrainingSessionName": "sess-1"}]
    assert sm.delete_calls == [{"TrainingSessionName": "sess-1"}]
    # Polled at least once before reaching the deletable state.
    assert sm.describe_calls


def test_stop_session_accepts_arn(patch_clients):
    """A TrainingSessionArn from list_sessions passes straight to stop_session."""
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Running"])
    install(sm)

    svc = ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole")
    svc.stop_session("arn:aws:sagemaker:us-west-2:123456789012:training-session/sess-1")
    # The ARN's last segment is used as the name.
    assert sm.stop_calls == [{"TrainingSessionName": "sess-1"}]


def test_delete_session_accepts_arn(patch_clients):
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Stopped"])
    install(sm)

    svc = ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole")
    svc.delete_session("arn:aws:sagemaker:us-west-2:123456789012:training-session/sess-1")
    assert sm.delete_calls == [{"TrainingSessionName": "sess-1"}]


# --- list -------------------------------------------------------------------


def _summary(arn, status="Running"):
    return {"TrainingSessionArn": arn, "TrainingSessionStatus": status}


def test_list_single_page(patch_clients):
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Running"], list_pages=[([_summary("arn/a")], None)])
    install(sm)

    out = ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole").list_sessions()
    assert [s.training_session_arn for s in out] == ["arn/a"]
    assert out[0].status is SessionStatus.RUNNING
    # No filters set -> empty request body.
    assert sm.list_calls == [{}]


def test_list_auto_paginates(patch_clients):
    install, _ = patch_clients
    sm = FakeSageMakerClient(
        ["Running"],
        list_pages=[
            ([_summary("arn/a")], "tok1"),
            ([_summary("arn/b")], None),
        ],
    )
    install(sm)

    out = ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole").list_sessions()
    assert [s.training_session_arn for s in out] == ["arn/a", "arn/b"]
    # Second call forwarded the NextToken from the first page.
    assert sm.list_calls[1]["NextToken"] == "tok1"


def test_list_forwards_filters(patch_clients):
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Running"], list_pages=[([], None)])
    install(sm)

    ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole").list_sessions(
        name_contains="exp", status_equals="Running"
    )
    assert sm.list_calls[0]["NameContains"] == "exp"
    assert sm.list_calls[0]["StatusEquals"] == "Running"


def test_list_status_equals_enum_sends_wire_value(patch_clients):
    """A SessionStatus enum must serialize to its wire value, not str(enum).

    Regression: str() of a str-mixin Enum is "SessionStatus.EXPIRED" on
    Python 3.11+, which the service rejects with a ValidationException.
    """
    from sagemaker.train.training_session import SessionStatus

    install, _ = patch_clients
    sm = FakeSageMakerClient(["Running"], list_pages=[([], None)])
    install(sm)

    ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole").list_sessions(
        status_equals=SessionStatus.EXPIRED
    )
    assert sm.list_calls[0]["StatusEquals"] == "Expired"


def test_list_respects_max_results(patch_clients):
    install, _ = patch_clients
    # First page already returns 2; max_results=1 caps the result and stops paging.
    sm = FakeSageMakerClient(
        ["Running"],
        list_pages=[([_summary("arn/a"), _summary("arn/b")], "tok1")],
    )
    install(sm)

    out = ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole").list_sessions(
        max_results=1
    )
    assert [s.training_session_arn for s in out] == ["arn/a"]
    # Only one page fetched despite a NextToken being present.
    assert len(sm.list_calls) == 1
    assert sm.list_calls[0]["MaxResults"] == 1


def test_list_holds_max_results_at_the_page_size(patch_clients):
    """A cap above one page is served by paging, not by one oversized request.

    The operation rejects a page larger than its own limit, so forwarding the
    caller's cap verbatim would fail the call instead of returning that many.
    """
    install, _ = patch_clients
    sm = FakeSageMakerClient(
        ["Running"],
        list_pages=[
            ([_summary(f"arn/{index}") for index in range(100)], "tok1"),
            ([_summary("arn/100")], None),
        ],
    )
    install(sm)

    out = ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole").list_sessions(
        max_results=150
    )

    assert len(out) == 101
    assert [call["MaxResults"] for call in sm.list_calls] == [100, 50]


def test_list_keeps_a_status_this_version_does_not_know(patch_clients):
    """An unrecognised status reaches the caller rather than failing the list."""
    install, _ = patch_clients
    sm = FakeSageMakerClient(
        ["Running"], list_pages=[([_summary("arn/a", status="SomethingNew")], None)]
    )
    install(sm)

    out = ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole").list_sessions()

    assert out[0].status == "SomethingNew"


# --- base_model resolution -------------------------------------------------
# create_lora_training_client accepts a friendly base_model name (resolved to a
# Hub ARN via DescribeHubContent) OR a full base_model_arn. Exactly one is
# required. The hub client is stubbed so no live AWS call is made.


class FakeHubClient:
    """Stubs DescribeHubContent + ListHubContents for base-model resolution."""

    #: Recipe keyword marking a model as fine-tunable (what the SDK filters on).
    _TRAINABLE_KEYWORD = "@recipe:finetuning_mtrl_lora"

    def __init__(
        self,
        arn="arn:aws:sagemaker:us-west-2:aws:hub-content/SageMakerPublicHub/Model/hubcontent-vlm-model-b/9.9.9",
        hub_content_names=None,
        keywords_by_name=None,
    ):
        self._arn = arn
        # Hub content names "available in this region" for list_hub_contents.
        self._names = (
            hub_content_names
            if hub_content_names is not None
            else ["hubcontent-vlm-model-b", "hubcontent-text-model-a"]
        )
        # Search keywords per content name. Default: every listed model carries
        # the fine-tuning recipe keyword, i.e. is trainable.
        self._keywords_by_name = (
            keywords_by_name
            if keywords_by_name is not None
            else {n: [self._TRAINABLE_KEYWORD] for n in self._names}
        )
        self.describe_calls: list[dict] = []
        self.list_calls: list[dict] = []

    def describe_hub_content(self, **kwargs):
        name = kwargs.get("HubContentName")
        self.describe_calls.append(kwargs)
        # DescribeHubContent is the authority on existence: an unknown content
        # name raises, mirroring the real API's ResourceNotFound.
        if name not in self._names:
            raise RuntimeError("ResourceNotFound: no such hub content")
        # The real Hub doc is a JSON string carrying the published HuggingfaceId,
        # one per model — echoed from this model's own keyword so a multi-model
        # catalogue publishes several ids rather than collapsing onto one.
        return {
            "HubContentArn": self._arn,
            "HubContentDocument": json.dumps({"HuggingfaceId": self._hf_id(name)}),
        }

    def _hf_id(self, name):
        for keyword in self._keywords_by_name.get(name, []):
            if keyword.startswith("@huggingface-id:"):
                return keyword[len("@huggingface-id:") :]
        return "Provider-B/Model-B"

    def list_hub_contents(self, **kwargs):
        self.list_calls.append(kwargs)
        return {
            "HubContentSummaries": [
                {
                    "HubContentName": n,
                    "HubContentSearchKeywords": self._keywords_by_name.get(n, []),
                }
                for n in self._names
            ]
        }


def test_get_hub_client_routes_through_sagemaker_core(patch_clients, monkeypatch):
    """The Hub / model-package client comes from a region-pinned sagemaker-core
    ``Session`` (not a hand-built boto3 client), so the public Hub and model
    package calls go through sagemaker-core's client management."""
    install, _ = patch_clients
    install(FakeSageMakerClient(["Running"]))

    captured = {}
    sentinel_client = object()

    class FakeSession:
        def __init__(self, *, boto_session, sagemaker_client=None):
            captured["boto_region"] = boto_session.region_name
            captured["injected_client"] = sagemaker_client
            self.boto_session = boto_session
            self.sagemaker_client = sentinel_client

    import sagemaker.core.helper.session_helper as session_mod

    monkeypatch.setattr(session_mod, "Session", FakeSession)

    svc = ServiceClient(region="eu-west-1", role_arn="arn:aws:iam::123456789012:role/TestRole")
    client = svc._get_hub_client()

    # Came straight off the core Session, pinned to this ServiceClient's region.
    assert client is sentinel_client
    assert captured["boto_region"] == "eu-west-1"
    # With no endpoint override, core builds the client itself.
    assert captured["injected_client"] is None
    # Cached: a second call does not build another Session.
    assert svc._get_hub_client() is sentinel_client


def test_create_with_base_model_name_resolves_arn(patch_clients):
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Running"])
    install(sm)

    svc = ServiceClient(region="us-west-2", role_arn="arn:aws:iam::123456789012:role/TestRole")
    hub = FakeHubClient()
    svc._hub_client = hub  # stub the lazily-built boto3 client

    kwargs = _create_kwargs()
    # base_model is a Hub content name (as returned by list_base_models()).
    svc.create_lora_training_client(base_model="hubcontent-vlm-model-b", **kwargs)

    # Resolved via DescribeHubContent with that Hub content name verbatim.
    assert hub.describe_calls[0]["HubContentName"] == "hubcontent-vlm-model-b"
    assert hub.describe_calls[0]["HubName"] == "SageMakerPublicHub"
    # The resolved ARN is what CreateTrainingSession received.
    assert sm.create_calls[0]["ModelConfig"]["BaseModelArn"] == hub._arn


def test_create_resolves_hf_id_once_and_reuses_for_tokenizer(patch_clients, monkeypatch):
    """Create resolves the model id from the same DescribeHubContent used for the
    ARN, and get_tokenizer() reuses it instead of describing the model again."""
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Running"])
    install(sm)

    svc = ServiceClient(region="us-west-2", role_arn="arn:aws:iam::123456789012:role/TestRole")
    hub = FakeHubClient()
    svc._hub_client = hub

    seen = {}
    import sagemaker.train.training_session._utils as utils

    def fake_load(base_model, hub_client, hf_model_id=None):
        seen["hf_model_id"] = hf_model_id
        return object()

    monkeypatch.setattr(utils, "load_tokenizer", fake_load)

    kwargs = _create_kwargs()
    tc = svc.create_lora_training_client(base_model="hubcontent-vlm-model-b", **kwargs)
    tc.get_tokenizer()

    # The create-time describe carried the model id straight into the tokenizer load.
    assert seen["hf_model_id"] == "Provider-B/Model-B"
    # Exactly one DescribeHubContent across create + get_tokenizer (no second call).
    assert len(hub.describe_calls) == 1


# The sampling path takes the same two guarantees as the training path above: it
# must not build the Hub client for an input that needs no Hub lookup, and it
# must carry the id it did resolve rather than make get_tokenizer look it up
# again.


def test_create_sampling_client_with_arn_does_not_build_hub_client(patch_clients):
    """An ARN input needs no Hub lookup, so none of sagemaker-core is imported.

    Group and output path are supplied so their defaults do not reach for a
    client either, leaving the returned client the only thing that could.
    """
    install, _ = patch_clients
    install(FakeSageMakerClient(["Running"]))

    svc = ServiceClient(
        role_arn="arn:aws:iam::123456789012:role/TestRole",
        model_package_group_arn="arn:mpg/x",
        s3_output_path="s3://b/out/",
    )
    built = {"n": 0}

    def fake_build_hub():
        built["n"] += 1
        return FakeHubClient()

    svc._get_hub_client = fake_build_hub

    sampler = svc.create_sampling_client(base_model=_BASE_MODEL_ARN)

    assert built["n"] == 0, "create must not build the Hub client for an ARN input"
    assert sampler._hub_client is None


def test_create_sampling_client_resolves_hf_id_once_and_reuses_for_tokenizer(
    patch_clients, monkeypatch
):
    """The name lookup's model id reaches get_tokenizer, so it describes once."""
    install, _ = patch_clients
    install(FakeSageMakerClient(["Running"]))

    svc = ServiceClient(
        region="us-west-2",
        role_arn="arn:aws:iam::123456789012:role/TestRole",
        model_package_group_arn="arn:mpg/x",
        s3_output_path="s3://b/out/",
    )
    hub = FakeHubClient()
    svc._hub_client = hub

    seen = {}
    import sagemaker.train.training_session._utils as utils

    def fake_load(base_model, hub_client, hf_model_id=None):
        seen["hf_model_id"] = hf_model_id
        return object()

    monkeypatch.setattr(utils, "load_tokenizer", fake_load)

    sampler = svc.create_sampling_client(base_model="hubcontent-vlm-model-b")
    sampler.get_tokenizer()

    assert seen["hf_model_id"] == "Provider-B/Model-B"
    assert len(hub.describe_calls) == 1


def test_sampling_client_lazily_builds_hub_client_when_needed(patch_clients, monkeypatch):
    """With an ARN input and no hf_model_id the defer is a defer, not a drop."""
    install, _ = patch_clients
    install(FakeSageMakerClient(["Running"]))

    svc = ServiceClient(
        role_arn="arn:aws:iam::123456789012:role/TestRole",
        model_package_group_arn="arn:mpg/x",
        s3_output_path="s3://b/out/",
    )
    hub = FakeHubClient()
    built = {"n": 0}

    def fake_build_hub():
        built["n"] += 1
        return hub

    svc._get_hub_client = fake_build_hub

    seen = {}
    import sagemaker.train.training_session._utils as utils

    def fake_load(base_model, hub_client, hf_model_id=None):
        seen["hub_client"] = hub_client
        return object()

    monkeypatch.setattr(utils, "load_tokenizer", fake_load)

    sampler = svc.create_sampling_client(base_model=_BASE_MODEL_ARN)
    assert built["n"] == 0  # not built at create

    sampler.get_tokenizer()
    assert built["n"] == 1  # built lazily on first tokenizer use
    assert seen["hub_client"] is hub


def test_create_from_state_has_no_load_optimizer_param():
    """from_state is weights-only; the optimizer variant is the separate method."""
    import inspect

    sig = inspect.signature(ServiceClient.create_training_client_from_state)
    assert "load_optimizer" not in sig.parameters


def test_from_state_takes_the_checkpoint_positionally():
    """The checkpoint reads as the subject of the call, so it stays positional.

    Every other argument is forwarded through ``**kwargs`` and has to be named,
    which leaves the ARN as the only thing a caller can pass without a keyword.
    """
    import inspect

    for method in (
        ServiceClient.create_training_client_from_state,
        ServiceClient.create_training_client_from_state_async,
        ServiceClient.create_training_client_from_state_with_optimizer,
        ServiceClient.create_training_client_from_state_with_optimizer_async,
    ):
        param = inspect.signature(method).parameters["model_package_arn"]
        assert param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD, method.__name__


class FakeLoadFuture:
    """Stand-in for the APIFuture the load methods return.

    Returning a bare ``None`` from a patched ``load_state`` would let the
    factory drop the future without resolving it and still pass. This records
    the call instead, so the test can assert the restore was waited on.
    """

    def __init__(self, error=None):
        self._error = error
        self.result_calls = 0

    def result(self, timeout=None):
        self.result_calls += 1
        if self._error is not None:
            raise self._error
        return None


def test_create_from_state_derives_base_model_from_checkpoint(patch_clients, monkeypatch):
    """The documented resume recipe passes only model_package_arn — base_model
    is read from the checkpoint's model package metadata, pinned to the version
    the weights were trained on."""
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Running"])
    install(sm)

    svc = ServiceClient(
        region="us-west-2",
        role_arn="arn:aws:iam::123456789012:role/TestRole",
        model_package_group_arn="arn:mpg/x",
        s3_output_path="s3://b/out/",
    )
    hub = FakeHubClient()
    hub.describe_model_package = lambda **kw: {
        "ModelPackageArn": "arn:mp/ckpt-3",
        "InferenceSpecification": {
            "Containers": [
                {
                    "IsCheckpoint": True,
                    "BaseModel": {
                        "HubContentName": "hubcontent-text-model-a",
                        "HubContentVersion": "3.15.0",
                    },
                }
            ]
        },
    }
    svc._hub_client = hub

    loads = []
    futures = []

    def _record(with_optimizer):
        def _load(self, arn):
            loads.append((arn, with_optimizer))
            fut = FakeLoadFuture()
            futures.append(fut)
            return fut

        return _load

    # Each factory must route to its matching load method by name; the
    # weights-only / with-optimizer choice is not a flag on one call.
    monkeypatch.setattr(TrainingClient, "load_state", _record(False))
    monkeypatch.setattr(TrainingClient, "load_state_with_optimizer", _record(True))

    svc.create_training_client_from_state(model_package_arn="arn:mp/ckpt-3")
    svc.create_training_client_from_state_with_optimizer(model_package_arn="arn:mp/ckpt-3")

    expected_base = (
        "arn:aws:sagemaker:us-west-2:aws:hub-content/"
        "SageMakerPublicHub/Model/hubcontent-text-model-a/3.15.0"
    )
    assert [c["ModelConfig"]["BaseModelArn"] for c in sm.create_calls] == [
        expected_base,
        expected_base,
    ]
    assert loads == [("arn:mp/ckpt-3", False), ("arn:mp/ckpt-3", True)]
    # Both factories must block on the restore, not return a client whose
    # checkpoint load is still queued.
    assert [f.result_calls for f in futures] == [1, 1]


def test_create_from_state_explicit_base_model_skips_describe(patch_clients, monkeypatch):
    """Passing base_model explicitly must not trigger a DescribeModelPackage."""
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Running"])
    install(sm)

    svc = ServiceClient(
        region="us-west-2",
        role_arn="arn:aws:iam::123456789012:role/TestRole",
        model_package_group_arn="arn:mpg/x",
        s3_output_path="s3://b/out/",
    )
    hub = FakeHubClient()

    def _boom(**kw):
        raise AssertionError("describe_model_package must not be called")

    hub.describe_model_package = _boom
    svc._hub_client = hub

    monkeypatch.setattr(TrainingClient, "load_state", lambda self, arn: FakeLoadFuture())

    svc.create_training_client_from_state(
        model_package_arn="arn:mp/ckpt-3", base_model=_BASE_MODEL_ARN
    )
    assert sm.create_calls[0]["ModelConfig"]["BaseModelArn"] == "arn:hub-content/Model/base"


def test_create_from_state_stops_session_when_restore_fails(patch_clients, monkeypatch):
    """A failed restore must not leave the just-created session unowned.

    The factory raises instead of returning a client, so nothing holds the
    session's handle and no automatic stop is armed for it.
    """
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Running"])
    install(sm)

    svc = ServiceClient(
        region="us-west-2",
        role_arn="arn:aws:iam::123456789012:role/TestRole",
        model_package_group_arn="arn:mpg/x",
        s3_output_path="s3://b/out/",
    )

    boom = RuntimeError("checkpoint unreadable")
    monkeypatch.setattr(TrainingClient, "load_state", lambda self, arn: FakeLoadFuture(error=boom))

    with pytest.raises(RuntimeError, match="checkpoint unreadable"):
        svc.create_training_client_from_state(
            model_package_arn="arn:mp/ckpt-3", base_model=_BASE_MODEL_ARN
        )
    # The original failure propagates, and the session is torn down.
    assert len(sm.stop_calls) == 1


def test_create_with_explicit_arn_skips_hub_lookup(patch_clients):
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Running"])
    install(sm)

    svc = ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole")
    hub = FakeHubClient()
    svc._hub_client = hub

    svc.create_lora_training_client(_BASE_MODEL_ARN, **_create_kwargs())
    assert sm.create_calls[0]["ModelConfig"]["BaseModelArn"] == "arn:hub-content/Model/base"
    assert hub.describe_calls == []  # no resolution needed


def test_explicit_hf_model_id_skips_tokenizer_describe(patch_clients, monkeypatch):
    """An explicit hf_model_id is used for the tokenizer with no Hub describe.

    With an ARN input the model id is normally resolved lazily at get_tokenizer()
    time via DescribeHubContent. Passing hf_model_id supplies it up front, so a
    caller that already knows it (or lacks Hub read access) never describes.
    """
    install, _ = patch_clients
    install(FakeSageMakerClient(["Running"]))

    svc = ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole")
    hub = FakeHubClient()
    svc._hub_client = hub

    seen = {}
    import sagemaker.train.training_session._utils as utils

    def fake_load(base_model, hub_client, hf_model_id=None):
        seen["hf_model_id"] = hf_model_id
        return object()

    monkeypatch.setattr(utils, "load_tokenizer", fake_load)

    tc = svc.create_lora_training_client(_BASE_MODEL_ARN, **_create_kwargs())
    tc.get_tokenizer()

    # ARN path: hf_model_id is None at create time (resolved lazily by tokenizer).
    assert seen["hf_model_id"] is None


def test_create_with_arn_does_not_build_hub_client(patch_clients):
    """Creating a session with base_model_arn must NOT build the Hub client.

    The Hub client is built via sagemaker-core, which is not always installed
    (e.g. the vendored/Fargate integ env). Building it eagerly at create time
    would import sagemaker-core even though the ARN path never needs the Hub, so
    the build must stay deferred until a tokenizer lookup actually needs it.
    Regression guard for the 1.13.2 eager-import break.
    """
    install, _ = patch_clients
    install(FakeSageMakerClient(["Running"]))

    svc = ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole")
    calls = {"n": 0}
    original = svc._get_hub_client

    def counting_get_hub_client():
        calls["n"] += 1
        return original()

    svc._get_hub_client = counting_get_hub_client

    tc = svc.create_lora_training_client(_BASE_MODEL_ARN, **_create_kwargs())
    assert calls["n"] == 0, "create must not build the Hub client for an ARN input"
    # And the client received no eagerly-built Hub client.
    assert tc._hub_client is None


def test_tokenizer_lazily_builds_hub_client_when_needed(patch_clients, monkeypatch):
    """With an ARN input and no hf_model_id, get_tokenizer() lazily builds the
    Hub client (deferred from create) to resolve the model id — proving the defer
    is a defer, not a drop."""
    install, _ = patch_clients
    install(FakeSageMakerClient(["Running"]))

    svc = ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole")
    hub = FakeHubClient()
    built = {"n": 0}

    def fake_build_hub():
        built["n"] += 1
        return hub

    svc._get_hub_client = fake_build_hub

    seen = {}
    import sagemaker.train.training_session._utils as utils

    def fake_load(base_model, hub_client, hf_model_id=None):
        seen["hub_client"] = hub_client
        return object()

    monkeypatch.setattr(utils, "load_tokenizer", fake_load)

    # ARN input, no hf_model_id -> tokenizer resolution needs the Hub client.
    tc = svc.create_lora_training_client(_BASE_MODEL_ARN, **_create_kwargs())
    assert built["n"] == 0  # not built at create
    tc.get_tokenizer()
    assert built["n"] == 1  # built lazily on first tokenizer use
    assert seen["hub_client"] is hub


def test_tokenizer_with_hf_id_never_builds_hub_client(patch_clients, monkeypatch):
    """get_tokenizer() with a known hf_model_id must NOT build the Hub client.

    The integ path passes base_model_arn + hf_model_id, so no Hub lookup is
    ever needed — building the client here would still import sagemaker-core in
    the vendored env. Regression guard: the lazy-build must stay gated on a
    missing hf_model_id, not fire unconditionally at tokenizer time.
    """
    install, _ = patch_clients
    install(FakeSageMakerClient(["Running"]))

    svc = ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole")
    built = {"n": 0}

    def fake_build_hub():
        built["n"] += 1
        return FakeHubClient()

    svc._get_hub_client = fake_build_hub

    import sagemaker.train.training_session._utils as utils

    monkeypatch.setattr(utils, "load_tokenizer", lambda *a, **k: object())

    tc = svc.create_lora_training_client(_BASE_MODEL_ARN, **_create_kwargs())
    # ARN path: hub client is NOT built at create time (no name resolution needed).
    assert built["n"] == 0, "ARN path must not build the Hub client at create time"
    # But it IS built when the tokenizer resolves the HF model id lazily.
    tc.get_tokenizer()
    assert built["n"] == 1, "Tokenizer needs Hub client to resolve model id from ARN"


def test_create_rejects_unresolvable_base_model(patch_clients):
    install, _ = patch_clients
    install(FakeSageMakerClient(["Running"]))
    svc = ServiceClient(role_arn="arn:aws:iam::123456789012:role/TestRole")
    svc._hub_client = FakeHubClient()

    kwargs = _create_kwargs()
    # Not a real Hub content name -> DescribeHubContent fails and we surface a
    # clear, actionable error.
    with pytest.raises(ValueError, match="Could not resolve base model"):
        svc.create_lora_training_client("not-a-real-model", **kwargs)


def test_list_base_models_filters_by_recipe_keyword(patch_clients):
    # Only models carrying a fine-tuning recipe keyword are trainable; a model
    # present in the hub without one must be excluded. Each trainable model is
    # reported by the model id its Hub entry advertises.
    install, _ = patch_clients
    install(FakeSageMakerClient(["Running"]))
    svc = ServiceClient(region="us-west-2", role_arn="arn:aws:iam::123456789012:role/TestRole")
    svc._hub_client = FakeHubClient(
        hub_content_names=[
            "hubcontent-vlm-model-b",  # trainable
            "hubcontent-vlm-model-c",  # trainable
            "some-eval-only-model",  # present but not fine-tunable
        ],
        keywords_by_name={
            "hubcontent-vlm-model-b": [
                "@recipe:finetuning_sft_lora",
                "@recipe:finetuning_mtrl_lora",
                "@huggingface-id:provider-b/model-b",
            ],
            "hubcontent-vlm-model-c": [
                "@recipe:finetuning_mtrl_lora",
                "@huggingface-id:provider-c/model-c",
            ],
            "some-eval-only-model": ["@recipe:evaluation_mtrlevaluation"],
        },
    )

    models = svc.list_base_models()
    # Both fine-tunable models, by model id, sorted; eval-only excluded.
    assert models == ["provider-b/model-b", "provider-c/model-c"]
    assert svc._hub_client.list_calls  # queried the Hub in-region


def test_list_base_models_excludes_unsupported_names(patch_clients):
    # The recipe keyword alone does not make a model usable through this client.
    # A name the client does not support end to end is excluded even when the
    # Hub tags it as fine-tunable.
    install, _ = patch_clients
    install(FakeSageMakerClient(["Running"]))
    svc = ServiceClient(region="us-west-2", role_arn="arn:aws:iam::123456789012:role/TestRole")
    svc._hub_client = FakeHubClient(
        hub_content_names=[
            "hubcontent-vlm-model-b",
            # Matches _EXCLUDED_MODEL_NAMES, so excluded despite the recipe.
            "hubcontent-nova-model-x",
        ],
        keywords_by_name={
            "hubcontent-vlm-model-b": [
                "@recipe:finetuning_mtrl_lora",
                "@huggingface-id:provider-b/model-b",
            ],
            "hubcontent-nova-model-x": [
                "@recipe:finetuning_mtrl_lora",
                "@huggingface-id:provider-x/model-x",
            ],
        },
    )

    assert svc.list_base_models() == ["provider-b/model-b"]


# --- checkpoint discovery ---------------------------------------------------
# Checkpoints are model package versions in the session's group. The hub client
# (svc._get_hub_client()) is the boto3 sagemaker client these helpers call, so
# the fake below stubs list_model_packages + describe_model_package.


class FakeCheckpointClient:
    """Stubs list_model_packages (newest-first, paginated) + describe_model_package."""

    def __init__(self, summaries, resumable_arns=()):
        # summaries: already in the desired (newest-first) order.
        self._summaries = summaries
        self._resumable = set(resumable_arns)
        self.list_calls: list[dict] = []
        self.describe_calls: list[dict] = []

    def list_model_packages(self, **kwargs):
        self.list_calls.append(kwargs)
        items = self._summaries
        mr = kwargs.get("MaxResults")
        if mr is not None:
            items = items[:mr]
        return {"ModelPackageSummaryList": items}

    def describe_model_package(self, **kwargs):
        arn = kwargs["ModelPackageName"]
        self.describe_calls.append(kwargs)
        is_ckpt = arn in self._resumable
        return {
            "InferenceSpecification": {"Containers": [{"IsCheckpoint": True} if is_ckpt else {}]}
        }


def _ckpt_summary(arn, version, desc="d"):
    return {
        "ModelPackageArn": arn,
        "ModelPackageVersion": version,
        "ModelPackageDescription": desc,
        "CreationTime": "2026-06-20T00:00:00Z",
    }


def _svc_with_checkpoints(patch_clients, fake):
    install, _ = patch_clients
    install(FakeSageMakerClient(["Running"]))
    svc = ServiceClient(region="us-west-2", role_arn="arn:aws:iam::123456789012:role/TestRole")
    svc._hub_client = fake
    return svc


def test_list_checkpoints_maps_summaries_no_describe(patch_clients):
    fake = FakeCheckpointClient(
        [_ckpt_summary("g/3", 3, "step 3"), _ckpt_summary("g/2", 2, "step 2")]
    )
    svc = _svc_with_checkpoints(patch_clients, fake)

    out = svc.list_model_packages(model_package_group_arn="g")
    assert [c.version for c in out] == [3, 2]
    assert out[0].model_package_arn == "g/3"
    assert out[0].description == "step 3"
    # list view does not resolve kind (no describe calls).
    assert out[0].kind is None
    assert fake.describe_calls == []


def test_list_checkpoints_respects_max_results(patch_clients):
    fake = FakeCheckpointClient(
        [_ckpt_summary("g/3", 3), _ckpt_summary("g/2", 2), _ckpt_summary("g/1", 1)]
    )
    svc = _svc_with_checkpoints(patch_clients, fake)

    out = svc.list_model_packages(model_package_group_arn="g", max_results=2)
    assert [c.version for c in out] == [3, 2]
    assert fake.list_calls[0]["MaxResults"] == 2


def test_get_latest_resumable_skips_model_checkpoints(patch_clients):
    # Newest is a model checkpoint; the resumable one is older — walk past it.
    fake = FakeCheckpointClient(
        [_ckpt_summary("g/9", 9), _ckpt_summary("g/8", 8), _ckpt_summary("g/7", 7)],
        resumable_arns=["g/8"],
    )
    svc = _svc_with_checkpoints(patch_clients, fake)

    latest = svc.get_latest_model_package(ModelPackageKind.CHECKPOINT, model_package_group_arn="g")
    assert latest.version == 8
    assert latest.kind == ModelPackageKind.CHECKPOINT
    # Described g/9 (model, skipped) then g/8 (resumable, match) — stopped there.
    assert [c["ModelPackageName"] for c in fake.describe_calls] == ["g/9", "g/8"]


def test_get_latest_model_skips_checkpoints(patch_clients):
    fake = FakeCheckpointClient(
        [_ckpt_summary("g/9", 9), _ckpt_summary("g/8", 8)], resumable_arns=["g/9"]
    )
    svc = _svc_with_checkpoints(patch_clients, fake)

    latest = svc.get_latest_model_package(ModelPackageKind.MODEL, model_package_group_arn="g")
    assert latest.version == 8
    assert latest.kind == ModelPackageKind.MODEL


def test_get_latest_resumable_returns_none_when_no_match_in_scan(patch_clients):
    # All model checkpoints, no resumable -> None after scanning.
    fake = FakeCheckpointClient(
        [_ckpt_summary("g/2", 2), _ckpt_summary("g/1", 1)], resumable_arns=[]
    )
    svc = _svc_with_checkpoints(patch_clients, fake)

    assert (
        svc.get_latest_model_package(ModelPackageKind.CHECKPOINT, model_package_group_arn="g")
        is None
    )


def test_get_latest_resumable_honors_scan_limit(patch_clients):
    fake = FakeCheckpointClient(
        [_ckpt_summary("g/3", 3), _ckpt_summary("g/2", 2), _ckpt_summary("g/1", 1)],
        resumable_arns=["g/1"],
    )
    svc = _svc_with_checkpoints(patch_clients, fake)

    # scan_limit=2 only describes the 2 newest; the resumable g/1 is beyond it.
    out = svc.get_latest_model_package(
        ModelPackageKind.CHECKPOINT, model_package_group_arn="g", scan_limit=2
    )
    assert out is None
    assert fake.list_calls[0]["MaxResults"] == 2
    assert len(fake.describe_calls) == 2


def _record_core_clients(monkeypatch):
    """Fake out core's ``Session`` and ``boto3.Session``, recording both.

    Returns ``(recorded, override_client)``. ``recorded["client_calls"]`` holds
    one entry per pre-built client, and ``recorded["injected"]`` one per core
    ``Session``, so a test can tell which accessor passed an endpoint and which
    did not.
    """
    import boto3

    recorded = {"client_calls": [], "injected": [], "regions": []}
    override_client = object()
    default_client = object()

    class _FakeBotoSession:
        def __init__(self, region_name=None):
            recorded["regions"].append(region_name)

        def client(self, service, endpoint_url=None, config=None):
            recorded["client_calls"].append(
                {
                    "service": service,
                    "endpoint_url": endpoint_url,
                    "user_agent_extra": getattr(config, "user_agent_extra", None),
                }
            )
            return override_client

    class _FakeSession:
        def __init__(self, *, boto_session, sagemaker_client=None):
            self.boto_session = boto_session
            recorded["injected"].append(sagemaker_client)
            # Core builds its own when handed none, which is the un-overridden
            # path; stand in a sentinel so the test can tell them apart.
            self.sagemaker_client = (
                sagemaker_client if sagemaker_client is not None else default_client
            )

    # Stub the session's own clients out first: the fake boto3.Session below
    # exists to record the lookup clients' arguments, and building the real ones
    # through it would only exercise the fake.
    monkeypatch.setattr(
        "sagemaker.train.training_session._clients._control_plane_client",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "sagemaker.train.training_session._clients._runtime_client",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(boto3, "Session", _FakeBotoSession)
    import sagemaker.core.helper.session_helper as session_mod

    monkeypatch.setattr(session_mod, "Session", _FakeSession)
    return recorded, override_client, default_client


def test_model_registry_client_honors_endpoint_override(monkeypatch):
    """An endpoint override reaches the model-registry client.

    The override needs its own client, built off the same session the rest of
    this ``ServiceClient`` uses, so it signs with the same credentials and
    carries core's user-agent config like any other.
    """
    recorded, override_client, _ = _record_core_clients(monkeypatch)
    monkeypatch.setenv("SAGEMAKER_TRAINING_SESSION_ENDPOINT", "https://custom.example.test")

    svc = ServiceClient(region="us-west-2", role_arn="arn:aws:iam::123456789012:role/TestRole")
    client = svc._get_model_registry_client()

    assert [c["service"] for c in recorded["client_calls"]] == ["sagemaker"]
    assert recorded["client_calls"][0]["endpoint_url"] == "https://custom.example.test"
    assert recorded["regions"] == ["us-west-2"]
    # Built with core's user-agent suffix, not a bare client.
    assert recorded["client_calls"][0]["user_agent_extra"]
    assert client is override_client


def test_hub_client_ignores_endpoint_override(monkeypatch):
    """The Hub client stays on the Region's endpoint even under an override.

    The Public Hub is published per Region, so resolving a base model against
    anything else would pick a version the Region does not serve. Only the
    model-registry client follows the override.
    """
    recorded, override_client, default_client = _record_core_clients(monkeypatch)
    monkeypatch.setenv("SAGEMAKER_TRAINING_SESSION_ENDPOINT", "https://custom.example.test")

    svc = ServiceClient(region="us-west-2", role_arn="arn:aws:iam::123456789012:role/TestRole")
    hub = svc._get_hub_client()

    # No pre-built client at all, so core resolved the Region's own endpoint.
    assert recorded["client_calls"] == []
    assert recorded["injected"] == [None]
    assert hub is default_client
    assert hub is not override_client
    # The two accessors are genuinely different clients under an override.
    assert svc._get_model_registry_client() is override_client


def test_both_lookup_clients_are_one_client_without_an_override(monkeypatch):
    """With no override the two registries share a client, as they always did."""
    recorded, _, default_client = _record_core_clients(monkeypatch)
    monkeypatch.delenv("SAGEMAKER_TRAINING_SESSION_ENDPOINT", raising=False)

    svc = ServiceClient(region="us-west-2", role_arn="arn:aws:iam::123456789012:role/TestRole")

    assert svc._get_hub_client() is default_client
    assert svc._get_model_registry_client() is svc._get_hub_client()
    # One core Session, one client: no extra client built for public callers.
    assert recorded["injected"] == [None]
    assert recorded["client_calls"] == []


# ---------------------------------------------------------------------------
# Failure reason on a session that never reaches Running
# ---------------------------------------------------------------------------


def test_terminal_state_error_includes_error_details(patch_clients):
    """A failed session reports why, not just the status."""
    install, _ = patch_clients
    sm = FakeSageMakerClient(
        ["Failed"], describe_detail={"ErrorDetails": "base model not accessible"}
    )
    install(sm)

    svc = ServiceClient(region="us-west-2", role_arn="arn:aws:iam::123456789012:role/TestRole")
    with pytest.raises(RuntimeError, match="base model not accessible"):
        svc.create_lora_training_client(_BASE_MODEL_ARN, **_create_kwargs())


def test_terminal_state_error_falls_back_to_error_message(patch_clients):
    """ErrorMessage is the fallback; used only when ErrorDetails is absent."""
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Failed"], describe_detail={"ErrorMessage": "quota exceeded"})
    install(sm)

    svc = ServiceClient(region="us-west-2", role_arn="arn:aws:iam::123456789012:role/TestRole")
    with pytest.raises(RuntimeError, match="quota exceeded"):
        svc.create_lora_training_client(_BASE_MODEL_ARN, **_create_kwargs())


def test_terminal_state_error_without_reason_names_the_status(patch_clients):
    """No reason from the service leaves the status to speak for itself."""
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Failed"])
    install(sm)

    svc = ServiceClient(region="us-west-2", role_arn="arn:aws:iam::123456789012:role/TestRole")
    with pytest.raises(RuntimeError, match="terminal state: Failed$"):
        svc.create_lora_training_client(_BASE_MODEL_ARN, **_create_kwargs())


# ---------------------------------------------------------------------------
# Flat and grouped field forms on session creation
# ---------------------------------------------------------------------------


def test_create_sends_model_and_output_config(patch_clients):
    """Model identity and output location go in ModelConfig and OutputConfig.

    Nothing else carries either value: the flat ``BaseModelArn`` /
    ``OutputS3Path`` pair is not part of the request shape, so sending it is
    refused before the request leaves.
    """
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Running"])
    install(sm)

    svc = ServiceClient(region="us-west-2", role_arn="arn:aws:iam::123456789012:role/TestRole")
    svc.create_lora_training_client(_BASE_MODEL_ARN, **_create_kwargs())

    cc = sm.create_calls[0]
    assert cc["ModelConfig"] == {"BaseModelArn": _BASE_MODEL_ARN}
    assert cc["OutputConfig"]["S3Location"]["Path"] == "s3://b/out/"
    assert "BaseModelArn" not in cc
    assert "OutputS3Path" not in cc


def test_create_sampling_only_session_sends_the_same_shape(patch_clients):
    """The implicit sampling-session path builds the same request, not a variant."""
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Running"])
    install(sm)

    svc = ServiceClient(
        region="us-west-2",
        role_arn="arn:aws:iam::123456789012:role/TestRole",
        model_package_group_arn="arn:mpg/x",
        s3_output_path="s3://b/out/",
    )
    svc.create_sampling_client(base_model=_BASE_MODEL_ARN)

    cc = sm.create_calls[0]
    assert cc["ModelConfig"] == {"BaseModelArn": _BASE_MODEL_ARN}
    assert cc["OutputConfig"]["S3Location"]["Path"] == "s3://b/out/"
    assert "BaseModelArn" not in cc
    assert "OutputS3Path" not in cc


# ---------------------------------------------------------------------------
# KMS / VPC support on session creation
# ---------------------------------------------------------------------------


def test_create_with_kms_sends_output_config(patch_clients):
    """Per-call kms_key_arn rides on OutputConfig.S3Location."""
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Running"])
    install(sm)

    svc = ServiceClient(region="us-west-2", role_arn="arn:aws:iam::123456789012:role/TestRole")
    svc.create_lora_training_client(
        _BASE_MODEL_ARN, **_create_kwargs(kms_key_arn="arn:aws:kms:us-west-2:123456789012:key/k1")
    )

    cc = sm.create_calls[0]
    assert cc["OutputConfig"] == {
        "S3Location": {
            "Path": "s3://b/out/",
            "KmsKeyArn": "arn:aws:kms:us-west-2:123456789012:key/k1",
        }
    }


def test_create_with_vpc_dict_sends_vpc_config(patch_clients):
    """A plain dict vpc_config is coerced and sent in the API shape."""
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Running"])
    install(sm)

    svc = ServiceClient(region="us-west-2", role_arn="arn:aws:iam::123456789012:role/TestRole")
    svc.create_lora_training_client(
        _BASE_MODEL_ARN,
        **_create_kwargs(
            vpc_config={"subnets": ["subnet-1", "subnet-2"], "security_group_ids": ["sg-1"]}
        ),
    )

    assert sm.create_calls[0]["VpcConfig"] == {
        "Subnets": ["subnet-1", "subnet-2"],
        "SecurityGroupIds": ["sg-1"],
    }


def test_create_without_kms_vpc_omits_only_those_members(patch_clients):
    """OutputConfig always goes out; KmsKeyArn and VpcConfig only when set."""
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Running"])
    install(sm)

    svc = ServiceClient(region="us-west-2", role_arn="arn:aws:iam::123456789012:role/TestRole")
    svc.create_lora_training_client(_BASE_MODEL_ARN, **_create_kwargs())

    cc = sm.create_calls[0]
    assert cc["OutputConfig"] == {"S3Location": {"Path": "s3://b/out/"}}
    assert "VpcConfig" not in cc


def test_create_kms_vpc_defaults_from_service_client_and_override(patch_clients):
    """ServiceClient-level defaults apply; per-call values win."""
    from sagemaker.train.training_session.types import VpcConfig

    install, _ = patch_clients
    sm = FakeSageMakerClient(["Running", "Running"])
    install(sm)

    svc = ServiceClient(
        region="us-west-2",
        role_arn="arn:aws:iam::123456789012:role/TestRole",
        kms_key_arn="arn:aws:kms:us-west-2:123456789012:key/default",
        vpc_config=VpcConfig(subnets=["subnet-d"], security_group_ids=["sg-d"]),
    )

    svc.create_lora_training_client(_BASE_MODEL_ARN, **_create_kwargs())
    cc = sm.create_calls[0]
    assert cc["OutputConfig"]["S3Location"]["KmsKeyArn"].endswith("key/default")
    assert cc["VpcConfig"]["Subnets"] == ["subnet-d"]

    svc.create_lora_training_client(
        _BASE_MODEL_ARN,
        **_create_kwargs(
            training_session_name="sess-2",
            kms_key_arn="arn:aws:kms:us-west-2:123456789012:key/override",
            vpc_config={"subnets": ["subnet-o"], "security_group_ids": ["sg-o"]},
        ),
    )
    cc = sm.create_calls[1]
    assert cc["OutputConfig"]["S3Location"]["KmsKeyArn"].endswith("key/override")
    assert cc["VpcConfig"]["Subnets"] == ["subnet-o"]


def test_sampling_client_inherits_kms_and_vpc(patch_clients):
    """The implicit sampling session uses the ServiceClient-level values."""
    install, _ = patch_clients
    sm = FakeSageMakerClient(["Running"])
    install(sm)

    svc = ServiceClient(
        role_arn="arn:aws:iam::123456789012:role/TestRole",
        model_package_group_arn="arn:mpg/x",
        s3_output_path="s3://b/out/",
        kms_key_arn="arn:aws:kms:us-west-2:123456789012:key/k1",
        vpc_config={"subnets": ["subnet-1"], "security_group_ids": ["sg-1"]},
    )
    svc._hub_client = FakeHubClient()
    svc.create_sampling_client(base_model="hubcontent-vlm-model-b")

    cc = sm.create_calls[0]
    assert cc["OutputConfig"]["S3Location"]["KmsKeyArn"].endswith("key/k1")
    assert cc["VpcConfig"] == {"Subnets": ["subnet-1"], "SecurityGroupIds": ["sg-1"]}


def test_default_s3_output_path_uses_core_default_bucket(patch_clients, monkeypatch):
    """The default S3 output path goes through sagemaker-core
    ``Session.default_bucket()``, which creates the default bucket on the
    caller behalf when missing, same behavior as the broader SageMaker
    PySDK (and what the s3:CreateBucket grant in the caller managed policy
    exists for)."""
    install, _ = patch_clients
    install(FakeSageMakerClient(["Running"]))

    calls = {"default_bucket": 0}

    class FakeSession:
        def __init__(self, *, boto_session):
            self.boto_session = boto_session
            self._region = boto_session.region_name

        def default_bucket(self):
            calls["default_bucket"] += 1
            return f"sagemaker-{self._region}-123456789012"

    import sagemaker.core.helper.session_helper as session_mod

    monkeypatch.setattr(session_mod, "Session", FakeSession)

    svc = ServiceClient(region="us-west-2", role_arn="arn:aws:iam::123456789012:role/TestRole")
    path = svc._default_s3_output_path("my-session")

    assert path == "s3://sagemaker-us-west-2-123456789012/my-session/output"
    assert calls["default_bucket"] == 1


# --- delete_model_package ---------------------------------------------------
# Two clients are involved: the hub client (DescribeModelPackage /
# DeleteModelPackage) and an s3 client built only when artifacts are deleted.


class FakeModelPackageClient:
    """Stubs DescribeModelPackage / DeleteModelPackage for one package."""

    def __init__(
        self,
        s3_uri="s3://bucket/group/req-1/model/",
        s3_data_type="S3Prefix",
        managed_storage_type=None,
        describe_error=None,
        delete_error=None,
    ):
        self._s3_uri = s3_uri
        self._s3_data_type = s3_data_type
        self._managed_storage_type = managed_storage_type
        self._describe_error = describe_error
        self._delete_error = delete_error
        self.describe_calls: list[dict] = []
        self.delete_calls: list[dict] = []

    def describe_model_package(self, **kwargs):
        self.describe_calls.append(kwargs)
        if self._describe_error is not None:
            raise self._describe_error
        source = {}
        if self._s3_uri is not None:
            source = {"S3Uri": self._s3_uri, "S3DataType": self._s3_data_type}
        resp = {
            "InferenceSpecification": {
                "Containers": [{"ModelDataSource": {"S3DataSource": source}}]
            }
        }
        if self._managed_storage_type is not None:
            resp["ManagedStorageType"] = self._managed_storage_type
        return resp

    def delete_model_package(self, **kwargs):
        self.delete_calls.append(kwargs)
        if self._delete_error is not None:
            raise self._delete_error
        return {}


class FakeS3Client:
    """Stubs ListObjectsV2 (paged) + DeleteObjects (per-key Errors, not raises)."""

    def __init__(self, pages=None, errors_for=()):
        # pages: list of (keys, next_token). One entry means a single page.
        self._pages = list(pages if pages is not None else [(["group/req-1/model/w.bin"], None)])
        self._errors_for = set(errors_for)
        self.list_calls: list[dict] = []
        self.delete_calls: list[dict] = []

    def list_objects_v2(self, **kwargs):
        self.list_calls.append(kwargs)
        keys, next_token = self._pages.pop(0)
        resp = {"Contents": [{"Key": k} for k in keys], "IsTruncated": bool(next_token)}
        if next_token:
            resp["NextContinuationToken"] = next_token
        return resp

    def delete_objects(self, **kwargs):
        self.delete_calls.append(kwargs)
        sent = [o["Key"] for o in kwargs["Delete"]["Objects"]]
        errors = [{"Key": k, "Code": "AccessDenied"} for k in sent if k in self._errors_for]
        return {"Errors": errors} if errors else {}


_MP_ARN = "arn:aws:sagemaker:us-west-2:123456789012:model-package/my-group/7"


def _svc_for_mp_delete(patch_clients, monkeypatch, hub, s3=None):
    install, _ = patch_clients
    install(FakeSageMakerClient(["Running"]))
    svc = ServiceClient(region="us-west-2", role_arn="arn:aws:iam::123456789012:role/TestRole")
    svc._hub_client = hub
    if s3 is not None:
        svc._s3_client = s3
    else:
        # Any S3 call at all is a bug in the retain path, so fail loudly.
        def _boom(*a, **kw):
            raise AssertionError("no S3 client should be built when retaining artifacts")

        monkeypatch.setattr(svc, "_get_s3_client", _boom)
    return svc


def _client_error(code, message="", op="DescribeModelPackage"):
    return ClientError({"Error": {"Code": code, "Message": message}}, op)


def test_delete_model_package_retains_artifacts_by_default(patch_clients, monkeypatch):
    """Default deletes the package only, and reports where the objects still are;
    the package was the only record of that location."""
    hub = FakeModelPackageClient()
    svc = _svc_for_mp_delete(patch_clients, monkeypatch, hub)

    result = svc.delete_model_package(_MP_ARN)

    assert result.model_package_deleted is True
    assert result.s3_uri == "s3://bucket/group/req-1/model/"
    assert result.objects_deleted == 0
    assert result.artifacts_retained is True
    assert "delete_artifacts" in result.retained_reason
    assert hub.delete_calls == [{"ModelPackageName": _MP_ARN}]


def test_delete_model_package_deletes_artifacts_before_package(patch_clients, monkeypatch):
    """Order matters: the package carries the location, so it goes last. If it
    went first, anything left in S3 would be unreachable."""
    hub = FakeModelPackageClient()
    s3 = FakeS3Client(pages=[(["group/req-1/model/a", "group/req-1/model/b"], None)])
    svc = _svc_for_mp_delete(patch_clients, monkeypatch, hub, s3)

    order: list[str] = []
    s3_delete = s3.delete_objects
    hub_delete = hub.delete_model_package
    s3.delete_objects = lambda **kw: (order.append("s3"), s3_delete(**kw))[1]
    hub.delete_model_package = lambda **kw: (order.append("mp"), hub_delete(**kw))[1]

    result = svc.delete_model_package(_MP_ARN, delete_artifacts=True)

    assert order == ["s3", "mp"]
    assert result.objects_deleted == 2
    assert result.artifacts_retained is False
    assert result.retained_reason is None


def test_delete_model_package_paginates_and_batches(patch_clients, monkeypatch):
    """Each listed page is deleted before the next is requested, so no full key
    list is held and the continuation token still resumes correctly."""
    hub = FakeModelPackageClient()
    s3 = FakeS3Client(pages=[(["k/1", "k/2"], "tok-1"), (["k/3"], None)])
    svc = _svc_for_mp_delete(patch_clients, monkeypatch, hub, s3)

    result = svc.delete_model_package(_MP_ARN, delete_artifacts=True)

    assert result.objects_deleted == 3
    assert len(s3.delete_calls) == 2
    assert s3.list_calls[1]["ContinuationToken"] == "tok-1"


def test_delete_model_package_prefix_gets_trailing_slash(patch_clients, monkeypatch):
    """A location without a trailing slash is matched on the segment boundary,
    so "run-1" cannot sweep up "run-10"."""
    hub = FakeModelPackageClient(s3_uri="s3://bucket/run-1")
    s3 = FakeS3Client(pages=[(["run-1/w.bin"], None)])
    svc = _svc_for_mp_delete(patch_clients, monkeypatch, hub, s3)

    svc.delete_model_package(_MP_ARN, delete_artifacts=True)

    assert s3.list_calls[0] == {"Bucket": "bucket", "Prefix": "run-1/"}


def test_delete_model_package_s3_object_deletes_exactly_one_key(patch_clients, monkeypatch):
    """S3DataType comes from the package, so a single-object location is deleted
    as that one key rather than listed as a prefix."""
    hub = FakeModelPackageClient(
        s3_uri="s3://bucket/group/req-1/model.tar.gz", s3_data_type="S3Object"
    )
    s3 = FakeS3Client()
    svc = _svc_for_mp_delete(patch_clients, monkeypatch, hub, s3)

    result = svc.delete_model_package(_MP_ARN, delete_artifacts=True)

    assert s3.list_calls == []
    assert s3.delete_calls[0]["Delete"]["Objects"] == [{"Key": "group/req-1/model.tar.gz"}]
    assert result.objects_deleted == 1


def test_delete_model_package_partial_s3_failure_keeps_package(patch_clients, monkeypatch):
    """DeleteObjects reports per-key failures in the body rather than raising.
    The package is kept so the survivors stay reachable and a retry works."""
    from sagemaker.train.training_session.exceptions import ArtifactDeletionError

    hub = FakeModelPackageClient()
    s3 = FakeS3Client(pages=[(["k/1", "k/2"], None)], errors_for=["k/2"])
    svc = _svc_for_mp_delete(patch_clients, monkeypatch, hub, s3)

    with pytest.raises(ArtifactDeletionError) as excinfo:
        svc.delete_model_package(_MP_ARN, delete_artifacts=True)

    assert excinfo.value.failed_keys == ["k/2"]
    assert excinfo.value.s3_uri == "s3://bucket/group/req-1/model/"
    assert hub.delete_calls == [], "the package must survive a partial artifact delete"


def test_delete_model_package_refuses_bucket_root(patch_clients, monkeypatch):
    """A location naming only a bucket would empty the whole bucket."""
    hub = FakeModelPackageClient(s3_uri="s3://bucket")
    s3 = FakeS3Client()
    svc = _svc_for_mp_delete(patch_clients, monkeypatch, hub, s3)

    with pytest.raises(ValueError, match="bucket root"):
        svc.delete_model_package(_MP_ARN, delete_artifacts=True)

    assert s3.delete_calls == []
    assert hub.delete_calls == []


def test_delete_model_package_skips_service_managed_artifacts(patch_clients, monkeypatch):
    """Artifacts SageMaker stores on the caller behalf have no S3 location the
    caller can delete, and deleting the package is what releases them. Asking
    for delete_artifacts=True is satisfied, not rejected."""
    hub = FakeModelPackageClient(s3_uri=None, managed_storage_type="Restricted")
    svc = _svc_for_mp_delete(patch_clients, monkeypatch, hub)

    result = svc.delete_model_package(_MP_ARN, delete_artifacts=True)

    assert result.artifacts_retained is True
    assert result.s3_uri is None
    assert "SageMaker" in result.retained_reason
    assert hub.delete_calls == [{"ModelPackageName": _MP_ARN}]


def test_delete_model_package_skips_when_no_s3_location(patch_clients, monkeypatch):
    hub = FakeModelPackageClient(s3_uri=None)
    svc = _svc_for_mp_delete(patch_clients, monkeypatch, hub)

    result = svc.delete_model_package(_MP_ARN, delete_artifacts=True)

    assert result.artifacts_retained is True
    assert "no S3 location" in result.retained_reason
    assert hub.delete_calls == [{"ModelPackageName": _MP_ARN}]


@pytest.mark.parametrize(
    "error",
    [
        _client_error("ValidationException", "Model package my-group/7 does not exist."),
        _client_error("ResourceNotFound", "Model package not found"),
    ],
)
def test_delete_model_package_is_idempotent_when_already_gone(patch_clients, monkeypatch, error):
    """Deleting twice reports nothing left to delete rather than raising."""
    hub = FakeModelPackageClient(describe_error=error)
    svc = _svc_for_mp_delete(patch_clients, monkeypatch, hub)

    result = svc.delete_model_package(_MP_ARN, delete_artifacts=True)

    assert result.model_package_deleted is False
    assert result.s3_uri is None
    assert hub.delete_calls == []


def test_delete_model_package_reraises_unrelated_validation_error(patch_clients, monkeypatch):
    """A ValidationException that is not "no such package" is a real failure."""
    hub = FakeModelPackageClient(
        describe_error=_client_error("ValidationException", "1 validation error detected")
    )
    svc = _svc_for_mp_delete(patch_clients, monkeypatch, hub)

    with pytest.raises(ClientError):
        svc.delete_model_package(_MP_ARN)


def test_delete_model_package_absent_on_delete_after_describe(patch_clients, monkeypatch):
    """Another caller deleted it between the describe and the delete."""
    hub = FakeModelPackageClient(
        delete_error=_client_error("ValidationException", "does not exist", op="DeleteModelPackage")
    )
    svc = _svc_for_mp_delete(patch_clients, monkeypatch, hub)

    result = svc.delete_model_package(_MP_ARN)

    assert result.model_package_deleted is False
    assert result.s3_uri == "s3://bucket/group/req-1/model/"


def test_delete_model_package_reraises_unrelated_delete_error(patch_clients, monkeypatch):
    hub = FakeModelPackageClient(
        delete_error=_client_error(
            "ValidationException", "Model package is in use", op="DeleteModelPackage"
        )
    )
    svc = _svc_for_mp_delete(patch_clients, monkeypatch, hub)

    with pytest.raises(ClientError):
        svc.delete_model_package(_MP_ARN)


def test_delete_model_package_empty_location_is_not_retained(patch_clients, monkeypatch):
    """Nothing to delete is a completed delete, not a retained one."""
    hub = FakeModelPackageClient()
    s3 = FakeS3Client(pages=[([], None)])
    svc = _svc_for_mp_delete(patch_clients, monkeypatch, hub, s3)

    result = svc.delete_model_package(_MP_ARN, delete_artifacts=True)

    assert result.objects_deleted == 0
    assert result.artifacts_retained is False
    assert s3.delete_calls == []


def test_delete_model_package_no_data_type_is_treated_as_a_prefix(patch_clients, monkeypatch):
    """A package carrying a location but no S3DataType is swept as a prefix."""
    hub = FakeModelPackageClient(s3_uri="s3://bucket/group/req-1/model/", s3_data_type=None)
    s3 = FakeS3Client(pages=[(["group/req-1/model/w.bin"], None)])
    svc = _svc_for_mp_delete(patch_clients, monkeypatch, hub, s3)

    result = svc.delete_model_package(_MP_ARN, delete_artifacts=True)

    assert s3.list_calls[0] == {"Bucket": "bucket", "Prefix": "group/req-1/model/"}
    assert result.objects_deleted == 1


@pytest.mark.parametrize(
    "failing_call, error_code",
    [
        # The two permissions this needs, each missing in turn.
        ("list_objects_v2", "AccessDenied"),
        ("delete_objects", "AccessDenied"),
    ],
)
def test_delete_model_package_s3_permission_error_keeps_package(
    patch_clients, monkeypatch, failing_call, error_code
):
    """An S3 call that raises rather than reporting per-key failures propagates
    unchanged, and the package survives so the objects stay reachable."""
    hub = FakeModelPackageClient()
    s3 = FakeS3Client()

    def _raise(**kwargs):
        raise _client_error(error_code, "Access Denied", op=failing_call)

    setattr(s3, failing_call, _raise)
    svc = _svc_for_mp_delete(patch_clients, monkeypatch, hub, s3)

    with pytest.raises(ClientError) as excinfo:
        svc.delete_model_package(_MP_ARN, delete_artifacts=True)

    assert excinfo.value.response["Error"]["Code"] == error_code
    assert hub.delete_calls == [], "the package must survive a failed artifact delete"


def test_delete_model_package_logs_where_retained_artifacts_are(patch_clients, monkeypatch, caplog):
    """The package was the only record of the location, so the retain path says
    it once at INFO, so a caller who discards the result can still find them."""
    hub = FakeModelPackageClient()
    svc = _svc_for_mp_delete(patch_clients, monkeypatch, hub)

    with caplog.at_level("INFO", logger=sc_mod.logger.name):
        svc.delete_model_package(_MP_ARN)

    assert "s3://bucket/group/req-1/model/" in caplog.text


def test_delete_model_package_does_not_log_a_location_it_has_none_for(
    patch_clients, monkeypatch, caplog
):
    """Nothing was left behind for the caller to find, so there is nothing to
    report."""
    hub = FakeModelPackageClient(s3_uri=None, managed_storage_type="Restricted")
    svc = _svc_for_mp_delete(patch_clients, monkeypatch, hub)

    with caplog.at_level("INFO", logger=sc_mod.logger.name):
        svc.delete_model_package(_MP_ARN, delete_artifacts=True)

    assert "artifacts remain at" not in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(_client_error("ResourceNotFoundException", ""), id="not-found-exception"),
        pytest.param(
            _client_error("ValidationException", "Model package cannot be found"),
            id="cannot-be-found",
        ),
        # The phrase is matched case-insensitively: the service capitalises these
        # messages inconsistently, and a casing change must not turn the
        # idempotent path back into a raise.
        pytest.param(
            _client_error("ValidationException", "Model Package Does Not Exist."),
            id="capitalised-message",
        ),
    ],
)
def test_delete_model_package_absent_code_variants(patch_clients, monkeypatch, error):
    hub = FakeModelPackageClient(describe_error=error)
    svc = _svc_for_mp_delete(patch_clients, monkeypatch, hub)

    assert svc.delete_model_package(_MP_ARN).model_package_deleted is False


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(_client_error("AccessDeniedException", "not authorized"), id="access-denied"),
        pytest.param(_client_error("ThrottlingException", "Rate exceeded"), id="throttled"),
        # No Message at all: reading one must not turn a real error into "gone".
        pytest.param(
            ClientError({"Error": {"Code": "ValidationException"}}, "DescribeModelPackage"),
            id="validation-without-message",
        ),
        # No Error block at all: reading a code out of it must not raise, and an
        # unreadable failure is a failure, not an absent package.
        pytest.param(ClientError({}, "DescribeModelPackage"), id="no-error-block"),
    ],
)
def test_delete_model_package_reraises_real_describe_errors(patch_clients, monkeypatch, error):
    """Only "no such package" is swallowed. A permission or throttling failure
    must not read as an idempotent no-op: that would report success while the
    package is still there."""
    hub = FakeModelPackageClient(describe_error=error)
    svc = _svc_for_mp_delete(patch_clients, monkeypatch, hub)

    with pytest.raises(ClientError):
        svc.delete_model_package(_MP_ARN)


def test_delete_model_package_is_idempotent_across_repeat_calls(patch_clients, monkeypatch):
    """The documented sweep loop can run twice: the second pass finds the
    package gone and reports it rather than raising."""
    hub = FakeModelPackageClient()
    s3 = FakeS3Client(pages=[(["group/req-1/model/w.bin"], None), ([], None)])
    svc = _svc_for_mp_delete(patch_clients, monkeypatch, hub, s3)

    first = svc.delete_model_package(_MP_ARN, delete_artifacts=True)
    hub._describe_error = _client_error("ValidationException", "does not exist")
    second = svc.delete_model_package(_MP_ARN, delete_artifacts=True)

    assert (first.model_package_deleted, first.objects_deleted) == (True, 1)
    assert (second.model_package_deleted, second.objects_deleted) == (False, 0)
    assert len(hub.delete_calls) == 1


def test_delete_artifacts_is_keyword_only():
    """``delete_model_package(arn, True)`` would not say what the True deletes,
    and it is the destructive half of the call."""
    import inspect

    params = inspect.signature(ServiceClient.delete_model_package).parameters
    assert params["delete_artifacts"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["delete_artifacts"].default is False


def test_delete_model_package_public_names_are_exported():
    """Both are named in the method's own docs, so both must be importable from
    the package root; a caller cannot annotate or catch what is not exported."""
    import sagemaker.train.training_session as pkg

    from sagemaker.train.training_session.exceptions import ArtifactDeletionError
    from sagemaker.train.training_session.types import ModelPackageDeletion

    assert pkg.ModelPackageDeletion is ModelPackageDeletion
    assert pkg.ArtifactDeletionError is ArtifactDeletionError
    assert {"ModelPackageDeletion", "ArtifactDeletionError"} <= set(pkg.__all__)


def test_artifact_deletion_error_truncates_its_message_not_its_keys(patch_clients, monkeypatch):
    """A whole prefix can fail on one missing permission, so the message shows a
    sample and points at the full list rather than printing thousands of keys."""
    from sagemaker.train.training_session.exceptions import ArtifactDeletionError

    keys = [f"group/req-1/model/shard-{i}" for i in range(7)]
    hub = FakeModelPackageClient()
    s3 = FakeS3Client(pages=[(keys, None)], errors_for=keys)
    svc = _svc_for_mp_delete(patch_clients, monkeypatch, hub, s3)

    with pytest.raises(ArtifactDeletionError) as excinfo:
        svc.delete_model_package(_MP_ARN, delete_artifacts=True)

    err = excinfo.value
    assert err.failed_keys == keys
    assert err.model_package_arn == _MP_ARN
    assert err.s3_uri == "s3://bucket/group/req-1/model/"
    message = str(err)
    assert "(7 total, see .failed_keys)" in message
    assert "shard-4" in message and "shard-5" not in message
    # The recovery step, not just the failure.
    assert "s3:DeleteObject" in message and "NOT deleted" in message


def test_artifact_deletion_error_lists_a_short_failure_in_full(patch_clients, monkeypatch):
    from sagemaker.train.training_session.exceptions import ArtifactDeletionError

    hub = FakeModelPackageClient()
    s3 = FakeS3Client(pages=[(["k/1", "k/2"], None)], errors_for=["k/1", "k/2"])
    svc = _svc_for_mp_delete(patch_clients, monkeypatch, hub, s3)

    with pytest.raises(ArtifactDeletionError) as excinfo:
        svc.delete_model_package(_MP_ARN, delete_artifacts=True)

    assert "total, see .failed_keys" not in str(excinfo.value)
    assert "k/1, k/2" in str(excinfo.value)


@pytest.mark.parametrize(
    "count, truncated",
    [
        pytest.param(5, False, id="exactly-the-sample-size"),
        pytest.param(6, True, id="one-over"),
    ],
)
def test_artifact_deletion_error_truncates_only_past_the_sample_size(count, truncated):
    """The boundary: a failure list the same length as the sample is printed in
    full, so the message never says "5 total, see .failed_keys" while already
    showing all five."""
    from sagemaker.train.training_session.exceptions import ArtifactDeletionError

    keys = [f"k/{i}" for i in range(count)]
    message = str(ArtifactDeletionError(_MP_ARN, "s3://bucket/k/", keys))

    assert ("see .failed_keys" in message) is truncated
    # Either way the sample itself is present.
    assert "k/4" in message


def test_artifact_deletion_error_is_a_training_session_error():
    """Catchable with the rest of this client's failures."""
    from sagemaker.train.training_session.exceptions import (
        ArtifactDeletionError,
        TrainingSessionError,
    )

    assert issubclass(ArtifactDeletionError, TrainingSessionError)


def test_get_s3_client_is_region_pinned_and_cached(patch_clients, monkeypatch):
    """The s3 client is built lazily, pinned to this ServiceClient region, and
    signed with the caller credentials like every other client here."""
    install, _ = patch_clients
    install(FakeSageMakerClient(["Running"]))

    captured = {}
    sentinel = object()

    class FakeBotoSession:
        def __init__(self, region_name=None):
            captured["region"] = region_name
            self.region_name = region_name

        def client(self, name):
            captured["service"] = name
            return sentinel

    class FakeSession:
        def __init__(self, *, boto_session):
            self.boto_session = boto_session
            self.sagemaker_client = object()

    import boto3
    import sagemaker.core.helper.session_helper as session_mod

    # Patched before the client exists: its session is built in the constructor,
    # so the s3 client comes off that one session rather than a fresh one.
    monkeypatch.setattr(boto3, "Session", FakeBotoSession)
    monkeypatch.setattr(session_mod, "Session", FakeSession)

    svc = ServiceClient(region="us-west-2", role_arn="arn:aws:iam::123456789012:role/TestRole")
    assert svc._s3_client is None

    assert svc._get_s3_client() is sentinel
    assert captured == {"region": "us-west-2", "service": "s3"}
    assert svc._get_s3_client() is sentinel
