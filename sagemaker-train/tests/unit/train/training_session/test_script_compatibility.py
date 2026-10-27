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
"""Existing training scripts run against this SDK unchanged.

Two guarantees, both load bearing for dropping such a script in with no edits:

* **Calling conventions match.** ``rank`` is positional-or-keyword, and the
  ``*_async`` twins of every client factory exist, because such code reaches
  for the coroutine form far more often than the blocking one.
* **Unhonoured parameters are loud.** Parameters we accept but cannot implement
  log a warning naming the parameter. Accepting them silently would let a run
  differ from what the caller asked for with nothing in the log to explain it.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from sagemaker.train.training_session import futures
from sagemaker.train.training_session import service_client as sc_mod
from sagemaker.train.training_session.sampling_client import SamplingClient
from sagemaker.train.training_session.service_client import ServiceClient
from sagemaker.train.training_session.training_client import TrainingClient

# --- calling conventions ----------------------------------------------------


def test_only_base_model_is_positional():
    """`rank` stays keyword-only on purpose.

    ``create_lora_training_client(model, 16)`` does not say what 16 is, and reads
    equally well as ``alpha``. Every surveyed call site passes ``rank=`` as a
    keyword already (8 of 8), so requiring it costs nothing in drop-in terms.
    ``base_model`` *is* positional, because 4 of those 8 pass it that way.
    """
    params = inspect.signature(ServiceClient.create_lora_training_client).parameters
    assert params["base_model"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("rank", "alpha", "seed", "user_metadata"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name


@pytest.mark.parametrize(
    "cls", [futures.APIFuture, futures.PendingAPIFuture, futures.CombinedFuture]
)
@pytest.mark.parametrize("method", ["result", "result_async"])
def test_future_wait_takes_only_timeout(cls, method):
    """``timeout`` is the one knob on a wait, and it is positional-or-keyword.

    Both ``future.result(30)`` and ``future.result(timeout=30)`` are how existing
    scripts bound a wait, so both must bind. A second parameter would also make
    ``result(30, 1)`` mean something here that it means nowhere else; the poll
    cadence is set when the future is built instead.
    """
    params = inspect.signature(getattr(cls, method)).parameters
    assert list(params) == ["self", "timeout"]
    assert params["timeout"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_positional_rank_is_rejected():
    """The ambiguous form must fail loudly, not silently bind to something."""
    sig = inspect.signature(ServiceClient.create_lora_training_client)
    with pytest.raises(TypeError):
        sig.bind(None, "Qwen/Qwen3-8B", 16)


def test_async_twin_also_rejects_positional_rank():
    """The coroutine twin must not be a back door to the positional form."""
    sig = inspect.signature(ServiceClient.create_lora_training_client_async)
    with pytest.raises(TypeError):
        sig.bind(None, "Qwen/Qwen3-8B", 16)


@pytest.mark.parametrize(
    "cls, name",
    [
        (ServiceClient, "create_lora_training_client_async"),
        (ServiceClient, "create_training_client_from_state_async"),
        (ServiceClient, "create_training_client_from_state_with_optimizer_async"),
        (ServiceClient, "create_sampling_client_async"),
        (TrainingClient, "save_weights_and_get_sampling_client_async"),
        (TrainingClient, "create_sampling_client_async"),
    ],
)
def test_async_factory_twins_exist_and_are_coroutines(cls, name):
    fn = getattr(cls, name, None)
    assert fn is not None, f"{cls.__name__}.{name} is missing"
    assert inspect.iscoroutinefunction(fn), f"{cls.__name__}.{name} must be async def"


@pytest.mark.parametrize(
    "cls, sync_name",
    [
        (TrainingClient, "forward_backward"),
        (TrainingClient, "optim_step"),
        (TrainingClient, "save_state"),
        (TrainingClient, "load_state"),
        (TrainingClient, "save_weights_for_sampler"),
        (TrainingClient, "forward"),
        (SamplingClient, "sample"),
        (SamplingClient, "compute_logprobs"),
    ],
)
def test_every_operation_has_an_async_twin(cls, sync_name):
    twin = getattr(cls, f"{sync_name}_async", None)
    assert twin is not None, f"{cls.__name__}.{sync_name}_async is missing"
    assert inspect.iscoroutinefunction(twin)


# --- unhonoured parameters are loud ----------------------------------------


UNSUPPORTED = [
    ("seed", 1234),
    ("train_mlp", False),
    ("train_attn", False),
    ("train_unembed", False),
    ("user_metadata", {"run": "exp-1"}),
]


@pytest.mark.parametrize("name, value", UNSUPPORTED)
def test_unsupported_parameter_is_accepted_and_warns(name, value, caplog):
    """Accepted so the script runs; warned about so the caller is not misled."""
    from sagemaker.train.training_session import service_client as sc_mod

    with caplog.at_level("WARNING"):
        sc_mod._warn_unsupported(**{name: value})

    assert name in caplog.text
    assert "not supported" in caplog.text


def test_defaults_do_not_warn(caplog):
    """Left unset, none of them fire — or the warning becomes background noise."""
    from sagemaker.train.training_session import service_client as sc_mod

    with caplog.at_level("WARNING"):
        sc_mod._warn_unsupported(
            seed=None,
            train_mlp=None,
            train_attn=None,
            train_unembed=None,
            user_metadata=None,
        )

    assert caplog.text == ""


@pytest.mark.parametrize(
    "args, kwargs",
    [
        # supervised training loop — all keyword
        ((), {"base_model": "Qwen/Qwen3-8B", "rank": 32, "user_metadata": {"a": "b"}}),
        # RL training loop — base_model positional
        (("Qwen/Qwen3-8B",), {"rank": 32, "user_metadata": {"a": "b"}}),
        # RL recipe loop — no metadata
        ((), {"base_model": "Qwen/Qwen3-8B", "rank": 32}),
    ],
)
def test_accepts_every_surveyed_call_shape(args, kwargs):
    """The exact shapes used across the 8 surveyed creation sites.

    Asserted against the signature rather than by creating a session: this is a
    calling-convention guarantee, and binding proves the call would not raise
    TypeError.
    """
    sig = inspect.signature(ServiceClient.create_lora_training_client)
    bound = sig.bind(None, *args, **kwargs)
    assert bound.arguments["base_model"] == "Qwen/Qwen3-8B"
    assert bound.arguments["rank"] == 32


# --- unhonoured operation parameters are loud -------------------------------


#: ``(method, kwargs, expected_warned_names)`` for the checkpoint-lifetime knobs
#: the save ops accept for drop-in compatibility but cannot honour.
SAVE_UNSUPPORTED = [
    ("save_state", {"ttl_seconds": 3600}, ["ttl_seconds"]),
    ("save_state", {"overwrite": True}, ["overwrite"]),
    ("save_state", {"ttl_seconds": 60, "overwrite": True}, ["ttl_seconds", "overwrite"]),
    ("save_state", {"name": "ckpt-1"}, ["name"]),
    ("save_weights_for_sampler", {"ttl_seconds": 3600}, ["ttl_seconds"]),
    ("save_weights_for_sampler", {"name": "ckpt-1"}, ["name"]),
]


@pytest.mark.parametrize("method, kwargs, warned", SAVE_UNSUPPORTED)
def test_save_op_lifetime_knob_is_accepted_and_warns(method, kwargs, warned, caplog):
    """Accepted so the script runs; warned about so the caller is not misled.

    A caller who asked for a 1-hour TTL and got platform retention has a cost
    surprise coming, so the discrepancy has to be in the log.
    """
    submitted = {}

    class FakeSelf:
        def _submit(self, op_config, *args, **kwargs):
            submitted["op_config"] = op_config
            return "future"

        def _cache_on_save(self, kind):
            return None

    with caplog.at_level("WARNING"):
        result = getattr(TrainingClient, method)(FakeSelf(), **kwargs)

    # Proceeds: the op is still submitted, and the unhonoured knob is not on the wire.
    assert result == "future"
    assert not any(
        key.lower() in ("ttl", "ttlseconds", "overwrite", "name")
        for config in submitted["op_config"].values()
        for key in config
    )
    for name in warned:
        assert name in caplog.text
    assert "not supported" in caplog.text
    # The name warning has to name the parameter that does work, or the caller
    # is told "no" with nowhere to go.
    if "name" in warned:
        assert "description" in caplog.text


@pytest.mark.parametrize("method", ["save_state", "save_weights_for_sampler"])
def test_save_op_defaults_do_not_warn(method, caplog):
    """``overwrite=False`` is what we already do, so it must stay silent.

    Warning on a default would fire on every save in the loop and train the
    caller to filter the log, which is how the TTL warning gets missed too.
    """

    class FakeSelf:
        def _submit(self, op_config, *args, **kwargs):
            return "future"

        def _cache_on_save(self, kind):
            return None

    with caplog.at_level("WARNING"):
        getattr(TrainingClient, method)(FakeSelf(), description="step 100")

    assert caplog.text == ""


@pytest.mark.parametrize(
    "method, kwargs",
    [
        ("save_state_async", {"ttl_seconds": 3600, "overwrite": True, "name": "c1"}),
        ("save_weights_for_sampler_async", {"ttl_seconds": 3600, "name": "c1"}),
    ],
)
def test_save_async_twin_forwards_lifetime_knobs(method, kwargs):
    """The twin must not drop them, or the warning silently stops firing."""
    seen = {}
    sync_name = method[: -len("_async")]

    class FakeSelf:
        pass

    setattr(FakeSelf, sync_name, lambda self, **kw: seen.update(kw) or "future")

    assert asyncio.run(getattr(TrainingClient, method)(FakeSelf(), **kwargs)) == "future"
    for name, value in kwargs.items():
        assert seen[name] == value


# --- the async twins actually run -------------------------------------------


def test_async_twin_delegates_to_the_sync_path(monkeypatch):
    """to_thread keeps the blocking wait off the loop but must still deliver."""
    calls = {}

    class FakeSelf:
        def create_sampling_client(self, **kwargs):
            calls.update(kwargs)
            return "sampling-client"

    coro = ServiceClient.create_sampling_client_async(
        FakeSelf(), base_model="arn:aws:sagemaker:::hub-content/m"
    )
    assert asyncio.run(coro) == "sampling-client"
    assert calls == {"base_model": "arn:aws:sagemaker:::hub-content/m"}


def test_async_twin_forwards_base_model_and_keywords():
    seen = {}

    class FakeSelf:
        def create_lora_training_client(self, base_model, **kwargs):
            seen["base_model"] = base_model
            seen.update(kwargs)
            return "training-client"

    coro = ServiceClient.create_lora_training_client_async(
        FakeSelf(), "Qwen/Qwen3-8B", rank=16, user_metadata={"a": "b"}
    )
    assert asyncio.run(coro) == "training-client"
    assert seen == {
        "base_model": "Qwen/Qwen3-8B",
        "rank": 16,
        "user_metadata": {"a": "b"},
    }


@pytest.mark.parametrize(
    "cls, method, kwargs",
    [
        (
            ServiceClient,
            "create_training_client_from_state_async",
            {"model_package_arn": "arn:aws:sagemaker:::model-package/g/1", "rank": 8},
        ),
        (
            ServiceClient,
            "create_training_client_from_state_with_optimizer_async",
            {"model_package_arn": "arn:aws:sagemaker:::model-package/g/1", "rank": 8},
        ),
        (
            TrainingClient,
            "save_weights_and_get_sampling_client_async",
            {"description": "step 100", "merge_weights": True},
        ),
        (
            TrainingClient,
            "create_sampling_client_async",
            {"model_package_arn": "arn:aws:sagemaker:::model-package/g/1"},
        ),
    ],
)
def test_client_returning_async_twin_forwards_every_keyword(cls, method, kwargs):
    """Every keyword must survive the hop onto the worker thread.

    These twins hand the blocking factory to ``asyncio.to_thread``, which
    forwards positionally and by keyword. A dropped keyword here would silently
    build the client with a default instead of what the caller asked for.
    """
    seen = {}
    sync_name = method[: -len("_async")]

    class FakeSelf:
        pass

    setattr(FakeSelf, sync_name, lambda self, **kw: seen.update(kw) or "client")

    assert asyncio.run(getattr(cls, method)(FakeSelf(), **kwargs)) == "client"
    for name, value in kwargs.items():
        assert seen[name] == value


def test_async_twin_does_not_swallow_a_keyword_named_func():
    """``to_thread(func, /, ...)`` takes ``func`` positionally-only.

    So a caller keyword that happens to be named ``func`` is forwarded to the
    factory rather than rebinding what ``to_thread`` runs. It fails in the
    factory, exactly as the synchronous call would.
    """

    class FakeSelf:
        def create_lora_training_client(self, base_model, **kwargs):
            raise TypeError(f"unexpected keyword: {sorted(kwargs)}")

    coro = ServiceClient.create_lora_training_client_async(
        FakeSelf(), "Qwen/Qwen3-8B", func="not-a-callable"
    )
    with pytest.raises(TypeError, match=r"unexpected keyword: \['func'\]"):
        asyncio.run(coro)


# --- auto-generated resource names -------------------------------------------


def test_session_name_prefix_only_prefixes_session_names():
    """The parameter names sessions and nothing else.

    It used to also feed the model package group name, which is why it was
    called ``resource_name``. Now that the group has its own shared default,
    the prefix has exactly one job and the name says so.
    """
    svc = ServiceClient.__new__(ServiceClient)
    svc._session_name_prefix = "my-run"

    first = ServiceClient._generate_session_name(svc)
    second = ServiceClient._generate_session_name(svc)

    assert first.startswith("my-run-")
    assert first != second, "each session needs its own name"


def test_default_session_name_prefix_is_used_when_unset():
    svc = ServiceClient.__new__(ServiceClient)
    svc._session_name_prefix = None

    assert ServiceClient._generate_session_name(svc).startswith(
        f"{sc_mod.DEFAULT_SESSION_NAME_PREFIX}-"
    )


def test_default_model_package_group_is_shared_not_per_session():
    """One group per account/region, like the SageMaker default bucket.

    A group per session would leak a group per run and scatter checkpoints so
    that a later process could not find the latest one.
    """
    created = []

    class FakeHub:
        exceptions = type("E", (), {"ClientError": RuntimeError})

        def create_model_package_group(self, **kwargs):
            created.append(kwargs["ModelPackageGroupName"])
            return {"ModelPackageGroupArn": "arn:group"}

    svc = ServiceClient.__new__(ServiceClient)
    svc._get_hub_client = lambda: FakeHub()

    assert ServiceClient._auto_create_mpg(svc) == "arn:group"
    assert ServiceClient._auto_create_mpg(svc) == "arn:group"
    assert created == [sc_mod.DEFAULT_MODEL_PACKAGE_GROUP_NAME] * 2


def test_default_model_package_group_name_shape():
    """Named per the convention agreed for this SDK.

    ``sagemaker`` prefix and ``default``, and no account id or region: the
    group ARN already carries both, so repeating them is noise. Feature-agnostic
    for the same reason the default bucket is.
    """
    name = sc_mod.DEFAULT_MODEL_PACKAGE_GROUP_NAME

    assert name == "sagemaker-default"
    assert not any(ch.isdigit() for ch in name), "no account id"
    assert "us-east" not in name and "us-west" not in name, "no region"


# --- retry_config is accepted and loud --------------------------------------


def test_service_client_create_sampling_client_warns_on_retry_config(caplog):
    """Existing code passes retry_config; it must not TypeError, and must warn."""
    svc = ServiceClient.__new__(ServiceClient)

    with caplog.at_level("WARNING"):
        # Reaches the ValueError only after the warning has been emitted, which
        # is the point: the warning fires on the way in, not as a side effect of
        # a successful session creation.
        with pytest.raises(ValueError):
            ServiceClient.create_sampling_client(svc, retry_config={"max": 5})

    assert "retry_config" in caplog.text
    assert "not supported" in caplog.text


def test_training_client_create_sampling_client_warns_once_on_retry_config(caplog):
    """One warning per call, not one per layer that forwards it."""

    class FakeSelf:
        _hub_client = object()
        _tokenizer = object()
        _hf_model_id = "m"
        _runtime = None
        _service_client = None
        _handle = None
        # None is a real value here, not a stand-in: it makes the new client
        # build its own submitter, which is what the standalone path does.
        _submitter = None
        training_session_name = "s"
        _base_model = "b"

    with caplog.at_level("WARNING"):
        TrainingClient.create_sampling_client(
            FakeSelf(), model_package_arn="", retry_config={"max": 5}
        )

    assert caplog.text.count("retry_config") == 1, caplog.text


def test_save_weights_and_get_sampling_client_forwards_retry_config():
    """Forwarding matters: dropping it would silently stop the warning."""
    seen = {}

    class FakeFuture:
        def result(self):
            return type("R", (), {"model_package_arn": "arn:mp/weights-1"})()

    class FakeSelf:
        def save_weights_for_sampler(self, **kwargs):
            return FakeFuture()

        def _load_weights_for_sampling(self, model_package_arn):
            seen["loaded"] = model_package_arn

        def create_sampling_client(self, **kwargs):
            seen.update(kwargs)
            return "sampling-client"

    result = TrainingClient.save_weights_and_get_sampling_client(
        FakeSelf(), retry_config={"max": 5}
    )

    assert result == "sampling-client"
    assert seen["retry_config"] == {"max": 5}


# --- kind-filtered discovery must not answer None by accident ----------------


def _fake_group(kinds):
    """A ServiceClient whose group holds ``kinds`` (newest first)."""
    from sagemaker.train.training_session.types import ModelPackageKind

    arns = [f"arn:mp/{i}" for i in range(len(kinds))]
    by_arn = dict(zip(arns, kinds))

    class FakeHub:
        def list_model_packages(self, **kwargs):
            limit = kwargs.get("MaxResults", len(arns))
            return {
                "ModelPackageSummaryList": [
                    {"ModelPackageArn": a, "ModelPackageVersion": i} for i, a in enumerate(arns)
                ][:limit]
            }

        def describe_model_package(self, ModelPackageName):
            is_ckpt = by_arn[ModelPackageName] == ModelPackageKind.CHECKPOINT
            return {"InferenceSpecification": {"Containers": [{"IsCheckpoint": is_ckpt}]}}

    svc = ServiceClient.__new__(ServiceClient)
    svc._get_hub_client = lambda: FakeHub()
    svc._model_package_group_arn = "arn:group"
    return svc


def test_buried_checkpoint_is_found_once_scan_limit_covers_it():
    """The RL-loop shape: weights saved every step, state every N steps.

    The newest CHECKPOINT sits under N-1 MODEL packages, so with N above the
    default scan limit it is out of reach until the caller widens the search.
    This pins the recovery route the warning points at: a larger scan_limit does
    find it, so None from a truncated scan really was a truncation artefact and
    not an absent checkpoint.
    """
    from sagemaker.train.training_session.types import ModelPackageKind

    kinds = [ModelPackageKind.MODEL] * 50 + [ModelPackageKind.CHECKPOINT]
    svc = _fake_group(kinds)

    assert (
        ServiceClient.get_latest_model_package(svc, ModelPackageKind.CHECKPOINT) is None
    ), "50 deep is past the default scan limit"

    found = ServiceClient.get_latest_model_package(svc, ModelPackageKind.CHECKPOINT, scan_limit=100)
    assert found is not None
    assert found.model_package_arn == "arn:mp/50"


def test_exhausted_scan_limit_warns_rather_than_implying_none_exist(caplog):
    """None + a truncated search is not evidence; say so at WARNING."""
    from sagemaker.train.training_session.types import ModelPackageKind

    svc = _fake_group([ModelPackageKind.MODEL] * 10)

    with caplog.at_level("WARNING"):
        found = ServiceClient.get_latest_model_package(
            svc, ModelPackageKind.CHECKPOINT, scan_limit=5
        )

    assert found is None
    assert "scan_limit" in caplog.text


def test_fully_scanned_group_with_no_match_does_not_warn(caplog):
    """Searched the whole group and found nothing, which IS evidence, so quiet."""
    from sagemaker.train.training_session.types import ModelPackageKind

    svc = _fake_group([ModelPackageKind.MODEL] * 3)

    with caplog.at_level("WARNING"):
        found = ServiceClient.get_latest_model_package(
            svc, ModelPackageKind.CHECKPOINT, scan_limit=20
        )

    assert found is None
    assert caplog.text == ""


def test_existing_default_group_is_reused_not_recreated():
    """Second caller must resolve the ARN rather than fail on AlreadyExists."""

    class AlreadyExists(RuntimeError):
        pass

    class FakeHub:
        exceptions = type("E", (), {"ClientError": AlreadyExists})

        def create_model_package_group(self, **kwargs):
            raise AlreadyExists("Model Package Group already exists")

        def describe_model_package_group(self, **kwargs):
            return {"ModelPackageGroupArn": f"arn:{kwargs['ModelPackageGroupName']}"}

    svc = ServiceClient.__new__(ServiceClient)
    svc._get_hub_client = lambda: FakeHub()

    assert ServiceClient._auto_create_mpg(svc) == (f"arn:{sc_mod.DEFAULT_MODEL_PACKAGE_GROUP_NAME}")
