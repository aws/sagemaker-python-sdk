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
"""Internal helpers for translating user-facing types to service API shapes.

Not part of the public API surface.
"""

from __future__ import annotations

import logging
import warnings
import weakref
from typing import Any, Dict, List, Optional

from sagemaker.train.training_session.types import (
    Datum,
    EncodedTextChunk,
    ModelInput,
    ModelInputChunk,
    _serialize_loss_fn_inputs,
)

logger = logging.getLogger(__name__)

_PUBLIC_HUB_NAME = "SageMakerPublicHub"

#: Hub content search-keyword prefix identifying a base model this client can
#: fine-tune. Each model's recipes are tagged
#: ``@recipe:{type}_{technique}_{strategy}`` (lowercase) on the Public Hub;
#: matching on the prefix selects models carrying a compatible fine-tuning
#: recipe regardless of the parameter-efficient strategy it uses.
_TRAINABLE_RECIPE_KEYWORD_PREFIX = "@recipe:finetuning_mtrl_"

#: Hub content search-keyword prefix carrying a model's published id, spelled
#: ``@huggingface-id:{provider}/{model}``. Free on the list summary, but keyword
#: values are lowercased; the model's Hub document has the exact case.
_HF_ID_KEYWORD_PREFIX = "@huggingface-id:"

#: Model-family names that :func:`list_base_models_in_region` does not report and
#: ``base_model`` does not accept, for models this client does not support end to
#: end. Matched against the dash-separated parts of a Hub content name.
_EXCLUDED_MODEL_NAMES = ("nova",)

#: Hub client -> {model id: Hub content name} for the fine-tunable set, filled by
#: a Hub scan. Only the mapping is cached, never the versioned ARN: that is
#: described on every resolution, so a session still starts on the model's
#: current version.
#:
#: Kept under the client itself, because the client is what decides the answer:
#: its Region, its endpoint, and its credentials all change which models come
#: back, and none of them is reliably readable to build a key from. The reference
#: is weak, so a catalogue lives exactly as long as the client it describes.
_HUB_MODEL_INDEX: "weakref.WeakKeyDictionary[Any, Dict[str, str]]" = weakref.WeakKeyDictionary()

#: Hub client -> {Hub content name: model id}. Survives a catalogue refresh,
#: which exists to find new models, not to re-read ids that do not change under
#: a published model. Successful reads only, so a failed one is retried rather
#: than remembered.
#:
#: That holds for a model publishing an id. For one that does not, the fallback
#: is memoised too, so a client keeps reporting the fallback for its lifetime
#: even if the model starts publishing an id. Bounded to one client, and the
#: fallback resolves, so the cost is a stale spelling rather than a stale model.
_MODEL_ID_BY_HUB_CONTENT: "weakref.WeakKeyDictionary[Any, Dict[str, str]]" = (
    weakref.WeakKeyDictionary()
)

#: Ceiling on concurrent ``DescribeHubContent`` calls for one Hub scan.
_MAX_HUB_DESCRIBE_WORKERS = 8


#: Consequence clause for ``retry_config``, shared by every factory returning a
#: SamplingClient. The retry policy is the SDK's: throttling and 5xx failures are
#: retried on a fixed budget on both submit and poll, and any other error
#: surfaces immediately.
_RETRY_CONFIG_CONSEQUENCE = (
    "the SDK's built-in transient-retry policy applies instead (fixed budget on "
    "throttling and 5xx; other errors surface immediately)"
)


def _warn_unsupported_args(consequence: str, **passed: Any) -> None:
    """Log a warning for each accepted-but-unhonoured keyword argument.

    These parameters are accepted but carry no behaviour. The warning names each
    one the caller actually set, so the log says plainly which knobs the run did
    not use.

    Only a value the caller chose warns. ``None``, or for a boolean the default
    the SDK already behaves as, carries no request and stays quiet, so the
    message keeps its signal.

    Args:
        consequence: Clause completing "... and has no effect; <consequence>",
            naming what happened instead (e.g. ``"the session was created
            without it"``).
        **passed: Parameter name to value. Values of ``None`` are skipped.
    """
    for name, value in passed.items():
        if value is None:
            continue
        logger.warning(
            "%s=%r is not supported by this SDK and has no effect; %s.",
            name,
            value,
            consequence,
        )


def _is_excluded_base_model(hub_content_name: str) -> bool:
    """True if any dash-separated part of the name is in :data:`_EXCLUDED_MODEL_NAMES`.

    Matched on whole parts rather than as a substring, so a name that merely
    contains an excluded one (as a longer word ending in it, say) is not caught
    with it. This decides both what the listing reports and what ``base_model``
    accepts, so an over-match makes a model unusable rather than just unlisted.
    """
    parts = set(hub_content_name.lower().split("-"))
    return any(name in parts for name in _EXCLUDED_MODEL_NAMES)


def _is_qualified_model_id(value: str) -> bool:
    """True if ``value`` is shaped like a model id (``provider/model``).

    The one predicate deciding both what an id is: :func:`resolve_base_model`
    routes on it, so a value the listing reports must satisfy it too. Reported
    under a value that fails it, a model would be listed and then rejected.
    """
    return "/" in value and not value.startswith("arn:")


def _hf_id_from_keywords(keywords: List[str]) -> Optional[str]:
    """The model id a Hub entry's search keywords advertise, or ``None``.

    The keyword's value is checked, not trusted: it is whatever the Hub
    publishes, and only a ``provider/model`` value resolves.
    """
    for keyword in keywords:
        if keyword.lower().startswith(_HF_ID_KEYWORD_PREFIX):
            hf_id = keyword[len(_HF_ID_KEYWORD_PREFIX) :].strip()
            return hf_id if hf_id and _is_qualified_model_id(hf_id) else None
    return None


def _known_model_ids(hub_client: Any) -> Dict[str, str]:
    """This client's ``{Hub content name: model id}`` memo, writable in place.

    A client that cannot be referenced weakly gets a throwaway dict, so writes
    are discarded rather than shared with another client.
    """
    # Membership is the only lookup a WeakKeyDictionary tolerates on a key that
    # cannot be referenced weakly; ``get`` raises on one just as ``[]`` does.
    try:
        if hub_client in _MODEL_ID_BY_HUB_CONTENT:
            return _MODEL_ID_BY_HUB_CONTENT[hub_client]
        memo: Dict[str, str] = {}
        _MODEL_ID_BY_HUB_CONTENT[hub_client] = memo
        return memo
    except TypeError:
        return {}


def _public_model_id(hub_content_name: str, keywords: List[str], hub_client: Any) -> str:
    """The id a Hub model is published under, and can be passed back as ``base_model``.

    Read from the model's Hub document, the only source that keeps the id's exact
    case. Falls back to the list summary's keyword (same id, lowercased), then to
    the Hub content name, so every entry is listed under an id that resolves.

    A failed read costs case and not identity where a keyword exists, since
    ``base_model`` matches case-insensitively. For a model publishing no keyword
    it does change the reported identity, to the Hub content name; that is warned
    about, not remembered, and the next call reports the id again.
    """
    known = _known_model_ids(hub_client)
    if hub_content_name in known:
        return known[hub_content_name]

    keyword_id = _hf_id_from_keywords(keywords)
    try:
        described = _extract_hf_model_id(_describe_hub_content(hub_content_name, hub_client))
    except Exception as e:  # noqa: BLE001 - any Hub failure degrades the same way
        # Logged: nothing else explains why this one id is spelled unlike the rest.
        logger.warning(
            "Could not read the published id for Hub model %r (%s); reporting it as "
            "%r, which still resolves.",
            hub_content_name,
            e,
            keyword_id or hub_content_name,
        )
        return keyword_id or hub_content_name

    model_id = described if described and _is_qualified_model_id(described) else None
    model_id = model_id or keyword_id or hub_content_name
    known[hub_content_name] = model_id
    return model_id


def _scan_trainable_hub_models(hub_client: Any) -> Dict[str, str]:
    """Return ``{model id: Hub content name}`` for the region's fine-tunable models.

    Queries the SageMaker Public Hub and keeps the models that advertise a
    compatible fine-tuning recipe (via their ``HubContentSearchKeywords``), so
    the set is discovered live rather than hardcoded and a newly enabled model
    appears with no SDK change.

    Args:
        hub_client: A boto3 ``sagemaker`` client (its region determines results).
    """
    trainable: List["tuple[str, List[str]]"] = []
    next_token = None
    while True:
        kwargs: dict = {"HubName": _PUBLIC_HUB_NAME, "HubContentType": "Model"}
        if next_token:
            kwargs["NextToken"] = next_token
        resp = hub_client.list_hub_contents(**kwargs)
        for item in resp.get("HubContentSummaries", []):
            hub_content_name = item.get("HubContentName")
            if not hub_content_name:
                continue
            if _is_excluded_base_model(hub_content_name):
                continue
            keywords = item.get("HubContentSearchKeywords", []) or []
            if not any(kw.lower().startswith(_TRAINABLE_RECIPE_KEYWORD_PREFIX) for kw in keywords):
                continue
            trainable.append((hub_content_name, keywords))
        next_token = resp.get("NextToken")
        if not next_token:
            break

    model_ids = _public_model_ids(trainable, hub_client)

    index: Dict[str, str] = {}
    # Walked in page order, so the tie-break stays deterministic however the ids
    # were read.
    for hub_content_name, _ in trainable:
        model_id = model_ids[hub_content_name]
        # Two Hub entries publishing one id would otherwise resolve to
        # whichever the Hub happened to page first; take the lower name so
        # the same call answers the same way twice. Logged because only one
        # of the two can be listed, and which one is not the caller's choice.
        existing = index.get(model_id)
        if existing is not None:
            logger.warning(
                "Base model id %r is published by more than one model in this "
                "region (%s); resolving it to %r. Pass a full Hub model ARN to "
                "select the other.",
                model_id,
                ", ".join(sorted([existing, hub_content_name])),
                min(existing, hub_content_name),
            )
        if existing is None or hub_content_name < existing:
            index[model_id] = hub_content_name
    return index


def _public_model_ids(trainable: List["tuple[str, List[str]]"], hub_client: Any) -> Dict[str, str]:
    """``{Hub content name: model id}`` for each entry, reading unknown ids at once.

    An unknown id costs one Hub call. The calls are independent, so they go out
    together and a cold scan costs roughly one round trip rather than one per
    model. Fewer than two go on this thread: a pool of one buys nothing.
    """
    known = _known_model_ids(hub_client)
    resolved = {name: known[name] for name, _ in trainable if name in known}
    pending = [(name, kws) for name, kws in trainable if name not in resolved]

    if len(pending) < 2:
        resolved.update({name: _public_model_id(name, kws, hub_client) for name, kws in pending})
        return resolved

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(
        max_workers=min(_MAX_HUB_DESCRIBE_WORKERS, len(pending)),
        thread_name_prefix="sagemaker-hub-model-id",
    ) as pool:
        futures = {
            name: pool.submit(_public_model_id, name, kws, hub_client) for name, kws in pending
        }
    resolved.update({name: future.result() for name, future in futures.items()})
    return resolved


def _is_hub_index_cached(hub_client: Any) -> bool:
    """True if this client's catalogue is already cached, so a miss is a real miss."""
    return hub_client in _HUB_MODEL_INDEX


def _hub_model_index(hub_client: Any, *, refresh: bool = False) -> Dict[str, str]:
    """``{model id: Hub content name}``, from this client's cache or a fresh scan.

    Args:
        hub_client: A boto3 ``sagemaker`` client (it determines the results, so it
            is also what the entry is kept under).
        refresh: Scan the Hub even when this client is already cached.
    """
    # Membership is the only lookup that tolerates a key which cannot be
    # referenced weakly; ``get`` would raise on one just as ``[]`` does.
    if not refresh and hub_client in _HUB_MODEL_INDEX:
        return _HUB_MODEL_INDEX[hub_client]
    index = _scan_trainable_hub_models(hub_client)
    try:
        _HUB_MODEL_INDEX[hub_client] = index
    except TypeError:
        # A client that cannot be referenced weakly is simply not cached, rather
        # than kept alive by this module or shared with another client.
        pass
    return index


def list_base_models_in_region(hub_client: Any) -> List[str]:
    """Return the ids of the base models available for fine-tuning in this region.

    Each model is named by its own model id (``provider/model``), spelled as its
    provider publishes it wherever the Hub carries one; pass one straight to
    :meth:`ServiceClient.create_lora_training_client` as ``base_model``. Results
    are region-specific.

    Always scans the Hub rather than answering from the catalogue cache, so a
    model enabled since the last call shows up without a new client.

    Args:
        hub_client: A boto3 ``sagemaker`` client (its region determines results).
    """
    # Sorted case-insensitively: a provider that capitalises its name would
    # otherwise sort ahead of every lowercase one, and the list is read by eye.
    return sorted(_hub_model_index(hub_client, refresh=True), key=str.lower)


def _describe_hub_content(hub_content_name: str, hub_client: Any) -> dict:
    """``DescribeHubContent`` for a Hub content name, errors unchanged.

    The raw call, for callers that must tell a lookup failure apart from a model
    that simply publishes nothing. :func:`_describe_base_model` is the one that
    turns a failure into an actionable message.
    """
    return hub_client.describe_hub_content(
        HubName=_PUBLIC_HUB_NAME,
        HubContentType="Model",
        HubContentName=hub_content_name,
    )


def _describe_base_model(base_model: str, hub_client: Any) -> dict:
    """Look up a base model on the Public Hub and return its content record.

    ``base_model`` may be a Hub content name or a full Hub content ARN. Returns
    the ``DescribeHubContent`` response, from which both the versioned ARN and
    the model's published id can be read.

    Raises:
        ValueError: If the name resolves to no Hub content in this region.
    """
    # An ARN already identifies a specific version; pull the content name out of
    # it (``...:hub-content/{hub}/Model/{name}[/version]``). Otherwise the value
    # is the Hub content name itself.
    if base_model.startswith("arn:"):
        parts = base_model.split("/")
        hub_content_name = parts[3] if len(parts) >= 4 else base_model
    else:
        hub_content_name = base_model

    try:
        return _describe_hub_content(hub_content_name, hub_client)
    except Exception as e:
        raise ValueError(
            f"Could not resolve base model {base_model!r} to a SageMaker Public "
            f"Hub model in this region. Pass a model id returned by "
            f"list_base_models(), or a full Hub model ARN."
        ) from e


def _extract_hf_model_id(describe_resp: dict) -> Optional[str]:
    """Pull the published ``HuggingfaceId`` out of a ``DescribeHubContent`` response.

    ``HubContentDocument`` is a JSON string; returns the ``HuggingfaceId`` field
    from it, or ``None`` if absent/unparseable.
    """
    import json

    doc = describe_resp.get("HubContentDocument")
    if not doc:
        return None
    try:
        return json.loads(doc).get("HuggingfaceId") or None
    except (TypeError, ValueError):
        return None


def resolve_base_model(base_model: str, hub_client: Any) -> tuple:
    """Resolve a base-model identifier to ``(versioned_arn, hf_model_id)`` in one call.

    Accepts three forms of input:

    - A model id (``"provider/model"``), as
      :func:`list_base_models_in_region` reports: looked up in the region's
      catalogue index, since Hub content names are NOT derivable from ids
      mechanically.
    - A Hub content name, which is what an ARN and a service-side error name the
      model.
    - A full Hub content ARN (``"arn:aws:sagemaker:..."``).

    For a Hub content *name*, a single ``DescribeHubContent`` yields both the
    versioned ARN and the published id, so callers that need both (session create
    + later tokenizer load) avoid describing the model twice.

    For an ARN input the ARN is returned unchanged with ``hf_model_id=None`` (no
    Describe). The id is then resolved lazily at tokenizer time via
    :func:`resolve_hf_model_id`.

    Returns:
        ``(arn, hf_model_id_or_None)``.

    Raises:
        ValueError: If a model id matches no supported Hub model, or a Hub
            content name resolves to no Hub content in this region. Either
            message lists the supported models under both names.
    """
    if base_model.startswith("arn:"):
        return base_model, None
    if _is_qualified_model_id(base_model):
        return _resolve_hf_id_to_hub_model(base_model, hub_client)
    if _is_excluded_base_model(base_model):
        # A Hub content name this client does not support end to end. Refused
        # here rather than at session create, and refused on the same rule the
        # listing filters on, so a name the listing omits does not resolve.
        raise _excluded_base_model_error(base_model)
    try:
        resp = _describe_base_model(base_model, hub_client)
    except ValueError as e:
        # A name the Hub does not carry. The Describe alone cannot say what this
        # region does offer, so the error lists the ids instead: the usual cause
        # is a model's display name, which is neither an id nor a content name.
        raise _unsupported_base_model_error(
            _hub_model_index(hub_client, refresh=True), base_model
        ) from e
    return resp["HubContentArn"], _extract_hf_model_id(resp)


def _hub_entry_for_model_id(index: Dict[str, str], hf_model_id: str) -> Optional["tuple[str, str]"]:
    """``(published id, Hub content name)`` listed under ``hf_model_id``, or ``None``.

    Matched case-insensitively, and the *published* id comes back rather than the
    caller's spelling of it: published ids are case-sensitive, so only that
    spelling loads a tokenizer.
    """
    want = hf_model_id.lower()
    for model_id, hub_content_name in index.items():
        if model_id.lower() == want:
            return model_id, hub_content_name
    return None


def _excluded_base_model_error(base_model: str) -> ValueError:
    """The error for a base model this client does not support for fine-tuning.

    Lists no catalogue. This input is refused on its name alone, without a Hub
    call, and inlining the supported set would turn the cheapest rejection on the
    path into a full Hub scan. The message names the one call that answers it.
    """
    return ValueError(
        f"Base model {base_model!r} is not supported for fine-tuning by this "
        f"client. Call list_base_models() for the models available in this region."
    )


def _unsupported_base_model_error(index: Dict[str, str], base_model: str) -> ValueError:
    """The error for a base model no Hub model in this region publishes.

    Lists the ids and nothing else. Every id here is one the caller can pass
    straight back, so the message answers "what should I have typed" without
    introducing a second way to name the same model.
    """
    # One model per line: a catalogue of any size stays readable, where a single
    # joined line grows past anything a terminal or a log viewer wraps well.
    if index:
        listing = "Supported models:" + "".join(
            f"\n  {model_id}" for model_id in sorted(index, key=str.lower)
        )
    else:
        listing = "Supported models: (none)"
    return ValueError(
        f"Could not resolve base model {base_model!r} to a model supported for "
        f"fine-tuning in this region. Pass one of these ids, or a full Hub model "
        f"ARN.\n{listing}"
    )


def _resolve_hf_id_to_hub_model(hf_model_id: str, hub_client: Any) -> tuple:
    """Find the supported Hub model published under ``hf_model_id``.

    A Hub content name is not derivable from a model id by rewriting it, so
    the id is looked up in the client's catalogue index, which one Hub scan builds
    and every later resolution on that client reuses. Only the name comes from the
    cache: it is then described, so the ARN is the model's current version, and a
    name the Hub no longer carries is caught rather than trusted.

    Returns:
        ``(versioned_arn, published_hf_model_id)``. The published id is returned
        rather than the caller's spelling of it, so a case-insensitive match
        still yields the canonical id a tokenizer loads by.

    Raises:
        ValueError: If no supported model publishes this id.
    """
    from_cache = _is_hub_index_cached(hub_client)
    index = _hub_model_index(hub_client)
    entry = _hub_entry_for_model_id(index, hf_model_id)
    if entry is not None:
        try:
            return _resolved_from_hub_name(entry[1], entry[0], hub_client)
        except ValueError:
            # The cached name no longer resolves (the entry was renamed or
            # withdrawn). Fall through to a fresh scan rather than reporting a
            # Hub content name the caller never typed.
            pass
    elif not from_cache:
        # The index above *was* the fresh scan, so a miss is already final and a
        # second scan would only ask the same question twice.
        raise _unsupported_base_model_error(index, hf_model_id)

    index = _hub_model_index(hub_client, refresh=True)
    entry = _hub_entry_for_model_id(index, hf_model_id)
    if entry is None:
        raise _unsupported_base_model_error(index, hf_model_id)
    return _resolved_from_hub_name(entry[1], entry[0], hub_client)


def _resolved_from_hub_name(hub_content_name: str, hf_model_id: str, hub_client: Any) -> tuple:
    """Describe ``hub_content_name`` and return ``(versioned_arn, hf_model_id)``.

    ``hf_model_id`` is the id the catalogue lists this model under, which is the
    id the model publishes. The Hub document is preferred where it carries one,
    and neither is the caller's own spelling: a tokenizer loads by exact case.
    """
    resp = _describe_base_model(hub_content_name, hub_client)
    return resp["HubContentArn"], _extract_hf_model_id(resp) or hf_model_id


def resolve_hf_model_id(base_model: str, hub_client: Any) -> str:
    """Resolve a base model to its HuggingFace model id for tokenizer loading.

    Reads the ``HuggingfaceId`` published on the model's Public Hub content
    record (via ``DescribeHubContent``), never a hand-maintained mapping,
    so the tokenizer tracks whatever the hub advertises. A value that is already
    a model id (``provider/model``, not an ARN or Hub content name) is returned
    as-is so callers can still pass one directly.

    Args:
        base_model: A Hub content name or Hub model ARN; or a model id (as
            :func:`list_base_models_in_region` reports) to pass through.
        hub_client: A boto3 ``sagemaker`` client used for the Hub lookup.

    Raises:
        ValueError: If the model has no published HuggingFace id (e.g. a model
            whose tokenizer is not publicly available); pass pre-tokenized token
            ids in that case.
    """
    # A bare HuggingFace id (``org/model``) is not an ARN and not a Hub content
    # name (those have no ``/``); pass it straight through.
    if _is_qualified_model_id(base_model):
        return base_model

    if hub_client is None:
        raise ValueError(
            f"No tokenizer is available for base model {base_model!r}: a "
            f"HuggingFace id could not be looked up. Provide a tokenizer or pass "
            f"pre-tokenized token ids instead."
        )

    hf_id = _extract_hf_model_id(_describe_base_model(base_model, hub_client))

    if not hf_id:
        raise ValueError(
            f"No public HuggingFace tokenizer is available for base model "
            f"{base_model!r}. Pass pre-tokenized token ids instead."
        )
    return hf_id


def load_tokenizer(base_model: str, hub_client: Any, hf_model_id: Optional[str] = None) -> Any:
    """Load a HuggingFace AutoTokenizer for the given base model.

    Args:
        base_model: A model id, a Hub content name, or a Hub model ARN.
        hub_client: A boto3 ``sagemaker`` client used to look up the model's
            published HuggingFace id. Only consulted when ``hf_model_id`` is not
            supplied.
        hf_model_id: The model's HuggingFace id, if already resolved (e.g. at
            session-create time). When given, the Hub lookup is skipped entirely.

    Returns:
        A ``transformers.PreTrainedTokenizer`` instance.

    Raises:
        ImportError: If ``transformers`` is not installed (install the optional
            tokenizer extra: ``pip install sagemaker-train[tokenizer]``).
        ValueError: If the model is not supported.
    """
    if not hf_model_id and not base_model:
        raise ValueError(
            "Cannot load tokenizer: no base model or HuggingFace model ID "
            "is configured for this session."
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*PyTorch.*")
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Loading a tokenizer requires the 'transformers' package, which "
                "is an optional dependency. Install it with "
                "`pip install sagemaker-train[tokenizer]`, or pass pre-tokenized "
                "token IDs instead of text."
            ) from e

    if not hf_model_id:
        hf_model_id = resolve_hf_model_id(base_model, hub_client)
    logger.info("Loading tokenizer for %r", hf_model_id)

    # Suppress noisy logs and warnings during download
    _loggers = ("httpx", "huggingface_hub", "huggingface_hub.utils._http")
    saved = {name: logging.getLogger(name).level for name in _loggers}
    for name in _loggers:
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return AutoTokenizer.from_pretrained(hf_model_id)
    finally:
        for name, level in saved.items():
            logging.getLogger(name).setLevel(level)


def build_lora_config(rank: int, alpha: int) -> dict[str, Any]:
    """Build the LoraConfig shape."""
    return {"Rank": rank, "Alpha": alpha}


#: Default cap on the number of checkpoints a single list_model_packages call
#: returns. A model package group can hold hundreds of versions; the latest
#: handful is what callers want, so the default is bounded instead of paginating
#: the whole group.
DEFAULT_LIST_MODEL_PACKAGES_LIMIT = 10

#: Largest ``MaxResults`` the list APIs accept on one request. A caller asking
#: for more than this pages up to their cap instead, since a larger request is
#: rejected rather than answered.
MAX_RESULTS_PER_PAGE = 100

#: How many of the newest packages the "get latest" helpers will describe while
#: looking for a matching kind. The kind is not in the list summary, so each
#: candidate costs one DescribeModelPackage, so this bounds that fan-out. A match
#: can sit well down the list (an RL loop saves sampler weights far more often
#: than state), so an exhausted scan warns instead of returning ``None``, and
#: the caller can retry with a larger ``scan_limit``.
DEFAULT_LATEST_SCAN_LIMIT = 20


def base_model_arn_from_model_package(mp_desc: Dict[str, Any], region: str) -> str:
    """Extract the versioned base-model Hub ARN a model package was trained on.

    ``DescribeModelPackage`` publishes the base model as
    ``InferenceSpecification.Containers[0].BaseModel`` (``HubContentName`` +
    ``HubContentVersion``), not as a top-level field. The version is pinned to
    what the weights were trained on; resolving the name via the Hub instead
    would silently return the latest version.

    Args:
        mp_desc: A ``DescribeModelPackage`` response.
        region: Region for the returned ARN.

    Raises:
        ValueError: If the response carries no base-model reference.
    """
    containers = (mp_desc.get("InferenceSpecification") or {}).get("Containers") or []
    base = (containers[0].get("BaseModel") or {}) if containers else {}
    hub_name = base.get("HubContentName")
    hub_version = base.get("HubContentVersion")
    if not hub_name or not hub_version:
        raise ValueError(
            "Model package "
            f"{mp_desc.get('ModelPackageArn', '<unknown>')!r} carries no "
            "base-model reference (InferenceSpecification.Containers[0].BaseModel)."
        )
    # Partition taken from the model package's own ARN rather than assumed:
    # ``aws`` is wrong in the China and GovCloud partitions, and every other ARN
    # this package builds resolves its partition rather than hardcoding one.
    partition = _arn_partition(mp_desc.get("ModelPackageArn"))
    return (
        f"arn:{partition}:sagemaker:{region}:aws:hub-content/"
        f"{_PUBLIC_HUB_NAME}/Model/{hub_name}/{hub_version}"
    )


#: Fields in a full ARN: ``arn:partition:service:region:account:resource``.
_ARN_FIELD_COUNT = 6


def _arn_partition(arn: Optional[str], default: str = "aws") -> str:
    """Partition field of ``arn``, or ``default`` when it cannot be read.

    The whole ARN shape is checked before the field is trusted, not just the
    presence of a colon: a short or malformed value would otherwise hand back
    part of the resource id as the partition.

    Defaults rather than raising: the partition only differs from ``aws`` outside
    the commercial regions, and the caller still needs an answer.
    """
    parts = (arn or "").split(":")
    if len(parts) < _ARN_FIELD_COUNT or parts[0] != "arn" or not parts[1]:
        return default
    return parts[1]


def list_model_package_summaries(
    sagemaker_client: Any,
    model_package_group_arn: str,
    max_results: Optional[int] = DEFAULT_LIST_MODEL_PACKAGES_LIMIT,
) -> List[Dict[str, Any]]:
    """Return ``ListModelPackages`` summaries for a group, newest first.

    Auto-paginates only as far as ``max_results``. ``max_results=None`` returns
    every version in the group (caller opts into a full scan). The summary
    carries arn / version / creation time / description, but NOT the checkpoint
    kind (``IsCheckpoint`` lives in the InferenceSpecification, returned only by
    ``DescribeModelPackage``).

    Args:
        sagemaker_client: A boto3 ``sagemaker`` client.
        model_package_group_arn: The session's model package group (name or ARN).
        max_results: Cap on summaries returned; None for all.
    """
    summaries: list = []
    next_token = None
    while True:
        kwargs: dict = {
            "ModelPackageGroupName": model_package_group_arn,
            "SortBy": "CreationTime",
            "SortOrder": "Descending",
        }
        if next_token:
            kwargs["NextToken"] = next_token
        if max_results is not None:
            remaining = max_results - len(summaries)
            if remaining <= 0:
                break
            kwargs["MaxResults"] = min(remaining, MAX_RESULTS_PER_PAGE)
        resp = sagemaker_client.list_model_packages(**kwargs)
        summaries.extend(resp.get("ModelPackageSummaryList", []))
        next_token = resp.get("NextToken")
        if not next_token:
            break
    if max_results is not None:
        return summaries[:max_results]
    return summaries


def describe_model_package_kind(sagemaker_client: Any, model_package_arn: str):
    """Return the :class:`ModelPackageKind` for a model package via Describe.

    ``IsCheckpoint=True`` marks a (save_state) checkpoint;
    absent/false marks a (save_weights_for_sampler) model.

    Args:
        sagemaker_client: A boto3 ``sagemaker`` client.
        model_package_arn: ARN of the model package version to describe.
    """
    from sagemaker.train.training_session.types import ModelPackageKind

    resp = sagemaker_client.describe_model_package(ModelPackageName=model_package_arn)
    is_checkpoint = False
    spec = resp.get("InferenceSpecification") or {}
    containers = spec.get("Containers") or []
    if containers and containers[0].get("IsCheckpoint"):
        is_checkpoint = True
    return ModelPackageKind.CHECKPOINT if is_checkpoint else ModelPackageKind.MODEL


#: ``ManagedStorageType`` of a model package whose weights SageMaker stores on
#: your behalf, not in your own S3 location. Describe does not return an
#: S3 URI for one, and deleting the package is what releases its storage.
MANAGED_STORAGE_RESTRICTED = "Restricted"

#: Max keys ``DeleteObjects`` accepts per call, which is also the max a
#: ``ListObjectsV2`` page returns, so one listed page is one delete batch.
_DELETE_OBJECTS_BATCH = 1000


def split_s3_uri(s3_uri: str) -> "tuple[str, str]":
    """Split ``s3://bucket/key/prefix`` into ``("bucket", "key/prefix")``.

    The key is empty when the URI names only a bucket.

    Args:
        s3_uri: An ``s3://`` URI.

    Raises:
        ValueError: If the URI is not an ``s3://`` URI or names no bucket.
    """
    scheme = "s3://"
    if not s3_uri.startswith(scheme):
        raise ValueError(f"{s3_uri!r} is not an s3:// URI.")
    bucket, _, key = s3_uri[len(scheme) :].partition("/")
    if not bucket:
        raise ValueError(f"{s3_uri!r} names no bucket.")
    return bucket, key


def describe_model_package_artifacts(
    sagemaker_client: Any, model_package_arn: str
) -> Dict[str, Optional[str]]:
    """Return where a model package's artifacts live, via Describe.

    Reads ``InferenceSpecification.Containers[0].ModelDataSource.S3DataSource``
    for the location and its type, plus the top-level ``ManagedStorageType``.
    ``s3_uri`` is ``None`` when the package exposes no location, which is the
    case for one whose storage SageMaker manages
    (:data:`MANAGED_STORAGE_RESTRICTED`).

    Args:
        sagemaker_client: A boto3 ``sagemaker`` client.
        model_package_arn: ARN of the model package version to describe.
    """
    resp = sagemaker_client.describe_model_package(ModelPackageName=model_package_arn)
    spec = resp.get("InferenceSpecification") or {}
    containers = spec.get("Containers") or []
    source = (
        ((containers[0].get("ModelDataSource") or {}).get("S3DataSource") or {})
        if containers
        else {}
    )
    return {
        "s3_uri": source.get("S3Uri") or None,
        # "S3Prefix" (everything under the location) or "S3Object" (one object).
        # Read from the package, not inferred from whether the URI ends in "/".
        "s3_data_type": source.get("S3DataType") or None,
        "managed_storage_type": resp.get("ManagedStorageType") or None,
    }


def delete_s3_objects_under(
    s3_client: Any, s3_uri: str, *, s3_data_type: Optional[str] = None
) -> "tuple[int, List[str]]":
    """Delete the S3 objects a model package's location covers.

    Returns ``(deleted_count, failed_keys)``. ``DeleteObjects`` reports per-key
    failures in the response body, so a partial failure comes back in
    ``failed_keys`` instead of raising.

    Each listed page is deleted before the next is requested, so no full key list
    is held in memory. The continuation token resumes past keys already returned,
    so the deletes do not disturb it.

    Args:
        s3_client: A boto3 ``s3`` client.
        s3_uri: Location to empty, as carried by the model package.
        s3_data_type: ``"S3Object"`` to delete exactly that one key; anything
            else (``"S3Prefix"`` or unset) treats the URI as a prefix.

    Raises:
        ValueError: If the URI names a bucket with no key under it, where
            deleting "its" objects would empty the whole bucket.
    """
    bucket, key = split_s3_uri(s3_uri)
    if not key:
        raise ValueError(
            f"{s3_uri!r} names a bucket root, not a location inside it. Refusing "
            "to delete: this would empty the whole bucket."
        )

    if s3_data_type == "S3Object":
        return _delete_keys(s3_client, bucket, [key])

    # A prefix is matched on the segment boundary. Without the trailing slash,
    # "run-1" would also match the contents of "run-10".
    prefix = key if key.endswith("/") else key + "/"
    deleted = 0
    failed: List[str] = []
    token: Optional[str] = None
    while True:
        kwargs: Dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3_client.list_objects_v2(**kwargs)
        keys = [obj["Key"] for obj in page.get("Contents") or []]
        if keys:
            page_deleted, page_failed = _delete_keys(s3_client, bucket, keys)
            deleted += page_deleted
            failed.extend(page_failed)
        token = page.get("NextContinuationToken")
        if not page.get("IsTruncated") or not token:
            break
    return deleted, failed


def _delete_keys(s3_client: Any, bucket: str, keys: List[str]) -> "tuple[int, List[str]]":
    """``DeleteObjects`` these keys, in batches of :data:`_DELETE_OBJECTS_BATCH`.

    Runs quiet, so the response carries only failures and the deleted count is
    the batch minus those.
    """
    deleted = 0
    failed: List[str] = []
    for start in range(0, len(keys), _DELETE_OBJECTS_BATCH):
        batch = keys[start : start + _DELETE_OBJECTS_BATCH]
        resp = s3_client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
        )
        batch_failed = [err["Key"] for err in resp.get("Errors") or [] if err.get("Key")]
        deleted += len(batch) - len(batch_failed)
        failed.extend(batch_failed)
    return deleted, failed


def _resolve_tokenizer(tokenizer: Any, tokenizer_loader: Any):
    """Return a tokenizer, lazily loading one via ``tokenizer_loader`` if needed.

    ``tokenizer`` (already cached) wins; otherwise the zero-arg
    ``tokenizer_loader`` (e.g. ``client.get_tokenizer``) is invoked on demand,
    so string input works without a manual ``get_tokenizer()`` call.
    Raises ValueError if neither yields a tokenizer.
    """
    if tokenizer is not None:
        return tokenizer
    if tokenizer_loader is not None:
        return tokenizer_loader()
    raise ValueError(
        "String input requires a tokenizer, but none is available for this "
        "session's base model. Provide pre-tokenized token IDs instead."
    )


def prompt_to_model_input(
    prompt: Any,
    tokenizer: Any = None,
    chunk_type: str = "encoded_text",
    tokenizer_loader: Any = None,
) -> dict[str, Any]:
    """Build a ModelInput shape from a sampling prompt.

    Accepts the same prompt forms as a single training datum:
      - A list of token IDs: ``[1, 2, 3]``
      - A string (tokenized via ``tokenizer`` or lazily via ``tokenizer_loader``)
      - A pre-built ModelInput dict (passthrough): ``{"chunks": [...]}``
      - A typed :class:`ModelInput` / :class:`EncodedTextChunk`, or any
        object with ``chunks``

    Returns the ``{"chunks": [...]}`` ModelInput shape expected by the sampling
    operation.
    """
    # Typed inputs: built-in types or compatible external objects. A typed
    # input built from text (ModelInput.from_text) carries the text rather than
    # chunks, so it is tokenized here too: this is the one place with a
    # tokenizer, and serializing it untokenized raises.
    if isinstance(prompt, (ModelInput, EncodedTextChunk, ModelInputChunk)):
        model_input = ModelInput.from_any(prompt)
        if model_input._pending_text is not None:
            model_input = model_input._tokenized_with(
                _resolve_tokenizer(tokenizer, tokenizer_loader)
            )
        return model_input._to_api()
    if not isinstance(prompt, (dict, list, str)) and hasattr(prompt, "chunks"):
        return ModelInput.from_any(prompt)._to_api()

    # Already a ModelInput dict
    if isinstance(prompt, dict) and "chunks" in prompt:
        return prompt

    # String: tokenize (lazily loading a tokenizer only when needed)
    if isinstance(prompt, str):
        token_ids = _resolve_tokenizer(tokenizer, tokenizer_loader).encode(prompt)
    # List of token IDs
    elif isinstance(prompt, list) and all(isinstance(t, int) for t in prompt):
        token_ids = prompt
    else:
        raise TypeError(
            f"Unsupported prompt type: {type(prompt).__name__}. "
            "Expected a list of token IDs, a string, or a ModelInput dict."
        )

    return {"chunks": [{"tokens": token_ids, "type": chunk_type}]}


def tokenize_to_datum(
    token_ids: list[int],
    loss_fn_inputs: dict[str, Any] | None = None,
    chunk_type: str = "encoded_text",
) -> dict[str, Any]:
    """Build a single Datum shape from token IDs.

    ``lossFnInputs`` is only included when ``loss_fn_inputs`` is provided.
    The service rejects an auto-generated ``labels`` entry. The accepted keys
    depend on the loss function (e.g. ``advantages``, ``weights``,
    ``target_tokens``, ``logprobs``, ``ref_log_prob``), so the caller supplies
    the inputs appropriate for their loss function.
    """
    datum: dict[str, Any] = {
        "modelInput": {"chunks": [{"tokens": token_ids, "type": chunk_type}]},
    }
    if loss_fn_inputs is not None:
        datum["lossFnInputs"] = _serialize_loss_fn_inputs(loss_fn_inputs)
    return datum


def _batch_element_kind(element: Any) -> str:
    """Classify one batch element, for detecting a batch that mixes kinds.

    Only the distinction between the interpretations :func:`batch_to_datum_list`
    applies matters here, so an unrecognised element gets its own kind and lands
    in the mixed path, where coercing it raises a clear error.
    """
    if isinstance(element, (Datum, ModelInput, EncodedTextChunk, ModelInputChunk)):
        return "typed"
    if isinstance(element, dict):
        return "dict"
    if isinstance(element, str):
        return "text"
    if isinstance(element, list) and all(isinstance(t, int) for t in element):
        return "tokens"
    if not isinstance(element, (dict, list, str, int)) and (
        hasattr(element, "model_input") or hasattr(element, "chunks")
    ):
        return "typed"
    return "unsupported"


def _typed_datum_to_api(
    element: Any,
    tokenizer: Any,
    tokenizer_loader: Any,
) -> dict[str, Any]:
    """Serialize one typed batch element, tokenizing text it still carries.

    A ``Datum`` or ``ModelInput`` built from text (``ModelInput.from_text``, or a
    caller's own object exposing a text prompt) has no chunks to send yet. The
    tokenizer lives out here, so this is where that text becomes tokens;
    otherwise serializing it raises.
    """
    datum = Datum.from_any(element)
    if datum._pending_text is not None:
        datum = datum._tokenized_with(_resolve_tokenizer(tokenizer, tokenizer_loader))
    return datum._to_api()


def _mixed_batch_to_datum_list(
    batch: Any,
    tokenizer: Any,
    loss_fn_inputs: dict[str, Any] | None,
    tokenizer_loader: Any,
) -> list[dict[str, Any]]:
    """Coerce a batch whose elements are not all the same kind, one at a time.

    Text is the one kind needing outside context (a tokenizer), so it keeps the
    dedicated path; everything else goes through :meth:`Datum.from_any`, which
    rejects an element it cannot read instead of emitting an empty datum.
    """
    tok = None
    datums: list[dict[str, Any]] = []
    for element in batch:
        if isinstance(element, str):
            if tok is None:
                tok = _resolve_tokenizer(tokenizer, tokenizer_loader)
            datums.append(tokenize_to_datum(tok.encode(element), loss_fn_inputs))
            continue
        datum = _typed_datum_to_api(element, tokenizer, tokenizer_loader)
        if loss_fn_inputs and "lossFnInputs" not in datum:
            datum = {**datum, "lossFnInputs": dict(_serialize_loss_fn_inputs(loss_fn_inputs))}
        datums.append(datum)
    return datums


#: ``lossFnInputs`` entries whose length the service requires to equal the token
#: count of ``modelInput``. Limited to what the service enforces: other entries
#: may legitimately carry their own length.
_PER_TOKEN_LOSS_FN_INPUTS = (
    "target_tokens",
    "weights",
    "logprobs",
    "old_logprobs",
)


def model_input_token_count(model_input: Any) -> int | None:
    """Token count of a serialized ``modelInput``, or ``None`` if not countable.

    ``None`` for anything that is not the ``{"chunks": [...]}`` shape, so a
    length check is skipped rather than guessed at.
    """
    if not isinstance(model_input, dict):
        return None
    chunks = model_input.get("chunks")
    if not isinstance(chunks, list):
        return None
    total = 0
    for chunk in chunks:
        if not isinstance(chunk, dict):
            return None
        tokens = chunk.get("tokens")
        if not isinstance(tokens, list):
            return None
        total += len(tokens)
    return total


def _wire_model_input_length(datum: dict[str, Any]) -> int | None:
    """Token count of a datum's serialized ``modelInput``."""
    return model_input_token_count(datum.get("modelInput"))


def _wire_value_length(value: Any) -> int | None:
    """Element count of a serialized ``lossFnInputs`` value, or ``None``."""
    if isinstance(value, dict):
        data = value.get("data")
        return len(data) if isinstance(data, list) else None
    if isinstance(value, list):
        return len(value)
    return None


#: The loss inputs each loss function requires on every datum. The field is
#: optional in general, but which entries a given loss needs is not: a request
#: missing one is refused, and the refusal names neither the datum nor the key.
_LOSS_FN_INPUT_KEYS = {
    "cross_entropy": ("target_tokens", "weights"),
    "importance_sampling": ("target_tokens", "weights", "logprobs", "advantages"),
    "ppo": ("target_tokens", "weights", "logprobs", "advantages"),
    "cispo": ("target_tokens", "weights", "logprobs", "advantages"),
}


def _validate_loss_fn_inputs(datums: list[dict[str, Any]], loss_fn_name: str) -> None:
    """Check every datum carries the loss inputs ``loss_fn_name`` requires.

    Raised here rather than left to the request so the message can name the datum
    and the keys. An operation that reaches the service without them is refused
    after it has been accepted and queued, and that refusal identifies neither.
    """
    required = _LOSS_FN_INPUT_KEYS.get(loss_fn_name)
    if not required:
        return
    for index, datum in enumerate(datums):
        if not isinstance(datum, dict):
            continue
        loss_fn_inputs = datum.get("lossFnInputs")
        present = loss_fn_inputs if isinstance(loss_fn_inputs, dict) else {}
        missing = [key for key in required if key not in present]
        if not missing:
            continue
        raise ValueError(
            f"datum[{index}]: loss function {loss_fn_name!r} requires "
            f"{', '.join(required)} in its loss inputs, and "
            f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} missing. "
            f"Pass them per datum, or for the whole batch with "
            f"loss_fn_inputs={{...}}; each is one value per token of the datum, "
            f"the prompt included."
        )


def _validate_datum_lengths(datums: list[dict[str, Any]]) -> None:
    """Check per-token ``lossFnInputs`` against the datum's token count.

    Every per-token entry must be as long as ``modelInput``, prompt positions
    included. Raised locally because the service accepts the mismatch and only
    fails the operation later.
    """
    for index, datum in enumerate(datums):
        if not isinstance(datum, dict):
            continue
        loss_fn_inputs = datum.get("lossFnInputs")
        if not isinstance(loss_fn_inputs, dict) or not loss_fn_inputs:
            continue
        expected = _wire_model_input_length(datum)
        if expected is None:
            continue
        for key in _PER_TOKEN_LOSS_FN_INPUTS:
            if key not in loss_fn_inputs:
                continue
            actual = _wire_value_length(loss_fn_inputs[key])
            if actual is None or actual == expected:
                continue
            raise ValueError(
                f"datum[{index}]: lossFnInputs[{key!r}] has {actual} values but "
                f"modelInput has {expected} tokens. Every per-token entry must "
                f"be the same length as modelInput, covering the prompt as well "
                f"as the response, e.g. weights=[0.0] * prompt_len + [1.0] * "
                f"response_len for a {expected}-token input. Response-only "
                f"values are the usual cause of this mismatch."
            )


def batch_to_datum_list(
    batch: Any,
    tokenizer: Any = None,
    loss_fn_inputs: dict[str, Any] | None = None,
    tokenizer_loader: Any = None,
    loss_fn_name: str | None = None,
) -> list[dict[str, Any]]:
    """Convert a user-provided batch to a list of Datum dicts.

    Accepts:
      - A list of token ID lists: ``[[1,2,3], [4,5,6]]``
      - A list of strings (tokenized via ``tokenizer`` or lazily via
        ``tokenizer_loader``): ``["hello", "world"]``
      - A list of pre-built datum dicts (passthrough)
      - A list of typed :class:`Datum` / :class:`ModelInput` objects, or
        compatible objects with the same attributes

    When ``loss_fn_inputs`` is provided, it is attached to every constructed
    datum. For pre-built datum dicts it fills in only those that do not already
    carry their own. When omitted, no ``lossFnInputs`` is emitted (the service
    rejects an auto-generated ``labels`` entry).

    Args:
        loss_fn_name: The loss function the batch will be sent with. Given, every
            datum is checked for the keys that loss needs. Omitted, only the
            lengths are checked.

    Raises:
        ValueError: A datum is missing a loss input ``loss_fn_name`` needs, or a
            per-token ``lossFnInputs`` entry is not the same length as the
            datum's ``modelInput``.
    """
    datums = _coerce_batch_to_datum_list(batch, tokenizer, loss_fn_inputs, tokenizer_loader)
    if loss_fn_name is not None:
        _validate_loss_fn_inputs(datums, loss_fn_name)
    _validate_datum_lengths(datums)
    return datums


def _coerce_batch_to_datum_list(
    batch: Any,
    tokenizer: Any = None,
    loss_fn_inputs: dict[str, Any] | None = None,
    tokenizer_loader: Any = None,
) -> list[dict[str, Any]]:
    """Coerce a batch to wire-shape datum dicts, per :func:`batch_to_datum_list`."""
    if not batch:
        return []

    first = batch[0]

    # A batch of bare token IDs is one datum, not a list of elements.
    if isinstance(first, int):
        # That reading holds only if every element is a token ID. A batch that
        # merely starts with one was read as a token list regardless, which put
        # the remaining elements inside a token array and sent them.
        others = sorted({type(e).__name__ for e in batch if not isinstance(e, int)})
        if others:
            raise TypeError(
                "Unsupported batch: one starting with a token ID must hold only "
                f"token IDs, but this one also holds {others}. Pass a list of token "
                "ID lists, a list of strings, or a list of datum dicts."
            )
    else:
        kinds = {_batch_element_kind(element) for element in batch}
        if len(kinds) > 1:
            # Mixed kinds: the homogeneous branches below read the whole batch
            # through the interpretation chosen for ``first``, which would apply
            # the wrong one to every other element. Coerce element by element
            # instead so each is read as what it actually is.
            return _mixed_batch_to_datum_list(batch, tokenizer, loss_fn_inputs, tokenizer_loader)

    # Typed Datum / ModelInput objects: built-in or compatible external objects.
    if isinstance(first, (Datum, ModelInput)) or (
        not isinstance(first, (dict, list, str, int))
        and (hasattr(first, "model_input") or hasattr(first, "chunks"))
    ):
        datums = [_typed_datum_to_api(d, tokenizer, tokenizer_loader) for d in batch]
        if not loss_fn_inputs:
            return datums
        serialized = _serialize_loss_fn_inputs(loss_fn_inputs)
        return [
            d if "lossFnInputs" in d else {**d, "lossFnInputs": dict(serialized)} for d in datums
        ]

    # Already a datum dict: fill loss inputs only where absent. Copy rather
    # than mutate the caller's dicts, and give each datum its own
    # loss_fn_inputs so a later edit to one does not bleed into the others.
    if isinstance(first, dict) and all(isinstance(d, dict) and "modelInput" in d for d in batch):
        if not loss_fn_inputs:
            return batch
        serialized = _serialize_loss_fn_inputs(loss_fn_inputs)
        return [
            d if "lossFnInputs" in d else {**d, "lossFnInputs": dict(serialized)} for d in batch
        ]

    # A datum dict spelled with ``model_input`` or ``prompt`` is read one
    # element at a time, the same way a mixed batch is. A dict naming no input
    # at all falls through to the error below, which lists what a batch may
    # hold; coercing it would instead build a datum with no input at all.
    if all(isinstance(d, dict) for d in batch) and any(
        "model_input" in d or "prompt" in d for d in batch
    ):
        return _mixed_batch_to_datum_list(batch, tokenizer, loss_fn_inputs, tokenizer_loader)

    # List of token ID lists
    if isinstance(first, list) and all(isinstance(t, int) for t in first):
        return [tokenize_to_datum(ids, loss_fn_inputs) for ids in batch]

    # List of strings: tokenize (lazily loading a tokenizer only when needed)
    if isinstance(first, str):
        tok = _resolve_tokenizer(tokenizer, tokenizer_loader)
        return [tokenize_to_datum(tok.encode(text), loss_fn_inputs) for text in batch]

    # Single token ID list (not wrapped)
    if isinstance(first, int):
        return [tokenize_to_datum(batch, loss_fn_inputs)]

    raise TypeError(
        f"Unsupported batch type: {type(first).__name__}. "
        "Expected list of token ID lists, list of strings, or list of datum dicts."
    )


# --- forward-backward chunking ----------------------------------------------
#
# A batch too large for one request is split into chunks and submitted as
# consecutive forwardBackward ops. Gradients accumulate until the next
# optim_step, so this is equivalent to sending the whole batch at once. Each
# chunk is capped by datum count and by an estimated byte budget.

#: Max datums per forward-backward chunk.
MAX_CHUNK_LEN = 1024
#: Metrics that count what a batch held rather than describe it per datum. A
#: chunked batch splits the count, so the whole-batch value is the total and
#: not an average. Named explicitly: an unrecognized metric is treated as a
#: per-datum quantity, which is the common case but still an assumption.
SUMMED_METRICS = frozenset({"global_token_num"})

#: Approximate serialized-byte budget per chunk (leaves headroom under the
#: runtime request-size limit for envelope + signing overhead).
MAX_CHUNK_BYTES = 10_000_000


def _estimate_datum_bytes(datum: Dict[str, Any]) -> int:
    """Roughly estimate a datum's serialized size in bytes.

    Counts the token payload (the dominant term: token IDs plus per-token
    loss-function inputs such as advantages/weights/logprobs) and adds a small
    fixed per-token allowance for JSON structure. Deliberately an estimate:
    it only needs to be monotonic in the real size to keep chunks under the
    limit, not exact.
    """
    tokens = 0
    for chunk in datum.get("modelInput", {}).get("chunks", []):
        tokens += len(chunk.get("tokens", []))
    lfi = datum.get("lossFnInputs") or {}
    for value in lfi.values():
        if isinstance(value, dict):
            data = value.get("data")
            if isinstance(data, list):
                tokens += len(data)
    # ~10 bytes per token/value covers the JSON digits, quoting, and commas.
    return tokens * 10


def chunk_datum_list(
    datum_list: List[Dict[str, Any]],
    *,
    max_len: int = MAX_CHUNK_LEN,
    max_bytes: int = MAX_CHUNK_BYTES,
) -> List[List[Dict[str, Any]]]:
    """Split a datum list into chunks bounded by count and estimated bytes.

    Each chunk holds at most ``max_len`` datums and stays under ``max_bytes`` of
    estimated payload, except that a single datum larger than ``max_bytes`` is
    kept on its own (it cannot be split further here). Datum order is preserved,
    so submitting the chunks in order reproduces the original batch.
    """
    if not datum_list:
        return []

    chunks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_bytes = 0
    for datum in datum_list:
        datum_bytes = _estimate_datum_bytes(datum)
        # Close the current chunk if adding this datum would exceed either
        # bound (but never emit an empty chunk just because one datum is huge).
        if current and (current_bytes + datum_bytes > max_bytes or len(current) >= max_len):
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append(datum)
        current_bytes += datum_bytes
    if current:
        chunks.append(current)
    return chunks


def _result_datum_count(result: Any) -> int:
    """Datums a chunk's result covers, inferred from its per-datum loss outputs.

    Zero when the chunk carries none, which is the usual case: loss outputs are
    only returned when the caller asked for them. Used only as a fallback, since
    inferring the count from an optional field silently makes every chunk weigh
    the same.
    """
    return len(result.loss_fn_outputs or [])


def combine_forward_backward_results(
    results: List[Any], datum_counts: Optional[List[int]] = None
) -> Any:
    """Fold per-chunk forward-backward results into one, in chunk order.

    Scalar metrics are combined so the caller sees the whole-batch step:
    ``loss`` and other per-example means are recomputed as a datum-count-
    weighted average over the chunks that reported them (each chunk reports
    the mean over its own datums), while ``grad_norm``, a property of the
    accumulated gradient after all chunks, is taken from the final chunk and
    omitted when that chunk does not report it. The counts in
    :data:`SUMMED_METRICS` are totalled, since each chunk counts part of the
    same batch. Non-scalar or unrecognized metrics fall back to the last
    chunk that reported them.
    ``loss_fn_outputs`` lists are concatenated in chunk order so per-datum
    outputs line up with the batch.

    Args:
        results: One result per chunk, in chunk order.
        datum_counts: Datums per chunk, in the same order. The caller built the
            chunks, so it is the only place the real sizes are known; without
            them the weights fall back to the per-datum loss outputs, which are
            usually absent and would make an uneven split (1024, 1024, 100)
            report a plain mean instead of the weighted one.

    Single-element input returns that result unchanged, so an unchunked call is
    unaffected.
    """
    if len(results) == 1:
        return results[0]

    # Local import avoids a module cycle (types imports nothing from here).
    from sagemaker.train.training_session.types import ForwardBackwardOutput

    per_chunk_metrics = [r.metrics for r in results]
    if datum_counts is not None and len(datum_counts) == len(results):
        weights = [max(count, 1) for count in datum_counts]
    else:
        weights = [max(_result_datum_count(r), 1) for r in results]

    combined_metrics: Dict[str, float] = {}
    # Keyed off every chunk, not just the last: a metric the earlier chunks
    # report and the final one omits would otherwise vanish from the result.
    for key in dict.fromkeys(k for m in per_chunk_metrics for k in m):
        present = [(m[key], w) for m, w in zip(per_chunk_metrics, weights) if key in m]
        values = [v for v, _ in present]
        if any(not isinstance(v, (int, float)) for v in values):
            combined_metrics[key] = values[-1]
        elif key == "grad_norm" or key.endswith("/grad_norm"):
            # A gradient norm describes the gradient accumulated across all
            # chunks, so only the final chunk's is the whole-batch value and an
            # average of norms is not one. Reported under a namespaced key, so
            # the suffix is matched rather than the whole name.
            if key in per_chunk_metrics[-1]:
                combined_metrics[key] = per_chunk_metrics[-1][key]
        elif key in SUMMED_METRICS:
            # The chunks each carry part of one batch, so the total is the
            # batch's. Averaging reported a count no chunk measured, smaller
            # than a single chunk's own.
            combined_metrics[key] = sum(values)
        else:
            # Weighted over the chunks that reported it, so a metric only some
            # chunks carry is still their mean and not diluted by the rest.
            key_w = sum(w for _, w in present)
            combined_metrics[key] = sum(v * w for v, w in present) / key_w

    # Per-datum loss outputs concatenate in chunk order, so they line up with
    # the batch and read exactly as an unchunked pass would.
    entries: List[Any] = []
    for r in results:
        entries.extend(r.loss_fn_outputs or [])
    combined_outputs: Optional[Any] = entries or None

    raw: Dict[str, Any] = {"forwardBackward": {"metrics": combined_metrics}}
    if combined_outputs is not None:
        raw["forwardBackward"]["lossFnOutputs"] = combined_outputs
    return ForwardBackwardOutput._from_api(raw)
