# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
"""Unit tests for sagemaker.train.training_session._utils helpers."""

from __future__ import annotations

import json
import logging
import re
import weakref
from unittest.mock import MagicMock

import pytest

from sagemaker.train.training_session._utils import (
    batch_to_datum_list,
    combine_forward_backward_results,
    _arn_partition,
    build_lora_config,
    describe_model_package_kind,
    list_base_models_in_region,
    list_model_package_summaries,
    prompt_to_model_input,
    resolve_base_model,
    resolve_hf_model_id,
    tokenize_to_datum,
)
from sagemaker.train.training_session.types import (
    Datum,
    EncodedTextChunk,
    ModelInput,
    ModelPackageKind,
    TensorData,
)


class TestBuildLoraConfig:
    def test_basic(self):
        assert build_lora_config(32, 64) == {"Rank": 32, "Alpha": 64}

    def test_same_rank_alpha(self):
        assert build_lora_config(16, 16) == {"Rank": 16, "Alpha": 16}


class _FakeHub:
    """Stubs describe_hub_content, returning a HubContentDocument with (or
    without) a HuggingfaceId so tokenizer resolution can be exercised offline."""

    def __init__(self, huggingface_id=None, raises=False, arn=None):
        self._hf_id = huggingface_id
        self._raises = raises
        self._arn = (
            arn or "arn:aws:sagemaker:us-west-2:aws:hub-content/SageMakerPublicHub/Model/stub/1.0"
        )

    def describe_hub_content(self, **kwargs):
        if self._raises:
            raise RuntimeError("ResourceNotFound")
        doc = {} if self._hf_id is None else {"HuggingfaceId": self._hf_id}
        return {"HubContentArn": self._arn, "HubContentDocument": json.dumps(doc)}


_MODEL_B_ARN = (
    "arn:aws:sagemaker:us-west-2:aws:hub-content/SageMakerPublicHub/Model/"
    "hubcontent-vlm-model-b/1.0.0"
)
_MODEL_A_ARN = (
    "arn:aws:sagemaker:us-west-2:aws:hub-content/SageMakerPublicHub/Model/"
    "hubcontent-text-model-a/3.15.0"
)
_MODEL_C_ARN = (
    "arn:aws:sagemaker:us-west-2:aws:hub-content/SageMakerPublicHub/Model/"
    "hubcontent-text-model-c/1.2.0"
)


class _FakeCatalogHub:
    """Stubs list_hub_contents + describe_hub_content over a model catalog.

    Mirrors the real Hub shape: content names are NOT derivable from model ids
    (e.g. ``Provider-B/Model-B`` lives at ``hubcontent-vlm-model-b``), so
    model-id resolution must go through the published model id.

    Args:
        models: ``{hub_content_name: (arn, hf_id_or_None)}``.
        advertise_hf_id: Put ``@huggingface-id:`` on the list summary, as the
            real Hub does. False leaves only the capability keyword, so the
            Describe fallback is what has to supply the id.
        region: Reported as ``meta.region_name``, as a real client does. The
            catalogue is cached per client rather than per Region, so this
            changes nothing about caching; None leaves the client region-less.
        keywords_by_name: Full keyword override per content name, for models
            that are not fine-tunable at all.
    """

    _CAPABILITY_KW = "@capability:training_session"

    def __init__(self, models, *, advertise_hf_id=False, region=None, keywords_by_name=None):
        self._models = models
        self._advertise_hf_id = advertise_hf_id
        self._keywords_by_name = keywords_by_name or {}
        self.list_calls = 0
        self.describe_calls: list[str] = []
        if region is not None:
            self.meta = MagicMock(region_name=region)

    def _keywords(self, name):
        if name in self._keywords_by_name:
            return self._keywords_by_name[name]
        keywords = [self._CAPABILITY_KW]
        hf_id = self._models[name][1]
        if self._advertise_hf_id and hf_id:
            keywords.append(f"@huggingface-id:{hf_id.lower()}")
        return keywords

    def list_hub_contents(self, **kwargs):
        self.list_calls += 1
        return {
            "HubContentSummaries": [
                {"HubContentName": n, "HubContentSearchKeywords": self._keywords(n)}
                for n in self._models
            ]
        }

    def describe_hub_content(self, **kwargs):
        name = kwargs["HubContentName"]
        self.describe_calls.append(name)
        if name not in self._models:
            raise RuntimeError("ResourceNotFound")
        arn, hf_id = self._models[name]
        doc = {} if hf_id is None else {"HuggingfaceId": hf_id}
        return {"HubContentArn": arn, "HubContentDocument": json.dumps(doc)}


class _SlottedHub:
    """A Hub client that cannot be referenced weakly, so it cannot be cached.

    ``__slots__`` on a class deriving straight from ``object`` is what drops
    ``__weakref__``; a subclass of a normal class would inherit it back.
    """

    __slots__ = ("list_calls", "describe_calls")

    def __init__(self):
        self.list_calls = 0
        self.describe_calls = []

    def list_hub_contents(self, **kwargs):
        self.list_calls += 1
        return {
            "HubContentSummaries": [
                {
                    "HubContentName": "hubcontent-text-model-a",
                    "HubContentSearchKeywords": [
                        "@capability:training_session",
                        "@huggingface-id:provider-a/model-a",
                    ],
                }
            ]
        }

    def describe_hub_content(self, **kwargs):
        self.describe_calls.append(kwargs["HubContentName"])
        return {
            "HubContentArn": _MODEL_A_ARN,
            "HubContentDocument": json.dumps({"HuggingfaceId": "provider-a/model-a"}),
        }


class TestListBaseModels:
    """The listing reports each model by the id its provider gave it."""

    _MODELS = {
        "hubcontent-vlm-model-b": (_MODEL_B_ARN, "Provider-B/Model-B"),
        "hubcontent-text-model-a": (_MODEL_A_ARN, "provider-a/model-a"),
    }

    def test_reports_the_id_as_its_provider_capitalises_it(self):
        # The keyword is free but lowercased; only the document has the exact
        # case, which is the spelling a customer copies.
        hub = _FakeCatalogHub(self._MODELS, advertise_hf_id=True)
        assert list_base_models_in_region(hub) == [
            "provider-a/model-a",
            "Provider-B/Model-B",
        ]
        assert sorted(hub.describe_calls) == sorted(self._MODELS)

    def test_reads_each_id_once_per_client(self):
        # Re-listing pays for the catalogue, not for every id in it again.
        hub = _FakeCatalogHub(self._MODELS, advertise_hf_id=True, region="us-west-2")
        list_base_models_in_region(hub)
        hub.describe_calls.clear()

        assert list_base_models_in_region(hub) == [
            "provider-a/model-a",
            "Provider-B/Model-B",
        ]
        assert hub.describe_calls == []

    def test_reads_only_the_ids_it_does_not_already_know(self):
        # A model enabled since the last call has its id read; the rest do not.
        # dict(): the fake holds what it is given, and this one grows.
        hub = _FakeCatalogHub(dict(self._MODELS), advertise_hf_id=True, region="us-west-2")
        list_base_models_in_region(hub)
        hub.describe_calls.clear()
        hub._models["hubcontent-text-model-c"] = (_MODEL_C_ARN, "Provider-C/Model-C")

        assert "Provider-C/Model-C" in list_base_models_in_region(hub)
        assert hub.describe_calls == ["hubcontent-text-model-c"]

    def test_a_client_that_cannot_be_referenced_weakly_still_lists(self):
        # Nothing is memoised for it, so it re-reads rather than sharing ids.
        hub = _SlottedHub()
        assert list_base_models_in_region(hub) == ["provider-a/model-a"]
        assert list_base_models_in_region(hub) == ["provider-a/model-a"]
        assert hub.describe_calls == ["hubcontent-text-model-a"] * 2

    def test_reports_the_document_id_with_no_keyword_to_fall_back_on(self):
        # No keyword means nothing to degrade to: the document path stands alone.
        hub = _FakeCatalogHub(self._MODELS)
        assert list_base_models_in_region(hub) == [
            "provider-a/model-a",
            "Provider-B/Model-B",
        ]
        assert sorted(hub.describe_calls) == sorted(self._MODELS)

    def test_falls_back_to_hub_content_name_without_any_hf_id(self):
        # A model publishing no model id anywhere is still listed, under the one
        # identifier it does have, which create_lora_training_client accepts.
        hub = _FakeCatalogHub({"some-private-model": (_MODEL_B_ARN, None)})
        assert list_base_models_in_region(hub) == ["some-private-model"]

    def test_excludes_models_without_a_finetuning_recipe(self):
        hub = _FakeCatalogHub(
            {
                "hubcontent-text-model-a": (_MODEL_A_ARN, "provider-a/model-a"),
                "some-eval-only-model": (_MODEL_B_ARN, "org/eval-only"),
            },
            advertise_hf_id=True,
            keywords_by_name={"some-eval-only-model": ["@recipe:evaluation_mtrlevaluation"]},
        )
        assert list_base_models_in_region(hub) == ["provider-a/model-a"]

    def test_static_region_list_surfaces_a_model_without_the_keyword(self):
        # Until the Hub publishes the capability, a Region's static list stands in.
        hub = _FakeCatalogHub(
            {"huggingface-vlm-qwen3-8-27b": (_MODEL_A_ARN, "Qwen/Qwen3.8-27B")},
            advertise_hf_id=True,
            region="us-west-2",
            keywords_by_name={"huggingface-vlm-qwen3-8-27b": ["@huggingface-id:qwen/qwen3.8-27b"]},
        )
        assert list_base_models_in_region(hub) == ["Qwen/Qwen3.8-27B"]
        assert resolve_base_model("Qwen/Qwen3.8-27B", hub) == (_MODEL_A_ARN, "Qwen/Qwen3.8-27B")

    def test_static_region_list_is_per_region(self):
        hub = _FakeCatalogHub(
            {"huggingface-vlm-qwen3-6-27b": (_MODEL_A_ARN, "Qwen/Qwen3.6-27B")},
            advertise_hf_id=True,
            region="us-east-1",
            keywords_by_name={"huggingface-vlm-qwen3-6-27b": ["@huggingface-id:qwen/qwen3.6-27b"]},
        )
        assert list_base_models_in_region(hub) == []

    def test_static_region_list_needs_a_region(self):
        hub = _FakeCatalogHub(
            {"huggingface-vlm-qwen3-8-27b": (_MODEL_A_ARN, "Qwen/Qwen3.8-27B")},
            advertise_hf_id=True,
            keywords_by_name={"huggingface-vlm-qwen3-8-27b": ["@huggingface-id:qwen/qwen3.8-27b"]},
        )
        assert list_base_models_in_region(hub) == []

    def test_paginates(self):
        hub = _FakeCatalogHub(self._MODELS, advertise_hf_id=True)
        pages = [
            {
                "HubContentSummaries": [
                    {
                        "HubContentName": "hubcontent-text-model-a",
                        "HubContentSearchKeywords": [
                            "@capability:training_session",
                            "@huggingface-id:provider-a/model-a",
                        ],
                    }
                ],
                "NextToken": "tok",
            },
            {
                "HubContentSummaries": [
                    {
                        "HubContentName": "hubcontent-vlm-model-b",
                        "HubContentSearchKeywords": [
                            "@capability:training_session",
                            "@huggingface-id:provider-b/model-b",
                        ],
                    }
                ]
            },
        ]
        hub.list_hub_contents = MagicMock(side_effect=pages)
        assert list_base_models_in_region(hub) == [
            "provider-a/model-a",
            "Provider-B/Model-B",
        ]
        assert hub.list_hub_contents.call_args_list[1][1]["NextToken"] == "tok"

    def test_rescans_the_hub_on_every_call(self):
        # A model enabled since the last call must show up without restarting
        # the process, so the listing never answers from the cache.
        hub = _FakeCatalogHub(self._MODELS, advertise_hf_id=True, region="us-west-2")
        list_base_models_in_region(hub)
        list_base_models_in_region(hub)
        assert hub.list_calls == 2

    def test_sorts_case_insensitively(self):
        # A provider that capitalises its name would otherwise sort ahead of
        # every lowercase one, and this list is read by eye.
        hub = _FakeCatalogHub(self._MODELS)
        assert list_base_models_in_region(hub) == [
            "provider-a/model-a",
            "Provider-B/Model-B",
        ]

    def test_ignores_a_keyword_that_is_not_a_model_id(self):
        # The keyword value is whatever the Hub publishes. A single-segment value
        # is not routed to the id path, so listing it would name it as supported
        # in the very error that rejects it; the content name is listed instead.
        hub = _FakeCatalogHub({"hubcontent-single": (_MODEL_A_ARN, None)})
        hub._keywords_by_name = {
            "hubcontent-single": ["@capability:training_session", "@huggingface-id:singleseg"]
        }

        listed = list_base_models_in_region(hub)

        assert listed == ["hubcontent-single"]
        # And the whole point: every id the listing reports resolves.
        assert resolve_base_model(listed[0], hub)[0] == _MODEL_A_ARN

    def test_a_failed_id_read_costs_case_and_never_the_model(self):
        # Falls back to the keyword's copy of the same id: lowercase, still
        # every model, still resolves.
        hub = _FakeCatalogHub(self._MODELS, advertise_hf_id=True, region="us-west-2")
        hub.describe_hub_content = MagicMock(
            side_effect=RuntimeError("ThrottlingException: Rate exceeded")
        )

        listed = list_base_models_in_region(hub)

        assert listed == ["provider-a/model-a", "provider-b/model-b"]
        del hub.describe_hub_content  # drops the stub, restoring the fake's own
        assert resolve_base_model("provider-b/model-b", hub)[0] == _MODEL_B_ARN

    def test_a_failed_id_read_is_retried_rather_than_remembered(self):
        # Memoising it would pin the model to lowercase for the client's life.
        hub = _FakeCatalogHub(self._MODELS, advertise_hf_id=True, region="us-west-2")
        hub.describe_hub_content = MagicMock(
            side_effect=RuntimeError("ThrottlingException: Rate exceeded")
        )
        list_base_models_in_region(hub)
        del hub.describe_hub_content  # drops the stub, restoring the fake's own

        assert list_base_models_in_region(hub) == [
            "provider-a/model-a",
            "Provider-B/Model-B",
        ]

    def test_a_failed_id_read_with_no_keyword_falls_back_to_the_content_name(self):
        # Nothing else names it, and that name resolves, so it is still listed.
        hub = _FakeCatalogHub({"hubcontent-text-model-a": (_MODEL_A_ARN, "Provider-A/Model-A")})
        hub.describe_hub_content = MagicMock(side_effect=RuntimeError("AccessDeniedException"))

        assert list_base_models_in_region(hub) == ["hubcontent-text-model-a"]

    def test_duplicate_id_resolves_deterministically_and_warns(self, caplog):
        # Two entries publishing one id: only one can be listed, and which one is
        # not the caller's choice, so the drop is logged rather than silent.
        hub = _FakeCatalogHub(
            {
                "hubcontent-model-v2": (_MODEL_B_ARN, "provider-a/model-a"),
                "hubcontent-model-v1": (_MODEL_A_ARN, "provider-a/model-a"),
            },
            advertise_hf_id=True,
            region="us-west-2",
        )
        with caplog.at_level(logging.WARNING):
            assert list_base_models_in_region(hub) == ["provider-a/model-a"]

        assert "published by more than one model in this region" in caplog.text
        assert "hubcontent-model-v1" in caplog.text
        assert "hubcontent-model-v2" in caplog.text
        # Deterministic across pagination order, which is what the tie-break buys.
        assert resolve_base_model("provider-a/model-a", hub)[0] == _MODEL_A_ARN


class TestResolveBaseModel:
    """Tests the 3 resolution paths: ARN, Hub content name, model id."""

    def _hub(self, **kwargs):
        return _FakeCatalogHub(
            {
                "hubcontent-vlm-model-b": (_MODEL_B_ARN, "Provider-B/Model-B"),
                "hubcontent-text-model-a": (_MODEL_A_ARN, "provider-a/model-a"),
            },
            **kwargs,
        )

    def test_arn_passthrough(self):
        result_arn, hf_id = resolve_base_model(_MODEL_B_ARN, self._hub())
        assert result_arn == _MODEL_B_ARN
        assert hf_id is None

    def test_hub_content_name(self):
        result_arn, hf_id = resolve_base_model("hubcontent-vlm-model-b", self._hub())
        assert result_arn == _MODEL_B_ARN
        assert hf_id == "Provider-B/Model-B"

    def test_huggingface_id_matched_via_published_id(self):
        # Hub content name is NOT slash-to-dash of the model id — resolution must
        # match the HuggingfaceId published on each supported model.
        result_arn, hf_id = resolve_base_model("Provider-B/Model-B", self._hub())
        assert result_arn == _MODEL_B_ARN
        assert hf_id == "Provider-B/Model-B"

    def test_huggingface_id_case_insensitive(self):
        result_arn, hf_id = resolve_base_model("provider-b/model-b", self._hub())
        assert result_arn == _MODEL_B_ARN
        assert hf_id == "Provider-B/Model-B"  # canonical published id returned

    def test_case_insensitive_match_returns_the_catalogued_id_not_the_caller_s(self):
        # published ids are case-sensitive, so only the published spelling
        # loads a tokenizer. When the keyword carries the id and the Hub document
        # omits it, the caller's spelling must still not be what comes back.
        class _KeywordOnlyHub(_FakeCatalogHub):
            """Advertises the canonical id; its Hub document carries none."""

            def _keywords(self, name):
                return ["@capability:training_session", "@huggingface-id:Provider-A/Model-A"]

            def describe_hub_content(self, **kwargs):
                self.describe_calls.append(kwargs["HubContentName"])
                return {"HubContentArn": _MODEL_A_ARN, "HubContentDocument": "{}"}

        hub = _KeywordOnlyHub(
            {"hubcontent-text-model-a": (_MODEL_A_ARN, "Provider-A/Model-A")},
            region="us-west-2",
        )

        _, hf_id = resolve_base_model("provider-a/MODEL-a", hub)

        assert hf_id == "Provider-A/Model-A"

    def test_excluded_hub_content_name_does_not_resolve(self):
        # The listing filters this model out, so resolving it by Hub content name
        # would let a name the listing never shows start a session anyway.
        hub = _FakeCatalogHub(
            {
                "hubcontent-text-model-a": (_MODEL_A_ARN, "provider-a/model-a"),
                "amazon-nova-model-x": (_MODEL_B_ARN, "amazon/nova-model-x"),
            },
            advertise_hf_id=True,
            region="us-west-2",
        )
        with pytest.raises(ValueError, match="not supported for fine-tuning"):
            resolve_base_model("amazon-nova-model-x", hub)
        # Refused on the name alone: the cheapest rejection on this path must not
        # pay for a Hub scan just to format its message.
        assert hub.describe_calls == []
        assert hub.list_calls == 0

    def test_a_name_merely_ending_in_an_excluded_one_still_resolves(self):
        # The exclusion matches whole dash-separated parts. Matched as a bare
        # substring, a longer word ending in an excluded name would be caught
        # with it, and the model would be neither listed nor accepted.
        hub = _FakeCatalogHub(
            {"vendor-supernova-lite": (_MODEL_A_ARN, "vendor/supernova-lite")},
            advertise_hf_id=True,
            region="us-west-2",
        )
        assert list_base_models_in_region(hub) == ["vendor/supernova-lite"]
        assert resolve_base_model("vendor-supernova-lite", hub)[0] == _MODEL_A_ARN

    def test_huggingface_id_unsupported_raises_with_supported_list(self):
        # re.DOTALL: the supported set is listed one model per line.
        with pytest.raises(
            ValueError, match=re.compile("provider-z/model-z.*Supported models", re.DOTALL)
        ):
            resolve_base_model("provider-z/model-z", self._hub())

    def test_unsupported_error_lists_one_model_per_line(self):
        # A joined single line grows past what a terminal or log viewer wraps.
        with pytest.raises(ValueError) as excinfo:
            resolve_base_model("provider-z/model-z", self._hub(advertise_hf_id=True))
        lines = str(excinfo.value).splitlines()
        assert lines[1] == "Supported models:"
        assert lines[2:] == [
            "  provider-a/model-a",
            "  Provider-B/Model-B",
        ]

    def test_unsupported_error_lists_ids_only(self):
        # Every value listed is one the caller can pass straight back. Naming the
        # models a second way as well would answer "what should I have typed"
        # with two answers, one of which the listing never shows.
        with pytest.raises(ValueError) as excinfo:
            resolve_base_model("provider-z/model-z", self._hub(advertise_hf_id=True))
        message = str(excinfo.value)
        assert "provider-a/model-a" in message
        assert "hubcontent" not in message

    def test_hub_name_not_found_raises(self):
        with pytest.raises(ValueError, match="Could not resolve"):
            resolve_base_model("nonexistent-model", self._hub())

    def test_hub_name_not_found_lists_supported_models(self):
        # A display name is neither an id nor a content name, and is the likeliest
        # mistake here, so the message says what this region does offer.
        with pytest.raises(ValueError, match="Supported models"):
            resolve_base_model("model-a", self._hub(advertise_hf_id=True))

    def test_hf_id_resolution_reuses_the_cached_catalogue(self):
        # The first resolution lists the region's models to find the Hub content
        # name behind the id; later ones only describe that name.
        hub = self._hub(advertise_hf_id=True, region="us-west-2")
        resolve_base_model("provider-a/model-a", hub)
        hub.describe_calls.clear()

        resolve_base_model("provider-b/model-b", hub)

        assert hub.list_calls == 1
        # The scan's reads are behind us; this describes only the name it needs.
        assert hub.describe_calls == ["hubcontent-vlm-model-b"]

    def test_one_client_never_answers_out_of_anothers_catalogue(self):
        # A catalogue is kept under the client that produced it, so it cannot
        # answer for a client reading a different Hub. That covers every axis at
        # once: Region, endpoint, and credentials all change the answer, and the
        # client is the only thing that carries all three.
        #
        # Asserted on the call counts, NOT the error text: were the catalogue
        # shared, the second lookup would describe the first's Hub content name,
        # fail, self-heal into a rescan, and raise a message matching "Could not
        # resolve" just as a legitimate miss does. The describe tells them apart.
        first = self._hub(advertise_hf_id=True, region="us-west-2")
        resolve_base_model("provider-b/model-b", first)
        second = _FakeCatalogHub(
            {"hubcontent-text-model-a": (_MODEL_A_ARN, "provider-a/model-a")},
            advertise_hf_id=True,
            region="us-east-1",
        )
        with pytest.raises(ValueError, match="Could not resolve"):
            resolve_base_model("provider-b/model-b", second)
        # Never reached for the other client's Hub content name (it does describe
        # its own, to read that model's id).
        assert "hubcontent-vlm-model-b" not in second.describe_calls
        # And one scan: its own miss is final, not retried against a cache.
        assert second.list_calls == 1

    def test_two_clients_reading_one_region_still_hold_separate_catalogues(self):
        # Same Region, but a different endpoint or different credentials means a
        # different Hub. Neither is reliably readable off a client, so neither is
        # trusted to build a key from.
        first = self._hub(advertise_hf_id=True, region="us-west-2")
        resolve_base_model("provider-b/model-b", first)
        same_region = _FakeCatalogHub(
            {"hubcontent-text-model-a": (_MODEL_A_ARN, "provider-a/model-a")},
            advertise_hf_id=True,
            region="us-west-2",
        )
        with pytest.raises(ValueError, match="Could not resolve"):
            resolve_base_model("provider-b/model-b", same_region)
        assert "hubcontent-vlm-model-b" not in same_region.describe_calls

    def test_a_client_reporting_no_region_is_still_cached(self):
        # Caching does not depend on reading a Region off the client, so a client
        # that reports none is not penalised with a rescan per resolution.
        hub = self._hub(advertise_hf_id=True)
        resolve_base_model("provider-a/model-a", hub)
        resolve_base_model("provider-b/model-b", hub)
        assert hub.list_calls == 1

    def test_a_client_that_cannot_be_referenced_weakly_still_resolves(self):
        # Not cacheable, so it rescans rather than being kept alive by this module
        # or sharing another client's catalogue. It must still resolve.
        #
        # __slots__ on a class deriving straight from object is what removes
        # weakref support; on a subclass it would be inherited from the base and
        # this would quietly test the ordinary cached path instead.
        hub = _SlottedHub()
        with pytest.raises(TypeError):
            weakref.ref(hub)

        assert resolve_base_model("provider-a/model-a", hub)[0] == _MODEL_A_ARN
        assert resolve_base_model("provider-a/model-a", hub)[0] == _MODEL_A_ARN
        assert hub.list_calls == 2  # uncached, so it rescans rather than sharing

    def test_warm_cache_still_resolves_a_newly_added_model(self):
        # The self-heal must not only produce good errors: a model added after
        # the cache warmed has to actually resolve.
        hub = self._hub(advertise_hf_id=True, region="us-west-2")
        resolve_base_model("provider-a/model-a", hub)
        hub._models["hubcontent-text-model-c"] = (_MODEL_C_ARN, "provider-c/model-c")

        result_arn, hf_id = resolve_base_model("provider-c/model-c", hub)

        assert result_arn == _MODEL_C_ARN
        assert hf_id == "provider-c/model-c"

    def test_unknown_id_on_a_cold_cache_scans_once(self):
        # The miss came from a scan that just ran, so asking again would only put
        # the same question to the Hub twice.
        hub = self._hub(advertise_hf_id=True, region="us-west-2")
        with pytest.raises(ValueError, match="Could not resolve"):
            resolve_base_model("provider-z/model-z", hub)
        assert hub.list_calls == 1

    def test_stale_cached_name_is_rescanned(self):
        # A Hub content name that has since been withdrawn must not surface as a
        # name the caller never typed; the catalogue is rescanned instead.
        hub = self._hub(advertise_hf_id=True, region="us-west-2")
        resolve_base_model("provider-a/model-a", hub)
        del hub._models["hubcontent-text-model-a"]
        with pytest.raises(ValueError, match="Could not resolve base model 'provider-a/model-a'"):
            resolve_base_model("provider-a/model-a", hub)


class TestResolveHfModelId:
    def test_direct_hf_model_id_passthrough(self):
        # A bare model id needs no Hub lookup.
        assert resolve_hf_model_id("provider-c/model-c", _FakeHub()) == "provider-c/model-c"

    def test_hub_arn_reads_huggingface_id_from_doc(self):
        arn = (
            "arn:aws:sagemaker:us-west-2:aws:hub-content/SageMakerPublicHub/Model/"
            "hubcontent-text-model-a/3.15.0"
        )
        hub = _FakeHub(huggingface_id="provider-a/model-a")
        assert resolve_hf_model_id(arn, hub) == "provider-a/model-a"

    def test_model_without_huggingface_id_raises(self):
        # A model whose Hub record carries no HuggingfaceId (tokenizer not public)
        # raises a clear, actionable error.
        with pytest.raises(ValueError, match="No public HuggingFace tokenizer"):
            resolve_hf_model_id("some-private-model", _FakeHub(huggingface_id=None))


class TestTokenizeToDatum:
    def test_basic_omits_loss_fn_inputs(self):
        # Without explicit loss_fn_inputs, no lossFnInputs key is emitted —
        # the service rejects an auto-generated `labels` entry.
        d = tokenize_to_datum([1, 2, 3])
        assert d["modelInput"]["chunks"] == [{"tokens": [1, 2, 3], "type": "encoded_text"}]
        assert "lossFnInputs" not in d

    def test_with_loss_fn_inputs(self):
        d = tokenize_to_datum([1], loss_fn_inputs={"advantages": [0.0]})
        assert d["lossFnInputs"] == {
            "advantages": {"data": [0.0], "dtype": "float32", "shape": [1]}
        }


class TestBatchToDatumList:
    def test_empty(self):
        assert batch_to_datum_list([]) == []

    def test_token_id_lists(self):
        result = batch_to_datum_list([[1, 2], [3, 4]])
        assert len(result) == 2
        assert result[0]["modelInput"]["chunks"][0]["tokens"] == [1, 2]
        assert all("lossFnInputs" not in d for d in result)

    def test_single_sequence(self):
        result = batch_to_datum_list([10, 20, 30])
        assert len(result) == 1
        assert result[0]["modelInput"]["chunks"][0]["tokens"] == [10, 20, 30]

    def test_datum_passthrough(self):
        datum = {"modelInput": {"chunks": [{"tokens": [1], "type": "encoded_text"}]}}
        result = batch_to_datum_list([datum])
        assert result == [datum]

    def test_datum_loss_inputs_do_not_mutate_caller(self):
        # Filling loss_fn_inputs must not write into the caller's dicts, and
        # each datum must get its own copy (no shared-reference aliasing).
        d1 = {"modelInput": {"chunks": [{"tokens": [1], "type": "encoded_text"}]}}
        d2 = {"modelInput": {"chunks": [{"tokens": [2], "type": "encoded_text"}]}}
        batch = [d1, d2]
        loss = {"labels": [0]}

        result = batch_to_datum_list(batch, loss_fn_inputs=loss)

        # Caller's dicts untouched.
        assert "lossFnInputs" not in d1 and "lossFnInputs" not in d2
        # Each result carries the loss inputs, in its own dict (not aliased).
        serialized = {"labels": {"data": [0], "dtype": "int64", "shape": [1]}}
        assert result[0]["lossFnInputs"] == serialized
        assert result[1]["lossFnInputs"] == serialized
        assert result[0]["lossFnInputs"] is not result[1]["lossFnInputs"]
        assert result[0]["lossFnInputs"] is not loss
        # Adding a key to one datum's loss inputs must not bleed into the
        # other datum or the caller's source dict.
        result[0]["lossFnInputs"]["extra"] = 1
        assert "extra" not in result[1]["lossFnInputs"]
        assert "extra" not in loss

    def test_datum_preserves_existing_loss_inputs(self):
        # A datum that already carries lossFnInputs is left as-is.
        own = {"labels": [7]}
        datum = {
            "modelInput": {"chunks": [{"tokens": [1], "type": "encoded_text"}]},
            "lossFnInputs": own,
        }
        result = batch_to_datum_list([datum], loss_fn_inputs={"labels": [0]})
        assert result[0]["lossFnInputs"] is own

    def test_strings_with_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = [[1, 2], [3, 4]]
        result = batch_to_datum_list(["hello", "world"], tokenizer=tokenizer)
        assert len(result) == 2
        assert result[0]["modelInput"]["chunks"][0]["tokens"] == [1, 2]

    def test_strings_without_tokenizer_raises(self):
        with pytest.raises(ValueError, match="tokenizer"):
            batch_to_datum_list(["hello"])

    def test_strings_auto_load_tokenizer_via_loader(self):
        # No tokenizer passed, but a loader is available — it's invoked lazily
        # so plain-text batches work without a manual get_tokenizer() call.
        loaded = MagicMock()
        loaded.encode.side_effect = [[7, 8]]
        calls = []

        def loader():
            calls.append(1)
            return loaded

        result = batch_to_datum_list(["hi"], tokenizer_loader=loader)
        assert result[0]["modelInput"]["chunks"][0]["tokens"] == [7, 8]
        assert calls == [1]  # loader invoked exactly once, on demand

    def test_token_ids_never_invoke_loader(self):
        # Token-ID input must NOT trigger tokenizer loading (no HF download).
        def loader():
            raise AssertionError("tokenizer_loader must not be called for token IDs")

        result = batch_to_datum_list([[1, 2]], tokenizer_loader=loader)
        assert result[0]["modelInput"]["chunks"][0]["tokens"] == [1, 2]

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported"):
            batch_to_datum_list([3.14])


def _per_token(n: int) -> dict:
    """Per-token loss inputs for an ``n``-token datum."""
    half = n // 2
    return {
        "target_tokens": list(range(1, n + 1)),
        "weights": [0.0] * half + [1.0] * (n - half),
        "logprobs": [-0.1] * n,
        "advantages": [1.0] * n,
    }


_TOKENS10 = list(range(100, 110))


def _wire_datum(tokens: list[int], loss_fn_inputs: dict | None = None) -> dict:
    datum = {"modelInput": {"chunks": [{"tokens": tokens, "type": "encoded_text"}]}}
    if loss_fn_inputs is not None:
        datum["lossFnInputs"] = loss_fn_inputs
    return datum


class TestPerTokenLengthValidation:
    """Per-token lossFnInputs entries must match the datum's token count.

    A mismatch is accepted by the request path and only fails once the
    operation runs, so these cover raising locally instead.
    """

    def test_shorter_than_model_input_raises(self):
        batch = [Datum(model_input=ModelInput.from_ints(_TOKENS10), loss_fn_inputs=_per_token(5))]
        with pytest.raises(ValueError, match=r"datum\[0\].*'target_tokens'.*5 values.*10 tokens"):
            batch_to_datum_list(batch)

    def test_longer_than_model_input_raises(self):
        """Too many values is the likelier mistake: response-length, not full-sequence.

        Per-token entries cover the whole sequence, so a caller sizing them to
        the response alone lands short, but one that concatenates prompt and
        response twice, or reuses a longer datum's inputs, lands long. Both
        directions have to raise locally.
        """
        batch = [Datum(model_input=ModelInput.from_ints(_TOKENS10), loss_fn_inputs=_per_token(14))]
        with pytest.raises(ValueError, match=r"datum\[0\].*'target_tokens'.*14 values.*10 tokens"):
            batch_to_datum_list(batch)

    def test_batch_level_longer_than_model_input_raises(self):
        with pytest.raises(ValueError, match=r"lossFnInputs\['target_tokens'\]"):
            batch_to_datum_list([_TOKENS10], loss_fn_inputs=_per_token(14))

    def test_prebuilt_wire_dict_longer_checked(self):
        with pytest.raises(ValueError, match=r"datum\[0\]"):
            batch_to_datum_list([_wire_datum(_TOKENS10, _per_token(14))])

    def test_matched_lengths_pass(self):
        batch = [Datum(model_input=ModelInput.from_ints(_TOKENS10), loss_fn_inputs=_per_token(10))]
        result = batch_to_datum_list(batch)
        assert len(result[0]["lossFnInputs"]["weights"]["data"]) == 10

    def test_batch_level_loss_fn_inputs_checked(self):
        # loss_fn_inputs passed alongside the batch is merged in after each
        # datum is built, so it has to be checked at the end of the batch path.
        with pytest.raises(ValueError, match=r"lossFnInputs\['target_tokens'\]"):
            batch_to_datum_list([_TOKENS10], loss_fn_inputs=_per_token(5))

    def test_batch_level_loss_fn_inputs_matched_passes(self):
        result = batch_to_datum_list([_TOKENS10], loss_fn_inputs=_per_token(10))
        assert len(result[0]["lossFnInputs"]["target_tokens"]["data"]) == 10

    def test_prebuilt_wire_dict_checked(self):
        with pytest.raises(ValueError, match=r"datum\[0\]"):
            batch_to_datum_list([_wire_datum(_TOKENS10, _per_token(5))])

    def test_offending_datum_index_reported(self):
        batch = [
            _wire_datum(_TOKENS10, _per_token(10)),
            _wire_datum(_TOKENS10, _per_token(10)),
            _wire_datum(_TOKENS10, _per_token(4)),
        ]
        with pytest.raises(ValueError, match=r"datum\[2\]"):
            batch_to_datum_list(batch)

    def test_mixed_batch_checked(self):
        batch = [
            Datum(model_input=ModelInput.from_ints(_TOKENS10), loss_fn_inputs=_per_token(10)),
            _wire_datum(_TOKENS10, _per_token(4)),
        ]
        with pytest.raises(ValueError, match=r"datum\[1\]"):
            batch_to_datum_list(batch)

    def test_tensor_data_values_checked(self):
        batch = [
            Datum(
                model_input=ModelInput.from_ints(_TOKENS10),
                loss_fn_inputs={
                    "weights": TensorData(data=[1.0] * 5, shape=[5]),
                    "target_tokens": TensorData(data=list(range(5)), shape=[5]),
                },
            )
        ]
        with pytest.raises(ValueError, match=r"5 values.*10 tokens"):
            batch_to_datum_list(batch)

    def test_tensor_data_matched_passes(self):
        batch = [
            Datum(
                model_input=ModelInput.from_ints(_TOKENS10),
                loss_fn_inputs={
                    "weights": TensorData(data=[1.0] * 10, shape=[10]),
                    "target_tokens": TensorData(data=list(range(10)), shape=[10]),
                },
            )
        ]
        assert len(batch_to_datum_list(batch)) == 1

    def test_multi_chunk_length_summed(self):
        split = {"modelInput": {"chunks": [{"tokens": _TOKENS10[:4]}, {"tokens": _TOKENS10[4:]}]}}
        ok = {**split, "lossFnInputs": _per_token(10)}
        assert len(batch_to_datum_list([ok])) == 1
        bad = {**split, "lossFnInputs": _per_token(9)}
        with pytest.raises(ValueError, match=r"9 values.*10 tokens"):
            batch_to_datum_list([bad])

    # --- entries that are not per-token must not be rejected ---

    def test_scalar_knob_not_checked(self):
        # A scalar sits alongside per-token entries with its own length.
        batch = [
            Datum(
                model_input=ModelInput.from_ints(_TOKENS10),
                loss_fn_inputs={**_per_token(10), "temperature": [0.7]},
            )
        ]
        assert len(batch_to_datum_list(batch)) == 1

    def test_unrecognised_key_not_checked(self):
        # Only known per-token keys are checked, so a key this version does not
        # know about is passed through for the service to interpret.
        batch = [
            Datum(
                model_input=ModelInput.from_ints(_TOKENS10),
                loss_fn_inputs={**_per_token(10), "some_future_knob": [1.0, 2.0]},
            )
        ]
        assert len(batch_to_datum_list(batch)) == 1

    def test_entries_the_service_does_not_enforce_are_not_checked(self):
        # ``advantages`` is reshaped rather than length-checked by the service,
        # so a request carrying a short one is still accepted and must not be
        # rejected here. Keeps this check from being stricter than the service.
        batch = [
            Datum(
                model_input=ModelInput.from_ints(_TOKENS10),
                loss_fn_inputs={"advantages": [0.5], "ref_log_prob": [0.1]},
            )
        ]
        assert len(batch_to_datum_list(batch)) == 1

    def test_text_prompt_not_checked(self):
        # Token count is not known until the text is tokenized.
        batch = [{"modelInput": {"prompt": "hello world"}, "lossFnInputs": _per_token(5)}]
        assert len(batch_to_datum_list(batch)) == 1

    def test_no_loss_fn_inputs_passes(self):
        assert len(batch_to_datum_list([_TOKENS10])) == 1

    def test_labels_only_not_checked(self):
        # ``labels`` is not a per-token entry, so it carries its own length.
        assert len(batch_to_datum_list([[1]], loss_fn_inputs={"labels": [0]})) == 1


class _FakeMP:
    """Stubs list_model_packages (paginated) + describe_model_package."""

    def __init__(self, pages=None, describe=None):
        # pages: list of (summaries, next_token) served in order.
        self._pages = list(pages or [])
        self._describe = describe or {}
        self.list_calls: list[dict] = []
        self.describe_calls: list[dict] = []

    def list_model_packages(self, **kwargs):
        self.list_calls.append(kwargs)
        summaries, next_token = self._pages.pop(0)
        resp = {"ModelPackageSummaryList": summaries}
        if next_token:
            resp["NextToken"] = next_token
        return resp

    def describe_model_package(self, **kwargs):
        self.describe_calls.append(kwargs)
        return self._describe[kwargs["ModelPackageName"]]


def _summary(arn, version, desc="d"):
    return {
        "ModelPackageArn": arn,
        "ModelPackageVersion": version,
        "ModelPackageDescription": desc,
        "CreationTime": "2026-06-20T00:00:00Z",
    }


class TestListCheckpointSummaries:
    def test_sorts_descending_and_caps(self):
        client = _FakeMP(pages=[([_summary("a/3", 3), _summary("a/2", 2)], "tok")])
        out = list_model_package_summaries(client, "grp", max_results=2)
        assert [s["ModelPackageVersion"] for s in out] == [3, 2]
        call = client.list_calls[0]
        assert call["ModelPackageGroupName"] == "grp"
        assert call["SortBy"] == "CreationTime"
        assert call["SortOrder"] == "Descending"
        assert call["MaxResults"] == 2
        # Capped at 2 -> stopped paging despite NextToken.
        assert len(client.list_calls) == 1

    def test_auto_paginates_until_limit(self):
        client = _FakeMP(pages=[([_summary("a/3", 3)], "tok"), ([_summary("a/2", 2)], None)])
        out = list_model_package_summaries(client, "grp", max_results=10)
        assert [s["ModelPackageVersion"] for s in out] == [3, 2]
        assert client.list_calls[1]["NextToken"] == "tok"

    def test_none_returns_all(self):
        client = _FakeMP(pages=[([_summary("a/2", 2)], "tok"), ([_summary("a/1", 1)], None)])
        out = list_model_package_summaries(client, "grp", max_results=None)
        assert len(out) == 2
        # No MaxResults sent when unbounded.
        assert "MaxResults" not in client.list_calls[0]

    def test_cap_above_one_page_pages_instead_of_asking_for_it(self):
        """A cap larger than the page size is served by paging.

        ListModelPackages accepts at most MAX_RESULTS_PER_PAGE per request and
        rejects anything above it, so sending the caller's cap verbatim would
        fail the call rather than return that many.
        """
        first = [_summary(f"a/{index}", index) for index in range(100)]
        client = _FakeMP(pages=[(first, "tok"), ([_summary("a/100", 100)], None)])

        out = list_model_package_summaries(client, "grp", max_results=150)

        assert len(out) == 101
        assert [call["MaxResults"] for call in client.list_calls] == [100, 50]


class TestDescribeModelPackageKind:
    def test_is_checkpoint_true_is_checkpoint_kind(self):
        client = _FakeMP(
            describe={"a/1": {"InferenceSpecification": {"Containers": [{"IsCheckpoint": True}]}}}
        )
        assert describe_model_package_kind(client, "a/1") == ModelPackageKind.CHECKPOINT

    def test_is_checkpoint_absent_is_model(self):
        client = _FakeMP(describe={"a/1": {"InferenceSpecification": {"Containers": [{}]}}})
        assert describe_model_package_kind(client, "a/1") == ModelPackageKind.MODEL

    def test_no_inference_spec_is_model(self):
        client = _FakeMP(describe={"a/1": {}})
        assert describe_model_package_kind(client, "a/1") == ModelPackageKind.MODEL


class _CompatChunk:
    def __init__(self, tokens):
        self.tokens = tokens


class _CompatModelInput:
    def __init__(self, chunks):
        self.chunks = chunks


class _CompatDatum:
    def __init__(self, model_input, loss_fn_inputs=None):
        self.model_input = model_input
        self.loss_fn_inputs = loss_fn_inputs or {}


class TestTypedInputWiring:
    """Typed objects (ours and attribute-compatible external ones) flow through the input paths."""

    def test_batch_of_datum_objects(self):
        batch = [Datum(model_input=ModelInput.from_ints([1, 2]))]
        out = batch_to_datum_list(batch)
        assert out == [{"modelInput": {"chunks": [{"tokens": [1, 2], "type": "encoded_text"}]}}]

    def test_batch_of_model_input_objects(self):
        out = batch_to_datum_list([ModelInput.from_ints([3])])
        assert out[0]["modelInput"]["chunks"][0]["tokens"] == [3]

    def test_batch_of_compatible_datums(self):
        batch = [
            _CompatDatum(
                model_input=_CompatModelInput([_CompatChunk([1, 2])]),
                loss_fn_inputs={"advantages": TensorData(data=[0.5])},
            )
        ]
        out = batch_to_datum_list(batch)
        assert out[0]["modelInput"]["chunks"][0]["tokens"] == [1, 2]
        assert out[0]["lossFnInputs"]["advantages"] == {"data": [0.5], "dtype": "float32"}

    def test_typed_batch_fills_shared_loss_inputs_only_where_absent(self):
        batch = [
            Datum(model_input=ModelInput.from_ints([1]), loss_fn_inputs={"w": [1.0]}),
            Datum(model_input=ModelInput.from_ints([2])),
        ]
        out = batch_to_datum_list(batch, loss_fn_inputs={"w": TensorData(data=[9.0])})
        assert out[0]["lossFnInputs"] == {"w": {"data": [1.0], "dtype": "float32", "shape": [1]}}
        assert out[1]["lossFnInputs"] == {"w": {"data": [9.0], "dtype": "float32"}}

    def test_typed_batch_shared_loss_inputs_not_aliased(self):
        batch = [
            Datum(model_input=ModelInput.from_ints([1])),
            Datum(model_input=ModelInput.from_ints([2])),
        ]
        out = batch_to_datum_list(batch, loss_fn_inputs={"w": [1.0]})
        out[0]["lossFnInputs"]["extra"] = True
        assert "extra" not in out[1]["lossFnInputs"]

    def test_tokenize_to_datum_serializes_tensor_data(self):
        datum = tokenize_to_datum([1], loss_fn_inputs={"advantages": TensorData(data=[0.1])})
        assert datum["lossFnInputs"]["advantages"] == {"data": [0.1], "dtype": "float32"}

    def test_dict_batch_shared_tensor_loss_inputs_serialized(self):
        batch = [{"modelInput": {"chunks": [{"tokens": [1], "type": "encoded_text"}]}}]
        out = batch_to_datum_list(batch, loss_fn_inputs={"adv": TensorData(data=[0.2])})
        assert out[0]["lossFnInputs"]["adv"] == {"data": [0.2], "dtype": "float32"}

    def test_prompt_model_input_object(self):
        out = prompt_to_model_input(ModelInput.from_ints([1, 2]))
        assert out == {"chunks": [{"tokens": [1, 2], "type": "encoded_text"}]}

    def test_prompt_encoded_text_chunk(self):
        out = prompt_to_model_input(EncodedTextChunk(tokens=[5]))
        assert out == {"chunks": [{"tokens": [5], "type": "encoded_text"}]}

    def test_prompt_compatible_model_input(self):
        compat = _CompatModelInput([_CompatChunk([7, 8])])
        out = prompt_to_model_input(compat)
        assert out == {"chunks": [{"tokens": [7, 8], "type": "encoded_text"}]}

    def test_prompt_dict_still_passes_through(self):
        raw = {"chunks": [{"tokens": [9], "type": "encoded_text"}]}
        assert prompt_to_model_input(raw) is raw


class TestBaseModelArnFromModelPackage:
    def _desc(self, base):
        return {
            "ModelPackageArn": "arn:mp/weights-7",
            "InferenceSpecification": {"Containers": [{"IsCheckpoint": False, "BaseModel": base}]},
        }

    def test_builds_versioned_hub_arn(self):
        from sagemaker.train.training_session._utils import (
            base_model_arn_from_model_package,
        )

        arn = base_model_arn_from_model_package(
            self._desc(
                {"HubContentName": "hubcontent-text-model-a", "HubContentVersion": "3.15.0"}
            ),
            region="us-west-2",
        )
        assert arn == (
            "arn:aws:sagemaker:us-west-2:aws:hub-content/"
            "SageMakerPublicHub/Model/hubcontent-text-model-a/3.15.0"
        )

    def test_missing_base_model_raises(self):
        from sagemaker.train.training_session._utils import (
            base_model_arn_from_model_package,
        )

        with pytest.raises(ValueError, match="no base-model reference"):
            base_model_arn_from_model_package(self._desc({}), region="us-west-2")

    def test_no_containers_raises(self):
        from sagemaker.train.training_session._utils import (
            base_model_arn_from_model_package,
        )

        with pytest.raises(ValueError, match="no base-model reference"):
            base_model_arn_from_model_package({"InferenceSpecification": {}}, region="us-west-2")


class TestSplitS3Uri:
    def test_bucket_and_key(self):
        from sagemaker.train.training_session._utils import split_s3_uri

        assert split_s3_uri("s3://b/a/b/c/") == ("b", "a/b/c/")

    def test_bucket_only_gives_empty_key(self):
        from sagemaker.train.training_session._utils import split_s3_uri

        assert split_s3_uri("s3://b") == ("b", "")
        assert split_s3_uri("s3://b/") == ("b", "")

    def test_non_s3_scheme_raises(self):
        from sagemaker.train.training_session._utils import split_s3_uri

        with pytest.raises(ValueError, match="not an s3:// URI"):
            split_s3_uri("https://bucket.s3.amazonaws.com/key")

    def test_no_bucket_raises(self):
        from sagemaker.train.training_session._utils import split_s3_uri

        with pytest.raises(ValueError, match="names no bucket"):
            split_s3_uri("s3:///key")


class TestDeleteS3ObjectsUnder:
    """``DeleteObjects`` caps at 1000 keys per call, so a larger page is split."""

    class _S3:
        def __init__(self, keys):
            self._keys = keys
            self.delete_calls: list[dict] = []

        def list_objects_v2(self, **kwargs):
            # One oversized page, to prove batching is not left to the lister.
            return {"Contents": [{"Key": k} for k in self._keys], "IsTruncated": False}

        def delete_objects(self, **kwargs):
            self.delete_calls.append(kwargs)
            return {}

    def test_batches_at_the_delete_objects_limit(self):
        from sagemaker.train.training_session._utils import (
            delete_s3_objects_under,
        )

        s3 = self._S3([f"p/{i}" for i in range(2500)])
        deleted, failed = delete_s3_objects_under(s3, "s3://b/p/")

        assert (deleted, failed) == (2500, [])
        sizes = [len(c["Delete"]["Objects"]) for c in s3.delete_calls]
        assert sizes == [1000, 1000, 500]

    def test_runs_quiet_so_only_failures_come_back(self):
        from sagemaker.train.training_session._utils import (
            delete_s3_objects_under,
        )

        s3 = self._S3(["p/1"])
        delete_s3_objects_under(s3, "s3://b/p/")

        assert s3.delete_calls[0]["Delete"]["Quiet"] is True


class TestDescribeModelPackageArtifacts:
    """Describe carries the location in a nested optional structure, so every
    level of it can be absent."""

    @staticmethod
    def _client(response):
        class _SageMaker:
            def __init__(self):
                self.calls: list[dict] = []

            def describe_model_package(self, **kwargs):
                self.calls.append(kwargs)
                return response

        return _SageMaker()

    def _describe(self, response, arn="arn:aws:sagemaker:us-west-2:123:model-package/g/1"):
        from sagemaker.train.training_session._utils import (
            describe_model_package_artifacts,
        )

        client = self._client(response)
        return describe_model_package_artifacts(client, arn), client

    def test_reads_location_type_and_managed_storage(self):
        artifacts, client = self._describe(
            {
                "ManagedStorageType": "None",
                "InferenceSpecification": {
                    "Containers": [
                        {
                            "ModelDataSource": {
                                "S3DataSource": {
                                    "S3Uri": "s3://b/g/req-1/model/",
                                    "S3DataType": "S3Prefix",
                                }
                            }
                        }
                    ]
                },
            }
        )

        assert artifacts == {
            "s3_uri": "s3://b/g/req-1/model/",
            "s3_data_type": "S3Prefix",
            "managed_storage_type": "None",
        }
        assert client.calls == [
            {"ModelPackageName": "arn:aws:sagemaker:us-west-2:123:model-package/g/1"}
        ]

    @pytest.mark.parametrize(
        "response",
        [
            pytest.param({}, id="no-inference-specification"),
            pytest.param({"InferenceSpecification": {}}, id="no-containers-key"),
            pytest.param({"InferenceSpecification": {"Containers": []}}, id="empty-containers"),
            pytest.param(
                {"InferenceSpecification": {"Containers": [{}]}}, id="no-model-data-source"
            ),
            pytest.param(
                {"InferenceSpecification": {"Containers": [{"ModelDataSource": {}}]}},
                id="no-s3-data-source",
            ),
            pytest.param(
                {
                    "InferenceSpecification": {
                        "Containers": [{"ModelDataSource": {"S3DataSource": {"S3Uri": ""}}}]
                    }
                },
                id="empty-s3-uri",
            ),
            pytest.param(
                # The legacy container shape. Nothing this SDK saves uses it,
                # and reporting no location keeps such a package's artifacts.
                {"InferenceSpecification": {"Containers": [{"ModelDataUrl": "s3://b/k"}]}},
                id="legacy-model-data-url",
            ),
        ],
    )
    def test_absent_at_any_level_reads_as_no_location(self, response):
        artifacts, _ = self._describe(response)

        assert artifacts == {
            "s3_uri": None,
            "s3_data_type": None,
            "managed_storage_type": None,
        }

    def test_managed_storage_without_a_location(self):
        from sagemaker.train.training_session._utils import (
            MANAGED_STORAGE_RESTRICTED,
        )

        artifacts, _ = self._describe({"ManagedStorageType": MANAGED_STORAGE_RESTRICTED})

        assert artifacts["managed_storage_type"] == MANAGED_STORAGE_RESTRICTED
        assert artifacts["s3_uri"] is None


class TestDeleteS3ObjectsUnderEdges:
    class _S3:
        """Serves scripted ``(keys, next_token)`` pages, optionally reporting
        per-key failures the way ``DeleteObjects`` does: in the body."""

        def __init__(self, pages, errors_for=(), truncated_without_token=False):
            self._pages = list(pages)
            self._errors_for = set(errors_for)
            self._truncated_without_token = truncated_without_token
            self.list_calls: list[dict] = []
            self.delete_calls: list[dict] = []

        def list_objects_v2(self, **kwargs):
            self.list_calls.append(kwargs)
            keys, next_token = self._pages.pop(0)
            resp = {"IsTruncated": bool(next_token) or self._truncated_without_token}
            if keys:
                resp["Contents"] = [{"Key": k} for k in keys]
            if next_token:
                resp["NextContinuationToken"] = next_token
            return resp

        def delete_objects(self, **kwargs):
            self.delete_calls.append(kwargs)
            sent = [o["Key"] for o in kwargs["Delete"]["Objects"]]
            errors = [{"Key": k, "Code": "AccessDenied"} for k in sent if k in self._errors_for]
            return {"Errors": errors} if errors else {}

    def _delete(self, s3, uri="s3://b/p/", **kwargs):
        from sagemaker.train.training_session._utils import (
            delete_s3_objects_under,
        )

        return delete_s3_objects_under(s3, uri, **kwargs)

    def test_truncated_without_a_token_stops(self):
        """A truncated page with no continuation token would otherwise re-request
        page one forever."""
        s3 = self._S3([(["p/1"], None)], truncated_without_token=True)

        deleted, failed = self._delete(s3)

        assert (deleted, failed) == (1, [])
        assert len(s3.list_calls) == 1

    def test_failures_accumulate_across_pages(self):
        """Each page is deleted as it is listed, so a later page's failures must
        join the earlier ones rather than replace them."""
        s3 = self._S3([(["p/1", "p/2"], "tok"), (["p/3"], None)], errors_for=["p/1", "p/3"])

        deleted, failed = self._delete(s3)

        assert deleted == 1
        assert sorted(failed) == ["p/1", "p/3"]

    def test_empty_page_deletes_nothing(self):
        s3 = self._S3([([], None)])

        assert self._delete(s3) == (0, [])
        assert s3.delete_calls == []

    def test_no_data_type_is_treated_as_a_prefix(self):
        """A package that carries a location but no S3DataType is swept, not read
        as a single key: the location is a directory in every case this SDK
        writes."""
        s3 = self._S3([(["p/1"], None)])

        self._delete(s3, s3_data_type=None)

        assert s3.list_calls == [{"Bucket": "b", "Prefix": "p/"}]

    def test_single_object_failure_comes_back(self):
        s3 = self._S3([], errors_for=["p/model.tar.gz"])

        deleted, failed = self._delete(s3, uri="s3://b/p/model.tar.gz", s3_data_type="S3Object")

        assert (deleted, failed) == (0, ["p/model.tar.gz"])
        assert s3.list_calls == []

    @pytest.mark.parametrize("uri", ["s3://b", "s3://b/"])
    def test_bucket_root_refused_before_any_call(self, uri):
        s3 = self._S3([])

        with pytest.raises(ValueError, match="bucket root"):
            self._delete(s3, uri=uri)

        assert s3.list_calls == []
        assert s3.delete_calls == []

    def test_a_token_on_an_untruncated_page_is_not_followed(self):
        """The loop stops when either signal says "done", so a final page that
        still carries a continuation token ends the sweep instead of re-listing
        from that token."""
        from sagemaker.train.training_session._utils import (
            delete_s3_objects_under,
        )

        class _S3:
            def __init__(self):
                self.list_calls = 0

            def list_objects_v2(self, **kwargs):
                self.list_calls += 1
                return {
                    "Contents": [{"Key": "p/1"}],
                    "IsTruncated": False,
                    "NextContinuationToken": "tok",
                }

            def delete_objects(self, **kwargs):
                return {}

        s3 = _S3()
        deleted, failed = delete_s3_objects_under(s3, "s3://b/p/")

        assert (deleted, failed) == (1, [])
        assert s3.list_calls == 1

    def test_a_failure_without_a_key_reads_as_deleted(self):
        """``DeleteObjects`` keys every error by object; an entry with no ``Key``
        names nothing to retry or report, so it is dropped rather than becoming a
        ``None`` in ``failed_keys`` that the caller would have to filter. Its key
        therefore counts as deleted, documented here because it means
        ``deleted`` is "not reported failed", not "confirmed gone"."""
        from sagemaker.train.training_session._utils import (
            delete_s3_objects_under,
        )

        class _S3:
            def list_objects_v2(self, **kwargs):
                return {"Contents": [{"Key": "p/1"}], "IsTruncated": False}

            def delete_objects(self, **kwargs):
                return {"Errors": [{"Code": "InternalError"}]}

        deleted, failed = delete_s3_objects_under(_S3(), "s3://b/p/")

        assert failed == []
        assert deleted == 1


class TestMixedBatchStartingWithATokenId:
    """A batch is only a bare token list if every element is a token.

    One that merely starts with an int skipped the mixed-kind check and was read
    as a single token list, putting the other elements inside a token array and
    sending that.
    """

    def test_a_string_after_a_token_id_raises(self):
        with pytest.raises(TypeError, match="only token IDs"):
            batch_to_datum_list([10, "hello"], tokenizer=None, loss_fn_inputs=None)

    def test_the_message_names_what_it_found(self):
        with pytest.raises(TypeError, match="str"):
            batch_to_datum_list([10, "hello"], tokenizer=None, loss_fn_inputs=None)

    def test_a_list_after_a_token_id_raises(self):
        with pytest.raises(TypeError, match="only token IDs"):
            batch_to_datum_list([10, [11, 12]], tokenizer=None, loss_fn_inputs=None)

    def test_an_all_token_batch_is_still_one_datum(self):
        datums = batch_to_datum_list([1, 2, 3], tokenizer=None, loss_fn_inputs=None)

        assert len(datums) == 1
        assert datums[0]["modelInput"]["chunks"][0]["tokens"] == [1, 2, 3]


class TestChunkedMetricsUseRealDatumCounts:
    """The combined loss is a datum-count-weighted mean over the chunks.

    The counts used to be inferred from per-datum loss outputs, which are absent
    unless the caller asked for them, so every chunk weighed the same and an
    uneven split reported a plain mean instead.
    """

    @staticmethod
    def _result(loss):
        from sagemaker.train.training_session.types import ForwardBackwardOutput

        return ForwardBackwardOutput._from_api({"forwardBackward": {"metrics": {"loss": loss}}})

    def test_uneven_chunks_are_weighted_by_size(self):
        results = [self._result(1.0), self._result(1.0), self._result(11.0)]

        combined = combine_forward_backward_results(results, datum_counts=[1024, 1024, 100])

        expected = (1.0 * 1024 + 1.0 * 1024 + 11.0 * 100) / 2148
        assert combined.metrics["loss"] == pytest.approx(expected)

    def test_without_counts_it_falls_back_to_equal_weight(self):
        """Documented fallback: no counts and no per-datum outputs means a mean."""
        results = [self._result(1.0), self._result(1.0), self._result(11.0)]

        combined = combine_forward_backward_results(results)

        assert combined.metrics["loss"] == pytest.approx(13.0 / 3)

    def test_grad_norm_still_comes_from_the_last_chunk(self):
        from sagemaker.train.training_session.types import ForwardBackwardOutput

        results = [
            ForwardBackwardOutput._from_api({"forwardBackward": {"metrics": {"grad_norm": 1.0}}}),
            ForwardBackwardOutput._from_api({"forwardBackward": {"metrics": {"grad_norm": 7.0}}}),
        ]

        combined = combine_forward_backward_results(results, datum_counts=[1000, 1])

        assert combined.metrics["grad_norm"] == 7.0


class TestHubContentArnPartition:
    """Every ARN this package builds resolves its partition rather than assuming."""

    def test_the_partition_comes_from_the_model_package_arn(self):
        assert _arn_partition("arn:aws-cn:sagemaker:cn-north-1:1:model-package/g/1") == "aws-cn"
        assert _arn_partition("arn:aws-us-gov:sagemaker:us-gov-west-1:1:x/y") == "aws-us-gov"

    def test_an_unreadable_arn_falls_back_to_the_commercial_partition(self):
        assert _arn_partition(None) == "aws"
        assert _arn_partition("") == "aws"
        assert _arn_partition("not-an-arn") == "aws"


class TestMissingLossFnInputsAreRefused:
    """A datum must carry the loss inputs its loss function requires.

    The service refuses an operation whose datums lack them, and that refusal
    names neither the datum nor the key. Catching it here can name both, before
    the operation is accepted and queued.
    """

    def test_a_subset_names_what_is_missing(self):
        with pytest.raises(ValueError, match="logprobs"):
            batch_to_datum_list(
                [[5, 6]],
                loss_fn_inputs={"advantages": [1.0, 1.0]},
                loss_fn_name="ppo",
            )

    def test_the_message_names_the_datum_and_what_is_present(self):
        with pytest.raises(ValueError, match=r"datum\[0\].*advantages"):
            batch_to_datum_list(
                [[5, 6]],
                loss_fn_inputs={"advantages": [1.0, 1.0]},
                loss_fn_name="ppo",
            )

    def test_no_loss_inputs_at_all_is_refused(self):
        """Absent is the case the quick starts hit, so it has to raise here.

        The shape leaves ``lossFnInputs`` optional, but a loss function's own
        requirement is not optional: the operation is refused for a datum
        without them.
        """
        with pytest.raises(ValueError, match="requires"):
            batch_to_datum_list([[5, 6]], loss_fn_name="cross_entropy")

    def test_a_batch_without_a_named_loss_is_still_left_alone(self):
        """No loss named, nothing to require: the caller has not chosen one yet."""
        datums = batch_to_datum_list([[5, 6]])

        assert "lossFnInputs" not in datums[0]

    def test_a_complete_set_passes(self):
        datums = batch_to_datum_list(
            [[5, 6]],
            loss_fn_inputs={
                "target_tokens": [6, 0],
                "weights": [0.0, 1.0],
                "logprobs": [-0.1, -0.2],
                "advantages": [1.0, 1.0],
            },
            loss_fn_name="ppo",
        )

        assert sorted(datums[0]["lossFnInputs"]) == [
            "advantages",
            "logprobs",
            "target_tokens",
            "weights",
        ]

    def test_cross_entropy_needs_only_its_own_two(self):
        datums = batch_to_datum_list(
            [[5, 6]],
            loss_fn_inputs={"target_tokens": [6, 0], "weights": [0.0, 1.0]},
            loss_fn_name="cross_entropy",
        )

        assert datums[0]["lossFnInputs"]

    def test_an_unnamed_loss_function_is_not_second_guessed(self):
        """No name given, nothing is assumed about which entries are wanted."""
        datums = batch_to_datum_list([[5, 6]], loss_fn_inputs={"advantages": [1.0, 1.0]})

        assert datums[0]["lossFnInputs"]


class TestHomogeneousDatumDictBatches:
    """A batch of datum dicts is read in every spelling ``from_any`` accepts.

    Only the wire spelling (``modelInput``) had a homogeneous branch, so a list
    of snake_case dicts was refused outright while the very same dict inside a
    mixed batch was read correctly.
    """

    _EXPECTED = [
        {"modelInput": {"chunks": [{"tokens": [1, 2, 3], "type": "encoded_text"}]}},
        {"modelInput": {"chunks": [{"tokens": [4, 5, 6], "type": "encoded_text"}]}},
    ]

    def test_snake_case_keys(self):
        batch = [{"model_input": [1, 2, 3]}, {"model_input": [4, 5, 6]}]

        assert batch_to_datum_list(batch) == self._EXPECTED

    def test_the_prompt_shorthand(self):
        batch = [{"prompt": [1, 2, 3]}, {"prompt": [4, 5, 6]}]

        assert batch_to_datum_list(batch) == self._EXPECTED

    def test_the_wire_spelling_is_still_passed_through_unchanged(self):
        batch = [dict(d) for d in self._EXPECTED]

        assert batch_to_datum_list(batch) == self._EXPECTED

    def test_loss_inputs_still_reach_a_snake_case_batch(self):
        batch = [{"model_input": [1, 2]}, {"model_input": [3, 4]}]

        datums = batch_to_datum_list(batch, loss_fn_inputs={"weights": [1.0, 1.0]})

        assert all("lossFnInputs" in d for d in datums)

    def test_a_batch_mixing_the_two_dict_spellings_reads_each_element(self):
        """Which spelling comes first must not decide how the rest are read.

        Keying the batch off its first element sent the later ones through the
        wrong reading of their own keys, so one order was coerced and the other
        refused.
        """
        wire = {"modelInput": {"chunks": [{"tokens": [1], "type": "encoded_text"}]}}
        snake = {"model_input": [2, 3]}
        expected = [
            wire,
            {"modelInput": {"chunks": [{"tokens": [2, 3], "type": "encoded_text"}]}},
        ]

        assert batch_to_datum_list([wire, snake]) == expected
        assert batch_to_datum_list([snake, wire]) == list(reversed(expected))

    def test_a_dict_naming_no_input_is_still_refused(self):
        with pytest.raises(TypeError, match="Unsupported batch type"):
            batch_to_datum_list([{"nothing_useful": 1}, {"nothing_useful": 2}])

    def test_loss_inputs_alone_do_not_make_a_datum(self):
        """``input_ids``/``messages`` are not spellings this accepts.

        Reading them as datums built an input-less example that only the
        service would refuse, so the batch is rejected here instead, where the
        error can name the shapes a batch may hold.
        """
        with pytest.raises(TypeError, match="Unsupported batch type"):
            batch_to_datum_list(
                [{"input_ids": [1, 2], "loss_fn_inputs": {"weights": [1.0, 1.0]}}] * 2
            )
        with pytest.raises(TypeError, match="Unsupported batch type"):
            batch_to_datum_list([{"messages": [{"role": "user"}]}] * 2)


class TestChunkedMetricsKeepEveryReportedKey:
    """Combining read only the last chunk's metric keys.

    A metric the earlier chunks reported and the last one omitted vanished from
    the combined result, so a chunked batch silently reported fewer metrics than
    the same batch sent whole.
    """

    @staticmethod
    def _result(metrics):
        from sagemaker.train.training_session.types import ForwardBackwardOutput

        return ForwardBackwardOutput._from_api({"forwardBackward": {"metrics": metrics}})

    def test_a_metric_only_an_earlier_chunk_reports_survives(self):
        results = [self._result({"loss": 1.0, "aux": 7.0}), self._result({"loss": 3.0})]

        combined = combine_forward_backward_results(results, datum_counts=[1, 1])

        assert combined.metrics["aux"] == pytest.approx(7.0)
        assert combined.metrics["loss"] == pytest.approx(2.0)

    def test_it_is_weighted_over_only_the_chunks_that_reported_it(self):
        """Weighting over every chunk would dilute it toward zero."""
        results = [self._result({"aux": 2.0}), self._result({"aux": 4.0}), self._result({})]

        combined = combine_forward_backward_results(results, datum_counts=[1, 3, 96])

        assert combined.metrics["aux"] == pytest.approx((2.0 * 1 + 4.0 * 3) / 4)

    def test_a_namespaced_grad_norm_comes_from_the_final_chunk(self):
        """The reported key is namespaced, so the suffix is what identifies it.

        Matching the bare name left the real one to be averaged, and the mean of
        two gradient norms is not a gradient norm.
        """
        results = [
            self._result({"actor/grad_norm": 1.0}),
            self._result({"actor/grad_norm": 7.0}),
        ]

        combined = combine_forward_backward_results(results, datum_counts=[1000, 1])

        assert combined.metrics["actor/grad_norm"] == 7.0

    def test_a_token_count_is_totalled_rather_than_averaged(self):
        """Each chunk counts part of one batch, so the batch value is the sum.

        Averaging a count reports a number no chunk measured: for the split
        below it gives 11502, which is lower than the first chunk's own 12288
        even though the batch held more tokens than that chunk did.
        """
        results = [
            self._result({"global_token_num": 12288.0}),
            self._result({"global_token_num": 912.0}),
        ]

        combined = combine_forward_backward_results(results, datum_counts=[1024, 76])

        assert combined.metrics["global_token_num"] == 13200.0

    def test_a_token_count_from_a_single_chunk_is_unchanged(self):
        """One chunk is not combined at all, so the count passes through."""
        results = [self._result({"global_token_num": 192.0})]

        combined = combine_forward_backward_results(results, datum_counts=[16])

        assert combined.metrics["global_token_num"] == 192.0

    def test_a_per_datum_metric_is_still_averaged(self):
        """The default stays a mean; only the named counts are totalled."""
        results = [
            self._result({"actor/loss": 2.0, "global_token_num": 100.0}),
            self._result({"actor/loss": 4.0, "global_token_num": 10.0}),
        ]

        combined = combine_forward_backward_results(results, datum_counts=[1, 1])

        assert combined.metrics["actor/loss"] == 3.0
        assert combined.metrics["global_token_num"] == 110.0

    def test_a_grad_norm_only_an_earlier_chunk_reports_is_omitted(self):
        """A partial norm is worse to report than none.

        The key union keeps every other metric an earlier chunk reported, but a
        gradient norm describes the gradient accumulated across all chunks, so
        one measured partway through is not the whole-batch value.
        """
        results = [
            self._result({"actor/grad_norm": 9.0, "actor/loss": 1.0}),
            self._result({"actor/loss": 3.0}),
        ]

        combined = combine_forward_backward_results(results, datum_counts=[1, 1])

        assert "actor/grad_norm" not in combined.metrics
        assert combined.metrics["actor/loss"] == 2.0
