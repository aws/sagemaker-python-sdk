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
"""``HubContent.get_all_cached`` serves a hub from the hub catalog cache.

Before this change every list of a hub described every item: 766 ``DescribeHubContent``
calls and 60 to 95 seconds for the public hub, on every call. ``HubCatalogCache`` follows
``JumpStartModelsCache``: an ``LRUCache`` of hubs in memory with an expiration horizon, and
a retrieval function that lists the hub and describes only the items whose version
changed. A JSON file per hub, content type, region, and account lets a new process start
from the previous catalog.
"""

from __future__ import absolute_import

import datetime
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.core import hub_catalog
from sagemaker.core.hub_catalog import (
    CACHE_FORMAT_VERSION,
    HUB_CATALOG_DEFAULT_CACHE_EXPIRATION_HORIZON,
    HUB_CATALOG_DEFAULT_MAX_CACHE_ITEMS,
    HubCatalogAccessor,
    HubCatalogCache,
    HubCatalogFile,
    HubCatalogKey,
    HubCatalogSource,
    default_cache_dir,
    reusable_entries,
)
from sagemaker.core.resources import Base, HubContent
from sagemaker.core.utilities.cache import LRUCache
from sagemaker.core.utils.utils import SageMakerClient

HUB = "SageMakerPublicHub"
ACCOUNT = "123456789012"
CREATED = datetime.datetime(2026, 3, 1, tzinfo=datetime.timezone.utc)
ORIGINAL = datetime.datetime(2025, 6, 1, tzinfo=datetime.timezone.utc)


def _summary(name, version="1.0.0"):
    return {
        "HubContentName": name,
        "HubContentArn": f"arn:aws:sagemaker:us-west-2:aws:hub-content/{HUB}/Model/{name}/{version}",
        "HubContentVersion": version,
        "HubContentType": "Model",
        "DocumentSchemaVersion": "2.0.0",
        "HubContentStatus": "Available",
        "HubContentSearchKeywords": ["@license:apache-2.0", "@model-size:1b-10b"],
        "CreationTime": CREATED,
        "OriginalCreationTime": ORIGINAL,
    }


class _Hub:
    """A fake SageMaker client over a dict of name to version."""

    def __init__(self, versions):
        self.versions = dict(versions)
        self.client = MagicMock()
        self.client.list_hub_contents.side_effect = self._list
        self.client.describe_hub_content.side_effect = self._describe
        self.described = []
        self.lists = 0
        self.in_flight = 0
        self.peak_in_flight = 0
        self._lock = threading.Lock()

    def _list(self, **kwargs):
        self.lists += 1
        names = sorted(self.versions)
        page = kwargs.get("NextToken")
        if page is None:
            first, token = names[:2], ("page-2" if len(names) > 2 else None)
        else:
            first, token = names[2:], None
        response = {"HubContentSummaries": [_summary(n, self.versions[n]) for n in first]}
        if token:
            response["NextToken"] = token
        return response

    def _describe(self, **kwargs):
        assert kwargs["HubName"]
        name = kwargs["HubContentName"]
        with self._lock:
            self.described.append(name)
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            summary = _summary(name, self.versions[name])
            summary.pop("OriginalCreationTime")
            return {
                **summary,
                "HubName": kwargs["HubName"],
                "HubArn": f"arn:aws:sagemaker:us-west-2:aws:hub/{HUB}",
                "HubContentDocument": json.dumps(
                    {"License": "Apache-2.0", "Version": self.versions[name]}
                ),
            }
        finally:
            with self._lock:
                self.in_flight -= 1


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv(hub_catalog.CACHE_DIR_ENV_VAR, str(tmp_path / "hub-cache"))
    SageMakerClient.reset()
    HubCatalogAccessor.reset_cache()
    with (
        patch.object(HubContent, "_account_id", return_value=ACCOUNT),
        patch(
            "sagemaker.core.telemetry.telemetry_logging._get_default_sagemaker_session",
            return_value=MagicMock(sagemaker_config=None),
        ),
        patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request"),
    ):
        yield tmp_path / "hub-cache"
    HubCatalogAccessor.reset_cache()
    SageMakerClient.reset()


@pytest.fixture
def telemetry_send():
    with patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request") as send:
        yield send


def _patched(hub):
    return patch.object(Base, "get_sagemaker_client", return_value=hub.client)


def _cache_payload(cache_dir, region="us-west-2", hub=HUB, content_type="Model"):
    return json.loads((cache_dir / f"{ACCOUNT}-{region}-{hub}-{content_type}.json").read_text())


def test_first_call_describes_every_item_and_writes_the_cache_file(_isolated_cache):
    hub = _Hub({"model-a": "1.0.0", "model-b": "2.0.0", "model-c": "1.0.0"})

    with _patched(hub):
        models = HubContent.get_all_cached(HUB, region="us-west-2")

    assert [m.hub_content_name for m in models] == ["model-a", "model-b", "model-c"]
    assert sorted(hub.described) == ["model-a", "model-b", "model-c"]
    for call in hub.client.describe_hub_content.call_args_list:
        assert call.kwargs["HubName"] == HUB
    assert models[0].original_creation_time == ORIGINAL
    assert models[0].hub_content_document == json.dumps(
        {"License": "Apache-2.0", "Version": "1.0.0"}
    )
    assert models[0].license == "apache-2.0"
    assert models[0]._region == "us-west-2"
    payload = _cache_payload(_isolated_cache)
    assert payload["format_version"] == CACHE_FORMAT_VERSION
    assert sorted(payload["entries"]) == ["model-a", "model-b", "model-c"]
    assert payload["entries"]["model-b"]["listed_version"] == "2.0.0"


def test_a_call_inside_the_horizon_makes_no_request():
    hub = _Hub({"model-a": "1.0.0", "model-b": "1.0.0"})
    with _patched(hub):
        first = HubContent.get_all_cached(HUB, region="us-west-2")
        hub.described.clear()
        hub.lists = 0
        second = HubContent.get_all_cached(HUB, region="us-west-2")

    assert (hub.lists, hub.described) == (0, [])
    assert [m.hub_content_name for m in second] == [m.hub_content_name for m in first]
    assert second[0] is not first[0]


def test_refresh_cache_lists_again_and_describes_only_new_and_changed_items():
    hub = _Hub({"model-a": "1.0.0", "model-b": "2.0.0", "model-c": "1.0.0"})
    with _patched(hub):
        HubContent.get_all_cached(HUB, region="us-west-2")
    hub.described.clear()
    hub.versions["model-b"] = "2.1.0"
    hub.versions["model-d"] = "1.0.0"
    del hub.versions["model-c"]

    with _patched(hub):
        refresh = HubContent.refresh_cache(HUB, region="us-west-2")
        models = HubContent.get_all_cached(HUB, region="us-west-2")

    assert sorted(hub.described) == ["model-b", "model-d"]
    assert (refresh.listed, refresh.reused, refresh.described, refresh.removed) == (3, 1, 2, 1)
    assert [m.hub_content_name for m in models] == ["model-a", "model-b", "model-d"]
    assert models[1].hub_content_version == "2.1.0"
    assert models[1].hub_content_document == json.dumps(
        {"License": "Apache-2.0", "Version": "2.1.0"}
    )
    assert models[0].original_creation_time == ORIGINAL


def test_an_expired_entry_lists_again_and_reuses_the_unchanged_items():
    hub = _Hub({"model-a": "1.0.0", "model-b": "1.0.0"})
    HubContent.set_cache_kwargs(cache_expiration_horizon=datetime.timedelta(milliseconds=50))
    with _patched(hub):
        HubContent.get_all_cached(HUB, region="us-west-2")
        assert hub.lists == 1
        hub.described.clear()
        hub.versions["model-b"] = "1.1.0"
        time.sleep(0.1)
        models = HubContent.get_all_cached(HUB, region="us-west-2")

    assert hub.lists == 2
    assert hub.described == ["model-b"]
    assert [m.hub_content_version for m in models] == ["1.0.0", "1.1.0"]


def test_a_new_process_starts_from_the_cache_file(_isolated_cache):
    hub = _Hub({"model-a": "1.0.0", "model-b": "1.0.0"})
    with _patched(hub):
        HubContent.get_all_cached(HUB, region="us-west-2")
    HubCatalogAccessor.reset_cache()
    hub.described.clear()

    with _patched(hub):
        models = HubContent.get_all_cached(HUB, region="us-west-2")

    assert hub.described == []
    assert [m.hub_content_name for m in models] == ["model-a", "model-b"]
    assert models[0].license == "apache-2.0"


def test_cache_files_are_separate_per_account_region_hub_and_type(_isolated_cache):
    hub = _Hub({"model-a": "1.0.0"})
    with _patched(hub):
        HubContent.get_all_cached(HUB, region="us-west-2")
        HubContent.get_all_cached(HUB, region="eu-west-1")
        HubContent.get_all_cached("my-hub", hub_content_type="Notebook", region="us-west-2")

    names = sorted(path.name for path in _isolated_cache.iterdir())
    assert names == [
        f"{ACCOUNT}-eu-west-1-{HUB}-Model.json",
        f"{ACCOUNT}-us-west-2-{HUB}-Model.json",
        f"{ACCOUNT}-us-west-2-my-hub-Notebook.json",
    ]


def test_clear_cache_drops_memory_and_deletes_the_file_once():
    hub = _Hub({"model-a": "1.0.0"})
    with _patched(hub):
        HubContent.get_all_cached(HUB, region="us-west-2")

        assert HubContent.clear_cache(HUB, region="us-west-2") is True
        assert HubContent.clear_cache(HUB, region="us-west-2") is False
        assert HubContent.clear_cache() is False
        hub.described.clear()
        HubContent.get_all_cached(HUB, region="us-west-2")

    assert hub.described == ["model-a"]


def test_an_unreadable_cache_file_is_ignored(_isolated_cache, caplog):
    _isolated_cache.mkdir(parents=True)
    (_isolated_cache / f"{ACCOUNT}-us-west-2-{HUB}-Model.json").write_text("{not json")
    hub = _Hub({"model-a": "1.0.0"})

    with _patched(hub), caplog.at_level("WARNING"):
        models = HubContent.get_all_cached(HUB, region="us-west-2")

    assert [m.hub_content_name for m in models] == ["model-a"]
    assert "Ignoring the unreadable hub content cache" in caplog.text


def test_a_failed_describe_raises_and_keeps_the_described_items_in_the_file(_isolated_cache):
    hub = _Hub({"model-a": "1.0.0", "model-b": "1.0.0", "model-c": "1.0.0"})

    def describe(**kwargs):
        if kwargs["HubContentName"] == "model-c":
            raise RuntimeError("describe of model-c failed")
        return hub._describe(**kwargs)

    hub.client.describe_hub_content.side_effect = describe
    HubContent.set_cache_kwargs(max_workers=1)
    with _patched(hub):
        with pytest.raises(RuntimeError, match="describe of model-c failed"):
            HubContent.get_all_cached(HUB, region="us-west-2")

    assert sorted(_cache_payload(_isolated_cache)["entries"]) == ["model-a", "model-b"]
    hub.client.describe_hub_content.side_effect = hub._describe
    hub.described.clear()
    with _patched(hub):
        refresh = HubContent.refresh_cache(HUB, region="us-west-2")

    assert hub.described == ["model-c"]
    assert (refresh.reused, refresh.described) == (2, 1)


def test_set_cache_kwargs_rebuilds_the_cache_and_rejects_bad_values():
    HubContent.set_cache_kwargs(max_workers=0)
    with pytest.raises(ValueError, match="max_workers must be at least 1"):
        HubCatalogAccessor.get_cache()

    HubContent.set_cache_kwargs(max_cache_items=3)
    cache = HubCatalogAccessor.get_cache()
    assert cache._content_cache._max_cache_items == 3
    assert HubCatalogAccessor.get_cache() is cache


def test_the_cache_is_an_lru_cache_with_the_jumpstart_defaults():
    cache = HubCatalogCache()

    assert isinstance(cache._content_cache, LRUCache)
    assert cache._content_cache._max_cache_items == HUB_CATALOG_DEFAULT_MAX_CACHE_ITEMS == 20
    assert cache._content_cache._expiration_horizon == HUB_CATALOG_DEFAULT_CACHE_EXPIRATION_HORIZON
    assert HUB_CATALOG_DEFAULT_CACHE_EXPIRATION_HORIZON == datetime.timedelta(hours=6)


def test_get_all_cached_and_refresh_cache_emit_telemetry(telemetry_send):
    hub = _Hub({"model-a": "1.0.0"})

    with _patched(hub):
        HubContent.get_all_cached(HUB, region="us-west-2")
        HubContent.refresh_cache(HUB, region="us-west-2")

    func_names = [call.args[5].split("&")[0] for call in telemetry_send.call_args_list]
    assert func_names == ["HubContent.get_all_cached", "HubContent.refresh_cache"]
    assert all(call.args[0] == 1 for call in telemetry_send.call_args_list)


def test_account_id_calls_sts_once_per_set_of_credentials(monkeypatch):
    import sagemaker.core.hub_content as hub_content_module

    monkeypatch.setattr(hub_content_module, "_ACCOUNT_IDS", {})
    account_id = hub_content_module.HubContentMixin.__dict__["_account_id"].__func__
    sts = MagicMock()
    sts.get_caller_identity.return_value = {"Account": ACCOUNT}
    session = MagicMock()
    session.region_name = "us-west-2"
    session.client.return_value = sts
    session.get_credentials.return_value.get_frozen_credentials.return_value = (
        "AKID",
        "secret",
        None,
    )

    first = account_id(HubContent, session, "us-west-2")
    second = account_id(HubContent, session, "us-west-2")

    assert (first, second) == (ACCOUNT, ACCOUNT)
    assert sts.get_caller_identity.call_count == 1
    assert session.client.call_args.kwargs["region_name"] == "us-west-2"


def test_default_cache_dir_honors_the_environment_variable(monkeypatch, tmp_path):
    monkeypatch.setenv(hub_catalog.CACHE_DIR_ENV_VAR, str(tmp_path / "override"))
    assert default_cache_dir() == str(tmp_path / "override")

    monkeypatch.delenv(hub_catalog.CACHE_DIR_ENV_VAR)
    assert default_cache_dir().endswith("hub-content")


def test_cache_key_file_name_is_file_system_safe():
    key = HubCatalogKey(
        hub_name="arn:aws:hub/x y", hub_content_type="Model", region="us-west-2", account_id=ACCOUNT
    )

    assert key.file_name() == f"{ACCOUNT}-us-west-2-arn_aws_hub_x_y-Model.json"


def test_reusable_entries_match_on_the_listed_version():
    entries = {
        "same": {"listed_version": "1.0.0", "resource": {}},
        "changed": {"listed_version": "1.0.0", "resource": {}},
        "gone": {"listed_version": "1.0.0", "resource": {}},
    }
    summaries = [_summary("same", "1.0.0"), _summary("changed", "1.1.0"), _summary("new", "1.0.0")]

    assert list(reusable_entries(entries, summaries)) == ["same"]


def test_cache_file_rejects_another_format_version(tmp_path):
    key = HubCatalogKey(HUB, "Model", "us-west-2", ACCOUNT)
    cache_file = HubCatalogFile(key, str(tmp_path))
    cache_file.save({"model-a": {"listed_version": "1.0.0", "resource": {"hub_content_name": "a"}}})
    payload = json.loads(open(cache_file.path).read())
    payload["format_version"] = CACHE_FORMAT_VERSION + 1
    open(cache_file.path, "w").write(json.dumps(payload))

    assert cache_file.load() == {}


def test_a_slow_retrieval_of_one_hub_does_not_block_a_hit_on_another_hub():
    cache = HubCatalogCache()
    fast_key = HubCatalogKey("fast-hub", "Model", "us-west-2", ACCOUNT)
    slow_key = HubCatalogKey("slow-hub", "Model", "us-west-2", ACCOUNT)
    started = threading.Event()

    def slow_list(key):
        started.set()
        time.sleep(0.5)
        return []

    fast_source = HubCatalogSource(lambda key: [], lambda key, summary: None)
    slow_source = HubCatalogSource(slow_list, lambda key, summary: None)
    cache.get(fast_key, fast_source)

    with ThreadPoolExecutor(max_workers=1) as executor:
        slow = executor.submit(cache.get, slow_key, slow_source)
        assert started.wait(timeout=5)
        hit_started = time.monotonic()
        cache.get(fast_key, fast_source)
        hit_elapsed = time.monotonic() - hit_started
        slow.result(timeout=5)

    assert hit_elapsed < 0.4


def test_two_threads_that_miss_the_same_hub_share_one_retrieval():
    cache = HubCatalogCache()
    key = HubCatalogKey("hub", "Model", "us-west-2", ACCOUNT)
    lists = []

    def list_hub(key):
        lists.append(threading.get_ident())
        return [_summary("model-a")]

    def describe(key, summary):
        resource = MagicMock()
        resource.model_dump.return_value = {"hub_content_name": "model-a"}
        return resource

    source = HubCatalogSource(list_hub, describe)
    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(lambda _: cache.get(key, source), range(4)))

    assert len(lists) == 1
    assert all(value.refresh.listed == 1 for value in values)
