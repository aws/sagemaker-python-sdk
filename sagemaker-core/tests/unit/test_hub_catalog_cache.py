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
"""``HubContent.get_all_cached`` describes only the hub contents the local cache lacks.

Before this change every list of a hub described every item: 766 ``DescribeHubContent``
calls and 60 to 95 seconds for the public hub, on every call. The cache keeps the described
resources on disk per hub, content type, region, and account, keyed by the version the
list summary reports, so an unchanged hub costs the list calls only.
"""

from __future__ import absolute_import

import datetime
import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.core import hub_catalog
from sagemaker.core.hub_catalog import (
    CACHE_FORMAT_VERSION,
    HubCatalogCache,
    HubCatalogKey,
    default_cache_dir,
    reusable_entries,
)
from sagemaker.core.resources import Base, HubContent
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
        self.in_flight = 0
        self.peak_in_flight = 0
        self._lock = threading.Lock()

    def _list(self, **kwargs):
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
    with (
        patch.object(HubContent, "_account_id", return_value=ACCOUNT),
        patch(
            "sagemaker.core.telemetry.telemetry_logging._get_default_sagemaker_session",
            return_value=MagicMock(sagemaker_config=None),
        ),
        patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request"),
    ):
        yield tmp_path / "hub-cache"
    SageMakerClient.reset()


@pytest.fixture
def telemetry_send():
    with patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request") as send:
        yield send


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


def _patched(hub):
    return patch.object(Base, "get_sagemaker_client", return_value=hub.client)


def test_first_call_describes_every_item_and_writes_the_cache(_isolated_cache):
    hub = _Hub({"model-a": "1.0.0", "model-b": "2.0.0", "model-c": "1.0.0"})

    with _patched(hub):
        models = HubContent.get_all_cached(HUB, max_workers=2, region="us-west-2")

    assert [m.hub_content_name for m in models] == ["model-a", "model-b", "model-c"]
    assert sorted(hub.described) == ["model-a", "model-b", "model-c"]
    assert hub.peak_in_flight >= 1
    for call in hub.client.describe_hub_content.call_args_list:
        assert call.kwargs["HubName"] == HUB
    assert models[0].original_creation_time == ORIGINAL
    assert models[0].hub_content_document == json.dumps(
        {"License": "Apache-2.0", "Version": "1.0.0"}
    )
    assert models[0].license == "apache-2.0"
    assert models[0]._region == "us-west-2"
    cache_path = _isolated_cache / f"{ACCOUNT}-us-west-2-{HUB}-Model.json"
    payload = json.loads(cache_path.read_text())
    assert payload["format_version"] == CACHE_FORMAT_VERSION
    assert sorted(payload["entries"]) == ["model-a", "model-b", "model-c"]
    assert payload["entries"]["model-b"]["listed_version"] == "2.0.0"


def test_second_call_describes_only_new_and_changed_items():
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


def test_a_stale_cache_describes_every_item_again():
    hub = _Hub({"model-a": "1.0.0"})
    with _patched(hub):
        HubContent.get_all_cached(HUB, region="us-west-2")
    hub.described.clear()

    with _patched(hub):
        HubContent.get_all_cached(HUB, max_age=datetime.timedelta(0), region="us-west-2")

    assert hub.described == ["model-a"]


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


def test_clear_cache_deletes_the_file_once():
    hub = _Hub({"model-a": "1.0.0"})
    with _patched(hub):
        HubContent.get_all_cached(HUB, region="us-west-2")

        assert HubContent.clear_cache(HUB, region="us-west-2") is True
        assert HubContent.clear_cache(HUB, region="us-west-2") is False


def test_an_unreadable_cache_file_is_ignored(_isolated_cache, caplog):
    _isolated_cache.mkdir(parents=True)
    (_isolated_cache / f"{ACCOUNT}-us-west-2-{HUB}-Model.json").write_text("{not json")
    hub = _Hub({"model-a": "1.0.0"})

    with _patched(hub), caplog.at_level("WARNING"):
        models = HubContent.get_all_cached(HUB, region="us-west-2")

    assert [m.hub_content_name for m in models] == ["model-a"]
    assert "Ignoring the unreadable hub content cache" in caplog.text


def test_a_failed_describe_raises_and_keeps_the_described_items_in_the_cache(_isolated_cache):
    hub = _Hub({"model-a": "1.0.0", "model-b": "1.0.0", "model-c": "1.0.0"})
    described_before_failure = []

    def describe(**kwargs):
        name = kwargs["HubContentName"]
        if name == "model-c":
            raise RuntimeError("describe of model-c failed")
        described_before_failure.append(name)
        return hub._describe(**kwargs)

    hub.client.describe_hub_content.side_effect = describe
    with _patched(hub):
        with pytest.raises(RuntimeError, match="describe of model-c failed"):
            HubContent.get_all_cached(HUB, max_workers=1, region="us-west-2")

    cache_path = _isolated_cache / f"{ACCOUNT}-us-west-2-{HUB}-Model.json"
    payload = json.loads(cache_path.read_text())
    assert sorted(payload["entries"]) == ["model-a", "model-b"]
    hub.client.describe_hub_content.side_effect = hub._describe
    hub.described.clear()
    with _patched(hub):
        refresh = HubContent.refresh_cache(HUB, region="us-west-2")

    assert hub.described == ["model-c"]
    assert (refresh.reused, refresh.described) == (2, 1)


def test_max_workers_below_one_is_rejected():
    with _patched(_Hub({})):
        with pytest.raises(ValueError, match="max_workers must be at least 1"):
            HubContent.get_all_cached(HUB, max_workers=0, region="us-west-2")


def test_default_cache_dir_honors_the_environment_variable(monkeypatch, tmp_path):
    monkeypatch.setenv(hub_catalog.CACHE_DIR_ENV_VAR, str(tmp_path / "override"))
    assert default_cache_dir() == str(tmp_path / "override")

    monkeypatch.delenv(hub_catalog.CACHE_DIR_ENV_VAR)
    assert (
        default_cache_dir().endswith("sagemaker/hub-content") or "sagemaker" in default_cache_dir()
    )


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


def test_cache_load_rejects_another_format_version(tmp_path):
    key = HubCatalogKey(HUB, "Model", "us-west-2", ACCOUNT)
    cache = HubCatalogCache(key, str(tmp_path))
    cache.save(
        {"model-a": {"listed_version": "1.0.0", "resource": {"hub_content_name": "model-a"}}}
    )
    payload = json.loads(open(cache.path).read())
    payload["format_version"] = CACHE_FORMAT_VERSION + 1
    open(cache.path, "w").write(json.dumps(payload))

    assert cache.load(datetime.timedelta(days=1)) == {}
