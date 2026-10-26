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
"""A local, on-disk cache of the described contents of a hub.

``HubContent.get_all`` costs one ``ListHubContents`` call per page plus one
``DescribeHubContent`` call per item, about 770 describes for the public hub. The list
is cheap (8 pages, about 1 second) and the describes are the cost (60 to 95 seconds).
``HubContent.get_all_cached`` keeps the described resources on disk, one JSON file per
hub, content type, region, and account. The next call lists the hub again and reuses
every cached item whose ``HubContentVersion`` matches the list summary. Only the items
that are new or have a new version are described. A model whose version did not change
has the same document, so the cache serves the same resource the service would return.

The cache lives under the ``platformdirs`` user cache directory (``~/.cache/sagemaker``
on Linux) or ``SAGEMAKER_HUB_CACHE_DIR``. It holds the service response for catalog
metadata and no credentials. A file is written to a temporary name and renamed into
place, so a reader never sees a partial file.
"""

from __future__ import absolute_import

import datetime
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

from boto3.session import Session
from platformdirs import user_cache_dir

logger = logging.getLogger(__name__)

CACHE_DIR_ENV_VAR = "SAGEMAKER_HUB_CACHE_DIR"
CACHE_APP_NAME = "sagemaker"
CACHE_SUBDIRECTORY = "hub-content"
CACHE_FORMAT_VERSION = 1
DEFAULT_MAX_AGE = datetime.timedelta(days=1)
_UNSAFE_FILE_NAME_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]+")


def default_cache_dir() -> str:
    """Return the directory of the hub content cache.

    ``SAGEMAKER_HUB_CACHE_DIR`` wins when set. The default is the ``hub-content``
    directory under the ``platformdirs`` user cache directory of the ``sagemaker``
    application, ``~/.cache/sagemaker/hub-content`` on Linux.
    """
    override = os.environ.get(CACHE_DIR_ENV_VAR)
    if override:
        return os.path.expanduser(override)
    return os.path.join(user_cache_dir(CACHE_APP_NAME), CACHE_SUBDIRECTORY)


@dataclass(frozen=True)
class HubCatalogKey:
    """What one cache file covers: one hub, one content type, in one region and account."""

    hub_name: str
    hub_content_type: str
    region: str
    account_id: str

    def file_name(self) -> str:
        """Return the cache file name, with every part reduced to file-system-safe text."""
        parts = (self.account_id, self.region, self.hub_name, self.hub_content_type)
        safe_parts = [_UNSAFE_FILE_NAME_CHARACTERS.sub("_", part) for part in parts]
        return "-".join(safe_parts) + ".json"


@dataclass(frozen=True)
class HubCatalogRequest:
    """What one ``HubContent.get_all_cached`` call asks for."""

    hub_name: str
    hub_content_type: str
    max_age: datetime.timedelta
    max_workers: int
    cache_dir: Optional[str]


@dataclass(frozen=True)
class HubCatalogRefresh:
    """What one ``HubContent.get_all_cached`` call did."""

    listed: int
    reused: int
    described: int
    removed: int
    cache_path: str


class HubCatalogCache:
    """Read and write the cache file of one :class:`HubCatalogKey`.

    The file holds the ``model_dump`` of every described ``HubContent`` keyed by name.
    Each entry carries the version the list summary reported, and the file carries the
    time of the write.
    """

    def __init__(self, key: HubCatalogKey, cache_dir: Optional[str] = None):
        self.key = key
        self.cache_dir = cache_dir or default_cache_dir()
        self.path = os.path.join(self.cache_dir, key.file_name())

    def load(self, max_age: datetime.timedelta) -> Dict[str, Dict[str, Any]]:
        """Return the cached entries by name, or an empty dict for a missing, stale, or unreadable file.

        Args:
            max_age: The longest time since the last write for which the file counts.
        """
        try:
            with open(self.path, "r", encoding="utf-8") as cache_file:
                payload = json.load(cache_file)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as error:
            logger.warning("Ignoring the unreadable hub content cache %s: %s", self.path, error)
            return {}
        if payload.get("format_version") != CACHE_FORMAT_VERSION:
            return {}
        written_at = _parse_time(payload.get("written_at"))
        if written_at is None or _now() - written_at > max_age:
            return {}
        entries = payload.get("entries")
        return entries if isinstance(entries, dict) else {}

    def save(self, entries: Dict[str, Dict[str, Any]]) -> None:
        """Write the entries to the cache file, through a temporary file and a rename."""
        os.makedirs(self.cache_dir, exist_ok=True)
        payload = {
            "format_version": CACHE_FORMAT_VERSION,
            "written_at": _now().isoformat(),
            "key": {
                "hub_name": self.key.hub_name,
                "hub_content_type": self.key.hub_content_type,
                "region": self.key.region,
                "account_id": self.key.account_id,
            },
            "entries": entries,
        }
        descriptor, temporary_path = tempfile.mkstemp(
            dir=self.cache_dir, prefix=self.key.file_name(), suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as cache_file:
                json.dump(payload, cache_file)
            os.replace(temporary_path, self.path)
        except BaseException:
            _remove_quietly(temporary_path)
            raise

    def clear(self) -> bool:
        """Delete the cache file. Return ``True`` when a file was deleted."""
        try:
            os.remove(self.path)
        except FileNotFoundError:
            return False
        return True


def cache_key_for(
    hub_name: str,
    hub_content_type: str,
    session: Optional[Session],
    region: Optional[str],
    account_id: str,
) -> HubCatalogKey:
    """Build the cache key of a hub from the call arguments and the resolved account."""
    resolved_region = region or (session.region_name if session is not None else None)
    if not resolved_region:
        resolved_region = Session().region_name or "default"
    return HubCatalogKey(
        hub_name=hub_name,
        hub_content_type=hub_content_type,
        region=str(resolved_region),
        account_id=account_id,
    )


def resource_entry(resource: Any, listed_version: str) -> Dict[str, Any]:
    """Return the cache entry of one described resource."""
    return {
        "listed_version": listed_version,
        "resource": resource.model_dump(exclude_unset=True, mode="json"),
    }


def reusable_entries(
    entries: Dict[str, Dict[str, Any]], summaries: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Return the cached entries whose listed version matches the current list summary."""
    reusable = {}
    for summary in summaries:
        name = summary.get("HubContentName")
        entry = entries.get(name)
        if entry and entry.get("listed_version") == summary.get("HubContentVersion"):
            reusable[name] = entry
    return reusable


def iterate_in_list_order(
    summaries: List[Dict[str, Any]], resources_by_name: Dict[str, Any]
) -> Iterator[Any]:
    """Yield the resources in the order the service listed them."""
    for summary in summaries:
        resource = resources_by_name.get(summary.get("HubContentName"))
        if resource is not None:
            yield resource


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _parse_time(value: Any) -> Optional[datetime.datetime]:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
