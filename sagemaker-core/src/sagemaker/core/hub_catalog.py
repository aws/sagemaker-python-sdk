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
"""A cache of the described contents of a hub, in the shape of the JumpStart models cache.

A list of a hub costs one ``ListHubContents`` call per page plus one ``DescribeHubContent``
call per item, about 770 describes for the public hub. The list is cheap (8 pages, about 1
second) and the describes are the cost (60 to 95 seconds). ``HubContent.get_all`` serves
the list from this cache.

``HubCatalogCache`` follows ``sagemaker.core.jumpstart.cache.JumpStartModelsCache``. Its
in-memory layer is the same ``sagemaker.core.utilities.cache.LRUCache``, with one entry per
hub and content type, a bounded item count, and an expiration horizon. The ``LRUCache``
calls the retrieval function on a miss, or on an expired entry with the old value. The
retrieval function lists the hub and keeps every described item whose ``HubContentVersion``
still matches the list summary. Only the new or changed items are described. The JumpStart
retrieval function reads the manifest again only when its ETag changed, in the same way.

On top of the JumpStart shape, the described items persist to one JSON file per hub,
content type, region, and account. A new process starts from the previous catalog instead
of describing every item again. The file lives under the ``platformdirs`` user cache
directory (``~/.cache/sagemaker/hub-content`` on Linux) or ``SAGEMAKER_HUB_CACHE_DIR``. It
holds the service response for catalog metadata and no credentials. A file is written to
a temporary name and renamed into place, so a reader never sees a partial file.

``HubCatalogAccessor`` holds the process-wide cache instance, as
``JumpStartModelsAccessor`` holds the JumpStart one.
"""

from __future__ import absolute_import

import datetime
import json
import logging
import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from botocore.config import Config
from packaging.version import InvalidVersion, Version
from platformdirs import user_cache_dir

from sagemaker.core.utilities.cache import LRUCache

logger = logging.getLogger(__name__)

CACHE_DIR_ENV_VAR = "SAGEMAKER_HUB_CACHE_DIR"
CACHE_APP_NAME = "sagemaker"
CACHE_SUBDIRECTORY = "hub-content"
CACHE_FORMAT_VERSION = 2
HUB_CATALOG_DEFAULT_MAX_CACHE_ITEMS = 20
HUB_CATALOG_DEFAULT_CACHE_EXPIRATION_HORIZON = datetime.timedelta(hours=6)
HUB_CATALOG_CLIENT_MAX_ATTEMPTS = 10
# The ThreadPoolExecutor default is at most 32 workers (``min(32, cpu_count + 4)``), so a
# pool of 32 connections holds one connection per worker of the default pool.
HUB_CATALOG_DEFAULT_MAX_POOL_CONNECTIONS = 32
LIST_PAGE_SIZE = 100
SORT_KEYS = {
    "HubContentName": lambda summary: summary.get("HubContentName") or "",
    "CreationTime": lambda summary: _sortable_time(summary.get("CreationTime")),
    "HubContentStatus": lambda summary: summary.get("HubContentStatus") or "",
}
_UNSAFE_FILE_NAME_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]+")

Summary = Dict[str, Any]
Entry = Dict[str, Any]


def default_cache_dir() -> str:
    """Return the directory of the hub content cache files.

    ``SAGEMAKER_HUB_CACHE_DIR`` wins when set. The default is the ``hub-content``
    directory under the ``platformdirs`` user cache directory of the ``sagemaker``
    application, ``~/.cache/sagemaker/hub-content`` on Linux.
    """
    override = os.environ.get(CACHE_DIR_ENV_VAR)
    if override:
        return os.path.expanduser(override)
    return os.path.join(user_cache_dir(CACHE_APP_NAME), CACHE_SUBDIRECTORY)


def catalog_client_config(max_workers: Optional[int]) -> Config:
    """Return the botocore config of the clients that list and describe a hub.

    The retry mode is ``adaptive``: botocore rate-limits the client after a
    ``ThrottlingException``, so a pool of describes backs off instead of failing once the
    standard retries run out. On the public hub (766 models) the default pool of 32 workers
    finished with 0 failures in adaptive mode, and failed 462 describes in standard mode.
    The connection pool holds one connection per worker, so urllib3 discards none. The
    explicit config outranks ``AWS_RETRY_MODE`` and ``AWS_MAX_ATTEMPTS`` for these calls.
    """
    return Config(
        retries={"max_attempts": HUB_CATALOG_CLIENT_MAX_ATTEMPTS, "mode": "adaptive"},
        max_pool_connections=max(HUB_CATALOG_DEFAULT_MAX_POOL_CONNECTIONS, max_workers or 0),
    )


@dataclass(frozen=True)
class HubCatalogKey:
    """The key of one cache entry: one hub and content type in one region and account."""

    hub_name: str
    hub_content_type: str
    region: str
    account_id: str

    def file_name(self) -> str:
        """Return the cache file name, with every part reduced to file-system-safe text."""
        parts = (self.account_id, self.region, self.hub_name, self.hub_content_type)
        safe_parts = [_UNSAFE_FILE_NAME_CHARACTERS.sub("_", part) for part in parts]
        return "-".join(safe_parts) + ".json"


DescribeFunction = Callable[[HubCatalogKey, Summary], Any]
ListFunction = Callable[[HubCatalogKey], List[Summary]]


@dataclass(frozen=True)
class HubCatalogSource:
    """The service calls of one retrieval: list a hub, and describe one listed summary.

    The calls carry the session of the caller. They are passed per call and not stored on
    the cache, so one process cache serves every session.
    """

    list_function: ListFunction
    describe_function: DescribeFunction


@dataclass(frozen=True)
class HubCatalogRefresh:
    """What one retrieval did: how many items it listed, reused, described, and dropped."""

    listed: int
    reused: int
    described: int
    removed: int
    cache_path: str


@dataclass(frozen=True)
class HubCatalogValue:
    """The value of one cache entry: the described items and the refresh that produced them.

    ``entries`` maps a content name to ``{"listed_version": ..., "resource": ...}``, where
    ``resource`` is the ``model_dump`` of the described ``HubContent``. ``summaries`` is the
    ``ListHubContents`` output of the retrieval, in service order.
    """

    entries: Dict[str, Entry]
    summaries: List[Summary]
    refresh: HubCatalogRefresh


@dataclass(frozen=True)
class HubCatalogFilters:
    """The ``ListHubContents`` filters, applied to the cached catalog instead of the service.

    ``name_contains`` is a case-sensitive substring of ``HubContentName``.
    ``max_schema_version`` keeps the items whose ``DocumentSchemaVersion`` is at most the
    given version. The two time bounds compare against the ``CreationTime`` of the list
    summary. ``sort_by`` is ``HubContentName``, ``CreationTime``, or ``HubContentStatus``,
    and ``sort_order`` is ``Ascending`` or ``Descending``. Without ``sort_by`` the service
    order of the list is kept.
    """

    name_contains: Optional[str] = None
    max_schema_version: Optional[str] = None
    creation_time_before: Optional[datetime.datetime] = None
    creation_time_after: Optional[datetime.datetime] = None
    sort_by: Optional[str] = None
    sort_order: Optional[str] = None

    def apply(self, pairs: List[Tuple[Summary, Entry]]) -> List[Tuple[Summary, Entry]]:
        """Return the ``(summary, entry)`` pairs that pass the filters, in the requested order."""
        kept = [pair for pair in pairs if self._matches(pair[0])]
        if self.sort_by is None:
            return kept
        if self.sort_by not in SORT_KEYS:
            raise ValueError(
                f"sort_by must be one of {', '.join(sorted(SORT_KEYS))}, got {self.sort_by!r}."
            )
        if self.sort_order not in (None, "Ascending", "Descending"):
            raise ValueError(
                f"sort_order must be Ascending or Descending, got {self.sort_order!r}."
            )
        sort_key = SORT_KEYS[self.sort_by]
        return sorted(
            kept, key=lambda pair: sort_key(pair[0]), reverse=self.sort_order == "Descending"
        )

    def _matches(self, summary: Summary) -> bool:
        if self.name_contains and self.name_contains not in (summary.get("HubContentName") or ""):
            return False
        if self.max_schema_version and not _version_at_most(
            summary.get("DocumentSchemaVersion"), self.max_schema_version
        ):
            return False
        created = _sortable_time(summary.get("CreationTime"))
        if self.creation_time_before and not created < _sortable_time(self.creation_time_before):
            return False
        if self.creation_time_after and not created > _sortable_time(self.creation_time_after):
            return False
        return True


class HubCatalogFile:
    """Read and write the on-disk copy of one :class:`HubCatalogKey`."""

    def __init__(self, key: HubCatalogKey, cache_dir: Optional[str] = None):
        self.key = key
        self.cache_dir = cache_dir or default_cache_dir()
        self.path = os.path.join(self.cache_dir, key.file_name())

    def load(self) -> Dict[str, Entry]:
        """Return the entries by name, or an empty dict for a missing or unreadable file."""
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
        entries = payload.get("entries")
        return entries if isinstance(entries, dict) else {}

    def save(self, entries: Dict[str, Entry]) -> None:
        """Write the entries, through a temporary file and a rename."""
        os.makedirs(self.cache_dir, exist_ok=True)
        payload = {
            "format_version": CACHE_FORMAT_VERSION,
            "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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
        """Delete the file. Return ``True`` when a file was deleted."""
        try:
            os.remove(self.path)
        except FileNotFoundError:
            return False
        return True


class HubCatalogCache:
    """Cache of the described contents of hubs, one ``LRUCache`` entry per hub and type.

    Built like ``JumpStartModelsCache``: an ``LRUCache`` with a bounded item count and an
    expiration horizon. The ``LRUCache`` calls the retrieval function on a miss, or on an
    expired entry with the old value. The retrieval function lists the hub and reuses the
    old items whose ``HubContentVersion`` the list still reports. The rest are described on
    a ``ThreadPoolExecutor`` of the host default size, with the adaptive-retry client of
    :attr:`client_config`, and the result persists to the cache file of the key.

    Args:
        max_cache_items: Maximum number of hubs to hold in memory. Default: 20.
        cache_expiration_horizon: Maximum age of an in-memory entry before the hub is
            listed again. Default: 6 hours.
        max_workers: How many describe calls to keep in flight. Default: the
            ``ThreadPoolExecutor`` default of the host (``min(32, cpu_count + 4)``).
        cache_dir: The directory of the cache files. Default: :func:`default_cache_dir`.

    Attributes:
        client_config: The botocore config of the list and describe clients, from
            :func:`catalog_client_config`.
    """

    def __init__(
        self,
        max_cache_items: int = HUB_CATALOG_DEFAULT_MAX_CACHE_ITEMS,
        cache_expiration_horizon: datetime.timedelta = HUB_CATALOG_DEFAULT_CACHE_EXPIRATION_HORIZON,
        max_workers: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        if max_workers is not None and max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {max_workers}.")
        self._max_workers = max_workers
        self._cache_dir = cache_dir
        self.client_config = catalog_client_config(max_workers)
        # The LRUCache is not thread safe, so its reads and writes run under this lock. A
        # retrieval runs outside it, with one lock per key so two threads that miss the
        # same hub share one list-and-describe pass while other hubs stay unblocked.
        self._cache_lock = threading.Lock()
        self._key_locks: Dict[HubCatalogKey, threading.Lock] = {}
        self._content_cache = LRUCache[HubCatalogKey, HubCatalogValue](
            max_cache_items=max_cache_items,
            expiration_horizon=cache_expiration_horizon,
            retrieval_function=self._retrieval_function,
        )

    def get(self, key: HubCatalogKey, source: HubCatalogSource) -> HubCatalogValue:
        """Return the catalog of a hub from memory, or retrieve it when absent or expired."""
        with self._cache_lock:
            # LRUCache.get pops an expired element before it raises, so read the old value
            # first. It lets the retrieval describe only the changed items without a file read.
            previous = self._previous_value(key)
            try:
                value, _ = self._content_cache.get(key, data_source_fallback=False)
                return value
            except KeyError:
                pass
        with self._key_lock(key):
            with self._cache_lock:
                try:
                    value, _ = self._content_cache.get(key, data_source_fallback=False)
                    return value
                except KeyError:
                    pass
            return self._retrieve_and_store(key, previous, source)

    def refresh(self, key: HubCatalogKey, source: HubCatalogSource) -> HubCatalogValue:
        """List the hub again, describe the changed items, and return the new catalog."""
        with self._cache_lock:
            previous = self._previous_value(key)
        with self._key_lock(key):
            return self._retrieve_and_store(key, previous, source)

    def clear(self, key: Optional[HubCatalogKey] = None) -> bool:
        """Drop the in-memory entries, and the cache file of ``key`` when one is given.

        Returns ``True`` when a file was deleted.
        """
        with self._cache_lock:
            self._content_cache.clear()
        if key is None:
            return False
        return HubCatalogFile(key, self._cache_dir).clear()

    def _previous_value(self, key: HubCatalogKey) -> Optional[HubCatalogValue]:
        """Return the stored value of ``key``, expired or not, without a retrieval."""
        element = self._content_cache._lru_cache.get(key)  # pylint: disable=protected-access
        return element.value if element is not None else None

    def _key_lock(self, key: HubCatalogKey) -> threading.Lock:
        with self._cache_lock:
            return self._key_locks.setdefault(key, threading.Lock())

    def _retrieve_and_store(
        self, key: HubCatalogKey, previous: Optional[HubCatalogValue], source: HubCatalogSource
    ) -> HubCatalogValue:
        """Run the retrieval outside the cache lock, then store the result under it."""
        value = self._retrieval_function(key, previous, source)
        with self._cache_lock:
            self._content_cache.put(key, value)
        return value

    def _retrieval_function(
        self,
        key: HubCatalogKey,
        value: Optional[HubCatalogValue],
        source: Optional[HubCatalogSource] = None,
    ) -> HubCatalogValue:
        """Bring the catalog of ``key`` up to date, from the old value and the cache file.

        The ``LRUCache`` holds this function as its ``retrieval_function`` and calls it with
        ``key`` and ``value`` only. :meth:`get` and :meth:`refresh` call it themselves with
        the ``source`` of the caller, so the ``LRUCache`` path raises for a missing source.
        """
        if source is None:
            raise RuntimeError("HubCatalogCache retrieves only through get() or refresh().")
        cache_file = HubCatalogFile(key, self._cache_dir)
        previous = value.entries if value is not None else cache_file.load()
        summaries = source.list_function(key)
        reused = reusable_entries(previous, summaries)
        missing = [s for s in summaries if s.get("HubContentName") not in reused]
        entries = dict(reused)
        failure = self._describe_all(source, key, missing, entries)
        # The file holds every item that was described, so after a failure the next
        # retrieval describes only the rest.
        cache_file.save(entries)
        if failure is not None:
            raise failure
        refresh = HubCatalogRefresh(
            listed=len(summaries),
            reused=len(reused),
            described=len(missing),
            removed=len(set(previous) - set(entries)),
            cache_path=cache_file.path,
        )
        return HubCatalogValue(entries=entries, summaries=summaries, refresh=refresh)

    def _describe_all(
        self,
        source: HubCatalogSource,
        key: HubCatalogKey,
        summaries: List[Summary],
        entries: Dict[str, Entry],
    ) -> Optional[BaseException]:
        """Describe every summary on a thread pool and put each result into ``entries``.

        Every describe runs to its end, as completed, so one failure drops no result that
        another worker already has. The first failure comes back instead of raising, so the
        caller persists the entries before it raises.
        """
        failure: Optional[BaseException] = None
        with ThreadPoolExecutor(
            max_workers=self._max_workers, thread_name_prefix="sagemaker-hub-describe"
        ) as executor:
            futures = {
                executor.submit(source.describe_function, key, summary): summary
                for summary in summaries
            }
            for future in as_completed(futures):
                summary = futures[future]
                try:
                    resource = future.result()
                except Exception as error:  # pylint: disable=broad-except
                    failure = failure or error
                    continue
                entries[summary["HubContentName"]] = resource_entry(
                    resource, summary.get("HubContentVersion")
                )
        return failure


class HubCatalogAccessor:
    """Static holder of the process-wide :class:`HubCatalogCache`.

    Mirrors ``JumpStartModelsAccessor``: one cache per process, built from the stored
    cache kwargs, rebuilt when :meth:`set_cache_kwargs` changes them, and dropped through
    :meth:`reset_cache`.
    """

    _cache: Optional[HubCatalogCache] = None
    _cache_kwargs: Dict[str, Any] = {}
    _lock = threading.Lock()

    @staticmethod
    def get_cache() -> HubCatalogCache:
        """Return the process cache, built on the first call with the stored kwargs."""
        with HubCatalogAccessor._lock:
            if HubCatalogAccessor._cache is None:
                HubCatalogAccessor._cache = HubCatalogCache(**HubCatalogAccessor._cache_kwargs)
            return HubCatalogAccessor._cache

    @staticmethod
    def set_cache_kwargs(**cache_kwargs: Any) -> None:
        """Store new ``HubCatalogCache`` kwargs and drop the cache, so the next call rebuilds it."""
        with HubCatalogAccessor._lock:
            HubCatalogAccessor._cache_kwargs = dict(cache_kwargs)
            HubCatalogAccessor._cache = None

    @staticmethod
    def reset_cache() -> None:
        """Drop the process cache and its kwargs, so the next call builds a default one."""
        with HubCatalogAccessor._lock:
            HubCatalogAccessor._cache = None
            HubCatalogAccessor._cache_kwargs = {}


def resource_entry(resource: Any, listed_version: str) -> Entry:
    """Return the cache entry of one described resource."""
    return {
        "listed_version": listed_version,
        "resource": resource.model_dump(exclude_unset=True, mode="json"),
    }


def reusable_entries(entries: Dict[str, Entry], summaries: List[Summary]) -> Dict[str, Entry]:
    """Return the entries whose listed version matches the current list summary."""
    reusable = {}
    for summary in summaries:
        name = summary.get("HubContentName")
        entry = entries.get(name)
        if entry and entry.get("listed_version") == summary.get("HubContentVersion"):
            reusable[name] = entry
    return reusable


def ordered_entries(value: HubCatalogValue) -> List[Tuple[Summary, Entry]]:
    """Return ``(summary, entry)`` pairs in the order the service listed the hub."""
    pairs = []
    for summary in value.summaries:
        entry = value.entries.get(summary.get("HubContentName"))
        if entry is not None:
            pairs.append((summary, entry))
    return pairs


def _sortable_time(value: Any) -> datetime.datetime:
    """Return ``value`` as an aware datetime, so cached ISO strings and datetimes compare."""
    if isinstance(value, str):
        value = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime.datetime):
        return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=datetime.timezone.utc)
    return value


def _version_at_most(version: Any, ceiling: str) -> bool:
    """Return ``True`` when ``version`` parses and is at most ``ceiling``."""
    try:
        return Version(str(version)) <= Version(ceiling)
    except InvalidVersion:
        return False


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
