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
"""``LRUCache.put`` on a present key hands the old value to the retrieval function.

Before this change the call raised ``AttributeError: 'Element' object has no attribute
'element'``, because ``put`` read ``curr_value.element`` from an ``Element`` whose attribute
is ``value``. ``JumpStartModelsCache`` never reached the line, because it refreshes
through ``get`` and ``_get_item``. ``HubCatalogCache.refresh`` reaches it.
"""

from __future__ import absolute_import

import datetime

import pytest

from sagemaker.core.utilities.cache import LRUCache


def _cache(calls, horizon=datetime.timedelta(hours=6)):
    def retrieve(key, value):
        calls.append((key, value))
        return f"value-{len(calls)}"

    return LRUCache(max_cache_items=2, expiration_horizon=horizon, retrieval_function=retrieve)


def test_put_on_a_present_key_passes_the_old_value_to_the_retrieval_function():
    calls = []
    cache = _cache(calls)
    assert cache.get("key") == ("value-1", False)

    cache.put("key")

    assert cache.get("key") == ("value-2", True)
    assert calls == [("key", None), ("key", "value-1")]


def test_put_with_a_value_stores_it_without_a_retrieval():
    calls = []
    cache = _cache(calls)

    cache.put("key", "given")

    assert cache.get("key") == ("given", True)
    assert calls == []


def test_put_evicts_the_least_recently_used_key_at_the_cap():
    calls = []
    cache = _cache(calls)
    cache.put("first")
    cache.put("second")

    cache.put("third")

    assert "first" not in cache
    assert "second" in cache and "third" in cache
    assert len(cache) == 2


def test_get_without_fallback_raises_for_an_absent_key():
    cache = _cache([])

    with pytest.raises(KeyError, match="not found in LRUCache"):
        cache.get("absent", data_source_fallback=False)
