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
from __future__ import absolute_import
from typing import Optional, Union
from mock.mock import MagicMock, patch
import pytest


from sagemaker.utilities import cache
import datetime


def retrieval_function(key: Optional[int] = None, value: Optional[str] = None) -> str:
    return str(hash(str(key)))


def test_cache_retrieves_item():
    my_cache = cache.LRUCache[int, Union[int, str]](
        max_cache_items=10,
        expiration_horizon=datetime.timedelta(hours=1),
        retrieval_function=retrieval_function,
    )

    my_cache.put(5)
    assert my_cache.get(5, False) == (retrieval_function(key=5), True)

    my_cache.put(6, 7)
    assert my_cache.get(6, False) == (7, True)
    assert len(my_cache) == 2

    my_cache.put(5, 6)
    assert my_cache.get(5, False) == (6, True)
    assert len(my_cache) == 2

    with pytest.raises(KeyError):
        my_cache.get(21, False)


def test_cache_invalidates_old_item():
    my_cache = cache.LRUCache[int, Union[int, str]](
        max_cache_items=10,
        expiration_horizon=datetime.timedelta(milliseconds=1),
        retrieval_function=retrieval_function,
    )

    mock_curr_time = datetime.datetime.fromtimestamp(1636730651.079551)
    with patch("datetime.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_curr_time
        my_cache.put(5)
        mock_datetime.now.return_value += datetime.timedelta(milliseconds=2)
        with pytest.raises(KeyError):
            my_cache.get(5, False)

    with patch("datetime.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_curr_time
        my_cache.put(5)
        mock_datetime.now.return_value += datetime.timedelta(milliseconds=0.5)
        assert my_cache.get(5, False) == (retrieval_function(key=5), True)


def test_cache_fetches_new_item():
    my_cache = cache.LRUCache[int, Union[int, str]](
        max_cache_items=10,
        expiration_horizon=datetime.timedelta(milliseconds=1),
        retrieval_function=retrieval_function,
    )

    mock_curr_time = datetime.datetime.fromtimestamp(1636730651.079551)
    with patch("datetime.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_curr_time
        my_cache.put(5, 10)
        mock_datetime.now.return_value += datetime.timedelta(milliseconds=2)
        assert my_cache.get(5) == (retrieval_function(key=5), True)

    with patch("datetime.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_curr_time
        my_cache.put(5, 10)
        mock_datetime.now.return_value += datetime.timedelta(milliseconds=0.5)
        assert my_cache.get(5, False) == (10, True)
        mock_datetime.now.return_value += datetime.timedelta(milliseconds=0.75)
        with pytest.raises(KeyError):
            my_cache.get(5, False)


def test_cache_removes_old_items_once_size_limit_reached():
    my_cache = cache.LRUCache[int, Union[int, str]](
        max_cache_items=5,
        expiration_horizon=datetime.timedelta(hours=1),
        retrieval_function=retrieval_function,
    )

    for i in [1, 2, 3, 4, 5]:
        my_cache.put(i)

    assert len(my_cache) == 5

    my_cache.put(6)
    assert len(my_cache) == 5
    with pytest.raises(KeyError):
        my_cache.get(1, False)
    assert my_cache.get(2, False) == (retrieval_function(key=2), True)


def test_cache_get_with_data_source_fallback():
    my_cache = cache.LRUCache[int, Union[int, str]](
        max_cache_items=5,
        expiration_horizon=datetime.timedelta(hours=1),
        retrieval_function=retrieval_function,
    )

    for i in range(10):
        val = my_cache.get(i)
        assert val == (retrieval_function(key=i), False)

    assert len(my_cache) == 5


def test_cache_gets_stored_value():
    my_cache = cache.LRUCache[int, Union[int, str]](
        max_cache_items=5,
        expiration_horizon=datetime.timedelta(hours=1),
        retrieval_function=retrieval_function,
    )

    for i in range(5):
        my_cache.put(i)

    my_cache._retrieval_function = MagicMock()
    my_cache.get(4)
    my_cache._retrieval_function.assert_not_called()

    my_cache._retrieval_function.reset_mock()
    my_cache.get(5)
    my_cache._retrieval_function.assert_called_with(key=5, value=None)

    my_cache._retrieval_function.reset_mock()
    my_cache.get(0)
    my_cache._retrieval_function.assert_called_with(key=0, value=None)


def test_cache_bad_retrieval_function():

    cache_no_retrieval_fx = cache.LRUCache[int, Union[int, str]](
        max_cache_items=5,
        expiration_horizon=datetime.timedelta(hours=1),
        retrieval_function=None,
    )

    with pytest.raises(TypeError):
        cache_no_retrieval_fx.put(1)

    cache_bad_retrieval_fx_signature = cache.LRUCache[int, Union[int, str]](
        max_cache_items=5,
        expiration_horizon=datetime.timedelta(hours=1),
        retrieval_function=lambda: 1,
    )

    with pytest.raises(TypeError):
        cache_bad_retrieval_fx_signature.put(1)

    cache_retrieval_fx_throws = cache.LRUCache[int, Union[int, str]](
        max_cache_items=5,
        expiration_horizon=datetime.timedelta(hours=1),
        retrieval_function=lambda key, value: exec("raise(RuntimeError())"),
    )

    with pytest.raises(RuntimeError):
        cache_retrieval_fx_throws.put(1)


def test_cache_clear_and_contains():
    my_cache = cache.LRUCache[int, Union[int, str]](
        max_cache_items=5,
        expiration_horizon=datetime.timedelta(hours=1),
        retrieval_function=retrieval_function,
    )

    for i in range(5):
        my_cache.put(i)
        assert i in my_cache

    my_cache.clear()
    assert len(my_cache) == 0
    with pytest.raises(KeyError):
        my_cache.get(1, False)
