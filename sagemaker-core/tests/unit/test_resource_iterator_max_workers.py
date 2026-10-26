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
"""``ResourceIterator(max_workers=N)`` describes N listed resources at a time.

Before this change the iterator described one resource per ``next()`` call, so a list of
the public hub (about 770 models) spent one ``DescribeHubContent`` round trip per model
in series, about 95 seconds. ``HubContent.get_all`` and ``get_all_versions`` take
``max_workers`` and hand it to the iterator, with 4 as the default.
"""

from __future__ import absolute_import

import inspect
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.core.resources import Base, HubContent
from sagemaker.core.utils.utils import ResourceIterator, SageMakerClient

HUB = "SageMakerPublicHub"
DESCRIBE_SECONDS = 0.05


def _summary(name):
    return {
        "HubContentName": name,
        "HubContentArn": f"arn:aws:sagemaker:us-west-2:aws:hub-content/{HUB}/Model/{name}/1.0.0",
        "HubContentVersion": "1.0.0",
        "HubContentType": "Model",
        "DocumentSchemaVersion": "2.0.0",
        "HubContentStatus": "Available",
    }


class _Recorder:
    """A fake client whose describe call sleeps and records its concurrency."""

    def __init__(self, names, fail_on=None):
        self.client = MagicMock()
        self.client.list_hub_contents.return_value = {
            "HubContentSummaries": [_summary(name) for name in names]
        }
        self.client.describe_hub_content.side_effect = self._describe
        self.fail_on = fail_on
        self.in_flight = 0
        self.peak_in_flight = 0
        self.started = []
        self._lock = threading.Lock()

    def _describe(self, **kwargs):
        with self._lock:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            self.started.append(kwargs["HubContentName"])
        try:
            time.sleep(DESCRIBE_SECONDS)
            if kwargs["HubContentName"] == self.fail_on:
                raise RuntimeError(f"describe of {self.fail_on} failed")
            return {
                **_summary(kwargs["HubContentName"]),
                "HubName": kwargs["HubName"],
                "HubContentDocument": "{}",
            }
        finally:
            with self._lock:
                self.in_flight -= 1


def _iterator(recorder, max_workers):
    return ResourceIterator(
        client=recorder.client,
        summaries_key="HubContentSummaries",
        summary_name="HubContentInfo",
        resource_cls=HubContent,
        list_method="list_hub_contents",
        list_method_kwargs={"HubName": HUB, "HubContentType": "Model"},
        extra_init_data={"hub_name": HUB},
        max_workers=max_workers,
    )


@pytest.fixture(autouse=True)
def _reset_client_cache():
    SageMakerClient.reset()
    yield
    SageMakerClient.reset()


def test_max_workers_describes_resources_concurrently_in_list_order():
    names = [f"model-{index:02d}" for index in range(12)]
    recorder = _Recorder(names)

    with patch.object(Base, "get_sagemaker_client", return_value=recorder.client):
        started = time.monotonic()
        contents = list(_iterator(recorder, max_workers=4))
        elapsed = time.monotonic() - started

    assert [content.hub_content_name for content in contents] == names
    assert all(content.hub_name == HUB for content in contents)
    assert recorder.peak_in_flight > 1
    assert recorder.client.describe_hub_content.call_count == len(names)
    assert elapsed < len(names) * DESCRIBE_SECONDS


def test_default_max_workers_describes_one_resource_at_a_time():
    recorder = _Recorder(["model-a", "model-b", "model-c"])

    with patch.object(Base, "get_sagemaker_client", return_value=recorder.client):
        contents = list(_iterator(recorder, max_workers=1))

    assert [content.hub_content_name for content in contents] == ["model-a", "model-b", "model-c"]
    assert recorder.peak_in_flight == 1


def test_stopping_early_describes_at_most_max_workers_resources():
    names = [f"model-{index:02d}" for index in range(20)]
    recorder = _Recorder(names)

    with patch.object(Base, "get_sagemaker_client", return_value=recorder.client):
        iterator = _iterator(recorder, max_workers=4)
        first = next(iterator)
        iterator._shutdown_executor()

    assert first.hub_content_name == "model-00"
    assert recorder.client.describe_hub_content.call_count <= 4


def test_a_describe_error_raises_in_list_order_and_releases_the_pool():
    recorder = _Recorder(["model-a", "model-b", "model-c"], fail_on="model-b")

    with patch.object(Base, "get_sagemaker_client", return_value=recorder.client):
        iterator = _iterator(recorder, max_workers=3)
        assert next(iterator).hub_content_name == "model-a"
        with pytest.raises(RuntimeError, match="describe of model-b failed"):
            next(iterator)
        iterator._shutdown_executor()

    assert iterator._executor is None


def test_max_workers_below_one_is_rejected():
    with pytest.raises(ValueError, match="max_workers must be at least 1"):
        _iterator(_Recorder([]), max_workers=0)


def test_hub_content_list_methods_take_max_workers():
    for method in (HubContent.get_all, HubContent.get_all_versions):
        parameter = inspect.signature(method).parameters["max_workers"]
        assert parameter.default == 4, method.__qualname__


def test_hub_content_get_all_hands_max_workers_to_the_iterator():
    recorder = _Recorder(["model-a"])

    with patch.object(Base, "get_sagemaker_client", return_value=recorder.client):
        iterator = HubContent.get_all(hub_name=HUB, hub_content_type="Model", max_workers=2)

    assert iterator.max_workers == 2
