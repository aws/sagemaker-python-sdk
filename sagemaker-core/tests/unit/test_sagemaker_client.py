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
"""Tests for SageMakerClient caching and session handling.

Regression coverage for aws/sagemaker-python-sdk#5986 and #6069: a client built
for one boto3 session must never be handed to a caller that passed a different
session, and the ``config`` argument must actually reach the boto3 clients.
"""

from __future__ import absolute_import

import boto3
import pytest
from botocore.config import Config

from sagemaker.core.utils.utils import SageMakerClient

REGION_A = "us-west-2"
REGION_B = "us-east-1"


def _session(access_key, region):
    """Offline boto3 session carrying distinguishable static credentials."""
    return boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key="secret-" + access_key,
        region_name=region,
    )


def _access_key(boto_client):
    return boto_client._request_signer._credentials.access_key


@pytest.fixture(autouse=True)
def _isolated_cache():
    SageMakerClient.reset()
    yield
    SageMakerClient.reset()


def test_explicit_session_is_honored_after_another_client_exists():
    """The second session must not receive the first session's client (#5986, #6069)."""
    session_a = _session("AKIAFIRSTSESSION0000", REGION_A)
    session_b = _session("AKIASECONDSESSION000", REGION_B)

    client_a = SageMakerClient(session=session_a)
    client_b = SageMakerClient(session=session_b)

    assert client_a is not client_b
    assert client_b.session is session_b
    assert client_b.region_name == REGION_B
    for service in (
        "sagemaker",
        "sagemaker-runtime",
        "sagemaker-featurestore-runtime",
        "sagemaker-metrics",
    ):
        assert _access_key(client_b.get_client(service)) == "AKIASECONDSESSION000"
        assert _access_key(client_a.get_client(service)) == "AKIAFIRSTSESSION0000"


def test_same_session_returns_cached_instance():
    session_a = _session("AKIAFIRSTSESSION0000", REGION_A)

    assert SageMakerClient(session=session_a) is SageMakerClient(session=session_a)
    assert SageMakerClient(session=session_a, region_name=REGION_A) is SageMakerClient(
        session=session_a, region_name=REGION_A
    )


def test_same_session_different_region_gets_distinct_client():
    session_a = _session("AKIAFIRSTSESSION0000", REGION_A)

    client_a = SageMakerClient(session=session_a, region_name=REGION_A)
    client_b = SageMakerClient(session=session_a, region_name=REGION_B)

    assert client_a is not client_b
    assert client_b.sagemaker_client.meta.region_name == REGION_B


def test_bare_call_returns_first_created_instance_as_default():
    """Configure-once pattern: SageMakerClient(session=...) then bare SageMakerClient()."""
    session_a = _session("AKIAFIRSTSESSION0000", REGION_A)

    configured = SageMakerClient(session=session_a)

    assert SageMakerClient() is configured
    assert SageMakerClient() is configured


def test_bare_call_default_is_not_replaced_by_later_explicit_session():
    session_a = _session("AKIAFIRSTSESSION0000", REGION_A)
    session_b = _session("AKIASECONDSESSION000", REGION_B)

    default = SageMakerClient(session=session_a)
    SageMakerClient(session=session_b)

    assert SageMakerClient() is default


def test_bare_call_creates_default_from_default_chain(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAENVIRONMENT00000")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION_A)

    default = SageMakerClient()

    assert SageMakerClient() is default
    assert default.region_name == REGION_A
    assert _access_key(default.sagemaker_client) == "AKIAENVIRONMENT00000"


def test_config_argument_is_applied_and_user_agent_suffix_appended():
    session_a = _session("AKIAFIRSTSESSION0000", REGION_A)
    config = Config(connect_timeout=3, read_timeout=7, user_agent_extra="mytool/1.0")

    client = SageMakerClient(session=session_a, config=config)

    meta_config = client.sagemaker_client.meta.config
    assert meta_config.connect_timeout == 3
    assert meta_config.read_timeout == 7
    assert meta_config.user_agent_extra.startswith("mytool/1.0 ")
    assert "sagemaker" in meta_config.user_agent_extra.lower()


def test_default_config_keeps_retry_settings():
    session_a = _session("AKIAFIRSTSESSION0000", REGION_A)

    client = SageMakerClient(session=session_a)

    # botocore normalizes {"max_attempts": N} to {"total_max_attempts": N + 1}.
    retries = client.sagemaker_client.meta.config.retries
    assert retries["total_max_attempts"] == 11
    assert retries["mode"] == "standard"


def test_distinct_config_objects_get_distinct_clients():
    session_a = _session("AKIAFIRSTSESSION0000", REGION_A)
    config_1 = Config(connect_timeout=1)
    config_2 = Config(connect_timeout=2)

    client_1 = SageMakerClient(session=session_a, config=config_1)
    client_2 = SageMakerClient(session=session_a, config=config_2)

    assert client_1 is not client_2
    assert client_1.sagemaker_client.meta.config.connect_timeout == 1
    assert client_2.sagemaker_client.meta.config.connect_timeout == 2


def test_reset_clears_default_and_keyed_entries():
    session_a = _session("AKIAFIRSTSESSION0000", REGION_A)
    client_a = SageMakerClient(session=session_a)
    default = SageMakerClient()

    SageMakerClient.reset()

    assert SageMakerClient(session=session_a) is not client_a
    assert SageMakerClient() is not default


def _cached_entries():
    from sagemaker.core.utils.utils import _ClientCacheMeta

    return _ClientCacheMeta._instances.get(SageMakerClient, {})


def test_keyed_cache_is_bounded_and_default_survives_eviction():
    from sagemaker.core.utils.utils import _ClientCacheMeta

    cap = _ClientCacheMeta._MAX_KEYED_ENTRIES
    sessions = [_session(f"AKIA{i:016d}", REGION_A) for i in range(cap + 5)]

    default = SageMakerClient(session=sessions[0])
    clients = [SageMakerClient(session=s) for s in sessions]

    entries = _cached_entries()
    keyed = [k for k in entries if k != _ClientCacheMeta._DEFAULT_KEY]
    assert len(keyed) == cap
    # The oldest keyed entries were dropped, the newest kept.
    assert SageMakerClient(session=sessions[-1]) is clients[-1]
    assert SageMakerClient(session=sessions[0]) is not clients[0]
    # The process default is pinned regardless of eviction.
    assert SageMakerClient() is default


def test_recently_used_entry_is_kept_over_stale_ones():
    from sagemaker.core.utils.utils import _ClientCacheMeta

    cap = _ClientCacheMeta._MAX_KEYED_ENTRIES
    sessions = [_session(f"AKIA{i:016d}", REGION_A) for i in range(cap)]
    clients = [SageMakerClient(session=s) for s in sessions]

    # Touch the oldest entry so it becomes most recently used, then overflow by one.
    assert SageMakerClient(session=sessions[0]) is clients[0]
    SageMakerClient(session=_session("AKIAOVERFLOW00000000", REGION_A))

    assert SageMakerClient(session=sessions[0]) is clients[0]
    assert SageMakerClient(session=sessions[1]) is not clients[1]


def test_concurrent_first_calls_share_one_instance():
    import threading

    session_a = _session("AKIAFIRSTSESSION0000", REGION_A)
    results = []
    start = threading.Barrier(8)

    def build():
        start.wait()
        results.append(SageMakerClient(session=session_a))

    threads = [threading.Thread(target=build) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 8
    assert all(r is results[0] for r in results)
