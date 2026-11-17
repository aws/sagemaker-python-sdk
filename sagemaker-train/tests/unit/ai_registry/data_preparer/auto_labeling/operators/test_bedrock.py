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
"""Unit tests for operators.common.bedrock: the Converse client and its retries."""

from __future__ import absolute_import

import boto3
import pytest
from botocore.exceptions import ClientError, ConnectionClosedError

from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _bedrock as bedrock


def _err(code, msg=""):
    return ClientError({"Error": {"Code": code, "Message": msg}}, "Converse")


class _Client:
    def __init__(self, script):
        self.script = list(script)
        self.sent = []

    def converse(self, **kw):
        self.sent.append(kw)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return {"output": {"message": {"content": item}}}


class TestInvoke:
    def test_joins_text_blocks_and_sends_defaults(self):
        c = _Client([[{"reasoning": "r"}, {"text": "a"}, {"text": "b"}]])
        assert bedrock.invoke(c, "m", "p") == "ab"
        assert c.sent == [
            {
                "modelId": "m",
                "messages": [{"role": "user", "content": [{"text": "p"}]}],
                "inferenceConfig": {"maxTokens": 512, "temperature": 0.0},
            }
        ]

    def test_retries_transient_errors_with_backoff(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(bedrock.time, "sleep", sleeps.append)
        c = _Client(
            [_err("ThrottlingException"), ConnectionClosedError(endpoint_url="u"), [{"text": "ok"}]]
        )
        assert bedrock.invoke(c, "m", "p") == "ok"
        assert sleeps == [1, 2]

    def test_non_retryable_and_exhausted(self, monkeypatch):
        monkeypatch.setattr(bedrock.time, "sleep", lambda s: None)
        with pytest.raises(ClientError):
            bedrock.invoke(_Client([_err("AccessDeniedException")]), "m", "p")
        with pytest.raises(ClientError):
            bedrock.invoke(_Client([_err("ThrottlingException")] * bedrock._MAX_ATTEMPTS), "m", "p")

    def test_temperature_refused_once_then_sent_without(self):
        c = _Client([_err("ValidationException", "temperature is not supported"), [{"text": "ok"}]])
        assert bedrock.invoke(c, "m", "p", system_prompt="s", temperature=0.3, max_tokens=9) == "ok"
        assert c.sent[0]["inferenceConfig"] == {"maxTokens": 9, "temperature": 0.3}
        assert c.sent[1]["inferenceConfig"] == {"maxTokens": 9} and c.sent[1]["system"] == [
            {"text": "s"}
        ]
        assert bedrock.rejects_temperature(_err("ValidationException", "bad temperature")) is False
        assert bedrock.rejects_temperature(object()) is False


class TestMakeClient:
    def test_config_retries_off_and_timeouts(self, monkeypatch):
        seen = {}

        def _client(service, region_name=None, config=None):
            seen.update(service=service, region=region_name, config=config)
            return "client"

        monkeypatch.setattr(boto3, "client", _client)
        assert bedrock.make_client("us-east-1", read_timeout=600, connect_timeout=30) == "client"
        assert seen["service"] == "bedrock-runtime" and seen["region"] == "us-east-1"
        assert seen["config"].retries == {"mode": "adaptive", "max_attempts": 1}
        assert (seen["config"].read_timeout, seen["config"].connect_timeout) == (600, 30)
