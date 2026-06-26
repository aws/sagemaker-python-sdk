# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
"""Unit tests for Secret.from_string()."""
from __future__ import absolute_import

import boto3
from botocore.stub import Stubber

from sagemaker.serve.ai_inference_recommender import Secret


class TestSecretFromString:
    def test_creates_secret_with_default_name(self):
        session = boto3.session.Session(region_name="us-east-1")
        client = session.client("secretsmanager")
        with Stubber(client) as stub:
            stub.add_response(
                "create_secret",
                {
                    "ARN": "arn:aws:secretsmanager:us-east-1:123:secret:sagemaker-workload-abc-AbCdEf",
                    "Name": "sagemaker-workload-abc",
                    "VersionId": "00000000-0000-0000-0000-000000000001",
                },
                expected_params={
                    "Name": _StartsWith("sagemaker-workload-"),
                    "SecretString": "my-token-value",
                },
            )
            # Inject the stubbed client by passing a session whose .client() returns it.

            class _SessionStub:
                def client(self, name):
                    assert name == "secretsmanager"
                    return client

            secret = Secret.from_string("my-token-value", session=_SessionStub())
            assert secret.arn == "arn:aws:secretsmanager:us-east-1:123:secret:sagemaker-workload-abc-AbCdEf"

    def test_creates_secret_with_custom_name(self):
        session = boto3.session.Session(region_name="us-east-1")
        client = session.client("secretsmanager")
        with Stubber(client) as stub:
            stub.add_response(
                "create_secret",
                {
                    "ARN": "arn:aws:secretsmanager:us-east-1:123:secret:my-name-AbCdEf",
                    "Name": "my-name",
                    "VersionId": "00000000-0000-0000-0000-000000000002",
                },
                expected_params={"Name": "my-name", "SecretString": "v"},
            )

            class _SessionStub:
                def client(self, name):
                    return client

            secret = Secret.from_string("v", name="my-name", session=_SessionStub())
            assert secret.arn.endswith(":my-name-AbCdEf")

    def test_delete_calls_force_delete_by_default(self):
        client = boto3.session.Session(region_name="us-east-1").client("secretsmanager")
        with Stubber(client) as stub:
            stub.add_response(
                "delete_secret",
                {"ARN": "arn:aws:secretsmanager:us-east-1:123:secret:foo", "Name": "foo"},
                expected_params={
                    "SecretId": "arn:aws:secretsmanager:us-east-1:123:secret:foo",
                    "ForceDeleteWithoutRecovery": True,
                },
            )

            class _SessionStub:
                def client(self, name):
                    return client

            Secret(arn="arn:aws:secretsmanager:us-east-1:123:secret:foo").delete(
                session=_SessionStub()
            )

    def test_context_manager_deletes_on_exit(self):
        deleted = {"called": False}

        def _spy_delete(self, *, force_delete_without_recovery=True, session=None):
            deleted["called"] = True

        original = Secret.delete
        try:
            Secret.delete = _spy_delete  # type: ignore[assignment]
            with Secret(arn="arn:aws:secretsmanager:us-east-1:123:secret:foo"):
                pass
            assert deleted["called"] is True
        finally:
            Secret.delete = original  # type: ignore[assignment]


class _StartsWith:
    """Helper for partial-string matching in Stubber expected_params."""

    def __init__(self, prefix: str):
        self.prefix = prefix

    def __eq__(self, other) -> bool:
        return isinstance(other, str) and other.startswith(self.prefix)

    def __repr__(self) -> str:
        return f"_StartsWith({self.prefix!r})"
