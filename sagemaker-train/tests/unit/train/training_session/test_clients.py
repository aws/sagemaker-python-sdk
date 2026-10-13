# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
"""Unit tests for sagemaker.train.training_session._clients.

Clients are built with dummy credentials and never reach the network, so these
tests assert on what botocore resolved: endpoint, region, retry budget, and the
operations the model exposes.

Endpoints are asserted on the prepared request rather than on
``client.meta.endpoint_url``. The two can differ: ``meta.endpoint_url`` is filled
in when the client is built, while the endpoint a request is actually sent to is
resolved per request from the service's endpoint rules. The prepared request is
what the caller's configuration has to reach, so that is what is checked here.
"""

from __future__ import annotations

import os

import boto3
import pytest
from botocore.awsrequest import AWSResponse
from botocore.config import Config
from botocore.exceptions import UnknownServiceError

from sagemaker.train.training_session._clients import (
    _CONNECT_TIMEOUT,
    _READ_TIMEOUT,
    _RUNTIME_SERVICE_NAME,
    _control_plane_client,
    _require_control_plane_operations,
    _runtime_client,
)

#: The runtime model comes from the installed botocore. Tests that build a real
#: runtime client are gated on it rather than skipped silently, so a botocore
#: without the model reports which coverage it cost.
requires_runtime_model = pytest.mark.skipif(
    _RUNTIME_SERVICE_NAME not in boto3.Session(region_name="us-west-2").get_available_services(),
    reason=f"the installed botocore does not model {_RUNTIME_SERVICE_NAME}",
)

#: A control-plane and a runtime call, with the arguments each needs to serialize.
_CONTROL_PLANE_CALL = ("describe_training_session", {"TrainingSessionName": "probe"})
_RUNTIME_CALL = (
    "get_training_operation_result",
    {"trainingSessionName": "probe", "operationId": "op"},
)


class _EmptyBody:
    """The streaming interface botocore reads a response body through."""

    def stream(self, **kwargs):
        yield b"{}"


def _prepared_url(client, call) -> str:
    """Return the URL ``client`` prepares for ``call``, sending nothing.

    The response is answered from ``before-send``, so the request is signed and
    fully prepared, exactly as it would go on the wire, and then dropped.
    """
    operation, params = call
    seen = {}

    def _capture(request, **kwargs):
        seen["url"] = request.url
        return AWSResponse(request.url, 200, {}, _EmptyBody())

    client.meta.events.register("before-send.*.*", _capture)
    getattr(client, operation)(**params)
    return seen["url"]


_CONTROL_PLANE_OPERATIONS = frozenset(
    {
        "CreateTrainingSession",
        "DescribeTrainingSession",
        "ListTrainingSessions",
        "StopTrainingSession",
        "DeleteTrainingSession",
    }
)

_RUNTIME_OPERATIONS = frozenset(
    {
        "ExecuteTrainingOperation",
        "GetTrainingOperationResult",
        "ExecuteSamplingOperation",
        "GetSamplingOperationResult",
    }
)


@pytest.fixture(autouse=True)
def _dummy_credentials(monkeypatch):
    """Let clients build without reaching the credential chain."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.delenv("AWS_PROFILE", raising=False)


# --- runtime client ---------------------------------------------------------


@requires_runtime_model
def test_runtime_client_exposes_the_operations_the_sdk_calls():
    client = _runtime_client(region="us-west-2")

    assert _RUNTIME_OPERATIONS <= set(client.meta.service_model.operation_names)


@requires_runtime_model
def test_runtime_client_targets_the_regional_production_endpoint():
    client = _runtime_client(region="eu-west-1")

    assert _prepared_url(client, _RUNTIME_CALL).startswith(
        "https://training-session-runtime.sagemaker.eu-west-1.api.aws/"
    )
    assert client.meta.region_name == "eu-west-1"


@requires_runtime_model
def test_runtime_client_honours_the_endpoint_override(monkeypatch):
    monkeypatch.setenv(
        "SAGEMAKER_TRAINING_SESSION_RUNTIME_ENDPOINT", "https://runtime.example.test"
    )

    client = _runtime_client(region="us-west-2")

    assert _prepared_url(client, _RUNTIME_CALL).startswith("https://runtime.example.test/")


@requires_runtime_model
def test_runtime_client_does_not_retry():
    """The submit and poll loops own the retry policy; a second one underneath
    them would multiply the wait."""
    client = _runtime_client(region="us-west-2")

    assert client.meta.config.retries["total_max_attempts"] == 1


@requires_runtime_model
def test_runtime_client_pool_is_sized_for_sampling_fan_out():
    client = _runtime_client(region="us-west-2")

    assert client.meta.config.max_pool_connections == 512


@requires_runtime_model
def test_runtime_client_pool_size_is_overridable():
    client = _runtime_client(region="us-west-2", max_pool_connections=8)

    assert client.meta.config.max_pool_connections == 8


@requires_runtime_model
def test_the_runtime_client_shortens_the_wait_for_a_connection():
    """The runtime client sends each call once, so this wait is the caller's wait."""
    assert _runtime_client(region="us-west-2").meta.config.connect_timeout == 10


def test_the_control_plane_client_shortens_it_too():
    """Asserted separately: this client needs no runtime model, so it is not gated."""
    assert _control_plane_client(region="us-west-2").meta.config.connect_timeout == 10


def test_shortening_the_connection_wait_is_a_departure_from_the_default():
    """Without which the assertions above would hold whether or not we set it.

    Ten seconds is a decision; sixty is what a client is given when nobody
    decides. Pinning the difference is what makes those two tests mean something.
    """
    assert Config().connect_timeout == 60
    assert _CONNECT_TIMEOUT == 10


def test_the_read_wait_is_pinned_to_what_a_client_is_given_by_default():
    """The read wait deliberately matches the default, so watch the default.

    Asserting the value on the client cannot show that we set it, because the
    number is the same either way. What is worth catching is the default moving
    underneath us: on that day this value stops being the status quo and becomes
    a change in behaviour, which should be a decision rather than a surprise.
    """
    assert _READ_TIMEOUT == Config().read_timeout
    assert _control_plane_client(region="us-west-2").meta.config.read_timeout == _READ_TIMEOUT


# --- control-plane client ---------------------------------------------------


def test_control_plane_client_exposes_the_training_session_operations():
    client = _control_plane_client(region="us-west-2")

    assert _CONTROL_PLANE_OPERATIONS <= set(client.meta.service_model.operation_names)


def test_the_create_request_the_sdk_builds_validates_against_the_api_model():
    """Every key in a built CreateTrainingSession request is one the API models.

    The other create tests drive a stub client, which accepts any key it is
    handed. A key the API does not model is refused before the request is sent,
    so a stub cannot see the difference between a field that works and one that
    fails every call -- only the real model can.
    """
    from botocore.validate import validate_parameters

    from sagemaker.train.training_session.service_client import _build_create_session_args

    args = _build_create_session_args(
        session_name="session",
        base_model_arn="arn:aws:sagemaker:us-west-2:aws:hub-content/H/Model/m/1.0.0",
        model_package_group_arn="arn:aws:sagemaker:us-west-2:123456789012:model-package-group/g",
        lora_config={"Rank": 32, "Alpha": 64},
        role_arn="arn:aws:iam::123456789012:role/R",
        s3_path="s3://bucket/prefix/",
        kms_key_arn="arn:aws:kms:us-west-2:123456789012:key/k",
        vpc_config=None,
    )

    shape = (
        _control_plane_client(region="us-west-2")
        .meta.service_model.operation_model("CreateTrainingSession")
        .input_shape
    )
    # Raises ParamValidationError naming any unmodelled or missing key.
    validate_parameters(args, shape)


def test_control_plane_client_targets_the_regional_production_endpoint():
    client = _control_plane_client(region="eu-west-1")

    assert _prepared_url(client, _CONTROL_PLANE_CALL).startswith(
        "https://api.sagemaker.eu-west-1.amazonaws.com/"
    )
    assert client.meta.region_name == "eu-west-1"


def test_control_plane_client_honours_the_endpoint_override(monkeypatch):
    monkeypatch.setenv("SAGEMAKER_TRAINING_SESSION_ENDPOINT", "https://cp.example.test")

    client = _control_plane_client(region="us-west-2")

    assert _prepared_url(client, _CONTROL_PLANE_CALL).startswith("https://cp.example.test/")


def test_control_plane_client_retries_throttling_and_5xx():
    client = _control_plane_client(region="us-west-2")

    assert client.meta.config.retries == {"mode": "standard", "total_max_attempts": 4}


# --- caller configuration both clients have to honour -----------------------


def test_control_plane_client_honours_a_fips_configuration(monkeypatch):
    """A caller who requires FIPS endpoints must not silently get plain ones.

    ``AWS_USE_FIPS_ENDPOINT`` is read by botocore, not by this package, so this
    holds only while the endpoint is left to botocore to resolve.
    """
    monkeypatch.setenv("AWS_USE_FIPS_ENDPOINT", "true")

    assert _prepared_url(_control_plane_client(region="eu-west-1"), _CONTROL_PLANE_CALL).startswith(
        "https://api-fips.sagemaker.eu-west-1.amazonaws.com/"
    )


@requires_runtime_model
def test_runtime_client_honours_a_fips_configuration(monkeypatch):
    monkeypatch.setenv("AWS_USE_FIPS_ENDPOINT", "true")

    assert _prepared_url(_runtime_client(region="eu-west-1"), _RUNTIME_CALL).startswith(
        "https://training-session-runtime.sagemaker-fips.eu-west-1.api.aws/"
    )


def test_control_plane_client_uses_the_regions_own_dns_suffix():
    """A Region outside the ``aws`` partition does not answer on amazonaws.com."""
    assert _prepared_url(
        _control_plane_client(region="cn-north-1"), _CONTROL_PLANE_CALL
    ).startswith("https://api.sagemaker.cn-north-1.amazonaws.com.cn/")


@requires_runtime_model
def test_runtime_client_uses_the_regions_own_dns_suffix():
    assert _prepared_url(_runtime_client(region="cn-north-1"), _RUNTIME_CALL).startswith(
        "https://training-session-runtime.sagemaker.cn-north-1.api.amazonwebservices.com.cn/"
    )


def test_no_configured_region_is_an_error_not_a_default(monkeypatch):
    """An unconfigured environment must not pick a Region for the caller.

    Creating a session in an unasked-for Region starts GPUs there, so this is
    raised rather than defaulted.
    """
    for name in ("AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", os.devnull)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", os.devnull)

    with pytest.raises(ValueError, match="No AWS Region configured"):
        _control_plane_client()
    with pytest.raises(ValueError, match="No AWS Region configured"):
        _runtime_client()


def test_older_botocore_fails_with_an_actionable_message():
    """Checked against the operations the model has, not its version string."""

    class _ModelWithoutTrainingSessions:
        operation_names = ["CreateEndpoint", "DescribeEndpoint"]

    class _Client:
        class meta:
            service_model = _ModelWithoutTrainingSessions()

    with pytest.raises(RuntimeError, match="does not describe the SageMaker training-session"):
        _require_control_plane_operations(_Client())


def test_the_control_plane_message_names_no_version_to_upgrade_to():
    """Naming a release only helps once one carries the operations.

    Pointing at a version that does not have them sends the caller through an
    upgrade that changes nothing, so the message reports what is missing and
    leaves the release out.
    """

    class _ModelWithoutTrainingSessions:
        operation_names = ["CreateEndpoint"]

    class _Client:
        class meta:
            service_model = _ModelWithoutTrainingSessions()

    with pytest.raises(RuntimeError) as caught:
        _require_control_plane_operations(_Client())

    message = str(caught.value)
    assert "pip install" not in message
    assert ">=" not in message


def test_current_botocore_passes_the_check():
    _require_control_plane_operations(_control_plane_client(region="us-west-2"))


def test_a_botocore_without_the_runtime_model_fails_with_an_actionable_message():
    """The lookup failure botocore raises names neither the cause nor the fix."""

    class _SessionWithoutTheRuntime:
        def client(self, service_name, **kwargs):
            raise UnknownServiceError(service_name=service_name, known_service_names=["sagemaker"])

    with pytest.raises(RuntimeError, match="does not describe the SageMaker training-session"):
        _runtime_client(region="us-west-2", boto_session=_SessionWithoutTheRuntime())


def test_the_supplied_sessions_region_is_used_when_none_is_named(monkeypatch):
    """Handing over a session is itself a choice of Region.

    Both factories resolved the Region from a fresh session instead of the one
    passed in, so a caller's explicitly pinned session reached another Region.
    """
    for name in ("AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", os.devnull)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", os.devnull)
    session = boto3.Session(region_name="eu-west-1")

    assert _control_plane_client(boto_session=session).meta.region_name == "eu-west-1"
    assert _runtime_client(boto_session=session).meta.region_name == "eu-west-1"


def test_the_supplied_session_outranks_an_ambient_aws_region(monkeypatch):
    """A session's own Region is not overridden by one named in the environment.

    Managed compute exports AWS_REGION, so without this these clients could end
    up in a different Region from every other client built off that session.
    """
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", os.devnull)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", os.devnull)
    session = boto3.Session(region_name="eu-west-1")

    assert _control_plane_client(boto_session=session).meta.region_name == "eu-west-1"
    assert _runtime_client(boto_session=session).meta.region_name == "eu-west-1"


def test_a_session_without_a_region_is_an_error_not_another_profiles(tmp_path, monkeypatch):
    """A supplied session answers alone, so an unset Region is not borrowed.

    Resolving on through the standard chain took the Region from whichever
    profile that chain lands on, while the credentials came from this one.
    """
    config = tmp_path / "config"
    config.write_text("[default]\nregion = us-west-2\n\n[profile noregion]\n")
    creds = tmp_path / "credentials"
    creds.write_text("[noregion]\naws_access_key_id = k\naws_secret_access_key = s\n")
    for name in ("AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(creds))
    session = boto3.Session(profile_name="noregion")

    with pytest.raises(ValueError, match="No AWS Region configured"):
        _control_plane_client(boto_session=session)
    with pytest.raises(ValueError, match="No AWS Region configured"):
        _runtime_client(boto_session=session)


def test_aws_region_still_answers_when_no_session_is_named(monkeypatch):
    """boto3 does not read AWS_REGION, so it stays worth reading explicitly."""
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", os.devnull)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", os.devnull)

    assert _control_plane_client().meta.region_name == "us-east-1"
    assert _runtime_client().meta.region_name == "us-east-1"


def test_an_explicit_region_still_wins_over_the_sessions(monkeypatch):
    for name in ("AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(name, raising=False)
    session = boto3.Session(region_name="eu-west-1")

    client = _control_plane_client(region="us-west-2", boto_session=session)

    assert client.meta.region_name == "us-west-2"
