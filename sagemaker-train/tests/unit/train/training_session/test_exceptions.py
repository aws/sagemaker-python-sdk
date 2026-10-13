# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
"""Unit tests for sagemaker.train.training_session.exceptions.

``is_retryable`` is the one classification the SDK still makes over a wire
error, and both the submit retry and the poll loops depend on it: a failure is
either transient, and worth sending again, or it is not.
"""

from __future__ import annotations

import pickle

import pytest
from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    ParamValidationError,
    ReadTimeoutError,
)

from sagemaker.train.training_session.exceptions import (
    ArtifactDeletionError,
    OperationError,
    ResultMismatchError,
    is_retryable,
)


def _client_error(code, message="", status=400):
    return ClientError(
        {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "ExecuteTrainingOperation",
    )


# --- is_retryable -----------------------------------------------------------


@pytest.mark.parametrize(
    "code,status",
    [("ThrottlingException", 429), ("InternalServerException", 500)],
)
def test_modelled_transient_codes_are_retryable(code, status):
    assert is_retryable(_client_error(code, "slow down", status=status))


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_unmodelled_transient_statuses_are_retryable(status):
    """A gateway ahead of the service can answer without a modelled code."""
    assert is_retryable(_client_error("SomethingUnmodelled", "gateway", status=status))


@pytest.mark.parametrize(
    "code,status",
    [
        ("ValidationException", 400),
        ("ConflictException", 409),
        ("ResourceNotFoundException", 404),
        ("AccessDeniedException", 403),
    ],
)
def test_request_rejections_are_not_retryable(code, status):
    """Resending a rejected request gets the same answer."""
    assert not is_retryable(_client_error(code, "no", status=status))


@pytest.mark.parametrize(
    "error",
    [
        ReadTimeoutError(endpoint_url="https://runtime.test"),
        ConnectTimeoutError(endpoint_url="https://runtime.test"),
    ],
)
def test_connection_failures_are_retryable(error):
    """No response means no status, and the operation may still be running."""
    assert is_retryable(error)


def test_local_request_validation_is_not_retryable():
    """botocore checks the request against the model before sending it, and that
    verdict does not change on a second try."""
    assert not is_retryable(
        ParamValidationError(report="Missing required parameter: operationConfig")
    )


def test_error_without_a_response_body_does_not_crash():
    """A malformed error must not fail inside the classifier."""
    assert not is_retryable(ClientError({}, "ExecuteTrainingOperation"))


def test_unrelated_exception_is_not_retryable():
    assert not is_retryable(ValueError("not a wire error"))


# --- OperationError ---------------------------------------------------------


def test_operation_error_carries_the_operation_id():
    """The id is what identifies the failed operation when asking for help."""
    error = OperationError("training", "out of memory", operation_id="op-7")

    assert error.operation_id == "op-7"
    assert error.operation_type == "training"
    assert "op-7" in str(error)
    assert "out of memory" in str(error)


def test_operation_error_message_omits_an_absent_id():
    error = OperationError("training", "out of memory")

    assert error.operation_id is None
    assert "None" not in str(error)
    assert str(error) == "training failed: out of memory"


# --- crossing a process boundary ---------------------------------------------
# A training loop that spreads work across processes moves exceptions between
# them, and each of these takes more constructor arguments than the single
# message the default rebuild would pass. Without an explicit rebuild the real
# failure is replaced by a TypeError about a missing argument.


@pytest.mark.parametrize(
    "error",
    [
        OperationError("training", "out of memory", operation_id="op-1"),
        OperationError("training", "out of memory"),
        ResultMismatchError("computeLogprobs", 3, 2),
        ArtifactDeletionError(
            "arn:aws:sagemaker:us-west-2:1:model-package/g/1", "s3://bucket/key", ["k1", "k2"]
        ),
    ],
    ids=["operation-with-id", "operation-without-id", "mismatch", "deletion"],
)
def test_every_error_survives_a_pickle_round_trip(error):
    restored = pickle.loads(pickle.dumps(error))

    assert type(restored) is type(error)
    assert str(restored) == str(error)


def test_a_round_trip_keeps_the_attributes_callers_read():
    """The attributes are the reason to catch these, so they have to survive."""
    original = OperationError("training", "out of memory", operation_id="op-1")

    restored = pickle.loads(pickle.dumps(original))

    assert restored.operation_type == "training"
    assert restored.operation_id == "op-1"
    assert restored.message == "out of memory"


def test_a_round_trip_keeps_the_full_failed_key_list():
    """The message truncates the keys; the attribute is what carries them all."""
    keys = [f"k{index}" for index in range(12)]
    original = ArtifactDeletionError("arn:mp/1", "s3://bucket/prefix/", keys)

    restored = pickle.loads(pickle.dumps(original))

    assert restored.failed_keys == keys
    assert restored.s3_uri == "s3://bucket/prefix/"
    assert restored.model_package_arn == "arn:mp/1"
