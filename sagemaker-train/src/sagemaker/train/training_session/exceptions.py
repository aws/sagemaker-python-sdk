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
"""Exceptions for the training_session client interface.

A call that fails on the wire raises the error ``botocore`` models for it:
``ValidationException``, ``ConflictException``, ``ResourceNotFoundException``,
``ThrottlingException``, ``InternalServerException``, reachable both as
attributes of the client (``client.exceptions.ValidationException``) and as
``botocore.exceptions.ClientError``, whose ``response`` carries the service's
own error code and message. The SDK passes those through untouched, so nothing
here has to be caught to read what the service said.

The exceptions defined here cover what a wire error cannot express: an
asynchronous operation that was accepted and later failed
(:class:`OperationError`), and two conditions the SDK detects on the client side
(:class:`ResultMismatchError`, :class:`ArtifactDeletionError`). A condition with
nothing specific to report, such as an operation asked of a client that is
already closed, raises the base :class:`TrainingSessionError` itself, so catching
that one catches every error defined here.

Running out of time waiting is the exception to that: a wait that outlives its
``timeout`` raises the built-in ``TimeoutError``, matching what
``concurrent.futures`` and ``asyncio`` raise for the same thing, so an existing
``except TimeoutError`` keeps working. Catch it alongside
:class:`TrainingSessionError` to cover both. Use :func:`is_retryable` to decide
whether a failed call is worth sending again.
"""

from __future__ import annotations

from typing import List, Optional

from botocore.exceptions import ClientError
from botocore.exceptions import ConnectionError as BotoConnectionError
from botocore.exceptions import HTTPClientError

#: Error codes for a failure that says nothing about the request's validity, so
#: sending it again may well succeed.
_RETRYABLE_ERROR_CODES = frozenset({"ThrottlingException", "InternalServerException"})

#: Status codes carrying the same meaning, for a failure returned without an
#: error code to read.
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def is_retryable(error: BaseException) -> bool:
    """Return whether sending the failed call again may succeed.

    True for throttling, an HTTP 5xx, and connection-level failures such as a
    read or connect timeout: the request did not arrive, or arrived and could not
    be served for a reason unrelated to what it asked for. False for everything
    else, including a rejected request, since resending that one gets the same
    answer.

    The SDK already retries its own internal calls on this basis. Use it to
    decide the same thing for a call of your own. Resolve the future inside the
    ``try``: submitting is non-blocking, so the operation's own failure surfaces
    where the result is awaited, not where it was queued::

        future = training_client.forward_backward(batch, loss_fn="cross_entropy")
        try:
            result = future.result()
        except Exception as error:
            if not is_retryable(error):
                raise
            ...

    Args:
        error: The exception raised by the failed call.

    Returns:
        ``True`` if the failure is transient, ``False`` otherwise.
    """
    if isinstance(error, (BotoConnectionError, HTTPClientError)):
        # No response, so no status code: the request may never have arrived.
        return True
    if isinstance(error, ClientError):
        response = error.response or {}
        code = (response.get("Error") or {}).get("Code") or ""
        status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
        return code in _RETRYABLE_ERROR_CODES or (
            isinstance(status, int) and status in _RETRYABLE_STATUS_CODES
        )
    return False


class TrainingSessionError(Exception):
    """Base exception for training session errors.

    Every error class in this module derives from this one, and it is also raised
    directly where the message says all there is to say: asking a closed client
    for another operation, for instance. A wait that runs out of time raises the
    built-in ``TimeoutError`` instead, so bound waits are caught the same way as
    with any other future.
    """


class OperationError(TrainingSessionError):
    """Raised when an asynchronous operation completed in a failed state.

    The operation was accepted, so nothing was raised when it was submitted; it
    then failed while running, and that surfaces here when the future for it is
    resolved. A request the service rejects outright raises ``ClientError``
    instead, at submit time.

    ``operation_id`` identifies the failed operation server-side and is the one
    detail to quote when asking for help with it.
    """

    def __init__(
        self,
        operation_type: str,
        message: str,
        *,
        operation_id: Optional[str] = None,
    ):
        self.operation_type = operation_type
        self.operation_id = operation_id
        self.message = message
        subject = f"{operation_type} operation {operation_id}" if operation_id else operation_type
        super().__init__(f"{subject} failed: {message}")

    def __reduce__(self) -> tuple:
        """Rebuild through ``__init__`` so the error survives pickling.

        A training loop that spreads work across processes moves exceptions
        between them. Without this the default rebuild calls ``__init__`` with
        the single formatted message and raises ``TypeError`` about a missing
        argument, replacing the real failure with an unrelated one.
        """
        return (
            _rebuild_operation_error,
            (type(self), self.operation_type, self.message, self.operation_id),
        )


def _rebuild_operation_error(
    cls: type, operation_type: str, message: str, operation_id: Optional[str]
) -> "OperationError":
    """Reconstruct an :class:`OperationError`; used only by pickling.

    Takes the class so a subclass round-trips as itself, the way the two
    siblings do by rebuilding through ``self.__class__``.
    """
    return cls(operation_type, message, operation_id=operation_id)


class ResultMismatchError(TrainingSessionError):
    """Raised when a completed operation's result does not match its request.

    A per-token result must be as long as the sequence sent. A different length
    means the result belongs to some other request, so it is raised instead of
    returned: the values would otherwise be read as this request's own.
    """

    def __init__(self, operation_type: str, expected: int, received: int):
        self.operation_type = operation_type
        self.expected = expected
        self.received = received
        super().__init__(
            f"{operation_type} returned {received} values for a {expected}-token "
            "sequence. The result does not belong to this request. Issuing these "
            "operations one at a time avoids it."
        )

    def __reduce__(self) -> tuple:
        """Rebuild through ``__init__``; see :meth:`OperationError.__reduce__`."""
        return (self.__class__, (self.operation_type, self.expected, self.received))


class ArtifactDeletionError(TrainingSessionError):
    """Raised when deleting a model package's S3 artifacts did not fully succeed.

    The model package is left in place, so it still points at the objects listed
    on ``failed_keys``. Re-run :meth:`ServiceClient.delete_model_package` once
    the cause is addressed (most often a missing S3 permission). The message
    truncates ``failed_keys``; the attribute is complete.
    """

    #: Keys listed in the message before it truncates.
    _KEYS_IN_MESSAGE = 5

    def __init__(self, model_package_arn: str, s3_uri: str, failed_keys: List[str]):
        self.model_package_arn = model_package_arn
        self.s3_uri = s3_uri
        self.failed_keys = failed_keys
        shown = ", ".join(failed_keys[: self._KEYS_IN_MESSAGE])
        if len(failed_keys) > self._KEYS_IN_MESSAGE:
            shown += f", ... ({len(failed_keys)} total, see .failed_keys)"
        super().__init__(
            f"Could not delete every object under {s3_uri} for model package "
            f"{model_package_arn}: {shown}. The model package was NOT deleted, so "
            "it still points at these objects; retry once the cause is "
            "addressed (commonly s3:DeleteObject on that location)."
        )

    def __reduce__(self) -> tuple:
        """Rebuild through ``__init__``; see :meth:`OperationError.__reduce__`."""
        return (
            self.__class__,
            (self.model_package_arn, self.s3_uri, self.failed_keys),
        )
