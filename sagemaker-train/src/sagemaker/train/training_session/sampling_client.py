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
"""SamplingClient: sample from a saved checkpoint (or base model).

Created via :meth:`TrainingClient.create_sampling_client`,
:meth:`TrainingClient.save_weights_and_get_sampling_client`, or
:meth:`ServiceClient.create_sampling_client` (sampling without training), not
directly by callers. Only the last of those creates the backing session, and so
only it stops that session when used as a context manager; the other two share
the TrainingClient's session handle, which is what marks them non-owning.
Turns a prompt + :class:`SamplingParams` into a sampling operation and returns
an :class:`APIFuture` over the
:class:`SampleResponse`. Submits are non-blocking: the wire call runs on a
background submitter thread, and sample ops are order-independent, so many
requests can be queued concurrently in seconds.

``model_package_arn`` selects which checkpoint to sample from; ``""`` means the
base model. It is sent on the wire as ``modelPackageArn``, or omitted entirely
for the base model.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar

from sagemaker.train.training_session._utils import (
    model_input_token_count,
    prompt_to_model_input,
)
from sagemaker.train.training_session.exceptions import (
    ResultMismatchError,
    TrainingSessionError,
)
from sagemaker.train.training_session._session_handle import _SessionHandle
from sagemaker.train.training_session._submitter import (
    _CLOSED_MESSAGE,
    BackgroundSubmitter,
    run_blocking,
)
from sagemaker.train.training_session.futures import (
    APIFuture,
    PendingAPIFuture,
    call_with_transient_retry,
)
from sagemaker.train.training_session.types import (
    SampleResponse,
    SamplingParams,
)

#: Sampling from weights that were just saved can briefly return a transient
#: error while those weights are still being made servable. Sampling carries no
#: clientToken and is side-effect-free, so the submit is safe to retry.
#: Budget = 150 attempts over the backoff, about 5 min of warmup tolerance.
_SAMPLE_SUBMIT_MAX_ATTEMPTS = 150

#: Result type of a submitted sampling op, fixed by the ``parse_fn`` passed to
#: :meth:`SamplingClient._submit_sample_op`.
_T = TypeVar("_T")

logger = logging.getLogger(__name__)


def _parse_sample_result(raw: Dict[str, Any]) -> SampleResponse:
    """Extract a :class:`SampleResponse` from a sampling-operation result body.

    The payload is returned under ``operationResult.sample``.
    """
    operation_result = raw.get("operationResult") or {}
    return SampleResponse._from_api(operation_result.get("sample") or {})


def _make_compute_logprobs_parser(
    expected_tokens: Optional[int],
) -> Callable[[Dict[str, Any]], List[Optional[float]]]:
    """Build the parser for a compute-logprobs result of a known token count.

    The payload is returned under ``operationResult.computeLogprobs`` (the
    compute-logprobs result variant). Returns the bare logprob list, matching
    what interactive-training loops expect from ``compute_logprobs``, rather
    than a wrapper type.

    The count is bound here so the parser can reject a result belonging to a
    different request; see :class:`ResultMismatchError`. ``None`` (a prompt the
    service tokenizes) leaves the result unchecked.
    """

    def parse(raw: Dict[str, Any]) -> List[Optional[float]]:
        operation_result = raw.get("operationResult") or {}
        logprobs = (operation_result.get("computeLogprobs") or {}).get("logprobs", [])
        if expected_tokens is not None and len(logprobs) != expected_tokens:
            raise ResultMismatchError("computeLogprobs", expected_tokens, len(logprobs))
        return logprobs

    return parse


class SamplingClient:
    """Samples from a checkpoint; ``model_package_arn=""`` = base model.

    Args:
        training_session_name: Name of the backing TrainingSession resource.
        runtime: Client for the training-session runtime.
        model_package_arn: Model package ARN of the checkpoint to sample from;
            ``""`` = base model.
        base_model: Base model (HF model ID or SageMaker Hub ARN), used to
            resolve a tokenizer via :meth:`get_tokenizer`.
        tokenizer: Optional pre-loaded tokenizer for string prompts.
        hub_client: Optional SageMaker Hub client, used to look up the
            HuggingFace model id behind a Hub ARN when a tokenizer is needed.
            Only consulted when neither ``tokenizer`` nor ``hf_model_id`` is set,
            and borrowed from ``service_client`` at that point if omitted here.
        hf_model_id: Optional HuggingFace model id to load the tokenizer from.
            Set it when ``base_model`` is an ARN whose id cannot be looked up, or
            to pin the tokenizer to a specific repo.
        service_client: The :class:`ServiceClient` that created this client,
            used to stop the session when this client owns it.
        session_handle: Existing :class:`_SessionHandle` to share, passed by
            :meth:`TrainingClient.create_sampling_client` so this client and its
            TrainingClient refcount one session. When omitted a fresh handle is
            created for ``training_session_name``.
        submitter: Existing :class:`BackgroundSubmitter` to share, passed by
            :meth:`TrainingClient.create_sampling_client` so refreshing a sampler
            does not add an event loop and thread each time. When omitted this
            client builds and owns its own.

    Whether this client *owns* its session is read off ``session_handle`` rather
    than passed separately: sharing a handle means a TrainingClient is driving
    the session, so this client must not stop it; a fresh handle means the
    session was created implicitly for this client alone and has no other owner.
    Ownership governs :meth:`stop` and ``__exit__`` only; the automatic stop is
    the handle's job either way. Submitter ownership is read the same way but
    tracked separately: a client can share a handle and still own its submitter.
    """

    def __init__(
        self,
        training_session_name: str,
        runtime: Any,
        *,
        model_package_arn: str = "",
        base_model: str = "",
        tokenizer: Any = None,
        hub_client: Any = None,
        hf_model_id: Optional[str] = None,
        service_client: Any = None,
        session_handle: Optional[_SessionHandle] = None,
        submitter: Optional[BackgroundSubmitter] = None,
    ) -> None:
        self.training_session_name = training_session_name
        self.model_package_arn = model_package_arn
        self._runtime = runtime
        self._base_model = base_model
        self._tokenizer = tokenizer
        self._hub_client = hub_client
        self._hf_model_id = hf_model_id
        self._service_client = service_client
        # Ownership IS "did I have to make my own handle?": a shared handle
        # means a TrainingClient is driving the same session, so the session
        # outlives whichever of the two is collected first and is stopped when
        # the last one goes.
        self._owns_session = session_handle is None
        # Sample ops are order-independent: submits run concurrently on the
        # background submitter and return their future immediately. A submitter
        # is an event loop on a daemon thread, so sharing one keeps a training
        # loop's sampler refresh down to a plain object allocation.
        self._owns_submitter = submitter is None
        self._submitter = submitter or BackgroundSubmitter(
            name=f"sagemaker-sampling-submit-{training_session_name}"
        )
        self._handle = session_handle or _SessionHandle(training_session_name, service_client)
        if self._owns_submitter:
            # A shared submitter is already registered by its owner, and the
            # handle keeps every submitter it is given, so registering again
            # would pile up references to the one submitter and drain it once
            # per entry.
            self._handle.register_submitter(self._submitter)
        # The TrainingClient keeps submitting on a shared submitter after this
        # client is done with it, so retirement is tracked here and not on the
        # submitter.
        self._closed = False

    def get_tokenizer(self) -> Any:
        """Return a HuggingFace tokenizer for the session's base model.

        The runtime accepts only token IDs, so callers with text must tokenize
        client-side before :meth:`sample`. The tokenizer is loaded once and
        cached (and reused for string prompts passed to :meth:`sample`).

        Returns:
            A ``transformers.PreTrainedTokenizer``.

        Raises:
            ValueError: If the base model has no tokenizer available (e.g. a
                model whose tokenizer is not published to HuggingFace); pass
                pre-tokenized token IDs in that case.
            ImportError: If ``transformers`` is not installed (install the
                optional extra: ``pip install sagemaker-train[tokenizer]``).
        """
        if self._tokenizer is None:
            from sagemaker.train.training_session._utils import (
                load_tokenizer,
            )

            # The Hub client is only needed to resolve the tokenizer id when no
            # hf_model_id was supplied; skip building it otherwise.
            hub_client = self._hub_client if self._hf_model_id else self._get_hub_client()
            self._tokenizer = load_tokenizer(self._base_model, hub_client, self._hf_model_id)
        return self._tokenizer

    def _get_hub_client(self) -> Any:
        """Return the Hub client, building it lazily via the service client.

        Usually ``None`` at construction and built here on first use, only when
        a base-model ARN's tokenizer id must be resolved and no ``hf_model_id``
        was supplied. Deferring it keeps sampling by token IDs working where
        sagemaker-core is not installed.
        """
        if self._hub_client is None and self._service_client is not None:
            self._hub_client = self._service_client._get_hub_client()
        return self._hub_client

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        prompt: Any,
        num_samples: int = 1,
        sampling_params: "Optional[SamplingParams | Dict[str, Any] | Any]" = None,
        *,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: Optional[int] = None,
    ) -> APIFuture[SampleResponse]:
        """Submit a sampling operation; returns an :class:`APIFuture`.

        Args:
            prompt: A list of token IDs, a string (requires a tokenizer), or a
                pre-built ModelInput dict.
            num_samples: Number of sequences to generate.
            sampling_params: Generation parameters. Uses defaults if omitted.
                Accepts a :class:`SamplingParams`, a dict of the same fields,
                or any object carrying at least one of them, so an existing
                config object can be passed as-is.
            include_prompt_logprobs: Request per-token logprobs over the prompt.
            topk_prompt_logprobs: If set, request the top-k logprobs per prompt
                position.

        Returns:
            An :class:`APIFuture` resolving to a :class:`SampleResponse`.
            Call ``.result()`` on it (or ``await`` it) to get the response. The
            submit itself runs on a background thread: this returns immediately, many
            requests can be queued concurrently, and a submit error surfaces
            when the future is resolved, not here.
        """
        # from_any: accept SamplingParams, a dict, or any object with the same attributes.
        params = (
            SamplingParams.from_any(sampling_params)
            if sampling_params is not None
            else SamplingParams()
        )
        model_input = prompt_to_model_input(
            prompt, tokenizer=self._tokenizer, tokenizer_loader=self.get_tokenizer
        )

        # The sampling operation takes an operationConfig union; we send the
        # ``sample`` variant.
        sample_config: dict[str, Any] = {
            "prompt": model_input,
            "numSamples": num_samples,
            "samplingParams": params._to_api(),
        }
        # Select which checkpoint to sample from; omit the field to sample the
        # base model.
        if self.model_package_arn:
            sample_config["modelPackageArn"] = self.model_package_arn
        if include_prompt_logprobs:
            sample_config["includePromptLogprobs"] = True
        if topk_prompt_logprobs is not None:
            sample_config["topkPromptLogprobs"] = topk_prompt_logprobs

        return self._submit_sample_op({"sample": sample_config}, _parse_sample_result)

    async def sample_async(
        self,
        prompt: Any,
        num_samples: int = 1,
        sampling_params: "Optional[SamplingParams | Dict[str, Any] | Any]" = None,
        *,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: Optional[int] = None,
    ) -> SampleResponse:
        """Coroutine twin of :meth:`sample`; awaits the response directly.

        ``resp = await sc.sample_async(...)`` resolves to the
        :class:`SampleResponse`. Sample ops are order-independent, so unlike
        the training ``*_async`` methods there is no sequence position to hold
        and therefore no intermediate future to return.
        """
        return await self.sample(
            prompt,
            num_samples=num_samples,
            sampling_params=sampling_params,
            include_prompt_logprobs=include_prompt_logprobs,
            topk_prompt_logprobs=topk_prompt_logprobs,
        )

    def compute_logprobs(self, prompt: Any) -> APIFuture[List[Optional[float]]]:
        """Submit a compute-logprobs operation; returns an :class:`APIFuture`.

        Unlike :meth:`sample`, this generates no new tokens; it scores the given
        prompt and returns one log probability per prompt token. Useful for
        evaluation, reranking, or importance weighting. Non-blocking, like
        :meth:`sample`.

        Args:
            prompt: A list of token IDs, a string (requires a tokenizer), or a
                pre-built ModelInput dict.

        Returns:
            An :class:`APIFuture` resolving to a list with one logprob per
            prompt token (the first entry is ``None``, since it has no preceding context).

        Raises:
            ResultMismatchError: On resolution, when the result's length does
                not match the sequence sent.
        """
        model_input = prompt_to_model_input(
            prompt, tokenizer=self._tokenizer, tokenizer_loader=self.get_tokenizer
        )

        # The sampling operation's operationConfig union, computeLogprobs
        # variant. Select the checkpoint the same way as sample: omit
        # modelPackageArn to score against the base model.
        logprobs_config: dict[str, Any] = {"prompt": model_input}
        if self.model_package_arn:
            logprobs_config["modelPackageArn"] = self.model_package_arn
        return self._submit_sample_op(
            {"computeLogprobs": logprobs_config},
            _make_compute_logprobs_parser(model_input_token_count(model_input)),
        )

    async def compute_logprobs_async(self, prompt: Any) -> List[Optional[float]]:
        """Coroutine twin of :meth:`compute_logprobs`; awaits the logprobs directly."""
        return await self.compute_logprobs(prompt)

    def _submit_sample_op(
        self,
        operation_config: Dict[str, Any],
        parse_fn: Callable[[Dict[str, Any]], _T],
    ) -> APIFuture[_T]:
        """Queue a sample-path op; immediately return a future over its result.

        The sampling operation takes no clientToken, unlike the training op:
        sampling is side-effect-free, so the service does not dedupe it. The
        submit retries through transient unavailability: right after
        save_weights_for_sampler the service may still be binding to the new
        checkpoint. Safe to retry: no clientToken, side-effect-free.
        """
        if self._closed:
            raise TrainingSessionError(_CLOSED_MESSAGE)

        def _wire_submit() -> APIFuture[_T]:
            resp = call_with_transient_retry(
                lambda: self._runtime.execute_sampling_operation(
                    trainingSessionName=self.training_session_name,
                    operationConfig=operation_config,
                ),
                max_attempts=_SAMPLE_SUBMIT_MAX_ATTEMPTS,
            )
            operation_id = resp["operationId"]
            return APIFuture(
                operation_id=operation_id,
                poll_fn=lambda: self._runtime.get_sampling_operation_result(
                    trainingSessionName=self.training_session_name,
                    operationId=operation_id,
                ),
                parse_fn=parse_fn,
                status_key="status",
                operation_type="sample",
            )

        return PendingAPIFuture(
            self._submitter.submit_unordered(run_blocking(_wire_submit)),
            operation_type="sample",
            keepalive=self._handle,
        )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _retire(self) -> None:
        """Refuse further work, and release the background submitter if it is ours.

        A shared submitter belongs to the :class:`TrainingClient` that handed
        this client out, so closing it would stop that client submitting and is
        left to its teardown. This client still stops accepting operations, so
        retirement fails fast here either way.
        """
        self._closed = True
        if self._owns_submitter:
            self._submitter.close()

    def stop(self) -> None:
        """Stop this client's training session.

        Only meaningful for a client created by
        :meth:`ServiceClient.create_sampling_client`, which creates a session
        implicitly and therefore owns it. For a client created by
        :meth:`TrainingClient.create_sampling_client` the session belongs to the
        TrainingClient, so this retires only this client; stop that session
        through its :class:`TrainingClient` instead.

        Drains the background submitter first (best effort) when this client owns
        it, so every operation already queued reaches the service before the stop
        request. The client accepts no new operations afterwards.
        """
        self._retire()
        if not self._owns_session:
            # Shared session: the handle belongs to the TrainingClient driving
            # it, so do NOT disarm: that would cancel the automatic stop for a
            # session this client has no say over, turning a no-op into a leak.
            return
        if self._service_client is None:
            # Nothing was stopped, so leave the automatic stop armed.
            logger.warning(
                "SamplingClient owns session %r but has no ServiceClient to "
                "stop it with; stop it via ServiceClient.stop_session(...).",
                self.training_session_name,
            )
            return
        try:
            self._service_client.stop_session(self.training_session_name)
        except Exception:
            # Keep the automatic stop armed: it is the only remaining
            # client-side attempt at releasing the session.
            self._handle.note_explicit_stop_failed()
            raise
        # Stopped, so a later garbage collection would only repeat the call.
        self._handle.disarm()

    def __enter__(self) -> "SamplingClient":
        """Return this client, so it can be used as a context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop the session on leaving the block, if this client created it."""
        # Drain queued submits so in-flight requests reach the service, then
        # stop the session iff this client created it. A client handed out by
        # TrainingClient.create_sampling_client shares the TrainingClient's
        # session and must not stop it, since that session outlives this block.
        if self._owns_session:
            try:
                self.stop()
            except Exception:
                logger.warning(
                    "Failed to stop implicitly-created session %r on exit; it "
                    "will be stopped when this client is collected, or after "
                    "its inactivity timeout.",
                    self.training_session_name,
                    exc_info=True,
                )
        else:
            self._retire()
