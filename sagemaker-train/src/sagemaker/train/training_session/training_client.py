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
"""TrainingClient: drives the training loop over a TrainingSession resource.

Owns the monotonic ``sequenceNumber`` for the session and turns each primitive
(``forward_backward`` / ``optim_step`` / ``save_weights_for_sampler`` / ``save_state`` /
``load_state``) into an ``ExecuteTrainingOperation`` call, returning a unified
:class:`APIFuture`. Submits are non-blocking: each method stamps its sequence
slot at call time, queues the wire call on a background submitter thread, and
returns its future immediately, so a training loop can queue a large batch of
operations in seconds and resolve results later. Operations can be pipelined:
submit ``optim_step`` before the preceding ``forward_backward`` future
resolves; the sequence numbers stamped at call time preserve order on the
service side.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from sagemaker.train.training_session.exceptions import TrainingSessionError
from sagemaker.train.training_session._utils import (
    batch_to_datum_list,
    chunk_datum_list,
    combine_forward_backward_results,
    _warn_unsupported_args,
    _RETRY_CONFIG_CONSEQUENCE,
)
from sagemaker.train.training_session._session_handle import _SessionHandle
from sagemaker.train.training_session._submitter import (
    BackgroundSubmitter,
    run_blocking,
)
from sagemaker.train.training_session.futures import (
    APIFuture,
    CombinedFuture,
    PendingAPIFuture,
    call_with_transient_retry,
)
from sagemaker.train.training_session.sampling_client import (
    SamplingClient,
    _SAMPLE_SUBMIT_MAX_ATTEMPTS,
)
from sagemaker.train.training_session.types import (
    AdamParams,
    ForwardBackwardOutput,
    LoadStateResponse,
    LossFunction,
    ModelPackageInfo,
    ModelPackageKind,
    OptimStepResponse,
    SaveWeightsResponse,
    SaveStateResponse,
    _OperationResult,
)

logger = logging.getLogger(__name__)

#: The first training op after the session flips to ``Running`` can briefly hit
#: transient unavailability while the runtime finishes warming up (the session
#: can report Running just before the runtime path accepts traffic). The submit
#: is idempotent (stable clientToken + sequenceNumber), so it is safe to retry
#: through the warmup window. Budget = 150 attempts over the backoff, about 5 min.
_TRAINING_SUBMIT_MAX_ATTEMPTS = 150

#: Consequence clause for the checkpoint-lifetime knobs the save ops accept but
#: do not honour. Named so ``save_state`` and ``save_weights_for_sampler`` cannot
#: drift into telling the caller two different things about the same behaviour.
_SAVE_TTL_CONSEQUENCE = "the checkpoint was saved under the platform's retention policy instead"

#: Consequence clause for ``name``. Points at the parameter that *does* carry a
#: caller-supplied label, so the warning is actionable, not just a refusal.
_SAVE_NAME_CONSEQUENCE = (
    "a checkpoint is identified by its model package ARN, not a caller-chosen "
    "name; pass description=... for a human-readable label"
)

#: Consequence clause for ``overwrite``. Distinct from the retention clause: this
#: knob is about versioning, and saying "retention policy" here would answer a
#: question the caller did not ask.
_SAVE_OVERWRITE_CONSEQUENCE = (
    "the save registered a new model package version instead, leaving earlier " "versions in place"
)

#: Result type of a submitted operation, fixed by the ``result_cls`` passed to
#: :meth:`TrainingClient._submit`. Bounded by the base that declares
#: ``_from_api``, which is what ``_submit`` calls to build the result.
_T = TypeVar("_T", bound=_OperationResult)


class TrainingClient:
    """Drives forward_backward / optim_step / save_weights_for_sampler over a session.

    Instances are created by :meth:`ServiceClient.create_lora_training_client`
    and its ``*_from_state`` variants, not directly by callers.

    Args:
        training_session_name: Name of the backing TrainingSession resource.
        runtime: Client for the training-session runtime.
        base_model: Base model (HF model ID or SageMaker Hub ARN), used to
            resolve a tokenizer via :meth:`get_tokenizer`.
        tokenizer: Optional pre-loaded tokenizer for string batches.
        hub_client: Optional SageMaker Hub client, used to look up the
            HuggingFace model id behind a Hub ARN when a tokenizer is needed.
            Only consulted when neither ``tokenizer`` nor ``hf_model_id`` is set.
        hf_model_id: Optional HuggingFace model id to load the tokenizer from.
            Set it when ``base_model`` is an ARN whose id cannot be looked up, or
            to pin the tokenizer to a specific repo.
        model_package_group_arn: The session's model package group, where
            save_state / save_weights_for_sampler register checkpoints. Lets
            :meth:`list_model_packages` resolve the group without the caller
            re-supplying it.
        service_client: The :class:`ServiceClient` that created this client,
            used to delegate checkpoint listing and to stop the session.

    The backing session is stopped once the last client referencing it becomes
    unreachable, including any :class:`SamplingClient` derived from this one,
    which shares this client's session handle. The automatic teardown *stops* the
    session, leaving it describable for later inspection; deleting one stays an
    explicit :meth:`delete` call.

    Releasing the session is handled for you, so a caller can focus on the
    training loop. Anything worth keeping goes in a checkpoint: model packages
    are durable on their own, so trained weights outlive both the session and
    the process, and a new session can be created from one at any time.
    """

    def __init__(
        self,
        training_session_name: str,
        runtime: Any,
        *,
        base_model: str = "",
        tokenizer: Any = None,
        hub_client: Any = None,
        hf_model_id: Optional[str] = None,
        model_package_group_arn: str = "",
        service_client: Any = None,
    ) -> None:
        self.training_session_name = training_session_name
        self._runtime = runtime
        self._base_model = base_model
        self._tokenizer = tokenizer
        self._hub_client = hub_client
        self._hf_model_id = hf_model_id
        self._sequence_counter = 0
        # Reentrant: a chunked forward-backward holds this across its whole chunk
        # loop so the chunks get consecutive sequence numbers, and each chunk's
        # submit takes it again to stamp its own.
        self._sequence_lock = threading.RLock()
        # Base methods queue their ExecuteTrainingOperation round trip here and
        # return a future immediately; ordered submits keep wire order equal to
        # the sequence numbers stamped at call time.
        self._submitter = BackgroundSubmitter(
            name=f"sagemaker-training-submit-{training_session_name}"
        )
        self._model_package_group_arn = model_package_group_arn
        self._service_client = service_client
        # Latest model package this client saved, per kind. Lets
        # get_latest_model_package answer without an API round-trip, and covers the
        # window where a just-saved package is not yet describable. Cache only: a
        # fresh client starts empty and does not imply a package it never saved.
        self._latest_by_kind: Dict[ModelPackageKind, ModelPackageInfo] = {}
        self._cache_lock = threading.Lock()
        # This session's lifetime, shared with every SamplingClient derived from
        # this client. The handle, not this client, is what CPython collects
        # last, so the session is stopped when the last of them goes away and
        # never while one is still driving it.
        self._handle = _SessionHandle(training_session_name, service_client)
        self._handle.register_submitter(self._submitter)

    def get_tokenizer(self) -> Any:
        """Return a HuggingFace tokenizer for the session's base model.

        The runtime accepts only token IDs, so callers with text must tokenize
        client-side: ``tok = tc.get_tokenizer(); ids = tok.encode("hello")``.
        The tokenizer is loaded once and cached (and reused for string batches
        passed to :meth:`forward_backward`).

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
        was supplied.
        """
        if self._hub_client is None and self._service_client is not None:
            self._hub_client = self._service_client._get_hub_client()
        return self._hub_client

    # ------------------------------------------------------------------
    # Training primitives
    # ------------------------------------------------------------------

    # Each training op comes in two forms: a sync method that queues the submit
    # and returns an :class:`APIFuture` immediately, and an ``*_async`` twin,
    # a coroutine for async code, where ``fut = await tc.op_async(...)``
    # completes the enqueue and ``await fut`` yields the result. The sync
    # methods do the work; the ``*_async`` twins are thin coroutine wrappers.

    def forward_backward(
        self,
        batch: Any,
        loss_fn: "LossFunction | str" = "cross_entropy",
        loss_fn_inputs: Optional[Dict[str, Any]] = None,
        loss_fn_config: Optional[Dict[str, float]] = None,
    ) -> APIFuture[ForwardBackwardOutput]:
        """Submit a forward-backward pass; returns an :class:`APIFuture`.

        Non-blocking: the submit runs on a background thread, so this returns
        immediately. Resolve the future with ``.result()`` (or ``await`` it)
        for the :class:`ForwardBackwardOutput`; a submit error surfaces there,
        not at the call site. Ops pipeline naturally: submit the next op
        before this future resolves; the sequence numbers stamped at call time
        preserve ordering on the service.

        Args:
            batch: Training data: a list of token ID lists, a single token ID
                list, a list of strings (requires a tokenizer), or raw datum
                dicts (advanced).
            loss_fn: Loss function name or :class:`LossFunction` enum.
            loss_fn_inputs: Loss inputs applied to each datum that does not
                already carry its own. Which keys ``loss_fn`` requires is
                checked before the request leaves.
            loss_fn_config: Optional loss function configuration (e.g.
                ``{"clip_low_threshold": 0.2, "clip_high_threshold": 5.0}``
                for CISPO/PPO clipping).

        Raises:
            ValueError: A per-token ``lossFnInputs`` entry is not the same length
                as its datum's ``modelInput``. Raised before the request is sent,
                naming the datum and the key.

        A batch too large to send in one request is split into chunks (bounded
        by datum count and estimated size) submitted as consecutive
        ``forwardBackward`` ops. The service accumulates gradients across
        consecutive forward-backward ops until the next ``optim_step``, so this
        is equivalent to sending the whole batch at once. When the batch fits in
        one chunk (the common case) the request is unchanged. The returned
        future resolves to a single combined :class:`ForwardBackwardOutput`.
        """
        loss_fn = LossFunction(loss_fn)
        datum_list = batch_to_datum_list(
            batch,
            tokenizer=self._tokenizer,
            loss_fn_inputs=loss_fn_inputs,
            tokenizer_loader=self.get_tokenizer,
            loss_fn_name=loss_fn.value,
        )
        chunks = chunk_datum_list(datum_list)
        # 0 chunks (empty batch) or 1 chunk: submit exactly one op.
        if len(chunks) <= 1:
            return self._submit_forward_backward(
                loss_fn.value,
                chunks[0] if chunks else datum_list,
                loss_fn_config=loss_fn_config,
            )
        # Multiple chunks. Gradients accumulate across consecutive
        # forward_backward ops, so the chunk ops must carry contiguous, in-order
        # sequence numbers: the whole loop holds the sequence lock, not just each
        # submit. Without that, an optim_step from another thread can take a
        # number in the middle of the batch and apply a partial gradient.
        with self._sequence_lock:
            futures = [
                self._submit_forward_backward(loss_fn.value, chunk, loss_fn_config=loss_fn_config)
                for chunk in chunks
            ]
        # The chunk sizes are known only here, and the combined loss is a
        # datum-count-weighted mean over them.
        return CombinedFuture(
            futures,
            functools.partial(
                combine_forward_backward_results,
                datum_counts=[len(chunk) for chunk in chunks],
            ),
        )

    async def forward_backward_async(
        self,
        batch: Any,
        loss_fn: "LossFunction | str" = "cross_entropy",
        loss_fn_inputs: Optional[Dict[str, Any]] = None,
        loss_fn_config: Optional[Dict[str, float]] = None,
    ) -> APIFuture[ForwardBackwardOutput]:
        """Coroutine twin of :meth:`forward_backward` for async code.

        ``fut = await tc.forward_backward_async(...)`` completes the enqueue, so
        the op is sequenced before the next line runs, and ``await fut``
        yields the :class:`ForwardBackwardOutput`.
        """
        return self.forward_backward(batch, loss_fn, loss_fn_inputs, loss_fn_config=loss_fn_config)

    def _submit_forward_backward(
        self,
        loss_fn_name: str,
        datum_list: List[Dict[str, Any]],
        loss_fn_config: Optional[Dict[str, float]] = None,
    ) -> APIFuture[ForwardBackwardOutput]:
        """Submit one forwardBackward op for an already-built datum list."""
        fb_config: Dict[str, Any] = {
            "lossFnName": loss_fn_name,
            "datumList": datum_list,
        }
        if loss_fn_config is not None:
            fb_config["lossFnConfig"] = loss_fn_config
        op_config = {"forwardBackward": fb_config}
        return self._submit(op_config, ForwardBackwardOutput)

    def optim_step(
        self, adam_params: "Optional[AdamParams | Dict[str, Any] | Any]" = None
    ) -> APIFuture[OptimStepResponse]:
        """Submit an optimizer step; returns an :class:`APIFuture`.

        Non-blocking; the future resolves to an :class:`OptimStepResponse`
        with ``metrics``. Pipelines with :meth:`forward_backward`; the
        sequence numbers stamped at call time preserve ordering on the service.

        Args:
            adam_params: Optimizer parameters. Uses defaults if omitted.
                Accepts an :class:`AdamParams`, a dict of the same fields, or
                any object carrying at least one of them, so an existing
                optimizer config object can be passed as-is.
        """
        params = AdamParams.from_any(adam_params) if adam_params is not None else AdamParams()
        op_config = {"optimizerStep": {"adamParamsConfig": params._to_api()}}
        return self._submit(op_config, OptimStepResponse)

    async def optim_step_async(
        self, adam_params: "Optional[AdamParams | Dict[str, Any] | Any]" = None
    ) -> APIFuture[OptimStepResponse]:
        """Coroutine twin of :meth:`optim_step` for async code.

        ``fut = await tc.optim_step_async(...)`` completes the enqueue,
        guaranteeing the step is sequenced after any forward-backward already
        submitted, and ``await fut`` yields the :class:`OptimStepResponse`.
        """
        return self.optim_step(adam_params)

    def save_weights_for_sampler(
        self,
        description: str = "",
        merge_weights: bool = False,
        *,
        ttl_seconds: Optional[int] = None,
        name: Optional[str] = None,
    ) -> APIFuture[SaveWeightsResponse]:
        """Submit a save-weights op; returns an :class:`APIFuture`.

        Non-blocking; the future resolves to a
        :class:`SaveWeightsResponse` carrying ``model_package_arn``.
        The op registers a model package version in the session's model package
        group.

        When ``merge_weights=True``, the service merges the LoRA adapter into the
        base model and writes full, adapter-free weights (``checkpoints/hf_merged/``)
        alongside the adapter. Use this when the checkpoint will be deployed as a
        single standalone model, e.g. on GPUs without a working LoRA serving path.

        Args:
            description: Optional human-readable description stored on the
                model package (e.g. ``"weights at step 100"``).
            merge_weights: Whether to also write full adapter-free weights for
                standalone deployment.
            ttl_seconds: Accepted but not used; setting it logs a warning.
                Checkpoint lifetime is governed by the platform's retention
                policy, not per-save. Delete the model package version when
                you no longer need it.
            name: Accepted but not used; setting it logs a warning.
                Checkpoints are model package *versions* addressed by ARN, so
                there is no caller-chosen name to set. Use ``description``.
        """
        _warn_unsupported_args(_SAVE_TTL_CONSEQUENCE, ttl_seconds=ttl_seconds)
        _warn_unsupported_args(_SAVE_NAME_CONSEQUENCE, name=name)
        save_weights_config: Dict[str, Any] = {}
        if merge_weights:
            save_weights_config["mergeWeights"] = True
        if description:
            save_weights_config["description"] = description
        op_config = {"saveWeights": save_weights_config}
        return self._submit(
            op_config,
            SaveWeightsResponse,
            record_fn=self._cache_on_save(ModelPackageKind.MODEL),
        )

    async def save_weights_for_sampler_async(
        self,
        description: str = "",
        merge_weights: bool = False,
        *,
        ttl_seconds: Optional[int] = None,
        name: Optional[str] = None,
    ) -> APIFuture[SaveWeightsResponse]:
        """Coroutine twin of :meth:`save_weights_for_sampler` for async code."""
        return self.save_weights_for_sampler(
            description=description,
            merge_weights=merge_weights,
            ttl_seconds=ttl_seconds,
            name=name,
        )

    def save_state(
        self,
        description: str = "",
        *,
        ttl_seconds: Optional[int] = None,
        overwrite: bool = False,
        name: Optional[str] = None,
    ) -> APIFuture[SaveStateResponse]:
        """Submit a save of full training state (weights + optimizer).

        Non-blocking; the future resolves to a :class:`SaveStateResponse`
        carrying ``model_package_arn``, always a model package version that
        :meth:`load_state` can resume from.

        Args:
            description: Optional human-readable description stored on the
                model package (e.g. ``"step 100 checkpoint"``).
            ttl_seconds: Accepted but not used; setting it logs a warning.
                Checkpoint lifetime is governed by the platform's retention
                policy, not per-save. Delete the model package version when
                you no longer need it.
            overwrite: Accepted but not used; setting it logs a warning.
                Every save registers a new model package *version*, so a save
                never overwrites an earlier one, which is what ``False`` (the
                default) asks for, so it does not warn. ``True`` warns and is
                ignored.
            name: Accepted but not used; setting it logs a warning.
                Checkpoints are model package *versions* addressed by ARN, so
                there is no caller-chosen name to set. Use ``description``.
        """
        _warn_unsupported_args(_SAVE_TTL_CONSEQUENCE, ttl_seconds=ttl_seconds)
        # None, not False: the default already describes what happens, so only an
        # explicit True is a request that goes unhonoured.
        _warn_unsupported_args(_SAVE_OVERWRITE_CONSEQUENCE, overwrite=True if overwrite else None)
        _warn_unsupported_args(_SAVE_NAME_CONSEQUENCE, name=name)
        save_state_config: Dict[str, Any] = {}
        if description:
            save_state_config["description"] = description
        return self._submit(
            {"saveState": save_state_config},
            SaveStateResponse,
            record_fn=self._cache_on_save(ModelPackageKind.CHECKPOINT),
        )

    async def save_state_async(
        self,
        description: str = "",
        *,
        ttl_seconds: Optional[int] = None,
        overwrite: bool = False,
        name: Optional[str] = None,
    ) -> APIFuture[SaveStateResponse]:
        """Coroutine twin of :meth:`save_state` for async code."""
        return self.save_state(
            description=description,
            ttl_seconds=ttl_seconds,
            overwrite=overwrite,
            name=name,
        )

    def load_state(self, model_package_arn: str) -> APIFuture[LoadStateResponse]:
        """Submit a load of a previously saved training state, weights only.

        Restores the LoRA weights from a :meth:`save_state` checkpoint and
        leaves the optimizer freshly initialized, so training continues from
        the saved weights with the optimizer reset. To also restore optimizer
        state (e.g. Adam momentum), use :meth:`load_state_with_optimizer`.

        The checkpoint itself carries both weights and optimizer state; which
        method you call decides what is restored from it.

        Non-blocking; the future resolves to a :class:`LoadStateResponse`.

        Args:
            model_package_arn: ARN of the model package to load.
        """
        return self._submit_load_state(model_package_arn, load_optimizer=False)

    async def load_state_async(self, model_package_arn: str) -> APIFuture[LoadStateResponse]:
        """Coroutine twin of :meth:`load_state` for async code."""
        return self.load_state(model_package_arn)

    def load_state_with_optimizer(self, model_package_arn: str) -> APIFuture[LoadStateResponse]:
        """Submit a load of saved training state, weights and optimizer.

        Like :meth:`load_state`, but also restores the optimizer state saved by
        :meth:`save_state`, so training resumes exactly where it left off
        instead of restarting the optimizer from scratch.

        Non-blocking; the future resolves to a :class:`LoadStateResponse`.

        Args:
            model_package_arn: ARN of the model package to load.
        """
        return self._submit_load_state(model_package_arn, load_optimizer=True)

    async def load_state_with_optimizer_async(
        self, model_package_arn: str
    ) -> APIFuture[LoadStateResponse]:
        """Coroutine twin of :meth:`load_state_with_optimizer`."""
        return self.load_state_with_optimizer(model_package_arn)

    def _submit_load_state(
        self, model_package_arn: str, *, load_optimizer: bool
    ) -> APIFuture[LoadStateResponse]:
        """Shared submit for the two public load-state methods.

        The ``loadOptimizer`` wire field is always sent explicitly; the choice
        is never a caller-facing knob; it is fixed by which of the two public
        load methods you call.
        """
        op_config = {
            "loadState": {
                "modelPackageArn": model_package_arn,
                "loadOptimizer": load_optimizer,
            }
        }
        return self._submit(op_config, LoadStateResponse)

    def forward(
        self,
        batch: Any,
        loss_fn: "LossFunction | str" = "cross_entropy",
        loss_fn_inputs: Optional[Dict[str, Any]] = None,
        loss_fn_config: Optional[Dict[str, float]] = None,
    ) -> APIFuture[ForwardBackwardOutput]:
        """Submit a forward-only pass (no gradient); returns an :class:`APIFuture`.

        Same input shape as :meth:`forward_backward`, but runs only the forward
        pass, useful for computing loss-fn outputs / logprobs without updating
        optimizer state. Non-blocking; the future resolves to a
        :class:`ForwardBackwardOutput` with ``metrics`` and ``loss_fn_outputs``.

        Args:
            batch: Training data.
            loss_fn: Loss function name or :class:`LossFunction` enum.
            loss_fn_inputs: Optional loss inputs applied to each datum.
            loss_fn_config: Optional loss function configuration (e.g.
                ``{"clip_low_threshold": 0.2, "clip_high_threshold": 5.0}``).

        Raises:
            ValueError: A per-token ``lossFnInputs`` entry is not the same length
                as its datum's ``modelInput``, as for :meth:`forward_backward`.

        A batch too large to send in one request is chunked as it is for
        :meth:`forward_backward`, and the returned future resolves to a single
        combined :class:`ForwardBackwardOutput`.
        """
        loss_fn = LossFunction(loss_fn)
        datum_list = batch_to_datum_list(
            batch,
            tokenizer=self._tokenizer,
            loss_fn_inputs=loss_fn_inputs,
            tokenizer_loader=self.get_tokenizer,
            loss_fn_name=loss_fn.value,
        )
        chunks = chunk_datum_list(datum_list)
        if len(chunks) <= 1:
            return self._submit_forward(
                loss_fn.value,
                chunks[0] if chunks else datum_list,
                loss_fn_config=loss_fn_config,
            )
        # A forward pass leaves no gradient to accumulate, so the chunks are
        # independent; unlike forward_backward, their order carries no meaning.
        futures = [
            self._submit_forward(loss_fn.value, chunk, loss_fn_config=loss_fn_config)
            for chunk in chunks
        ]
        # Weighted by chunk size for the same reason forward_backward is: an
        # uneven split would otherwise report a plain mean over the chunks and
        # understate whichever one carried the most datums.
        return CombinedFuture(
            futures,
            functools.partial(
                combine_forward_backward_results,
                datum_counts=[len(chunk) for chunk in chunks],
            ),
        )

    def _submit_forward(
        self,
        loss_fn_name: str,
        datum_list: List[Dict[str, Any]],
        loss_fn_config: Optional[Dict[str, float]] = None,
    ) -> APIFuture[ForwardBackwardOutput]:
        """Submit one forward op for an already-built datum list."""
        fwd_config: Dict[str, Any] = {
            "lossFnName": loss_fn_name,
            "datumList": datum_list,
        }
        if loss_fn_config is not None:
            fwd_config["lossFnConfig"] = loss_fn_config
        op_config = {"forward": fwd_config}
        return self._submit(op_config, ForwardBackwardOutput)

    async def forward_async(
        self,
        batch: Any,
        loss_fn: "LossFunction | str" = "cross_entropy",
        loss_fn_inputs: Optional[Dict[str, Any]] = None,
        loss_fn_config: Optional[Dict[str, float]] = None,
    ) -> APIFuture[ForwardBackwardOutput]:
        """Coroutine twin of :meth:`forward` for async code."""
        return self.forward(batch, loss_fn, loss_fn_inputs, loss_fn_config=loss_fn_config)

    def _load_weights_for_sampling(self, model_package_arn: str) -> LoadStateResponse:
        """Point the sampling path at a checkpoint and block until complete.

        Internal method used by :meth:`save_weights_and_get_sampling_client` to
        direct the session's sampling path at a saved checkpoint before returning
        a :class:`SamplingClient`.

        This operation is routed through the sampling endpoint (not the training
        endpoint) and does NOT consume a training sequence number.
        """
        return self._submit_load_weights_for_sampling(model_package_arn).result()

    def _submit_load_weights_for_sampling(
        self, model_package_arn: str
    ) -> APIFuture[LoadStateResponse]:
        """Submit a load-weights-for-sampling op; returns an :class:`APIFuture`.

        Routed via the sampling-operations endpoint, so no sequence number
        consumed, so it takes the sampling submit budget, not the training one.
        Not an ``*_async`` twin: it is the submitting half of
        :meth:`_load_weights_for_sampling`, not a coroutine.
        """
        op_config = {"loadWeightsForSampling": {"modelPackageArn": model_package_arn}}
        resp = call_with_transient_retry(
            lambda: self._runtime.execute_sampling_operation(
                trainingSessionName=self.training_session_name,
                operationConfig=op_config,
            ),
            max_attempts=_SAMPLE_SUBMIT_MAX_ATTEMPTS,
        )
        operation_id = resp["operationId"]

        def parse_fn(r: Dict[str, Any]) -> LoadStateResponse:
            return LoadStateResponse._from_api(r.get("operationResult", {}))

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

    # ------------------------------------------------------------------
    # Model package discovery (delegates to ServiceClient with this session's group)
    # ------------------------------------------------------------------

    def list_model_packages(
        self,
        *,
        kind: "Optional[ModelPackageKind]" = None,
        max_results: Optional[int] = None,
    ) -> List[ModelPackageInfo]:
        """List this session's model packages, newest first.

        Convenience over :meth:`ServiceClient.list_model_packages` that supplies
        the session's own model package group.

        Args:
            kind: Optional filter; only return packages of this kind
                (e.g. ``ModelPackageKind.CHECKPOINT`` or ``ModelPackageKind.MODEL``).
            max_results: Cap on packages returned. ``None`` uses the
                ServiceClient default (10 newest).
        """
        kwargs: Dict[str, Any] = {"model_package_group_arn": self._require_group()}
        if kind is not None:
            kwargs["kind"] = kind
        if max_results is not None:
            kwargs["max_results"] = max_results
        return self._require_service_client().list_model_packages(**kwargs)

    def get_latest_model_package(self, kind: ModelPackageKind) -> Optional[ModelPackageInfo]:
        """Return the latest model package of the given kind THIS client saved.

        Answered purely from this client's in-memory save cache, so no API call,
        and immune to the brief lag before a just-saved model package is
        describable. Returns ``None`` if this client has not saved a package of
        that kind in this process.

        A fresh client starts empty: a package from a session this
        client did not drive is not implied here. Use
        :meth:`ServiceClient.get_latest_model_package` or
        :meth:`ServiceClient.list_model_packages` to discover packages in a
        model package group.

        Args:
            kind: The kind of model package to look up
                (e.g. ``ModelPackageKind.CHECKPOINT`` or ``ModelPackageKind.MODEL``).

        Returns:
            A :class:`ModelPackageInfo` with ``model_package_arn`` and ``kind``,
            or ``None`` if nothing of that kind was saved by this client.
        """
        return self._latest_cached(kind)

    # ------------------------------------------------------------------
    # Save cache
    # ------------------------------------------------------------------

    def _cache_on_save(self, kind: ModelPackageKind) -> Callable[[Any], None]:
        """Build a record_fn that caches a saved checkpoint of ``kind``.

        Composed into the save_state / save_weights_for_sampler future's ``parse_fn`` (see
        :meth:`_submit`), so it runs once the op completes (sync or async).
        """

        def _record(result: Any) -> None:
            arn = result.model_package_arn
            if not arn:
                return
            with self._cache_lock:
                self._latest_by_kind[kind] = ModelPackageInfo(model_package_arn=arn, kind=kind)

        return _record

    def _latest_cached(self, kind: ModelPackageKind) -> Optional[ModelPackageInfo]:
        """Latest cached checkpoint of ``kind``, or ``None`` if none saved."""
        with self._cache_lock:
            return self._latest_by_kind.get(kind)

    def _require_group(self) -> str:
        if not self._model_package_group_arn:
            raise ValueError(
                "This client has no model package group associated, so checkpoints "
                "cannot be listed. Use ServiceClient.list_model_packages(...) with an "
                "explicit model_package_group_arn instead."
            )
        return self._model_package_group_arn

    def _require_service_client(self) -> Any:
        if self._service_client is None:
            raise ValueError(
                "This client was not created by a ServiceClient, so checkpoint "
                "lookups are unavailable. Use ServiceClient.list_model_packages(...) "
                "directly."
            )
        return self._service_client

    def _require_lifecycle_client(self) -> Any:
        """The ServiceClient used to stop or delete the session.

        Starting and stopping a session are control-plane operations, and this
        client holds only the runtime client, so the ServiceClient that created
        the session is what carries them out.
        """
        if self._service_client is None:
            raise ValueError(
                "This client was not created by a ServiceClient, so it cannot "
                "stop or delete the session. Use ServiceClient.stop_session(name) "
                "or ServiceClient.delete_session(name) instead."
            )
        return self._service_client

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Stop this client's training session.

        Convenience over :meth:`ServiceClient.stop_session` for this session, with no
        need to repeat the session name. Safe to call on an already-stopped
        session.

        Drains the background submitter first (best effort), so every operation
        already queued reaches the service before the stop request; the client
        accepts no new operations afterwards.
        """
        self._submitter.close()
        try:
            self._require_lifecycle_client().stop_session(self.training_session_name)
        except Exception:
            # Keep the automatic stop armed: it is the only remaining
            # client-side attempt at releasing the session.
            self._handle.note_explicit_stop_failed()
            raise
        # Stopped, so a later garbage collection would only repeat the call.
        self._handle.disarm()

    def delete(self, *, stop_if_active: bool = False) -> None:
        """Delete this client's training session.

        Convenience over :meth:`ServiceClient.delete_session` for this session.
        The service only deletes terminal sessions; pass ``stop_if_active=True``
        to stop and wait for a terminal state first. Drains the background
        submitter first; the client accepts no new operations afterwards.
        """
        self._submitter.close()
        self._require_lifecycle_client().delete_session(
            self.training_session_name, stop_if_active=stop_if_active
        )
        # Disarmed only once the delete has succeeded, or a running session the
        # service refused to delete (the default, stop_if_active=False) was left
        # with nothing client-side still trying to release it; detaching is
        # one-way, so it held its GPUs until the inactivity timeout.
        self._handle.disarm()

    def save_weights_and_get_sampling_client(
        self,
        description: str = "",
        merge_weights: bool = False,
        *,
        name: Optional[str] = None,
        retry_config: Optional[Any] = None,
    ) -> "SamplingClient":
        """Save weights, load them for sampling, and return a SamplingClient.

        Blocks until ``save_weights_for_sampler`` completes, points the sampling
        path at the freshly saved checkpoint, then returns a :class:`SamplingClient`
        bound to it.

        Args:
            description: Optional human-readable description stored on the
                model package (passed through to :meth:`save_weights_for_sampler`).
            merge_weights: Whether to also write full adapter-free weights for
                standalone deployment (passed through to :meth:`save_weights_for_sampler`).
            name: Accepted but not used; setting it logs a warning.
                Checkpoints are model package *versions* addressed by ARN, so
                there is no caller-chosen name to set. Use ``description``.
            retry_config: Accepted but not used; setting it logs a warning.
                Throttling and 5xx failures are retried on a fixed budget; any
                other error surfaces immediately.

        Returns:
            A :class:`SamplingClient` for sampling from the saved checkpoint.

        Raises:
            TrainingSessionError: The save completed without a checkpoint ARN.
                Raised rather than falling back: an empty ARN means the base
                model to :class:`SamplingClient`, so the caller would have
                sampled untrained weights believing they were the trained ones.
        """
        result = self.save_weights_for_sampler(
            description=description,
            merge_weights=merge_weights,
            name=name,
        ).result()
        model_package_arn = result.model_package_arn
        if not model_package_arn:
            raise TrainingSessionError(
                "The weights were saved but the service returned no checkpoint "
                "ARN, so there is nothing to sample from. Retry the save, or "
                "call save_weights_for_sampler() and inspect its result."
            )
        # Point the sampling path at this checkpoint before sampling.
        self._load_weights_for_sampling(model_package_arn)
        # Forwarded, not warned about here: create_sampling_client owns the
        # warning, so passing it on keeps exactly one line in the log.
        return self.create_sampling_client(
            model_package_arn=model_package_arn, retry_config=retry_config
        )

    async def save_weights_and_get_sampling_client_async(
        self,
        description: str = "",
        merge_weights: bool = False,
        *,
        name: Optional[str] = None,
        retry_config: Optional[Any] = None,
    ) -> "SamplingClient":
        """Coroutine twin of :meth:`save_weights_and_get_sampling_client`.

        The save has to complete before the checkpoint can be sampled, so this
        blocks on a worker thread and not on the event loop, letting an
        RL loop keep servicing rollouts while the weights are written.
        """
        return await asyncio.to_thread(
            self.save_weights_and_get_sampling_client,
            description=description,
            merge_weights=merge_weights,
            name=name,
            retry_config=retry_config,
        )

    # ------------------------------------------------------------------
    # Sampling client factory
    # ------------------------------------------------------------------

    def create_sampling_client(
        self,
        model_package_arn: str = "",
        *,
        retry_config: Optional[Any] = None,
    ) -> "SamplingClient":
        """Create a SamplingClient; ``model_package_arn=""`` = base model.

        Args:
            model_package_arn: Model package ARN of the checkpoint to sample
                from. Empty string samples from the base model.
            retry_config: Accepted but not used; setting it logs a warning.
                Throttling and 5xx failures are retried on a fixed budget; any
                other error surfaces immediately.

        Returns:
            A :class:`SamplingClient` bound to this session and checkpoint.
        """
        _warn_unsupported_args(_RETRY_CONFIG_CONSEQUENCE, retry_config=retry_config)
        # The SamplingClient only needs a Hub client to resolve a tokenizer when
        # it has neither a cached tokenizer nor an hf_model_id; build one only in
        # that case. Best-effort: if sagemaker-core is not installed, skip the
        # Hub client: token-ID prompts still work, but string prompts will fail
        # later with a clear error.
        hub_client = self._hub_client
        if hub_client is None and self._tokenizer is None and not self._hf_model_id:
            try:
                hub_client = self._get_hub_client()
            except ImportError:  # ModuleNotFoundError is a subclass
                pass
        return SamplingClient(
            training_session_name=self.training_session_name,
            runtime=self._runtime,
            model_package_arn=model_package_arn,
            base_model=self._base_model,
            tokenizer=self._tokenizer,
            hub_client=hub_client,
            hf_model_id=self._hf_model_id,
            service_client=self._service_client,
            # Sharing the handle refcounts the session across both clients, so it
            # survives whichever is collected first and is stopped when the last
            # one goes, in either drop order. It also marks the new client as
            # NOT owning the session, so leaving a `with` block on it cannot stop
            # a session this training loop is still driving.
            session_handle=self._handle,
            # Shared for the same reason the handle is: a loop that refreshes
            # its sampler calls this every step, and a submitter is an event
            # loop on a daemon thread. Safe to share because wire calls run on
            # the loop's executor, so the loop itself only dispatches, and
            # sampling submits unordered, taking no turn from training's.
            submitter=self._submitter,
        )

    async def create_sampling_client_async(
        self,
        model_package_arn: str = "",
        *,
        retry_config: Optional[Any] = None,
    ) -> "SamplingClient":
        """Coroutine twin of :meth:`create_sampling_client`.

        Binding to an existing session issues no control-plane call, so this is
        cheap; it exists so an ``asyncio`` caller can use one calling convention
        throughout without remembering which factories block.
        """
        return await asyncio.to_thread(
            self.create_sampling_client,
            model_package_arn=model_package_arn,
            retry_config=retry_config,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _submit(
        self,
        operation_config: Dict[str, Any],
        result_cls: Type[_T],
        *,
        record_fn: Optional[Callable[[_T], None]] = None,
    ) -> APIFuture[_T]:
        """Queue an operation; immediately return a future over its result.

        The sequence slot is stamped here, atomically with scheduling on the
        ordered background submitter, so wire order always equals call order;
        the ExecuteTrainingOperation round trip itself happens off-thread. A
        submit error (e.g. a ConflictException) surfaces when the returned
        future is resolved, not at call time.

        ``record_fn`` is composed into the future's ``parse_fn``, so it runs
        exactly once when the op completes, on whichever path resolves the
        future. Used to cache saved checkpoints.
        """
        client_token = str(uuid.uuid4())

        def parse_fn(r: Dict[str, Any]) -> _T:
            result = result_cls._from_api(r.get("operationResult", {}))
            if record_fn is not None:
                record_fn(result)
            return result

        with self._sequence_lock:
            # Every allocated seq is dispatched exactly once: the ordered
            # submitter releases wire calls strictly in allocation order, and a
            # failed submit still consumed its slot (the service blocks on the
            # gap, matching a lost request's semantics). Use
            # `current_sequence_number` to inspect without consuming.
            self._sequence_counter += 1
            seq = self._sequence_counter

            def _wire_submit() -> APIFuture[_T]:
                # Transient 5xx/throttling are retried; the stable clientToken
                # + sequenceNumber make the call idempotent, so a retry never
                # double-submits or skips a sequence.
                resp = call_with_transient_retry(
                    lambda: self._runtime.execute_training_operation(
                        trainingSessionName=self.training_session_name,
                        sequenceNumber=seq,
                        clientToken=client_token,
                        operationConfig=operation_config,
                    ),
                    max_attempts=_TRAINING_SUBMIT_MAX_ATTEMPTS,
                )
                operation_id = resp["operationId"]
                return APIFuture(
                    operation_id=operation_id,
                    poll_fn=lambda: self._runtime.get_training_operation_result(
                        trainingSessionName=self.training_session_name,
                        operationId=operation_id,
                    ),
                    parse_fn=parse_fn,
                    status_key="status",
                    operation_type="training",
                )

            submit_future = self._submitter.submit_ordered(run_blocking(_wire_submit))
        return PendingAPIFuture(submit_future, operation_type="training", keepalive=self._handle)

    @property
    def current_sequence_number(self) -> int:
        """The last sequence number assigned to an operation (read-only).

        Safe to read for diagnostics: unlike the internal sequence allocator,
        this does NOT advance the counter, so inspecting it never consumes a
        sequence the service expects to see dispatched.
        """
        return self._sequence_counter

    def __enter__(self) -> "TrainingClient":
        """Return this client, so it can be used as a context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop the session on leaving the block."""
        # A failed stop must not replace whatever exception is leaving the
        # block, but it must not vanish either: the session is still running.
        try:
            self.stop()
        except Exception:
            logger.warning(
                "Failed to stop session %r on exit; it will be stopped when "
                "this client is collected, or after its inactivity timeout.",
                self.training_session_name,
                exc_info=True,
            )
