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
"""ServiceClient: entry point for creating and managing training sessions.

Holds session-level defaults (role, MPG, S3 output path) so that child-client
creation is concise. Use :meth:`ServiceClient.create_lora_training_client` to
start a training session and get back a :class:`TrainingClient`, or
:meth:`ServiceClient.create_sampling_client` to sample without training.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from sagemaker.train.training_session._endpoints import (
    _SAGEMAKER_ENDPOINT_ENV,
    _resolve_region,
)

if TYPE_CHECKING:
    from sagemaker.core.helper.session_helper import Session
from sagemaker.train.training_session.exceptions import ArtifactDeletionError
from sagemaker.train.training_session._utils import (
    DEFAULT_LATEST_SCAN_LIMIT,
    DEFAULT_LIST_MODEL_PACKAGES_LIMIT,
    MANAGED_STORAGE_RESTRICTED,
    MAX_RESULTS_PER_PAGE,
    base_model_arn_from_model_package,
    build_lora_config,
    delete_s3_objects_under,
    describe_model_package_artifacts,
    describe_model_package_kind,
    list_base_models_in_region,
    list_model_package_summaries,
    resolve_base_model,
    _warn_unsupported_args,
    _RETRY_CONFIG_CONSEQUENCE,
)
from sagemaker.train.training_session.sampling_client import SamplingClient
from sagemaker.train.training_session.training_client import TrainingClient
from sagemaker.train.training_session.types import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_RANK,
    ModelPackageDeletion,
    ModelPackageInfo,
    ModelPackageKind,
    SessionStatus,
    TrainingSessionSummary,
    VpcConfig,
)

logger = logging.getLogger(__name__)

_RUNNING_STATUS = "Running"

_TERMINAL_STATUSES = frozenset({"Failed", "Stopped", "Stopping", "Blocked"})
#: States from which the service accepts a DeleteTrainingSession call.
_DELETABLE_STATUSES = frozenset({"Stopped", "Failed", "DeletionFailed"})

#: States in which a StopTrainingSession call has nothing left to do: the session
#: is already on its way down, already terminal, or being removed. The service
#: rejects the state transition instead of treating the call as a no-op, so
#: :meth:`ServiceClient.stop_session` absorbs it.
_STOP_IS_NOOP_STATUSES = frozenset({"Stopping", "Stopped", "Failed", "Deleting", "DeletionFailed"})

#: Name of the model package group used when the caller supplies none. One shared
#: group per account/region, so kind-filtered discovery has to look past packages
#: it did not write; see :data:`DEFAULT_LATEST_SCAN_LIMIT`.
DEFAULT_MODEL_PACKAGE_GROUP_NAME = "sagemaker-default"

#: Prefix for auto-generated session names, when no ``session_name_prefix`` is set.
DEFAULT_SESSION_NAME_PREFIX = "sagemaker-training-session"


def _session_name_of(session: str) -> str:
    """Accept either a session name or its ARN and return the name.

    ``list_sessions`` hands back ``TrainingSessionArn`` while the stop/delete
    APIs take a name, so callers otherwise have to split the ARN themselves. A
    session ARN ends ``.../training-session/<name>``; the name is the final
    segment. A bare name has no ``/`` and is returned unchanged.
    """
    return session.rsplit("/", 1)[-1] if session.startswith("arn:") else session


def _session_and_region(region: Optional[str]) -> "tuple[Session, str]":
    """Return the session every call goes through, and the Region it runs in.

    One session serves this client: its clients carry the configuration every
    other sagemaker-core caller gets, and it answers for the default output
    bucket. It is pinned to the resolved Region so that the Region a session
    runs in is the Region its output is written to.

    Args:
        region: Region to run in, resolved from the environment when not given.
    """
    from sagemaker.core.helper.session_helper import Session as _Session
    from sagemaker.train.defaults import TrainDefaults

    resolved = _resolve_region(region)
    session = TrainDefaults.get_sagemaker_session(
        sagemaker_session=_Session(boto_session=boto3.Session(region_name=resolved))
    )
    return session, resolved


def _stop_already_done(error: ClientError) -> bool:
    """True when a StopTrainingSession failure only means "nothing left to stop".

    The control plane validates the session state transition and answers a stop
    it cannot apply (``Stopping -> Stopping``, ``Stopped -> Stopping``) with a
    ``ValidationException`` instead of ignoring it. Reaching that is normal, not a
    failure: two teardown paths can legitimately target the same session, an
    explicit stop and the automatic stop that runs when the last client
    referencing the session is collected, and they can run in either order.

    Matching is on the state named in the message, not the message as a whole, so
    it holds if the service rewords the error.
    """
    err = error.response.get("Error") or {}
    if err.get("Code") != "ValidationException":
        return False
    message = err.get("Message") or ""
    return any(
        f"invalid session state transition from {status} to " in message
        for status in _STOP_IS_NOOP_STATUSES
    )


#: Not-found phrases that mean a model package is already gone. A
#: ``ResourceNotFound`` code says so on its own; ``ValidationException`` also
#: covers real problems (a malformed ARN, a package still referenced), so it
#: counts as absent only when the message says the package is not there.
_ABSENT_MODEL_PACKAGE_PHRASES = ("does not exist", "not found", "cannot be found")


def _model_package_absent(error: ClientError) -> bool:
    """True when a model package Describe/Delete failed only because it is gone.

    Lets :meth:`ServiceClient.delete_model_package` be idempotent: deleting a
    package twice, or racing another caller doing the same, reports "nothing
    left to delete" instead of raising.
    """
    err = error.response.get("Error") or {}
    code = err.get("Code")
    if code in ("ResourceNotFound", "ResourceNotFoundException"):
        return True
    if code != "ValidationException":
        return False
    message = (err.get("Message") or "").lower()
    return any(phrase in message for phrase in _ABSENT_MODEL_PACKAGE_PHRASES)


def _model_package_group_exists(error: Exception) -> bool:
    """True when creating a model package group failed only because it is there.

    Lets the default group be created on demand without racing another caller
    doing the same. ``ResourceInUse`` is the modelled answer, so it is checked
    first; the message is only consulted when the code does not settle it.

    Reads the response defensively: the registry client's error classes are its
    own, and one may carry the reason in its string alone.
    """
    err = getattr(error, "response", None) or {}
    if (err.get("Error") or {}).get("Code") in ("ResourceInUse", "ResourceInUseException"):
        return True
    message = (err.get("Error") or {}).get("Message") or str(error)
    return "already exists" in message.lower()


def _build_create_session_args(
    *,
    session_name: str,
    base_model_arn: str,
    model_package_group_arn: str,
    lora_config: Dict[str, Any],
    role_arn: str,
    s3_path: str,
    kms_key_arn: Optional[str],
    vpc_config: Optional[VpcConfig],
) -> Dict[str, Any]:
    """Build a CreateTrainingSession request.

    Shared by both create paths, so the training and sampling-only requests
    cannot drift apart. Model identity goes in ``ModelConfig`` and the output
    location in ``OutputConfig``; both are required by the API.
    """
    s3_location: Dict[str, Any] = {"Path": s3_path}
    if kms_key_arn:
        s3_location["KmsKeyArn"] = kms_key_arn

    args: Dict[str, Any] = {
        "TrainingSessionName": session_name,
        "ModelPackageGroupArn": model_package_group_arn,
        "LoraConfig": lora_config,
        "RoleArn": role_arn,
        "ModelConfig": {"BaseModelArn": base_model_arn},
        "OutputConfig": {"S3Location": s3_location},
    }
    if vpc_config is not None:
        args["VpcConfig"] = vpc_config._to_api()
    return args


def _warn_unsupported(**passed: Any) -> None:
    """Warn for each accepted-but-unhonoured session-creation keyword argument.

    Thin wrapper over :func:`_warn_unsupported_args` carrying this module's
    consequence clause.
    """
    _warn_unsupported_args("the session was created without it", **passed)


def _warm_hub_imports() -> None:
    """Import the sagemaker-core / boto3 modules the Hub client needs, once.

    Called from :meth:`ServiceClient.__init__`, not left to
    :meth:`ServiceClient._get_hub_client`, and the reason is subtle: CPython
    retains the frame chain of a first-time import, which pins the frame that
    triggered it, and therefore whatever that frame returns. Performed on
    first use inside ``create_sampling_client`` or ``create_lora_training_client``
    (which call ``_get_hub_client`` to resolve a tokenizer or a base-model
    name), it pins the client those factories return, so the client is never
    collected and its session is not auto-stopped until interpreter exit.

    Warming here moves the pin onto this call's frame, which holds a
    ``ServiceClient`` that lives for the process anyway. Only the first
    ``ServiceClient`` in a process pays it; later ones hit the already-imported
    fast path and retain nothing.

    Guarded, so sagemaker-core stays an optional dependency, and this keeps
    the *package* import cheap, which is what ``_get_hub_client``'s laziness was
    protecting. Only constructing a ``ServiceClient`` pays the import cost.
    """
    try:
        import boto3  # noqa: F401
        import sagemaker.core.helper.session_helper  # noqa: F401
    except Exception:  # pragma: no cover - optional dependency
        logger.debug("sagemaker-core not importable; Hub lookups may fail later")


def _extract_describe_summary(resp: dict) -> dict:
    """Extract the session info dict from a DescribeTrainingSession response."""
    if "TrainingSessionDetail" in resp:
        return resp["TrainingSessionDetail"]
    return {k: v for k, v in resp.items() if k != "ResponseMetadata"}


def _describe_failure_suffix(summary: dict) -> str:
    """Return the reason a session failed, as a message suffix.

    Empty string when the service reported no reason, leaving the status to
    speak for itself.
    """
    reason = summary.get("ErrorDetails") or summary.get("ErrorMessage")
    return f" ({reason})" if reason else ""


class ServiceClient:
    """Entry point for creating and managing training sessions.

    ``ServiceClient`` holds session-level defaults (``role_arn``, optional
    ``model_package_group_arn``, ``s3_output_path``, ``kms_key_arn``,
    ``vpc_config``) so that child-client creation needs only the model
    identity.

    Anything left unset gets a SageMaker-conventional default:

    * **Session name**: ``{session_name_prefix}-{8 hex}``, prefix defaulting to
      :data:`DEFAULT_SESSION_NAME_PREFIX`. ``session_name_prefix`` names sessions
      only; it has no effect on the model package group or the S3 path.
    * **Model package group**: the account's shared
      :data:`DEFAULT_MODEL_PACKAGE_GROUP_NAME`, created on first use. Pass
      ``model_package_group_arn`` to keep one experiment's checkpoints in their
      own group.
    * **S3 output**: ``s3://{sagemaker default bucket}/{session name}/output``,
      resolved through the same ``Session.default_bucket()`` the broader
      SageMaker PySDK uses, so an existing ``sagemaker_config`` override applies.

    Every client this ``ServiceClient`` creates cleans up after itself: a session
    is stopped once the last client referencing it becomes unreachable, so one is
    not left Running by a script that exits or a notebook cell that rebinds its
    client. The automatic path stops the session and leaves deletion to an
    explicit call, so a stopped session stays describable and a run that went
    wrong can still be inspected. Prefer a ``with`` block or an explicit
    ``.stop()`` where you can: the automatic stop is a backstop, not a substitute
    for scoping the session. Durability comes from a saved checkpoint, which
    :meth:`create_sampling_client` takes directly.

    Example::

        from sagemaker.train.training_session import ServiceClient

        svc = ServiceClient(
            region="us-west-2",
            role_arn="arn:aws:iam::123456789012:role/SageMakerTrainingSessionsRole",
            model_package_group_arn="arn:aws:sagemaker:us-west-2:123456789012:model-package-group/my-group",
            s3_output_path="s3://my-bucket/output",
        )
        # rank and alpha are keyword-only, defaulting to DEFAULT_LORA_RANK
        # and DEFAULT_LORA_ALPHA.
        tc = svc.create_lora_training_client(
            base_model="<id from list_base_models()>",
            rank=8,
            alpha=16,
        )
        sp = svc.create_sampling_client(base_model="<id from list_base_models()>")
    """

    def __init__(
        self,
        region: Optional[str] = None,
        *,
        role_arn: str,
        model_package_group_arn: Optional[str] = None,
        s3_output_path: Optional[str] = None,
        kms_key_arn: Optional[str] = None,
        vpc_config: "Optional[VpcConfig | dict[str, Any]]" = None,
        session_name_prefix: Optional[str] = None,
        max_pool_connections: Optional[int] = None,
    ) -> None:
        # Imported lazily to keep ``import`` of the package cheap.
        from sagemaker.train.training_session._clients import (
            _control_plane_client,
            _runtime_client,
        )

        self._sagemaker_session, self._region = _session_and_region(region)
        boto_session = self._sagemaker_session.boto_session
        self._role_arn = role_arn
        self._model_package_group_arn = model_package_group_arn
        self._s3_output_path = s3_output_path
        self._kms_key_arn = kms_key_arn
        self._vpc_config = VpcConfig.from_any(vpc_config) if vpc_config is not None else None
        self._session_name_prefix = session_name_prefix
        self._sagemaker_client = _control_plane_client(
            region=self._region, boto_session=boto_session
        )
        runtime_kwargs: Dict[str, Any] = {
            "region": self._region,
            "boto_session": boto_session,
        }
        if max_pool_connections is not None:
            runtime_kwargs["max_pool_connections"] = max_pool_connections
        self._runtime = _runtime_client(**runtime_kwargs)
        # Lazily-built ``sagemaker`` clients (via sagemaker-core). The Hub one
        # serves model discovery and name -> ARN resolution; the model-registry
        # one serves model-package and checkpoint lookups. They are separate
        # because the two registries are not always reachable at the same
        # endpoint; see the accessors.
        self._hub_client: Optional[Any] = None
        self._model_registry_client: Optional[Any] = None
        # Built on first use by delete_model_package(delete_artifacts=True).
        # Nothing else here touches S3, so most callers never build it.
        self._s3_client: Optional[Any] = None
        # Must happen here, not on first use in a factory; see the docstring.
        # Without it, the first client a factory returns is never collected and
        # its session is not auto-stopped until interpreter exit.
        _warm_hub_imports()

    def _build_core_sagemaker_client(self, endpoint_url: Optional[str]) -> Any:
        """Return the ``sagemaker`` client used for Hub and model-registry lookups.

        Normally this is the client on the
        :class:`~sagemaker.core.helper.session_helper.Session` this
        ``ServiceClient`` holds, so lookups carry the same configuration and
        credentials as every other call it makes.

        An ``endpoint_url`` needs its own client, which is built from that same
        session so it still signs with those credentials.
        """
        # Imported here, not at module scope: this is the expensive import in
        # the package, and only these lookups need it.
        from sagemaker.core.user_agent import get_user_agent_extra_suffix

        if not endpoint_url:
            return self._sagemaker_session.sagemaker_client
        return self._sagemaker_session.boto_session.client(
            "sagemaker",
            endpoint_url=endpoint_url,
            config=Config(user_agent_extra=get_user_agent_extra_suffix()),
        )

    def _get_hub_client(self) -> Any:
        """Return the client used for SageMaker Public Hub lookups.

        The Public Hub is published per Region, so this client resolves through
        the Region's own endpoint: a base model's latest version is whatever the
        Region publishes, and reading it anywhere else can name a version the
        Region does not serve.
        """
        if self._hub_client is None:
            self._hub_client = self._build_core_sagemaker_client(None)
        return self._hub_client

    def _get_model_registry_client(self) -> Any:
        """Return the client used for model-package and checkpoint lookups.

        Model packages are registered against the same endpoint the sessions
        are, so this client follows that endpoint rather than the Region default.
        Otherwise it would read a different registry than the one the session
        wrote to.
        """
        endpoint = os.environ.get(_SAGEMAKER_ENDPOINT_ENV)
        if not endpoint:
            # Both registries answer on the Region's own endpoint, so one client
            # serves both. This is the only path a default configuration takes.
            return self._get_hub_client()
        if self._model_registry_client is None:
            self._model_registry_client = self._build_core_sagemaker_client(endpoint)
        return self._model_registry_client

    def _get_s3_client(self) -> Any:
        """Return a region-pinned ``s3`` client, built on first use.

        Signs with the same credentials as the rest of this client: the
        caller's, not the ``role_arn`` the sessions run under. Deleting a model
        package's artifacts therefore needs ``s3:ListBucket`` and
        ``s3:DeleteObject`` on that location for whoever is calling.
        """
        if self._s3_client is None:
            self._s3_client = self._sagemaker_session.boto_session.client("s3")
        return self._s3_client

    def list_base_models(self) -> List[str]:
        """List the base models available for fine-tuning in this client's region.

        Queries the SageMaker Public Hub and returns the models this client can
        fine-tune end to end, which is narrower than every model the Hub tags as
        fine-tunable. Results are region-specific.

        Each model is named by its own model id (``"provider/model"``), spelled as
        its provider publishes it wherever the Hub carries one. Pass one straight to
        :meth:`create_lora_training_client` as ``base_model``. The set changes as
        models are added, so call this rather than hard-coding an id.
        """
        return list_base_models_in_region(self._get_hub_client())

    # ------------------------------------------------------------------
    # Model package discovery
    # ------------------------------------------------------------------
    # These wrap ListModelPackages (newest first). The kind (checkpoint vs model)
    # is not in the list summary, so the kind-specific helpers describe
    # candidates lazily until one matches.

    def list_model_packages(
        self,
        *,
        kind: Optional[ModelPackageKind] = None,
        model_package_group_arn: Optional[str] = None,
        max_results: Optional[int] = DEFAULT_LIST_MODEL_PACKAGES_LIMIT,
    ) -> List[ModelPackageInfo]:
        """List model packages, newest first.

        Each :class:`ModelPackageInfo` carries the ARN, version, creation time,
        and description. Pass its ``model_package_arn`` to
        :meth:`TrainingClient.load_state` or
        :meth:`ServiceClient.create_sampling_client`.

        When ``kind`` is supplied, the method describes each candidate to
        resolve its kind and returns only those matching the filter. Without
        ``kind``, the ``kind`` field on each result is left unset (stays at one
        API call regardless of group size).

        Args:
            kind: Optional filter; only return packages of this kind.
            model_package_group_arn: The model package group (name or ARN).
                When omitted, uses the value set on this :class:`ServiceClient`.
            max_results: Cap on packages returned (default 10, newest first).
                Pass ``None`` to return every version in the group.
        """
        resolved_group = model_package_group_arn or self._model_package_group_arn
        if not resolved_group:
            raise ValueError("No model_package_group_arn provided and none set on ServiceClient.")
        summaries = list_model_package_summaries(
            self._get_model_registry_client(), resolved_group, max_results
        )
        results = [ModelPackageInfo._from_summary(s) for s in summaries]
        if kind is None:
            return results
        filtered: List[ModelPackageInfo] = []
        for info in results:
            info.kind = describe_model_package_kind(
                self._get_model_registry_client(), info.model_package_arn
            )
            if info.kind == kind:
                filtered.append(info)
        return filtered

    def get_latest_model_package(
        self,
        kind: ModelPackageKind,
        *,
        model_package_group_arn: Optional[str] = None,
        scan_limit: int = DEFAULT_LATEST_SCAN_LIMIT,
    ) -> Optional[ModelPackageInfo]:
        """Return the latest model package of a given kind, or ``None``.

        Lists newest first and describes candidates in order, returning the
        first whose kind matches. Use :attr:`ModelPackageKind.CHECKPOINT` for
        resumable state (``save_state``) or :attr:`ModelPackageKind.MODEL` for
        deployable weights (``save_weights_for_sampler``).

        Args:
            kind: The kind of package to search for.
            model_package_group_arn: The model package group. When omitted,
                uses the value set on this :class:`ServiceClient`.
            scan_limit: Most-recent packages to describe while searching
                (default :data:`DEFAULT_LATEST_SCAN_LIMIT`). Exhausting it
                without a match logs a warning: ``None`` from a truncated search
                does not mean no such package exists.
        """
        resolved_group = model_package_group_arn or self._model_package_group_arn
        if not resolved_group:
            raise ValueError("No model_package_group_arn provided and none set on ServiceClient.")
        return self._latest_of_kind(resolved_group, kind, scan_limit)

    def _latest_of_kind(
        self, model_package_group_arn: str, kind: ModelPackageKind, scan_limit: int
    ) -> Optional[ModelPackageInfo]:
        """Newest model package matching ``kind``, describing at most ``scan_limit``."""
        summaries = list_model_package_summaries(
            self._get_model_registry_client(),
            model_package_group_arn,
            max_results=scan_limit,
        )
        for summary in summaries:
            ckpt = ModelPackageInfo._from_summary(summary)
            ckpt.kind = describe_model_package_kind(
                self._get_model_registry_client(), ckpt.model_package_arn
            )
            if ckpt.kind == kind:
                return ckpt
        if len(summaries) >= scan_limit:
            # Exhausted the budget without a match, so a matching package may
            # well exist just past it. WARNING, not INFO: the caller is about to
            # read None as "nothing to resume from" and start over from the base
            # model, and that is not a conclusion to reach from an INFO line.
            logger.warning(
                "No %s package found in the %d most recent versions of group %s, "
                "but the group has more; scan_limit stopped the search, so this "
                "is not proof that none exists. Retry with a larger scan_limit, or "
                "pass a group dedicated to this run.",
                kind.value,
                len(summaries),
                model_package_group_arn,
            )
        elif summaries:
            logger.info(
                "No %s package among the %d versions of group %s.",
                kind.value,
                len(summaries),
                model_package_group_arn,
            )
        return None

    def create_lora_training_client(
        self,
        base_model: str,
        *,
        rank: int = DEFAULT_LORA_RANK,
        alpha: int = DEFAULT_LORA_ALPHA,
        training_session_name: Optional[str] = None,
        model_package_group_arn: Optional[str] = None,
        s3_output_path: Optional[str] = None,
        kms_key_arn: Optional[str] = None,
        vpc_config: "Optional[VpcConfig | dict[str, Any]]" = None,
        seed: Optional[int] = None,
        train_mlp: Optional[bool] = None,
        train_attn: Optional[bool] = None,
        train_unembed: Optional[bool] = None,
        user_metadata: Optional[Dict[str, str]] = None,
    ) -> TrainingClient:
        """Create a TrainingSession resource and return a TrainingClient.

        Blocks until the session reaches ``Running`` before returning.

        Every parameter after ``base_model`` is keyword-only, deliberately:
        ``create_lora_training_client(model, 16)`` does not say what 16 is, and a
        bare number at a call site is exactly the kind of thing that gets
        mis-read as ``alpha``. Requiring ``rank=16`` costs a caller nothing.

        ``seed``, ``train_mlp``, ``train_attn``, ``train_unembed`` and
        ``user_metadata`` are accepted for compatibility with existing training
        scripts but are **not honoured**; each one that is set logs a warning
        naming it.

        Session-level parameters (``model_package_group_arn``, ``s3_output_path``)
        default to the values set on this :class:`ServiceClient`. Pass them here
        to override per-session.

        Args:
            base_model: A model id from :meth:`list_base_models`
                (``"provider/model"``), or a full Hub model ARN to pin a specific
                model version. An id is resolved through the Hub to that model's
                current version.
            rank: LoRA rank (default 32).
            alpha: LoRA alpha / scaling factor (default 64).
            training_session_name: Name for the new session. When omitted, the
                SDK auto-generates ``{session_name_prefix}-{8 hex}``.
            model_package_group_arn: Model package group ARN for checkpoints.
                Falls back to :class:`ServiceClient`'s default. When neither is
                set, the SDK uses the account's shared default group
                (:data:`DEFAULT_MODEL_PACKAGE_GROUP_NAME`), creating it on first
                use.
            s3_output_path: S3 path for outputs. Falls back to
                :class:`ServiceClient`'s default. When neither is set, the SDK
                uses the SageMaker default bucket convention.
            kms_key_arn: ARN of an AWS KMS key used to encrypt the session's
                outputs (saved weights and checkpoints) in Amazon S3. Falls
                back to :class:`ServiceClient`'s default. When neither is set,
                outputs use Amazon S3 managed encryption.
            vpc_config: :class:`VpcConfig` (or a dict with ``subnets`` and
                ``security_group_ids``) to run the session inside your VPC.
                Falls back to :class:`ServiceClient`'s default.
            seed: Accepted but not used; setting it logs a warning. LoRA
                initialisation is seeded by the service and is not caller-settable.
            train_mlp: Accepted but not used; setting it logs a warning.
                LoRA is applied to the layer set the service selects.
            train_attn: Accepted but not used; setting it logs a warning.
            train_unembed: Accepted but not used; setting it logs a warning.
            user_metadata: Accepted but not used; setting it logs a warning.
                Nothing is recorded against the session. Tag the model package
                group instead if you need run metadata to persist.

        Returns:
            A :class:`TrainingClient` bound to the running session.
        """
        _warn_unsupported(
            seed=seed,
            train_mlp=train_mlp,
            train_attn=train_attn,
            train_unembed=train_unembed,
            user_metadata=user_metadata,
        )
        resolved_mpg = model_package_group_arn or self._model_package_group_arn
        resolved_s3 = s3_output_path or self._s3_output_path
        resolved_kms = kms_key_arn or self._kms_key_arn
        resolved_vpc = (
            VpcConfig.from_any(vpc_config) if vpc_config is not None else self._vpc_config
        )
        resolved_name = training_session_name or self._generate_session_name()

        # Resolve the base model: ARN passthrough vs name lookup
        if base_model.startswith("arn:"):
            resolved_arn, resolved_hf_model_id = base_model, None
        else:
            resolved_arn, resolved_hf_model_id = resolve_base_model(
                base_model, self._get_hub_client()
            )

        if not resolved_mpg:
            resolved_mpg = self._auto_create_mpg()
        if not resolved_s3:
            resolved_s3 = self._default_s3_output_path(resolved_name)

        args = _build_create_session_args(
            session_name=resolved_name,
            base_model_arn=resolved_arn,
            model_package_group_arn=resolved_mpg,
            lora_config=build_lora_config(rank, alpha),
            role_arn=self._role_arn,
            s3_path=resolved_s3,
            kms_key_arn=resolved_kms,
            vpc_config=resolved_vpc,
        )

        resp = self._sagemaker_client.create_training_session(**args)
        logger.info(
            "Created training session %r (arn=%s)",
            resolved_name,
            resp.get("TrainingSessionArn", ""),
        )
        self._wait_until_running_or_stop(resolved_name)
        return TrainingClient(
            training_session_name=resolved_name,
            runtime=self._runtime,
            base_model=resolved_arn,
            hub_client=None,
            hf_model_id=resolved_hf_model_id,
            model_package_group_arn=resolved_mpg,
            service_client=self,
        )

    async def create_lora_training_client_async(
        self, base_model: str, **kwargs: Any
    ) -> TrainingClient:
        """Coroutine twin of :meth:`create_lora_training_client`.

        Session creation waits for the session to reach ``Running``, which can
        take minutes on a cold model. Running it on a worker thread keeps that
        wait off the event loop, so an ``asyncio`` caller can provision and
        prepare data concurrently.

        ``base_model`` is the only positional parameter here too, and forwarding
        ``**kwargs`` is what keeps ``rank`` keyword-only through this path rather
        than quietly reintroducing a positional form the sync method rejects.
        """
        return await asyncio.to_thread(self.create_lora_training_client, base_model, **kwargs)

    def create_training_client_from_state(
        self,
        model_package_arn: str,
        **kwargs: Any,
    ) -> TrainingClient:
        """Create a session from saved model weights.

        Convenience over :meth:`create_lora_training_client` + ``load_state``:
        creates a fresh session (forwarding all the same keyword arguments) and
        loads ``model_package_arn`` into it, so the returned client resumes from
        that checkpoint.

        This restores **only the model weights, not optimizer state**; the
        optimizer resets. To also restore optimizer state (e.g. Adam momentum),
        use :meth:`create_training_client_from_state_with_optimizer`. The
        checkpoint carries both; which factory you call decides what is
        restored from it.

        Args:
            model_package_arn: ARN of the model package to load on startup.
            **kwargs: Forwarded to :meth:`create_lora_training_client`. When
                ``base_model`` is omitted, it is read from the checkpoint's
                model package metadata (pinned to the version the weights were
                trained on).
        """
        return self._create_training_client_from_state(
            model_package_arn, load_optimizer=False, kwargs=kwargs
        )

    async def create_training_client_from_state_async(
        self, model_package_arn: str, **kwargs: Any
    ) -> TrainingClient:
        """Coroutine twin of :meth:`create_training_client_from_state`."""
        return await asyncio.to_thread(
            self.create_training_client_from_state,
            model_package_arn=model_package_arn,
            **kwargs,
        )

    def create_training_client_from_state_with_optimizer(
        self, model_package_arn: str, **kwargs: Any
    ) -> TrainingClient:
        """Like :meth:`create_training_client_from_state`, restoring the optimizer."""
        return self._create_training_client_from_state(
            model_package_arn, load_optimizer=True, kwargs=kwargs
        )

    async def create_training_client_from_state_with_optimizer_async(
        self, model_package_arn: str, **kwargs: Any
    ) -> TrainingClient:
        """Coroutine twin of :meth:`create_training_client_from_state_with_optimizer`."""
        return await asyncio.to_thread(
            self.create_training_client_from_state_with_optimizer,
            model_package_arn=model_package_arn,
            **kwargs,
        )

    def _create_training_client_from_state(
        self, model_package_arn: str, *, load_optimizer: bool, kwargs: Dict[str, Any]
    ) -> TrainingClient:
        if "base_model" not in kwargs:

            mp_desc = self._get_model_registry_client().describe_model_package(
                ModelPackageName=model_package_arn
            )
            kwargs = {
                **kwargs,
                "base_model": base_model_arn_from_model_package(mp_desc, self._region),
            }
        tc = self.create_lora_training_client(**kwargs)
        # Wait for the restore, so the client returned is ready to train.
        try:
            if load_optimizer:
                tc.load_state_with_optimizer(model_package_arn).result()
            else:
                tc.load_state(model_package_arn).result()
        except BaseException:
            # No client is being returned, so nothing would own the session the
            # line above created; stop it instead of leaving it to the inactivity
            # timeout. Never masks the original failure.
            try:
                tc.stop()
            except Exception:
                logger.warning(
                    "Could not stop training session %r after restoring state "
                    "from %r failed; it will be stopped after its inactivity "
                    "timeout.",
                    tc.training_session_name,
                    model_package_arn,
                    exc_info=True,
                )
            raise
        return tc

    def create_sampling_client(
        self,
        *,
        base_model: Optional[str] = None,
        model_package_arn: Optional[str] = None,
        retry_config: Optional[Any] = None,
    ) -> SamplingClient:
        """Create a SamplingClient, implicitly creating a training session.

        The SDK creates a training session under the hood and returns a
        :class:`SamplingClient` bound to it. The session uses SDK-generated
        defaults for MPG and S3 output (or the values on this
        :class:`ServiceClient`).

        Because the session is implicit, the returned client *owns* it: use the
        client as a context manager (or call :meth:`SamplingClient.stop`) so the
        session is stopped at a point you chose::

            with svc.create_sampling_client(base_model="<id from list_base_models()>") as sp:
                out = sp.sample("hello").result()
            # session stopped here

        If you neither scope nor stop it, the SDK stops the session once the
        client becomes unreachable (on garbage collection or at interpreter exit)
        instead of leaving it Running until its inactivity timeout stops it.
        That is a backstop for interactive work, not a replacement
        for the ``with`` block above: it fires at a time the interpreter picks,
        and never at all if the process is killed with ``os._exit`` or
        ``SIGKILL``.

        Specify exactly one of ``base_model`` or ``model_package_arn``:

        * ``base_model``: sample from the base model directly.
        * ``model_package_arn``: sample from saved weights (the SDK resolves
          the base model from the package metadata).

        Args:
            base_model: A model id from :meth:`list_base_models`, or a full Hub
                model ARN.
            model_package_arn: ARN of a model package (saved weights) to sample
                from.
            retry_config: Accepted but not used; setting it logs a warning.
                Throttling and 5xx failures are retried on a fixed budget; any
                other error surfaces immediately.

        Returns:
            A :class:`SamplingClient` bound to the implicit session.

        Raises:
            ValueError: If neither or both of ``base_model`` /
                ``model_package_arn`` are supplied.
        """
        _warn_unsupported_args(_RETRY_CONFIG_CONSEQUENCE, retry_config=retry_config)
        if bool(base_model) == bool(model_package_arn):
            raise ValueError("Specify exactly one of base_model or model_package_arn.")

        resolved_hf_model_id = None
        if model_package_arn:
            mp_desc = self._get_model_registry_client().describe_model_package(
                ModelPackageName=model_package_arn
            )
            resolved_base_arn = base_model_arn_from_model_package(mp_desc, self._region)
            checkpoint_ref = model_package_arn
        else:
            if base_model.startswith("arn:"):
                resolved_base_arn = base_model
            else:
                resolved_base_arn, resolved_hf_model_id = resolve_base_model(
                    base_model, self._get_hub_client()
                )
            checkpoint_ref = ""

        session_name = self._generate_session_name()
        resolved_mpg = self._model_package_group_arn or self._auto_create_mpg()
        resolved_s3 = self._s3_output_path or self._default_s3_output_path(session_name)

        args = _build_create_session_args(
            session_name=session_name,
            base_model_arn=resolved_base_arn,
            model_package_group_arn=resolved_mpg,
            lora_config=build_lora_config(DEFAULT_LORA_RANK, DEFAULT_LORA_ALPHA),
            role_arn=self._role_arn,
            s3_path=resolved_s3,
            kms_key_arn=self._kms_key_arn,
            vpc_config=self._vpc_config,
        )
        self._sagemaker_client.create_training_session(**args)
        logger.info(
            "Created implicit sampling session %r for base model %s",
            session_name,
            resolved_base_arn,
        )
        self._wait_until_running_or_stop(session_name)

        return SamplingClient(
            training_session_name=session_name,
            runtime=self._runtime,
            model_package_arn=checkpoint_ref,
            base_model=resolved_base_arn,
            # Carry the id resolved above so a later get_tokenizer() does not
            # describe the Hub again, and leave the Hub client for the client to
            # borrow lazily if it ever needs one, as the training path does.
            hub_client=None,
            hf_model_id=resolved_hf_model_id,
            # This session is implicit: no TrainingClient owns it, so no handle
            # is passed in and the SamplingClient builds its own. That is exactly
            # what marks it the owner, responsible for the stop on __exit__.
            service_client=self,
        )

    async def create_sampling_client_async(self, **kwargs: Any) -> SamplingClient:
        """Coroutine twin of :meth:`create_sampling_client`.

        Creates the implicit session on a worker thread, so the wait for it to
        reach ``Running`` does not block the event loop.
        """
        return await asyncio.to_thread(self.create_sampling_client, **kwargs)

    def list_sessions(
        self,
        *,
        name_contains: Optional[str] = None,
        status_equals: "Optional[str | SessionStatus]" = None,
        max_results: Optional[int] = None,
    ) -> "list[TrainingSessionSummary]":
        """List training sessions, auto-paginating.

        Args:
            name_contains: Only return sessions whose name contains this string.
            status_equals: Only return sessions in this status. Accepts a
                :class:`SessionStatus` enum or a raw string (e.g. ``"Running"``).
            max_results: Cap the total number of summaries returned. When None,
                every matching session is returned. Larger than one page, this
                pages until the cap is reached.

        Returns:
            A list of :class:`TrainingSessionSummary`.
        """
        filters: dict[str, Any] = {}
        if name_contains is not None:
            filters["NameContains"] = name_contains
        if status_equals is not None:
            filters["StatusEquals"] = getattr(status_equals, "value", status_equals)

        summaries: list[TrainingSessionSummary] = []
        next_token: Optional[str] = None
        while True:
            args = dict(filters)
            if next_token:
                args["NextToken"] = next_token
            if max_results is not None:
                remaining = max_results - len(summaries)
                if remaining <= 0:
                    break
                args["MaxResults"] = min(remaining, MAX_RESULTS_PER_PAGE)
            resp = self._sagemaker_client.list_training_sessions(**args)
            summaries.extend(
                TrainingSessionSummary._from_summary(summary)
                for summary in resp.get("TrainingSessionSummaries", [])
            )
            next_token = resp.get("NextToken")
            if not next_token:
                break

        if max_results is not None:
            return summaries[:max_results]
        return summaries

    def stop_session(self, training_session: str) -> None:
        """Stop a running training session.

        Transitions the session toward a terminal ``Stopped`` state.

        Idempotent: a session that is already stopping, already terminal, or
        being deleted is left as it is and the call returns normally. That makes
        stop order-independent, which is what the rest of the lifecycle relies
        on: an explicit ``stop_session`` here and the automatic stop that
        follows the last client's collection both land on the same session, and
        neither has to know whether the other went first.

        Args:
            training_session: Session name or ARN. A ``TrainingSessionArn`` from
                :meth:`list_sessions` works as-is; the name is taken from the
                ARN's last segment.
        """
        self._stop_session_if_active(_session_name_of(training_session))

    def _stop_session_if_active(self, training_session_name: str) -> bool:
        """:meth:`stop_session`, reporting whether a stop was actually issued.

        ``False`` means the session was already stopping or terminal, so nothing
        happened. The automatic teardown needs to know the difference: telling a
        caller "stopped it for you" about a session that was already down is
        noise, and warning them they forgot to clean up is simply wrong.

        Kept private; ``stop_session``'s contract is "the session is not
        running when this returns", and that holds either way.
        """
        try:
            self._sagemaker_client.stop_training_session(
                TrainingSessionName=training_session_name,
            )
        except ClientError as e:
            if not _stop_already_done(e):
                raise
            logger.debug(
                "Training session %r is already stopping or terminal; " "nothing to stop.",
                training_session_name,
            )
            return False
        logger.info("Requested stop for training session %r", training_session_name)
        return True

    def delete_session(self, training_session: str, *, stop_if_active: bool = False) -> None:
        """Delete a training session.

        The service only deletes sessions in a terminal state (``Stopped``,
        ``Failed``, or ``DeletionFailed``); deletion is asynchronous: the
        session flips to ``Deleting`` and a background worker removes it
        shortly after. Pass ``stop_if_active=True`` to stop a still-running
        session before deleting it.

        Args:
            training_session: Session name or ARN. A ``TrainingSessionArn`` from
                :meth:`list_sessions` works as-is; the name is taken from the
                ARN's last segment.
            stop_if_active: When True, issue a stop and wait for a terminal
                state before deleting.
        """
        training_session_name = _session_name_of(training_session)
        if stop_if_active:
            self.stop_session(training_session_name)
            self._wait_until_terminal(training_session_name)

        self._sagemaker_client.delete_training_session(
            TrainingSessionName=training_session_name,
        )
        logger.info("Requested delete for training session %r", training_session_name)

    def delete_model_package(
        self, model_package_arn: str, *, delete_artifacts: bool = False
    ) -> ModelPackageDeletion:
        """Delete a model package, optionally deleting its S3 artifacts too.

        The model package carries the only reference to the S3 location its
        weights were written to, so by default this deletes the package and
        leaves the objects: they keep costing storage with nothing pointing at
        them. Keep the returned :attr:`~ModelPackageDeletion.s3_uri` if you
        intend to clean up later.

        With ``delete_artifacts=True`` the objects go first and the package
        second, so a failure part-way leaves the package still pointing at
        whatever survived and the call can be retried.

        Idempotent: a package that is already gone returns a result with
        ``model_package_deleted=False`` instead of raising.

        .. note::
            :meth:`ServiceClient.create_sampling_client` and
            :meth:`TrainingClient.load_state` read a package's artifacts, so
            once they are deleted do not rely on this package for either.

        Args:
            model_package_arn: ARN of the model package version to delete.
            delete_artifacts: When True, delete the S3 objects under the
                package's location before deleting the package. Objects whose
                storage SageMaker manages on your behalf are left to it;
                deleting the package is what releases those.

        Returns:
            A :class:`ModelPackageDeletion` describing what was deleted and, when
            the artifacts were kept, where they still are.

        Raises:
            ArtifactDeletionError: If some object could not be deleted. The
                model package is left in place, so a retry can find them again.
            ValueError: If the package's location names a bucket root, where
                deleting "its" artifacts would empty the whole bucket.

        Example::

            # Delete the package, keep the weights
            result = svc.delete_model_package(ckpt.model_package_arn)
            print("weights still at", result.s3_uri)

            # Delete both
            result = svc.delete_model_package(
                ckpt.model_package_arn, delete_artifacts=True
            )
            print("removed", result.objects_deleted, "objects")
        """
        try:
            artifacts = describe_model_package_artifacts(
                self._get_model_registry_client(), model_package_arn
            )
        except ClientError as e:
            if not _model_package_absent(e):
                raise
            logger.info(
                "Model package %r does not exist; nothing to delete.",
                model_package_arn,
            )
            return ModelPackageDeletion(
                model_package_arn=model_package_arn,
                model_package_deleted=False,
                retained_reason="the model package does not exist",
            )

        s3_uri = artifacts["s3_uri"]
        objects_deleted = 0
        retained_reason: Optional[str] = None
        if not delete_artifacts:
            retained_reason = "delete_artifacts was not set"
        elif artifacts["managed_storage_type"] == MANAGED_STORAGE_RESTRICTED:
            # Describe returns no location for these and the caller has no
            # access to one: deleting the package is what releases the storage.
            # A skip, not an error, so the same call works for any kind.
            retained_reason = (
                "this model package's artifacts are stored by SageMaker on your "
                "behalf, not in your own S3 location; deleting the package "
                "releases them"
            )
        elif not s3_uri:
            retained_reason = "the model package carries no S3 location"
        else:
            objects_deleted, failed_keys = delete_s3_objects_under(
                self._get_s3_client(),
                s3_uri,
                s3_data_type=artifacts["s3_data_type"],
            )
            if failed_keys:
                raise ArtifactDeletionError(model_package_arn, s3_uri, failed_keys)
            logger.info(
                "Deleted %d object(s) under %s for model package %r.",
                objects_deleted,
                s3_uri,
                model_package_arn,
            )

        try:
            self._get_model_registry_client().delete_model_package(
                ModelPackageName=model_package_arn
            )
            deleted = True
        except ClientError as e:
            if not _model_package_absent(e):
                raise
            # Another caller deleted it between the describe and here.
            deleted = False

        if retained_reason is not None and s3_uri:
            # The package carried the only record of this location and is now
            # gone, so say it once at INFO even if the result is discarded.
            logger.info(
                "Model package %r deleted; its artifacts remain at %s (%s).",
                model_package_arn,
                s3_uri,
                retained_reason,
            )
        return ModelPackageDeletion(
            model_package_arn=model_package_arn,
            model_package_deleted=deleted,
            s3_uri=s3_uri,
            objects_deleted=objects_deleted,
            artifacts_retained=retained_reason is not None,
            retained_reason=retained_reason,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate_session_name(self) -> str:
        """Auto-generate a unique training session name."""

        base = self._session_name_prefix or DEFAULT_SESSION_NAME_PREFIX
        short_id = uuid.uuid4().hex[:8]
        return f"{base}-{short_id}"

    def _auto_create_mpg(self) -> str:
        """Return the ARN of the account's default model package group.

        One shared group per account/region, mirroring how the broader SageMaker
        PySDK shares one default bucket: a group per session would leave a
        never-collected group behind for every session an unconfigured script
        creates, and would scatter a run's checkpoints so that
        :meth:`get_latest_model_package` could not find them from a later
        process. Created on first use; returns the existing ARN thereafter.
        """
        group_name = DEFAULT_MODEL_PACKAGE_GROUP_NAME
        try:
            resp = self._get_model_registry_client().create_model_package_group(
                ModelPackageGroupName=group_name,
                ModelPackageGroupDescription=(
                    "Default group for SageMaker training session checkpoints, "
                    "created by the sagemaker-train SDK."
                ),
            )
            return resp["ModelPackageGroupArn"]
        except self._get_model_registry_client().exceptions.ClientError as e:
            if _model_package_group_exists(e):
                resp = self._get_model_registry_client().describe_model_package_group(
                    ModelPackageGroupName=group_name
                )
                return resp["ModelPackageGroupArn"]
            raise

    def _default_s3_output_path(self, session_name: str) -> str:
        """Generate a default S3 output path using the SageMaker default bucket.

        Follows the same convention as the broader SageMaker PySDK:
        ``s3://sagemaker-{region}-{account}/{session_name}/output``, and created
        if it does not exist yet, exactly like the broader SageMaker PySDK.
        """
        bucket = self._sagemaker_session.default_bucket()
        return f"s3://{bucket}/{session_name}/output"

    def _wait_until_running_or_stop(self, training_session_name: str) -> None:
        """Wait for ``Running``, stopping the session if it never gets there.

        The create call has already returned by this point, so a failed wait
        would otherwise leave a session running with no client bound to it:
        nothing holds its handle, so no automatic stop is armed and only the
        inactivity timeout would release it. Includes ``KeyboardInterrupt``:
        interrupting a cold-model wait is the likeliest way to reach this.

        Best effort, and never masks the original failure, which is what tells
        the caller why they have no client.
        """
        try:
            self._wait_until_running(training_session_name)
        except BaseException:
            try:
                self._stop_session_if_active(training_session_name)
            except Exception:
                logger.warning(
                    "Could not stop training session %r after it failed to "
                    "reach Running; it will be stopped after its inactivity "
                    "timeout.",
                    training_session_name,
                    exc_info=True,
                )
            raise

    def _wait_until_running(
        self,
        training_session_name: str,
        timeout: int = 600,
        poll_interval: int = 10,
    ) -> None:
        """Block until the session reaches ``Running`` status."""
        deadline = time.monotonic() + timeout
        last_status = None
        while time.monotonic() < deadline:
            resp = self._sagemaker_client.describe_training_session(
                TrainingSessionName=training_session_name,
            )
            summary = _extract_describe_summary(resp)
            status = summary["TrainingSessionStatus"]
            if status != last_status:
                logger.info("Session %r status: %s", training_session_name, status)
                last_status = status
            if status == _RUNNING_STATUS:
                return
            if status in _TERMINAL_STATUSES:
                raise RuntimeError(
                    f"Session {training_session_name!r} entered terminal state: "
                    f"{status}{_describe_failure_suffix(summary)}"
                )
            time.sleep(poll_interval)

        raise TimeoutError(
            f"Session {training_session_name!r} did not reach Running within {timeout}s"
        )

    def _wait_until_terminal(
        self,
        training_session_name: str,
        timeout: int = 600,
        poll_interval: int = 10,
    ) -> None:
        """Block until the session reaches a deletable terminal status."""
        deadline = time.monotonic() + timeout
        last_status = None
        while time.monotonic() < deadline:
            resp = self._sagemaker_client.describe_training_session(
                TrainingSessionName=training_session_name,
            )
            status = _extract_describe_summary(resp)["TrainingSessionStatus"]
            if status != last_status:
                logger.info("Session %r status: %s", training_session_name, status)
                last_status = status
            if status in _DELETABLE_STATUSES:
                return
            time.sleep(poll_interval)

        raise TimeoutError(
            f"Session {training_session_name!r} did not reach a terminal state "
            f"within {timeout}s"
        )
