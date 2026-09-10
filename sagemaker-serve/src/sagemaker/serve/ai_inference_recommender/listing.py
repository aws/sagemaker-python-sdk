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
"""List and filter AI benchmark / recommendation jobs.

The underlying ``ListAIBenchmarkJobs`` / ``ListAIRecommendationJobs`` APIs only
filter by name substring, status, and creation-time window — the endpoint a
benchmark targeted, or the model a recommendation ran on, live on the full
Describe response, not the list summary. So ``endpoint`` / ``model`` /
``model_package`` filtering is done client-side: the native filters narrow the
list server-side, then each candidate is described (hydrated by sagemaker-core's
resource iterator) and matched on the nested field. Two bounds keep the cost in
hand: ``max_results`` caps the matches returned, and ``max_scan`` caps how many
candidates are described — so a rarely-matching filter cannot fan out across the
whole account.
"""
from __future__ import absolute_import

import logging
from typing import List, Optional, Union

import boto3

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.telemetry.constants import Feature
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.serve.ai_inference_recommender.jobs import BenchmarkJob, RecommendationJob

logger = logging.getLogger(__name__)

# Default cap on how many matching jobs are returned.
DEFAULT_MAX_RESULTS = 100
# Default cap on how many candidates are described (each is one Describe, issued
# by the resource iterator). Bounds the fan-out when a client-side filter
# (endpoint / model / model_package) matches rarely.
DEFAULT_MAX_SCAN = 1000
# How many undescribable candidates to log individually before deferring the
# rest to the end-of-scan summary (avoids one warning per candidate).
_MAX_SKIP_WARNINGS = 5


def _endpoint_matches(job, endpoint: str) -> bool:
    """True if a benchmark job targeted ``endpoint`` (by name or ARN)."""
    target = getattr(job, "benchmark_target", None)
    ep = getattr(target, "endpoint", None) if target else None
    identifier = getattr(ep, "identifier", None) if ep else None
    if not identifier:
        return False
    # identifier may be a name or an ARN; match either the exact value or the
    # name suffix of an ARN (endpoint/<name>).
    return endpoint == identifier or identifier.endswith(f"/{endpoint}")


def _model_matches(job, model: str) -> bool:
    """True if a recommendation job ran on ``model`` (its source S3 URI)."""
    source = getattr(job, "model_source", None)
    s3 = getattr(source, "s3", None) if source else None
    s3_uri = getattr(s3, "s3_uri", None) if s3 else None
    if not s3_uri:
        return False
    return model == s3_uri or s3_uri.rstrip("/") == model.rstrip("/")


def _model_package_matches(job, model_package: str) -> bool:
    """True if a recommendation job is associated with ``model_package``.

    Matches either the output model-package group the job registers into, or a
    model-package ARN produced on one of the job's recommendation rows.
    """
    output = getattr(job, "output_config", None)
    group = getattr(output, "model_package_group_identifier", None) if output else None
    if group and (model_package == group or group.endswith(f"/{model_package}")):
        return True
    for row in getattr(job, "recommendations", None) or []:
        details = getattr(row, "model_details", None)
        arn = getattr(details, "model_package_arn", None) if details else None
        if arn and (model_package == arn or arn.endswith(f"/{model_package}")):
            return True
    return False


def _collect(iterator, predicate, max_results: int, max_scan: int, subclass) -> list:
    """Keep candidates from ``iterator`` matching ``predicate``, re-typed to
    ``subclass`` (so ``show_result`` is available).

    The iterator hydrates each object as it yields it, so this loop does not
    Describe again. ``max_results`` bounds matches returned; ``max_scan`` bounds
    candidates examined. A candidate that fails to Describe is skipped, not
    fatal. All three limits are logged when hit.
    """
    matches: list = []
    scanned = 0
    skipped = 0
    hit_result_cap = False
    it = iter(iterator)
    while scanned < max_scan:
        # next() drives the iterator's per-object Describe, so a failure for one
        # candidate is caught here rather than aborting the listing.
        try:
            job = next(it)
        except StopIteration:
            break
        except Exception as exc:  # noqa: BLE001 - per-job hydration is best-effort
            skipped += 1
            scanned += 1
            # Log the first few individually for diagnosis; the rest are covered
            # by the summary below, so a fully-denied role does not emit one
            # warning per candidate up to max_scan.
            if skipped <= _MAX_SKIP_WARNINGS:
                logger.warning(
                    "Skipping a %s that could not be described: %s", subclass.__name__, exc
                )
            continue

        scanned += 1
        job.__class__ = subclass
        if predicate is None or predicate(job):
            matches.append(job)
        if len(matches) >= max_results:
            hit_result_cap = True
            break
    else:
        logger.warning(
            "Scanned max_scan=%d jobs and stopped; raise max_scan to look further.", max_scan
        )

    if hit_result_cap:
        logger.info(
            "Returned max_results=%d matches and stopped; there may be more. "
            "Raise max_results to return more.",
            max_results,
        )
    if skipped:
        logger.warning("%d job(s) were skipped because they could not be described.", skipped)
    return matches


@_telemetry_emitter(
    feature=Feature.INFERENCE_RECOMMENDER, func_name="ai_inference_recommender.list_benchmarks"
)
def list_benchmarks(
    *,
    endpoint: Optional[str] = None,
    status: Optional[str] = None,
    name_contains: Optional[str] = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    max_scan: int = DEFAULT_MAX_SCAN,
    sagemaker_session: Optional[Union[boto3.session.Session, Session]] = None,
) -> List[BenchmarkJob]:
    """List benchmark jobs, optionally filtered by the endpoint they targeted.

    Args:
        endpoint: Endpoint name or ARN. Client-side filter (each candidate is
            described, since the endpoint is not on the list summary).
        status: ``StatusEquals`` filter, applied server-side.
        name_contains: ``NameContains`` filter, applied server-side.
        max_results: Cap on the number of matching jobs returned. Defaults to
            ``DEFAULT_MAX_RESULTS``.
        max_scan: Cap on how many candidates are described while filtering (each
            is one Describe). Bounds the fan-out when ``endpoint`` matches
            rarely; a warning is logged if the scan is truncated. Defaults to
            ``DEFAULT_MAX_SCAN``.
        sagemaker_session: Optional session — a ``boto3.session.Session`` or a
            sagemaker ``Session`` wrapping one (unwrapped automatically). A
            default is created if omitted.

    Returns:
        A list of ``BenchmarkJob`` (newest first), each with ``show_result``.
    """
    iterator = BenchmarkJob.get_all(
        **_native_filters(name_contains, status),
        sort_by="CreationTime",
        sort_order="Descending",
        session=_boto_session(sagemaker_session),
    )
    predicate = (lambda job: _endpoint_matches(job, endpoint)) if endpoint else None
    return _collect(iterator, predicate, max_results, max_scan, BenchmarkJob)


@_telemetry_emitter(
    feature=Feature.INFERENCE_RECOMMENDER,
    func_name="ai_inference_recommender.list_recommendations",
)
def list_recommendations(
    *,
    model: Optional[str] = None,
    model_package: Optional[str] = None,
    status: Optional[str] = None,
    name_contains: Optional[str] = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    max_scan: int = DEFAULT_MAX_SCAN,
    sagemaker_session: Optional[Union[boto3.session.Session, Session]] = None,
) -> List[RecommendationJob]:
    """List recommendation jobs, optionally filtered by model or model package.

    Args:
        model: Model source S3 URI the job ran on. Client-side filter.
        model_package: Model-package ARN or group identifier associated with the
            job (its output group, or a package produced on a recommendation
            row). Client-side filter.
        status: ``StatusEquals`` filter, applied server-side.
        name_contains: ``NameContains`` filter, applied server-side.
        max_results: Cap on the number of matching jobs returned. Defaults to
            ``DEFAULT_MAX_RESULTS``.
        max_scan: Cap on how many candidates are described while filtering (each
            is one Describe). Bounds the fan-out when a client-side filter
            matches rarely; a warning is logged if the scan is truncated.
            Defaults to ``DEFAULT_MAX_SCAN``.
        sagemaker_session: Optional session — a ``boto3.session.Session`` or a
            sagemaker ``Session`` wrapping one (unwrapped automatically). A
            default is created if omitted.

    Returns:
        A list of ``RecommendationJob`` (newest first), each with ``show_result``.
    """
    if model and model_package:
        raise ValueError("Pass only one of `model` or `model_package` to list_recommendations().")
    iterator = RecommendationJob.get_all(
        **_native_filters(name_contains, status),
        sort_by="CreationTime",
        sort_order="Descending",
        session=_boto_session(sagemaker_session),
    )
    predicate = None
    if model:
        predicate = lambda job: _model_matches(job, model)  # noqa: E731
    elif model_package:
        predicate = lambda job: _model_package_matches(job, model_package)  # noqa: E731
    return _collect(iterator, predicate, max_results, max_scan, RecommendationJob)


def _boto_session(sagemaker_session):
    """Resolve the session for ``get_all``, which requires a boto3 session.

    Accepts a boto3 session (passed through) or a sagemaker ``Session``
    (unwrapped to its ``.boto_session``). ``None`` stays ``None``.
    """
    if sagemaker_session is None or isinstance(sagemaker_session, boto3.session.Session):
        return sagemaker_session
    boto_session = getattr(sagemaker_session, "boto_session", None)
    if isinstance(boto_session, boto3.session.Session):
        return boto_session
    raise TypeError(
        "sagemaker_session must be a boto3.session.Session or a sagemaker "
        f"Session wrapping one; got {type(sagemaker_session).__name__}."
    )


def _native_filters(name_contains: Optional[str], status: Optional[str]) -> dict:
    """Build the server-side ``get_all`` filter kwargs, omitting unset ones so
    each defaults to the sagemaker-core ``Unassigned`` sentinel."""
    kwargs = {}
    if name_contains:
        kwargs["name_contains"] = name_contains
    if status:
        kwargs["status_equals"] = status
    return kwargs
