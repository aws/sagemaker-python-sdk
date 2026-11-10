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
"""Submitting a labeling job, re-attaching to one, and reading its result back.

The job is a SyntheticDataGeneration job in labeling mode, created through
:class:`~sagemaker.ai_registry.data_preparer.preparer.DataPreparer`. This module adds
what a long-running client loop needs around that call:

* **Transfer**: upload a spec or demos text to S3, size-checked where the service has a
  limit (the byte transfers themselves are :mod:`._s3`).
* **Submission**: a unique job name recorded *before* ``CreateJob`` so a crashed run can
  re-attach; transient account-limit and throttle errors retried with jittered backoff
  within a wait budget; a re-attach that reads a Completed job, waits on a running one and
  resubmits only a Failed, Stopped or unknown one; and the classification of a refusal
  (:class:`CreateJobRefused`, :class:`OptionRefused`, :class:`LabelingBackendUnavailable`).
* **Read-back**: :func:`read_job_results` walks the Hub pointer through
  :class:`~sagemaker.ai_registry.data_preparer.auto_labeling.LabelingSummary` and downloads
  the artifacts the job's index names.

Nothing here writes a local file except the downloads the caller asks for by path.
"""

from __future__ import absolute_import

import json
import logging
import random
import re
import time
import uuid
from pathlib import Path

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _s3
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._bedrock import (
    _set_telemetry_attribution,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.request import (
    INDEX_ARTIFACTS,
    INDEX_FILENAME,
    LabelingRequestError,
    to_dataset_job_config,
)

_log = logging.getLogger(__name__)

# The three errors below are also ``RuntimeError``s, as the command line's were, so a caller
# that catches ``RuntimeError`` (the skill's ``except RuntimeError``) still catches them.


class CreateJobRefused(DataPreparerError, RuntimeError):
    """``CreateJob`` refused the request, so no job exists (the caller drops its job record)."""


class OptionRefused(CreateJobRefused):
    """``CreateJob`` refused an opt-in option field the caller sent (``option_fields``)."""


class LabelingBackendUnavailable(DataPreparerError, RuntimeError):
    """The control plane refused labeling mode itself.

    Its published SDG schema does not carry ``DatasetOptions`` yet, or labeling is not
    deployed there. A caller halts on this: there is no other labeling path.
    """


class JobWaitError(DataPreparerError, RuntimeError):
    """A job could not be described, waited on or resubmitted safely.

    The message says what to do: re-run with the same run id, or stop the job in the
    console.
    """


def _region_bound_session(region):
    """A region-bound V3 SDK ``Session`` so the job runs in ``region``, not the ambient default.

    ``DataPreparer`` has no region parameter, so the region arrives through its session.
    It tags the process as the SageMaker agent plugin (telemetry attribution).
    """
    import boto3

    _set_telemetry_attribution()
    from sagemaker.core.helper.session_helper import Session

    return Session(boto_session=boto3.Session(region_name=_s3.resolve_region(region)))


def upload_text(text, s3_uri, region, *, field, max_bytes=None):
    """Upload ``text`` to ``s3_uri`` after checking its size against ``max_bytes``.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import MAX_SPEC_BYTES, upload_text

        upload_text(spec_text, spec_uri, "us-west-2", field="Spec", max_bytes=MAX_SPEC_BYTES)

    Raises:
        LabelingRequestError: Over ``max_bytes``; names ``field`` (the service's spec and
            demos limits).
    """
    data = text.encode("utf-8")
    if max_bytes is not None and len(data) > max_bytes:
        raise LabelingRequestError(
            f"{field} is {len(data):,} bytes; the service takes at most {max_bytes:,}"
        )
    _s3.put_object(s3_uri, data, region)
    return s3_uri


# ---------------------------------------------------------------------------
# Submission: retry, re-attach, and the HALT on an un-upgraded control plane
# ---------------------------------------------------------------------------

# Retry job submission on transient account-limit / throttle errors. The account limit
# is shared by every run on the account, so the retry is bounded by a total wait
# budget with jittered backoff, and capped at _MAX_SUBMIT_ATTEMPTS.
_MAX_SUBMIT_ATTEMPTS = 40
_SUBMIT_MAX_WAIT_S = 3600
_SUBMIT_BACKOFF_BASE_S = 30
_SUBMIT_BACKOFF_CAP_S = 300
_JOB_RETRY_MARKERS = (
    "ResourceLimitExceeded",
    "ThrottlingException",
    "ThrottledException",
    "LimitExceededException",
    "TooManyRequests",
    "service limit",
    "concurrent synthetic data generation jobs",
)
_SDG_JOB_CATEGORY = "SyntheticDataGeneration"
_JOB_DONE = "Completed"
_JOB_DEAD = ("Failed", "Stopped")
_REATTACH_POLL_S = 60
_REATTACH_MAX_WAIT_S = 24 * 3600
_clock = time.monotonic
_DESCRIBE_ATTEMPTS = 5
_NOT_FOUND_CODES = ("ResourceNotFound", "ResourceNotFoundException")
_NOT_FOUND_PHRASES = ("does not exist", "not found")
# The service rejects a ModelId outside its allow-list synchronously at submit.
_UNSUPPORTED_MODEL_MARKER = "contains unsupported model"
# A job that produced nothing ends Failed with this generic message.
_INVALID_INPUT_MARKER = "invalid input or configuration"
# The control plane does not carry labeling mode: the schema gate (server side, or the
# SDK's client-side copy of the published schema) rejects the labeling fields by name.
_LABELING_FIELDS = ("DatasetOptions", "InputDatasetArn")
_SCHEMA_REJECTION_MARKERS = (
    "not allowed by the schema",
    "additionalProperties",
    "Extra inputs are not permitted",
    "extra_forbidden",
    "NumSamples",
    "schema version",
)


def _is_job_not_found(e, job_name):
    code = ((getattr(e, "response", None) or {}).get("Error") or {}).get("Code")
    if code in _NOT_FOUND_CODES or type(e).__name__ in _NOT_FOUND_CODES:
        return True
    msg = str(e).lower()
    return job_name.lower() in msg and any(p in msg for p in _NOT_FOUND_PHRASES)


def job_name_for(base_job_name):
    """A unique, pattern-valid SDG job name (<= 63 chars)."""
    base = re.sub(r"[^A-Za-z0-9-]+", "-", base_job_name or "autolabel").strip("-")
    return f"{(base or 'autolabel')[:54]}-{uuid.uuid4().hex[:8]}"


def _submit_backoff_s(attempt, rng=random):
    """Jittered exponential backoff before retry number `attempt` (1-based)."""
    ceiling = min(_SUBMIT_BACKOFF_CAP_S, _SUBMIT_BACKOFF_BASE_S * (2 ** (attempt - 1)))
    return ceiling / 2.0 + rng.uniform(0, ceiling / 2.0)


def _invalid_input_hint(msg):
    return (
        f"{msg}\n\nThe SDG service also reports this when the labeling job produced no "
        "row. Read the job's CloudWatch log for the container's reason (a spec the "
        "program could not parse, a corpus row missing a text field, a model that "
        "returned no text) and fix the input before resubmitting."
    )


def is_labeling_schema_rejection(exc):
    """True when ``exc`` (and its cause chain) is a schema-gate rejection of the labeling fields.

    The control plane this request went to does not accept labeling mode yet.
    """
    seen, e = [], exc
    while e is not None and e not in seen:
        seen.append(e)
        e = e.__cause__ or e.__context__
    hay = " ".join(f"{type(x).__name__} {x}" for x in seen)
    names_field = any(f in hay for f in _LABELING_FIELDS)
    schema_shaped = any(m.lower() in hay.lower() for m in _SCHEMA_REJECTION_MARKERS)
    return names_field and schema_shaped


def labeling_unavailable_message(exc, endpoint=None):
    """The HALT message for a control plane that rejected labeling mode."""
    where = f" at {endpoint}" if endpoint else ""
    return (
        f"The SDG control plane{where} does not accept labeling mode: its published "
        f"SyntheticDataGeneration schema rejected the labeling fields "
        f"({type(exc).__name__}: {str(exc)[:400]}).\n"
        "HALT. Labeling runs only through the SDG labeling job; do not label the rows "
        "another way (no direct Bedrock call, no inline labels). Point the SDK at a "
        "stage that carries the labeling schema (AWS_ENDPOINT_URL_SAGEMAKER) or wait "
        "for the deployment, then re-run with the same --run-id."
    )


def _describe_job(session, job_name):
    """One ``DescribeJob`` for an SDG job, or ``None`` when the service has no such job.

    Retries throttling; any other error is raised, because resubmitting a job we cannot
    see could start a duplicate.
    """
    client = session.sagemaker_client
    for attempt in range(1, _DESCRIBE_ATTEMPTS + 1):  # pragma: no branch - returns or raises
        try:
            return client.describe_job(JobName=job_name, JobCategory=_SDG_JOB_CATEGORY)
        except Exception as e:  # noqa: BLE001 - classify, then retry / None / raise
            hay = (type(e).__name__ + " " + str(e)).lower()
            if _is_job_not_found(e, job_name):
                return None
            if any(t.lower() in hay for t in _JOB_RETRY_MARKERS) and attempt < _DESCRIBE_ATTEMPTS:
                time.sleep(_submit_backoff_s(attempt))
                continue
            raise JobWaitError(
                f"could not describe the SDG job {job_name!r} ({type(e).__name__}: {e}); "
                "not resubmitting it, because a duplicate job could run. Check the job in "
                "the console, then re-run with the same --run-id."
            ) from e


def _job_arn_field(job, key):
    """``key`` from a ``DescribeJob`` response.

    Top level, else the overlay Hub finalization puts on the echoed
    ``JobConfigDocument`` once it has minted the ARN.
    """
    if not job:
        return None
    if job.get(key):
        return job[key]
    doc = job.get("JobConfigDocument")
    if isinstance(doc, str):
        try:
            doc = json.loads(doc)
        except ValueError:
            doc = None
    if isinstance(doc, dict):
        return doc.get(key) or None
    return None


def job_dataset_arn(job):
    """The labeled DataSet ARN a DescribeJob response carries (`DatasetArn`)."""
    return _job_arn_field(job, "DatasetArn")


def job_review_dataset_arn(job):
    """The review DataSet ARN a routing job's ``DescribeJob`` carries, or ``None``."""
    return _job_arn_field(job, "ReviewDatasetArn")


def _wait_for_job(session, job_name, poll_s=_REATTACH_POLL_S, max_wait_s=_REATTACH_MAX_WAIT_S):
    """Poll an in-flight job to a terminal state; returns the DescribeJob response."""
    deadline = _clock() + max_wait_s
    while True:
        job = _describe_job(session, job_name)
        status = (job or {}).get("JobStatus")
        if job is None or status in _JOB_DEAD or status == _JOB_DONE:
            return job
        if _clock() >= deadline:
            raise JobWaitError(
                f"SDG job {job_name} is still {status} after waiting {max_wait_s}s; not "
                "submitting a duplicate. Stop the job in the SageMaker console and re-run "
                "with the same --run-id, or re-run with a new --run-id."
            )
        time.sleep(min(poll_s, max(0.0, deadline - _clock())))


def classify_submit_error(e, option_fields=(), endpoint=None, job_name=None):
    """What a submit error stands for, or ``None`` when it is not a known refusal.

    A synchronous refusal means no job exists: :class:`OptionRefused` when the rejected
    key or path is an option field the caller sent (checked before the schema reading),
    :class:`LabelingBackendUnavailable`, or :class:`CreateJobRefused` (any other
    ``ValidationException``). The generic "invalid input" failure comes from a job that
    ran and ended Failed: a :class:`JobWaitError` naming the job.
    """
    msg = str(e)
    resp = getattr(e, "response", None)
    code = (resp.get("Error") or {}).get("Code", "") if isinstance(resp, dict) else ""
    path = re.sub(
        r"input_value=.*?(input_type=\w+|$)", "", msg, flags=re.S
    )  # the echoed request is no path
    if _UNSUPPORTED_MODEL_MARKER in msg:
        # The service names the allowed ids in its own message; relay it as is.
        return CreateJobRefused(msg)
    for f in option_fields:
        if re.search(rf"(^|[^\w]){f}([^\w]|$)", path) and (
            "ValidationException" in f"{code} {msg}" or f"'{f}' was unexpected" in msg
        ):
            return OptionRefused(msg)
    if is_labeling_schema_rejection(e):
        return LabelingBackendUnavailable(labeling_unavailable_message(e, endpoint))
    if (code == "ValidationException" or "(ValidationException)" in msg) and not any(
        t.lower() in msg.lower() for t in _JOB_RETRY_MARKERS
    ):  # a limit is retried, not refused
        return CreateJobRefused(
            f"CreateJob refused the request, so no job was created; the service says: "
            f"{msg}\nFix what it names and rerun; the same --run-id works."
        )
    if _INVALID_INPUT_MARKER in msg.lower():
        return JobWaitError(f"SDG job {job_name}: {_invalid_input_hint(msg)}")
    return None


def submit_labeling_job(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    document,
    *,
    region,
    role_arn=None,
    base_job_name="autolabel",
    session=None,
    submit_max_wait_s=None,
    on_submit=None,
    resume_job=None,
    endpoint=None,
    option_fields=(),
):
    """Create the labeling job and wait for it; the job record.

    Use it for every labeling job once :func:`build_labeling_request` passed.
    ``on_submit({"job_name", "output_s3_uri", "dataset_name"})`` is called *before*
    ``CreateJob`` so the caller can persist the name; a later run passes that record back
    as ``resume_job``: a Completed job is read instead of resubmitted, a running one is
    waited on, and only a Failed, Stopped or unknown job is submitted again. Never call it
    twice for the same round without ``resume_job``: that is how duplicate jobs happen.
    It tags the process as the SageMaker agent plugin (telemetry attribution) before the
    first service call.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import submit_labeling_job

        job = submit_labeling_job(document, region="us-west-2", role_arn=role_arn,
                                  on_submit=lambda text: open("proj/job/dev/job.json", "w").write(text))
        job["status"], job["dataset_arn"]

    Args:
        document: :func:`build_labeling_request`'s return.
        region: The region the job runs in.
        role_arn: The job's execution role; the SDK's default when ``None``.
        base_job_name: The job name's prefix; a unique suffix is added.
        session: A region-bound session, when the caller has one.
        submit_max_wait_s: The budget for retrying transient account-limit and throttle
            errors (jittered backoff), one hour by default.
        on_submit: Called with the job record before ``CreateJob``.
        resume_job: An earlier run's record, to re-attach.
        endpoint: The control-plane endpoint, named in a HALT message.
        option_fields: The opt-in request fields sent, so a refusal naming one is an
            :class:`OptionRefused`.

    Returns:
        ``{"job_name", "job_arn", "job_status", "dataset_arn", "review_dataset_arn",
        "output_s3_uri", "reattached", "failure_reason"}``.

    Raises:
        CreateJobRefused: The service refused the request; no job exists.
        LabelingBackendUnavailable: The control plane does not carry labeling mode; halt.
        JobWaitError: A job could not be described or waited on safely; the message says
            how to re-attach.
    """
    DataPreparer = _data_preparer_class()  # pylint: disable=invalid-name
    _set_telemetry_attribution()
    if session is None:
        session = _region_bound_session(region)
    output_s3_uri = document["OutputDataConfig"]["OutputS3Uri"]
    if resume_job and resume_job.get("job_name"):
        name = resume_job["job_name"]
        job = _describe_job(session, name)
        status = (job or {}).get("JobStatus")
        if job is not None and status not in _JOB_DEAD:
            if status != _JOB_DONE:
                _log.warning(
                    "SDG job %s from an earlier run is still %s; waiting for it "
                    "instead of submitting a duplicate",
                    name,
                    status,
                )
                job = _wait_for_job(session, name)
                status = (job or {}).get("JobStatus")
            if job is not None and status == _JOB_DONE:
                _log.info("reusing the output of SDG job %s submitted by an earlier run", name)
                return _job_record(job, name, output_s3_uri, reattached=True)
        _log.warning(
            "in-flight SDG job %s from an earlier run is %s; submitting it again",
            name,
            status or "not found",
        )

    job_name = job_name_for(base_job_name)
    if on_submit is not None:
        on_submit(
            {
                "job_name": job_name,
                "output_s3_uri": output_s3_uri,
                "dataset_name": document["DatasetName"],
            }
        )
    cfg = to_dataset_job_config(document)
    dp_kwargs = {"base_job_name": base_job_name, "sagemaker_session": session}
    if role_arn:
        dp_kwargs["role"] = role_arn
    budget = _SUBMIT_MAX_WAIT_S if submit_max_wait_s is None else max(0, submit_max_wait_s)
    waited = 0.0
    for attempt in range(1, _MAX_SUBMIT_ATTEMPTS + 1):  # pragma: no branch - returns or raises
        try:
            ds = DataPreparer(**dp_kwargs).generate(cfg, job_name=job_name)
            break
        except Exception as e:  # noqa: BLE001 - classify, re-raise if not transient
            ended = _job_ended_record(e, session, job_name, output_s3_uri)
            if ended is not None:
                return ended
            msg = str(e)
            refused = classify_submit_error(e, option_fields, endpoint, job_name)
            if refused is not None:
                raise refused from e
            cause = e.__cause__ or e.__context__
            hay = " ".join(f"{type(x).__name__} {x}" for x in (e, cause) if x is not None).lower()
            transient = any(t.lower() in hay for t in _JOB_RETRY_MARKERS)
            if not transient or attempt == _MAX_SUBMIT_ATTEMPTS:
                raise
            delay = min(_submit_backoff_s(attempt), max(0.0, budget - waited))
            if delay <= 0:
                raise
            _log.warning(
                "SDG CreateJob for %s hit a transient limit (attempt %d, waited "
                "%.0fs of %ss); retrying in %.0fs: %s",
                document["DatasetName"],
                attempt,
                waited,
                budget,
                delay,
                msg[:200],
            )
            waited += delay
            time.sleep(delay)
    job = _describe_job(session, job_name) or {}
    record = _job_record(job, job_name, output_s3_uri, reattached=False)
    # The SDK rehydrates the DataSet by name; its ARN is the fallback when DescribeJob
    # carries none.
    record["dataset_arn"] = record["dataset_arn"] or getattr(ds, "arn", None)
    return record


# DataPreparer.generate(wait=True) raises these (sagemaker.ai_registry exceptions) when the
# job it waited on ends Failed or Stopped, or the wait times out; it returns only on success.
_SDK_JOB_ENDED = {"DataPreparerJobFailedError": "Failed", "DataPreparerJobStoppedError": "Stopped"}
_SDK_JOB_TIMEOUT = "DataPreparerJobTimeoutError"


def _job_ended_record(e, session, job_name, output_s3_uri):
    """The job record for an SDK error that reports the submitted job ended Failed or Stopped.

    ``None`` for any other error. A wait timeout raises :class:`JobWaitError`: the job may
    still be running, so it is re-attached by the same run id, never resubmitted.
    """
    names = {c.__name__ for c in type(e).__mro__}
    if _SDK_JOB_TIMEOUT in names:
        raise JobWaitError(
            f"SDG job {job_name} did not finish within the SDK's wait ({e}); it may still be "
            "running. Re-run with the same --run-id and --stage to wait for it again; that "
            "does not submit a duplicate."
        ) from e
    status = next((v for k, v in _SDK_JOB_ENDED.items() if k in names), None)
    if status is None:
        return None
    try:
        job = _describe_job(session, job_name) or {}
    except JobWaitError:
        job = {}
    record = _job_record(job, job_name, output_s3_uri, reattached=False)
    record["job_status"] = job.get("JobStatus") or status
    record["job_arn"] = record["job_arn"] or getattr(e, "job_arn", None)
    record["failure_reason"] = (
        record["failure_reason"] or getattr(e, "failure_reason", None) or str(e)
    )
    return record


def _job_record(job, job_name, output_s3_uri, *, reattached):
    job = job or {}
    return {
        "job_name": job_name,
        "job_arn": job.get("JobArn"),
        "job_status": job.get("JobStatus"),
        "dataset_arn": job_dataset_arn(job),
        "review_dataset_arn": job_review_dataset_arn(job),
        "output_s3_uri": output_s3_uri,
        "reattached": reattached,
        "failure_reason": job.get("FailureReason"),
    }


# ---------------------------------------------------------------------------
# Reading the result
# ---------------------------------------------------------------------------


def _data_preparer_class():
    """The SDK's ``DataPreparer`` (one place, so tests can stub the submission)."""
    from sagemaker.ai_registry.data_preparer.preparer import DataPreparer

    return DataPreparer


def _labeling_summary_class():
    """The SDK's ``LabelingSummary`` (one place, so tests can stub the read-back)."""
    from sagemaker.ai_registry.data_preparer.auto_labeling import LabelingSummary

    return LabelingSummary


# The fields `LabelingSummary.from_dataset` resolves itself, not part of the container's
# labeling_summary.json; `read_job_results` returns them under their own keys.
_SUMMARY_RESOLVED_FIELDS = frozenset({"artifacts", "labeled_dataset_arn", "review_dataset_arn"})


def read_job_results(*, region, dataset_arn, dest_dir, names=None, review_dataset_arn=None):
    """Read a finished labeling job's result through the SDK and download its artifacts.

    ``LabelingSummary.from_dataset`` walks the Hub pointer (DataSet document ->
    ``JobMetadata.MetadataS3Uri`` -> ``labeling/index.json`` -> ``labeling_summary.json``)
    and resolves the review DataSet ARN from the labeled entry's keywords; this downloads
    the artifacts the index names (all, or ``names``) under ``dest_dir`` with their
    contract file names (``INDEX_ARTIFACTS``).

    Args:
        region: The region.
        dataset_arn: The labeled DataSet ARN the job registered.
        dest_dir: Where to download the artifacts.
        names: The artifact names to download; all when ``None``.
        review_dataset_arn: ``DescribeJob``'s ``ReviewDatasetArn`` when the caller has it.

    Returns:
        A dict with ``index_uri``, ``files`` (artifact name to local path), ``summary``,
        ``labeled_dataset_arn``, ``review_dataset_arn`` and ``artifact_uris``; ``summary``
        is the container's ``labeling_summary.json`` document as written (PascalCase keys).

    Raises:
        FileNotFoundError: No DataSet ARN, or the DataSet carries no labeling result to
            read; the job may have failed before the finalizer wrote its summary.
    """
    if not dataset_arn:
        raise FileNotFoundError(
            "the job returned no labeled DataSet ARN, so its labeling result cannot be "
            "located. The job may have failed before the finalizer registered its output; "
            "read its FailureReason."
        )
    LabelingSummary = _labeling_summary_class()  # pylint: disable=invalid-name
    try:
        summary = LabelingSummary.from_dataset(
            dataset_arn, sagemaker_session=_region_bound_session(region)
        )
    except Exception as e:  # noqa: BLE001 - the SDK's DataPreparerError, or a Describe failure
        raise FileNotFoundError(
            f"cannot read the labeling result of {dataset_arn}: {type(e).__name__}: {e}. The "
            "job may have failed before the finalizer wrote its summary; read its "
            "FailureReason."
        ) from e
    uris = {name: uri for name, uri in (summary.artifacts or {}).items() if uri}
    if not uris.get("LabelingSummary"):
        raise FileNotFoundError(f"the labeling index of {dataset_arn} names no summary")
    index_uri = uris["LabelingSummary"].rsplit("/", 1)[0] + "/" + INDEX_FILENAME
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, uri in uris.items():
        if names and name not in names:
            continue
        local = dest / INDEX_ARTIFACTS.get(name, uri.rsplit("/", 1)[-1])
        _s3.download_file(uri, local, region)
        files[name] = str(local)
    document = summary.model_dump(
        by_alias=True, exclude_none=True, exclude=_SUMMARY_RESOLVED_FIELDS
    )
    return {
        "index_uri": index_uri,
        "files": files,
        "summary": document,
        "labeled_dataset_arn": summary.labeled_dataset_arn or dataset_arn,
        "review_dataset_arn": review_dataset_arn or summary.review_dataset_arn,
        "artifact_uris": uris,
    }
