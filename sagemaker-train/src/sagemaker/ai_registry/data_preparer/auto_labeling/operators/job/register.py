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
"""The Hub doors of an auto-labeling project: dataset references in, labeled output out.

The labeling job itself reads S3 only, so a project meets the AI Registry Hub at its
edges. This module is those edges, as functions:

* :func:`resolve_dataset_ref` turns a local path, an ``s3://`` URI or a DataSet ARN into
  the object the pipeline reads. A remote reference stops at the resolved S3 URI (one
  ``HeadObject`` for its ETag and size, so the caller can write a pointer) unless the
  caller asks for a download, which is for a client-readable seed or sample, never the
  corpus. :func:`corpus_ref_document` builds that pointer.
* :func:`register_labeled_output` uploads a labels file (and its reports) to a per-run S3
  prefix and imports a new version of the ``<name>-labeled`` DataSet, after a lineage
  guard that refuses to append onto an unrelated entry and a consent check so appending
  is always deliberate. After the uploads a Hub failure degrades (``hub_registered`` false,
  the S3 copy is the deliverable) instead of failing the run.

The dataset-transformation doors are :mod:`.dt_handoff`; the scratch versions a project
registers along the way are :mod:`._scratch`.

Every refusal is a :class:`~sagemaker.ai_registry.data_preparer.exceptions.DataPreparerOperatorError`
raised before any upload or import. Notes for the person running it (a Hub that was
unreachable, an Admin caller whose validation manifest will show an error) go to this
module's logger. The HubContentDocument carries only the fields the Hub's v4 document
schema knows; provenance rides ``HubContentDescription`` and lowercase search keywords.
"""

from __future__ import absolute_import

import json
import logging
import re
import time
import uuid
from typing import Any, Dict, Optional, Tuple

from sagemaker.core.jumpstart.hub.utils import get_info_from_hub_resource_arn

from sagemaker.ai_registry.air_hub import AIRHub
from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _digest
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _s3
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.client import _NOT_FOUND_CODES

_log = logging.getLogger(__name__)

DOCUMENT_SCHEMA_VERSION = "4.0.0"
DATASET_FILE_FORMAT = "Custom"  # labeled JSONL; serializes to UNKNOWN on the wire enum

# Fields the service writes after an import; a new version's document leaves them out.
_SERVICE_FIELDS = ("ValidationManifest",)

# Provenance rides HubContentDescription, not the document: the Hub v4 document schema
# rejects unknown fields at import. The description allows any text up to 1023 chars and
# is returned by both describe and list. Search keywords forbid uppercase, so an ARN
# cannot be one.
_DESC_PREFIX = "auto-labeled"
_DESC_SOURCE_RE = re.compile(r"source=([^;]+)")
_DESC_LABELS_RE = re.compile(r"labels-sha256=([0-9a-f]+)")
_HUB_NAME_RE = re.compile(r"^[a-zA-Z0-9](-*[a-zA-Z0-9])*$")


# ---------------------------------------------------------------------------
# clients (module-level, so tests stub them)
# ---------------------------------------------------------------------------


def _default_region():
    """The boto session's configured region, else ``us-west-2``."""
    return _s3.resolve_region(None)


def _sts_account(region):
    import boto3

    return boto3.client("sts", region_name=region).get_caller_identity()["Account"]


def _sts_caller_arn(region):
    import boto3

    return boto3.client("sts", region_name=region).get_caller_identity()["Arn"]


def _not_found(e):
    return e.response.get("Error", {}).get("Code", "") in _NOT_FOUND_CODES


def _sm_client(region):
    import boto3

    return boto3.client("sagemaker", region_name=region)


def ai_registry_hub_name(account_id, region):
    """The account's default AI Registry hub, where a job with no ``HubName`` registers.

    :meth:`~sagemaker.ai_registry.air_hub.AIRHub.hub_name_for`, with the arguments in the
    order the labeling operators pass them.
    """
    return AIRHub.hub_name_for(region, account_id)


def existing_datasets(names, region, hub_name=None):
    """The DataSets among ``names`` that already exist in the hub the job registers into.

    Hub finalization adds a version to an existing name, so a corpus run whose dataset
    name or review dataset name is taken would put its rows on another DataSet as its
    latest version. A hub that does not exist yet holds nothing.

    Returns:
        ``[{name, hub, version, arn}]``.
    """
    import boto3

    sm = _sm_client(region)
    hub = hub_name or ai_registry_hub_name(
        boto3.client("sts", region_name=region).get_caller_identity()["Account"], region
    )
    found = []
    for name in dict.fromkeys(n for n in names if n):
        try:
            resp = sm.describe_hub_content(
                HubName=hub, HubContentType="DataSet", HubContentName=name
            )
        except Exception as e:  # noqa: BLE001 - only "not found" means the name is free
            code = (
                getattr(e, "response", {}).get("Error", {}).get("Code", "")
                if hasattr(e, "response")
                else ""
            )
            if (
                code in ("ResourceNotFound", "ResourceNotFoundException")
                or "ResourceNotFound" in f"{type(e).__name__} {e}"
            ):
                continue
            raise
        found.append(
            {
                "name": name,
                "hub": hub,
                "version": resp.get("HubContentVersion"),
                "arn": resp.get("HubContentArn"),
            }
        )
    return found


def delete_hub_content_version(arn, region=None):
    """Delete one DataSet version from the Hub (scratch cleanup).

    The catalog entry goes, the S3 files stay. The ARN must carry the version: a name
    alone could delete another run's entry.

    Raises:
        DataPreparerOperatorError: Not a DataSet ARN, or no version.
    """
    hub, ctype, name, version, arn_region = parse_hub_content_arn(arn)
    if ctype != "DataSet":
        raise DataPreparerOperatorError(f"{arn} is not a DataSet ARN")
    if not version:
        raise DataPreparerOperatorError(
            f"{arn} has no version; cleanup deletes one version, never a name"
        )
    _sm_client(region or arn_region).delete_hub_content(
        HubName=hub, HubContentType="DataSet", HubContentName=name, HubContentVersion=version
    )
    return {"deleted": arn, "hub": hub, "name": name, "version": version}


# ---------------------------------------------------------------------------
# hub-content ARNs
# ---------------------------------------------------------------------------


def parse_hub_content_arn(arn: str) -> Tuple[str, str, str, Optional[str], str]:
    """``(hub, type, name, version or None, region)`` of a Hub content ARN.

    Delegates to sagemaker-core's shared hub-ARN parser rather than splitting on ``/``
    here, so the accepted shape stays in one place:
    ``arn:{partition}:sagemaker:{region}:{account}:hub-content/{hub}/{type}/{name}[/{version}]``.
    That parser expects the version segment, so an ARN without one is read with an
    empty version appended.

    Raises:
        DataPreparerOperatorError: Not a hub-content ARN.
    """
    text = arn or ""
    if ":hub-content/" in text and text.count("/") == 3:
        text += "/"
    info = get_info_from_hub_resource_arn(text)
    if (
        info is None
        or not info.hub_content_name
        or not info.hub_content_type
        or "/" in (info.hub_content_version or "")
    ):
        raise DataPreparerOperatorError(f"not a hub-content ARN: {arn!r}")
    return (
        info.hub_name,
        info.hub_content_type,
        info.hub_content_name,
        info.hub_content_version or None,
        info.region,
    )


def parse_dataset_arn(ref):
    """Split a DataSet HubContent ARN.

    HubContent ARNs are region-scoped, so the ARN's own region is the one to resolve it in.

    Args:
        ref: ``arn:aws:sagemaker:<region>:<account>:hub-content/<hub>/DataSet/<name>[/<version>]``.

    Returns:
        ``(hub, name, version or None, region)``.

    Raises:
        DataPreparerOperatorError: Not a hub-content ARN, or not a DataSet one.
    """
    hub, ctype, name, version, region = parse_hub_content_arn(ref)
    if ctype != "DataSet":
        raise DataPreparerOperatorError(f"hub-content type is {ctype!r}, expected DataSet: {ref}")
    return hub, name, version, region


def _s3_from_document(doc):
    """The dataset's S3 location from a HubContentDocument (URI or bucket+prefix)."""
    if doc.get("DatasetS3Uri"):
        return doc["DatasetS3Uri"]
    if doc.get("DatasetS3Bucket"):
        prefix = (doc.get("DatasetS3Prefix") or "").strip("/")
        return f"s3://{doc['DatasetS3Bucket']}/{prefix}"
    raise DataPreparerOperatorError(
        "HubContentDocument has neither DatasetS3Uri nor DatasetS3Bucket"
    )


def resolve_dataset_ref(ref, out_path, region, download=False):
    """Resolve a local path, an ``s3://`` URI or a DataSet ARN to what the pipeline reads.

    A local ``ref`` is copied to ``out_path``. A remote one stops at the resolved S3 URI,
    with the object's ETag and size from one ``HeadObject``, so the caller writes a
    pointer (:func:`corpus_ref_document`) and the corpus stays in the bucket; with
    ``download`` it is downloaded to ``out_path`` instead, which is for a client-readable
    seed or sample, never the corpus. An ARN is resolved in its own region, whatever
    ``region`` says: HubContent ARNs are region-scoped.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import register

        provenance = register.resolve_dataset_ref(
            "arn:aws:sagemaker:us-west-2:123456789012:hub-content/H/DataSet/tickets", out_path=None, region="us-west-2"
        )
        pointer = register.corpus_ref_document(provenance)  # corpus.ref.json: the corpus stays in the bucket

    Args:
        ref: A local path, ``s3://bucket/key.jsonl`` (or a prefix holding one ``.jsonl``),
            or a DataSet HubContent ARN, version-pinned or not.
        out_path: Where a local copy or a download lands; unused for a remote ``ref``
            without ``download``.
        region: The region for an ``s3://`` ``ref``.
        download: Download a remote ``ref`` to ``out_path``.

    Returns:
        The provenance: ``{}`` for a local ``ref``; ``{"resolved_s3_uri"}`` plus ``etag`` and
        ``size`` (when not downloading) for an ``s3://`` one; for an ARN also
        ``source_arn``, ``source_name``, ``resolved_version`` and ``dataset_file_format``.

    Raises:
        DataPreparerOperatorError: A bad ARN, a document without an S3 location, or a
            prefix that does not hold exactly one ``.jsonl``.
    """
    provenance: Dict[str, Any] = {}
    if ref.startswith("arn:"):
        hub, name, version, arn_region = parse_dataset_arn(ref)
        region = arn_region
        kwargs = dict(HubName=hub, HubContentType="DataSet", HubContentName=name)
        if version:
            kwargs["HubContentVersion"] = version
        resp = _sm_client(region).describe_hub_content(**kwargs)
        doc = json.loads(resp["HubContentDocument"])
        s3_uri = _s3.resolve_object(_s3_from_document(doc), region)
        provenance = {
            "source_arn": ref,
            "source_name": name,
            "resolved_version": resp.get("HubContentVersion", version),
            "dataset_file_format": doc.get("DatasetFileFormat"),
            "resolved_s3_uri": s3_uri,
        }
        ref = s3_uri
    if ref.startswith("s3://"):
        resolved = _s3.resolve_object(ref, region)
        bucket, key = _s3.split_s3_uri(resolved)
        provenance.setdefault("resolved_s3_uri", f"s3://{bucket}/{key}")
        if not download:
            provenance.update(_s3.head_object(resolved, region))
            return provenance
        _s3.s3_client(region).download_file(bucket, key, str(out_path))
        return provenance
    with open(ref, "rb") as src, open(out_path, "wb") as dst:
        dst.write(src.read())
    return provenance


def corpus_ref_document(provenance, resolved_at=None):
    """The ``corpus.ref.json`` pointer a resolve without download leaves in place of the body.

    Args:
        provenance: :func:`resolve_dataset_ref`'s return value for a remote reference.
        resolved_at: The pointer's timestamp (``%Y-%m-%dT%H:%M:%SZ``); now when ``None``.

    Returns:
        ``{"source_s3_uri", "etag", "size", "hub_arn", "hub_version", "resolved_at"}``.
    """
    return {
        "source_s3_uri": provenance.get("resolved_s3_uri"),
        "etag": provenance.get("etag"),
        "size": provenance.get("size"),
        "hub_arn": provenance.get("source_arn"),
        "hub_version": provenance.get("resolved_version"),
        "resolved_at": resolved_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ---------------------------------------------------------------------------
# register (output edge)
# ---------------------------------------------------------------------------


def _canonical_ai_registry_hub_name(account_id, region):
    """The account's AI Registry hub name (the same derivation as data-quality's)."""
    return ai_registry_hub_name(account_id, region)


def _resolve_or_create_hub(client, account_id, region, override=None):
    """The hub to import into: the override (must exist) or the canonical one, created if missing."""
    from botocore.exceptions import ClientError

    if override:
        try:
            client.describe_hub(HubName=override)
        except ClientError as e:
            if _not_found(e):
                # A config error, not a transient Hub problem: fail before any upload.
                raise DataPreparerOperatorError(
                    f"Hub '{override}' not found. Create it first, or drop the "
                    f"override to use the canonical AiRegistry hub."
                ) from e
            raise
        return override
    canonical = _canonical_ai_registry_hub_name(account_id, region)
    try:
        client.describe_hub(HubName=canonical)
        return canonical
    except ClientError as e:
        if not _not_found(e):
            raise
    display = AIRHub.hub_display_name_for(region, account_id)
    client.create_hub(
        HubName=canonical,
        HubDescription=display,
        HubDisplayName=display,
        HubSearchKeywords=["AiRegistry"],
    )
    return canonical


def _label_set_digest(label_set):
    """A short stable digest of a label set for lineage comparison (12 hex chars)."""
    return _digest.sha256_json(sorted(label_set))[:12]


def _build_description(source_arn, label_digest, run_id):
    parts = [_DESC_PREFIX]
    if source_arn:
        parts.append(f"source={source_arn}")
    if label_digest:
        parts.append(f"labels-sha256={label_digest}")
    parts.append(f"run-id={run_id}")
    return "; ".join(parts)[:1023]


def _normalize_source(arn):
    """A version-insensitive identity of a source dataset.

    ``hub/name`` for a hub-content ARN (a re-label of the same dataset resolved at another
    version is the same lineage), the raw string otherwise.
    """
    if not arn:
        return arn
    try:
        hub, name, _version, _region = parse_dataset_arn(arn)
        return f"{hub}/{name}"
    except ValueError:
        return arn


def _append_guard(client, hub, content_name, source_arn, label_digest):
    """The lineage check before appending to an existing content name.

    Its latest version's description must carry the auto-labeled marker, the same source
    and the same label-set digest when both are known.

    Returns:
        ``("new", None)`` or ``("append", <latest version>)``.
    """
    from botocore.exceptions import ClientError

    try:
        resp = client.describe_hub_content(
            HubName=hub, HubContentType="DataSet", HubContentName=content_name
        )
    except ClientError as e:
        if _not_found(e):
            return "new", None
        raise
    desc = resp.get("HubContentDescription") or ""
    if not desc.startswith(_DESC_PREFIX):
        raise DataPreparerOperatorError(
            f"'{content_name}' already exists but was not registered by this "
            f"skill (no auto-labeled provenance marker); registering this run "
            f"there would append onto an unrelated dataset. Use a different --name."
        )
    m = _DESC_SOURCE_RE.search(desc)
    prior_source = m.group(1).strip() if m else None
    m = _DESC_LABELS_RE.search(desc)
    prior_digest = m.group(1) if m else None
    if (
        source_arn
        and prior_source
        and _normalize_source(source_arn) != _normalize_source(prior_source)
    ):
        raise DataPreparerOperatorError(
            f"'{content_name}' already exists with a different source dataset "
            f"({prior_source}); registering this run there would append a "
            f"wrong-lineage version. Use a different --name."
        )
    if label_digest and prior_digest and label_digest != prior_digest:
        raise DataPreparerOperatorError(
            f"'{content_name}' already exists with a different label set "
            f"(digest {prior_digest}); use a different --name."
        )
    return "append", resp.get("HubContentVersion")


def _require_append_consent(verdict, prior_version, content_name, allow_append):
    """Appending is deliberate: even with matching lineage the caller must opt in."""
    if verdict == "append" and not allow_append:
        raise DataPreparerOperatorError(
            f"'{content_name}' already exists (latest version {prior_version}) "
            f"with matching lineage. Confirm with the user whether this run "
            f"should be a new version of that dataset -- if yes, re-run with "
            f"--append; if it should be a separate dataset, pick a different "
            f"--name."
        )


def _validate_content_name(name):
    """``<name>-labeled``, or a refusal with a suggested name (never a silent rename)."""
    content_name = f"{name}-labeled"
    if _HUB_NAME_RE.match(content_name):
        return content_name
    suggestion = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    raise DataPreparerOperatorError(
        f"name '{name}' would produce Hub content name '{content_name}', which "
        f"violates the Hub naming pattern (alphanumerics and hyphens only). "
        f"Confirm a valid name with the user -- e.g. --name {suggestion}"
    )


def _validate_hub_name(name):
    """A Hub content name: 1 to 63 letters, digits and hyphens, starting and ending with one."""
    if len(name) > 63 or not _HUB_NAME_RE.match(name):
        raise DataPreparerOperatorError(
            f"'{name}' is not a valid Hub content name (1 to 63 letters, "
            f"digits and hyphens, starting and ending with a letter or digit)"
        )


def _warn_if_untrusted_caller(region):
    """Log a note when the caller is the Admin role: its validation manifest will show an error."""
    try:
        arn = _sts_caller_arn(region)
    except Exception:  # noqa: BLE001 - advisory only
        return
    if ":assumed-role/Admin/" in arn or ":role/Admin" in arn:
        _log.warning(
            "NOTE: registering as the Admin role. The service runs an automatic "
            "data-quality validation job under your role after import; Admin "
            "does not trust that service, so the content's ValidationManifest "
            "will show JobStatus ERROR (harmless, but noisy). For a clean "
            "manifest, run register with credentials for "
            "AmazonSageMakerBedrockExecutionRole."
        )


def register_labeled_output(  # pylint: disable=too-many-locals,too-many-arguments,too-many-branches,too-many-statements
    *,
    labels_path,
    name,
    s3_prefix,
    region,
    report_path=None,
    report_html_path=None,
    provenance=None,
    label_set=None,
    hub_name_override=None,
    run_id=None,
    extra_doc=None,
    allow_append=False,
):
    """Upload labeled rows to a per-run S3 prefix and import them as ``<name>-labeled``.

    The order is the safety rail. First the lineage guard and the consent check, with
    nothing uploaded yet: an existing ``<name>-labeled`` must carry this module's
    provenance marker, the same source and label set, and ``allow_append`` must be set.
    Then the bytes, under ``<s3_prefix>/labeled/<name>/<run_id>/`` so a re-registration
    never overwrites a prior version's files. Then the import, with the version omitted
    (the service increments). A Hub failure after the uploads, or a Hub the guard could
    not reach, gives ``hub_registered`` false with ``hub_error``: the S3 copy is the
    deliverable, and a version whose lineage was never verified is not appended.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import register

        res = register.register_labeled_output(
            labels_path="proj/labels.jsonl", name="tickets", s3_prefix="s3://bucket/prefix",
            region="us-west-2", label_set=["TP", "FP"], run_id="r-01",
        )
        res["dataset_arn"] if res["hub_registered"] else res["hub_error"]

    Args:
        labels_path: The local ``labels.jsonl``.
        name: The dataset's base name (from the conversation, not a file name).
        s3_prefix: ``s3://bucket/prefix`` for the outputs.
        region: The region.
        report_path: The local ``held_out_report.json``, uploaded alongside.
        report_html_path: The local ``report.html``; when given, the document's
            ``ValidationReportS3Uri`` points at it rather than the JSON.
        provenance: :func:`resolve_dataset_ref`'s dict for the input (its ``source_arn``
            is the lineage).
        label_set: The label names; their digest is part of the lineage.
        hub_name_override: A hub to import into instead of the account's canonical one;
            it must exist.
        run_id: The run's id; a timestamp plus a random suffix when ``None``.
        extra_doc: Further HubContentDocument fields (v4 schema only).
        allow_append: Consent to add a version to an existing ``<name>-labeled``.

    Returns:
        ``{"labels_s3_uri", "report_s3_uri", "report_json_s3_uri", "report_html_s3_uri",
        "run_id", "hub_registered"}`` plus, when registered, ``hub_name``, ``content_name``
        and ``dataset_arn``; otherwise ``hub_error``.

    Raises:
        DataPreparerOperatorError: An invalid name, a missing override hub, a lineage
            mismatch, or an append without consent; nothing was uploaded.
    """
    provenance = provenance or {}
    run_id = run_id or time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    content_name = _validate_content_name(name)
    label_digest = _label_set_digest(label_set) if label_set else None
    source_arn = provenance.get("source_arn")

    # 0) the lineage check, before any upload; an unreachable Hub skips registration.
    sm = hub = None
    precheck_error = None
    try:
        sm = _sm_client(region)
        hub = _resolve_or_create_hub(sm, _sts_account(region), region, hub_name_override)
        verdict, prior_version = _append_guard(sm, hub, content_name, source_arn, label_digest)
        _require_append_consent(verdict, prior_version, content_name, allow_append)
    except ValueError as e:
        raise DataPreparerOperatorError(f"{e} (nothing was uploaded)") from None
    except Exception as e:  # noqa: BLE001 - Hub is best-effort until after S3
        precheck_error = f"{type(e).__name__}: {e}"
        sm = hub = None

    # 1) bytes first: per-run prefix so a re-register never overwrites a prior version
    base = s3_prefix.rstrip("/") + f"/labeled/{name}/{run_id}"
    labels_uri = base + "/labels.jsonl"
    _s3.upload_file(labels_path, labels_uri, region)
    report_json_uri = None
    if report_path:
        report_json_uri = base + "/held_out_report.json"
        _s3.upload_file(report_path, report_json_uri, region)
    report_html_uri = None
    if report_html_path:
        report_html_uri = base + "/report.html"
        _s3.upload_file(report_html_path, report_html_uri, region)
    report_uri = report_html_uri or report_json_uri  # HTML preferred for the pointer
    result: Dict[str, Any] = {
        "labels_s3_uri": labels_uri,
        "report_s3_uri": report_uri,
        "report_json_s3_uri": report_json_uri,
        "report_html_s3_uri": report_html_uri,
        "run_id": run_id,
        "hub_registered": False,
    }

    # 2) Hub registration, only with a guard that was actually evaluated.
    if sm is None or hub is None:
        result["hub_error"] = (
            f"Hub unreachable during the lineage pre-check ({precheck_error}); "
            f"registration skipped -- never appending an unverified version. "
            f"Re-run register once the Hub is reachable."
        )
        _log.warning("labeled data is in S3 (%s) but %s", labels_uri, result["hub_error"])
        return result
    try:
        _warn_if_untrusted_caller(region)
        bucket, key = _s3.split_s3_uri(labels_uri)
        document = {
            "DatasetFileFormat": DATASET_FILE_FORMAT,
            "DatasetS3Bucket": bucket,
            "DatasetS3Prefix": key.rsplit("/", 1)[0],
            "DatasetS3Uri": labels_uri,
        }
        if report_uri:
            document["ValidationReportS3Uri"] = report_uri
        document.update(extra_doc or {})
        keywords = ["auto-labeled"]
        if label_digest:
            keywords.append(f"labels-sha256:{label_digest}")
        if source_arn:
            keywords.append(
                f"source-name:{provenance.get('source_name', '').lower()}"
                if provenance.get("source_name")
                else "source:hub-dataset"
            )
        arn = sm.import_hub_content(
            HubName=hub,
            HubContentName=content_name,
            HubContentType="DataSet",
            DocumentSchemaVersion=DOCUMENT_SCHEMA_VERSION,
            HubContentDescription=_build_description(source_arn, label_digest, run_id),
            HubContentSearchKeywords=keywords[:50],
            HubContentDocument=json.dumps(document),
        )[
            "HubContentArn"
        ]  # HubContentVersion omitted -> service auto-increments
        result.update(
            {
                "hub_registered": True,
                "hub_name": hub,
                "content_name": content_name,
                "dataset_arn": arn,
            }
        )
    except Exception as e:  # noqa: BLE001 - degradation contract: Hub is best-effort
        result["hub_error"] = f"{type(e).__name__}: {e}"
        _log.warning(
            "labeled data is in S3 (%s) but Hub registration failed: %s",
            labels_uri,
            result["hub_error"],
        )
    return result
