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
"""The dataset-transformation doors of an auto-labeling project.

The labeling loop drives a DT overlay over its labeled output; these are the Hub
operations that overlay needs: :func:`dt_view` registers a scratch view as a DT job's
first input, :func:`register_transform_output` registers the job's output under a new
name, :func:`promote_transform_output` adds the overlay's output as the labeled DataSet's
next version, and :func:`register_s3_object` registers an ``s3://`` object without reading
it. Metadata only: nothing here reads a row.

This is the first boundary to revisit when the transformation primitive has operators of
its own.
"""

from __future__ import absolute_import

import json
import logging

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _s3
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import register
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.register import (
    DATASET_FILE_FORMAT,
    DOCUMENT_SCHEMA_VERSION,
    _SERVICE_FIELDS,
    _not_found,
    _s3_from_document,
    _validate_hub_name,
    hub_arn_from_dataset_arn,
    parse_dataset_arn,
)

_log = logging.getLogger(__name__)

DT_VIEW_SUFFIX = "-dt-input"
DT_VIEW_KEYWORD = "auto-labeling-dt-input"


def _view_name(name):
    """The dt-view name of a DataSet named ``name``: ``<name>-dt-input``, within 63."""
    return f"{name[:63 - len(DT_VIEW_SUFFIX)].rstrip('-')}{DT_VIEW_SUFFIX}"


def _version_kwargs(version):
    return {"HubContentVersion": version} if version else {}


def _require_new_name(sm, hub, name, hint):
    """Refuse when ``name`` exists in ``hub``: a new name is never appended to."""
    from botocore.exceptions import ClientError

    try:
        sm.describe_hub_content(HubName=hub, HubContentType="DataSet", HubContentName=name)
    except ClientError as e:
        if _not_found(e):
            return
        raise
    raise DataPreparerOperatorError(f"'{name}' already exists in the hub; {hint}")


def promote_transform_output(*, from_arn, to_arn, region=None):  # pylint: disable=too-many-locals
    """Add the overlay's output as the labeled DataSet's next version.

    ``to_arn``'s document, description and keywords plus the output's ``Transformation*``
    fields, ``DatasetS3Uri`` at the output's ``.jsonl``, at ``to_arn``'s
    ``DocumentSchemaVersion``. Metadata only: nothing is read.

    Args:
        from_arn: The overlay job's ``OutputDatasetArns`` entry, a version of ``to_arn``'s
            dt-view.
        to_arn: The labeled DataSet version the view was taken of.
        region: The region; the ARN's own when ``None``.

    Returns:
        ``{"dataset_arn", "source_arn", "transform_output_arn", "dataset_s3_uri"}``.

    Raises:
        DataPreparerOperatorError: ``from_arn`` is not on ``to_arn``'s view, or the
            labeled DataSet's latest version is not ``to_arn``'s (someone added a version
            after the overlay was built against ``to_arn``'s rows). Nothing imported.
    """
    hub, name, version, arn_region = parse_dataset_arn(to_arn)
    f_hub, f_name, f_version, _r = parse_dataset_arn(from_arn)
    # NOTE: f_hub is a bare HubName from parse_dataset_arn(from_arn); DescribeHubContent
    # resolves it in the caller's account, so a cross-account SOURCE arn reads the wrong
    # hub below.  `hub` comes from to_arn and names the local DESTINATION, so its bare
    # form is correct -- do not "fix" it.  The source-side defect is fixed in
    # resolve_dataset_ref and dt_view via hub_arn_from_dataset_arn; this one site is
    # knowingly unaddressed pending the owning team's decision.
    if (f_hub, f_name) != (hub, _view_name(name)):
        raise DataPreparerOperatorError(
            f"{from_arn} is not a version of '{_view_name(name)}', the dt-view "
            f"of '{name}': pass the overlay job's OutputDatasetArns entry"
        )
    region = region or arn_region
    sm = register._sm_client(region)
    kwargs = dict(HubName=hub, HubContentType="DataSet", HubContentName=name)
    dst = sm.describe_hub_content(**kwargs, **_version_kwargs(version))
    latest = sm.describe_hub_content(**kwargs).get("HubContentVersion")
    if latest and dst.get("HubContentVersion") and latest != dst["HubContentVersion"]:
        raise DataPreparerOperatorError(
            f"'{name}' is at version {latest}, not {dst['HubContentVersion']} (the "
            f"version passed). For a later export of the same review set, the overlay "
            f"runs on a view of the latest version and is promoted onto it (the overlay is "
            f"rebuilt from the whole seed, and a row people labeled keeps its model "
            f"answer). Otherwise confirm with the user which version the corrections "
            f"belong on."
        )
    src = sm.describe_hub_content(
        HubName=f_hub,
        HubContentType="DataSet",
        HubContentName=f_name,
        **_version_kwargs(f_version),
    )
    src_doc = json.loads(src.get("HubContentDocument") or "{}")
    try:
        location = _s3_from_document(src_doc)
    except ValueError as e:
        raise DataPreparerOperatorError(f"the transform output {from_arn}: {e}") from None
    obj = _s3.resolve_object(location, region)
    bucket, key = _s3.split_s3_uri(obj)
    dst_doc = json.loads(dst.get("HubContentDocument") or "{}")
    document = {
        k: v
        for k, v in dst_doc.items()
        if k not in _SERVICE_FIELDS and not k.startswith("Transformation")
    }
    document["DatasetS3Uri"] = obj
    if "DatasetS3Bucket" in document or "DatasetS3Prefix" in document:
        document.update(DatasetS3Bucket=bucket, DatasetS3Prefix=key.rsplit("/", 1)[0])
    document.update({k: v for k, v in src_doc.items() if k.startswith("Transformation")})
    new = dict(
        HubName=hub,
        HubContentName=name,
        HubContentType="DataSet",
        DocumentSchemaVersion=dst.get("DocumentSchemaVersion") or DOCUMENT_SCHEMA_VERSION,
        HubContentDocument=json.dumps(document),
    )
    if dst.get("HubContentDescription"):
        new["HubContentDescription"] = dst["HubContentDescription"][:1023]
    # The Hub adds its own sagemaker: keywords at import and refuses them in a request.
    keywords = [
        k for k in dst.get("HubContentSearchKeywords") or [] if not k.startswith("sagemaker:")
    ]
    if keywords:
        new["HubContentSearchKeywords"] = keywords[:50]
    new_arn = sm.import_hub_content(**new)["HubContentArn"]  # the service increments
    return {
        "dataset_arn": new_arn,
        "source_arn": dst.get("HubContentArn") or to_arn,
        "transform_output_arn": src.get("HubContentArn") or from_arn,
        "dataset_s3_uri": obj,
    }


def register_s3_object(*, s3_uri, name, hub_name_override=None, region=None):
    """Register an ``s3://`` JSONL object (a corpus, a seed) as a DataSet without reading it.

    One list call at most and one ``ImportHubContent``; the ranged read that checks the
    rows is :mod:`~sagemaker.ai_registry.data_preparer.auto_labeling.operators.inspect.inputs`'s.

    Args:
        s3_uri: The object, or a folder holding one ``.jsonl``.
        name: The DataSet name; refused when it exists.
        hub_name_override: A hub instead of the account's canonical one; it must exist.
        region: The region; the session's when ``None``.

    Returns:
        ``{"dataset_arn", "dataset_name", "hub_name", "dataset_s3_uri"}``.

    Raises:
        DataPreparerOperatorError: An invalid or taken name, a missing override hub, or a
            folder without exactly one ``.jsonl``.
    """
    _validate_hub_name(name)
    region = region or register._default_region()
    sm = register._sm_client(region)
    hub = register._resolve_or_create_hub(
        sm, register._sts_account(region), region, hub_name_override
    )
    _require_new_name(sm, hub, name, "use its ARN, or pick a new name with the user")
    obj = _s3.resolve_object(s3_uri, region)
    bucket, key = _s3.split_s3_uri(obj)
    document = {
        "DatasetFileFormat": DATASET_FILE_FORMAT,
        "DatasetS3Uri": obj,
        "DatasetS3Bucket": bucket,
        "DatasetS3Prefix": key.rsplit("/", 1)[0],
    }
    arn = sm.import_hub_content(
        HubName=hub,
        HubContentName=name,
        HubContentType="DataSet",
        DocumentSchemaVersion=DOCUMENT_SCHEMA_VERSION,
        HubContentDescription=f"registered by auto-labeling from {obj} (not read)"[:1023],
        HubContentSearchKeywords=["auto-labeling-input"],
        HubContentDocument=json.dumps(document),
    )["HubContentArn"]
    return {"dataset_arn": arn, "dataset_name": name, "hub_name": hub, "dataset_s3_uri": obj}


def dt_view(  # pylint: disable=too-many-locals,too-many-branches
    *, arn=None, s3_uri=None, name=None, hub_name_override=None, region=None
):
    """Register the DataSet a dataset-transformation job takes as its first input.

    A DT job registers its output as the next version of its first input, so on a view,
    ``<name>-dt-input``, it does not land on the customer's DataSet; and the view points at
    the one ``.jsonl`` of a labeling job's output folder, which a DT job cannot size. The
    view and its outputs are scratch (:func:`scratch_cleanup`). Nothing is downloaded.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import dt_handoff

        view = dt_handoff.dt_view(arn=labeled_dataset_arn, region="us-west-2")
        run_log.append(_scratch.scratch_event(view["dataset_arn"], "dt-view"))

    Args:
        arn: The DataSet the DT job would take (the corpus, a labeled DataSet, a seed).
        s3_uri: Instead of ``arn``: an ``s3://`` object that is not a DataSet; needs
            ``name``.
        name: With ``s3_uri``: the dataset's name.
        hub_name_override: With ``s3_uri``: a hub instead of the canonical one.
        region: The region; the ARN's own, else the session's, when ``None``.

    Returns:
        ``{"dataset_arn", "source_arn", "dataset_s3_uri"}``.

    Raises:
        DataPreparerOperatorError: Not exactly one of ``arn`` and ``s3_uri``; ``s3_uri``
            without ``name``; an existing ``<name>-dt-input`` this module did not
            register; or an object that is not ``.jsonl``.
    """
    from botocore.exceptions import ClientError

    if bool(arn) == bool(s3_uri):
        raise DataPreparerOperatorError(
            "pass exactly one of --arn (a DataSet) and --s3-uri (an object)"
        )
    if arn:
        hub, name, version, arn_region = parse_dataset_arn(arn)
        region = region or arn_region
        sm = register._sm_client(region)
        resp = sm.describe_hub_content(
            HubName=hub_arn_from_dataset_arn(arn),
            HubContentType="DataSet",
            HubContentName=name,
            **_version_kwargs(version),
        )
        doc = json.loads(resp.get("HubContentDocument") or "{}")
        pinned, source = resp.get("HubContentArn") or arn, _s3_from_document(doc)
    else:
        if not name:
            raise DataPreparerOperatorError(
                "--s3-uri needs --name (the dataset's name, from the conversation)"
            )
        _validate_hub_name(name)
        region = region or register._default_region()
        sm = register._sm_client(region)
        hub = register._resolve_or_create_hub(
            sm, register._sts_account(region), region, hub_name_override
        )
        doc, pinned, source = {}, s3_uri, s3_uri
    obj = _s3.resolve_object(source, region)
    if not obj.endswith(".jsonl"):
        raise DataPreparerOperatorError(
            f"{obj} is not a .jsonl object: a dataset-transformation job "
            f"takes JSONL, CSV, TSV or Parquet"
        )
    bucket, key = _s3.split_s3_uri(obj)
    view = _view_name(name)
    try:
        versions = (
            sm.list_hub_content_versions(
                HubName=hub, HubContentType="DataSet", HubContentName=view, MaxResults=100
            ).get("HubContentSummaries")
            or []
        )
        # Ours when a version carries the keyword: the DT outputs that land on a view
        # carry only the Hub's own keywords.
        if versions and not any(
            DT_VIEW_KEYWORD in (v.get("HubContentSearchKeywords") or []) for v in versions
        ):
            raise DataPreparerOperatorError(
                f"'{view}' exists and was not registered by this skill; "
                f"rename it or pick another DataSet"
            )
    except ClientError as e:
        if not _not_found(e):
            raise
    document = {
        "DatasetFileFormat": doc.get("DatasetFileFormat") or DATASET_FILE_FORMAT,
        "DatasetS3Uri": obj,
        "DatasetS3Bucket": bucket,
        "DatasetS3Prefix": key.rsplit("/", 1)[0],
    }
    new_arn = sm.import_hub_content(
        HubName=hub,
        HubContentName=view,
        HubContentType="DataSet",
        DocumentSchemaVersion=DOCUMENT_SCHEMA_VERSION,
        HubContentDescription=f"dataset-transformation input for {pinned}"[:1023],
        HubContentSearchKeywords=[DT_VIEW_KEYWORD],
        HubContentDocument=json.dumps(document),
    )["HubContentArn"]
    return {"dataset_arn": new_arn, "source_arn": pinned, "dataset_s3_uri": obj}


def register_transform_output(*, from_arn, name, source_arn=None, region=None):
    """Register a DT output that landed on a dt-view under a new name, as version 1 of its own.

    Its document is the output's row location and the job's ``Transformation*`` fields.
    One ``DescribeHubContent`` and one ``ImportHubContent``; nothing downloaded.

    Args:
        from_arn: The DataSet ARN the transform job returned (``OutputDatasetArns``).
        name: The new DataSet name; refused when it exists (a derived dataset is never
            added as a version of another).
        source_arn: The DataSet the job ran on, for the description.
        region: The region; the ARN's own when ``None``.

    Returns:
        ``{"dataset_arn", "dataset_name", "hub_name", "registered_from", "dataset_s3_uri"}``.

    Raises:
        DataPreparerOperatorError: An invalid or taken name, or an output document
            without an S3 location. Nothing imported.
    """
    _validate_hub_name(name)
    hub, _n, version, arn_region = parse_dataset_arn(from_arn)
    # NOTE: hub is a bare HubName; same cross-account caveat as promote_transform_output.
    region = region or arn_region
    sm = register._sm_client(region)
    src = sm.describe_hub_content(
        HubName=hub, HubContentType="DataSet", HubContentName=_n, **_version_kwargs(version)
    )
    src_doc = json.loads(src.get("HubContentDocument") or "{}")
    try:
        _s3_from_document(src_doc)
    except ValueError as e:
        raise DataPreparerOperatorError(f"the transform output {from_arn}: {e}") from None
    _require_new_name(
        sm,
        hub,
        name,
        "pick a new name (a derived dataset is never added as a version of another)",
    )
    document = {
        k: src_doc[k]
        for k in ("DatasetS3Uri", "DatasetS3Bucket", "DatasetS3Prefix", "DatasetFileFormat")
        if src_doc.get(k)
    }
    document.update({k: v for k, v in src_doc.items() if k.startswith("Transformation")})
    desc = f"derived by {src_doc.get('TransformationJobArn') or from_arn}" + (
        f" from {source_arn}" if source_arn else ""
    )
    arn = sm.import_hub_content(
        HubName=hub,
        HubContentName=name,
        HubContentType="DataSet",
        DocumentSchemaVersion=DOCUMENT_SCHEMA_VERSION,
        HubContentDescription=desc[:1023],
        HubContentSearchKeywords=["auto-labeled", "derived"],
        HubContentDocument=json.dumps(document),
    )["HubContentArn"]
    return {
        "dataset_arn": arn,
        "dataset_name": name,
        "hub_name": hub,
        "registered_from": src.get("HubContentArn") or from_arn,
        "dataset_s3_uri": _s3_from_document(document),
    }
