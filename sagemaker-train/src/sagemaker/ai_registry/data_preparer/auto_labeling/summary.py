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
"""Reader for an auto-labeling job's summary report.

An auto-labeling job's result is not in ``DescribeJob``: the container writes a
``labeling_summary.json`` to S3 — the certificate status, held-out accuracy and
its confidence bound, the row counts, and where routed rows went — and Hub
finalization records a pointer to it on the labeled DataSet. Reading it by hand
means four hops: DataSet ARN, the DataSet document's ``JobMetadata``, the
labeling ``index.json`` that entry points at, and finally the summary the index
names under ``Artifacts``.

:class:`LabelingSummary` walks those hops::

    dataset = DataPreparer.generate(config)
    summary = LabelingSummary.from_dataset(dataset)

    print(summary.status, summary.held_out_accuracy, summary.rows_routed)

Also reachable from a job handle, which reads the DataSet ARN off the job::

    job = DataPreparer.generate(config, wait=False)
    job.wait()
    summary = job.labeling_summary

The fields the container always writes are typed; the rest of the document —
including the per-task fields, such as summarization's grader statistics — is
kept as model extras, so a summary key added by a newer container is still
readable as an attribute.
"""

from __future__ import absolute_import

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict

from sagemaker.ai_registry.air_constants import (
    DATASET_HUB_CONTENT_TYPE,
    RESPONSE_KEY_HUB_CONTENT_DOCUMENT,
    RESPONSE_KEY_HUB_CONTENT_SEARCH_KEYWORDS,
    TAG_KEY_REVIEW_DATASET,
)
from sagemaker.ai_registry.air_hub import AIRHub
from sagemaker.ai_registry.air_utils import _parse_keywords
from sagemaker.ai_registry.data_preparer._case_conversion import _snake_to_pascal
from sagemaker.ai_registry.data_preparer.defaults import DataPreparerDefaults
from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerError
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.s3.utils import parse_s3_url

logger = logging.getLogger(__name__)

# The JobMetadata entry an auto-labeling job leaves on the labeled DataSet. A
# DataSet can carry entries from other job types (data-quality evaluation, for
# one), so the JobType is what identifies the labeling run.
_LABELING_JOB_TYPE = "SyntheticDataGeneration"
_JOB_METADATA_KEY = "JobMetadata"
_METADATA_URI_KEY = "MetadataS3Uri"
_SUMMARY_ARTIFACT_KEY = "LabelingSummary"


def _split_s3_uri(uri: str) -> Tuple[str, str]:
    """Split ``s3://bucket/key`` into ``(bucket, key)``.

    Delegates the scheme parsing to sagemaker-core's shared ``parse_s3_url`` so
    the accepted shape stays in one place, and re-raises as a DataPreparer error
    rather than the bare ``ValueError`` it throws.

    Raises:
        DataPreparerError: If *uri* is not an ``s3://`` URI naming both a bucket
            and a key.
    """
    try:
        bucket, key = parse_s3_url(uri)
    except (ValueError, AttributeError) as e:
        raise DataPreparerError(
            f"Expected an s3:// URI for the labeling artifacts, got {uri!r}."
        ) from e
    if not bucket or not key:
        raise DataPreparerError(f"Malformed S3 URI for the labeling artifacts: {uri!r}.")
    return bucket, key


def _read_s3_json(session: Session, uri: str) -> Dict[str, Any]:
    """Read and parse the JSON document at *uri*.

    Raises:
        DataPreparerError: If *uri* is malformed or does not hold a JSON object.
    """
    bucket, key = _split_s3_uri(uri)
    parsed = json.loads(session.read_s3_file(bucket, key))
    if not isinstance(parsed, dict):
        raise DataPreparerError(f"Expected a JSON object at {uri}, got {type(parsed).__name__}.")
    return parsed


def _review_dataset_arn(describe_response: Dict[str, Any]) -> Optional[str]:
    """Return the review DataSet's ARN from the labeled DataSet's keywords.

    Hub finalization records it as a ``review-dataset:<arn>`` search keyword,
    because the summary the container wrote predates both ARNs. ``None`` when the
    job does not route, so there is no review DataSet.
    """
    keywords = describe_response.get(RESPONSE_KEY_HUB_CONTENT_SEARCH_KEYWORDS)
    if not isinstance(keywords, list):
        return None
    arn = _parse_keywords(keywords).get(TAG_KEY_REVIEW_DATASET)
    if not isinstance(arn, str):
        return None
    return arn.strip() or None


def _created_at_key(entry: Dict[str, Any]) -> Tuple[int, datetime]:
    """Sort key ordering ``JobMetadata`` entries by ``CreatedAt``, oldest first.

    ``CreatedAt`` is an ISO-8601 timestamp. Parsed rather than compared as a
    string so the order stays chronological whatever form the writer emits — a
    ``+00:00`` offset instead of ``Z``, or a different fractional-second
    precision, both of which sort wrongly as text. A naive timestamp is read as
    UTC, since that is what the writer records. An absent or unparseable value
    sorts oldest, behind every entry that has one, rather than winning by
    accident.
    """
    raw = entry.get("CreatedAt")
    oldest = (0, datetime.min.replace(tzinfo=timezone.utc))
    if not isinstance(raw, str) or not raw:
        return oldest
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return oldest
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (1, parsed)


def _newest_labeling_entry(document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the most recent labeling ``JobMetadata`` entry, or ``None``.

    Entries from other job types, and any entry with no ``MetadataS3Uri``, are
    skipped. The newest of what remains is chosen by parsed ``CreatedAt``; see
    :func:`_created_at_key`.
    """
    entries = document.get(_JOB_METADATA_KEY)
    if not isinstance(entries, list):
        return None

    labeling = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("JobType") == _LABELING_JOB_TYPE
        and entry.get(_METADATA_URI_KEY)
    ]
    if not labeling:
        return None
    return max(labeling, key=_created_at_key)


class LabelingSummary(BaseModel):
    """An auto-labeling job's summary report (``labeling_summary.json``).

    Construct it with :meth:`from_dataset`, or read it off a job handle through
    ``Job.labeling_summary``. Unrecognized keys in the document are kept as
    model extras rather than dropped.

    Attributes:
        status: The certificate outcome: ``Certified``, ``NotCertified`` or
            ``Partial``.
        status_reason: Why the job landed on that status, as a code such as
            ``spot_check_pending``, ``target_not_met``,
            ``insufficient_eval_rows``, ``task_not_certifiable`` or
            ``no_target``.
        type: The labeling ``Type`` the job ran, such as
            ``labeling.classification``.
        rows_labeled: Rows the job produced a label for.
        rows_accepted: Labeled rows that met the accept threshold.
        rows_routed: Rows sent to the review DataSet instead of being accepted.
        rows_failed: Rows the job could not label.
        rows_missing: Rows left without a prediction.
        held_out_accuracy: Measured accuracy on the held-out evaluation rows.
        held_out_lower_bound95: Lower 95% bound on ``held_out_accuracy``, over
            all evaluation rows.
        accepted_held_out_accuracy: Held-out accuracy restricted to accepted rows.
        accepted_lower_bound95: Lower 95% bound on accepted-row accuracy; the
            certified promise, compared against ``target``. Adjusted for the
            cutoff grid the job searched, and ``None`` on a run that certified
            nothing.
        target: The accuracy the job was asked to certify.
        accept_threshold: The confidence cut-off a row had to clear to be accepted.
        held_out_rows: Evaluation rows the accuracy was measured on.
        rows_human_labeled: Rows whose final label came from a person.
        spot_check_rows: Accepted rows drawn for a human spot-check.
        juror_model_ids: The models that voted on each label.
        spec_sha256: Digest of the labeling spec the rows were labeled with.
        review_dataset_name: The DataSet routed rows were written to.
        labeled_dataset_arn: The labeled DataSet this summary belongs to.
        review_dataset_arn: The review DataSet routed rows went to, when the job
            routes. Both ARNs are filled in by :meth:`from_dataset`: Hub mints
            them after the container exits, so the document itself carries
            ``null`` for each.
        warnings: Non-fatal warnings the job recorded.
        artifacts: The labeling index's ``Artifacts`` map — the S3 location of
            each output the job wrote, with ``None`` for the ones it did not.

    Keys the container writes that are not listed above stay reachable as model
    extras under their original PascalCase name — ``summary.GradedRows``, not
    ``summary.graded_rows``. The fields above are the snake_case surface.
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        alias_generator=_snake_to_pascal,
    )

    status: Optional[str] = None
    status_reason: Optional[str] = None
    type: Optional[str] = None

    rows_labeled: Optional[int] = None
    rows_accepted: Optional[int] = None
    rows_routed: Optional[int] = None
    rows_failed: Optional[int] = None
    rows_missing: Optional[int] = None

    held_out_accuracy: Optional[float] = None
    held_out_lower_bound95: Optional[float] = None
    accepted_held_out_accuracy: Optional[float] = None
    accepted_lower_bound95: Optional[float] = None

    target: Optional[float] = None
    accept_threshold: Optional[float] = None

    held_out_rows: Optional[int] = None
    rows_human_labeled: Optional[int] = None
    spot_check_rows: Optional[int] = None
    juror_model_ids: List[str] = []
    spec_sha256: Optional[str] = None

    review_dataset_name: Optional[str] = None
    labeled_dataset_arn: Optional[str] = None
    review_dataset_arn: Optional[str] = None
    warnings: List[str] = []
    artifacts: Dict[str, Optional[str]] = {}

    @classmethod
    def from_dataset(
        cls,
        dataset_or_arn: Union[str, Any],
        sagemaker_session: Optional[Session] = None,
    ) -> "LabelingSummary":
        """Read the labeling summary behind a labeled DataSet.

        Args:
            dataset_or_arn: A ``DataSet`` returned by ``DataPreparer.generate``,
                or its hub-content ARN.
            sagemaker_session: Session to read with. Defaults to the DataSet's
                own session, then to a new default session.

        Returns:
            The parsed :class:`LabelingSummary`.

        Raises:
            DataPreparerError: If the DataSet was not produced by an
                auto-labeling job, or if its labeling index names no summary.
        """
        if isinstance(dataset_or_arn, str):
            arn = dataset_or_arn
            session = DataPreparerDefaults.get_sagemaker_session(sagemaker_session)
        else:
            arn = dataset_or_arn.arn
            session = DataPreparerDefaults.get_sagemaker_session(
                sagemaker_session or dataset_or_arn.sagemaker_session
            )

        # Imported here: preparer imports configs, which imports
        # auto_labeling.shapes, so a module-level import would close a cycle
        # back through this module.
        from sagemaker.ai_registry.data_preparer.preparer import _parse_dataset_arn

        name, hub, version = _parse_dataset_arn(arn)
        response = AIRHub.describe_hub_content(
            DATASET_HUB_CONTENT_TYPE,
            name,
            version,
            session=session,
            hub_name=hub,
        )

        document = response.get(RESPONSE_KEY_HUB_CONTENT_DOCUMENT) or "{}"
        if isinstance(document, str):
            document = json.loads(document)

        entry = _newest_labeling_entry(document)
        if entry is None:
            raise DataPreparerError(
                f"DataSet '{name}' carries no labeling job metadata, so it was not produced "
                "by an auto-labeling job. Only a DataSet a labeling job registered has a "
                "summary to read."
            )

        index_uri = entry[_METADATA_URI_KEY]
        index = _read_s3_json(session, index_uri)
        artifacts = _artifact_uris(index, index_uri)
        summary_uri = artifacts.get(_SUMMARY_ARTIFACT_KEY)
        if not summary_uri:
            raise DataPreparerError(
                f"The labeling index for DataSet '{name}' names no {_SUMMARY_ARTIFACT_KEY} "
                "artifact, so the job did not finish writing its summary."
            )

        summary = _read_s3_json(session, summary_uri)
        return cls(
            **{
                **summary,
                "Artifacts": artifacts,
                # The container writes both ARNs as null: Hub mints them after it
                # exits. Fill them from what was resolved here, so a routed job's
                # review hand-off does not read None.
                "LabeledDatasetArn": arn,
                "ReviewDatasetArn": _review_dataset_arn(response)
                or summary.get("ReviewDatasetArn"),
            }
        )


def _artifact_uris(index: Dict[str, Any], index_uri: str) -> Dict[str, Any]:
    """The index's artifact URIs, each resolved from where the index itself sits.

    The index lives at ``<job output>/labeling/index.json`` and its ``RelativePaths`` are
    relative to ``<job output>``. Resolving them against the index's own URI finds the
    files wherever the service placed the job's output folder; the container's absolute
    ``Artifacts`` URIs can omit a folder level the service adds (a
    ``synthetic-data-generation/<job>/<timestamp>/`` prefix), so a relative path wins
    over them. An ``Artifacts`` entry with no relative path is kept as written.
    """
    artifacts = index.get("Artifacts")
    artifacts = dict(artifacts) if isinstance(artifacts, dict) else {}
    relative = index.get("RelativePaths")
    suffix = "/labeling/index.json"
    if isinstance(relative, dict) and isinstance(index_uri, str) and index_uri.endswith(suffix):
        root = index_uri[: -len(suffix)]
        for key, rel in relative.items():
            if isinstance(rel, str) and rel:
                artifacts[key] = f"{root}/{rel.lstrip('/')}"
    return artifacts


__all__ = ["LabelingSummary"]
