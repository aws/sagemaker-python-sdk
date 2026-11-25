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
"""S3 transfer for the auto-labeling operators, over sagemaker-core's URI parsing.

URI parsing is :func:`sagemaker.core.s3.parse_s3_url`. The transfers stay thin boto3 calls
rather than :class:`~sagemaker.core.s3.S3Uploader` / :class:`~sagemaker.core.s3.S3Downloader`
because an operator is bound to a region by argument (a DataSet ARN's own region), and those
take a ``Session``; constructing one per transfer would add the session's clients to every
call. The clients are module-level functions so a test stubs one place
(``monkeypatch.setattr(_s3, "s3_client", ...)``); callers reach them through the module
(``_s3.s3_client(region)``), never by a name bound at import.
"""

from __future__ import absolute_import

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from sagemaker.core.s3 import is_s3_url, parse_s3_url

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError

DEFAULT_REGION = "us-west-2"


def split_s3_uri(uri: str) -> Tuple[str, str]:
    """``(bucket, key)`` of an ``s3://bucket/key`` URI (sagemaker-core's parser, key required).

    Raises:
        DataPreparerOperatorError: ``uri`` is not an ``s3://`` URI with a bucket and a key.
    """
    try:
        bucket, key = parse_s3_url(uri or "")
    except ValueError as e:
        raise DataPreparerOperatorError(f"not an s3:// URI: {uri!r}") from e
    if not bucket or not key:
        raise DataPreparerOperatorError(f"an s3:// URI needs a bucket and a key: {uri!r}")
    return bucket, key


def is_s3_uri(uri: Any) -> bool:
    """Whether ``uri`` is an ``s3://`` URI with a bucket and a key."""
    if not isinstance(uri, str) or not is_s3_url(uri):
        return False
    bucket, key = parse_s3_url(uri)
    return bool(bucket and key)


def resolve_region(region: Optional[str]) -> str:
    """``region``, else the ambient boto3 default region, else ``us-west-2``."""
    if region:
        return region
    import boto3

    return boto3.session.Session().region_name or DEFAULT_REGION


def s3_client(region: Optional[str]):
    """The region-bound boto3 S3 client every transfer here uses (no AWS CLI needed)."""
    import boto3

    return boto3.client("s3", region_name=region)


def upload_file(local, s3_uri: str, region: Optional[str]) -> None:
    """Upload the local file at ``local`` to ``s3_uri`` byte for byte, without reading it."""
    bucket, key = split_s3_uri(s3_uri)
    s3_client(region).upload_file(str(local), bucket, key)


def download_file(s3_uri: str, local, region: Optional[str]) -> None:
    """Copy one S3 object to a local path (its parent folder is created)."""
    bucket, key = split_s3_uri(s3_uri)
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    s3_client(region).download_file(bucket, key, str(local))


def put_object(s3_uri: str, body, region: Optional[str]) -> None:
    """Write ``body`` (text or bytes) as the object at ``s3_uri``."""
    bucket, key = split_s3_uri(s3_uri)
    data = body.encode("utf-8") if isinstance(body, str) else body
    s3_client(region).put_object(Bucket=bucket, Key=key, Body=data)


def resolve_object(s3_uri: str, region: Optional[str]) -> str:
    """A concrete JSONL object URI for a dataset S3 location.

    A direct object key passes through; a prefix (folder) resolves to the single ``.jsonl``
    under it. Ambiguity is an error, not a guess.
    """
    if s3_uri.rstrip("/").endswith(".jsonl"):
        return s3_uri
    bucket, _, key = s3_uri.rstrip("/")[len("s3://") :].partition(
        "/"
    )  # a bucket root lists the bucket
    if not s3_uri.startswith("s3://") or not bucket:
        raise DataPreparerOperatorError(f"not an s3:// URI: {s3_uri!r}")
    key = key + "/" if key else ""
    hits = []
    for page in (
        s3_client(region).get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=key)
    ):
        hits.extend(o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(".jsonl"))
    if len(hits) != 1:
        raise DataPreparerOperatorError(
            f"expected exactly one .jsonl under {s3_uri} (found {len(hits)}); "
            f"pass the object URI directly"
        )
    return f"s3://{bucket}/{hits[0]}"


def head_object(s3_uri: str, region: Optional[str]) -> Dict[str, Any]:
    """``{etag, size}`` of an object from one ``HeadObject``, no body.

    Best-effort: ``{}`` on any error, so a resolve never fails because the identity probe
    did not land (the URI itself is the deliverable). The ETag's quotes are stripped.
    """
    try:
        bucket, key = split_s3_uri(s3_uri)
        resp = s3_client(region).head_object(Bucket=bucket, Key=key)
    except Exception:  # noqa: BLE001 - identity probe is advisory, never fatal
        return {}
    etag = (resp.get("ETag") or "").strip('"')
    out: Dict[str, Any] = {}
    if etag:
        out["etag"] = etag
    if resp.get("ContentLength") is not None:
        out["size"] = int(resp["ContentLength"])
    return out
