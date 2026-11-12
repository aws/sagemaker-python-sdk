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
"""Shared S3 access helpers for the data_preparer job flows.

PRIVATE — not exported. Imports nothing from this package, so both
``preparer`` and ``_job_resource`` can depend on it without a cycle.

This is the Session-aware counterpart to the region-only
``auto_labeling/operators/_s3.py`` module: unlike that module, it needs a
SageMaker ``Session`` object to resolve default-bucket ownership.
"""

from __future__ import absolute_import

# Chunk size for streaming line counts. Bounds peak memory to one chunk plus
# one partial line, while keeping the iteration count low: botocore's default
# of 1 KB would run ~1M iterations per GB, each concatenating and splitting.
_STREAM_CHUNK_BYTES = 1024 * 1024


def _s3_owner_kwargs(session, bucket: str) -> dict:
    """Return ``{"ExpectedBucketOwner": account}`` when ``bucket`` is the session's
    SDK-generated default bucket, else ``{}``.

    Matches the settled pattern in session_helper.py:605-611 and common_utils.py:436,641,798.
    Only asserts ownership on the predictable default-bucket name to guard against
    bucket-squatting; cross-account dataset buckets are left unasserted so legitimate
    cross-account reads are not broken.
    """
    expected_owner = session._get_account_id_if_default_bucket(bucket)
    if expected_owner:
        return {"ExpectedBucketOwner": expected_owner}
    return {}
