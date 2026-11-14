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
"""Tests for the cross-account hub ARN fix (SEC-90E04A7E, SEC-07CCC6EC).

Covers:
- hub_arn_from_dataset_arn returns a fully-qualified hub ARN using the source
  ARN's own account and region, never the caller's session.
- resolve_dataset_ref passes the qualified hub ARN to describe_hub_content.
- dt_view passes the qualified hub ARN to describe_hub_content.
- Regression: _normalize_source still returns a bare hub/name label (trap 1).
- Regression: dt_view's import_hub_content still receives the bare hub name (trap 2).

No network, no credentials.
"""

from __future__ import absolute_import

import json

import pytest
from botocore.exceptions import ClientError

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _s3
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import dt_handoff
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import register

# Cross-account ARN: account 999888777666, region eu-west-1, hub EUHub.
_XACCT = "999888777666"
_XACCT_REGION = "eu-west-1"
_XHUB = "EUHub"
_XARN = f"arn:aws:sagemaker:{_XACCT_REGION}:{_XACCT}:hub-content/{_XHUB}/DataSet/my-ds/2.0.0"
_XARN_NOVER = f"arn:aws:sagemaker:{_XACCT_REGION}:{_XACCT}:hub-content/{_XHUB}/DataSet/my-ds"
_XHUB_ARN = f"arn:aws:sagemaker:{_XACCT_REGION}:{_XACCT}:hub/{_XHUB}"


# ---------------------------------------------------------------------------
# Minimal fakes — no network, no boto3
# ---------------------------------------------------------------------------


def _nf(op):
    return ClientError({"Error": {"Code": "ResourceNotFound"}}, op)


class _Sm:
    """Fake SageMaker client that records every call."""

    def __init__(self, hub_doc=None):
        self.hub_doc = hub_doc or {"DatasetS3Uri": "s3://b/data.jsonl"}
        self.calls = []

    def describe_hub_content(self, **kw):
        self.calls.append(("describe_hub_content", kw))
        return {
            "HubContentArn": _XARN,
            "HubContentVersion": "2.0.0",
            "HubContentDocument": json.dumps(self.hub_doc),
        }

    def list_hub_content_versions(self, **kw):
        self.calls.append(("list_hub_content_versions", kw))
        raise _nf("ListHubContentVersions")  # view does not exist yet

    def import_hub_content(self, **kw):
        self.calls.append(("import_hub_content", kw))
        return {
            "HubContentArn": f"arn:aws:sagemaker:{_XACCT_REGION}:{_XACCT}:hub-content/{_XHUB}/DataSet/out/0.0.1"
        }


class _FakeS3:
    def __init__(self):
        self.heads = []

    def head_object(self, Bucket, Key):  # noqa: N803
        self.heads.append((Bucket, Key))
        return {"ETag": '"etag-x"', "ContentLength": 100}


# ---------------------------------------------------------------------------
# helper: hub_arn_from_dataset_arn
# ---------------------------------------------------------------------------


def test_hub_arn_from_dataset_arn_cross_account():
    # The returned ARN must carry the SOURCE ARN's account and region,
    # not the caller's session.  _XACCT is a different account.
    result = register.hub_arn_from_dataset_arn(_XARN)
    assert result == _XHUB_ARN


def test_hub_arn_from_dataset_arn_versionless():
    # ARNs without a version segment are accepted.
    result = register.hub_arn_from_dataset_arn(_XARN_NOVER)
    assert result == _XHUB_ARN


def test_hub_arn_from_dataset_arn_round_trips_region_and_account():
    # Build an ARN in a third region/account to confirm the helper always reads
    # the ARN's own coordinates, not any default.
    third = "arn:aws:sagemaker:ap-southeast-1:111222333444:hub-content/HubX/DataSet/d/0.1.0"
    result = register.hub_arn_from_dataset_arn(third)
    assert result == "arn:aws:sagemaker:ap-southeast-1:111222333444:hub/HubX"


def test_hub_arn_from_dataset_arn_rejects_non_arn():
    with pytest.raises(DataPreparerOperatorError, match="not a hub-content ARN"):
        register.hub_arn_from_dataset_arn("s3://bucket/key.jsonl")


def test_hub_arn_from_dataset_arn_rejects_empty():
    with pytest.raises(DataPreparerOperatorError, match="not a hub-content ARN"):
        register.hub_arn_from_dataset_arn("")


def test_hub_arn_from_dataset_arn_non_aws_partition():
    """A non-aws partition (e.g. aws-cn) must be preserved verbatim in the hub ARN;
    the partition must not be re-derived from the region."""
    cn_arn = "arn:aws-cn:sagemaker:cn-north-1:123456789012:hub-content/HubCN/DataSet/cn-ds/1.0.0"
    result = register.hub_arn_from_dataset_arn(cn_arn)
    assert result == "arn:aws-cn:sagemaker:cn-north-1:123456789012:hub/HubCN"


# ---------------------------------------------------------------------------
# SEC-90E04A7E: resolve_dataset_ref uses qualified HubName
# ---------------------------------------------------------------------------


def test_resolve_dataset_ref_describe_uses_qualified_hub_arn(monkeypatch):
    sm = _Sm({"DatasetS3Uri": "s3://b/data.jsonl"})
    s3 = _FakeS3()
    monkeypatch.setattr(register, "_sm_client", lambda region: sm)
    monkeypatch.setattr(_s3, "s3_client", lambda region: s3)

    register.resolve_dataset_ref(_XARN, None, "us-west-2")

    describes = [kw for op, kw in sm.calls if op == "describe_hub_content"]
    assert len(describes) == 1
    # Finding site: HubName must be the fully-qualified hub ARN, not bare "EUHub".
    assert describes[0]["HubName"] == _XHUB_ARN
    assert describes[0]["HubName"] != _XHUB  # bare name must NOT be used
    assert describes[0]["HubContentName"] == "my-ds"
    assert describes[0]["HubContentVersion"] == "2.0.0"


def test_resolve_dataset_ref_describe_qualified_hub_arn_versionless(monkeypatch):
    sm = _Sm({"DatasetS3Uri": "s3://b/data.jsonl"})
    s3 = _FakeS3()
    monkeypatch.setattr(register, "_sm_client", lambda region: sm)
    monkeypatch.setattr(_s3, "s3_client", lambda region: s3)

    register.resolve_dataset_ref(_XARN_NOVER, None, "us-west-2")

    describes = [kw for op, kw in sm.calls if op == "describe_hub_content"]
    assert describes[0]["HubName"] == _XHUB_ARN
    assert "HubContentVersion" not in describes[0]


# ---------------------------------------------------------------------------
# SEC-07CCC6EC: dt_view uses qualified HubName for the lookup
# ---------------------------------------------------------------------------


def test_dt_view_describe_uses_qualified_hub_arn(monkeypatch):
    sm = _Sm({"DatasetS3Uri": "s3://b/out.jsonl"})
    monkeypatch.setattr(register, "_sm_client", lambda region: sm)
    monkeypatch.setattr(_s3, "s3_client", lambda region: _FakeS3())

    dt_handoff.dt_view(arn=_XARN, region=_XACCT_REGION)

    describes = [kw for op, kw in sm.calls if op == "describe_hub_content"]
    assert len(describes) == 1
    # Finding site: lookup must use the fully-qualified hub ARN.
    assert describes[0]["HubName"] == _XHUB_ARN
    assert describes[0]["HubName"] != _XHUB  # bare name must NOT be used
    assert describes[0]["HubContentName"] == "my-ds"


# ---------------------------------------------------------------------------
# Trap 1 (register.py): _normalize_source still returns bare hub/name
# ---------------------------------------------------------------------------


def test_normalize_source_returns_bare_hub_slash_name():
    # register.py:406 builds f"{hub}/{name}" for a lineage label.
    # If hub became an ARN this would emit arn:.../hub/EUHub/my-ds instead of EUHub/my-ds.
    result = register._normalize_source(_XARN)
    assert result == f"{_XHUB}/my-ds"
    assert not result.startswith("arn:")


def test_normalize_source_returns_bare_hub_slash_name_versionless():
    result = register._normalize_source(_XARN_NOVER)
    assert result == f"{_XHUB}/my-ds"


# ---------------------------------------------------------------------------
# Trap 2 (dt_handoff.py): dt_view's import_hub_content still uses bare hub
# ---------------------------------------------------------------------------


def test_dt_view_import_hub_content_uses_bare_hub_name(monkeypatch):
    # dt_view writes the view to the SAME hub as the source DataSet, using the
    # bare hub name so we do not silently import into an unverified ARN target.
    sm = _Sm({"DatasetS3Uri": "s3://b/out.jsonl"})
    monkeypatch.setattr(register, "_sm_client", lambda region: sm)
    monkeypatch.setattr(_s3, "s3_client", lambda region: _FakeS3())

    dt_handoff.dt_view(arn=_XARN, region=_XACCT_REGION)

    imports = [kw for op, kw in sm.calls if op == "import_hub_content"]
    assert len(imports) == 1
    # ImportHubContent uses the bare hub name, not an ARN.
    assert imports[0]["HubName"] == _XHUB
    assert not imports[0]["HubName"].startswith("arn:")
