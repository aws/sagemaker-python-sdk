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
"""Conftest for DataPreparer integration tests.

Prerequisites:
    AWS credentials for an account where the DataPreparer feature is enabled.
    The following environment variables are read by the suite:
        DATA_PREPARER_INTEG_ROLE_ARN   — execution role (required for most tests)
        DATA_PREPARER_INTEG_DT_ROLE_ARN — role that trusts sagemaker.amazonaws.com
                                          (required for DataTransformation tests;
                                          falls back to DATA_PREPARER_INTEG_ROLE_ARN)
        DATA_PREPARER_INTEG_BUCKET     — S3 bucket the role can read and write
                                          (defaults to the account's SageMaker bucket)
        DATA_PREPARER_INTEG_HUB_NAME   — hub to use for dataset imports/exports
                                          (defaults to the AI Registry hub)
"""

from __future__ import absolute_import

import json
import os
import uuid
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pytest

from sagemaker.core.helper.session_helper import Session

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sagemaker_session():
    """Create a SageMaker session."""
    return Session()


@pytest.fixture(scope="session")
def test_role(sagemaker_session):
    """Get the execution role."""
    from sagemaker.core.helper.session_helper import get_execution_role

    return get_execution_role(sagemaker_session)


@pytest.fixture(scope="session")
def dt_role():
    """Resolve the role the service will assume to run a DataTransformation job.

    DataTransformation needs its own knob because ``test_role`` resolves to
    ``get_execution_role``, which echoes back the *caller's* role — outside a
    SageMaker-hosted environment that is whatever you authenticated as, e.g. an
    ``Admin`` or ``ReadOnly`` role. Those do not trust
    ``sagemaker.amazonaws.com``, so ``CreateJob`` accepts the ARN (it only checks
    the syntax) and the job then fails at task launch with a generic
    "Internal error launching transformation task" and no customer-visible logs.

    Set ``DATA_PREPARER_INTEG_DT_ROLE_ARN`` to a role that trusts
    ``sagemaker.amazonaws.com`` and can read the input S3 data.
    ``DATA_PREPARER_INTEG_ROLE_ARN`` is honoured as a fallback so that a single
    variable still configures the whole data_preparer suite; the DT-specific
    variable exists for when that shared role cannot serve DT as well.

    Deliberately separate from ``test_role`` rather than layered onto it: that
    fixture is also consumed by the SDG, DQ and schema-validation tests, and
    re-pointing their role from a DT-motivated setting would change their
    behaviour without any change in their own files.

    Unlike ``test_role`` this does *not* fall back to ``get_execution_role``. DT
    needs three coordinated things that cannot be discovered — a role trusting
    ``sagemaker.amazonaws.com``, a bucket that role can read and write, and a hub
    it can import into — and every one of them fails as the same opaque task-launch
    error. Skipping when unconfigured keeps that misconfiguration from reading as a
    product regression; see the module docstring above for the environment variables
    the suite reads. Running inside SageMaker Studio does not change this:
    point the variable at that environment's execution role.
    """
    role_arn = os.environ.get("DATA_PREPARER_INTEG_DT_ROLE_ARN") or os.environ.get(
        "DATA_PREPARER_INTEG_ROLE_ARN"
    )
    if not role_arn:
        pytest.skip(
            "DataTransformation integ tests need an execution role that trusts "
            "sagemaker.amazonaws.com. Set DATA_PREPARER_INTEG_DT_ROLE_ARN (or "
            "DATA_PREPARER_INTEG_ROLE_ARN), along with DATA_PREPARER_INTEG_BUCKET "
            "and DATA_PREPARER_INTEG_HUB_NAME, to run them."
        )
    return role_arn


@pytest.fixture(scope="session")
def integ_bucket(sagemaker_session):
    """Bucket for job inputs and outputs.

    The execution role is typically scoped to one bucket (the bug-bash role grants
    S3 only on ``arn:aws:s3:::<bugbash-bucket>/*``). Uploading fixtures to the
    account's default SageMaker bucket therefore produces input the service cannot
    read, and the job dies at task launch with a generic "Internal error launching
    transformation task" rather than an access-denied message.

    Set ``DATA_PREPARER_INTEG_BUCKET`` to the bucket that role can read and write.
    """
    return os.environ.get("DATA_PREPARER_INTEG_BUCKET") or sagemaker_session.default_bucket()


@pytest.fixture(scope="session")
def integ_hub_name(sagemaker_session):
    """Hub that input datasets are imported into and outputs published to.

    Defaults to the AI Registry hub. Override with ``DATA_PREPARER_INTEG_HUB_NAME``
    when the execution role's ``sagemaker:ImportHubContent`` grant is scoped to a
    specific hub — publishing the output dataset fails otherwise.
    """
    hub_name = os.environ.get("DATA_PREPARER_INTEG_HUB_NAME")
    if hub_name:
        return hub_name

    from sagemaker.ai_registry.air_hub import AIRHub

    AIRHub._ensure_hub_name_initialized()
    return AIRHub.hubName


@pytest.fixture
def import_dataset(sagemaker_session, integ_hub_name):
    """Factory: register a JSONL already in S3 as a DataSet, return its ARN.

    DT takes its input as a ``DatasetArn``, not an S3 URI, so every DT test needs
    a hub-registered DataSet wrapping its fixture. Each call uses a fresh content
    name so tests never collide on hub content or contend over versions.

    Imports the document directly rather than calling ``DataSet.create`` so each
    test can pin ``DatasetFileFormat`` and ``DocumentSchemaVersion`` without going
    through Hub data validation (``create`` requires ``validation_role_arn`` for a
    known format on 4.0.0/5.0.0). The document must locate the input: it carries
    ``DatasetS3Uri`` (a document with neither ``DatasetS3Uri`` nor ``DatasetType``
    fails DT with "Unable to size inputs for compute routing"). ``DataSet.create``
    itself is covered by ``test_dt_happy_path.py::TestDTOnDataSetCreate``.

    Args:
        s3_uri: S3 URI of the JSONL, e.g. from the ``s3_fixtures`` dict.
        file_format: ``DatasetFileFormat`` to declare. Operators such as
            SchemaConversion and SchemaFilter branch on it, so it must describe
            the fixture's actual layout.
        document_schema_version: ``DocumentSchemaVersion`` to import under.
            Defaults to the version DT reads today; pass it explicitly to pin a
            test to a different one, so this factory does not need renaming each
            time the document schema moves.
    """

    def _import(s3_uri, file_format="HF_Prompt_Completion", document_schema_version="4.0.0"):
        parsed = urlparse(s3_uri)
        response = sagemaker_session.sagemaker_client.import_hub_content(
            HubName=integ_hub_name,
            HubContentName=f"dt-integ-{uuid.uuid4().hex[:8]}",
            HubContentType="DataSet",
            DocumentSchemaVersion=document_schema_version,
            HubContentDocument=json.dumps(
                {
                    "DatasetFileFormat": file_format,
                    "DatasetS3Bucket": parsed.netloc,
                    "DatasetS3Prefix": parsed.path.lstrip("/"),
                    "DatasetS3Uri": s3_uri,
                }
            ),
            HubContentDescription=f"v{document_schema_version} BYOD dataset for DT integ test",
        )
        return response["HubContentArn"]

    return _import


@pytest.fixture(scope="session")
def s3_fixtures(sagemaker_session, integ_bucket):
    """Upload fixture files to a temp S3 prefix. Clean up after."""
    bucket = integ_bucket
    prefix = f"data-preparer-integ-{uuid.uuid4().hex[:8]}"
    s3 = boto3.client("s3", region_name=sagemaker_session.boto_region_name)

    uploaded = {}
    for fixture_file in FIXTURES_DIR.iterdir():
        if fixture_file.is_file():
            key = f"{prefix}/{fixture_file.name}"
            s3.upload_file(str(fixture_file), bucket, key)
            uploaded[fixture_file.name] = f"s3://{bucket}/{key}"

    yield uploaded

    # Cleanup
    for fixture_file in FIXTURES_DIR.iterdir():
        if fixture_file.is_file():
            key = f"{prefix}/{fixture_file.name}"
            try:
                s3.delete_object(Bucket=bucket, Key=key)
            except Exception:
                pass
