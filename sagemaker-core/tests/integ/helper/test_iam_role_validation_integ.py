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
"""Integration tests: interfaces validate roles and never auto-create them.

Each test drives a real interface (ModelTrainer, SFTTrainer, ModelBuilder) with a
role that does not exist, against REAL AWS IAM, and asserts that the SDK surfaces
a clear error (RoleValidationError / ValueError) instead of silently creating an
IAM role. It also confirms no ``SageMaker-AutoRole-*`` role was created as a side
effect — the core of the security change.

Run under pytest (the integ marker keeps it out of the unit run)::

    PYTHONPATH=src python -m pytest tests/integ/helper/test_iam_role_validation_integ.py -m integ -s

Requires AWS credentials with iam:GetRole (read-only).
"""
from __future__ import absolute_import

import logging
import os
import sys
import uuid

import boto3
import pytest
from botocore.exceptions import ClientError, NoCredentialsError

if __package__ in (None, ""):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from sagemaker.core.helper.iam_role_resolver import RoleValidationError  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iam_role_validation_integ")

# A role name that does not exist in the account, so resolution must fail rather
# than create anything. Unique per run to avoid any collision.
NONEXISTENT_ROLE = f"sdk-does-not-exist-{uuid.uuid4().hex[:12]}"


def _credentials_available() -> bool:
    try:
        boto3.Session().client("sts").get_caller_identity()
        return True
    except (NoCredentialsError, ClientError) as e:
        logger.warning("No usable AWS credentials (%s). Integ test will be skipped.", e)
        return False


def _no_autorole_created(role_type_name: str) -> bool:
    """Return True if the well-known SageMaker-AutoRole-* role was NOT created."""
    iam = boto3.Session().client("iam")
    try:
        iam.get_role(RoleName=role_type_name)
        return False  # it exists — would indicate auto-creation
    except ClientError as e:
        return e.response["Error"]["Code"] in ("NoSuchEntity", "NoSuchEntityException")


pytestmark = pytest.mark.skipif(
    not _credentials_available(), reason="AWS credentials not available"
)


@pytest.mark.integ
def test_model_trainer_validation_error_no_autocreate():
    """ModelTrainer with a non-existent role raises and creates no auto-role."""
    from sagemaker.train.model_trainer import ModelTrainer

    with pytest.raises((RoleValidationError, ValueError)):
        ModelTrainer(
            training_image="123456789012.dkr.ecr.us-east-1.amazonaws.com/img:latest",
            role=NONEXISTENT_ROLE,
        ).train()
    assert _no_autorole_created("SageMaker-AutoRole-Training")


@pytest.mark.integ
def test_sft_trainer_validation_error_no_autocreate():
    """SFTTrainer with a non-existent role raises and creates no auto-role."""
    from sagemaker.train.sft_trainer import SFTTrainer
    from sagemaker.train.common import TrainingType

    with pytest.raises((RoleValidationError, ValueError)):
        SFTTrainer(
            model="nova-textgeneration-lite-v2",
            training_type=TrainingType.LORA,
            training_dataset="s3://example-bucket/train.jsonl",
            role=NONEXISTENT_ROLE,
        ).train(wait=False)
    assert _no_autorole_created("SageMaker-AutoRole-Training")


@pytest.mark.integ
def test_model_builder_validation_error_no_autocreate():
    """ModelBuilder with a non-existent role raises and creates no auto-role."""
    from sagemaker.serve.model_builder import ModelBuilder

    with pytest.raises((RoleValidationError, ValueError)):
        ModelBuilder(
            model="meta-textgeneration-llama-3-8b",
            role_arn=NONEXISTENT_ROLE,
        ).build()
    assert _no_autorole_created("SageMaker-AutoRole-Serving")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-m", "integ", "-s", "-v"]))
