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
"""This module contains the Integ Tests for JumpStart Training.

Coverage:
  * Public JumpStart models (model_id only).
  * Private-hub ModelReference (a pointer to a public model), including an
    aliased reference whose hub content name differs from the public model_id.
  * A privately-owned Model authored directly into a private hub.

The private-hub / private-model tests each create their own temporary hub,
populate it, run training, and tear the hub down. They skip gracefully if the
environment lacks permissions to create hubs or import content.
"""

from __future__ import absolute_import

import time
import uuid
import logging

import pytest
from botocore.exceptions import ClientError

from sagemaker.core.jumpstart import JumpStartConfig
from sagemaker.core.helper.session_helper import Session
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute

logger = logging.getLogger(__name__)

# A trainable classical-ML model keeps these tests fast/cheap on CPU.
TRAINABLE_MODEL_ID = "catboost-regression-model"
# Gated, trainable model reused from the v2 private-hub parity tests. Exercises
# the accept_eula / ModelAccessConfig path for a gated ModelReference. Gated
# models resolve to GPU and require real EULA acceptance, so the test that uses
# it runs a real training job and is marked slow_test + gpu_intensive (scheduled
# CI, not PR checks). The instance type is intentionally left to the SDK:
# from_jumpstart_config validates a supplied instance_type against the model's
# SupportedTrainingInstanceTypes and raises if it is not in the list, so
# resolving the model's own default is safer than hardcoding one here.
GATED_TRAINABLE_MODEL_ID = "meta-textgeneration-llama-3-2-1b"
HUB_NAME_PREFIX = "sdk-integ-train-hub"
ALIASED_REFERENCE_NAME = "sdk-integ-aliased-catboost"
PRIVATE_MODEL_NAME = "sdk-integ-private-catboost"

# Only these error codes represent "this restricted account is not allowed to
# set up the fixture" and warrant a graceful skip. Any other ClientError is a
# real failure (e.g. a service-side regression in create_hub_content_reference
# or import_hub_content) and must fail loudly so the test does not silently
# stop guarding the fix while CI stays green.
_SKIPPABLE_SETUP_ERROR_CODES = frozenset(
    {
        "AccessDeniedException",
        "AccessForbiddenException",
        "UnauthorizedOperation",
    }
)


def _skip_if_unauthorized(e, message):
    """Skip only on an expected authorization error; re-raise everything else."""
    if e.response.get("Error", {}).get("Code") in _SKIPPABLE_SETUP_ERROR_CODES:
        pytest.skip(f"{message}: {e}")
    raise


def _assert_reference_channels(model_trainer):
    """Assert the SDK resolved a ModelReference into hub-aware training channels.

    Resolving a ModelReference must produce a container image and attach a
    HubAccessConfig(hub_content_arn=...) to the model channel (defaults.py,
    hub_content_type == "ModelReference" branch). Asserting this on the
    SDK-resolved channels — rather than passing an explicit training channel to
    train() — is what actually guards the fix; a non-gated model would train
    fine even if this plumbing regressed.
    """
    assert model_trainer.training_image
    model_channels = [
        c for c in model_trainer.input_data_config if getattr(c, "channel_name", None) == "model"
    ]
    assert len(model_channels) == 1
    hub_access_config = model_channels[0].data_source.s3_data_source.hub_access_config
    # A resolved reference must carry a real HubAccessConfig. Use a truthy check
    # (not `is not None`): the field's unset default is the Unassigned() sentinel,
    # so `is not None` would wrongly pass if the plumbing regressed and left it
    # unset. Unassigned() is falsy, a real HubAccessConfig is truthy.
    assert hub_access_config
    assert hub_access_config.hub_content_arn


def _assert_gated_reference_channels(model_trainer):
    """Assert accept_eula flowed into the model channel's ModelAccessConfig.

    Verified against defaults.py get_model_artifact_input: the resolved "model"
    channel always carries
    data_source.s3_data_source.model_access_config = ModelAccessConfig(
        accept_eula=jumpstart_config.accept_eula). Using a gated model_id is what
    makes accept_eula=True meaningful (a gated model is unusable without it); the
    assertion itself is the same plumbing every JumpStart model uses.

    Asserted before .train() so the gated ModelAccessConfig plumbing is pinned
    even if the training job itself later fails for an unrelated capacity/quota
    reason.
    """
    assert model_trainer.training_image
    model_channels = [
        c for c in model_trainer.input_data_config if getattr(c, "channel_name", None) == "model"
    ]
    assert len(model_channels) == 1
    model_access_config = model_channels[0].data_source.s3_data_source.model_access_config
    # Truthy check rather than `is not None`: the unset default is Unassigned()
    # (falsy), which `is not None` would let through; a real ModelAccessConfig is
    # truthy.
    assert model_access_config, "gated reference resolved without a ModelAccessConfig"
    assert model_access_config.accept_eula is True


def _assert_owned_model_channels(model_trainer):
    """Assert a privately-owned Model resolved into direct (non-brokered) channels.

    An owned Model (not a reference) must resolve to a model channel with a real
    S3 artifact URI and NO HubAccessConfig — the inverse of the reference case.
    """
    assert model_trainer.training_image
    model_channels = [
        c for c in model_trainer.input_data_config if getattr(c, "channel_name", None) == "model"
    ]
    assert len(model_channels) == 1
    s3_source = model_channels[0].data_source.s3_data_source
    assert s3_source.s3_uri
    # An owned Model gets no HubAccessConfig. The field's default is the
    # Unassigned() sentinel (not None), so assert it was never populated via a
    # falsy check — both Unassigned() and None are falsy — rather than `is None`.
    assert not s3_source.hub_access_config


def _sm_client(sagemaker_session):
    return sagemaker_session.boto_session.client("sagemaker")


def _region(sagemaker_session):
    return sagemaker_session.boto_region_name


def _execution_role(sagemaker_session):
    """Resolve a SageMaker execution role from the running environment."""
    return sagemaker_session.get_caller_identity_arn()


def _public_model_arn(region, model_id):
    return f"arn:aws:sagemaker:{region}:aws:hub-content/" f"SageMakerPublicHub/Model/{model_id}"


def _wait_for_content(sm, hub_name, name, content_type, timeout=300, poll=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = sm.describe_hub_content(
                HubName=hub_name,
                HubContentName=name,
                HubContentType=content_type,
            )
            if resp.get("HubContentStatus") == "Available":
                return True
        except ClientError:
            pass
        time.sleep(poll)
    return False


def _delete_hub(sm, hub_name):
    for content_type in ("ModelReference", "Model"):
        try:
            resp = sm.list_hub_contents(HubName=hub_name, HubContentType=content_type)
        except ClientError:
            continue
        for c in resp.get("HubContentSummaries", []):
            try:
                if content_type == "ModelReference":
                    sm.delete_hub_content_reference(
                        HubName=hub_name,
                        HubContentType=content_type,
                        HubContentName=c["HubContentName"],
                    )
                else:
                    sm.delete_hub_content(
                        HubName=hub_name,
                        HubContentType=content_type,
                        HubContentName=c["HubContentName"],
                        HubContentVersion=c["HubContentVersion"],
                    )
            except ClientError as e:
                logger.warning("Failed to delete hub content %s: %s", c, e)
    try:
        sm.delete_hub(HubName=hub_name)
    except ClientError as e:
        logger.warning("Failed to delete hub %s: %s", hub_name, e)


@pytest.fixture(scope="module")
def sagemaker_session():
    return Session()


@pytest.fixture(scope="module")
def private_hub(sagemaker_session):
    """Create a temporary private hub; tear it (and its contents) down after."""
    sm = _sm_client(sagemaker_session)
    hub_name = f"{HUB_NAME_PREFIX}-{uuid.uuid4().hex[:8]}"
    try:
        sm.create_hub(
            HubName=hub_name,
            HubDescription="SDK integ test JumpStart training private hub",
        )
    except ClientError as e:
        _skip_if_unauthorized(e, "Cannot create private hub (missing permissions?)")

    for _ in range(30):
        if sm.describe_hub(HubName=hub_name)["HubStatus"] == "InService":
            break
        time.sleep(2)
    else:
        pytest.fail(f"Hub {hub_name} did not reach InService")

    yield hub_name
    _delete_hub(sm, hub_name)


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "model_id": "huggingface-spc-bert-base-cased",
            "hyperparameters": {
                "epochs": 1,  # Set to 1 for testing purposes
            },
            # Override default instance type; the model's default
            # (ml.p3.2xlarge) is deprecated.
            "compute": Compute(instance_type="ml.g4dn.xlarge"),
        },
        {"model_id": "xgboost-classification-model"},
        {"model_id": "catboost-regression-model"},
    ],
    ids=[
        "huggingface-spc-bert-base-cased",
        "xgboost-classification-model",
        "catboost-regression-model",
    ],
)
def test_jumpstart_train(test_case):
    """Test JumpStart training from a public model_id."""
    jumpstart = JumpStartConfig(
        model_id=test_case["model_id"],
        accept_eula=test_case.get("accept_eula", False),
    )
    model_trainer = ModelTrainer.from_jumpstart_config(
        jumpstart,
        base_job_name=test_case["model_id"],
        hyperparameters=test_case.get("hyperparameters", {}),
        compute=test_case.get("compute"),
    )
    model_trainer.train()


def test_jumpstart_train_from_private_hub_reference(private_hub, sagemaker_session):
    """Train from a ModelReference (pointer to a public model) in a private hub."""
    sm = _sm_client(sagemaker_session)
    region = _region(sagemaker_session)

    try:
        sm.create_hub_content_reference(
            HubName=private_hub,
            SageMakerPublicHubContentArn=_public_model_arn(region, TRAINABLE_MODEL_ID),
        )
    except ClientError as e:
        _skip_if_unauthorized(e, "Cannot create hub content reference")
    if not _wait_for_content(sm, private_hub, TRAINABLE_MODEL_ID, "ModelReference"):
        pytest.fail(
            f"ModelReference {TRAINABLE_MODEL_ID} did not become Available in {private_hub}"
        )

    jumpstart = JumpStartConfig(model_id=TRAINABLE_MODEL_ID, hub_name=private_hub, accept_eula=True)
    model_trainer = ModelTrainer.from_jumpstart_config(
        jumpstart,
        role=_execution_role(sagemaker_session),
        base_job_name="sdk-integ-train-ref",
        compute=Compute(instance_type="ml.m5.xlarge"),
        sagemaker_session=sagemaker_session,
    )

    # Assert the fix's plumbing on the SDK-resolved channels before training:
    # resolving a ModelReference must attach a HubAccessConfig(hub_content_arn=...)
    # to the model channel. This is the assertion that would catch the plumbing
    # regressing; a non-gated model would otherwise train fine even if it broke.
    _assert_reference_channels(model_trainer)

    # Train on the SDK-resolved channels (no explicit training channel), so the
    # hub-aware channel construction under test is actually exercised.
    model_trainer.train()


@pytest.mark.slow_test
@pytest.mark.gpu_intensive
def test_jumpstart_train_from_gated_reference(private_hub, sagemaker_session):
    """Train from a GATED ModelReference in a private hub, verifying the
    accept_eula / ModelAccessConfig path.

    Gated models resolve to a GPU instance and require real EULA acceptance, so
    this runs a real training job and is marked gpu_intensive (submits a real job
    that consumes training capacity; scheduled CI, not PR checks) as well as
    slow_test. The resolved channels are asserted before training so the fix's
    ModelAccessConfig/HubAccessConfig plumbing is pinned even if the job itself
    later fails for an unrelated capacity/quota reason."""
    sm = _sm_client(sagemaker_session)
    region = _region(sagemaker_session)

    try:
        sm.create_hub_content_reference(
            HubName=private_hub,
            SageMakerPublicHubContentArn=_public_model_arn(region, GATED_TRAINABLE_MODEL_ID),
        )
    except ClientError as e:
        _skip_if_unauthorized(e, "Cannot create gated hub content reference")
    if not _wait_for_content(sm, private_hub, GATED_TRAINABLE_MODEL_ID, "ModelReference"):
        pytest.fail(
            f"Gated reference {GATED_TRAINABLE_MODEL_ID} did not become Available in {private_hub}"
        )

    jumpstart = JumpStartConfig(
        model_id=GATED_TRAINABLE_MODEL_ID,
        hub_name=private_hub,
        accept_eula=True,
    )
    model_trainer = ModelTrainer.from_jumpstart_config(
        jumpstart,
        role=_execution_role(sagemaker_session),
        base_job_name="sdk-integ-train-gated-ref",
        # No compute: let from_jumpstart_config resolve the gated model's own
        # default (GPU) instance type. Passing one risks a ValueError if it is
        # not in the model's SupportedTrainingInstanceTypes.
        sagemaker_session=sagemaker_session,
    )

    # Pin the fix's plumbing on the SDK-resolved channels before training: a gated
    # ModelReference must resolve with accept_eula flowed into a ModelAccessConfig,
    # plus the HubAccessConfig every reference gets.
    _assert_reference_channels(model_trainer)
    _assert_gated_reference_channels(model_trainer)

    # Train on the SDK-resolved channels (no explicit training channel), so the
    # hub-aware, gated channel construction under test is actually exercised
    # end-to-end against a real training job.
    model_trainer.train()


def test_jumpstart_train_from_aliased_reference(private_hub, sagemaker_session):
    """Train from a ModelReference filed under an alias that differs from the
    public model_id (exercises hub_content_name resolution)."""
    sm = _sm_client(sagemaker_session)
    region = _region(sagemaker_session)

    try:
        sm.create_hub_content_reference(
            HubName=private_hub,
            SageMakerPublicHubContentArn=_public_model_arn(region, TRAINABLE_MODEL_ID),
            HubContentName=ALIASED_REFERENCE_NAME,
        )
    except ClientError as e:
        _skip_if_unauthorized(e, "Cannot create aliased hub content reference")
    if not _wait_for_content(sm, private_hub, ALIASED_REFERENCE_NAME, "ModelReference"):
        pytest.fail(f"Aliased reference {ALIASED_REFERENCE_NAME} did not become Available")

    jumpstart = JumpStartConfig(
        model_id=TRAINABLE_MODEL_ID,
        hub_name=private_hub,
        hub_content_name=ALIASED_REFERENCE_NAME,
        accept_eula=True,
    )
    model_trainer = ModelTrainer.from_jumpstart_config(
        jumpstart,
        role=_execution_role(sagemaker_session),
        base_job_name="sdk-integ-train-alias",
        compute=Compute(instance_type="ml.m5.xlarge"),
        sagemaker_session=sagemaker_session,
    )

    # The alias must have been threaded through resolution (not the model_id).
    assert model_trainer._jumpstart_config.hub_content_name == ALIASED_REFERENCE_NAME
    # ...and resolving it as a ModelReference must attach the hub-aware channels.
    _assert_reference_channels(model_trainer)

    # Train on the SDK-resolved channels (no explicit training channel).
    model_trainer.train()


def test_jumpstart_train_from_private_owned_model(private_hub, sagemaker_session):
    """Train from a privately-owned Model authored directly into a private hub
    (content-type Model, not a ModelReference). Exercises the document.py
    fallback-to-Model resolution probe."""
    sm = _sm_client(sagemaker_session)

    # Author a private Model by importing a trainable public model's document
    # into the private hub as content-type Model.
    try:
        public = sm.describe_hub_content(
            HubName="SageMakerPublicHub",
            HubContentType="Model",
            HubContentName=TRAINABLE_MODEL_ID,
        )
    except ClientError as e:
        _skip_if_unauthorized(e, "Cannot read public model document")

    try:
        sm.import_hub_content(
            HubName=private_hub,
            HubContentName=PRIVATE_MODEL_NAME,
            HubContentType="Model",
            HubContentDocument=public["HubContentDocument"],
            DocumentSchemaVersion=public.get("DocumentSchemaVersion", "2.0.0"),
            HubContentDisplayName=public.get("HubContentDisplayName", PRIVATE_MODEL_NAME),
            HubContentDescription="Privately owned model for integ test",
            HubContentMarkdown=public.get("HubContentMarkdown", ""),
            HubContentSearchKeywords=public.get("HubContentSearchKeywords", []),
        )
    except ClientError as e:
        # Only skip if the account is simply not allowed to author a private
        # Model. Any other failure is a real regression in the owned-Model path
        # (the core case this fix enables) and must fail loudly.
        _skip_if_unauthorized(e, "import_hub_content for a private Model not permitted")
    if not _wait_for_content(sm, private_hub, PRIVATE_MODEL_NAME, "Model"):
        pytest.fail(f"Private Model {PRIVATE_MODEL_NAME} did not become Available in {private_hub}")

    jumpstart = JumpStartConfig(
        model_id=TRAINABLE_MODEL_ID,
        hub_name=private_hub,
        hub_content_name=PRIVATE_MODEL_NAME,
        accept_eula=True,
    )
    model_trainer = ModelTrainer.from_jumpstart_config(
        jumpstart,
        role=_execution_role(sagemaker_session),
        base_job_name="sdk-integ-train-private",
        compute=Compute(instance_type="ml.m5.xlarge"),
        sagemaker_session=sagemaker_session,
    )

    # Owned Model (fallback-to-Model probe): direct S3 artifact, no HubAccessConfig.
    _assert_owned_model_channels(model_trainer)

    # Train on the SDK-resolved channels (no explicit training channel).
    model_trainer.train()
