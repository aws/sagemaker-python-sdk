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
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, InputData

logger = logging.getLogger(__name__)

# A trainable classical-ML model keeps these tests fast/cheap on CPU.
TRAINABLE_MODEL_ID = "catboost-regression-model"
HUB_NAME_PREFIX = "sdk-integ-train-hub"
ALIASED_REFERENCE_NAME = "sdk-integ-aliased-catboost"
PRIVATE_MODEL_NAME = "sdk-integ-private-catboost"


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
            resp = sm.list_hub_contents(HubName=hub_name, HubContentType=content_type)
            if any(
                c["HubContentName"] == name and c.get("HubContentStatus") == "Available"
                for c in resp.get("HubContentSummaries", [])
            ):
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


def _default_training_dataset(region, model_id):
    """Resolve the model's default training dataset S3 URI from JS metadata."""
    from sagemaker.core.jumpstart.accessors import JumpStartModelsAccessor

    specs = JumpStartModelsAccessor.get_model_specs(region=region, model_id=model_id, version="*")
    if not getattr(specs, "training_supported", False):
        return None
    key = getattr(specs, "default_training_dataset_key", None)
    if not key:
        return None
    return f"s3://jumpstart-cache-prod-{region}/{key}"


@pytest.fixture(scope="module")
def sagemaker_session():
    from sagemaker.core.helper.session_helper import Session

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
        pytest.skip(f"Cannot create private hub (missing permissions?): {e}")

    for _ in range(30):
        if sm.describe_hub(HubName=hub_name)["HubStatus"] == "InService":
            break
        time.sleep(2)
    else:
        pytest.skip(f"Hub {hub_name} did not reach InService")

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
        pytest.skip(f"Cannot create hub content reference: {e}")
    if not _wait_for_content(sm, private_hub, TRAINABLE_MODEL_ID, "ModelReference"):
        pytest.skip(f"ModelReference {TRAINABLE_MODEL_ID} not available in {private_hub}")

    dataset = _default_training_dataset(region, TRAINABLE_MODEL_ID)
    if dataset is None:
        pytest.skip(f"{TRAINABLE_MODEL_ID} is not trainable / has no default dataset")

    jumpstart = JumpStartConfig(model_id=TRAINABLE_MODEL_ID, hub_name=private_hub, accept_eula=True)
    model_trainer = ModelTrainer.from_jumpstart_config(
        jumpstart,
        role=_execution_role(sagemaker_session),
        base_job_name="sdk-integ-train-ref",
        compute=Compute(instance_type="ml.m5.xlarge"),
        sagemaker_session=sagemaker_session,
    )
    model_trainer.train(input_data_config=[InputData(channel_name="training", data_source=dataset)])


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
        pytest.skip(f"Cannot create aliased hub content reference: {e}")
    if not _wait_for_content(sm, private_hub, ALIASED_REFERENCE_NAME, "ModelReference"):
        pytest.skip(f"Aliased reference {ALIASED_REFERENCE_NAME} not available")

    dataset = _default_training_dataset(region, TRAINABLE_MODEL_ID)
    if dataset is None:
        pytest.skip(f"{TRAINABLE_MODEL_ID} is not trainable / has no default dataset")

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
    model_trainer.train(input_data_config=[InputData(channel_name="training", data_source=dataset)])


def test_jumpstart_train_from_private_owned_model(private_hub, sagemaker_session):
    """Train from a privately-owned Model authored directly into a private hub
    (content-type Model, not a ModelReference). Exercises the document.py
    fallback-to-Model resolution probe."""
    sm = _sm_client(sagemaker_session)
    region = _region(sagemaker_session)

    # Author a private Model by importing a trainable public model's document
    # into the private hub as content-type Model.
    try:
        public = sm.describe_hub_content(
            HubName="SageMakerPublicHub",
            HubContentType="Model",
            HubContentName=TRAINABLE_MODEL_ID,
        )
    except ClientError as e:
        pytest.skip(f"Cannot read public model document: {e}")

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
        pytest.skip(f"import_hub_content for a private Model not permitted/supported: {e}")
    if not _wait_for_content(sm, private_hub, PRIVATE_MODEL_NAME, "Model"):
        pytest.skip(f"Private Model {PRIVATE_MODEL_NAME} not available in {private_hub}")

    dataset = _default_training_dataset(region, TRAINABLE_MODEL_ID)
    if dataset is None:
        pytest.skip(f"{TRAINABLE_MODEL_ID} is not trainable / has no default dataset")

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
    model_trainer.train(input_data_config=[InputData(channel_name="training", data_source=dataset)])
