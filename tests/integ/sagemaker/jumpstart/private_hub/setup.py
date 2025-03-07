from __future__ import absolute_import

import os

import pytest
from sagemaker.jumpstart.hub.hub import Hub

from tests.integ.sagemaker.jumpstart.constants import (
    ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME,
)
from tests.integ.sagemaker.jumpstart.utils import (
    get_public_hub_model_arn,
    get_sm_session,
    with_exponential_backoff,
)


TEST_MODEL_IDS = {
    "catboost-classification-model",
    "huggingface-txt2img-conflictx-complex-lineart",
    "meta-textgeneration-llama-2-7b",
    "meta-textgeneration-llama-3-2-1b",
    "catboost-regression-model",
    "huggingface-spc-bert-base-cased",
}


@with_exponential_backoff()
def create_model_reference(hub_instance, model_arn):
    hub_instance.create_model_reference(model_arn=model_arn)


@pytest.fixture(scope="session")
def add_model_references():
    # Create Model References to test in Hub
    hub_instance = Hub(
        hub_name=os.environ[ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME], sagemaker_session=get_sm_session()
    )
    for model in TEST_MODEL_IDS:
        model_arn = get_public_hub_model_arn(hub_instance, model)
        create_model_reference(hub_instance, model_arn)
