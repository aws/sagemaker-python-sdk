"""Tests for SAGEMAKER_HUB_NAME env-var override via get_sagemaker_hub_name."""
from __future__ import absolute_import

import os
from unittest.mock import patch

from sagemaker.train.constants import get_sagemaker_hub_name


def test_get_sagemaker_hub_name_defaults_to_public_hub():
    """When SAGEMAKER_HUB_NAME is unset, returns SageMakerPublicHub."""
    env = {k: v for k, v in os.environ.items() if k != "SAGEMAKER_HUB_NAME"}
    with patch.dict(os.environ, env, clear=True):
        assert get_sagemaker_hub_name() == "SageMakerPublicHub"


def test_get_sagemaker_hub_name_overridden_by_env_var():
    """When SAGEMAKER_HUB_NAME is set, returns the override value."""
    with patch.dict(os.environ, {"SAGEMAKER_HUB_NAME": "MyPrivateHub"}):
        assert get_sagemaker_hub_name() == "MyPrivateHub"
