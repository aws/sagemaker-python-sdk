"""Tests for SAGEMAKER_HUB_NAME env-var override of the HUB_NAME constant."""
from __future__ import absolute_import

import importlib
import os
from unittest.mock import patch


def _reload_hub_name():
    """Reload the constants module under the current env and return HUB_NAME."""
    from sagemaker.train import constants
    importlib.reload(constants)
    return constants.HUB_NAME


def test_hub_name_defaults_to_public_hub():
    """When SAGEMAKER_HUB_NAME is unset, HUB_NAME is SageMakerPublicHub."""
    env = {k: v for k, v in os.environ.items() if k != "SAGEMAKER_HUB_NAME"}
    with patch.dict(os.environ, env, clear=True):
        assert _reload_hub_name() == "SageMakerPublicHub"


def test_hub_name_overridden_by_env_var():
    """When SAGEMAKER_HUB_NAME is set, HUB_NAME reflects the override."""
    with patch.dict(os.environ, {"SAGEMAKER_HUB_NAME": "MyPrivateHub"}):
        assert _reload_hub_name() == "MyPrivateHub"
