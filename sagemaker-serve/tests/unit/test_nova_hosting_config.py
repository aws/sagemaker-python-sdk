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
"""Unit tests for Nova hosting config resolution in ModelBuilder.

Verifies that hosting configs published in the JumpStart hub document take
priority over the hardcoded ``_NOVA_HOSTING_CONFIGS`` fallback.
"""

import unittest
from unittest.mock import MagicMock, patch

from sagemaker.serve.model_builder import ModelBuilder


def _make_builder(region="us-east-1"):
    """Create a ModelBuilder without running __init__."""
    mb = ModelBuilder.__new__(ModelBuilder)
    mb.image_uri = None
    mb.env_vars = None
    mb.instance_type = None
    session = MagicMock()
    session.boto_region_name = region
    mb.sagemaker_session = session
    return mb


def _make_model_package(recipe_name="", hub_content_name="nova-textgeneration-lite"):
    pkg = MagicMock()
    base_model = MagicMock()
    base_model.recipe_name = recipe_name
    base_model.hub_content_name = hub_content_name
    pkg.inference_specification.containers = [MagicMock(base_model=base_model)]
    return pkg


class TestNovaHostingConfigResolution(unittest.TestCase):
    """Tests for ModelBuilder._get_nova_hosting_config priority behavior."""

    def test_hub_recipe_collection_config_takes_priority(self):
        """Hosting config from RecipeCollection in the hub doc is preferred."""
        mb = _make_builder()
        hub_doc = {
            "RecipeCollection": [
                {
                    "Name": "my-nova-recipe",
                    "HostingConfigs": [
                        {
                            "Profile": "Default",
                            "EcrAddress": "111.dkr.ecr.us-east-1.amazonaws.com/custom:tag",
                            "InstanceType": "ml.p5.48xlarge",
                            "Environment": {
                                "CONTEXT_LENGTH": "999",
                                "MAX_CONCURRENCY": "3",
                            },
                        }
                    ],
                }
            ]
        }
        mp = _make_model_package(
            recipe_name="my-nova-recipe", hub_content_name="nova-textgeneration-lite"
        )
        with patch.object(
            ModelBuilder, "_fetch_hub_document_for_custom_model", return_value=hub_doc
        ), patch.object(ModelBuilder, "_fetch_model_package", return_value=mp):
            cfg = mb._get_nova_hosting_config()

        self.assertEqual(
            cfg["image_uri"], "111.dkr.ecr.us-east-1.amazonaws.com/custom:tag"
        )
        self.assertEqual(
            cfg["env_vars"], {"CONTEXT_LENGTH": "999", "MAX_CONCURRENCY": "3"}
        )
        self.assertEqual(cfg["instance_type"], "ml.p5.48xlarge")

    def test_top_level_hosting_configs_used_when_no_recipe_match(self):
        """Top-level HostingConfigs is used when no RecipeCollection matches."""
        mb = _make_builder()
        hub_doc = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "EcrAddress": "222.dkr.ecr.us-east-1.amazonaws.com/top:tag",
                    "InstanceType": "ml.g6.24xlarge",
                    "Environment": {"CONTEXT_LENGTH": "100"},
                }
            ]
        }
        mp = _make_model_package(
            recipe_name="unmatched", hub_content_name="nova-textgeneration-micro"
        )
        with patch.object(
            ModelBuilder, "_fetch_hub_document_for_custom_model", return_value=hub_doc
        ), patch.object(ModelBuilder, "_fetch_model_package", return_value=mp):
            cfg = mb._get_nova_hosting_config()

        self.assertEqual(
            cfg["image_uri"], "222.dkr.ecr.us-east-1.amazonaws.com/top:tag"
        )

    def test_hardcoded_fallback_when_hub_has_no_hosting_config(self):
        """Hardcoded escrow config is used when the hub doc has no hosting config."""
        mb = _make_builder()
        mp = _make_model_package(hub_content_name="nova-textgeneration-lite")
        with patch.object(
            ModelBuilder, "_fetch_hub_document_for_custom_model", return_value={}
        ), patch.object(ModelBuilder, "_fetch_model_package", return_value=mp):
            cfg = mb._get_nova_hosting_config()

        self.assertIn("nova-inference-repo:SM-Inference-latest", cfg["image_uri"])
        self.assertEqual(cfg["instance_type"], "ml.g6.48xlarge")

    def test_hardcoded_fallback_when_hub_fetch_raises(self):
        """Hardcoded config is used defensively when hub fetch raises."""
        mb = _make_builder()
        mp = _make_model_package(hub_content_name="nova-textgeneration-pro")
        with patch.object(
            ModelBuilder,
            "_fetch_hub_document_for_custom_model",
            side_effect=RuntimeError("hub unavailable"),
        ), patch.object(ModelBuilder, "_fetch_model_package", return_value=mp):
            cfg = mb._get_nova_hosting_config()

        self.assertEqual(cfg["instance_type"], "ml.p5.48xlarge")
        self.assertIn("nova-inference-repo:SM-Inference-latest", cfg["image_uri"])

    def test_missing_ecr_address_falls_through_to_hardcoded(self):
        """A hub hosting config without EcrAddress falls back to the escrow image."""
        mb = _make_builder()
        hub_doc = {
            "RecipeCollection": [
                {
                    "Name": "r",
                    "HostingConfigs": [
                        {"Profile": "Default", "InstanceType": "ml.p5.48xlarge"}
                    ],
                }
            ]
        }
        mp = _make_model_package(
            recipe_name="r", hub_content_name="nova-textgeneration-pro"
        )
        with patch.object(
            ModelBuilder, "_fetch_hub_document_for_custom_model", return_value=hub_doc
        ), patch.object(ModelBuilder, "_fetch_model_package", return_value=mp):
            cfg = mb._get_nova_hosting_config()

        self.assertIn("nova-inference-repo:SM-Inference-latest", cfg["image_uri"])

    def test_instance_type_match_in_hub_config(self):
        """A requested instance type selects the matching hub config entry."""
        mb = _make_builder()
        hub_doc = {
            "RecipeCollection": [
                {
                    "Name": "r",
                    "HostingConfigs": [
                        {
                            "Profile": "Default",
                            "EcrAddress": "333.dkr.ecr.us-east-1.amazonaws.com/a:tag",
                            "InstanceType": "ml.p5.48xlarge",
                            "Environment": {"CONTEXT_LENGTH": "1"},
                        },
                        {
                            "EcrAddress": "333.dkr.ecr.us-east-1.amazonaws.com/b:tag",
                            "InstanceType": "ml.g6.48xlarge",
                            "Environment": {"CONTEXT_LENGTH": "2"},
                        },
                    ],
                }
            ]
        }
        mp = _make_model_package(
            recipe_name="r", hub_content_name="nova-textgeneration-lite"
        )
        with patch.object(
            ModelBuilder, "_fetch_hub_document_for_custom_model", return_value=hub_doc
        ), patch.object(ModelBuilder, "_fetch_model_package", return_value=mp):
            cfg = mb._get_nova_hosting_config(instance_type="ml.g6.48xlarge")

        self.assertEqual(
            cfg["image_uri"], "333.dkr.ecr.us-east-1.amazonaws.com/b:tag"
        )
        self.assertEqual(cfg["instance_type"], "ml.g6.48xlarge")

    def test_unsupported_instance_type_raises(self):
        """Requesting an unsupported instance type raises ValueError (fallback path)."""
        mb = _make_builder()
        mp = _make_model_package(hub_content_name="nova-textgeneration-pro")
        with patch.object(
            ModelBuilder, "_fetch_hub_document_for_custom_model", return_value={}
        ), patch.object(ModelBuilder, "_fetch_model_package", return_value=mp):
            with self.assertRaises(ValueError):
                mb._get_nova_hosting_config(instance_type="ml.invalid.type")


if __name__ == "__main__":
    unittest.main()
