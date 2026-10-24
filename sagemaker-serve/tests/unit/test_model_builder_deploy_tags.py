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
"""``ModelBuilder.deploy(tags=...)`` is one tag contract for every deploy path.

Before this change ``tags`` reached ``deploy()`` only through ``**kwargs``, so
``inspect.signature`` could not validate it, the fine-tuned (model customization) and
Nova paths ignored it, and ``add_tags`` raised ``KeyError: 'Key'`` on the
``{"key": ..., "value": ...}`` form that the ``sagemaker.core.resources`` classes use.
Every representation now comes out with string keys and values, and the tags a call
passes are merged for that call only, so a second ``deploy()`` on the same builder does
not carry the first call's tags.
"""

from __future__ import absolute_import

import inspect
import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest

from sagemaker.core.common_utils import _validate_new_tags, format_tags, remove_tag_with_key
from sagemaker.core.enums import EndpointType
from sagemaker.core.inference_config import ResourceRequirements
from sagemaker.core.resources import Endpoint
from sagemaker.core.shapes import Tag
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer

ROLE_ARN = "arn:aws:iam::123456789012:role/TestRole"
REGION = "us-east-2"
IMAGE_URI = "123456789012.dkr.ecr.us-east-2.amazonaws.com/inference:latest"
BOTO_TAG = {"Key": "project", "Value": "example"}
CORE_TAG = {"key": "project", "value": "example"}


def _session():
    session = Mock()
    session.boto_region_name = REGION
    session.default_bucket.return_value = "test-bucket"
    session.default_bucket_prefix = None
    session.config = {}
    session.sagemaker_config = {}
    session.settings = Mock(include_jumpstart_tags=False)
    session.settings._local_download_dir = None
    session.sagemaker_client = MagicMock()
    session.boto_session = Mock()
    session.boto_session.region_name = REGION
    return session


def _make_builder():
    builder = ModelBuilder(
        model=Mock(),
        model_server=ModelServer.TORCHSERVE,
        image_uri=IMAGE_URI,
        s3_model_data_url="s3://test-bucket/model.tar.gz",
        role_arn=ROLE_ARN,
        instance_type="ml.m5.large",
        sagemaker_session=_session(),
    )
    builder.region = REGION
    builder.built_model = Mock(model_name="built-model")
    builder.model_server = None
    return builder


class TestTagNormalization(unittest.TestCase):
    """``format_tags`` and ``add_tags`` accept every tag representation the SDK uses."""

    def test_format_tags_accepts_every_representation(self):
        expected = [BOTO_TAG]

        self.assertEqual(format_tags({"project": "example"}), expected)
        self.assertEqual(format_tags(BOTO_TAG), expected)
        self.assertEqual(format_tags([BOTO_TAG]), expected)
        self.assertEqual(format_tags([CORE_TAG]), expected)
        self.assertEqual(format_tags([Tag(key="project", value="example")]), expected)
        self.assertEqual(format_tags([("project", "example")]), expected)
        self.assertEqual(
            format_tags({"n": 1}),
            format_tags([("n", 1)]),
            format_tags([{"key": "n", "value": 1}]),
        )
        self.assertEqual(format_tags([("n", 1)]), [{"Key": "n", "Value": "1"}])
        self.assertEqual(
            format_tags([BOTO_TAG, {"key": "team", "value": "ml"}]),
            [BOTO_TAG, {"Key": "team", "Value": "ml"}],
        )
        self.assertIsNone(format_tags(None))

    def test_format_tags_rejects_a_tag_without_a_key(self):
        with pytest.raises(TypeError, match="'Key'"):
            format_tags([{"Value": "no-key"}])
        with pytest.raises(TypeError, match="Tag object"):
            format_tags(["project=example"])

    def test_validate_new_tags_merges_mixed_casing_by_key(self):
        merged = _validate_new_tags([CORE_TAG, {"key": "team", "value": "ml"}], [BOTO_TAG])

        self.assertEqual(merged, [BOTO_TAG, {"Key": "team", "Value": "ml"}])

    def test_validate_new_tags_with_tag_objects(self):
        merged = _validate_new_tags([Tag(key="team", value="ml")], [CORE_TAG])

        self.assertEqual(merged, [BOTO_TAG, {"Key": "team", "Value": "ml"}])

    def test_remove_tag_with_key_accepts_core_casing(self):
        self.assertIsNone(remove_tag_with_key("project", [CORE_TAG]))

    def test_add_tags_normalizes_core_casing_to_boto(self):
        builder = _make_builder()

        builder.add_tags([CORE_TAG])
        builder.add_tags({"team": "ml"})
        builder.add_tags([Tag(key="env", value="prod")])

        self.assertEqual(
            builder._tags,
            [BOTO_TAG, {"Key": "team", "Value": "ml"}, {"Key": "env", "Value": "prod"}],
        )


class TestDeployTagsParameter(unittest.TestCase):
    """``deploy(tags=...)`` is a named parameter and reaches every deploy path."""

    def test_deploy_has_a_tags_parameter(self):
        parameters = inspect.signature(ModelBuilder.deploy).parameters

        self.assertIn("tags", parameters)
        self.assertIsNone(parameters["tags"].default)

    def test_deploy_merges_tags_into_the_builder_before_the_core_path(self):
        builder = _make_builder()
        builder.add_tags({"team": "ml"})
        seen = {}

        def _capture(**kwargs):
            seen["tags"] = list(builder._tags)
            seen["kwargs_tags"] = kwargs.get("tags")
            return Mock(spec=Endpoint)

        with (
            patch.object(ModelBuilder, "_deploy", side_effect=_capture),
            patch.object(ModelBuilder, "_is_model_customization", return_value=False),
            patch.object(ModelBuilder, "_resolve_model_source_id", return_value=None),
        ):
            builder.deploy(endpoint_name="ep", tags=[CORE_TAG])

        self.assertEqual(seen["tags"], [{"Key": "team", "Value": "ml"}, BOTO_TAG])
        # The core path reads self._tags; no duplicate copy travels through kwargs.
        self.assertIsNone(seen["kwargs_tags"])

    def test_deploy_core_endpoint_normalizes_tags_passed_through_kwargs(self):
        builder = _make_builder()
        builder.built_model.model_name = "built-model"
        builder.model_name = "built-model"
        builder.sagemaker_session.endpoint_in_service_or_not.return_value = False
        seen = {}

        def _capture(**kwargs):
            seen["tags"] = kwargs.get("tags")
            return "ep"

        builder.sagemaker_session.endpoint_from_production_variants = Mock(side_effect=_capture)
        with (
            patch(
                "sagemaker.serve.model_builder.session_helper.production_variant",
                return_value={"VariantName": "AllTraffic"},
            ),
            patch("sagemaker.serve.model_builder.Endpoint.get", return_value=Mock(spec=Endpoint)),
            patch.object(builder, "_wait_for_endpoint"),
        ):
            builder._deploy_core_endpoint(
                instance_type="ml.g5.xlarge",
                initial_instance_count=1,
                endpoint_name="ep",
                endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
                resources=ResourceRequirements(requests={"memory": 1024, "copies": 1}),
                inference_component_name="ic",
                tags=[CORE_TAG, Tag(key="team", value="ml")],
            )

        self.assertEqual(seen["tags"], [BOTO_TAG, {"Key": "team", "Value": "ml"}])
        self.assertEqual(builder._tags, [BOTO_TAG, {"Key": "team", "Value": "ml"}])

    def test_deploy_passes_the_builder_tags_to_a_recommendation_deploy(self):
        builder = _make_builder()
        builder._recommendation_job = Mock()
        seen = {}

        def _capture(**kwargs):
            seen["tags"] = kwargs["tags"]
            return Mock(spec=Endpoint)

        with patch.object(ModelBuilder, "_deploy_recommendation", side_effect=_capture):
            builder.deploy(tags={"project": "example"})

        self.assertEqual(seen["tags"], [BOTO_TAG])

    def test_core_endpoint_tags_use_the_core_resource_casing(self):
        builder = _make_builder()
        builder.add_tags([BOTO_TAG, Tag(key="team", value="ml")])

        self.assertEqual(
            builder._core_endpoint_tags(),
            [CORE_TAG, {"key": "team", "value": "ml"}],
        )

    def test_model_customization_restricted_package_endpoint_is_tagged(self):
        builder = _make_builder()
        builder.instance_type = "ml.g5.xlarge"
        created = {}

        def _create_endpoint(**kwargs):
            created.update(kwargs)
            endpoint = Mock(spec=Endpoint)
            endpoint.wait_for_status = Mock()
            return endpoint

        with (
            patch.object(ModelBuilder, "_is_nova_model", return_value=False),
            patch.object(ModelBuilder, "_fetch_model_package", return_value=Mock()),
            patch(
                "sagemaker.serve.utils.model_package_utils.is_restricted_model_package",
                return_value=True,
            ),
            patch("sagemaker.serve.model_builder.EndpointConfig.create"),
            patch("sagemaker.serve.model_builder.Endpoint.create", side_effect=_create_endpoint),
        ):
            builder.add_tags([CORE_TAG])
            builder._deploy_model_customization(endpoint_name="ep", wait=False)

        self.assertEqual(created["tags"], [CORE_TAG])

    def test_nova_endpoint_merges_the_builder_tags(self):
        builder = _make_builder()
        builder.instance_type = "ml.g5.xlarge"
        created = {}

        def _create_endpoint(**kwargs):
            created.update(kwargs)
            return Mock(spec=Endpoint)

        with (
            patch.object(ModelBuilder, "_base_model_name", return_value="nova-micro"),
            patch.object(ModelBuilder, "_fetch_model_package", return_value=None),
            patch("sagemaker.serve.model_builder.EndpointConfig.create"),
            patch("sagemaker.serve.model_builder.Endpoint.create", side_effect=_create_endpoint),
        ):
            builder.add_tags([CORE_TAG, {"Key": "sagemaker-sdk:jumpstart-model-id", "Value": "x"}])
            builder._deploy_nova_model(endpoint_name="ep", wait=False)

        self.assertEqual(
            created["tags"],
            [{"key": "sagemaker-sdk:jumpstart-model-id", "value": "nova-micro"}, CORE_TAG],
        )

    def test_deploy_tags_do_not_outlive_the_call(self):
        builder = _make_builder()
        builder.add_tags({"team": "ml"})
        seen = {}

        def _capture(**kwargs):
            seen["tags"] = list(builder._tags)
            return Mock(spec=Endpoint)

        with (
            patch.object(ModelBuilder, "_deploy", side_effect=_capture),
            patch.object(ModelBuilder, "_is_model_customization", return_value=False),
            patch.object(ModelBuilder, "_resolve_model_source_id", return_value="s3://b/m"),
        ):
            builder.deploy(endpoint_name="ep", tags={"run": "1"})

        self.assertEqual(
            [tag["Key"] for tag in seen["tags"]],
            ["team", "run", "sagemaker.amazonaws.com/model-source"],
        )
        self.assertEqual(builder._tags, [{"Key": "team", "Value": "ml"}])

    def test_deploy_tags_are_restored_when_the_path_raises(self):
        builder = _make_builder()
        builder.add_tags({"team": "ml"})

        with (
            patch.object(ModelBuilder, "_deploy", side_effect=RuntimeError("boom")),
            patch.object(ModelBuilder, "_is_model_customization", return_value=False),
            patch.object(ModelBuilder, "_resolve_model_source_id", return_value=None),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                builder.deploy(endpoint_name="ep", tags={"run": "1"})

        self.assertEqual(builder._tags, [{"Key": "team", "Value": "ml"}])
