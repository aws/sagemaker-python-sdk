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
"""Unit tests for BedrockModelBuilder."""

import json
import pytest
from unittest.mock import Mock, patch
from botocore.exceptions import ClientError

from sagemaker.serve.bedrock_model_builder import BedrockModelBuilder, _is_nova_model

MODULE = "sagemaker.serve.bedrock_model_builder"


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_container(recipe_name=None, hub_content_name=None, s3_uri=None):
    """Build a mock container with optional base_model and model_data_source."""
    container = Mock()
    if recipe_name is not None or hub_content_name is not None:
        base_model = Mock()
        base_model.recipe_name = recipe_name
        base_model.hub_content_name = hub_content_name
        container.base_model = base_model
    else:
        container.base_model = None

    if s3_uri:
        s3_data = Mock()
        s3_data.s3_uri = s3_uri
        data_source = Mock()
        data_source.s3_data_source = s3_data
        container.model_data_source = data_source
    else:
        container.model_data_source = None
    return container


def _make_model_package(container):
    """Wrap a container in a mock ModelPackage."""
    pkg = Mock()
    pkg.inference_specification.containers = [container]
    return pkg


def _builder():
    """Return a BedrockModelBuilder(model=None) — no side-effects."""
    return BedrockModelBuilder(model=None)


# ── _is_nova_model ──────────────────────────────────────────────────────────


class TestIsNovaModel:
    def test_nova_via_recipe_name(self):
        assert _is_nova_model(_make_container(recipe_name="amazon-nova-micro-v1")) is True

    def test_nova_via_hub_content_name(self):
        assert _is_nova_model(_make_container(hub_content_name="amazon-nova-lite")) is True

    def test_non_nova(self):
        assert _is_nova_model(_make_container(recipe_name="llama-3-8b", hub_content_name="llama")) is False

    def test_no_base_model(self):
        assert _is_nova_model(_make_container()) is False

    def test_none_fields(self):
        assert _is_nova_model(_make_container(recipe_name=None, hub_content_name=None)) is False

    def test_case_insensitive(self):
        assert _is_nova_model(_make_container(recipe_name="NOVA-PRO")) is True


# ── __init__ ────────────────────────────────────────────────────────────────


class TestInit:
    def test_none_model(self):
        b = _builder()
        assert b.model is None
        assert b.model_package is None
        assert b.s3_model_artifacts is None

    def test_with_model(self):
        m = Mock()
        with patch.object(BedrockModelBuilder, "_fetch_model_package", return_value=Mock()), \
             patch.object(BedrockModelBuilder, "_get_s3_artifacts", return_value="s3://b/k"):
            b = BedrockModelBuilder(model=m)
        assert b.model is m
        assert b.s3_model_artifacts == "s3://b/k"


# ── Client singletons ──────────────────────────────────────────────────────


class TestClients:
    def test_bedrock_client_cached(self):
        b = _builder()
        b.boto_session = Mock()
        b.boto_session.client.return_value = Mock()
        c1 = b._get_bedrock_client()
        c2 = b._get_bedrock_client()
        assert c1 is c2
        b.boto_session.client.assert_called_once_with("bedrock")

    def test_sagemaker_client_cached(self):
        b = _builder()
        b.boto_session = Mock()
        b.boto_session.client.return_value = Mock()
        c1 = b._get_sagemaker_client()
        c2 = b._get_sagemaker_client()
        assert c1 is c2
        b.boto_session.client.assert_called_once_with("sagemaker")

    def test_injected_bedrock_client(self):
        b = _builder()
        injected = Mock()
        b._bedrock_client = injected
        assert b._get_bedrock_client() is injected


# ── _fetch_model_package ────────────────────────────────────────────────────


# Sentinel classes used to control isinstance checks in _fetch_model_package tests.
class _SentinelA:
    pass


class _SentinelB:
    pass


class _SentinelC:
    pass


class TestFetchModelPackage:
    def test_model_package_returned_directly(self):
        """When model is a ModelPackage, return it as-is."""
        b = _builder()
        b.model = Mock()
        # ModelPackage = type(b.model) so isinstance matches; others are sentinels
        with patch(f"{MODULE}.ModelPackage", type(b.model)), \
             patch(f"{MODULE}.TrainingJob", _SentinelA), \
             patch(f"{MODULE}.ModelTrainer", _SentinelB):
            result = b._fetch_model_package()
        assert result is b.model

    def test_from_training_job(self):
        b = _builder()
        b.model = Mock()
        b.model.output_model_package_arn = "arn:pkg"
        expected = Mock()

        # We need ModelPackage to NOT match but still have a .get() method.
        # Use a class with a get classmethod.
        class _FakeModelPackage:
            @staticmethod
            def get(arn):
                return expected

        with patch(f"{MODULE}.ModelPackage", _FakeModelPackage), \
             patch(f"{MODULE}.TrainingJob", type(b.model)), \
             patch(f"{MODULE}.ModelTrainer", _SentinelA):
            result = b._fetch_model_package()
        assert result is expected

    def test_from_model_trainer(self):
        b = _builder()
        b.model = Mock()
        b.model._latest_training_job.output_model_package_arn = "arn:pkg"
        expected = Mock()

        class _FakeModelPackage:
            @staticmethod
            def get(arn):
                return expected

        with patch(f"{MODULE}.ModelPackage", _FakeModelPackage), \
             patch(f"{MODULE}.TrainingJob", _SentinelA), \
             patch(f"{MODULE}.ModelTrainer", type(b.model)):
            result = b._fetch_model_package()
        assert result is expected

    def test_unknown_type_returns_none(self):
        b = _builder()
        b.model = "unknown"
        assert b._fetch_model_package() is None


# ── _get_s3_artifacts ───────────────────────────────────────────────────────


class TestGetS3Artifacts:
    def test_none_when_no_model_package(self):
        b = _builder()
        b.model_package = None
        assert b._get_s3_artifacts() is None

    def test_non_nova_returns_s3_uri(self):
        c = _make_container(recipe_name="llama", hub_content_name="llama", s3_uri="s3://b/m.tar.gz")
        b = _builder()
        b.model_package = _make_model_package(c)
        assert b._get_s3_artifacts() == "s3://b/m.tar.gz"

    def test_non_nova_no_data_source(self):
        c = _make_container(recipe_name="llama", hub_content_name="llama")
        b = _builder()
        b.model_package = _make_model_package(c)
        assert b._get_s3_artifacts() is None

    def test_nova_training_job_delegates_to_manifest(self):
        c = _make_container(recipe_name="nova-micro")
        b = _builder()
        b.model = Mock()
        b.model_package = _make_model_package(c)
        with patch(f"{MODULE}.TrainingJob", type(b.model)), \
             patch.object(BedrockModelBuilder, "_get_checkpoint_uri_from_manifest",
                          return_value="s3://b/ckpt"):
            result = b._get_s3_artifacts()
        assert result == "s3://b/ckpt"

    def test_nova_non_training_job_falls_through(self):
        c = _make_container(recipe_name="nova-micro", s3_uri="s3://b/fallback")
        b = _builder()
        b.model = "not-a-training-job"
        b.model_package = _make_model_package(c)
        assert b._get_s3_artifacts() == "s3://b/fallback"


# ── _get_checkpoint_uri_from_manifest ───────────────────────────────────────


class TestGetCheckpointUri:
    def _make_builder(self, s3_artifacts, manifest_body=None, s3_error=None):
        mock_job = Mock()
        mock_job.model_artifacts = Mock()
        mock_job.model_artifacts.s3_model_artifacts = s3_artifacts

        mock_s3 = Mock()
        # Always set exceptions.NoSuchKey to a real exception class so
        # `except s3_client.exceptions.NoSuchKey` works in the source code.
        mock_s3.exceptions = Mock()
        mock_s3.exceptions.NoSuchKey = ClientError

        if s3_error:
            mock_s3.get_object.side_effect = s3_error
        elif manifest_body is not None:
            body = Mock()
            body.read.return_value = json.dumps(manifest_body).encode()
            mock_s3.get_object.return_value = {"Body": body}

        session = Mock()
        session.client.return_value = mock_s3

        b = _builder()
        b.model = mock_job
        b.boto_session = session
        return b, mock_s3

    def test_success(self):
        b, s3 = self._make_builder(
            "s3://bucket/path/output/model.tar.gz",
            manifest_body={"checkpoint_s3_bucket": "s3://bucket/ckpt/step_4"},
        )
        with patch(f"{MODULE}.TrainingJob", type(b.model)):
            result = b._get_checkpoint_uri_from_manifest()
        assert result == "s3://bucket/ckpt/step_4"
        s3.get_object.assert_called_once_with(
            Bucket="bucket", Key="path/output/output/manifest.json"
        )

    def test_missing_checkpoint_key(self):
        b, _ = self._make_builder(
            "s3://bucket/path/output/model.tar.gz",
            manifest_body={"other_key": "value"},
        )
        with patch(f"{MODULE}.TrainingJob", type(b.model)):
            with pytest.raises(ValueError, match="checkpoint_s3_bucket"):
                b._get_checkpoint_uri_from_manifest()

    def test_manifest_not_found(self):
        err = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        b, _ = self._make_builder("s3://bucket/path/output/model.tar.gz", s3_error=err)
        with patch(f"{MODULE}.TrainingJob", type(b.model)):
            with pytest.raises(ValueError, match="manifest.json not found"):
                b._get_checkpoint_uri_from_manifest()

    def test_not_training_job_raises(self):
        b = _builder()
        b.model = "not-a-training-job"
        with pytest.raises(ValueError, match="TrainingJob"):
            b._get_checkpoint_uri_from_manifest()

    def test_no_s3_artifacts_raises(self):
        b, _ = self._make_builder(None)
        with patch(f"{MODULE}.TrainingJob", type(b.model)):
            with pytest.raises(ValueError, match="No S3 model artifacts"):
                b._get_checkpoint_uri_from_manifest()

    def test_invalid_json_raises(self):
        mock_job = Mock()
        mock_job.model_artifacts = Mock()
        mock_job.model_artifacts.s3_model_artifacts = "s3://bucket/path/output/m.tar.gz"

        body = Mock()
        body.read.return_value = b"not-json"
        mock_s3 = Mock()
        mock_s3.get_object.return_value = {"Body": body}
        mock_s3.exceptions = Mock()
        mock_s3.exceptions.NoSuchKey = ClientError

        session = Mock()
        session.client.return_value = mock_s3

        b = _builder()
        b.model = mock_job
        b.boto_session = session

        with patch(f"{MODULE}.TrainingJob", type(b.model)):
            with pytest.raises(ValueError, match="Failed to parse manifest.json"):
                b._get_checkpoint_uri_from_manifest()


# ── _wait_for_model_active ──────────────────────────────────────────────────


class TestWaitForModelActive:
    def test_immediate_active(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}
        b._wait_for_model_active("arn:model")
        b._bedrock_client.get_custom_model.assert_called_once()

    def test_polls_then_active(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_custom_model.side_effect = [
            {"modelStatus": "Creating"},
            {"modelStatus": "Creating"},
            {"modelStatus": "Active"},
        ]
        with patch(f"{MODULE}.time.sleep"):
            b._wait_for_model_active("arn:model", poll_interval=1, max_wait=10)
        assert b._bedrock_client.get_custom_model.call_count == 3

    def test_failed_raises(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Failed"}
        with pytest.raises(RuntimeError, match="failed"):
            b._wait_for_model_active("arn:model")

    def test_timeout_raises(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Creating"}
        with patch(f"{MODULE}.time.sleep"):
            with pytest.raises(RuntimeError, match="Timed out"):
                b._wait_for_model_active("arn:model", poll_interval=1, max_wait=2)


# ── create_deployment ───────────────────────────────────────────────────────


class TestCreateDeployment:
    def test_polls_then_creates(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}
        b._bedrock_client.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "arn:dep"
        }
        result = b.create_deployment(model_arn="arn:model", deployment_name="dep")
        b._bedrock_client.get_custom_model.assert_called_once()
        b._bedrock_client.create_custom_model_deployment.assert_called_once()
        assert result["customModelDeploymentArn"] == "arn:dep"

    def test_passes_extra_kwargs(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}
        b._bedrock_client.create_custom_model_deployment.return_value = {}
        b.create_deployment(model_arn="arn:model", deployment_name="d", commitmentDuration="ONE_MONTH")
        kw = b._bedrock_client.create_custom_model_deployment.call_args[1]
        assert kw["commitmentDuration"] == "ONE_MONTH"

    def test_empty_model_arn_raises(self):
        with pytest.raises(ValueError, match="model_arn is required"):
            _builder().create_deployment(model_arn="", deployment_name="d")

    def test_none_model_arn_raises(self):
        with pytest.raises(ValueError, match="model_arn is required"):
            _builder().create_deployment(model_arn=None, deployment_name="d")


# ── deploy ──────────────────────────────────────────────────────────────────


class TestDeploy:
    def test_non_nova(self):
        c = _make_container(s3_uri="s3://b/m.tar.gz")
        b = _builder()
        b.model_package = _make_model_package(c)
        b.s3_model_artifacts = "s3://b/m.tar.gz"
        b._bedrock_client = Mock()
        b._bedrock_client.create_model_import_job.return_value = {"jobArn": "arn:job"}
        result = b.deploy(job_name="j", imported_model_name="m", role_arn="r")
        assert result == {"jobArn": "arn:job"}

    def test_nova_full_chain(self):
        c = _make_container(recipe_name="nova-micro", hub_content_name="nova")
        b = _builder()
        b.model_package = _make_model_package(c)
        b.s3_model_artifacts = "s3://b/ckpt"
        b._bedrock_client = Mock()
        b._bedrock_client.create_custom_model.return_value = {"modelArn": "arn:m"}
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}
        b._bedrock_client.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "arn:dep"
        }
        result = b.deploy(custom_model_name="nova-m", role_arn="r")
        b._bedrock_client.create_custom_model.assert_called_once()
        b._bedrock_client.get_custom_model.assert_called_once()
        b._bedrock_client.create_custom_model_deployment.assert_called_once()
        assert result["customModelDeploymentArn"] == "arn:dep"

    def test_nova_via_hub_content_name(self):
        c = _make_container(recipe_name=None, hub_content_name="amazon-nova-lite")
        b = _builder()
        b.model_package = _make_model_package(c)
        b.s3_model_artifacts = "s3://b/ckpt"
        b._bedrock_client = Mock()
        b._bedrock_client.create_custom_model.return_value = {"modelArn": "arn:m"}
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}
        b._bedrock_client.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "arn:dep"
        }
        result = b.deploy(custom_model_name="n", role_arn="r")
        assert result["customModelDeploymentArn"] == "arn:dep"

    def test_nova_default_deployment_name(self):
        c = _make_container(recipe_name="nova-micro")
        b = _builder()
        b.model_package = _make_model_package(c)
        b.s3_model_artifacts = "s3://b/k"
        b._bedrock_client = Mock()
        b._bedrock_client.create_custom_model.return_value = {"modelArn": "arn"}
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}
        b._bedrock_client.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "dep"
        }
        b.deploy(custom_model_name="my-model", role_arn="r")
        kw = b._bedrock_client.create_custom_model_deployment.call_args[1]
        assert kw["modelDeploymentName"] == "my-model-deployment"

    def test_nova_with_tags(self):
        c = _make_container(recipe_name="nova-micro")
        b = _builder()
        b.model_package = _make_model_package(c)
        b.s3_model_artifacts = "s3://b/k"
        b._bedrock_client = Mock()
        b._bedrock_client.create_custom_model.return_value = {"modelArn": "arn"}
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}
        b._bedrock_client.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "dep"
        }
        tags = [{"Key": "env", "Value": "test"}]
        b.deploy(custom_model_name="m", role_arn="r", model_tags=tags)
        kw = b._bedrock_client.create_custom_model.call_args[1]
        assert kw["modelTags"] == tags

    def test_no_model_package_raises(self):
        b = _builder()
        b.model_package = None
        with pytest.raises(ValueError, match="model_package is not set"):
            b.deploy(job_name="j", role_arn="r")

    def test_nova_missing_custom_model_name_raises(self):
        c = _make_container(recipe_name="nova-micro")
        b = _builder()
        b.model_package = _make_model_package(c)
        b.s3_model_artifacts = "s3://b/k"
        with pytest.raises(ValueError, match="custom_model_name is required"):
            b.deploy(role_arn="r")

    def test_nova_missing_role_arn_raises(self):
        c = _make_container(recipe_name="nova-micro")
        b = _builder()
        b.model_package = _make_model_package(c)
        b.s3_model_artifacts = "s3://b/k"
        with pytest.raises(ValueError, match="role_arn is required"):
            b.deploy(custom_model_name="m")

    def test_non_nova_strips_none_params(self):
        c = _make_container()
        b = _builder()
        b.model_package = _make_model_package(c)
        b.s3_model_artifacts = "s3://b/k"
        b._bedrock_client = Mock()
        b._bedrock_client.create_model_import_job.return_value = {"jobArn": "arn"}
        b.deploy(job_name="j", imported_model_name="m", role_arn="r")
        kw = b._bedrock_client.create_model_import_job.call_args[1]
        assert "importedModelKmsKeyId" not in kw
        assert "clientRequestToken" not in kw
