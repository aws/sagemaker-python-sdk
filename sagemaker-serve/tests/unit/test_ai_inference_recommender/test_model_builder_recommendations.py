# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
"""Unit tests for ModelBuilder.from_recommendation_job
and the new recommendation_job / recommendation_spec_name kwargs on deploy()."""
from __future__ import absolute_import

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mb_class():
    """Import ModelBuilder lazily to keep import errors isolated to this fixture."""
    from sagemaker.serve import ModelBuilder

    return ModelBuilder


def _rec_row(spec_name=None, model_package_arn="arn:aws:sm:us-west-2:1:model-package/p/1"):
    return SimpleNamespace(
        model_details=SimpleNamespace(
            model_package_arn=model_package_arn,
            inference_specification_name=spec_name,
        ),
        deployment_configuration=SimpleNamespace(
            instance_type="ml.g6.12xlarge",
            instance_count=1,
            copy_count_per_instance=1,
        ),
        expected_performance=[],
    )


class TestFromRecommendationJob:
    def test_accepts_string_name_and_calls_get(self, mb_class):
        session = MagicMock()
        with patch("sagemaker.core.resources.AIRecommendationJob") as RecJob:
            fake = MagicMock()
            RecJob.get.return_value = fake
            mb = mb_class.from_recommendation_job(
                "my-rec-job", sagemaker_session=session
            )
            RecJob.get.assert_called_once_with(
                ai_recommendation_job_name="my-rec-job", session=session
            )
            assert mb._recommendation_job is fake

    def test_accepts_resource_object_directly(self, mb_class):
        fake = MagicMock()
        session = MagicMock()
        with patch("sagemaker.core.resources.AIRecommendationJob") as RecJob:
            mb = mb_class.from_recommendation_job(fake, sagemaker_session=session)
            RecJob.get.assert_not_called()
            assert mb._recommendation_job is fake


class TestDeployRecommendationRowSelection:
    """_deploy_recommendation row selection: recommendation_spec_name beats index."""

    def _make_builder_with_rows(self, mb_class, rows):
        mb = mb_class(sagemaker_session=MagicMock())
        mb._recommendation_job = SimpleNamespace(
            recommendations=rows,
            ai_recommendation_job_status="Completed",
        )
        return mb

    def _patch_aws_calls(self):
        # Patch resource creation so the test runs without real AWS.
        return [
            patch("sagemaker.core.resources.Model.create"),
            patch("sagemaker.serve.model_builder.EndpointConfig.create"),
            patch(
                "sagemaker.serve.model_builder.Endpoint.create",
                return_value=MagicMock(),
            ),
        ]

    def test_recommendation_spec_name_picks_matching_row(self, mb_class):
        rows = [
            _rec_row(spec_name="A", model_package_arn="arn:.../p/A"),
            _rec_row(spec_name="B", model_package_arn="arn:.../p/B"),
        ]
        mb = self._make_builder_with_rows(mb_class, rows)
        sm_mock = mb.sagemaker_session.sagemaker_client
        sm_mock.describe_model_package.return_value = {"ModelApprovalStatus": "Approved"}

        patches = self._patch_aws_calls()
        for p in patches:
            p.start()
        try:
            mb._deploy_recommendation(
                recommendation_index=0,
                recommendation_spec_name="B",
                endpoint_name=None,
                model_name=None,
                endpoint_config_name=None,
                role="arn:role",
                instance_type=None,
                initial_instance_count=None,
                tags=None,
                wait=False,
            )
            # describe_model_package should have been called with row B's ARN.
            described_arn = sm_mock.describe_model_package.call_args.kwargs["ModelPackageName"]
            assert described_arn == "arn:.../p/B"
        finally:
            for p in patches:
                p.stop()

    def test_recommendation_spec_name_with_no_match_raises(self, mb_class):
        rows = [
            _rec_row(spec_name="A"),
            _rec_row(spec_name="B"),
        ]
        mb = self._make_builder_with_rows(mb_class, rows)
        with pytest.raises(ValueError, match="recommendation_spec_name='C'"):
            mb._deploy_recommendation(
                recommendation_index=0,
                recommendation_spec_name="C",
                endpoint_name=None,
                model_name=None,
                endpoint_config_name=None,
                role="arn:role",
                instance_type=None,
                initial_instance_count=None,
                tags=None,
                wait=False,
            )

    def test_index_used_when_no_recommendation_spec_name_given(self, mb_class):
        rows = [
            _rec_row(spec_name="A", model_package_arn="arn:.../p/A"),
            _rec_row(spec_name="B", model_package_arn="arn:.../p/B"),
        ]
        mb = self._make_builder_with_rows(mb_class, rows)
        sm_mock = mb.sagemaker_session.sagemaker_client
        sm_mock.describe_model_package.return_value = {"ModelApprovalStatus": "Approved"}

        patches = self._patch_aws_calls()
        for p in patches:
            p.start()
        try:
            mb._deploy_recommendation(
                recommendation_index=1,
                recommendation_spec_name=None,
                endpoint_name=None,
                model_name=None,
                endpoint_config_name=None,
                role="arn:role",
                instance_type=None,
                initial_instance_count=None,
                tags=None,
                wait=False,
            )
            described_arn = sm_mock.describe_model_package.call_args.kwargs["ModelPackageName"]
            assert described_arn == "arn:.../p/B"
        finally:
            for p in patches:
                p.stop()


class TestInferTokenizer:
    """_infer_tokenizer resolves the workload tokenizer when omitted."""

    def test_recovers_hf_id_behind_jumpstart_redirect(self, mb_class):
        mb = mb_class(sagemaker_session=MagicMock())
        mb.model = "huggingface-reasoning-qwen3-06b"
        mb._jumpstart_mapping = {
            "Qwen/Qwen3-0.6B": {"jumpstart-model-id": "huggingface-reasoning-qwen3-06b"}
        }
        assert mb._infer_tokenizer() == "Qwen/Qwen3-0.6B"

    def test_uses_bare_hf_id_directly(self, mb_class):
        mb = mb_class(sagemaker_session=MagicMock())
        mb.model = "Qwen/Qwen3-0.6B"
        assert mb._infer_tokenizer() == "Qwen/Qwen3-0.6B"

    def test_returns_none_for_jumpstart_only_id(self, mb_class):
        mb = mb_class(sagemaker_session=MagicMock())
        mb.model = "huggingface-reasoning-qwen3-06b"
        assert mb._infer_tokenizer() is None

    def test_returns_none_when_model_unset(self, mb_class):
        mb = mb_class(sagemaker_session=MagicMock())
        mb.model = None
        assert mb._infer_tokenizer() is None


class TestDeployRecommendationRoleFallback:
    """_deploy_recommendation falls back to the builder's role_arn."""

    def _make_builder(self, mb_class):
        mb = mb_class(sagemaker_session=MagicMock())
        mb.role_arn = "arn:aws:iam::1:role/builder"
        mb._recommendation_job = SimpleNamespace(
            recommendations=[_rec_row(spec_name="A", model_package_arn="arn:.../p/A")],
            ai_recommendation_job_status="Completed",
        )
        return mb

    def test_builder_role_used_when_role_omitted(self, mb_class):
        mb = self._make_builder(mb_class)
        mb.sagemaker_session.sagemaker_client.describe_model_package.return_value = {
            "ModelApprovalStatus": "Approved"
        }
        with patch("sagemaker.core.resources.Model.create") as model_create, patch(
            "sagemaker.serve.model_builder.EndpointConfig.create"
        ), patch(
            "sagemaker.serve.model_builder.Endpoint.create", return_value=MagicMock()
        ):
            mb._deploy_recommendation(
                recommendation_index=0,
                recommendation_spec_name=None,
                endpoint_name=None,
                model_name=None,
                endpoint_config_name=None,
                role=None,
                instance_type=None,
                initial_instance_count=None,
                tags=None,
                wait=False,
            )
        assert (
            model_create.call_args.kwargs["execution_role_arn"]
            == "arn:aws:iam::1:role/builder"
        )
