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
        # Patch all the AWS-touching calls inside _deploy_recommendation so we
        # can drive it without real boto3.
        return [
            patch("boto3.client"),
            patch("sagemaker.core.resources.Model.create"),
            patch(
                "sagemaker.serve.model_builder.EndpointConfig.create"
            ),
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

        patches = self._patch_aws_calls()
        for p in patches:
            p.start()
        try:
            sm_mock = MagicMock()
            sm_mock.describe_model_package.return_value = {"ModelApprovalStatus": "Approved"}
            from boto3 import client as _boto3_client_unused  # noqa: F401
            with patch("boto3.client", return_value=sm_mock):
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

        patches = self._patch_aws_calls()
        for p in patches:
            p.start()
        try:
            sm_mock = MagicMock()
            sm_mock.describe_model_package.return_value = {"ModelApprovalStatus": "Approved"}
            with patch("boto3.client", return_value=sm_mock):
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
