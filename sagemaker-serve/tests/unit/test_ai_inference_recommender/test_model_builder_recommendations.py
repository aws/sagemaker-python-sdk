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


@pytest.fixture(autouse=True)
def _ambient_execution_role():
    """Simulate running where a SageMaker execution role is resolvable.

    ModelBuilder resolves a default serving role at construction when none is
    supplied; these unit tests build with a mock session and no role, so stand
    in a resolved role the way an ambient execution role would.
    """
    with patch(
        "sagemaker.serve.model_builder.resolve_and_validate_role",
        return_value="arn:aws:iam::123456789012:role/ambient-execution-role",
    ):
        yield


@pytest.fixture
def mb_class():
    """Import ModelBuilder lazily to keep import errors isolated to this fixture."""
    from sagemaker.serve import ModelBuilder

    return ModelBuilder


def _mock_session():
    """A mock SageMaker session whose ``sagemaker_config`` is a real dict, so the
    full ``deploy()`` path (which validates that config) runs against a mock."""
    session = MagicMock()
    session.sagemaker_config = {}
    return session


def _rec_row(
    spec_name=None,
    model_package_arn="arn:aws:sm:us-west-2:1:model-package/p/1",
    instance_count=1,
):
    return SimpleNamespace(
        model_details=SimpleNamespace(
            model_package_arn=model_package_arn,
            inference_specification_name=spec_name,
        ),
        deployment_configuration=SimpleNamespace(
            instance_type="ml.g6.12xlarge",
            instance_count=instance_count,
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
            mb = mb_class.from_recommendation_job("my-rec-job", sagemaker_session=session)
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

    def test_recommendation_row_selects_exact_row_against_describe(self, mb_class):
        """Passing recommendation_row deploys that exact row's ModelPackage even
        when another row shares its spec name — verified against the real
        describe_model_package call, not just a forwarded kwarg."""
        rows = [
            _rec_row(spec_name="dup", model_package_arn="arn:.../p/0"),
            _rec_row(spec_name="dup", model_package_arn="arn:.../p/1"),
        ]
        mb = self._make_builder_with_rows(mb_class, rows)
        sm_mock = mb.sagemaker_session.sagemaker_client
        sm_mock.describe_model_package.return_value = {"ModelApprovalStatus": "Approved"}

        patches = self._patch_aws_calls()
        for p in patches:
            p.start()
        try:
            mb._deploy_recommendation(
                recommendation_index=0,  # default; must be ignored in favor of the row
                recommendation_spec_name=None,
                recommendation_row=rows[1],
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
            assert described_arn == "arn:.../p/1"
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
        ), patch("sagemaker.serve.model_builder.Endpoint.create", return_value=MagicMock()):
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
        assert model_create.call_args.kwargs["execution_role_arn"] == "arn:aws:iam::1:role/builder"


class TestDeployRecommendationSpeculativeDecoding:
    """Optimized recommendations (kernel tuning / speculative decoding) carry
    the base weights and any draft model as AdditionalModelDataSources on the
    ModelPackage. The hosting stack (including Inference Components) now resolves
    those channels itself, so deploy just passes the ModelPackage through — no
    client-side collapsing of the additional sources into a primary source."""

    def _make_builder(self, mb_class, described):
        mb = mb_class(sagemaker_session=MagicMock())
        mb.role_arn = "arn:aws:iam::1:role/builder"
        mb._recommendation_job = SimpleNamespace(
            recommendations=[_rec_row(spec_name="A", model_package_arn="arn:.../p/A")],
            ai_recommendation_job_status="Completed",
        )
        mb.sagemaker_session.sagemaker_client.describe_model_package.return_value = described
        return mb

    def _deploy(self, mb):
        with patch("sagemaker.core.resources.Model.create") as model_create, patch(
            "sagemaker.serve.model_builder.EndpointConfig.create"
        ), patch("sagemaker.serve.model_builder.Endpoint.create", return_value=MagicMock()):
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
        return model_create

    @staticmethod
    def _channel(name, uri):
        return {
            "ChannelName": name,
            "S3DataSource": {
                "S3Uri": uri,
                "S3DataType": "S3Prefix",
                "CompressionType": "None",
            },
        }

    def test_optimized_recommendation_deploys_via_model_package(self, mb_class):
        # An optimized rec exposes base_model + draft_model as additional data
        # sources; deploy no longer collapses them — it passes the ModelPackage
        # through and lets the hosting stack resolve the channels.
        #
        # The AdditionalModelDataSources below are intentionally present as a
        # regression guard: the current path reads only ModelApprovalStatus and
        # ignores them, so the two negative assertions (no model_data_source /
        # no additional_model_data_sources) only fail if the client-side
        # collapse logic is ever re-introduced.
        described = {
            "ModelApprovalStatus": "Approved",
            "InferenceSpecification": {
                "Containers": [
                    {
                        "Image": "123.dkr.ecr.us-west-2.amazonaws.com/djl-inference:latest",
                        "AdditionalModelDataSources": [
                            self._channel("base_model", "s3://base/weights/"),
                            self._channel("draft_model", "s3://draft/model/"),
                        ],
                    }
                ]
            },
        }
        mb = self._make_builder(mb_class, described)
        container = self._deploy(mb).call_args.kwargs["primary_container"]

        # Deployed straight from the ModelPackage — no client-side collapsing.
        assert container.model_package_name == "arn:.../p/A"
        # The row's inference_specification_name is threaded onto the container
        # (guards the `if inference_specification_name:` branch from regressing).
        assert container.inference_specification_name == "A"
        # No collapse: the additional sources above must NOT become a primary
        # model_data_source / additional_model_data_sources on the container.
        assert not getattr(container, "model_data_source", None)
        assert not getattr(container, "additional_model_data_sources", None)

    def test_no_additional_sources_uses_model_package(self, mb_class):
        described = {
            "ModelApprovalStatus": "Approved",
            "InferenceSpecification": {"Containers": [{"Image": "123.dkr.ecr/x:1"}]},
        }
        mb = self._make_builder(mb_class, described)
        model_create = self._deploy(mb)
        container = model_create.call_args.kwargs["primary_container"]
        # Plain (non-optimized) recommendation also deploys via the ModelPackage.
        assert container.model_package_name == "arn:.../p/A"


class TestDeployRecommendationInstanceCount:
    """The recommended instance count must survive deploy()'s default of 1.

    deploy() defaults initial_instance_count=1; the recommendation path must
    not let that silently override a recommendation that asks for multiple
    instances. These drive the public deploy() the way a real caller does
    (rather than calling the private method with initial_instance_count=None).
    """

    def _make_builder_with_row(self, mb_class, instance_count):
        session = MagicMock()
        # deploy() is telemetry-wrapped and reads sagemaker_config; give it a
        # real (empty) config so the wrapper doesn't choke on a MagicMock.
        session.sagemaker_config = {}
        mb = mb_class(sagemaker_session=session)
        mb.role_arn = "arn:aws:iam::1:role/builder"
        mb._recommendation_job = SimpleNamespace(
            recommendations=[_rec_row(spec_name="A", instance_count=instance_count)],
            ai_recommendation_job_status="Completed",
        )
        mb.sagemaker_session.sagemaker_client.describe_model_package.return_value = {
            "ModelApprovalStatus": "Approved",
            "InferenceSpecification": {"Containers": [{"Image": "123.dkr.ecr/x:1"}]},
        }
        return mb

    @staticmethod
    def _deployed_instance_count(ep_config_create):
        variants = ep_config_create.call_args.kwargs["production_variants"]
        return variants[0].initial_instance_count

    def test_multi_instance_recommendation_not_forced_to_one(self, mb_class):
        # deploy() default initial_instance_count=1 must not clobber a rec of 3.
        mb = self._make_builder_with_row(mb_class, instance_count=3)
        with patch("sagemaker.core.resources.Model.create"), patch(
            "sagemaker.serve.model_builder.EndpointConfig.create"
        ) as ep_cfg_create, patch(
            "sagemaker.serve.model_builder.Endpoint.create", return_value=MagicMock()
        ):
            mb.deploy(endpoint_name="ep", wait=False)
        assert self._deployed_instance_count(ep_cfg_create) == 3

    def test_explicit_multi_instance_override_wins(self, mb_class):
        # An explicit initial_instance_count > 1 overrides the recommendation.
        mb = self._make_builder_with_row(mb_class, instance_count=2)
        with patch("sagemaker.core.resources.Model.create"), patch(
            "sagemaker.serve.model_builder.EndpointConfig.create"
        ) as ep_cfg_create, patch(
            "sagemaker.serve.model_builder.Endpoint.create", return_value=MagicMock()
        ):
            mb.deploy(endpoint_name="ep", initial_instance_count=5, wait=False)
        assert self._deployed_instance_count(ep_cfg_create) == 5

    def test_single_instance_recommendation_deploys_one(self, mb_class):
        mb = self._make_builder_with_row(mb_class, instance_count=1)
        with patch("sagemaker.core.resources.Model.create"), patch(
            "sagemaker.serve.model_builder.EndpointConfig.create"
        ) as ep_cfg_create, patch(
            "sagemaker.serve.model_builder.Endpoint.create", return_value=MagicMock()
        ):
            mb.deploy(endpoint_name="ep", wait=False)
        assert self._deployed_instance_count(ep_cfg_create) == 1


class TestDeployRecommendationIndexBounds:
    """recommendation_index is bounds-checked with an actionable error."""

    def _make_builder(self, mb_class, n_rows):
        mb = mb_class(sagemaker_session=MagicMock())
        mb.role_arn = "arn:aws:iam::1:role/builder"
        mb._recommendation_job = SimpleNamespace(
            recommendations=[_rec_row(spec_name=str(i)) for i in range(n_rows)],
            ai_recommendation_job_status="Completed",
        )
        return mb

    def _deploy_at(self, mb, index):
        mb._deploy_recommendation(
            recommendation_index=index,
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

    def test_index_past_end_raises_valueerror(self, mb_class):
        mb = self._make_builder(mb_class, n_rows=2)
        with pytest.raises(ValueError, match="out of range"):
            self._deploy_at(mb, 5)

    def test_negative_index_raises_valueerror(self, mb_class):
        # Without a bounds check a negative index would wrap silently to a
        # different row; it must be rejected instead.
        mb = self._make_builder(mb_class, n_rows=2)
        with pytest.raises(ValueError, match="out of range"):
            self._deploy_at(mb, -1)


class TestRegisterReturnsArn:
    """register() must return the created ModelPackageArn (regression guard).

    A prior change dropped the trailing return, so callers doing
    arn = mb.register(...) silently got None. Lock the contract.
    """

    def _make_builder(self, mb_class):
        session = MagicMock()
        # register() is telemetry-wrapped and reads sagemaker_config.
        session.sagemaker_config = {}
        mb = mb_class(sagemaker_session=session)
        mb.image_uri = "123.dkr.ecr.us-west-2.amazonaws.com/img:latest"
        mb.s3_model_data_url = "s3://bucket/model/"
        mb.content_types = ["application/json"]
        mb.response_types = ["application/json"]
        mb.framework = None
        mb.framework_version = None
        return mb

    def test_register_returns_model_package_arn(self, mb_class):
        # A MagicMock session is not a PipelineSession, so register() runs the
        # non-pipeline path and must return the ARN (not fall off the end).
        mb = self._make_builder(mb_class)
        arn = "arn:aws:sagemaker:us-west-2:1:model-package/g/1"
        with patch(
            "sagemaker.serve.model_builder.create_model_package_from_containers",
            return_value={"ModelPackageArn": arn},
        ), patch("sagemaker.serve.model_builder.get_model_package_args", return_value={}), patch(
            "sagemaker.serve.model_builder.update_container_with_inference_params",
            side_effect=lambda **kw: kw.get("container_def"),
        ), patch.object(
            mb, "_prepare_container_def", return_value={"Image": mb.image_uri}
        ):
            result = mb.register(model_package_group_name="g")
        assert result == arn


class TestDeployRecommendationApproval:
    """Governance: an unapproved ModelPackage must not be silently approved."""

    def _make_builder(self, mb_class, approval_status):
        mb = mb_class(sagemaker_session=MagicMock())
        mb.role_arn = "arn:aws:iam::1:role/builder"
        mb._recommendation_job = SimpleNamespace(
            recommendations=[_rec_row(spec_name="A", model_package_arn="arn:.../p/A")],
            ai_recommendation_job_status="Completed",
        )
        mb.sagemaker_session.sagemaker_client.describe_model_package.return_value = {
            "ModelApprovalStatus": approval_status,
            "InferenceSpecification": {"Containers": [{"Image": "123.dkr.ecr/x:1"}]},
        }
        return mb

    def _deploy(self, mb, **overrides):
        kwargs = dict(
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
        kwargs.update(overrides)
        with patch("sagemaker.core.resources.Model.create"), patch(
            "sagemaker.serve.model_builder.EndpointConfig.create"
        ), patch("sagemaker.serve.model_builder.Endpoint.create", return_value=MagicMock()):
            return mb._deploy_recommendation(**kwargs)

    def test_unapproved_package_raises_by_default(self, mb_class):
        mb = self._make_builder(mb_class, "PendingManualApproval")
        with pytest.raises(ValueError, match="approval status"):
            self._deploy(mb)
        # Must NOT have silently approved.
        mb.sagemaker_session.sagemaker_client.update_model_package.assert_not_called()

    def test_already_approved_deploys_without_update(self, mb_class):
        mb = self._make_builder(mb_class, "Approved")
        self._deploy(mb)
        mb.sagemaker_session.sagemaker_client.update_model_package.assert_not_called()

    def test_auto_approve_opts_in_to_update(self, mb_class):
        mb = self._make_builder(mb_class, "PendingManualApproval")
        self._deploy(mb, auto_approve=True)
        update = mb.sagemaker_session.sagemaker_client.update_model_package
        update.assert_called_once()
        assert update.call_args.kwargs["ModelApprovalStatus"] == "Approved"


class TestDeployUseRecommendationSelector:
    """deploy(use_recommendation=...) controls the recommendation takeover."""

    def _builder_with_job(self, mb_class):
        session = MagicMock()
        session.sagemaker_config = {}
        mb = mb_class(sagemaker_session=session)
        mb.role_arn = "arn:aws:iam::1:role/builder"
        mb._recommendation_job = SimpleNamespace(
            recommendations=[_rec_row(spec_name="A")],
            ai_recommendation_job_status="Completed",
        )
        return mb

    def test_use_recommendation_false_skips_rec_path(self, mb_class):
        mb = self._builder_with_job(mb_class)
        # With the rec path disabled and nothing built, deploy() must fall
        # through to the built-model check and raise (not deploy the rec).
        with patch.object(mb, "_deploy_recommendation") as rec_deploy:
            with pytest.raises(ValueError, match="Model needs to be built"):
                mb.deploy(endpoint_name="ep", use_recommendation=False, wait=False)
            rec_deploy.assert_not_called()

    def test_use_recommendation_true_without_job_raises(self, mb_class):
        session = MagicMock()
        session.sagemaker_config = {}
        mb = mb_class(sagemaker_session=session)
        with pytest.raises(ValueError, match="no recommendation job is attached"):
            mb.deploy(endpoint_name="ep", use_recommendation=True, wait=False)

    def test_default_uses_rec_path_when_job_attached(self, mb_class):
        mb = self._builder_with_job(mb_class)
        with patch.object(mb, "_deploy_recommendation", return_value="EP") as rec_deploy:
            result = mb.deploy(endpoint_name="ep", wait=False)
            rec_deploy.assert_called_once()
            assert result == "EP"


class TestDeployRecommendationSpecNameMultiMatch:
    """Multiple spec_name matches warn but proceed with the first."""

    def test_multiple_matches_warn_and_use_first(self, mb_class, caplog):
        mb = mb_class(sagemaker_session=MagicMock())
        mb.role_arn = "arn:aws:iam::1:role/builder"
        mb._recommendation_job = SimpleNamespace(
            recommendations=[
                _rec_row(spec_name="dup", model_package_arn="arn:.../p/A"),
                _rec_row(spec_name="dup", model_package_arn="arn:.../p/B"),
            ],
            ai_recommendation_job_status="Completed",
        )
        sm = mb.sagemaker_session.sagemaker_client
        sm.describe_model_package.return_value = {
            "ModelApprovalStatus": "Approved",
            "InferenceSpecification": {"Containers": [{"Image": "123.dkr.ecr/x:1"}]},
        }
        with patch("sagemaker.core.resources.Model.create"), patch(
            "sagemaker.serve.model_builder.EndpointConfig.create"
        ), patch("sagemaker.serve.model_builder.Endpoint.create", return_value=MagicMock()):
            mb._deploy_recommendation(
                recommendation_index=0,
                recommendation_spec_name="dup",
                endpoint_name=None,
                model_name=None,
                endpoint_config_name=None,
                role=None,
                instance_type=None,
                initial_instance_count=None,
                tags=None,
                wait=False,
            )
        # First match (row A) is the one described/deployed.
        assert sm.describe_model_package.call_args.kwargs["ModelPackageName"] == "arn:.../p/A"


class TestDeployRecommendationRowObject:
    """deploy(recommendation=<row>) resolves a recommendation row to its
    positional index — the exact-row identifier — instead of making callers
    hand-copy a magic index. Forwarding the index (not the spec name) is what
    guarantees the exact row passed is the one deployed, even when spec names
    repeat."""

    def _make_builder(self, mb_class):
        mb = mb_class(sagemaker_session=_mock_session())
        mb._recommendation_job = SimpleNamespace(
            recommendations=[
                _rec_row(spec_name="A", model_package_arn="arn:.../p/A"),
                _rec_row(spec_name="B", model_package_arn="arn:.../p/B"),
            ],
            ai_recommendation_job_status="Completed",
        )
        return mb

    def _row(self, mb, index):
        # A real recommendation view row, as mb.recommendations[index] returns.
        return mb.recommendations[index]

    def test_best_row_forwards_its_raw_shape(self, mb_class):
        mb = self._make_builder(mb_class)
        with patch.object(mb_class, "_deploy_recommendation") as deploy_rec:
            mb.deploy(recommendation=mb.recommendations.best)
        # The row's underlying shape is passed through (not an index/spec name),
        # so the exact row deploys regardless of list order or refresh.
        assert deploy_rec.call_args.kwargs["recommendation_row"] is mb.recommendations.best.raw
        assert deploy_rec.call_args.kwargs["recommendation_spec_name"] is None

    def test_indexed_row_forwards_its_raw_shape(self, mb_class):
        mb = self._make_builder(mb_class)
        row = self._row(mb, 1)
        with patch.object(mb_class, "_deploy_recommendation") as deploy_rec:
            mb.deploy(recommendation=row)
        assert deploy_rec.call_args.kwargs["recommendation_row"] is row.raw
        assert deploy_rec.call_args.kwargs["recommendation_spec_name"] is None

    def test_non_row_value_raises_type_error(self, mb_class):
        """A spec-name string / raw shape / int / dict is not a row view; it
        must raise rather than silently defaulting to row 0."""
        mb = self._make_builder(mb_class)
        for bad in ("B", 1, {"recommendation_spec_name": "B"}):
            with pytest.raises(TypeError, match="recommendation must be a recommendation row"):
                mb.deploy(recommendation=bad)

    def test_duplicate_spec_deploys_the_exact_row_passed(self, mb_class):
        """Two rows sharing an inference_specification_name (the normal fan-out
        shape). Passing view[1] must carry row 1's shape — resolving by spec
        name would pick matches[0] (row 0), the wrong hardware."""
        mb = mb_class(sagemaker_session=_mock_session())
        mb._recommendation_job = SimpleNamespace(
            recommendations=[
                _rec_row(spec_name="dup", model_package_arn="arn:.../p/0"),
                _rec_row(spec_name="dup", model_package_arn="arn:.../p/1"),
            ],
            ai_recommendation_job_status="Completed",
        )
        row1 = mb.recommendations[1]
        with patch.object(mb_class, "_deploy_recommendation") as deploy_rec:
            mb.deploy(recommendation=row1)
        # Row 1's exact shape flows through; the shared spec name is not used.
        assert deploy_rec.call_args.kwargs["recommendation_row"] is row1.raw
        assert (
            deploy_rec.call_args.kwargs["recommendation_row"].model_details.model_package_arn
            == "arn:.../p/1"
        )
        assert deploy_rec.call_args.kwargs["recommendation_spec_name"] is None

    def test_recommendation_conflicts_with_index(self, mb_class):
        mb = self._make_builder(mb_class)
        with pytest.raises(ValueError, match="only one of"):
            mb.deploy(recommendation=mb.recommendations.best, recommendation_index=1)

    def test_recommendation_conflicts_with_spec_name(self, mb_class):
        mb = self._make_builder(mb_class)
        with pytest.raises(ValueError, match="only one of"):
            mb.deploy(recommendation=mb.recommendations.best, recommendation_spec_name="B")
