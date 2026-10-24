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
"""Unit tests for ``sagemaker.serve.bedrock_deployment`` and the builder deployment handle.

Before this change ``sagemaker.serve`` had no ``bedrock_deployment`` module, so the import
below raised ``ModuleNotFoundError``. ``BedrockModelBuilder`` had no ``deployment``
attribute and no ``latest_deployment()`` method. Callers kept the response dict and called
``bedrock`` and ``bedrock-runtime`` by hand to list models, to wait out
``ModelNotReadyException``, and to invoke the model.
"""

import io
import json
from unittest.mock import Mock, call, patch

import pytest
from botocore.exceptions import ClientError

from sagemaker.serve.bedrock_deployment import (
    BedrockDeployment,
    BedrockFoundationModel,
    list_invocable_foundation_models,
)
from .test_bedrock_model_builder import _builder, _make_container, _make_model_package

MODULE = "sagemaker.serve.bedrock_deployment"
MODEL_ARN = "arn:aws:bedrock:us-west-2:123456789012:imported-model/abcdef123456"
DEPLOYMENT_ARN = "arn:aws:bedrock:us-west-2:123456789012:custom-model-deployment/abcdef123456"
CUSTOM_MODEL_ARN = "arn:aws:bedrock:us-west-2:123456789012:custom-model/nova/abcdef123456"
HAIKU_ID = "anthropic.claude-3-haiku-20240307-v1:0"
NOVA_PRO_ID = "amazon.nova-pro-v1:0"
PROVISIONED_ONLY_ID = "meta.llama3-70b-instruct-v1:0"


def _foundation_model_arn(model_id, region="us-west-2"):
    return f"arn:aws:bedrock:{region}::foundation-model/{model_id}"


def _summary(model_id, inference_types):
    return {
        "modelArn": _foundation_model_arn(model_id),
        "modelId": model_id,
        "modelName": model_id.split(".")[1],
        "providerName": model_id.split(".")[0].title(),
        "inputModalities": ["TEXT"],
        "outputModalities": ["TEXT"],
        "inferenceTypesSupported": inference_types,
    }


def _profile(profile_id, model_id, status="ACTIVE", profile_type="SYSTEM_DEFINED"):
    return {
        "inferenceProfileId": profile_id,
        "inferenceProfileArn": f"arn:aws:bedrock:us-west-2:123456789012:inference-profile/{profile_id}",
        "status": status,
        "type": profile_type,
        "models": [
            {"modelArn": _foundation_model_arn(model_id, "us-east-1")},
            {"modelArn": _foundation_model_arn(model_id, "us-west-2")},
        ],
    }


def _not_ready_error(operation):
    return ClientError(
        {"Error": {"Code": "ModelNotReadyException", "Message": "Model is not ready"}},
        operation,
    )


def _invoke_response(payload):
    return {
        "body": io.BytesIO(json.dumps(payload).encode("utf-8")),
        "contentType": "application/json",
    }


def _session_with_bedrock(bedrock):
    session = Mock()
    session.client.return_value = bedrock
    return session


class TestListInvocableFoundationModels:
    def test_two_pages_join_one_profile_and_drop_provisioned_only_models(self):
        bedrock = Mock()
        bedrock.list_foundation_models.side_effect = [
            {
                "modelSummaries": [
                    _summary(HAIKU_ID, ["ON_DEMAND"]),
                    _summary(PROVISIONED_ONLY_ID, ["PROVISIONED"]),
                ],
                "nextToken": "page-2",
            },
            {"modelSummaries": [_summary(NOVA_PRO_ID, ["PROVISIONED"])]},
        ]
        bedrock.list_inference_profiles.return_value = {
            "inferenceProfileSummaries": [_profile(f"us.{NOVA_PRO_ID}", NOVA_PRO_ID)]
        }
        session = _session_with_bedrock(bedrock)

        models = list_invocable_foundation_models(region="us-west-2", boto_session=session)

        session.client.assert_called_once_with("bedrock", region_name="us-west-2")
        assert bedrock.list_foundation_models.call_args_list == [
            call(byOutputModality="TEXT"),
            call(byOutputModality="TEXT", nextToken="page-2"),
        ]
        bedrock.list_inference_profiles.assert_called_once_with(typeEquals="SYSTEM_DEFINED")
        assert [model.model_id for model in models] == [HAIKU_ID, NOVA_PRO_ID]
        assert models[0] == BedrockFoundationModel(
            model_id=HAIKU_ID,
            model_arn=_foundation_model_arn(HAIKU_ID),
            model_name="claude-3-haiku-20240307-v1:0",
            provider_name="Anthropic",
            input_modalities=["TEXT"],
            output_modalities=["TEXT"],
            inference_types=["ON_DEMAND"],
            inference_profile_id=None,
        )
        assert models[1].inference_types == ["PROVISIONED", "INFERENCE_PROFILE"]
        assert models[1].inference_profile_id == f"us.{NOVA_PRO_ID}"

    def test_output_modality_filter_is_passed_through(self):
        bedrock = Mock()
        bedrock.list_foundation_models.return_value = {"modelSummaries": []}
        bedrock.list_inference_profiles.return_value = {"inferenceProfileSummaries": []}

        models = list_invocable_foundation_models(
            boto_session=_session_with_bedrock(bedrock), output_modality="IMAGE"
        )

        assert models == []
        bedrock.list_foundation_models.assert_called_once_with(byOutputModality="IMAGE")

    def test_geographic_profile_wins_over_global_and_inactive_profiles_are_ignored(self):
        bedrock = Mock()
        bedrock.list_foundation_models.return_value = {
            "modelSummaries": [_summary(NOVA_PRO_ID, ["PROVISIONED"])]
        }
        bedrock.list_inference_profiles.side_effect = [
            {
                "inferenceProfileSummaries": [_profile(f"global.{NOVA_PRO_ID}", NOVA_PRO_ID)],
                "nextToken": "page-2",
            },
            {
                "inferenceProfileSummaries": [
                    _profile(f"us.{NOVA_PRO_ID}", NOVA_PRO_ID),
                    _profile(f"eu.{NOVA_PRO_ID}", NOVA_PRO_ID, status="INACTIVE"),
                    _profile("my-app-profile", NOVA_PRO_ID, profile_type="APPLICATION"),
                ]
            },
        ]

        models = list_invocable_foundation_models(boto_session=_session_with_bedrock(bedrock))

        assert bedrock.list_inference_profiles.call_args_list == [
            call(typeEquals="SYSTEM_DEFINED"),
            call(typeEquals="SYSTEM_DEFINED", nextToken="page-2"),
        ]
        assert [model.inference_profile_id for model in models] == [f"us.{NOVA_PRO_ID}"]

    def test_default_session_is_created_when_none_is_given(self):
        bedrock = Mock()
        bedrock.list_foundation_models.return_value = {"modelSummaries": []}
        bedrock.list_inference_profiles.return_value = {"inferenceProfileSummaries": []}
        with patch(f"{MODULE}.boto3.Session", return_value=_session_with_bedrock(bedrock)):
            assert list_invocable_foundation_models(region="eu-west-1") == []
        bedrock.list_foundation_models.assert_called_once()


class TestBedrockDeploymentWaitUntilReady:
    def test_imported_model_retries_one_not_ready_then_returns(self):
        runtime = Mock()
        runtime.invoke_model.side_effect = [
            _not_ready_error("InvokeModel"),
            _invoke_response({"generation": "Hi"}),
        ]
        deployment = BedrockDeployment(MODEL_ARN, _session_with_bedrock(runtime))

        with patch(f"{MODULE}.time.sleep") as sleep:
            deployment.wait_until_ready(timeout=600, poll=30)

        assert runtime.invoke_model.call_count == 2
        sleep.assert_called_once_with(30)
        probe_kwargs = runtime.invoke_model.call_args.kwargs
        assert probe_kwargs["modelId"] == MODEL_ARN
        assert json.loads(probe_kwargs["body"]) == {"prompt": "Hi", "max_tokens": 1}
        runtime.converse.assert_not_called()

    def test_custom_model_deployment_probes_converse_with_the_deployment_arn(self):
        runtime = Mock()
        runtime.converse.side_effect = [_not_ready_error("Converse"), {"output": {}}]
        deployment = BedrockDeployment(
            CUSTOM_MODEL_ARN, _session_with_bedrock(runtime), deployment_arn=DEPLOYMENT_ARN
        )

        with patch(f"{MODULE}.time.sleep"):
            deployment.wait_until_ready(timeout=600, poll=30)

        assert runtime.converse.call_count == 2
        assert runtime.converse.call_args.kwargs["modelId"] == DEPLOYMENT_ARN
        assert runtime.converse.call_args.kwargs["inferenceConfig"] == {"maxTokens": 1}
        runtime.invoke_model.assert_not_called()

    def test_timeout_raises_with_the_arn(self):
        runtime = Mock()
        runtime.invoke_model.side_effect = _not_ready_error("InvokeModel")
        deployment = BedrockDeployment(MODEL_ARN, _session_with_bedrock(runtime))

        with patch(f"{MODULE}.time.sleep") as sleep:
            with pytest.raises(TimeoutError, match=f"{MODEL_ARN} was not ready after 60s"):
                deployment.wait_until_ready(timeout=60, poll=30)

        assert runtime.invoke_model.call_count == 3
        assert sleep.call_count == 2

    def test_custom_probe_body_replaces_the_completion_default(self):
        runtime = Mock()
        runtime.invoke_model.return_value = _invoke_response({"embedding": [0.1]})
        deployment = BedrockDeployment(MODEL_ARN, _session_with_bedrock(runtime))

        deployment.wait_until_ready(timeout=60, poll=30, probe_body={"inputText": "Hi"})

        runtime.invoke_model.assert_called_once()
        assert json.loads(runtime.invoke_model.call_args.kwargs["body"]) == {"inputText": "Hi"}

    def test_other_client_errors_are_raised_at_once(self):
        runtime = Mock()
        runtime.invoke_model.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "bad body"}}, "InvokeModel"
        )
        deployment = BedrockDeployment(MODEL_ARN, _session_with_bedrock(runtime))

        with patch(f"{MODULE}.time.sleep") as sleep:
            with pytest.raises(ClientError, match="ValidationException"):
                deployment.wait_until_ready(timeout=600, poll=30)

        sleep.assert_not_called()


class TestBedrockDeploymentInvoke:
    def test_dict_body_round_trips_through_json(self):
        runtime = Mock()
        runtime.invoke_model.return_value = _invoke_response({"outputs": [{"text": "Paris"}]})
        session = _session_with_bedrock(runtime)
        deployment = BedrockDeployment(MODEL_ARN, session, region="us-west-2")

        result = deployment.invoke({"prompt": "Capital of France?", "max_tokens": 16})

        session.client.assert_called_once_with("bedrock-runtime", region_name="us-west-2")
        runtime.invoke_model.assert_called_once_with(
            modelId=MODEL_ARN,
            body=json.dumps({"prompt": "Capital of France?", "max_tokens": 16}),
            contentType="application/json",
            accept="application/json",
        )
        assert result == {"outputs": [{"text": "Paris"}]}

    def test_string_body_and_non_json_accept_return_raw_bytes(self):
        runtime = Mock()
        runtime.invoke_model.return_value = {
            "body": io.BytesIO(b"plain text"),
            "contentType": "text/plain",
        }
        deployment = BedrockDeployment(
            CUSTOM_MODEL_ARN, _session_with_bedrock(runtime), deployment_arn=DEPLOYMENT_ARN
        )

        result = deployment.invoke('{"prompt": "x"}', accept="text/plain")

        assert runtime.invoke_model.call_args.kwargs["body"] == '{"prompt": "x"}'
        assert runtime.invoke_model.call_args.kwargs["modelId"] == DEPLOYMENT_ARN
        assert result == {"body": b"plain text", "contentType": "text/plain"}

    def test_runtime_client_is_created_once(self):
        runtime = Mock()
        runtime.invoke_model.side_effect = [_invoke_response({}), _invoke_response({})]
        session = _session_with_bedrock(runtime)
        deployment = BedrockDeployment(MODEL_ARN, session)

        deployment.invoke({"prompt": "a"})
        deployment.invoke({"prompt": "b"})

        session.client.assert_called_once()


class TestBedrockDeploymentConverse:
    def test_inference_config_and_system_pass_through(self):
        runtime = Mock()
        runtime.converse.return_value = {"output": {"message": {"content": [{"text": "Hi"}]}}}
        deployment = BedrockDeployment(
            CUSTOM_MODEL_ARN, _session_with_bedrock(runtime), deployment_arn=DEPLOYMENT_ARN
        )
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]

        result = deployment.converse(
            messages,
            inference_config={"maxTokens": 100, "temperature": 0.7},
            system=[{"text": "Answer briefly."}],
        )

        runtime.converse.assert_called_once_with(
            modelId=DEPLOYMENT_ARN,
            messages=messages,
            inferenceConfig={"maxTokens": 100, "temperature": 0.7},
            system=[{"text": "Answer briefly."}],
        )
        assert result["output"]["message"]["content"][0]["text"] == "Hi"

    def test_optional_maps_are_omitted_when_none(self):
        runtime = Mock()
        runtime.converse.return_value = {}
        deployment = BedrockDeployment(MODEL_ARN, _session_with_bedrock(runtime))
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]

        deployment.converse(messages)

        runtime.converse.assert_called_once_with(modelId=MODEL_ARN, messages=messages)


class TestBuilderDeploymentHandle:
    def test_latest_deployment_raises_before_deploy(self):
        builder = _builder()
        assert builder.deployment is None
        with pytest.raises(ValueError, match="Call deploy\\(\\) first"):
            builder.latest_deployment()

    def test_oss_deploy_sets_the_imported_model_handle(self):
        builder = _builder()
        builder.model_package = _make_model_package(_make_container(s3_uri="s3://b/model/"))
        builder.s3_model_artifacts = "s3://b/model/"
        builder.boto_session = Mock()
        builder._bedrock_client = Mock()
        builder._bedrock_client.create_model_import_job.return_value = {"jobArn": "arn:job"}
        builder._bedrock_client.get_model_import_job.return_value = {
            "status": "Completed",
            "importedModelName": "my-imported-model",
            "importedModelArn": MODEL_ARN,
        }
        runtime = builder.boto_session.client.return_value
        runtime.invoke_model.return_value = _invoke_response({"generation": "Paris"})

        with patch("sagemaker.serve.bedrock_model_builder.time.sleep"):
            result = builder.deploy(job_name="j", imported_model_name="m", role_arn="r")

        assert result["importedModelArn"] == MODEL_ARN
        assert builder.deployment.model_arn == MODEL_ARN
        assert builder.deployment.deployment_arn is None
        assert builder.latest_deployment() is builder.deployment
        assert builder.deployment.invoke({"prompt": "Capital of France?"}) == {
            "generation": "Paris"
        }
        builder.boto_session.client.assert_any_call("bedrock-runtime", region_name=None)
        assert runtime.invoke_model.call_args.kwargs["modelId"] == MODEL_ARN

    def test_nova_deploy_sets_the_custom_model_deployment_handle(self):
        builder = _builder()
        builder.model_package = _make_model_package(_make_container(recipe_name="nova-micro"))
        builder.s3_model_artifacts = "s3://b/ckpt"
        builder.boto_session = Mock()
        builder._bedrock_client = Mock()
        builder._bedrock_client.create_custom_model.return_value = {"modelArn": CUSTOM_MODEL_ARN}
        builder._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}
        builder._bedrock_client.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": DEPLOYMENT_ARN
        }
        builder._bedrock_client.get_custom_model_deployment.return_value = {"status": "Active"}

        result = builder.deploy(custom_model_name="nova-m", role_arn="r")

        assert result["customModelDeploymentArn"] == DEPLOYMENT_ARN
        assert builder.latest_deployment().model_arn == CUSTOM_MODEL_ARN
        assert builder.latest_deployment().deployment_arn == DEPLOYMENT_ARN

    def test_reused_nova_model_and_deployment_set_the_handle(self):
        builder = _builder()
        builder.model_package = _make_model_package(_make_container(recipe_name="nova-micro"))
        builder.s3_model_artifacts = "s3://b/ckpt"
        builder.boto_session = Mock()
        builder._bedrock_client = Mock()

        with (
            patch(
                "sagemaker.serve.bedrock_model_builder.find_existing_bedrock_model",
                return_value=CUSTOM_MODEL_ARN,
            ),
            patch(
                "sagemaker.serve.bedrock_model_builder.find_active_bedrock_deployment_for_model",
                return_value=DEPLOYMENT_ARN,
            ),
        ):
            builder.deploy(custom_model_name="nova-m", role_arn="r", reuse_resources=True)

        builder._bedrock_client.create_custom_model.assert_not_called()
        assert builder.latest_deployment().model_arn == CUSTOM_MODEL_ARN
        assert builder.latest_deployment().deployment_arn == DEPLOYMENT_ARN

    def test_missing_model_arn_resets_a_stale_handle(self):
        builder = _builder()
        builder.model_package = _make_model_package(_make_container(s3_uri="s3://b/model/"))
        builder.s3_model_artifacts = "s3://b/model/"
        builder.boto_session = Mock()
        builder.deployment = BedrockDeployment(MODEL_ARN, builder.boto_session)
        builder._bedrock_client = Mock()
        builder._bedrock_client.create_model_import_job.return_value = {"jobArn": "arn:job"}
        builder._bedrock_client.get_model_import_job.return_value = {
            "status": "Completed",
            "importedModelName": "my-imported-model",
        }

        with patch("sagemaker.serve.bedrock_model_builder.time.sleep"):
            builder.deploy(job_name="j", imported_model_name="m", role_arn="r")

        assert builder.deployment is None
        with pytest.raises(ValueError):
            builder.latest_deployment()
