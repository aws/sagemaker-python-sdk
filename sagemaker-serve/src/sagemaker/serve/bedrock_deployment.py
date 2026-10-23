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
"""Bedrock foundation model listing and the deployment handle of ``BedrockModelBuilder``."""

from __future__ import absolute_import

import json
import logging
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

ON_DEMAND_INFERENCE_TYPE = "ON_DEMAND"
INFERENCE_PROFILE_INFERENCE_TYPE = "INFERENCE_PROFILE"
SYSTEM_DEFINED_PROFILE_TYPE = "SYSTEM_DEFINED"
ACTIVE_PROFILE_STATUS = "ACTIVE"
GLOBAL_PROFILE_PREFIX = "global."
MODEL_NOT_READY_ERROR_CODE = "ModelNotReadyException"
JSON_CONTENT_TYPE = "application/json"
PROBE_TEXT = "Hi"
# A readiness probe asks for one token so that it costs almost nothing.
PROBE_MAX_TOKENS = 1


@dataclass(frozen=True)
class BedrockFoundationModel:
    """One foundation model that the region can invoke.

    Attributes:
        model_id: The Bedrock model id, for example ``anthropic.claude-3-haiku-20240307-v1:0``.
        model_arn: The foundation model ARN.
        model_name: The branded model name.
        provider_name: The model provider.
        input_modalities: ``TEXT``, ``IMAGE``, or ``EMBEDDING`` values the model accepts.
        output_modalities: ``TEXT``, ``IMAGE``, or ``EMBEDDING`` values the model returns.
        inference_types: ``ON_DEMAND`` and ``PROVISIONED`` from the model summary, plus
            ``INFERENCE_PROFILE`` when a system-defined inference profile routes to the model.
        inference_profile_id: The cross-region inference profile id that routes to the model,
            or None. Pass it as ``modelId`` when ``ON_DEMAND`` is absent.
    """

    model_id: str
    model_arn: str
    model_name: str
    provider_name: str
    input_modalities: List[str]
    output_modalities: List[str]
    inference_types: List[str]
    inference_profile_id: Optional[str]


def _collect_pages(list_call: Callable[..., Dict[str, Any]], result_key: str) -> List[Dict]:
    """Follow ``nextToken`` through every page of a Bedrock list call.

    ``ListInferenceProfiles`` pages with ``nextToken``. ``ListFoundationModels`` has no
    token in botocore 1.43.118 and returns one page, which this loop also handles.

    Args:
        list_call: The client method with its filter arguments already bound.
        result_key: The response key that holds the page items.

    Returns:
        The items of every page in order.
    """
    items: List[Dict] = []
    params: Dict[str, str] = {}
    while True:
        response = list_call(**params)
        items.extend(response.get(result_key, []))
        next_token = response.get("nextToken")
        if not next_token:
            return items
        params = {"nextToken": next_token}


def _profile_ids_by_model_id(profiles: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Map each model id to the sorted active system-defined profile ids that route to it.

    Args:
        profiles: The ``inferenceProfileSummaries`` items.

    Returns:
        Model id to sorted profile ids. The model id is the last ARN path segment of
        ``models[].modelArn``, because a cross-region profile lists ARNs of other regions.
    """
    profile_ids: Dict[str, set] = {}
    for profile in profiles:
        if profile.get("status") != ACTIVE_PROFILE_STATUS:
            continue
        if profile.get("type") != SYSTEM_DEFINED_PROFILE_TYPE:
            continue
        for routed_model in profile.get("models", []):
            model_id = routed_model.get("modelArn", "").rsplit("/", 1)[-1]
            profile_ids.setdefault(model_id, set()).add(profile["inferenceProfileId"])
    return {model_id: sorted(ids) for model_id, ids in profile_ids.items()}


def _choose_profile_id(profile_ids: List[str]) -> Optional[str]:
    """Prefer the geographic profile to the global one, then the first sorted id.

    Args:
        profile_ids: The sorted profile ids that route to one model.

    Returns:
        The chosen profile id, or None when the list is empty.
    """
    if not profile_ids:
        return None
    geographic_ids = [
        profile_id for profile_id in profile_ids if not profile_id.startswith(GLOBAL_PROFILE_PREFIX)
    ]
    return (geographic_ids or profile_ids)[0]


def _to_foundation_model(
    summary: Dict[str, Any], profile_ids: List[str]
) -> Optional[BedrockFoundationModel]:
    """Build the dataclass for one model summary, or None when nothing can invoke the model.

    Args:
        summary: One ``FoundationModelSummary`` item.
        profile_ids: The system-defined profile ids that route to the model.

    Returns:
        The model when its inference types include ``ON_DEMAND`` or a profile routes to it,
        else None.
    """
    inference_types = list(summary.get("inferenceTypesSupported", []))
    profile_id = _choose_profile_id(profile_ids)
    if profile_id:
        inference_types.append(INFERENCE_PROFILE_INFERENCE_TYPE)
    if ON_DEMAND_INFERENCE_TYPE not in inference_types and not profile_id:
        return None
    return BedrockFoundationModel(
        model_id=summary["modelId"],
        model_arn=summary.get("modelArn", ""),
        model_name=summary.get("modelName", ""),
        provider_name=summary.get("providerName", ""),
        input_modalities=list(summary.get("inputModalities", [])),
        output_modalities=list(summary.get("outputModalities", [])),
        inference_types=inference_types,
        inference_profile_id=profile_id,
    )


def list_invocable_foundation_models(
    region: Optional[str] = None,
    boto_session: Optional[boto3.Session] = None,
    output_modality: str = "TEXT",
) -> List[BedrockFoundationModel]:
    """List the foundation models that ``InvokeModel`` or ``Converse`` accepts in the region.

    Calls ``ListFoundationModels`` filtered by output modality and ``ListInferenceProfiles``
    filtered to system-defined profiles. The result keeps a model when
    ``inferenceTypesSupported`` includes ``ON_DEMAND`` or an active cross-region profile
    routes to it.

    Args:
        region: The region to list. None keeps the session region.
        boto_session: The session to use. None creates ``boto3.Session()``.
        output_modality: ``TEXT``, ``IMAGE``, or ``EMBEDDING``. Defaults to ``TEXT``.

    Returns:
        The invocable models in the order that ``ListFoundationModels`` returned them.
    """
    session = boto_session or boto3.Session()
    client = session.client("bedrock", region_name=region)
    summaries = _collect_pages(
        partial(client.list_foundation_models, byOutputModality=output_modality),
        "modelSummaries",
    )
    profiles = _collect_pages(
        partial(client.list_inference_profiles, typeEquals=SYSTEM_DEFINED_PROFILE_TYPE),
        "inferenceProfileSummaries",
    )
    profile_ids_by_model = _profile_ids_by_model_id(profiles)
    models = []
    for summary in summaries:
        model = _to_foundation_model(summary, profile_ids_by_model.get(summary["modelId"], []))
        if model is not None:
            models.append(model)
    logger.info("%d of %d %s models are invocable", len(models), len(summaries), output_modality)
    return models


def _error_code(error: ClientError) -> Optional[str]:
    """Return the service error code of a ``ClientError``."""
    return error.response.get("Error", {}).get("Code")


def _is_json_media_type(media_type: str) -> bool:
    """Return True when the media type is ``application/json`` with or without parameters."""
    return media_type.split(";", 1)[0].strip().lower() == JSON_CONTENT_TYPE


class BedrockDeployment:
    """A handle on a Bedrock model that ``BedrockModelBuilder.deploy()`` made invocable.

    ``invoke`` and ``converse`` send requests to ``deployment_arn`` when it is set (a Nova
    custom model deployment) and to ``model_arn`` otherwise (an imported model). The first
    call creates the ``bedrock-runtime`` client.

    Args:
        model_arn: The custom model ARN or the imported model ARN.
        boto_session: The session that creates the ``bedrock-runtime`` client.
        region: The region override. None keeps the session region.
        deployment_arn: The custom model deployment ARN, or None for an imported model.
    """

    def __init__(
        self,
        model_arn: str,
        boto_session: boto3.Session,
        region: Optional[str] = None,
        deployment_arn: Optional[str] = None,
    ):
        """Store the identifiers and the session. The constructor sends no request."""
        self._model_arn = model_arn
        self._deployment_arn = deployment_arn
        self._boto_session = boto_session
        self._region = region
        self._runtime_client = None

    @property
    def model_arn(self) -> str:
        """The custom model ARN or the imported model ARN."""
        return self._model_arn

    @property
    def deployment_arn(self) -> Optional[str]:
        """The custom model deployment ARN, or None for an imported model."""
        return self._deployment_arn

    @property
    def _invocation_target(self) -> str:
        """The ``modelId`` that ``InvokeModel`` and ``Converse`` take."""
        return self._deployment_arn or self._model_arn

    def _runtime(self):
        """Return the ``bedrock-runtime`` client, created on first use."""
        if self._runtime_client is None:
            self._runtime_client = self._boto_session.client(
                "bedrock-runtime", region_name=self._region
            )
        return self._runtime_client

    def wait_until_ready(
        self,
        timeout: int = 1800,
        poll: int = 30,
        probe_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Block until the model answers a one-token request.

        Bedrock returns ``ModelNotReadyException`` for some minutes after an import or a
        deployment. The probe calls ``Converse`` for a custom model deployment, and
        ``InvokeModel`` with ``probe_body`` for an imported model. The method raises every
        other service error at once, so pass a ``probe_body`` that the model accepts.

        Args:
            timeout: Maximum seconds to wait. Defaults to 1800.
            poll: Seconds between probes. Defaults to 30.
            probe_body: The ``InvokeModel`` body for an imported model. Defaults to
                ``{"prompt": "Hi", "max_tokens": 1}``, the completion schema of imported
                text models. An embedding model takes ``{"inputText": "Hi"}``.

        Raises:
            TimeoutError: When the model is not ready after ``timeout`` seconds.
            botocore.exceptions.ClientError: When a probe fails for another reason.
        """
        if probe_body is None:
            probe_body = {"prompt": PROBE_TEXT, "max_tokens": PROBE_MAX_TOKENS}
        elapsed = 0
        while not self._probe_ready(probe_body):
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Bedrock model {self._invocation_target} was not ready after {timeout}s."
                )
            logger.info(
                "Bedrock model %s is not ready yet (elapsed %ds)", self._invocation_target, elapsed
            )
            time.sleep(poll)
            elapsed += poll

    def _probe_ready(self, probe_body: Dict[str, Any]) -> bool:
        """Send one probe. Return False on ``ModelNotReadyException``, raise other errors."""
        try:
            self._send_probe(probe_body)
        except ClientError as error:
            if _error_code(error) != MODEL_NOT_READY_ERROR_CODE:
                raise
            return False
        return True

    def _send_probe(self, probe_body: Dict[str, Any]) -> None:
        """Send the one-token probe that fits the deployment type."""
        if self._deployment_arn:
            self.converse(
                messages=[{"role": "user", "content": [{"text": PROBE_TEXT}]}],
                inference_config={"maxTokens": PROBE_MAX_TOKENS},
            )
            return
        self.invoke(probe_body)

    def invoke(
        self,
        body: Union[str, bytes, Dict[str, Any]],
        content_type: str = JSON_CONTENT_TYPE,
        accept: str = JSON_CONTENT_TYPE,
    ) -> Dict[str, Any]:
        """Call ``InvokeModel`` and read the response body.

        Args:
            body: The request body. ``json.dumps`` serializes a dict.
            content_type: The request media type. Defaults to ``application/json``.
            accept: The response media type. Defaults to ``application/json``.

        Returns:
            The parsed JSON body when ``accept`` is JSON. Otherwise a dict with the raw
            ``body`` bytes and the response ``contentType``.
        """
        payload = json.dumps(body) if isinstance(body, dict) else body
        response = self._runtime().invoke_model(
            modelId=self._invocation_target,
            body=payload,
            contentType=content_type,
            accept=accept,
        )
        raw_body = response["body"].read()
        if _is_json_media_type(accept):
            return json.loads(raw_body)
        return {"body": raw_body, "contentType": response.get("contentType")}

    def converse(
        self,
        messages: List[Dict[str, Any]],
        inference_config: Optional[Dict[str, Any]] = None,
        system: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Call ``Converse`` and return its response.

        Args:
            messages: The conversation, for example
                ``[{"role": "user", "content": [{"text": "Hi"}]}]``.
            inference_config: The ``inferenceConfig`` map (``maxTokens``, ``temperature``,
                ``topP``, ``stopSequences``), or None.
            system: The ``system`` content blocks, or None.

        Returns:
            The ``Converse`` response with ``output``, ``stopReason``, ``usage``, and ``metrics``.
        """
        params: Dict[str, Any] = {"modelId": self._invocation_target, "messages": messages}
        if inference_config is not None:
            params["inferenceConfig"] = inference_config
        if system is not None:
            params["system"] = system
        return self._runtime().converse(**params)
