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
"""Evaluator entity for AI Registry Hub."""
from __future__ import annotations

import io
import json
import os
import zipfile
from collections.abc import Sequence
from datetime import datetime
from enum import Enum
from itertools import islice
from typing import List, Optional

import boto3

from sagemaker.ai_registry.air_hub import AIRHub
from sagemaker.ai_registry.air_utils import _determine_new_version
from sagemaker.ai_registry.air_constants import (
    EVALUATOR_HUB_CONTENT_TYPE, EVALUATOR_HUB_CONTENT_SUBTYPE,
    HubContentStatus,
    EVALUATOR_DEFAULT_S3_PREFIX, EVALUATOR_DEFAULT_RUNTIME,
    EVALUATOR_DOCUMENT_SCHEMA_VERSION,
    EVALUATOR_DEFAULT_METHOD,
    EVALUATOR_BYOCODE,
    EVALUATOR_BYOLAMBDA,
    LAMBDA_ARN_PREFIX, TAG_KEY_METHOD, TAG_KEY_DOMAIN_ID, RESPONSE_KEY_HUB_CONTENT_VERSION,
    RESPONSE_KEY_HUB_CONTENT_ARN, RESPONSE_KEY_CREATION_TIME,
    RESPONSE_KEY_LAST_MODIFIED_TIME, RESPONSE_KEY_FUNCTION_ARN,
    RESPONSE_KEY_HUB_CONTENT_NAME, RESPONSE_KEY_HUB_CONTENT_STATUS,
    RESPONSE_KEY_HUB_CONTENT_DOCUMENT, RESPONSE_KEY_HUB_CONTENT_SEARCH_KEYWORDS,
    DOC_KEY_JSON_CONTENT,
    DOC_KEY_REFERENCE, DOC_KEY_SUB_TYPE, REWARD_FUNCTION, REWARD_PROMPT,
)
from sagemaker.ai_registry.air_hub_entity import AIRHubEntity
from sagemaker.ai_registry.air_utils import _get_default_bucket
from sagemaker.core.utils.utils import ResourceIterator
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.core.telemetry.constants import Feature
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.common_utils.finetune_utils import _get_current_domain_id
from sagemaker.train.defaults import TrainDefaults

class EvaluatorMethod(Enum):
    """Enum for Evaluator method types."""
    BYOC = "byoc"
    LAMBDA = "lambda"


class EvaluatorList(Sequence):
    """List-like wrapper for evaluators with pagination support."""
    
    def __init__(self, evaluators: List["Evaluator"], next_token: Optional[str]):
        self._evaluators = evaluators
        self.next_token = next_token
    
    def __getitem__(self, index):
        return self._evaluators[index]
    
    def __len__(self):
        return len(self._evaluators)
    
    def __repr__(self):
        return repr(self._evaluators)
    
    def __str__(self):
        return str(self._evaluators)


class Evaluator(AIRHubEntity):
    """Evaluator entity for AI Registry."""
    
    name: str
    version: str
    arn: str
    type: Optional[str]
    method: Optional[EvaluatorMethod]
    reference: Optional[str]
    status: Optional[HubContentStatus]
    created_time: Optional[datetime]
    updated_time: Optional[datetime]
    sagemaker_session: Optional[Session] = None

    def __init__(
        self,
        name: Optional[str] = None,
        version: Optional[str] = None,
        arn: Optional[str] = None,
        type: Optional[str] = None,
        method: Optional[EvaluatorMethod] = None,
        reference: Optional[str] = None,
        status: Optional[HubContentStatus] = None,
        created_time: Optional[datetime] = None,
        updated_time: Optional[datetime] = None,
        sagemaker_session: Optional[Session] = None
    ) -> None:
        """Initialize Evaluator entity.
        
        Args:
            name: Name of the evaluator
            version: Version of the evaluator
            arn: ARN of the evaluator
            type: Type of evaluator (e.g., RewardFunction, RewardPrompt)
            method: Method used by the evaluator (BYOC, Lambda, etc.)
            reference: Reference to the evaluator implementation (ARN, S3 URI, etc.)
            status: Current status of the evaluator
            created_time: Creation timestamp
            updated_time: Last update timestamp
            sagemaker_session: Optional SageMaker session.
        """
        super().__init__(name, version, arn, status, created_time, updated_time,sagemaker_session)
        self.method = method
        self.type = type
        self.reference = reference

    def __repr__(self):
        return (
            f"Evaluator(\n"
            f"  name={self.name!r},\n"
            f"  version={self.version!r},\n"
            f"  status={self.status!r},\n"
            f"  type={self.type!r},\n"
            f"  method={self.method.value if self.method else None!r},\n"
            f"  arn={self.arn!r},\n"
            f"  reference={self.reference!r},\n"
            f"  created_time={self.created!r}\n"
            f")"
        )

    def __str__(self):
        return self.__repr__()
    
    def refresh(self):
        """Load full evaluator details from API."""
        if not self.name:
            return self
        
        response = AIRHub.describe_hub_content(EVALUATOR_HUB_CONTENT_TYPE, self.name, session=self.sagemaker_session)
        doc = json.loads(response[RESPONSE_KEY_HUB_CONTENT_DOCUMENT])
        try:
            keywords = {kw.split(":")[0]: kw.split(":")[1] for kw in response.get(RESPONSE_KEY_HUB_CONTENT_SEARCH_KEYWORDS, []) if ":" in kw}
        except (IndexError, AttributeError):
            keywords = {}
        json_content = json.loads(doc.get(DOC_KEY_JSON_CONTENT, "{}"))
        
        self.name = response[RESPONSE_KEY_HUB_CONTENT_NAME]
        self.arn = response[RESPONSE_KEY_HUB_CONTENT_ARN]
        self.version = response[RESPONSE_KEY_HUB_CONTENT_VERSION]
        self.reference = json_content.get(DOC_KEY_REFERENCE, "")
        self.type = json_content.get(DOC_KEY_SUB_TYPE, "")
        self.status = response[RESPONSE_KEY_HUB_CONTENT_STATUS]
        method_str = keywords.get(TAG_KEY_METHOD)
        self.method = EvaluatorMethod(method_str) if method_str else None
        self.created = response.get(RESPONSE_KEY_CREATION_TIME)
        self.updated = response.get(RESPONSE_KEY_LAST_MODIFIED_TIME)
        
        return self

    @property
    def hub_content_type(self) -> str:
        return EVALUATOR_HUB_CONTENT_TYPE

    @classmethod
    def _get_hub_content_type_for_list(cls) -> str:
        return EVALUATOR_HUB_CONTENT_TYPE

    @classmethod
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="Evaluator.get")
    def get(cls, name: str, sagemaker_session=None) -> "Evaluator":
        """Get evaluator by name."""
        sagemaker_session = TrainDefaults.get_sagemaker_session(sagemaker_session=sagemaker_session)
        response = AIRHub.describe_hub_content(EVALUATOR_HUB_CONTENT_TYPE, name, session=sagemaker_session)
        doc = json.loads(response[RESPONSE_KEY_HUB_CONTENT_DOCUMENT])
        try:
            keywords = {kw.split(":")[0]: kw.split(":")[1] for kw in response.get(RESPONSE_KEY_HUB_CONTENT_SEARCH_KEYWORDS, []) if ":" in kw}
        except (IndexError, AttributeError):
            keywords = {}
        json_content = json.loads(doc.get(DOC_KEY_JSON_CONTENT, "{}"))
        reference = json_content.get(DOC_KEY_REFERENCE, "")
        type = json_content.get(DOC_KEY_SUB_TYPE, "")
        method_str = keywords.get(TAG_KEY_METHOD)
        return cls(
            name=response[RESPONSE_KEY_HUB_CONTENT_NAME],
            arn=response[RESPONSE_KEY_HUB_CONTENT_ARN],
            version=response[RESPONSE_KEY_HUB_CONTENT_VERSION],
            type=type,
            method=EvaluatorMethod(method_str) if method_str else None,
            reference=reference,
            status=response[RESPONSE_KEY_HUB_CONTENT_STATUS],
            created_time=response.get(RESPONSE_KEY_CREATION_TIME),
            updated_time=response.get(RESPONSE_KEY_LAST_MODIFIED_TIME),
        )

    @classmethod
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="Evaluator.create")
    def create(
        cls,
        name: str,
        type: str,
        source: Optional[str] = None,
        wait: bool = True,
        role: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> "Evaluator":
        """Create a new Evaluator entity in the AI Registry.
        
        Args:
            name: Name of the evaluator
            type: Type of evaluator (RewardFunction or RewardPrompt)
            source: Lambda ARN, S3 URI, or local file path depending on evaluator type
            wait: Whether to wait for the evaluator to be available

        Returns:
            Evaluator: Newly created Evaluator instance
            
        Raises:
            ValueError: If source is required but not provided, or if type is unsupported
        """
        # Get or create session for domain ID extraction
        if sagemaker_session is None:
            sagemaker_session = Session()

        # Extract domain ID if available (only works in Studio environments)
        domain_id = _get_current_domain_id(sagemaker_session)
        sagemaker_session = TrainDefaults.get_sagemaker_session(sagemaker_session=sagemaker_session)
        role = TrainDefaults.get_role(role=role, sagemaker_session=sagemaker_session)
        
        method = None
        reference = None
        
        if type == REWARD_PROMPT:
            reference = cls._handle_reward_prompt(name, source)
        elif type == REWARD_FUNCTION:
            method, reference = cls._handle_reward_function(name, source, role)
        else:
            raise ValueError(f"Unsupported evaluator type: {type}")

        # Create hub content document
        json_content = {"Reference": reference, "EvaluatorType": type}
        hub_content_document = json.dumps({
            "SubType": EVALUATOR_HUB_CONTENT_SUBTYPE,
            "JsonContent": json.dumps(json_content)
        })
        
        content_type = EVALUATOR_BYOCODE  # Default content type
        if source and source.startswith(LAMBDA_ARN_PREFIX):
            content_type = EVALUATOR_BYOLAMBDA

        # Prepare tags for SearchKeywords
        if method is not None:
            tags = [
                (TAG_KEY_METHOD, method.value),
                ("@" + "subtype", EVALUATOR_HUB_CONTENT_SUBTYPE.lower()),
                ("@" + DOC_KEY_SUB_TYPE.lower(), type.lower()),
                ("@" + "contenttype", content_type.lower())
            ]
        else:
            tags = []
        
        # Add domain-id to SearchKeywords if available
        if domain_id:
            tags.append((TAG_KEY_DOMAIN_ID, domain_id))

        # Determine new version
        new_version = _determine_new_version(EVALUATOR_HUB_CONTENT_TYPE, name, sagemaker_session)

        # Import hub content
        AIRHub.import_hub_content(
            hub_content_type=EVALUATOR_HUB_CONTENT_TYPE,
            hub_content_name=name,
            hub_content_version=new_version,
            document_schema_version=EVALUATOR_DOCUMENT_SCHEMA_VERSION,
            hub_content_document=hub_content_document,
            tags=tags,
            session=sagemaker_session,
        )
        
        # Get the created evaluator details
        describe_response = AIRHub.describe_hub_content(EVALUATOR_HUB_CONTENT_TYPE, name, session=sagemaker_session)

        evaluator = cls(
            name=name,
            version=describe_response[RESPONSE_KEY_HUB_CONTENT_VERSION],
            arn=describe_response[RESPONSE_KEY_HUB_CONTENT_ARN],
            type=type,
            method=method,
            status=HubContentStatus.IMPORTING,
            created_time=describe_response[RESPONSE_KEY_CREATION_TIME],
            updated_time=describe_response[RESPONSE_KEY_LAST_MODIFIED_TIME],
            reference=reference,
            sagemaker_session=sagemaker_session
        )
        
        if wait:
            evaluator.wait()
        
        return evaluator

    @classmethod
    def _handle_reward_prompt(cls, name: str, source: Optional[str]) -> str:
        """Handle creation of reward prompt evaluator.
        
        Args:
            name: Name of the evaluator
            source: S3 URI or local file path
            
        Returns:
            Reference to the prompt source
        """
        if source is None:
            raise ValueError("source must be provided for RewardPrompt")
            
        if source.startswith("s3://"):
            return source
        else:
            # Upload local file to S3
            try:
                return AIRHub.upload_to_s3(
                    _get_default_bucket(),
                    f"{EVALUATOR_DEFAULT_S3_PREFIX}/{name}", 
                    source
                )
            except Exception as e:
                raise ValueError(f"[PySDK Error] Failed to upload prompt source to S3: {str(e)}") from e

    @classmethod
    def _handle_reward_function(cls, name: str, source: Optional[str], role: Optional[str]) -> tuple[EvaluatorMethod, str]:
        """Handle creation of reward function evaluator.
        
        Args:
            name: Name of the evaluator
            source: Lambda ARN or local file path
            
        Returns:
            Tuple of (method, reference)
        """
        if source is None:
            raise ValueError("source must be provided for RewardFunction")
            
        if source.startswith(LAMBDA_ARN_PREFIX):
            # Use existing Lambda function
            return EvaluatorMethod.LAMBDA, source
        else:
            # BYOC - create Lambda function from local file
            return cls._create_lambda_function(name, source, role)

    @classmethod
    def _create_lambda_function(cls, name: str, source_file: str, role: Optional[str]) -> tuple[EvaluatorMethod, str]:
        """Create Lambda function from local Python file.
        
        Args:
            name: Name of the evaluator
            source_file: Path to local Python file
            
        Returns:
            Tuple of (EvaluatorMethod.BYOC, lambda_arn)
        """
        # Upload function file to S3 for backup
        AIRHub.upload_to_s3(
            _get_default_bucket(),
            f"{EVALUATOR_DEFAULT_S3_PREFIX}/{name}", 
            source_file
        )

        # Create ZIP file from Python code
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.write(source_file, 'lambda_function.py')
        zip_buffer.seek(0)

        # Create Lambda function
        lambda_client = boto3.client("lambda")
        function_name = f"SageMaker-evaluator-{name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        handler_name = f"{os.path.splitext(os.path.basename(source_file))[0]}.lambda_handler"

        try:
            lambda_response = lambda_client.create_function(
                FunctionName=function_name,
                Runtime=EVALUATOR_DEFAULT_RUNTIME,
                Role=role,
                Handler=handler_name,
                Code={"ZipFile": zip_buffer.read()},
            )
        except lambda_client.exceptions.ResourceConflictException:
            # Function exists, update it
            zip_buffer.seek(0)
            lambda_response = lambda_client.update_function_code(
                FunctionName=function_name,
                ZipFile=zip_buffer.read()
            )
        
        return EvaluatorMethod.BYOC, lambda_response[RESPONSE_KEY_FUNCTION_ARN]

    @classmethod
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="Evaluator.get_all")
    def get_all(cls, type: Optional[str] = None, max_results: Optional[int] = None, sagemaker_session=None):
        """List all evaluator entities in the hub.
        
        Args:
            max_results: Maximum number of results to return
            type: Filter by evaluator type (REWARD_PROMPT or REWARD_FUNCTION)
            
        Returns:
            Iterator for listed Evaluator resources
        """
        AIRHub._ensure_hub_name_initialized()

        sagemaker_session = TrainDefaults.get_sagemaker_session(sagemaker_session=sagemaker_session)
        client = sagemaker_session.sagemaker_client
        
        operation_input_args = {
            "HubName": AIRHub.hubName,
            "HubContentType": cls._get_hub_content_type_for_list(),
        }
        
        iterator = ResourceIterator(
            client=client,
            list_method="list_hub_contents",
            summaries_key="HubContentSummaries",
            summary_name="HubContentInfo",
            resource_cls=cls,
            list_method_kwargs=operation_input_args,
            custom_key_mapping={
                "hub_content_name": "name",
                "hub_content_arn": "arn",
                "hub_content_version": "version",
                "hub_content_status": "status",
                "creation_time": "created_time",
                "last_modified_time": "updated_time",
            },
        )
        
        if type:
            iterator = (e for e in iterator if e.type == type)
        
        return islice(iterator, max_results) if max_results else iterator
    
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="Evaluator.get_versions")
    def get_versions(self) -> List["Evaluator"]:
        """
        List all versions of this evaluator.
        
        Returns:
            List[Evaluator]: List of all versions of this evaluator
        """
        versions = AIRHub.list_hub_content_versions(self.hub_content_type, self.name, session=self.sagemaker_session)
        
        evaluators = []
        for v in versions:
            response = AIRHub.describe_hub_content(self.hub_content_type, self.name, v.get(RESPONSE_KEY_HUB_CONTENT_VERSION), session=self.sagemaker_session)
            doc = json.loads(response[RESPONSE_KEY_HUB_CONTENT_DOCUMENT])
            try:
                keywords = {kw.split(":")[0]: kw.split(":")[1] for kw in response.get(RESPONSE_KEY_HUB_CONTENT_SEARCH_KEYWORDS, []) if ":" in kw}
            except (IndexError, AttributeError):
                keywords = {}
            json_content = json.loads(doc.get(DOC_KEY_JSON_CONTENT, "{}"))
            reference = json_content.get(DOC_KEY_REFERENCE, "")
            type = doc.get(DOC_KEY_SUB_TYPE, "")
            method_str = keywords.get(TAG_KEY_METHOD, EVALUATOR_DEFAULT_METHOD)
            
            evaluators.append(Evaluator(
                name=response[RESPONSE_KEY_HUB_CONTENT_NAME],
                arn=response[RESPONSE_KEY_HUB_CONTENT_ARN],
                version=response[RESPONSE_KEY_HUB_CONTENT_VERSION],
                type=type,
                status=response[RESPONSE_KEY_HUB_CONTENT_STATUS],
                method=EvaluatorMethod(method_str) if method_str else None,
                reference=reference,
                created_time=response.get(RESPONSE_KEY_CREATION_TIME),
                updated_time=response.get(RESPONSE_KEY_LAST_MODIFIED_TIME)
            ))
        
        return evaluators

    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="Evaluator.create_version")
    def create_version(self, source: str) -> bool:
        """Create a new version of this evaluator.
        
        Args:
            source: Lambda ARN or local file path for the function

        Returns:
            bool: True if version created successfully, False otherwise
        """
        try:
            Evaluator.create(
                name=self.name,
                type=self.type,
                source=source,
            )
            return True
        except Exception as e:
            raise RuntimeError(f"[PySDK Error] Failed to create new version: {str(e)}")
