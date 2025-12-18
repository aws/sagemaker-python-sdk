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

"""Dataset entity for AI Registry Hub."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from itertools import islice
from typing import List, Optional, Tuple, Union
from urllib.parse import urlparse

import pandas as pd

from sagemaker.ai_registry.dataset_format_detector import DatasetFormatDetector
from sagemaker.ai_registry.air_hub import AIRHub
from sagemaker.ai_registry.air_utils import _determine_new_version, _get_default_bucket
from sagemaker.ai_registry.air_constants import (
    HubContentStatus, DATASET_HUB_CONTENT_TYPE,
    DATASET_DEFAULT_TYPE,
    DATASET_DEFAULT_CONVERSATION_ID,
    DATASET_DEFAULT_CHECKPOINT_ID, DATASET_DOCUMENT_SCHEMA_VERSION,
    DATASET_DEFAULT_METHOD, DATASET_MAX_FILE_SIZE_BYTES, DATASET_SUPPORTED_EXTENSIONS,
    TAG_KEY_METHOD, TAG_KEY_CUSTOMIZATION_TECHNIQUE, TAG_KEY_DOMAIN_ID,
    RESPONSE_KEY_HUB_CONTENT_NAME, RESPONSE_KEY_HUB_CONTENT_ARN,
    RESPONSE_KEY_HUB_CONTENT_VERSION, RESPONSE_KEY_HUB_CONTENT_STATUS,
    RESPONSE_KEY_HUB_CONTENT_DOCUMENT, RESPONSE_KEY_HUB_CONTENT_DESCRIPTION,
    RESPONSE_KEY_HUB_CONTENT_SEARCH_KEYWORDS, RESPONSE_KEY_CREATION_TIME,
    RESPONSE_KEY_LAST_MODIFIED_TIME,
    DOC_KEY_DATASET_S3_BUCKET, DOC_KEY_DATASET_S3_PREFIX
)
from sagemaker.ai_registry.air_hub_entity import AIRHubEntity
from sagemaker.ai_registry.dataset_utils import CustomizationTechnique, DataSetMethod, DataSetHubContentDocument, \
    DataSetList, _get_default_s3_prefix
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.common_utils.finetune_utils import _get_current_domain_id
from sagemaker.ai_registry.dataset_validation import validate_dataset
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.core.telemetry.constants import Feature
from sagemaker.core.utils.utils import (
    ResourceIterator,
)
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.defaults import TrainDefaults


class DataSet(AIRHubEntity):
    """Dataset entity for AI Registry."""
    
    name: str
    arn: str
    version: str
    source: Optional[str]
    status: HubContentStatus
    description: Optional[str]
    customization_technique: Optional[CustomizationTechnique]
    method: Optional[DataSetMethod]
    created_time: Optional[datetime]
    updated_time: Optional[datetime]
    sagemaker_session: Optional[Session] = None,

    def __init__(
        self,
        name: str,
        arn: str,
        version: str,
        status: HubContentStatus,
        source: Optional[str] = None,
        description: Optional[str] = None,
        customization_technique: Optional[CustomizationTechnique] = None,
        method: Optional[DataSetMethod] = None,
        created_time: Optional[datetime] = None,
        updated_time: Optional[datetime] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> None:
        """Initialize DataSet entity.
        
        Args:
            name: Name of the dataset
            arn: ARN of the dataset
            version: Version of the dataset
            source: S3 location of the dataset
            status: Current status of the dataset
            description: Description of the dataset
            customization_technique: Customization technique used
            method: Method used to create the dataset
            created_time: Creation timestamp
            updated_time: Last update timestamp
            sagemaker_session: Optional SageMaker session.
        """
        super().__init__(name, version, arn, status, created_time, updated_time, description, sagemaker_session)
        self.source = source
        self.customization_technique = customization_technique
        self.method = method
    
    def refresh(self):
        """Load full dataset details from API."""
        if not self.name:
            return self

        response = AIRHub.describe_hub_content(DATASET_HUB_CONTENT_TYPE, self.name, session=self.sagemaker_session)
        doc = json.loads(response[RESPONSE_KEY_HUB_CONTENT_DOCUMENT])
        try:
            keywords = {kw.split(":")[0]: kw.split(":")[1] for kw in response.get(RESPONSE_KEY_HUB_CONTENT_SEARCH_KEYWORDS, []) if ":" in kw}
        except (IndexError, AttributeError):
            keywords = {}
        
        self.name = response[RESPONSE_KEY_HUB_CONTENT_NAME]
        self.arn = response[RESPONSE_KEY_HUB_CONTENT_ARN]
        self.version = response[RESPONSE_KEY_HUB_CONTENT_VERSION]
        self.source = f"s3://{doc.get(DOC_KEY_DATASET_S3_BUCKET, '')}/{doc.get(DOC_KEY_DATASET_S3_PREFIX, '')}"
        self.status = response[RESPONSE_KEY_HUB_CONTENT_STATUS]
        self.description = response.get(RESPONSE_KEY_HUB_CONTENT_DESCRIPTION, "")
        self.customization_technique = CustomizationTechnique(keywords.get(TAG_KEY_CUSTOMIZATION_TECHNIQUE)) if keywords.get(TAG_KEY_CUSTOMIZATION_TECHNIQUE) else None
        self.method = DataSetMethod(keywords.get(TAG_KEY_METHOD, DATASET_DEFAULT_METHOD))
        self.created = response.get(RESPONSE_KEY_CREATION_TIME)
        self.updated = response.get(RESPONSE_KEY_LAST_MODIFIED_TIME)

        return self

    def __repr__(self):
        return (
            f"DataSet(\n"
            f"  name={self.name!r},\n"
            f"  version={self.version!r},\n"
            f"  status={self.status!r},\n"
            f"  method={self.method.value if self.method else None!r},\n"
            f"  technique={self.customization_technique.value if self.customization_technique else None!r},\n"
            f"  source={self.source!r},\n"
            f"  created_time={self.created!r},\n"
            f"  updated_time={self.updated!r},\n"
            f"  arn={self.arn!r}\n"
            f")"
        )

    def __str__(self):
        return self.__repr__()

    @property
    def hub_content_type(self) -> str:
        return DATASET_HUB_CONTENT_TYPE

    @classmethod
    def _get_hub_content_type_for_list(cls) -> str:
        return DATASET_HUB_CONTENT_TYPE

    @classmethod
    def _validate_dataset_file(cls, file_path: str) -> None:
        """Validate dataset file extension and size.
        
        Args:
            file_path: Path to the dataset file (local or S3 path component)
            
        Raises:
            ValueError: If file extension is not supported or file size exceeds limit
        """
        # Validate file extension
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension not in DATASET_SUPPORTED_EXTENSIONS:
            supported_extensions = ', '.join(DATASET_SUPPORTED_EXTENSIONS)
            raise ValueError(f"Unsupported file extension: {file_extension}. Supported extensions: {supported_extensions}")
        
        # Validate file size for local files
        if not file_path.startswith("s3://") and os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            if file_size > DATASET_MAX_FILE_SIZE_BYTES:
                file_size_mb = file_size / (1024 * 1024)
                max_size_mb = DATASET_MAX_FILE_SIZE_BYTES / (1024 * 1024)
                raise ValueError(f"File size {file_size_mb:.2f} MB exceeds maximum allowed size of {max_size_mb:.0f} MB")

    @classmethod
    def _validate_dataset_format(cls, file_path: str) -> None:
        """Validate dataset format using DatasetFormatDetector.

        Args:
            file_path: Path to the dataset file (local path)

        Raises:
            ValueError: If dataset format cannot be detected
        """
        detector = DatasetFormatDetector()
        format_name = detector.validate_dataset(file_path)
        if format_name is False:
            raise ValueError(f"Unable to detect format for {file_path}. Please provide a valid dataset file.")

    @classmethod
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="DataSet.get")
    def get(cls, name: str, sagemaker_session=None) -> "DataSet":
        """Get dataset by name."""
        sagemaker_session = TrainDefaults.get_sagemaker_session(sagemaker_session=sagemaker_session)
        response = AIRHub.describe_hub_content(hub_content_type=DATASET_HUB_CONTENT_TYPE, hub_content_name=name, session=sagemaker_session)
        doc = json.loads(response[RESPONSE_KEY_HUB_CONTENT_DOCUMENT])
        try:
            keywords = {kw.split(":")[0]: kw.split(":")[1] for kw in response.get(RESPONSE_KEY_HUB_CONTENT_SEARCH_KEYWORDS, []) if ":" in kw}
        except (IndexError, AttributeError):
            keywords = {}
        return cls(
            name=response[RESPONSE_KEY_HUB_CONTENT_NAME],
            arn=response[RESPONSE_KEY_HUB_CONTENT_ARN],
            version=response[RESPONSE_KEY_HUB_CONTENT_VERSION],
            source=f"s3://{doc.get(DOC_KEY_DATASET_S3_BUCKET, '')}/{doc.get(DOC_KEY_DATASET_S3_PREFIX, '')}",
            status=response[RESPONSE_KEY_HUB_CONTENT_STATUS],
            description=response.get(RESPONSE_KEY_HUB_CONTENT_DESCRIPTION, ""),
            customization_technique=CustomizationTechnique(keywords.get(TAG_KEY_CUSTOMIZATION_TECHNIQUE)) if keywords.get(TAG_KEY_CUSTOMIZATION_TECHNIQUE) else None,
            method=DataSetMethod(keywords.get(TAG_KEY_METHOD, DATASET_DEFAULT_METHOD)),
            created_time=response.get(RESPONSE_KEY_CREATION_TIME),
            updated_time=response.get(RESPONSE_KEY_LAST_MODIFIED_TIME),
        )

    @classmethod
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="DataSet.create")
    def create(
        cls,
        name: str,
        source: str,
        customization_technique: Optional[CustomizationTechnique] = None,
        wait: bool = True,
        description: str = "",
        tags: Optional[List[Tuple[str, str]]] = None,
        role: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> "DataSet":
        """Create a new DataSet Hub AIR entity.
        
        Creates a new version if entity already exists. This is the primary entry point
        for users. Uploads to S3 internally if local file input is provided.
        
        Args:
            name: Name of the dataset
            source: S3 URI or local file path for the dataset
            customization_technique: Customization technique to use
            wait: Whether to wait for the dataset to be available
            description: Description of the dataset
            tags: Optional list of (key, value) tag tuples
            role: Optional IAM role ARN. If not provided, uses default execution role.
            sagemaker_session: Optional SageMaker session. If not provided, uses default session.
            
        Returns:
            DataSet: The created dataset instance
            
        Raises:
            ValueError: If validation fails or required parameters are missing
        """
        # Get or create session for domain ID extraction
        if sagemaker_session is None:
            sagemaker_session = Session()
        
        # Extract domain ID if available (only works in Studio environments)
        domain_id = _get_current_domain_id(sagemaker_session)
        
        # Validate dataset file
        cls._validate_dataset_file(source)
        sagemaker_session = TrainDefaults.get_sagemaker_session(sagemaker_session=sagemaker_session)
        role = TrainDefaults.get_role(role=role, sagemaker_session=sagemaker_session)
        
        # Parse S3 URL to extract bucket and prefix
        if source.startswith("s3://"):
            parsed = urlparse(source)
            bucket_name = parsed.netloc
            s3_key = parsed.path.lstrip("/")
            s3_prefix = s3_key  # Use full path including filename
            method = DataSetMethod.GENERATED
            
            # Download and validate format
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(s3_key)[1]
            ) as tmp_file:
                local_path = tmp_file.name

            try:
                AIRHub.download_from_s3(source, local_path)
                cls._validate_dataset_format(local_path)
            finally:
                if os.path.exists(local_path):
                    os.remove(local_path)
        else:
            # Local file - upload to S3
            bucket_name = _get_default_bucket()
            s3_prefix = _get_default_s3_prefix(name)
            method = DataSetMethod.UPLOADED
            
            cls._validate_dataset_format(source)
            AIRHub.upload_to_s3(bucket_name, s3_prefix, source)

        # Create hub content document
        # TODO: Clean up hardcoded values - should come from intelligent defaults
        hub_content_document = DataSetHubContentDocument(
            dataset_s3_bucket=bucket_name,
            dataset_s3_prefix=s3_prefix,
            dataset_context_s3_uri="\"\"",
            dataset_type=DATASET_DEFAULT_TYPE,
            dataset_role_arn=role,
            conversation_id=DATASET_DEFAULT_CONVERSATION_ID,  # Required for now, needs cleanup
            conversation_checkpoint_id=DATASET_DEFAULT_CHECKPOINT_ID,
            dependencies=[],
        )
        
        document_str = hub_content_document.to_json()

        # Prepare tags for SearchKeywords
        if tags is None:
            tags = []
            if customization_technique is not None:
                tags.append((TAG_KEY_CUSTOMIZATION_TECHNIQUE, customization_technique.value))
            if method is not None:
                tags.insert(0, (TAG_KEY_METHOD, method.value))
        
        # Add domain-id to SearchKeywords if available
        if domain_id:
            tags.append((TAG_KEY_DOMAIN_ID, domain_id))

        # Determine new version
        new_version = _determine_new_version(DATASET_HUB_CONTENT_TYPE, name, sagemaker_session)
        # Import hub content
        AIRHub.import_hub_content(
            hub_content_type=DATASET_HUB_CONTENT_TYPE,
            hub_content_name=name,
            hub_content_version=new_version,
            document_schema_version=DATASET_DOCUMENT_SCHEMA_VERSION,
            hub_content_document=document_str,
            tags=tags,
            session=sagemaker_session
        )
        
        # Get the created dataset details
        describe_response = AIRHub.describe_hub_content(
            hub_content_type=DATASET_HUB_CONTENT_TYPE, 
            hub_content_name=name,
            session=sagemaker_session
        )
        
        dataset = cls(
            name=name,
            arn=describe_response[RESPONSE_KEY_HUB_CONTENT_ARN],
            version=describe_response[RESPONSE_KEY_HUB_CONTENT_VERSION],
            source=source,
            status=HubContentStatus.IMPORTING,
            description=description or f"Dataset {name}",
            customization_technique=customization_technique,
            method=method,
            created_time=describe_response[RESPONSE_KEY_CREATION_TIME],
            updated_time=describe_response[RESPONSE_KEY_LAST_MODIFIED_TIME],
            sagemaker_session=sagemaker_session,
        )
        
        if wait:
            dataset.wait()
        
        return dataset

    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="DataSet.get_versions")
    def get_versions(self) -> List["DataSet"]:
        """List all versions of this dataset."""
        versions = AIRHub.list_hub_content_versions(self.hub_content_type, self.name, session=self.sagemaker_session)
        
        datasets = []
        for v in versions:
            response = AIRHub.describe_hub_content(self.hub_content_type, self.name, v.get(RESPONSE_KEY_HUB_CONTENT_VERSION), session=self.sagemaker_session)
            doc = json.loads(response[RESPONSE_KEY_HUB_CONTENT_DOCUMENT])
            try:
                keywords = {kw.split(":")[0]: kw.split(":")[1] for kw in response.get(RESPONSE_KEY_HUB_CONTENT_SEARCH_KEYWORDS, []) if ":" in kw}
            except (IndexError, AttributeError):
                keywords = {}
            
            datasets.append(DataSet(
                name=response[RESPONSE_KEY_HUB_CONTENT_NAME],
                arn=response[RESPONSE_KEY_HUB_CONTENT_ARN],
                version=response[RESPONSE_KEY_HUB_CONTENT_VERSION],
                source=f"s3://{doc.get(DOC_KEY_DATASET_S3_BUCKET)}/{doc.get(DOC_KEY_DATASET_S3_PREFIX)}",
                status=response[RESPONSE_KEY_HUB_CONTENT_STATUS],
                description=response.get(RESPONSE_KEY_HUB_CONTENT_DESCRIPTION, ""),
                customization_technique=CustomizationTechnique(keywords.get(TAG_KEY_CUSTOMIZATION_TECHNIQUE)) if keywords.get(TAG_KEY_CUSTOMIZATION_TECHNIQUE) else None,
                method=DataSetMethod(keywords.get(TAG_KEY_METHOD, DATASET_DEFAULT_METHOD)),
                created_time=response.get(RESPONSE_KEY_CREATION_TIME),
                updated_time=response.get(RESPONSE_KEY_LAST_MODIFIED_TIME)
            ))
        
        return datasets

    @classmethod
    @classmethod
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="DataSet.get_all")
    def get_all(cls, max_results: Optional[int] = None, sagemaker_session=None):
        """List all entities of this type.
        
        Args:
            max_results: Maximum number of results to return
            
        Returns:
            Iterator for listed DataSet resources
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
        
        return islice(iterator, max_results) if max_results else iterator

    @classmethod
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="DataSet.split")
    def split(
        cls, 
        source: str, 
        train_split_ratio: float = 0.8
    ) -> Tuple["DataSet", "DataSet"]:
        """Split dataset into train and validation sets.
        
        Args:
            source: Path to the CSV dataset file
            train_split_ratio: Ratio of data to use for training (0.0-1.0)
            
        Returns:
            Tuple of (train_dataset, validation_dataset)
            
        Raises:
            ValueError: If split ratio is not between 0.0 and 1.0
            FileNotFoundError: If source file doesn't exist
            
        Note:
            This method currently only supports CSV files.
            TODO: Add support for JSONL files and test split functionality.
        """
        if not 0.0 < train_split_ratio < 1.0:
            raise ValueError("train_split_ratio must be between 0.0 and 1.0")
        
        if not os.path.exists(source):
            raise FileNotFoundError(f"Dataset file not found: {source}")
        
        # Read and split the dataset
        df = pd.read_csv(source)
        train_size = int(len(df) * train_split_ratio)
        train_df = df[:train_size]
        val_df = df[train_size:]

        # Create split file paths
        base_name = os.path.splitext(source)[0]
        train_path = f"{base_name}_train.csv"
        val_path = f"{base_name}_validation.csv"

        # Save split datasets
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        # Create DataSet objects
        train_dataset = cls.create(
            name=f"{os.path.basename(base_name)}_train",
            source=train_path,
            customization_technique=CustomizationTechnique.SFT,
        )
        val_dataset = cls.create(
            name=f"{os.path.basename(base_name)}_validation",
            source=val_path,
            customization_technique=CustomizationTechnique.SFT,
        )

        return (train_dataset, val_dataset)

    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="DataSet.create_version")
    def create_version(
        self, 
        source: str,
        customization_technique: Optional[CustomizationTechnique] = None
    ) -> bool:
        """Create a new version of this dataset.
        
        Args:
            source: S3 URI or local file path for the dataset
            customization_technique: Customization technique to use. If None, uses existing technique.
            
        Returns:
            True if version created successfully, False otherwise
        """
        try:
            # Get current dataset metadata
            response = AIRHub.describe_hub_content(
                hub_content_type=DATASET_HUB_CONTENT_TYPE,
                hub_content_name=self.name,
                session=self.sagemaker_session
            )
            
            # Parse existing keywords
            keywords = self._parse_keywords(response.get(RESPONSE_KEY_HUB_CONTENT_SEARCH_KEYWORDS, []))
            
            # Use provided technique or fall back to existing one
            existing_technique = keywords.get(TAG_KEY_CUSTOMIZATION_TECHNIQUE)
            technique = customization_technique or (CustomizationTechnique(existing_technique) if existing_technique else None)
            
            # Create new version
            DataSet.create(
                name=self.name,
                source=source,
                customization_technique=technique,
                tags=[
                    (TAG_KEY_CUSTOMIZATION_TECHNIQUE, technique.value),
                    (TAG_KEY_METHOD, keywords.get(TAG_KEY_METHOD, ""))
                ] if technique else [(TAG_KEY_METHOD, keywords.get(TAG_KEY_METHOD, ""))]
            )
            return True
        except Exception as e:
            print(f"Failed to create new version for dataset {self.name} with exception : {e}")
            return False

    @staticmethod
    def _parse_keywords(search_keywords: List[str]) -> dict:
        """Parse search keywords into a dictionary.
        
        Args:
            search_keywords: List of keyword strings in format "key:value"
            
        Returns:
            Dictionary mapping keyword keys to values
        """
        keywords = {}
        for kw in search_keywords:
            if ":" in kw:
                try:
                    key, value = kw.split(":", 1)
                    keywords[key] = value
                except (IndexError, AttributeError):
                    continue
        return keywords
