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
"""Dataset wrappers and helpers for the AI Registry."""

from typing import List, Optional, TYPE_CHECKING
from enum import Enum
from collections.abc import Sequence
import json
import warnings

from sagemaker.ai_registry.air_constants import (
    DATASET_DOCUMENT_SCHEMA_VERSION,
    DATASET_SUPPORTED_SCHEMA_VERSIONS,
)

if TYPE_CHECKING:
    from sagemaker.ai_registry.dataset import DataSet


class CustomizationTechnique(str, Enum):
    """Customization technique for dataset."""

    SFT = "sft"
    DPO = "dpo"
    RLVR = "rlvr"


class DataSetMethod(Enum):
    """Enum for DataSet method types."""

    UPLOADED = "uploaded"
    GENERATED = "generated"


class DataSetList(Sequence):
    """List-like wrapper for datasets with pagination support."""

    def __init__(self, datasets: List["DataSet"], next_token: Optional[str]):
        self._datasets = datasets
        self.next_token = next_token

    def __getitem__(self, index):
        """Return the dataset at the given index."""
        return self._datasets[index]

    def __len__(self):
        """Return the number of datasets."""
        return len(self._datasets)

    def __repr__(self):
        """Return the repr of the underlying datasets list."""
        return repr(self._datasets)

    def __str__(self):
        """Return the string form of the underlying datasets list."""
        return str(self._datasets)


class DataSetHubContentDocument:
    """Hub content document for dataset.

    Supports both v2 (legacy) and v5 (current) document schemas.
    v5 requires DatasetFileFormat and does not allow legacy fields like
    ConversationId, DatasetRoleArn, Dependencies, etc.
    """

    def __init__(
        self,
        dataset_s3_bucket: Optional[str] = None,
        dataset_s3_prefix: Optional[str] = None,
        dataset_file_format: Optional[str] = None,
        dataset_type: Optional[str] = None,
        specification_arn: Optional[str] = None,
        # Legacy v2 fields — only used when schema_version < 5.0.0
        dataset_role_arn: Optional[str] = None,
        dataset_context_s3_uri: Optional[str] = None,
        conversation_id: Optional[str] = None,
        conversation_checkpoint_id: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        content_metadata: Optional[dict] = None,
    ):
        self.dataset_s3_bucket = dataset_s3_bucket
        self.dataset_s3_prefix = dataset_s3_prefix
        self.dataset_file_format = dataset_file_format
        self.dataset_type = dataset_type
        self.specification_arn = specification_arn
        # Legacy v2 fields
        self.dataset_role_arn = dataset_role_arn
        self.dataset_context_s3_uri = dataset_context_s3_uri
        self.conversation_id = conversation_id
        self.conversation_checkpoint_id = conversation_checkpoint_id
        self.dependencies = dependencies or []
        self.content_metadata = content_metadata

    def to_json(self, schema_version: str = DATASET_DOCUMENT_SCHEMA_VERSION) -> str:
        """Convert to JSON string compatible with the given schema version.

        Args:
            schema_version: The document schema version to target. Defaults to the
                current version (:data:`DATASET_DOCUMENT_SCHEMA_VERSION`).
                v5.0.0 produces a minimal document with DatasetFileFormat (required).
                v2.0.0 produces the legacy document with all fields.
                In every version, ``content_metadata`` (when set) is emitted as
                ``ContentMetadata``; the v5.0.0 schema defines this object
                (SourceFeatureGroups required; ExtractionMethod and
                AthenaQueryExecutionId optional) for Feature Store lineage.

                Any version older than the current one is out of date and emits a
                ``DeprecationWarning``. The shapes differ materially rather than
                cosmetically -- the legacy document carries ``DatasetRoleArn``,
                ``DatasetContextS3Uri``, ``ConversationId``,
                ``ConversationCheckpointId``, ``Dependencies`` and
                ``SpecificationArn``, none of which the current schema accepts,
                and it renames ``SpecificationArn`` to ``SpecS3Uri``. Pin an older
                version only to talk to a Hub that has not moved.
        """
        if schema_version != DATASET_DOCUMENT_SCHEMA_VERSION:
            known = (
                ""
                if schema_version in DATASET_SUPPORTED_SCHEMA_VERSIONS
                else f" It is also not one of the supported versions "
                f"{DATASET_SUPPORTED_SCHEMA_VERSIONS}."
            )
            warnings.warn(
                f"Dataset document schema version {schema_version!r} is out of date; "
                f"the current version is {DATASET_DOCUMENT_SCHEMA_VERSION!r}. The "
                f"documents differ in which fields they carry, not just in version "
                f"number, so a consumer expecting the current shape will not find the "
                f"legacy fields.{known}",
                DeprecationWarning,
                stacklevel=2,
            )
        if schema_version >= "3.0.0":
            # v3+ / v5 schema: only allowed fields, DatasetFileFormat required
            content = {}
            if self.dataset_file_format:
                content["DatasetFileFormat"] = self.dataset_file_format
            if self.dataset_type:
                content["DatasetType"] = self.dataset_type
            if self.dataset_s3_bucket:
                content["DatasetS3Bucket"] = self.dataset_s3_bucket
            if self.dataset_s3_prefix:
                content["DatasetS3Prefix"] = self.dataset_s3_prefix
            if self.specification_arn:
                content["SpecS3Uri"] = self.specification_arn
            if self.content_metadata:
                content["ContentMetadata"] = self.content_metadata
            return json.dumps(content)
        else:
            # v2 legacy schema
            content = {"DatasetType": self.dataset_type or "CUSTOMER_PROVIDED"}
            if self.dataset_role_arn:
                content["DatasetRoleArn"] = self.dataset_role_arn
            if self.dataset_s3_bucket:
                content["DatasetS3Bucket"] = self.dataset_s3_bucket
            if self.dataset_s3_prefix:
                content["DatasetS3Prefix"] = self.dataset_s3_prefix
            if self.dataset_context_s3_uri:
                content["DatasetContextS3Uri"] = self.dataset_context_s3_uri
            if self.specification_arn:
                # The legacy key. v3+ renamed this to SpecS3Uri above; the v2
                # rewrite dropped it entirely, so a caller that set
                # specification_arn and targeted v2 lost the field silently.
                content["SpecificationArn"] = self.specification_arn
            if self.conversation_id:
                content["ConversationId"] = self.conversation_id
            if self.conversation_checkpoint_id:
                content["ConversationCheckpointId"] = self.conversation_checkpoint_id
            content["Dependencies"] = self.dependencies
            if self.content_metadata:
                content["ContentMetadata"] = self.content_metadata
            return json.dumps(content)


def _get_default_s3_prefix(name: str) -> str:
    """Get default S3 prefix in format datasets/{name}/{current_date_time}.jsonl."""
    from datetime import datetime

    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"datasets/{name}/{current_datetime}.jsonl"
