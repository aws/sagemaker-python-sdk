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

from typing import List, Optional
from enum import Enum
from collections.abc import Sequence
import json


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
        return self._datasets[index]

    def __len__(self):
        return len(self._datasets)

    def __repr__(self):
        return repr(self._datasets)

    def __str__(self):
        return str(self._datasets)


class DataSetHubContentDocument:
    """Hub content document for dataset."""
    
    def __init__(
        self,
        dataset_type: Optional[str] = "AGENT_GENERATED",
        dataset_role_arn: Optional[str] = None,
        dataset_s3_bucket: Optional[str] = None,
        dataset_s3_prefix: Optional[str] = None,
        dataset_context_s3_uri: Optional[str] = None,
        specification_arn: Optional[str] = None,
        conversation_id: Optional[str] = None,
        conversation_checkpoint_id: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
    ):
        self.dataset_type = dataset_type
        self.dataset_role_arn = dataset_role_arn
        self.dataset_s3_bucket = dataset_s3_bucket
        self.dataset_s3_prefix = dataset_s3_prefix
        self.dataset_context_s3_uri = dataset_context_s3_uri
        self.specification_arn = specification_arn
        self.conversation_id = conversation_id
        self.conversation_checkpoint_id = conversation_checkpoint_id
        self.dependencies = dependencies or []
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        content = {"DatasetType": self.dataset_type}
        if self.dataset_role_arn:
            content["DatasetRoleArn"] = self.dataset_role_arn
        if self.dataset_s3_bucket:
            content["DatasetS3Bucket"] = self.dataset_s3_bucket
        if self.dataset_s3_prefix:
            content["DatasetS3Prefix"] = self.dataset_s3_prefix
        if self.dataset_context_s3_uri:
            content["DatasetContextS3Uri"] = self.dataset_context_s3_uri
        if self.specification_arn:
            content["SpecificationArn"] = self.specification_arn
        if self.conversation_id:
            content["ConversationId"] = self.conversation_id
        if self.conversation_checkpoint_id:
            content["ConversationCheckpointId"] = self.conversation_checkpoint_id
        content["Dependencies"] = self.dependencies
        return json.dumps(content)


def _get_default_s3_prefix(name: str) -> str:
    """Get default S3 prefix in format datasets/{name}/{current_date_time}.jsonl."""
    from datetime import datetime
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"datasets/{name}/{current_datetime}.jsonl"
