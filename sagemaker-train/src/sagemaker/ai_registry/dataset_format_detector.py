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

import json
from typing import Dict, Any, Optional
from pathlib import Path


class DatasetFormatDetector:
    """Utility class for detecting dataset formats."""
    
    # Schema directory
    SCHEMA_DIR = Path(__file__).parent / "schemas"
    
    @staticmethod
    def _load_schema(format_name: str) -> Dict[str, Any]:
        """Load JSON schema for a format."""
        schema_path = DatasetFormatDetector.SCHEMA_DIR / f"{format_name}.json"
        if schema_path.exists():
            with open(schema_path) as f:
                return json.load(f)
        return {}
    
    @staticmethod
    def detect_format(file_path: str) -> Optional[str]:
        """Detect the dataset format from the first record in a JSONL file.

        Args:
            file_path: Path to the JSONL file.

        Returns:
            Format name (e.g. 'genqa', 'openai_chat') if detected, None otherwise.
        """
        import jsonschema

        schema_formats = [
            "dpo", "converse", "hf_preference", "hf_prompt_completion",
            "verl", "openai_chat", "genqa"
        ]

        try:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)

                    for format_name in schema_formats:
                        schema = DatasetFormatDetector._load_schema(format_name)
                        if schema:
                            try:
                                jsonschema.validate(instance=data, schema=schema)
                                return format_name
                            except jsonschema.exceptions.ValidationError:
                                continue

                    if DatasetFormatDetector._is_rft_format(data):
                        return "rft"
                    break
        except (json.JSONDecodeError, FileNotFoundError, IOError):
            pass
        return None

    @staticmethod
    def validate_dataset(file_path: str) -> bool:
        """Validate if the dataset adheres to any known format.

        Args:
            file_path: Path to the JSONL file.

        Returns:
            True if dataset is valid according to any known format, False otherwise.
        """
        return DatasetFormatDetector.detect_format(file_path) is not None
    
    @staticmethod
    def _is_rft_format(data: Dict[str, Any]) -> bool:
        """Check if data matches RFT format pattern."""
        if not isinstance(data, dict) or "messages" not in data:
            return False
        
        messages = data["messages"]
        if not isinstance(messages, list) or not messages:
            return False
        
        # Check message structure
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if "role" not in msg or "content" not in msg:
                return False
            if not isinstance(msg["role"], str) or not isinstance(msg["content"], str):
                return False
        
        return True
