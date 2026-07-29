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
    def validate_dataset(file_path: str) -> bool:
        """
        Validate if the dataset adheres to any known format.
        
        Args:
            file_path: Path to the JSONL, Parquet, JSON, or CSV file
            
        Returns:
            True if dataset is valid according to any known format, False otherwise
        """
        # Parquet files are binary — validate by reading the header
        if file_path.endswith(".parquet"):
            return DatasetFormatDetector._validate_parquet(file_path)

        # CSV files — validate by checking header row exists
        if file_path.endswith(".csv"):
            return DatasetFormatDetector._validate_csv(file_path)

        # JSON array files
        if file_path.endswith(".json"):
            return DatasetFormatDetector._validate_json(file_path)

        import jsonschema
        
        # Schema-based formats (JSONL)
        schema_formats = [
            "dpo", "converse", "hf_preference", "hf_prompt_completion",
            "verl", "openai_chat", "genqa"
        ]
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        
                        # Try schema validation first
                        for format_name in schema_formats:
                            schema = DatasetFormatDetector._load_schema(format_name)
                            if schema:
                                try:
                                    jsonschema.validate(instance=data, schema=schema)
                                    return True
                                except jsonschema.exceptions.ValidationError:
                                    continue
                        
                        # Check for RFT-style format (messages + additional fields)
                        if DatasetFormatDetector._is_rft_format(data):
                            return True
                        break
            return False
        except (json.JSONDecodeError, FileNotFoundError, IOError):
            return False
    
    @staticmethod
    def _validate_parquet(file_path: str) -> bool:
        """Validate that a file is a valid Parquet file by checking its magic bytes."""
        try:
            with open(file_path, "rb") as f:
                magic = f.read(4)
                return magic == b"PAR1"
        except (FileNotFoundError, IOError):
            return False

    @staticmethod
    def _validate_csv(file_path: str) -> bool:
        """Validate that a file is a valid CSV with a header row."""
        import csv

        try:
            with open(file_path, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                return header is not None and len(header) > 0
        except (FileNotFoundError, IOError):
            return False

    @staticmethod
    def _validate_json(file_path: str) -> bool:
        """Validate that a file is a valid JSON array of objects."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                return isinstance(data, list) and len(data) > 0
        except (json.JSONDecodeError, FileNotFoundError, IOError):
            return False

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
