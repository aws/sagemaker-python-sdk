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
"""Per-category, version-specific extraction strategies for job config documents.

Each strategy reads one ``JobConfigDocument`` shape.

PRIVATE — not exported. This is where anything that has to branch on a
category's ``JobConfigSchemaVersion`` lives. Version dispatch happens once at
the call boundary via ``{Cat}Shape.for_version(...)``; every downstream caller
holds a strategy object and never re-checks the version string.

Layout. One class hierarchy per category that has a versioned shape. Today
only DataQualityEvaluation qualifies (``DqShape`` → ``V1DqShape`` /
``V2DqShape`` / ``V3DqShape``). When another category grows a shape variant, add
``{Cat}Shape`` + ``V1{Cat}Shape`` / ``V2{Cat}Shape`` in this file with the
same ``for_version`` static-factory pattern. Callers stay category-local and
stateless — no cross-category dispatcher.

DQ shapes today, plus a v1 fallback for missing / legacy / unknown versions:

* v1 (any version not listed below): scalar ``DatasetArn`` at the root,
  ``OutputDataConfig.OutputS3Uri``, bare-string ``Metrics``.
* v2 (``"0.2.0"``, ``"2.0.0"``): ``InputDataConfig: [{DatasetArn}]``,
  ``OutputDataConfig.S3OutputPath``, ``Metrics: [{Metric, Config?}]``.
* v3 (``"3.0.0"``): ``EvaluationConfig.Operations: [{Operator, Config?}]``, with v2's
  ``InputDataConfig`` and ``OutputDataConfig``.
"""

from __future__ import absolute_import

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

# Version strings that select the v2 DQ shape. "0.2.0" is the preview number,
# kept below "1.0.0" per semver so ``ListJobSchemaVersions`` "latest" callers
# stay on v1 until they explicitly opt in. The service plans to rename the
# schema file to 2.0.0 "without other changes" when v2 becomes the default —
# accept both now so that rename does not silently re-route v2 jobs to the
# v1 parser.
_V2_DQ_SCHEMA_VERSIONS = ("0.2.0", "2.0.0")

# Version strings that select the v3 DQ shape.
_V3_DQ_SCHEMA_VERSIONS = ("3.0.0",)


class DqShape(ABC):
    """Version-specific extraction for DQ ``JobConfigDocument`` shapes.

    Callers resolve the shape once via :meth:`for_version` and read the two
    fields that flip between v1 and v2 through :meth:`extract_output_s3_uri`
    and :meth:`extract_dataset_arn`. The strategy never mutates its input.
    """

    @staticmethod
    def for_version(schema_version: Optional[str]) -> "DqShape":
        """Return the shape strategy for a resolved schema version.

        Dispatch rules (in priority order):
        * ``"0.2.0"`` (legacy preview, semver < 1.0.0) → v2 shape
        * major == 2 → v2 shape (accepts "2", "2.0", "2.0.0", etc.)
        * major == 3 → v3 shape (accepts "3", "3.0", "3.0.0", etc.)
        * everything else (None, empty, "1.x", unknown/garbage) → v1 shape

        The service has historically returned both two-part ("2.0") and
        three-part ("2.0.0") version strings for the same schema, causing
        exact-string membership tests to silently fall back to the v1
        parser. Major-version dispatch avoids that fragility.
        """
        # Special-case the pre-1.0 preview alias before parsing.
        if schema_version == "0.2.0":
            return _V2_DQ_SHAPE

        if not schema_version:
            return _V1_DQ_SHAPE

        # Extract the major version component; any non-numeric prefix falls to v1.
        try:
            major = int(schema_version.split(".")[0])
        except (ValueError, IndexError):
            return _V1_DQ_SHAPE

        if major == 2:
            return _V2_DQ_SHAPE
        if major == 3:
            return _V3_DQ_SHAPE
        return _V1_DQ_SHAPE

    @abstractmethod
    def extract_output_s3_uri(self, document: Dict[str, Any]) -> Optional[str]:
        """Return the resolved-payload S3 output URI, or ``None``.

        ``None`` means the field this version expects is missing or malformed.
        Never falls back to the other version's field name: mixing shapes is a
        validation error and should surface as ``None`` here.
        """

    @abstractmethod
    def extract_dataset_arn(self, document: Dict[str, Any]) -> Optional[str]:
        """Return the resolved-payload input dataset ARN, or ``None``.

        ``None`` means the field is missing or malformed. Same no-fallback
        contract as :meth:`extract_output_s3_uri`.
        """


class V1DqShape(DqShape):
    """Legacy DQ shape: scalar ``DatasetArn``, ``OutputDataConfig.OutputS3Uri``."""

    def extract_output_s3_uri(self, document: Dict[str, Any]) -> Optional[str]:
        """Return ``OutputDataConfig.OutputS3Uri``, or ``None``."""
        output_cfg = document.get("OutputDataConfig")
        if not isinstance(output_cfg, dict):
            return None
        value = output_cfg.get("OutputS3Uri")
        return value if isinstance(value, str) else None

    def extract_dataset_arn(self, document: Dict[str, Any]) -> Optional[str]:
        """Return the top-level ``DatasetArn``, or ``None``."""
        value = document.get("DatasetArn")
        return value if isinstance(value, str) else None


class V2DqShape(DqShape):
    """v2 (0.2.0) DQ shape.

    ``InputDataConfig: [{DatasetArn}]`` and ``OutputDataConfig.S3OutputPath``.
    """

    def extract_output_s3_uri(self, document: Dict[str, Any]) -> Optional[str]:
        """Return ``OutputDataConfig.S3OutputPath``, or ``None``."""
        output_cfg = document.get("OutputDataConfig")
        if not isinstance(output_cfg, dict):
            return None
        value = output_cfg.get("S3OutputPath")
        return value if isinstance(value, str) else None

    def extract_dataset_arn(self, document: Dict[str, Any]) -> Optional[str]:
        """Return ``InputDataConfig[0].DatasetArn``, or ``None``."""
        input_cfg = document.get("InputDataConfig")
        if not isinstance(input_cfg, list) or not input_cfg:
            return None
        first = input_cfg[0]
        if not isinstance(first, dict):
            return None
        value = first.get("DatasetArn")
        return value if isinstance(value, str) else None


class V3DqShape(V2DqShape):
    """v3 (3.0.0) DQ shape.

    ``EvaluationConfig.Operations`` replaces v2's metric groups. ``InputDataConfig``
    and ``OutputDataConfig`` are unchanged from v2, so both extractors are inherited.
    """


# Singletons — the strategies are stateless, so we avoid per-call allocation.
_V1_DQ_SHAPE: DqShape = V1DqShape()
_V2_DQ_SHAPE: DqShape = V2DqShape()
_V3_DQ_SHAPE: DqShape = V3DqShape()
