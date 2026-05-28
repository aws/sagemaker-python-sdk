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

"""Factory for high-level DataSet orchestration workflows in AIR Hub."""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

from sagemaker.ai_registry.dataset_transformation import DatasetFormat
from sagemaker.ai_registry.dataset_utils import CustomizationTechnique
from sagemaker.core.helper.session_helper import Session

if TYPE_CHECKING:
    from sagemaker.ai_registry.dataset import DataSet

logger = logging.getLogger(__name__)


class DataSetHubFactory:
    """Orchestration layer for multi-step DataSet workflows.

    Composes DataSet entity primitives (create, create_version, get_versions)
    into higher-level operations like transformation, splitting, scoring,
    and generation. Keeps the DataSet entity focused on CRUD while this
    factory owns the complex workflow logic.
    """

    @classmethod
    def _resolve_dataset(
        cls,
        dataset: Union[DataSet, str],
        sagemaker_session: Optional[Session] = None,
    ) -> DataSet:
        """Resolve a DataSet reference to a hydrated DataSet instance.

        Args:
            dataset: A DataSet instance, dataset name, or dataset ARN.
            sagemaker_session: Optional SageMaker session.

        Returns:
            A hydrated DataSet instance.
        """
        if isinstance(dataset, str):
            from sagemaker.ai_registry.dataset import DataSet
            return DataSet.get(name=dataset, sagemaker_session=sagemaker_session)
        return dataset

    @classmethod
    def _download_to_local(cls, s3_uri: str) -> str:
        """Download an S3 file to a local temp path.

        Args:
            s3_uri: S3 URI to download.

        Returns:
            Local file path of the downloaded file.
        """
        suffix = os.path.splitext(s3_uri)[-1] or ".jsonl"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.close()
        from sagemaker.ai_registry.air_hub import AIRHub
        AIRHub.download_from_s3(s3_uri, tmp.name)
        return tmp.name

    @classmethod
    def transform_dataset(
        cls,
        name: str,
        target_format: DatasetFormat,
        source: Optional[str] = None,
        dataset: Optional[Union[DataSet, str]] = None,
        customization_technique: Optional[CustomizationTechnique] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> DataSet:
        """Transform a dataset to a target format and persist the result.

        Scenario 1 — source path provided (local or S3):
            Detect format → transform → DataSet.create() with the transformed file.

        Scenario 2 — existing dataset (DataSet instance, name, or ARN):
            Rehydrate → download from S3 → detect format → transform →
            DataSet.create_version() on the existing dataset.

        Args:
            name: Name for the resulting dataset.
            target_format: The desired output format.
            source: S3 URI or local file path. Mutually exclusive with dataset.
            dataset: Existing DataSet instance, name, or ARN. Mutually exclusive with source.
            customization_technique: Customization technique to apply.
            sagemaker_session: Optional SageMaker session.

        Returns:
            The transformed DataSet instance.

        Raises:
            ValueError: If neither or both of source and dataset are provided.
        """
        from sagemaker.ai_registry.dataset_transformation import DatasetTransformation
        from sagemaker.ai_registry.dataset import DataSet

        if (source is None) == (dataset is None):
            raise ValueError(
                "Exactly one of 'source' or 'dataset' must be provided, not both or neither."
            )

        # ----------------------------------------------------------
        # Scenario 1: raw source path (local file or S3 URI)
        # ----------------------------------------------------------
        if source is not None:
            local_path = source
            is_temp = False

            # If S3, download to a local temp file for detection + transformation
            if source.startswith("s3://"):
                local_path = cls._download_to_local(source)
                is_temp = True

            try:
                source_format = DatasetTransformation.detect_format(local_path)
                logger.info("Detected source format: %s", source_format.value)

                transformed_path = DatasetTransformation.transform_file(
                    file_path=local_path,
                    source_format=source_format,
                    target_format=target_format,
                )

                return DataSet.create(
                    name=name,
                    source=transformed_path,
                    customization_technique=customization_technique,
                    sagemaker_session=sagemaker_session,
                )
            finally:
                if is_temp and os.path.exists(local_path):
                    os.remove(local_path)

        # ----------------------------------------------------------
        # Scenario 2: existing dataset (instance, name, or ARN)
        # ----------------------------------------------------------
        resolved = cls._resolve_dataset(dataset, sagemaker_session)
        local_path = cls._download_to_local(resolved.source)

        try:
            source_format = DatasetTransformation.detect_format(local_path)
            logger.info("Detected source format: %s", source_format.value)

            transformed_path = DatasetTransformation.transform_file(
                file_path=local_path,
                source_format=source_format,
                target_format=target_format,
            )

            resolved.create_version(
                source=transformed_path,
                customization_technique=customization_technique,
            )

            # Re-fetch to return the latest version
            return DataSet.get(name=resolved.name, sagemaker_session=sagemaker_session)
        finally:
            if os.path.exists(local_path):
                os.remove(local_path)

    @classmethod
    def split_dataset(
        cls,
        name: str,
        source: Optional[str] = None,
        dataset: Optional[Union[DataSet, str]] = None,
        train_split_ratio: float = 0.8,
        customization_technique: Optional[CustomizationTechnique] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> Tuple[DataSet, DataSet]:
        """Split a dataset into train and validation sets.

        Args:
            name: Base name for the resulting datasets (suffixed with _train/_validation).
            source: S3 URI or local file path to split.
            dataset: Existing DataSet instance, name, or ARN. Mutually exclusive with source.
            train_split_ratio: Ratio of data for training (0.0-1.0).
            customization_technique: Customization technique for the resulting datasets.
            sagemaker_session: Optional SageMaker session.

        Returns:
            Tuple of (train_dataset, validation_dataset).

        Raises:
            ValueError: If neither or both of source and dataset are provided,
                or if train_split_ratio is not between 0.0 and 1.0.
        """
        raise NotImplementedError

    @classmethod
    def score_dataset(
        cls,
        name: str,
        scoring_params: Dict,
        customization_technique: Optional[CustomizationTechnique] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> DataSet:
        """Score a dataset by creating a new dataset that triggers a backend scoring workflow.

        Args:
            name: Name for the scored dataset.
            scoring_params: Parameters defining the scoring configuration.
            customization_technique: Customization technique for the scored dataset.
            sagemaker_session: Optional SageMaker session.

        Returns:
            The created DataSet representing the scoring job.
        """
        raise NotImplementedError

    @classmethod
    def generate_dataset(
        cls,
        name: str,
        generation_params: Dict,
        dataset: Optional[Union[DataSet, str]] = None,
        customization_technique: Optional[CustomizationTechnique] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> DataSet:
        """Generate a dataset using the provided generation parameters.

        Args:
            name: Name for the generated dataset.
            generation_params: Parameters defining the generation configuration.
            dataset: Optional existing DataSet (instance, name, or ARN) to version.
            customization_technique: Customization technique for the generated dataset.
            sagemaker_session: Optional SageMaker session.

        Returns:
            The created or versioned DataSet.
        """
        raise NotImplementedError
