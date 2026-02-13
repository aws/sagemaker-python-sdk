# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Multi-threaded data ingestion for FeatureStore using SageMaker Core."""
import logging
import math
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Any, Dict, Iterable, List, Sequence, Union

import pandas as pd
from pandas import DataFrame
from pandas.api.types import is_list_like

from sagemaker.core.resources import FeatureGroup as CoreFeatureGroup
from sagemaker.core.shapes import FeatureValue
from sagemaker.core.utils.utils import Unassigned
from sagemaker.core.telemetry import Feature, _telemetry_emitter

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """Exception raised for errors during ingestion.

    Attributes:
        failed_rows: List of row indices that failed to ingest.
        message: Error message.
    """

    def __init__(self, failed_rows: List[int], message: str):
        self.failed_rows = failed_rows
        self.message = message
        super().__init__(self.message)


@dataclass
class IngestionManagerPandas:
    """Class to manage the multi-threaded data ingestion process.

    This class will manage the data ingestion process which is multi-threaded.

    Attributes:
        feature_group_name (str): name of the Feature Group.
        feature_definitions (Dict[str, Dict[Any, Any]]): dictionary of feature definitions
            where the key is the feature name and the value is the FeatureDefinition.
            The FeatureDefinition contains the data type of the feature.
        max_workers (int): number of threads to create.
        max_processes (int): number of processes to create. Each process spawns
            ``max_workers`` threads.
    """

    feature_group_name: str
    feature_definitions: Dict[str, Dict[Any, Any]]
    max_workers: int = 1
    max_processes: int = 1
    _async_result: Any = field(default=None, init=False)
    _processing_pool: Pool = field(default=None, init=False)
    _failed_indices: List[int] = field(default_factory=list, init=False)

    @property
    def failed_rows(self) -> List[int]:
        """Get rows that failed to ingest.

        Returns:
            List of row indices that failed to be ingested.
        """
        return self._failed_indices

    @_telemetry_emitter(Feature.FEATURE_STORE, "IngestionManagerPandas.run")
    def run(
        self,
        data_frame: DataFrame,
        target_stores: List[str] = None,
        wait: bool = True,
        timeout: Union[int, float] = None,
    ):
        """Start the ingestion process.

        Args:
            data_frame (DataFrame): source DataFrame to be ingested.
            target_stores (List[str]): list of target stores ("OnlineStore", "OfflineStore").
                If None, the default target store is used.
            wait (bool): whether to wait for the ingestion to finish or not.
            timeout (Union[int, float]): ``concurrent.futures.TimeoutError`` will be raised
                if timeout is reached.
        """
        if self.max_workers == 1 and self.max_processes == 1:
            self._run_single_process_single_thread(data_frame=data_frame, target_stores=target_stores)
        else:
            self._run_multi_process(data_frame=data_frame, target_stores=target_stores, wait=wait, timeout=timeout)

    def wait(self, timeout: Union[int, float] = None):
        """Wait for the ingestion process to finish.

        Args:
            timeout (Union[int, float]): ``concurrent.futures.TimeoutError`` will be raised
                if timeout is reached.
        """
        try:
            results = self._async_result.get(timeout=timeout)
        except KeyboardInterrupt as e:
            self._processing_pool.terminate()
            self._processing_pool.join()
            raise e
        else:
            self._processing_pool.close()
            self._processing_pool.join()

        self._failed_indices = [idx for failed in results for idx in failed]

        if self._failed_indices:
            raise IngestionError(
                self._failed_indices,
                f"Failed to ingest some data into FeatureGroup {self.feature_group_name}",
            )

    def _run_single_process_single_thread(
        self,
        data_frame: DataFrame,
        target_stores: List[str] = None,
    ):
        """Ingest utilizing a single process and a single thread."""
        logger.info("Started single-threaded ingestion for %d rows", len(data_frame))
        failed_rows = []

        fg = CoreFeatureGroup(feature_group_name=self.feature_group_name)

        for row in data_frame.itertuples():
            self._ingest_row(
                data_frame=data_frame,
                row=row,
                feature_group=fg,
                feature_definitions=self.feature_definitions,
                failed_rows=failed_rows,
                target_stores=target_stores,
            )

        self._failed_indices = failed_rows
        if self._failed_indices:
            raise IngestionError(
                self._failed_indices,
                f"Failed to ingest some data into FeatureGroup {self.feature_group_name}",
            )

    def _run_multi_process(
        self,
        data_frame: DataFrame,
        target_stores: List[str] = None,
        wait: bool = True,
        timeout: Union[int, float] = None,
    ):
        """Start the ingestion process with the specified number of processes."""
        batch_size = math.ceil(data_frame.shape[0] / self.max_processes)

        args = []
        for i in range(self.max_processes):
            start_index = min(i * batch_size, data_frame.shape[0])
            end_index = min(i * batch_size + batch_size, data_frame.shape[0])
            args.append((
                self.max_workers,
                self.feature_group_name,
                self.feature_definitions,
                data_frame[start_index:end_index],
                target_stores,
                start_index,
                timeout,
            ))

        def init_worker():
            signal.signal(signal.SIGINT, signal.SIG_IGN)

        self._processing_pool = Pool(self.max_processes, init_worker)

        self._async_result = self._processing_pool.starmap_async(
            IngestionManagerPandas._run_multi_threaded,
            args,
        )

        if wait:
            self.wait(timeout=timeout)

    @staticmethod
    def _run_multi_threaded(
        max_workers: int,
        feature_group_name: str,
        feature_definitions: Dict[str, Dict[Any, Any]],
        data_frame: DataFrame,
        target_stores: List[str] = None,
        row_offset: int = 0,
        timeout: Union[int, float] = None,
    ) -> List[int]:
        """Start multi-threaded ingestion within a single process."""
        executor = ThreadPoolExecutor(max_workers=max_workers)
        batch_size = math.ceil(data_frame.shape[0] / max_workers)

        futures = {}
        for i in range(max_workers):
            start_index = min(i * batch_size, data_frame.shape[0])
            end_index = min(i * batch_size + batch_size, data_frame.shape[0])
            future = executor.submit(
                IngestionManagerPandas._ingest_single_batch,
                data_frame=data_frame,
                feature_group_name=feature_group_name,
                feature_definitions=feature_definitions,
                start_index=start_index,
                end_index=end_index,
                target_stores=target_stores,
            )
            futures[future] = (start_index + row_offset, end_index + row_offset)

        failed_indices = []
        for future in as_completed(futures, timeout=timeout):
            start, end = futures[future]
            failed_rows = future.result()
            if not failed_rows:
                logger.info("Successfully ingested row %d to %d", start, end)
            failed_indices.extend(failed_rows)

        executor.shutdown(wait=False)
        return failed_indices

    @staticmethod
    def _ingest_single_batch(
        data_frame: DataFrame,
        feature_group_name: str,
        feature_definitions: Dict[str, Dict[Any, Any]],
        start_index: int,
        end_index: int,
        target_stores: List[str] = None,
    ) -> List[int]:
        """Ingest a single batch of DataFrame rows into FeatureStore."""
        logger.info("Started ingesting index %d to %d", start_index, end_index)
        failed_rows = []

        fg = CoreFeatureGroup(feature_group_name=feature_group_name)

        for row in data_frame[start_index:end_index].itertuples():
            IngestionManagerPandas._ingest_row(
                data_frame=data_frame,
                row=row,
                feature_group=fg,
                feature_definitions=feature_definitions,
                failed_rows=failed_rows,
                target_stores=target_stores,
            )

        return failed_rows

    @staticmethod
    def _ingest_row(
        data_frame: DataFrame,
        row: Iterable,
        feature_group: CoreFeatureGroup,
        feature_definitions: Dict[str, Dict[Any, Any]],
        failed_rows: List[int],
        target_stores: List[str] = None,
    ):
        """Ingest a single DataFrame row into FeatureStore using SageMaker Core."""
        try:
            record = []
            for index in range(1, len(row)):
                feature_name = data_frame.columns[index - 1]
                feature_value = row[index]

                if not IngestionManagerPandas._feature_value_is_not_none(feature_value):
                    continue

                if IngestionManagerPandas._is_feature_collection_type(feature_name, feature_definitions):
                    record.append(FeatureValue(
                        feature_name=feature_name,
                        value_as_string_list=IngestionManagerPandas._convert_to_string_list(feature_value),
                    ))
                else:
                    record.append(FeatureValue(
                        feature_name=feature_name,
                        value_as_string=str(feature_value),
                    ))

            # Use SageMaker Core's put_record directly
            feature_group.put_record(
                record=record,
                target_stores=target_stores,
            )

        except Exception as e:
            logger.error("Failed to ingest row %d: %s", row[0], e)
            failed_rows.append(row[0])

    @staticmethod
    def _is_feature_collection_type(
        feature_name: str,
        feature_definitions: Dict[str, Dict[Any, Any]],
    ) -> bool:
        """Check if the feature is a collection type."""
        feature_def = feature_definitions.get(feature_name)
        if feature_def:
            collection_type = feature_def.get("CollectionType")
            if isinstance(collection_type, Unassigned) or collection_type is None or collection_type == "":
                return False
            return True
        return False

    @staticmethod
    def _feature_value_is_not_none(feature_value: Any) -> bool:
        """Check if the feature value is not None.

        For Collection Type features, we check if the value is not None.
        For Scalar values, we use pd.notna() to keep the behavior same.
        """
        if not is_list_like(feature_value):
            return pd.notna(feature_value)
        return feature_value is not None

    @staticmethod
    def _convert_to_string_list(feature_value: List[Any]) -> List[str]:
        """Convert a list of feature values to a list of strings."""
        if not is_list_like(feature_value):
            raise ValueError(
                f"Invalid feature value: {feature_value} for a collection type feature "
                f"must be an Array, but was {type(feature_value)}"
            )
        return [str(v) if v is not None else None for v in feature_value]
