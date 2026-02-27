import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict
from urllib.parse import urlparse
import pandas as pd
from pandas import DataFrame

from sagemaker.mlops.feature_store.feature_utils import (
    start_query_execution,
    get_query_execution,
    wait_for_athena_query,
    download_athena_query_result,
)

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.telemetry import Feature, _telemetry_emitter

@dataclass
class AthenaQuery:
    """Class to manage querying of feature store data with AWS Athena.

    This class instantiates a AthenaQuery object that is used to retrieve data from feature store
    via standard SQL queries.

    Attributes:
        catalog (str): name of the data catalog.
        database (str): name of the database.
        table_name (str): name of the table.
        sagemaker_session (Session): instance of the Session class to perform boto calls.
    """

    catalog: str
    database: str
    table_name: str
    sagemaker_session: Session
    _current_query_execution_id: str = field(default=None, init=False)
    _result_bucket: str = field(default=None, init=False)
    _result_file_prefix: str = field(default=None, init=False)

    @_telemetry_emitter(Feature.FEATURE_STORE, "AthenaQuery.run")
    def run(
        self, query_string: str, output_location: str, kms_key: str = None, workgroup: str = None
    ) -> str:
        """Execute a SQL query given a query string, output location and kms key.

        This method executes the SQL query using Athena and outputs the results to output_location
        and returns the execution id of the query.

        Args:
            query_string: SQL query string.
            output_location: S3 URI of the query result.
            kms_key: KMS key id. If set, will be used to encrypt the query result file.
            workgroup (str): The name of the workgroup in which the query is being started.

        Returns:
            Execution id of the query.
        """
        response = start_query_execution(
            session=self.sagemaker_session,
            catalog=self.catalog,
            database=self.database,
            query_string=query_string,
            output_location=output_location,
            kms_key=kms_key,
            workgroup=workgroup,
        )

        self._current_query_execution_id = response["QueryExecutionId"]
        parsed_result = urlparse(output_location, allow_fragments=False)
        self._result_bucket = parsed_result.netloc
        self._result_file_prefix = parsed_result.path.strip("/")
        return self._current_query_execution_id

    def wait(self):
        """Wait for the current query to finish."""
        wait_for_athena_query(self.sagemaker_session, self._current_query_execution_id)

    def get_query_execution(self) -> Dict[str, Any]:
        """Get execution status of the current query.

        Returns:
            Response dict from Athena.
        """
        return get_query_execution(self.sagemaker_session, self._current_query_execution_id)

    @_telemetry_emitter(Feature.FEATURE_STORE, "AthenaQuery.as_dataframe")
    def as_dataframe(self, **kwargs) -> DataFrame:
        """Download the result of the current query and load it into a DataFrame.

        Args:
            **kwargs (object): key arguments used for the method pandas.read_csv to be able to
                    have a better tuning on data. For more info read:
                    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html

        Returns:
            A pandas DataFrame contains the query result.
        """
        state = self.get_query_execution()["QueryExecution"]["Status"]["State"]
        if state != "SUCCEEDED":
            if state in ("QUEUED", "RUNNING"):
                raise RuntimeError(f"Query {self._current_query_execution_id} still executing.")
            raise RuntimeError(f"Query {self._current_query_execution_id} failed.")

        output_file = os.path.join(tempfile.gettempdir(), f"{self._current_query_execution_id}.csv")
        download_athena_query_result(
            session=self.sagemaker_session,
            bucket=self._result_bucket,
            prefix=self._result_file_prefix,
            query_execution_id=self._current_query_execution_id,
            filename=output_file,
        )
        kwargs.pop("delimiter", None)
        return pd.read_csv(output_file, delimiter=",", **kwargs)

