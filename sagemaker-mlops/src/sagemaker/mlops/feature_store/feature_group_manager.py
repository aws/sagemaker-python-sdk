# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""FeatureGroup with Lake Formation support."""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import botocore.exceptions

from sagemaker.core.resources import FeatureGroup
from sagemaker.core.resources import Base
from sagemaker.core.shapes import (
    FeatureDefinition,
    OfflineStoreConfig,
    OnlineStoreConfig,
    Tag,
    ThroughputConfig,
)
from sagemaker.core.shapes import Unassigned
from sagemaker.core.helper.pipeline_variable import StrPipeVar
from sagemaker.core.s3.utils import parse_s3_url
from sagemaker.core.common_utils import aws_partition
from boto3 import Session


logger = logging.getLogger(__name__)


class LakeFormationConfig(Base):
    """Configuration for Lake Formation governance on Feature Group offline stores.

    Attributes:
        enabled: If True, enables Lake Formation governance for the offline store.
            Requires offline_store_config and role_arn to be set on the Feature Group.
        use_service_linked_role: Whether to use the Lake Formation service-linked role
            for S3 registration. If True, Lake Formation uses its service-linked role.
            If False, registration_role_arn must be provided. Default is True.
        registration_role_arn: IAM role ARN to use for S3 registration with Lake Formation.
            Required when use_service_linked_role is False. This can be different from the
            Feature Group's execution role.
    """

    enabled: bool = False
    use_service_linked_role: bool = True
    registration_role_arn: Optional[str] = None
    disable_hybrid_access_mode = False


class FeatureGroupManager(FeatureGroup):
    """FeatureGroup with extended management capabilities."""

    @staticmethod
    def _s3_uri_to_arn(s3_uri: str, region: Optional[str] = None) -> str:
        """
        Convert S3 URI to S3 ARN format for Lake Formation.

        Args:
            s3_uri: S3 URI in format s3://bucket/path or already an ARN
            region: AWS region name (e.g., 'us-west-2'). Used to determine the correct
                partition for the ARN. If not provided, defaults to 'aws' partition.

        Returns:
            S3 ARN in format arn:{partition}:s3:::bucket/path

        Note:
            This format is specifically used for Lake Formation resource registration.
            The triple colon (:::) after 's3' is correct - S3 ARNs don't include
            region or account ID fields.
        """
        if s3_uri.startswith("arn:"):
            return s3_uri
        
        # Determine partition based on region
        partition = aws_partition(region) if region else "aws"
        
        bucket, key = parse_s3_url(s3_uri)
        # Reconstruct as ARN - key may be empty string
        s3_path = f"{bucket}/{key}" if key else bucket
        return f"arn:{partition}:s3:::{s3_path}"

    @staticmethod
    def _extract_account_id_from_arn(arn: str) -> str:
        """
        Extract AWS account ID from an ARN.

        Args:
            arn: AWS ARN in format arn:aws:service:region:account:resource

        Returns:
            AWS account ID (the 5th colon-separated field)

        Raises:
            ValueError: If ARN format is invalid (fewer than 5 colon-separated parts)
        """
        parts = arn.split(":")
        if len(parts) < 5:
            raise ValueError(f"Invalid ARN format: {arn}")
        return parts[4]

    @staticmethod
    def _get_lake_formation_service_linked_role_arn(
        account_id: str, region: Optional[str] = None
    ) -> str:
        """
        Generate the Lake Formation service-linked role ARN for an account.

        Args:
            account_id: AWS account ID
            region: AWS region name (e.g., 'us-west-2'). Used to determine the correct
                partition for the ARN. If not provided, defaults to 'aws' partition.

        Returns:
            Lake Formation service-linked role ARN in format:
            arn:{partition}:iam::{account}:role/aws-service-role/lakeformation.amazonaws.com/
            AWSServiceRoleForLakeFormationDataAccess
        """
        partition = aws_partition(region) if region else "aws"
        return (
            f"arn:{partition}:iam::{account_id}:role/aws-service-role/"
            f"lakeformation.amazonaws.com/AWSServiceRoleForLakeFormationDataAccess"
        )

    def _get_lake_formation_client(
        self,
        session: Optional[Session] = None,
        region: Optional[str] = None,
    ):
        """
        Get a Lake Formation client.

        Args:
            session: Boto3 session. If not provided, a new session will be created.
            region: AWS region name.

        Returns:
            A boto3 Lake Formation client.
        """
        boto_session = session or Session()
        return boto_session.client("lakeformation", region_name=region)

    def _get_s3_client(
        self,
        session: Optional[Session] = None,
        region: Optional[str] = None,
    ):
        """
        Get an S3 client.

        Args:
            session: Boto3 session. If not provided, a new session will be created.
            region: AWS region name.

        Returns:
            A boto3 S3 client.
        """
        boto_session = session or Session()
        return boto_session.client("s3", region_name=region)

    def _get_cloudtrail_client(
        self,
        session: Optional[Session] = None,
        region: Optional[str] = None,
    ):
        """Get a CloudTrail client.

        Args:
            session: Boto3 session. If not provided, a new session will be created.
            region: AWS region name.

        Returns:
            A boto3 CloudTrail client.
        """
        boto_session = session or Session()
        return boto_session.client("cloudtrail", region_name=region)

    def _get_athena_client(
        self,
        session: Optional[Session] = None,
        region: Optional[str] = None,
    ):
        """Get an Athena client.

        Args:
            session: Boto3 session. If not provided, a new session will be created.
            region: AWS region name.

        Returns:
            A boto3 Athena client.
        """
        boto_session = session or Session()
        return boto_session.client("athena", region_name=region)

    def _get_sagemaker_client(
        self,
        session: Optional[Session] = None,
        region: Optional[str] = None,
    ):
        """Get a SageMaker client.

        Args:
            session: Boto3 session. If not provided, a new session will be created.
            region: AWS region name.

        Returns:
            A boto3 SageMaker client.
        """
        boto_session = session or Session()
        return boto_session.client("sagemaker", region_name=region)
    def _get_glue_client(
        self,
        session: Optional[Session] = None,
        region: Optional[str] = None,
    ):
        """Get a Glue client.

        Args:
            session: Boto3 session. If not provided, a new session will be created.
            region: AWS region name.

        Returns:
            A boto3 Glue client.
        """
        boto_session = session or Session()
        return boto_session.client("glue", region_name=region)

    def _query_glue_table_accessors(
        self,
        database_name: str,
        table_name: str,
        session: Optional[Session] = None,
        region: Optional[str] = None,
        lookback_days: int = 30,
    ) -> dict:
        """Query CloudTrail for IAM principals that accessed a Glue table.

        Queries CloudTrail LookupEvents filtered by EventSource=glue.amazonaws.com,
        then client-side filters for events targeting the specified table. Extracts
        distinct principal ARNs with their last access time.

        Args:
            database_name: Glue database name.
            table_name: Glue table name.
            session: Boto3 session.
            region: AWS region name.
            lookback_days: Number of days to look back in CloudTrail. Default 30.

        Returns:
            Dict with keys:
                - accessors: list of dicts with principal_arn and last_access_time
                - warnings: list of warning strings
        """
        warnings = []
        try:
            client = self._get_cloudtrail_client(session=session, region=region)
        except Exception as e:
            logger.warning(f"Failed to create CloudTrail client: {e}")
            return {"accessors": [], "warnings": [f"Failed to create CloudTrail client: {e}"]}

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=lookback_days)

        # principal_arn -> latest access time
        principals: dict[str, str] = {}
        events_scanned = 0
        max_events = 1000

        try:
            paginator_kwargs = {
                "LookupAttributes": [
                    {"AttributeKey": "EventSource", "AttributeValue": "glue.amazonaws.com"}
                ],
                "StartTime": start_time,
                "EndTime": end_time,
                "MaxResults": 50,
            }

            next_token = None
            glue_table_read_events = {
                "GetTable", "GetTables", "GetTableVersion", "GetTableVersions",
                "BatchGetTable", "GetPartition", "GetPartitions",
                "BatchGetPartition", "SearchTables",
            }
            while events_scanned < max_events:
                kwargs = dict(paginator_kwargs)
                if next_token:
                    kwargs["NextToken"] = next_token

                response = client.lookup_events(**kwargs)

                for event in response.get("Events", []):
                    events_scanned += 1
                    if events_scanned > max_events:
                        break

                    # Client-side filter: check if event references our table
                    # Only consider read-access events on the table itself
                    event_name = event.get("EventName", "")
                    if event_name not in glue_table_read_events:
                        continue

                    resources = event.get("Resources", [])
                    matches_table = False
                    for resource in resources:
                        resource_name = resource.get("ResourceName", "")
                        # Exact match on table name or database/table path
                        if resource_name == table_name:
                            matches_table = True
                            break
                        if resource_name == f"{database_name}/{table_name}":
                            matches_table = True
                            break

                    if not matches_table:
                        # Check requestParameters for exact database + table match
                        cloud_trail_event = event.get("CloudTrailEvent", "")
                        if cloud_trail_event:
                            try:
                                event_detail = json.loads(cloud_trail_event)
                                req_params = event_detail.get("requestParameters", {})
                                req_db = req_params.get("databaseName", "")
                                req_tbl = (
                                    req_params.get("tableName", "")
                                    or req_params.get("name", "")
                                )
                                if req_db == database_name and req_tbl == table_name:
                                    matches_table = True
                            except (json.JSONDecodeError, AttributeError):
                                pass

                    if matches_table:
                        # Prefer the full ARN from CloudTrailEvent.userIdentity
                        # because event["Username"] for AssumedRole events is just
                        # the session name (e.g. "SageMaker..."), not a real ARN.
                        principal_arn = ""
                        try:
                            event_detail = json.loads(event.get("CloudTrailEvent", "{}"))
                            user_identity = event_detail.get("userIdentity", {})
                            principal_arn = user_identity.get("arn", "")
                        except (json.JSONDecodeError, AttributeError):
                            pass

                        if not principal_arn:
                            principal_arn = event.get("Username", "")

                        if not principal_arn:
                            continue

                        event_time = event.get("EventTime", "")
                        event_time_str = (
                            event_time.isoformat()
                            if hasattr(event_time, "isoformat")
                            else str(event_time)
                        )
                        # Keep the latest access time per principal
                        if principal_arn not in principals:
                            principals[principal_arn] = event_time_str
                        else:
                            if event_time_str > principals[principal_arn]:
                                principals[principal_arn] = event_time_str

                next_token = response.get("NextToken")
                if not next_token:
                    break

        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "AccessDeniedException":
                logger.warning(
                    "CloudTrail access denied. Results may be incomplete. "
                    "Ensure the caller has cloudtrail:LookupEvents permission."
                )
                warnings.append("CloudTrail access denied, results may be incomplete")
            else:
                logger.warning(f"CloudTrail query failed: {e}")
                warnings.append(f"CloudTrail query failed: {e}")

        accessors = [
            {"principal_arn": arn, "last_access_time": time}
            for arn, time in principals.items()
        ]

        return {"accessors": accessors, "warnings": warnings}

    def _query_athena_query_principals(
        self,
        database_name: str,
        table_name: str,
        session: Optional[Session] = None,
        region: Optional[str] = None,
        lookback_days: int = 30,
    ) -> dict:
        """Query CloudTrail + Athena for recent queries that reference a specific table.

        Uses CloudTrail to discover StartQueryExecution events from all principals
        (not just the caller), then uses batch_get_query_execution to retrieve
        query details and filter by table reference.

        Args:
            database_name: Glue database name.
            table_name: Glue table name.
            session: Boto3 session.
            region: AWS region name.
            lookback_days: Number of days to look back. Default 30.

        Returns:
            Dict with keys:
                - principals: list of dicts with principal_arn, query_execution_id,
                  query, submission_time, state
                - running_queries: list of dicts with query_execution_id, query, state
                - warnings: list of warning strings
        """
        warnings = []

        # --- Phase 1: CloudTrail discovery ---
        try:
            ct_client = self._get_cloudtrail_client(session=session, region=region)
        except Exception as e:
            logger.warning(f"Failed to create CloudTrail client: {e}")
            return {
                "principals": [],
                "running_queries": [],
                "warnings": [f"Failed to create CloudTrail client: {e}"],
            }

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=lookback_days)

        # query_execution_id -> principal_arn
        candidate_queries: dict[str, str] = {}
        events_scanned = 0
        max_events_scanned = 10000
        max_candidates = 500

        try:
            paginator_kwargs = {
                "LookupAttributes": [
                    {"AttributeKey": "EventSource", "AttributeValue": "athena.amazonaws.com"}
                ],
                "StartTime": start_time,
                "EndTime": end_time,
                "MaxResults": 50,
            }

            next_token = None
            while events_scanned < max_events_scanned and len(candidate_queries) < max_candidates:
                kwargs = dict(paginator_kwargs)
                if next_token:
                    kwargs["NextToken"] = next_token

                response = ct_client.lookup_events(**kwargs)

                for event in response.get("Events", []):
                    events_scanned += 1
                    if events_scanned > max_events_scanned:
                        break
                    if len(candidate_queries) >= max_candidates:
                        break

                    if event.get("EventName") != "StartQueryExecution":
                        continue

                    cloud_trail_event_str = event.get("CloudTrailEvent", "{}")
                    try:
                        event_detail = json.loads(cloud_trail_event_str)
                    except (json.JSONDecodeError, AttributeError):
                        continue

                    resp_elements = event_detail.get("responseElements", {})
                    query_id = resp_elements.get("queryExecutionId", "")
                    if not query_id:
                        continue

                    user_identity = event_detail.get("userIdentity", {})
                    principal_arn = user_identity.get("arn", "")
                    if not principal_arn:
                        principal_arn = event.get("Username", "")
                    if not principal_arn:
                        continue

                    candidate_queries[query_id] = principal_arn

                next_token = response.get("NextToken")
                if not next_token:
                    break

        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "AccessDeniedException":
                logger.warning(
                    "CloudTrail access denied. Athena query principal results may be incomplete."
                )
                warnings.append("CloudTrail access denied, Athena query principal results may be incomplete")
            else:
                logger.warning(f"CloudTrail query for Athena events failed: {e}")
                warnings.append(f"CloudTrail query for Athena events failed: {e}")

        if not candidate_queries:
            return {"principals": [], "running_queries": [], "warnings": warnings}

        # --- Phase 2: Athena query detail lookup ---
        try:
            athena_client = self._get_athena_client(session=session, region=region)
        except Exception as e:
            logger.warning(f"Failed to create Athena client: {e}")
            warnings.append(f"Failed to create Athena client: {e}")
            return {"principals": [], "running_queries": [], "warnings": warnings}

        principals = []
        running_queries = []
        query_ids = list(candidate_queries.keys())

        for i in range(0, len(query_ids), 50):
            batch = query_ids[i : i + 50]
            try:
                response = athena_client.batch_get_query_execution(QueryExecutionIds=batch)
            except botocore.exceptions.ClientError as e:
                logger.warning(f"Athena batch_get_query_execution failed: {e}")
                warnings.append(f"Athena batch_get_query_execution failed: {e}")
                continue

            for qe in response.get("QueryExecutions", []):
                query_sql = qe.get("Query", "")
                status = qe.get("Status", {})
                state = status.get("State", "")
                submission_time = status.get("SubmissionDateTime")
                qe_id = qe.get("QueryExecutionId", "")

                # Match if the table name appears anywhere in the SQL
                # (handles quoted identifiers like "db"."table")
                if table_name.lower() not in query_sql.lower():
                    continue

                if state in ("QUEUED", "RUNNING"):
                    running_queries.append({
                        "query_execution_id": qe_id,
                        "query": query_sql[:200],
                        "state": state,
                    })

                submission_str = (
                    submission_time.isoformat()
                    if hasattr(submission_time, "isoformat")
                    else str(submission_time)
                ) if submission_time else ""

                principals.append({
                    "principal_arn": candidate_queries.get(qe_id, ""),
                    "query_execution_id": qe_id,
                    "query": query_sql[:200],
                    "submission_time": submission_str,
                    "state": state,
                })

        return {"principals": principals, "running_queries": running_queries, "warnings": warnings}

    @staticmethod
    def _job_references_s3_path(detail: dict, s3_path: str, job_type: str) -> bool:
        """Check if a described SageMaker job references the given S3 path.

        Args:
            detail: Response from describe_training_job / describe_processing_job /
                describe_transform_job.
            s3_path: S3 URI prefix of the feature group offline store.
            job_type: One of TrainingJob, ProcessingJob, TransformJob.

        Returns:
            True if any input or output channel references s3_path.
        """
        uris: list[str] = []

        if job_type == "TrainingJob":
            for channel in detail.get("InputDataConfig", []):
                ds = channel.get("DataSource", {}).get("S3DataSource", {})
                if ds.get("S3Uri"):
                    uris.append(ds["S3Uri"])
            output_path = detail.get("OutputDataConfig", {}).get("S3OutputPath", "")
            if output_path:
                uris.append(output_path)

        elif job_type == "ProcessingJob":
            for inp in detail.get("ProcessingInputs", []):
                s3_input = inp.get("S3Input", {})
                if s3_input.get("S3Uri"):
                    uris.append(s3_input["S3Uri"])
            for out in detail.get("ProcessingOutputConfig", {}).get("Outputs", []):
                s3_output = out.get("S3Output", {})
                if s3_output.get("S3Uri"):
                    uris.append(s3_output["S3Uri"])

        elif job_type == "TransformJob":
            transform_input = detail.get("TransformInput", {}).get("DataSource", {}).get("S3DataSource", {})
            if transform_input.get("S3Uri"):
                uris.append(transform_input["S3Uri"])
            transform_output = detail.get("TransformOutput", {}).get("S3OutputPath", "")
            if transform_output:
                uris.append(transform_output)

        normalized = s3_path.rstrip("/")
        return any(uri.rstrip("/").startswith(normalized) or normalized.startswith(uri.rstrip("/")) for uri in uris)

    def _query_sagemaker_jobs_for_s3_path(
        self,
        s3_path: str,
        session: Optional[Session] = None,
        region: Optional[str] = None,
        lookback_days: int = 30,
    ) -> dict:
        """Query SageMaker for recent jobs that reference a specific S3 path.

        Lists recent training, processing, and transform jobs within the lookback
        window, describes each, and returns only those whose input/output channels
        reference the given S3 path.

        Args:
            s3_path: S3 URI prefix of the feature group offline store.
            session: Boto3 session.
            region: AWS region name.
            lookback_days: Number of days to look back. Default 30.

        Returns:
            Dict with keys:
                - jobs: list of dicts with job_type, job_name, role_arn, and s3_uris
                - warnings: list of warning strings
        """
        warnings: list[str] = []
        try:
            client = self._get_sagemaker_client(session=session, region=region)
        except Exception as e:
            logger.warning(f"Failed to create SageMaker client: {e}")
            return {"jobs": [], "warnings": [f"Failed to create SageMaker client: {e}"]}

        creation_time_after = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        matching_jobs: list[dict] = []

        job_listers = [
            ("TrainingJob", "list_training_jobs", "TrainingJobSummaries", "TrainingJobName", "describe_training_job"),
            ("ProcessingJob", "list_processing_jobs", "ProcessingJobSummaries", "ProcessingJobName", "describe_processing_job"),
            ("TransformJob", "list_transform_jobs", "TransformJobSummaries", "TransformJobName", "describe_transform_job"),
        ]

        for job_type, api_method, summary_key, name_key, describe_method in job_listers:
            try:
                paginator = client.get_paginator(api_method)
                page_kwargs = {
                    "CreationTimeAfter": creation_time_after,
                    "SortBy": "CreationTime",
                    "SortOrder": "Descending",
                }

                for page in paginator.paginate(**page_kwargs):
                    for job_summary in page.get(summary_key, []):
                        job_name = job_summary.get(name_key, "")
                        try:
                            detail = getattr(client, describe_method)(**{name_key: job_name})
                        except Exception:
                            continue

                        if self._job_references_s3_path(detail, s3_path, job_type):
                            matching_jobs.append({
                                "job_type": job_type,
                                "job_name": job_name,
                                "role_arn": detail.get("RoleArn"),
                                "status": detail.get(f"{job_type}Status", "Unknown"),
                            })

            except botocore.exceptions.ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code == "AccessDeniedException":
                    logger.warning(
                        f"SageMaker {api_method} access denied. "
                        "Results may be incomplete."
                    )
                    warnings.append(
                        f"SageMaker {api_method} access denied, results may be incomplete"
                    )
                else:
                    logger.warning(f"SageMaker {api_method} failed: {e}")
                    warnings.append(f"SageMaker {api_method} failed: {e}")

        return {"jobs": matching_jobs, "warnings": warnings}

    def _query_running_jobs(
        self,
        s3_path: str,
        session: Optional[Session] = None,
        region: Optional[str] = None,
    ) -> dict:
        """Query SageMaker for currently running jobs that reference a specific S3 path.

        Lists in-progress training, processing, and transform jobs, describes each,
        and returns only those whose input/output channels reference the given S3 path.

        Args:
            s3_path: S3 URI prefix of the feature group offline store.
            session: Boto3 session.
            region: AWS region name.

        Returns:
            Dict with keys:
                - running_jobs: list of dicts with service, job_type, job_name,
                  status, and role_arn
                - warnings: list of warning strings
        """
        warnings: list[str] = []
        try:
            client = self._get_sagemaker_client(session=session, region=region)
        except Exception as e:
            logger.warning(f"Failed to create SageMaker client: {e}")
            return {"running_jobs": [], "warnings": [f"Failed to create SageMaker client: {e}"]}

        running_jobs: list[dict] = []

        job_listers = [
            ("TrainingJob", "list_training_jobs", "TrainingJobSummaries", "TrainingJobName", "describe_training_job"),
            ("ProcessingJob", "list_processing_jobs", "ProcessingJobSummaries", "ProcessingJobName", "describe_processing_job"),
            ("TransformJob", "list_transform_jobs", "TransformJobSummaries", "TransformJobName", "describe_transform_job"),
        ]

        for job_type, api_method, summary_key, name_key, describe_method in job_listers:
            try:
                paginator = client.get_paginator(api_method)
                for page in paginator.paginate(StatusEquals="InProgress"):
                    for job_summary in page.get(summary_key, []):
                        job_name = job_summary.get(name_key, "")
                        try:
                            detail = getattr(client, describe_method)(**{name_key: job_name})
                        except Exception:
                            continue

                        if self._job_references_s3_path(detail, s3_path, job_type):
                            running_jobs.append({
                                "service": "SageMaker",
                                "job_type": job_type,
                                "job_name": job_name,
                                "status": "InProgress",
                                "role_arn": detail.get("RoleArn"),
                            })
            except botocore.exceptions.ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code == "AccessDeniedException":
                    logger.warning(
                        f"SageMaker {api_method} access denied. "
                        "Results may be incomplete."
                    )
                    warnings.append(
                        f"SageMaker {api_method} access denied, results may be incomplete"
                    )
                else:
                    logger.warning(f"SageMaker {api_method} failed: {e}")
                    warnings.append(f"SageMaker {api_method} failed: {e}")

        return {"running_jobs": running_jobs, "warnings": warnings}

    def _query_glue_etl_jobs(
        self,
        database_name: str,
        table_name: str,
        session: Optional[Session] = None,
        region: Optional[str] = None,
    ) -> dict:
        """Query Glue for visual ETL jobs that reference a specific table.

        Paginates through all Glue jobs, checks visual-mode jobs
        (those with CodeGenConfigurationNodes) for references to the
        specified database and table, and checks for running job runs.

        Args:
            database_name: Glue database name.
            table_name: Glue table name.
            session: Boto3 session.
            region: AWS region name.

        Returns:
            Dict with keys:
                - jobs: list of dicts with job_name and role
                - running_job_runs: list of dicts with job_name, run_id, and state
                - warnings: list of warning strings
        """
        warnings = []
        try:
            client = self._get_glue_client(session=session, region=region)
        except Exception as e:
            logger.warning(f"Failed to create Glue client: {e}")
            return {
                "jobs": [],
                "running_job_runs": [],
                "warnings": [f"Failed to create Glue client: {e}"],
            }

        matching_jobs = []
        running_job_runs = []

        catalog_node_keys = (
            "CatalogSource", "CatalogTarget",
            "GovernedCatalogSource", "GovernedCatalogTarget",
            "S3CatalogSource", "S3CatalogTarget",
        )

        try:
            paginator = client.get_paginator("get_jobs")
            for page in paginator.paginate():
                for job in page.get("Jobs", []):
                    nodes = job.get("CodeGenConfigurationNodes")
                    if not nodes:
                        continue

                    job_name = job.get("Name", "")
                    matched = False

                    for node_value in nodes.values():
                        for catalog_key in catalog_node_keys:
                            if catalog_key not in node_value:
                                continue
                            node_config = node_value[catalog_key]
                            db = node_config.get("Database") or node_config.get("DatabaseName")
                            tbl = node_config.get("Table")
                            tables = node_config.get("Tables")
                            if db == database_name and (
                                tbl == table_name
                                or (isinstance(tables, list) and table_name in tables)
                            ):
                                matched = True
                                break
                        if matched:
                            break

                    if matched:
                        matching_jobs.append({
                            "job_name": job_name,
                            "role": job.get("Role", ""),
                        })

                        try:
                            runs_resp = client.get_job_runs(
                                JobName=job_name, MaxResults=20
                            )
                            for run in runs_resp.get("JobRuns", []):
                                if run.get("JobRunState") in (
                                    "STARTING", "RUNNING", "STOPPING", "WAITING"
                                ):
                                    running_job_runs.append({
                                        "job_name": job_name,
                                        "run_id": run.get("Id", ""),
                                        "state": run.get("JobRunState", ""),
                                    })
                        except Exception as e:
                            logger.warning(f"Failed to get job runs for {job_name}: {e}")

        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "AccessDeniedException":
                logger.warning("Glue get_jobs access denied. Results may be incomplete.")
                warnings.append("Glue get_jobs access denied, results may be incomplete")
            else:
                logger.warning(f"Glue get_jobs failed: {e}")
                warnings.append(f"Glue get_jobs failed: {e}")

        return {
            "jobs": matching_jobs,
            "running_job_runs": running_job_runs,
            "warnings": warnings,
        }

    def _run_all_audit_queries(
        self,
        database_name: str,
        table_name: str,
        s3_path: str,
        session: Optional[Session] = None,
        region: Optional[str] = None,
        lookback_days: int = 30,
    ) -> dict:
        """Run all audit queries and aggregate results.

        Args:
            database_name: Glue database name.
            table_name: Glue table name.
            s3_path: S3 URI of the offline store.
            session: Boto3 session.
            region: AWS region name.
            lookback_days: Number of days to look back. Default 30.

        Returns:
            Dict with keys: glue_table_accessors, sagemaker_jobs,
            athena_query_principals, athena_running_queries, glue_etl_jobs,
            glue_running_job_runs, sagemaker_running_jobs, glue_database,
            glue_table, s3_path, warnings.
        """
        warnings = []

        cloudtrail_result = self._query_glue_table_accessors(
            database_name=database_name,
            table_name=table_name,
            session=session,
            region=region,
            lookback_days=lookback_days,
        )
        warnings.extend(cloudtrail_result.get("warnings", []))

        sagemaker_result = self._query_sagemaker_jobs_for_s3_path(
            s3_path=s3_path,
            session=session,
            region=region,
            lookback_days=lookback_days,
        )
        warnings.extend(sagemaker_result.get("warnings", []))

        athena_result = self._query_athena_query_principals(
            database_name=database_name,
            table_name=table_name,
            session=session,
            region=region,
            lookback_days=lookback_days,
        )
        warnings.extend(athena_result.get("warnings", []))

        glue_result = self._query_glue_etl_jobs(
            database_name=database_name,
            table_name=table_name,
            session=session,
            region=region,
        )
        warnings.extend(glue_result.get("warnings", []))

        running_result = self._query_running_jobs(
            s3_path=s3_path,
            session=session,
            region=region,
        )
        warnings.extend(running_result.get("warnings", []))

        return {
            "glue_table_accessors": cloudtrail_result.get("accessors", []),
            "sagemaker_jobs": sagemaker_result.get("jobs", []),
            "athena_query_principals": athena_result.get("principals", []),
            "athena_running_queries": athena_result.get("running_queries", []),
            "glue_etl_jobs": glue_result.get("jobs", []),
            "glue_running_job_runs": glue_result.get("running_job_runs", []),
            "sagemaker_running_jobs": running_result.get("running_jobs", []),
            "glue_database": database_name,
            "glue_table": table_name,
            "s3_path": s3_path,
            "warnings": warnings,
        }

    def _format_risk_report(self, audit_results: dict) -> str:
        """Format audit results into a human-readable risk report.

        Args:
            audit_results: Dict returned by _run_all_audit_queries.

        Returns:
            Formatted report string.
        """
        lines = [
            "=== Lake Formation Impact Report ===",
            f"Feature Group : {self.feature_group_name}",
            f"Glue table    : {audit_results.get('glue_table', '')}",
            f"S3 path       : {audit_results.get('s3_path', '')}",
        ]

        accessors = audit_results.get("glue_table_accessors", [])
        if accessors:
            lines.append("")
            lines.append("Glue table accessors (via CloudTrail):")
            for entry in accessors:
                lines.append(f"  - {entry.get('principal_arn', 'unknown')}")

        jobs = audit_results.get("sagemaker_jobs", [])
        if jobs:
            lines.append("")
            lines.append("SageMaker jobs referencing this S3 path:")
            for entry in jobs:
                role = entry.get("role_arn", "unknown")
                lines.append(
                    f"  - {entry.get('job_type', '')}: {entry.get('job_name', 'unknown')} "
                    f"(role: {role}, status: {entry.get('status', 'unknown')})"
                )

        athena_principals = audit_results.get("athena_query_principals", [])
        if athena_principals:
            lines.append("")
            lines.append("Athena query principals:")
            for entry in athena_principals:
                principal = entry.get("principal_arn", "")
                prefix = f"{principal} — " if principal else ""
                lines.append(
                    f"  - {prefix}{entry.get('query_execution_id', 'unknown')}: "
                    f"{entry.get('query', '')[:100]}"
                )

        glue_jobs = audit_results.get("glue_etl_jobs", [])
        if glue_jobs:
            lines.append("")
            lines.append("Glue ETL jobs:")
            for entry in glue_jobs:
                lines.append(f"  - {entry.get('job_name', 'unknown')}")

        running = []
        running.extend(audit_results.get("sagemaker_running_jobs", []))
        running.extend(audit_results.get("athena_running_queries", []))
        running.extend(audit_results.get("glue_running_job_runs", []))
        if running:
            lines.append("")
            lines.append("[!] Running jobs/queries:")
            for entry in running:
                name = entry.get("job_name", entry.get("query_execution_id", "unknown"))
                state = entry.get("status", entry.get("state", ""))
                lines.append(f"  - {name} ({state})")

        warn = audit_results.get("warnings", [])
        if warn:
            lines.append("")
            lines.append("Warnings:")
            for w in warn:
                lines.append(f"  - {w}")

        lines.append("")
        lines.append("[!] WARNING - Limitations:")
        lines.append("  - CloudTrail has 15-minute delivery delay")
        lines.append("  - SageMaker job scan uses S3 path matching on input/output channels")
        lines.append("  - Athena query detection uses CloudTrail + SQL text matching (may have false positives)")
        lines.append("  - Glue ETL detection only covers VISUAL mode jobs")

        return "\n".join(lines)

    def audit_lake_formation_impact(
        self,
        session: Optional[Session] = None,
        region: Optional[str] = None,
        lookback_days: int = 30,
    ) -> dict:
        """Audit the impact of enabling Lake Formation on this Feature Group.

        Runs all audit queries and prints a formatted risk report.

        Args:
            session: Boto3 session.
            region: AWS region name.
            lookback_days: Number of days to look back. Default 30.

        Returns:
            Dict with audit results from _run_all_audit_queries.

        Raises:
            ValueError: If Feature Group is not in 'Created' status or has no offline store.
        """
        self.refresh()

        if self.feature_group_status not in ["Created"]:
            raise ValueError(
                f"Feature Group '{self.feature_group_name}' must be in 'Created' status. "
                f"Current status: '{self.feature_group_status}'."
            )

        if self.offline_store_config is None or isinstance(self.offline_store_config, Unassigned):
            raise ValueError(
                f"Feature Group '{self.feature_group_name}' does not have an offline store configured."
            )

        data_catalog_config = self.offline_store_config.data_catalog_config
        s3_config = self.offline_store_config.s3_storage_config
        database_name = str(data_catalog_config.database)
        table_name = str(data_catalog_config.table_name)
        s3_path = str(s3_config.resolved_output_s3_uri)

        audit_results = self._run_all_audit_queries(
            database_name=database_name,
            table_name=table_name,
            s3_path=s3_path,
            session=session,
            region=region,
            lookback_days=lookback_days,
        )

        print(self._format_risk_report(audit_results))

        return audit_results

    def _register_s3_with_lake_formation(
        self,
        s3_location: str,
        session: Optional[Session] = None,
        region: Optional[str] = None,
        use_service_linked_role: bool = True,
        role_arn: Optional[str] = None,
    ) -> bool:
        """
        Register an S3 location with Lake Formation.

        Args:
            s3_location: S3 URI or ARN to register.
            session: Boto3 session.
            region: AWS region. If not provided, will be inferred from the session.
            use_service_linked_role: Whether to use the Lake Formation service-linked role.
                If True, Lake Formation uses its service-linked role for registration.
                If False, role_arn must be provided.
            role_arn: IAM role ARN to use for registration. Required when
                use_service_linked_role is False.

        Returns:
            True if registration succeeded or location already registered.

        Raises:
            ValueError: If use_service_linked_role is False but role_arn is not provided.
            ClientError: If registration fails for unexpected reasons.
        """
        if not use_service_linked_role and not role_arn:
            raise ValueError("role_arn must be provided when use_service_linked_role is False")

        # Get region from session if not provided
        if region is None and session is not None:
            region = session.region_name

        client = self._get_lake_formation_client(session, region)
        resource_arn = self._s3_uri_to_arn(s3_location, region)

        try:
            register_params = {"ResourceArn": resource_arn, "WithFederation": True}

            if use_service_linked_role:
                register_params["UseServiceLinkedRole"] = True
            else:
                register_params["RoleArn"] = role_arn
            # print(register_params)
            client.register_resource(**register_params)
            logger.info(f"Successfully registered S3 location: {resource_arn}")
            return True
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "AlreadyExistsException":
                logger.info(f"S3 location already registered: {resource_arn}")
                return True
            raise

    def _revoke_iam_allowed_principal(
        self,
        database_name: str,
        table_name: str,
        session: Optional[Session] = None,
        region: Optional[str] = None,
    ) -> bool:
        """
        Revoke IAMAllowedPrincipal permissions from a Glue table.

        Checks for existing IAMAllowedPrincipal permissions via list_permissions
        before attempting revocation. If no permissions exist, skips the revoke call.

        Args:
            database_name: Glue database name.
            table_name: Glue table name.
            session: Boto3 session.
            region: AWS region. If not provided, will be inferred from the session.

        Returns:
            True if revocation succeeded or permissions didn't exist.

        Raises:
            ClientError: If list_permissions or revoke_permissions fails.
        """
        # Get region from session if not provided
        if region is None and session is not None:
            region = session.region_name

        client = self._get_lake_formation_client(session, region)

        response = client.list_permissions(
            Principal={"DataLakePrincipalIdentifier": "IAM_ALLOWED_PRINCIPALS"},
            Resource={
                "Table": {
                    "DatabaseName": database_name,
                    "Name": table_name,
                }
            },
        )

        if not response.get("PrincipalResourcePermissions", []):
            logger.info(
                f"No IAMAllowedPrincipal permissions found on: {database_name}.{table_name}"
            )
            return True

        client.revoke_permissions(
            Principal={"DataLakePrincipalIdentifier": "IAM_ALLOWED_PRINCIPALS"},
            Resource={
                "Table": {
                    "DatabaseName": database_name,
                    "Name": table_name,
                }
            },
            Permissions=["ALL"],
        )
        logger.info(f"Disabled Lake Formation hybrid-access mode on table: {database_name}.{table_name}")
        return True

    def _grant_lake_formation_permissions(
        self,
        role_arn: str,
        database_name: str,
        table_name: str,
        session: Optional[Session] = None,
        region: Optional[str] = None,
    ) -> bool:
        """
        Grant permissions to a role on a Glue table via Lake Formation.

        Args:
            role_arn: IAM role ARN to grant permissions to.
            database_name: Glue database name.
            table_name: Glue table name.
            session: Boto3 session.
            region: AWS region. If not provided, will be inferred from the session.

        Returns:
            True if grant succeeded or permissions already exist.

        Raises:
            ClientError: If grant fails for unexpected reasons.
        """
        # Get region from session if not provided
        if region is None and session is not None:
            region = session.region_name

        client = self._get_lake_formation_client(session, region)
        permissions = ["SELECT", "INSERT", "DELETE", "DESCRIBE", "ALTER"]

        try:
            client.grant_permissions(
                Principal={"DataLakePrincipalIdentifier": role_arn},
                Resource={
                    "Table": {
                        "DatabaseName": database_name,
                        "Name": table_name,
                    }
                },
                Permissions=permissions,
                PermissionsWithGrantOption=[],
            )
            logger.info(f"Granted permissions to {role_arn} on table: {database_name}.{table_name}")
            return True
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "InvalidInputException":
                logger.info(
                    f"Permissions may already exist for {role_arn} on: {database_name}.{table_name}"
                )
                return True
            raise
    
    def _generate_s3_deny_statements(
        self,
        bucket_name: str,
        s3_prefix: str,
        lake_formation_role_arn: str,
        feature_store_role_arn: str,
        region: Optional[str] = None,
        caller_arn: Optional[str] = None,
    ) -> list:
        """
        Generate S3 deny statements for Lake Formation governance.

        These statements deny S3 access to the offline store data prefix except for
        the Lake Formation role, Feature Store execution role, and optionally the caller.

        Args:
            bucket_name: S3 bucket name.
            s3_prefix: S3 prefix path (without bucket name).
            lake_formation_role_arn: Lake Formation registration role ARN.
            feature_store_role_arn: Feature Store execution role ARN.
            region: AWS region name (e.g., 'us-west-2'). Used to determine the correct
                partition for S3 ARNs. If not provided, defaults to 'aws' partition.
            caller_arn: ARN of the caller to exempt from deny statements.

        Returns:
            List of statement dicts containing:
            1. Deny GetObject, PutObject, DeleteObject on data prefix except allowed principals
            2. Deny ListBucket on bucket with prefix condition except allowed principals
        """
        partition = aws_partition(region) if region else "aws"

        allowed_principals = [lake_formation_role_arn, feature_store_role_arn]
        if caller_arn:
            allowed_principals.append(caller_arn)

        sid_suffix = s3_prefix.rstrip("/").rsplit("/", 1)[-1]

        return [
            {
                "Sid": f"DenyFSObjectAccess_{sid_suffix}",
                "Effect": "Deny",
                "Principal": "*",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                "Resource": f"arn:{partition}:s3:::{bucket_name}/{s3_prefix}/*",
                "Condition": {
                    "StringNotEquals": {"aws:PrincipalArn": allowed_principals}
                },
            },
            {
                "Sid": f"DenyFSListAccess_{sid_suffix}",
                "Effect": "Deny",
                "Principal": "*",
                "Action": "s3:ListBucket",
                "Resource": f"arn:{partition}:s3:::{bucket_name}",
                "Condition": {
                    "StringLike": {"s3:prefix": f"{s3_prefix}/*"},
                    "StringNotEquals": {"aws:PrincipalArn": allowed_principals},
                },
            },
        ]
    
    def _apply_bucket_policy(
        self,
        bucket_name: str,
        s3_prefix: str,
        lake_formation_role_arn: str,
        feature_store_role_arn: str,
        session: Optional[Session] = None,
        region: Optional[str] = None,
    ) -> bool:
        """
        Apply S3 deny statements to the bucket policy for Lake Formation governance.

        Fetches the existing bucket policy, appends deny statements idempotently
        (by Sid), and puts the merged policy back.

        Args:
            bucket_name: S3 bucket name.
            s3_prefix: S3 prefix path (without bucket name).
            lake_formation_role_arn: Lake Formation registration role ARN.
            feature_store_role_arn: Feature Store execution role ARN.
            session: Boto3 session.
            region: AWS region name.

        Returns:
            True on success.

        Raises:
            ClientError: If S3 operations fail.
        """
        s3_client = self._get_s3_client(session, region)

        # Get existing bucket policy
        try:
            response = s3_client.get_bucket_policy(Bucket=bucket_name)
            policy = json.loads(response["Policy"])
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchBucketPolicy":
                policy = {"Version": "2012-10-17", "Statement": []}
            else:
                raise

        # Generate deny statements
        deny_statements = self._generate_s3_deny_statements(
            bucket_name=bucket_name,
            s3_prefix=s3_prefix,
            lake_formation_role_arn=lake_formation_role_arn,
            feature_store_role_arn=feature_store_role_arn,
            region=region,
        )

        # Filter out statements whose Sid already exists
        existing_sids = {stmt.get("Sid") for stmt in policy.get("Statement", [])}
        new_statements = [s for s in deny_statements if s.get("Sid") not in existing_sids]

        if not new_statements:
            logger.info("Bucket policy already contains the deny statements. Skipping update.")
            return True

        policy["Statement"].extend(new_statements)
        s3_client.put_bucket_policy(Bucket=bucket_name, Policy=json.dumps(policy))
        logger.info(f"Applied bucket policy: {json.dumps(policy, indent=2)}")
        return True

    @Base.add_validate_call
    def enable_lake_formation(
        self,
        session: Optional[Session] = None,
        region: Optional[str] = None,
        use_service_linked_role: bool = True,
        registration_role_arn: Optional[str] = None,
        wait_for_active: bool = False,
        lookback_days: int = 30,
        auto_confirm: bool = False,
    ) -> dict:
        """
        Enable Lake Formation governance for this Feature Group's offline store.

        This method:
        1. Optionally waits for Feature Group to reach 'Created' status
        2. Validates Feature Group status is 'Created'
        3. Registers the offline store S3 location as data lake location
        4. Grants the execution role permissions on the Glue table
        5. Runs access audit to discover affected principals
        6. Revokes IAMAllowedPrincipal permissions from the Glue table
        7. Applies S3 deny bucket policy

        After Phase 2, an access audit runs automatically. If any IAM principals
        are found that currently access the Glue table or any SageMaker execution
        roles are in active use, the user is prompted for confirmation before
        proceeding (unless auto_confirm=True). Use audit_lake_formation_impact()
        to preview affected principals before calling this method.

        The role ARN is automatically extracted from the Feature Group's configuration.
        Each phase depends on the success of the previous phase - if any phase fails,
        subsequent phases are not executed.

        Parameters:
            session: Boto3 session.
            region: Region name.
            use_service_linked_role: Whether to use the Lake Formation service-linked role
                for S3 registration. If True, Lake Formation uses its service-linked role.
                If False, registration_role_arn must be provided. Default is True.
            registration_role_arn: IAM role ARN to use for S3 registration with Lake Formation.
                Required when use_service_linked_role is False. This can be different from the
                Feature Group's execution role (role_arn)
            wait_for_active: If True, waits for the Feature Group to reach 'Created' status
                before enabling Lake Formation. Default is False.
            lookback_days: Number of days to look back when auditing CloudTrail
                and SageMaker job history. Default is 30.
            auto_confirm: If True, skip the interactive confirmation prompt when
                audit findings exist and proceed directly to Phase 3. Default is False.

        Returns:
            Dict with status of each Lake Formation operation:
            - s3_registration: bool
            - permissions_granted: bool
            - iam_principal_revoked: bool
            - bucket_policy_applied: bool

        Raises:
            ValueError: If the Feature Group has no offline store configured,
                if role_arn is not set on the Feature Group, if use_service_linked_role
                is False but registration_role_arn is not provided, or if the Feature
                Group is not in 'Created' status.
            ClientError: If Lake Formation operations fail.
            RuntimeError: If a phase fails and subsequent phases cannot proceed.
        """
        # Get region from session if not provided
        if region is None and session is not None:
            region = session.region_name

        # Wait for Created status if requested
        if wait_for_active:
            self.wait_for_status(target_status="Created")

        # Refresh to get latest state
        self.refresh()

        # Validate Feature Group status
        if self.feature_group_status not in ["Created"]:
            raise ValueError(
                f"Feature Group '{self.feature_group_name}' must be in 'Created' status "
                f"to enable Lake Formation. Current status: '{self.feature_group_status}'. "
                f"Use wait_for_active=True to automatically wait for the Feature Group to be ready."
            )

        # Validate offline store exists
        if self.offline_store_config is None or isinstance(self.offline_store_config, Unassigned):
            raise ValueError(
                f"Feature Group '{self.feature_group_name}' does not have an offline store configured. "
                "Lake Formation can only be enabled for Feature Groups with offline stores."
            )

        # Get role ARN from Feature Group config
        if self.role_arn is None or isinstance(self.role_arn, Unassigned):
            raise ValueError(
                f"Feature Group '{self.feature_group_name}' does not have a role_arn configured. "
                "Lake Formation requires a role ARN to grant permissions."
            )
        if not use_service_linked_role and registration_role_arn is None:
            raise ValueError(
                "Either 'use_service_linked_role' must be True or 'registration_role_arn' must be provided."
            )

        # Extract required configuration
        s3_config = self.offline_store_config.s3_storage_config
        if s3_config is None:
            raise ValueError("Offline store S3 configuration is missing")

        resolved_s3_uri = s3_config.resolved_output_s3_uri
        if resolved_s3_uri is None or isinstance(resolved_s3_uri, Unassigned):
            raise ValueError(
                "Resolved S3 URI not available. Ensure the Feature Group is in 'Created' status."
            )

        data_catalog_config = self.offline_store_config.data_catalog_config
        if data_catalog_config is None:
            raise ValueError("Data catalog configuration is missing from offline store config")

        database_name = data_catalog_config.database
        table_name = data_catalog_config.table_name

        if not database_name or not table_name:
            raise ValueError("Database name and table name are required from data catalog config")

        # Convert to str to handle PipelineVariable types
        resolved_s3_uri_str = str(resolved_s3_uri)
        database_name_str = str(database_name)
        table_name_str = str(table_name)
        role_arn_str = str(self.role_arn)

        # Determine the actual S3 location to register with Lake Formation.
        # For Iceberg tables, the Glue table's StorageDescriptor.Location is the parent
        # path (without /data suffix), while resolved_output_s3_uri always ends with /data.
        s3_location_to_register = resolved_s3_uri_str
        if (
            self.offline_store_config.table_format is not None
            and str(self.offline_store_config.table_format) == "Iceberg"
            and resolved_s3_uri_str.endswith("/data")
        ):
            s3_location_to_register = resolved_s3_uri_str[: -len("/data")]
            logger.info(
                f"Iceberg table format detected. Using parent S3 path for LF registration: "
                f"{s3_location_to_register}"
            )

        results = {
            "s3_registration": False,
            "iam_principal_revoked": False,
            "permissions_granted": False,
            "bucket_policy_applied": False,
        }

        # # --- Audit Gate ---
        # audit_results = self._run_all_audit_queries(
        #     database_name=database_name_str,
        #     table_name=table_name_str,
        #     s3_path=resolved_s3_uri_str,
        #     session=session,
        #     region=region,
        #     lookback_days=lookback_days,
        # )

        # has_findings = any([
        #     audit_results.get("glue_table_accessors"),
        #     audit_results.get("sagemaker_jobs"),
        #     audit_results.get("athena_query_principals"),
        #     audit_results.get("glue_etl_jobs"),
        #     audit_results.get("sagemaker_running_jobs"),
        #     audit_results.get("athena_running_queries"),
        #     audit_results.get("glue_running_job_runs"),
        # ])

        # if has_findings:
        #     print(self._format_risk_report(audit_results))
        #     if auto_confirm:
        #         logger.info("auto_confirm=True, proceeding without user confirmation")
        #     else:
        #         response = input("Proceed with enabling Lake Formation? [y/N]: ")
        #         if response.strip().lower() != "y":
        #             return {"aborted": True, "audit_results": audit_results, **results}


        # Execute Lake Formation setup with fail-fast behavior

        # Phase 1: Register S3 with Lake Formation
        try:
            results["s3_registration"] = self._register_s3_with_lake_formation(
                s3_location_to_register,
                session,
                region,
                use_service_linked_role=use_service_linked_role,
                role_arn=registration_role_arn,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to register S3 location with Lake Formation. "
                f"Subsequent phases skipped. Results: {results}. Error: {e}"
            ) from e

        if not results["s3_registration"]:
            raise RuntimeError(
                f"Failed to register S3 location with Lake Formation. "
                f"Subsequent phases skipped. Results: {results}"
            )

        # Phase 2: Grant Lake Formation permissions to the role
        try:
            results["permissions_granted"] = self._grant_lake_formation_permissions(
                role_arn_str, database_name_str, table_name_str, session, region
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to grant Lake Formation permissions. "
                f"Subsequent phases skipped. Results: {results}. Error: {e}"
            ) from e

        if not results["permissions_granted"]:
            raise RuntimeError(
                f"Failed to grant Lake Formation permissions. "
                f"Subsequent phases skipped. Results: {results}"
            )


        # Phase 3: Revoke IAMAllowedPrincipal permissions
        try:
            results["iam_principal_revoked"] = self._revoke_iam_allowed_principal(
                database_name_str, table_name_str, session, region
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to revoke IAMAllowedPrincipal permissions. Results: {results}. Error: {e}"
            ) from e

        if not results["iam_principal_revoked"]:
            raise RuntimeError(
                f"Failed to revoke IAMAllowedPrincipal permissions. Results: {results}"
            )

        # Phase 4: Apply S3 bucket policy
        # Validate feature_group_arn is available for account ID extraction
        if self.feature_group_arn is None or isinstance(self.feature_group_arn, Unassigned):
            raise ValueError(
                "Feature Group ARN is required to apply bucket policy. "
                "Ensure the Feature Group is in 'Created' status."
            )

        bucket_name, s3_prefix = parse_s3_url(resolved_s3_uri_str)
        feature_group_arn_str = str(self.feature_group_arn) if self.feature_group_arn else ""
        account_id = self._extract_account_id_from_arn(feature_group_arn_str)

        if use_service_linked_role:
            lf_role_arn = self._get_lake_formation_service_linked_role_arn(account_id, region)
        else:
            lf_role_arn = str(registration_role_arn)

        try:
            results["bucket_policy_applied"] = self._apply_bucket_policy(
                bucket_name=bucket_name,
                s3_prefix=s3_prefix,
                lake_formation_role_arn=lf_role_arn,
                feature_store_role_arn=role_arn_str,
                session=session,
                region=region,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to apply bucket policy. Results: {results}. Error: {e}"
            ) from e

        if not results["bucket_policy_applied"]:
            raise RuntimeError(
                f"Failed to apply bucket policy. Results: {results}"
            )

        logger.info(f"Lake Formation setup complete for {self.feature_group_name}: {results}")

        return results

    @classmethod
    @Base.add_validate_call
    def create(
        cls,
        feature_group_name: StrPipeVar,
        record_identifier_feature_name: StrPipeVar,
        event_time_feature_name: StrPipeVar,
        feature_definitions: List[FeatureDefinition],
        online_store_config: Optional[OnlineStoreConfig] = None,
        offline_store_config: Optional[OfflineStoreConfig] = None,
        throughput_config: Optional[ThroughputConfig] = None,
        role_arn: Optional[StrPipeVar] = None,
        description: Optional[StrPipeVar] = None,
        tags: Optional[List[Tag]] = None,
        use_pre_prod_offline_store_replicator_lambda: Optional[bool] = None,
        lake_formation_config: Optional[LakeFormationConfig] = None,
        session: Optional[Session] = None,
        region: Optional[StrPipeVar] = None,
    ) -> Optional["FeatureGroupManager"]:
        """
        Create a FeatureGroupManager resource with optional Lake Formation governance.

        Parameters:
            feature_group_name: The name of the FeatureGroup.
            record_identifier_feature_name: The name of the Feature whose value uniquely
                identifies a Record.
            event_time_feature_name: The name of the feature that stores the EventTime.
            feature_definitions: A list of Feature names and types.
            online_store_config: Configuration for the OnlineStore.
            offline_store_config: Configuration for the OfflineStore.
            throughput_config: Throughput configuration.
            role_arn: IAM execution role ARN for the OfflineStore.
            description: A free-form description of the FeatureGroup.
            tags: Tags used to identify Features in each FeatureGroup.
            use_pre_prod_offline_store_replicator_lambda: Pre-prod replicator flag.
            lake_formation_config: Optional LakeFormationConfig to configure Lake Formation
                governance. When enabled=True, requires offline_store_config and role_arn.
            session: Boto3 session.
            region: Region name.

        Returns:
            The FeatureGroupManager resource.

        Raises:
            botocore.exceptions.ClientError: This exception is raised for AWS service related errors.
                The error message and error code can be parsed from the exception as follows:
                ```
                try:
                    # AWS service call here
                except botocore.exceptions.ClientError as e:
                    error_message = e.response['Error']['Message']
                    error_code = e.response['Error']['Code']
                ```
            ResourceInUse: Resource being accessed is in use.
            ResourceLimitExceeded: You have exceeded an SageMaker resource limit.
                For example, you might have too many training jobs created.
            ConfigSchemaValidationError: Raised when a configuration file does not adhere to the schema
            LocalConfigNotFoundError: Raised when a configuration file is not found in local file system
            S3ConfigNotFoundError: Raised when a configuration file is not found in S3
        """
        # Validation for Lake Formation
        if lake_formation_config is not None and lake_formation_config.enabled:
            if offline_store_config is None:
                raise ValueError(
                    "lake_formation_config with enabled=True requires offline_store_config to be configured"
                )
            if role_arn is None:
                raise ValueError(
                    "lake_formation_config with enabled=True requires role_arn to be specified"
                )
            if (
                not lake_formation_config.use_service_linked_role
                and not lake_formation_config.registration_role_arn
            ):
                raise ValueError(
                    "registration_role_arn must be provided in lake_formation_config "
                    "when use_service_linked_role is False"
                )

        # Build kwargs, only including non-None values so parent uses its defaults
        create_kwargs = {
            "feature_group_name": feature_group_name,
            "record_identifier_feature_name": record_identifier_feature_name,
            "event_time_feature_name": event_time_feature_name,
            "feature_definitions": feature_definitions,
            "session": session,
            "region": region,
        }
        if online_store_config is not None:
            create_kwargs["online_store_config"] = online_store_config
        if offline_store_config is not None:
            create_kwargs["offline_store_config"] = offline_store_config
        if throughput_config is not None:
            create_kwargs["throughput_config"] = throughput_config
        if role_arn is not None:
            create_kwargs["role_arn"] = role_arn
        if description is not None:
            create_kwargs["description"] = description
        if tags is not None:
            create_kwargs["tags"] = tags
        if use_pre_prod_offline_store_replicator_lambda is not None:
            create_kwargs["use_pre_prod_offline_store_replicator_lambda"] = use_pre_prod_offline_store_replicator_lambda

        super().create(**create_kwargs)

        # Get as FeatureGroupManager instance (super().create() returns FeatureGroup)
        feature_group = cls.get(feature_group_name=feature_group_name, session=session, region=region)

        # Enable Lake Formation if requested
        if lake_formation_config is not None and lake_formation_config.enabled:
            feature_group.wait_for_status(target_status="Created")
            feature_group.enable_lake_formation(
                session=session,
                region=region,
                use_service_linked_role=lake_formation_config.use_service_linked_role,
                registration_role_arn=lake_formation_config.registration_role_arn,
            )
        return feature_group
