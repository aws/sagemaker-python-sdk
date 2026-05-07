# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""FeatureGroup with Lake Formation support."""

import json
import logging
from collections import Counter
from typing import Dict, List, Optional

from pydantic import model_validator

import botocore.exceptions

from sagemaker.core.resources import FeatureGroup
from sagemaker.core.resources import Base
from sagemaker.core.shapes import (
    FeatureDefinition,
    OfflineStoreConfig,
    OnlineStoreConfig,
    OnlineStoreConfigUpdate,
    Tag,
    ThroughputConfig,
    ThroughputConfigUpdate,
)
from sagemaker.core.shapes import Unassigned
from sagemaker.core.helper.pipeline_variable import StrPipeVar
from sagemaker.core.s3.utils import parse_s3_url
from sagemaker.core.common_utils import aws_partition
from boto3 import Session
from botocore.exceptions import ClientError
from pyiceberg.catalog import load_catalog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sagemaker.mlops.feature_store.feature_utils import (
    _ALLOWED_ICEBERG_PROPERTIES,
    _ICEBERG_PERMISSIONS_ERROR_MESSAGE,
)


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
        hybrid_access_mode_enabled: If True, IAM-based access remains alongside
            Lake Formation permissions (hybrid access mode). If False, revokes
            IAMAllowedPrincipal permissions from the Glue table, enforcing Lake
            Formation-only governance. Warning: setting this to False may break
            existing jobs that access the table via IAM-based permissions. After
            this change, all principals must be granted access through Lake Formation.
        acknowledge_risk: If True, acknowledges the risks of the Lake Formation
            operation and proceeds. When hybrid_access_mode_enabled is False, this
            acknowledges that revoking IAMAllowedPrincipal permissions may break
            existing jobs (e.g., training, processing, ETL) that access the table
            via IAM-based permissions. When hybrid_access_mode_enabled is True,
            this acknowledges that IAM-based access remains alongside Lake Formation
            permissions. If False, raises RuntimeError without proceeding.
    """

    enabled: bool = False
    use_service_linked_role: bool = True
    registration_role_arn: Optional[str] = None
    hybrid_access_mode_enabled: bool
    acknowledge_risk: bool


class IcebergProperties(Base):
    """Configuration for Iceberg table properties in a Feature Group offline store.

    Use this to customize Iceberg table behavior such as compaction settings,
    snapshot retention, and other Iceberg-specific configurations.

    Attributes:
        properties: A dictionary mapping Iceberg property names to their values.
            Common properties include:
            - 'write.target-file-size-bytes': Target size for data files
            - 'commit.manifest.min-count-to-merge': Min manifests before merging
            - 'history.expire.max-snapshot-age-ms': Max age for snapshot expiration
    """

    properties: Optional[Dict[str, str]] = None

    @model_validator(mode="after")
    def validate_property_keys(self):
        if self.properties is None:
            return self
        invalid_keys = set(self.properties.keys()) - _ALLOWED_ICEBERG_PROPERTIES
        if invalid_keys:
            raise ValueError(
                f"Invalid iceberg properties: {invalid_keys}. "
                f"Allowed properties are: {_ALLOWED_ICEBERG_PROPERTIES}"
            )
        # Check for no duplicate keys
        keys = list(self.properties.keys())
        duplicates = {k for k, count in Counter(keys).items() if count > 1}
        if duplicates:
            raise ValueError(
                f"Invalid duplicate properties: {duplicates}. Please only have 1 of each property."
            )
        return self


class FeatureGroupManager(FeatureGroup):
    """FeatureGroup with extended management capabilities."""
     # Inherit parent docstring and append our additions
    if FeatureGroup.__doc__ and __doc__:
        __doc__ = FeatureGroup.__doc__
        
    # Attribute for Iceberg table properties (populated by get() when include_iceberg_properties=True)
    iceberg_properties: Optional[IcebergProperties] = None

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
    ) -> list:
        """
        Generate S3 deny statements for Lake Formation governance.

        These statements deny S3 access to the offline store data prefix except for
        the Lake Formation role and Feature Store execution role.

        Args:
            bucket_name: S3 bucket name.
            s3_prefix: S3 prefix path (without bucket name).
            lake_formation_role_arn: Lake Formation registration role ARN.
            feature_store_role_arn: Feature Store execution role ARN.
            region: AWS region name (e.g., 'us-west-2'). Used to determine the correct
                partition for S3 ARNs. If not provided, defaults to 'aws' partition.

        Returns:
            List of statement dicts containing:
            1. Deny GetObject, PutObject, DeleteObject on data prefix except allowed principals
            2. Deny ListBucket on bucket with prefix condition except allowed principals
        """
        partition = aws_partition(region) if region else "aws"

        allowed_principals = [lake_formation_role_arn, feature_store_role_arn]

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
    

    @Base.add_validate_call
    def enable_lake_formation(
        self,
        hybrid_access_mode_enabled: bool,
        acknowledge_risk: bool,
        session: Optional[Session] = None,
        region: Optional[str] = None,
        use_service_linked_role: bool = True,
        registration_role_arn: Optional[str] = None,
        wait_for_active: bool = False
    ) -> dict:
        """
        Enable Lake Formation governance for this Feature Group's offline store.

        This method:
        1. Optionally waits for Feature Group to reach 'Created' status
        2. Validates Feature Group status is 'Created'
        3. Registers the offline store S3 location as data lake location
        4. Grants the execution role permissions on the Glue table
        5. Optionally revokes IAMAllowedPrincipal permissions from the Glue table
        6. Prints recommended S3 deny bucket policy

        Parameters:
            hybrid_access_mode_enabled: If True, IAM-based access remains alongside
                Lake Formation permissions (hybrid access mode). If False, revokes
                IAMAllowedPrincipal permissions from the Glue table, enforcing Lake
                Formation-only governance. Warning: setting this to False may break
                existing jobs (e.g., training, processing, ETL) that access the table
                via IAM-based permissions. After this change, all principals must be
                granted access through Lake Formation.
            acknowledge_risk: If True, acknowledges the risks of the Lake Formation
                operation and proceeds. When hybrid_access_mode_enabled is False, this
                acknowledges that revoking IAMAllowedPrincipal permissions may break
                existing jobs (e.g., training, processing, ETL) that access the table
                via IAM-based permissions. When hybrid_access_mode_enabled is True,
                this acknowledges that IAM-based access remains alongside Lake Formation
                permissions. If False, raises RuntimeError without proceeding.
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

        Returns:
            Dict with status of each Lake Formation operation:
            - s3_location_registered: bool
            - lf_permissions_granted: bool
            - hybrid_access_mode_enabled: bool

        Raises:
            ValueError: If the Feature Group has no offline store configured,
                if role_arn is not set on the Feature Group, if use_service_linked_role
                is False but registration_role_arn is not provided, or if the Feature
                Group is not in 'Created' status.
            ClientError: If Lake Formation operations fail.
            RuntimeError: If a phase fails and subsequent phases cannot proceed,
                or if the user declines to proceed without disabling hybrid access mode.
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

        if hybrid_access_mode_enabled:
            logger.warning(
                f"Hybrid access mode is enabled for table: {database_name_str}.{table_name_str}. "
                f"IAMAllowedPrincipal permissions remain in effect, which means IAM-based access "
                f"to the table is still allowed alongside Lake Formation permissions. "
                f"For more info: https://docs.aws.amazon.com/lake-formation/latest/dg/hybrid-access-mode.html"
            )
            proceed = acknowledge_risk
            if not proceed:
                raise RuntimeError(
                    "User chose not to proceed with hybrid access mode enabled. "
                    "Re-run with hybrid_access_mode_enabled=False to revoke IAMAllowedPrincipal permissions."
                )
        else:
            logger.warning(
                f"Disabling hybrid access mode for table: {database_name_str}.{table_name_str}. "
                f"This will revoke IAMAllowedPrincipal permissions, which may break existing jobs "
                f"(e.g., training, processing, ETL) that access this table via IAM-based permissions. "
                f"After this change, all principals must be granted access through Lake Formation. "
                f"For more info: https://docs.aws.amazon.com/lake-formation/latest/dg/hybrid-access-mode.html"
            )
            proceed = acknowledge_risk
            if not proceed:
                raise RuntimeError(
                    "User chose not to proceed with disabling hybrid access mode. "
                    "Re-run with hybrid_access_mode_enabled=True to keep IAMAllowedPrincipal permissions."
                )


        results = {
            "s3_location_registered": False,
            "lf_permissions_granted": False,
            "hybrid_access_mode_enabled": True
        }

        # Execute Lake Formation setup with fail-fast behavior.
        # On failure, log warnings for incomplete steps before re-raising.
        phase_error = None

        # Phase 1: Register S3 with Lake Formation
        try:
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

            results["s3_location_registered"] = self._register_s3_with_lake_formation(
                s3_location_to_register,
                session,
                region,
                use_service_linked_role=use_service_linked_role,
                role_arn=registration_role_arn,
            )
        except Exception as e:
            phase_error = RuntimeError(
                f"Failed to register S3 location with Lake Formation. "
                f"Subsequent phases skipped. Results: {results}. Error: {e}"
            )
            phase_error.__cause__ = e

        # Phase 2: Grant Lake Formation permissions to the role
        if phase_error is None:
            try:
                results["lf_permissions_granted"] = self._grant_lake_formation_permissions(
                    role_arn_str, database_name_str, table_name_str, session, region
                )
            except Exception as e:
                phase_error = RuntimeError(
                    f"Failed to grant Lake Formation permissions. "
                    f"Subsequent phases skipped. Results: {results}. Error: {e}"
                )
                phase_error.__cause__ = e

        # Phase 3: Revoke IAMAllowedPrincipal permissions
        if phase_error is None and not hybrid_access_mode_enabled:
            try:
                revoked = self._revoke_iam_allowed_principal(
                    database_name_str, table_name_str, session, region
                )
                results["hybrid_access_mode_enabled"] = not revoked
            except Exception as e:
                phase_error = RuntimeError(
                    f"Failed to revoke IAMAllowedPrincipal permissions. Results: {results}. Error: {e}"
                )
                phase_error.__cause__ = e

        # Warn about any steps that were not completed
        if not results["s3_location_registered"]:
            logger.warning(
                "S3 location was not registered with Lake Formation. "
                "Re-run enable_lake_formation() after fixing the issue."
            )
        if not results["lf_permissions_granted"]:
            logger.warning(
                "Lake Formation permissions were not granted to the "
                "execution role. Re-run enable_lake_formation() after fixing the issue."
            )
        if not hybrid_access_mode_enabled and results["hybrid_access_mode_enabled"]:
            logger.warning(
                "Failed to disable hybrid access mode. IAM-based access "
                "to the Glue table is still allowed alongside Lake "
                "Formation permissions. Re-run with "
                "hybrid_access_mode_enabled=False to retry. For more info: "
                "https://docs.aws.amazon.com/lake-formation/latest/dg/hybrid-access-mode.html"
            )

        # Re-raise after warnings so the caller sees what was incomplete
        if phase_error is not None:
            raise phase_error

        # Phase 4: Print S3 bucket policy
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

    
        bucket_deny_policy = self._generate_s3_deny_statements(
            bucket_name=bucket_name,
            s3_prefix=s3_prefix,
            lake_formation_role_arn=lf_role_arn,
            feature_store_role_arn=role_arn_str,
            region=region
        )

        policy_json = json.dumps(bucket_deny_policy, indent=2)
        logger.warning(
            "RECOMMENDED S3 BUCKET POLICY\n"
            "\n"
            "Lake Formation governs access through the Glue Data Catalog\n"
            "but does not block direct S3 access. To enforce Lake\n"
            "Formation-only access to the offline store data, add the\n"
            "following deny statements to your S3 bucket policy for\n"
            "bucket '%s'.\n"
            "\n"
            "WARNING: Applying this policy will deny direct S3 access\n"
            "to any principal not in the allowed list. This may cause\n"
            "existing jobs (e.g., training, processing, ETL) that read\n"
            "from or write to this path to fail if they rely on direct\n"
            "S3 access instead of Lake Formation vended credentials.\n"
            "\n"
            "%s",
            bucket_name,
            policy_json,
        )

        logger.info(f"Lake Formation setup complete for {self.feature_group_name}: {results}")

        return results

    def _validate_table_ownership(self, table, database_name: str, table_name: str):
        """Validate that the Iceberg table belongs to this feature group by checking S3 location."""
        table_location = table.metadata.location if table.metadata else None
        s3_config = self.offline_store_config.s3_storage_config
        if s3_config and s3_config.s3_uri:
            expected_prefix = str(s3_config.s3_uri).rstrip("/")
            if table_location and not table_location.startswith(expected_prefix):
                logger.error(
                    f"Table ownership validation failed for feature group "
                    f"'{self.feature_group_name}'. The Glue table "
                    f"'{database_name}.{table_name}' has location '{table_location}' "
                    f"but the feature group's offline store is configured with "
                    f"S3 URI '{expected_prefix}'. This may indicate that the "
                    f"data_catalog_config is pointing to a table that does not belong "
                    f"to this feature group. To fix this, verify that the "
                    f"data_catalog_config.database and data_catalog_config.table_name "
                    f"in your feature group's offline_store_config match the correct "
                    f"Glue table for this feature group."
                )
                raise ValueError(
                    f"Table '{database_name}.{table_name}' location '{table_location}' "
                    f"does not match the feature group's S3 URI '{expected_prefix}'. "
                    f"The table may not belong to feature group '{self.feature_group_name}'."
                )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(RuntimeError),
        reraise=True,
    )
    def _get_iceberg_properties(
        self,
        session: Optional[Session] = None,
        region: Optional[StrPipeVar] = None,
    ) -> Dict[str, any]:
        """
        Fetch the current Iceberg catalog table definition for the Feature Group's Iceberg offline store.

        Validates that the Feature Group has an Iceberg-formatted offline store
        and retrieves the table via the Iceberg catalog.

        Parameters:
            session: Optional boto3 session. If not provided, uses default credentials.
            region: Optional AWS region. If not provided, uses default region.

        Returns:
            Dict with keys:
            - 'database_name': The Iceberg catalog database name
            - 'table_name': The Iceberg catalog table name
            - 'table': The pyiceberg Table object
            - 'properties': The table properties dict

        Raises:
            ValueError: If offline_store_config is not configured or table_format is not Iceberg.
            RuntimeError: If the Iceberg catalog table load fails.
        """
        # Validate offline store is configured
        if self.offline_store_config is None or self.offline_store_config == Unassigned():
            raise ValueError(
                "Cannot update Iceberg properties: offline_store_config is not configured"
            )

        # Validate table format is Iceberg
        if (
            self.offline_store_config.table_format is None
            or str(self.offline_store_config.table_format) != "Iceberg"
        ):
            raise ValueError(
                "Cannot update Iceberg properties: table_format must be 'Iceberg'"
            )

        # Get database and table name from data_catalog_config
        data_catalog_config = self.offline_store_config.data_catalog_config
        if data_catalog_config is None:
            raise ValueError(
                "Cannot update Iceberg properties: data_catalog_config is not available"
            )

        database_name = str(data_catalog_config.database)
        table_name = str(data_catalog_config.table_name)

        if session is None:
            session = Session()
        region_str = str(region) if region else session.region_name
        catalog = load_catalog("glue", **{"type": "glue", "client.region": region_str})

        try:
            table = catalog.load_table(f"{database_name}.{table_name}")
            self._validate_table_ownership(table, database_name, table_name)

            return {
                "database_name": database_name,
                "table_name": table_name,
                "table": table,
                "properties": dict(table.properties),
            }

        except ClientError as e:
            if e.response["Error"]["Code"] == "AccessDeniedException":
                raise PermissionError(
                    f"Access denied reading Iceberg properties for '{self.feature_group_name}'. "
                    f"{_ICEBERG_PERMISSIONS_ERROR_MESSAGE}"
                ) from e
            raise RuntimeError(
                f"Failed to get Iceberg properties for '{self.feature_group_name}': {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to get Iceberg properties for '{self.feature_group_name}': {e}"
            ) from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(RuntimeError),
        reraise=True,
    )
    def _update_iceberg_properties(
        self,
        iceberg_properties: IcebergProperties,
        session: Optional[Session] = None,
        region: Optional[StrPipeVar] = None,
    ) -> Dict[str, any]:
        """
        Update Iceberg table properties for the Feature Group's offline store.

        This method updates the Glue table properties for an Iceberg-formatted
        offline store. The Feature Group must have an offline store configured
        with table_format='Iceberg'.

        Parameters:
            iceberg_properties: IcebergProperties object containing the properties to set.
            session: Optional boto3 session. If not provided, uses default credentials.
            region: Optional AWS region. If not provided, uses default region.

        Returns:
            Dict containing the update results with keys:
            - 'database': The Glue database name
            - 'table': The Glue table name
            - 'properties_updated': The properties that were updated

        Raises:
            ValueError: If offline_store_config is not configured or table_format is not Iceberg.
            RuntimeError: If the Glue table update fails.
        """
        # Validate iceberg_properties has properties to update
        if iceberg_properties is None or not iceberg_properties.properties:
            raise ValueError(
                "iceberg_properties must contain at least one property to update"
            )

        invalid_keys = set(iceberg_properties.properties.keys()) - _ALLOWED_ICEBERG_PROPERTIES
        if invalid_keys:
            raise ValueError(
                f"Invalid iceberg properties: {invalid_keys}. "
                f"Allowed properties are: {_ALLOWED_ICEBERG_PROPERTIES}"
            )

         # Check for no duplicate keys
        keys = list(iceberg_properties.properties.keys())
        duplicates = {k for k, count in Counter(keys).items() if count > 1}
        if duplicates:
            raise ValueError(
                f"Invalid duplicate properties: {duplicates}. Please only have 1 of each property."
            )

        result = self._get_iceberg_properties(session=session, region=region)
        database_name = result["database_name"]
        table_name = result["table_name"]
        table = result["table"]
        current_properties = result["properties"]

        self._validate_table_ownership(table, database_name, table_name)

        # Compute before/after diff for audit logging
        changed = {}
        for key, new_value in iceberg_properties.properties.items():
            old_value = current_properties.get(key)
            if old_value != new_value:
                changed[key] = {"old": old_value, "new": new_value}

        logger.info(
            f"Updating Iceberg properties for feature group '{self.feature_group_name}' "
            f"(database={database_name}, table={table_name}). "
            f"Property changes: {changed}"
        )

        try:
            with table.transaction() as txn:
                txn.set_properties(**iceberg_properties.properties)

            logger.info(
                f"Successfully updated Iceberg properties for feature group "
                f"'{self.feature_group_name}'. Properties applied: {changed}"
            )

            return {
                "database": database_name,
                "table": table_name,
                "properties_updated": iceberg_properties.properties,
            }

        except ClientError as e:
            if e.response["Error"]["Code"] == "AccessDeniedException":
                raise PermissionError(
                    f"Access denied updating Iceberg properties for '{self.feature_group_name}'. "
                    f"{_ICEBERG_PERMISSIONS_ERROR_MESSAGE}"
                ) from e
            logger.error(
                f"Failed to update Iceberg properties for feature group "
                f"'{self.feature_group_name}'. Attempted changes: {changed}. Error: {e}"
            )
            raise RuntimeError(
                f"Failed to update Iceberg properties for '{self.feature_group_name}': {e}"
            ) from e
        except Exception as e:
            logger.error(
                f"Failed to update Iceberg properties for feature group "
                f"'{self.feature_group_name}'. Attempted changes: {changed}. Error: {e}"
            )
            raise RuntimeError(
                f"Failed to update Iceberg properties for {self.feature_group_name}: {e}"
            ) from e

    @classmethod
    def get(
        cls,
        *args,
        include_iceberg_properties: bool = False,
        **kwargs,
    ) -> Optional["FeatureGroup"]:
        """
        Get a FeatureGroup resource with optional Iceberg property retrieval.

        Accepts all parameters from FeatureGroup.get(), plus:

        Parameters:
            include_iceberg_properties: If True, fetches Iceberg table properties
                from Glue and stores them in the iceberg_properties attribute.
                Only applies to Feature Groups with table_format='Iceberg'.

        Returns:
            The FeatureGroup resource.
        """
        session = kwargs.get("session")
        region = kwargs.get("region")

        feature_group = super().get(*args, **kwargs)

        if include_iceberg_properties:
            result = feature_group._get_iceberg_properties(session=session, region=region)
            all_properties = result["properties"]
            allowed_properties = {
                k: v for k, v in all_properties.items()
                if k in _ALLOWED_ICEBERG_PROPERTIES
            }
            feature_group.iceberg_properties = IcebergProperties(
                properties=allowed_properties
            )

        return feature_group

    @classmethod
    def create(
        cls,
        *args,
        lake_formation_config: Optional[LakeFormationConfig] = None,
        iceberg_properties: Optional[IcebergProperties] = None,
        **kwargs,
    ) -> Optional["FeatureGroupManager"]:
        """
        Create a FeatureGroupManager resource with optional Lake Formation governance and Iceberg properties.

        Accepts all parameters from FeatureGroup.create(), plus:

        Parameters:
            lake_formation_config: Optional LakeFormationConfig to configure Lake Formation
                governance. When enabled=True, requires offline_store_config and role_arn.
                The config fields control the following behavior:

                - **enabled** (bool, default False): When True, automatically enables Lake
                  Formation governance after the Feature Group is created. This registers
                  the offline store S3 location with Lake Formation and grants the execution
                  role permissions on the Glue table.
                - **use_service_linked_role** (bool, default True): When True, uses the Lake
                  Formation service-linked role for S3 registration. When False,
                  ``registration_role_arn`` must be provided.
                - **registration_role_arn** (str, optional): Custom IAM role ARN for S3
                  registration with Lake Formation. Required when ``use_service_linked_role``
                  is False.
                - **hybrid_access_mode_enabled** (bool, required): When True, IAM-based
                  access remains alongside Lake Formation permissions (hybrid access mode).
                  When False, revokes IAMAllowedPrincipal permissions from the Glue table,
                  enforcing Lake Formation-only governance. **Warning**: setting this to
                  False may break existing jobs (e.g., training, processing, ETL) that
                  access the table via IAM-based permissions. After this change, all
                  principals must be granted access through Lake Formation.
            iceberg_properties: Optional IcebergProperties to configure Iceberg table
                properties for the offline store. Requires offline_store_config with
                table_format='Iceberg'.

        Returns:
            The FeatureGroupManager resource.

        Raises:
            botocore.exceptions.ClientError: This exception is raised for AWS service related errors.
            ResourceInUse: Resource being accessed is in use.
            ResourceLimitExceeded: You have exceeded an SageMaker resource limit.
        """
        # Extract parent kwargs we need for validation
        offline_store_config = kwargs.get("offline_store_config")
        role_arn = kwargs.get("role_arn")
        session = kwargs.get("session")
        region = kwargs.get("region")

        # Validation for Lake Formation
        if lake_formation_config is not None and lake_formation_config.enabled:
            if not lake_formation_config.acknowledge_risk:
                raise RuntimeError(
                    "Lake Formation is enabled but acknowledge_risk is False. "
                    "Set acknowledge_risk=True to proceed with Feature Group creation."
                )
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

        # Validation for Iceberg properties
        if iceberg_properties is not None and iceberg_properties.properties:
            if offline_store_config is None:
                raise ValueError(
                    "iceberg_properties requires offline_store_config to be configured"
                )
            if (
                offline_store_config.table_format is None
                or str(offline_store_config.table_format) != "Iceberg"
            ):
                raise ValueError(
                    "iceberg_properties requires offline_store_config.table_format to be 'Iceberg'"
                )

        feature_group = super().create(*args, **kwargs)

        # Enable Lake Formation if requested
        if lake_formation_config is not None and lake_formation_config.enabled:
            feature_group.wait_for_status(target_status="Created")
            feature_group.enable_lake_formation(
                session=session,
                region=region,
                use_service_linked_role=lake_formation_config.use_service_linked_role,
                registration_role_arn=lake_formation_config.registration_role_arn,
                hybrid_access_mode_enabled=lake_formation_config.hybrid_access_mode_enabled,
                acknowledge_risk=lake_formation_config.acknowledge_risk,
            )

        # Update Iceberg properties if requested
        if iceberg_properties is not None and iceberg_properties.properties:
            feature_group.wait_for_status(target_status="Created")
            try:
                feature_group._update_iceberg_properties(
                    iceberg_properties=iceberg_properties,
                    session=session,
                    region=region,
                )
            except Exception as e:
                logger.error(
                    f"Feature group '{feature_group.feature_group_name}' was created "
                    f"successfully but failed to update Iceberg properties: {e}. "
                    f"Please now run update on the created Feature Group with the "
                    f"Iceberg Properties to avoid recreating your Feature Group again."
                )
                raise

        return feature_group

    def update(
        self,
        *args,
        iceberg_properties: Optional[IcebergProperties] = None,
        session: Optional[Session] = None,
        region: Optional[StrPipeVar] = None,
        **kwargs,
    ) -> Optional["FeatureGroup"]:
        """
        Update a FeatureGroup resource with optional Iceberg property updates.

        Accepts all parameters from FeatureGroup.update(), plus:

        Parameters:
            iceberg_properties: Optional IcebergProperties to update Iceberg table
                properties for the offline store. Requires offline_store_config with
                table_format='Iceberg'.
            session: Boto3 session for Iceberg property updates.
            region: Region name for Iceberg property updates.

        Returns:
            The FeatureGroup resource.
        """

        offline_store_config = self.offline_store_config

        # Validation for Iceberg properties
        if iceberg_properties is not None and iceberg_properties.properties:
            if offline_store_config is None or offline_store_config == Unassigned():
                raise ValueError(
                    "iceberg_properties requires offline_store_config to be configured"
                )
            if (
                offline_store_config.table_format is None
                or str(offline_store_config.table_format) != "Iceberg"
            ):
                raise ValueError(
                    "iceberg_properties requires offline_store_config.table_format to be 'Iceberg'"
                )

        # Only call parent update if there are non-iceberg args to pass
        result = None
        if args or kwargs:
            try:
                result = super().update(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Feature group '{self.feature_group_name}' was not updated successfully: {e}"
                )

        # Update Iceberg properties if requested
        if iceberg_properties is not None and iceberg_properties.properties:
            try:
                self._update_iceberg_properties(
                    iceberg_properties=iceberg_properties,
                    session=session,
                    region=region,
                )
            except Exception as e:
                logger.error(
                    f"Feature group '{self.feature_group_name}' was updated successfully "
                    f"but failed to update Iceberg properties: {e}"
                )
                raise

        return result if result is not None else self
