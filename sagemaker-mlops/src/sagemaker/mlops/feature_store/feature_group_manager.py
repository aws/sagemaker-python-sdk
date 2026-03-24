# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""FeatureGroup with Lake Formation support."""

import json
import logging
from typing import Dict, List, Optional

import botocore.exceptions
from pydantic import model_validator

from sagemaker.core.resources import FeatureGroup
from sagemaker.core.resources import Base
from sagemaker.core.shapes import (
    AddOnlineStoreReplicaAction,
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
from sagemaker.mlops.feature_store.feature_utils import _APPROVED_ICEBERG_PROPERTIES


logger = logging.getLogger(__name__)


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
        invalid_keys = set(self.properties.keys()) - _APPROVED_ICEBERG_PROPERTIES
        if invalid_keys:
            raise ValueError(
                f"Invalid iceberg properties: {invalid_keys}. "
                f"Approved properties are: {_APPROVED_ICEBERG_PROPERTIES}"
            )
        return self


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
        show_s3_policy: If True, prints the S3 deny policy to the console after successful
            Lake Formation setup. This policy should be added to your S3 bucket to restrict
            access to only the allowed principals. Default is False.
        disable_hybrid_access_mode: If True, revokes IAMAllowedPrincipal permissions from
            the Glue table, moving it to Lake Formation-only access mode. If False, keeps
            hybrid access mode where both IAM and Lake Formation control access.
            Default is True (LF-only mode).
    """

    enabled: bool = False
    use_service_linked_role: bool = True
    registration_role_arn: Optional[str] = None
    show_s3_policy: bool = False
    disable_hybrid_access_mode: bool = True


class FeatureGroupManager(FeatureGroup):

    # Attribute for Iceberg table properties (populated by get() when include_iceberg_properties=True)
    iceberg_properties: Optional[IcebergProperties] = None

    # Inherit parent docstring and append our additions
    if FeatureGroup.__doc__ and __doc__:
        __doc__ = FeatureGroup.__doc__

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

    def _generate_s3_deny_policy(
        self,
        bucket_name: str,
        s3_prefix: str,
        lake_formation_role_arn: str,
        feature_store_role_arn: str,
        region: Optional[str] = None,
    ) -> dict:
        """
        Generate an S3 deny policy for Lake Formation governance.

        This policy denies S3 access to the offline store data prefix except for
        the Lake Formation role and Feature Store execution role.

        Args:
            bucket_name: S3 bucket name.
            s3_prefix: S3 prefix path (without bucket name).
            lake_formation_role_arn: Lake Formation registration role ARN.
            feature_store_role_arn: Feature Store execution role ARN.
            region: AWS region name (e.g., 'us-west-2'). Used to determine the correct
                partition for S3 ARNs. If not provided, defaults to 'aws' partition.

        Returns:
            S3 bucket policy as a dict with valid JSON structure containing:
            - Version: "2012-10-17"
            - Statement: List with two deny statements:
              1. Deny GetObject, PutObject, DeleteObject on data prefix except allowed principals
              2. Deny ListBucket on bucket with prefix condition except allowed principals
        """
        partition = aws_partition(region) if region else "aws"

        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "DenyAllAccessToFeatureStorePrefixExceptAllowedPrincipals",
                    "Effect": "Deny",
                    "Principal": "*",
                    "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                    "Resource": f"arn:{partition}:s3:::{bucket_name}/{s3_prefix}/*",
                    "Condition": {
                        "StringNotEquals": {
                            "aws:PrincipalArn": [
                                lake_formation_role_arn,
                                feature_store_role_arn,
                            ]
                        }
                    },
                },
                {
                    "Sid": "DenyListOnPrefixExceptAllowedPrincipals",
                    "Effect": "Deny",
                    "Principal": "*",
                    "Action": "s3:ListBucket",
                    "Resource": f"arn:{partition}:s3:::{bucket_name}",
                    "Condition": {
                        "StringLike": {"s3:prefix": f"{s3_prefix}/*"},
                        "StringNotEquals": {
                            "aws:PrincipalArn": [
                                lake_formation_role_arn,
                                feature_store_role_arn,
                            ]
                        },
                    },
                },
            ],
        }
        return policy

    def _get_lake_formation_client(
        self,
        session: Optional[Session] = None,
        region: Optional[str] = None,
    ):
        """
        Get a Lake Formation client, reusing a cached client when possible.

        The client is cached on the instance keyed by (session, region). Subsequent
        calls with the same arguments return the existing client instead of creating
        a new one.

        Args:
            session: Boto3 session. If not provided, a new session will be created.
            region: AWS region name.

        Returns:
            A boto3 Lake Formation client.
        """
        cache_key = (id(session), region)
        if not hasattr(self, "_lf_client_cache"):
            self._lf_client_cache: dict = {}

        if cache_key not in self._lf_client_cache:
            boto_session = session or Session()
            self._lf_client_cache[cache_key] = boto_session.client(
                "lakeformation", region_name=region
            )

        return self._lf_client_cache[cache_key]

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
            register_params = {"ResourceArn": resource_arn}

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

        Args:
            database_name: Glue database name.
            table_name: Glue table name.
            session: Boto3 session.
            region: AWS region. If not provided, will be inferred from the session.

        Returns:
            True if revocation succeeded or permissions didn't exist.

        Raises:
            ClientError: If revocation fails for unexpected reasons.
        """
        # Get region from session if not provided
        if region is None and session is not None:
            region = session.region_name

        client = self._get_lake_formation_client(session, region)

        try:
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
        except botocore.exceptions.ClientError as e:
            # if the Table doesn't have that permission because the user already revoked it
            # then just return True
            if e.response["Error"]["Code"] == "InvalidInputException":
                logger.info(
                    f"IAMAllowedPrincipal permissions may not exist on: {database_name}.{table_name}"
                )
                return True
            raise

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

    @Base.add_validate_call
    def enable_lake_formation(
        self,
        session: Optional[Session] = None,
        region: Optional[str] = None,
        use_service_linked_role: bool = True,
        registration_role_arn: Optional[str] = None,
        wait_for_active: bool = False,
        show_s3_policy: bool = False,
        disable_hybrid_access_mode: bool = True,
    ) -> dict:
        """
        Enable Lake Formation governance for this Feature Group's offline store.

        This method:
        1. Optionally waits for Feature Group to reach 'Created' status
        2. Validates Feature Group status is 'Created'
        3. Registers the offline store S3 location as data lake location
        4. Grants the execution role permissions on the Glue table
        5. Optionally revokes IAMAllowedPrincipal permissions from the Glue table
           (controlled by disable_hybrid_access_mode)

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
            show_s3_policy: If True, prints the S3 deny policy to the console after successful
                Lake Formation setup. This policy should be added to your S3 bucket to restrict
                access to only the allowed principals. Default is False.
            disable_hybrid_access_mode: If True, revokes IAMAllowedPrincipal permissions from
                the Glue table, moving it to Lake Formation-only access mode. If False, keeps
                hybrid access mode where both IAM and Lake Formation control access.
                Default is True.

        Returns:
            Dict with status of each Lake Formation operation:
            - s3_registration: bool
            - iam_principal_revoked: bool or None (None when disable_hybrid_access_mode=False)
            - permissions_granted: bool

        Raises:
            ValueError: If the Feature Group has no offline store configured,
                if role_arn is not set on the Feature Group, if use_service_linked_role
                is False but registration_role_arn is not provided, or if the Feature Group
                is not in 'Created' status.
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
        if self.offline_store_config is None or self.offline_store_config == Unassigned():
            raise ValueError(
                f"Feature Group '{self.feature_group_name}' does not have an offline store configured. "
                "Lake Formation can only be enabled for Feature Groups with offline stores."
            )

        # Get role ARN from Feature Group config
        if self.role_arn is None or self.role_arn == Unassigned():
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
        if resolved_s3_uri is None or resolved_s3_uri == Unassigned():
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

        # Execute Lake Formation setup with fail-fast behavior
        results = {
            "s3_registration": False,
            "iam_principal_revoked": False,
            "permissions_granted": False,
        }

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

        # Phase 3: Revoke IAMAllowedPrincipal permissions (if disabling hybrid access mode)
        if disable_hybrid_access_mode:
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
        else:
            results["iam_principal_revoked"] = None
            logger.info(
                "Skipping IAMAllowedPrincipal revocation - hybrid access mode preserved."
            )

        logger.info(f"Lake Formation setup complete for {self.feature_group_name}: {results}")

        # Generate and optionally print S3 deny policy
        if show_s3_policy:
            # Extract bucket name and prefix from resolved S3 URI using core utility
            bucket_name, s3_prefix = parse_s3_url(resolved_s3_uri_str)

            # Extract account ID from Feature Group ARN
            feature_group_arn_str = str(self.feature_group_arn) if self.feature_group_arn else ""
            account_id = self._extract_account_id_from_arn(feature_group_arn_str)

            # Determine Lake Formation role ARN based on use_service_linked_role flag
            if use_service_linked_role:
                lf_role_arn = self._get_lake_formation_service_linked_role_arn(account_id, region)
            else:
                # registration_role_arn is validated earlier when use_service_linked_role is False
                lf_role_arn = str(registration_role_arn)

            # Generate the S3 deny policy
            policy = self._generate_s3_deny_policy(
                bucket_name=bucket_name,
                s3_prefix=s3_prefix,
                lake_formation_role_arn=lf_role_arn,
                feature_store_role_arn=role_arn_str,
                region=region,
            )

            # Log policy with clear instructions
            policy_json = json.dumps(policy, indent=2)
            policy_message = (
                "\n" + "=" * 50
                + "\nS3 Bucket Policy Update recommended"
                + "\n" + "=" * 50
                + "\n\nTo complete Lake Formation setup, add the following"
                + "\ndeny policy to your S3 bucket."
                + "\nThis restricts access to the offline store data to"
                + "\nonly the allowed principals."
                + f"\n\nBucket: {bucket_name}"
                + f"\n\nPolicy to add:\n{policy_json}"
                + "\n\nNote: Merge this with your existing bucket policy if one exists."
                + "\n" + "=" * 50 + "\n"
            )
            logger.info(policy_message)

        return results

    def _get_iceberg_properties(
        self,
        session: Optional[Session] = None,
        region: Optional[StrPipeVar] = None,
    ) -> Dict[str, any]:
        """
        Fetch the current Glue table definition for the Feature Group's Iceberg offline store.

        Validates that the Feature Group has an Iceberg-formatted offline store,
        retrieves the Glue table, and strips non-TableInput fields.

        Parameters:
            session: Optional boto3 session. If not provided, uses default credentials.
            region: Optional AWS region. If not provided, uses default region.

        Returns:
            Dict with keys:
            - 'database_name': The Glue database name
            - 'table_name': The Glue table name
            - 'table_input': The cleaned Glue TableInput dict
            - 'glue_client': The Glue client used for the request

        Raises:
            ValueError: If offline_store_config is not configured or table_format is not Iceberg.
            RuntimeError: If the Glue get_table call fails.
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

        # Get Glue client
        if session is None:
            session = Session()

        region_str = str(region) if region else session.region_name
        glue_client = session.client("glue", region_name=region_str)

        try:
            # Get current table definition
            response = glue_client.get_table(
                DatabaseName=database_name,
                Name=table_name,
            )
            table_input = response["Table"]

            # Remove fields that shouldn't be in TableInput
            fields_to_remove = [
                "DatabaseName",
                "CreateTime",
                "UpdateTime",
                "CreatedBy",
                "IsRegisteredWithLakeFormation",
                "CatalogId",
                "VersionId",
                "FederatedTable",
            ]
            for field in fields_to_remove:
                table_input.pop(field, None)

            return {
                "database_name": database_name,
                "table_name": table_name,
                "table_input": table_input,
                "glue_client": glue_client,
            }

        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            raise RuntimeError(
                f"Failed to update Iceberg properties for {self.feature_group_name}: "
                f"[{error_code}] {error_message}"
            ) from e

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

        invalid_keys = set(iceberg_properties.properties.keys()) - _APPROVED_ICEBERG_PROPERTIES
        if invalid_keys:
            raise ValueError(
                f"Invalid iceberg properties: {invalid_keys}. "
                f"Approved properties are: {_APPROVED_ICEBERG_PROPERTIES}"
            )

        result = self._get_iceberg_properties(session=session, region=region)
        database_name = result["database_name"]
        table_name = result["table_name"]
        table_input = result["table_input"]
        glue_client = result["glue_client"]

        logger.info(
            f"Updating Iceberg properties for {self.feature_group_name} "
            f"(database={database_name}, table={table_name})"
        )

        try:
            # Update parameters with new Iceberg properties
            if "Parameters" not in table_input:
                table_input["Parameters"] = {}

            for key, value in iceberg_properties.properties.items():
                table_input["Parameters"][key] = value

            # Update the table
            glue_client.update_table(
                DatabaseName=database_name,
                TableInput=table_input,
            )

            logger.info(
                f"Successfully updated Iceberg properties for {self.feature_group_name}"
            )

            return {
                "database": database_name,
                "table": table_name,
                "properties_updated": iceberg_properties.properties,
            }

        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            raise RuntimeError(
                f"Failed to update Iceberg properties for {self.feature_group_name}: "
                f"[{error_code}] {error_message}"
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
            feature_group.iceberg_properties = IcebergProperties(
                properties=result["table_input"].get("Parameters", {})
            )

        return feature_group

    @classmethod
    def create(
        cls,
        *args,
        lake_formation_config: Optional[LakeFormationConfig] = None,
        iceberg_properties: Optional[IcebergProperties] = None,
        **kwargs,
    ) -> Optional["FeatureGroup"]:
        """
        Create a FeatureGroup resource with optional Lake Formation governance and Iceberg properties.

        Accepts all parameters from FeatureGroup.create(), plus:

        Parameters:
            lake_formation_config: Optional LakeFormationConfig to configure Lake Formation
                governance. When enabled=True, requires offline_store_config and role_arn.
            iceberg_properties: Optional IcebergProperties to configure Iceberg table
                properties for the offline store. Requires offline_store_config with
                table_format='Iceberg'.

        Returns:
            The FeatureGroup resource.
        """
        offline_store_config = kwargs.get("offline_store_config")
        role_arn = kwargs.get("role_arn")
        session = kwargs.get("session")
        region = kwargs.get("region")

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
                show_s3_policy=lake_formation_config.show_s3_policy,
                disable_hybrid_access_mode=lake_formation_config.disable_hybrid_access_mode,
            )
        
        # Update Iceberg properties if requested
        if iceberg_properties is not None and iceberg_properties.properties:
            # Wait for feature group to be created before updating Iceberg properties
            feature_group.wait_for_status(target_status="Created")
            feature_group._update_iceberg_properties(
                iceberg_properties=iceberg_properties,
                session=session,
                region=region,
            )

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

        result = super().update(*args, **kwargs)

        # Update Iceberg properties if requested
        if iceberg_properties is not None and iceberg_properties.properties:
            self._update_iceberg_properties(
                iceberg_properties=iceberg_properties,
                session=session,
                region=region,
            )

        return result
