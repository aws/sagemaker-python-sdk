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
        disable_hybrid_access_mode: If True, revokes IAMAllowedPrincipal permissions
            from the Glue table, enforcing Lake Formation-only governance. Warning: this
            may break existing jobs that access the table via IAM-based permissions. After
            this change, all principals must be granted access through Lake Formation.
            If False, IAM-based access remains alongside Lake Formation permissions.
    """

    enabled: bool = False
    use_service_linked_role: bool = True
    registration_role_arn: Optional[str] = None
    disable_hybrid_access_mode: bool


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
        disable_hybrid_access_mode: bool,
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
        6. Prints optional S3 deny bucket policy

        Parameters:
            disable_hybrid_access_mode: If True, revokes IAMAllowedPrincipal permissions
                from the Glue table, enforcing Lake Formation-only governance. Warning:
                this may break existing jobs (e.g., training, processing, ETL) that access
                the table via IAM-based permissions. After this change, all principals must
                be granted access through Lake Formation. If False, prompts the user for
                confirmation before proceeding with hybrid access.
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
            - hybrid_access_mode_disabled: bool

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

        if not disable_hybrid_access_mode:
            logger.warning(
                f"Hybrid access mode is not disabled for table: {database_name_str}.{table_name_str}. "
                f"IAMAllowedPrincipal permissions remain in effect, which means IAM-based access "
                f"to the table is still allowed alongside Lake Formation permissions. "
                f"For more info: https://docs.aws.amazon.com/lake-formation/latest/dg/hybrid-access-mode.html"
            )
            proceed = input(
                "Hybrid access mode is not disabled. IAM-based access to the Glue table will "
                "still be allowed. Do you want to proceed without revoking IAMAllowedPrincipal "
                "permissions? (y/n): "
            ).strip().lower()
            if proceed != "y":
                raise RuntimeError(
                    "User chose not to proceed without disabling hybrid access mode. "
                    "Re-run with disable_hybrid_access_mode=True to revoke IAMAllowedPrincipal permissions."
                )
        else:
            logger.warning(
                f"Disabling hybrid access mode for table: {database_name_str}.{table_name_str}. "
                f"This will revoke IAMAllowedPrincipal permissions, which may break existing jobs "
                f"(e.g., training, processing, ETL) that access this table via IAM-based permissions. "
                f"After this change, all principals must be granted access through Lake Formation. "
                f"For more info: https://docs.aws.amazon.com/lake-formation/latest/dg/hybrid-access-mode.html"
            )


        results = {
            "s3_location_registered": False,
            "lf_permissions_granted": False,
            "hybrid_access_mode_disabled": False
        }

        # Execute Lake Formation setup with fail-fast behavior

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
            raise RuntimeError(
                f"Failed to register S3 location with Lake Formation. "
                f"Subsequent phases skipped. Results: {results}. Error: {e}"
            ) from e

        if not results["s3_location_registered"]:
            raise RuntimeError(
                f"Failed to register S3 location with Lake Formation. "
                f"Subsequent phases skipped. Results: {results}"
            )

        # Phase 2: Grant Lake Formation permissions to the role
        try:
            results["lf_permissions_granted"] = self._grant_lake_formation_permissions(
                role_arn_str, database_name_str, table_name_str, session, region
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to grant Lake Formation permissions. "
                f"Subsequent phases skipped. Results: {results}. Error: {e}"
            ) from e

        if not results["lf_permissions_granted"]:
            raise RuntimeError(
                f"Failed to grant Lake Formation permissions. "
                f"Subsequent phases skipped. Results: {results}"
            )


        # Phase 3: Revoke IAMAllowedPrincipal permissions
        if disable_hybrid_access_mode:
            try:
                results["hybrid_access_mode_disabled"] = self._revoke_iam_allowed_principal(
                    database_name_str, table_name_str, session, region
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to revoke IAMAllowedPrincipal permissions. Results: {results}. Error: {e}"
                ) from e
    
            if not results["hybrid_access_mode_disabled"]:
                raise RuntimeError(
                    f"Failed to revoke IAMAllowedPrincipal permissions. Results: {results}"
                )
            

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

        # Warn about any steps that were not completed
        if not results["s3_location_registered"]:
            logger.warning(
                "S3 location was not registered with Lake Formation. "
                "Re-run enable_lake_formation() or manually register "
                "the S3 location via the Lake Formation console."
            )
        if not results["lf_permissions_granted"]:
            logger.warning(
                "Lake Formation permissions were not granted to the "
                "execution role. Grant permissions manually via the "
                "Lake Formation console or re-run the method after fixing the issue."
            )
        if not results["hybrid_access_mode_disabled"]:
            logger.warning(
                "Hybrid access mode was not disabled. IAM-based access "
                "to the Glue table is still allowed alongside Lake "
                "Formation permissions. To disable, re-run with "
                "disable_hybrid_access_mode=True. For more info: "
                "https://docs.aws.amazon.com/lake-formation/latest/dg/hybrid-access-mode.html"
            )

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
                - **disable_hybrid_access_mode** (bool, required): When True, revokes
                  IAMAllowedPrincipal permissions from the Glue table, enforcing Lake
                  Formation-only governance. **Warning**: this may break existing jobs
                  (e.g., training, processing, ETL) that access the table via IAM-based
                  permissions. After this change, all principals must be granted access
                  through Lake Formation. When False, an interactive prompt asks the user
                  to confirm proceeding with hybrid access mode.
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
                disable_hybrid_access_mode=lake_formation_config.disable_hybrid_access_mode,
            )
        return feature_group
