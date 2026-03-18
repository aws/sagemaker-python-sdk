# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""FeatureGroupManager with Lake Formation support."""

import json
import logging
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
from sagemaker.core.telemetry import Feature, _telemetry_emitter
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


class FeatureGroupManager:
    """Standalone wrapper for FeatureGroup operations with Lake Formation support.

    Provides create, describe, list, and enable_lake_formation methods that
    delegate to the core FeatureGroup resource class via composition.
    """

    @classmethod
    @_telemetry_emitter(Feature.FEATURE_STORE, "FeatureGroupManager.create_feature_group")
    def create_feature_group(
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
    ) -> Optional[FeatureGroup]:
        """Create a FeatureGroup with optional Lake Formation governance.

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
            The FeatureGroup resource.

        Raises:
            ValueError: If lake_formation_config.enabled=True but prerequisites are missing.
            botocore.exceptions.ClientError: For AWS service related errors.
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

        # Build kwargs, only including non-None values so FeatureGroup.create uses its defaults
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
            create_kwargs["use_pre_prod_offline_store_replicator_lambda"] = (
                use_pre_prod_offline_store_replicator_lambda
            )

        feature_group = FeatureGroup.create(**create_kwargs)

        # Enable Lake Formation if requested
        if lake_formation_config is not None and lake_formation_config.enabled:
            feature_group.wait_for_status(target_status="Created")
            manager = cls()
            manager.enable_lake_formation(
                feature_group, lake_formation_config, session=session, region=region
            )

        return feature_group

    @classmethod
    @_telemetry_emitter(Feature.FEATURE_STORE, "FeatureGroupManager.describe_feature_group")
    def describe_feature_group(
        cls,
        feature_group_name: StrPipeVar,
        next_token: Optional[StrPipeVar] = None,
        session: Optional[Session] = None,
        region: Optional[StrPipeVar] = None,
    ) -> Optional[FeatureGroup]:
        """Describe/retrieve an existing FeatureGroup by name.

        Parameters:
            feature_group_name: The name or ARN of the FeatureGroup.
            next_token: A token to resume pagination of the list of Features.
            session: Boto3 session.
            region: Region name.

        Returns:
            The FeatureGroup instance.
        """
        return FeatureGroup.get(
            feature_group_name=feature_group_name,
            next_token=next_token,
            session=session,
            region=region,
        )

    @_telemetry_emitter(Feature.FEATURE_STORE, "FeatureGroupManager.enable_lake_formation")
    def enable_lake_formation(
        self,
        feature_group: FeatureGroup,
        lake_formation_config: LakeFormationConfig,
        session: Optional[Session] = None,
        region: Optional[str] = None,
    ) -> dict:
        """Enable Lake Formation governance on an existing FeatureGroup.

        This method executes a 3-phase workflow:
        1. Registers the offline store S3 location with Lake Formation
        2. Grants the execution role permissions on the Glue table
        3. Optionally revokes IAMAllowedPrincipal permissions from the Glue table

        Each phase depends on the success of the previous phase - if any phase fails,
        subsequent phases are not executed.

        Parameters:
            feature_group: A FeatureGroup instance to enable Lake Formation on.
            lake_formation_config: LakeFormationConfig with governance settings.
            session: Optional Boto3 session for Lake Formation API calls.
            region: Optional region name for Lake Formation API calls.

        Returns:
            Dict with status of each Lake Formation operation:
            - s3_registration: bool
            - iam_principal_revoked: bool or None (None when disable_hybrid_access_mode=False)
            - permissions_granted: bool

        Raises:
            ValueError: If the Feature Group has no offline store configured,
                if role_arn is not set, or if the Feature Group is not in 'Created' status.
            RuntimeError: If a phase fails and subsequent phases cannot proceed.
        """
        use_service_linked_role = lake_formation_config.use_service_linked_role
        registration_role_arn = lake_formation_config.registration_role_arn
        show_s3_policy = lake_formation_config.show_s3_policy
        disable_hybrid_access_mode = lake_formation_config.disable_hybrid_access_mode

        # Refresh to get latest state
        feature_group.refresh()

        # Validate Feature Group status
        if feature_group.feature_group_status not in ["Created"]:
            raise ValueError(
                f"Feature Group '{feature_group.feature_group_name}' must be in 'Created' status "
                f"to enable Lake Formation. Current status: '{feature_group.feature_group_status}'."
            )

        # Validate offline store exists
        if feature_group.offline_store_config is None or feature_group.offline_store_config == Unassigned():
            raise ValueError(
                f"Feature Group '{feature_group.feature_group_name}' does not have an offline store configured. "
                "Lake Formation can only be enabled for Feature Groups with offline stores."
            )

        # Get role ARN from Feature Group config
        if feature_group.role_arn is None or feature_group.role_arn == Unassigned():
            raise ValueError(
                f"Feature Group '{feature_group.feature_group_name}' does not have a role_arn configured. "
                "Lake Formation requires a role ARN to grant permissions."
            )
        if not use_service_linked_role and registration_role_arn is None:
            raise ValueError(
                "Either 'use_service_linked_role' must be True or 'registration_role_arn' must be provided."
            )

        # Extract required configuration
        s3_config = feature_group.offline_store_config.s3_storage_config
        if s3_config is None:
            raise ValueError("Offline store S3 configuration is missing")

        resolved_s3_uri = s3_config.resolved_output_s3_uri
        if resolved_s3_uri is None or resolved_s3_uri == Unassigned():
            raise ValueError(
                "Resolved S3 URI not available. Ensure the Feature Group is in 'Created' status."
            )

        data_catalog_config = feature_group.offline_store_config.data_catalog_config
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
        role_arn_str = str(feature_group.role_arn)

        # Determine the actual S3 location to register with Lake Formation.
        # For Iceberg tables, the Glue table's StorageDescriptor.Location is the parent
        # path (without /data suffix), while resolved_output_s3_uri always ends with /data.
        s3_location_to_register = resolved_s3_uri_str
        if (
            feature_group.offline_store_config.table_format is not None
            and str(feature_group.offline_store_config.table_format) == "Iceberg"
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

        logger.info(f"Lake Formation setup complete for {feature_group.feature_group_name}: {results}")

        # Generate and optionally print S3 deny policy
        if show_s3_policy:
            # Extract bucket name and prefix from resolved S3 URI using core utility
            bucket_name, s3_prefix = parse_s3_url(resolved_s3_uri_str)

            # Extract account ID from Feature Group ARN
            feature_group_arn_str = str(feature_group.feature_group_arn) if feature_group.feature_group_arn else ""
            account_id = self._extract_account_id_from_arn(feature_group_arn_str)

            # Determine Lake Formation role ARN based on use_service_linked_role flag
            if use_service_linked_role:
                lf_role_arn = self._get_lake_formation_service_linked_role_arn(account_id, region)
            else:
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

    # --- Internal Lake Formation helper methods ---

    @staticmethod
    def _s3_uri_to_arn(s3_uri: str, region: Optional[str] = None) -> str:
        """Convert S3 URI to S3 ARN format for Lake Formation.

        Args:
            s3_uri: S3 URI in format s3://bucket/path or already an ARN
            region: AWS region name. Used to determine the correct partition.

        Returns:
            S3 ARN in format arn:{partition}:s3:::bucket/path
        """
        if s3_uri.startswith("arn:"):
            return s3_uri

        partition = aws_partition(region) if region else "aws"

        bucket, key = parse_s3_url(s3_uri)
        s3_path = f"{bucket}/{key}" if key else bucket
        return f"arn:{partition}:s3:::{s3_path}"

    @staticmethod
    def _extract_account_id_from_arn(arn: str) -> str:
        """Extract AWS account ID from an ARN.

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
        """Generate the Lake Formation service-linked role ARN for an account.

        Args:
            account_id: AWS account ID
            region: AWS region name. Used to determine the correct partition.

        Returns:
            Lake Formation service-linked role ARN.
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
        """Generate an S3 deny policy for Lake Formation governance.

        Args:
            bucket_name: S3 bucket name.
            s3_prefix: S3 prefix path (without bucket name).
            lake_formation_role_arn: Lake Formation registration role ARN.
            feature_store_role_arn: Feature Store execution role ARN.
            region: AWS region name.

        Returns:
            S3 bucket policy dict with two deny statements.
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
        """Get a Lake Formation client, reusing a cached client when possible.

        Args:
            session: Boto3 session.
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
        """Register an S3 location with Lake Formation.

        Args:
            s3_location: S3 URI or ARN to register.
            session: Boto3 session.
            region: AWS region.
            use_service_linked_role: Whether to use the LF service-linked role.
            role_arn: IAM role ARN for registration (required when SLR is False).

        Returns:
            True if registration succeeded or location already registered.

        Raises:
            ValueError: If use_service_linked_role is False but role_arn is not provided.
            ClientError: If registration fails for unexpected reasons.
        """
        if not use_service_linked_role and not role_arn:
            raise ValueError("role_arn must be provided when use_service_linked_role is False")

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
        """Revoke IAMAllowedPrincipal permissions from a Glue table.

        Args:
            database_name: Glue database name.
            table_name: Glue table name.
            session: Boto3 session.
            region: AWS region.

        Returns:
            True if revocation succeeded or permissions didn't exist.

        Raises:
            ClientError: If revocation fails for unexpected reasons.
        """
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
        """Grant permissions to a role on a Glue table via Lake Formation.

        Args:
            role_arn: IAM role ARN to grant permissions to.
            database_name: Glue database name.
            table_name: Glue table name.
            session: Boto3 session.
            region: AWS region.

        Returns:
            True if grant succeeded or permissions already exist.

        Raises:
            ClientError: If grant fails for unexpected reasons.
        """
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
