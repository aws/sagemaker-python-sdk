# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""FeatureGroup with Lake Formation support."""

import logging
from typing import List, Optional

import botocore.exceptions

from sagemaker.core.resources import FeatureGroup as CoreFeatureGroup
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


class LakeFormationConfig:
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
    """

    enabled: bool = False
    use_service_linked_role: bool = True
    registration_role_arn: Optional[str] = None
    show_s3_policy: bool = False


class FeatureGroup(CoreFeatureGroup):

    # Inherit parent docstring and append our additions
    if CoreFeatureGroup.__doc__ and __doc__:
        __doc__ = CoreFeatureGroup.__doc__

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

        Returns:
            S3 bucket policy as a dict with valid JSON structure containing:
            - Version: "2012-10-17"
            - Statement: List with two deny statements:
              1. Deny GetObject, PutObject, DeleteObject on data prefix except allowed principals
              2. Deny ListBucket on bucket with prefix condition except allowed principals
        """
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "DenyAllAccessToFeatureStorePrefixExceptAllowedPrincipals",
                    "Effect": "Deny",
                    "Principal": "*",
                    "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                    "Resource": f"arn:aws:s3:::{bucket_name}/{s3_prefix}/*",
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
                    "Resource": f"arn:aws:s3:::{bucket_name}",
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
        Get a Lake Formation client.

        Args:
            session: Boto3 session. If not provided, a new session will be created.
            region: AWS region name.

        Returns:
            A boto3 Lake Formation client.
        """
        # TODO: don't create w new client for each call
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
            region = session.region_name()

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
            region = session.region_name()

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
            logger.info(f"Revoked IAMAllowedPrincipal from table: {database_name}.{table_name}")
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
            region = session.region_name()

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
    ) -> dict:
        """
        Enable Lake Formation governance for this Feature Group's offline store.

        This method:
        1. Optionally waits for Feature Group to reach 'Created' status
        2. Validates Feature Group status is 'Created'
        3. Registers the offline store S3 location as data lake location
        4. Grants the execution role permissions on the Glue table
        5. Revokes IAMAllowedPrincipal permissions from the Glue table

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

        Returns:
            Dict with status of each Lake Formation operation:
            - s3_registration: bool
            - iam_principal_revoked: bool
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
            region = session.region_name()

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

        # Execute Lake Formation setup with fail-fast behavior
        results = {
            "s3_registration": False,
            "iam_principal_revoked": False,
            "permissions_granted": False,
        }

        # Phase 1: Register S3 with Lake Formation
        try:
            results["s3_registration"] = self._register_s3_with_lake_formation(
                resolved_s3_uri_str,
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
            )

            # Print policy with clear instructions
            import json

            print("\n" + "=" * 80)
            print("S3 Bucket Policy Update recommended")
            print("=" * 80)
            print(
                "\nTo complete Lake Formation setup, add the following deny policy to your S3 bucket."
            )
            print(
                "This policy restricts access to the offline store data to only the allowed principals."
            )
            print("\nBucket:", bucket_name)
            print("\nPolicy to add:")
            print("-" * 40)
            print(json.dumps(policy, indent=2))
            print("-" * 40)
            print("\nNote: Merge this with your existing bucket policy if one exists.")
            print("=" * 80 + "\n")

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
    ) -> Optional["FeatureGroup"]:
        """
        Create a FeatureGroup resource with optional Lake Formation governance.

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

        feature_group = super().create(**create_kwargs)

        # Enable Lake Formation if requested
        if lake_formation_config is not None and lake_formation_config.enabled:
            feature_group.wait_for_status(target_status="Created")
            feature_group.enable_lake_formation(
                session=session,
                region=region,
                use_service_linked_role=lake_formation_config.use_service_linked_role,
                registration_role_arn=lake_formation_config.registration_role_arn,
                show_s3_policy=lake_formation_config.show_s3_policy,
            )
        return feature_group
