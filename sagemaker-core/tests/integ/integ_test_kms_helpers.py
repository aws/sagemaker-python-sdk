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
"""KMS test helpers for integration tests.

Ported from SageMakerHulkPythonSDK/tests/integ/kms_utils.py.

NOTE: KMS keys created by these helpers use a fixed alias and are intentionally
reused across test runs rather than deleted after each run. This is because KMS
keys have a mandatory 7-day minimum deletion window (schedule_key_deletion), so
per-run create/delete is not practical. The persistent shared key approach avoids
accumulating orphaned keys and unnecessary costs.
"""
from __future__ import absolute_import

import json

from sagemaker.core.common_utils import aws_partition, sts_regional_endpoint

PRINCIPAL_TEMPLATE = (
    '["{account_id}", "{role_arn}", '
    '"arn:{partition}:iam::{account_id}:role/{sagemaker_role}"] '
)

KEY_ALIAS = "SageMakerTestKMSKey"
POLICY_NAME = "default"
KEY_POLICY = """
{{
  "Version": "2012-10-17",
  "Id": "{id}",
  "Statement": [
    {{
      "Sid": "Enable IAM User Permissions",
      "Effect": "Allow",
      "Principal": {{
        "AWS": {principal}
      }},
      "Action": "kms:*",
      "Resource": "*"
    }}
  ]
}}
"""


def _get_kms_key_arn(kms_client, alias):
    try:
        response = kms_client.describe_key(KeyId="alias/" + alias)
        return response["KeyMetadata"]["Arn"]
    except kms_client.exceptions.NotFoundException:
        return None


def _get_kms_key_id(kms_client, alias):
    try:
        response = kms_client.describe_key(KeyId="alias/" + alias)
        return response["KeyMetadata"]["KeyId"]
    except kms_client.exceptions.NotFoundException:
        return None


def _create_kms_key(
    kms_client, account_id, region, role_arn=None, sagemaker_role="SageMakerRole", alias=KEY_ALIAS
):
    if role_arn:
        principal = PRINCIPAL_TEMPLATE.format(
            partition=aws_partition(region),
            account_id=account_id,
            role_arn=role_arn,
            sagemaker_role=sagemaker_role,
        )
    else:
        principal = '"{account_id}"'.format(account_id=account_id)

    response = kms_client.create_key(
        Policy=KEY_POLICY.format(
            id=POLICY_NAME, principal=principal, sagemaker_role=sagemaker_role
        ),
        Description="KMS key for SageMaker Python SDK integ tests",
    )
    key_arn = response["KeyMetadata"]["Arn"]

    if alias:
        kms_client.create_alias(AliasName="alias/" + alias, TargetKeyId=key_arn)
    return key_arn


def _add_role_to_policy(
    kms_client, account_id, role_arn, region, alias=KEY_ALIAS, sagemaker_role="SageMakerRole"
):
    key_id = _get_kms_key_id(kms_client, alias)
    policy = kms_client.get_key_policy(KeyId=key_id, PolicyName=POLICY_NAME)
    policy = json.loads(policy["Policy"])
    principal = policy["Statement"][0]["Principal"]["AWS"]

    if role_arn not in principal or sagemaker_role not in principal:
        principal = PRINCIPAL_TEMPLATE.format(
            partition=aws_partition(region),
            account_id=account_id,
            role_arn=role_arn,
            sagemaker_role=sagemaker_role,
        )

        kms_client.put_key_policy(
            KeyId=key_id,
            PolicyName=POLICY_NAME,
            Policy=KEY_POLICY.format(id=POLICY_NAME, principal=principal),
        )


def get_or_create_kms_key(
    sagemaker_session, role_arn=None, alias=KEY_ALIAS, sagemaker_role="SageMakerRole"
):
    kms_client = sagemaker_session.boto_session.client("kms")
    kms_key_arn = _get_kms_key_arn(kms_client, alias)

    region = sagemaker_session.boto_region_name
    sts_client = sagemaker_session.boto_session.client(
        "sts", region_name=region, endpoint_url=sts_regional_endpoint(region)
    )
    account_id = sts_client.get_caller_identity()["Account"]

    if kms_key_arn is None:
        return _create_kms_key(kms_client, account_id, region, role_arn, sagemaker_role, alias)

    if role_arn:
        _add_role_to_policy(kms_client, account_id, role_arn, region, alias, sagemaker_role)

    return kms_key_arn
